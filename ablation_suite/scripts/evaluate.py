#!/usr/bin/env python3
"""Evaluate WHISP checkpoints; append JSONL metrics (defaults: all runs under runs/<cat>/<slug>/)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_suite.models.whisp_ablated import WHISPAblated
from core.csv_tensor_cache import load_or_build_cache
from core.figures_path import figures_dir
from core.device import resolve_device
from scripts.train_whisp import (
    airfoil_row_splits,
    move_bundle,
    polar_mask_for_rows,
    required_bundle_keys,
)


def discover_checkpoints(suite_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    runs = suite_root / "runs"
    if not runs.is_dir():
        return out
    for ck in sorted(runs.glob("*/*/model.pt")):
        slug = ck.parent.name
        cat = ck.parent.parent.name
        out.append((f"{cat}/{slug}", ck))
    return out


def resolve_encoder_ckpts_from_meta(meta: dict, root: Path) -> tuple[Path, ...]:
    """Resolve encoder checkpoint paths across machines/workspaces."""
    resolved: list[Path] = []
    for raw in meta["encoder_ckpts"]:
        p = Path(str(raw))
        if p.is_file():
            resolved.append(p)
            continue
        fallback = root / "models" / p.name
        if fallback.is_file():
            resolved.append(fallback)
            continue
        raise FileNotFoundError(
            f"Missing encoder checkpoint {p} (fallback also missing: {fallback})."
        )
    return tuple(resolved)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.square(y_true - y_pred).sum())
    y_mean = float(y_true.mean())
    ss_tot = float(np.square(y_true - y_mean).sum())
    return 1.0 - _safe_div(ss_res, ss_tot)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "original.csv")
    p.add_argument("--cache", type=Path, default=ROOT / "models" / "original_tensors.pt")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac-train", type=float, default=0.8)
    p.add_argument("--frac-val", type=float, default=0.1)
    p.add_argument("--suite-root", type=Path, default=ROOT / "ablation_suite")
    p.add_argument("--metrics-path", type=Path, default=None)
    p.add_argument(
        "--run-ids",
        nargs="*",
        default=None,
        help="Explicit run ids (category/slug). Default: discover all checkpoints.",
    )
    args = p.parse_args()

    device = resolve_device(args.device)
    bundle_cpu = load_or_build_cache(args.csv, args.cache, args.max_rows, args.rebuild_cache)
    missing = required_bundle_keys() - set(bundle_cpu.keys())
    if missing:
        raise SystemExit(f"Cache missing keys {sorted(missing)}.")

    n_rows = int(bundle_cpu["n_rows"])
    bundle = move_bundle(bundle_cpu, device)
    polar_row = bundle["polar_row_idx"]
    re_log_all = bundle["re_log_flat"]

    logs_dir = args.suite_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_path or (logs_dir / "metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, Path]]
    if args.run_ids:
        targets = []
        for rid in args.run_ids:
            cat, _, slug = rid.partition("/")
            if not slug:
                print(f"skip bad run-id {rid!r}")
                continue
            pth = args.suite_root / "runs" / cat / slug / "model.pt"
            if pth.is_file():
                targets.append((rid, pth))
            else:
                print(f"skip missing {pth}")
    else:
        targets = discover_checkpoints(args.suite_root)

    fd = figures_dir()
    summary_rows: list[dict[str, object]] = []
    detailed_path = logs_dir / "metrics_detailed.json"
    detailed_rows: list[dict[str, object]] = []

    for run_key, ckpt_path in targets:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        meta = blob["meta"]
        spec = meta["spec"]
        train_cfg = meta.get("train", {})
        frac_train = float(meta.get("frac_train", args.frac_train))
        frac_val = float(meta.get("frac_val", args.frac_val))
        train_rows, val_rows, _ = airfoil_row_splits(n_rows, args.seed, frac_train, frac_val)
        val_mask = polar_mask_for_rows(polar_row, val_rows)
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)
        enc_paths = resolve_encoder_ckpts_from_meta(meta, ROOT)
        model = WHISPAblated(enc_paths, spec).to(device)
        model.load_state_dict(blob["model"])
        model.eval()

        v_idx = val_idx
        if train_cfg.get("extrapolation") and train_cfg.get("re_log_val_min") is not None:
            va_min_t = torch.tensor(float(train_cfg["re_log_val_min"]), device=device, dtype=re_log_all.dtype)
            v_idx = val_idx[re_log_all[val_idx] >= va_min_t]
        if v_idx.numel() == 0:
            print(f"skip empty val for {run_key}")
            continue

        sum_cst_l2 = 0.0
        sum_cst_abs = 0.0
        sum_cst_sq = 0.0
        sum_cl_mse = 0.0
        sum_cl_abs = 0.0
        n_cl = 0
        sum_cd_mse_const = 0.0
        n_tot = 0
        cd_mean = bundle["cd_flat"][v_idx].mean()

        t_cst0: list[torch.Tensor] = []
        p_cst0: list[torch.Tensor] = []
        cl_list: list[torch.Tensor] = []
        clp_list: list[torch.Tensor] = []
        cst_err_rows: list[torch.Tensor] = []
        max_pts = 5000
        n_collected = 0

        for s in range(0, v_idx.numel(), args.batch):
            sl = v_idx[s : s + args.batch]
            cl = bundle["cl_flat"][sl]
            cd = bundle["cd_flat"][sl]
            re_log = bundle["re_log_flat"][sl]
            mach = bundle["mach_flat"][sl]
            alpha = bundle["alpha_flat"][sl]
            row = polar_row[sl]
            cst_tgt = bundle["cst18"][row]
            cst_pred, aux = model(cl, cd, re_log, mach, alpha)
            err = cst_pred - cst_tgt
            sum_cst_l2 += float(torch.norm(err, dim=-1).pow(2).sum())
            sum_cst_abs += float(torch.abs(err).sum())
            sum_cst_sq += float(torch.square(err).sum())
            n_tot += sl.numel()
            last_k = model.n_outer - 1
            key = f"cl_gamma_{last_k}"
            if key in aux:
                sum_cl_mse += float(F.mse_loss(aux[key], cl, reduction="sum").detach())
                sum_cl_abs += float(torch.abs(aux[key] - cl).sum().detach())
                n_cl += sl.numel()
            sum_cd_mse_const += float(((cd - cd_mean) ** 2).sum())
            if n_collected < max_pts:
                take = min(sl.numel(), max_pts - n_collected)
                t_cst0.append(cst_tgt[:take, 0].detach().cpu())
                p_cst0.append(cst_pred[:take, 0].detach().cpu())
                cl_list.append(cl[:take].detach().cpu())
                clp_list.append(aux["cl_direct"][:take].detach().cpu())
                n_collected += take
            cst_err_rows.append(err.detach().cpu())

        cst_error = (sum_cst_l2 / max(1, n_tot)) ** 0.5
        cl_error = (sum_cl_mse / max(1, n_cl)) if n_cl else None
        cd_error = sum_cd_mse_const / max(1, n_tot)
        n_cst_vals = max(1, n_tot * int(bundle["cst18"].shape[1]))
        cst_rmse = math.sqrt(_safe_div(sum_cst_sq, float(n_cst_vals)))
        cst_mae = _safe_div(sum_cst_abs, float(n_cst_vals))
        cl_rmse = math.sqrt(_safe_div(sum_cl_mse, float(max(1, n_cl)))) if n_cl else float("nan")
        cl_mae = _safe_div(sum_cl_abs, float(max(1, n_cl))) if n_cl else float("nan")

        cst_err_np = torch.cat(cst_err_rows, dim=0).numpy() if cst_err_rows else np.zeros((1, 18), dtype=np.float32)
        cst_abs_np = np.abs(cst_err_np)
        cst_abs_p90 = float(np.percentile(cst_abs_np, 90))
        cst_abs_p95 = float(np.percentile(cst_abs_np, 95))
        cst_abs_p99 = float(np.percentile(cst_abs_np, 99))
        cst_bias_mean = float(np.mean(cst_err_np))

        cat, _, slug = run_key.partition("/")
        record: dict[str, object] = {
            "model": run_key,
            "category": cat,
            "slug": slug,
            "cst_error": round(cst_error, 6),
            "cst_rmse": round(float(cst_rmse), 6),
            "cst_mae": round(float(cst_mae), 6),
            "cst_abs_p90": round(float(cst_abs_p90), 6),
            "cst_abs_p95": round(float(cst_abs_p95), 6),
            "cst_abs_p99": round(float(cst_abs_p99), 6),
            "cst_bias_mean": round(float(cst_bias_mean), 6),
            "cl_error": (None if cl_error is None else round(float(cl_error), 6)),
            "cl_rmse": (None if n_cl == 0 else round(float(cl_rmse), 6)),
            "cl_mae": (None if n_cl == 0 else round(float(cl_mae), 6)),
            "cd_error": round(float(cd_error), 6),
            "epoch": int(meta.get("epochs", -1)),
            "trainable_params": int(meta.get("trainable_params", -1)),
        }
        if "frac_train" in meta:
            record["frac_train"] = float(meta["frac_train"])
        # Composite score: lower is better; balances shape and lift fidelity.
        record["paper_score"] = round(
            float(record["cst_rmse"]) + 0.25 * (float(record["cl_rmse"]) if n_cl else 0.0),
            6,
        )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        summary_rows.append(record)

        t0 = torch.cat(t_cst0).numpy() if t_cst0 else None
        p0 = torch.cat(p_cst0).numpy() if p_cst0 else None
        ctrue = torch.cat(cl_list).numpy() if cl_list else None
        cpred = torch.cat(clp_list).numpy() if clp_list else None
        cst0_r2 = _safe_r2(t0, p0) if t0 is not None and p0 is not None else float("nan")
        cl_r2 = _safe_r2(ctrue, cpred) if ctrue is not None and cpred is not None else float("nan")
        cl_corr = (
            float(np.corrcoef(ctrue, cpred)[0, 1])
            if ctrue is not None and cpred is not None and ctrue.size > 1
            else float("nan")
        )

        detailed_rows.append(
            {
                "model": run_key,
                "checkpoint": str(ckpt_path),
                "metrics": {
                    **record,
                    "cst0_r2": cst0_r2,
                    "cl_direct_r2": cl_r2,
                    "cl_direct_corr": cl_corr,
                },
            }
        )

        if t0 is not None and p0 is not None:
            safe = run_key.replace("/", "__")
            residual = p0 - t0
            fig, (ax, axr) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.45, 1.0]})
            hb = ax.hexbin(t0, p0, gridsize=48, cmap="viridis", mincnt=1)
            lims = [min(t0.min(), p0.min()), max(t0.max(), p0.max())]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6)
            ax.set_xlabel("CST target dim 0")
            ax.set_ylabel("CST pred dim 0")
            ax.set_title(f"{run_key}: CST dim-0 parity")
            ax.set_aspect("equal", adjustable="box")
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Sample count")
            ax.text(
                0.02,
                0.97,
                f"R²={cst0_r2:.3f}\nN={t0.size}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            axr.hist(residual, bins=40, color="#4C72B0", alpha=0.82)
            axr.set_title("Residual distribution")
            axr.set_xlabel("Pred - Target")
            axr.set_ylabel("Count")
            axr.axvline(0.0, color="k", ls="--", lw=0.9)
            fig.tight_layout()
            fig.savefig(fd / f"eval_ablation_{safe}_cst0_diagnostics.png", dpi=260)
            plt.close(fig)
        if ctrue is not None and cpred is not None:
            safe = run_key.replace("/", "__")
            cl_residual = cpred - ctrue
            fig2, (ax2, ax2r) = plt.subplots(
                1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.45, 1.0]}
            )
            hb2 = ax2.hexbin(ctrue, cpred, gridsize=48, cmap="magma", mincnt=1)
            lo = min(ctrue.min(), cpred.min())
            hi = max(ctrue.max(), cpred.max())
            ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
            ax2.set_xlabel("Cl true")
            ax2.set_ylabel("Cl direct head")
            ax2.set_title(f"{run_key}: Cl parity")
            cb2 = fig2.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.04)
            cb2.set_label("Sample count")
            ax2.text(
                0.02,
                0.97,
                f"R²={cl_r2:.3f}\nr={cl_corr:.3f}",
                transform=ax2.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            ax2r.hist(cl_residual, bins=40, color="#DD8452", alpha=0.84)
            ax2r.set_title("Residual distribution")
            ax2r.set_xlabel("Pred - True")
            ax2r.set_ylabel("Count")
            ax2r.axvline(0.0, color="k", ls="--", lw=0.9)
            fig2.tight_layout()
            fig2.savefig(fd / f"eval_ablation_{safe}_cl_diagnostics.png", dpi=260)
            plt.close(fig2)

    if summary_rows:
        figs, axs = plt.subplots(figsize=(8.8, 5.4))
        xs = [float(r["cst_rmse"]) for r in summary_rows]
        ys = [float(r["cl_rmse"]) if r.get("cl_rmse") is not None else float("nan") for r in summary_rows]
        labs = [str(r["model"]) for r in summary_rows]
        axs.scatter(xs, ys, s=50, alpha=0.8, c=np.arange(len(xs)), cmap="tab20")
        for i, lb in enumerate(labs):
            axs.annotate(lb, (xs[i], ys[i]), fontsize=7, xytext=(2, 2), textcoords="offset points")
        axs.set_xlabel("CST RMSE (lower is better)")
        axs.set_ylabel("Cl RMSE (lower is better)")
        axs.set_title("Evaluation landscape (this invocation)")
        axs.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        figs.tight_layout()
        figs.savefig(fd / "eval_ablation_batch_summary_landscape.png", dpi=260)
        plt.close(figs)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_rows, f, indent=2)
    print(f"Saved detailed metrics JSON: {detailed_path}")
    print(f"Saved evaluation figures under {fd}")


if __name__ == "__main__":
    main()
