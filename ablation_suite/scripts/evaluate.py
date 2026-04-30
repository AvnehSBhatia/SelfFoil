#!/usr/bin/env python3
"""Evaluate WHISP checkpoints; append JSONL metrics (defaults: all runs under runs/<cat>/<slug>/)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        enc_paths = tuple(Path(x) for x in meta["encoder_ckpts"])
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
        sum_cl_mse = 0.0
        n_cl = 0
        sum_cd_mse_const = 0.0
        n_tot = 0
        cd_mean = bundle["cd_flat"][v_idx].mean()

        t_cst0: list[torch.Tensor] = []
        p_cst0: list[torch.Tensor] = []
        cl_list: list[torch.Tensor] = []
        clp_list: list[torch.Tensor] = []
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
            n_tot += sl.numel()
            last_k = model.n_outer - 1
            key = f"cl_gamma_{last_k}"
            if key in aux:
                sum_cl_mse += float(F.mse_loss(aux[key], cl, reduction="sum").detach())
                n_cl += sl.numel()
            sum_cd_mse_const += float(((cd - cd_mean) ** 2).sum())
            if n_collected < max_pts:
                take = min(sl.numel(), max_pts - n_collected)
                t_cst0.append(cst_tgt[:take, 0].detach().cpu())
                p_cst0.append(cst_pred[:take, 0].detach().cpu())
                cl_list.append(cl[:take].detach().cpu())
                clp_list.append(aux["cl_direct"][:take].detach().cpu())
                n_collected += take

        cst_error = (sum_cst_l2 / max(1, n_tot)) ** 0.5
        cl_error = (sum_cl_mse / max(1, n_cl)) if n_cl else None
        cd_error = sum_cd_mse_const / max(1, n_tot)
        cat, _, slug = run_key.partition("/")
        record: dict[str, object] = {
            "model": run_key,
            "category": cat,
            "slug": slug,
            "cst_error": round(cst_error, 6),
            "cl_error": (None if cl_error is None else round(float(cl_error), 6)),
            "cd_error": round(float(cd_error), 6),
            "epoch": int(meta.get("epochs", -1)),
            "trainable_params": int(meta.get("trainable_params", -1)),
        }
        if "frac_train" in meta:
            record["frac_train"] = float(meta["frac_train"])
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        summary_rows.append(record)

        t0 = torch.cat(t_cst0).numpy() if t_cst0 else None
        p0 = torch.cat(p_cst0).numpy() if p_cst0 else None
        if t0 is not None and p0 is not None:
            safe = run_key.replace("/", "__")
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(t0, p0, s=4, alpha=0.35)
            lims = [min(t0.min(), p0.min()), max(t0.max(), p0.max())]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.6)
            ax.set_xlabel("CST target dim 0")
            ax.set_ylabel("CST pred dim 0")
            ax.set_title(f"Eval {run_key} (val)")
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            fig.savefig(fd / f"eval_ablation_{safe}_cst0_scatter.png", dpi=200)
            plt.close(fig)
        if cl_list:
            ctrue = torch.cat(cl_list).numpy()
            cpred = torch.cat(clp_list).numpy()
            safe = run_key.replace("/", "__")
            fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
            ax2.scatter(ctrue, cpred, s=4, alpha=0.35)
            lo = min(ctrue.min(), cpred.min())
            hi = max(ctrue.max(), cpred.max())
            ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
            ax2.set_xlabel("Cl true")
            ax2.set_ylabel("Cl direct head")
            ax2.set_title(f"Eval {run_key} — Cl head")
            fig2.tight_layout()
            fig2.savefig(fd / f"eval_ablation_{safe}_cl_scatter.png", dpi=200)
            plt.close(fig2)

    if summary_rows:
        figs, axs = plt.subplots(figsize=(8, 5))
        xs = [float(r["cst_error"]) for r in summary_rows]
        ys = [float(r["cl_error"]) if r.get("cl_error") is not None else 0.0 for r in summary_rows]
        labs = [str(r["slug"]) for r in summary_rows]
        axs.scatter(xs, ys, s=36, alpha=0.75)
        for i, lb in enumerate(labs):
            axs.annotate(lb, (xs[i], ys[i]), fontsize=7, xytext=(2, 2), textcoords="offset points")
        axs.set_xlabel("CST RMSE (this eval batch)")
        axs.set_ylabel("Cl MSE (or 0)")
        axs.set_title("Eval batch summary (this invocation)")
        figs.tight_layout()
        figs.savefig(fd / "eval_ablation_batch_summary_landscape.png", dpi=200)
        plt.close(figs)

    print(f"Saved evaluation figures under {fd}")


if __name__ == "__main__":
    main()
