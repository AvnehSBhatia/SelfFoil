#!/usr/bin/env python3
"""Benchmark all ablation/baseline checkpoints on data/test.csv (CPU batched inference)."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_suite.models.whisp_ablated import WHISPAblated
from core.csv_tensor_cache import load_or_build_cache
from core.device import resolve_device
from core.figures_path import figures_dir


def resolve_encoder_ckpts_from_meta(meta: dict, root: Path) -> tuple[Path, ...]:
    """Resolve encoder checkpoint paths across machines/workspaces.

    Checkpoints may store absolute paths from a different host (e.g. /workspace/...).
    If a stored path doesn't exist, fallback to root/models/<filename>.
    """
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


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.square(y_true - y_pred).sum())
    y_mean = float(y_true.mean())
    ss_tot = float(np.square(y_true - y_mean).sum())
    return 1.0 - _safe_div(ss_res, ss_tot)


def _last_val_loss(run_dir: Path) -> float:
    p = run_dir / "history.json"
    if not p.is_file():
        return float("nan")
    try:
        with p.open(encoding="utf-8") as f:
            hist = json.load(f)
        if not hist:
            return float("nan")
        return float(hist[-1].get("val_loss", float("nan")))
    except Exception:
        return float("nan")


@torch.no_grad()
def evaluate_one(
    run_key: str,
    ckpt_path: Path,
    bundle: dict[str, torch.Tensor | int],
    device: torch.device,
    batch_size: int,
    speed_repeats: int,
    speed_warmup_batches: int,
) -> tuple[dict[str, float | int | str], dict[str, object]]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = blob["meta"]
    spec = meta["spec"]
    encoder_ckpts = resolve_encoder_ckpts_from_meta(meta, ROOT)
    model = WHISPAblated(encoder_ckpts, spec).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    n_total = int(bundle["total_polar"])
    idx_all = torch.arange(n_total, device=device, dtype=torch.long)
    polar_row = bundle["polar_row_idx"].to(device)

    sum_cst_sq = 0.0
    sum_cst_abs = 0.0
    sum_cld_sq = 0.0
    sum_cld_abs = 0.0
    sum_clg_sq = 0.0
    n_clg = 0

    cst_targets: list[torch.Tensor] = []
    cst_preds: list[torch.Tensor] = []
    cl_true_list: list[torch.Tensor] = []
    cl_direct_list: list[torch.Tensor] = []

    last_k = model.n_outer - 1
    clg_key = f"cl_gamma_{last_k}"

    for s in range(0, n_total, batch_size):
        sl = idx_all[s : s + batch_size]
        cl = bundle["cl_flat"][sl].to(device)
        cd = bundle["cd_flat"][sl].to(device)
        re_log = bundle["re_log_flat"][sl].to(device)
        mach = bundle["mach_flat"][sl].to(device)
        alpha = bundle["alpha_flat"][sl].to(device)
        rows = polar_row[sl]
        cst_tgt = bundle["cst18"][rows].to(device)

        cst_pred, aux = model(cl, cd, re_log, mach, alpha)
        err_cst = cst_pred - cst_tgt
        err_cld = aux["cl_direct"] - cl

        sum_cst_sq += float(torch.square(err_cst).sum().item())
        sum_cst_abs += float(torch.abs(err_cst).sum().item())
        sum_cld_sq += float(torch.square(err_cld).sum().item())
        sum_cld_abs += float(torch.abs(err_cld).sum().item())

        if clg_key in aux:
            err_clg = aux[clg_key] - cl
            sum_clg_sq += float(torch.square(err_clg).sum().item())
            n_clg += int(sl.numel())

        cst_targets.append(cst_tgt.detach().cpu())
        cst_preds.append(cst_pred.detach().cpu())
        cl_true_list.append(cl.detach().cpu())
        cl_direct_list.append(aux["cl_direct"].detach().cpu())

    cst_t = torch.cat(cst_targets, dim=0).numpy()
    cst_p = torch.cat(cst_preds, dim=0).numpy()
    cl_t = torch.cat(cl_true_list, dim=0).numpy()
    cl_p = torch.cat(cl_direct_list, dim=0).numpy()

    mean_dim = np.mean(np.abs(cst_t), axis=0) + 1e-9
    cst_nrmse = float(np.mean(np.sqrt(np.mean((cst_p - cst_t) ** 2, axis=0)) / mean_dim))
    cl_corr = float(np.corrcoef(cl_t, cl_p)[0, 1]) if cl_t.size >= 2 else float("nan")

    # Speed benchmark: repeat full forward passes, excluding warmup.
    warmup = max(0, speed_warmup_batches)
    for s in range(0, min(n_total, warmup * batch_size), batch_size):
        sl = idx_all[s : s + batch_size]
        cl = bundle["cl_flat"][sl].to(device)
        cd = bundle["cd_flat"][sl].to(device)
        re_log = bundle["re_log_flat"][sl].to(device)
        mach = bundle["mach_flat"][sl].to(device)
        alpha = bundle["alpha_flat"][sl].to(device)
        _ = model(cl, cd, re_log, mach, alpha)

    elapsed_runs: list[float] = []
    for _ in range(max(1, speed_repeats)):
        t0 = time.perf_counter()
        for s in range(0, n_total, batch_size):
            sl = idx_all[s : s + batch_size]
            cl = bundle["cl_flat"][sl].to(device)
            cd = bundle["cd_flat"][sl].to(device)
            re_log = bundle["re_log_flat"][sl].to(device)
            mach = bundle["mach_flat"][sl].to(device)
            alpha = bundle["alpha_flat"][sl].to(device)
            _ = model(cl, cd, re_log, mach, alpha)
        elapsed_runs.append(time.perf_counter() - t0)

    elapsed_mean = float(statistics.mean(elapsed_runs))
    throughput = _safe_div(float(n_total), elapsed_mean)
    latency_per_sample_ms = _safe_div(elapsed_mean * 1000.0, float(n_total))

    run_dir = ckpt_path.parent
    final_val_loss = _last_val_loss(run_dir)

    cat, _, slug = run_key.partition("/")
    n_cst_vals = cst_t.shape[0] * cst_t.shape[1]
    cst_err = cst_p - cst_t
    cst_rmse_per_dim = np.sqrt(np.mean(np.square(cst_err), axis=0))
    cst_mae_per_dim = np.mean(np.abs(cst_err), axis=0)
    cst_bias_per_dim = np.mean(cst_err, axis=0)
    cl_err = cl_p - cl_t
    cl_abs_err = np.abs(cl_err)
    cl_sq_err = np.square(cl_err)
    elapsed_std = float(statistics.pstdev(elapsed_runs)) if len(elapsed_runs) > 1 else 0.0
    elapsed_min = float(min(elapsed_runs))
    elapsed_max = float(max(elapsed_runs))
    record: dict[str, float | int | str] = {
        "model": run_key,
        "category": cat,
        "slug": slug,
        "samples": int(n_total),
        "trainable_params": int(meta.get("trainable_params", -1)),
        "epochs": int(meta.get("epochs", -1)),
        "batch_size": int(batch_size),
        "cpu_elapsed_s": elapsed_mean,
        "throughput_samples_per_s": throughput,
        "latency_ms_per_sample": latency_per_sample_ms,
        "cst_rmse": math.sqrt(_safe_div(sum_cst_sq, float(n_cst_vals))),
        "cst_mae": _safe_div(sum_cst_abs, float(n_cst_vals)),
        "cst_nrmse": cst_nrmse,
        "cl_direct_mse": _safe_div(sum_cld_sq, float(n_total)),
        "cl_direct_rmse": math.sqrt(_safe_div(sum_cld_sq, float(n_total))),
        "cl_direct_mae": _safe_div(sum_cld_abs, float(n_total)),
        "cl_direct_r2": _safe_r2(cl_t, cl_p),
        "cl_direct_corr": cl_corr,
        "cl_gamma_mse": _safe_div(sum_clg_sq, float(n_clg)) if n_clg else float("nan"),
        "final_train_val_loss": final_val_loss,
    }
    detailed: dict[str, object] = {
        "model": run_key,
        "checkpoint_path": str(ckpt_path),
        "meta": {
            "category": cat,
            "slug": slug,
            "samples": int(n_total),
            "trainable_params": int(meta.get("trainable_params", -1)),
            "epochs": int(meta.get("epochs", -1)),
            "batch_size": int(batch_size),
            "device": str(device),
        },
        "timing": {
            "speed_repeats": int(max(1, speed_repeats)),
            "warmup_batches": int(max(0, speed_warmup_batches)),
            "elapsed_runs_s": [float(x) for x in elapsed_runs],
            "elapsed_mean_s": elapsed_mean,
            "elapsed_std_s": elapsed_std,
            "elapsed_min_s": elapsed_min,
            "elapsed_max_s": elapsed_max,
            "throughput_samples_per_s": throughput,
            "latency_ms_per_sample": latency_per_sample_ms,
        },
        "cst_metrics": {
            "rmse": float(record["cst_rmse"]),
            "mae": float(record["cst_mae"]),
            "nrmse": float(record["cst_nrmse"]),
            "rmse_per_dim": [float(x) for x in cst_rmse_per_dim.tolist()],
            "mae_per_dim": [float(x) for x in cst_mae_per_dim.tolist()],
            "bias_per_dim": [float(x) for x in cst_bias_per_dim.tolist()],
            "error_abs_p50": float(np.percentile(np.abs(cst_err), 50)),
            "error_abs_p90": float(np.percentile(np.abs(cst_err), 90)),
            "error_abs_p95": float(np.percentile(np.abs(cst_err), 95)),
            "error_abs_p99": float(np.percentile(np.abs(cst_err), 99)),
        },
        "cl_direct_metrics": {
            "mse": float(record["cl_direct_mse"]),
            "rmse": float(record["cl_direct_rmse"]),
            "mae": float(record["cl_direct_mae"]),
            "r2": float(record["cl_direct_r2"]),
            "corr": float(record["cl_direct_corr"]),
            "abs_error_p50": float(np.percentile(cl_abs_err, 50)),
            "abs_error_p90": float(np.percentile(cl_abs_err, 90)),
            "abs_error_p95": float(np.percentile(cl_abs_err, 95)),
            "abs_error_p99": float(np.percentile(cl_abs_err, 99)),
            "error_bias": float(np.mean(cl_err)),
            "error_std": float(np.std(cl_err)),
            "squared_error_mean": float(np.mean(cl_sq_err)),
        },
        "cl_gamma_metrics": {
            "mse": float(record["cl_gamma_mse"]),
            "count": int(n_clg),
        },
        "training_loss_reference": {
            "final_val_loss": final_val_loss,
        },
    }
    return record, detailed


def _scatter_with_labels(
    xs: list[float],
    ys: list[float],
    labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.scatter(xs, ys, s=46, alpha=0.8)
    for i, label in enumerate(labels):
        ax.annotate(label, (xs[i], ys[i]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _bar_by_model(
    rows: list[dict[str, float | int | str]],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    labels = [str(r["model"]).replace("/", "\n") for r in rows]
    vals = [float(r[key]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(10.0, 0.3 * len(labels)), 5.2))
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _line_losses_per_category(
    rows: list[dict[str, float | int | str]],
    metric_key: str,
    out_path: Path,
) -> None:
    by_cat: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        cat = str(r["category"])
        by_cat.setdefault(cat, []).append((str(r["slug"]), float(r[metric_key])))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for cat in sorted(by_cat.keys()):
        items = sorted(by_cat[cat], key=lambda x: x[0])
        xs = np.arange(len(items))
        ys = [v for _, v in items]
        ax.plot(xs, ys, marker="o", label=cat)
    ax.set_xlabel("Model index within category (sorted by slug)")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_key} by category")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "test.csv")
    p.add_argument("--cache", type=Path, default=ROOT / "models" / "test_tensors.pt")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--suite-root", type=Path, default=ROOT / "ablation_suite")
    p.add_argument("--device", default="cpu", choices=["cpu", "auto", "mps", "cuda"])
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--speed-repeats", type=int, default=3)
    p.add_argument("--speed-warmup-batches", type=int, default=2)
    p.add_argument("--run-ids", nargs="*", default=None, help="Optional subset: category/slug")
    args = p.parse_args()

    device = resolve_device(args.device)
    if device.type != "cpu":
        print(f"[warn] requested device resolved to {device}; continuing anyway.")
    bundle_cpu = load_or_build_cache(args.csv, args.cache, args.max_rows, args.rebuild_cache)

    if args.run_ids:
        targets: list[tuple[str, Path]] = []
        for rid in args.run_ids:
            cat, _, slug = rid.partition("/")
            if not slug:
                print(f"skip invalid run-id {rid!r}")
                continue
            ck = args.suite_root / "runs" / cat / slug / "model.pt"
            if ck.is_file():
                targets.append((rid, ck))
            else:
                print(f"skip missing checkpoint {ck}")
    else:
        targets = discover_checkpoints(args.suite_root)

    if not targets:
        raise SystemExit(f"No checkpoints discovered under {(args.suite_root / 'runs')}.")

    fig_dir = figures_dir()
    out_json = fig_dir / "benchmark_all_models_testcsv.json"
    out_json_full = fig_dir / "benchmark_all_models_testcsv_all_metrics.json"
    out_csv = fig_dir / "benchmark_all_models_testcsv.csv"
    records: list[dict[str, float | int | str]] = []
    detailed_records: list[dict[str, object]] = []

    # Move once to target device to avoid repeated host-device transfers.
    bundle = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in bundle_cpu.items()}

    for run_key, ckpt in targets:
        rec, rec_detailed = evaluate_one(
            run_key=run_key,
            ckpt_path=ckpt,
            bundle=bundle,
            device=device,
            batch_size=args.batch,
            speed_repeats=args.speed_repeats,
            speed_warmup_batches=args.speed_warmup_batches,
        )
        records.append(rec)
        detailed_records.append(rec_detailed)
        print(json.dumps(rec))

    records = sorted(records, key=lambda r: float(r["cst_rmse"]))
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with out_json_full.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_csv": str(args.csv),
                "device": str(device),
                "batch_size": int(args.batch),
                "speed_repeats": int(args.speed_repeats),
                "speed_warmup_batches": int(args.speed_warmup_batches),
                "num_models": len(records),
                "models": detailed_records,
            },
            f,
            indent=2,
        )

    headers = list(records[0].keys())
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in records:
            vals = []
            for h in headers:
                v = r[h]
                if isinstance(v, str):
                    vals.append(v)
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

    _bar_by_model(
        records,
        key="cst_rmse",
        ylabel="CST RMSE",
        title="CPU test.csv benchmark: CST RMSE by model",
        out_path=fig_dir / "benchmark_testcsv_cst_rmse_bar.png",
    )
    _bar_by_model(
        records,
        key="throughput_samples_per_s",
        ylabel="samples / second",
        title="CPU test.csv benchmark: throughput by model",
        out_path=fig_dir / "benchmark_testcsv_throughput_bar.png",
    )
    _bar_by_model(
        records,
        key="final_train_val_loss",
        ylabel="final val loss from history.json",
        title="Training final loss by model",
        out_path=fig_dir / "benchmark_testcsv_final_train_val_loss_bar.png",
    )

    labels = [str(r["model"]) for r in records]
    _scatter_with_labels(
        xs=[float(r["cst_rmse"]) for r in records],
        ys=[float(r["throughput_samples_per_s"]) for r in records],
        labels=labels,
        title="Accuracy-speed Pareto view (lower RMSE, higher throughput)",
        xlabel="CST RMSE",
        ylabel="Throughput (samples/s)",
        out_path=fig_dir / "benchmark_testcsv_accuracy_vs_speed.png",
    )
    _scatter_with_labels(
        xs=[float(r["cl_direct_mse"]) for r in records],
        ys=[float(r["final_train_val_loss"]) for r in records],
        labels=labels,
        title="Cl direct MSE vs final training val loss",
        xlabel="Cl direct MSE (test.csv)",
        ylabel="Final val loss (training history)",
        out_path=fig_dir / "benchmark_testcsv_cl_mse_vs_train_val_loss.png",
    )
    _scatter_with_labels(
        xs=[float(r["trainable_params"]) for r in records],
        ys=[float(r["cst_rmse"]) for r in records],
        labels=labels,
        title="Model size vs error",
        xlabel="Trainable parameters",
        ylabel="CST RMSE",
        out_path=fig_dir / "benchmark_testcsv_error_vs_params.png",
    )
    _line_losses_per_category(
        records,
        metric_key="cst_rmse",
        out_path=fig_dir / "benchmark_testcsv_cst_rmse_by_category_line.png",
    )
    _line_losses_per_category(
        records,
        metric_key="final_train_val_loss",
        out_path=fig_dir / "benchmark_testcsv_final_train_val_loss_by_category_line.png",
    )

    print(f"Saved benchmark artifacts: {out_json}, {out_json_full}, {out_csv}")
    print(f"Saved benchmark figures under {fig_dir}")


if __name__ == "__main__":
    main()
