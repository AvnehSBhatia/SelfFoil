#!/usr/bin/env python3
"""Figures: overview + one CST bar chart per ablation category (matplotlib only)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_suite.catalog import VARIANT_CATALOG
from core.figures_path import figures_dir


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def last_per_run(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        m = str(r["model"])
        out[m] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite-root", type=Path, default=ROOT / "ablation_suite")
    args = p.parse_args()

    fig_dir = figures_dir()
    logs = args.suite_root / "logs"
    metrics_path = logs / "metrics.jsonl"
    eff_path = logs / "efficiency.jsonl"

    rows = read_jsonl(metrics_path)
    latest = last_per_run(rows)
    order = sorted(latest.keys())
    if "baseline/full" in order and "baseline/base" in order:
        order = [k for k in order if k != "baseline/base"]

    # --- Overview bar ---
    fig0, ax0 = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(order))
    ys = [latest[k]["cst_error"] for k in order]
    ax0.bar(xs, ys, tick_label=[k.replace("/", "\n") for k in order])
    ax0.set_ylabel("CST RMSE (validation)")
    ax0.set_title("WHISP ablations (all runs)")
    ax0.tick_params(axis="x", labelsize=6)
    fig0.tight_layout()
    fig0.savefig(fig_dir / "ablation_suite_barplot_overview.png", dpi=300)
    plt.close(fig0)

    # --- Per-category bars ---
    by_cat: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for k, r in latest.items():
        cat = str(r.get("category") or k.split("/")[0])
        slug = str(r.get("slug") or (k.split("/", 1)[1] if "/" in k else k))
        by_cat[cat].append((slug, float(r["cst_error"])))
    for cat, items in sorted(by_cat.items()):
        items.sort(key=lambda x: x[0])
        fig, ax = plt.subplots(figsize=(7, 4.2))
        sx = np.arange(len(items))
        ax.bar(sx, [v for _, v in items], tick_label=[s for s, _ in items])
        ax.set_ylabel("CST RMSE (validation)")
        ax.set_title(f"WHISP — {cat}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        safe = cat.replace("/", "_")
        fig.savefig(fig_dir / f"ablation_suite_category_{safe}_bar.png", dpi=300)
        plt.close(fig)

    # --- Convergence (+ optional efficiency) ---
    eff = read_jsonl(eff_path)
    if eff:
        fig2, (ax2, ax2b) = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        ax2b = None
    for k in order:
        parts = k.split("/")
        if len(parts) != 2:
            continue
        hist_path = args.suite_root / "runs" / parts[0] / parts[1] / "history.json"
        if not hist_path.is_file():
            continue
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)
        ep = [h["epoch"] for h in hist]
        loss = [h["val_loss"] for h in hist]
        ax2.plot(ep, loss, label=k.replace("/", "/"))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss")
    ax2.set_title("Convergence")
    ax2.legend(loc="best", fontsize=6)
    if ax2b is not None:
        fracs = sorted({float(r["frac_train"]) for r in eff})
        errs = []
        for fr in fracs:
            block = [r for r in eff if float(r.get("frac_train", -1)) == fr]
            errs.append(block[-1]["cst_error"] if block else float("nan"))
        ax2b.plot([100 * f for f in fracs], errs, marker="o")
        ax2b.set_xlabel("Training fraction (% of airfoils)")
        ax2b.set_ylabel("CST RMSE")
        ax2b.set_title("Data efficiency")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "ablation_suite_convergence_curves.png", dpi=300)
    plt.close(fig2)

    # --- Error vs size ---
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    px = [latest[k].get("trainable_params", 0) for k in order]
    py = [latest[k]["cst_error"] for k in order]
    ax3.scatter(px, py)
    for i, k in enumerate(order):
        ax3.annotate(k.split("/")[-1], (px[i], py[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax3.set_xlabel("Trainable parameters")
    ax3.set_ylabel("CST RMSE")
    ax3.set_title("Error vs model size")
    fig3.tight_layout()
    fig3.savefig(fig_dir / "ablation_suite_error_vs_model_size.png", dpi=300)
    plt.close(fig3)

    # --- Metric landscape (2D "cluster" view of latest runs) ---
    xs_c: list[float] = []
    ys_cl: list[float] = []
    labels: list[str] = []
    for k in order:
        r = latest[k]
        xs_c.append(float(r["cst_error"]))
        ce = r.get("cl_error")
        ys_cl.append(float(ce) if ce is not None else 0.0)
        labels.append(k.split("/")[-1])
    if xs_c:
        fig4, ax4 = plt.subplots(figsize=(7, 5.5))
        ax4.scatter(xs_c, ys_cl, s=42, alpha=0.75, c=np.arange(len(xs_c)), cmap="tab20")
        for i, lab in enumerate(labels):
            ax4.annotate(lab, (xs_c[i], ys_cl[i]), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax4.set_xlabel("CST RMSE (validation)")
        ax4.set_ylabel("Cl MSE (or 0 if n/a)")
        ax4.set_title("Ablation metric landscape (latest eval)")
        fig4.tight_layout()
        fig4.savefig(fig_dir / "ablation_suite_metric_landscape.png", dpi=300)
        plt.close(fig4)

    # --- Manifest (counts) ---
    with open(fig_dir / "ablation_suite_manifest.txt", "w", encoding="utf-8") as f:
        f.write(f"catalog_runs={len(VARIANT_CATALOG)} evaluated={len(latest)}\n")

    print(f"Wrote figures under {fig_dir}")


if __name__ == "__main__":
    main()
