#!/usr/bin/env python3
"""Figures: overview + one CST bar chart per ablation category (matplotlib only)."""

from __future__ import annotations

import argparse
import json
import math
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


def _safe_float(v: object, fallback: float = float("nan")) -> float:
    try:
        if v is None:
            return fallback
        return float(v)
    except Exception:
        return fallback


def _rank(values: list[float], reverse: bool = False) -> list[float]:
    valid = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    valid_sorted = sorted(valid, key=lambda x: x[1], reverse=reverse)
    out = [float("nan")] * len(values)
    for r, (i, _) in enumerate(valid_sorted, start=1):
        out[i] = float(r)
    return out


def _composite_paper_index(rows: list[dict], keys: list[str]) -> list[float]:
    collected = {k: [_safe_float(r.get(k)) for r in rows] for k in keys}
    rank_sum = [0.0] * len(rows)
    for k in keys:
        ranks = _rank(collected[k], reverse=False)
        for i, rv in enumerate(ranks):
            if math.isfinite(rv):
                rank_sum[i] += rv
    return rank_sum


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

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Overview ranked bar (paper-ready) ---
    ranked = sorted(order, key=lambda k: _safe_float(latest[k].get("cst_rmse", latest[k].get("cst_error"))))
    fig0, ax0 = plt.subplots(figsize=(10.8, 6.6))
    ys = [_safe_float(latest[k].get("cst_rmse", latest[k].get("cst_error"))) for k in ranked]
    labels = [k.replace("/", "\n") for k in ranked]
    bars = ax0.barh(np.arange(len(ranked)), ys, color="#4C72B0", alpha=0.9)
    ax0.set_yticks(np.arange(len(ranked)))
    ax0.set_yticklabels(labels, fontsize=8)
    ax0.invert_yaxis()
    ax0.set_xlabel("CST RMSE (validation, lower is better)")
    ax0.set_title("Ablation ranking by shape accuracy")
    for i, b in enumerate(bars):
        ax0.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2, f"{ys[i]:.4f}", va="center", fontsize=7)
    fig0.tight_layout()
    fig0.savefig(fig_dir / "ablation_suite_barplot_overview.png", dpi=340)
    plt.close(fig0)

    # --- Per-category bars ---
    by_cat: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for k, r in latest.items():
        cat = str(r.get("category") or k.split("/")[0])
        slug = str(r.get("slug") or (k.split("/", 1)[1] if "/" in k else k))
        by_cat[cat].append((slug, float(r["cst_error"])))
    for cat, items in sorted(by_cat.items()):
        items.sort(key=lambda x: x[0])
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        sx = np.arange(len(items))
        vals = [v for _, v in items]
        colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(vals)))
        ax.bar(sx, vals, tick_label=[s for s, _ in items], color=colors)
        ax.set_ylabel("CST RMSE (validation)")
        ax.set_title(f"{cat} ablation: shape accuracy")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        for i, y in enumerate(vals):
            ax.text(i, y + 0.002, f"{y:.3f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        safe = cat.replace("/", "_")
        fig.savefig(fig_dir / f"ablation_suite_category_{safe}_bar.png", dpi=340)
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
        ax2.plot(ep, loss, label=k, linewidth=1.3, alpha=0.9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss")
    ax2.set_title("Training convergence by run")
    ax2.legend(loc="best", fontsize=6)
    if ax2b is not None:
        fracs = sorted({float(r["frac_train"]) for r in eff})
        errs = []
        for fr in fracs:
            block = [r for r in eff if float(r.get("frac_train", -1)) == fr]
            errs.append(block[-1]["cst_error"] if block else float("nan"))
        ax2b.plot([100 * f for f in fracs], errs, marker="o", linewidth=1.8, color="#2A9D8F")
        ax2b.set_xlabel("Training fraction (% of airfoils)")
        ax2b.set_ylabel("CST RMSE (lower is better)")
        ax2b.set_title("Data efficiency curve")
        ax2b.grid(alpha=0.25, linestyle="--")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "ablation_suite_convergence_curves.png", dpi=340)
    plt.close(fig2)

    # --- Error vs size ---
    fig3, ax3 = plt.subplots(figsize=(8.6, 5.0))
    px = [_safe_float(latest[k].get("trainable_params", 0), 0.0) for k in order]
    py = [_safe_float(latest[k].get("cst_rmse", latest[k]["cst_error"])) for k in order]
    cat_labels = [str(latest[k].get("category", "other")) for k in order]
    cats = sorted(set(cat_labels))
    cmap = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cats)}
    for cat in cats:
        idx = [i for i, c in enumerate(cat_labels) if c == cat]
        ax3.scatter([px[i] for i in idx], [py[i] for i in idx], label=cat, s=44, alpha=0.85, color=cmap[cat])
    for i, k in enumerate(order):
        ax3.annotate(k.split("/")[-1], (px[i], py[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax3.set_xlabel("Trainable parameters")
    ax3.set_ylabel("CST RMSE (validation)")
    ax3.set_title("Model complexity vs shape error")
    ax3.legend(fontsize=7, ncol=2)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "ablation_suite_error_vs_model_size.png", dpi=340)
    plt.close(fig3)

    # --- Metric landscape (paper view: shape vs lift) ---
    xs_c: list[float] = []
    ys_cl: list[float] = []
    labels: list[str] = []
    categories: list[str] = []
    for k in order:
        r = latest[k]
        xs_c.append(_safe_float(r.get("cst_rmse", r.get("cst_error"))))
        ce = r.get("cl_rmse", r.get("cl_error"))
        ys_cl.append(_safe_float(ce))
        labels.append(k)
        categories.append(str(r.get("category", "other")))
    if xs_c:
        fig4, ax4 = plt.subplots(figsize=(8.6, 6.1))
        unique_cats = sorted(set(categories))
        cat_to_color = {c: plt.cm.tab10(i % 10) for i, c in enumerate(unique_cats)}
        for cat in unique_cats:
            idx = [i for i, c in enumerate(categories) if c == cat]
            ax4.scatter(
                [xs_c[i] for i in idx],
                [ys_cl[i] for i in idx],
                s=54,
                alpha=0.85,
                label=cat,
                color=cat_to_color[cat],
            )
        for i, lab in enumerate(labels):
            ax4.annotate(lab.split("/")[-1], (xs_c[i], ys_cl[i]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax4.set_xlabel("CST RMSE (shape fidelity)")
        ax4.set_ylabel("Cl RMSE (aerodynamic fidelity)")
        ax4.set_title("Accuracy trade-off landscape")
        ax4.legend(fontsize=7, ncol=2)
        ax4.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        fig4.tight_layout()
        fig4.savefig(fig_dir / "ablation_suite_metric_landscape.png", dpi=340)
        plt.close(fig4)

    # --- Pareto frontier (shape vs lift) ---
    pareto_candidates = []
    for k in order:
        r = latest[k]
        x = _safe_float(r.get("cst_rmse", r.get("cst_error")))
        y = _safe_float(r.get("cl_rmse", r.get("cl_error")))
        if math.isfinite(x) and math.isfinite(y):
            pareto_candidates.append((k, x, y))
    if pareto_candidates:
        pareto_sorted = sorted(pareto_candidates, key=lambda t: (t[1], t[2]))
        frontier: list[tuple[str, float, float]] = []
        best_y = float("inf")
        for row in pareto_sorted:
            if row[2] <= best_y:
                frontier.append(row)
                best_y = row[2]
        fig5, ax5 = plt.subplots(figsize=(8.3, 5.7))
        ax5.scatter([x for _, x, _ in pareto_candidates], [y for _, _, y in pareto_candidates], s=36, alpha=0.5)
        ax5.plot([x for _, x, _ in frontier], [y for _, _, y in frontier], color="#D62728", lw=2.0, marker="o")
        for k, x, y in frontier:
            ax5.annotate(k.split("/")[-1], (x, y), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax5.set_xlabel("CST RMSE")
        ax5.set_ylabel("Cl RMSE")
        ax5.set_title("Pareto frontier: shape vs lift error")
        ax5.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        fig5.tight_layout()
        fig5.savefig(fig_dir / "ablation_suite_pareto_shape_vs_lift.png", dpi=340)
        plt.close(fig5)

    # --- Category deltas vs reference ---
    by_cat_rows: dict[str, list[dict]] = defaultdict(list)
    for k in order:
        by_cat_rows[str(latest[k].get("category", k.split("/")[0]))].append(latest[k])
    delta_rows: list[tuple[str, float]] = []
    for cat, rows_cat in sorted(by_cat_rows.items()):
        refs = [r for r in rows_cat if str(r.get("slug")) == "reference"]
        if not refs:
            continue
        ref = refs[0]
        ref_err = _safe_float(ref.get("cst_rmse", ref.get("cst_error")))
        for r in rows_cat:
            slug = str(r.get("slug"))
            if slug == "reference":
                continue
            err = _safe_float(r.get("cst_rmse", r.get("cst_error")))
            delta_rows.append((f"{cat}/{slug}", err - ref_err))
    if delta_rows:
        delta_rows.sort(key=lambda x: x[1])
        fig6, ax6 = plt.subplots(figsize=(10.0, 5.8))
        vals = [d for _, d in delta_rows]
        cols = ["#2CA02C" if v < 0 else "#D62728" for v in vals]
        ax6.barh(np.arange(len(delta_rows)), vals, color=cols, alpha=0.9)
        ax6.set_yticks(np.arange(len(delta_rows)))
        ax6.set_yticklabels([k for k, _ in delta_rows], fontsize=8)
        ax6.axvline(0.0, color="k", lw=1.0)
        ax6.set_xlabel("Delta CST RMSE vs category reference (negative is better)")
        ax6.set_title("Ablation impact relative to reference")
        fig6.tight_layout()
        fig6.savefig(fig_dir / "ablation_suite_delta_vs_reference.png", dpi=340)
        plt.close(fig6)

    # --- Paper table (composite ranking) ---
    rows_for_table = [latest[k] for k in order]
    comp = _composite_paper_index(rows_for_table, ["cst_rmse", "cl_rmse", "cst_abs_p95"])
    table_rows: list[dict[str, object]] = []
    for i, k in enumerate(order):
        r = latest[k]
        table_rows.append(
            {
                "model": k,
                "category": r.get("category"),
                "trainable_params": int(_safe_float(r.get("trainable_params"), -1)),
                "cst_rmse": _safe_float(r.get("cst_rmse", r.get("cst_error"))),
                "cl_rmse": _safe_float(r.get("cl_rmse", r.get("cl_error"))),
                "cst_abs_p95": _safe_float(r.get("cst_abs_p95")),
                "paper_rank_index": comp[i],
            }
        )
    table_rows.sort(key=lambda r: _safe_float(r["paper_rank_index"]))
    with open(fig_dir / "ablation_suite_paper_metrics_table.json", "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)
    with open(fig_dir / "ablation_suite_paper_metrics_table.csv", "w", encoding="utf-8") as f:
        headers = list(table_rows[0].keys()) if table_rows else []
        if headers:
            f.write(",".join(headers) + "\n")
            for r in table_rows:
                f.write(",".join(str(r[h]) for h in headers) + "\n")

    # --- Manifest (counts + generated set) ---
    with open(fig_dir / "ablation_suite_manifest.txt", "w", encoding="utf-8") as f:
        f.write(f"catalog_runs={len(VARIANT_CATALOG)} evaluated={len(latest)}\n")
        f.write("generated=overview,category,convergence,error_vs_size,landscape,pareto,delta_vs_reference,paper_table\n")

    print(f"Wrote figures under {fig_dir}")


if __name__ == "__main__":
    main()
