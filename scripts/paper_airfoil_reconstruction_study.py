#!/usr/bin/env python3
"""Paper-focused inverse reconstruction study on named airfoils.

Given named airfoils (e.g. S1223, NACA0012), generate reference polars with NeuralFoil,
run WHISP inverse inference from (Cl, Cd, Re, Mach, AoA), and quantify:
- geometry reconstruction error (pointwise and CST-space),
- direct lift-head fidelity,
- aerodynamic consistency of reconstructed geometry via NeuralFoil.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import aerosandbox as asb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_suite.models.whisp_ablated import WHISPAblated
from core.cst_kulfan import CSTDecoder18, fit_cst18_from_xy_batched
from core.figures_path import figures_dir


def _default_checkpoint() -> Path:
    candidates = sorted((ROOT / "ablation_suite" / "runs").glob("*/*/model.pt"))
    if not candidates:
        raise FileNotFoundError("No checkpoints found under ablation_suite/runs/*/*/model.pt")
    # Pick baseline/full if available, else first checkpoint.
    preferred = ROOT / "ablation_suite" / "runs" / "baseline" / "full" / "model.pt"
    return preferred if preferred.is_file() else candidates[0]


def _discover_checkpoints(suite_root: Path) -> list[Path]:
    return sorted((suite_root / "runs").glob("*/*/model.pt"))


def _resolve_encoder_ckpts(meta: dict[str, Any], root: Path) -> tuple[Path, ...]:
    out: list[Path] = []
    for raw in meta["encoder_ckpts"]:
        p = Path(str(raw))
        if p.is_file():
            out.append(p)
            continue
        fallback = root / "models" / p.name
        if fallback.is_file():
            out.append(fallback)
            continue
        raise FileNotFoundError(f"Missing encoder checkpoint {p} (fallback missing: {fallback}).")
    return tuple(out)


def _load_decoder_first_branch(meta_path: Path) -> str:
    if not meta_path.is_file():
        return "upper"
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict):
            branch = str(meta.get("first_branch", "upper"))
            if branch in ("upper", "lower"):
                return branch
    except Exception:
        pass
    return "upper"


def _load_x_coords_from_cache(cache_path: Path, device: torch.device) -> torch.Tensor | None:
    if not cache_path.is_file():
        return None
    try:
        bundle = torch.load(cache_path, map_location="cpu", weights_only=False)
        x = bundle.get("x_coords")
        if isinstance(x, torch.Tensor) and x.ndim == 1 and x.numel() > 8:
            return x.to(device=device, dtype=torch.float32)
    except Exception:
        return None
    return None


def _build_x_coords(n_points_per_side: int, device: torch.device) -> torch.Tensor:
    # TE(1)->LE(0)->TE(1) cosine-spaced synthetic fallback grid.
    theta = torch.linspace(0.0, math.pi, n_points_per_side, device=device)
    x_first = 0.5 * (1.0 + torch.cos(theta))  # 1 -> 0
    x_second = x_first.flip(0)[1:]  # 0 -> 1 (skip duplicate LE)
    return torch.cat([x_first, x_second], dim=0)


def _load_model(ckpt_path: Path, device: torch.device) -> WHISPAblated:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = blob["meta"]
    spec = meta["spec"]
    encoder_ckpts = _resolve_encoder_ckpts(meta, ROOT)
    model = WHISPAblated(encoder_ckpts, spec).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def _try_neuralfoil_import() -> Any:
    try:
        import neuralfoil as nf  # type: ignore

        return nf
    except Exception as e:
        raise SystemExit(
            "NeuralFoil import failed. Install it in the active environment first.\n" f"Import error: {e}"
        )


def _run_nf_from_coordinates(
    nf: Any,
    xy: np.ndarray,
    re: float,
    mach: float,
    alphas_deg: np.ndarray,
    model_size: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Compatibility wrapper across NeuralFoil versions.
    if hasattr(nf, "get_aero_from_coordinates"):
        fn = nf.get_aero_from_coordinates  # type: ignore[attr-defined]
        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {
            "coordinates": xy,
            "alpha": alphas_deg,
            "Re": re,
            "model_size": model_size,
        }
        if "mach" in sig.parameters:
            kwargs["mach"] = mach
        out = fn(**kwargs)
        return np.asarray(out["CL"]), np.asarray(out["CD"])
    raise RuntimeError("This environment does not expose neuralfoil.get_aero_from_coordinates.")


def _run_nf_from_airfoil(
    nf: Any,
    airfoil_name: str,
    re: float,
    alphas_deg: np.ndarray,
    model_size: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # NeuralFoil's get_aero_from_airfoil does not use Mach in this signature.
    af = asb.Airfoil(airfoil_name)
    out = nf.get_aero_from_airfoil(airfoil=af, alpha=alphas_deg, Re=re, model_size=model_size)  # type: ignore[attr-defined]
    return np.asarray(out["CL"]), np.asarray(out["CD"]), np.asarray(af.coordinates)


def _point_errors(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch for error computation: {pred.shape} vs {true.shape}")
    finite = np.isfinite(pred) & np.isfinite(true)
    if not np.any(finite):
        return float("nan"), float("nan")
    diff = pred[finite] - true[finite]
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def _plot_airfoil_polar_case(
    out_path: Path,
    airfoil_name: str,
    alphas: np.ndarray,
    cl_true: np.ndarray,
    cl_recon: np.ndarray,
    cd_true: np.ndarray,
    cd_recon: np.ndarray,
) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax0.plot(alphas, cl_true, lw=2.0, label="Reference (NeuralFoil)")
    ax0.plot(alphas, cl_recon, lw=1.6, marker="o", ms=3.2, label="Reconstructed geometry")
    ax0.set_title(f"{airfoil_name} lift curve")
    ax0.set_xlabel("AoA (deg)")
    ax0.set_ylabel("Cl")
    ax0.grid(alpha=0.25, linestyle="--")
    ax0.legend(fontsize=8)

    ax1.plot(cl_true, cd_true, lw=2.0, label="Reference polar")
    ax1.plot(cl_recon, cd_recon, lw=1.6, marker="o", ms=3.2, label="Reconstructed polar")
    ax1.set_title(f"{airfoil_name} drag polar")
    ax1.set_xlabel("Cl")
    ax1.set_ylabel("Cd")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def _plot_geometry_overlay(out_path: Path, airfoil_name: str, xy_true: np.ndarray, xy_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.plot(xy_true[:, 0], xy_true[:, 1], lw=2.0, label="Reference geometry")
    ax.plot(xy_pred[:, 0], xy_pred[:, 1], lw=1.7, linestyle="--", label="Reconstructed (mean over AoA)")
    ax.set_title(f"{airfoil_name} geometry overlay")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


def _plot_summary_bars(out_path: Path, rows: list[dict[str, Any]]) -> None:
    names = [str(r["airfoil"]) for r in rows]
    geom_mae = [float(r["geometry_xy_mae"]) for r in rows]
    cl_rmse = [float(r["cl_recon_rmse"]) for r in rows]
    cd_rmse = [float(r["cd_recon_rmse"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
    metrics = [
        ("Geometry MAE (x,y)", geom_mae, "#4C72B0"),
        ("Cl RMSE", cl_rmse, "#55A868"),
        ("Cd RMSE", cd_rmse, "#C44E52"),
    ]
    for ax, (title, vals, color) in zip(axes, metrics):
        idx = np.arange(len(names))
        ax.bar(idx, vals, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(idx)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.suptitle("Named-airfoil inverse reconstruction accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_metric_heatmap(
    out_path: Path,
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
) -> None:
    models = sorted({str(r["model"]) for r in rows})
    airfoils = sorted({str(r["airfoil"]) for r in rows})
    mat = np.full((len(models), len(airfoils)), np.nan, dtype=np.float64)
    idx_m = {m: i for i, m in enumerate(models)}
    idx_a = {a: i for i, a in enumerate(airfoils)}
    for r in rows:
        m = str(r["model"])
        a = str(r["airfoil"])
        mat[idx_m[m], idx_a[a]] = float(r.get(metric_key, float("nan")))

    fig, ax = plt.subplots(figsize=(1.2 * len(airfoils) + 4.5, 0.35 * len(models) + 4.0))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(airfoils)))
    ax.set_xticklabels(airfoils, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(metric_key, rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _plot_model_mean_rankings(out_path: Path, rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, list[float]] = {}
    for r in rows:
        score = float(r["geometry_xy_mae"]) + 0.5 * float(r["cl_recon_rmse"])
        by_model.setdefault(str(r["model"]), []).append(score)
    models = sorted(by_model.keys(), key=lambda m: float(np.mean(by_model[m])))
    means = [float(np.mean(by_model[m])) for m in models]

    fig, ax = plt.subplots(figsize=(10.5, 0.35 * len(models) + 2.8))
    y = np.arange(len(models))
    ax.barh(y, means, color="#4C72B0", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean composite score = geometry_mae + 0.5 * cl_rmse")
    ax.set_title("All-model ranking across named airfoils")
    fig.tight_layout()
    fig.savefig(out_path, dpi=280)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--all-models", action="store_true", help="Run all checkpoints under ablation_suite/runs/*/*/model.pt")
    p.add_argument("--suite-root", type=Path, default=ROOT / "ablation_suite")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument(
        "--airfoils",
        nargs="+",
        default=["naca0012", "s1223", "naca2412", "naca4412", "e216", "clarky"],
        help="Named airfoils available through AeroSandbox/NeuralFoil.",
    )
    p.add_argument("--re", type=float, default=1_000_000.0)
    p.add_argument("--mach", type=float, default=0.10)
    p.add_argument("--alpha-start", type=float, default=-10.0)
    p.add_argument("--alpha-end", type=float, default=20.0)
    p.add_argument("--alpha-step", type=float, default=1.0)
    p.add_argument("--n-points-per-side", type=int, default=121)
    p.add_argument("--x-grid-cache", type=Path, default=ROOT / "models" / "original_tensors.pt")
    p.add_argument("--decoder-meta", type=Path, default=ROOT / "models" / "decoder_coords.pt")
    p.add_argument("--neuralfoil-model-size", default="xxxlarge")
    p.add_argument("--tag", default="paper_airfoil_study")
    p.add_argument(
        "--save-per-case-plots",
        action="store_true",
        help="Save per-(model,airfoil) polar and geometry plots (many files).",
    )
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoints: list[Path]
    if args.all_models:
        checkpoints = _discover_checkpoints(args.suite_root)
        if not checkpoints:
            raise SystemExit(f"No checkpoints found under {args.suite_root / 'runs'}.")
    else:
        checkpoints = [args.checkpoint or _default_checkpoint()]
    nf = _try_neuralfoil_import()

    x_1d = _load_x_coords_from_cache(args.x_grid_cache, device=device)
    if x_1d is None:
        x_1d = _build_x_coords(args.n_points_per_side, device=device)
    first_branch = _load_decoder_first_branch(args.decoder_meta)
    decoder = CSTDecoder18(first_branch=first_branch).to(device)
    x_coords = x_1d.unsqueeze(0)

    alphas = np.arange(args.alpha_start, args.alpha_end + 1e-9, args.alpha_step, dtype=np.float64)
    fig_dir = figures_dir()
    safe_tag = args.tag.replace("/", "_")
    out_json = fig_dir / f"{safe_tag}_summary.json"
    out_csv = fig_dir / f"{safe_tag}_summary.csv"

    rows: list[dict[str, Any]] = []
    ref_cache: dict[str, dict[str, Any]] = {}
    for airfoil_name in args.airfoils:
        cl_ref, cd_ref, xy_true = _run_nf_from_airfoil(
            nf=nf,
            airfoil_name=airfoil_name,
            re=args.re,
            alphas_deg=alphas,
            model_size=args.neuralfoil_model_size,
        )
        # Fit target CST from the reference geometry.
        xy_true_t = torch.tensor(xy_true, dtype=torch.float32).unsqueeze(0)
        cst_true = fit_cst18_from_xy_batched(xy_true_t).squeeze(0).numpy()
        # Resample reference geometry once.
        x_ref_n = x_1d.numel()
        xy_true_resampled = asb.Airfoil(airfoil_name).repanel(n_points_per_side=(x_ref_n + 1) // 2).coordinates
        ref_cache[airfoil_name] = {
            "cl_ref": cl_ref,
            "cd_ref": cd_ref,
            "xy_true": xy_true,
            "xy_true_resampled": xy_true_resampled,
            "cst_true": cst_true,
        }

    for checkpoint in checkpoints:
        model = _load_model(checkpoint, device=device)
        model_id = f"{checkpoint.parent.parent.name}/{checkpoint.parent.name}"
        for airfoil_name in args.airfoils:
            ref = ref_cache[airfoil_name]
            cl_ref = np.asarray(ref["cl_ref"])
            cd_ref = np.asarray(ref["cd_ref"])
            cst_true = np.asarray(ref["cst_true"])
            xy_true_resampled = np.asarray(ref["xy_true_resampled"])
            cl_direct_pred: list[float] = []
            cst_pred_all: list[np.ndarray] = []
            xy_pred_all: list[np.ndarray] = []
            cl_recon: list[float] = []
            cd_recon: list[float] = []

            for i, alpha in enumerate(alphas):
                cl_in = float(cl_ref[i])
                cd_in = float(cd_ref[i])
                cl_t = torch.tensor([cl_in], dtype=torch.float32, device=device)
                cd_t = torch.tensor([cd_in], dtype=torch.float32, device=device)
                re_log_t = torch.tensor([math.log10(args.re)], dtype=torch.float32, device=device)
                mach_t = torch.tensor([args.mach], dtype=torch.float32, device=device)
                aoa_t = torch.tensor([float(alpha)], dtype=torch.float32, device=device)

                cst_pred, aux = model(cl_t, cd_t, re_log_t, mach_t, aoa_t)
                cst_v = cst_pred[0].detach().cpu().numpy()
                cst_pred_all.append(cst_v)
                cl_direct_pred.append(float(aux["cl_direct"][0].detach().cpu().item()))

                xy_flat = decoder(cst_pred, x_coords)[0]
                xy_pred = xy_flat.view(-1, 2).detach().cpu().numpy()
                xy_pred_all.append(xy_pred)

                cl_r, cd_r = _run_nf_from_coordinates(
                    nf=nf,
                    xy=xy_pred,
                    re=args.re,
                    mach=args.mach,
                    alphas_deg=np.array([alpha], dtype=np.float64),
                    model_size=args.neuralfoil_model_size,
                )
                cl_recon.append(float(cl_r[0]))
                cd_recon.append(float(cd_r[0]))

            cst_pred_np = np.stack(cst_pred_all, axis=0)
            cst_true_rep = np.repeat(cst_true[None, :], cst_pred_np.shape[0], axis=0)
            cst_rmse, cst_mae = _point_errors(cst_pred_np, cst_true_rep)
            cl_direct_rmse, cl_direct_mae = _point_errors(np.array(cl_direct_pred), cl_ref)
            cl_recon_rmse, cl_recon_mae = _point_errors(np.array(cl_recon), cl_ref)
            cd_recon_rmse, cd_recon_mae = _point_errors(np.array(cd_recon), cd_ref)

            # Geometry pointwise error against reference geometry using mean reconstructed geometry.
            xy_pred_mean = np.mean(np.stack(xy_pred_all, axis=0), axis=0)
            n_comp = min(xy_true_resampled.shape[0], xy_pred_mean.shape[0])
            geom_rmse, geom_mae = _point_errors(xy_pred_mean[:n_comp], xy_true_resampled[:n_comp])

            case = {
                "model": model_id,
                "checkpoint": str(checkpoint),
                "airfoil": airfoil_name,
                "re": args.re,
                "mach": args.mach,
                "alphas_deg": alphas.tolist(),
                "cst_rmse_vs_reference": cst_rmse,
                "cst_mae_vs_reference": cst_mae,
                "cl_direct_rmse": cl_direct_rmse,
                "cl_direct_mae": cl_direct_mae,
                "cl_recon_rmse": cl_recon_rmse,
                "cl_recon_mae": cl_recon_mae,
                "cd_recon_rmse": cd_recon_rmse,
                "cd_recon_mae": cd_recon_mae,
                "geometry_xy_rmse": geom_rmse,
                "geometry_xy_mae": geom_mae,
            }
            rows.append(case)

            if args.save_per_case_plots:
                safe_model = model_id.replace("/", "__")
                _plot_airfoil_polar_case(
                    fig_dir / f"{safe_tag}_{safe_model}_{airfoil_name}_polar_comparison.png",
                    airfoil_name=f"{airfoil_name} [{model_id}]",
                    alphas=alphas,
                    cl_true=cl_ref,
                    cl_recon=np.array(cl_recon),
                    cd_true=cd_ref,
                    cd_recon=np.array(cd_recon),
                )
                _plot_geometry_overlay(
                    fig_dir / f"{safe_tag}_{safe_model}_{airfoil_name}_geometry_overlay.png",
                    airfoil_name=f"{airfoil_name} [{model_id}]",
                    xy_true=xy_true_resampled[:n_comp],
                    xy_pred=xy_pred_mean[:n_comp],
                )

            print(
                json.dumps(
                    {
                        "model": model_id,
                        "airfoil": airfoil_name,
                        "geometry_xy_mae": round(geom_mae, 6),
                        "cl_recon_rmse": round(cl_recon_rmse, 6),
                        "cd_recon_rmse": round(cd_recon_rmse, 6),
                    }
                )
            )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "all_models": bool(args.all_models),
                "num_models": len(checkpoints),
                "device": str(device),
                "neuralfoil_model_size": args.neuralfoil_model_size,
                "rows": rows,
            },
            f,
            indent=2,
        )
    if rows:
        headers = [
            "model",
            "airfoil",
            "cst_rmse_vs_reference",
            "cst_mae_vs_reference",
            "cl_direct_rmse",
            "cl_direct_mae",
            "cl_recon_rmse",
            "cl_recon_mae",
            "cd_recon_rmse",
            "cd_recon_mae",
            "geometry_xy_rmse",
            "geometry_xy_mae",
            "re",
            "mach",
            "checkpoint",
        ]
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")

    # Aggregate by airfoil across all models for quick paper digest.
    by_airfoil: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_airfoil.setdefault(str(r["airfoil"]), []).append(r)
    agg_rows: list[dict[str, Any]] = []
    for airfoil_name in sorted(by_airfoil.keys()):
        block = by_airfoil[airfoil_name]
        agg_rows.append(
            {
                "airfoil": airfoil_name,
                "geometry_xy_mae": float(np.nanmean([float(x["geometry_xy_mae"]) for x in block])),
                "cl_recon_rmse": float(np.nanmean([float(x["cl_recon_rmse"]) for x in block])),
                "cd_recon_rmse": float(np.nanmean([float(x["cd_recon_rmse"]) for x in block])),
            }
        )
    _plot_summary_bars(fig_dir / f"{safe_tag}_summary_bars.png", agg_rows)
    _plot_metric_heatmap(
        fig_dir / f"{safe_tag}_heatmap_cl_recon_rmse.png",
        rows,
        metric_key="cl_recon_rmse",
        title="Cl reconstruction RMSE across models and airfoils",
    )
    _plot_metric_heatmap(
        fig_dir / f"{safe_tag}_heatmap_geometry_mae.png",
        rows,
        metric_key="geometry_xy_mae",
        title="Geometry MAE across models and airfoils",
    )
    _plot_model_mean_rankings(fig_dir / f"{safe_tag}_model_ranking.png", rows)
    print(f"Saved paper study summary JSON: {out_json}")
    print(f"Saved paper study summary CSV: {out_csv}")
    print(f"Saved paper study figures under {fig_dir}")


if __name__ == "__main__":
    main()

