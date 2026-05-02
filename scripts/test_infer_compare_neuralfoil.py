#!/usr/bin/env python3
"""Interactive test utility: infer airfoil from (Cl, Cd, Re, Mach, AoA), compare with NeuralFoil polars.

Loads checkpoints via ``meta["arch"]``:

- ``cst_struct32``: ``CstStruct32`` — uses **Cl, Cd, AoA, log10(Re)** only; **Mach is ignored** for inference
  (still pass ``--mach`` for NeuralFoil polars on the decoded geometry).
- ``cst_mlp``: five-input MLP to CST18.
- Otherwise: ``WHISPAblated`` (requires ``meta["spec"]`` + encoder ``.pt`` files).

Small CST heads omit ``aux["cl_direct"]``; WHISP ablations include it when present.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_suite.models.whisp_ablated import WHISPAblated
from core.cst_kulfan import CSTDecoder18
from core.figures_path import figures_dir
from core.whisp_net import CstMLP, CstStruct32


def _default_checkpoint() -> Path:
    runs = sorted((ROOT / "ablation_suite" / "runs").glob("*/*/model.pt"))
    if runs:
        return runs[0]
    for path in (ROOT / "models" / "cst_struct32.pt", ROOT / "models" / "cst_mlp.pt"):
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No checkpoint: install ablation_suite/runs/*/*/model.pt or models/cst_struct32.pt / cst_mlp.pt "
        "or pass --checkpoint explicitly."
    )


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
        raise FileNotFoundError(
            f"Missing encoder checkpoint {p} (fallback missing: {fallback})."
        )
    return tuple(out)


def _build_x_coords(n_points_per_side: int, device: torch.device) -> torch.Tensor:
    # TE(1)->LE(0)->TE(1) synthetic fallback grid.
    theta = torch.linspace(0.0, math.pi, n_points_per_side, device=device)
    x_first = 0.5 * (1.0 + torch.cos(theta))  # 1 -> 0
    x_second = x_first.flip(0)[1:]  # 0 -> 1, skip duplicate LE point
    return torch.cat([x_first, x_second], dim=0)


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


def _load_decoder_first_branch(meta_path: Path) -> str:
    if not meta_path.is_file():
        return "upper"
    try:
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict):
            fb = str(meta.get("first_branch", "upper"))
            if fb in ("upper", "lower"):
                return fb
    except Exception:
        pass
    return "upper"


def _load_inverse_checkpoint(
    ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, str, str]:
    """Returns ``(model, label_for_plots, arch_slug)`` where ``arch_slug`` is normalized ``meta['arch']``."""
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = blob["meta"]
    arch_raw = str(meta.get("arch", "")).lower().replace("-", "_")
    if arch_raw == "cst_mlp":
        model = CstMLP().to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        try:
            rid = str(ckpt_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        except ValueError:
            rid = ckpt_path.name
        return model, rid, "cst_mlp"
    if arch_raw == "cst_struct32":
        model = CstStruct32().to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        try:
            rid = str(ckpt_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        except ValueError:
            rid = ckpt_path.name
        return model, rid, "cst_struct32"
    spec = meta["spec"]
    encoder_ckpts = _resolve_encoder_ckpts(meta, ROOT)
    model = WHISPAblated(encoder_ckpts, spec).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    model_id = f"{ckpt_path.parent.parent.name}/{ckpt_path.parent.name}"
    arch_slug = arch_raw if arch_raw else "whisp_ablated"
    return model, model_id, arch_slug


def _try_neuralfoil_import() -> Any:
    try:
        import neuralfoil as nf  # type: ignore

        return nf
    except Exception as e:
        raise SystemExit(
            "NeuralFoil is not installed or import failed. Install it in your active env first.\n"
            f"Import error: {e}"
        )


def _run_neuralfoil_polars(
    nf: Any,
    xy: np.ndarray,
    re: float,
    mach: float,
    alphas_deg: np.ndarray,
    model_size: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Try a few known API shapes for portability across NeuralFoil versions.
    if hasattr(nf, "get_aero_from_coordinates"):
        fn = nf.get_aero_from_coordinates  # type: ignore[attr-defined]
        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {
            "coordinates": xy,
            "alpha": alphas_deg,
            "Re": re,
            "model_size": model_size,
        }
        # Some NeuralFoil builds don't expose Mach in this API.
        if "mach" in sig.parameters:
            kwargs["mach"] = mach
        else:
            print("[warn] NeuralFoil API does not expose `mach`; running without Mach correction.")
        out = fn(**kwargs)
        return np.asarray(out["CL"]), np.asarray(out["CD"])

    if hasattr(nf, "get_aero"):
        out = nf.get_aero(  # type: ignore[attr-defined]
            coordinates=xy,
            alpha=alphas_deg,
            Re=re,
            mach=mach,
            model_size=model_size,
        )
        return np.asarray(out["CL"]), np.asarray(out["CD"])

    raise RuntimeError(
        "Unsupported NeuralFoil API in this environment. "
        "Expected `get_aero_from_coordinates` or `get_aero`."
    )


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="WHISP ablation model.pt, or models/cst_struct32.pt / cst_mlp.pt (default: first ablation run, else models/cst_struct32.pt)",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--cl", type=float, required=True)
    p.add_argument("--cd", type=float, required=True)
    p.add_argument("--re", type=float, required=True)
    p.add_argument(
        "--mach",
        type=float,
        required=True,
        help="Mach number (always used for NeuralFoil polars; ignored by CstStruct32 inverse pass only).",
    )
    p.add_argument("--aoa", type=float, required=True, help="AoA in degrees")
    p.add_argument("--alpha-start", type=float, default=-5.0)
    p.add_argument("--alpha-end", type=float, default=15.0)
    p.add_argument("--alpha-step", type=float, default=0.5)
    p.add_argument("--n-points-per-side", type=int, default=101)
    p.add_argument("--x-grid-cache", type=Path, default=ROOT / "models" / "original_tensors.pt")
    p.add_argument("--decoder-meta", type=Path, default=ROOT / "models" / "decoder_coords.pt")
    p.add_argument("--neuralfoil-model-size", default="xxxlarge")
    p.add_argument("--tag", default="manual_case")
    args = p.parse_args()

    if args.re <= 0:
        raise SystemExit("--re must be positive.")
    if args.alpha_step <= 0:
        raise SystemExit("--alpha-step must be > 0.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = args.checkpoint or _default_checkpoint()
    model, model_label, arch_slug = _load_inverse_checkpoint(checkpoint, device=device)

    cl_t = torch.tensor([args.cl], dtype=torch.float32, device=device)
    cd_t = torch.tensor([args.cd], dtype=torch.float32, device=device)
    re_log_t = torch.tensor([math.log10(args.re)], dtype=torch.float32, device=device)
    mach_t = torch.tensor([args.mach], dtype=torch.float32, device=device)
    aoa_t = torch.tensor([args.aoa], dtype=torch.float32, device=device)

    cst_pred, aux = model(cl_t, cd_t, re_log_t, mach_t, aoa_t)
    cl_direct: float | None = None
    if isinstance(aux, dict) and "cl_direct" in aux and aux["cl_direct"].numel():
        cl_direct = float(aux["cl_direct"][0].detach().cpu().item())
    cst = cst_pred[0]

    x_1d = _load_x_coords_from_cache(args.x_grid_cache, device=device)
    if x_1d is None:
        x_1d = _build_x_coords(args.n_points_per_side, device=device)
    first_branch = _load_decoder_first_branch(args.decoder_meta)
    x_coords = x_1d.unsqueeze(0)
    decoder = CSTDecoder18(first_branch=first_branch).to(device)
    xy_flat = decoder(cst.unsqueeze(0), x_coords)[0]
    xy = xy_flat.view(-1, 2).detach().cpu().numpy()

    alphas = np.arange(args.alpha_start, args.alpha_end + 1e-9, args.alpha_step, dtype=np.float64)
    nf = _try_neuralfoil_import()
    cl_nf, cd_nf = _run_neuralfoil_polars(
        nf=nf,
        xy=xy,
        re=args.re,
        mach=args.mach,
        alphas_deg=alphas,
        model_size=args.neuralfoil_model_size,
    )

    fig_dir = figures_dir()
    safe_tag = args.tag.replace("/", "_")
    out_json = fig_dir / f"test_infer_neuralfoil_{safe_tag}.json"
    out_geom = fig_dir / f"test_infer_neuralfoil_{safe_tag}_airfoil_xy.csv"
    out_png = fig_dir / f"test_infer_neuralfoil_{safe_tag}_polars.png"

    with out_geom.open("w", encoding="utf-8") as f:
        f.write("x,y\n")
        for x, y in xy:
            f.write(f"{x},{y}\n")

    payload = {
        "input": {
            "cl": args.cl,
            "cd": args.cd,
            "re": args.re,
            "mach": args.mach,
            "aoa_deg": args.aoa,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_arch": arch_slug,
        "model_label": model_label,
        "device": str(device),
        "neuralfoil_model_size": args.neuralfoil_model_size,
        "whisp_outputs_at_input": {
            **({"cl_direct": cl_direct} if cl_direct is not None else {}),
            "cst18": cst.detach().cpu().tolist(),
        },
        "polar_comparison": {
            "alphas_deg": alphas.tolist(),
            "cl_neuralfoil": cl_nf.tolist(),
            "cd_neuralfoil": cd_nf.tolist(),
        },
        "outputs": {
            "airfoil_xy_csv": str(out_geom),
            "polar_plot_png": str(out_png),
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle(f"test_infer_compare_neuralfoil — {model_label} ({arch_slug})", fontsize=10, y=1.02)
    ax0.plot(xy[:, 0], xy[:, 1], lw=1.4)
    ax0.set_title("Reconstructed airfoil")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    ax1.plot(alphas, cl_nf, lw=1.5, label="NeuralFoil Cl(alpha)")
    ax1.scatter([args.aoa], [args.cl], s=38, c="C3", marker="x", label="Input target")
    if cl_direct is not None:
        ax1.scatter([args.aoa], [cl_direct], s=30, c="C2", marker="o", label="Model cl_direct")
    ax1.set_title("Lift polar")
    ax1.set_xlabel("AoA (deg)")
    ax1.set_ylabel("Cl")
    ax1.legend(fontsize=8)

    ax2.plot(cl_nf, cd_nf, lw=1.5, label="NeuralFoil polar")
    ax2.scatter([args.cl], [args.cd], s=38, c="C3", marker="x", label="Input target")
    ax2.set_title("Drag polar")
    ax2.set_xlabel("Cl")
    ax2.set_ylabel("Cd")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)

    print(f"checkpoint_arch={arch_slug} model_label={model_label}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved geometry CSV: {out_geom}")
    print(f"Saved polar figure: {out_png}")


if __name__ == "__main__":
    main()
