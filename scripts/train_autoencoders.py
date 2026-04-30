#!/usr/bin/env python3
"""Train pair tanh autoencoders; evaluate CST reconstruction. One CSV read + tensor cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.csv_tensor_cache import load_or_build_cache
from core.figures_path import figures_dir
from core.cst_kulfan import CSTDecoder18, CSTEncoder18
from core.pair_tanh_autoencoder import PairTanhAutoencoder


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def move_bundle_to_device(bundle: dict, device: torch.device) -> dict:
    out = {}
    for k, v in bundle.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_pair_epochs(
    bundle: dict,
    device: torch.device,
    mode: str,
    epochs: int,
    lr: float,
    batch_points: int,
    out_path: Path,
) -> list[float]:
    if mode == "Cl":
        feat = bundle["cl_flat"]
    elif mode == "Cd":
        feat = bundle["cd_flat"]
    elif mode == "mach":
        feat = bundle["mach_flat"]
    else:
        feat = bundle["re_log_flat"]
    aoa = bundle["alpha_flat"]

    model = PairTanhAutoencoder(latent_dim=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    p = feat.shape[0]

    curve: list[float] = []
    for ep in range(epochs):
        total = 0.0
        n_batches = 0
        for s in range(0, p, batch_points):
            e = min(s + batch_points, p)
            x = torch.stack([feat[s:e], aoa[s:e]], dim=1)
            opt.zero_grad(set_to_none=True)
            recon, _ = model(x)
            loss = mae_loss(recon, x)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        m = total / max(n_batches, 1)
        curve.append(m)
        print(f"  [{mode}] epoch {ep + 1}/{epochs} mean_batch_mae={m:.6e}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), out_path)
    print(f"  saved encoder -> {out_path}")
    return curve


def eval_cst_mae_batched(
    bundle: dict,
    device: torch.device,
    batch_rows: int,
    coord_dim: int,
) -> float:
    coords = bundle["coords"]
    x_coords = bundle["x_coords"]
    n = coords.shape[0]
    encoder = CSTEncoder18(coord_dim, n_weights_per_side=8, n1=0.5, n2=1.0).to(device)
    decoder = CSTDecoder18(n_weights_per_side=8, n1=0.5, n2=1.0).to(device)

    total_mae = 0.0
    n_batches = 0
    x_expand = x_coords.unsqueeze(0)
    for s in range(0, n, batch_rows):
        e = min(s + batch_rows, n)
        xb = coords[s:e]
        bsz = xb.shape[0]
        with torch.no_grad():
            z = encoder(xb)
            recon = decoder(z, x_expand.expand(bsz, -1))
            total_mae += float(mae_loss(recon, xb).detach())
        n_batches += 1
    return total_mae / max(n_batches, 1)


def save_cst_meta(out_path: Path, decoder: CSTDecoder18) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "analytic_cst_decoder",
            "n_weights_per_side": decoder.n_weights_per_side,
            "n1": decoder.n1,
            "n2": decoder.n2,
        },
        out_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "original.csv")
    p.add_argument("--cache", type=Path, default=ROOT / "models" / "original_tensors.pt")
    p.add_argument("--rebuild-cache", action="store_true", help="Rebuild tensor cache from CSV")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-rows", type=int, default=None, help="Cap CSV rows before caching")
    p.add_argument("--batch-points", type=int, default=131072, help="Polar samples per pair-AE step")
    p.add_argument("--batch-rows", type=int, default=512, help="Airfoil rows per CST MAE batch")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = p.parse_args()

    device = resolve_device(args.device)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading / building tensor cache -> {args.cache}")
    bundle_cpu = load_or_build_cache(args.csv, args.cache, args.max_rows, args.rebuild_cache)
    coord_dim = int(bundle_cpu["coords"].shape[1])
    bundle = move_bundle_to_device(bundle_cpu, device)

    print(f"Training pair tanh autoencoders on {device} (save encoder only, MAE)...")
    modes_fnames = [
        ("Cl", "encoder_cl_alpha.pt"),
        ("Cd", "encoder_cd_alpha.pt"),
        ("mach", "encoder_mach_alpha.pt"),
        ("Re", "encoder_re_alpha.pt"),
    ]
    curves: dict[str, list[float]] = {}
    for mode, fname in modes_fnames:
        curves[mode] = train_pair_epochs(
            bundle,
            device,
            mode,
            args.epochs,
            args.lr,
            args.batch_points,
            models_dir / fname,
        )

    fd = figures_dir()
    fig_p, axes_p = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, (mode, _) in zip(axes_p.ravel(), modes_fnames):
        ep = list(range(1, args.epochs + 1))
        ax.plot(ep, curves[mode], marker=".", ms=2)
        ax.set_title(f"{mode}+α encoder MAE")
        ax.set_ylabel("batch MAE")
    for ax in axes_p[-1, :]:
        ax.set_xlabel("epoch")
    fig_p.suptitle("Pair autoencoder training curves")
    fig_p.tight_layout()
    fig_p.savefig(fd / "train_autoencoders_pair_mae_curves.png", dpi=220)
    plt.close(fig_p)

    decoder = CSTDecoder18(n_weights_per_side=8, n1=0.5, n2=1.0)
    print(f"CST reconstruction MAE (coord_dim={coord_dim}) on {device}...")
    cst_mae_curve: list[float] = []
    for ep in range(args.epochs):
        mae = eval_cst_mae_batched(bundle, device, args.batch_rows, coord_dim)
        cst_mae_curve.append(mae)
        print(f"  [CST] pass {ep + 1}/{args.epochs} mean_batch_mae={mae:.6e}")

    fig_c, ax_c = plt.subplots(figsize=(8, 4))
    ax_c.plot(range(1, args.epochs + 1), cst_mae_curve, marker=".", color="C2")
    ax_c.set_xlabel("pass")
    ax_c.set_ylabel("mean batch MAE (coords)")
    ax_c.set_title("CST encoder/decoder coordinate reconstruction (batched MAE)")
    fig_c.tight_layout()
    fig_c.savefig(fd / "train_autoencoders_cst_coord_mae.png", dpi=220)
    plt.close(fig_c)

    save_cst_meta(models_dir / "decoder_coords.pt", decoder)
    print(f"  saved CST meta -> {models_dir / 'decoder_coords.pt'}")
    print(f"Saved figures under {fd}")
    print("Done.")


if __name__ == "__main__":
    main()
