#!/usr/bin/env python3
"""Load encoders + tensor cache; scatter latents and CST recon plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.csv_tensor_cache import load_or_build_cache
from core.cst_kulfan import CSTDecoder18, CSTEncoder18
from core.pair_tanh_autoencoder import PairTanhAutoencoder


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def collect_pair_latents(
    bundle: dict,
    encoder_path: Path,
    mode: str,
    max_rows: int,
    device: torch.device,
) -> torch.Tensor:
    m = PairTanhAutoencoder(latent_dim=8).to(device)
    m.encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    m.eval()

    n_rows = int(bundle["n_rows"])
    n_use = min(max_rows, n_rows)
    ptr = bundle["polar_ptr"]
    zs = []
    for i in range(n_use):
        s = int(ptr[i].item())
        e = int(ptr[i + 1].item())
        if mode == "Cl":
            feat = bundle["cl_flat"][s:e]
        elif mode == "Cd":
            feat = bundle["cd_flat"][s:e]
        elif mode == "mach":
            feat = bundle["mach_flat"][s:e]
        else:
            feat = bundle["re_log_flat"][s:e]
        aoa = bundle["alpha_flat"][s:e]
        x = torch.stack([feat, aoa], dim=1)
        with torch.no_grad():
            z = m.encoder(x)
        zs.append(z.cpu())
    return torch.cat(zs, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "original.csv")
    ap.add_argument("--cache", type=Path, default=ROOT / "models" / "original_tensors.pt")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--max-rows", type=int, default=800)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    device = resolve_device(args.device)
    models_dir = ROOT / "models"
    fig_dir = ROOT / "test" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    bundle_cpu = load_or_build_cache(
        args.csv,
        args.cache,
        max_rows=args.max_rows if args.rebuild_cache else None,
        rebuild=args.rebuild_cache,
    )
    bundle = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in bundle_cpu.items()}
    coord_dim = int(bundle["coords"].shape[1])

    for mode, fname in [
        ("Cl", "encoder_cl_alpha.pt"),
        ("Cd", "encoder_cd_alpha.pt"),
        ("mach", "encoder_mach_alpha.pt"),
        ("Re", "encoder_re_alpha.pt"),
    ]:
        path = models_dir / fname
        if not path.is_file():
            print(f"skip {fname}: missing")
            continue
        z = collect_pair_latents(bundle, path, mode, args.max_rows, device)
        print(f"{mode}: latent shape {tuple(z.shape)} mean_abs={z.abs().mean():.4f}")
        plt.figure(figsize=(5, 4))
        plt.scatter(z[:, 0].numpy(), z[:, 1].numpy(), s=2, alpha=0.35)
        plt.xlabel("latent dim 0")
        plt.ylabel("latent dim 1")
        plt.title(f"{mode}+AoA encoder (first two dims)")
        plt.tight_layout()
        out = fig_dir / f"latent_{mode.lower()}_scatter.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  wrote {out}")

    dec_path = models_dir / "decoder_coords.pt"
    if dec_path.is_file():
        meta = torch.load(dec_path, map_location=device, weights_only=True)
        if isinstance(meta, dict) and meta.get("type") == "analytic_cst_decoder":
            print(
                "CST:",
                f"n_weights_per_side={meta.get('n_weights_per_side')}, n1={meta.get('n1')}, n2={meta.get('n2')}",
            )
        enc = CSTEncoder18(coord_dim, n_weights_per_side=8, n1=0.5, n2=1.0).to(device)
        dec = CSTDecoder18(n_weights_per_side=8, n1=0.5, n2=1.0).to(device)
        xgrid = bundle["x_coords"].unsqueeze(0)
        n_plot = min(3, int(bundle["n_rows"]))
        plt.figure(figsize=(10, 3))
        for k in range(n_plot):
            xb = bundle["coords"][k : k + 1]
            with torch.no_grad():
                z = enc(xb)
                pred = dec(z, xgrid)
            xy_true = xb.cpu().numpy().reshape(-1, 2)
            xy_pred = pred.cpu().numpy().reshape(-1, 2)
            plt.subplot(1, n_plot, k + 1)
            plt.plot(xy_true[:, 0], xy_true[:, 1], "-", lw=1.0, label="true")
            plt.plot(xy_pred[:, 0], xy_pred[:, 1], "--", lw=1.0, label="cst")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title(f"recon row {k}")
            if k == 0:
                plt.legend(fontsize=7, loc="best")
        plt.tight_layout()
        out = fig_dir / "decoder_coords_samples.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"wrote {out}")
    else:
        print("skip decoder_coords.pt: missing")


if __name__ == "__main__":
    main()
