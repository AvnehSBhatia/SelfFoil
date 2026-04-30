#!/usr/bin/env python3
"""Train WHISP: (Cl, Cd, Re, Mach, alpha) -> CST18 using frozen pair encoders + stabilized physics losses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.csv_tensor_cache import load_or_build_cache
from core.whisp_net import WHISP


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def move_bundle(bundle: dict, device: torch.device) -> dict:
    out = {}
    for k, v in bundle.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def airfoil_row_splits(n_rows: int, seed: int, frac_train: float, frac_val: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n_rows < 3:
        raise ValueError("Need at least 3 airfoil rows for train/val/test splits")
    if not (0.0 < frac_train < 1.0 and 0.0 < frac_val < 1.0 and frac_train + frac_val < 1.0):
        raise ValueError("Invalid split fractions (need room for test)")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_rows, generator=g)
    n_tr = max(1, int(frac_train * n_rows))
    n_va = max(1, int(frac_val * n_rows))
    n_te = n_rows - n_tr - n_va
    while n_te < 1 and n_va > 1:
        n_va -= 1
        n_te = n_rows - n_tr - n_va
    while n_te < 1 and n_tr > 1:
        n_tr -= 1
        n_te = n_rows - n_tr - n_va
    if n_te < 1:
        raise RuntimeError("Could not form non-empty test split")
    train_rows = perm[:n_tr]
    val_rows = perm[n_tr : n_tr + n_va]
    test_rows = perm[n_tr + n_va :]
    return train_rows, val_rows, test_rows


def polar_mask_for_rows(polar_row_idx: torch.Tensor, allowed_rows: torch.Tensor) -> torch.Tensor:
    return torch.isin(polar_row_idx, allowed_rows.to(polar_row_idx.device))


def required_bundle_keys() -> set[str]:
    return {
        "alpha_flat",
        "cl_flat",
        "cd_flat",
        "mach_flat",
        "re_log_flat",
        "polar_row_idx",
        "cst18",
        "polar_ptr",
        "total_polar",
        "n_rows",
    }


def route_tau_schedule(epoch: int, epochs: int, tau_start: float, tau_end: float) -> float:
    """Linear cooldown: high τ early (soft routing) → lower τ (sharper)."""
    if epochs <= 1:
        return tau_end
    t = epoch / max(epochs - 1, 1)
    return tau_start + (tau_end - tau_start) * t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "original.csv")
    p.add_argument("--cache", type=Path, default=ROOT / "models" / "original_tensors.pt")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-ns", type=float, default=0.05, dest="lambda_ns")
    p.add_argument("--lambda-cl-gamma", type=float, default=0.02, dest="lambda_cl_gamma")
    p.add_argument("--lambda-cl-direct", type=float, default=0.15, dest="lambda_cl_direct")
    p.add_argument("--tau-start", type=float, default=2.0, dest="tau_start", help="Routing softmax temperature at epoch 0")
    p.add_argument("--tau-end", type=float, default=0.5, dest="tau_end", help="Routing temperature at last epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac-train", type=float, default=0.8)
    p.add_argument("--frac-val", type=float, default=0.1)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out", type=Path, default=ROOT / "models" / "whisp.pt")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models", help="Directory with encoder_*.pt from train_autoencoders")
    p.add_argument("--enc-cl", type=Path, default=None)
    p.add_argument("--enc-cd", type=Path, default=None)
    p.add_argument("--enc-re", type=Path, default=None)
    p.add_argument("--enc-mach", type=Path, default=None)
    args = p.parse_args()

    device = resolve_device(args.device)
    md = args.models_dir
    enc_cl = args.enc_cl or md / "encoder_cl_alpha.pt"
    enc_cd = args.enc_cd or md / "encoder_cd_alpha.pt"
    enc_re = args.enc_re or md / "encoder_re_alpha.pt"
    enc_mach = args.enc_mach or md / "encoder_mach_alpha.pt"
    encoder_ckpts = (enc_cl, enc_cd, enc_re, enc_mach)
    for path in encoder_ckpts:
        if not path.is_file():
            raise SystemExit(
                f"Missing pair encoder checkpoint: {path}\n"
                "Train them first: python scripts/train_autoencoders.py\n"
                "Or pass --enc-cl, --enc-cd, --enc-re, --enc-mach."
            )

    bundle_cpu = load_or_build_cache(args.csv, args.cache, args.max_rows, args.rebuild_cache)
    missing = required_bundle_keys() - set(bundle_cpu.keys())
    if missing:
        raise SystemExit(
            f"Tensor cache missing keys {sorted(missing)}. Rebuild with: "
            f"python {Path(__file__).name} --rebuild-cache"
        )

    n_rows = int(bundle_cpu["n_rows"])
    train_rows, val_rows, test_rows = airfoil_row_splits(n_rows, args.seed, args.frac_train, args.frac_val)
    bundle = move_bundle(bundle_cpu, device)

    polar_row = bundle["polar_row_idx"]
    train_mask = polar_mask_for_rows(polar_row, train_rows)
    val_mask = polar_mask_for_rows(polar_row, val_rows)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise SystemExit("Empty train or val polar split; adjust fractions or dataset size.")

    model = WHISP(encoder_ckpts=encoder_ckpts).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    last_k = model.n_outer - 1

    def forward_losses(idx: torch.Tensor, route_tau: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cl = bundle["cl_flat"][idx]
        cd = bundle["cd_flat"][idx]
        re_log = bundle["re_log_flat"][idx]
        mach = bundle["mach_flat"][idx]
        alpha = bundle["alpha_flat"][idx]
        row = polar_row[idx]
        cst_tgt = bundle["cst18"][row]

        cst_pred, aux = model(cl, cd, re_log, mach, alpha, route_tau=route_tau)
        l_geo = (cst_pred - cst_tgt).abs().mean()
        l_ns = torch.stack(aux["L_ns_list"]).mean()
        l_cl_g = nn.functional.mse_loss(aux[f"cl_gamma_{last_k}"], cl)
        l_cl_d = nn.functional.mse_loss(aux["cl_direct"], cl)
        loss = (
            l_geo
            + args.lambda_ns * l_ns
            + args.lambda_cl_gamma * l_cl_g
            + args.lambda_cl_direct * l_cl_d
        )
        parts = {"geo": l_geo, "ns": l_ns, "clg": l_cl_g, "cld": l_cl_d}
        return loss, parts

    print(f"Device={device} | train polar={train_idx.numel()} val polar={val_idx.numel()} | rows={n_rows}")
    n_trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"WHISP trainable parameters: {n_trainable} (pair encoders frozen)")

    for ep in range(args.epochs):
        tau = route_tau_schedule(ep, args.epochs, args.tau_start, args.tau_end)
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        run_loss = 0.0
        run_parts: dict[str, float] = {"geo": 0.0, "ns": 0.0, "clg": 0.0, "cld": 0.0}
        n_batches = 0
        for s in range(0, perm.numel(), args.batch):
            sl = perm[s : s + args.batch]
            opt.zero_grad(set_to_none=True)
            loss, parts = forward_losses(sl, tau)
            loss.backward()
            opt.step()

            run_loss += float(loss.detach())
            run_parts["geo"] += float(parts["geo"].detach())
            run_parts["ns"] += float(parts["ns"].detach())
            run_parts["clg"] += float(parts["clg"].detach())
            run_parts["cld"] += float(parts["cld"].detach())
            n_batches += 1

        model.eval()
        val_tau = args.tau_end
        with torch.no_grad():
            v_loss = 0.0
            v_geo = 0.0
            for s in range(0, val_idx.numel(), args.batch):
                sl = val_idx[s : s + args.batch]
                lf, parts = forward_losses(sl, val_tau)
                v_loss += float(lf)
                v_geo += float(parts["geo"])
            v_n = max(1, (val_idx.numel() + args.batch - 1) // args.batch)
            v_loss /= v_n
            v_geo /= v_n

        if n_batches:
            run_loss /= n_batches
            for k in run_parts:
                run_parts[k] /= n_batches

        print(
            f"epoch {ep + 1}/{args.epochs}  τ={tau:.3f}  train_loss={run_loss:.5e}  "
            f"[geo={run_parts['geo']:.4e} ns={run_parts['ns']:.4e} clg={run_parts['clg']:.4e} cld={run_parts['cld']:.4e}]  "
            f"val_loss={v_loss:.5e} val_geo_mae={v_geo:.5e}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "n_outer": model.n_outer,
                "n_inner": model.stages[0].n_inner,
                "d": model.d,
                "encoder_ckpts": [str(x) for x in encoder_ckpts],
                "tau_start": args.tau_start,
                "tau_end": args.tau_end,
                "train_rows": train_rows.cpu(),
                "val_rows": val_rows.cpu(),
                "test_rows": test_rows.cpu(),
            },
        },
        args.out,
    )
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
