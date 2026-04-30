#!/usr/bin/env python3
"""Train WHISP: (Cl, Cd, Re, Mach, alpha) -> CST18 using frozen pair encoders + stabilized physics losses."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.csv_tensor_cache import load_or_build_cache
from core.device import configure_cuda_training, resolve_device
from core.figures_path import figures_dir
from core.whisp_net import WHISP


def move_bundle(bundle: dict, device: torch.device) -> dict:
    out = {}
    for k, v in bundle.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _fetch_batch(bundle_cpu: dict, idx: torch.Tensor, key: str, device: torch.device) -> torch.Tensor:
    return bundle_cpu[key][idx].to(device, non_blocking=True)


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


def maybe_compile_model(model: nn.Module, device: torch.device, enabled: bool, backend: str) -> nn.Module:
    if not enabled or not hasattr(torch, "compile"):
        return model
    if backend == "auto":
        # ROCm+Inductor can spend minutes in async kernel compilation for this model.
        # Use AOT eager by default so --compile does not make startup look frozen.
        backend = "aot_eager" if device.type == "cuda" and getattr(torch.version, "hip", None) is not None else "inductor"
    try:
        if backend == "inductor":
            try:
                import torch._inductor.config as inductor_config

                inductor_config.max_autotune = False
            except Exception:
                pass
            return torch.compile(model, mode="reduce-overhead")
        if backend == "none":
            return model
        return torch.compile(model, backend=backend)
    except Exception as e:
        print(f"[warn] torch.compile setup failed ({e}); falling back to eager.")
        return model


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_batch_count(n_items: int, batch: int, max_batches: int | None) -> int:
    n = (n_items + batch - 1) // batch
    if max_batches is not None:
        n = min(n, max_batches)
    return max(1, n)


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
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out", type=Path, default=ROOT / "models" / "whisp.pt")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models", help="Directory with encoder_*.pt from train_autoencoders")
    p.add_argument("--enc-cl", type=Path, default=None)
    p.add_argument("--enc-cd", type=Path, default=None)
    p.add_argument("--enc-re", type=Path, default=None)
    p.add_argument("--enc-mach", type=Path, default=None)
    p.add_argument("--compile", action="store_true", help="Compile model with torch.compile for faster training")
    p.add_argument(
        "--compile-backend",
        default="auto",
        choices=["auto", "inductor", "aot_eager", "eager", "none"],
        help="torch.compile backend; auto avoids slow ROCm Inductor startup",
    )
    p.add_argument("--warmup", action="store_true", help="Run one no-grad warmup batch before training")
    p.add_argument("--log-every", type=int, default=100, help="Print progress every N train batches (0 disables)")
    p.add_argument("--max-train-batches", type=int, default=None, help="Cap train batches per epoch for quick CUDA checks")
    p.add_argument("--max-val-batches", type=int, default=None, help="Cap validation batches per epoch for quick CUDA checks")
    args = p.parse_args()

    device = resolve_device(args.device)
    configure_cuda_training(device, deterministic=False)
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

    polar_row_cpu = bundle_cpu["polar_row_idx"]
    train_mask = polar_mask_for_rows(polar_row_cpu, train_rows)
    val_mask = polar_mask_for_rows(polar_row_cpu, val_rows)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise SystemExit("Empty train or val polar split; adjust fractions or dataset size.")

    model_base = WHISP(encoder_ckpts=encoder_ckpts).to(device)
    model = maybe_compile_model(model_base, device, enabled=args.compile, backend=args.compile_backend)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    last_k = model_base.n_outer - 1

    def forward_losses(idx: torch.Tensor, route_tau: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cl = _fetch_batch(bundle_cpu, idx, "cl_flat", device)
        cd = _fetch_batch(bundle_cpu, idx, "cd_flat", device)
        re_log = _fetch_batch(bundle_cpu, idx, "re_log_flat", device)
        mach = _fetch_batch(bundle_cpu, idx, "mach_flat", device)
        alpha = _fetch_batch(bundle_cpu, idx, "alpha_flat", device)
        row = bundle_cpu["polar_row_idx"][idx]
        cst_tgt = bundle_cpu["cst18"][row].to(device, non_blocking=True)

        cst_pred, aux = model(cl, cd, re_log, mach, alpha, route_tau=route_tau)
        l_geo = (cst_pred - cst_tgt).abs().mean()
        l_ns = aux["L_ns"]
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
    n_trainable = sum(x.numel() for x in model_base.parameters() if x.requires_grad)
    print(f"WHISP trainable parameters: {n_trainable} (pair encoders frozen)")
    total_train_batches = train_batch_count(train_idx.numel(), args.batch, args.max_train_batches)
    total_val_batches = train_batch_count(val_idx.numel(), args.batch, args.max_val_batches)
    print(
        f"Batch={args.batch} | train_batches/epoch={total_train_batches} "
        f"val_batches/epoch={total_val_batches} | compile_backend={args.compile_backend if args.compile else 'off'}"
    )
    if args.warmup:
        with torch.no_grad():
            warm_idx = train_idx[: min(args.batch, train_idx.numel())]
            _ = forward_losses(warm_idx, args.tau_start)
            sync_if_needed(device)

    hist_train_loss: list[float] = []
    hist_val_loss: list[float] = []
    hist_val_geo: list[float] = []
    hist_tau: list[float] = []
    hist_train_geo: list[float] = []
    hist_train_ns: list[float] = []
    hist_train_clg: list[float] = []
    hist_train_cld: list[float] = []

    for ep in range(args.epochs):
        tau = route_tau_schedule(ep, args.epochs, args.tau_start, args.tau_end)
        model.train()
        perm_idx = torch.randperm(train_idx.numel(), device="cpu")
        perm = train_idx[perm_idx]
        run_loss = torch.zeros((), device=device, dtype=torch.float32)
        run_parts: dict[str, torch.Tensor] = {
            "geo": torch.zeros((), device=device, dtype=torch.float32),
            "ns": torch.zeros((), device=device, dtype=torch.float32),
            "clg": torch.zeros((), device=device, dtype=torch.float32),
            "cld": torch.zeros((), device=device, dtype=torch.float32),
        }
        n_batches = 0
        t_epoch = time.perf_counter()
        for s in range(0, perm.numel(), args.batch):
            if args.max_train_batches is not None and n_batches >= args.max_train_batches:
                break
            sl = perm[s : s + args.batch]
            opt.zero_grad(set_to_none=True)
            loss, parts = forward_losses(sl, tau)
            loss.backward()
            opt.step()

            run_loss = run_loss + loss.detach()
            run_parts["geo"] = run_parts["geo"] + parts["geo"].detach()
            run_parts["ns"] = run_parts["ns"] + parts["ns"].detach()
            run_parts["clg"] = run_parts["clg"] + parts["clg"].detach()
            run_parts["cld"] = run_parts["cld"] + parts["cld"].detach()
            n_batches += 1
            if args.log_every > 0 and (n_batches == 1 or n_batches % args.log_every == 0):
                sync_if_needed(device)
                elapsed = time.perf_counter() - t_epoch
                batches_per_s = n_batches / max(elapsed, 1e-9)
                print(
                    f"  train batch {n_batches}/{total_train_batches} "
                    f"({batches_per_s:.2f} batch/s, loss={loss.detach().item():.4e})",
                    flush=True,
                )

        model.eval()
        val_tau = args.tau_end
        with torch.no_grad():
            v_loss = torch.zeros((), device=device, dtype=torch.float32)
            v_geo = torch.zeros((), device=device, dtype=torch.float32)
            for vi, s in enumerate(range(0, val_idx.numel(), args.batch)):
                if args.max_val_batches is not None and vi >= args.max_val_batches:
                    break
                sl = val_idx[s : s + args.batch]
                lf, parts = forward_losses(sl, val_tau)
                v_loss = v_loss + lf.detach()
                v_geo = v_geo + parts["geo"].detach()
            v_n = total_val_batches
            v_loss = (v_loss / v_n).item()
            v_geo = (v_geo / v_n).item()

        if n_batches:
            run_loss = (run_loss / n_batches).item()
            for k in run_parts:
                run_parts[k] = (run_parts[k] / n_batches).item()
        else:
            run_loss = float(run_loss.item())
            for k in run_parts:
                run_parts[k] = float(run_parts[k].item())

        print(
            f"epoch {ep + 1}/{args.epochs}  τ={tau:.3f}  train_loss={run_loss:.5e}  "
            f"[geo={run_parts['geo']:.4e} ns={run_parts['ns']:.4e} clg={run_parts['clg']:.4e} cld={run_parts['cld']:.4e}]  "
            f"val_loss={v_loss:.5e} val_geo_mae={v_geo:.5e}"
        )
        hist_train_loss.append(run_loss)
        hist_val_loss.append(v_loss)
        hist_val_geo.append(v_geo)
        hist_tau.append(tau)
        hist_train_geo.append(run_parts["geo"])
        hist_train_ns.append(run_parts["ns"])
        hist_train_clg.append(run_parts["clg"])
        hist_train_cld.append(run_parts["cld"])

    fd = figures_dir()
    ep_axis = np.arange(1, args.epochs + 1)
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ep_axis, hist_train_loss, label="train loss")
    ax1.plot(ep_axis, hist_val_loss, label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("WHISP training / validation loss")
    fig1.tight_layout()
    fig1.savefig(fd / "train_whisp_loss_curves.png", dpi=220)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(ep_axis, hist_val_geo, color="C2", label="val CST MAE (geo)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.set_title("WHISP validation CST error")
    fig2.tight_layout()
    fig2.savefig(fd / "train_whisp_val_cst_mae.png", dpi=220)
    plt.close(fig2)

    fig3, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    axes[0, 0].plot(ep_axis, hist_train_geo, color="C0")
    axes[0, 0].set_ylabel("train geo")
    axes[0, 1].plot(ep_axis, hist_train_ns, color="C1")
    axes[0, 1].set_ylabel("train L_ns")
    axes[1, 0].plot(ep_axis, hist_train_clg, color="C2")
    axes[1, 0].set_ylabel("train cl_γ MSE")
    axes[1, 1].plot(ep_axis, hist_train_cld, color="C3")
    axes[1, 1].set_ylabel("train cl_direct MSE")
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
    fig3.suptitle("WHISP loss components (train batch means)")
    fig3.tight_layout()
    fig3.savefig(fd / "train_whisp_loss_components.png", dpi=220)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 3.5))
    ax4.plot(ep_axis, hist_tau, color="C4", marker=".", ms=3)
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("τ (routing softmax temp)")
    ax4.set_title("WHISP routing temperature schedule")
    fig4.tight_layout()
    fig4.savefig(fd / "train_whisp_routing_tau.png", dpi=220)
    plt.close(fig4)

    print(f"Saved figures under {fd}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_base.state_dict(),
            "meta": {
                "n_outer": model_base.n_outer,
                "n_inner": model_base.stages[0].n_inner,
                "d": model_base.d,
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
