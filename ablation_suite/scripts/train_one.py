#!/usr/bin/env python3
"""Train one WHISP ablation from `ablation_suite.catalog` (runs/<category>/<slug>/)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_suite.catalog import get_spec
from ablation_suite.models.registry import resolve_run_key, run_path_parts
from ablation_suite.models.whisp_ablated import WHISPAblated
from core.csv_tensor_cache import load_or_build_cache
from core.device import configure_cuda_training, resolve_device
from core.figures_path import figures_dir
from scripts.train_whisp import (
    airfoil_row_splits,
    aux_scale_schedule,
    build_lr_scheduler,
    fluctuation_plateau,
    maybe_compile_model,
    polar_mask_for_rows,
    required_bundle_keys,
    set_two_stage_trainable,
    sync_if_needed,
    train_batch_count,
)


def _fetch_batch(bundle_cpu: dict, idx: torch.Tensor, key: str, device: torch.device) -> torch.Tensor:
    return bundle_cpu[key][idx].to(device, non_blocking=True)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def count_trainable(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", type=str, help="Catalog key, e.g. physics/no_bl or baseline/full")
    g.add_argument("--model", type=str, help="Legacy baseline slug only, e.g. full, no_physics")
    p.add_argument("--csv", type=Path, default=ROOT / "data" / "original.csv")
    p.add_argument("--cache", type=Path, default=ROOT / "models" / "original_tensors.pt")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda-ns", type=float, default=0.05, dest="lambda_ns")
    p.add_argument("--lambda-cl-gamma", type=float, default=0.02, dest="lambda_cl_gamma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frac-train", type=float, default=0.8)
    p.add_argument("--frac-val", type=float, default=0.1)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--enc-cl", type=Path, default=None)
    p.add_argument("--enc-cd", type=Path, default=None)
    p.add_argument("--enc-re", type=Path, default=None)
    p.add_argument("--enc-mach", type=Path, default=None)
    p.add_argument("--dropout-start", type=float, default=0.05, help="Dropout probability in WHISP blocks/heads")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--suite-root", type=Path, default=ROOT / "ablation_suite")
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
    p.add_argument("--aux-ramp-epochs", type=int, default=10, help="Ramp auxiliary losses from 0 to full weight over this many epochs")
    p.add_argument("--early-stop-patience", type=int, default=0, help="Early stop patience (0 disables)")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Minimum improvement to reset patience")
    p.add_argument(
        "--early-stop-mode",
        default="delta",
        choices=["delta", "fluctuation"],
        help="delta: patience on min-delta improvement, fluctuation: stop when rolling window range is small",
    )
    p.add_argument("--early-stop-window", type=int, default=8, help="Window size for fluctuation-based stopping")
    p.add_argument("--early-stop-fluctuation-tol", type=float, default=1e-4, help="Range tolerance for fluctuation-based stopping")
    p.add_argument(
        "--early-stop-monitor",
        default="val_geo",
        choices=["val_geo", "val_loss"],
        help="Metric monitored by early stopping",
    )
    p.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine", "onecycle"], help="Dynamic LR schedule")
    p.add_argument("--lr-min-factor", type=float, default=0.1, help="Min LR as a fraction of base LR for cosine/onecycle")
    p.add_argument("--geo-loss", default="huber", choices=["mae", "huber"], help="Geometry loss type")
    p.add_argument("--huber-delta", type=float, default=0.02, help="Huber delta for geometry loss")
    p.add_argument("--aux-ema-beta", type=float, default=0.98, help="EMA beta for aux loss normalization")
    p.add_argument("--aux-norm-eps", type=float, default=1e-6, help="Epsilon for aux loss normalization")
    p.add_argument("--two-stage", action="store_true", help="Enable two-stage training with early-stop-driven stage switching")
    p.add_argument("--stage-cycles", type=int, default=2, help="Number of geo<->aux cycles in two-stage mode")
    args = p.parse_args()

    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    configure_cuda_training(device, deterministic=args.deterministic)
    run_key = resolve_run_key(args.run_id, args.model)
    spec = get_spec(run_key)
    spec = {**spec, "dropout_p": args.dropout_start}
    train_cfg = dict(spec["train"])
    frac_train = float(train_cfg.get("frac_train") or args.frac_train)
    frac_val = float(train_cfg.get("frac_val") or args.frac_val)
    lam_ns = float(train_cfg["lambda_ns"] if train_cfg.get("lambda_ns") is not None else args.lambda_ns)
    lam_cl = float(
        train_cfg["lambda_cl_gamma"] if train_cfg.get("lambda_cl_gamma") is not None else args.lambda_cl_gamma
    )
    noise_std = float(train_cfg.get("input_noise_std") or 0.0)
    stride = int(train_cfg.get("sparse_aoa_stride") or 1)
    extrap = bool(train_cfg.get("extrapolation"))
    tr_max = train_cfg.get("re_log_train_max")
    va_min = train_cfg.get("re_log_val_min")
    loss_profile = str(train_cfg.get("loss_profile") or "full")
    anneal = bool(train_cfg.get("anneal_physics"))

    md = args.models_dir
    enc_cl = args.enc_cl or md / "encoder_cl_alpha.pt"
    enc_cd = args.enc_cd or md / "encoder_cd_alpha.pt"
    enc_re = args.enc_re or md / "encoder_re_alpha.pt"
    enc_mach = args.enc_mach or md / "encoder_mach_alpha.pt"
    encoder_ckpts = (enc_cl, enc_cd, enc_re, enc_mach)
    if str(spec.get("encoder_mode", "frozen")) != "scratch":
        for path in encoder_ckpts:
            if not path.is_file():
                raise SystemExit(f"Missing encoder checkpoint: {path}")

    bundle_cpu = load_or_build_cache(args.csv, args.cache, args.max_rows, args.rebuild_cache)
    missing = required_bundle_keys() - set(bundle_cpu.keys())
    if missing:
        raise SystemExit(f"Cache missing keys {sorted(missing)}; use --rebuild-cache.")

    n_rows = int(bundle_cpu["n_rows"])
    train_rows, val_rows, test_rows = airfoil_row_splits(n_rows, args.seed, frac_train, frac_val)
    polar_row_cpu = bundle_cpu["polar_row_idx"]
    train_mask = polar_mask_for_rows(polar_row_cpu, train_rows)
    val_mask = polar_mask_for_rows(polar_row_cpu, val_rows)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise SystemExit("Empty train or val split.")

    re_log_all = bundle_cpu["re_log_flat"]
    if extrap and tr_max is not None and va_min is not None:
        tr_max_t = torch.tensor(float(tr_max), device=device, dtype=re_log_all.dtype)
        va_min_t = torch.tensor(float(va_min), device=device, dtype=re_log_all.dtype)
        train_idx = train_idx[re_log_all[train_idx] < float(tr_max)]
        val_idx = val_idx[re_log_all[val_idx] >= float(va_min)]
    if stride > 1:
        train_idx = train_idx[torch.arange(train_idx.numel()) % stride == 0]

    model_base = WHISPAblated(encoder_ckpts, spec).to(device)
    model = maybe_compile_model(model_base, device, enabled=args.compile, backend=args.compile_backend)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    distill_w = float(spec.get("distill_weight", 0.0) or 0.0)
    model_for_meta = model_base
    ema_ns = torch.tensor(1.0, device=device)
    ema_clg = torch.tensor(1.0, device=device)
    ema_cld = torch.tensor(1.0, device=device)
    ema_beta = float(args.aux_ema_beta)

    def forward_losses(
        idx: torch.Tensor, ep: int, aux_scale: float, *, update_ema: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        nonlocal ema_ns, ema_clg, ema_cld
        cl = _fetch_batch(bundle_cpu, idx, "cl_flat", device)
        cd = _fetch_batch(bundle_cpu, idx, "cd_flat", device)
        re_log = _fetch_batch(bundle_cpu, idx, "re_log_flat", device)
        mach = _fetch_batch(bundle_cpu, idx, "mach_flat", device)
        alpha = _fetch_batch(bundle_cpu, idx, "alpha_flat", device)
        if noise_std > 0.0:
            cl = cl + torch.randn_like(cl) * noise_std
            cd = cd + torch.randn_like(cd) * noise_std
            re_log = re_log + torch.randn_like(re_log) * noise_std
            mach = mach + torch.randn_like(mach) * noise_std
            alpha = alpha + torch.randn_like(alpha) * noise_std
        row = bundle_cpu["polar_row_idx"][idx]
        cst_tgt = bundle_cpu["cst18"][row].to(device, non_blocking=True)

        cst_pred, aux = model(cl, cd, re_log, mach, alpha)
        if args.geo_loss == "huber":
            l_geo = nn.functional.huber_loss(cst_pred, cst_tgt, delta=args.huber_delta, reduction="mean")
        else:
            l_geo = (cst_pred - cst_tgt).abs().mean()
        parts: dict[str, torch.Tensor] = {"geo": l_geo}
        l_ns_mean = aux["L_ns"] if model_for_meta.use_physics else torch.tensor(0.0, device=device)
        parts["ns"] = l_ns_mean
        cl_keys = [k for k in aux if k.startswith("cl_gamma_")]
        last_k = max((int(k.split("_")[-1]) for k in cl_keys), default=model_for_meta.n_outer - 1)
        l_cl_g = torch.tensor(0.0, device=device)
        if f"cl_gamma_{last_k}" in aux:
            l_cl_g = nn.functional.mse_loss(aux[f"cl_gamma_{last_k}"], cl)
        if update_ema:
            ema_ns = ema_beta * ema_ns + (1.0 - ema_beta) * l_ns_mean.detach()
            ema_clg = ema_beta * ema_clg + (1.0 - ema_beta) * l_cl_g.detach()
        parts["clg"] = l_cl_g
        l_cl_head = nn.functional.mse_loss(aux["cl_direct"], cl)
        if update_ema:
            ema_cld = ema_beta * ema_cld + (1.0 - ema_beta) * l_cl_head.detach()
        l_ns_norm = l_ns_mean / (ema_ns.detach() + args.aux_norm_eps)
        l_cl_g_norm = l_cl_g / (ema_clg.detach() + args.aux_norm_eps)
        l_cl_head_norm = l_cl_head / (ema_cld.detach() + args.aux_norm_eps)

        t = 1.0 - ep / max(1, args.epochs - 1) if anneal and args.epochs > 1 else 1.0
        lam_ns_ep = lam_ns * t
        lam_cl_ep = lam_cl * t

        if loss_profile == "geo_only":
            loss = l_geo
        elif loss_profile == "aero_only":
            loss = l_cl_g_norm if model_for_meta.use_physics and f"cl_gamma_{last_k}" in aux else l_cl_head_norm
        elif loss_profile == "no_geo":
            loss = (
                aux_scale * lam_ns_ep * l_ns_norm + aux_scale * lam_cl_ep * l_cl_g_norm
                if model_for_meta.use_physics
                else l_cl_head_norm
            )
        else:
            loss = l_geo
            if model_for_meta.use_physics:
                loss = loss + aux_scale * lam_ns_ep * l_ns_norm + aux_scale * lam_cl_ep * l_cl_g_norm
        if distill_w > 0.0 and "embed_distill" in aux:
            loss = loss + distill_w * aux["embed_distill"]
        return loss, parts

    cat, slug = run_path_parts(run_key)
    run_dir = args.suite_root / "runs" / cat / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    print(
        f"WHISP run={run_key} device={device} train={train_idx.numel()} val={val_idx.numel()} "
        f"trainable={count_trainable(model_for_meta)} loss={loss_profile}"
    )
    total_train_batches = train_batch_count(train_idx.numel(), args.batch, args.max_train_batches)
    total_val_batches = train_batch_count(val_idx.numel(), args.batch, args.max_val_batches)
    print(
        f"Batch={args.batch} | train_batches/epoch={total_train_batches} "
        f"val_batches/epoch={total_val_batches} | compile_backend={args.compile_backend if args.compile else 'off'}"
    )
    print(
        f"Aux ramp epochs={args.aux_ramp_epochs} | early_stop monitor={args.early_stop_monitor} "
        f"patience={args.early_stop_patience} mode={args.early_stop_mode}"
    )
    total_train_steps = max(1, total_train_batches * max(1, args.epochs))
    lr_scheduler = build_lr_scheduler(
        opt,
        schedule=args.lr_schedule,
        base_lr=args.lr,
        total_train_steps=total_train_steps,
        min_factor=args.lr_min_factor,
    )
    print(f"LR schedule={args.lr_schedule} | base_lr={args.lr:.3e}")
    if args.warmup:
        with torch.no_grad():
            warm_idx = train_idx[: min(args.batch, train_idx.numel())]
            _ = forward_losses(warm_idx, 0, aux_scale=1.0, update_ema=False)
            sync_if_needed(device)

    best_monitor = float("inf")
    stale_epochs = 0
    monitor_hist: list[float] = []
    stage = "geo"
    cycle_idx = 0
    stage_best = float("inf")
    stage_stale = 0
    stage_monitor_hist: list[float] = []
    if args.two_stage:
        set_two_stage_trainable(model_base, stage)
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        lr_scheduler = build_lr_scheduler(
            opt,
            schedule=args.lr_schedule,
            base_lr=args.lr,
            total_train_steps=total_train_steps,
            min_factor=args.lr_min_factor,
        )
        print(
            f"Two-stage enabled: cycles={args.stage_cycles}, stage-switch monitor={args.early_stop_monitor}, "
            f"patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta:g}"
        )

    for ep in range(args.epochs):
        aux_scale = 0.1 if (args.two_stage and stage == "geo") else (1.0 if args.two_stage else aux_scale_schedule(ep, args.aux_ramp_epochs))
        model.train()
        perm_idx = torch.randperm(train_idx.numel(), device="cpu")
        perm = train_idx[perm_idx]
        run_loss = torch.zeros((), device=device, dtype=torch.float32)
        run_geo = torch.zeros((), device=device, dtype=torch.float32)
        n_batches = 0
        t_epoch = time.perf_counter()
        for s in range(0, perm.numel(), args.batch):
            if args.max_train_batches is not None and n_batches >= args.max_train_batches:
                break
            sl = perm[s : s + args.batch]
            opt.zero_grad(set_to_none=True)
            loss, parts = forward_losses(sl, ep, aux_scale=aux_scale, update_ema=True)
            loss.backward()
            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            run_loss = run_loss + loss.detach()
            run_geo = run_geo + parts["geo"].detach()
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
        v_loss = torch.zeros((), device=device, dtype=torch.float32)
        v_geo = torch.zeros((), device=device, dtype=torch.float32)
        with torch.no_grad():
            for vi, s in enumerate(range(0, val_idx.numel(), args.batch)):
                if args.max_val_batches is not None and vi >= args.max_val_batches:
                    break
                sl = val_idx[s : s + args.batch]
                lf, parts = forward_losses(sl, ep, aux_scale=aux_scale, update_ema=False)
                v_loss = v_loss + lf.detach()
                v_geo = v_geo + parts["geo"].detach()
        v_n = total_val_batches
        v_loss = (v_loss / v_n).item()
        v_geo = (v_geo / v_n).item()
        if n_batches:
            run_loss = (run_loss / n_batches).item()
            run_geo = (run_geo / n_batches).item()
        else:
            run_loss = float(run_loss.item())
            run_geo = float(run_geo.item())

        history.append(
            {
                "epoch": ep + 1,
                "train_loss": run_loss,
                "train_geo_mae": run_geo,
                "val_loss": v_loss,
                "val_geo_mae": v_geo,
            }
        )
        print(
            f"epoch {ep + 1}/{args.epochs}{(' stage=' + stage) if args.two_stage else ''} aux={aux_scale:.2f} train_loss={run_loss:.5e} "
            f"val_loss={v_loss:.5e} val_geo_mae={v_geo:.5e} lr={opt.param_groups[0]['lr']:.3e}"
        )
        monitor_val = v_geo if args.early_stop_monitor == "val_geo" else v_loss
        monitor_hist.append(float(monitor_val))
        if monitor_val < (best_monitor - args.early_stop_min_delta):
            best_monitor = monitor_val
            stale_epochs = 0
        else:
            stale_epochs += 1
        if args.two_stage:
            stage_monitor_hist.append(float(monitor_val))
            if monitor_val < (stage_best - args.early_stop_min_delta):
                stage_best = monitor_val
                stage_stale = 0
            else:
                stage_stale += 1
            stage_should_switch = False
            if args.early_stop_mode == "fluctuation":
                stage_should_switch = fluctuation_plateau(
                    stage_monitor_hist, args.early_stop_window, args.early_stop_fluctuation_tol
                )
            elif args.early_stop_patience > 0 and stage_stale >= args.early_stop_patience:
                stage_should_switch = True
            if stage_should_switch:
                if stage == "geo":
                    stage = "aux"
                    print(f"[two-stage] stage-early-stop -> switch to aux at epoch {ep + 1}")
                else:
                    cycle_idx += 1
                    if cycle_idx >= args.stage_cycles:
                        print(f"[two-stage] completed {cycle_idx} cycle(s); stopping.")
                        break
                    stage = "geo"
                    print(f"[two-stage] stage-early-stop -> switch to geo at epoch {ep + 1} (cycle {cycle_idx + 1}/{args.stage_cycles})")
                stage_best = float("inf")
                stage_stale = 0
                stage_monitor_hist = []
                set_two_stage_trainable(model_base, stage)
                opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
                lr_scheduler = build_lr_scheduler(
                    opt,
                    schedule=args.lr_schedule,
                    base_lr=args.lr,
                    total_train_steps=max(1, total_train_batches * max(1, args.epochs - ep - 1)),
                    min_factor=args.lr_min_factor,
                )
                continue
        else:
            should_stop = False
            if args.early_stop_mode == "fluctuation":
                should_stop = fluctuation_plateau(monitor_hist, args.early_stop_window, args.early_stop_fluctuation_tol)
            elif args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
                should_stop = True
            if should_stop:
                if args.early_stop_mode == "fluctuation":
                    print(
                        f"Early stopping at epoch {ep + 1}: {args.early_stop_monitor} fluctuated within "
                        f"{args.early_stop_fluctuation_tol:g} for {args.early_stop_window} epoch(s)."
                    )
                else:
                    print(
                        f"Early stopping at epoch {ep + 1}: {args.early_stop_monitor} did not improve by "
                        f"{args.early_stop_min_delta:g} for {args.early_stop_patience} epoch(s)."
                    )
                break

    meta = {
        "run_id": run_key,
        "category": cat,
        "slug": slug,
        "spec": {k: v for k, v in spec.items() if k != "train"},
        "train": train_cfg,
        "n_outer": model_for_meta.n_outer,
        "n_inner": model_for_meta.n_inner,
        "d": model_for_meta.d,
        "encoder_ckpts": [str(x) for x in encoder_ckpts],
        "train_rows": train_rows.cpu(),
        "val_rows": val_rows.cpu(),
        "test_rows": test_rows.cpu(),
        "trainable_params": count_trainable(model_for_meta),
        "epochs": args.epochs,
        "frac_train": frac_train,
        "frac_val": frac_val,
        "seed": args.seed,
        "variant": run_key,
    }
    ckpt_path = run_dir / "model.pt"
    torch.save({"model": model_for_meta.state_dict(), "meta": meta}, ckpt_path)
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=0)
    first_low = next((h["epoch"] for h in history if h["val_geo_mae"] < 0.02), args.epochs)
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_key,
                "epoch": args.epochs,
                "final_val_geo_mae": history[-1]["val_geo_mae"],
                "convergence_epoch_val_geo_mae_below_0.02": first_low,
            },
            f,
            indent=0,
        )
    print(f"Saved checkpoint -> {ckpt_path}")

    fd = figures_dir()
    safe = f"{cat}__{slug}".replace("/", "_")
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val loss")
    axes[0].set_ylabel("loss")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"WHISP ablation train/val — {run_key}")
    axes[1].plot(ep, [h["val_geo_mae"] for h in history], color="C2", label="val CST MAE")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val geo MAE")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fd / f"ablation_train_curves_{safe}.png", dpi=220)
    plt.close(fig)
    print(f"Saved figure -> {fd / f'ablation_train_curves_{safe}.png'}")


if __name__ == "__main__":
    main()
