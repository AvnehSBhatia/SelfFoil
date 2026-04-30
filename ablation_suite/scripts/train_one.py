#!/usr/bin/env python3
"""Train one WHISP ablation from `ablation_suite.catalog` (runs/<category>/<slug>/)."""

from __future__ import annotations

import argparse
import json
import random
import sys
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
    move_bundle,
    polar_mask_for_rows,
    required_bundle_keys,
)


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
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--suite-root", type=Path, default=ROOT / "ablation_suite")
    args = p.parse_args()

    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    configure_cuda_training(device, deterministic=args.deterministic)
    run_key = resolve_run_key(args.run_id, args.model)
    spec = get_spec(run_key)
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
    bundle = move_bundle(bundle_cpu, device)
    polar_row = bundle["polar_row_idx"]
    train_mask = polar_mask_for_rows(polar_row, train_rows)
    val_mask = polar_mask_for_rows(polar_row, val_rows)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise SystemExit("Empty train or val split.")

    re_log_all = bundle["re_log_flat"]
    if extrap and tr_max is not None and va_min is not None:
        tr_max_t = torch.tensor(float(tr_max), device=device, dtype=re_log_all.dtype)
        va_min_t = torch.tensor(float(va_min), device=device, dtype=re_log_all.dtype)
        train_idx = train_idx[re_log_all[train_idx] < tr_max_t]
        val_idx = val_idx[re_log_all[val_idx] >= va_min_t]
    if stride > 1:
        train_idx = train_idx[torch.arange(train_idx.numel(), device=device) % stride == 0]

    model = WHISPAblated(encoder_ckpts, spec).to(device)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    distill_w = float(spec.get("distill_weight", 0.0) or 0.0)

    def forward_losses(idx: torch.Tensor, ep: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cl = bundle["cl_flat"][idx]
        cd = bundle["cd_flat"][idx]
        re_log = bundle["re_log_flat"][idx]
        mach = bundle["mach_flat"][idx]
        alpha = bundle["alpha_flat"][idx]
        if noise_std > 0.0:
            cl = cl + torch.randn_like(cl) * noise_std
            cd = cd + torch.randn_like(cd) * noise_std
            re_log = re_log + torch.randn_like(re_log) * noise_std
            mach = mach + torch.randn_like(mach) * noise_std
            alpha = alpha + torch.randn_like(alpha) * noise_std
        row = polar_row[idx]
        cst_tgt = bundle["cst18"][row]

        cst_pred, aux = model(cl, cd, re_log, mach, alpha)
        l_geo = (cst_pred - cst_tgt).abs().mean()
        parts: dict[str, torch.Tensor] = {"geo": l_geo}
        l_ns_mean = torch.tensor(0.0, device=device)
        if model.use_physics and aux["L_ns_list"]:
            l_ns_mean = torch.stack(aux["L_ns_list"]).mean()
        parts["ns"] = l_ns_mean
        cl_keys = [k for k in aux if k.startswith("cl_gamma_")]
        last_k = max((int(k.split("_")[-1]) for k in cl_keys), default=model.n_outer - 1)
        l_cl_g = torch.tensor(0.0, device=device)
        if f"cl_gamma_{last_k}" in aux:
            l_cl_g = nn.functional.mse_loss(aux[f"cl_gamma_{last_k}"], cl)
        parts["clg"] = l_cl_g
        l_cl_head = nn.functional.mse_loss(aux["cl_direct"], cl)

        t = 1.0 - ep / max(1, args.epochs - 1) if anneal and args.epochs > 1 else 1.0
        lam_ns_ep = lam_ns * t
        lam_cl_ep = lam_cl * t

        if loss_profile == "geo_only":
            loss = l_geo
        elif loss_profile == "aero_only":
            loss = l_cl_g if model.use_physics and f"cl_gamma_{last_k}" in aux else l_cl_head
        elif loss_profile == "no_geo":
            loss = lam_ns_ep * l_ns_mean + lam_cl_ep * l_cl_g if model.use_physics else l_cl_head
        else:
            loss = l_geo
            if model.use_physics and aux["L_ns_list"]:
                loss = loss + lam_ns_ep * l_ns_mean + lam_cl_ep * l_cl_g
        if distill_w > 0.0 and "embed_distill" in aux:
            loss = loss + distill_w * aux["embed_distill"]
        return loss, parts

    cat, slug = run_path_parts(run_key)
    run_dir = args.suite_root / "runs" / cat / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    print(
        f"WHISP run={run_key} device={device} train={train_idx.numel()} val={val_idx.numel()} "
        f"trainable={count_trainable(model)} loss={loss_profile}"
    )

    for ep in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        run_loss = 0.0
        run_geo = 0.0
        n_batches = 0
        for s in range(0, perm.numel(), args.batch):
            sl = perm[s : s + args.batch]
            opt.zero_grad(set_to_none=True)
            loss, parts = forward_losses(sl, ep)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach())
            run_geo += float(parts["geo"].detach())
            n_batches += 1

        model.eval()
        v_loss = 0.0
        v_geo = 0.0
        with torch.no_grad():
            for s in range(0, val_idx.numel(), args.batch):
                sl = val_idx[s : s + args.batch]
                lf, parts = forward_losses(sl, ep)
                v_loss += float(lf)
                v_geo += float(parts["geo"])
        v_n = max(1, (val_idx.numel() + args.batch - 1) // args.batch)
        v_loss /= v_n
        v_geo /= v_n
        if n_batches:
            run_loss /= n_batches
            run_geo /= n_batches

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
            f"epoch {ep + 1}/{args.epochs} train_loss={run_loss:.5e} val_loss={v_loss:.5e} val_geo_mae={v_geo:.5e}"
        )

    meta = {
        "run_id": run_key,
        "category": cat,
        "slug": slug,
        "spec": {k: v for k, v in spec.items() if k != "train"},
        "train": train_cfg,
        "n_outer": model.n_outer,
        "n_inner": model.n_inner,
        "d": model.d,
        "encoder_ckpts": [str(x) for x in encoder_ckpts],
        "train_rows": train_rows.cpu(),
        "val_rows": val_rows.cpu(),
        "test_rows": test_rows.cpu(),
        "trainable_params": count_trainable(model),
        "epochs": args.epochs,
        "frac_train": frac_train,
        "frac_val": frac_val,
        "seed": args.seed,
        "variant": run_key,
    }
    ckpt_path = run_dir / "model.pt"
    torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
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
