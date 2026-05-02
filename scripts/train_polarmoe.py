#!/usr/bin/env python3
"""
Train :class:`~core.regime_transformer.RegimeConditionedAeroTransformer` on polar CSVs.

Example (CUDA):
    cd SelfFoil && python3 scripts/train_polarmoe.py --device cuda --epochs 10

CPU fallback:
    python3 scripts/train_polarmoe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.airfoil_embedding import AirfoilFourierEmbedding
from core.dataset import PolarAirfoilDataset, make_polar_collate_fn
from core.device import (
    dataloader_pin_memory,
    default_training_device,
    polar_batch_to_device,
)
from core.regime_transformer import RegimeConditionedAeroTransformer, fourier_mse_loss


@torch.inference_mode()
def evaluate(model: RegimeConditionedAeroTransformer, loader: DataLoader, dev: torch.device) -> float:
    model.eval()
    pin = dev.type == "cuda"
    total, seen = 0.0, 0
    for batch in loader:
        batch = polar_batch_to_device(batch, dev, non_blocking=pin)
        y = model(batch["polar"], batch["padding_mask"], batch["lengths"])
        loss = fourier_mse_loss(y, batch["target_fourier"]).item() * batch["polar"].size(0)
        total += loss
        seen += batch["polar"].size(0)
    return total / max(seen, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=_REPO_ROOT / "data" / "train.csv")
    ap.add_argument("--val", type=Path, default=_REPO_ROOT / "data" / "val.csv")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "regime_transformer.pt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument(
        "--device",
        type=str,
        default=str(default_training_device()),
        help="cuda, mps (Apple), or cpu; default prefers CUDA/MPS when available.",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    if not args.train.is_file():
        raise SystemExit(f"Missing {args.train} (place training CSV under data/train.csv).")
    if not args.val.is_file():
        raise SystemExit(f"Missing {args.val} (place validation CSV under data/val.csv).")

    dev = torch.device(args.device)
    pin = dataloader_pin_memory(dev)
    nb = pin

    # Shared Fourier engine for batched analytic targets on both splits.
    fourier = AirfoilFourierEmbedding()
    polar_collate = make_polar_collate_fn(fourier)

    tr_ds = PolarAirfoilDataset(args.train, max_rows=args.max_train_rows, fourier_engine=fourier)
    va_ds = PolarAirfoilDataset(args.val, max_rows=args.max_val_rows, fourier_engine=fourier)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise SystemExit("Empty train or val dataset — check CSV paths and file content.")

    tr_load = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=polar_collate,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    va_load = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=polar_collate,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = RegimeConditionedAeroTransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for batch in tr_load:
            batch = polar_batch_to_device(batch, dev, non_blocking=nb)
            opt.zero_grad()
            y = model(batch["polar"], batch["padding_mask"], batch["lengths"])
            loss = fourier_mse_loss(y, batch["target_fourier"])
            loss.backward()
            opt.step()
            run += loss.item() * batch["polar"].size(0)
            n += batch["polar"].size(0)
        tr_mse = run / max(n, 1)
        val_mse = evaluate(model, va_load, dev)
        print(f"epoch {epoch}  train MSE {tr_mse:.6e}  val MSE {val_mse:.6e}")
        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_mse": val_mse,
                    "epoch": epoch,
                    "architecture": "RegimeConditionedAeroTransformer",
                },
                args.out,
            )
    print(f"best val MSE {best:.6e} → saved {args.out}")


if __name__ == "__main__":
    main()
