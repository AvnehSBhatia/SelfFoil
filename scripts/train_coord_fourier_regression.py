#!/usr/bin/env python3
"""
Optional supervised run: **coordinate** MLP → match 50D FFT targets.

This is NOT the polar ``[Cl, Cd, α, Re, Ma] → 16D`` embedding; that is learned end-to-end inside
:class:`~core.regime_transformer.RegimeConditionedAeroTransformer` as :class:`~core.polar_token_embedding.PolarTokenEmbedding`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.airfoil_embedding import AirfoilFourierEmbedding
from core.coord_fourier_regression import CoordFourierRegressionMLP
from core.dataset import CoordFourierSupervisedDataset, make_coord_fourier_collate_fn
from core.device import (
    coord_fourier_batch_to_device,
    dataloader_pin_memory,
    default_training_device,
)


@torch.inference_mode()
def evaluate(
    model: CoordFourierRegressionMLP, loader: DataLoader, dev: torch.device, *, nb: bool
) -> float:
    model.eval()
    t, s = 0.0, 0
    for batch in loader:
        batch = coord_fourier_batch_to_device(batch, dev, non_blocking=nb)
        t += F.mse_loss(model(batch["coords"]), batch["target_fourier"], reduction="sum").item()
        s += batch["coords"].size(0)
    return t / max(s, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=_REPO_ROOT / "data" / "train.csv")
    ap.add_argument("--val", type=Path, default=_REPO_ROOT / "data" / "val.csv")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "coord_fourier_regression.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument(
        "--device",
        type=str,
        default=str(default_training_device()),
        help="cuda, mps, or cpu; default prefers CUDA/MPS when available.",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    if not args.train.is_file() or not args.val.is_file():
        raise SystemExit("Missing data/train.csv or data/val.csv (same format as data/test.csv).")

    dev = torch.device(args.device)
    pin = dataloader_pin_memory(dev)
    nb = pin

    fourier = AirfoilFourierEmbedding()
    coord_collate = make_coord_fourier_collate_fn(fourier)

    tr = CoordFourierSupervisedDataset(
        args.train, max_rows=args.max_train_rows, fourier_engine=fourier
    )
    va = CoordFourierSupervisedDataset(
        args.val, max_rows=args.max_val_rows, fourier_engine=fourier
    )

    kw = dict(collate_fn=coord_collate, num_workers=args.num_workers, pin_memory=pin)
    if args.num_workers > 0:
        kw["persistent_workers"] = True

    tr_l = DataLoader(tr, batch_size=args.batch_size, shuffle=True, **kw)
    va_l = DataLoader(va, batch_size=args.batch_size, shuffle=False, **kw)

    model = CoordFourierRegressionMLP(hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for batch in tr_l:
            batch = coord_fourier_batch_to_device(batch, dev, non_blocking=nb)
            opt.zero_grad()
            y = model(batch["coords"])
            loss = F.mse_loss(y, batch["target_fourier"])
            loss.backward()
            opt.step()
            run += loss.item() * batch["coords"].size(0)
            n += batch["coords"].size(0)
        v = evaluate(model, va_l, dev, nb=nb)
        print(f"epoch {epoch}  train MSE {run/max(n,1):.6e}  val MSE {v:.6e}")
        if v < best:
            best = v
            torch.save(
                {"state_dict": model.state_dict(), "val_mse": v, "hidden": args.hidden},
                args.out,
            )
    print(f"best val MSE {best:.6e} → {args.out}")


if __name__ == "__main__":
    main()
