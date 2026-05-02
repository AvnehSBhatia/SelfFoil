#!/usr/bin/env python3
"""Supervised training of :class:`CoordToFourierMLP` to match the analytic airfoil Fourier target."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.dataset import CoordFourierSupervisedDataset
from core.neural_embedding import CoordToFourierMLP


@torch.inference_mode()
def evaluate(model: CoordToFourierMLP, loader: DataLoader, dev: torch.device) -> float:
    model.eval()
    t, s = 0.0, 0
    for b in loader:
        x, y = b["coords"].to(dev), b["target_fourier"].to(dev)
        t += F.mse_loss(model(x), y, reduction="sum").item()
        s += x.size(0)
    return t / max(s, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=_REPO_ROOT / "data" / "train.csv")
    ap.add_argument("--val", type=Path, default=_REPO_ROOT / "data" / "val.csv")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "coord_fourier_mlp.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    if not args.train.is_file() or not args.val.is_file():
        raise SystemExit("Missing data/train.csv or data/val.csv (same format as data/test.csv).")

    dev = torch.device(args.device)
    tr = CoordFourierSupervisedDataset(args.train, max_rows=args.max_train_rows)
    va = CoordFourierSupervisedDataset(args.val, max_rows=args.max_val_rows)

    def _collate(items: list) -> dict:
        c = torch.stack([it["coords"] for it in items], dim=0)
        t = torch.stack([it["target_fourier"] for it in items], dim=0)
        return {"coords": c, "target_fourier": t}

    tr_l = DataLoader(
        tr, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, num_workers=args.num_workers
    )
    va_l = DataLoader(
        va, batch_size=args.batch_size, shuffle=False, collate_fn=_collate, num_workers=args.num_workers
    )

    model = CoordToFourierMLP(hidden=args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for b in tr_l:
            x, yg = b["coords"].to(dev), b["target_fourier"].to(dev)
            opt.zero_grad()
            y = model(x)
            loss = F.mse_loss(y, yg)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        v = evaluate(model, va_l, dev)
        print(f"epoch {epoch}  train MSE {run/max(n,1):.6e}  val MSE {v:.6e}")
        if v < best:
            best = v
            torch.save({"state_dict": model.state_dict(), "val_mse": v, "hidden": args.hidden}, args.out)
    print(f"best val MSE {best:.6e} → {args.out}")


if __name__ == "__main__":
    main()
