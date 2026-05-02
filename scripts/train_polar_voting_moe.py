#!/usr/bin/env python3
"""Train :class:`PolarVotingMoETransformer` (5-expert MoE, 10-vote, 3 macro stages) on polar CSVs."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.airfoil_embedding import AirfoilFourierEmbedding
from core.dataset import PolarAirfoilDataset, make_polar_collate_fn
from core.polar_voting_moe import PolarVotingMoETransformer, fourier_mse_loss


@torch.inference_mode()
def evaluate(
    model: PolarVotingMoETransformer, loader: DataLoader, dev: torch.device, *, desc: str = "val"
) -> float:
    model.eval()
    total, seen = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        polar = batch["polar"].to(dev)
        m = batch["padding_mask"].to(dev)
        tgt = batch["target_fourier"].to(dev)
        lengths = batch["lengths"].to(dev)
        y = model(polar, m, lengths)
        loss = fourier_mse_loss(y, tgt).item() * polar.size(0)
        total += loss
        seen += polar.size(0)
    return total / max(seen, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=_REPO_ROOT / "data" / "train.csv")
    ap.add_argument("--val", type=Path, default=_REPO_ROOT / "data" / "val.csv")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "models" / "polar_voting_moe.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    if not args.train.is_file():
        raise SystemExit(f"Missing {args.train}")
    if not args.val.is_file():
        raise SystemExit(f"Missing {args.val}")

    dev = torch.device(args.device)
    fourier = AirfoilFourierEmbedding()
    polar_collate_fn = make_polar_collate_fn(fourier)
    tr_ds = PolarAirfoilDataset(args.train, max_rows=args.max_train_rows, fourier_engine=fourier)
    va_ds = PolarAirfoilDataset(args.val, max_rows=args.max_val_rows, fourier_engine=fourier)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise SystemExit("Empty train or val dataset.")

    tr_load = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=polar_collate_fn,
        num_workers=args.num_workers,
    )
    va_load = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=polar_collate_fn,
        num_workers=args.num_workers,
    )

    tqdm.write(
        f"train rows={len(tr_ds)}  val rows={len(va_ds)}  batches/train={len(tr_load)}  "
        f"device={dev}"
    )

    model = PolarVotingMoETransformer().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        run, n = 0.0, 0
        batch_bar = tqdm(
            tr_load,
            desc=f"train e{epoch}/{args.epochs}",
            leave=False,
            unit="batch",
        )
        for batch in batch_bar:
            polar = batch["polar"].to(dev)
            m = batch["padding_mask"].to(dev)
            tgt = batch["target_fourier"].to(dev)
            lengths = batch["lengths"].to(dev)
            opt.zero_grad()
            y = model(polar, m, lengths)
            loss = fourier_mse_loss(y, tgt)
            loss.backward()
            opt.step()
            li = loss.item()
            run += li * polar.size(0)
            n += polar.size(0)
            batch_bar.set_postfix(loss=f"{li:.4e}", avg=f"{run / max(n, 1):.4e}")

        tr_mse = run / max(n, 1)
        val_mse = evaluate(model, va_load, dev, desc=f"val e{epoch}/{args.epochs}")
        epoch_bar.set_postfix(train_mse=f"{tr_mse:.4e}", val_mse=f"{val_mse:.4e}")
        tqdm.write(f"epoch {epoch}/{args.epochs}  train MSE {tr_mse:.6e}  val MSE {val_mse:.6e}")
        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                },
                args.out,
            )
            tqdm.write(f"  → saved best val MSE {best:.6e} to {args.out}")

    tqdm.write(f"done. best val MSE {best:.6e} → {args.out}")


if __name__ == "__main__":
    main()
