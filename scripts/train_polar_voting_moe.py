#!/usr/bin/env python3
"""Train :class:`PolarVotingMoETransformer` (5-expert MoE, 10-vote, 3 macro stages) on polar CSVs."""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.airfoil_embedding import AirfoilFourierEmbedding
from core.dataset import PolarAirfoilDataset, make_polar_collate_fn
from core.polar_voting_moe import PolarVotingMoETransformer, fourier_mse_loss


def _move_polar_batch(
    batch: dict[str, torch.Tensor], dev: torch.device, *, non_blocking: bool
) -> dict[str, torch.Tensor]:
    return {k: v.to(dev, non_blocking=non_blocking) for k, v in batch.items()}


@torch.inference_mode()
def compute_normalization_stats(
    ds: PolarAirfoilDataset,
    base_collate,
    *,
    batch_size: int,
    max_rows: int | None,
) -> dict[str, torch.Tensor]:
    """
    Train-set stats for valid polar tokens and Fourier targets.

    This keeps raw Re/Mach/alpha and FFT coefficients from dominating the first
    few optimization steps. Padded polar slots stay zero after normalization.
    """
    n_rows = len(ds) if max_rows is None else min(len(ds), max_rows)
    if n_rows <= 0:
        raise ValueError("cannot compute stats from an empty dataset")

    polar_sum = torch.zeros(5, dtype=torch.float64)
    polar_sq = torch.zeros(5, dtype=torch.float64)
    polar_count = 0
    target_sum = torch.zeros(50, dtype=torch.float64)
    target_sq = torch.zeros(50, dtype=torch.float64)
    target_count = 0

    for start in tqdm(range(0, n_rows, batch_size), desc="stats", unit="batch"):
        items = [ds[i] for i in range(start, min(start + batch_size, n_rows))]
        batch = base_collate(items)
        valid = ~batch["padding_mask"]
        polar = batch["polar"][valid].double()
        target = batch["target_fourier"].double()

        polar_sum += polar.sum(dim=0)
        polar_sq += (polar * polar).sum(dim=0)
        polar_count += polar.size(0)
        target_sum += target.sum(dim=0)
        target_sq += (target * target).sum(dim=0)
        target_count += target.size(0)

    polar_mean = polar_sum / max(polar_count, 1)
    polar_var = (polar_sq / max(polar_count, 1)) - polar_mean.square()
    target_mean = target_sum / max(target_count, 1)
    target_var = (target_sq / max(target_count, 1)) - target_mean.square()

    return {
        "polar_mean": polar_mean.float(),
        "polar_std": polar_var.clamp_min(1e-12).sqrt().float().clamp_min(1e-6),
        "target_mean": target_mean.float(),
        "target_std": target_var.clamp_min(1e-12).sqrt().float().clamp_min(1e-6),
    }


def make_normalized_collate(base_collate, stats: dict[str, torch.Tensor]):
    polar_mean = stats["polar_mean"].view(1, 1, -1)
    polar_std = stats["polar_std"].view(1, 1, -1)
    target_mean = stats["target_mean"].view(1, -1)
    target_std = stats["target_std"].view(1, -1)

    def _collate(batch):
        out = base_collate(batch)
        valid = ~out["padding_mask"]
        out["polar"] = (out["polar"] - polar_mean) / polar_std
        out["polar"][~valid] = 0.0
        out["target_fourier"] = (out["target_fourier"] - target_mean) / target_std
        return out

    return _collate


@torch.inference_mode()
def evaluate(
    model: PolarVotingMoETransformer,
    loader: DataLoader,
    dev: torch.device,
    *,
    desc: str = "val",
    amp: bool = False,
    non_blocking: bool = False,
) -> float:
    model.eval()
    total, seen = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        batch = _move_polar_batch(batch, dev, non_blocking=non_blocking)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            y = model(batch["polar"], batch["padding_mask"], batch["lengths"])
            loss = fourier_mse_loss(y, batch["target_fourier"])
        loss = loss.item() * batch["polar"].size(0)
        total += loss
        seen += batch["polar"].size(0)
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
    ap.add_argument("--stats-rows", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    args = ap.parse_args()
    if not args.train.is_file():
        raise SystemExit(f"Missing {args.train}")
    if not args.val.is_file():
        raise SystemExit(f"Missing {args.val}")

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    pin = dev.type == "cuda"
    amp = dev.type == "cuda" and not args.no_amp
    nb = pin

    fourier = AirfoilFourierEmbedding()
    base_collate = make_polar_collate_fn(fourier)
    tr_ds = PolarAirfoilDataset(args.train, max_rows=args.max_train_rows, fourier_engine=fourier)
    va_ds = PolarAirfoilDataset(args.val, max_rows=args.max_val_rows, fourier_engine=fourier)
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise SystemExit("Empty train or val dataset.")

    stats = compute_normalization_stats(
        tr_ds, base_collate, batch_size=args.batch_size, max_rows=args.stats_rows
    )
    polar_collate_fn = make_normalized_collate(base_collate, stats)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    tr_load = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=polar_collate_fn,
        **loader_kwargs,
    )
    va_load = DataLoader(
        va_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=polar_collate_fn,
        **loader_kwargs,
    )

    tqdm.write(
        f"train rows={len(tr_ds)}  val rows={len(va_ds)}  batches/train={len(tr_load)}  "
        f"device={dev}  amp={amp}  workers={args.num_workers}"
    )
    tqdm.write(
        "normalized loss: zero-pred target MSE is about 1.0; "
        f"raw target |mean|={stats['target_mean'].abs().mean().item():.4g}, "
        f"target std mean={stats['target_std'].mean().item():.4g}"
    )

    raw_model = PolarVotingMoETransformer().to(dev)
    model = raw_model
    if args.compile:
        model = torch.compile(model)  # type: ignore[assignment]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
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
            batch = _move_polar_batch(batch, dev, non_blocking=nb)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                y = model(batch["polar"], batch["padding_mask"], batch["lengths"])
                loss = fourier_mse_loss(y, batch["target_fourier"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            li = loss.item()
            run += li * batch["polar"].size(0)
            n += batch["polar"].size(0)
            batch_bar.set_postfix(loss=f"{li:.4e}", avg=f"{run / max(n, 1):.4e}")

        tr_mse = run / max(n, 1)
        val_mse = evaluate(
            model,
            va_load,
            dev,
            desc=f"val e{epoch}/{args.epochs}",
            amp=amp,
            non_blocking=nb,
        )
        epoch_bar.set_postfix(train_mse=f"{tr_mse:.4e}", val_mse=f"{val_mse:.4e}")
        tqdm.write(f"epoch {epoch}/{args.epochs}  train MSE {tr_mse:.6e}  val MSE {val_mse:.6e}")
        if val_mse < best:
            best = val_mse
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                    "normalization": stats,
                },
                args.out,
            )
            tqdm.write(f"  → saved best val MSE {best:.6e} to {args.out}")

    tqdm.write(f"done. best val MSE {best:.6e} → {args.out}")


if __name__ == "__main__":
    main()
