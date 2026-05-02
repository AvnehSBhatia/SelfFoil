"""CSV polar datasets matching ``data/test.csv`` schema."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .airfoil_embedding import AirfoilFourierEmbedding
from .constants import POLAR_DIM


def _parse_float_field(raw: str) -> float:
    return float(raw)


def _parse_float_list(raw: str) -> list[float]:
    return json.loads(raw)


def build_line_index(csv_path: str | Path, max_rows: int | None) -> tuple[str, list[int]]:
    """``(header_line, byte_offsets)`` for one-line-per-row UTF-8 CSVs."""
    offsets: list[int] = []
    path = Path(csv_path)
    with path.open("rb") as f:
        header_str = f.readline().decode("utf-8")
        while max_rows is None or len(offsets) < max_rows:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    return header_str, offsets


def read_row_by_index(csv_path: Path, header_str: str, byte_offsets: list[int], idx: int) -> dict[str, str]:
    with Path(csv_path).open("rb") as f:
        f.seek(byte_offsets[idx])
        line = f.readline().decode("utf-8")
    buf = io.StringIO(header_str + line)
    return next(csv.DictReader(buf))


class PolarAirfoilDataset(Dataset):
    """
    One CSV row = one airfoil with polar samples ``(N, 5)`` and Fourier target ``(50,)``.

    Columns: ``coords``, ``Re``, ``alpha``, ``Cl``, ``Cd``, ``mach``.
    ``alpha``, ``Cl``, ``Cd`` are JSON lists of equal length ``N``.

    Indexing assumes **one physical line per row** (no embedded newlines inside quoted fields).
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        max_rows: int | None = None,
        fourier_engine: AirfoilFourierEmbedding | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.fourier = fourier_engine or AirfoilFourierEmbedding()
        self._header_str, self._offsets = build_line_index(self.csv_path, max_rows)

    def __len__(self) -> int:
        return len(self._offsets)

    def _row_at(self, idx: int) -> dict[str, str]:
        return read_row_by_index(self.csv_path, self._header_str, self._offsets, idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._row_at(idx)
        alpha = _parse_float_list(row["alpha"])
        cl = _parse_float_list(row["Cl"])
        cd = _parse_float_list(row["Cd"])
        re = _parse_float_field(row["Re"])
        mach = _parse_float_field(row["mach"])
        n = len(alpha)
        if not (len(cl) == n and len(cd) == n):
            raise ValueError(f"Mismatched polar lengths in row {idx}")

        polar = np.stack(
            [cl, cd, alpha, np.full(n, re, dtype=np.float64), np.full(n, mach, dtype=np.float64)],
            axis=-1,
        )
        coords = np.asarray(json.loads(row["coords"]), dtype=np.float32)

        return {
            "polar": torch.from_numpy(polar.astype(np.float32)),
            "coords": torch.from_numpy(coords),
            "length": torch.tensor(n, dtype=torch.int64),
        }


def make_polar_collate_fn(
    fourier: AirfoilFourierEmbedding,
) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    """
    Batched analytic Fourier targets via :meth:`AirfoilFourierEmbedding.encode_batch`
    (vectorized ``np.fft.fft`` over the batch).
    """

    def polar_collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        max_n = max(int(b["length"]) for b in batch)
        bsz = len(batch)
        polar = torch.zeros(bsz, max_n, POLAR_DIM, dtype=torch.float32)
        mask = torch.ones(bsz, max_n, dtype=torch.bool)
        lengths = torch.stack([b["length"] for b in batch], dim=0)
        coords_np = np.stack([b["coords"].numpy() for b in batch], axis=0)
        targets = torch.as_tensor(
            fourier.encode_batch(coords_np, resample=True), dtype=torch.float32
        )

        for i, b in enumerate(batch):
            n = int(b["length"])
            polar[i, :n] = b["polar"]
            mask[i, :n] = False

        return {
            "polar": polar,
            "padding_mask": mask,
            "target_fourier": targets,
            "lengths": lengths,
        }

    return polar_collate_fn
