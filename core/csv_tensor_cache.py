"""Build and load CPU tensor bundles from original.csv-style rows (one JSON parse pass)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from .cst_kulfan import fit_cst18_from_xy_batched


def cache_stale(csv_path: Path, cache_path: Path) -> bool:
    if not cache_path.is_file():
        return True
    return csv_path.stat().st_mtime > cache_path.stat().st_mtime


def _fill_cst18_batched(
    cst18: torch.Tensor,
    xy_list: list[torch.Tensor],
    *,
    chunk_size: int = 512,
) -> None:
    """Write CST rows into cst18[in_rows] using batched least squares."""
    by_len: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for i, xy in enumerate(xy_list):
        by_len[int(xy.shape[0])].append((i, xy))
    for _length, items in by_len.items():
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            batch_xy = torch.stack([t[1] for t in chunk], dim=0)
            out = fit_cst18_from_xy_batched(batch_xy)
            row_ids = torch.tensor([t[0] for t in chunk], dtype=torch.long)
            cst18[row_ids] = out


def build_tensor_bundle(rows: list[dict]) -> dict[str, torch.Tensor | int]:
    n = len(rows)
    polar_lens_list: list[int] = []
    for row in rows:
        polar_lens_list.append(len(json.loads(row["alpha"])))
    polar_lens = torch.tensor(polar_lens_list, dtype=torch.int32)
    polar_ptr = torch.zeros(n + 1, dtype=torch.int64)
    if n:
        polar_ptr[1:] = torch.cumsum(polar_lens.to(torch.int64), dim=0)
    total_polar = int(polar_ptr[-1].item()) if n else 0

    alpha_flat = torch.empty(total_polar, dtype=torch.float32)
    cl_flat = torch.empty(total_polar, dtype=torch.float32)
    cd_flat = torch.empty(total_polar, dtype=torch.float32)
    mach_flat = torch.empty(total_polar, dtype=torch.float32)
    re_log_flat = torch.empty(total_polar, dtype=torch.float32)
    re_flat = torch.empty(total_polar, dtype=torch.float32)
    polar_row_idx = torch.empty(total_polar, dtype=torch.int64)

    off = 0
    coord_rows: list[torch.Tensor] = []
    xy_list: list[torch.Tensor] = []
    x_coords_template: torch.Tensor | None = None

    for row_i, row in enumerate(rows):
        alpha = json.loads(row["alpha"])
        cl = json.loads(row["Cl"])
        cd = json.loads(row["Cd"])
        t = len(alpha)
        mach_v = float(row["mach"])
        re_v = float(row["Re"])
        re_log = float(torch.log10(torch.tensor(re_v, dtype=torch.float32)).item())

        alpha_flat[off : off + t] = torch.tensor(alpha, dtype=torch.float32)
        cl_flat[off : off + t] = torch.tensor(cl, dtype=torch.float32)
        cd_flat[off : off + t] = torch.tensor(cd, dtype=torch.float32)
        mach_flat[off : off + t] = mach_v
        re_log_flat[off : off + t] = re_log
        re_flat[off : off + t] = re_v
        polar_row_idx[off : off + t] = row_i
        off += t

        xy = torch.tensor(json.loads(row["coords"]), dtype=torch.float32).reshape(-1, 2)
        coord_rows.append(xy.reshape(-1))
        xy_list.append(xy)
        if x_coords_template is None:
            x_coords_template = xy[:, 0].clone()

    coords = torch.stack(coord_rows, dim=0)
    cst18 = torch.empty(n, 18, dtype=torch.float32)
    if n:
        _fill_cst18_batched(cst18, xy_list)
    assert x_coords_template is not None
    return {
        "alpha_flat": alpha_flat,
        "cl_flat": cl_flat,
        "cd_flat": cd_flat,
        "mach_flat": mach_flat,
        "re_log_flat": re_log_flat,
        "re_flat": re_flat,
        "polar_row_idx": polar_row_idx,
        "polar_lens": polar_lens,
        "polar_ptr": polar_ptr,
        "coords": coords,
        "cst18": cst18,
        "x_coords": x_coords_template,
        "n_rows": n,
        "total_polar": total_polar,
    }


def load_or_build_cache(
    csv_path: Path,
    cache_path: Path,
    max_rows: int | None,
    rebuild: bool,
) -> dict[str, torch.Tensor | int]:
    if not rebuild and not cache_stale(csv_path, cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    with csv_path.open(newline="") as f:
        data = list(csv.DictReader(f))
    if max_rows is not None:
        data = data[:max_rows]

    bundle = build_tensor_bundle(data)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    return bundle
