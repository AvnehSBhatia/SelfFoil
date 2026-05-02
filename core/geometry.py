"""Closed-polygon sampling helpers for airfoil contours."""

from __future__ import annotations

import numpy as np


def remove_duplicate_closure(xy: np.ndarray) -> np.ndarray:
    if len(xy) >= 2 and np.allclose(xy[0], xy[-1]):
        return xy[:-1].copy()
    return xy.copy()


def resample_closed_poly(xy: np.ndarray, m: int) -> np.ndarray:
    """Uniform arc-length samples around closed polygon (m×2). Vectorized (no per-point Python)."""
    xy = remove_duplicate_closure(xy)
    n = len(xy)
    nxt = np.roll(xy, -1, axis=0)
    seg_len = np.linalg.norm(nxt - xy, axis=1)
    total = float(seg_len.sum())
    verts_dist = np.zeros(n, dtype=np.float64)
    if n > 1:
        verts_dist[1:] = np.cumsum(seg_len[:-1], dtype=np.float64)

    targets = (np.arange(m, dtype=np.float64) / m) * total
    if total <= 0.0 or n < 1:
        return np.tile(xy[:1], (m, 1))
    j = np.searchsorted(verts_dist, targets, side="right") - 1
    j = np.clip(j, 0, n - 1)
    d0 = verts_dist[j]
    sl = seg_len[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (targets - d0) / sl
    alpha = np.where(sl > 1e-15, alpha, 0.0)
    rows = (1.0 - alpha)[:, None] * xy[j] + alpha[:, None] * nxt[j]
    return rows


def resample_closed_poly_batched(
    xy: np.ndarray, m: int, *, resample: bool
) -> np.ndarray:
    """
    Parameters
    ----------
    xy
        ``(B, 250, 2)`` closed polylines (trailing point may duplicate the first).
    resample
        If False, return **xy** unchanged (already on a uniform grid).

    Returns
    -------
    array
        ``(B, m, 2)`` float64.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if not resample:
        return np.asarray(xy, dtype=np.float64)
    b = xy.shape[0]
    out = np.empty((b, m, 2), dtype=np.float64)
    for k in range(b):
        out[k] = resample_closed_poly(xy[k], m)
    return out
