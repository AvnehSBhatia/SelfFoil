"""Kulfan CST with LEM: 18D = 8 lower + 8 upper + LE weight + TE thickness."""

from __future__ import annotations

import torch
import torch.nn as nn


def _bernstein_basis(psi: torch.Tensor, degree: int) -> torch.Tensor:
    """psi in (0,1); returns (len(psi), degree + 1)."""
    n = degree
    one_m = 1.0 - psi
    # B_{n,k}(psi) = C(n,k) psi^k (1-psi)^{n-k}
    B = torch.zeros(psi.shape[0], n + 1, dtype=psi.dtype, device=psi.device)
    for k in range(n + 1):
        logc = torch.lgamma(torch.tensor(n + 1.0, device=psi.device, dtype=psi.dtype))
        logc -= torch.lgamma(torch.tensor(k + 1.0, device=psi.device, dtype=psi.dtype))
        logc -= torch.lgamma(torch.tensor(n - k + 1.0, device=psi.device, dtype=psi.dtype))
        c = torch.exp(logc)
        B[:, k] = c * (psi**k) * (one_m ** (n - k))
    return B


def _class_function(psi: torch.Tensor, n1: float, n2: float) -> torch.Tensor:
    psi = psi.clamp(min=1e-8, max=1.0 - 1e-8)
    return (psi**n1) * ((1.0 - psi) ** n2)


def _split_upper_lower(xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """xy (N,2) TE-upper->...->LE->...->TE-lower. Returns upper (TE->LE, excludes LE) and lower (LE->TE)."""
    x = xy[:, 0]
    i = int(torch.argmin(x))
    upper = xy[:i]  # TE upper .. point before LE
    lower = xy[i:]  # LE .. TE lower
    return upper, lower


def _chordwise_psi(x: torch.Tensor) -> torch.Tensor:
    """Map monotonic x from LE (small) to TE (1) to psi in [0,1]."""
    x0, x1 = x[0], x[-1]
    span = (x1 - x0).clamp(min=1e-9)
    return ((x - x0) / span).clamp(0.0, 1.0)


def _kulfan_matrix_rows(
    x: torch.Tensor,
    is_upper: torch.Tensor,
    n_weights_per_side: int,
    n1: float,
    n2: float,
) -> torch.Tensor:
    """
    Build least-squares matrix rows for Kulfan CST+LEM.
    Unknown ordering:
        [lower_weights(8), upper_weights(8), leading_edge_weight, TE_thickness]
    """
    degree = n_weights_per_side - 1
    C = _class_function(x, n1, n2)
    S = _bernstein_basis(x, degree)  # (N, n_weights_per_side)

    lower_block = torch.where(
        is_upper.unsqueeze(1),
        torch.zeros_like(S),
        C.unsqueeze(1) * S,
    )
    upper_block = torch.where(
        is_upper.unsqueeze(1),
        C.unsqueeze(1) * S,
        torch.zeros_like(S),
    )

    le_col = (x * torch.clamp(1.0 - x, min=0.0) ** (n_weights_per_side + 0.5)).unsqueeze(1)
    te_col = torch.where(is_upper, x / 2.0, -x / 2.0).unsqueeze(1)

    return torch.cat([lower_block, upper_block, le_col, te_col], dim=1)


def fit_cst18_from_xy(
    xy: torch.Tensor,
    n_weights_per_side: int = 8,
    n1: float = 0.5,
    n2: float = 1.0,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """
    xy: (N,2) airfoil polyline in chord order (same convention as CSV).
    Returns (18,) = [lower(8), upper(8), leading_edge_weight, TE_thickness].
    """
    x = xy[:, 0].clamp(0.0, 1.0)
    y = xy[:, 1]
    i_le = int(torch.argmin(x))
    is_upper = torch.arange(x.shape[0], device=x.device) <= i_le

    A = _kulfan_matrix_rows(x, is_upper, n_weights_per_side, n1, n2)
    I = torch.eye(A.shape[1], dtype=xy.dtype, device=xy.device)
    coeffs = torch.linalg.solve(A.T @ A + ridge * I, A.T @ y)

    # Enforce non-negative TE thickness; refit without TE column if needed.
    if float(coeffs[-1]) < 0.0:
        A2 = A[:, :-1]
        I2 = torch.eye(A2.shape[1], dtype=xy.dtype, device=xy.device)
        c2 = torch.linalg.solve(A2.T @ A2 + ridge * I2, A2.T @ y)
        coeffs = torch.cat([c2, torch.zeros(1, dtype=xy.dtype, device=xy.device)], dim=0)

    return coeffs


class CSTEncoder18(nn.Module):
    """
    Non-trainable encoder: flattened coords -> 18 CST coefficients (9 upper + 9 lower Bernstein terms).
    Uses the same panel ordering as `data/original.csv` (TE upper .. LE .. TE lower).
    """

    def __init__(
        self,
        coord_dim: int,
        n_weights_per_side: int = 8,
        n1: float = 0.5,
        n2: float = 1.0,
        ridge: float = 1e-6,
    ) -> None:
        super().__init__()
        if coord_dim % 2 != 0:
            raise ValueError("coord_dim must be even (x,y pairs)")
        self.coord_dim = coord_dim
        self.n_weights_per_side = n_weights_per_side
        self.n1 = n1
        self.n2 = n2
        self.ridge = ridge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, coord_dim); no trainable weights.
        batch = []
        for b in range(x.shape[0]):
            xy = x[b].view(-1, 2)
            z = fit_cst18_from_xy(
                xy,
                n_weights_per_side=self.n_weights_per_side,
                n1=self.n1,
                n2=self.n2,
                ridge=self.ridge,
            )
            batch.append(z)
        return torch.stack(batch, dim=0)


class CSTDecoder18(nn.Module):
    """Analytic CST decoder: (18 coeffs + x-grid) -> flattened (x,y) coordinates."""

    def __init__(self, n_weights_per_side: int = 8, n1: float = 0.5, n2: float = 1.0) -> None:
        super().__init__()
        self.n_weights_per_side = n_weights_per_side
        self.n1 = n1
        self.n2 = n2

    def _decode_single(self, z: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        # z: [lower(8), upper(8), leading_edge_weight, TE_thickness]
        n = self.n_weights_per_side
        lower = z[:n]
        upper = z[n : 2 * n]
        le_weight = z[2 * n]
        te_thickness = z[2 * n + 1]

        x = x_coords.clamp(0.0, 1.0)
        i_le = int(torch.argmin(x))
        is_upper = torch.arange(x.shape[0], device=x.device) <= i_le
        C = _class_function(x, self.n1, self.n2)
        S = _bernstein_basis(x, n - 1)

        y = torch.zeros_like(x)
        y[is_upper] = (C[is_upper].unsqueeze(1) * S[is_upper]) @ upper
        y[~is_upper] = (C[~is_upper].unsqueeze(1) * S[~is_upper]) @ lower

        y = y + le_weight * x * torch.clamp(1.0 - x, min=0.0) ** (n + 0.5)
        y = y + torch.where(is_upper, x * te_thickness / 2.0, -x * te_thickness / 2.0)

        xy = torch.stack([x_coords, y], dim=1)
        return xy.reshape(-1)

    def forward(self, z: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        # z: (B,18), x_coords: (B,N)
        if z.ndim != 2 or x_coords.ndim != 2:
            raise ValueError("Expected z=(B,18) and x_coords=(B,N).")
        if z.shape[0] != x_coords.shape[0]:
            raise ValueError("Batch size mismatch between z and x_coords.")

        outs = []
        for b in range(z.shape[0]):
            outs.append(self._decode_single(z[b], x_coords[b]))
        return torch.stack(outs, dim=0)
