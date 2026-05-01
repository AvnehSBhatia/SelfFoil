"""Kulfan CST with LEM: 18D = 8 lower + 8 upper + LE weight + TE thickness."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _kulfan_matrix_rows_batched(
    x: torch.Tensor,
    is_upper: torch.Tensor,
    n_weights_per_side: int,
    n1: float,
    n2: float,
) -> torch.Tensor:
    """x (B,N), is_upper (B,N) -> A (B,N,18)."""
    B, N = x.shape
    degree = n_weights_per_side - 1
    xf = x.reshape(-1)
    C = _class_function(xf, n1, n2).reshape(B, N)
    S = _bernstein_basis(xf, degree).reshape(B, N, degree + 1)

    lower_block = torch.where(
        is_upper.unsqueeze(-1),
        torch.zeros(B, N, degree + 1, dtype=x.dtype, device=x.device),
        C.unsqueeze(-1) * S,
    )
    upper_block = torch.where(
        is_upper.unsqueeze(-1),
        C.unsqueeze(-1) * S,
        torch.zeros(B, N, degree + 1, dtype=x.dtype, device=x.device),
    )
    le_col = (x * torch.clamp(1.0 - x, min=0.0) ** (n_weights_per_side + 0.5)).unsqueeze(-1)
    te_col = torch.where(is_upper, x / 2.0, -x / 2.0).unsqueeze(-1)
    return torch.cat([lower_block, upper_block, le_col, te_col], dim=-1)


def _infer_upper_mask_from_branches(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Infer upper/lower branch membership from contour branch means.

    The contour is expected in TE -> LE -> TE order, but either branch can come first.
    """
    n = int(x.shape[0])
    i_le = int(torch.argmin(x).item())
    j = torch.arange(n, device=x.device, dtype=torch.long)
    branch_a = j <= i_le
    branch_b = ~branch_a
    # Robust fallback if LE is at an endpoint (degenerate ordering).
    if not branch_b.any():
        branch_b = j >= i_le
    mean_a = y[branch_a].mean()
    mean_b = y[branch_b].mean()
    if mean_a >= mean_b:
        return branch_a
    return branch_b


def _infer_upper_mask_from_branches_batched(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vectorized branch inference for TE->LE->TE contours."""
    B, N = x.shape
    i_le = x.argmin(dim=1)
    j = torch.arange(N, device=x.device, dtype=torch.long).view(1, N).expand(B, N)
    branch_a = j <= i_le.unsqueeze(1)
    branch_b = ~branch_a

    # Degenerate safeguard: if LE is at one end, ensure both masks have support.
    cnt_b = branch_b.sum(dim=1)
    deg = cnt_b == 0
    if deg.any():
        alt_b = j >= i_le.unsqueeze(1)
        branch_b = torch.where(deg.unsqueeze(1), alt_b, branch_b)
        branch_a = ~branch_b

    cnt_a = branch_a.sum(dim=1).clamp_min(1)
    cnt_b = branch_b.sum(dim=1).clamp_min(1)
    mean_a = (y * branch_a.to(y.dtype)).sum(dim=1) / cnt_a.to(y.dtype)
    mean_b = (y * branch_b.to(y.dtype)).sum(dim=1) / cnt_b.to(y.dtype)
    a_is_upper = mean_a >= mean_b
    return torch.where(a_is_upper.unsqueeze(1), branch_a, branch_b)


def fit_cst18_from_xy_batched(
    xy: torch.Tensor,
    n_weights_per_side: int = 8,
    n1: float = 0.5,
    n2: float = 1.0,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """
    Batched ridge-normal least squares for Kulfan CST+LEM.

    xy: (B, N, 2) chord-ordered polylines (same convention as CSV).
    Returns (B, 18) = [lower(8), upper(8), leading_edge_weight, TE_thickness] per row.
    """
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("fit_cst18_from_xy_batched expects xy with shape (B, N, 2).")
    B, N, _ = xy.shape
    x = xy[..., 0].clamp(0.0, 1.0)
    y = xy[..., 1]
    is_upper = _infer_upper_mask_from_branches_batched(x, y)

    A = _kulfan_matrix_rows_batched(x, is_upper, n_weights_per_side, n1, n2)
    d = A.shape[-1]
    AtA = torch.bmm(A.transpose(1, 2), A)
    rhs = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1))
    eye = torch.eye(d, dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, d, d)
    coeffs = torch.linalg.solve(AtA + ridge * eye, rhs).squeeze(-1)

    bad = coeffs[:, -1] < 0.0
    if bad.any():
        d2 = d - 1
        A2 = A[bad, :, :-1]
        y2 = y[bad]
        AtA2 = torch.bmm(A2.transpose(1, 2), A2)
        rhs2 = torch.bmm(A2.transpose(1, 2), y2.unsqueeze(-1))
        nb = A2.shape[0]
        eye2 = torch.eye(d2, dtype=x.dtype, device=x.device).unsqueeze(0).expand(nb, d2, d2)
        c2 = torch.linalg.solve(AtA2 + ridge * eye2, rhs2).squeeze(-1)
        coeffs = coeffs.clone()
        coeffs[bad, :d2] = c2
        coeffs[bad, -1] = 0.0

    return coeffs


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
    return fit_cst18_from_xy_batched(
        xy.unsqueeze(0),
        n_weights_per_side=n_weights_per_side,
        n1=n1,
        n2=n2,
        ridge=ridge,
    ).squeeze(0)


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
        xy = x.view(x.shape[0], -1, 2)
        return fit_cst18_from_xy_batched(
            xy,
            n_weights_per_side=self.n_weights_per_side,
            n1=self.n1,
            n2=self.n2,
            ridge=self.ridge,
        )


class CSTDecoder18(nn.Module):
    """Analytic CST decoder: (18 coeffs + x-grid) -> flattened (x,y) coordinates."""

    def __init__(
        self,
        n_weights_per_side: int = 8,
        n1: float = 0.5,
        n2: float = 1.0,
        first_branch: str = "upper",
    ) -> None:
        super().__init__()
        self.n_weights_per_side = n_weights_per_side
        self.n1 = n1
        self.n2 = n2
        if first_branch not in ("lower", "upper"):
            raise ValueError("first_branch must be 'lower' or 'upper'.")
        self.first_branch = first_branch

    def _decode_single(self, z: torch.Tensor, x_coords: torch.Tensor) -> torch.Tensor:
        # z: [lower(8), upper(8), leading_edge_weight, TE_thickness]
        n = self.n_weights_per_side
        lower = z[:n]
        upper = z[n : 2 * n]
        le_weight = z[2 * n]
        te_thickness = z[2 * n + 1]

        x = x_coords.clamp(0.0, 1.0)
        i_le = int(torch.argmin(x))
        branch_a = torch.arange(x.shape[0], device=x.device) <= i_le
        if self.first_branch == "upper":
            is_upper = branch_a
        else:
            is_upper = ~branch_a
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


def build_analytic_cst_decoder(models_dir: Path, device: torch.device) -> CSTDecoder18:
    """Load `first_branch` from `decoder_coords.pt` if present; else default upper (matches training script)."""
    meta_path = models_dir / "decoder_coords.pt"
    first_branch: str = "upper"
    if meta_path.is_file():
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict) and meta.get("first_branch") in ("upper", "lower"):
            first_branch = str(meta["first_branch"])
    dec = CSTDecoder18(n_weights_per_side=8, n1=0.5, n2=1.0, first_branch=first_branch)
    return dec.to(device).eval()


def coord_geo_loss_from_cst(
    decoder: CSTDecoder18,
    cst_pred: torch.Tensor,
    coords_gt_flat: torch.Tensor,
    *,
    loss: str,
    huber_delta: float,
) -> torch.Tensor:
    """Geometry loss in flattened (x,y) space after analytic CST decode (not coefficient MAE)."""
    if cst_pred.ndim != 2 or cst_pred.shape[-1] != 18:
        raise ValueError(f"Expected cst_pred (B, 18), got {tuple(cst_pred.shape)}")
    bsz = cst_pred.shape[0]
    if coords_gt_flat.shape[0] != bsz:
        raise ValueError("Batch size mismatch between cst_pred and coords_gt_flat.")
    if coords_gt_flat.shape[1] % 2 != 0:
        raise ValueError("coords_gt_flat must have an even last dim (x,y pairs).")
    n_pts = coords_gt_flat.shape[1] // 2
    x_grid = coords_gt_flat.view(bsz, n_pts, 2)[:, :, 0].contiguous()
    pred_flat = decoder(cst_pred, x_grid)
    if loss == "huber":
        return F.huber_loss(pred_flat, coords_gt_flat, delta=huber_delta, reduction="mean")
    if loss == "mae":
        return (pred_flat - coords_gt_flat).abs().mean()
    raise ValueError(f"Unknown loss={loss!r} (use 'mae' or 'huber').")
