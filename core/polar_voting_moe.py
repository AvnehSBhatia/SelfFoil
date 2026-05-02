"""
Polar sequence (Cl, Cd, α, Re, Ma) → 50D Fourier-style target.

Architecture (user spec):
- Per-token 5D → 16D embed.
- Expert block: self-attn → outer-product + tanh + row-weighted mix (vector u) →
  16→64→16 FFN → second outer mix (vector v).
- 5 such experts in parallel; router on concatenated last valid 16D states → softmax 5 → fuse.
- 10 independent MoE stacks in parallel; vote router on 10×16 last states → softmax 10 → fuse sequences.
- Two pressure branches on N×16; first 4 dims per token give ∫₀¹ Taylor poly (deg ≤ 3); upper−lower
  vs Cl → residual through 1→8→16 MLP, added to both branch tensors.
- Last-vector router (2-way softmax) fuses the two branches.
- Above macro block is duplicated (MoE10 → pressure/residual/fuse → MoE10 again).
- Full macro repeated ``N_STAGE`` times (default 3).
- Final single expert block, then 16→16→32→32→50 on last valid token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import POLAR_CL, POLAR_DIM

D_MODEL = 16
N_EXPERTS = 5
N_VOTES = 10
N_BRANCH_VOTE = 2
N_STAGES = 3
N_POLY_COEFFS = 4  # first 4 embedding dims only; ∫₀¹ Σ c_k x^k = Σ c_k/(k+1)


def polynomial_integral_first4(c4: torch.Tensor) -> torch.Tensor:
    """
    ∫₀¹ (c0 + c1 x + c2 x² + c3 x³) dx = c0/1 + c1/2 + c2/3 + c3/4.

    Parameters
    ----------
    c4
        (..., 4)
    """
    k = torch.arange(1, N_POLY_COEFFS + 1, device=c4.device, dtype=c4.dtype)
    w = 1.0 / k
    return (c4 * w).sum(dim=-1)


class OuterRowMix(nn.Module):
    """M_ij = tanh(h_i u_j); update vector is column mix weighted by rows: out_j = Σ_i row_w[i] M_ij."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.zeros(d_model))
        self.row_w = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.row_w, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: B,N,D — M[b,n,i,j] = tanh(h[b,n,i] * u[j])
        M = torch.tanh(h.unsqueeze(-1) * self.u.view(1, 1, 1, -1))
        return torch.einsum("bnij,i->bnj", M, self.row_w)


class SauceExpertBlock(nn.Module):
    """Attention → outer(u) → 16→64→16 → outer(v)."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=True
        )
        self.outer_u = OuterRowMix(d_model)
        self.outer_v = OuterRowMix(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_u = nn.LayerNorm(d_model)
        self.norm_m = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        t = self.norm_a(h)
        a, _ = self.attn(t, t, t, key_padding_mask=key_padding_mask, need_weights=False)
        h = h + self.dropout(a)
        t = self.norm_u(h)
        h = h + self.outer_u(t)
        t = self.norm_m(h)
        h = h + self.dropout(self.ffn(t))
        t = self.norm_v(h)
        h = h + self.outer_v(t)
        return h


class FiveExpertMoELayer(nn.Module):
    """Five parallel :class:`SauceExpertBlock`, router on 5×16 last valid tokens → softmax 5 → fuse."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [SauceExpertBlock(d_model, nhead, ffn_dim, dropout) for _ in range(N_EXPERTS)]
        )
        self.router = nn.Sequential(
            nn.Linear(N_EXPERTS * d_model, 64),
            nn.GELU(),
            nn.Linear(64, N_EXPERTS),
        )

    def forward(
        self, h: torch.Tensor, key_padding_mask: torch.Tensor | None, lengths: torch.Tensor
    ) -> torch.Tensor:
        b, _, d = h.shape
        outs = [ex(h, key_padding_mask) for ex in self.experts]
        last_rows = [_last_token_state(t, lengths) for t in outs]
        logits = self.router(torch.cat(last_rows, dim=-1))
        w = F.softmax(logits, dim=-1).view(b, 1, 1, N_EXPERTS)
        stack = torch.stack(outs, dim=-1)
        return (stack * w).sum(dim=-1)


class TenVoteMoEStack(nn.Module):
    """Ten parallel :class:`FiveExpertMoELayer`; vote on 10×16 last states → softmax 10 → fuse."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FiveExpertMoELayer(d_model, nhead, ffn_dim, dropout) for _ in range(N_VOTES)]
        )
        self.vote_router = nn.Sequential(
            nn.Linear(N_VOTES * d_model, 128),
            nn.GELU(),
            nn.Linear(128, N_VOTES),
        )

    def forward(
        self, h: torch.Tensor, key_padding_mask: torch.Tensor | None, lengths: torch.Tensor
    ) -> torch.Tensor:
        b = h.size(0)
        fused = [ly(h, key_padding_mask, lengths) for ly in self.layers]
        lasts = [_last_token_state(t, lengths) for t in fused]
        vw = F.softmax(self.vote_router(torch.cat(lasts, dim=-1)), dim=-1).view(b, 1, 1, N_VOTES)
        stack = torch.stack(fused, dim=-1)
        return (stack * vw).sum(dim=-1)


class PressureDualBranch(nn.Module):
    """Two sequence transformers; integral from first 4 dims; Cl residual 1→8→16 broadcast add."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.upper = SauceExpertBlock(d_model, nhead, ffn_dim, dropout)
        self.lower = SauceExpertBlock(d_model, nhead, ffn_dim, dropout)
        self.res_head = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, d_model),
        )
        self.branch_router = nn.Sequential(
            nn.Linear(2 * d_model, 32),
            nn.GELU(),
            nn.Linear(32, N_BRANCH_VOTE),
        )

    def forward(
        self,
        h: torch.Tensor,
        polar: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        p1 = self.upper(h, key_padding_mask)
        p2 = self.lower(h, key_padding_mask)
        i1 = polynomial_integral_first4(p1[..., :N_POLY_COEFFS])
        i2 = polynomial_integral_first4(p2[..., :N_POLY_COEFFS])
        cl = polar[:, :, POLAR_CL]
        valid = ~key_padding_mask
        # Estimated section lift contribution vs measured Cl at each α sample.
        est = i1 - i2
        residual = (est - cl) * valid.float()
        phi = self.res_head(residual.unsqueeze(-1))
        p1c = p1 + phi
        p2c = p2 + phi
        b = h.size(0)
        z1 = _last_token_state(p1c, lengths)
        z2 = _last_token_state(p2c, lengths)
        bw = F.softmax(self.branch_router(torch.cat([z1, z2], dim=-1)), dim=-1).view(b, 1, 1, N_BRANCH_VOTE)
        return (torch.stack([p1c, p2c], dim=-1) * bw).sum(dim=-1)


class MacroDuplicatedStage(nn.Module):
    """Ten-vote MoE → dual pressure + residual + branch vote → ten-vote MoE again."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.moe_a = TenVoteMoEStack(d_model, nhead, ffn_dim, dropout)
        self.pressure = PressureDualBranch(d_model, nhead, ffn_dim, dropout)
        self.moe_b = TenVoteMoEStack(d_model, nhead, ffn_dim, dropout)

    def forward(
        self,
        h: torch.Tensor,
        polar: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        t = self.moe_a(h, key_padding_mask, lengths)
        t = self.pressure(t, polar, key_padding_mask, lengths)
        return self.moe_b(t, key_padding_mask, lengths)


def _last_token_state(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    b = h.size(0)
    idx = (lengths - 1).clamp(min=0)
    return h[torch.arange(b, device=h.device), idx]


class PolarVotingMoETransformer(nn.Module):
    """
    Polar ``(B, N, 5)`` [Cl, Cd, α, Re, Ma] + padding → ``(B, 50)`` Fourier target.

    See module docstring for the full recipe.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = 4,
        ffn_dim: int = 64,
        dropout: float = 0.1,
        n_stages: int = N_STAGES,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.embed = nn.Linear(POLAR_DIM, d_model)
        self.stages = nn.ModuleList(
            [MacroDuplicatedStage(d_model, nhead, ffn_dim, dropout) for _ in range(n_stages)]
        )
        self.final_block = SauceExpertBlock(d_model, nhead, ffn_dim, dropout)
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 50),
        )

    def forward(
        self, polar: torch.Tensor, key_padding_mask: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        h = self.embed(polar)
        for st in self.stages:
            h = st(h, polar, key_padding_mask, lengths)
        h = self.final_block(h, key_padding_mask)
        z = _last_token_state(h, lengths)
        return self.out_mlp(z)


def fourier_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="mean")
