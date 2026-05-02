"""
Regime-conditioned aerodynamic polar → 50D Fourier-style output coefficients.

Polar **token embedding** ``[Cl, Cd, α, Re, Ma] → ℝ^{16}`` is :class:`~core.polar_token_embedding.PolarTokenEmbedding`.
This module adds experts, routers, pressure readouts, and the final 50-vector head.

Implements: 3 parallel experts, router, MoE fusion,
two pressure branches with adaptive polynomial integration, lift residual
correction, final transformer, last-row bottleneck, output MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import POLAR_CL
from .polar_token_embedding import PolarTokenEmbedding


D_MODEL = 16
D_ROUTER_IN = 3 * D_MODEL
N_EXPERTS = 3
N_DEGREES = 16  # polynomial orders 0..15
DEG_COEFFS = 16  # a_0..a_15 (latent width)


class OuterProductMix(nn.Module):
    """Bilinear feature coupling: h ⊙ h^T (outer) → linear → 16D residual."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d = d_model
        self.lin = nn.Linear(d_model * d_model, d_model)
        with torch.no_grad():
            self.lin.weight.mul_(0.1)
            if self.lin.bias is not None:
                self.lin.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        o = h.unsqueeze(-1) * h.unsqueeze(-2)  # B,N,D,D
        o = o.reshape(b, n, d * d)
        return self.lin(o)


class ExpertBlock(nn.Module):
    """Self-attn → outer → MLP → outer."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=True
        )
        self.op1 = OuterProductMix(d_model)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_op1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.op2 = OuterProductMix(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.norm_op2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._d_model = d_model

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        # h: B,N,D
        t = self.norm_a(h)
        a, _ = self.attn(t, t, t, key_padding_mask=key_padding_mask, need_weights=False)
        h = h + self.dropout(a)
        t = self.norm_op1(h)
        h = h + self.op1(t)
        t = self.norm_mlp(h)
        h = h + self.dropout(self.ffn(t))
        t = self.norm_op2(h)
        h = h + self.op2(t)
        return h


class SmallPressureTransform(nn.Module):
    """Lightweight sequence transform for a pressure stream."""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        return self.layer(h, src_key_padding_mask=key_padding_mask)


def polynomial_integrals(p: torch.Tensor) -> torch.Tensor:
    """
    ``csum[...,d] = Σ_{k=0}^d a_k/(k+1)`` using token coefficients ``p[..., 0..15]``.

    Shape ``(B, N, 16)`` — index ``d`` is the integral truncated at degree ``d``.
    """
    inv = torch.arange(1, DEG_COEFFS + 1, device=p.device, dtype=p.dtype).reciprocal()
    return torch.cumsum(p * inv, dim=-1)


def integrated_pressure(p: torch.Tensor, degree_lin: nn.Module) -> torch.Tensor:
    """Soft degree mixture over ``Σ_k p_k softmax · I_k``."""
    csum = polynomial_integrals(p)
    logits = degree_lin(p)
    pr = F.softmax(logits, dim=-1)
    return (pr * csum).sum(dim=-1)


class RegimeConditionedAeroTransformer(nn.Module):
    """
    Polar ``(B, N, 5)``: each row is embedded with :class:`~core.polar_token_embedding.PolarTokenEmbedding`
    ``(ℝ^5 → ℝ^{16} per token)``, then fused / physics blocks produce a Fourier-style ``(B, 50)`` vector.

    The architecture follows the "Final architecture" spec: three experts, router
    on last non-padding expert rows (per-sample last *sequence* index), two pressure
    paths, one residual ``ΔI - Cl`` broadcast correction, then final head.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = 4,
        ffn_expert: int = 64,
        ffn_pressure: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.polar_token_embed = PolarTokenEmbedding(d_latent=d_model)
        self.experts = nn.ModuleList(
            [ExpertBlock(d_model, nhead, ffn_expert, dropout) for _ in range(N_EXPERTS)]
        )
        self.router = nn.Sequential(
            nn.Linear(D_ROUTER_IN, 32),
            nn.GELU(),
            nn.Linear(32, N_EXPERTS),
        )
        self.pt1 = SmallPressureTransform(d_model, nhead, ffn_pressure, dropout)
        self.pt2 = SmallPressureTransform(d_model, nhead, ffn_pressure, dropout)
        self.deg1 = nn.Linear(d_model, N_DEGREES)
        self.deg2 = nn.Linear(d_model, N_DEGREES)
        self.final_tf = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_pressure * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 50),
        )

    def load_state_dict(self, state_dict: dict, strict: bool = True):  # type: ignore[override]
        if any(str(k).startswith("embed.") for k in state_dict):
            remap = {}
            for k, v in state_dict.items():
                ks = str(k)
                if ks.startswith("embed."):
                    remap["polar_token_embed.proj." + ks[len("embed.") :]] = v
                else:
                    remap[k] = v
            state_dict = remap
        return super().load_state_dict(state_dict, strict=strict)

    def _last_token_state(self, h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        s_k = H at last *valid* sequence index: ``h[b, L_b-1]`` (not the padded end).
        """
        b = h.size(0)
        idx = (lengths - 1).clamp(min=0)
        return h[torch.arange(b, device=h.device), idx]

    def forward(
        self, polar: torch.Tensor, key_padding_mask: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        polar
            ``(B, N, 5)`` — [Cl, Cd, α, Re, Ma] per valid token, zeros in pad slots.
        key_padding_mask
            ``(B, N)``, ``True`` where positions are **padding** (ignore in attention).
        lengths
            ``(B,)`` valid token count per sample.
        """
        b, n, _ = polar.shape
        h0 = self.polar_token_embed(polar)  # B,N,16  (learned Cl,Cd,α,Re,Ma → 16D per token)
        hks = [expert(h0, key_padding_mask) for expert in self.experts]

        last_rows = [self._last_token_state(hk, lengths) for hk in hks]  # each (B,16)
        s = torch.cat(last_rows, dim=-1)  # (B,48)
        w = F.softmax(self.router(s), dim=-1)  # (B,3)

        stack = torch.stack(hks, dim=-1)  # B,N,16,3
        wf = w.view(b, 1, 1, N_EXPERTS)
        h1 = (stack * wf).sum(dim=-1)  # B,N,16

        p1 = self.pt1(h1, key_padding_mask)
        p2 = self.pt2(h1, key_padding_mask)

        i1 = integrated_pressure(p1, self.deg1)  # B,N
        i2 = integrated_pressure(p2, self.deg2)
        cl = polar[:, :, POLAR_CL]
        valid = ~key_padding_mask
        r = (i1 - i2) - cl
        r = r * valid.float()

        phi = r.unsqueeze(-1).expand(b, n, self.d_model)
        p1c = p1 + phi
        p2c = p2 + phi
        h2 = 0.5 * (p1c + p2c)

        h3 = self.final_tf(h2, src_key_padding_mask=key_padding_mask)
        z = self._last_token_state(h3, lengths)  # B,16
        return self.out_mlp(z)


def fourier_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="mean")
