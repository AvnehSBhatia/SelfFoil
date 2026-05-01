"""WHISP: weighted hydrodynamic iterative structured propagation (inverse airfoil design stack)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn

from .pair_encoder_loaders import pretrained_pair_embedders
from .whisp_physics import DeltaTransformer, PreDeltaPhysics


class CstMLP(nn.Module):
    """Baseline MLP: (Cl, Cd, Re_log, Mach, alpha) -> CST18 coefficients (5→32→64→18)."""

    def __init__(
        self,
        *,
        in_dim: int = 5,
        hidden1: int = 32,
        hidden2: int = 64,
        cst_dim: int = 18,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden2, cst_dim),
        )

    def forward(
        self,
        cl: torch.Tensor,
        cd: torch.Tensor,
        re_log: torch.Tensor,
        mach: torch.Tensor,
        alpha: torch.Tensor,
        route_tau: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del route_tau  # unused; kept for call-site parity with WHISP.
        x = torch.stack([cl, cd, re_log, mach, alpha], dim=-1)
        return self.net(x), {}


class InnerBlock(nn.Module):
    """One bilinear + projection + routing residual block (shared 5× per outer stage)."""

    def __init__(self, d: int = 8, n_emb: int = 4, dropout_p: float = 0.05) -> None:
        super().__init__()
        self.d = d
        self.n_emb = n_emb
        self.B = nn.Parameter(torch.zeros(n_emb, d, d))
        self.W_proj = nn.Linear(d * d, d)
        self.route = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_emb))
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, E: torch.Tensor, u: torch.Tensor, route_tau: float = 1.0) -> torch.Tensor:
        # E: (B, 4, 8), u: (B, 8); route_tau > 0 softens softmax (less collapse early in training).
        tau = max(float(route_tau), 1e-3)
        inv_tau = 1.0 / tau
        B, n_emb, d = E.shape
        u_exp = u.unsqueeze(1).unsqueeze(2)
        E_exp = E.unsqueeze(3)
        outer = E_exp * u_exp
        M = outer + self.B.unsqueeze(0)
        H = self.W_proj(M.contiguous().view(B, n_emb, d * d))
        H = self.dropout(H)
        inv_tau = 1.0 / tau
        route_logits = self.route(H) * inv_tau
        S = torch.softmax(route_logits, dim=-1)
        mix = torch.einsum("bij,bjd->bid", S, H)
        return E + mix


class OuterStage(nn.Module):
    """Θ_k: inner weights + pre-delta embedding mixer for this outer index."""

    def __init__(self, d: int = 8, n_emb: int = 4, n_inner: int = 5, dropout_p: float = 0.05) -> None:
        super().__init__()
        self.inner = InnerBlock(d=d, n_emb=n_emb, dropout_p=dropout_p)
        self.n_inner = n_inner
        self.w_aero_logits = nn.Linear(n_emb * d, n_emb)

    def run_inner(self, E: torch.Tensor, u: torch.Tensor, route_tau: float = 1.0) -> torch.Tensor:
        for _ in range(self.n_inner):
            E = self.inner(E, u, route_tau)
        return E

    def mix_aero(self, E: torch.Tensor) -> torch.Tensor:
        logits = self.w_aero_logits(E.view(E.shape[0], -1))
        w = torch.softmax(logits, dim=-1)
        return (w.unsqueeze(-1) * E).sum(dim=1)


class WHISP(nn.Module):
    """
    (Cl, Cd, Re_log, Mach, alpha) -> frozen pair encoders -> E (4×8), iterative core,
    physics auxiliary losses, damped latent-only delta updates, CST head (18).
    """

    def __init__(
        self,
        encoder_ckpts: Sequence[str | Path],
        d: int = 8,
        n_emb: int = 4,
        n_outer: int = 3,
        n_inner: int = 5,
        cst_dim: int = 18,
        freeze_encoders: bool = True,
        dropout_p: float = 0.05,
    ) -> None:
        super().__init__()
        if len(encoder_ckpts) != 4:
            raise ValueError("encoder_ckpts must list four paths: Cl, Cd, Re, Mach.")
        self.d = d
        self.n_emb = n_emb
        self.n_outer = n_outer
        self.embeds = pretrained_pair_embedders(encoder_ckpts, latent_dim=d, freeze=freeze_encoders)
        self.stages = nn.ModuleList(
            [OuterStage(d=d, n_emb=n_emb, n_inner=n_inner, dropout_p=dropout_p) for _ in range(n_outer)]
        )
        self.w_out_logits = nn.Linear(n_emb * d, n_emb)
        self.post_stage_ln = nn.LayerNorm(d)
        self.final_norm = nn.LayerNorm(d)
        self.final_dropout = nn.Dropout(p=dropout_p)
        self.head_cst = nn.Linear(d, cst_dim)
        self.head_cl = nn.Linear(d, 1)
        self.pre_physics = PreDeltaPhysics(z_dim=d)
        self.delta_transformer = DeltaTransformer(z_dim=d)
        self.register_buffer("delta_damping", torch.tensor(0.3, dtype=torch.float32))
        self.delta_damping_value = 0.3

    def _stack_pairs(
        self,
        cl: torch.Tensor,
        cd: torch.Tensor,
        re_log: torch.Tensor,
        mach: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        pairs = [
            torch.stack([cl, alpha], dim=-1),
            torch.stack([cd, alpha], dim=-1),
            torch.stack([re_log, alpha], dim=-1),
            torch.stack([mach, alpha], dim=-1),
        ]
        return torch.stack(pairs, dim=1)

    def embed(self, pairs: torch.Tensor) -> torch.Tensor:
        parts = [self.embeds[i](pairs[:, i, :]) for i in range(self.n_emb)]
        return torch.stack(parts, dim=1)

    def forward(
        self,
        cl: torch.Tensor,
        cd: torch.Tensor,
        re_log: torch.Tensor,
        mach: torch.Tensor,
        alpha: torch.Tensor,
        route_tau: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pairs = self._stack_pairs(cl, cd, re_log, mach, alpha)
        E = self.embed(pairs)
        u = E.mean(dim=1)
        tau = max(float(route_tau), 1e-3)
        inv_tau = 1.0 / tau

        aux: dict[str, torch.Tensor] = {}
        l_ns_acc: torch.Tensor | None = None
        for k, stage in enumerate(self.stages):
            E = stage.run_inner(E, u, route_tau=tau)
            a = stage.mix_aero(E)
            logits = (E * a.unsqueeze(1)).sum(dim=-1)
            scores = torch.softmax(logits * inv_tau, dim=-1)
            raw_delta = (scores.unsqueeze(-1) * E).sum(dim=1)

            L_ns, cl_gamma = self.pre_physics(raw_delta, return_predelta_feats=False)
            l_ns_acc = L_ns if l_ns_acc is None else l_ns_acc + L_ns
            aux[f"cl_gamma_{k}"] = cl_gamma

            du = self.delta_transformer(raw_delta)
            u = raw_delta + self.delta_damping_value * du

            B = E.shape[0]
            E = self.post_stage_ln(E.contiguous().view(-1, self.d)).view(B, self.n_emb, self.d)

        assert l_ns_acc is not None
        aux["L_ns"] = l_ns_acc.mean() / self.n_outer

        logits_out = self.w_out_logits(E.view(E.shape[0], -1))
        w_out = torch.softmax(logits_out, dim=-1)
        a_final = (w_out.unsqueeze(-1) * E).sum(dim=1)
        a_final = self.final_norm(a_final)
        a_final = self.final_dropout(a_final)
        cst_pred = self.head_cst(a_final)
        aux["cl_direct"] = self.head_cl(a_final).squeeze(-1)
        aux["a_final"] = a_final
        aux["E_final"] = E
        return cst_pred, aux
