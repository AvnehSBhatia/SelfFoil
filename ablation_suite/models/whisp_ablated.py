"""Catalog-driven WHISP for ablations (kept in sync with `core.whisp_net` stack)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.pair_encoder_loaders import pretrained_pair_embedders
from core.whisp_physics import DeltaTransformer, PreDeltaPhysics

DEFAULT_MODEL_FLAGS: dict[str, object] = {
    "use_physics": True,
    "use_delta": True,
    "routing": "softmax",
    "delta_mode": "mlp",
    "random_delta_std": 0.05,
    "freeze_delta": False,
    "interaction": "bilinear",
    "shared_B_matrix": False,
    "n_outer": 3,
    "n_inner": 5,
    "shared_outer": False,
    "outer_decay": 1.0,
    "outer_order": "forward",
    "latent_p_mode": "full",
    "p_noise_std": 0.1,
    "encoder_mode": "frozen",
    "physics_fidelity": "full",
    "integration": "trapz",
    "physics_nx": 32,
    "adaptive_x_grid": False,
    "distill_weight": 0.0,
    "d": 8,
    "n_emb": 4,
}


def merge_model_spec(spec: dict[str, object]) -> dict[str, object]:
    out = {**DEFAULT_MODEL_FLAGS}
    for k, v in spec.items():
        if k in ("category", "slug", "train"):
            continue
        out[k] = v
    return out


class InnerBlockAblated(nn.Module):
    def __init__(
        self,
        d: int,
        n_emb: int,
        *,
        interaction: str = "bilinear",
        shared_B: bool = False,
        routing: str = "softmax",
    ) -> None:
        super().__init__()
        self.d = d
        self.n_emb = n_emb
        self.interaction = interaction
        self.routing = routing
        if shared_B:
            self.B_shared = nn.Parameter(torch.zeros(d, d))
            self.B = None
        else:
            self.B = nn.Parameter(torch.zeros(n_emb, d, d))
            self.B_shared = None
        self.W_proj = nn.Linear(d * d, d)
        self.route = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_emb))
        if interaction == "linear_concat":
            self.mix_lin = nn.Linear(d + d, d)
        elif interaction == "hadamard":
            self.mix_lin = nn.Linear(d, d)
        elif interaction == "attention":
            self.attn_scale = d**-0.5
        elif interaction == "no_cross":
            self.mix_lin = nn.Linear(d, d)

    def _B(self, i: int) -> torch.Tensor:
        if self.B_shared is not None:
            return self.B_shared
        assert self.B is not None
        return self.B[i]

    def forward(self, E: torch.Tensor, u: torch.Tensor, route_tau: float = 1.0) -> torch.Tensor:
        tau = max(float(route_tau), 1e-3)
        h_parts = []
        for i in range(self.n_emb):
            ei = E[:, i, :]
            if self.interaction == "bilinear":
                outer = torch.bmm(ei.unsqueeze(2), u.unsqueeze(1))
                M = outer + self._B(i)
                hi = self.W_proj(M.reshape(E.shape[0], -1))
            elif self.interaction == "linear_concat":
                hi = self.mix_lin(torch.cat([ei, u], dim=-1))
            elif self.interaction == "hadamard":
                hi = self.mix_lin(ei * u)
            elif self.interaction == "attention":
                outer = torch.bmm(ei.unsqueeze(2), u.unsqueeze(1))
                M = outer + self._B(i)
                hi = self.W_proj(M.reshape(E.shape[0], -1))
            elif self.interaction == "no_cross":
                hi = self.mix_lin(ei * u)
            else:
                raise ValueError(f"unknown interaction {self.interaction}")
            h_parts.append(hi)
        H = torch.stack(h_parts, dim=1)
        updated = []
        if self.interaction == "attention":
            dots = (E * u.unsqueeze(1)).sum(dim=-1) * self.attn_scale
            a = torch.softmax(dots / tau, dim=-1)
            mix = (a.unsqueeze(-1) * E).sum(dim=1)
            for i in range(self.n_emb):
                updated.append(E[:, i, :] + mix)
            return torch.stack(updated, dim=1)
        if self.routing == "mean_h":
            mix = H.mean(dim=1)
            for i in range(self.n_emb):
                updated.append(E[:, i, :] + mix)
            return torch.stack(updated, dim=1)
        for i in range(self.n_emb):
            s = torch.softmax(self.route(H[:, i, :]) / tau, dim=-1)
            mix = (s.unsqueeze(-1) * H).sum(dim=1)
            updated.append(E[:, i, :] + mix)
        return torch.stack(updated, dim=1)


class OuterStageAblated(nn.Module):
    def __init__(
        self,
        d: int,
        n_emb: int,
        n_inner: int,
        *,
        interaction: str,
        shared_B: bool,
        routing: str,
    ) -> None:
        super().__init__()
        self.inner = InnerBlockAblated(d, n_emb, interaction=interaction, shared_B=shared_B, routing=routing)
        self.n_inner = n_inner
        self.w_aero_logits = nn.Linear(n_emb * d, n_emb)

    def run_inner(self, E: torch.Tensor, u: torch.Tensor, route_tau: float) -> torch.Tensor:
        for _ in range(self.n_inner):
            E = self.inner(E, u, route_tau)
        return E

    def mix_aero(self, E: torch.Tensor) -> torch.Tensor:
        logits = self.w_aero_logits(E.reshape(E.shape[0], -1))
        w = torch.softmax(logits, dim=-1)
        return (w.unsqueeze(-1) * E).sum(dim=1)


class LinearDelta(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(z_dim, z_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.lin(z)


class WHISPAblated(nn.Module):
    """WHISP with ablation flags from `ablation_suite.catalog` merged specs."""

    def __init__(self, encoder_ckpts: Sequence[str | Path], spec: dict[str, object]) -> None:
        super().__init__()
        if len(encoder_ckpts) != 4:
            raise ValueError("encoder_ckpts must have length 4.")
        m = merge_model_spec(spec)
        self._spec = m
        self.d = int(m.get("d", 8))
        d = self.d
        self.n_emb = int(m.get("n_emb", 4))
        n_emb = self.n_emb
        self.n_outer = int(m.get("n_outer", 3))
        self.n_inner = int(m.get("n_inner", 5))
        self.use_physics = bool(m["use_physics"])
        self.use_delta = bool(m["use_delta"])
        self.delta_mode = str(m["delta_mode"])
        self.random_delta_std = float(m["random_delta_std"])
        self.outer_decay = float(m["outer_decay"])
        self.outer_order = str(m["outer_order"])
        self.latent_p_mode = str(m["latent_p_mode"])
        self.p_noise_std = float(m["p_noise_std"])
        self.distill_weight = float(m["distill_weight"])
        routing = "mean_h" if m.get("routing") == "mean_h" else "softmax"
        interaction = str(m["interaction"])
        shared_b_matrix = bool(m.get("shared_B_matrix", False))
        shared_outer = bool(m.get("shared_outer", False))
        enc_mode = str(m["encoder_mode"])
        if enc_mode == "scratch":
            embeds = nn.ModuleList()
            for _ in range(4):
                mod = nn.Sequential(nn.Linear(2, d), nn.Tanh())
                for layer in mod.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                embeds.append(mod)
            self.embeds = embeds
        else:
            if enc_mode == "frozen":
                self.embeds = pretrained_pair_embedders(encoder_ckpts, latent_dim=d, freeze=True)
            elif enc_mode == "partial_freeze":
                self.embeds = pretrained_pair_embedders(encoder_ckpts, latent_dim=d, freeze=False)
                for enc in self.embeds:
                    lin = enc[0]
                    assert isinstance(lin, nn.Linear)
                    lin.weight.requires_grad_(False)
            else:
                self.embeds = pretrained_pair_embedders(encoder_ckpts, latent_dim=d, freeze=False)
        if shared_outer:
            self.stages = nn.ModuleList(
                [
                    OuterStageAblated(
                        d,
                        n_emb,
                        self.n_inner,
                        interaction=interaction,
                        shared_B=shared_b_matrix,
                        routing=routing,
                    )
                ]
            )
        else:
            self.stages = nn.ModuleList(
                [
                    OuterStageAblated(
                        d, n_emb, self.n_inner, interaction=interaction, shared_B=shared_b_matrix, routing=routing
                    )
                    for _ in range(self.n_outer)
                ]
            )
        self._shared_outer = shared_outer
        self.w_out_logits = nn.Linear(n_emb * d, n_emb)
        self.post_stage_ln = nn.LayerNorm(d)
        self.final_norm = nn.LayerNorm(d)
        self.head_cst = nn.Linear(d, 18)
        self.head_cl = nn.Linear(d, 1)
        self.register_buffer("delta_damping", torch.tensor(0.3, dtype=torch.float32))
        nx = int(m["physics_nx"])
        if self.use_physics:
            self.pre_physics = PreDeltaPhysics(
                z_dim=d,
                nx=nx,
                fidelity_mode=str(m["physics_fidelity"]),
                integration=str(m["integration"]),
                adaptive_x_grid=bool(m["adaptive_x_grid"]),
            )
        else:
            self.pre_physics = None
        if self.delta_mode == "linear":
            self.delta_transformer = LinearDelta(d)
            self.lat_expand = None
        elif self.delta_mode == "expanded":
            self.lat_expand = nn.Linear(d, d)
            self.delta_transformer = nn.Linear(d * 2, d)
        else:
            self.delta_transformer = DeltaTransformer(z_dim=d)
            self.lat_expand = None
        if self.freeze_delta_bool():
            for p in self.delta_transformer.parameters():
                p.requires_grad_(False)
        if self.distill_weight > 0.0 and enc_mode != "scratch":
            self.teacher_embeds = pretrained_pair_embedders(encoder_ckpts, latent_dim=d, freeze=True)
        else:
            self.teacher_embeds = None
        if self.latent_p_mode == "shuffle":
            self.register_buffer("_latent_perm", torch.randperm(d, dtype=torch.long))

    def freeze_delta_bool(self) -> bool:
        return bool(self._spec.get("freeze_delta", False))

    def _stack_pairs(
        self,
        cl: torch.Tensor,
        cd: torch.Tensor,
        re_log: torch.Tensor,
        mach: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack([cl, alpha], dim=-1),
                torch.stack([cd, alpha], dim=-1),
                torch.stack([re_log, alpha], dim=-1),
                torch.stack([mach, alpha], dim=-1),
            ],
            dim=1,
        )

    def embed(self, pairs: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.embeds[i](pairs[:, i, :]) for i in range(self.n_emb)], dim=1)

    def _z_for_delta(self, raw_delta: torch.Tensor) -> torch.Tensor:
        z = raw_delta
        mode = self.latent_p_mode
        if mode == "noise":
            z = z + torch.randn_like(z) * self.p_noise_std
        elif mode == "shuffle" and hasattr(self, "_latent_perm"):
            z = z[:, self._latent_perm.to(z.device)]
        elif mode == "scalar":
            z = z.mean(dim=-1, keepdim=True).expand_as(z)
        elif mode == "zeros":
            z = torch.zeros_like(z)
        return z

    def _delta_du(self, raw_delta: torch.Tensor) -> torch.Tensor:
        if self.delta_mode == "expanded":
            assert self.lat_expand is not None and isinstance(self.delta_transformer, nn.Linear)
            z_in = torch.cat([raw_delta, self.lat_expand(raw_delta)], dim=-1)
            return self.delta_transformer(z_in)
        if self.delta_mode == "linear":
            return self.delta_transformer(raw_delta)
        z_in = self._z_for_delta(raw_delta)
        if self.delta_mode == "random":
            return torch.randn_like(raw_delta) * self.random_delta_std
        if self.delta_mode == "sign":
            du = self.delta_transformer(raw_delta)
            return torch.sign(du.detach()) * 0.1 + du - du.detach()
        return self.delta_transformer(z_in)

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
        aux: dict[str, torch.Tensor] = {"L_ns_list": []}
        if self.teacher_embeds is not None and self.distill_weight > 0.0:
            with torch.no_grad():
                Et = torch.stack([self.teacher_embeds[i](pairs[:, i, :]) for i in range(self.n_emb)], dim=1)
            aux["embed_distill"] = F.mse_loss(E, Et.detach())
        u = E.mean(dim=1)
        u_mean_fixed = u
        damp = float(self.delta_damping.item())
        tau = max(float(route_tau), 1e-3)
        outer_indices = list(range(self.n_outer))
        if self.outer_order == "reversed":
            outer_indices = outer_indices[::-1]
        for step_i, k in enumerate(outer_indices):
            stage = self.stages[0] if self._shared_outer else self.stages[k]
            u_loop = u_mean_fixed if not self.use_delta else u
            E = stage.run_inner(E, u_loop, route_tau=tau)
            a = stage.mix_aero(E)
            logits = (E * a.unsqueeze(1)).sum(dim=-1)
            scores = torch.softmax(logits / tau, dim=-1)
            raw_delta = (scores.unsqueeze(-1) * E).sum(dim=1)
            if self.use_physics and self.pre_physics is not None:
                L_ns, cl_gamma = self.pre_physics(raw_delta)
                aux["L_ns_list"].append(L_ns)
                aux[f"cl_gamma_{k}"] = cl_gamma
            if self.delta_mode == "identity":
                u = raw_delta
            elif not self.use_delta:
                u = u_mean_fixed
            else:
                du = self._delta_du(raw_delta)
                decay = self.outer_decay**step_i
                u = raw_delta + damp * decay * du
            B, four, d_ = E.shape
            E = self.post_stage_ln(E.reshape(-1, d_)).reshape(B, four, d_)
        logits_out = self.w_out_logits(E.reshape(E.shape[0], -1))
        w_out = torch.softmax(logits_out, dim=-1)
        a_final = (w_out.unsqueeze(-1) * E).sum(dim=1)
        a_final = self.final_norm(a_final)
        cst_pred = self.head_cst(a_final)
        aux["cl_direct"] = self.head_cl(a_final).squeeze(-1)
        aux["a_final"] = a_final
        aux["E_final"] = E
        return cst_pred, aux
