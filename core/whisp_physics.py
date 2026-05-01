"""Reduced boundary-layer momentum residual + lift channel (physics as auxiliary loss only)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _trapz_yx(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y (B, N), x (N,) -> (B,) trapezoidal integral."""
    dx = x[1:] - x[:-1]
    return ((y[:, 1:] + y[:, :-1]) * dx.unsqueeze(0)).sum(dim=-1) * 0.5


def _simpson_yx(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Composite Simpson along non-uniform x (N even for interior Simpson pairs). y (B,N), x (N,)."""
    if x.shape[0] < 3:
        return _trapz_yx(y, x)
    n = x.shape[0]
    if n % 2 == 0:
        return _simpson_yx(y[:, :-1], x[:-1]) + _trapz_yx(y[:, -2:], x[-2:])
    acc = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
    for i in range(0, n - 2, 2):
        h0 = x[i + 1] - x[i]
        h1 = x[i + 2] - x[i + 1]
        acc = acc + (h0 + h1) / 6.0 * (y[:, i] + 4.0 * y[:, i + 1] + y[:, i + 2])
    return acc


def _fourier_phi(x: torch.Tensor, n_modes: int = 8) -> torch.Tensor:
    """x: (Nx,) in [0,1]. Returns (8, Nx) basis rows."""
    out = []
    for k in range(n_modes):
        w = (k + 1) * math.pi
        if k % 2 == 0:
            out.append(torch.cos(w * x))
        else:
            out.append(torch.sin(w * x))
    return torch.stack(out, dim=0)


class PreDeltaPhysics(nn.Module):
    """
    z -> U_e, H, C_f -> momentum thickness march -> dimensionless NS residual loss + circulation Cl.

    By default this module only contributes auxiliary losses (L_ns, cl_gamma). Optionally it can also
    expose cheap pooled summaries of (U_e, H, C_f) for downstream delta conditioning.
    """

    def __init__(
        self,
        z_dim: int = 8,
        nx: int = 32,
        hidden: int = 16,
        *,
        fidelity_mode: str = "full",
        integration: str = "trapz",
        adaptive_x_grid: bool = False,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.nx = nx
        self.fidelity_mode = fidelity_mode
        self.integration = integration
        self.adaptive_x_grid = adaptive_x_grid
        if adaptive_x_grid:
            i = torch.arange(nx, dtype=torch.float32)
            x = 0.5 * (1.0 - torch.cos(math.pi * i / max(1, nx - 1)))
        else:
            x = torch.linspace(0.0, 1.0, nx, dtype=torch.float32)
        self.register_buffer("x_grid", x)
        phi = _fourier_phi(x, n_modes=z_dim)
        self.register_buffer("phi", phi)
        self.U_inf = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        in_loc = z_dim + 1
        self.mlp_H = nn.Sequential(
            nn.Linear(in_loc, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.mlp_Cf = nn.Sequential(
            nn.Linear(in_loc, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.theta0 = nn.Parameter(torch.tensor(1e-4, dtype=torch.float32))
        self.cl_amp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.cl_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        # Blend explicit Euler toward previous θ (stiffness / stability).
        self.register_buffer("theta_damp_mix", torch.tensor(0.1, dtype=torch.float32))
        if fidelity_mode == "shuffled_x":
            self.register_buffer("_x_perm", torch.randperm(nx))

    def _integrate(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.integration == "simpson":
            return _simpson_yx(y, x)
        return _trapz_yx(y, x)

    def _predelta_channel_stats(self, U_e: torch.Tensor, H: torch.Tensor, C_f: torch.Tensor) -> torch.Tensor:
        """Cheap pooled summaries of the pre-delta physics channels (B, 12)."""
        def _stats(t: torch.Tensor) -> list[torch.Tensor]:
            return [t.mean(dim=-1), t.std(dim=-1), t.min(dim=-1).values, t.max(dim=-1).values]

        parts: list[torch.Tensor] = []
        parts.extend(_stats(U_e))
        parts.extend(_stats(H))
        parts.extend(_stats(torch.log(C_f.clamp(min=1e-6))))
        return torch.stack(parts, dim=-1)

    def forward(
        self, z: torch.Tensor, *, return_predelta_feats: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        z: (B, 8)
        Returns:
            L_ns: (B,) log1p of mean squared residual, scaled by U_inf^2 (dimensionless-ish).
            cl_from_gamma: (B,) lift surrogate from integrated U_e.
        """
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"z must be (B,{self.z_dim})")

        B, _ = z.shape
        device, dtype = z.device, z.dtype
        x = self.x_grid.to(device=device, dtype=dtype)
        nx = x.shape[0]
        phi = self.phi.to(device=device, dtype=dtype)
        if self.fidelity_mode == "shuffled_x" and hasattr(self, "_x_perm"):
            perm = self._x_perm.to(device)
            phi = phi[:, perm]
            x = x[perm]
        mix = self.theta_damp_mix.to(device=device, dtype=dtype)

        U_e = self.U_inf + torch.matmul(z, phi)
        U_e = U_e.clamp(min=0.05, max=20.0)

        zx = z.unsqueeze(1).expand(B, nx, self.z_dim)
        xx = x.view(1, nx, 1).expand(B, nx, 1)
        loc = torch.cat([zx, xx], dim=-1)
        H = 2.0 + 0.5 * torch.tanh(self.mlp_H(loc).squeeze(-1))
        C_f = 1e-3 + torch.nn.functional.softplus(self.mlp_Cf(loc).squeeze(-1))
        predelta_feats = self._predelta_channel_stats(U_e, H, C_f) if return_predelta_feats else None

        dUedx = torch.zeros(B, nx, device=device, dtype=dtype)
        if nx > 2:
            dUedx[:, 1:-1] = (U_e[:, 2:] - U_e[:, :-2]) / (x[2:] - x[:-2]).clamp(min=1e-6).view(1, -1)
        if nx > 1:
            dUedx[:, 0] = (U_e[:, 1] - U_e[:, 0]) / (x[1] - x[0]).clamp(min=1e-6)
            dUedx[:, -1] = (U_e[:, -1] - U_e[:, -2]) / (x[-1] - x[-2]).clamp(min=1e-6)

        U_safe = U_e.clamp(min=0.05)
        t0 = (self.theta0.abs() + 1e-6).expand(B).clamp(max=0.05)

        if self.fidelity_mode == "no_bl":
            Gamma = self._integrate(U_e, x)
            chord = 1.0
            V = self.U_inf.abs().clamp(min=1e-3)
            cl_gamma = (2.0 * Gamma / (V * chord)) * self.cl_amp + self.cl_bias
            L_ns = torch.zeros(B, device=device, dtype=dtype)
            if return_predelta_feats:
                assert predelta_feats is not None
                return L_ns, cl_gamma, predelta_feats
            return L_ns, cl_gamma

        theta_seq = [t0]
        for k in range(nx - 1):
            theta_k = theta_seq[-1].clamp(1e-6, 0.25)
            Uk = U_safe[:, k]
            Hk = H[:, k]
            Cf_k = C_f[:, k]
            dU = dUedx[:, k]
            entrain = (theta_k / Uk) * dU
            entrain = entrain.clamp(-80.0, 80.0)
            rhs = 0.5 * Cf_k - (2.0 + Hk) * entrain
            rhs = rhs.clamp(-500.0, 500.0)
            dxk = x[k + 1] - x[k]
            euler = theta_k + dxk * rhs
            theta_next = (1.0 - mix) * theta_k + mix * euler
            theta_next = theta_next.clamp(1e-6, 0.25)
            theta_seq.append(theta_next)
        theta = torch.stack(theta_seq, dim=-1)

        dthetadx = torch.zeros_like(theta)
        if nx > 2:
            dthetadx[:, 1:-1] = (theta[:, 2:] - theta[:, :-2]) / (x[2:] - x[:-2]).clamp(min=1e-6).view(1, -1)
        if nx > 1:
            dthetadx[:, 0] = (theta[:, 1] - theta[:, 0]) / (x[1] - x[0]).clamp(min=1e-6)
            dthetadx[:, -1] = (theta[:, -1] - theta[:, -2]) / (x[-1] - x[-2]).clamp(min=1e-6)

        cf_term = 0.5 * C_f if self.fidelity_mode not in ("no_energy_term", "no_energy") else torch.zeros_like(C_f)
        residual = dthetadx + (2.0 + H) * (theta / U_safe) * dUedx - cf_term
        residual = residual.clamp(-500.0, 500.0)
        msr = (residual[:, 1:-1] ** 2).mean(dim=-1)
        scale = self.U_inf * self.U_inf + 1e-6
        L_ns = torch.log1p(msr / scale)

        Gamma = self._integrate(U_e, x)
        chord = 1.0
        V = self.U_inf.abs().clamp(min=1e-3)
        cl_gamma = (2.0 * Gamma / (V * chord)) * self.cl_amp + self.cl_bias
        if self.fidelity_mode == "no_circulation":
            cl_gamma = torch.zeros_like(cl_gamma) + self.cl_bias

        if return_predelta_feats:
            assert predelta_feats is not None
            return L_ns, cl_gamma, predelta_feats
        return L_ns, cl_gamma


class DeltaTransformer(nn.Module):
    """Latent-only residual correction (no physics vector p — avoids double physics in gradients)."""

    def __init__(self, z_dim: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
