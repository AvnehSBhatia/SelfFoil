"""Per-sample polar-token embedding ``E``: ``[C_l, C_d, α, Re, M_∞]`` → latent ``ℝ^{16}``."""

from __future__ import annotations

import torch
import torch.nn as nn

from .constants import POLAR_DIM


class PolarTokenEmbedding(nn.Module):
    """
    Stage-1 embedding from the architecture spec:

    ``h_i^(0) = E(x_i)`` with ``x_i = [Cl, Cd, α, Re, Ma]``, each token ``ℝ^5 → ℝ^{d}``.

    This is **not** the Fourier / coordinate representation; outputs are Transformer inputs.
    """

    def __init__(self, d_latent: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(POLAR_DIM, d_latent)

    def forward(self, polar: torch.Tensor) -> torch.Tensor:
        """``polar``: ``(B, N, 5)`` → ``(B, N, d_latent)``."""
        return self.proj(polar)
