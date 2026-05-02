"""Neural encoder from raw coordinates to 50D Fourier-style targets (supervised)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .airfoil_embedding import N_COORDS, N_FOURIER_REAL


class CoordToFourierMLP(nn.Module):
    """Flattened ``(250·2)`` coordinates → MLP → ``50`` (matches analytic Fourier layout)."""

    def __init__(self, hidden: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        flat = N_COORDS * 2
        self.net = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_FOURIER_REAL),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        xy
            ``(B, 250, 2)`` or ``(B, 500)``.
        """
        if xy.dim() == 3:
            xy = xy.reshape(xy.size(0), -1)
        return self.net(xy)
