"""Optional supervised MLP: flat airfoil coordinates → 50D analytic-Fourier targets (NOT polar 5→16)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .airfoil_embedding import N_COORDS, N_FOURIER_REAL


class CoordFourierRegressionMLP(nn.Module):
    """Flatten ``(250·2)`` coordinates → small MLP → ``50`` (learned match to FFT targets)."""

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
        if xy.dim() == 3:
            xy = xy.reshape(xy.size(0), -1)
        return self.net(xy)


# Back-compat alias — avoid “embedding” for this pathway.
CoordToFourierMLP = CoordFourierRegressionMLP
