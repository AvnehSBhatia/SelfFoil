"""Two-dimensional linear autoencoder: (scalar, angle of attack) <-> latent."""

from __future__ import annotations

import torch.nn as nn


class PairLinearAutoencoder(nn.Module):
    """Encoder 2 -> `latent_dim`, decoder `latent_dim` -> 2. Pure linear layers (no activations)."""

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, latent_dim)
        self.decoder = nn.Linear(latent_dim, 2)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
