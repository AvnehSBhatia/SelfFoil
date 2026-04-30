"""2D -> 8D -> 2D autoencoder with tanh nonlinearities (feature, AoA)."""

from __future__ import annotations

import torch.nn as nn


class PairTanhAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, latent_dim), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 2), nn.Tanh())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
