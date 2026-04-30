"""Load pretrained PairTanhAutoencoder.encoder checkpoints (2→8 + Tanh)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn


class _NormalizedPairEmbedder(nn.Module):
    def __init__(self, encoder: nn.Module, input_mean: torch.Tensor, input_std: torch.Tensor) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("input_mean", input_mean.view(1, 2))
        self.register_buffer("input_std", input_std.view(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / self.input_std.clamp_min(1e-6)
        return self.encoder(x_norm)


def pretrained_pair_embedders(
    ckpts: Sequence[str | Path],
    latent_dim: int = 8,
    freeze: bool = True,
) -> nn.ModuleList:
    """Each ckpt is state_dict for nn.Sequential(Linear(2, latent_dim), Tanh())."""
    if len(ckpts) != 4:
        raise ValueError("Expected four encoder checkpoints: Cl, Cd, Re, Mach order.")
    embeds = nn.ModuleList()
    for path in ckpts:
        enc = nn.Sequential(nn.Linear(2, latent_dim), nn.Tanh())
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            sd = payload["state_dict"]
            input_mean = payload.get("input_mean")
            input_std = payload.get("input_std")
        else:
            sd = payload
            input_mean = None
            input_std = None
        enc.load_state_dict(sd)
        embed: nn.Module = enc
        if input_mean is not None and input_std is not None:
            embed = _NormalizedPairEmbedder(enc, input_mean.to(dtype=torch.float32), input_std.to(dtype=torch.float32))
        if freeze:
            for p in embed.parameters():
                p.requires_grad_(False)
        embeds.append(embed)
    return embeds
