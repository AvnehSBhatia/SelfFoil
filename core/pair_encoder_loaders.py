"""Load pretrained PairTanhAutoencoder.encoder checkpoints (2→8 + Tanh)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn


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
        sd = torch.load(Path(path), map_location="cpu", weights_only=True)
        enc.load_state_dict(sd)
        if freeze:
            for p in enc.parameters():
                p.requires_grad_(False)
        embeds.append(enc)
    return embeds
