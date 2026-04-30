"""CUDA-first device selection for training and evaluation scripts."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    """
    Map CLI device string to `torch.device`.

    ``auto`` prefers **CUDA**, then Apple **MPS**, then **CPU** (CUDA-first policy).
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def configure_cuda_training(device: torch.device, *, deterministic: bool = False) -> None:
    """Enable cuDNN autotuner on CUDA unless a deterministic run is requested."""
    if device.type != "cuda":
        return
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
