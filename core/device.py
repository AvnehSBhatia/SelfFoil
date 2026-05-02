"""Training device helpers (CUDA, optional MPS, CPU) and host→device transfer."""

from __future__ import annotations

import torch

TensorDict = dict[str, torch.Tensor]


def default_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dataloader_pin_memory(dev: torch.device) -> bool:
    return dev.type == "cuda"


def tensor_to(
    t: torch.Tensor, dev: torch.device, *, non_blocking: bool
) -> torch.Tensor:
    return t.to(dev, non_blocking=non_blocking)


def polar_batch_to_device(
    batch: TensorDict, dev: torch.device, *, non_blocking: bool
) -> TensorDict:
    return {
        "polar": tensor_to(batch["polar"], dev, non_blocking=non_blocking),
        "padding_mask": tensor_to(batch["padding_mask"], dev, non_blocking=non_blocking),
        "target_fourier": tensor_to(
            batch["target_fourier"], dev, non_blocking=non_blocking
        ),
        "lengths": tensor_to(batch["lengths"], dev, non_blocking=non_blocking),
    }


def coord_fourier_batch_to_device(
    batch: TensorDict, dev: torch.device, *, non_blocking: bool
) -> TensorDict:
    return {
        "coords": tensor_to(batch["coords"], dev, non_blocking=non_blocking),
        "target_fourier": tensor_to(
            batch["target_fourier"], dev, non_blocking=non_blocking
        ),
    }
