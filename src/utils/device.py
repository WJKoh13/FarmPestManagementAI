"""Device selection for Intel XPU, CUDA, Apple MPS, and CPU."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Select an available accelerator, or honour an explicit device name."""
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
