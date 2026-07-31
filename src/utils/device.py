"""Device selection. The same model code runs on CUDA, Apple MPS and CPU."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """'auto' picks cuda -> mps -> cpu. Anything else is honoured verbatim."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
