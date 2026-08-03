"""Seeding and device selection.

Everyone runs this on a different machine -- one M-series Mac, one CUDA laptop,
one Colab session. None of that may change the result, so device choice is
automatic and seeding covers every library that draws random numbers.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all accelerators)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - signature fixed by DataLoader
    """DataLoader ``worker_init_fn`` so augmentation is reproducible across workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """``auto`` picks cuda -> mps -> cpu. Anything else is honoured verbatim."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_environment() -> dict[str, str]:
    """Recorded into every ``results.json`` so a surprising number can be traced."""
    import platform

    device = resolve_device("auto")
    name = str(device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
    return {
        "device": str(device),
        "device_name": name,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
