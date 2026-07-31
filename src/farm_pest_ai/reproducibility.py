"""Seeding and environment capture for reproducible experiments.

PyTorch and NumPy are optional at import time so that the harness, the
configuration layer and the tests remain usable before the training environment
is provisioned in Phase 3. Whatever is installed gets seeded; whatever is not is
reported as absent in the environment snapshot rather than raising.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_SEED",
    "SeedState",
    "seed_everything",
    "worker_init_fn",
    "derive_seed",
    "environment_snapshot",
    "git_revision",
]

#: Seed used when configuration does not specify one.
DEFAULT_SEED = 1337

#: Upper bound for seeds accepted by NumPy's legacy generator.
_SEED_MODULUS = 2**32


@dataclass(frozen=True)
class SeedState:
    """Record of what was seeded, stored alongside every checkpoint."""

    seed: int
    deterministic: bool
    cudnn_benchmark: bool
    seeded: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable child seed from a base seed and arbitrary parts.

    Used for per-worker and per-fold seeds so that changing one does not perturb
    the others. The result is deterministic across processes and platforms,
    unlike :func:`hash`.

    Args:
        base_seed: The run's global seed.
        *parts: Values identifying the sub-stream, e.g. ``"loader", 3``.

    Returns:
        A seed in ``[0, 2**32)``.
    """
    payload = "|".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % _SEED_MODULUS


def seed_everything(
    seed: int = DEFAULT_SEED,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> SeedState:
    """Seed every random source that is available.

    Seeds :mod:`random`, ``PYTHONHASHSEED``, NumPy and PyTorch (CPU and CUDA)
    when installed.

    Args:
        seed: The global seed.
        deterministic: Request deterministic kernels. Slower, but makes runs
            comparable; disable for throughput benchmarking.
        cudnn_benchmark: Allow cuDNN autotuning. Mutually exclusive in spirit
            with ``deterministic`` and ignored when it is set.

    Returns:
        A :class:`SeedState` describing what was actually seeded.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    seed = int(seed) % _SEED_MODULUS

    seeded: list[str] = []

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded.append("python")

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
        seeded.append("numpy")

    try:
        import torch
    except ImportError:
        return SeedState(seed, deterministic, cudnn_benchmark, tuple(seeded))

    torch.manual_seed(seed)
    seeded.append("torch")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        seeded.append("torch.cuda")

    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = bool(deterministic)
        cudnn.benchmark = False if deterministic else bool(cudnn_benchmark)

    if deterministic:
        # cuBLAS needs this to make matmul reductions reproducible; setting it
        # after CUDA initialisation has no effect, hence the early assignment.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            seeded.append("torch.deterministic")
        except (AttributeError, RuntimeError):
            # Older builds, or kernels without a deterministic implementation.
            pass

    return SeedState(seed, deterministic, cudnn_benchmark, tuple(seeded))


def worker_init_fn(worker_id: int, base_seed: int = DEFAULT_SEED) -> None:
    """Seed a DataLoader worker deterministically.

    Bind ``base_seed`` with :func:`functools.partial` before handing this to a
    ``DataLoader``.
    """
    seed = derive_seed(base_seed, "loader_worker", worker_id)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % _SEED_MODULUS)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def git_revision(cwd: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Capture the current Git revision, if the project is a repository."""
    info: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }
    try:
        for key, command in commands.items():
            result = subprocess.run(
                command, cwd=cwd, capture_output=True, text=True, timeout=10, check=False
            )
            if result.returncode == 0:
                info[key] = result.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=False,
        )
        if status.returncode == 0:
            info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # Git absent or not a repository; leave the fields as None.
        pass
    return info


def environment_snapshot(include_git: bool = True) -> dict[str, Any]:
    """Describe the runtime environment for the experiment record.

    Every checkpoint and metrics file embeds this so a result can be traced back
    to the exact interpreter, library versions and GPU that produced it.
    """
    snapshot: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import numpy as np
    except ImportError:
        snapshot["numpy_version"] = None
    else:
        snapshot["numpy_version"] = np.__version__

    try:
        import torch
    except ImportError:
        snapshot.update(
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            cudnn_version=None,
            gpu_count=0,
            gpus=[],
        )
    else:
        cuda_available = torch.cuda.is_available()
        gpus: list[dict[str, Any]] = []
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                gpus.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory_mib": round(properties.total_memory / 2**20),
                        "capability": f"{properties.major}.{properties.minor}",
                    }
                )
        cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
        snapshot.update(
            torch_version=torch.__version__,
            cuda_available=cuda_available,
            cuda_version=getattr(torch.version, "cuda", None),
            cudnn_version=cudnn.version() if cudnn is not None and cuda_available else None,
            gpu_count=len(gpus),
            gpus=gpus,
        )

    try:
        import torchvision
    except ImportError:
        snapshot["torchvision_version"] = None
    else:
        snapshot["torchvision_version"] = torchvision.__version__

    if include_git:
        snapshot["git"] = git_revision()

    return snapshot
