"""Guard the environment contract established in Phase 3.

These tests protect two things that are easy to break silently: the dependency
lock staying usable (it pins CUDA wheels that are not on PyPI, so it needs its
extra index) and the interpreter actually being able to run the training stack.

The GPU-dependent assertions are marked ``gpu`` and skipped when PyTorch or CUDA
is absent, so the suite still passes on a CPU-only checkout or in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

#: Wheels published only on the PyTorch index, never on PyPI.
CUDA_PINNED = ("torch", "torchvision")

#: The CUDA build Phase 3 selected for the RTX 4070 Laptop (sm_89).
EXPECTED_CUDA_TAG = "+cu126"


@pytest.fixture()
def lock_lines(project_root: Path) -> list[str]:
    """Non-empty, non-comment lines of the dependency lock."""
    lock = project_root / "requirements-lock.txt"
    assert lock.is_file(), "requirements-lock.txt is missing; regenerate it (Phase 3)"
    return [
        line.strip()
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_lock_carries_pytorch_extra_index(lock_lines: list[str]) -> None:
    """The CUDA wheels are not on PyPI, so the lock must name their index."""
    index_lines = [line for line in lock_lines if line.startswith("--extra-index-url")]
    assert index_lines, (
        "requirements-lock.txt pins CUDA wheels but declares no --extra-index-url, "
        "so a clean `pip install -r` cannot resolve them"
    )
    assert any("download.pytorch.org" in line for line in index_lines)


def test_lock_pins_exact_versions(lock_lines: list[str]) -> None:
    """Every requirement is pinned with ``==``, never a floor or a range."""
    requirements = [line for line in lock_lines if not line.startswith("-")]
    assert requirements, "lock file contains no requirements"
    unpinned = [line for line in requirements if "==" not in line]
    assert not unpinned, f"lock entries are not pinned exactly: {unpinned}"


@pytest.mark.parametrize("package", CUDA_PINNED)
def test_lock_pins_cuda_build(lock_lines: list[str], package: str) -> None:
    """Both PyTorch packages must be the CUDA build, not the CPU default."""
    matches = [line for line in lock_lines if line.lower().startswith(f"{package}==")]
    assert matches, f"{package} is absent from requirements-lock.txt"
    assert EXPECTED_CUDA_TAG in matches[0], (
        f"{matches[0]} is not the {EXPECTED_CUDA_TAG} build; a CPU wheel would "
        "silently make training far slower"
    )


def test_lock_excludes_the_project_itself(lock_lines: list[str]) -> None:
    """The project is installed with ``-e . --no-deps`` and must not self-pin."""
    assert not [line for line in lock_lines if line.lower().startswith("farm-pest-ai")]


def test_interpreter_is_supported() -> None:
    """The harness targets Python 3.10+."""
    assert sys.version_info >= (3, 10)


def test_interpreter_is_not_mingw() -> None:
    """MSYS2/MinGW Python cannot run PyTorch; Phase 3 requires official CPython.

    Guards risk 1: an MSYS2 interpreter shadows CPython on this machine's PATH,
    and its ``mingw_*`` platform tag has no PyTorch wheels at all.
    """
    import sysconfig

    platform_tag = sysconfig.get_platform()
    assert "mingw" not in platform_tag.lower(), (
        f"running under {platform_tag}; use the project venv built from official "
        "CPython (.venv\\Scripts\\python.exe), not MSYS2 Python"
    )


@pytest.mark.gpu
def test_torch_reports_cuda() -> None:
    """PyTorch must see CUDA, and must be a CUDA build rather than CPU-only."""
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    assert torch.version.cuda is not None
    assert torch.cuda.device_count() >= 1


@pytest.mark.gpu
def test_conv_runs_on_gpu_at_project_input_size() -> None:
    """A conv at 160x160 must actually execute on the GPU.

    ``torch.cuda.is_available()`` alone does not prove kernels launch; a driver
    or toolkit mismatch surfaces only when real work runs.
    """
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    conv = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
    batch = torch.randn(2, 3, 160, 160, device="cuda")
    output = conv(batch)
    torch.cuda.synchronize()

    assert output.shape == (2, 8, 160, 160)
    assert torch.isfinite(output).all()
