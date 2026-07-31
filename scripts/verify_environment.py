#!/usr/bin/env python3
"""Report on the runtime environment and the project harness.

Checks the interpreter, optional scientific stack, GPU availability and the
project's own configuration wiring, then prints a table and exits non-zero if a
required check failed. Optional checks (PyTorch, CUDA) are reported but do not
fail the run before Phase 3 provisions them.

Examples:
    python scripts/verify_environment.py
    python scripts/verify_environment.py --scope full102 --json
    python scripts/verify_environment.py --require-torch
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from farm_pest_ai.cli import base_parser, config_from_args  # noqa: E402
from farm_pest_ai.config import ConfigError  # noqa: E402
from farm_pest_ai.logging_config import configure_logging, get_logger  # noqa: E402
from farm_pest_ai.reproducibility import environment_snapshot  # noqa: E402
from farm_pest_ai.scopes import SCOPES, num_classes_for  # noqa: E402

#: Packages required for the harness itself to function.
REQUIRED_PACKAGES = ("yaml",)

#: Packages needed from Phase 3 onward; absence is reported, not fatal.
OPTIONAL_PACKAGES = (
    "torch", "torchvision", "numpy", "PIL", "pandas", "sklearn",
    "fastapi", "uvicorn", "streamlit", "pydantic", "pytest", "httpx",
)


@dataclass
class Check:
    """The outcome of a single environment check."""

    name: str
    ok: bool
    detail: str
    required: bool = True
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Human-readable status token."""
        if self.ok:
            return "OK"
        return "FAIL" if self.required else "ABSENT"


def _module_version(name: str) -> str:
    """Best-effort version string for an imported module."""
    module = importlib.import_module(name)
    for attribute in ("__version__", "VERSION", "version"):
        value = getattr(module, attribute, None)
        if isinstance(value, str):
            return value
    return "unknown"


def check_python() -> Check:
    """Verify the interpreter is Python 3.10 or newer."""
    version = sys.version_info
    ok = version >= (3, 10)
    return Check(
        name="python",
        ok=ok,
        detail=f"{version.major}.{version.minor}.{version.micro} at {sys.executable}",
        data={"executable": sys.executable},
    )


def check_packages() -> list[Check]:
    """Report which required and optional packages import successfully."""
    checks: list[Check] = []
    for name in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
        required = name in REQUIRED_PACKAGES
        try:
            version = _module_version(name)
        except ImportError:
            checks.append(
                Check(f"package:{name}", False, "not installed", required=required)
            )
        else:
            checks.append(
                Check(f"package:{name}", True, version, required=required,
                      data={"version": version})
            )
    return checks


def check_torch_cuda() -> list[Check]:
    """Report PyTorch CUDA availability and GPU properties."""
    try:
        import torch
    except ImportError:
        return [
            Check(
                "cuda",
                False,
                "torch not installed; provisioned in Phase 3",
                required=False,
            )
        ]

    available = torch.cuda.is_available()
    if not available:
        return [
            Check(
                "cuda",
                False,
                f"torch {torch.__version__} present but CUDA unavailable "
                f"(built for CUDA {getattr(torch.version, 'cuda', None)})",
                required=False,
            )
        ]

    checks = [
        Check(
            "cuda",
            True,
            f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
            f"{torch.cuda.device_count()} device(s)",
            required=False,
        )
    ]
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        total = properties.total_memory / 2**20
        free, _ = torch.cuda.mem_get_info(index)
        checks.append(
            Check(
                f"gpu:{index}",
                True,
                f"{properties.name}, {total:.0f} MiB total, "
                f"{free / 2**20:.0f} MiB free, sm_{properties.major}{properties.minor}",
                required=False,
                data={"name": properties.name, "total_mib": round(total)},
            )
        )
    return checks


def check_harness(args: argparse.Namespace) -> list[Check]:
    """Verify the project's own configuration and scope wiring."""
    checks: list[Check] = []

    for name, spec in SCOPES.items():
        try:
            spec.validate()
            derived = num_classes_for(name)
            checks.append(
                Check(
                    f"scope:{name}",
                    True,
                    f"{derived} classes, mapping {'identity' if spec.is_identity else 'remapped'}",
                    data={"num_classes": derived},
                )
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            checks.append(Check(f"scope:{name}", False, f"{type(exc).__name__}: {exc}"))

    try:
        config = config_from_args(args)
        dataset = config.dataset
        checks.append(
            Check(
                "config",
                True,
                f"scope={dataset.scope_name} num_classes={dataset.num_classes} "
                f"image_size={dataset.image_size[0]}x{dataset.image_size[1]} "
                f"seed={config.seed}",
                data={
                    "scope": dataset.scope_name,
                    "num_classes": dataset.num_classes,
                    "sources": [str(p) for p in config.sources],
                },
            )
        )
    except (ConfigError, OSError) as exc:
        checks.append(Check("config", False, f"{type(exc).__name__}: {exc}"))
        return checks

    paths = config.paths
    checks.append(
        Check(
            "dataset_root",
            paths.classification_root.is_dir(),
            str(paths.classification_root),
            required=False,
        )
    )
    checks.append(
        Check("images_dir", paths.images_dir.is_dir(), str(paths.images_dir), required=False)
    )

    usage = shutil.disk_usage(paths.project_root)
    free_gb = usage.free / 2**30
    checks.append(
        Check(
            "disk",
            free_gb >= 20,
            f"{free_gb:.1f} GiB free at {paths.project_root}",
            data={"free_gib": round(free_gb, 1)},
        )
    )
    return checks


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script."""
    parser = base_parser(
        "Verify the runtime environment and project harness.",
        default_configs=("base.yaml",),
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table."
    )
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="Treat PyTorch and CUDA as required, failing the run when absent.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_file=getattr(args, "log_file", None))
    logger = get_logger("verify_environment")

    if args.print_config:
        print(config_from_args(args).to_yaml())
        return 0

    checks: list[Check] = [check_python()]
    checks.extend(check_packages())
    checks.extend(check_torch_cuda())
    checks.extend(check_harness(args))

    if args.require_torch:
        for check in checks:
            if check.name in {"cuda", "package:torch", "package:torchvision"}:
                check.required = True

    failures = [c for c in checks if c.required and not c.ok]

    if args.json:
        print(
            json.dumps(
                {
                    "checks": [
                        {
                            "name": c.name,
                            "ok": c.ok,
                            "status": c.status,
                            "required": c.required,
                            "detail": c.detail,
                            **({"data": c.data} if c.data else {}),
                        }
                        for c in checks
                    ],
                    "environment": environment_snapshot(),
                    "failures": [c.name for c in failures],
                },
                indent=2,
                default=str,
            )
        )
    else:
        width = max(len(c.name) for c in checks) + 2
        print(f"\n{'CHECK'.ljust(width)}{'STATUS'.ljust(9)}DETAIL")
        print("-" * (width + 9 + 60))
        for check in checks:
            print(f"{check.name.ljust(width)}{check.status.ljust(9)}{check.detail}")
        print()
        if failures:
            print(f"{len(failures)} required check(s) failed: "
                  f"{', '.join(c.name for c in failures)}")
        else:
            print("All required checks passed.")
        absent = [c for c in checks if not c.required and not c.ok]
        if absent:
            print(f"Optional/absent: {', '.join(c.name for c in absent)}")

    logger.info(
        "environment verification finished",
        extra={"event": "verify_environment", "failures": len(failures)},
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
