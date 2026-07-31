"""Shared command-line plumbing for the project scripts.

Every entry point in ``scripts/`` builds its parser from :func:`base_parser` so
that configuration selection, scope overrides, seeding and logging behave
identically no matter which script is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .config import Config, load_config
from .logging_config import configure_logging, get_logger
from .reproducibility import SeedState, seed_everything
from .scopes import scope_names

__all__ = [
    "base_parser",
    "add_bootstrap_path",
    "config_from_args",
    "bootstrap",
]


def add_bootstrap_path() -> None:
    """Make ``src`` importable when a script is run without installation.

    Phase 3 installs the package properly; until then, scripts call this so the
    harness can be exercised from a bare checkout.
    """
    import sys

    src = Path(__file__).resolve().parents[1]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def base_parser(
    description: str, *, default_configs: Sequence[str] = ("base.yaml",)
) -> argparse.ArgumentParser:
    """Build an argument parser carrying the common project options.

    Args:
        description: Text shown by ``--help``.
        default_configs: Configuration files used when ``--config`` is omitted.

    Returns:
        A parser with ``--config``, ``--set``, ``--scope``, ``--seed``,
        ``--log-level``, ``--log-file`` and ``--print-config`` already defined.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        metavar="PATH",
        help=(
            "YAML configuration file; repeat to layer files left to right. "
            f"Defaults to {' '.join(default_configs)}."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="Override a config value by dotted key, e.g. --set training.epochs=40.",
    )
    parser.add_argument(
        "--scope",
        choices=list(scope_names()),
        help="Shorthand for --set dataset.scope=<scope>.",
    )
    parser.add_argument(
        "--seed", type=int, help="Shorthand for --set reproducibility.seed=<int>."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging threshold.",
    )
    parser.add_argument(
        "--log-file", metavar="PATH", help="Optional JSON Lines log destination."
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the fully resolved configuration and exit.",
    )
    parser.set_defaults(_default_configs=tuple(default_configs))
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Resolve configuration from parsed arguments.

    ``--scope`` and ``--seed`` are folded into the override list so that a
    single precedence rule applies: files, then environment, then ``--set``.
    """
    overrides: list[str] = list(getattr(args, "overrides", []) or [])
    if getattr(args, "scope", None):
        overrides.append(f"dataset.scope={args.scope}")
    if getattr(args, "seed", None) is not None:
        overrides.append(f"reproducibility.seed={args.seed}")

    configs = args.config or list(getattr(args, "_default_configs", ("base.yaml",)))
    return load_config(configs, cli_overrides=overrides)


def bootstrap(args: argparse.Namespace) -> tuple[Config, SeedState]:
    """Perform the standard start-up sequence for a script.

    Configures logging, resolves configuration, seeds every random source and
    reports the active scope.

    Returns:
        The resolved configuration and the resulting seed state.
    """
    configure_logging(args.log_level, json_file=getattr(args, "log_file", None))
    logger = get_logger("cli")

    config = config_from_args(args)
    reproducibility: dict[str, Any] = config.section("reproducibility")
    state = seed_everything(
        config.seed,
        deterministic=bool(reproducibility.get("deterministic", True)),
        cudnn_benchmark=bool(reproducibility.get("cudnn_benchmark", False)),
    )

    dataset = config.dataset
    logger.info(
        "scope=%s num_classes=%d seed=%d image_size=%dx%d",
        dataset.scope_name,
        dataset.num_classes,
        state.seed,
        *dataset.image_size,
        extra={
            "event": "bootstrap",
            "scope": dataset.scope_name,
            "num_classes": dataset.num_classes,
            "seed": state.seed,
            "config_sources": [str(p) for p in config.sources],
        },
    )
    return config, state
