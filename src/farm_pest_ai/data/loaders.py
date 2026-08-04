"""Building ``DataLoader`` objects and the statistics derived from them.

This module assembles the pieces: a derived manifest becomes a
:class:`~farm_pest_ai.data.dataset.PestImageDataset`, gets the transform
pipeline for its split, and is wrapped in a ``DataLoader`` configured from the
``runtime`` section.

Three project rules are enforced here rather than left to the caller:

Training-only randomness
    Only the training loader shuffles and only the training pipeline augments.
    Validation and test loaders keep official manifest order, so a per-image
    prediction file lines up with the manifest row by row.

Training-only statistics
    Class weights and sampler weights are computed from the training dataset and
    from nothing else. :func:`build_loaders` never looks at validation or test
    labels for this purpose.

Reproducible workers
    Worker seeds derive from the run seed through
    :func:`farm_pest_ai.reproducibility.derive_seed`, and the shuffle uses an
    explicit generator, so two runs with the same seed see batches in the same
    order.

Torch is imported inside the functions that need it, keeping this module
importable in an environment without the training extras.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from ..reproducibility import derive_seed, worker_init_fn
from ..scopes import CLASS_MAPPING_VERSION, ScopeSpec
from .dataset import PestImageDataset, class_weights
from .manifests import SPLITS
from .transforms import (
    EVAL_SPLITS,
    PREPROCESSING_VERSION,
    PreprocessingConfig,
    build_transform,
    describe_transform,
    preprocessing_config_from_config,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from ..config import Config

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "LoaderBundle",
    "LoaderError",
    "RuntimeConfig",
    "build_dataset",
    "build_datasets",
    "build_loader",
    "build_loaders",
    "resolve_device",
    "runtime_config_from_config",
    "sampler_weights",
]

#: Used when ``training.batch_size`` is absent. Phase 1 measured only ~4.1 GB of
#: the RTX 4070 Laptop's 8 GB free under a normal desktop session, and TRAINING.md
#: fixes 64 at 160x160 as the starting point.
DEFAULT_BATCH_SIZE = 64

#: Used when ``training.class_weighting_beta`` is absent. Matches the
#: :func:`~farm_pest_ai.data.dataset.class_weights` default, so an existing
#: configuration that names only ``class_weighting`` behaves exactly as before.
#: Beta governs the entire strength of the ``effective`` scheme: on full102's 82x
#: imbalance 0.9999 gives a 69.5x weight ratio and 0.999 gives 23.5x.
DEFAULT_CLASS_WEIGHT_BETA = 0.9999


class LoaderError(RuntimeError):
    """Raised when a loader cannot be built from the given configuration."""


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved ``runtime`` section governing DataLoader behaviour.

    Attributes:
        device: ``"cuda"``, ``"cpu"`` or ``"auto"`` as written in configuration.
            Resolution to a concrete device happens in :func:`resolve_device`,
            which refuses to fall back silently.
        num_workers: DataLoader worker processes. Forced to 0 on the evaluation
            path only if configuration says so; the default applies everywhere.
        pin_memory: Whether to pin host memory. Only meaningful for CUDA, and
            disabled automatically on CPU to avoid a pointless copy.
        persistent_workers: Whether workers survive between epochs. Requires
            ``num_workers > 0``.
        prefetch_factor: Batches prefetched per worker. Requires
            ``num_workers > 0``.
        drop_last: Whether the training loader drops a short final batch. On for
            training so BatchNorm never sees a single-sample batch; always off
            for evaluation, where every image must be scored exactly once.
    """

    device: str = "auto"
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True

    def validate(self) -> RuntimeConfig:
        """Check the section for self-consistency.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            LoaderError: If a value is negative or a worker-dependent option is
                requested with no workers.
        """
        if self.num_workers < 0:
            raise LoaderError(f"runtime.num_workers must be >= 0, got {self.num_workers}")
        if self.prefetch_factor < 1:
            raise LoaderError(
                f"runtime.prefetch_factor must be >= 1, got {self.prefetch_factor}"
            )
        if self.num_workers == 0 and self.persistent_workers:
            raise LoaderError(
                "runtime.persistent_workers requires runtime.num_workers > 0; set "
                "num_workers or disable persistent_workers"
            )
        if self.device not in ("auto", "cpu", "cuda") and not self.device.startswith(
            "cuda:"
        ):
            raise LoaderError(
                f"runtime.device must be 'auto', 'cpu', 'cuda' or 'cuda:<n>', got "
                f"{self.device!r}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "device": self.device,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "drop_last": self.drop_last,
        }


def runtime_config_from_config(config: Config) -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` from the ``runtime`` section."""
    section = config.section("runtime")
    defaults = RuntimeConfig()
    resolved = RuntimeConfig(
        device=str(section.get("device", defaults.device)),
        num_workers=int(section.get("num_workers", defaults.num_workers)),
        pin_memory=bool(section.get("pin_memory", defaults.pin_memory)),
        persistent_workers=bool(
            section.get("persistent_workers", defaults.persistent_workers)
        ),
        prefetch_factor=int(section.get("prefetch_factor", defaults.prefetch_factor)),
        drop_last=bool(section.get("drop_last", defaults.drop_last)),
    )
    return resolved.validate()


def resolve_device(requested: str, *, allow_cpu_fallback: bool = True) -> str:
    """Resolve a configured device string to a concrete one.

    ``"auto"`` becomes ``"cuda"`` when CUDA is available and ``"cpu"``
    otherwise. An explicit ``"cuda"`` request with no CUDA present raises unless
    ``allow_cpu_fallback`` is set: the project rules forbid an approved full
    training run from silently degrading to CPU, which would take days instead
    of hours and produce results the log would not explain.

    Args:
        requested: Device string from configuration.
        allow_cpu_fallback: Whether an explicit CUDA request may fall back.

    Returns:
        ``"cpu"``, ``"cuda"`` or ``"cuda:<n>"``.

    Raises:
        LoaderError: If CUDA is explicitly requested but unavailable and
            fallback is not allowed.
    """
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        available = False
    else:
        available = torch.cuda.is_available()

    if requested == "auto":
        return "cuda" if available else "cpu"
    if available:
        return requested
    if allow_cpu_fallback:
        return "cpu"
    raise LoaderError(
        f"device {requested!r} was requested explicitly but CUDA is not available; "
        f"refusing to fall back to CPU silently"
    )


# -- datasets -----------------------------------------------------------


def build_dataset(
    config: Config,
    split: str,
    *,
    preprocessing: PreprocessingConfig | None = None,
    augment: bool | None = None,
    verify_files: bool = False,
) -> PestImageDataset:
    """Build the dataset for one split from a resolved configuration.

    Args:
        config: Resolved project configuration.
        split: One of :data:`farm_pest_ai.data.manifests.SPLITS`.
        preprocessing: Preprocessing to use. Derived from ``config`` when
            omitted.
        augment: Force augmentation on or off for the training split. ``None``
            uses the configured setting. Passing ``False`` yields the
            deterministic pipeline, which is how the determinism check compares
            two passes over the training data.
        verify_files: Whether to stat every referenced image at construction.

    Returns:
        The dataset, with the split's transform already attached.

    Raises:
        LoaderError: If ``split`` is unknown or an evaluation split is asked to
            augment.
        DatasetError: If the derived manifest is missing or inconsistent.
    """
    if split not in SPLITS:
        raise LoaderError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    if augment and split in EVAL_SPLITS:
        raise LoaderError(
            f"refusing to augment {split!r}: validation and test preprocessing must "
            f"stay deterministic"
        )

    resolved = preprocessing or preprocessing_config_from_config(config)
    if augment is not None and split == "train":
        resolved = replace(
            resolved,
            augmentation=replace(resolved.augmentation, enabled=bool(augment)),
        ).validate()

    paths = config.paths
    return PestImageDataset.from_manifest(
        paths.processed_dir,
        config.dataset.scope,
        split,
        paths.images_dir,
        transform=build_transform(resolved, split),
        manifest_version=config.dataset.manifest_version,
        verify_files=verify_files,
    )


def build_datasets(
    config: Config,
    splits: tuple[str, ...] = SPLITS,
    *,
    preprocessing: PreprocessingConfig | None = None,
    verify_files: bool = False,
) -> dict[str, PestImageDataset]:
    """Build datasets for several splits, sharing one preprocessing config.

    Note that including ``"test"`` reads the test manifest. Nothing before
    Phase 9 should do so; the default here is all three splits only because the
    verification script needs to prove that the test pipeline is deterministic
    without ever training on it.
    """
    resolved = preprocessing or preprocessing_config_from_config(config)
    return {
        split: build_dataset(
            config, split, preprocessing=resolved, verify_files=verify_files
        )
        for split in splits
    }


# -- sampling and weighting ---------------------------------------------


def sampler_weights(
    dataset: PestImageDataset, *, scheme: str = "inverse_sqrt"
) -> tuple[float, ...]:
    """Per-sample weights for a ``WeightedRandomSampler``.

    Computed from the **training** dataset's labels. Each sample receives its
    class's weight, so rare classes are drawn more often.

    Args:
        dataset: The training dataset.
        scheme: Passed to :func:`~farm_pest_ai.data.dataset.class_weights`.

    Returns:
        One weight per sample, in dataset order.

    Raises:
        LoaderError: If ``dataset`` is not the training split, since sampling
            from an evaluation split would change which images are scored.
    """
    if dataset.split != "train":
        raise LoaderError(
            f"sampler weights may only be derived from the training split, got "
            f"{dataset.split!r}"
        )
    per_class = class_weights(dataset.targets, dataset.num_classes, scheme=scheme)
    return tuple(per_class[label] for label in dataset.targets)


# -- loaders ------------------------------------------------------------


def build_loader(
    dataset: PestImageDataset,
    runtime: RuntimeConfig,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 1337,
    shuffle: bool | None = None,
    sampler: Any = None,
    device: str | None = None,
) -> Any:
    """Wrap a dataset in a ``DataLoader``.

    Args:
        dataset: The dataset to iterate.
        runtime: Resolved runtime settings.
        batch_size: Images per batch.
        seed: Run seed; worker seeds and the shuffle generator derive from it.
        shuffle: Force shuffling on or off. ``None`` shuffles the training split
            and preserves manifest order for evaluation splits.
        sampler: Optional sampler. Mutually exclusive with shuffling, which
            PyTorch enforces; this function raises the clearer error first.
        device: Concrete device. Used only to decide whether pinning host memory
            is worthwhile; pinning is disabled on CPU.

    Returns:
        A configured ``torch.utils.data.DataLoader``.

    Raises:
        LoaderError: If ``batch_size`` is not positive, or both a sampler and
            shuffling are requested.
    """
    from torch.utils.data import DataLoader

    if batch_size <= 0:
        raise LoaderError(f"batch_size must be positive, got {batch_size}")

    is_train = dataset.split == "train"
    if shuffle is None:
        shuffle = is_train
    if sampler is not None and shuffle:
        raise LoaderError(
            "a sampler and shuffle=True are mutually exclusive; the sampler already "
            "decides the order"
        )

    resolved_device = device or resolve_device(runtime.device)
    workers = runtime.num_workers
    # A short final batch gives BatchNorm too little to normalise over, so it is
    # dropped for training. Evaluation must score every image exactly once, so
    # it never drops.
    drop_last = bool(runtime.drop_last) if is_train else False

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "drop_last": drop_last,
        "pin_memory": bool(runtime.pin_memory) and resolved_device.startswith("cuda"),
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = bool(shuffle)

    if workers > 0:
        kwargs["persistent_workers"] = bool(runtime.persistent_workers)
        kwargs["prefetch_factor"] = runtime.prefetch_factor
        # Each worker gets its own deterministic stream, derived from the run
        # seed and the split, so the training and validation loaders do not draw
        # from the same sequence.
        kwargs["worker_init_fn"] = partial(
            worker_init_fn, base_seed=derive_seed(seed, "loader", dataset.split)
        )

    if shuffle:
        import torch

        generator = torch.Generator()
        generator.manual_seed(derive_seed(seed, "shuffle", dataset.split))
        kwargs["generator"] = generator

    return DataLoader(dataset, **kwargs)


@dataclass(frozen=True)
class LoaderBundle:
    """Everything a training run needs from the data layer.

    Attributes:
        loaders: Split name to ``DataLoader``.
        datasets: Split name to :class:`PestImageDataset`.
        preprocessing: The preprocessing configuration actually applied.
        runtime: The resolved runtime settings.
        scope: The active scope.
        device: The concrete device the loaders were configured for.
        batch_size: Images per batch.
        seed: The run seed the loaders were derived from.
        class_weights: Per-class loss weights derived from the **training**
            split, or ``None`` when ``training.class_weighting`` is ``none``.
    """

    loaders: Mapping[str, Any]
    datasets: Mapping[str, PestImageDataset]
    preprocessing: PreprocessingConfig
    runtime: RuntimeConfig
    scope: ScopeSpec
    device: str
    batch_size: int
    seed: int
    class_weights: tuple[float, ...] | None = None
    _extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_classes(self) -> int:
        """Number of output classes, derived from the scope."""
        return self.scope.num_classes

    def describe(self) -> dict[str, Any]:
        """Return the JSON-serialisable record stored with every run.

        Includes the preprocessing fingerprint and the ordered step names of
        each pipeline, which together make it possible to prove after the fact
        that evaluation was deterministic.
        """
        return {
            "scope": self.scope.name,
            "num_classes": self.num_classes,
            "class_mapping_version": CLASS_MAPPING_VERSION,
            "preprocessing_version": self.preprocessing.version,
            "preprocessing_module_version": PREPROCESSING_VERSION,
            "preprocessing_fingerprint": self.preprocessing.fingerprint,
            "preprocessing": self.preprocessing.to_dict(),
            "runtime": self.runtime.to_dict(),
            "device": self.device,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "class_weighting": self._extra.get("class_weighting", "none"),
            # Recorded even when the scheme ignores it, so a run's summary always
            # states the exact parameters its weight vector came from rather
            # than leaving beta to be inferred from a default that may change.
            "class_weighting_beta": self._extra.get(
                "class_weighting_beta", DEFAULT_CLASS_WEIGHT_BETA
            ),
            "class_weights": (
                list(self.class_weights) if self.class_weights is not None else None
            ),
            "splits": {
                split: {
                    **dataset.describe(),
                    "batches": len(self.loaders[split]) if split in self.loaders else None,
                    "shuffled": split == "train",
                    "augmented": (
                        split == "train" and self.preprocessing.augmentation.enabled
                    ),
                    "pipeline": list(describe_transform(dataset.transform)),
                }
                for split, dataset in self.datasets.items()
            },
        }


def build_loaders(
    config: Config,
    splits: tuple[str, ...] = ("train", "validation"),
    *,
    batch_size: int | None = None,
    seed: int | None = None,
    preprocessing: PreprocessingConfig | None = None,
    augment: bool | None = None,
    verify_files: bool = False,
    allow_cpu_fallback: bool = True,
) -> LoaderBundle:
    """Build datasets and loaders for the requested splits.

    The default omits ``"test"`` deliberately: nothing before Phase 9 may read
    it, so a caller has to name it explicitly.

    Class weights, when ``training.class_weighting`` requests them, are computed
    from the training dataset alone. If ``"train"`` is not among ``splits`` the
    weights are ``None`` rather than derived from whatever else is present.

    Args:
        config: Resolved project configuration.
        splits: Splits to build.
        batch_size: Override ``training.batch_size``.
        seed: Override ``reproducibility.seed``.
        preprocessing: Override the derived preprocessing configuration.
        augment: Force training augmentation on or off.
        verify_files: Whether to stat every referenced image.
        allow_cpu_fallback: Whether an explicit CUDA request may fall back to
            CPU. Training scripts pass ``False`` for an approved run.

    Returns:
        A :class:`LoaderBundle`.

    Raises:
        LoaderError: If a split is unknown or the configuration is unusable.
        DatasetError: If a derived manifest is missing or inconsistent.
    """
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise LoaderError(f"unknown split(s) {unknown}; expected from {list(SPLITS)}")
    if not splits:
        raise LoaderError("at least one split must be requested")

    resolved_preprocessing = preprocessing or preprocessing_config_from_config(config)
    runtime = runtime_config_from_config(config)
    device = resolve_device(runtime.device, allow_cpu_fallback=allow_cpu_fallback)

    training = config.section("training")
    resolved_batch = int(
        batch_size
        if batch_size is not None
        else training.get("batch_size", DEFAULT_BATCH_SIZE)
    )
    resolved_seed = int(seed if seed is not None else config.seed)

    datasets = {
        split: build_dataset(
            config,
            split,
            preprocessing=resolved_preprocessing,
            augment=augment if split == "train" else None,
            verify_files=verify_files,
        )
        for split in splits
    }

    scheme = str(training.get("class_weighting", "none"))
    # The effective-number scheme's strength is governed entirely by beta, and
    # the difference is large: on full102's 82x imbalance the default 0.9999
    # produces a 69.5x weight ratio while 0.999 produces 23.5x. Leaving it
    # hard-coded would make "effective" a single fixed intensity rather than a
    # tunable correction, so it is configuration like every other knob.
    beta = float(training.get("class_weighting_beta", DEFAULT_CLASS_WEIGHT_BETA))
    weights: tuple[float, ...] | None = None
    if scheme != "none":
        train_dataset = datasets.get("train")
        if train_dataset is None:
            raise LoaderError(
                f"training.class_weighting={scheme!r} needs the training split, but "
                f"only {list(splits)} were requested; class statistics may never come "
                f"from validation or test data"
            )
        try:
            weights = class_weights(
                train_dataset.targets,
                train_dataset.num_classes,
                scheme=scheme,
                beta=beta,
            )
        except ValueError as exc:
            raise LoaderError(str(exc)) from exc

    loaders = {
        split: build_loader(
            dataset,
            runtime,
            batch_size=resolved_batch,
            seed=resolved_seed,
            device=device,
        )
        for split, dataset in datasets.items()
    }

    return LoaderBundle(
        loaders=loaders,
        datasets=datasets,
        preprocessing=resolved_preprocessing,
        runtime=runtime,
        scope=config.dataset.scope,
        device=device,
        batch_size=resolved_batch,
        seed=resolved_seed,
        class_weights=weights,
        _extra={"class_weighting": scheme, "class_weighting_beta": beta},
    )
