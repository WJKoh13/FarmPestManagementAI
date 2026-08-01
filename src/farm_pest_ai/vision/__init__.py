"""Custom CNN models, training engine and inference (Phases 6-9).

The public surface is re-exported here so callers write
``from farm_pest_ai.vision import build_model`` rather than reaching into
submodules. Torch is imported eagerly by everything in this package: unlike the
data layer, no path into these modules is reachable without it.
"""

from .blocks import (
    ConvBNAct,
    DepthwiseSeparableConv,
    DropPath,
    ResidualSeparableBlock,
    SqueezeExcite,
)
from .checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    CheckpointMetadata,
    best_checkpoint_path,
    last_checkpoint_path,
    load_checkpoint,
    load_model_for_inference,
    read_metadata,
    save_checkpoint,
)
from .metrics import (
    ClassificationMetrics,
    MetricsAccumulator,
    MetricsError,
    confusion_matrix,
    macro_f1,
    top_k_accuracy,
)
from .models import (
    MODEL_NAMES,
    BaselineCNN,
    CustomCNN,
    ModelConfig,
    ModelError,
    build_model,
    count_parameters,
    model_config_from_config,
    summarize_model,
)
from .training import (
    EpochResult,
    Trainer,
    TrainingConfig,
    TrainingError,
    build_optimizer,
    build_scheduler,
    build_trainer,
    training_config_from_config,
)

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "MODEL_NAMES",
    "BaselineCNN",
    "CheckpointError",
    "CheckpointMetadata",
    "ClassificationMetrics",
    "ConvBNAct",
    "CustomCNN",
    "DepthwiseSeparableConv",
    "DropPath",
    "EpochResult",
    "MetricsAccumulator",
    "MetricsError",
    "ModelConfig",
    "ModelError",
    "ResidualSeparableBlock",
    "SqueezeExcite",
    "Trainer",
    "TrainingConfig",
    "TrainingError",
    "best_checkpoint_path",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "build_trainer",
    "confusion_matrix",
    "count_parameters",
    "last_checkpoint_path",
    "load_checkpoint",
    "load_model_for_inference",
    "macro_f1",
    "model_config_from_config",
    "read_metadata",
    "save_checkpoint",
    "summarize_model",
    "top_k_accuracy",
    "training_config_from_config",
]
