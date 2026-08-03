"""Shared harness for benchmarking pest-classification models on IP102.

The deal: this package owns the data pipeline, the training protocol and the
metrics. Your notebook owns the architecture. Because everyone imports the same
loaders and calls the same ``save_run``, the numbers in the comparison table
differ only by the thing being compared.

Typical notebook::

    from ip102_bench import load_protocol, build_dataloaders, train_model, save_run

    protocol = load_protocol()
    loaders  = build_dataloaders(protocol)

    model  = MyNet(num_classes=protocol.num_classes)
    result = train_model(model, protocol, loaders)

    save_run(model=model, model_name="mynet", protocol=protocol, result=result,
             test_loader=loaders["test"], pretrained=False, author="your name")
"""

from .artifacts import new_run_id, predict_split, save_run
from .data import IP102Dataset, build_dataloaders, build_dataset
from .metrics import compute_metrics, count_parameters, measure_cpu_latency
from .plots import plot_confusion_matrix, plot_curves, plot_per_class_f1
from .protocol import Protocol, load_protocol, resolve_path
from .runtime import describe_environment, resolve_device, seed_everything
from .training import TrainingResult, result_from_external_run, train_model
from .transforms import build_eval_transform, build_train_transform

__all__ = [
    "IP102Dataset",
    "Protocol",
    "TrainingResult",
    "build_dataloaders",
    "build_dataset",
    "build_eval_transform",
    "build_train_transform",
    "compute_metrics",
    "count_parameters",
    "describe_environment",
    "load_protocol",
    "measure_cpu_latency",
    "new_run_id",
    "plot_confusion_matrix",
    "plot_curves",
    "plot_per_class_f1",
    "predict_split",
    "resolve_device",
    "resolve_path",
    "save_run",
    "seed_everything",
    "train_model",
]
