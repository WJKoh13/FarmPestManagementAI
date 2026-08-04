from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Preprocessing the app falls back to when a checkpoint does not record its own.
# These are ProPestNet's; see protocol.yaml -> subsets.detection_top15.
DEFAULT_IMAGE_SIZE = 128
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


@dataclass
class LoadedModel:
    """A model plus everything needed to feed it the way it was trained.

    The app never guesses preprocessing. If a checkpoint cannot say how it was
    trained, ``model`` is None and the UI says so -- serving predictions through
    the wrong transform is worse than serving none, because it fails quietly.
    """

    model: object | None = None
    path: Path | None = None
    is_trained: bool = False
    under_trained: bool = False
    reason: str = ""
    class_names: list[str] = field(default_factory=list)
    display_names: list[str] = field(default_factory=list)
    image_size: int = DEFAULT_IMAGE_SIZE
    mean: list[float] = field(default_factory=lambda: list(DEFAULT_MEAN))
    std: list[float] = field(default_factory=lambda: list(DEFAULT_STD))
    inference_views: list[str] = field(default_factory=lambda: ["centre", "whole"])
    tta_flip: bool = True
    use_box_crop: bool = True
    protocol_warning: str = ""
    results: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _score(results: dict) -> float:
    """Rank a run by the best macro F1 it can evidence.

    Runs write this in three different shapes: nested under ``test``, nested
    under ``test_with_tta``, or flat at the top level. Read all three rather
    than silently scoring every run -1.
    """
    for section in ("test_with_tta", "test"):
        block = results.get(section)
        if isinstance(block, dict) and block.get("macro_f1") is not None:
            return float(block["macro_f1"])
    for key in ("macro_f1", "test_macro_f1", "best_val_macro_f1"):
        if results.get(key) is not None:
            return float(results[key])
    return -1.0


def find_runs(runs_dir: Path = PROJECT_ROOT / "runs") -> list[tuple[float, Path, dict]]:
    """Every run holding both a checkpoint and a results.json, best first.

    Searches ``runs/<run>/`` and ``runs/<model>/<run>/`` -- both layouts exist
    in this repo, and globbing only the deeper one made real runs invisible.
    """
    candidates: list[tuple[float, Path, dict]] = []
    seen: set[Path] = set()
    for pattern in ("*/results.json", "*/*/results.json"):
        for results_path in sorted(runs_dir.glob(pattern)):
            checkpoint = results_path.parent / "best_model.pt"
            if not checkpoint.exists() or checkpoint in seen:
                continue
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            seen.add(checkpoint)
            candidates.append((_score(results), checkpoint, results))
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def describe_runs(num_classes: int | None = None,
                  runs_dir: Path = PROJECT_ROOT / "runs") -> list[dict]:
    """Summarize every run for a model picker, without loading any weights.

    Reads results.json only. Loading each checkpoint just to list it would mean
    reading hundreds of megabytes to draw a dropdown, so incompatibility is
    reported from the metadata and confirmed at load time.
    """
    described: list[dict] = []
    for score, checkpoint, results in find_runs(runs_dir):
        classes = results.get("classes") or []
        label = str(checkpoint.parent.relative_to(runs_dir))
        problem = ""
        if not results.get("model_name"):
            problem = ("records no model_name, so its architecture is unknown — "
                       "re-import it with scripts/import_propestnet_run.py")
        elif num_classes is not None and classes and len(classes) != num_classes:
            problem = f"trained on {len(classes)} classes, this app serves {num_classes}"

        described.append({
            "label": label,
            "path": checkpoint,
            "score": score,
            "model": results.get("model") or results.get("model_name") or "unknown",
            "num_classes": len(classes) or None,
            "under_trained": bool(results.get("under_trained")),
            "usable": not problem,
            "problem": problem,
        })
    return described


def load_best_model(
    *, num_classes: int | None = None, device: str = "cpu", model_path: str | Path | None = None
) -> LoadedModel:
    """Load the highest-scoring usable run in ``runs/``.

    ``num_classes``, when given, rejects runs trained on a different class list.
    A ten-class checkpoint loaded under fifteen class names would answer every
    photo with a confidently wrong name.
    """
    if torch is None:
        return LoadedModel(reason="PyTorch is not installed in the environment running this app.")

    if model_path:
        path = Path(model_path)
        if not path.exists():
            return LoadedModel(reason=f"No checkpoint at {path}.")
        results = {}
        results_path = path.parent / "results.json"
        if results_path.is_file():
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                results = {}
        candidates = [(_score(results), path, results)]
    else:
        candidates = find_runs()

    if not candidates:
        return LoadedModel(
            reason="No run in runs/ has both a best_model.pt and a results.json. "
                   "Import one with scripts/import_propestnet_run.py."
        )

    skipped: list[str] = []
    for _, checkpoint_path, results in candidates:
        loaded = _try_load(checkpoint_path, results, num_classes, device)
        if loaded.model is not None:
            return loaded
        skipped.append(f"{checkpoint_path.parent.name} ({loaded.reason})")

    return LoadedModel(reason="No usable checkpoint in runs/ — skipped " + "; ".join(skipped))


def _try_load(
    checkpoint_path: Path, results: dict, num_classes: int | None, device: str
) -> LoadedModel:
    from ip102_bench.models import build_model

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as error:  # noqa: BLE001 - one broken file must not take the app down
        return LoadedModel(reason=f"unreadable: {error}")

    if not isinstance(checkpoint, dict):
        return LoadedModel(reason="not a checkpoint dictionary")

    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    model_name = checkpoint.get("model_name") or results.get("model_name")
    if not model_name:
        return LoadedModel(
            reason="records no model_name, so its architecture is unknown — "
                   "re-import it with scripts/import_propestnet_run.py"
        )
    model_name = {"alexnet_cnn": "alexnet", "vgg16_cnn": "vgg16", "vgg19_cnn": "vgg19"}.get(
        str(model_name), str(model_name)
    )

    class_names = checkpoint.get("class_names") or results.get("classes") or []
    checkpoint_classes = checkpoint.get("num_classes") or len(class_names) or None
    if not checkpoint_classes:
        return LoadedModel(reason="records no class list")
    if num_classes is not None and checkpoint_classes != num_classes:
        return LoadedModel(
            reason=f"trained on {checkpoint_classes} classes, app expects {num_classes}"
        )

    try:
        model_kwargs = checkpoint.get("model_kwargs") or results.get("model_kwargs") or {}
        model = build_model(model_name, num_classes=checkpoint_classes, **model_kwargs)
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, NotImplementedError, RuntimeError) as error:
        return LoadedModel(reason=f"weights do not fit {model_name}: {type(error).__name__}")

    display_names = checkpoint.get("display_names") or [
        name.replace("_", " ").capitalize() for name in class_names
    ]
    return LoadedModel(
        model=model.to(device).eval(),
        path=checkpoint_path,
        is_trained=True,
        under_trained=bool(results.get("under_trained")),
        reason=str(results.get("under_trained_note", "")),
        class_names=list(class_names),
        display_names=list(display_names),
        image_size=int(checkpoint.get("image_size", results.get("image_size", DEFAULT_IMAGE_SIZE))),
        mean=list(checkpoint.get("mean", DEFAULT_MEAN)),
        std=list(checkpoint.get("std", DEFAULT_STD)),
        inference_views=list(
            checkpoint.get("inference_views", results.get("inference_views", ["centre", "whole"]))
        ),
        tta_flip=bool(checkpoint.get("tta_flip", results.get("tta_flip", True))),
        use_box_crop=bool(checkpoint.get("use_box_crop", results.get("use_box_crop", True))),
        protocol_warning=(
            str(results.get("benchmark_note", ""))
            if results.get("benchmark_compatible") is False else ""
        ),
        results=results,
    )
