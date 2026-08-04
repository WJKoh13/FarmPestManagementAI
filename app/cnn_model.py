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
    # Section 13 of the notebook selects `tau` on validation and records the
    # training class prior next to it. Both absent means no correction, which is
    # what every checkpoint written before that section existed gets.
    logit_adjust_tau: float = 0.0
    train_class_prior: list[float] = field(default_factory=list)
    results: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def adjusts_for_prior(self) -> bool:
        """Whether this checkpoint carries a usable prior correction."""
        return bool(self.train_class_prior) and self.logit_adjust_tau != 0.0


def _score(results: dict) -> float:
    """Rank a run by the best macro F1 it can evidence.

    Runs write this in several shapes: nested under ``test``, under
    ``test_with_tta``, under ``test_with_tta_and_prior``, or flat at the top
    level. Read all of them rather than silently scoring every run -1. Most
    corrected first, so a run is ranked by the setting the app will actually
    serve it under.
    """
    for section in ("test_with_tta_and_prior", "test_with_tta", "test"):
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


def _test_accuracy(results: dict) -> float | None:
    """The test accuracy of the setting the app actually serves, or None.

    Same order as :func:`_score`, so the number shown next to a run is the one
    that run is ranked by -- a picker that ranked on the corrected score and
    displayed the uncorrected one would be quietly lying.

    Returns None rather than falling back to a validation figure: a run that was
    never scored on test has no accuracy, and presenting a macro-F1 as an
    accuracy would be a different number under the same name.
    """
    for section in ("test_with_tta_and_prior", "test_with_tta", "test"):
        block = results.get(section)
        if isinstance(block, dict) and block.get("accuracy") is not None:
            return float(block["accuracy"])
    return None


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
        # The class count is checked first because it is the one problem that
        # re-importing cannot fix: a model trained on a different set of pests
        # can never answer for this one, however well its metadata is recorded.
        # Reporting the metadata gap first would send someone to run an import
        # that leaves the run exactly as unusable as it was.
        problem = ""
        if num_classes is not None and classes and len(classes) != num_classes:
            problem = (f"trained on {len(classes)} pests, this app serves {num_classes} — "
                       "it belongs to a different dataset, so it cannot be used here")
        elif not results.get("model_name"):
            problem = ("does not record which architecture it is — "
                       "re-import it with scripts/import_propestnet_run.py")

        model = results.get("model") or results.get("model_name") or "unknown"
        under_trained = bool(results.get("under_trained"))
        accuracy = _test_accuracy(results)

        # What a picker shows. `label` stays the folder path -- it is what an
        # error message has to name so the folder can be found -- but nobody
        # using this app should have to read a datestamped directory to choose
        # a model.
        if under_trained:
            display_label = f"{model} — early test build, not for advice"
        elif accuracy is None:
            display_label = f"{model} — not scored on the test set"
        else:
            display_label = f"{model} — {accuracy:.0%} accurate"

        described.append({
            "label": label,
            "display_label": display_label,
            "path": checkpoint,
            "score": score,
            "accuracy": accuracy,
            "model": model,
            "num_classes": len(classes) or None,
            "under_trained": under_trained,
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
        model = build_model(model_name, num_classes=checkpoint_classes)
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, NotImplementedError, RuntimeError) as error:
        return LoadedModel(reason=f"weights do not fit {model_name}: {type(error).__name__}")

    display_names = checkpoint.get("display_names") or [
        name.replace("_", " ").capitalize() for name in class_names
    ]
    # Written by section 13 of the notebook. Read from the checkpoint first and
    # results.json second, the same order every other setting here uses.
    adjustment = results.get("logit_adjustment") or {}
    prior = checkpoint.get("train_class_prior") or adjustment.get("train_class_prior") or []
    tau = checkpoint.get("logit_adjust_tau", adjustment.get("tau", 0.0))
    if len(prior) != checkpoint_classes:
        # A prior of the wrong length would silently reweight the wrong classes.
        prior, tau = [], 0.0
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
        logit_adjust_tau=float(tau),
        train_class_prior=[float(value) for value in prior],
        results=results,
    )
