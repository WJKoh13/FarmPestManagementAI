from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def best_checkpoint(runs_dir: Path = PROJECT_ROOT / "runs") -> Path | None:
    """Return the checkpoint from the run with the highest recorded macro F1."""
    candidates: list[tuple[float, Path]] = []
    for results_path in runs_dir.glob("*/*/results.json"):
        checkpoint = results_path.parent / "best_model.pt"
        if not checkpoint.exists():
            continue
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            score = float(results.get("macro_f1", results.get("test_macro_f1", -1)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((score, checkpoint))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def load_best_model(
    *, num_classes: int, device: str = "cpu", model_path: str | Path | None = None
):
    """Load the best checkpoint, or an explicitly labelled untrained demo CNN."""
    if torch is None:
        return None, None, False

    checkpoint_path = Path(model_path) if model_path else best_checkpoint()
    if checkpoint_path is None or not checkpoint_path.exists():
        from ip102_bench.models import build_model

        return build_model("alexnet", num_classes=num_classes).to(device).eval(), None, False

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    model_name = checkpoint.get("model_name")

    # Saved benchmark artifacts include model_name. The current repository has a
    # completed AlexNet-style scratch model; other completed registry models work too.
    if not model_name:
        return None, checkpoint_path, False
    model_name = {"alexnet_cnn": "alexnet", "vgg16_cnn": "vgg16", "vgg19_cnn": "vgg19"}.get(
        str(model_name), str(model_name)
    )
    try:
        from ip102_bench.models import build_model

        model = build_model(model_name, num_classes=num_classes)
        model.load_state_dict(state_dict)
    except (KeyError, NotImplementedError, RuntimeError):
        return None, checkpoint_path, False

    return model.to(device).eval(), checkpoint_path, True
