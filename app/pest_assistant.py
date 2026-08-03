from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight environments
    torch = None
    F = None

from PIL import Image

from app.cnn_model import load_best_model
from app.ollama_client import OllamaClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES_PATH = PROJECT_ROOT / "data_manifests" / "classes.json"


@dataclass
class DetectionResult:
    pest_name: str
    confidence: float
    response: str


def load_class_names() -> list[str]:
    data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return [entry["class_name"] for entry in data["classes"]]


def load_eval_transform():
    stats = json.loads((PROJECT_ROOT / "data_manifests" / "norm_stats.json").read_text(encoding="utf-8"))
    from ip102_bench.transforms import build_eval_transform

    return build_eval_transform(stats["image_size"], stats["mean"], stats["std"])


TREATMENT_GUIDES = {
    "aphids": """**Do today**
- Check the undersides of young leaves and new shoots.
- Spray small groups off with a firm stream of clean water.
- Remove only leaves that are badly covered.

**Organic treatment**
- If numbers keep rising, use insecticidal soap or neem oil labelled for your crop. Cover leaf undersides.
- Avoid spraying when bees or helpful insects are active.""",
    "corn_borer": """**Do today**
- Check 10 plants in different parts of the field for small holes and sawdust-like droppings near stems, tassels, or cobs.
- Remove badly damaged shoots or cobs and take them away from the field.

**Organic treatment**
- For small caterpillars, apply a Bt product (Bacillus thuringiensis) approved for your crop. Aim into the leaf whorl or feeding area, preferably late afternoon.
- Treat early; it is much less effective after caterpillars enter the stem.""",
    "grub": """**Do today**
- Gently inspect soil around wilting plants and hand-pick grubs you find.
- Remove dead roots and crop waste that can shelter pests.

**Organic treatment**
- Ask a local organic supplier about beneficial nematodes suitable for your pest and soil temperature. Water the soil as directed after applying them.""",
    "wireworm": """**Do today**
- Dig near weak plants and look for thin, hard, yellow-brown larvae in the soil.
- Use cut potato pieces as bait traps; lift and destroy larvae every few days.

**Organic prevention**
- Keep weeds and grassy cover crops under control before planting, and rotate crops where possible.""",
}


def treatment_guide(pest_name: str) -> str:
    return TREATMENT_GUIDES.get(
        pest_name,
        """**Do today**
- Check several plants, including the undersides of leaves, and remove pests by hand where practical.
- Remove badly damaged plant parts and keep them away from the field.

**Organic treatment**
- Start with the least harmful option: water spray, hand removal, barriers, or traps.
- If the problem is spreading, choose a product approved by your local organic-certification body and follow its label exactly.

**Keep watch**
- Recheck the same plants in 2 to 3 days. Take another photo if damage increases.""",
    )


def is_fallback_reply(reply: str) -> bool:
    return reply.startswith("Local language model is unavailable.") or "offline fallback mode" in reply


class PestAssistant:
    """Offline pest classifier plus organic-treatment conversational assistant."""

    def __init__(self, device: str | None = None, model_path: str | Path | None = None) -> None:
        if torch is not None:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = "cpu"
        self.class_names = load_class_names()
        self.model, self.model_path, self.model_is_trained = load_best_model(
            num_classes=len(self.class_names), device=str(self.device), model_path=model_path
        )
        self.transform = load_eval_transform() if self.model is not None else None
        self.llm = OllamaClient()

    def _predict_label(self, image_path: Path) -> tuple[str | None, float | None]:
        if self.model is None or self.transform is None:
            return None, None
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = F.softmax(self.model(tensor), dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        return self.class_names[index.item()], confidence.item()

    def analyze_image(self, image_path: str | os.PathLike[str], farmer_note: str = "") -> dict[str, Any]:
        image_path = Path(image_path)
        pest_name, confidence = self._predict_label(image_path)
        if self.model is None:
            return {
                "pest_name": None,
                "confidence": None,
                "response": (
                    "I received the photo, but the CNN is unavailable because PyTorch is not installed in "
                    "the Python environment running this app. Install the project requirements, then restart."
                ),
            }

        display_name = pest_name.replace("_", " ").title()
        demo_notice = ""
        if not self.model_is_trained:
            demo_notice = (
                "**Testing mode:** This is a sample prediction from an untrained model. Use it to test the app only.\n\n"
            )
        base_response = demo_notice + f"**Possible pest: {display_name}**\n\n{treatment_guide(pest_name)}"
        prompt = (
            "You are a practical organic-farming adviser writing for a farmer with no technical background. "
            f"The possible pest is {display_name}. The farmer's note is: {farmer_note or 'No note provided'}. "
            "Use the headings: Do today, Organic treatment, Keep watch. Give short, concrete steps. "
            "Mention safe timing if relevant and say to follow the product label and local organic rules. "
            "Do not mention AI, model confidence, or technical terms."
        )
        llm_reply = self.llm.generate(prompt)
        response = base_response if is_fallback_reply(llm_reply) else f"**Possible pest: {display_name}**\n\n{llm_reply}"
        return {
            "pest_name": pest_name,
            "confidence": round(confidence, 2),
            "response": response,
        }

    def chat_reply(self, user_message: str, image_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        if image_path:
            detection = self.analyze_image(image_path, farmer_note=user_message)
            return {
                "message": detection["response"],
                "pest_name": detection["pest_name"],
                "confidence": detection["confidence"],
            }

        if "treatment" in user_message.lower() or "pest" in user_message.lower():
            prompt = (
                "You are an organic-farming assistant. Respond with one short paragraph about how to "
                "identify pests and choose organic treatment options safely."
            )
            llm_reply = self.llm.generate(prompt)
            return {
                "message": llm_reply,
                "pest_name": None,
                "confidence": None,
            }

        return {
            "message": "Please upload a photo of the pest so I can identify it and recommend organic treatment options.",
            "pest_name": None,
            "confidence": None,
        }
