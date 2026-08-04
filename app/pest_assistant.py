from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight environments
    torch = None

from app.cnn_model import load_best_model
from app.ollama_client import OllamaClient
from app.treatment_guides import treatment_guide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES_PATH = PROJECT_ROOT / "data_manifests" / "classes_top15.json"

# Below this top-1 probability the assistant offers candidates instead of an
# identification. The model is 69.2% top-1 against 86.5% top-3 even when fully
# trained, so naming one pest confidently is the wrong shape of answer near the
# margin -- and a farmer acting on a wrong name wastes a treatment.
CONFIDENCE_FLOOR = 0.35


def load_class_metadata(path: Path = CLASSES_PATH) -> tuple[list[str], list[str]]:
    """(slugs, display names) in project-label order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["classes"]
    slugs = [entry["class_name"] for entry in entries]
    display = [
        entry.get("display_name") or entry["class_name"].replace("_", " ").capitalize()
        for entry in entries
    ]
    return slugs, display


def is_fallback_reply(reply: str) -> bool:
    return reply.startswith("Local language model is unavailable.") or "offline fallback mode" in reply


def format_candidates(candidates: list[tuple[str, float]], display_names: dict[str, str],
                      header: str = "Possible pest") -> str:
    """The top-k list as a short markdown table of names and confidences."""
    if not candidates:
        return ""
    lines = [f"| {header} | Confidence |", "|---|---:|"]
    lines += [f"| {display_names.get(name, name)} | {conf:.0%} |" for name, conf in candidates]
    return "\n".join(lines)


class PestAssistant:
    """Offline pest classifier plus organic-treatment conversational assistant."""

    def __init__(self, device: str | None = None, model_path: str | Path | None = None) -> None:
        if torch is not None:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = "cpu"

        self.class_names, display = load_class_metadata()
        self.display_names = dict(zip(self.class_names, display))

        self.loaded = load_best_model(
            num_classes=len(self.class_names), device=str(self.device), model_path=model_path
        )
        self.model = self.loaded.model

        # The checkpoint's own class list wins: it is what the weights were
        # actually trained against, and a manifest edited afterwards must not
        # silently relabel its outputs.
        if self.loaded.class_names:
            self.class_names = self.loaded.class_names
            self.display_names = dict(zip(self.loaded.class_names, self.loaded.display_names))

        self.views = None
        if self.model is not None:
            from app.propest_inference import build_views

            self.views = build_views(self.loaded.image_size, self.loaded.mean, self.loaded.std)

        self.llm = OllamaClient()

    # ------------------------------------------------------------------ status
    @property
    def model_is_trained(self) -> bool:
        return self.loaded.is_trained

    @property
    def status_message(self) -> str:
        """What to tell the user about the model serving them, or "" if all is well."""
        if self.model is None:
            return f"No pest model is loaded. {self.loaded.reason}"
        if self.loaded.under_trained:
            return self.loaded.reason or "This checkpoint is under-trained; treat results as a test only."
        return ""

    # --------------------------------------------------------------- inference
    def predict(self, image_path: str | os.PathLike[str], k: int = 3,
                box: list[float] | None = None, tta: bool = True) -> list[tuple[str, float]]:
        """Top-k (slug, confidence) for a photo. Empty when no model is loaded."""
        if self.model is None or self.views is None:
            return []
        from app.propest_inference import predict_topk

        return predict_topk(
            self.model, Path(image_path), self.class_names, self.views,
            k=k, tta=tta, device=self.device, box=box,
        )

    def analyze_image(self, image_path: str | os.PathLike[str], farmer_note: str = "") -> dict[str, Any]:
        candidates = self.predict(image_path)
        if not candidates:
            return {
                "pest_name": None,
                "confidence": None,
                "alternatives": [],
                "uncertain": True,
                "response": (
                    "I received the photo, but no pest model is loaded, so I cannot identify it. "
                    f"{self.loaded.reason}"
                ),
            }

        pest_name, confidence = candidates[0]
        display_name = self.display_names.get(pest_name, pest_name.replace("_", " ").title())
        uncertain = confidence < CONFIDENCE_FLOOR

        notice = ""
        if self.loaded.under_trained:
            notice = (
                "**Testing mode:** this checkpoint is under-trained, so the identification below "
                "is not reliable. Use it to test the app only.\n\n"
            )

        if uncertain:
            heading = (
                "**I am not certain what this is.** Here are the closest matches:\n\n"
                + format_candidates(candidates, self.display_names)
                + "\n\nA closer, well-lit photo of the insect filling most of the frame usually "
                "settles it. The general steps below are safe for any of these."
            )
            # Advice must not commit to a pest the model could not name.
            guide = treatment_guide("")
        else:
            heading = f"**Possible pest: {display_name}** ({confidence:.0%} confidence)"
            others = format_candidates(candidates[1:], self.display_names, header="Also possible")
            if others:
                heading += "\n\n" + others
            guide = treatment_guide(pest_name)

        base_response = f"{notice}{heading}\n\n{guide}"

        # The LLM is an enhancement, never the source of truth: it rephrases the
        # guide around the farmer's note, and the guide stands on its own when
        # Ollama is not running.
        prompt = (
            "You are a practical organic-farming adviser writing for a farmer with no technical "
            f"background. The likely pest is {display_name}"
            f"{' but the identification is uncertain' if uncertain else ''}. "
            f"The farmer's note is: {farmer_note or 'No note provided'}. "
            "Rewrite this guidance around their note, keeping the headings "
            "'Do today', 'Organic treatment' and 'Keep watch', and keeping every safety point:\n\n"
            f"{guide}\n\n"
            "Give short, concrete steps. Say to follow the product label and local organic rules. "
            "Do not mention AI, model confidence, or technical terms."
        )
        llm_reply = self.llm.generate(prompt)
        response = base_response if is_fallback_reply(llm_reply) else f"{notice}{heading}\n\n{llm_reply}"

        return {
            "pest_name": pest_name,
            "confidence": round(confidence, 2),
            "alternatives": [
                {"pest_name": name, "confidence": round(conf, 2)} for name, conf in candidates[1:]
            ],
            "uncertain": uncertain,
            "response": response,
        }

    def chat_reply(self, user_message: str,
                   image_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        if image_path:
            detection = self.analyze_image(image_path, farmer_note=user_message)
            return {
                "message": detection["response"],
                "pest_name": detection["pest_name"],
                "confidence": detection["confidence"],
                "alternatives": detection["alternatives"],
                "uncertain": detection["uncertain"],
            }

        if "treatment" in user_message.lower() or "pest" in user_message.lower():
            llm_reply = self.llm.generate(
                "You are an organic-farming assistant. Respond with one short paragraph about how "
                "to identify pests and choose organic treatment options safely."
            )
            if is_fallback_reply(llm_reply):
                llm_reply = (
                    "Start by looking closely: check the undersides of leaves, the growing tips and "
                    "the soil at the base of a few plants, at the time of day the damage appears. "
                    "Remove what you can by hand, and try barriers or traps before any spray. If you "
                    "do need a product, choose one approved by your local organic-certification "
                    "body, follow its label exactly, and apply it at dusk so bees are not flying. "
                    "Send me a photo and I will tell you what I think it is."
                )
            return {"message": llm_reply, "pest_name": None, "confidence": None,
                    "alternatives": [], "uncertain": False}

        return {
            "message": "Please upload a photo of the pest so I can identify it and recommend "
                       "organic treatment options.",
            "pest_name": None,
            "confidence": None,
            "alternatives": [],
            "uncertain": False,
        }
