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
from app.conversation import Conversation, PestContext
from app.ollama_client import OllamaClient
from app.retrieval import guidance_block
from app.treatment_guides import treatment_guide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES_PATH = PROJECT_ROOT / "data_manifests" / "classes_top15.json"

# Below this top-1 probability the assistant offers candidates instead of an
# identification. The model is 69.2% top-1 against 86.5% top-3 even when fully
# trained, so naming one pest confidently is the wrong shape of answer near the
# margin -- and a farmer acting on a wrong name wastes a treatment.
CONFIDENCE_FLOOR = 0.35

# The rules the language model works under. It rephrases the vetted guidance
# around the farmer's own situation; it is never the source of the advice.
SYSTEM_RULES = """You are a practical organic-farming adviser talking to a farmer with no \
technical background. Reply in plain language, in short paragraphs or short bullets.

Rules you must follow:
- Answer only from the approved guidance below. Do not name any product, treatment or dose \
that does not appear in it.
- Always say to follow the product label and the rules of the farmer's local \
organic-certification body.
- Keep every safety warning in the guidance. Never drop one to be brief.
- If the guidance does not cover what was asked, say so plainly and suggest what to check \
in the field instead of guessing.
- Never mention AI, models, confidence scores, percentages or anything technical about how \
the pest was identified.
- Do not invent what the farmer's crop, region or season is. If it matters, ask."""

# The answer when the language model is not running and the question is not
# about a specific pest. It has to stand on its own: this is what most users see.
GENERAL_FALLBACK = (
    "Start by looking closely: check the undersides of leaves, the growing tips and "
    "the soil at the base of a few plants, at the time of day the damage appears. "
    "Remove what you can by hand, and try barriers or traps before any spray. If you "
    "do need a product, choose one approved by your local organic-certification "
    "body, follow its label exactly, and apply it at dusk so bees are not flying. "
    "Send me a photo and I will tell you what I think it is."
)


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
    """Whether a reply is the offline stand-in rather than the model's own words.

    Retained for callers holding only a string. Anything with access to the
    client should branch on ``LLMReply.ok`` instead -- matching on wording is
    what this function is, and it breaks the moment the wording changes.
    """
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

    def __init__(self, device: str | None = None, model_path: str | Path | None = None,
                 llm: OllamaClient | None = None) -> None:
        if torch is not None:
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch, "xpu") and torch.xpu.is_available():
                    device = "xpu"
                else:
                    device = "cpu"
            self.device = torch.device(device)
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

        # Injectable so tests can pin the offline path instead of depending on
        # whether the developer happens to have Ollama running.
        self.llm = llm if llm is not None else OllamaClient()

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
        if self.loaded.protocol_warning:
            return self.loaded.protocol_warning
        return ""

    def display_name_for(self, slug: str) -> str:
        return self.display_names.get(slug, slug.replace("_", " ").title())

    # --------------------------------------------------------------- inference
    def predict(self, image_path: str | os.PathLike[str], k: int = 3,
                box: list[float] | None = None, tta: bool = True) -> list[tuple[str, float]]:
        """Top-k (slug, confidence) for a photo. Empty when no model is loaded."""
        if self.model is None or self.views is None:
            return []
        from app.propest_inference import predict_topk

        return predict_topk(
            self.model, Path(image_path), self.class_names, self.views,
            k=k, tta=tta, device=self.device,
            box=box if self.loaded.use_box_crop else None,
            view_names=self.loaded.inference_views,
            tta_flip=self.loaded.tta_flip,
        )

    # ----------------------------------------------------------------- context
    def system_prompt(self, query: str, conversation: Conversation | None = None) -> str:
        """The rules, what we know about the pest in hand, and the vetted guidance."""
        pest = conversation.pest if conversation else None
        sections = [SYSTEM_RULES]
        if pest:
            sections.append(pest.summary())
        sections.append(
            "Approved guidance — answer from this and do not contradict it:\n\n"
            + guidance_block(query, self.display_names, pest.slug if pest else None)
        )
        return "\n\n".join(sections)

    def prepare_turn(self, user_message: str, image_path: str | os.PathLike[str] | None = None,
                     conversation: Conversation | None = None) -> dict[str, Any]:
        """Everything needed to render one turn, before the language model runs.

        Returns the classifier's verdict, the messages to send, and the text to
        show if the model never answers. Both the blocking path (`chat_reply`)
        and the streaming UI go through this, so they can never diverge.
        """
        candidates: list[tuple[str, float]] = []
        pest_name: str | None = None
        confidence: float | None = None
        uncertain = False
        heading = ""
        heading_plain = ""
        note = ""
        guide = ""

        if image_path is not None:
            candidates = self.predict(image_path)
            if not candidates:
                message = (
                    "I received the photo, but no pest model is loaded, so I cannot identify it. "
                    f"{self.loaded.reason}"
                )
                return {
                    "messages": [], "fallback": message, "fallback_body": message,
                    "heading": "", "heading_plain": "", "note": "", "candidates": [],
                    "pest_name": None, "confidence": None, "uncertain": True, "no_model": True,
                }

            pest_name, confidence = candidates[0]
            display_name = self.display_name_for(pest_name)
            uncertain = confidence < CONFIDENCE_FLOOR

            notice = ""
            if self.loaded.under_trained:
                notice = (
                    "**Testing mode:** this checkpoint is under-trained, so the identification below "
                    "is not reliable. Use it to test the app only.\n\n"
                )

            if uncertain:
                opening = "**I am not certain what this is.** Here are the closest matches:"
                # Sits *below* the candidate list, because it is about that list.
                note = (
                    "A closer, well-lit photo of the insect filling most of the frame usually "
                    "settles it. The general steps below are safe for any of these."
                )
                heading = (
                    f"{notice}{opening}\n\n"
                    + format_candidates(candidates, self.display_names)
                    + f"\n\n{note}"
                )
                heading_plain = f"{notice}{opening}"
                # Advice must not commit to a pest the model could not name.
                guide = treatment_guide("")
            else:
                opening = f"**Possible pest: {display_name}** ({confidence:.0%} confidence)"
                others = format_candidates(candidates[1:], self.display_names, header="Also possible")
                heading = f"{notice}{opening}" + (f"\n\n{others}" if others else "")
                heading_plain = f"{notice}{opening}"
                guide = treatment_guide(pest_name)

            # The pest in hand, for every later turn in this conversation.
            if conversation is not None:
                conversation.pest = PestContext(
                    slug=pest_name, display_name=display_name, confidence=float(confidence),
                    uncertain=uncertain, image_path=str(image_path),
                )

        query = user_message or (pest_name or "")
        system = self.system_prompt(query, conversation)

        if image_path is not None:
            instruction = (
                f"A photo was just identified as {self.display_name_for(pest_name)}"
                f"{', though not with certainty' if uncertain else ''}. "
                f"The farmer's note is: {user_message or 'No note provided'}. "
                "Rewrite the approved guidance around their note, keeping the headings "
                "'Do today', 'Organic treatment' and 'Keep watch'. Give short, concrete steps."
            )
            fallback_body = f"{note}\n\n{guide}" if note else guide
            fallback = f"{heading}\n\n{guide}" if heading else guide
        else:
            instruction = user_message
            fallback = fallback_body = self._text_fallback(user_message, conversation)

        if conversation is not None:
            messages = conversation.to_llm_messages(system, latest=instruction)
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": instruction}]

        return {
            # `fallback` is the whole answer (API, tests); `fallback_body` omits
            # the heading, for a UI that has already drawn it above the bars.
            "messages": messages, "fallback": fallback, "fallback_body": fallback_body,
            "heading": heading, "heading_plain": heading_plain, "note": note,
            "candidates": candidates, "pest_name": pest_name, "confidence": confidence,
            "uncertain": uncertain, "no_model": False,
        }

    def _text_fallback(self, user_message: str, conversation: Conversation | None) -> str:
        """A useful answer to a text-only question with the language model down.

        Falls back to the vetted guide for whichever pest the question is about,
        so "how often do I spray it?" still gets the right advice rather than an
        apology.
        """
        pest = conversation.pest if conversation else None
        from app.retrieval import relevant_guides

        guides = relevant_guides(
            user_message, self.display_names, pest.slug if pest else None, k=1
        )
        if guides:
            name, guide = guides[0]
            return f"Here is the organic guidance for **{name}**:\n\n{guide}"
        return GENERAL_FALLBACK

    # ------------------------------------------------------------------ replies
    def analyze_image(self, image_path: str | os.PathLike[str], farmer_note: str = "",
                      conversation: Conversation | None = None) -> dict[str, Any]:
        turn = self.prepare_turn(farmer_note, image_path=image_path, conversation=conversation)
        if turn["no_model"]:
            return {
                "pest_name": None, "confidence": None, "alternatives": [],
                "uncertain": True, "response": turn["fallback"],
            }

        # The LLM is an enhancement, never the source of truth: it rephrases the
        # guide around the farmer's note, and the guide stands on its own when
        # Ollama is not running.
        reply = self.llm.chat(turn["messages"])
        response = f"{turn['heading']}\n\n{reply.text}" if reply.ok else turn["fallback"]

        return {
            "pest_name": turn["pest_name"],
            "confidence": round(turn["confidence"], 2),
            "alternatives": [
                {"pest_name": name, "confidence": round(conf, 2)}
                for name, conf in turn["candidates"][1:]
            ],
            "uncertain": turn["uncertain"],
            "response": response,
        }

    def chat_reply(self, user_message: str,
                   image_path: str | os.PathLike[str] | None = None,
                   conversation: Conversation | None = None) -> dict[str, Any]:
        """One turn of conversation, with history and the pest in hand applied."""
        if image_path:
            detection = self.analyze_image(image_path, farmer_note=user_message,
                                           conversation=conversation)
            return {
                "message": detection["response"],
                "pest_name": detection["pest_name"],
                "confidence": detection["confidence"],
                "alternatives": detection["alternatives"],
                "uncertain": detection["uncertain"],
            }

        turn = self.prepare_turn(user_message, conversation=conversation)
        reply = self.llm.chat(turn["messages"])
        message = reply.text if reply.ok else turn["fallback"]
        pest = conversation.pest if conversation else None
        return {
            "message": message,
            "pest_name": pest.slug if pest else None,
            "confidence": pest.confidence if pest else None,
            "alternatives": [],
            "uncertain": bool(pest.uncertain) if pest else False,
        }
