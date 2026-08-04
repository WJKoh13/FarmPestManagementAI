"""The three things the language model is allowed to do for itself.

This is what replaces the `if image_path is not None:` in `pest_assistant.py`.
Nothing here decides to classify a photo; the model does, by asking for
``classify_pest_image``. Python's remaining job is to make that safe: to validate
what the model asks for, to run it, and to hand back a payload the model can act
on without being told anything a farmer should not hear.

Two layers of knowledge, deliberately not merged:

*The guides* (``lookup_treatment_guide``) are vetted and safety-critical. They
are the only place a product name, a dose or an interval may come from.

*The knowledge base* (``search_knowledge_base``) is broad background -- why a
treatment works, how a pest lives. It explains; it never instructs. Collapsing
the two would let a passage about neem become the source of a dose, which is
exactly the failure the guides exist to prevent.

The schemas and their implementations live side by side because a schema that
has drifted from the function it describes fails silently: the model sends
well-formed arguments to a signature that no longer exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.conversation import IMAGES_DIR, Conversation, PestContext
from app.ollama_client import ToolCall
from app.pest_assistant import PestAssistant, identification_view
from app.retrieval import relevant_guides
from app.treatment_guides import TREATMENT_GUIDES, treatment_guide

# How many knowledge passages one search returns. Three is about the ceiling for
# a 3B model: past that the transcript grows faster than the model's ability to
# pick the relevant line out of it.
KNOWLEDGE_K = 3

# Repeated in two payloads, so it is written once. The model is told the boundary
# every time it reaches for background, not just in the system prompt where it
# would sit hundreds of tokens away from the passage it applies to.
GUIDE_AUTHORITY_NOTE = (
    "Any product, dose or interval you state must still come from "
    "lookup_treatment_guide, never from this background."
)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "classify_pest_image",
            "description": (
                "Identify the pest in a photograph the farmer has attached. You cannot see "
                "photographs yourself, so you must call this whenever a photo is mentioned. "
                "Returns the pest name, or the closest matches when it is not certain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": (
                            "The path the farmer's photo was saved to, exactly as it was given "
                            "to you in the message. Do not invent a path."
                        ),
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_treatment_guide",
            "description": (
                "Fetch the approved organic treatment guide for a pest. This is the only "
                "source of treatments, products, doses and timings you may use. Call it "
                "before giving any advice about what to apply or when."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    # An enum is worth far more here than a description. A 3B model
                    # picks reliably from a fixed list and invents freely without one,
                    # and the slug set is fixed anyway, so it costs nothing.
                    "pest_slug": {
                        "type": "string",
                        "description": (
                            "The pest to fetch the guide for. Use an empty string for general "
                            "organic guidance when no pest has been identified."
                        ),
                        "enum": sorted(TREATMENT_GUIDES) + [""],
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "What the farmer actually asked, used to pick a guide when no "
                            "pest_slug is known."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the organic-farming reference library for background: why a treatment "
                "works, how a pest lives and breeds, what damage looks like, certification and "
                "spray-safety practice. Use it to explain and to answer 'why' questions. It is "
                "background only and is never a source of products or doses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up, in the farmer's own words where possible.",
                    },
                    "pest_slug": {
                        "type": "string",
                        "description": "The pest being discussed, if one is known. Optional.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_NAMES = frozenset(schema["function"]["name"] for schema in TOOL_SCHEMAS)


@dataclass
class ToolResult:
    """What one tool call produced.

    ``payload`` is what the model sees, and it is the only field that reaches the
    transcript. ``candidates`` and ``view`` are the side channel: raw confidences
    and rendered headings that the *user interface* needs and the model must
    never be given, because the system rules forbid it repeating a number.
    """

    name: str
    payload: dict[str, Any]
    candidates: list[tuple[str, float]] = field(default_factory=list)
    view: dict[str, Any] | None = None
    # True when the app made this call itself rather than the model asking for it.
    # Surfaced in the UI trace so the demo can be honest about which is which.
    auto: bool = False

    def as_message(self) -> dict[str, str]:
        """The transcript turn Ollama expects after a tool has run."""
        return {"role": "tool", "tool_name": self.name,
                "content": json.dumps(self.payload, ensure_ascii=False)}


@dataclass
class ToolBox:
    """The tools bound to one conversation, for one turn.

    ``allowed_images`` is the security boundary and is built fresh per turn by
    the caller -- see `resolve_image`.
    """

    assistant: PestAssistant
    conversation: Conversation | None = None
    knowledge: Any = None  # KnowledgeIndex; optional so Part A stands alone
    allowed_images: frozenset[str] = frozenset()
    # Pests the classifier offered but was not confident about. Looking any of
    # them up must not produce that pest's treatment advice -- see
    # `lookup_treatment_guide`. Populated by `classify_pest_image`.
    unconfirmed: set[str] = field(default_factory=set)
    # Whether a photo was identified during *this* turn, which decides how the
    # guide should be presented. Also set by `classify_pest_image`.
    identified_this_turn: bool = False

    @property
    def unconfirmed_slugs(self) -> set[str]:
        """Pests that must not yield their own treatment guide.

        Two sources, because uncertainty has to outlive the turn it happened in.
        `unconfirmed` holds this turn's candidate list, which only exists if the
        model classified something just now; the conversation's pest carries the
        verdict forward, so a follow-up three turns later still cannot quietly
        collect the advice for a pest nobody ever confirmed.
        """
        slugs = set(self.unconfirmed)
        pest = self.conversation.pest if self.conversation else None
        if pest and pest.uncertain:
            slugs.add(pest.slug)
        return slugs

    # ------------------------------------------------------------- dispatch
    def run(self, call: ToolCall) -> ToolResult:
        """Execute one tool call. Never raises -- a bad call is a payload, not a crash.

        This is the single dispatch point on purpose: everything the model can
        reach passes through here, so validation cannot be forgotten at one call
        site, and an alternative transport (an MCP server) can replace one method
        rather than three.
        """
        arguments = call.arguments or {}
        try:
            if call.name == "classify_pest_image":
                return self.classify_pest_image(str(arguments.get("image_path") or ""))
            if call.name == "lookup_treatment_guide":
                return self.lookup_treatment_guide(
                    str(arguments.get("pest_slug") or ""),
                    str(arguments.get("question") or ""),
                    # A guide asked for in the same turn as an identification is
                    # answering "what is this and what do I do?", which is the
                    # full-structure case. Later turns are follow-up questions.
                    fresh_identification=self.identified_this_turn,
                )
            if call.name == "search_knowledge_base":
                return self.search_knowledge_base(
                    str(arguments.get("query") or ""),
                    str(arguments.get("pest_slug") or ""),
                )
        except Exception as error:  # noqa: BLE001 - one bad call must not end the turn
            return ToolResult(call.name, {"error": "tool_failed", "message": str(error)})

        # A model that invents a tool name gets told so and carries on. Raising
        # here would turn a recoverable slip into a failed turn.
        return ToolResult(call.name, {
            "error": "unknown_tool",
            "message": f"There is no tool called {call.name!r}.",
            "available": sorted(TOOL_NAMES),
        })

    # -------------------------------------------------------------- security
    def resolve_image(self, raw: str) -> Path | None:
        """The photo the model is allowed to read, or None.

        The model emits this path as free text, so the tool is a *selector over
        an allowlist*, not a file reader. Four checks, in order:

        1. ``resolve()`` -- collapses ``..`` and follows symlinks, so a symlink
           planted inside the image store points at its real target for check 3.
        2. Membership in ``allowed_images``. This is the real gate. A directory
           check alone would let the model read another conversation's photo by
           guessing a sixteen-character name.
        3. Containment in the image store, so a caller that builds the allowlist
           wrongly still cannot escape it.
        4. It exists and is a file.

        Then one deliberate leniency: if the path fails but exactly one image is
        allowed this turn, use it. Small models hallucinate "pest.jpg" and
        "/path/to/image" constantly, and this cannot widen the allowlist -- it
        can only pick the photo the farmer just uploaded. With two or more, refuse
        rather than guess which one they meant.
        """
        if self.allowed_images:
            try:
                candidate = Path(raw).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                candidate = None

            if candidate is not None and str(candidate) in self.allowed_images:
                try:
                    store = IMAGES_DIR.resolve()
                except (OSError, RuntimeError):
                    store = None
                inside = store is not None and candidate.is_relative_to(store)
                if inside and candidate.is_file():
                    return candidate

            if len(self.allowed_images) == 1:
                only = Path(next(iter(self.allowed_images)))
                if only.is_file():
                    return only
        return None

    # ------------------------------------------------------------------ CNN
    def classify_pest_image(self, image_path: str) -> ToolResult:
        """Run the CNN on an attached photo, because the model asked us to."""
        name = "classify_pest_image"

        if self.assistant.model is None:
            return ToolResult(name, {
                "error": "no_classifier",
                "message": self.assistant.loaded.reason
                           or "No pest model is loaded, so photos cannot be identified.",
            })

        path = self.resolve_image(image_path)
        if path is None:
            # Never echo the model's string back: it is untrusted text, and a
            # farmer should not be reading file paths either way.
            return ToolResult(name, {
                "error": "no_photo",
                "message": "No photograph is attached to this turn. Ask the farmer to send one.",
            })

        candidates = self.assistant.predict(path)
        if not candidates:
            return ToolResult(name, {
                "error": "unreadable_photo",
                "message": "That photograph could not be read. Ask the farmer to send another.",
            })

        view = identification_view(candidates, self.assistant.display_names,
                                   under_trained=self.assistant.loaded.under_trained)
        self.identified_this_turn = True

        # The pest in hand, for every later turn -- set from the classifier's own
        # top-1, never from anything the language model says.
        if self.conversation is not None:
            self.conversation.pest = PestContext(
                slug=view["pest_name"], display_name=view["display_name"],
                confidence=view["confidence"], uncertain=view["uncertain"],
                image_path=str(path),
            )

        others = [self.assistant.display_name_for(slug) for slug, _ in candidates[1:]]

        # Note what is not in here: a single number. The system rules forbid the
        # model mentioning confidence, and a small model handed 0.81 will
        # paraphrase it. The certainty decision is made here, in Python, against
        # CONFIDENCE_FLOOR; the raw floats ride the side channel to the UI.
        if view["uncertain"]:
            # Remember what we were unsure about. Naming these back to the model
            # is unavoidable -- it has to tell the farmer what the options are --
            # but a follow-up lookup of one of them must not return that pest's
            # treatment, or an uncertain guess becomes confident advice.
            self.unconfirmed = {slug for slug, _ in candidates}
            payload = {
                "identified": False,
                "certain": False,
                "closest_matches": [view["display_name"], *others],
                "next_step": (
                    "Do not name a pest. Call lookup_treatment_guide with an empty pest_slug "
                    "for general organic guidance, and tell the farmer that a closer, well-lit "
                    "photo of the insect filling the frame would settle it."
                ),
            }
        else:
            payload = {
                "identified": True,
                "certain": True,
                "pest_name": view["display_name"],
                "pest_slug": view["pest_name"],
                "also_possible": others,
                "next_step": (
                    f"Call lookup_treatment_guide with pest_slug {view['pest_name']!r} "
                    "before giving any advice."
                ),
            }

        return ToolResult(name, payload, candidates=candidates, view=view)

    # --------------------------------------------------------------- guides
    def lookup_treatment_guide(self, pest_slug: str = "", question: str = "",
                               fresh_identification: bool = False) -> ToolResult:
        """The vetted guide for a pest: the only sanctioned source of treatments.

        ``fresh_identification`` says a photo has just been named, which is the
        only situation where the full 'Do today / Organic treatment / Keep watch'
        structure is the right shape of answer. On a follow-up ("how often do I
        spray it?") demanding those headings makes the model re-paste the guide
        instead of answering, so it is asked for a direct answer instead.
        """
        name = "lookup_treatment_guide"
        slug = pest_slug.strip().lower()
        unconfirmed = self.unconfirmed_slugs

        # Only steer retrieval towards the pest in hand when we are sure of it.
        # relevant_guides weights a context pest at 200 against a full name match
        # at 100, so passing an unconfirmed slug here would drag its guide back
        # in through the side door even when the slug above was refused.
        pest = self.conversation.pest if self.conversation else None
        context_slug = pest.slug if (pest and not pest.uncertain) else None

        # The classifier offered this pest without confidence. Handing back its
        # treatment would launder a 14%-confident guess into firm instructions --
        # the same reason prepare_turn calls treatment_guide("") on this path.
        if slug and slug in unconfirmed:
            return ToolResult(name, {
                "source": "approved organic guide",
                "pest_name": "General organic pest management",
                "guide": treatment_guide(""),
                "withheld": (
                    f"The photo was not identified confidently, so the specific guide for "
                    f"{self.assistant.display_name_for(slug)} is not available."
                ),
                "instructions": (
                    "Give this general guidance, which is safe whichever of the candidates it "
                    "turns out to be. Do not give advice specific to any one of them, and do "
                    "not state which pest it is. " + GUIDE_AUTHORITY_NOTE
                ),
            })

        if slug in TREATMENT_GUIDES:
            display = self.assistant.display_name_for(slug)
            guide = treatment_guide(slug)
        else:
            # A model that passes "aphids on my kale" as a slug still meant aphids,
            # so fall through to the same keyword retrieval the offline path uses
            # rather than punishing it with the generic guide.
            guides = relevant_guides(question or pest_slug, self.assistant.display_names,
                                     context_slug, k=1)
            if guides:
                display, guide = guides[0]
            else:
                display, guide = "General organic pest management", treatment_guide("")

        if fresh_identification:
            # Lifted from the wording prepare_turn uses, so the three headings
            # survive into the answer without a second system message.
            instructions = (
                "Rewrite this around the farmer's own situation. Keep the headings "
                "'Do today', 'Organic treatment' and 'Keep watch'. Give short, concrete steps. "
                "Do not name any product, treatment or dose that is not in this text."
            )
        else:
            instructions = (
                "Answer the farmer's actual question using this guide. Do not repeat the "
                "whole guide back to them and do not use headings -- they have already seen "
                "it. Two or three sentences is usually right. If the guide does not answer "
                "what they asked, say so plainly. Do not name any product, treatment or dose "
                "that is not in this text."
            )

        return ToolResult(name, {
            "source": "approved organic guide",
            "pest_name": display,
            "guide": guide,
            "instructions": instructions,
        })

    # ------------------------------------------------------------ knowledge
    def search_knowledge_base(self, query: str, pest_slug: str = "") -> ToolResult:
        """Background reference. Explains why; never prescribes what."""
        name = "search_knowledge_base"
        context_slug = pest_slug.strip().lower() or (
            self.conversation.pest.slug if (self.conversation and self.conversation.pest) else "")

        if self.knowledge is None:
            return ToolResult(name, {
                "error": "no_knowledge_base",
                "message": "The reference library is not available. Answer from the treatment "
                           "guide instead, and say plainly if it does not cover the question.",
            })

        hits = self.knowledge.search(query, self.assistant.llm, k=KNOWLEDGE_K,
                                     pest_slug=context_slug)
        if not hits:
            return ToolResult(name, {
                "source": "background reference, NOT a treatment authority",
                "passages": [],
                "instructions": "Nothing in the library covers this. Say so plainly rather "
                                "than guessing, and suggest what to check in the field.",
            })

        return ToolResult(name, {
            "source": "background reference, NOT a treatment authority",
            "passages": [{"title": passage.title, "text": passage.text} for _, passage in hits],
            "instructions": (
                "Use these to explain and to answer 'why' questions. " + GUIDE_AUTHORITY_NOTE
            ),
        })
