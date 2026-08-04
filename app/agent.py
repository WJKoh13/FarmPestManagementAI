"""The loop that lets the language model drive.

This is the app's primary path. `PestAssistant.prepare_turn` is still here and
still works, but it decides in Python whether to run the classifier; this module
does not decide anything of the sort. It hands the model a set of tools and a
message that *mentions* a photo, and the model asks for the classifier itself.

A turn has two phases, and they are separate for a concrete reason:

*The tool phase* is not streamed. Ollama only fills ``message.tool_calls`` on a
non-streaming response, so streaming it would mean parsing tool calls out of a
half-arrived body.

*The answer phase* is streamed, with no tools offered. `plan()` stops once the
tools have run and hands back a transcript; the caller streams the reply from
it. That keeps `stream_chat` and the whole UI exactly as they were.

Two safety nets sit around the model's judgement, and both are visible in the
trace rather than hidden:

*The photo re-prompt* -- a photo arrived and the model did not classify it, so it
is told to, once. Small models skip this more often than is comfortable.

*Forced grounding* -- the model is about to answer without a treatment guide, so
one is fetched for it and marked ``auto``. The model still chooses whether and
when to classify, which guide it wants, and whether to consult the knowledge
base at all; what it cannot do is give treatment advice from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent_tools import ToolBox, ToolResult, TOOL_SCHEMAS
from app.conversation import Conversation
from app.ollama_client import MAX_TOOL_ROUNDS, ToolCall
from app.pest_assistant import PestAssistant


@dataclass
class AgentTurn:
    """Everything one agent turn produced, short of the streamed answer itself."""

    # The transcript to stream the answer from, tool results included.
    messages: list[dict[str, Any]] = field(default_factory=list)
    # [{"name": ..., "auto": bool}] -- what the model asked for, in order. This is
    # the evidence that the classifier was invoked by the model and not by an if.
    trace: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[tuple[str, float]] = field(default_factory=list)
    view: dict[str, Any] = field(default_factory=dict)
    classified: bool = False
    grounded: bool = False
    # False means the caller should fall back to prepare_turn entirely.
    ok: bool = True

    @property
    def tools_used(self) -> list[str]:
        return [step["name"] for step in self.trace]


_KNOWLEDGE_CACHE: Any = None


def default_knowledge() -> Any:
    """The reference library, loaded once per process.

    Cached at module level because every `PestAgent` wants the same index and
    Streamlit rebuilds its objects on each rerun -- re-reading a few hundred
    kilobytes of vectors on every keystroke is a cost with no benefit.
    """
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE is None:
        from app.knowledge import load_index

        _KNOWLEDGE_CACHE = load_index()
    return _KNOWLEDGE_CACHE


@dataclass
class PestAgent:
    """Runs the tool phase of a turn. Owns no state between turns."""

    assistant: PestAssistant
    knowledge: Any = None
    max_rounds: int = MAX_TOOL_ROUNDS

    def __post_init__(self) -> None:
        # Loaded here rather than defaulted in the signature so a caller can pass
        # an explicit index (or an empty one, in tests) and be obeyed.
        if self.knowledge is None:
            self.knowledge = default_knowledge()

    # ---------------------------------------------------------------- helpers
    def allowed_images(self, image_path: str | None,
                       conversation: Conversation | None) -> frozenset[str]:
        """Every photo the model may read this turn, resolved to absolute paths.

        The current upload, whatever the last identification was about, and every
        photo earlier in this conversation -- so "what about the one I sent
        before?" works, while photos from *other* conversations stay unreachable.
        """
        paths: set[str] = set()

        def add(raw: str | None) -> None:
            if not raw:
                return
            try:
                paths.add(str(Path(raw).expanduser().resolve()))
            except (OSError, RuntimeError, ValueError):
                return

        add(image_path)
        if conversation is not None:
            if conversation.pest:
                add(conversation.pest.image_path)
            for message in conversation.messages:
                add(message.image_path)
        return frozenset(paths)

    def opening_instruction(self, user_message: str, image_path: str | None) -> str:
        """The user turn the model actually answers.

        When a photo is attached this *names the file and stops*. That single
        sentence is what replaces the `if image_path is not None:` branch: the
        model is told a photo exists and where it is, and has to decide for
        itself to call the classifier. Nothing here identifies anything.
        """
        if not image_path:
            return user_message
        note = user_message.strip() or "no note given"
        return (
            f"The farmer has attached a photograph, saved at {image_path}. "
            f"Their note: {note}."
        )

    def _merge(self, turn: AgentTurn, result: ToolResult) -> None:
        """Fold one tool result into the turn's UI-facing state."""
        turn.trace.append({"name": result.name, "auto": result.auto})
        if result.candidates:
            turn.candidates = result.candidates
        if result.view:
            turn.view = result.view
        if result.name == "classify_pest_image" and not result.payload.get("error"):
            turn.classified = True
        if result.name == "lookup_treatment_guide":
            turn.grounded = True

    # ------------------------------------------------------------------- plan
    def plan(self, user_message: str, image_path: str | None = None,
             conversation: Conversation | None = None) -> AgentTurn:
        """Run the tool phase and return the transcript to stream the answer from."""
        llm = self.assistant.llm
        tools = ToolBox(
            assistant=self.assistant,
            conversation=conversation,
            knowledge=self.knowledge,
            allowed_images=self.allowed_images(image_path, conversation),
        )

        system = self.assistant.agent_system_prompt(conversation)
        instruction = self.opening_instruction(user_message, image_path)
        if conversation is not None:
            messages = conversation.to_llm_messages(system, latest=instruction)
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": instruction}]

        turn = AgentTurn(messages=messages)
        offer_tools = True

        for _ in range(self.max_rounds):
            reply = llm.chat_with_tools(turn.messages, TOOL_SCHEMAS if offer_tools else [])

            if reply.unsupported and offer_tools:
                # Not an outage: this model simply cannot call tools. Retry once
                # without them, and let forced grounding supply the guide it can
                # no longer fetch for itself.
                offer_tools = False
                continue

            if not reply.ok:
                turn.ok = False
                return turn

            if not reply.tool_calls:
                break

            turn.messages.append(reply.raw_message or
                                 {"role": "assistant", "content": reply.text})
            for call in reply.tool_calls:
                result = tools.run(call)
                turn.messages.append(result.as_message())
                self._merge(turn, result)

        # A photo arrived and the model talked around it. Say so once, plainly,
        # and give it one more round -- on a 3B this is the difference between a
        # reliable demo and a coin flip.
        if image_path and not turn.classified and offer_tools:
            turn.messages.append({
                "role": "user",
                "content": "You have not looked at the photograph yet. Call "
                           "classify_pest_image on it before you answer.",
            })
            reply = llm.chat_with_tools(turn.messages, TOOL_SCHEMAS)
            if reply.ok and reply.tool_calls:
                turn.messages.append(reply.raw_message or
                                     {"role": "assistant", "content": reply.text})
                for call in reply.tool_calls:
                    result = tools.run(call)
                    turn.messages.append(result.as_message())
                    self._merge(turn, result)

        # About to answer with no vetted guide in the transcript. Fetch one and
        # mark it: the model still chose whether to classify and what to look up,
        # but it may not invent a treatment from nothing.
        if not turn.grounded:
            slug = ""
            if conversation is not None and conversation.pest and not conversation.pest.uncertain:
                slug = conversation.pest.slug
            result = tools.lookup_treatment_guide(
                slug, user_message, fresh_identification=turn.classified)
            result.auto = True
            turn.messages.append(result.as_message())
            self._merge(turn, result)

        return turn
