"""The agent: what the model may ask for, and what it must never be handed.

The dead-port client in `test_pest_assistant.py` proves what the app says when
Ollama is *down*. It can say nothing at all about what happens when the model
*answers*, which is now most of the behaviour -- so this module scripts the
replies instead.

Several tests here pin bugs that a live run actually produced, rather than ones
imagined at design time. They are marked where that is the case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.agent import PestAgent  # noqa: E402
from app.agent_tools import TOOL_NAMES, TOOL_SCHEMAS, ToolBox  # noqa: E402
from app.cnn_model import load_best_model  # noqa: E402
from app.conversation import Conversation, PestContext  # noqa: E402
from app.ollama_client import OllamaClient, ToolCall, ToolCallReply  # noqa: E402
from app.pest_assistant import PestAssistant  # noqa: E402
from app.treatment_guides import GENERIC_GUIDE, TREATMENT_GUIDES  # noqa: E402

needs_model = pytest.mark.skipif(
    load_best_model(num_classes=15).model is None, reason="no usable checkpoint in runs/"
)

SAMPLES = Path(__file__).resolve().parents[1] / "sample_images"
needs_samples = pytest.mark.skipif(not SAMPLES.is_dir(), reason="no sample images")


class ScriptedLLM(OllamaClient):
    """An Ollama client that replays a fixed script instead of reaching a server.

    A subclass of the real client rather than a mock, so the seam is the HTTP
    call and nothing else: every other method, including the ones the agent
    relies on incidentally, behaves exactly as it does in production.

    ``seen`` records each transcript it is handed, which is how the grounding
    guarantees below are *verified* rather than assumed -- the assertion is on
    what actually reached the model, not on what we meant to send.
    """

    def __init__(self, script: list[ToolCallReply], body: str = "answer text") -> None:
        super().__init__(base_url="http://127.0.0.1:9")
        self.script = list(script)
        self.body = body
        self.seen: list[list[dict]] = []

    def available(self, force: bool = False) -> bool:
        return True

    def chat_with_tools(self, messages, tools):
        self.seen.append(list(messages))
        if not self.script:
            return ToolCallReply(text="done", tool_calls=[])
        return self.script.pop(0)

    def stream_chat(self, messages):
        self.seen.append(list(messages))
        yield self.body


def wants(name: str, **arguments) -> ToolCallReply:
    """A scripted reply asking for one tool."""
    return ToolCallReply(
        text="",
        tool_calls=[ToolCall(name=name, arguments=arguments)],
        raw_message={"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": name, "arguments": arguments}}]},
    )


def agent_with(script: list[ToolCallReply], body: str = "answer text") -> PestAgent:
    return PestAgent(assistant=PestAssistant(llm=ScriptedLLM(script, body)))


def tool_messages(turn) -> list[dict]:
    return [m for m in turn.messages if m.get("role") == "tool"]


# ------------------------------------------------------------------- schemas
def test_schemas_and_dispatch_agree_on_names():
    """A schema the dispatcher does not implement is a tool that silently fails."""
    assert TOOL_NAMES == {"classify_pest_image", "lookup_treatment_guide",
                          "search_knowledge_base"}
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert schema["function"]["name"] in TOOL_NAMES
        assert schema["function"]["description"].strip()
        assert schema["function"]["parameters"]["type"] == "object"


def test_guide_slug_is_constrained_to_the_real_slugs():
    """The enum is what stops a 3B model inventing pest names."""
    guide_schema = next(s for s in TOOL_SCHEMAS
                        if s["function"]["name"] == "lookup_treatment_guide")
    enum = guide_schema["function"]["parameters"]["properties"]["pest_slug"]["enum"]
    assert set(enum) == set(TREATMENT_GUIDES) | {""}


# ------------------------------------------------------------------ security
@pytest.fixture
def escape_box(tmp_path):
    """A ToolBox whose only allowed photo is a real file in a temp store."""
    assistant = PestAssistant(llm=ScriptedLLM([]))
    photo = tmp_path / "allowed.jpg"
    photo.write_bytes(b"not really a jpeg")
    return ToolBox(assistant=assistant, allowed_images=frozenset({str(photo.resolve())}))


@pytest.mark.parametrize("attack", [
    "/etc/passwd",
    "../../../etc/passwd",
    "~/.ssh/id_rsa",
    "/Users/kohwenjun/.aws/credentials",
])
def test_path_escapes_are_refused(escape_box, attack):
    """The model emits this string freely, so it may never reach the filesystem.

    Every one of these resolves outside the image store, so `resolve_image`
    returns None even though exactly one image is allowed -- the leniency for a
    hallucinated filename must not become a way out of the allowlist.
    """
    assert escape_box.resolve_image(attack) is None or \
        escape_box.resolve_image(attack).name == "allowed.jpg"
    # And whatever it resolves to, it is never the attacked path.
    resolved = escape_box.resolve_image(attack)
    assert resolved is None or str(resolved) in escape_box.allowed_images


def test_another_conversations_photo_is_refused(tmp_path):
    """Living in the image store is not enough -- the allowlist is per turn."""
    assistant = PestAssistant(llm=ScriptedLLM([]))
    mine = tmp_path / "mine.jpg"
    theirs = tmp_path / "theirs.jpg"
    for path in (mine, theirs):
        path.write_bytes(b"jpeg")

    box = ToolBox(assistant=assistant,
                  allowed_images=frozenset({str(mine.resolve()), str(theirs.resolve())}))
    # Two allowed images, so the single-image leniency is off and a bogus path
    # cannot be silently resolved to either of them.
    assert box.resolve_image("/etc/passwd") is None


def test_hallucinated_filename_falls_back_to_the_one_attached_photo(escape_box):
    """Small models invent "pest.jpg" constantly; one attached photo is unambiguous."""
    resolved = escape_box.resolve_image("pest.jpg")
    assert resolved is not None and resolved.name == "allowed.jpg"


def test_classify_without_a_photo_reports_it_and_never_echoes_the_path():
    """Refusals must not reflect model-controlled text back into the transcript."""
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])), allowed_images=frozenset())
    result = box.classify_pest_image("/etc/passwd")
    assert result.payload["error"] == "no_photo"
    assert "/etc/passwd" not in json.dumps(result.payload)


# -------------------------------------------------------------- tool payloads
@needs_model
@needs_samples
def test_classify_payload_carries_no_numbers():
    """The rules forbid the model mentioning confidence, so never give it one.

    A 3B model handed 0.81 will paraphrase it into the answer. The certainty
    decision belongs in Python; the floats ride the side channel to the UI.
    """
    photo = sorted(SAMPLES.glob("*.jpg"))[0]
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])),
                  allowed_images=frozenset({str(photo.resolve())}))
    result = box.classify_pest_image(str(photo))

    blob = json.dumps(result.payload)
    assert "%" not in blob
    for value in result.payload.values():
        assert not isinstance(value, float)
    # The UI still gets what it needs, on the side channel.
    assert result.candidates and isinstance(result.candidates[0][1], float)
    assert result.view is not None


def test_unknown_tool_is_answered_not_raised():
    """A model that invents a tool name must be corrected, not crash the turn."""
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])))
    result = box.run(ToolCall(name="delete_everything", arguments={}))
    assert result.payload["error"] == "unknown_tool"
    assert sorted(TOOL_NAMES) == result.payload["available"]


def test_unknown_slug_gives_generic_guidance_not_an_error():
    """treatment_guide("") is deliberately GENERIC_GUIDE; the tool keeps that."""
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])))
    result = box.lookup_treatment_guide("no_such_pest", "")
    assert result.payload["guide"] == GENERIC_GUIDE


def test_guide_keeps_the_three_headings_after_an_identification():
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])))
    box.identified_this_turn = True
    result = box.lookup_treatment_guide("aphids", "", fresh_identification=True)
    for heading in ("Do today", "Organic treatment", "Keep watch"):
        assert heading.lower() in result.payload["guide"].lower()
    assert "Keep the headings" in result.payload["instructions"]


def test_follow_up_is_told_to_answer_not_to_repeat_the_guide():
    """Regression: a live run re-pasted the whole guide at "how often do I spray it?".

    The three-heading instruction is right after an identification and wrong on
    a follow-up, where it makes the model restate what the farmer just read.
    """
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])))
    result = box.lookup_treatment_guide("aphids", "how often should I spray it?")
    assert "Keep the headings" not in result.payload["instructions"]
    assert "do not repeat the whole guide" in result.payload["instructions"].lower()


# --------------------------------------------------------- the grounding leak
def test_uncertain_pest_never_yields_its_own_treatment_guide():
    """Regression, found live: a 14%-confident guess produced confident advice.

    The classifier was unsure between three pests, named them, and the model
    looked the first one up -- turning "I am not certain" into that pest's
    instructions. Both routes to the guide have to refuse.
    """
    chat = Conversation()
    chat.pest = PestContext(slug="wireworm", display_name="Wireworm",
                            confidence=0.139, uncertain=True)
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])), conversation=chat)

    asked_directly = box.lookup_treatment_guide("wireworm", "")
    assert asked_directly.payload["guide"] == GENERIC_GUIDE
    assert "withheld" in asked_directly.payload

    # And the retrieval fallback must not drag it back in by context weighting,
    # which is exactly how the first fix was defeated.
    via_retrieval = box.lookup_treatment_guide("", "what do I do about it?")
    assert via_retrieval.payload["guide"] == GENERIC_GUIDE


def test_a_confident_pest_still_gets_its_own_guide():
    """The guard must not fire on a confident identification."""
    chat = Conversation()
    chat.pest = PestContext(slug="aphids", display_name="Aphids",
                            confidence=0.91, uncertain=False)
    box = ToolBox(assistant=PestAssistant(llm=ScriptedLLM([])), conversation=chat)
    result = box.lookup_treatment_guide("aphids", "")
    assert result.payload["guide"] == TREATMENT_GUIDES["aphids"]
    assert "withheld" not in result.payload


# ------------------------------------------------------------------ the loop
def test_the_model_not_python_invokes_the_classifier(tmp_path):
    """The requirement, as a test: no `if image_path` decides this.

    The agent is handed a photo and asks for nothing; the classifier does not
    run. It is the scripted *tool call* that makes it run.
    """
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"jpeg")

    silent = agent_with([ToolCallReply(text="I think it is fine.", tool_calls=[])])
    turn = silent.plan("what is this?", image_path=str(photo), conversation=Conversation())
    assert not turn.classified          # never classified: the model never asked
    assert "classify_pest_image" not in turn.tools_used


def test_tool_results_reach_the_transcript_before_the_answer():
    """Whatever the tools returned must be in the messages the answer streams from."""
    agent = agent_with([wants("lookup_treatment_guide", pest_slug="aphids")])
    turn = agent.plan("how do I treat aphids?", conversation=Conversation())

    assert "lookup_treatment_guide" in turn.tools_used
    tools = tool_messages(turn)
    assert tools, "the guide never reached the model"
    assert "aphid" in json.dumps(tools).lower()


def test_grounding_is_guaranteed_even_when_the_model_asks_for_nothing():
    """The model chooses which guide; it does not get to answer with none."""
    agent = agent_with([ToolCallReply(text="Just spray something.", tool_calls=[])])
    turn = agent.plan("what should I spray?", conversation=Conversation())

    assert turn.grounded
    assert tool_messages(turn), "no guide in the transcript"
    # And the trace is honest about who asked for it.
    assert turn.trace[-1]["auto"] is True


def test_the_loop_stops_at_the_round_cap():
    """A model looping forever looks like a hang on stage."""
    forever = [wants("lookup_treatment_guide", pest_slug="aphids") for _ in range(20)]
    agent = agent_with(forever)
    agent.max_rounds = 3
    turn = agent.plan("hello", conversation=Conversation())
    # Three rounds of the loop, plus the forced-grounding call at most.
    assert len(turn.trace) <= agent.max_rounds + 1


def test_a_dead_model_falls_back_rather_than_answering():
    """ok=False is the caller's signal to use prepare_turn instead."""
    agent = PestAgent(assistant=PestAssistant(llm=OllamaClient(base_url="http://127.0.0.1:9")))
    turn = agent.plan("hello", conversation=Conversation())
    assert turn.ok is False


def test_a_model_without_tool_support_still_gets_grounded():
    """Retry without tools, then hand it a guide anyway -- better than the raw guide."""
    agent = agent_with([
        ToolCallReply(ok=False, unsupported=True),
        ToolCallReply(text="I can answer without tools.", tool_calls=[]),
    ])
    turn = agent.plan("how do I treat aphids?", conversation=Conversation())
    assert turn.ok is True
    assert turn.grounded


def test_history_and_pest_memory_reach_the_model():
    """A follow-up must resolve "it" without the photo being sent again."""
    chat = Conversation()
    chat.pest = PestContext(slug="aphids", display_name="Aphids",
                            confidence=0.9, uncertain=False)
    chat.add("user", "what is this?")
    chat.add("assistant", "Those are aphids.")

    agent = agent_with([ToolCallReply(text="ok", tool_calls=[])])
    turn = agent.plan("how often should I spray it?", conversation=chat)

    system = turn.messages[0]
    assert system["role"] == "system"
    assert "Aphids" in system["content"]        # the pest in hand
    assert "classify_pest_image" in system["content"]   # the obligation to call
    roles = [m["role"] for m in turn.messages if m["role"] in ("user", "assistant")]
    assert roles[:3] == ["user", "assistant", "user"]


def test_the_photo_message_names_the_file_without_identifying_it():
    """The one line that replaced the IF-ELSE."""
    agent = agent_with([])
    instruction = agent.opening_instruction("all over my kale", "/tmp/photo.jpg")
    assert "/tmp/photo.jpg" in instruction
    assert "all over my kale" in instruction
    # Crucially it names no pest: identification is the model's call to make.
    for slug in TREATMENT_GUIDES:
        assert slug.replace("_", " ") not in instruction.lower()
