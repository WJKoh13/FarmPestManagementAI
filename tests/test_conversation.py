"""The chat's context: history, the pest in hand, grounding, and persistence.

None of this needs a checkpoint or the IP102 images, so the whole file runs on a
bare clone.
"""

from __future__ import annotations

import json

import pytest

from app.conversation import (
    NEW_CHAT_TITLE,
    Conversation,
    Message,
    PestContext,
    load_all,
    prune_orphan_images,
    store_image,
)
from app.ollama_client import DEFAULT_MODEL, LLMReply, OllamaClient
from app.retrieval import guidance_block, relevant_guides, score_guides
from app.treatment_guides import TREATMENT_GUIDES


@pytest.fixture
def chats_dir(tmp_path):
    return tmp_path / "chats"


def aphid_context() -> PestContext:
    return PestContext(slug="aphids", display_name="Aphids", confidence=0.81)


# --------------------------------------------------------------------- history
def test_history_is_sent_to_the_model_in_order():
    chat = Conversation()
    chat.add("user", "what is eating my beans?")
    chat.add("assistant", "Check the undersides of the leaves.")

    messages = chat.to_llm_messages("RULES", latest="how often do I look?")

    assert messages[0] == {"role": "system", "content": "RULES"}
    assert [message["role"] for message in messages[1:]] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == "how often do I look?"


def test_history_is_trimmed_to_the_recent_turns():
    chat = Conversation()
    for index in range(20):
        chat.add("user", f"question {index}")
        chat.add("assistant", f"answer {index}")

    messages = chat.to_llm_messages("RULES", history_turns=3)

    assert len(messages) == 1 + 3 * 2
    assert messages[-1]["content"] == "answer 19"


def test_photo_turns_are_described_not_dropped():
    """The model is text-only, but it must know a photo was sent."""
    chat = Conversation()
    chat.add("user", "is this bad?", image_path="/tmp/x.jpg", image_name="x.jpg")

    content = chat.to_llm_messages("RULES")[1]["content"]

    assert "sent a photo" in content
    assert "is this bad?" in content


def test_the_current_question_is_not_duplicated():
    chat = Conversation()
    chat.add("user", "is neem safe?")

    messages = chat.to_llm_messages("RULES", latest="is neem safe?")

    assert [message["content"] for message in messages].count("is neem safe?") == 1


# ----------------------------------------------------------------- pest memory
def test_pest_context_reads_as_the_pest_in_hand():
    summary = aphid_context().summary()
    assert "Aphids" in summary
    assert "'it'" in summary


def test_an_uncertain_identification_is_not_stated_as_fact():
    context = PestContext("aphids", "Aphids", 0.19, uncertain=True)
    assert "not state the name as certain" in context.summary()


# ------------------------------------------------------------------ retrieval
def test_a_named_pest_wins_retrieval():
    guides = relevant_guides("my beans are covered in aphids", {"aphids": "Aphids"})
    assert guides and guides[0][0] == "Aphids"
    assert guides[0][1] == TREATMENT_GUIDES["aphids"]


def test_the_display_name_is_matched_too():
    guides = relevant_guides("what do I do about a black cutworm?",
                             {"black_cutworm": "Black cutworm"})
    assert guides and guides[0][1] == TREATMENT_GUIDES["black_cutworm"]


def test_the_pest_in_hand_wins_a_question_that_names_nothing():
    """'is it safe for bees?' has to resolve against the last identification."""
    guides = relevant_guides("is it safe for bees?", {"aphids": "Aphids"},
                             context_slug="aphids")
    assert guides and guides[0][1] == TREATMENT_GUIDES["aphids"]


def test_a_general_question_retrieves_nothing_specific():
    """Attaching an arbitrary pest guide to a general question would mislead."""
    assert relevant_guides("hello", {}) == []
    block = guidance_block("hello", {})
    assert "General organic pest guidance" in block
    assert "Background" not in block


def test_a_partial_match_is_background_but_never_a_diagnosis():
    """'is neem oil safe for bees?' is answered in the aphid guide, not about aphids."""
    names = {"aphids": "Aphids"}
    question = "is neem oil safe for bees?"

    assert relevant_guides(question, names) == []       # nothing to name the farmer
    block = guidance_block(question, names)
    assert "bees" in block                              # but the fact is given to the model
    assert "do not tell the farmer they have Aphids" in block


def test_every_guide_is_scored():
    assert len(score_guides("aphids", {})) == len(TREATMENT_GUIDES)


def test_guidance_block_carries_the_guide_text():
    block = guidance_block("aphids on my kale", {"aphids": "Aphids"})
    assert "insecticidal soap" in block


# ---------------------------------------------------------------- persistence
def test_a_conversation_survives_a_round_trip(chats_dir):
    chat = Conversation(title="Aphids on kale")
    chat.add("user", "what is this?", image_path="/photos/a.jpg", image_name="a.jpg")
    chat.add("assistant", "Aphids.", heading="**Possible pest: Aphids**",
             candidates=[("aphids", 0.81), ("miridae", 0.07)])
    chat.pest = aphid_context()
    chat.save(chats_dir)

    (restored,) = load_all(chats_dir)

    assert restored.id == chat.id
    assert restored.title == "Aphids on kale"
    assert restored.pest.slug == "aphids"
    assert restored.messages[0].image_name == "a.jpg"
    assert restored.messages[1].heading == "**Possible pest: Aphids**"
    assert restored.messages[1].candidates == [("aphids", 0.81), ("miridae", 0.07)]


def test_a_corrupt_chat_file_does_not_hide_the_others(chats_dir):
    Conversation(title="good").save(chats_dir)
    (chats_dir / "broken.json").write_text("{not json", encoding="utf-8")

    chats = load_all(chats_dir)

    assert [chat.title for chat in chats] == ["good"]


def test_chats_are_listed_most_recent_first(chats_dir):
    older = Conversation(title="older", updated_at=100.0)
    newer = Conversation(title="newer", updated_at=200.0)
    older.save(chats_dir)
    newer.save(chats_dir)

    assert [chat.title for chat in load_all(chats_dir)] == ["newer", "older"]


def test_saving_leaves_no_partial_file(chats_dir):
    Conversation(title="x").save(chats_dir)
    assert not list(chats_dir.glob("*.tmp"))
    assert json.loads(next(chats_dir.glob("*.json")).read_text())["title"] == "x"


def test_a_stored_photo_outlives_the_upload(tmp_path):
    path = store_image(b"not really a jpeg", "pest.JPG", tmp_path)
    assert path.exists() and path.suffix == ".jpg"
    assert path.read_bytes() == b"not really a jpeg"


def test_orphan_photos_are_pruned_and_referenced_ones_kept(tmp_path):
    chats_dir, images_dir = tmp_path / "chats", tmp_path / "images"
    kept = store_image(b"a", "kept.jpg", images_dir)
    orphan = store_image(b"b", "orphan.jpg", images_dir)
    chat = Conversation()
    chat.add("user", "this one", image_path=str(kept))
    chat.save(chats_dir)

    assert prune_orphan_images(chats_dir, images_dir) == 1
    assert kept.exists() and not orphan.exists()


# --------------------------------------------------------------------- titling
def test_an_untouched_chat_is_named_after_its_first_question():
    chat = Conversation()
    assert chat.title == NEW_CHAT_TITLE
    chat.retitle_from("  what is  eating\nmy kale?  ")
    assert chat.title == "what is eating my kale?"

    chat.retitle_from("something else entirely")
    assert chat.title == "what is eating my kale?"


def test_a_chat_is_empty_until_the_farmer_speaks():
    chat = Conversation(messages=[Message("assistant", "Hello!")])
    assert chat.is_empty
    chat.add("user", "hi")
    assert not chat.is_empty


# ------------------------------------------------------------------ llm client
def test_the_model_environment_variable_is_honoured(monkeypatch):
    """This was a real bug: the parameter default shadowed the variable."""
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    assert OllamaClient().model == "llama3.2:3b"

    monkeypatch.delenv("OLLAMA_MODEL")
    assert OllamaClient().model == DEFAULT_MODEL
    assert OllamaClient(model="qwen2.5:3b").model == "qwen2.5:3b"


def test_a_dead_server_reports_not_ready_rather_than_raising():
    client = OllamaClient(base_url="http://127.0.0.1:9")
    assert client.available() is False
    assert "not running" in client.status_line
    assert client.chat([{"role": "user", "content": "hi"}]).ok is False
    # An empty stream is how the UI learns to fall back.
    assert list(client.stream_chat([{"role": "user", "content": "hi"}])) == []


def test_an_llm_reply_is_falsy_when_it_did_not_come_from_the_model():
    assert LLMReply("hello", ok=True)
    assert not LLMReply("hello", ok=False)
