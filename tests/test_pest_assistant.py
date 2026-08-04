"""The chatbot's end of the integration: names, guidance, and honest failure.

Tests that need the checkpoint or the IP102 images skip when they are absent,
so a bare clone still runs the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.cnn_model import find_runs, load_best_model  # noqa: E402
from app.conversation import Conversation, PestContext  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402
from app.pest_assistant import CONFIDENCE_FLOOR, PestAssistant, load_class_metadata  # noqa: E402
from app.treatment_guides import GENERIC_GUIDE, TREATMENT_GUIDES, treatment_guide  # noqa: E402


def offline_assistant() -> PestAssistant:
    """An assistant pinned to the no-language-model path.

    Pointing the client at a closed port is deliberate. These tests assert what
    the app says when Ollama is down, and reading that off the real client would
    make them pass or fail depending on whether the developer happens to have it
    running.
    """
    return PestAssistant(llm=OllamaClient(base_url="http://127.0.0.1:9"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "IP102_v1.1" / "Detection" / "VOC2007" / "JPEGImages"
SPLITS = PROJECT_ROOT / "data_manifests" / "splits_top15.json"

needs_model = pytest.mark.skipif(
    load_best_model(num_classes=15).model is None, reason="no usable checkpoint in runs/"
)
needs_images = pytest.mark.skipif(not IMAGE_DIR.is_dir(), reason="IP102 images not present")


# ------------------------------------------------------------------ class list
def test_every_class_has_a_treatment_guide():
    """A farmer must never be shown a pest name with no advice attached."""
    slugs, _ = load_class_metadata()
    missing = [slug for slug in slugs if slug not in TREATMENT_GUIDES]
    assert not missing, f"no treatment guide for: {missing}"


def test_guides_carry_the_three_headings():
    for slug, guide in TREATMENT_GUIDES.items():
        assert "**Do today**" in guide, f"{slug} has no 'Do today' section"
        assert "**Organic treatment**" in guide or "**Organic prevention**" in guide, slug


def test_unknown_pest_falls_back_to_safe_general_advice():
    assert treatment_guide("not_a_pest") == GENERIC_GUIDE


def test_class_list_matches_the_split_file():
    slugs, display = load_class_metadata()
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    labels = {label for rows in ("train", "val", "test") for _, label in splits[rows]}
    assert labels == set(range(len(slugs)))
    assert len(display) == len(slugs)


# --------------------------------------------------------------- model loading
def test_loader_rejects_a_checkpoint_with_the_wrong_class_count():
    """A 10-class checkpoint under 15 class names would mislabel every photo."""
    loaded = load_best_model(num_classes=999)
    assert loaded.model is None
    assert "classes" in loaded.reason


def test_run_discovery_finds_runs_at_both_depths():
    """runs/<run>/ and runs/<model>/<run>/ both exist in this repo."""
    runs = find_runs()
    if not runs:
        pytest.skip("no runs on this machine")
    assert all(path.name == "best_model.pt" for _, path, _ in runs)


@needs_model
def test_loaded_model_reports_its_own_preprocessing():
    loaded = load_best_model(num_classes=15)
    assert loaded.image_size == 128
    assert loaded.mean == pytest.approx([0.485, 0.456, 0.406])
    assert len(loaded.class_names) == 15


# -------------------------------------------------------------------- the chat
def test_text_only_reply_asks_for_a_photo():
    reply = offline_assistant().chat_reply("hello")
    assert reply["pest_name"] is None
    assert "photo" in reply["message"].lower()


def test_advice_question_answers_without_a_language_model():
    """Ollama is optional; with it down the reply must still be useful."""
    reply = offline_assistant().chat_reply("how do I choose an organic treatment?")
    assert len(reply["message"]) > 100
    assert "offline fallback mode" not in reply["message"]


def test_a_follow_up_resolves_against_the_pest_already_identified():
    """'how often do I spray it?' must not need the photo sent again."""
    assistant = offline_assistant()
    chat = Conversation(pest=PestContext("aphids", "Aphids", 0.81))
    chat.add("user", "what is this?", image_path="/photos/a.jpg")
    chat.add("assistant", "Aphids.")

    reply = assistant.chat_reply("how often should I spray it?", conversation=chat)

    assert reply["pest_name"] == "aphids"
    # With Ollama down the answer is the aphid guide itself, not an apology.
    assert "insecticidal soap" in reply["message"]


def test_the_pest_in_hand_reaches_the_system_prompt():
    assistant = offline_assistant()
    chat = Conversation(pest=PestContext("aphids", "Aphids", 0.81))

    prompt = assistant.system_prompt("is it safe for bees?", chat)

    assert "Aphids" in prompt
    assert "insecticidal soap" in prompt          # the guide is attached
    assert "do not contradict it" in prompt       # and it is binding


def test_a_text_turn_sends_history_and_the_rules():
    assistant = offline_assistant()
    chat = Conversation()
    chat.add("user", "my kale has aphids")
    chat.add("assistant", "Check the leaf undersides.")

    turn = assistant.prepare_turn("and now?", conversation=chat)

    assert turn["messages"][0]["role"] == "system"
    assert [message["role"] for message in turn["messages"][1:]] == ["user", "assistant", "user"]
    assert turn["messages"][-1]["content"] == "and now?"


@needs_model
@needs_images
def test_known_image_is_identified_within_the_top_three():
    """The measurement that proves preprocessing matches training.

    Uses the bounding box, as the notebook's own evaluation does, so a failure
    here means the app's transform disagrees with the one the weights were
    trained under -- not merely that the model is weak.
    """
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    boxes = json.loads((PROJECT_ROOT / "data_manifests" / "boxes_top15.json").read_text())
    assistant = PestAssistant()

    sample = splits["test"][:40]
    hits = 0
    for filename, label in sample:
        candidates = assistant.predict(IMAGE_DIR / filename, box=boxes.get(filename))
        names = [name for name, _ in candidates]
        assert len(names) == 3
        hits += assistant.class_names[label] in names

    # The imported checkpoint measures ~69% top-3; the finished model reaches
    # 86.5%. Well below half means the pipeline is broken, not under-trained.
    assert hits / len(sample) > 0.45, f"top-3 hit rate {hits}/{len(sample)} is too low to be preprocessing-correct"


@needs_model
@needs_images
def test_low_confidence_answers_with_candidates_not_a_diagnosis():
    assistant = PestAssistant()
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    filename, _ = splits["test"][0]
    result = assistant.analyze_image(IMAGE_DIR / filename)

    assert len(result["alternatives"]) == 2
    if result["confidence"] < CONFIDENCE_FLOOR:
        assert result["uncertain"]
        assert "not certain" in result["response"]
    else:
        assert "Possible pest" in result["response"]
