"""The chatbot's end of the integration: names, guidance, and honest failure.

Tests that need the checkpoint or the IP102 images skip when they are absent,
so a bare clone still runs the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.cnn_model import describe_runs, find_runs, load_best_model  # noqa: E402
from app.conversation import Conversation, PestContext  # noqa: E402
from app.ollama_client import OllamaClient  # noqa: E402
from app.pest_assistant import CONFIDENCE_FLOOR, PestAssistant, load_class_metadata  # noqa: E402
from app.treatment_guides import GENERIC_GUIDE, TREATMENT_GUIDES, treatment_guide  # noqa: E402
from tests.run_bundles import write_legacy_custom_cnn_bundle, write_run_bundle  # noqa: E402


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

# The two end-to-end tests below measure a *real* trained checkpoint against
# real photos -- a synthetic bundle of random weights cannot tell you whether
# preprocessing matches training. They are therefore the only tests that read
# the repository's runs/, and they skip when it is empty. Every other test
# builds what it needs under tmp_path.
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
# These build the runs they need under tmp_path. Reading the repository's real
# runs/ made them pass or fail on what a developer happened to have imported.
def test_loader_rejects_a_checkpoint_with_the_wrong_class_count(tmp_path):
    """A 15-class checkpoint under 999 class names would mislabel every photo."""
    write_run_bundle(tmp_path, num_classes=15)

    loaded = load_best_model(num_classes=999, runs_dir=tmp_path)

    assert loaded.model is None
    assert "classes" in loaded.reason


def test_loader_reports_when_there_is_nothing_to_load(tmp_path):
    loaded = load_best_model(num_classes=15, runs_dir=tmp_path)
    assert loaded.model is None
    assert "No run in runs/" in loaded.reason


def test_run_discovery_finds_runs_at_both_depths(tmp_path):
    """runs/<run>/ and runs/<model>/<run>/ both exist in this repo."""
    write_run_bundle(tmp_path, model_name="propestnet", run_id="deep")
    # The shallow layout: runs/<run>/ with no model directory above it.
    shallow = write_run_bundle(tmp_path, model_name="propestnet", run_id="tmp")
    shallow.rename(tmp_path / "shallow")

    runs = find_runs(tmp_path)

    assert len(runs) == 2
    assert all(path.name == "best_model.pt" for _, path, _ in runs)


def test_loaded_model_reports_its_own_preprocessing(tmp_path):
    """Preprocessing comes off the checkpoint, not from a constant in the app."""
    write_run_bundle(tmp_path, image_size=128, mean=[0.485, 0.456, 0.406])

    loaded = load_best_model(num_classes=15, runs_dir=tmp_path)

    assert loaded.model is not None
    assert loaded.image_size == 128
    assert loaded.mean == pytest.approx([0.485, 0.456, 0.406])
    assert len(loaded.class_names) == 15


# ------------------------------------------------- automatic vs explicit choice
def test_an_ineligible_run_is_never_chosen_automatically(tmp_path):
    """The legacy import scores higher, and must still lose to the official run.

    Its 0.66 was measured under a different protocol. If discovery ranked on
    score alone, an explicitly non-comparable number would take the top slot.
    """
    write_run_bundle(tmp_path, model_name="propestnet", run_id="official",
                     results_extra={"best_val_macro_f1": 0.42})
    write_legacy_custom_cnn_bundle(tmp_path, best_val_macro_f1=0.99)

    loaded = load_best_model(num_classes=15, runs_dir=tmp_path)

    assert loaded.model is not None
    assert loaded.path.parent.name == "official"
    # The official run's 128px, not the legacy bundle's 160px.
    assert loaded.image_size == 128
    assert loaded.crop_margin == pytest.approx(0.25)


def test_discovery_omits_ineligible_runs(tmp_path):
    write_legacy_custom_cnn_bundle(tmp_path)

    assert find_runs(tmp_path) == []
    # ...but they are still discoverable when explicitly asked for.
    assert len(find_runs(tmp_path, include_ineligible=True)) == 1


def test_an_ineligible_run_is_not_served_even_when_it_is_the_only_one(tmp_path):
    write_legacy_custom_cnn_bundle(tmp_path)

    loaded = load_best_model(num_classes=15, runs_dir=tmp_path)

    assert loaded.model is None


def test_an_ineligible_run_still_loads_when_selected_by_path(tmp_path):
    """Excluded from automatic selection is not the same as unusable."""
    run_dir = write_legacy_custom_cnn_bundle(tmp_path)

    loaded = load_best_model(num_classes=15, model_path=run_dir / "best_model.pt")

    assert loaded.model is not None
    assert loaded.image_size == 160
    assert loaded.mean == pytest.approx([0.485, 0.456, 0.406])
    assert loaded.std == pytest.approx([0.229, 0.224, 0.225])
    assert loaded.crop_margin == pytest.approx(0.15)


def test_bundles_without_the_new_field_stay_eligible(tmp_path):
    """Backwards compatibility: an older bundle has no opinion, so it is usable."""
    write_run_bundle(tmp_path)  # writes no eligible_for_automatic_selection

    loaded = load_best_model(num_classes=15, runs_dir=tmp_path)

    assert loaded.model is not None


def test_the_picker_lists_an_ineligible_run_and_flags_it(tmp_path):
    write_legacy_custom_cnn_bundle(tmp_path)

    described = describe_runs(num_classes=15, runs_dir=tmp_path)

    assert len(described) == 1
    assert described[0]["usable"] is True
    assert described[0]["eligible_for_automatic_selection"] is False
    assert described[0]["comparable_to_main"] is False


# ------------------------------------------------------------- the crop contract
def test_an_imported_custom_cnn_crops_with_its_own_margin(tmp_path):
    run_dir = write_legacy_custom_cnn_bundle(tmp_path)

    loaded = load_best_model(num_classes=15, model_path=run_dir / "best_model.pt")

    assert loaded.crop_mode == "box"
    assert loaded.crop_margin == pytest.approx(0.15)


def test_a_bundle_without_crop_metadata_keeps_the_protocol_default(tmp_path):
    """Existing checkpoints must be fed exactly what they were fed before."""
    write_run_bundle(tmp_path)  # writes no crop_mode / crop_margin

    loaded = load_best_model(num_classes=15, runs_dir=tmp_path)

    assert loaded.crop_margin == pytest.approx(0.25)


def test_the_loaded_margin_reaches_the_crop(tmp_path, monkeypatch):
    """The margin must travel from the checkpoint into crop_to_box, not be re-typed."""
    from PIL import Image

    import app.propest_inference as inference

    run_dir = write_legacy_custom_cnn_bundle(tmp_path)
    assistant = PestAssistant(model_path=run_dir / "best_model.pt",
                              llm=OllamaClient(base_url="http://127.0.0.1:9"))

    seen: list[float] = []
    original = inference.crop_to_box

    def spy(image, box, margin=0.25):
        seen.append(margin)
        return original(image, box, margin=margin)

    monkeypatch.setattr(inference, "crop_to_box", spy)

    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (400, 300), "green").save(photo)
    assistant.predict(photo, box=[50, 50, 250, 200], tta=False)

    assert seen == [pytest.approx(0.15)]


def test_a_box_crop_model_warns_about_uncropped_photos(tmp_path):
    """A farmer's upload has no box; the app must say that is out of distribution."""
    run_dir = write_legacy_custom_cnn_bundle(tmp_path)
    assistant = PestAssistant(model_path=run_dir / "best_model.pt",
                              llm=OllamaClient(base_url="http://127.0.0.1:9"))

    message = assistant.status_message

    assert "cropped insect boxes" in message
    assert "0.15" in message


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
    # 93.3%. Well below half means the pipeline is broken, not under-trained.
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


# ------------------------------------------------------- prior correction (13)
def test_prior_correction_is_identity_without_a_tau_or_a_prior():
    """Every checkpoint written before section 13 must behave exactly as before."""
    from app.propest_inference import adjust_for_prior

    probabilities = torch.tensor([0.5, 0.3, 0.2])
    prior = [0.7, 0.2, 0.1]

    assert torch.equal(adjust_for_prior(probabilities, prior, 0.0), probabilities)
    assert torch.equal(adjust_for_prior(probabilities, None, 0.5), probabilities)
    assert torch.equal(adjust_for_prior(probabilities, [], 0.5), probabilities)


def test_prior_correction_moves_mass_the_documented_direction():
    """Positive tau favours rare classes, negative tau favours common ones."""
    from app.propest_inference import adjust_for_prior

    probabilities = torch.tensor([0.5, 0.3, 0.2])
    prior = [0.7, 0.2, 0.1]   # class 0 common, class 2 rare

    towards_rare = adjust_for_prior(probabilities, prior, 0.5)
    towards_common = adjust_for_prior(probabilities, prior, -0.5)

    assert towards_rare[2] > probabilities[2] > towards_common[2]
    assert towards_common[0] > probabilities[0] > towards_rare[0]
    for adjusted in (towards_rare, towards_common):
        assert float(adjusted.sum()) == pytest.approx(1.0)


def test_prior_correction_matches_subtracting_tau_log_prior_from_logits():
    """The app's formula and the textbook one have to be the same operation.

    Section 13 selects tau against `p / prior ** tau`. If that ever stops
    agreeing with logit adjustment proper, the app is applying a correction
    nobody measured.
    """
    from app.propest_inference import adjust_for_prior

    logits = torch.tensor([2.0, 0.5, -1.0])
    prior = torch.tensor([0.7, 0.2, 0.1])
    tau = 0.4

    from_probabilities = adjust_for_prior(torch.softmax(logits, dim=0), prior.tolist(), tau)
    from_logits = torch.softmax(logits - tau * prior.log(), dim=0)

    assert from_probabilities == pytest.approx(from_logits, abs=1e-6)


def test_a_mismatched_prior_is_dropped_rather_than_applied():
    """A prior of the wrong length would silently reweight the wrong classes."""
    loaded = load_best_model(num_classes=15)
    if loaded.model is None:
        pytest.skip("no usable checkpoint in runs/")
    assert not loaded.train_class_prior or len(loaded.train_class_prior) == 15
    if not loaded.train_class_prior:
        assert loaded.logit_adjust_tau == 0.0
        assert not loaded.adjusts_for_prior


def test_prior_correction_takes_a_list_or_a_tensor_and_a_row_or_a_batch():
    """The notebook holds the prior as a tensor, the checkpoint stores a list.

    Section 13's demo cell calls this with a tensor prior and a batched row;
    the app calls it with a list and a single row. A version that accepted only
    one of those would be a trap for whichever caller reached for the other.
    """
    from app.propest_inference import adjust_for_prior

    batched = torch.tensor([[0.5, 0.3, 0.2]])
    prior = [0.7, 0.2, 0.1]

    from_list = adjust_for_prior(batched, prior, -1.0)
    from_tensor = adjust_for_prior(batched, torch.tensor(prior), -1.0)
    from_row = adjust_for_prior(batched[0], prior, -1.0)

    assert torch.allclose(from_list, from_tensor)
    assert torch.allclose(from_list[0], from_row)
    assert float(from_row.sum()) == pytest.approx(1.0)


# --------------------------------------------------------- the model picker
def test_picker_labels_a_run_by_what_it_is_and_how_good_it_is():
    """No folder paths or datestamps in the name a user reads."""
    from app.cnn_model import describe_runs

    for run in describe_runs(num_classes=15):
        assert run["display_label"], f"{run['label']} has no display label"
        assert "/" not in run["display_label"], "a folder path leaked into the picker"
        assert not any(ch.isdigit() and len(part) == 8
                       for part in run["display_label"].split() for ch in part), \
            "a datestamp leaked into the picker"


def test_picker_never_reports_a_validation_score_as_accuracy():
    """A run with no test split has no accuracy, and must say so instead."""
    from app.cnn_model import _test_accuracy, describe_runs

    assert _test_accuracy({"best_val_macro_f1": 0.9}) is None
    assert _test_accuracy({"test": {"accuracy": 0.5}}) == 0.5
    # The corrected setting wins, because that is the one the app serves.
    assert _test_accuracy({"test": {"accuracy": 0.5},
                           "test_with_tta_and_prior": {"accuracy": 0.77}}) == 0.77

    for run in describe_runs(num_classes=15):
        if run["accuracy"] is None and not run["under_trained"]:
            assert "accurate" not in run["display_label"]


def test_an_incompatible_class_count_outranks_a_missing_model_name():
    """The reason shown must be the one that is actually blocking.

    A model trained on another set of pests cannot be fixed by re-importing it,
    so reporting the metadata gap first would send someone to run an import that
    changes nothing.
    """
    from app.cnn_model import describe_runs

    for run in describe_runs(num_classes=15):
        if not run["usable"] and run["num_classes"] not in (None, 15):
            assert "different dataset" in run["problem"]
            assert "re-import" not in run["problem"]
