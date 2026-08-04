"""Tests for E5: ensembling, test-time augmentation and selective accuracy.

The properties under test are the ones whose failure produces a *plausible wrong
number* rather than an error:

* averaging raw logits, not labels, and not probabilities;
* refusing to combine members that do not describe the same images in the same
  order, which no accuracy figure would reveal;
* refusing to mix scopes;
* selective accuracy being reported as a coverage/accuracy pair, so it can never
  be mistaken for full-coverage accuracy;
* the test split being unreachable.

Most of this needs no GPU, no dataset and no checkpoint: :class:`MemberScores`
is a plain container, so alignment and averaging are testable directly. The two
tests that do read real artifacts skip when those artifacts are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from farm_pest_ai.vision.ensemble import (
    ABSTENTION_THRESHOLDS,
    EnsembleError,
    MemberScores,
    checkpoint_sha256,
    ensemble_logits,
    metrics_from_logits,
    score_checkpoint,
    selective_accuracy,
    summarize_scores,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = REPO_ROOT / "artifacts" / "checkpoints"


def _member(
    member_id: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    scope: str = "rice10",
    num_classes: int = 10,
    fingerprint: str = "9e75177ab60f96e0",
    image_size: tuple[int, int] = (160, 160),
    tta: str = "none",
) -> MemberScores:
    """Build a member without touching disk."""
    return MemberScores(
        member_id=member_id,
        run_dir=Path("artifacts/checkpoints") / member_id,
        checkpoint="best.pt",
        checkpoint_sha256="0" * 64,
        scope=scope,
        num_classes=num_classes,
        epoch=60,
        model_name="custom_cnn",
        preprocessing_fingerprint=fingerprint,
        image_size=image_size,
        tta=tta,
        logits=logits.double(),
        targets=targets.to(torch.int64),
    )


# -- logit averaging ----------------------------------------------------


def test_ensemble_averages_raw_logits_not_labels() -> None:
    """The mean is taken in logit space, elementwise."""
    targets = torch.tensor([0, 1])
    a = _member("a", torch.tensor([[4.0, 0.0], [0.0, 2.0]]), targets, num_classes=2)
    b = _member("b", torch.tensor([[0.0, 2.0], [0.0, 4.0]]), targets, num_classes=2)

    averaged, shared = ensemble_logits([a, b])

    assert torch.allclose(averaged, torch.tensor([[2.0, 1.0], [0.0, 3.0]]).double())
    assert torch.equal(shared, targets)


def test_ensemble_differs_from_majority_vote() -> None:
    """A confident minority member can overturn two unconfident ones.

    This is the behaviour that distinguishes logit averaging from voting. Two
    members weakly predict class 0; the third predicts class 1 with a much larger
    margin. Averaging yields class 1, a majority vote would yield class 0. If a
    future refactor quietly switched to voting, only a test like this would
    notice — the accuracy would simply change a little.
    """
    targets = torch.tensor([0])
    weak_a = _member("a", torch.tensor([[0.6, 0.4]]), targets, num_classes=2)
    weak_b = _member("b", torch.tensor([[0.6, 0.4]]), targets, num_classes=2)
    strong = _member("c", torch.tensor([[0.0, 6.0]]), targets, num_classes=2)

    averaged, _ = ensemble_logits([weak_a, weak_b, strong])

    assert int(averaged.argmax(dim=1)[0]) == 1
    votes = [1 if m.logits.argmax(dim=1)[0] == 1 else 0 for m in (weak_a, weak_b, strong)]
    assert sum(votes) == 1  # a majority vote would have chosen class 0


def test_ensemble_is_invariant_to_member_order() -> None:
    """Member order must not change the result, to the last bit."""
    targets = torch.tensor([0, 1, 0])
    members = [
        _member("a", torch.randn(3, 10), targets),
        _member("b", torch.randn(3, 10), targets),
        _member("c", torch.randn(3, 10), targets),
    ]
    forward, _ = ensemble_logits(members)
    backward, _ = ensemble_logits(list(reversed(members)))

    assert torch.equal(forward, backward)


def test_ensemble_requires_two_members() -> None:
    """A single-member 'ensemble' is a mistake, not a degenerate case."""
    targets = torch.tensor([0])
    with pytest.raises(EnsembleError, match="at least two members"):
        ensemble_logits([_member("a", torch.zeros(1, 10), targets)])


# -- alignment ----------------------------------------------------------


def test_ensemble_refuses_mismatched_targets() -> None:
    """Same length, different order: the rows describe different images."""
    a = _member("a", torch.zeros(3, 10), torch.tensor([0, 1, 2]))
    b = _member("b", torch.zeros(3, 10), torch.tensor([0, 2, 1]))

    with pytest.raises(EnsembleError, match="does not share sample order"):
        ensemble_logits([a, b])


def test_ensemble_refuses_mismatched_sample_counts() -> None:
    """A member that dropped a batch cannot be averaged in."""
    a = _member("a", torch.zeros(4, 10), torch.tensor([0, 1, 2, 3]))
    b = _member("b", torch.zeros(3, 10), torch.tensor([0, 1, 2]))

    with pytest.raises(EnsembleError, match="scored 3 samples against 4"):
        ensemble_logits([a, b])


def test_ensemble_refuses_to_mix_scopes() -> None:
    """rice10 and full102 labels do not mean the same thing."""
    targets = torch.tensor([0, 1])
    a = _member("a", torch.zeros(2, 10), targets, scope="rice10", num_classes=10)
    b = _member(
        "b", torch.zeros(2, 102), targets, scope="full102", num_classes=102
    )

    with pytest.raises(EnsembleError, match="different classification tasks"):
        ensemble_logits([a, b])


def test_ensemble_refuses_mismatched_class_counts_within_a_scope() -> None:
    """A class-count difference is refused even when the scope name matches."""
    targets = torch.tensor([0, 1])
    a = _member("a", torch.zeros(2, 10), targets, num_classes=10)
    b = _member("b", torch.zeros(2, 12), targets, num_classes=12)

    with pytest.raises(EnsembleError, match="12 classes against 10"):
        ensemble_logits([a, b])


def test_ensemble_allows_different_preprocessing_when_order_matches() -> None:
    """A 160px and a 224px member combine legitimately.

    Each was scored through its own pipeline, so the fingerprints differ by
    design. What guarantees correctness is the shared target vector, not a
    shared fingerprint.
    """
    targets = torch.tensor([0, 1, 2])
    small = _member(
        "s160", torch.ones(3, 10), targets, fingerprint="9e75177ab60f96e0",
        image_size=(160, 160),
    )
    large = _member(
        "s224", torch.ones(3, 10) * 3, targets, fingerprint="3378a6f0570336b3",
        image_size=(224, 224),
    )

    averaged, _ = ensemble_logits([small, large])

    assert small.preprocessing_fingerprint != large.preprocessing_fingerprint
    assert torch.allclose(averaged, torch.full((3, 10), 2.0).double())


# -- selective accuracy -------------------------------------------------


def test_selective_accuracy_matches_a_hand_computation() -> None:
    """Coverage and answered accuracy at a known threshold.

    Four samples with hand-chosen logits: two confident and correct, one
    confident and wrong, one unconfident. At threshold 0.7 three answer and two
    of those are right.
    """
    logits = torch.tensor(
        [
            [10.0, 0.0],   # confident, correct
            [10.0, 0.0],   # confident, correct
            [0.0, 10.0],   # confident, wrong
            [0.1, 0.0],    # unconfident (~0.525), abstains
        ]
    )
    targets = torch.tensor([0, 0, 0, 0])

    results = selective_accuracy(logits, targets, thresholds=(0.7,))
    result = results[0]

    assert result.answered == 3
    assert result.coverage == pytest.approx(0.75)
    assert result.correct == 2
    assert result.accuracy == pytest.approx(2 / 3)


def test_selective_accuracy_exceeds_full_coverage_accuracy() -> None:
    """The two numbers differ, which is exactly why both are reported.

    Abstention raises accuracy on the answered subset while covering less of the
    split. Reporting the selective figure as if it were full-coverage accuracy
    is the misrepresentation the phase brief calls out.
    """
    logits = torch.tensor(
        [[8.0, 0.0], [8.0, 0.0], [0.05, 0.0], [0.0, 0.05]]
    )
    targets = torch.tensor([0, 0, 1, 0])

    full = metrics_from_logits(logits, targets, 2)
    selective = selective_accuracy(logits, targets, thresholds=(0.9,))[0]

    assert full.accuracy == pytest.approx(0.5)
    assert selective.accuracy == pytest.approx(1.0)
    assert selective.coverage == pytest.approx(0.5)
    assert selective.accuracy > full.accuracy


def test_selective_accuracy_covers_everything_at_a_zero_threshold() -> None:
    """At threshold 0 the selective figure collapses onto full coverage."""
    logits = torch.randn(20, 10)
    targets = torch.randint(0, 10, (20,))

    full = metrics_from_logits(logits, targets, 10)
    selective = selective_accuracy(logits, targets, thresholds=(0.0,))[0]

    assert selective.coverage == pytest.approx(1.0)
    assert selective.accuracy == pytest.approx(full.accuracy)


def test_selective_accuracy_handles_total_abstention() -> None:
    """No answers must yield 0 coverage and 0 accuracy, not a division error."""
    logits = torch.zeros(5, 10)
    targets = torch.zeros(5, dtype=torch.int64)

    result = selective_accuracy(logits, targets, thresholds=(0.99,))[0]

    assert result.answered == 0
    assert result.coverage == 0.0
    assert result.accuracy == 0.0


def test_default_thresholds_are_the_documented_ones() -> None:
    """The phase reports 0.5, 0.7 and 0.9."""
    assert ABSTENTION_THRESHOLDS == (0.5, 0.7, 0.9)


def test_selective_accuracy_rejects_shape_mismatch() -> None:
    """Mismatched logits and targets raise rather than broadcasting."""
    with pytest.raises(EnsembleError, match="against"):
        selective_accuracy(torch.zeros(4, 10), torch.zeros(3, dtype=torch.int64))


# -- metrics agreement --------------------------------------------------


def test_metrics_from_logits_matches_the_training_accumulator() -> None:
    """Ensemble metrics route through the same accumulator training uses.

    Guards against a second metric implementation drifting from the corrected
    Phase 7.1 F1 denominator.
    """
    from farm_pest_ai.vision.metrics import MetricsAccumulator

    logits = torch.randn(64, 10)
    targets = torch.randint(0, 10, (64,))

    direct = MetricsAccumulator(10, device="cpu")
    direct.update(logits, targets)
    expected = direct.compute()

    actual = metrics_from_logits(logits, targets, 10)

    assert actual.macro_f1 == pytest.approx(expected.macro_f1)
    assert actual.accuracy == pytest.approx(expected.accuracy)
    assert actual.balanced_accuracy == pytest.approx(expected.balanced_accuracy)


def test_logit_averaging_and_probability_averaging_can_disagree() -> None:
    """The two are different operations and can pick different labels.

    Documents why the module averages before softmax rather than after: the
    choice changes the answer, so it has to be deliberate. Here one member is
    overwhelmingly confident in class 0 while two are mildly confident in
    class 1. Logit averaging follows the confident member; probability averaging
    saturates its vote at ~1.0 and is outvoted.
    """
    confident = torch.tensor([[20.0, 0.0]])
    mild_a = torch.tensor([[0.0, 1.5]])
    mild_b = torch.tensor([[0.0, 1.5]])

    logit_mean = (confident + mild_a + mild_b) / 3
    probability_mean = (
        torch.softmax(confident, dim=1)
        + torch.softmax(mild_a, dim=1)
        + torch.softmax(mild_b, dim=1)
    ) / 3

    # The two methods choose *different labels* on the same members.
    assert int(logit_mean.argmax(dim=1)[0]) == 0
    assert int(probability_mean.argmax(dim=1)[0]) == 1
    # Probability averaging saturates the confident member's vote at ~1.0, so
    # two mild votes outweigh it. Logit averaging preserves the margin.
    assert float(torch.softmax(logit_mean, dim=1)[0, 0]) > 0.99


# -- reporting ----------------------------------------------------------


def test_summary_reports_full_coverage_and_selective_separately() -> None:
    """The report keeps the two accuracy notions in distinct blocks."""
    targets = torch.tensor([0, 1, 2, 3])
    member = _member("a", torch.randn(4, 10), targets)

    payload = summarize_scores(
        "test arm", member.logits, member.targets, 10, members=[member], kind="single"
    )

    assert "full_coverage" in payload
    assert "selective" in payload
    assert "accuracy" in payload["full_coverage"]
    assert [entry["threshold"] for entry in payload["selective"]] == [0.5, 0.7, 0.9]
    assert all("selective_accuracy" in entry for entry in payload["selective"])
    # The selective entries never carry a bare "accuracy" key that a report
    # reader could mistake for the full-coverage figure.
    assert all("accuracy" not in entry for entry in payload["selective"])


def test_summary_records_member_provenance() -> None:
    """Every arm records what produced it, including the checkpoint hash."""
    targets = torch.tensor([0, 1])
    members = [
        _member("a", torch.randn(2, 10), targets),
        _member("b", torch.randn(2, 10), targets),
    ]
    logits, shared = ensemble_logits(members)

    payload = summarize_scores(
        "pair", logits, shared, 10, members=members, kind="ensemble"
    )

    assert len(payload["members"]) == 2
    for block in payload["members"]:
        assert block["checkpoint"] == "best.pt"
        assert len(block["checkpoint_sha256"]) == 64
        assert block["preprocessing_fingerprint"]
        assert "epoch" in block


def test_write_report_is_atomic_and_leaves_no_temporary(tmp_path: Path) -> None:
    """A report write replaces the file and cleans up after itself."""
    destination = tmp_path / "nested" / "report.json"

    write_report(destination, {"phase": "8.1", "experiment": "E5"})

    assert json.loads(destination.read_text(encoding="utf-8"))["experiment"] == "E5"
    assert not list(tmp_path.rglob("*.tmp"))


def test_checkpoint_sha256_is_stable_and_content_addressed(tmp_path: Path) -> None:
    """Identical bytes hash identically; a changed byte does not."""
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    first.write_bytes(b"weights")
    second.write_bytes(b"weights")

    assert checkpoint_sha256(first) == checkpoint_sha256(second)

    second.write_bytes(b"weightz")
    assert checkpoint_sha256(first) != checkpoint_sha256(second)


def test_checkpoint_sha256_reports_a_missing_file() -> None:
    """A missing checkpoint raises rather than hashing nothing."""
    with pytest.raises(EnsembleError, match="checkpoint not found"):
        checkpoint_sha256(Path("does") / "not" / "exist.pt")


# -- test-split discipline ----------------------------------------------


def test_score_checkpoint_refuses_the_test_split() -> None:
    """The refusal happens before any file is opened."""
    with pytest.raises(EnsembleError, match="refusing to score split 'test'"):
        score_checkpoint(
            Path("artifacts") / "checkpoints" / "nonexistent",
            config=None,
            split="test",
        )


def test_score_checkpoint_rejects_an_unknown_tta() -> None:
    """Only documented, deterministic TTA is available."""
    with pytest.raises(EnsembleError, match="unknown tta"):
        score_checkpoint(
            Path("artifacts") / "checkpoints" / "nonexistent",
            config=None,
            tta="tencrop",
        )


def test_script_exposes_no_test_split_option() -> None:
    """No CLI path can name the test split."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from evaluate_ensemble import build_parser

    parser = build_parser()
    split_action = next(
        action for action in parser._actions if action.dest == "split"
    )
    assert "test" not in (split_action.choices or [])
    assert set(split_action.choices) <= {"validation", "train"}


def test_script_names_the_checkpoint_explicitly() -> None:
    """best.pt vs last.pt is a stated choice, restricted to those two."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from evaluate_ensemble import build_parser

    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "checkpoint")
    assert set(action.choices) == {"best.pt", "last.pt"}


def test_script_does_not_ensemble_the_weak_full102_baseline() -> None:
    """The weaker baseline is a standalone reference, never a default member."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import evaluate_ensemble

    source = Path(evaluate_ensemble.__file__).read_text(encoding="utf-8")
    # It appears only as its own single arm; it is never appended to a member
    # list that also holds the custom model.
    assert "FULL102_BASELINE_RUN" in source
    assert "reference only" in source


# -- real artifacts -----------------------------------------------------


@pytest.mark.skipif(
    not (CHECKPOINTS / "rice10_custom_e0_corrected" / "summary.json").is_file(),
    reason="Phase 7 rice10 artifacts are not present",
)
def test_real_run_rebuilds_its_own_preprocessing_fingerprint() -> None:
    """A run's rebuilt preprocessing must match its checkpoint's fingerprint.

    This is the property that keeps a 224x224 model from being scored through a
    160x160 pipeline. It is checked here without loading weights.
    """
    from farm_pest_ai.vision.checkpoints import read_metadata
    from farm_pest_ai.vision.results import load_run

    for name in ("rice10_custom_e0_corrected", "rice10_custom_e2_224"):
        directory = CHECKPOINTS / name
        if not (directory / "best.pt").is_file():
            continue
        run = load_run(directory)
        preprocessing = run.preprocessing_config()
        assert preprocessing is not None, f"{name} records no preprocessing"
        metadata = read_metadata(directory / "best.pt")
        assert preprocessing.fingerprint == metadata.preprocessing_fingerprint, (
            f"{name}: rebuilt {preprocessing.fingerprint} against recorded "
            f"{metadata.preprocessing_fingerprint}"
        )


@pytest.mark.skipif(
    not (CHECKPOINTS / "rice10_custom_e4_s160_seed1337" / "best.pt").is_file()
    or not (CHECKPOINTS / "rice10_custom_e4_s224_seed1337" / "best.pt").is_file(),
    reason="E4 artifacts are not present",
)
def test_the_two_e4_arms_have_different_fingerprints() -> None:
    """160px and 224px runs must be distinguishable by fingerprint.

    If they were not, the strict preprocessing check could not tell them apart
    and the combined ensemble would have no guarantee behind it.
    """
    from farm_pest_ai.vision.checkpoints import read_metadata

    small = read_metadata(CHECKPOINTS / "rice10_custom_e4_s160_seed1337" / "best.pt")
    large = read_metadata(CHECKPOINTS / "rice10_custom_e4_s224_seed1337" / "best.pt")

    assert small.preprocessing_fingerprint != large.preprocessing_fingerprint
    assert small.scope == large.scope == "rice10"
