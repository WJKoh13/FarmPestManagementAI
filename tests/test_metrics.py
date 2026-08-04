"""Tests for classification metrics.

Macro F1 is the project's model-selection metric, so the headline figures are
cross-checked against scikit-learn rather than only against hand-computed
values. The zero-division convention is pinned explicitly: a class the model
never predicts contributes zero to the macro average instead of being dropped.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from farm_pest_ai.vision.metrics import (  # noqa: E402
    MetricsAccumulator,
    MetricsError,
    confusion_matrix,
    label_smoothing_loss_floor,
    macro_f1,
    metrics_from_predictions,
    top_k_accuracy,
)

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def _one_hot_logits(predictions: list[int], num_classes: int) -> torch.Tensor:
    """Build logits whose argmax equals ``predictions``."""
    logits = torch.zeros(len(predictions), num_classes)
    for row, label in enumerate(predictions):
        logits[row, label] = 10.0
    return logits


# -- confusion matrix ---------------------------------------------------


def test_confusion_matrix_is_indexed_true_by_predicted() -> None:
    matrix = confusion_matrix(
        torch.tensor([1, 1]), torch.tensor([0, 0]), num_classes=2
    )
    # Two samples of true class 0 predicted as class 1.
    assert int(matrix[0, 1]) == 2
    assert int(matrix[1, 0]) == 0


def test_confusion_matrix_rejects_out_of_range_label() -> None:
    with pytest.raises(MetricsError, match="outside"):
        confusion_matrix(torch.tensor([5]), torch.tensor([0]), num_classes=3)


def test_confusion_matrix_rejects_shape_mismatch() -> None:
    with pytest.raises(MetricsError, match="does not match"):
        confusion_matrix(torch.tensor([0, 1]), torch.tensor([0]), num_classes=2)


def test_confusion_matrix_handles_empty_input() -> None:
    matrix = confusion_matrix(torch.tensor([]), torch.tensor([]), num_classes=3)
    assert matrix.shape == (3, 3)
    assert int(matrix.sum()) == 0


# -- agreement with scikit-learn ----------------------------------------

CASES = [
    # (targets, predictions, num_classes) - each exercises a different shape of
    # disagreement, including classes that are never predicted.
    ([0, 0, 1, 1], [0, 0, 1, 1], 2),
    ([0, 0, 1, 1], [1, 1, 0, 0], 2),
    (
        [0] * 50 + [1] * 20 + [2] * 5 + [3] * 3,
        [0] * 50 + [1] * 18 + [0] * 2 + [0] * 5 + [1] * 3,
        4,
    ),
    ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], 5),
    ([0, 0, 0, 1], [0, 0, 0, 0], 2),
]


@pytest.mark.parametrize(("targets", "predictions", "num_classes"), CASES)
def test_macro_f1_matches_sklearn(
    targets: list[int], predictions: list[int], num_classes: int
) -> None:
    ours = metrics_from_predictions(predictions, targets, num_classes)
    theirs = sklearn_metrics.f1_score(
        targets,
        predictions,
        average="macro",
        labels=list(range(num_classes)),
        zero_division=0,
    )
    assert ours.macro_f1 == pytest.approx(theirs)


@pytest.mark.parametrize(("targets", "predictions", "num_classes"), CASES)
def test_weighted_f1_matches_sklearn(
    targets: list[int], predictions: list[int], num_classes: int
) -> None:
    ours = metrics_from_predictions(predictions, targets, num_classes)
    theirs = sklearn_metrics.f1_score(
        targets,
        predictions,
        average="weighted",
        labels=list(range(num_classes)),
        zero_division=0,
    )
    assert ours.weighted_f1 == pytest.approx(theirs)


@pytest.mark.parametrize(("targets", "predictions", "num_classes"), CASES)
def test_accuracy_matches_sklearn(
    targets: list[int], predictions: list[int], num_classes: int
) -> None:
    ours = metrics_from_predictions(predictions, targets, num_classes)
    assert ours.accuracy == pytest.approx(
        sklearn_metrics.accuracy_score(targets, predictions)
    )


@pytest.mark.parametrize(("targets", "predictions", "num_classes"), CASES)
def test_balanced_accuracy_matches_sklearn(
    targets: list[int], predictions: list[int], num_classes: int
) -> None:
    ours = metrics_from_predictions(predictions, targets, num_classes)
    assert ours.balanced_accuracy == pytest.approx(
        sklearn_metrics.balanced_accuracy_score(targets, predictions)
    )


@pytest.mark.parametrize(("targets", "predictions", "num_classes"), CASES)
def test_per_class_f1_matches_sklearn(
    targets: list[int], predictions: list[int], num_classes: int
) -> None:
    ours = metrics_from_predictions(predictions, targets, num_classes)
    theirs = sklearn_metrics.f1_score(
        targets,
        predictions,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    assert list(ours.per_class_f1) == pytest.approx(list(theirs))


# -- project conventions ------------------------------------------------


def test_never_predicted_class_scores_zero_not_excluded() -> None:
    """The stricter convention: an abandoned class drags the macro average down."""
    targets = [0] * 10 + [1] * 10
    predictions = [0] * 20
    metrics = metrics_from_predictions(predictions, targets, 2)
    assert metrics.classes_never_predicted == (1,)
    # Class 1 scores 0.0, so the macro average is half of class 0's F1.
    assert metrics.macro_f1 == pytest.approx(metrics.per_class_f1[0] / 2)


def test_macro_f1_penalises_ignoring_rare_classes_more_than_accuracy() -> None:
    """The reason macro F1 is the selection metric."""
    targets = [0] * 95 + [1] * 5
    predictions = [0] * 100
    metrics = metrics_from_predictions(predictions, targets, 2)
    assert metrics.accuracy == pytest.approx(0.95)
    assert metrics.macro_f1 < 0.5


def test_classes_absent_from_split_do_not_affect_balanced_accuracy() -> None:
    """Balanced accuracy averages over classes actually present."""
    targets = [0, 0, 1, 1]
    predictions = [0, 0, 1, 1]
    metrics = metrics_from_predictions(predictions, targets, 4)
    assert metrics.balanced_accuracy == pytest.approx(1.0)


def test_perfect_predictions_score_one() -> None:
    metrics = metrics_from_predictions([0, 1, 2], [0, 1, 2], 3)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.accuracy == pytest.approx(1.0)


# -- accumulator --------------------------------------------------------


def test_accumulator_matches_single_shot_computation() -> None:
    """Accumulating in batches must equal scoring everything at once."""
    torch.manual_seed(0)
    targets = torch.randint(0, 10, (128,))
    logits = torch.randn(128, 10)

    accumulator = MetricsAccumulator(10)
    for start in range(0, 128, 16):
        accumulator.update(logits[start : start + 16], targets[start : start + 16])
    batched = accumulator.compute()

    single = metrics_from_predictions(logits.argmax(1), targets, 10)
    assert batched.macro_f1 == pytest.approx(single.macro_f1)
    assert batched.accuracy == pytest.approx(single.accuracy)


def test_accumulator_rejects_wrong_logit_width() -> None:
    """A model and scope that disagree must fail loudly."""
    accumulator = MetricsAccumulator(10)
    with pytest.raises(MetricsError, match="the model and the active scope disagree"):
        accumulator.update(torch.randn(4, 102), torch.zeros(4, dtype=torch.int64))


def test_accumulator_tracks_weighted_mean_loss() -> None:
    """Loss must be weighted by batch size, not averaged over batches."""
    accumulator = MetricsAccumulator(4)
    accumulator.update(
        _one_hot_logits([0] * 10, 4), torch.zeros(10, dtype=torch.int64), loss=1.0
    )
    accumulator.update(
        _one_hot_logits([0] * 2, 4), torch.zeros(2, dtype=torch.int64), loss=4.0
    )
    # (1.0*10 + 4.0*2) / 12
    assert accumulator.compute().loss == pytest.approx(1.5)


def test_accumulator_disables_top5_for_small_scopes() -> None:
    """Top-5 over 5 classes is trivially 1.0 and carries no information."""
    assert MetricsAccumulator(5).track_top5 is False
    assert MetricsAccumulator(10).track_top5 is True


# -- the Phase 7.1 F1 denominator correction ----------------------------
#
# The shared safe-division helper used to clamp its denominator to `min=1`.
# For precision and recall that is a no-op, because their denominators are
# integer counts. F1's denominator is `precision + recall`, a fraction, and
# clamping rewrote every value in (0, 1) as 1 — under-reporting F1 for exactly
# the weakest classes and dragging the macro average down with them.
#
# The cases above never caught it: each produces classes whose precision and
# recall sum to 0 or to at least 1, so the clamp never engaged. These tests
# target the gap directly.


def _weak_class_case() -> tuple[list[int], list[int], int]:
    """A case whose class 0 has precision + recall strictly between 0 and 1.

    Class 0: 10 true, predicted 20 times, 2 correct.
        precision 2/20 = 0.10, recall 2/10 = 0.20, sum 0.30 < 1.
        Correct F1 = 2*0.1*0.2 / 0.3 = 0.1333...
        Under the clamp the denominator became 1.0, giving 0.04 — a 3.3x
        under-report of the class the metric most needs to see.
    """
    targets = [0] * 10 + [1] * 40
    predictions = [0] * 2 + [1] * 8 + [0] * 18 + [1] * 22
    return targets, predictions, 2


def test_f1_denominator_below_one_is_not_clamped() -> None:
    """The regression itself, checked against the closed-form value."""
    targets, predictions, num_classes = _weak_class_case()
    metrics = metrics_from_predictions(predictions, targets, num_classes)

    precision, recall = metrics.per_class_precision[0], metrics.per_class_recall[0]
    assert precision == pytest.approx(0.10)
    assert recall == pytest.approx(0.20)
    # The condition that triggered the bug.
    assert 0.0 < precision + recall < 1.0

    expected = 2 * precision * recall / (precision + recall)
    assert expected == pytest.approx(0.13333333333333333)
    assert metrics.per_class_f1[0] == pytest.approx(expected)
    # The clamped value, pinned so a reintroduction cannot pass silently.
    assert metrics.per_class_f1[0] != pytest.approx(2 * precision * recall)


@pytest.mark.parametrize("average", ["macro", "weighted", None])
def test_weak_class_f1_matches_sklearn(average: str | None) -> None:
    """scikit-learn agreement on the case the original suite did not cover."""
    targets, predictions, num_classes = _weak_class_case()
    ours = metrics_from_predictions(predictions, targets, num_classes)
    theirs = sklearn_metrics.f1_score(
        targets,
        predictions,
        average=average,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    if average == "macro":
        assert ours.macro_f1 == pytest.approx(theirs)
    elif average == "weighted":
        assert ours.weighted_f1 == pytest.approx(theirs)
    else:
        assert list(ours.per_class_f1) == pytest.approx(list(theirs))


# Reproduced from the real Phase 7 `custom_cnn` run, epoch 1 validation
# (artifacts/checkpoints/rice10_custom_protocolA/metrics.jsonl, first line).
# The epoch-1 model predicted almost everything as class 0, so class 0 had
# precision 0.1801 and recall 0.6847 — summing to 0.865, just under 1, which is
# precisely where the clamp did its damage. The run recorded F1 0.2466 for that
# class; the correct value is 0.2853.
#
# Supports come straight from the manifest. The true-positive and prediction
# counts are recovered from the recorded precision and recall, which pin them
# uniquely: class 0 is 76/422, class 2 is 39/286, class 8 is 2/9.
#
# Those three account for 717 of the split's 721 predictions. The remaining 4
# went to classes 5 and 6, which the run listed as *predicted* — its
# `classes_never_predicted` was [1, 3, 4, 7, 9] — but which scored no true
# positives, so their precision recorded as 0.0 and their exact counts are not
# recoverable. Splitting the remainder between them is the one free choice here,
# and it changes nothing that is being tested: a class with no true positives
# has F1 zero however many times it was guessed.
PHASE7_EPOCH1_SUPPORT = [111, 48, 106, 50, 51, 83, 90, 56, 86, 40]
PHASE7_EPOCH1_CORRECT = [76, 0, 39, 0, 0, 0, 0, 0, 2, 0]
PHASE7_EPOCH1_PREDICTED = [422, 0, 286, 0, 0, 2, 2, 0, 9, 0]


def _phase7_epoch1_predictions() -> tuple[list[int], list[int]]:
    """Rebuild a label sequence reproducing that epoch's confusion pattern.

    Only the per-class true-positive, support and predicted-count totals
    determine precision, recall and F1, so the off-diagonal mass is laid out by
    filling a confusion matrix row by row: each true class keeps its recorded
    true positives and spends the rest of its support on whichever predicted
    class still has budget left.
    """
    predicted_classes = [
        label for label, count in enumerate(PHASE7_EPOCH1_PREDICTED) if count
    ]
    budget = {
        label: PHASE7_EPOCH1_PREDICTED[label] - PHASE7_EPOCH1_CORRECT[label]
        for label in predicted_classes
    }

    targets: list[int] = []
    predictions: list[int] = []
    for label, support in enumerate(PHASE7_EPOCH1_SUPPORT):
        correct = PHASE7_EPOCH1_CORRECT[label]
        targets.extend([label] * support)
        predictions.extend([label] * correct)

        remaining = support - correct
        for spender in predicted_classes:
            if spender == label or remaining == 0:
                continue
            take = min(remaining, budget[spender])
            budget[spender] -= take
            remaining -= take
            predictions.extend([spender] * take)
        assert remaining == 0, f"could not place {remaining} predictions for {label}"

    assert sum(budget.values()) == 0, "prediction counts must be fully consumed"
    return targets, predictions


def test_phase7_run_confusion_pattern_is_reproduced() -> None:
    """The rebuilt sequence must reproduce the recorded run exactly."""
    targets, predictions = _phase7_epoch1_predictions()
    metrics = metrics_from_predictions(predictions, targets, 10)

    assert metrics.samples == 721
    assert list(metrics.per_class_support) == PHASE7_EPOCH1_SUPPORT
    # Exactly what the run recorded for this epoch.
    assert metrics.classes_never_predicted == (1, 3, 4, 7, 9)
    # Precision and recall were recorded correctly by the buggy run: the clamp
    # never touched their integer-count denominators.
    assert metrics.per_class_precision[0] == pytest.approx(0.18009478672985782)
    assert metrics.per_class_recall[0] == pytest.approx(0.6846846846846847)
    assert metrics.accuracy == pytest.approx(0.1622746185852982)


def test_phase7_run_f1_was_under_reported() -> None:
    """The corrected F1 for the recorded epoch, and the value it replaces."""
    targets, predictions = _phase7_epoch1_predictions()
    metrics = metrics_from_predictions(predictions, targets, 10)

    # What the run recorded for class 0 against what it should have been.
    assert metrics.per_class_f1[0] == pytest.approx(0.2851782363977486)
    assert metrics.per_class_f1[0] != pytest.approx(0.24661628453097648)

    # The macro average moves with it, in the direction the bug guaranteed:
    # clamping an under-1 denominator can only shrink F1, never grow it.
    assert metrics.macro_f1 == pytest.approx(0.052626309139237805)
    assert metrics.macro_f1 > 0.03572952550168798

    theirs = sklearn_metrics.f1_score(
        targets, predictions, average="macro", labels=list(range(10)), zero_division=0
    )
    assert metrics.macro_f1 == pytest.approx(theirs)


def test_accumulator_reports_top5_for_large_scopes() -> None:
    accumulator = MetricsAccumulator(102)
    accumulator.update(_one_hot_logits([3, 4], 102), torch.tensor([3, 4]))
    assert accumulator.compute().top5_accuracy == pytest.approx(1.0)


def test_accumulator_empty_returns_zeros_without_raising() -> None:
    metrics = MetricsAccumulator(10).compute()
    assert metrics.samples == 0
    assert metrics.macro_f1 == 0.0


def test_accumulator_ignores_empty_batch() -> None:
    accumulator = MetricsAccumulator(10)
    accumulator.update(torch.zeros(0, 10), torch.zeros(0, dtype=torch.int64))
    assert accumulator.samples == 0


def test_accumulator_reset_clears_state() -> None:
    accumulator = MetricsAccumulator(4)
    accumulator.update(_one_hot_logits([0, 1], 4), torch.tensor([0, 1]))
    accumulator.reset()
    assert accumulator.samples == 0
    assert int(accumulator.matrix.sum()) == 0


# -- top-k --------------------------------------------------------------


def test_top_k_accuracy_counts_a_hit_below_the_argmax() -> None:
    logits = torch.tensor([[1.0, 0.9, 0.0]])
    # True class 1 is not the argmax but is within the top 2.
    assert top_k_accuracy(logits, torch.tensor([1]), k=1) == pytest.approx(0.0)
    assert top_k_accuracy(logits, torch.tensor([1]), k=2) == pytest.approx(1.0)


def test_top_k_accuracy_clamps_k_to_class_count() -> None:
    logits = torch.tensor([[1.0, 0.0]])
    assert top_k_accuracy(logits, torch.tensor([1]), k=99) == pytest.approx(1.0)


def test_top_k_accuracy_handles_empty_batch() -> None:
    assert top_k_accuracy(torch.zeros(0, 10), torch.zeros(0, dtype=torch.int64)) == 0.0


# -- label smoothing floor ----------------------------------------------


def test_label_smoothing_floor_is_zero_without_smoothing() -> None:
    assert label_smoothing_loss_floor(0.0, 10) == 0.0


@pytest.mark.parametrize(("eps", "num_classes"), [(0.1, 10), (0.1, 102), (0.05, 10)])
def test_label_smoothing_floor_matches_empirical_minimum(
    eps: float, num_classes: int
) -> None:
    """The floor must equal the loss a fully converged model actually reaches.

    Minimising the real cross-entropy directly is the ground truth here; an
    analytically wrong floor would make the Phase 6 overfit gate either
    unreachable or trivially passable.
    """
    logits = torch.zeros(1, num_classes, requires_grad=True)
    target = torch.tensor([0])
    optimizer = torch.optim.Adam([logits], lr=0.1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=eps)
    for _ in range(3000):
        optimizer.zero_grad()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
    assert label_smoothing_loss_floor(eps, num_classes) == pytest.approx(
        float(loss.detach()), abs=1e-3
    )


def test_label_smoothing_floor_grows_with_class_count() -> None:
    """A fixed loss threshold cannot serve both scopes."""
    assert label_smoothing_loss_floor(0.1, 102) > label_smoothing_loss_floor(0.1, 10)


def test_label_smoothing_floor_equals_target_entropy() -> None:
    eps, num_classes = 0.1, 10
    on = 1 - eps + eps / num_classes
    off = eps / num_classes
    expected = -(on * math.log(on) + (num_classes - 1) * off * math.log(off))
    assert label_smoothing_loss_floor(eps, num_classes) == pytest.approx(expected)


def test_label_smoothing_floor_rejects_invalid_epsilon() -> None:
    with pytest.raises(MetricsError, match=r"\[0, 1\)"):
        label_smoothing_loss_floor(1.0, 10)


# -- helpers ------------------------------------------------------------


def test_macro_f1_from_matrix_matches_full_computation() -> None:
    targets = torch.tensor([0, 0, 1, 1, 2])
    predictions = torch.tensor([0, 1, 1, 1, 0])
    matrix = confusion_matrix(predictions, targets, 3)
    assert macro_f1(matrix) == pytest.approx(
        metrics_from_predictions(predictions, targets, 3).macro_f1
    )


def test_metrics_to_dict_can_omit_per_class_arrays() -> None:
    metrics = metrics_from_predictions([0, 1], [0, 1], 2)
    assert "per_class" in metrics.to_dict(per_class=True)
    assert "per_class" not in metrics.to_dict(per_class=False)


def test_metrics_get_rejects_uncomputed_metric() -> None:
    metrics = metrics_from_predictions([0, 1], [0, 1], 2)
    with pytest.raises(MetricsError, match="was not computed"):
        metrics.get("top5_accuracy")


def test_metrics_get_rejects_unknown_metric() -> None:
    metrics = metrics_from_predictions([0, 1], [0, 1], 2)
    with pytest.raises(MetricsError, match="unknown metric"):
        metrics.get("auroc")
