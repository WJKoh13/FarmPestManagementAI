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
