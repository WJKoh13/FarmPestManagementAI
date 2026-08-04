"""E5: inference-only ensembling, test-time augmentation and selective accuracy.

Nothing here trains, fine-tunes or rewrites a checkpoint. Every function reads
completed run artifacts, scores them over the **validation** split, and returns
metrics. That is what makes E5 runnable before any training approval: it can only
observe models that already exist.

Three properties this module is responsible for.

**Each member is scored through its own recorded preprocessing.** A checkpoint
trained at 224x224 loads cleanly into a 160x160 pipeline — ``strict_preprocessing``
defaults off — and yields a plausible, wrong result. Phase 7.2 hit exactly that
while plotting a confusion matrix. Here every member rebuilds its preprocessing
from its own run summary and verifies the resulting fingerprint against the
fingerprint embedded in its checkpoint, with ``strict_preprocessing=True``. A
mismatch raises rather than degrading quietly.

**Raw logits are averaged, never predicted labels.** Voting discards the
model's confidence, which is the part an ensemble is supposed to combine, and it
ties badly at small member counts. Averaging happens before any softmax, on
float64, so member order cannot change the result.

**Sample alignment is proven, not assumed.** Members may run through different
preprocessing and therefore different loaders. Evaluation loaders preserve
official manifest order with ``drop_last=False``, so row *i* is the same image
for every member — but "so it should line up" is precisely the assumption that
produces a silently wrong ensemble. Every member therefore carries its own target
vector, and :func:`ensemble_logits` refuses to combine members whose targets,
sample counts, class counts or scopes disagree.

Scope discipline
    ``rice10`` and ``full102`` members can never be mixed: the scope check
    rejects it before any logit is touched. The two tasks' metrics are never
    combined or ranked against one another.

The test split
    :func:`score_checkpoint` refuses any split but ``train`` and ``validation``.
    Phase 8.1 makes no test-split access of any kind.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..logging_config import get_logger
from .metrics import ClassificationMetrics, MetricsAccumulator
from .results import ResultsError, RunResults, load_run

__all__ = [
    "ABSTENTION_THRESHOLDS",
    "EnsembleError",
    "MemberScores",
    "SelectiveResult",
    "checkpoint_sha256",
    "ensemble_logits",
    "metrics_from_logits",
    "score_checkpoint",
    "selective_accuracy",
    "summarize_scores",
]

logger = get_logger("ensemble")

#: Confidence thresholds the phase reports coverage and answered accuracy at.
ABSTENTION_THRESHOLDS: tuple[float, ...] = (0.5, 0.7, 0.9)

#: Splits E5 may score. The test split is reserved for Phase 9 and is refused.
SCORABLE_SPLITS = ("train", "validation")


class EnsembleError(RuntimeError):
    """Raised when members cannot be scored or safely combined."""


def checkpoint_sha256(path: Path | str) -> str:
    """Hash a checkpoint file, so a report identifies exactly which weights ran.

    Args:
        path: Checkpoint file.

    Returns:
        The full hex SHA-256 digest.

    Raises:
        EnsembleError: If the file does not exist.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise EnsembleError(f"checkpoint not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MemberScores:
    """One checkpoint's raw logits over one split, with its provenance.

    Attributes:
        member_id: Human-readable identifier, usually ``run_id/checkpoint``.
        run_dir: Directory the checkpoint was read from.
        checkpoint: Which checkpoint file was scored, ``best.pt`` or ``last.pt``.
        checkpoint_sha256: Digest of the exact file that produced these logits.
        scope: Scope recorded in the checkpoint.
        num_classes: Output width recorded in the checkpoint.
        epoch: The epoch the checkpoint holds. Recorded because the numerically
            best epoch under the corrected metric is **not** always the epoch
            stored in ``best.pt`` — Phase 7.1 deliberately left one stale.
        model_name: Architecture name.
        preprocessing_fingerprint: Fingerprint of the pipeline actually used,
            verified against the checkpoint's own recorded fingerprint.
        image_size: Input size the member was scored at.
        tta: Name of the test-time augmentation applied, ``"none"`` when the
            member was scored once on the unmodified image.
        logits: ``(N, C)`` float64 logits in official manifest order. Float64
            because these get averaged and compared across members; float32
            summation order would make a member-order difference visible in the
            last digits.
        targets: ``(N,)`` int64 ground-truth project labels, in the same order.
        seconds: Wall-clock scoring time.
    """

    member_id: str
    run_dir: Path
    checkpoint: str
    checkpoint_sha256: str
    scope: str
    num_classes: int
    epoch: int
    model_name: str
    preprocessing_fingerprint: str
    image_size: tuple[int, int]
    tta: str
    logits: Tensor = field(repr=False)
    targets: Tensor = field(repr=False)
    seconds: float = 0.0

    @property
    def samples(self) -> int:
        """Number of scored samples."""
        return int(self.logits.shape[0])

    def describe(self) -> dict[str, Any]:
        """Return the JSON-serialisable provenance block, without the tensors."""
        return {
            "member_id": self.member_id,
            "run_dir": str(self.run_dir),
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "scope": self.scope,
            "num_classes": self.num_classes,
            "epoch": self.epoch,
            "model": self.model_name,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "image_size": list(self.image_size),
            "tta": self.tta,
            "samples": self.samples,
            "seconds": round(self.seconds, 2),
        }


@dataclass(frozen=True)
class SelectiveResult:
    """Coverage and answered accuracy at one confidence threshold.

    Attributes:
        threshold: Softmax-confidence cut-off. Predictions below it abstain.
        coverage: Fraction of the split the model answered on.
        answered: Number of answered predictions.
        accuracy: Accuracy **among answered predictions only**. This is
            *selective* accuracy and is never reported as full-coverage accuracy.
        correct: Correct answered predictions.
    """

    threshold: float
    coverage: float
    answered: int
    accuracy: float
    correct: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "threshold": self.threshold,
            "coverage": self.coverage,
            "answered": self.answered,
            "selective_accuracy": self.accuracy,
            "correct": self.correct,
        }


def _resolve_preprocessing(run: RunResults) -> Any:
    """Rebuild a run's own preprocessing, refusing to guess.

    Raises:
        EnsembleError: If the run summary records no preprocessing block. An
            ensemble member whose pipeline is unknown cannot be scored safely,
            and falling back to the ambient configuration is the exact mistake
            Phase 7.2 caught.
    """
    preprocessing = run.preprocessing_config()
    if preprocessing is None:
        raise EnsembleError(
            f"{run.run_dir} records no preprocessing block, so the pipeline it "
            f"trained under is unknown; refusing to score it through the ambient "
            f"configuration"
        )
    return preprocessing


def score_checkpoint(
    run: RunResults | str | Path,
    config: Any,
    *,
    checkpoint: str = "best.pt",
    split: str = "validation",
    tta: str = "none",
    device: str | None = None,
) -> MemberScores:
    """Score one checkpoint over one split through its **own** preprocessing.

    Args:
        run: A loaded run, or a run directory to load.
        config: Resolved project configuration supplying manifests and runtime.
            Its scope must match the checkpoint's.
        checkpoint: ``best.pt`` or ``last.pt``. Named explicitly by every caller
            rather than defaulted silently, because the two answer different
            questions: ``best.pt`` is the epoch the run's monitored metric chose,
            ``last.pt`` is the final epoch.
        split: ``train`` or ``validation``. The test split is refused.
        tta: ``"none"`` for a single clean pass, or ``"hflip"`` for deterministic
            horizontal-flip TTA, which averages the logits of the original and
            the flipped image.
        device: Device override. Defaults to CUDA when available.

    Returns:
        The member's logits, targets and provenance.

    Raises:
        EnsembleError: If the split is the test split, the checkpoint is
            missing, the TTA name is unknown, or the rebuilt preprocessing
            disagrees with the checkpoint's recorded fingerprint.
    """
    if split not in SCORABLE_SPLITS:
        raise EnsembleError(
            f"refusing to score split {split!r}; Phase 8.1 uses "
            f"{list(SCORABLE_SPLITS)} only and the test split is reserved for "
            f"Phase 9"
        )
    if tta not in ("none", "hflip"):
        raise EnsembleError(
            f"unknown tta {tta!r}; expected 'none' or 'hflip'"
        )

    import time

    from ..data.loaders import build_loaders
    from .checkpoints import CheckpointError, load_checkpoint

    results = run if isinstance(run, RunResults) else load_run(run)
    path = results.run_dir / checkpoint
    if not path.is_file():
        raise EnsembleError(f"checkpoint not found: {path}")

    preprocessing = _resolve_preprocessing(results)
    fingerprint = preprocessing.fingerprint

    # strict_preprocessing=True is the whole point: this is what turns "the
    # pipeline drifted" from a silent accuracy loss into an error.
    try:
        model, metadata, _ = load_checkpoint(
            path,
            scope=config.scope,
            map_location="cpu",
            preprocessing_fingerprint=fingerprint,
            strict_preprocessing=True,
        )
    except CheckpointError as exc:
        raise EnsembleError(f"cannot score {path}: {exc}") from exc

    bundle = build_loaders(config, (split,), preprocessing=preprocessing)
    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(resolved_device).eval()

    started = time.perf_counter()
    chunks: list[Tensor] = []
    labels: list[Tensor] = []
    with torch.no_grad():
        for images, batch_targets in bundle.loaders[split]:
            images = images.to(resolved_device)
            logits = model(images).double()
            if tta == "hflip":
                # dims=(3,) is the width axis of an (N, C, H, W) batch. Averaging
                # the two logit sets — not the two argmaxes — keeps the
                # confidence information the average is meant to combine.
                flipped = model(torch.flip(images, dims=(3,))).double()
                logits = (logits + flipped) / 2.0
            chunks.append(logits.cpu())
            labels.append(batch_targets.reshape(-1).to(torch.int64).cpu())
    elapsed = time.perf_counter() - started

    if not chunks:
        raise EnsembleError(f"{split!r} split produced no batches for {path}")

    member_logits = torch.cat(chunks, dim=0)
    member_targets = torch.cat(labels, dim=0)

    expected = len(bundle.datasets[split])
    if member_logits.shape[0] != expected:
        raise EnsembleError(
            f"{path} scored {member_logits.shape[0]} samples but the {split!r} "
            f"dataset holds {expected}; predictions would not join to the manifest"
        )

    return MemberScores(
        member_id=f"{results.run_dir.name}/{checkpoint}"
        + ("" if tta == "none" else f"+{tta}"),
        run_dir=results.run_dir,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256(path),
        scope=metadata.scope,
        num_classes=metadata.num_classes,
        epoch=metadata.epoch,
        model_name=str(metadata.model.get("name", results.model_name)),
        preprocessing_fingerprint=fingerprint,
        image_size=tuple(preprocessing.image_size),
        tta=tta,
        logits=member_logits,
        targets=member_targets,
        seconds=elapsed,
    )


def ensemble_logits(members: Sequence[MemberScores]) -> tuple[Tensor, Tensor]:
    """Average raw logits across members, after proving they align.

    Uniform weights only. Tuning per-member weights on the same validation split
    that then judges the ensemble would fit the split rather than measure a
    method, so this function deliberately exposes no weight argument.

    Members are permitted to differ in preprocessing — a 160x160 and a 224x224
    model can be combined — because each was scored through its own pipeline and
    the resulting logits describe the same images in the same order. What they
    may not differ in is scope, class count, sample count or targets.

    Args:
        members: Two or more scored members.

    Returns:
        The mean logits ``(N, C)`` and the shared targets ``(N,)``.

    Raises:
        EnsembleError: If fewer than two members are given, or if their scopes,
            class counts, sample counts or targets disagree.
    """
    if len(members) < 2:
        raise EnsembleError(
            f"an ensemble needs at least two members, got {len(members)}"
        )

    first = members[0]
    for member in members[1:]:
        if member.scope != first.scope:
            raise EnsembleError(
                f"cannot ensemble scope {member.scope!r} with {first.scope!r}; "
                f"rice10 and full102 are different classification tasks and their "
                f"labels do not mean the same thing"
            )
        if member.num_classes != first.num_classes:
            raise EnsembleError(
                f"member {member.member_id!r} has {member.num_classes} classes "
                f"against {first.num_classes} for {first.member_id!r}"
            )
        if member.samples != first.samples:
            raise EnsembleError(
                f"member {member.member_id!r} scored {member.samples} samples "
                f"against {first.samples} for {first.member_id!r}"
            )
        if not torch.equal(member.targets, first.targets):
            # Same length, different labels: the two members saw the split in
            # different orders, so averaging row i would combine two different
            # images. This is the failure an ensemble cannot detect from its
            # own accuracy, which is why it is checked rather than assumed.
            raise EnsembleError(
                f"member {member.member_id!r} does not share sample order with "
                f"{first.member_id!r}; their target vectors differ, so their rows "
                f"describe different images"
            )

    stacked = torch.stack([member.logits for member in members], dim=0)
    return stacked.mean(dim=0), first.targets


def metrics_from_logits(
    logits: Tensor, targets: Tensor, num_classes: int
) -> ClassificationMetrics:
    """Compute the project's metric set from logits and targets.

    Routes through :class:`MetricsAccumulator` rather than reimplementing the
    aggregation, so ensemble metrics and training metrics cannot drift apart and
    both carry the Phase 7.1 corrected F1 denominator.
    """
    accumulator = MetricsAccumulator(num_classes, device="cpu")
    accumulator.update(logits.float(), targets)
    return accumulator.compute()


def selective_accuracy(
    logits: Tensor,
    targets: Tensor,
    thresholds: Iterable[float] = ABSTENTION_THRESHOLDS,
) -> list[SelectiveResult]:
    """Coverage and answered accuracy at each confidence threshold.

    Confidence is the maximum softmax probability. Softmax is applied **here**,
    at the policy layer, not inside the model: the models emit raw logits and
    the ensemble average happens in logit space before this point.

    The result is *selective* accuracy — accuracy over the answered subset. It
    rises with the threshold precisely because the model abstains on the cases it
    is unsure about, so quoting it as a full-coverage figure overstates the
    system substantially. Both numbers are returned together for that reason.

    Args:
        logits: ``(N, C)`` raw logits.
        targets: ``(N,)`` ground-truth labels.
        thresholds: Confidence cut-offs.

    Returns:
        One :class:`SelectiveResult` per threshold, in the order given.

    Raises:
        EnsembleError: If shapes disagree.
    """
    if logits.ndim != 2:
        raise EnsembleError(
            f"expected logits of shape (N, C), got {tuple(logits.shape)}"
        )
    if targets.reshape(-1).shape[0] != logits.shape[0]:
        raise EnsembleError(
            f"{logits.shape[0]} logit rows against "
            f"{targets.reshape(-1).shape[0]} targets"
        )

    probabilities = torch.softmax(logits.double(), dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions == targets.reshape(-1).to(predictions.dtype)
    total = int(logits.shape[0])

    results: list[SelectiveResult] = []
    for threshold in thresholds:
        answered_mask = confidence >= float(threshold)
        answered = int(answered_mask.sum())
        hits = int((correct & answered_mask).sum())
        results.append(
            SelectiveResult(
                threshold=float(threshold),
                coverage=answered / total if total else 0.0,
                answered=answered,
                accuracy=hits / answered if answered else 0.0,
                correct=hits,
            )
        )
    return results


def summarize_scores(
    label: str,
    logits: Tensor,
    targets: Tensor,
    num_classes: int,
    *,
    members: Sequence[MemberScores] = (),
    kind: str = "single",
    per_class: bool = True,
) -> dict[str, Any]:
    """Build the JSON report block for one arm.

    Args:
        label: Arm name, such as ``"rice10 E0 best.pt"``.
        logits: The arm's logits, already averaged for an ensemble.
        targets: Ground-truth labels.
        num_classes: Class count for the arm's scope.
        members: Members that produced the logits, recorded for reproducibility.
        kind: ``"single"``, ``"tta"`` or ``"ensemble"``.
        per_class: Whether to include per-class arrays.

    Returns:
        A JSON-serialisable mapping carrying both full-coverage and selective
        figures, plus every member's checkpoint hash.
    """
    metrics = metrics_from_logits(logits, targets, num_classes)
    selective = selective_accuracy(logits, targets)
    return {
        "label": label,
        "kind": kind,
        "scope": members[0].scope if members else None,
        "num_classes": num_classes,
        "samples": int(logits.shape[0]),
        "members": [member.describe() for member in members],
        "full_coverage": metrics.to_dict(per_class=per_class),
        "selective": [result.to_dict() for result in selective],
    }


def write_report(path: Path | str, payload: Mapping[str, Any]) -> Path:
    """Write a report atomically, so an interrupted write leaves the old file.

    Args:
        path: Destination JSON file.
        payload: The report body.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)
    return destination


def load_member_run(run_dir: Path | str) -> RunResults:
    """Load a run directory, translating a results error into an ensemble one.

    Raises:
        EnsembleError: If the directory is not a usable completed run.
    """
    try:
        return load_run(run_dir)
    except ResultsError as exc:
        raise EnsembleError(f"cannot use {run_dir} as an ensemble member: {exc}") from exc
