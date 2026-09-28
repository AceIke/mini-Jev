"""Calibration metrics: is ``probability 0.8`` right about 80% of the time?

Calibration is a property of a *group* of predictions, not of a single answer.
Every metric here takes (predicted distribution, realised outcome) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _assign_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, n_bins - 1), edges


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    mean_accuracy: float

    @property
    def gap(self) -> float:
        if self.count == 0:
            return 0.0
        return self.mean_confidence - self.mean_accuracy


def reliability_bins(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Bin ``probs`` and compare mean confidence with the observed frequency."""

    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("probs and labels must have the same shape")
    idx, edges = _assign_bins(p, n_bins)
    out: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            out.append(ReliabilityBin(edges[b], edges[b + 1], 0, float("nan"), float("nan")))
            continue
        out.append(
            ReliabilityBin(
                lo=float(edges[b]),
                hi=float(edges[b + 1]),
                count=n,
                mean_confidence=float(p[mask].mean()),
                mean_accuracy=float(y[mask].mean()),
            )
        )
    return out


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Binary ECE: weighted mean of ``|mean_confidence - mean_accuracy|``."""

    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0.0
    bins = reliability_bins(p, labels, n_bins=n_bins)
    return float(
        sum(b.count * abs(b.gap) for b in bins) / p.size
    )


def maximum_calibration_error(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
) -> float:
    bins = reliability_bins(probs, labels, n_bins=n_bins)
    return float(max((abs(b.gap) for b in bins if b.count > 0), default=0.0))


def top_label_ece(
    prob_matrix: np.ndarray,
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Multiclass confidence calibration on the predicted ("top") label.

    Confidence is ``max(p)``; accuracy is whether ``argmax(p)`` was right.
    """

    P = np.asarray(prob_matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(np.float64)
    return expected_calibration_error(conf, correct, n_bins=n_bins)


def classwise_ece(
    prob_matrix: np.ndarray,
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Average over classes of the one-vs-rest binary ECE (Guo et al. style)."""

    P = np.asarray(prob_matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if P.size == 0:
        return 0.0
    scores = [
        expected_calibration_error(P[:, k], (y == k).astype(np.float64), n_bins=n_bins)
        for k in range(P.shape[1])
    ]
    return float(np.mean(scores))


def brier_multiclass(prob_matrix: np.ndarray, labels: Sequence[int]) -> float:
    """Mean squared error between the predicted distribution and the one-hot truth."""

    P = np.asarray(prob_matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    onehot = np.zeros_like(P)
    onehot[np.arange(P.shape[0]), y] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def brier_binary(probs: Sequence[float], labels: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def nll_multiclass(prob_matrix: np.ndarray, labels: Sequence[int]) -> float:
    """Mean negative log-likelihood of the realised label."""

    P = np.asarray(prob_matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    eps = 1e-12
    picked = np.clip(P[np.arange(P.shape[0]), y], eps, 1.0)
    return float(-np.mean(np.log(picked)))


def selective_accuracy(
    confidence: Sequence[float],
    correct: Sequence[float],
    coverage: float,
) -> float:
    """Accuracy on the most-confident ``coverage`` fraction of predictions.

    Calibration and *usefulness* are different properties. A model can be
    perfectly calibrated and still useless for routing (if confidence is
    independent of correctness). This metric answers the routing question:
    "if I only act on the top 50% by confidence, how often am I right?"
    """

    conf = np.asarray(confidence, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    k = max(1, int(round(coverage * conf.size)))
    order = np.argsort(-conf, kind="stable")[:k]
    return float(hit[order].mean())




def reliability_diagram(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
    width: int = 18,
) -> str:
    """A small ASCII reliability diagram (``C`` = confidence, ``A`` = accuracy)."""

    bins = reliability_bins(probs, labels, n_bins=n_bins)
    lines = [
        " bin            n     conf    acc     gap   C(confidence) / A(accuracy)",
        " " + "-" * 72,
    ]
    for b in bins:
        head = f" {b.lo:4.2f}-{b.hi:4.2f} {b.count:6d}"
        if b.count == 0:
            lines.append(head + "        -       -       -")
            continue
        cbar = "#" * max(0, round(b.mean_confidence * width))
        abar = "#" * max(0, round(b.mean_accuracy * width))
        lines.append(
            f"{head}   {b.mean_confidence:5.3f}   {b.mean_accuracy:5.3f}"
            f"  {b.gap:+6.3f}  C{cbar:<{width}} A{abar}"
        )
    return "\n".join(lines)
