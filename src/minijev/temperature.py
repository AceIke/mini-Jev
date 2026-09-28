"""Post-hoc temperature scaling.

Temperature scaling is the cheapest calibration fix there is: one parameter,
fitted after training, no retraining and no architecture change. Given a
distribution ``p`` it returns

    softmax(log(p) / T)

which is the same as dividing the model's logits by ``T``. ``T > 1`` softens an
over-confident model, ``T < 1`` sharpens an under-confident one, and ``T = 1``
is a no-op. With a single parameter it can rescale confidence but it cannot
reorder anything: the argmax, and therefore the answer, never changes.

That last property is the point of having this in the repository. It puts a
number on a question the training ablations cannot answer on their own: how much
of a calibration gap is recoverable after the fact, and how much is baked into
the model's probabilities?

Fit it on data that was not used to evaluate it. ``TemperatureScaler.fit`` takes
an explicit held-out split; ``experiments.temperature_report`` and
``python -m minijev temperature`` do the split for you.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .calibration import nll_multiclass
from .confidence import choice_confidence
from .types import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResult,
)

_EPS = 1e-12
# Below and above these bounds the fitted temperature stops being meaningful and
# starts chasing numerical noise, so the search is clamped rather than unbounded.
_DEFAULT_LO = 0.05
_DEFAULT_HI = 50.0


def scale_probs(
    probs: np.ndarray | Sequence[float], temperature: float
) -> np.ndarray:
    """``softmax(log(probs) / T)``, preserving the input's shape."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, None)
    z = np.log(p) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def scale_binary(p: np.ndarray | float, temperature: float) -> np.ndarray:
    """Rescale P(yes) by applying the temperature to its log-odds."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    arr = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    z = np.log(arr / (1.0 - arr)) / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))


def binary_nll(p: Sequence[float], labels: Sequence[float]) -> float:
    """Mean negative log-likelihood of a set of Bernoulli predictions."""

    arr = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(arr) + (1.0 - y) * np.log(1.0 - arr)))


def _golden_section_min(
    fn, lo: float, hi: float, iters: int = 120
) -> float:
    """Minimise a unimodal function on ``[lo, hi]``. No scipy required."""

    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def fit_temperature(
    prob_matrix: np.ndarray,
    labels: Sequence[int],
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> float:
    """Fit ``T`` by minimising NLL on ``(prob_matrix, labels)``.

    NLL is convex in the logits, and ``logits / T`` traces a straight line
    through logit space, so the objective is unimodal in ``log T`` and a
    golden-section search finds the optimum without gradients.
    """

    P = np.asarray(prob_matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if P.ndim != 2:
        raise ValueError("prob_matrix must be 2-D (n_samples, n_options)")
    if P.shape[0] != y.shape[0]:
        raise ValueError("prob_matrix and labels must have the same length")

    def objective(log_t: float) -> float:
        return nll_multiclass(scale_probs(P, float(np.exp(log_t))), y)

    return float(np.exp(_golden_section_min(objective, np.log(lo), np.log(hi))))


def fit_temperature_binary(
    probs: Sequence[float],
    labels: Sequence[float],
    lo: float = _DEFAULT_LO,
    hi: float = _DEFAULT_HI,
) -> float:
    """Same as :func:`fit_temperature` for a Noul-style P(yes)."""

    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    def objective(log_t: float) -> float:
        return binary_nll(scale_binary(p, float(np.exp(log_t))), y)

    return float(np.exp(_golden_section_min(objective, np.log(lo), np.log(hi))))


@dataclass(frozen=True)
class TemperatureScaler:
    """A fitted temperature, plus helpers to apply it to answers."""

    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not self.temperature > 0.0:
            raise ValueError("temperature must be positive")

    @classmethod
    def fit(cls, prob_matrix, labels, **kwargs) -> "TemperatureScaler":
        return cls(fit_temperature(prob_matrix, labels, **kwargs))

    # -- applying it -----------------------------------------------------
    def transform_probs(self, probs) -> np.ndarray:
        if self.temperature == 1.0:
            return np.asarray(probs, dtype=np.float64)
        return scale_probs(probs, self.temperature)

    def transform_noul(self, p):
        """Rescale P(yes). Accepts a scalar or an array, returns the same shape."""

        if self.temperature == 1.0:
            out = np.asarray(p, dtype=np.float64)
        else:
            out = scale_binary(p, self.temperature)
        # Keep scalars scalar so `NoulAnswer(noul=...)` stays a float.
        return float(out) if np.ndim(out) == 0 else out

    def rescale_answer(self, answer):
        """Return a rescaled copy of one answer. The choice itself is unchanged."""

        if isinstance(answer, NoulAnswer):
            return NoulAnswer(noul=self.transform_noul(answer.noul))

        # Confidence is derived from the distribution, so it has to be
        # recomputed rather than carried over.
        names = list(answer.probabilities)
        scaled = self.transform_probs([answer.probabilities[n] for n in names])
        dist = {name: float(p) for name, p in zip(names, scaled)}
        confidence = choice_confidence(list(dist.values()))

        if isinstance(answer, ChoiceAnswer):
            # uniform scaling cannot change the argmax
            return ChoiceAnswer(
                choice=answer.choice,
                probabilities=dist,
                confidence=confidence,
            )
        if isinstance(answer, ScoreAnswer):
            levels = np.arange(len(names), dtype=np.float64)
            expected = float(np.dot(scaled, levels))
            return ScoreAnswer(
                score=expected,
                legend=dict(answer.legend),
                probabilities=dist,
                confidence=confidence,
            )
        raise TypeError(f"unsupported answer: {type(answer).__name__}")

    def rescale(self, result: SystemOneResult, model_id: str | None = None) -> SystemOneResult:
        """Rescale every answer in a :class:`SystemOneResult`."""

        return SystemOneResult(
            model=model_id or result.model,
            answers={k: self.rescale_answer(v) for k, v in result.answers.items()},
            usage=dict(result.usage),
        )
