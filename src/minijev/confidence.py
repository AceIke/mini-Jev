"""Confidence reported alongside Choice and Score answers.

TypeSafe derives ``confidence`` from the probability distribution the answer
already carries: a distribution concentrated on one option is confident, a flat
one is not. The docs state the three-option case as ``(3 * peak - 1) / 2`` and
explicitly note you are *not* locked into their definition. We implement the
general-n version of that same statistic and ship an entropy alternative.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_probs(probs: Sequence[float]) -> np.ndarray:
    """Return ``probs`` as a non-negative float array summing to 1."""

    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size)
    return arr / total


def choice_confidence(probs: Sequence[float]) -> float:
    """Generalised peak confidence in ``[0, 1]``.

    ``(n * peak - 1) / (n - 1)``, clamped to ``[0, 1]``.

    * a uniform distribution over ``n`` options maps to ``0.0``;
    * all mass on one option maps to ``1.0``;
    * for ``n == 3`` and ``peak == 0.9`` this is ``(3*0.9 - 1)/2 = 0.85``,
      matching the formula printed in the TypeSafe confidence docs.
    """

    arr = normalize_probs(probs)
    n = arr.size
    if n == 1:
        return 1.0
    peak = float(arr.max())
    return float(min(1.0, max(0.0, (n * peak - 1.0) / (n - 1.0))))


def entropy_confidence(probs: Sequence[float]) -> float:
    """Alternative confidence: ``1 - H(p) / log(n)``.

    Same endpoints as :func:`choice_confidence` (uniform -> 0, one-hot -> 1) but
    a different shape. Included to make the point that the statistic is a
    *choice*, and that the raw ``probabilities`` are the thing that matters.
    """

    arr = normalize_probs(probs)
    n = arr.size
    if n == 1:
        return 1.0
    nz = arr[arr > 0.0]
    entropy = float(-(nz * np.log(nz)).sum())
    max_entropy = float(np.log(n))
    return float(min(1.0, max(0.0, 1.0 - entropy / max_entropy)))
