"""The two functions that turn scores into a distribution.

They live in their own module so that the synthetic task generator can compute
an exact posterior without importing the model, and so the package has one
definition of softmax instead of a copy in each module that needs it.
"""

from __future__ import annotations

import numpy as np

# Added to probabilities before taking a log, so a hard 0 does not give -inf and
# a hard 1 does not divide by zero.
EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array of logits."""

    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(x: float) -> float:
    """Logistic function for a scalar logit."""

    return float(1.0 / (1.0 + np.exp(-x)))
