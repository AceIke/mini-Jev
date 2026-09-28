"""The model: one shared text encoder, then non-autoregressive scoring.

The architecture exists to demonstrate three properties of a System One model:

1. The state is read once. ``_context_vector`` turns the state bag and the
   question's instruction bag into a single vector ``u``, and every option is
   scored against that same ``u``.
2. Nothing is generated left to right. All ``k`` option scores come from one
   matmul, ``O @ u``. There is no loop over output positions.
3. Questions do not interact. Each question is a separate forward pass against
   the same state vector, so asking more questions cannot change the answer to
   an existing one (see ``tests/test_runtime.py``).

Everything is plain NumPy with a hand-written backward pass, so the gradient
derivation is inspectable and verifiable (``tests/test_gradcheck.py`` compares
the analytic gradients against finite differences).

The cost of the simplification is the encoder: it is a bag of words, so word
order and syntax are discarded. ``docs/DESIGN.md`` covers what swapping in a
transformer encoder would involve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .numerics import EPS as _EPS
from .numerics import sigmoid, softmax


@dataclass
class Item:
    """One (state, question, target) training row.

    ``target`` is a full probability distribution, not a hard label: the
    synthetic task in ``tasks.py`` knows the exact Bayes posterior, and we train
    the model to match it. That is what makes calibration measurable rather
    than aspirational.
    """

    kind: str  # "choice" | "score" | "noul"
    state_bag: np.ndarray
    instr_bag: np.ndarray
    option_bags: np.ndarray | None
    target: np.ndarray | float


class Grads:
    """Gradient accumulator with the same layout as the parameter dict."""

    __slots__ = ("E", "P_s", "P_q", "P_o", "w_n", "b_n")

    def __init__(self, model: "MiniJevModel") -> None:
        self.E = np.zeros_like(model.E)
        self.P_s = np.zeros_like(model.P_s)
        self.P_q = np.zeros_like(model.P_q)
        self.P_o = np.zeros_like(model.P_o)
        self.w_n = np.zeros_like(model.w_n)
        self.b_n = np.zeros_like(model.b_n)

    def scaled(self, factor: float) -> "Grads":
        out = object.__new__(Grads)
        for name in Grads.__slots__:
            setattr(out, name, getattr(self, name) * factor)
        return out

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in Grads.__slots__}


class MiniJevModel:
    """A tiny bilinear decision model over bag-of-words features."""

    def __init__(self, vocab_size: int, dim: int = 48, seed: int = 0) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self.dim)
        self.E = rng.normal(0.0, scale, (self.vocab_size, self.dim))
        self.P_s = rng.normal(0.0, scale, (self.dim, self.dim))
        self.P_q = rng.normal(0.0, scale, (self.dim, self.dim))
        self.P_o = rng.normal(0.0, scale, (self.dim, self.dim))
        self.w_n = rng.normal(0.0, scale, (self.dim,))
        self.b_n = np.zeros(1, dtype=np.float64)

    # -- parameters ------------------------------------------------------
    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "E": self.E,
            "P_s": self.P_s,
            "P_q": self.P_q,
            "P_o": self.P_o,
            "w_n": self.w_n,
            "b_n": self.b_n,
        }

    def n_parameters(self) -> int:
        return int(sum(p.size for p in self.parameters().values()))

    # -- forward ---------------------------------------------------------
    def _context_vector(
        self, state_bag: np.ndarray, instr_bag: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode state once, mix in the question, return the context vector."""

        e_s = state_bag @ self.E
        s = self.P_s @ e_s
        e_q = instr_bag @ self.E
        q = self.P_q @ e_q
        return s + q, e_s, e_q

    def score_options(
        self,
        state_bag: np.ndarray,
        instr_bag: np.ndarray,
        option_bags: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Score every option in one matmul. Returns raw logits and a cache."""

        u, e_s, e_q = self._context_vector(state_bag, instr_bag)
        e_o = option_bags @ self.E
        o = e_o @ self.P_o.T
        logits = (o @ u) / np.sqrt(self.dim)
        cache = {
            "u": u,
            "e_s": e_s,
            "e_q": e_q,
            "e_o": e_o,
            "o": o,
            "state_bag": state_bag,
            "instr_bag": instr_bag,
            "option_bags": option_bags,
        }
        return logits, cache

    def score_noul(
        self, state_bag: np.ndarray, instr_bag: np.ndarray
    ) -> tuple[float, dict]:
        u, e_s, e_q = self._context_vector(state_bag, instr_bag)
        logit = float(self.w_n @ u + self.b_n[0])
        cache = {
            "u": u,
            "e_s": e_s,
            "e_q": e_q,
            "state_bag": state_bag,
            "instr_bag": instr_bag,
        }
        return logit, cache

    def predict_options(
        self,
        state_bag: np.ndarray,
        instr_bag: np.ndarray,
        option_bags: np.ndarray,
    ) -> np.ndarray:
        logits, _ = self.score_options(state_bag, instr_bag, option_bags)
        return softmax(logits)

    def predict_noul(self, state_bag: np.ndarray, instr_bag: np.ndarray) -> float:
        logit, _ = self.score_noul(state_bag, instr_bag)
        return sigmoid(logit)

    # -- backward --------------------------------------------------------
    def _backward_context(self, cache: dict, d_u: np.ndarray, grads: Grads) -> None:
        """dL/d(context vector) -> parameter gradients for the shared encoder."""

        # u = s + q
        d_s = d_u
        d_q = d_u
        # s = P_s @ e_s ; e_s = state_bag @ E
        grads.P_s += np.outer(d_s, cache["e_s"])
        d_e_s = self.P_s.T @ d_s
        grads.E += np.outer(cache["state_bag"], d_e_s)
        # q = P_q @ e_q ; e_q = instr_bag @ E
        grads.P_q += np.outer(d_q, cache["e_q"])
        d_e_q = self.P_q.T @ d_q
        grads.E += np.outer(cache["instr_bag"], d_e_q)

    def backward_options(self, cache: dict, d_logits: np.ndarray, grads: Grads) -> None:
        inv = 1.0 / np.sqrt(self.dim)
        o = cache["o"]
        # logits = (o @ u) * inv
        d_o = np.outer(d_logits, cache["u"]) * inv
        d_u = (o.T @ d_logits) * inv
        # o = e_o @ P_o.T ; e_o = option_bags @ E
        grads.P_o += d_o.T @ cache["e_o"]
        grads.E += cache["option_bags"].T @ (d_o @ self.P_o)
        self._backward_context(cache, d_u, grads)

    def backward_noul(self, cache: dict, d_logit: float, grads: Grads) -> None:
        u = cache["u"]
        # logit = w_n . u + b_n
        grads.w_n += d_logit * u
        grads.b_n[0] += d_logit
        self._backward_context(cache, d_logit * self.w_n, grads)

    # -- loss ------------------------------------------------------------
    def loss_and_grads(self, items: Sequence[Item]) -> tuple[float, Grads]:
        """Mean soft-target cross-entropy and its gradient over ``items``.

        For a Choice/Score item the target is a distribution ``t`` over the
        options, so ``dL/dlogits = p - t``. For a Noul item the target is a
        scalar ``t`` and ``dL/dlogit = p - t``. Both are exact, not clipped.
        """

        if not items:
            raise ValueError("loss_and_grads needs at least one item")
        grads = Grads(self)
        total = 0.0
        for item in items:
            if item.kind == "noul":
                logit, cache = self.score_noul(item.state_bag, item.instr_bag)
                p = sigmoid(logit)
                t = float(item.target)
                total += -(t * np.log(p + _EPS) + (1.0 - t) * np.log(1.0 - p + _EPS))
                self.backward_noul(cache, p - t, grads)
            else:
                logits, cache = self.score_options(
                    item.state_bag, item.instr_bag, item.option_bags
                )
                p = softmax(logits)
                t = np.asarray(item.target, dtype=np.float64)
                total += float(-np.sum(t * np.log(p + _EPS)))
                self.backward_options(cache, p - t, grads)
        scale = 1.0 / len(items)
        return total * scale, grads.scaled(scale)

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        payload = {name: p for name, p in self.parameters().items()}
        payload["__meta__"] = np.array(
            [json.dumps({"vocab_size": self.vocab_size, "dim": self.dim})]
        )
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> "MiniJevModel":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"][0]))
            model = cls(meta["vocab_size"], dim=meta["dim"])
            for name in model.parameters():
                model.parameters()[name][...] = data[name]
        return model


class Adam:
    """Adam, written out so the training loop has no hidden dependencies."""

    def __init__(
        self,
        model: MiniJevModel,
        lr: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.model = model
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in model.parameters().items()}
        self.v = {k: np.zeros_like(v) for k, v in model.parameters().items()}

    def step(self, grads: Grads) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1**self.t
        bc2 = 1.0 - b2**self.t
        for name, param in self.model.parameters().items():
            g = grads.as_dict()[name]
            self.m[name] = b1 * self.m[name] + (1.0 - b1) * g
            self.v[name] = b2 * self.v[name] + (1.0 - b2) * (g * g)
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
