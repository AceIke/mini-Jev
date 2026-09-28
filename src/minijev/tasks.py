"""A synthetic decision task whose Bayes-optimal answers are known exactly.

This is the piece that makes the whole repository falsifiable.

Normally you cannot check whether a model's probabilities are *right*, because
you do not know the true conditional distribution ``P(label | input)``. Here we
define the data-generating process, so we do:

* Latent variables: ``department`` (3 topics), ``is_urgent`` (binary),
  ``severity`` (3 ordered levels).
* The state is a bag of words assembled from three disjoint vocabularies, one
  per latent variable. Each latent variable only influences *its own* segment,
  so the three questions are statistically independent given the state. That is
  what makes each posterior closed-form.
* Every token's class-conditional probability is a known constant, so

      log P(department = j | tokens) = log P(j) + sum_v count_v * log P(v | j)

  is exact. No sampling is needed to know the target distribution.

Consequences we can test:

* The Bayes-optimal predictor for this task is *linear in token counts*, which
  is inside the hypothesis class of ``model.MiniJevModel``. So a trained model
  should approach the optimum, and the residual gap is a real, measurable
  number rather than a story.
* Because we also keep the sampled latent label, we can measure ordinary
  empirical calibration (top-label ECE against realised outcomes), not just
  distance to the posterior.

The task is deliberately word-salad-ish. It is a statistics demo, not a
language benchmark. ``docs/DESIGN.md`` discusses what a language-flavoured
version would require.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .numerics import sigmoid, softmax

TOPICS: tuple[str, ...] = ("billing", "technical", "sales")

TOPIC_CUES: dict[str, tuple[str, ...]] = {
    "billing": ("charge", "invoice", "refund", "payment"),
    "technical": ("error", "crash", "timeout", "bug"),
    "sales": ("pricing", "quote", "upgrade", "demo"),
}

FILLER: tuple[str, ...] = (
    "please",
    "hello",
    "order",
    "account",
    "need",
    "help",
    "today",
    "again",
    "review",
    "question",
)

URGENCY_WORDS: tuple[str, ...] = ("asap", "urgent", "whenever", "sometime")
SEVERITY_WORDS: tuple[str, ...] = ("minor", "moderate", "blocking", "usual")
BOILERPLATE: tuple[str, ...] = ("customer", "writes")


@dataclass
class _Distribution:
    """A categorical distribution with a stable name -> probability index."""

    names: tuple[str, ...]
    table: np.ndarray  # (n_states, n_names)

    def row(self, state: int) -> np.ndarray:
        return self.table[state]

    def index(self, word: str) -> int | None:
        try:
            return self.names.index(word)
        except ValueError:
            return None

    def counts(self, tokens: Iterable[str]) -> np.ndarray:
        out = np.zeros(len(self.names), dtype=np.float64)
        for tok in tokens:
            i = self.index(tok)
            if i is not None:
                out[i] += 1.0
        return out

    def log_row(self, state: int) -> np.ndarray:
        return np.log(self.row(state))


@dataclass
class Sample:
    """One generated state, the latent labels, and the exact posteriors."""

    text: str
    tokens: list[str]
    posteriors: dict[str, np.ndarray | float]
    labels: dict[str, int]


@dataclass
class NoisyTriageTask:
    """Configurable generator; every knob changes the Bayes posterior too."""

    n_topic_slots: int = 12
    n_urgency_slots: int = 4
    n_severity_slots: int = 4
    # Chosen (by grid search over the generator, see docs/DESIGN.md) so that the
    # Bayes-optimal top-1 accuracy is ~0.74 with a genuinely spread posterior:
    # max-posterior quartiles are roughly 0.58 / 0.75 / 0.90. A task that is too
    # easy would make every reliability bin empty and every ECE trivially zero.
    cue_signal: float = 0.10
    cue_confuse: float = 0.05
    topic_prior: tuple[float, ...] = (0.45, 0.35, 0.20)
    urgent_prior: float = 0.30
    severity_prior: tuple[float, ...] = (0.50, 0.30, 0.20)
    urgency_hit: float = 0.25
    urgency_false_alarm: float = 0.05

    def __post_init__(self) -> None:
        topic_vocab = tuple(w for t in TOPICS for w in TOPIC_CUES[t])
        self._topic_vocab = topic_vocab
        n_cue = len(topic_vocab)
        n_filler = len(FILLER)
        cue_mass = self.cue_signal * (n_cue // len(TOPICS))
        confuse_mass = self.cue_confuse * (
            n_cue - n_cue // len(TOPICS)
        )
        filler_mass = 1.0 - cue_mass - confuse_mass
        if filler_mass <= 0.0:
            raise ValueError(
                f"cue_signal={self.cue_signal} and cue_confuse={self.cue_confuse} "
                "over-allocate probability mass; lower them"
            )
        table = np.zeros((len(TOPICS), n_cue + n_filler), dtype=np.float64)
        for ti, topic in enumerate(TOPICS):
            for vi, word in enumerate(topic_vocab):
                owner = topic_vocab.index(word) // (n_cue // len(TOPICS))
                table[ti, vi] = (
                    self.cue_signal if owner == ti else self.cue_confuse
                )
            table[ti, n_cue:] = filler_mass / n_filler
        self.topic_dist = _Distribution(topic_vocab + FILLER, table)

        # Urgency: one marker is "asap"/"urgent", the rest are neutral words.
        urg = np.zeros((2, len(URGENCY_WORDS)), dtype=np.float64)
        for wi, word in enumerate(URGENCY_WORDS):
            hit = word in ("asap", "urgent")
            if hit:
                urg[1, wi] = self.urgency_hit
                urg[0, wi] = self.urgency_false_alarm
            else:
                urg[1, wi] = 0.5 - self.urgency_hit
                urg[0, wi] = 0.5 - self.urgency_false_alarm
        self.urgency_dist = _Distribution(URGENCY_WORDS, urg)

        sev = np.array(
            [
                [0.40, 0.10, 0.02, 0.48],
                [0.20, 0.35, 0.15, 0.30],
                [0.05, 0.20, 0.50, 0.25],
            ],
            dtype=np.float64,
        )
        self.severity_dist = _Distribution(SEVERITY_WORDS, sev)
        self.topic_vocab = topic_vocab

    # -- generation ------------------------------------------------------
    @property
    def vocab_words(self) -> tuple[str, ...]:
        return (
            self.topic_vocab
            + FILLER
            + URGENCY_WORDS
            + SEVERITY_WORDS
            + BOILERPLATE
        )

    def sample(self, rng: np.random.Generator) -> Sample:
        topic = int(rng.choice(len(TOPICS), p=np.asarray(self.topic_prior)))
        urgent = int(rng.random() < self.urgent_prior)
        severity = int(rng.choice(3, p=np.asarray(self.severity_prior)))

        topic_tokens = rng.choice(
            self.topic_dist.names,
            size=self.n_topic_slots,
            p=self.topic_dist.row(topic),
        )
        urgency_tokens = rng.choice(
            self.urgency_dist.names,
            size=self.n_urgency_slots,
            p=self.urgency_dist.row(urgent),
        )
        severity_tokens = rng.choice(
            self.severity_dist.names,
            size=self.n_severity_slots,
            p=self.severity_dist.row(severity),
        )

        tokens = [str(t) for t in topic_tokens]
        tokens += [str(t) for t in urgency_tokens]
        tokens += [str(t) for t in severity_tokens]
        tokens += list(BOILERPLATE)
        order = rng.permutation(len(tokens))
        shuffled = [tokens[i] for i in order]
        text = "customer writes " + " ".join(shuffled)

        return Sample(
            text=text,
            tokens=shuffled,
            posteriors=self.posteriors_from_tokens(shuffled),
            labels={
                "department": topic,
                "is_urgent": urgent,
                "severity": severity,
            },
        )

    # -- exact inference -------------------------------------------------
    def posteriors_from_tokens(self, tokens: Sequence[str]) -> dict:
        """Closed-form ``P(latent | tokens)`` for the three latent variables."""

        c_topic = self.topic_dist.counts(tokens)
        log_joint = np.log(np.asarray(self.topic_prior)) + c_topic @ np.log(
            self.topic_dist.table.T
        )
        p_topic = softmax(log_joint)

        c_urg = self.urgency_dist.counts(tokens)
        ll_yes = np.log(self.urgent_prior) + float(
            c_urg @ self.urgency_dist.log_row(1)
        )
        ll_no = np.log(1.0 - self.urgent_prior) + float(
            c_urg @ self.urgency_dist.log_row(0)
        )
        p_urgent = sigmoid(ll_yes - ll_no)

        c_sev = self.severity_dist.counts(tokens)
        log_sev = np.log(np.asarray(self.severity_prior)) + c_sev @ np.log(
            self.severity_dist.table.T
        )
        p_severity = softmax(log_sev)

        return {
            "department": p_topic,
            "is_urgent": p_urgent,
            "severity": p_severity,
        }

    # -- the questions a caller would ask --------------------------------
    def questions(self) -> dict:
        from .types import Choice, Noul, Score

        return {
            "department": Choice(
                instructions="Which team should handle this ticket?",
                criteria={
                    "billing": "Charges, invoices, refunds and payment problems.",
                    "technical": "Bugs, crashes, timeouts and integration failures.",
                    "sales": "Pricing, quotes, upgrades and demo requests.",
                },
            ),
            "is_urgent": Noul(
                instructions="Does the ticket convey urgency or time-sensitivity?",
            ),
            "severity": Score(
                instructions="How severe is the problem described in the ticket?",
                criteria=[
                    "Minor: cosmetic or trivial, no user impact.",
                    "Moderate: annoying or degraded, a workaround exists.",
                    "Blocking: stops work or affects many users.",
                ],
            ),
        }
