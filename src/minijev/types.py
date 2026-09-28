"""The typed question / answer contract.

This mirrors the wire format documented at ``docs.typesafe.ai/api``: a request
carries ``state``, ``model`` and a map of ``questions``; the response carries
``answers`` under the same ids, plus ``usage``.

Three question types:

======== ========================================== ================================
 Type     Question                                   Answer
======== ========================================== ================================
 choice   pick one of a fixed set of options         ``choice`` + ``probabilities``
 score    rate the state on ordered levels            ``score`` + ``probabilities``
 noul     is this statement true?                     ``noul`` in ``[0, 1]``
======== ========================================== ================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

Json = Any


def _as_text(value: Json) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """Base class for the three primitives."""

    TYPE: ClassVar[str] = "question"

    instructions: Json

    def option_texts(self) -> list[str]:
        """The text of each scorable option, in order. Empty for Noul."""

        return []

    def option_names(self) -> list[str]:
        return []

    def to_wire(self) -> dict:
        payload: dict[str, Any] = {
            "type": self.TYPE,
            "instructions": self.instructions,
        }
        payload.update(self.extra_wire_fields())
        return payload

    def extra_wire_fields(self) -> dict:
        return {}


@dataclass(frozen=True)
class Choice(Question):
    """Pick one option from a fixed set.

    ``criteria`` maps option name -> description. The description may be
    ``None`` when the option name speaks for itself.
    """

    TYPE: ClassVar[str] = "choice"

    criteria: Mapping[str, Json] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Choice requires at least one option in `criteria`")
        if len(self.criteria) > 255:
            raise ValueError("Choice accepts at most 255 options")

    def option_names(self) -> list[str]:
        return [str(k) for k in self.criteria]

    def option_texts(self) -> list[str]:
        out = []
        for name, description in self.criteria.items():
            desc = _as_text(description).strip()
            out.append(f"{name}: {desc}" if desc else str(name))
        return out

    def extra_wire_fields(self) -> dict:
        return {"criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score(Question):
    """Rate the state on ordered levels.

    ``criteria`` is an ordered sequence of level descriptions. The answer's
    ``score`` is the probability-weighted mean level index, so it can land
    between levels.
    """

    TYPE: ClassVar[str] = "score"

    criteria: Sequence[Json] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        levels = list(self.criteria)
        if len(levels) < 2:
            raise ValueError("Score requires at least 2 levels")
        if len(levels) > 10:
            raise ValueError("Score accepts at most 10 levels")

    def option_names(self) -> list[str]:
        return [str(i) for i in range(len(self.criteria))]

    def option_texts(self) -> list[str]:
        return [f"{i}: {_as_text(level)}" for i, level in enumerate(self.criteria)]

    def extra_wire_fields(self) -> dict:
        return {"criteria": list(self.criteria)}


@dataclass(frozen=True)
class Noul(Question):
    """A yes/no question. Returns ``P(yes)`` on ``[0, 1]``."""

    TYPE: ClassVar[str] = "noul"

    criteria: Mapping[str, Json] | None = None

    def extra_wire_fields(self) -> dict:
        return {"criteria": dict(self.criteria)} if self.criteria else {}


def choice(instructions: Json, criteria: Mapping[str, Json]) -> Choice:
    """Shorthand matching the JS SDK's ``choice()`` helper."""

    return Choice(instructions=instructions, criteria=criteria)


def score(instructions: Json, criteria: Sequence[Json]) -> Score:
    """Shorthand matching the JS SDK's ``score()`` helper."""

    return Score(instructions=instructions, criteria=list(criteria))


def noul(instructions: Json, criteria: Mapping[str, Json] | None = None) -> Noul:
    """Shorthand matching the JS SDK's ``noul()`` helper."""

    return Noul(instructions=instructions, criteria=criteria)


# --------------------------------------------------------------------------
# answers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceAnswer:
    TYPE: ClassVar[str] = "choice"

    choice: str
    probabilities: dict[str, float]
    confidence: float

    def to_wire(self) -> dict:
        return {
            "type": self.TYPE,
            "choice": self.choice,
            "probabilities": dict(self.probabilities),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class ScoreAnswer:
    TYPE: ClassVar[str] = "score"

    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float

    def to_wire(self) -> dict:
        return {
            "type": self.TYPE,
            "score": float(self.score),
            "legend": dict(self.legend),
            "probabilities": dict(self.probabilities),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class NoulAnswer:
    TYPE: ClassVar[str] = "noul"

    noul: float

    def to_wire(self) -> dict:
        return {"type": self.TYPE, "noul": float(self.noul)}


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


def answer_from_wire(payload: Mapping[str, Json]) -> Answer:
    """Parse one answer from a Jev-shaped response (ours or TypeSafe's)."""

    kind = payload.get("type")
    if kind == "choice":
        return ChoiceAnswer(
            choice=str(payload["choice"]),
            probabilities={str(k): float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(payload["score"]),
            legend={str(k): str(v) for k, v in payload["legend"].items()},
            probabilities={str(k): float(v) for k, v in payload["probabilities"].items()},
            confidence=float(payload["confidence"]),
        )
    if kind == "noul":
        return NoulAnswer(noul=float(payload["noul"]))
    raise ValueError(f"unknown answer type: {kind!r}")


@dataclass
class SystemOneResult:
    """A whole response: one answer per question id, plus token usage."""

    model: str
    answers: dict[str, Answer]
    usage: dict[str, int] = field(default_factory=dict)

    # -- wire ------------------------------------------------------------
    def to_wire(self) -> dict:
        return {
            "model": self.model,
            "answers": {k: v.to_wire() for k, v in self.answers.items()},
            "usage": dict(self.usage),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Json]) -> "SystemOneResult":
        return cls(
            model=str(payload.get("model", "")),
            answers={
                str(k): answer_from_wire(v)
                for k, v in dict(payload.get("answers", {})).items()
            },
            usage={str(k): int(v) for k, v in dict(payload.get("usage", {})).items()},
        )

    # -- convenience -----------------------------------------------------
    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]

    def __contains__(self, key: object) -> bool:
        return key in self.answers

    def choice(self, key: str) -> str:
        return self._typed(key, ChoiceAnswer).choice

    def score(self, key: str) -> float:
        return self._typed(key, ScoreAnswer).score

    def noul(self, key: str) -> float:
        return self._typed(key, NoulAnswer).noul

    def confidence(self, key: str) -> float:
        answer = self.answers[key]
        if isinstance(answer, NoulAnswer):
            raise TypeError("Noul answers carry no confidence (see docs/confidence)")
        return answer.confidence

    def probabilities(self, key: str) -> dict[str, float]:
        answer = self.answers[key]
        if isinstance(answer, NoulAnswer):
            raise TypeError("Noul answers carry no probability distribution")
        return answer.probabilities

    def _typed(self, key: str, cls):
        answer = self.answers[key]
        if not isinstance(answer, cls):
            raise TypeError(
                f"answer {key!r} is a {type(answer).__name__}, not a {cls.__name__}"
            )
        return answer
