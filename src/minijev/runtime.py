"""Runtime: turn a ``(state, questions)`` request into typed answers.

This is the layer a caller actually talks to. It is deliberately shaped like
the TypeSafe client SDKs::

    from minijev import MiniSystemOne, Choice, Noul, Score

    mini = MiniSystemOne.load("runs/triage")
    result = mini.system_one(
        state="charged twice, please refund asap",
        questions={
            "department": Choice(...),
            "is_urgent": Noul(...),
            "severity": Score(...),
        },
    )
    result.choice("department"), result.noul("is_urgent"), result.score("severity")

``usage.input_tokens`` counts the tokens the model actually had to read. Jev
bills input tokens only and gives output tokens away for free; we mirror that
by always reporting ``output_tokens = 0``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .confidence import choice_confidence
from .model import MiniJevModel
from .text import Vocab, normalize_instruction, normalize_state, tokenize
from .types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResult,
)


@dataclass
class QuestionPlan:
    """A question encoded once, so it can be scored against many states."""

    qid: str
    kind: str
    instr_bag: np.ndarray
    option_bags: np.ndarray | None = None
    option_names: list[str] | None = None
    option_labels: list[str] | None = None

    def input_tokens(self) -> int:
        n = float(self.instr_bag.sum())
        if self.option_bags is not None:
            n += float(self.option_bags.sum())
        return int(n)


def plan_questions(
    vocab: Vocab, questions: dict[str, Question]
) -> dict[str, QuestionPlan]:
    """Encode a question set against a fixed vocabulary."""

    plans: dict[str, QuestionPlan] = {}
    for qid, question in questions.items():
        instr_tokens = tokenize(normalize_instruction(question.instructions))
        instr_bag = vocab.bag(instr_tokens)
        if isinstance(question, Noul):
            plans[qid] = QuestionPlan(qid, "noul", instr_bag)
            continue
        names = question.option_names()
        texts = question.option_texts()
        bags = np.stack([vocab.bag(tokenize(t)) for t in texts], axis=0)
        if isinstance(question, Choice):
            labels = [
                normalize_instruction(v) if v is not None else ""
                for v in question.criteria.values()
            ]
            kind = "choice"
        elif isinstance(question, Score):
            labels = [normalize_instruction(v) for v in question.criteria]
            kind = "score"
        else:
            raise TypeError(f"unsupported question: {type(question).__name__}")
        plans[qid] = QuestionPlan(qid, kind, instr_bag, bags, names, labels)
    return plans


class MiniSystemOne:
    """A trained mini-jev model plus everything needed to answer requests."""

    def __init__(
        self,
        model: MiniJevModel,
        vocab: Vocab,
        model_id: str = "mini-jev-0.1",
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.model_id = model_id

    # -- construction ----------------------------------------------------
    @classmethod
    def new(
        cls,
        vocab: Vocab | None = None,
        dim: int = 48,
        seed: int = 0,
        model_id: str = "mini-jev-0.1",
    ) -> "MiniSystemOne":
        vocab = vocab if vocab is not None else Vocab()
        return cls(MiniJevModel(vocab.size, dim=dim, seed=seed), vocab, model_id)

    def save(self, directory: str | os.PathLike) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "model.npz"))
        (path / "vocab.json").write_text(
            json.dumps(self.vocab.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (path / "meta.json").write_text(
            json.dumps({"model_id": self.model_id}, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, directory: str | os.PathLike) -> "MiniSystemOne":
        path = Path(directory)
        vocab = Vocab.from_dict(json.loads((path / "vocab.json").read_text("utf-8")))
        meta_path = path / "meta.json"
        model_id = "mini-jev-0.1"
        if meta_path.exists():
            model_id = json.loads(meta_path.read_text("utf-8")).get("model_id", model_id)
        return cls(MiniJevModel.load(str(path / "model.npz")), vocab, model_id)

    # -- answering -------------------------------------------------------
    def answer_state_bag(
        self,
        state_bag: np.ndarray,
        plans: dict[str, QuestionPlan],
        model_id: str | None = None,
        state_tokens: int | None = None,
    ) -> SystemOneResult:
        """Score a pre-encoded state. Used by both the API and evaluation."""

        answers = {}
        for qid, plan in plans.items():
            if plan.kind == "noul":
                p = self.model.predict_noul(state_bag, plan.instr_bag)
                answers[qid] = NoulAnswer(noul=p)
                continue
            probs = self.model.predict_options(
                state_bag, plan.instr_bag, plan.option_bags
            )
            if plan.kind == "choice":
                dist = {
                    name: float(p) for name, p in zip(plan.option_names, probs)
                }
                best = max(dist, key=lambda k: dist[k])
                answers[qid] = ChoiceAnswer(
                    choice=best,
                    probabilities=dist,
                    confidence=choice_confidence(list(dist.values())),
                )
            else:
                dist = {str(i): float(p) for i, p in enumerate(probs)}
                expected = float(np.dot(probs, np.arange(probs.size)))
                answers[qid] = ScoreAnswer(
                    score=expected,
                    legend={
                        str(i): label
                        for i, label in enumerate(plan.option_labels or [])
                    },
                    probabilities=dist,
                    confidence=choice_confidence(list(dist.values())),
                )

        input_tokens = int(state_tokens or 0) + sum(
            plan.input_tokens() for plan in plans.values()
        )
        return SystemOneResult(
            model=model_id or self.model_id,
            answers=answers,
            usage={"input_tokens": input_tokens, "output_tokens": 0},
        )

    def system_one(
        self,
        state,
        questions: dict[str, Question],
        model: str | None = None,
    ) -> SystemOneResult:
        """Evaluate ``state`` against ``questions``; one answer per question id."""

        if not isinstance(questions, dict) or not questions:
            raise ValueError("`questions` must be a non-empty mapping")
        text = normalize_state(state)
        tokens = tokenize(text)
        state_bag = self.vocab.bag(tokens)
        plans = plan_questions(self.vocab, questions)
        return self.answer_state_bag(
            state_bag, plans, model_id=model, state_tokens=len(tokens)
        )
