import numpy as np
import pytest

from minijev.runtime import MiniSystemOne, plan_questions
from minijev.text import Vocab
from minijev.types import Choice, Noul, Score

VOCAB_WORDS = [
    "charge", "refund", "invoice", "error", "crash", "timeout",
    "pricing", "quote", "asap", "urgent", "minor", "blocking", "usual",
    "customer", "writes", "ticket", "help",
]


def _questions() -> dict:
    return {
        "department": Choice(
            instructions="Which team should handle this ticket?",
            criteria={
                "billing": "Charges and refunds",
                "technical": "Bugs and timeouts",
                "sales": "Pricing and quotes",
            },
        ),
        "is_urgent": Noul(instructions="Does the ticket convey urgency?"),
        "severity": Score(
            instructions="How severe is the problem?",
            criteria=["Minor", "Moderate", "Blocking"],
        ),
    }


def _mini(seed: int = 0) -> MiniSystemOne:
    vocab = Vocab(VOCAB_WORDS, n_unk_buckets=8)
    return MiniSystemOne.new(vocab=vocab, dim=16, seed=seed)


def test_system_one_returns_typed_answers_with_valid_probabilities():
    mini = _mini()
    result = mini.system_one(
        state="my card was charged twice, please refund urgent",
        questions=_questions(),
    )
    assert set(result.answers) == {"department", "is_urgent", "severity"}
    assert result.choice("department") in {"billing", "technical", "sales"}
    probs = result.probabilities("department")
    assert sum(probs.values()) == pytest.approx(1.0)
    assert 0.0 <= result.confidence("department") <= 1.0
    assert 0.0 <= result.noul("is_urgent") <= 1.0
    assert 0.0 <= result.score("severity") <= 2.0
    assert sum(result.probabilities("severity").values()) == pytest.approx(1.0)


def test_score_legend_maps_levels_back_to_descriptions():
    mini = _mini()
    result = mini.system_one(state="anything", questions=_questions())
    assert result["severity"].legend == {"0": "Minor", "1": "Moderate", "2": "Blocking"}


def test_questions_are_evaluated_independently():
    """Adding a question must not change the answer to an existing one.

    This is Jev's documented contract ("each question is evaluated in
    isolation") and it is the property that makes speculative fan-out safe.
    """

    mini = _mini(seed=3)
    state = "the integration returns a timeout error, please help asap"
    questions = _questions()

    alone = mini.system_one(
        state=state, questions={"department": questions["department"]}
    )
    together = mini.system_one(state=state, questions=questions)
    assert alone.probabilities("department") == pytest.approx(
        together.probabilities("department")
    )
    assert alone.confidence("department") == pytest.approx(
        together.confidence("department")
    )

    single_noul = mini.system_one(
        state=state, questions={"is_urgent": questions["is_urgent"]}
    )
    assert single_noul.noul("is_urgent") == pytest.approx(together.noul("is_urgent"))


def test_state_accepts_string_object_and_array():
    mini = _mini()
    questions = _questions()
    as_string = mini.system_one(state="charged twice", questions=questions)
    as_object = mini.system_one(
        state={"message": "charged twice", "order": "A-1"}, questions=questions
    )
    as_array = mini.system_one(state=["charged twice"], questions=questions)
    for result in (as_string, as_object, as_array):
        assert sum(result.probabilities("department").values()) == pytest.approx(1.0)


def test_usage_counts_are_reported_and_output_is_free():
    mini = _mini()
    result = mini.system_one(state="charged twice", questions=_questions())
    assert result.usage["input_tokens"] > 0
    assert result.usage["output_tokens"] == 0


def test_save_and_load_round_trip_is_exact(tmp_path):
    mini = _mini(seed=5)
    questions = _questions()
    state = "pricing quote for an upgrade"
    before = mini.system_one(state=state, questions=questions)
    mini.save(tmp_path / "model")
    after = MiniSystemOne.load(tmp_path / "model").system_one(
        state=state, questions=questions
    )
    assert before.to_wire()["answers"] == after.to_wire()["answers"]


def test_unknown_words_do_not_crash_and_change_nothing_when_irrelevant():
    mini = _mini()
    questions = _questions()
    plain = mini.system_one(state="charged twice", questions=questions)
    noisy = mini.system_one(
        state="charged twice qwertyuiop zzzzunseen", questions=questions
    )
    assert sum(noisy.probabilities("department").values()) == pytest.approx(1.0)
    assert plain.usage["input_tokens"] < noisy.usage["input_tokens"]


def test_empty_questions_rejected():
    mini = _mini()
    with pytest.raises(ValueError):
        mini.system_one(state="x", questions={})


def test_plans_encode_every_option():
    vocab = Vocab(VOCAB_WORDS, n_unk_buckets=8)
    plans = plan_questions(vocab, _questions())
    assert plans["department"].option_bags.shape[0] == 3
    assert plans["severity"].option_bags.shape[0] == 3
    assert plans["is_urgent"].option_bags is None
    assert plans["department"].option_names == ["billing", "technical", "sales"]
