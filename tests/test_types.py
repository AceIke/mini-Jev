import pytest

from minijev.types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResult,
    choice,
    noul,
    score,
)


def test_choice_wire_shape_matches_documented_api():
    q = Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "Charges and invoices", "sales": None},
    )
    wire = q.to_wire()
    assert wire == {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and invoices", "sales": None},
    }
    assert q.option_names() == ["billing", "sales"]
    assert q.option_texts() == ["billing: Charges and invoices", "sales"]


def test_score_wire_shape_and_levels():
    q = Score(instructions="How bad?", criteria=["calm", "frustrated", "angry"])
    assert q.to_wire()["criteria"] == ["calm", "frustrated", "angry"]
    assert q.option_names() == ["0", "1", "2"]
    assert q.option_texts()[1] == "1: frustrated"


def test_noul_wire_shape_omits_empty_criteria():
    assert Noul(instructions="Is it urgent?").to_wire() == {
        "type": "noul",
        "instructions": "Is it urgent?",
    }
    assert Noul(instructions="x", criteria={"true": "yes"}).to_wire()["criteria"] == {
        "true": "yes"
    }


def test_question_validation():
    with pytest.raises(ValueError):
        Choice(instructions="x", criteria={})
    with pytest.raises(ValueError):
        Score(instructions="x", criteria=["only one"])
    with pytest.raises(ValueError):
        Score(instructions="x", criteria=[str(i) for i in range(11)])


def test_factory_helpers():
    assert isinstance(choice("x", {"a": "b"}), Choice)
    assert isinstance(score("x", ["a", "b"]), Score)
    assert isinstance(noul("x"), Noul)


def test_system_one_result_round_trip():
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.88, "technical": 0.12},
                "confidence": 0.81,
            },
            "frustration": {
                "type": "score",
                "score": 1.05,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                "confidence": 0.92,
            },
            "is_urgent": {"type": "noul", "noul": 0.95},
        },
        "usage": {"input_tokens": 307, "output_tokens": 20},
    }
    result = SystemOneResult.from_wire(payload)
    assert result.choice("department") == "billing"
    assert result.confidence("department") == pytest.approx(0.81)
    assert result.score("frustration") == pytest.approx(1.05)
    assert result.noul("is_urgent") == pytest.approx(0.95)
    assert result.probabilities("frustration")["1"] == pytest.approx(0.95)
    assert result.usage["input_tokens"] == 307
    assert result.to_wire() == payload


def test_result_typing_errors_are_explicit():
    result = SystemOneResult(
        model="m",
        answers={"a": NoulAnswer(noul=0.5), "b": ChoiceAnswer("x", {"x": 1.0}, 1.0)},
    )
    with pytest.raises(TypeError):
        result.confidence("a")
    with pytest.raises(TypeError):
        result.probabilities("a")
    with pytest.raises(TypeError):
        result._typed("b", ScoreAnswer)
    with pytest.raises(TypeError):
        result.score("b")
