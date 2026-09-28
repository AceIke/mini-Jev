import numpy as np
import pytest

from minijev.calibration import nll_multiclass
from minijev.temperature import (
    TemperatureScaler,
    binary_nll,
    fit_temperature,
    fit_temperature_binary,
    scale_binary,
    scale_probs,
)
from minijev.types import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResult,
)


def test_temperature_one_is_identity():
    probs = np.array([0.7, 0.2, 0.1])
    assert scale_probs(probs, 1.0) == pytest.approx(probs)
    assert float(scale_binary(0.3, 1.0)) == pytest.approx(0.3)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        scale_probs([0.5, 0.5], 0.0)
    with pytest.raises(ValueError):
        TemperatureScaler(0.0)


def test_scaling_never_reorders():
    rng = np.random.default_rng(0)
    for _ in range(50):
        probs = rng.dirichlet(np.ones(4))
        for temperature in (0.2, 0.5, 1.0, 2.0, 8.0):
            scaled = scale_probs(probs, temperature)
            assert scaled.argmax() == probs.argmax()
            assert scaled.sum() == pytest.approx(1.0)


def test_raising_temperature_flattens_and_lowering_sharpens():
    probs = [0.8, 0.15, 0.05]
    assert scale_probs(probs, 5.0).max() < probs[0]
    assert scale_probs(probs, 0.2).max() > probs[0]


def test_binary_scaling_keeps_the_ordering_and_the_endpoints():
    assert float(scale_binary(0.5, 7.0)) == pytest.approx(0.5)
    sharp = scale_binary(np.array([0.2, 0.8]), 0.3)
    assert sharp[0] < 0.2 and sharp[1] > 0.8


def test_transform_noul_accepts_scalars_and_arrays():
    scaler = TemperatureScaler(4.0)
    single = scaler.transform_noul(0.9)
    assert isinstance(single, float) and single < 0.9
    batch = scaler.transform_noul(np.array([0.9, 0.1]))
    assert batch.shape == (2,)
    assert batch[0] < 0.9 and batch[1] > 0.1
    assert TemperatureScaler(1.0).transform_noul(0.3) == pytest.approx(0.3)


def _overconfident_dataset(factor: float, n: int, seed: int):
    """Labels drawn from a true distribution; model over-sharpened by ``factor``."""

    rng = np.random.default_rng(seed)
    truth = rng.dirichlet(np.ones(3), size=n)
    labels = np.array([rng.choice(3, p=row) for row in truth])
    model = scale_probs(truth, 1.0 / factor)  # dividing log p by T<1 sharpens
    return model, labels


@pytest.mark.parametrize("factor", [2.0, 3.0, 5.0])
def test_fit_recovers_a_known_overconfidence_factor(factor):
    model, labels = _overconfident_dataset(factor, n=20000, seed=7)
    temperature = fit_temperature(model, labels)
    # The model is softmax(log p / T_model) with T_model = 1/factor, so the
    # optimal correction is T = factor.
    assert temperature == pytest.approx(factor, rel=0.15)


def test_fit_on_already_calibrated_data_returns_about_one():
    rng = np.random.default_rng(3)
    truth = rng.dirichlet(np.ones(3), size=20000)
    labels = np.array([rng.choice(3, p=row) for row in truth])
    assert fit_temperature(truth, labels) == pytest.approx(1.0, rel=0.1)


def test_fitting_reduces_nll_on_held_out_data():
    rng = np.random.default_rng(11)
    truth = rng.dirichlet(np.ones(3), size=4000)
    labels = np.array([rng.choice(3, p=row) for row in truth])
    overconfident = scale_probs(truth, 0.2)  # T_model = 0.2 -> very sharp
    fit, test = slice(0, 2000), slice(2000, None)
    temperature = fit_temperature(overconfident[fit], labels[fit])
    calibrated = scale_probs(overconfident[test], temperature)
    assert nll_multiclass(calibrated, labels[test]) < nll_multiclass(
        overconfident[test], labels[test]
    )


def test_binary_fit_recovers_overconfidence():
    rng = np.random.default_rng(5)
    truth = rng.random(20000)
    y = (rng.random(20000) < truth).astype(float)
    overconfident = scale_binary(truth, 0.25)  # T_model = 0.25
    assert fit_temperature_binary(overconfident, y) == pytest.approx(4.0, rel=0.2)


def test_binary_nll_reference_value():
    p = np.array([1.0, 0.0, 0.5])
    y = np.array([1.0, 0.0, 1.0])
    # Two near-perfect predictions plus one coin flip.
    assert binary_nll(p, y) == pytest.approx(np.log(2) / 3, abs=1e-6)


def test_rescale_preserves_the_choice_and_recomputes_confidence():
    answer = ChoiceAnswer(
        choice="b",
        probabilities={"a": 0.7, "b": 0.2, "c": 0.1},
        confidence=0.55,
    )
    scaler = TemperatureScaler(4.0)
    rescaled = scaler.rescale_answer(answer)
    assert rescaled.choice == "b"  # rescaling cannot reorder
    assert rescaled.probabilities["a"] < 0.7
    assert sum(rescaled.probabilities.values()) == pytest.approx(1.0)
    assert rescaled.confidence != answer.confidence
    assert 0.0 <= rescaled.confidence <= 1.0


def test_rescale_score_recomputes_the_expected_level():
    answer = ScoreAnswer(
        score=0.02 * 0 + 0.08 * 1 + 0.90 * 2,
        legend={"0": "minor", "1": "moderate", "2": "blocking"},
        probabilities={"0": 0.02, "1": 0.08, "2": 0.90},
        confidence=0.85,
    )
    rescaled = TemperatureScaler(3.0).rescale_answer(answer)
    assert rescaled.legend == answer.legend
    # Softening pulls mass off the extreme level, so the expected level falls.
    assert rescaled.score < answer.score
    assert rescaled.probabilities["2"] < 0.90
    assert rescaled.probabilities["1"] > 0.08
    assert sum(rescaled.probabilities.values()) == pytest.approx(1.0)


def test_rescale_whole_result_and_noul():
    result = SystemOneResult(
        model="mini-jev-0.2",
        answers={
            "department": ChoiceAnswer("a", {"a": 0.9, "b": 0.1}, 0.8),
            "is_urgent": NoulAnswer(noul=0.9),
        },
        usage={"input_tokens": 10, "output_tokens": 0},
    )
    scaled = TemperatureScaler(5.0).rescale(result)
    assert scaled.model == "mini-jev-0.2"
    assert scaled.usage == result.usage
    assert scaled.answers["is_urgent"].noul < 0.9
    assert scaled.answers["department"].choice == "a"
    assert scaled.answers["department"].probabilities["a"] < 0.9


def test_rescale_is_a_no_op_at_one():
    answer = ChoiceAnswer("a", {"a": 0.6, "b": 0.4}, 0.2)
    assert TemperatureScaler(1.0).rescale_answer(answer).probabilities == pytest.approx(
        answer.probabilities
    )
