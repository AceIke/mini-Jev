import pytest

from minijev.confidence import choice_confidence, entropy_confidence, normalize_probs


def test_matches_the_formula_printed_in_the_typesafe_docs():
    # The docs' confidence explorer uses (3 * peak - 1) / 2 for three options
    # and shows 0.85 for [90, 6, 4].
    assert choice_confidence([0.90, 0.06, 0.04]) == pytest.approx(0.85)


def test_endpoints():
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)
    assert choice_confidence([1.0]) == pytest.approx(1.0)
    assert choice_confidence([0.5, 0.5]) == pytest.approx(0.0)


def test_confidence_is_monotone_in_the_peak():
    peaks = [1 / 4, 0.4, 0.55, 0.7, 0.9, 1.0]
    values = [
        choice_confidence([p] + [(1 - p) / 3] * 3) for p in peaks
    ]
    assert values == sorted(values)


def test_normalize_probs_handles_unnormalised_input():
    assert normalize_probs([2.0, 2.0]).tolist() == pytest.approx([0.5, 0.5])
    assert normalize_probs([-1.0, 1.0]).tolist() == pytest.approx([0.0, 1.0])
    assert normalize_probs([0.0, 0.0]).tolist() == pytest.approx([0.5, 0.5])


def test_normalize_probs_rejects_empty():
    with pytest.raises(ValueError):
        normalize_probs([])


def test_entropy_alternative_has_the_same_endpoints():
    assert entropy_confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert entropy_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert 0.0 < entropy_confidence([0.7, 0.2, 0.1]) < 1.0
