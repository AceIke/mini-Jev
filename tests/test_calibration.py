import numpy as np
import pytest

from minijev.calibration import (
    brier_binary,
    brier_multiclass,
    classwise_ece,
    expected_calibration_error,
    maximum_calibration_error,
    nll_multiclass,
    reliability_bins,
    reliability_diagram,
    selective_accuracy,
    top_label_ece,
)


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    # Each bin contains 100 predictions whose mean confidence equals the
    # observed frequency exactly.
    probs, labels = [], []
    for conf in (0.1, 0.3, 0.5, 0.7, 0.9):
        n_pos = round(conf * 100)
        probs += [conf] * 100
        labels += [1.0] * n_pos + [0.0] * (100 - n_pos)
    assert expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.0)
    assert maximum_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.0)


def test_overconfidence_produces_positive_ece():
    probs = [0.95] * 200
    labels = [1.0] * 100 + [0.0] * 100  # right only half the time
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.45)


def test_reliability_bins_partition_every_prediction():
    rng = np.random.default_rng(0)
    probs = rng.random(500)
    labels = (rng.random(500) < probs).astype(float)
    bins = reliability_bins(probs, labels, n_bins=10)
    assert len(bins) == 10
    assert sum(b.count for b in bins) == 500
    assert all(0.0 <= b.lo < b.hi <= 1.0 for b in bins)


def test_top_label_ece_endpoints():
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert top_label_ece(perfect, labels) == pytest.approx(0.0)

    confidently_wrong = np.array([[0.95, 0.05], [0.9, 0.1]])
    assert top_label_ece(confidently_wrong, np.array([1, 1])) > 0.8


def test_brier_and_nll_reference_values():
    P = np.array([[0.8, 0.2], [0.1, 0.9]])
    y = np.array([0, 1])
    # (0.2^2 + 0.2^2) and (0.1^2 + 0.1^2) averaged
    assert brier_multiclass(P, y) == pytest.approx((0.08 + 0.02) / 2)
    assert nll_multiclass(P, y) == pytest.approx((-np.log(0.8) - np.log(0.9)) / 2)
    assert brier_binary([1.0, 0.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_classwise_ece_is_zero_for_a_perfect_predictor():
    P = np.eye(3)
    y = np.array([0, 1, 2])
    assert classwise_ece(P, y) == pytest.approx(0.0)


def test_selective_accuracy_improves_when_confidence_is_informative():
    # Confident predictions are the correct ones.
    conf = np.array([0.9, 0.9, 0.9, 0.6, 0.6, 0.2, 0.2, 0.2])
    correct = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
    assert selective_accuracy(conf, correct, 0.5) == pytest.approx(1.0)
    assert selective_accuracy(conf, correct, 1.0) == pytest.approx(5 / 8)
    accs = [selective_accuracy(conf, correct, k / 4) for k in (1, 2, 3, 4)]
    assert accs == sorted(accs, reverse=True)
    assert accs[-1] == pytest.approx(5 / 8)


def test_reliability_diagram_is_text_and_reports_empty_bins():
    diagram = reliability_diagram([0.05] * 10, [0.0] * 10, n_bins=5, width=8)
    assert "confidence" in diagram
    assert "0.00-0.20" in diagram
    assert "-" in diagram  # empty bins are rendered, not skipped silently
