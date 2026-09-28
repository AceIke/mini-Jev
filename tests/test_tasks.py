"""The synthetic task must actually generate the posterior it claims to."""

import numpy as np
import pytest

from minijev.calibration import expected_calibration_error, top_label_ece
from minijev.tasks import FILLER, TOPICS, URGENCY_WORDS, NoisyTriageTask


def test_likelihood_tables_are_normalised():
    task = NoisyTriageTask()
    assert task.topic_dist.table.sum(axis=1) == pytest.approx(np.ones(len(TOPICS)))
    assert task.urgency_dist.table.sum(axis=1) == pytest.approx(np.ones(2))
    assert task.severity_dist.table.sum(axis=1) == pytest.approx(np.ones(3))


def test_posteriors_are_proper_distributions():
    task = NoisyTriageTask()
    rng = np.random.default_rng(0)
    for _ in range(50):
        sample = task.sample(rng)
        assert sum(sample.posteriors["department"]) == pytest.approx(1.0)
        assert sum(sample.posteriors["severity"]) == pytest.approx(1.0)
        assert 0.0 <= float(sample.posteriors["is_urgent"]) <= 1.0
        assert all(p > 0.0 for p in sample.posteriors["department"])


def test_generated_state_contains_only_known_vocabulary():
    task = NoisyTriageTask()
    rng = np.random.default_rng(1)
    allowed = set(task.vocab_words) | {"customer", "writes"}
    for _ in range(20):
        sample = task.sample(rng)
        assert set(sample.tokens) <= allowed
        assert len(sample.tokens) == (
            task.n_topic_slots
            + task.n_urgency_slots
            + task.n_severity_slots
            + 2
        )


def test_bayes_posterior_is_calibrated_against_realised_labels():
    """If the closed-form posterior is right, it must be calibrated by construction.

    This is the strongest end-to-end check available without integration: it
    ties the likelihood tables, the sampler, and the posterior computation
    together and would catch an error in any of them.
    """

    task = NoisyTriageTask()
    rng = np.random.default_rng(2)
    n = 6000
    samples = [task.sample(rng) for _ in range(n)]

    P = np.stack([np.asarray(s.posteriors["department"]) for s in samples])
    y = np.array([s.labels["department"] for s in samples])
    assert top_label_ece(P, y, n_bins=10) < 0.06

    p_u = np.array([float(s.posteriors["is_urgent"]) for s in samples])
    y_u = np.array([float(s.labels["is_urgent"]) for s in samples])
    assert expected_calibration_error(p_u, y_u, n_bins=10) < 0.06


def test_the_task_is_neither_trivial_nor_impossible():
    """A too-easy task would make every reliability bin empty."""

    task = NoisyTriageTask()
    rng = np.random.default_rng(3)
    samples = [task.sample(rng) for _ in range(3000)]
    P = np.stack([np.asarray(s.posteriors["department"]) for s in samples])
    y = np.array([s.labels["department"] for s in samples])
    accuracy = float((P.argmax(axis=1) == y).mean())
    peak = P.max(axis=1)
    assert 0.60 < accuracy < 0.90, accuracy
    assert peak.mean() < 0.90, "posterior is too sharp to make a useful demo"
    assert (peak < 0.8).mean() > 0.15, "too few ambiguous samples"


def test_vocabulary_has_no_duplicates():
    task = NoisyTriageTask()
    words = list(task.vocab_words)
    assert len(words) == len(set(words))
    assert set(FILLER) <= set(task.topic_dist.names)
    assert set(URGENCY_WORDS) == set(task.urgency_dist.names)


def test_misconfigured_mass_is_rejected():
    with pytest.raises(ValueError):
        NoisyTriageTask(cue_signal=0.3, cue_confuse=0.3)
