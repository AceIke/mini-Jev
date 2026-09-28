"""End-to-end: does a short CPU run actually become calibrated and accurate?"""

import numpy as np

from minijev.tasks import NoisyTriageTask
from minijev.train import TrainConfig, train


def _quick_run(tmp_path, **overrides):
    config = TrainConfig(
        n_train=800, n_eval=400, epochs=6, batch_size=128, dim=32, seed=0
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    run = train(config, verbose=False)
    return run


def test_training_reduces_the_loss_and_reaches_the_bayes_optimum(tmp_path):
    run = _quick_run(tmp_path)
    metrics = run.metrics
    history = metrics["train"]["history"]
    assert history[-1]["train_nll"] < history[0]["train_nll"]

    dept = metrics["questions"]["department"]
    # The Bayes-optimal predictor for this generator is linear in token counts,
    # so even this small run should land close to it. Give it a wide margin:
    # the point of the assertion is "close", not a specific decimal.
    assert abs(dept["accuracy"] - dept["bayes_accuracy"]) < 0.05
    assert dept["mean_posterior_l1"] < 0.05
    assert abs(dept["excess_nll_over_bayes"]) < 0.05


def test_predictions_are_empirically_calibrated(tmp_path):
    run = _quick_run(tmp_path)
    metrics = run.metrics["questions"]
    assert metrics["department"]["top_label_ece"] < 0.08
    assert metrics["is_urgent"]["ece"] < 0.08
    assert metrics["severity"]["top_label_ece"] < 0.08


def test_confidence_is_useful_for_routing(tmp_path):
    run = _quick_run(tmp_path)
    dept = run.metrics["questions"]["department"]
    # Acting only on the most confident half should be visibly better than
    # acting on everything.
    assert dept["accuracy_top_50pct_by_confidence"] > dept["accuracy"]


def test_hard_targets_are_further_from_the_bayes_posterior(tmp_path):
    """The central ablation, as a regression test.

    Measured across three training budgets, swapping the soft Bayes target for
    the realised one-hot outcome makes the model move much further from the
    posterior, worse on the Brier score, and worse calibrated on the Noul head.

    The Choice *top-label* ECE is deliberately not asserted. On a small budget
    it can move either way, because the posterior-target run is itself underfit
    and underfit models are miscalibrated in their own way. That is a real
    finding, and it is reported rather than hidden.
    """

    posterior = _quick_run(tmp_path, target_mode="posterior")
    hard = _quick_run(tmp_path, target_mode="hard")
    p = posterior.metrics["questions"]
    h = hard.metrics["questions"]
    assert h["department"]["mean_posterior_l1"] > 5 * p["department"]["mean_posterior_l1"]
    assert h["department"]["brier"] > p["department"]["brier"]
    assert h["severity"]["brier"] > p["severity"]["brier"]
    assert h["is_urgent"]["ece"] > p["is_urgent"]["ece"]


def test_distribution_shift_increases_miscalibration(tmp_path):
    config = TrainConfig(
        n_train=1500, n_eval=600, epochs=8, batch_size=128, dim=32, seed=0
    )
    in_domain = train(config, verbose=False)
    shifted = train(
        config,
        eval_task=NoisyTriageTask(cue_signal=0.07, cue_confuse=0.05),
        verbose=False,
    )
    base = in_domain.metrics["questions"]["department"]["top_label_ece"]
    shift = shifted.metrics["questions"]["department"]["top_label_ece"]
    assert shift > base


# --- post-hoc temperature scaling -----------------------------------------


def _temperature_by_question(**overrides) -> dict:
    from minijev.experiments import temperature_rows

    # batch_size=256 keeps the "one epoch" case genuinely under-trained: it is
    # about six optimiser steps, which is what makes the model over-confident.
    # A smaller batch size trains enough to be calibrated in one epoch, and the
    # effect disappears.
    config = TrainConfig(
        n_train=1500, n_eval=800, epochs=8, batch_size=256, dim=48, seed=0
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    run = train(config, verbose=False)
    return {row["question"]: row for row in temperature_rows(run)}


def test_temperature_scaling_is_close_to_a_no_op_on_a_converged_model():
    """A model that already matches the posterior should not want rescaling."""

    rows = _temperature_by_question()
    assert 0.8 < rows["department"]["temperature"] < 1.3
    assert rows["department"]["ece_after"] < 0.06


def test_temperature_scaling_repairs_an_overconfident_model():
    """A badly underfit model is over-confident, and that part is fixable."""

    rows = _temperature_by_question(epochs=1)
    department = rows["department"]
    assert department["temperature"] > 1.3
    assert department["ece_after"] < department["ece_before"] / 2
    assert department["nll_after"] < department["nll_before"]


def test_temperature_scaling_cannot_fix_a_wrong_distribution():
    """Training on realised outcomes makes the shape wrong, not the scale.

    The fitted temperature stays near 1 (there is no global over-confidence to
    remove) and the distance to the Bayes posterior is left where it was. This is
    the boundary of what post-hoc scaling can do, and it is the reason the
    repository ships both this module and the ablations.
    """

    rows = _temperature_by_question(target_mode="hard")
    department = rows["department"]
    assert 0.7 < department["temperature"] < 1.4
    assert department["l1_after"] > 0.8 * department["l1_before"]
    assert department["l1_after"] > 0.1
