"""Controlled ablations: which mechanisms actually break calibration?

Each variant changes exactly one thing and re-measures. The point is that
"calibrated" is not a property you get for free, and it is not a property you
can claim from a single number -- you have to say *what you did* to the model.

Run ``python -m minijev experiments`` to regenerate the table with your own
numbers. Measured shape of the result:

* training on the Bayes posterior  -> distance to the posterior ~0;
* training on realised hard labels -> the same architecture, same seed, moves
  far from the posterior, over-confidently, and every proper score gets worse;
* training too briefly             -> underfit and badly miscalibrated;
* weaker evidence at eval time     -> overconfidence from distribution shift.

The first two variants share identical architecture, data, optimiser and seed.
Only the target changes, which is what makes the comparison a fair test of the
mechanism rather than of the tuning.

One honest caveat, found by running this at three training budgets: the Choice
*top-label ECE* is the least reliable of these signals. At a small budget the
posterior-target run is underfit, and an underfit model can happen to post a
lower top-label ECE than a hard-target run. Brier, NLL, distance to the
posterior and the Noul ECE are all monotone in the effect; the Choice ECE is
not. A single headline number would hide exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import calibration as cal
from .tasks import NoisyTriageTask
from .temperature import (
    TemperatureScaler,
    binary_nll,
    fit_temperature,
    fit_temperature_binary,
)
from .train import RunResult, TrainConfig, build_fold, predict_fold, train


@dataclass
class Variant:
    name: str
    description: str
    config: TrainConfig
    eval_task: NoisyTriageTask | None = None


def default_variants(quick: bool = False) -> list[Variant]:
    n_train, n_eval = (1500, 800) if quick else (4000, 2000)
    epochs = 6 if quick else 12
    base = dict(n_train=n_train, n_eval=n_eval, seed=0, dim=48, lr=0.05)
    return [
        Variant(
            name="posterior_targets",
            description="Train directly on the known Bayes posterior.",
            config=TrainConfig(epochs=epochs, **base),
        ),
        Variant(
            name="hard_targets",
            description=(
                "Identical run, but the target is the realised outcome "
                "(one-hot) instead of the posterior."
            ),
            config=TrainConfig(epochs=epochs, target_mode="hard", **base),
        ),
        Variant(
            name="underfit",
            description=(
                "Posterior targets, but a single pass over 1500 samples: the "
                "model never reaches the optimum, which is its own kind of "
                "miscalibration."
            ),
            config=TrainConfig(
                n_train=1500, n_eval=n_eval, epochs=1, seed=0, dim=48, lr=0.05
            ),
        ),
        Variant(
            name="evidence_shift",
            description=(
                "Trained on the default generator, evaluated where the cue "
                "signal is weaker, so the true posterior is flatter than the "
                "model was taught to expect."
            ),
            config=TrainConfig(epochs=epochs, **base),
            eval_task=NoisyTriageTask(cue_signal=0.075, cue_confuse=0.05),
        ),
    ]


def run_ablations(
    quick: bool = False,
    verbose: bool = False,
    on_done: Callable[[Variant, RunResult], None] | None = None,
) -> list[dict]:
    """Train every variant and collect one comparable summary row each."""

    rows: list[dict] = []
    for variant in default_variants(quick=quick):
        if verbose:
            print(f"\n=== {variant.name}: {variant.description}")
        result = train(variant.config, eval_task=variant.eval_task, verbose=verbose)
        q = result.metrics["questions"]
        rows.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "target_mode": variant.config.target_mode,
                "shifted_eval": variant.eval_task is not None,
                "department_accuracy": q["department"]["accuracy"],
                "department_ece": q["department"]["top_label_ece"],
                "department_brier": q["department"]["brier"],
                "department_posterior_l1": q["department"]["mean_posterior_l1"],
                "urgent_accuracy": q["is_urgent"]["accuracy"],
                "urgent_ece": q["is_urgent"]["ece"],
                "urgent_brier": q["is_urgent"]["brier"],
                "severity_ece": q["severity"]["top_label_ece"],
                "severity_brier": q["severity"]["brier"],
                "severity_mae_vs_posterior": q["severity"][
                    "expected_score_mae_vs_posterior"
                ],
            }
        )
        if on_done is not None:
            on_done(variant, result)
    return rows


def ablations_markdown(rows: list[dict]) -> str:
    lines = [
        "# mini-jev ablations: what breaks calibration?\n",
        "Every row below is a measured run. Only the listed mechanism changes.\n",
        "| variant | target | shifted eval | dept acc | dept ECE | dept Brier | "
        "dept L1 to Bayes | urgent acc | urgent ECE | severity ECE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {variant} | {target_mode} | {shifted_eval} | {department_accuracy:.3f} | "
            "{department_ece:.3f} | {department_brier:.3f} | "
            "{department_posterior_l1:.4f} | {urgent_accuracy:.3f} | "
            "{urgent_ece:.3f} | {severity_ece:.3f} |".format(**r)
        )
    lines.append("")
    lines.append("## What each variant isolates\n")
    for r in rows:
        lines.append(f"* **{r['variant']}** -- {r['description']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# temperature scaling
# ---------------------------------------------------------------------------


def temperature_rows(run: RunResult, n_bins: int = 10) -> list[dict]:
    """Fit a temperature on half a run's eval fold, then measure on the other half.

    The split matters. Fitting ``T`` on the same samples you report it on would
    guarantee an improvement on paper and mean nothing in practice, so the first
    half is used only for fitting and every number below is measured on the
    second half.
    """

    fold = run.eval_fold
    raw = run.raw
    n = len(fold.samples)
    half = n // 2
    if half < 10:
        raise ValueError("eval fold is too small to split for a temperature fit")

    rows: list[dict] = []
    for qid, plan in fold.plans.items():
        labels = np.array([s.labels[qid] for s in fold.samples])
        if plan.kind == "noul":
            posterior = np.array(
                [float(s.posteriors[qid]) for s in fold.samples], dtype=np.float64
            )
        else:
            posterior = np.stack(
                [np.asarray(s.posteriors[qid], dtype=np.float64) for s in fold.samples]
            )

        if plan.kind == "noul":
            p = np.asarray(raw[qid], dtype=np.float64)
            y = labels.astype(np.float64)
            temperature = fit_temperature_binary(p[:half], y[:half])
            before = p[half:]
            after = TemperatureScaler(temperature).transform_noul(before)

            def metrics(values: np.ndarray) -> dict:
                return {
                    "ece": cal.expected_calibration_error(values, y[half:], n_bins),
                    "brier": cal.brier_binary(values, y[half:]),
                    "nll": binary_nll(values, y[half:]),
                    "l1": float(np.mean(np.abs(values - posterior[half:]))),
                }
        else:
            P = np.asarray(raw[qid], dtype=np.float64)
            temperature = fit_temperature(P[:half], labels[:half])
            before = P[half:]
            after = TemperatureScaler(temperature).transform_probs(before)

            def metrics(values: np.ndarray) -> dict:
                return {
                    "ece": cal.top_label_ece(values, labels[half:], n_bins),
                    "brier": cal.brier_multiclass(values, labels[half:]),
                    "nll": cal.nll_multiclass(values, labels[half:]),
                    "l1": float(np.mean(np.abs(values - posterior[half:]).sum(1))),
                }

        m_before, m_after = metrics(before), metrics(after)
        rows.append(
            {
                "question": qid,
                "type": plan.kind,
                "temperature": float(temperature),
                **{f"{k}_before": v for k, v in m_before.items()},
                **{f"{k}_after": v for k, v in m_after.items()},
            }
        )
    return rows


def temperature_markdown(rows: list[dict], title: str | None = None) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}\n")
    lines += [
        "Temperature fitted on the first half of the held-out fold; every number "
        "below is measured on the second half.\n",
        "| question | type | T | ECE before | ECE after | Brier before | "
        "Brier after | NLL before | NLL after | L1 to Bayes before | L1 after |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {question} | {type} | {temperature:.3f} | {ece_before:.4f} | "
            "{ece_after:.4f} | {brier_before:.4f} | {brier_after:.4f} | "
            "{nll_before:.4f} | {nll_after:.4f} | {l1_before:.4f} | "
            "{l1_after:.4f} |".format(**r)
        )
    lines.append("")
    return "\n".join(lines)


def run_experiments(quick: bool = False, verbose: bool = False) -> dict:
    """Train every variant once, then derive both report tables from that run."""

    runs: dict[str, RunResult] = {}
    rows = run_ablations(
        quick=quick,
        verbose=verbose,
        on_done=lambda variant, result: runs.__setitem__(variant.name, result),
    )
    temperature: list[dict] = []
    for name, result in runs.items():
        for row in temperature_rows(result):
            temperature.append({"variant": name, **row})
    return {"ablations": rows, "temperature": temperature}


def temperature_report_for_model(
    mini,
    task: NoisyTriageTask | None = None,
    n: int = 2000,
    seed: int = 1234,
    n_bins: int = 10,
) -> list[dict]:
    """Fit temperatures for an already-trained model on a fresh held-out fold.

    Used by ``python -m minijev temperature``. Generates new samples rather than
    reusing a training split, so the fold is genuinely unseen.
    """

    from .runtime import MiniSystemOne  # local import keeps the module import-light

    if not isinstance(mini, MiniSystemOne):
        raise TypeError("mini must be a MiniSystemOne")
    task = task or NoisyTriageTask()
    fold = build_fold(
        task, mini.vocab, task.questions(), n, np.random.default_rng(seed), "posterior"
    )
    run = RunResult(
        mini=mini, metrics={}, eval_fold=fold, raw=predict_fold(mini, fold)
    )
    return temperature_rows(run, n_bins=n_bins)
