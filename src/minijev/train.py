"""Train and evaluate mini-jev on the synthetic triage task.

The training target is the exact Bayes posterior, so the loss is a proper scoring
rule against truth. Evaluation then reports two independent things:

1. Empirical calibration: does a stated 0.8 come true about 80% of the time?
   (top-label ECE, classwise ECE, Brier, NLL, on held-out samples)
2. Distance to the optimum: how far are the model's probabilities from the known
   posterior? (mean L1, and excess NLL over the posterior's entropy)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from . import calibration as cal
from .model import Adam, Item, MiniJevModel
from .runtime import MiniSystemOne, QuestionPlan, plan_questions
from .tasks import NoisyTriageTask, Sample
from .text import Vocab


@dataclass
class TrainConfig:
    n_train: int = 4000
    n_eval: int = 2000
    epochs: int = 12
    batch_size: int = 256
    lr: float = 0.05
    dim: int = 48
    seed: int = 0
    unk_buckets: int = 16
    target_mode: str = "posterior"  # "posterior" | "hard"

    def __post_init__(self) -> None:
        if self.target_mode not in ("posterior", "hard"):
            raise ValueError("target_mode must be 'posterior' or 'hard'")


@dataclass
class Fold:
    """Everything needed to train or score one split."""

    vocab: Vocab
    plans: dict[str, QuestionPlan]
    samples: list[Sample]
    items: list[Item]
    state_bags: np.ndarray


@dataclass
class RunResult:
    """Everything a caller needs to inspect or persist a trained run."""

    mini: MiniSystemOne
    metrics: dict
    eval_fold: Fold
    raw: dict[str, np.ndarray]


def _target_for(
    plan: QuestionPlan, sample: Sample, mode: str
) -> np.ndarray | float:
    """Soft (Bayes posterior) or hard (realised outcome) training target."""

    if mode == "posterior":
        return sample.posteriors[plan.qid]
    if plan.kind == "noul":
        return float(sample.labels[plan.qid])
    n = len(plan.option_names or [])
    onehot = np.zeros(n, dtype=np.float64)
    onehot[sample.labels[plan.qid]] = 1.0
    return onehot


def build_fold(
    task: NoisyTriageTask,
    vocab: Vocab,
    questions: dict,
    n: int,
    rng: np.random.Generator,
    target_mode: str = "posterior",
) -> Fold:
    plans = plan_questions(vocab, questions)
    samples: list[Sample] = []
    items: list[Item] = []
    bags: list[np.ndarray] = []
    for _ in range(n):
        sample = task.sample(rng)
        state_bag = vocab.bag(sample.tokens)
        bags.append(state_bag)
        samples.append(sample)
        for qid, plan in plans.items():
            items.append(
                Item(
                    kind=plan.kind,
                    state_bag=state_bag,
                    instr_bag=plan.instr_bag,
                    option_bags=plan.option_bags,
                    target=_target_for(plan, sample, target_mode),
                )
            )
    return Fold(vocab, plans, samples, items, np.stack(bags, axis=0))


def train(
    config: TrainConfig | None = None,
    task: NoisyTriageTask | None = None,
    eval_task: NoisyTriageTask | None = None,
    verbose: bool = True,
) -> RunResult:
    """Train, then evaluate.

    ``eval_task`` lets you evaluate on a *different* generator, which is how the
    distribution-shift ablation induces miscalibration without touching the
    training code.
    """

    config = config or TrainConfig()
    task = task or NoisyTriageTask()
    rng = np.random.default_rng(config.seed)

    questions = task.questions()
    vocab = Vocab(task.vocab_words, n_unk_buckets=config.unk_buckets)
    train_fold = build_fold(
        task, vocab, questions, config.n_train, rng, config.target_mode
    )
    eval_fold = build_fold(
        eval_task or task, vocab, questions, config.n_eval, rng, "posterior"
    )

    model = MiniJevModel(vocab.size, dim=config.dim, seed=config.seed)
    optimizer = Adam(model, lr=config.lr)

    history: list[dict] = []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(train_fold.items))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            batch = [train_fold.items[i] for i in idx]
            loss, grads = model.loss_and_grads(batch)
            optimizer.step(grads)
            epoch_loss += loss
            n_batches += 1
        mean_loss = epoch_loss / max(1, n_batches)
        history.append({"epoch": epoch, "train_nll": mean_loss})
        if verbose:
            print(f"epoch {epoch:3d}/{config.epochs}  train nll {mean_loss:.4f}")
    elapsed = time.time() - started

    mini = MiniSystemOne(model, vocab)
    raw = predict_fold(mini, eval_fold)
    metrics = evaluate(mini, eval_fold, raw=raw)
    metrics["train"] = {
        "epochs": config.epochs,
        "seconds": round(elapsed, 2),
        "n_train": config.n_train,
        "n_eval": config.n_eval,
        "parameters": model.n_parameters(),
        "target_mode": config.target_mode,
        "distribution_shift": eval_task is not None,
        "history": history,
    }
    metrics["config"] = asdict(config)
    return RunResult(mini=mini, metrics=metrics, eval_fold=eval_fold, raw=raw)


def predict_fold(
    mini: MiniSystemOne, fold: Fold
) -> dict[str, np.ndarray]:
    """Per-question predictions: a (n, k) matrix, or (n,) for Noul."""

    raw: dict[str, list] = {qid: [] for qid in fold.plans}
    for bag in fold.state_bags:
        result = mini.answer_state_bag(bag, fold.plans)
        for qid, plan in fold.plans.items():
            answer = result.answers[qid]
            if plan.kind == "noul":
                raw[qid].append(float(answer.noul))
            else:
                raw[qid].append(
                    [answer.probabilities[name] for name in plan.option_names]
                )
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def evaluate(
    mini: MiniSystemOne,
    fold: Fold,
    n_bins: int = 10,
    raw: dict[str, np.ndarray] | None = None,
) -> dict:
    if raw is None:
        raw = predict_fold(mini, fold)
    metrics: dict = {"questions": {}}
    nu = 1e-12

    # --- department (Choice) -------------------------------------------
    qid = "department"
    P = raw[qid]
    labels = np.array([s.labels[qid] for s in fold.samples])
    posterior = np.stack([np.asarray(s.posteriors[qid]) for s in fold.samples])
    metrics["questions"][qid] = {
        "type": "choice",
        "n": int(len(labels)),
        "accuracy": float((P.argmax(1) == labels).mean()),
        "top_label_ece": cal.top_label_ece(P, labels, n_bins),
        "classwise_ece": cal.classwise_ece(P, labels, n_bins),
        "brier": cal.brier_multiclass(P, labels),
        "nll": cal.nll_multiclass(P, labels),
        "mean_posterior_l1": float(np.mean(np.abs(P - posterior).sum(1))),
        "bayes_accuracy": float(
            (posterior.argmax(1) == labels).mean()
        ),
        "excess_nll_over_bayes": float(
            cal.nll_multiclass(P, labels)
            - cal.nll_multiclass(np.clip(posterior, nu, 1.0), labels)
        ),
        "mean_confidence": float(P.max(1).mean()),
        "accuracy_top_50pct_by_confidence": cal.selective_accuracy(
            P.max(1), (P.argmax(1) == labels).astype(float), 0.5
        ),
        "accuracy_top_80pct_by_confidence": cal.selective_accuracy(
            P.max(1), (P.argmax(1) == labels).astype(float), 0.8
        ),
    }

    # --- is_urgent (Noul) ----------------------------------------------
    qid = "is_urgent"
    p = np.asarray(raw[qid], dtype=np.float64)
    y = np.array([s.labels[qid] for s in fold.samples], dtype=np.float64)
    q_posterior = np.array([float(s.posteriors[qid]) for s in fold.samples])
    metrics["questions"][qid] = {
        "type": "noul",
        "n": int(len(y)),
        "accuracy": float(((p > 0.5).astype(float) == y).mean()),
        "ece": cal.expected_calibration_error(p, y, n_bins),
        "mce": cal.maximum_calibration_error(p, y, n_bins),
        "brier": cal.brier_binary(p, y),
        "brier_vs_posterior": cal.brier_binary(p, q_posterior),
        "mean_posterior_l1": float(np.mean(np.abs(p - q_posterior))),
        "mean_confidence": float(p.mean()),
        "accuracy_top_50pct_by_confidence": cal.selective_accuracy(
            np.maximum(p, 1.0 - p), ((p > 0.5) == (y > 0.5)).astype(float), 0.5
        ),
        "accuracy_top_80pct_by_confidence": cal.selective_accuracy(
            np.maximum(p, 1.0 - p), ((p > 0.5) == (y > 0.5)).astype(float), 0.8
        ),
    }

    # --- severity (Score) ----------------------------------------------
    qid = "severity"
    S = raw[qid]
    s_labels = np.array([s.labels[qid] for s in fold.samples])
    s_posterior = np.stack([np.asarray(s.posteriors[qid]) for s in fold.samples])
    levels = np.arange(S.shape[1], dtype=np.float64)
    metrics["questions"][qid] = {
        "type": "score",
        "n": int(len(s_labels)),
        "argmax_accuracy": float((S.argmax(1) == s_labels).mean()),
        "top_label_ece": cal.top_label_ece(S, s_labels, n_bins),
        "classwise_ece": cal.classwise_ece(S, s_labels, n_bins),
        "brier": cal.brier_multiclass(S, s_labels),
        "nll": cal.nll_multiclass(S, s_labels),
        "mean_posterior_l1": float(np.mean(np.abs(S - s_posterior).sum(1))),
        "expected_score_mae_vs_posterior": float(
            np.mean(
                np.abs(S @ levels - s_posterior @ levels)
            )
        ),
        "excess_nll_over_bayes": float(
            cal.nll_multiclass(S, s_labels)
            - cal.nll_multiclass(np.clip(s_posterior, nu, 1.0), s_labels)
        ),
        "accuracy_top_50pct_by_confidence": cal.selective_accuracy(
            S.max(1), (S.argmax(1) == s_labels).astype(float), 0.5
        ),
    }
    return metrics


def _fmt(value) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if abs(value) < 1e-3 and value != 0.0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def report_markdown(
    metrics: dict,
    raw: dict[str, np.ndarray],
    fold: Fold,
    n_bins: int = 10,
) -> str:
    """A human-readable report. Numbers come from an actual run, never typed in."""

    lines: list[str] = []
    lines.append("# mini-jev calibration report\n")
    lines.append("Generated by `python -m minijev train`. Every number is measured.\n")
    train = metrics.get("train", {})
    lines.append("## Run\n")
    lines.append(f"* parameters: {train.get('parameters')}")
    lines.append(f"* train samples: {train.get('n_train')}   eval samples: {train.get('n_eval')}")
    lines.append(f"* epochs: {train.get('epochs')}   wall clock: {train.get('seconds')}s\n")

    lines.append("## Headline metrics\n")
    lines.append("| question | type | metric | value |")
    lines.append("| --- | --- | --- | --- |")
    for qid, qm in metrics["questions"].items():
        for key in (
            "accuracy",
            "argmax_accuracy",
            "top_label_ece",
            "classwise_ece",
            "ece",
            "mce",
            "brier",
            "nll",
            "mean_posterior_l1",
            "excess_nll_over_bayes",
            "expected_score_mae_vs_posterior",
            "mean_confidence",
            "accuracy_top_50pct_by_confidence",
            "accuracy_top_80pct_by_confidence",
        ):
            if key in qm:
                lines.append(f"| {qid} | {qm['type']} | {key} | {_fmt(qm[key])} |")
    lines.append("")

    lines.append("## Reliability: is 0.8 right 80% of the time?\n")
    for qid, plan in fold.plans.items():
        labels = np.array([s.labels[qid] for s in fold.samples])
        predictions = raw[qid]
        lines.append(f"### {qid} ({plan.kind})\n")
        if plan.kind == "noul":
            probs = predictions
            correct = np.array(
                [float(s.labels[qid]) for s in fold.samples], dtype=np.float64
            )
        else:
            probs = predictions.max(axis=1)
            correct = (predictions.argmax(axis=1) == labels).astype(np.float64)
        lines.append("```")
        lines.append(cal.reliability_diagram(probs, correct, n_bins=n_bins))
        lines.append("```\n")
    return "\n".join(lines)
