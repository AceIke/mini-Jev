"""Command line entry points.

    python -m minijev train       # fit a model, write runs/<name>/report.md
    python -m minijev demo        # ask a trained model real questions
    python -m minijev experiments # the calibration ablations
    python -m minijev temperature # how much of the gap post-hoc scaling can fix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="minijev",
        description="A tiny, educational System One decision model.",
    )
    parser.add_argument(
        "--version", action="version", version=f"minijev {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="train on the synthetic triage task")
    p.add_argument("--out", default="runs/triage", help="output directory")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target-mode",
        choices=("posterior", "hard"),
        default="posterior",
        help="train against the Bayes posterior, or against realised outcomes",
    )
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("demo", help="ask a trained model a few hand-written tickets")
    p.add_argument("--model", default="runs/triage")
    p.set_defaults(func=_cmd_demo)

    p = sub.add_parser("experiments", help="run the calibration ablations")
    p.add_argument("--out", default="runs/experiments")
    p.add_argument("--quick", action="store_true", help="smaller, faster run")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=_cmd_experiments)

    p = sub.add_parser(
        "temperature",
        help="fit post-hoc temperature scaling on a fresh held-out fold",
    )
    p.add_argument("--model", default="runs/triage")
    p.add_argument("--n", type=int, default=2000, help="held-out samples")
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=_cmd_temperature)

    return parser


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import TrainConfig, report_markdown, train

    config = TrainConfig(
        n_train=args.n_train,
        n_eval=args.n_eval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dim=args.dim,
        seed=args.seed,
        target_mode=args.target_mode,
    )
    run = train(config, verbose=not args.quiet)
    out = Path(args.out)
    run.mini.save(out)
    (out / "metrics.json").write_text(
        json.dumps(run.metrics, indent=2), encoding="utf-8"
    )
    (out / "report.md").write_text(
        report_markdown(run.metrics, run.raw, run.eval_fold), encoding="utf-8"
    )

    train_info = run.metrics["train"]
    print()
    print(f"parameters      {train_info['parameters']}")
    print(f"wall clock      {train_info['seconds']}s")
    for qid, qm in run.metrics["questions"].items():
        if qm["type"] == "noul":
            print(
                f"{qid:<12} noul  acc {qm['accuracy']:.3f}  ece {qm['ece']:.3f}  "
                f"brier {qm['brier']:.3f}"
            )
        else:
            acc = qm.get("accuracy", qm.get("argmax_accuracy"))
            print(
                f"{qid:<12} {qm['type']:<5} acc {acc:.3f}  "
                f"ece {qm['top_label_ece']:.3f}  brier {qm['brier']:.3f}"
            )
    print(f"\nwrote {out / 'report.md'} and {out / 'metrics.json'}")
    return 0


DEMO_TICKETS = [
    "my card was charged twice for the same order, please refund asap",
    "the integration keeps returning a timeout error and is blocking our launch",
    "could i get a pricing quote before we upgrade the account?",
    "tiny cosmetic typo in the invoice header, no rush at all",
]


def _cmd_demo(args: argparse.Namespace) -> int:
    from .runtime import MiniSystemOne
    from .tasks import NoisyTriageTask

    try:
        mini = MiniSystemOne.load(args.model)
    except FileNotFoundError:
        print(
            f"no trained model at {args.model!r}. Run `python -m minijev train` first.",
            file=sys.stderr,
        )
        return 2

    questions = NoisyTriageTask().questions()
    print(f"model: {mini.model_id}   parameters: {mini.model.n_parameters()}\n")
    header = f"{'state':<52} {'department':<10} {'conf':>5}  {'urgent':>6}  {'severity':>8}"
    print(header)
    print("-" * len(header))
    for ticket in DEMO_TICKETS:
        result = mini.system_one(state=ticket, questions=questions)
        state = ticket if len(ticket) <= 50 else ticket[:47] + "..."
        print(
            f"{state:<52} {result.choice('department'):<10} "
            f"{result.confidence('department'):>5.2f}  "
            f"{result.noul('is_urgent'):>6.3f}  {result.score('severity'):>8.3f}"
        )
    print(
        "\nNote: this model only knows the synthetic vocabulary it was trained on."
        "\nUnseen words hash into unknown buckets, so treat these outputs as a"
        "\ndemonstration of the pipeline, not as real triage advice."
    )
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    from .experiments import ablations_markdown, run_experiments, temperature_markdown

    report = run_experiments(quick=args.quick, verbose=args.verbose)
    ablation_md = ablations_markdown(report["ablations"])
    temperature_md = temperature_markdown(
        report["temperature"], title="Does post-hoc scaling fix the gap?"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(ablation_md + "\n" + temperature_md, encoding="utf-8")
    (out / "ablations.json").write_text(
        json.dumps(report["ablations"], indent=2), encoding="utf-8"
    )
    (out / "temperature.json").write_text(
        json.dumps(report["temperature"], indent=2), encoding="utf-8"
    )
    print(ablation_md)
    print(temperature_md)
    print(f"wrote {out / 'report.md'}")
    return 0


def _cmd_temperature(args: argparse.Namespace) -> int:
    from .experiments import temperature_markdown, temperature_report_for_model
    from .runtime import MiniSystemOne

    try:
        mini = MiniSystemOne.load(args.model)
    except FileNotFoundError:
        print(
            f"no trained model at {args.model!r}. Run `python -m minijev train` first.",
            file=sys.stderr,
        )
        return 2
    rows = temperature_report_for_model(mini, n=args.n, seed=args.seed)
    print(temperature_markdown(rows, title=f"temperature scaling for {args.model}"))
    print(
        "Reading this: T above 1 means the model was over-confident and T below 1 "
        "means it was under-confident. ECE, Brier and NLL are the metrics "
        "post-hoc scaling can move. When T lands near 1 there was nothing to "
        "correct and any movement in those columns is noise from the split, "
        "which is why T is fitted on one half of the fold and reported on the "
        "other. Rescaling is monotone, so it can never change an answer, only "
        "the confidence attached to it."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
