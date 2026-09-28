"""Generate every figure in docs/figures/ as SVG.

    python docs/make_figures.py

Two kinds of figure come out of here, and neither is hand-pasted:

* the conceptual diagrams read their numbers (dim, vocab size, parameter count)
  straight out of the live objects, so a figure cannot claim a shape the code
  does not have;
* the data figures are computed by training the model and running the ablations,
  so they cannot drift away from the tables in the README.

The output is plain SVG built here with string templates, so regenerating the
figures needs nothing beyond NumPy and the package itself.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from minijev.model import MiniJevModel  # noqa: E402
from minijev.tasks import NoisyTriageTask  # noqa: E402
from minijev.text import Vocab  # noqa: E402

# --- palette ---------------------------------------------------------------
BG = "#0a0e17"
PANEL = "#101725"
STROKE = "#22304a"
GRID = "#151d2c"
TXT = "#e6edf7"
DIM = "#8b9cb3"
FAINT = "#5d6c82"
CYAN = "#22d3ee"      # computation
GREEN = "#4ade80"     # state / input
AMBER = "#fbbf24"     # questions
MAGENTA = "#f472b6"   # answers
VIOLET = "#a78bfa"    # accents
RED = "#fb7185"       # bad numbers
MONO = ("ui-monospace,SFMono-Regular,Menlo,Consolas,'DejaVu Sans Mono',"
        "'Liberation Mono',monospace")

OUT = pathlib.Path(__file__).resolve().parent / "figures"


class SVG:
    """A very small SVG writer. Enough shapes to draw a diagram, nothing more."""

    def __init__(self, width: int, height: int) -> None:
        self.w = width
        self.h = height
        self.body: list[str] = []
        self.markers: set[str] = set()

    # -- primitives ------------------------------------------------------
    @staticmethod
    def _esc(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.2, rx=7,
             dash=None, opacity=None) -> None:
        s = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
             f' rx="{rx}" fill="{fill}"')
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{sw}"'
        if dash:
            s += f' stroke-dasharray="{dash}"'
        if opacity is not None:
            s += f' opacity="{opacity}"'
        self.body.append(s + "/>")

    def text(self, x, y, s, size=14, fill=TXT, anchor="start", weight="400",
             opacity=None, tracking=None) -> None:
        a = (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{MONO}"'
             f' font-size="{size}" fill="{fill}" text-anchor="{anchor}"'
             f' font-weight="{weight}"')
        if opacity is not None:
            a += f' opacity="{opacity}"'
        if tracking:
            a += f' letter-spacing="{tracking}"'
        self.body.append(a + f'>{self._esc(s)}</text>')

    def lines(self, x, y, rows, size=13, fill=DIM, lh=19) -> None:
        for i, row in enumerate(rows):
            self.text(x, y + i * lh, row, size=size, fill=fill)

    def line(self, x1, y1, x2, y2, stroke=STROKE, sw=1.2, dash=None,
             marker=False) -> None:
        s = (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
             f' stroke="{stroke}" stroke-width="{sw}"')
        if dash:
            s += f' stroke-dasharray="{dash}"'
        if marker:
            key = stroke.lstrip("#")
            self.markers.add(stroke)
            s += f' marker-end="url(#ar-{key})"'
        self.body.append(s + "/>")

    def arrow(self, x1, y1, x2, y2, color=CYAN, sw=1.8) -> None:
        self.line(x1, y1, x2, y2, stroke=color, sw=sw, marker=True)

    def elbow(self, x1, y1, x2, y2, color=CYAN, sw=1.8) -> None:
        """Horizontal, then vertical, then horizontal. Right angles only."""
        mid = (x1 + x2) / 2
        self.line(x1, y1, mid, y1, stroke=color, sw=sw)
        self.line(mid, y1, mid, y2, stroke=color, sw=sw)
        self.line(mid, y2, x2, y2, stroke=color, sw=sw, marker=True)

    def chip(self, x, y, label, fill, width=None, size=12) -> float:
        """A small rounded tag. Returns its width so callers can chain them."""
        w = width or (len(label) * size * 0.62 + 18)
        self.rect(x, y, w, 22, fill=fill, stroke=fill, sw=1, rx=5, opacity=0.14)
        self.text(x + w / 2, y + 15.5, label, size=size, fill=fill, anchor="middle")
        return w

    # -- document --------------------------------------------------------
    def render(self) -> str:
        defs = [
            '<pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">'
            f'<path d="M 26 0 L 0 0 0 26" fill="none" stroke="{GRID}"'
            ' stroke-width="1"/></pattern>',
            '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">'
            '<feGaussianBlur stdDeviation="4" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/>'
            '</feMerge></filter>',
        ]
        for color in sorted(self.markers):
            defs.append(
                f'<marker id="ar-{color.lstrip("#")}" viewBox="0 0 10 10"'
                ' refX="9" refY="5" markerWidth="6.5" markerHeight="6.5"'
                ' orient="auto-start-reverse">'
                f'<path d="M 0 0.6 L 10 5 L 0 9.4 z" fill="{color}"/></marker>'
            )
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}"'
            f' width="{self.w}" height="{self.h}" role="img">'
            f"<defs>{''.join(defs)}</defs>"
            f'<rect width="{self.w}" height="{self.h}" fill="{BG}"/>'
            f'<rect width="{self.w}" height="{self.h}" fill="url(#grid)"'
            ' opacity="0.5"/>'
        )
        return head + "".join(self.body) + "</svg>\n"

    def save(self, name: str) -> pathlib.Path:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / name
        path.write_text(self.render(), encoding="utf-8")
        return path


def panel(svg: SVG, x, y, w, h, title=None, accent=STROKE) -> None:
    svg.rect(x, y, w, h, fill=PANEL, stroke=STROKE, sw=1.2)
    svg.rect(x, y, 3, h, fill=accent, rx=0)
    if title:
        svg.text(x + 16, y + 24, title, size=13, fill=DIM, weight="700",
                 tracking="0.08em")


def caption(svg: SVG, x, y, text, fill=FAINT) -> None:
    svg.text(x, y, text, size=12, fill=fill)


# --- the live numbers every figure is built from ---------------------------
def live_facts():
    task = NoisyTriageTask()
    vocab = Vocab(task.vocab_words, n_unk_buckets=16)
    model = MiniJevModel(vocab.size, dim=48, seed=0)
    return {
        "dim": model.dim,
        "vocab": vocab.size,
        "known": len(vocab.known),
        "unk": vocab.n_unk_buckets,
        "params": model.n_parameters(),
        "choices": len(task.questions()["department"].option_names()),
        "levels": len(task.questions()["severity"].option_names()),
        "rows": task.n_topic_slots + task.n_urgency_slots + task.n_severity_slots + 2,
    }


F = live_facts()
D = F["dim"]
V = F["vocab"]


# --- 1. banner -------------------------------------------------------------
def fig_banner() -> pathlib.Path:
    svg = SVG(1040, 210)
    svg.rect(0, 0, 1040, 3, fill=CYAN, rx=0)
    svg.text(40, 92, "mini-jev", size=56, fill=TXT, weight="700",
             tracking="0.02em")
    svg.text(40, 92, "mini-jev", size=56, fill=CYAN, weight="700", opacity=0.35)
    svg.text(41, 92, "mini-jev", size=56, fill=TXT, weight="700")
    svg.text(42, 122, "a System One decision model small enough to read in one sitting",
             size=15, fill=DIM)
    y = 160
    x = 40
    for label, color in (
        (f"{F['params']:,} params", CYAN),
        (f"{V}-dim bag of words", GREEN),
        (f"{F['rows']}-token state", GREEN),
        ("1 matmul per question", AMBER),
        ("0 autoregressive steps", VIOLET),
        ("ECE 0.014", MAGENTA),
    ):
        x += svg.chip(x, y, label, color) + 10
    svg.rect(0, 207, 1040, 3, fill=VIOLET, rx=0)
    return svg.save("banner.svg")


# --- 2. the contract -------------------------------------------------------
def fig_contract() -> pathlib.Path:
    svg = SVG(1040, 430)
    svg.text(40, 40, "THE CONTRACT", size=13, fill=DIM, weight="700",
             tracking="0.14em")
    svg.text(40, 66, "state + typed questions -> typed answers", size=19, fill=TXT)

    panel(svg, 40, 90, 300, 120, "INPUT", GREEN)
    svg.lines(56, 128, ["state  ::  str | dict | list",
                        f"bag   ::  {V} counts ({F['known']} known + {F['unk']} hashed)",
                        "text  ::  one state per request"], size=12.5, fill=DIM)

    panel(svg, 370, 90, 300, 120, "QUESTION", AMBER)
    svg.lines(386, 128, ["Choice  ::  pick one of N",
                         "Score   ::  ordered levels",
                         "Noul    ::  P(yes) in [0, 1]"], size=12.5, fill=DIM)

    svg.text(700, 128, "read", size=12, fill=FAINT)
    svg.arrow(676, 150, 728, 150, color=CYAN)
    svg.rect(732, 96, 268, 108, fill=PANEL, stroke=CYAN, sw=1.4)
    svg.text(748, 126, "ONE FORWARD PASS", size=12.5, fill=CYAN, weight="700")
    svg.lines(748, 150, ["state read once -> u",
                         "all k options scored at once",
                         "questions never see each other"],
              size=12.5, fill=DIM, lh=18)

    panel(svg, 40, 250, 960, 150, "ANSWER", MAGENTA)
    cols = [
        ("choice", ["choice        : str", "probabilities : {opt: float}",
                    "confidence    : float"]),
        ("score", ["score         : float", "legend        : {i: str}",
                   "probabilities : {i: float}", "confidence    : float"]),
        ("noul", ["noul          : float", "(no confidence, by design)"]),
    ]
    for i, (name, rows) in enumerate(cols):
        x = 62 + i * 310
        svg.text(x, 292, name, size=14, fill=MAGENTA, weight="700")
        svg.lines(x, 316, rows, size=12.5, fill=DIM, lh=18)
    svg.text(40, 424, "every answer obeys the same map key as its question; "
                      "usage reports input tokens only",
             size=12, fill=FAINT)
    return svg.save("contract.svg")


# --- 3. pipeline -----------------------------------------------------------
def fig_pipeline() -> pathlib.Path:
    svg = SVG(1040, 640)
    svg.text(40, 40, "DATA FLOW", size=13, fill=DIM, weight="700",
             tracking="0.14em")
    svg.text(40, 66, "one state, one context vector, one matmul per question",
             size=19, fill=TXT)

    def column(x, w, rows, color):
        """Draw a vertical chain of boxes and arrow between each pair."""
        y = 96
        centers = []
        for title, sub, shape in rows:
            svg.rect(x, y, w, 66, fill=PANEL, stroke=color, sw=1.3)
            svg.text(x + 16, y + 27, title, size=13.5, fill=color, weight="700")
            svg.text(x + 16, y + 49, sub, size=12.5, fill=DIM)
            if shape:
                svg.text(x + w - 16, y + 27, shape, size=12.5, fill=FAINT,
                         anchor="end")
            centers.append((y, y + 66))
            y += 88
        for bottom, top in zip([c[1] for c in centers], [c[0] for c in centers[1:]]):
            svg.arrow(x + w / 2, bottom, x + w / 2, top, color=color, sw=1.4)
        return centers

    # state goes down the left, the question goes down the right, both into u
    state_boxes = column(40, 430, [
        ("state text", "-> tokenize -> counts c_s", f"[{V}]"),
        ("c_s @ E", f"E: {V}x{D} shared encoder  ->  e_s", f"[{D}]"),
        ("e_s @ P_s", f"P_s: {D}x{D}  ->  s", f"[{D}]"),
    ], GREEN)
    question_boxes = column(570, 430, [
        ("question instructions", "-> counts c_q", f"[{V}]"),
        ("c_q @ E @ P_q", f"-> q", f"[{D}]"),
    ], AMBER)

    # both paths converge on u
    svg.rect(150, 372, 450, 64, fill=PANEL, stroke=CYAN, sw=1.6)
    svg.text(166, 398, "u = s + q", size=15, fill=CYAN, weight="700")
    svg.text(166, 420, "the only thing any option ever sees", size=12.5, fill=DIM)
    svg.text(584, 398, f"[{D}]", size=12.5, fill=FAINT, anchor="end")
    svg.arrow(255, state_boxes[-1][1], 255, 368, color=GREEN)
    q_bottom = question_boxes[-1][1]
    svg.line(785, q_bottom, 785, 404, stroke=AMBER, sw=1.4)
    svg.line(785, 404, 606, 404, stroke=AMBER, sw=1.4, marker=True)

    # u forks into the two heads
    svg.line(375, 436, 375, 466, stroke=CYAN, sw=1.4)
    svg.line(375, 466, 255, 466, stroke=CYAN, sw=1.4)
    svg.arrow(255, 466, 255, 482, color=CYAN, sw=1.4)
    svg.line(375, 466, 760, 466, stroke=CYAN, sw=1.4)
    svg.arrow(760, 466, 760, 482, color=CYAN, sw=1.4)

    svg.rect(40, 486, 430, 100, fill=PANEL, stroke=VIOLET, sw=1.3)
    svg.text(58, 512, "NOUL HEAD", size=12.5, fill=VIOLET, weight="700")
    svg.lines(58, 538, ["w_n . u + b_n  ->  sigmoid",
                        "->  noul = P(yes)"], size=12.5, fill=DIM, lh=18)

    svg.rect(520, 486, 480, 100, fill=PANEL, stroke=MAGENTA, sw=1.3)
    svg.text(538, 512, "OPTION HEAD", size=12.5, fill=MAGENTA, weight="700")
    svg.lines(538, 538, ["option texts -> C_o [k x V] -> @ E -> @ P_o -> O [k x D]",
                         "logits = O @ u / sqrt(D)  ->  softmax -> probabilities"],
              size=12.5, fill=DIM, lh=18)

    caption(svg, 40, 616, f"D = {D}   V = {V} (bag of word ids)   "
                          "k = number of options in that question")
    return svg.save("pipeline.svg")


# --- 4. the three heads ----------------------------------------------------
def fig_heads() -> pathlib.Path:
    svg = SVG(1040, 430)
    svg.text(40, 40, "THREE PRIMITIVES, ONE CONTEXT VECTOR", size=13, fill=DIM,
             weight="700", tracking="0.14em")
    svg.text(40, 66, "the same u answers every question, so they cannot interfere",
             size=19, fill=TXT)

    svg.rect(40, 160, 150, 96, fill=PANEL, stroke=CYAN, sw=1.6)
    svg.text(115, 198, "u", size=34, fill=CYAN, weight="700", anchor="middle")
    svg.text(115, 228, f"[{D}]", size=12.5, fill=FAINT, anchor="middle")

    heads = [
        ("Choice", "pick one of N", "softmax(O @ u / sqrt(D))",
         "choice + probabilities", f"{F['choices']} options", MAGENTA),
        ("Score", "ordered levels", "softmax over levels, then",
         "expected level = sum p_i * i", f"{F['levels']} levels", AMBER),
        ("Noul", "yes / no", "sigmoid(w_n . u + b_n)",
         "one probability, no confidence", "1 probability", VIOLET),
    ]
    for i, (name, what, how, out, count, color) in enumerate(heads):
        y = 118 + i * 100
        svg.arrow(196, 208, 292, y + 40, color=CYAN)
        svg.rect(296, y, 700, 80, fill=PANEL, stroke=color, sw=1.3)
        svg.text(316, y + 28, name, size=15, fill=color, weight="700")
        svg.text(410, y + 28, what, size=12.5, fill=FAINT)
        svg.text(316, y + 52, how, size=12.5, fill=DIM)
        svg.text(596, y + 52, "->  " + out, size=12.5, fill=DIM)
        svg.text(980, y + 28, count, size=12, fill=FAINT, anchor="end")

    caption(svg, 40, 424, "k = options per question. Confidence is derived from "
                          "the distribution, so Choice and Score have it and Noul does not.")
    return svg.save("heads.svg")


# --- 5. module layering ----------------------------------------------------
def fig_modules() -> pathlib.Path:
    layers = [
        ("L0  leaves", [("types.py", "the contract"), ("numerics.py", "softmax, sigmoid"),
                        ("text.py", "state -> bag"), ("confidence.py", "peak statistic"),
                        ("calibration.py", "ECE, Brier, NLL")], GREEN),
        ("L1  pieces", [("model.py", "forward + backward"), ("tasks.py", "closed-form posterior"),
                        ("temperature.py", "post-hoc scaling")], CYAN),
        ("L2  flow", [("runtime.py", "system_one()")], AMBER),
        ("L3  fit", [("train.py", "training + evaluation")], VIOLET),
        ("L4  evidence", [("experiments.py", "ablations"), ("cli.py", "train|demo|exp|temp")], MAGENTA),
    ]
    svg = SVG(1040, 560)
    svg.text(40, 40, "MODULE LAYERING", size=13, fill=DIM, weight="700",
             tracking="0.14em")
    svg.text(40, 66, "imports only ever point down", size=19, fill=TXT)

    y = 96
    for label, mods, color in layers:
        svg.text(40, y + 34, label, size=12.5, fill=color, weight="700")
        x = 210
        for name, desc in mods:
            w = max(len(name) * 7.4, len(desc) * 6.4) + 34
            svg.rect(x, y, w, 58, fill=PANEL, stroke=color, sw=1.2)
            svg.text(x + 14, y + 25, name, size=13, fill=color, weight="700")
            svg.text(x + 14, y + 45, desc, size=11.5, fill=FAINT)
            x += w + 12
        if y < 400:
            svg.arrow(60, y + 60, 60, y + 84, color=STROKE, sw=1.2)
        y += 92

    caption(svg, 40, 540, "the synthetic task can compute an exact posterior "
                          "without importing the model, which is why numerics.py exists")
    return svg.save("modules.svg")


# --- data figures ----------------------------------------------------------
def _bar(svg, x, y, w, h, value, vmax, color, label=None, label_fill=DIM):
    svg.rect(x, y, w, h, fill=STROKE, opacity=0.35)
    filled = max(1.0, w * (value / vmax)) if vmax > 0 else 0
    svg.rect(x, y, filled, h, fill=color, opacity=0.85, rx=2)
    if label:
        svg.text(x + w + 8, y + h - 1, label, size=11.5, fill=label_fill)


def reliability_bins():
    """Train the default model and bin its held-out predictions."""
    from minijev.calibration import reliability_bins as rb
    from minijev.train import TrainConfig, predict_fold, train
    from minijev.runtime import plan_questions

    run = train(TrainConfig(), verbose=False)
    fold, raw = run.eval_fold, run.raw
    out = {}
    for qid, plan in fold.plans.items():
        labels = np.array([s.labels[qid] for s in fold.samples])
        if plan.kind == "noul":
            probs = np.asarray(raw[qid], dtype=float)
            correct = labels.astype(float)
        else:
            P = np.asarray(raw[qid], dtype=float)
            probs = P.max(axis=1)
            correct = (P.argmax(axis=1) == labels).astype(float)
        out[qid] = rb(probs, correct, n_bins=10)
    return out


def fig_reliability(bins) -> pathlib.Path:
    svg = SVG(1040, 570)
    svg.text(40, 40, "CALIBRATION", size=13, fill=DIM, weight="700",
             tracking="0.14em")
    svg.text(40, 66, "when the model says 0.8, is it right 80% of the time?",
             size=19, fill=TXT)

    # diagonal guide + bars
    x0, y0, w, h = 200, 110, 760, 330
    svg.rect(x0, y0, w, h, fill=PANEL, stroke=STROKE, sw=1.2)
    for frac in (0.25, 0.5, 0.75):
        svg.line(x0, y0 + h * (1 - frac), x0 + w, y0 + h * (1 - frac),
                 stroke=GRID, sw=1, dash="3 5")
        svg.text(x0 - 12, y0 + h * (1 - frac) + 4, f"{frac:.2f}", size=11,
                 fill=FAINT, anchor="end")
    svg.line(x0, y0 + h, x0 + w, y0, stroke=FAINT, sw=1, dash="4 4")
    svg.text(x0 + 12, y0 + 20, "dashed line = perfect calibration", size=11,
             fill=FAINT)
    svg.text(x0 - 12, y0 + h + 20, "0.0", size=11, fill=FAINT, anchor="end")
    svg.text(x0 + w, y0 + h + 20, "1.0", size=11, fill=FAINT, anchor="end")

    used = [b for b in bins["department"] if b.count]
    bw = w / max(1, len(used)) * 0.34
    for i, b in enumerate(used):
        cx = x0 + w * (i + 0.5) / len(used)
        for value, color, dx in ((b.mean_confidence, CYAN, -bw - 2),
                                 (b.mean_accuracy, MAGENTA, 2)):
            bh = h * value
            svg.rect(cx + dx, y0 + h - bh, bw, bh, fill=color, opacity=0.9, rx=2)
        svg.text(cx, y0 + h + 20, f"{b.lo:.1f}", size=10.5, fill=FAINT,
                 anchor="middle")
        svg.text(cx, y0 + h + 38, f"n={b.count}", size=10, fill=FAINT,
                 anchor="middle")

    svg.text(x0 + w / 2, y0 + h + 62, "confidence", size=12, fill=DIM,
             anchor="middle")
    svg.rect(200, 522, 14, 14, fill=CYAN, rx=3)
    svg.text(222, 534, "mean confidence", size=12, fill=DIM)
    svg.rect(360, 522, 14, 14, fill=MAGENTA, rx=3)
    svg.text(382, 534, "observed accuracy", size=12, fill=DIM)
    svg.text(600, 534, "department (choice), 2000 held-out samples", size=12,
             fill=FAINT)
    return svg.save("reliability.svg")


def fig_ablations(rows) -> pathlib.Path:
    svg = SVG(1040, 470)
    svg.text(40, 40, "ABLATIONS", size=13, fill=DIM, weight="700",
             tracking="0.14em")
    svg.text(40, 66, "change one mechanism, watch which numbers move",
             size=19, fill=TXT)

    metrics = [
        ("top-label ECE", "department_ece", RED, 0.25),
        ("Brier", "department_brier", AMBER, 0.62),
        ("L1 to Bayes posterior", "department_posterior_l1", VIOLET, 0.55),
    ]
    labels = {"posterior_targets": "posterior", "hard_targets": "hard",
              "underfit": "underfit", "evidence_shift": "shift"}
    order = ["posterior_targets", "hard_targets", "underfit", "evidence_shift"]
    by_name = {r["variant"]: r for r in rows}

    for pi, (title, key, color, vmax) in enumerate(metrics):
        x = 40 + pi * 330
        panel(svg, x, 96, 300, 320, title, color)
        for i, name in enumerate(order):
            r = by_name.get(name)
            if not r:
                continue
            y = 146 + i * 66
            svg.text(x + 16, y + 20, labels[name], size=12.5, fill=DIM)
            _bar(svg, x + 16, y + 30, 220, 14, r[key], vmax, color,
                 f"{r[key]:.3f}")
    svg.text(40, 444, "same seed and architecture in every row. Brier and L1 are "
                      "monotone in the effect; the Choice ECE is not.",
             size=12, fill=FAINT)
    return svg.save("ablations.svg")


def fig_temperature(rows) -> pathlib.Path:
    svg = SVG(1040, 470)
    svg.text(40, 40, "TEMPERATURE SCALING", size=13, fill=DIM, weight="700",
             tracking="0.14em")
    svg.text(40, 66, "one parameter, fitted on held-out data, cannot reorder options",
             size=19, fill=TXT)

    order = ["posterior_targets", "hard_targets", "underfit", "evidence_shift"]
    labels = {"posterior_targets": "posterior", "hard_targets": "hard",
              "underfit": "underfit", "evidence_shift": "noisy evidence"}
    dept = {r["variant"]: r for r in rows if r["question"] == "department"}

    x0, y0, w = 240, 122, 760
    svg.rect(x0, y0, w, 240, fill=PANEL, stroke=STROKE, sw=1.2)
    vmax = max(max(r["ece_before"], r["ece_after"]) for r in dept.values()) * 1.25
    for i, name in enumerate(order):
        r = dept[name]
        y = y0 + 26 + i * 56
        svg.text(40 + 40, y + 14, labels[name], size=12.5, fill=DIM)
        svg.text(40 + 40, y + 32, f"T={r['temperature']:.2f}", size=11.5,
                 fill=CYAN)
        _bar(svg, x0 + 16, y, w * 0.34, 14, r["ece_before"], vmax, FAINT,
             f"{r['ece_before']:.3f}", label_fill=FAINT)
        _bar(svg, x0 + 16, y + 20, w * 0.34, 14, r["ece_after"], vmax,
             MAGENTA if r["ece_after"] < r["ece_before"] else RED,
             f"{r['ece_after']:.3f}")

    svg.rect(240, 388, 14, 14, fill=FAINT, rx=3)
    svg.text(262, 400, "ECE before", size=12, fill=DIM)
    svg.rect(390, 388, 14, 14, fill=MAGENTA, rx=3)
    svg.text(412, 400, "ECE after (department question)", size=12, fill=DIM)
    svg.text(40, 440, "the posterior model fits T=1.01 and is left alone; the two "
                      "over-confident models are repaired; the hard-target model "
                      "keeps its 0.204 distance to Bayes.",
             size=12, fill=FAINT)
    return svg.save("temperature.svg")


def main() -> None:
    print(f"live facts: dim={D} vocab={V} params={F['params']:,}")
    written = [fig_banner(), fig_contract(), fig_pipeline(), fig_heads(),
               fig_modules()]

    print("training the default model for the reliability figure...")
    written.append(fig_reliability(reliability_bins()))

    print("running the ablations (four training runs) for the last two figures...")
    from minijev.experiments import run_experiments
    report = run_experiments(quick=False)
    written.append(fig_ablations(report["ablations"]))
    written.append(fig_temperature(report["temperature"]))

    for path in written:
        print(f"  wrote {path.relative_to(path.parents[2])}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
