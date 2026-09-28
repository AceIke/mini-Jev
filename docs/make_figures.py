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
STROKE = "#2a3b58"
GRID = "#151d2c"
TXT = "#eef4fc"
DIM = "#a3b4cb"
FAINT = "#7b8ca4"
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
        self.clips: set[tuple[float, float, float, float]] = set()
        self.extra_defs: list[str] = []
        # every figure gets the same corner brackets and CRT lines unless it
        # draws its own
        self.framed = True

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
    def raw(self, markup: str) -> None:
        """Escape hatch for effects the helper methods do not cover."""

        self.body.append(markup)

    def defs(self, markup: str) -> None:
        """Register gradients, filters and clip paths for this figure."""

        self.extra_defs.append(markup)

    def scanlines(self, x, y, w, h, opacity=0.10) -> None:
        """Horizontal CRT lines over a region."""

        self.raw(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
            f' fill="url(#scan)" opacity="{opacity}"/>'
        )

    def hud(self, x, y, w, h, color=CYAN, size=22, sw=2.2) -> None:
        """Four corner brackets, the standard sci-fi frame."""

        for cx, sx in ((x, 1), (x + w, -1)):
            for cy, sy in ((y, 1), (y + h, -1)):
                self.raw(
                    f'<path d="M {cx:.1f} {cy + sy * size:.1f} '
                    f'L {cx:.1f} {cy:.1f} L {cx + sx * size:.1f} {cy:.1f}"'
                    f' fill="none" stroke="{color}" stroke-width="{sw}"/>'
                )

    def glow_text(self, x, y, s, size=14, fill=TXT, weight="400",
                  anchor="start", halo=0.45, dx=0.0, tracking=None) -> None:
        """Text with a soft halo behind it. Cheap glow without filters."""

        for off, op in ((2.0, 0.16), (1.0, 0.22)):
            self.text(x + dx + off, y + off, s, size=size, fill=fill,
                      anchor=anchor, weight=weight, opacity=op)
        self.text(x + dx, y, s, size=size, fill=fill, anchor=anchor,
                  weight=weight, tracking=tracking)

    def glitch(self, x, y, s, size=34, fill=TXT, weight="700") -> None:
        """Chromatic-aberration title: cyan and magenta copies behind the text."""

        self.text(x - 2.5, y, s, size=size, fill=CYAN, weight=weight, opacity=0.75)
        self.text(x + 2.5, y, s, size=size, fill=MAGENTA, weight=weight, opacity=0.75)
        self.text(x, y, s, size=size, fill=fill, weight=weight)

    def cursor(self, x, y, w=11, h=18, color=GREEN, dur="1.1s") -> None:
        """A block cursor that blinks."""

        self.raw(
            f'<rect x="{x:.1f}" y="{y - h + 4:.1f}" width="{w}" height="{h}"'
            f' fill="{color}" opacity="0.9">'
            f'<animate attributeName="opacity" values="0.9;0.9;0;0;0.9"'
            f' dur="{dur}" repeatCount="indefinite"/></rect>'
        )

    def sweep(self, x, y, w, h, color=CYAN, dur="7s", band=54) -> None:
        """A slow scan band travelling down the region."""

        self.clips.add((x, y, w, h))
        self.raw(
            f'<g clip-path="url(#clip-{int(x)}-{int(y)})">'
            f'<rect x="{x:.1f}" y="{y - band:.1f}" width="{w:.1f}"'
            f' height="{band}" fill="{color}" opacity="0.055">'
            f'<animateTransform attributeName="transform" type="translate"'
            f' from="0 0" to="0 {h + band:.1f}" dur="{dur}"'
            f' repeatCount="indefinite"/></rect></g>'
        )

    def render(self) -> str:
        defs = [
            '<pattern id="grid" width="26" height="26" patternUnits="userSpaceOnUse">'
            f'<path d="M 26 0 L 0 0 0 26" fill="none" stroke="{GRID}"'
            ' stroke-width="1"/></pattern>',
            '<pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse">'
            f'<rect width="4" height="1" fill="{TXT}" opacity="0.55"/></pattern>',
            '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">'
            '<feGaussianBlur stdDeviation="4" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/>'
            '</feMerge></filter>',
        ]
        for clip in self.clips:
            x, y, w, h = clip
            defs.append(
                f'<clipPath id="clip-{int(x)}-{int(y)}">'
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}"/></clipPath>'
            )
        defs.extend(self.extra_defs)
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
        if self.framed:
            self.hud(10, 10, self.w - 20, self.h - 20, color=STROKE, size=18,
                     sw=1.6)
            self.scanlines(0, 0, self.w, self.h, opacity=0.03)
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


# --- 1. the terminal, and the scene ----------------------------------------
TW, TH = 1040, 440      # terminal figure
SW, SH = 1600, 900      # hero


def _terminal_lines(svg: SVG, x: float, y: float, size: float = 14.5,
                    lh: float = 22) -> None:
    """Lay out a terminal transcript, segment by segment, in real colours."""

    cw = size * 0.6005
    rows = [
        [("$ python -m minijev train", GREEN)],
        [("parameters", CYAN), ("        9265", TXT), ("   (numpy only)", FAINT)],
        [("department", CYAN), ("   choice  acc 0.736  ece 0.014  brier 0.369", DIM)],
        [("is_urgent", CYAN), ("    noul    acc 0.876  ece 0.013  brier 0.098", DIM)],
        [("severity", CYAN), ("     score   acc 0.767  ece 0.015  brier 0.321", DIM)],
        [],
        [("$ python -m minijev demo --model runs/triage", GREEN)],
        [("my card was charged twice, please refund asap", DIM),
         ("  billing", MAGENTA), ("    0.17  0.745  0.775", AMBER)],
        [("the integration keeps returning a timeout ...", DIM),
         ("  technical", MAGENTA), ("  0.83  0.420  1.418", AMBER)],
        [("could i get a pricing quote before we upgrade", DIM),
         ("  sales", MAGENTA), ("      0.82  0.549  0.457", AMBER)],
        [],
        [("$ python -m pytest -q", GREEN), ("    74 passed in 16.40s", DIM)],
    ]
    for i, row in enumerate(rows):
        cx = x
        for text, color in row:
            svg.text(cx, y + i * lh, text, size=size, fill=color)
            cx += len(text) * cw
    last = y + (len(rows) - 1) * lh
    svg.cursor(cx + 6, last, w=10, h=size + 2, color=GREEN)


def fig_terminal() -> pathlib.Path:
    svg = SVG(TW, TH)
    svg.text(40, 40, "// RUNTIME", size=13, fill=DIM, weight="700",
             tracking="0.14em")
    svg.text(40, 68, "what a request actually comes back with", size=19, fill=TXT)

    tx, ty, tw, th = 40, 96, 960, 300
    svg.rect(tx + 6, ty + 8, tw, th, fill="#000000", opacity=0.5, rx=10)
    svg.rect(tx, ty, tw, th, fill="#070b13", stroke=CYAN, sw=1.5, rx=10)
    svg.rect(tx, ty, tw, 30, fill="#0d1420", stroke=CYAN, sw=1.5, rx=10)
    svg.rect(tx, ty + 20, tw, 10, fill="#0d1420")
    for i, dot in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        svg.raw(f'<circle cx="{tx + 20 + i * 18}" cy="{ty + 15}" r="5.5"'
                f' fill="{dot}" opacity="0.9"/>')
    svg.text(tx + tw / 2, ty + 20, "mini-jev  ::  system-one runtime",
             size=12.5, fill=DIM, anchor="middle")
    _terminal_lines(svg, tx + 26, ty + 60, size=14, lh=21)
    svg.scanlines(tx, ty, tw, th, opacity=0.05)
    svg.sweep(tx, ty, tw, th, color=CYAN, dur="7.5s", band=70)
    caption(svg, 40, 424, "every number above is printed by the commands shown, "
                          "on a laptop CPU")
    return svg.save("terminal.svg")


# the silhouette at the window, in local coordinates with the head near origin
FIGURE = [
    '<path d="M-52 -14 q-26 74 -12 170 q8 60 26 96 l58 0 q14 -52 10 -118'
    ' q-4 -74 -28 -134 z"/>',
    '<path d="M-104 96 q104 -56 208 0 l26 300 l-260 0 z"/>',
    '<rect x="-20" y="28" width="40" height="48" rx="14"/>',
    '<path d="M76 152 q46 -36 38 -94 l26 -6 q16 76 -38 124 z"/>',
    '<ellipse cx="122" cy="52" rx="17" ry="20"/>',
    '<ellipse cx="0" cy="-24" rx="46" ry="54"/>',
    '<path d="M-50 -30 q6 -58 52 -56 q44 2 50 54 q-14 -30 -50 -30 q-36 0 -52 32 z"/>',
    '<path d="M-46 -30 q46 -66 92 0" fill="none" stroke-width="9"/>',
    '<rect x="-62" y="-36" width="26" height="44" rx="10"/>',
    '<rect x="36" y="-36" width="26" height="44" rx="10"/>',
]


def _figure(svg: SVG, dx, dy, scale, fill, stroke, sw, opacity) -> None:
    svg.raw(f'<g transform="translate({dx},{dy}) scale({scale})" fill="{fill}"'
            f' stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}">')
    for prim in FIGURE:
        svg.raw(prim)
    svg.raw("</g>")


def _drop(svg: SVG, x, y, rx, ry, hot=False) -> None:
    svg.raw(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"'
            f' fill="#dceeff" opacity="0.13"/>')
    svg.raw(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx * 0.7:.1f}"'
            f' ry="{ry * 0.7:.1f}" fill="#0b1220" opacity="0.18"/>')
    svg.raw(f'<circle cx="{x - rx * 0.30:.1f}" cy="{y - ry * 0.40:.1f}"'
            f' r="{max(1.0, rx * 0.20):.1f}" fill="#ffffff" opacity="0.60"/>')
    if hot:
        svg.raw(f'<circle cx="{x + rx * 0.24:.1f}" cy="{y + ry * 0.22:.1f}"'
                f' r="{max(1.2, rx * 0.26):.1f}" fill="{MAGENTA}"'
                ' opacity="0.30"/>')


def _skyline(svg: SVG, rng, y_base, height, wmin, wmax, rows, lit) -> None:
    """One band of buildings. Each window row is a single dashed line."""

    palette = ("#ffd9a0", "#9ae6ff", "#f9a8d4", "#c4b5fd")
    x = -70.0
    while x < SW + 70:
        w = float(rng.uniform(wmin, wmax))
        h = float(rng.uniform(height * 0.5, height))
        top = y_base - h
        svg.rect(x, top, w, h, fill="#080d19", stroke="#0f1b2e", sw=0.8, rx=1.5)
        step = max(5.0, h / (rows + 1))
        for r in range(rows):
            wy = top + step * (r + 0.9)
            if wy > y_base - 2:
                continue
            on = rng.random() < lit
            col = palette[int(rng.integers(0, len(palette)))] if on else "#16243c"
            svg.raw(f'<line x1="{x + 3:.1f}" y1="{wy:.1f}"'
                    f' x2="{x + w - 3:.1f}" y2="{wy:.1f}" stroke="{col}"'
                    f' stroke-width="1.6" stroke-dasharray="2.5 3.5"'
                    f' opacity="{0.55 if on else 0.45:.2f}"/>')
        x += w + float(rng.uniform(3, 10))


def _traffic(svg: SVG, rng, y, half_band, color, n, length) -> None:
    for _ in range(n):
        y0 = y + float(rng.uniform(-half_band, half_band))
        x0 = float(rng.uniform(-260, SW))
        ln = float(rng.uniform(length * 0.45, length))
        svg.raw(f'<line x1="{x0:.0f}" y1="{y0:.1f}" x2="{x0 + ln:.0f}"'
                f' y2="{y0:.1f}" stroke="{color}"'
                f' stroke-width="{rng.uniform(1.2, 3.0):.1f}"'
                f' opacity="{rng.uniform(0.22, 0.72):.2f}"'
                ' stroke-linecap="round"/>')
        svg.raw(f'<line x1="{x0:.0f}" y1="{y0 + 7:.1f}" x2="{x0 + ln * 0.8:.0f}"'
                f' y2="{y0 + 7:.1f}" stroke="{color}"'
                f' stroke-width="{rng.uniform(0.7, 1.8):.1f}"'
                f' opacity="{rng.uniform(0.07, 0.20):.2f}"'
                ' stroke-linecap="round"/>')


def fig_hero() -> pathlib.Path:
    svg = SVG(SW, SH)
    svg.framed = False  # the scene draws its own frame
    rng = np.random.default_rng(20260928)
    svg.defs(f'''
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#060912"/>
        <stop offset="38%" stop-color="#0f1733"/>
        <stop offset="66%" stop-color="#2a1546"/>
        <stop offset="88%" stop-color="#3d1338"/>
        <stop offset="100%" stop-color="#160d22"/>
      </linearGradient>
      <radialGradient id="haze" cx="52%" cy="56%" r="60%">
        <stop offset="0%" stop-color="{MAGENTA}" stop-opacity="0.26"/>
        <stop offset="42%" stop-color="{VIOLET}" stop-opacity="0.15"/>
        <stop offset="100%" stop-color="#000000" stop-opacity="0"/>
      </radialGradient>
      <linearGradient id="fog" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#2a1030" stop-opacity="0"/>
        <stop offset="55%" stop-color="#331236" stop-opacity="0.30"/>
        <stop offset="100%" stop-color="#0a0812" stop-opacity="0.96"/>
      </linearGradient>
      <linearGradient id="titlescrim" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#04060c" stop-opacity="0.86"/>
        <stop offset="60%" stop-color="#04060c" stop-opacity="0.34"/>
        <stop offset="100%" stop-color="#04060c" stop-opacity="0"/>
      </linearGradient>
      <radialGradient id="phone" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#a5f3fc" stop-opacity="0.85"/>
        <stop offset="45%" stop-color="{CYAN}" stop-opacity="0.30"/>
        <stop offset="100%" stop-color="{CYAN}" stop-opacity="0"/>
      </radialGradient>
      <linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#d7ecff" stop-opacity="0.10"/>
        <stop offset="34%" stop-color="#d7ecff" stop-opacity="0.015"/>
        <stop offset="52%" stop-color="#ffffff" stop-opacity="0.075"/>
        <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
      </linearGradient>
      <radialGradient id="vig" cx="50%" cy="52%" r="74%">
        <stop offset="48%" stop-color="#000000" stop-opacity="0"/>
        <stop offset="100%" stop-color="#000000" stop-opacity="0.78"/>
      </radialGradient>
      <linearGradient id="scrim" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#03060c" stop-opacity="0.88"/>
        <stop offset="60%" stop-color="#03060c" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#03060c" stop-opacity="0"/>
      </linearGradient>
    ''')

    # sky, glow, deep skyline, nearer skyline
    svg.rect(0, 0, SW, SH, fill="url(#sky)")
    svg.rect(0, 0, SW, SH, fill="url(#haze)")
    _skyline(svg, rng, y_base=330, height=96, wmin=16, wmax=42, rows=4, lit=0.55)
    _skyline(svg, rng, y_base=430, height=150, wmin=30, wmax=78, rows=6, lit=0.45)
    _skyline(svg, rng, y_base=580, height=240, wmin=52, wmax=132, rows=8, lit=0.38)
    # distance fog hides the hard base of the skyline and adds depth
    svg.rect(0, 420, SW, 310, fill="url(#fog)")

    # rain falling towards the camera, behind the glass
    streaks = []
    for _ in range(190):
        x = float(rng.uniform(-140, SW + 140))
        y = float(rng.uniform(-SH, SH))
        ln = float(rng.uniform(24, 92))
        streaks.append((x, y, ln, float(rng.uniform(0.7, 1.7)),
                        float(rng.uniform(0.06, 0.30))))
    body = "".join(
        f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x - ln * 0.26:.0f}"'
        f' y2="{y + ln:.0f}" stroke="#d6ecff" stroke-width="{w:.1f}"'
        f' opacity="{o:.2f}" stroke-linecap="round"/>'
        for x, y, ln, w, o in streaks
    )
    svg.raw(f'<g>{body}<animateTransform attributeName="transform"'
            f' type="translate" from="0 0" to="0 {SH}" dur="2.6s"'
            ' repeatCount="indefinite"/></g>')

    # the street, far below, with long-exposure traffic
    svg.rect(0, 706, SW, 194, fill="#04070d")
    _traffic(svg, rng, 736, 12, "#ffd9a0", 46, 420)
    _traffic(svg, rng, 776, 16, "#ff5f7e", 58, 520)
    _traffic(svg, rng, 828, 22, "#9ae6ff", 42, 300)
    svg.rect(0, 706, SW, 194, fill="#04070d", opacity=0.42)

    # the person: two rimlight passes, then the solid silhouette on top
    fx, fy, fs = 196, 486, 0.94
    _figure(svg, fx - 3, fy - 2, fs, "#05070c", CYAN, 2.6, 0.48)
    _figure(svg, fx + 3, fy + 2, fs, "#05070c", MAGENTA, 2.0, 0.28)
    _figure(svg, fx, fy, fs, "#04060b", "none", 0, 1.0)

    # phone light, positioned from the figure transform so the two cannot drift
    pcx, pcy = fx + 124 * fs, fy + 36 * fs
    svg.raw(f'<circle cx="{pcx:.0f}" cy="{pcy:.0f}" r="138" fill="url(#phone)">'
            '<animate attributeName="opacity" values="0.85;1;0.85" dur="4.5s"'
            ' repeatCount="indefinite"/></circle>')
    svg.rect(pcx - 14, pcy - 28, 28, 56, fill="#0b1a24", stroke=CYAN, sw=1.4,
             rx=6)
    svg.rect(pcx - 10, pcy - 23, 20, 42, fill="#7dd3fc", opacity=0.85, rx=3)

    # glass in front of everything
    _drop_layer = [(float(rng.uniform(0, SW)), float(rng.uniform(0, SH))) for _ in range(56)]
    for dx, dy in _drop_layer:
        rx = float(rng.uniform(5, 21))
        _drop(svg, dx, dy, rx, rx * float(rng.uniform(1.05, 1.5)),
              hot=rng.random() < 0.34)
    for i, (dx, dy) in enumerate(_drop_layer[:4]):
        svg.raw(
            f'<g><ellipse cx="{dx:.0f}" cy="{dy:.0f}" rx="11" ry="17"'
            ' fill="#dceeff" opacity="0.12"/>'
            f'<animateTransform attributeName="transform" type="translate"'
            f' from="0 0" to="0 {120 + i * 40}" dur="{10 + i * 2.5}s"'
            ' repeatCount="indefinite"/></g>'
        )
    svg.rect(0, 0, SW, SH, fill="url(#sheen)")

    # grade, title, caption
    svg.rect(0, 0, SW, SH, fill="url(#vig)")
    svg.rect(22, 22, 760, 168, fill="url(#titlescrim)")
    # the room: a dark window opening around the view
    svg.rect(0, 0, SW, 22, fill="#04060c")
    svg.rect(0, SH - 24, SW, 24, fill="#04060c")
    svg.rect(0, 0, 22, SH, fill="#04060c")
    svg.rect(SW - 22, 0, 22, SH, fill="#04060c")
    svg.rect(22, 22, SW - 44, 1.6, fill="#bfe6ff", opacity=0.18)
    svg.rect(22, 22, 1.6, SH - 46, fill="#bfe6ff", opacity=0.10)
    svg.glitch(96, 106, "mini-jev", size=64)
    svg.text(98, 140, "a System One decision model small enough to read in one sitting",
             size=16, fill=DIM)
    for i, (tag, col) in enumerate((("// TYPED DECISIONS", CYAN),
                                    ("// CALIBRATED PROBABILITIES", VIOLET),
                                    ("// NO AUTOREGRESSION", MAGENTA))):
        svg.text(SW - 96, 96 + i * 24, tag, size=13, fill=col, anchor="end",
                 tracking="0.16em")
    svg.text(96, SH - 46, "// 02:40, still raining, department ECE 0.014",
             size=13, fill=FAINT)
    svg.text(SW - 96, SH - 46, "9,265 parameters · numpy only",
             size=13, fill=FAINT, anchor="end")
    svg.hud(34, 34, SW - 68, SH - 68, color=CYAN, size=30, sw=2.0)
    svg.scanlines(0, 0, SW, SH, opacity=0.03)
    return svg.save("hero.svg")


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
    written = [fig_hero(), fig_contract(), fig_pipeline(), fig_heads(),
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
