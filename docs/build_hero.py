"""Compose the README hero and rasterise it, still or moving.

    python docs/build_hero.py              # moving WebP + GIF, plus a still JPEG
    python docs/build_hero.py --still      # only the still JPEG

Two CC0 photographs supply the outside world (see docs/hero/CREDITS.md). The
person, the window frame, the blown rain, the grade and the type are drawn here,
reusing the same palette and silhouette as the vector figures so the README
looks like one document rather than two.

Nothing here depends on the browser's own animation clock. Every frame is
generated from a phase in ``[0, 1)`` that drives the falling rain, the drifting
traffic and the phone glow, so the loop is exact and the output is reproducible
frame for frame.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from make_figures import CYAN, FIGURE, MAGENTA, VIOLET  # noqa: E402

W, H = 1600, 900          # css canvas
RENDER_SCALE = 1.25       # screenshot at 2000x1125, then downsample
OUT_W = 1400              # final asset width
FRAMES = 24
FPS = 12
QUALITY = 90
GIF_W = 900          # the GIF is a compatibility fallback, so it stays small
GIF_STRIDE = 2       # every other frame, at half the frame rate

FIGURES = HERE / "figures"
STILL = FIGURES / "hero.jpg"
MOVING_WEBP = FIGURES / "hero.webp"
MOVING_GIF = FIGURES / "hero.gif"
OUT_HTML = HERE / "hero.html"


def _chrome() -> str:
    candidates = [
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for path in candidates:
        if path and pathlib.Path(path).exists():
            return path
    raise SystemExit("no Chrome or Edge found; install one to rasterise the hero")


# --- vector layers ---------------------------------------------------------
def _figure_markup(phase: float) -> str:
    """The silhouette, plus a phone that breathes."""

    def layer(dx, dy, fill, stroke, sw, opacity, scale=1.10):
        body = "".join(FIGURE)
        return (f'<g transform="translate({dx},{dy}) scale({scale})" fill="{fill}"'
                f' stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}">'
                f"{body}</g>")

    fx, fy, fs = 262, 470, 1.10
    glow = 0.78 + 0.22 * math.sin(2 * math.pi * phase)
    return f'''
    {layer(fx - 4, fy - 3, "#05070c", CYAN, 3.4, 0.68, fs)}
    {layer(fx + 4, fy + 3, "#05070c", MAGENTA, 2.6, 0.40, fs)}
    {layer(fx, fy, "#04060b", "none", 0, 1.0, fs)}
    <g transform="translate({fx + 124 * fs:.1f},{fy + 36 * fs:.1f})">
      <circle r="150" fill="url(#phone)" opacity="{glow:.3f}"/>
      <circle r="88" fill="url(#phone)" opacity="{glow * 0.5:.3f}"/>
      <rect x="-14" y="-28" width="28" height="56" rx="6" fill="#0b1a24"
            stroke="{CYAN}" stroke-width="1.4"/>
      <rect x="-10" y="-23" width="20" height="42" rx="3" fill="#7dd3fc"
            opacity="{0.75 + 0.2 * (1 - glow):.3f}"/>
    </g>
    '''


def _rain_markup(phase: float) -> str:
    """Wind-blown rain. The field spans two canvas heights so the loop is seamless."""

    rng = np.random.default_rng(4)
    offset = phase * H
    out = []
    for _ in range(190):
        x = rng.uniform(-120, W + 120)
        y = rng.uniform(-H, H)
        ln = rng.uniform(26, 96)
        sw = rng.uniform(0.8, 2.0)
        op = rng.uniform(0.10, 0.44)
        out.append(
            f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x - ln * 0.30:.0f}"'
            f' y2="{y + ln:.0f}" stroke="#dceeff" stroke-width="{sw:.1f}"'
            f' opacity="{op:.2f}" stroke-linecap="round"/>'
        )
    return f'<g transform="translate(0,{offset:.1f})">{"".join(out)}</g>'


def _droplets(phase: float) -> str:
    """A couple of drops that crawl down the pane, fading as they go."""

    rng = np.random.default_rng(19)
    out = []
    for i in range(3):
        x = float(rng.uniform(180, W - 180))
        y0 = float(rng.uniform(140, 420))
        run = 150 + i * 70
        y = y0 + phase * run
        fade = max(0.0, 1.0 - phase * 1.25)
        rx = 11 + i * 2
        out.append(
            f'<ellipse cx="{x:.0f}" cy="{y:.0f}" rx="{rx}" ry="{rx * 1.5:.0f}"'
            f' fill="#dceeff" opacity="{0.14 * fade:.3f}"/>'
            f'<circle cx="{x - rx * 0.3:.0f}" cy="{y - rx * 0.6:.0f}" r="2.2"'
            f' fill="#ffffff" opacity="{0.6 * fade:.3f}"/>'
        )
    return "".join(out)


def build_html(phase: float) -> str:
    # the street layer drifts, which reads as traffic moving far below
    drift = phase * 190.0
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>mini-jev hero</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ width:{W}px; height:{H}px; overflow:hidden; background:#04060c; }}
  .stage {{ position:relative; width:{W}px; height:{H}px; overflow:hidden; }}
  .layer {{ position:absolute; inset:0; }}

  /* the outside: a wet pane in focus, the city behind it out of focus */
  .glass {{
    background:url("hero/glass.jpg") center 42% / cover no-repeat;
    filter:saturate(0.85) contrast(1.12) brightness(0.78);
  }}
  /* the street, far below, seen down through the glass */
  .street {{
    background-image:url("hero/street.jpg");
    background-size:175% auto;
    background-position:{480 - drift:.0f}px 78%;
    background-repeat:repeat-x;
    mix-blend-mode:screen; opacity:0.66;
    filter:blur(1.9px) saturate(1.45) contrast(1.18) brightness(1.12);
    transform:perspective(1150px) rotateX(24deg) scale(1.3);
    transform-origin:50% 100%;
    -webkit-mask-image:linear-gradient(to bottom,
      transparent 0%, transparent 30%, #000 64%, #000 100%);
    mask-image:linear-gradient(to bottom,
      transparent 0%, transparent 30%, #000 64%, #000 100%);
  }}
  .grade {{
    background:
      radial-gradient(120% 82% at 50% 104%, rgba(244,114,182,0.26), transparent 62%),
      radial-gradient(90% 70% at 22% 60%, rgba(34,211,238,0.12), transparent 66%),
      linear-gradient(to bottom, rgba(4,6,12,0.90) 0%, rgba(6,9,20,0.26) 32%,
                      rgba(4,6,12,0.30) 60%, rgba(3,5,10,0.80) 100%);
  }}
  .vig {{ background:radial-gradient(74% 74% at 50% 52%,
      transparent 46%, rgba(0,0,0,0.80) 100%); }}
  .frame {{ position:absolute; inset:0;
      box-shadow: inset 0 0 0 22px #04060c,
                  inset 0 0 0 24px rgba(191,230,255,0.14); }}
  svg {{ display:block; }}
  .type {{ position:absolute; left:96px; top:74px; }}
  .kicker {{ font:700 13px/1.4 ui-monospace,Menlo,Consolas,monospace;
      letter-spacing:0.18em; color:{CYAN}; }}
  h1 {{ font:700 64px/1 ui-monospace,Menlo,Consolas,monospace;
      color:#eef4fc; margin:14px 0 10px;
      text-shadow:-2.5px 0 0 rgba(34,211,238,0.75), 2.5px 0 0 rgba(244,114,182,0.75); }}
  .sub {{ font:400 16px/1.4 ui-monospace,Menlo,Consolas,monospace; color:#a3b4cb; }}
  .note {{ font:400 13px/1.6 ui-monospace,Menlo,Consolas,monospace;
      color:#8b9cb3; margin-top:14px;
      text-shadow:0 1px 4px rgba(3,5,10,0.95), 0 0 14px rgba(3,5,10,0.85); }}
  .tags {{ position:absolute; right:96px; top:86px; text-align:right; }}
  .tags div {{ font:700 13px/1.9 ui-monospace,Menlo,Consolas,monospace;
      letter-spacing:0.16em; }}
  .foot {{ position:absolute; right:96px; bottom:44px; text-align:right;
      font:400 13px/1.5 ui-monospace,Menlo,Consolas,monospace; color:#8b9cb3; }}
</style></head>
<body><div class="stage">
  <div class="layer glass"></div>
  <div class="layer street"></div>
  <div class="layer grade"></div>

  <svg class="layer" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <radialGradient id="phone" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#a5f3fc" stop-opacity="0.80"/>
        <stop offset="45%" stop-color="{CYAN}" stop-opacity="0.26"/>
        <stop offset="100%" stop-color="{CYAN}" stop-opacity="0"/>
      </radialGradient>
      <linearGradient id="sheen" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#d7ecff" stop-opacity="0.08"/>
        <stop offset="38%" stop-color="#d7ecff" stop-opacity="0.01"/>
        <stop offset="56%" stop-color="#ffffff" stop-opacity="0.06"/>
        <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
      </linearGradient>
    </defs>
    {_figure_markup(phase)}
    <rect width="{W}" height="{H}" fill="url(#sheen)"/>
    {_droplets(phase)}
    {_rain_markup(phase)}
  </svg>

  <div class="layer vig"></div>
  <div class="frame"></div>
  <div class="type">
    <div class="kicker">// MINI-JEV</div>
    <h1>mini-jev</h1>
    <div class="sub">a System One decision model small enough to read in one sitting</div>
    <div class="note">// 02:40, still raining, department ECE 0.014</div>
  </div>
  <div class="tags">
    <div style="color:{CYAN}">// TYPED DECISIONS</div>
    <div style="color:{VIOLET}">// CALIBRATED PROBABILITIES</div>
    <div style="color:{MAGENTA}">// NO AUTOREGRESSION</div>
  </div>
  <div class="foot">9,265 parameters &middot; numpy only</div>
</div></body></html>
"""


# --- rasterising -----------------------------------------------------------
def render_frame(chrome: str, phase: float, out_png: pathlib.Path) -> None:
    OUT_HTML.write_text(build_html(phase), encoding="utf-8")
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--force-device-scale-factor={RENDER_SCALE}",
         f"--window-size={W},{H}",
         f"--screenshot={out_png}",
         OUT_HTML.as_uri()],
        check=True, capture_output=True,
    )


def _load(png: pathlib.Path) -> Image.Image:
    image = Image.open(png).convert("RGB")
    height = round(image.height * OUT_W / image.width)
    return image.resize((OUT_W, height), Image.LANCZOS)


def _gif_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """One shared palette, so the animation does not shimmer between frames."""

    palette = frames[0].quantize(colors=256, method=Image.MEDIANCUT)
    return [f.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for f in frames]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--still", action="store_true",
                        help="only render the still JPEG")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    chrome = _chrome()

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)

        print("rendering the still...")
        render_frame(chrome, 0.0, tmpdir / "still.png")
        still = _load(tmpdir / "still.png")
        still.save(STILL, "JPEG", quality=QUALITY, optimize=True, progressive=True)
        print(f"  {STILL.relative_to(REPO)}  {still.width}x{still.height}  "
              f"{STILL.stat().st_size / 1024:.0f} KB")

        if args.still:
            return

        print(f"rendering {args.frames} frames...")
        frames = []
        for i in range(args.frames):
            png = tmpdir / f"f{i:03d}.png"
            render_frame(chrome, i / args.frames, png)
            frames.append(_load(png))
            print(f"\r  frame {i + 1}/{args.frames}", end="", flush=True)
        print()

        duration = round(1000 / args.fps)
        frames[0].save(
            MOVING_WEBP, format="WEBP", save_all=True, append_images=frames[1:],
            duration=duration, loop=0, quality=QUALITY, method=6,
        )
        print(f"  {MOVING_WEBP.relative_to(REPO)}  {frames[0].width}x{frames[0].height}  "
              f"{args.frames} frames @ {args.fps} fps  "
              f"{MOVING_WEBP.stat().st_size / 1024 / 1024:.1f} MB")

        gif_src = [
            f.resize((GIF_W, round(f.height * GIF_W / f.width)), Image.LANCZOS)
            for f in frames[::GIF_STRIDE]
        ]
        gif = _gif_frames(gif_src)
        gif[0].save(
            MOVING_GIF, format="GIF", save_all=True, append_images=gif[1:],
            duration=round(1000 / (args.fps / GIF_STRIDE)), loop=0, optimize=False,
        )
        print(f"  {MOVING_GIF.relative_to(REPO)}  {gif[0].width}x{gif[0].height}  "
              f"{len(gif)} frames  "
              f"{MOVING_GIF.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"wrote {OUT_HTML.relative_to(REPO)} (gitignored, regenerated each run)")


if __name__ == "__main__":
    main()
