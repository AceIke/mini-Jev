"""Compose the README hero and rasterise it.

    python docs/build_hero.py

Two CC0 photographs supply the outside world (see docs/hero/CREDITS.md). The
person, the window frame, the blown rain, the grade and the type are drawn here,
reusing the same palette and silhouette as the vector figures so the README
looks like one document rather than two.

The page is written to docs/hero.html, screenshotted with headless Chrome, and
re-encoded with Pillow so the committed file stays a few hundred kilobytes
instead of several megabytes.
"""

from __future__ import annotations

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

W, H = 1600, 900
SCALE = 1.5                     # renders at 2400x1350, then downsampled
OUT_JPG = HERE / "figures" / "hero.jpg"
OUT_HTML = HERE / "hero.html"
QUALITY = 88


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


def _figure_markup() -> str:
    """The silhouette, in the same local coordinates the SVG figures use."""

    def layer(dx, dy, fill, stroke, sw, opacity, scale=1.10):
        body = "".join(FIGURE)
        return (f'<g transform="translate({dx},{dy}) scale({scale})" fill="{fill}"'
                f' stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}">'
                f"{body}</g>")

    fx, fy, fs = 262, 470, 1.10
    phone_x, phone_y = fx + 124 * fs, fy + 36 * fs
    return f'''
    {layer(fx - 4, fy - 3, "#05070c", CYAN, 3.4, 0.68, fs)}
    {layer(fx + 4, fy + 3, "#05070c", MAGENTA, 2.6, 0.40, fs)}
    {layer(fx, fy, "#04060b", "none", 0, 1.0, fs)}
    <circle cx="{phone_x:.0f}" cy="{phone_y:.0f}" r="150" fill="url(#phone)">
      <animate attributeName="opacity" values="0.80;1;0.80" dur="4.5s"
               repeatCount="indefinite"/>
    </circle>
    <rect x="{phone_x - 14:.0f}" y="{phone_y - 28:.0f}" width="28" height="56"
          rx="6" fill="#0b1a24" stroke="{CYAN}" stroke-width="1.4"/>
    <rect x="{phone_x - 10:.0f}" y="{phone_y - 23:.0f}" width="20" height="42"
          rx="3" fill="#7dd3fc" opacity="0.85"/>
    '''


def _rain_markup() -> str:
    rng = np.random.default_rng(4)
    out = []
    for _ in range(170):
        x = rng.uniform(-120, W + 120)
        y = rng.uniform(-H, H)
        ln = rng.uniform(26, 96)
        sw = rng.uniform(0.8, 2.0)
        op = rng.uniform(0.10, 0.42)
        out.append(
            f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x - ln * 0.30:.0f}"'
            f' y2="{y + ln:.0f}" stroke="#dceeff" stroke-width="{sw:.1f}"'
            f' opacity="{op:.2f}" stroke-linecap="round"/>'
        )
    return (f'<g>{"".join(out)}<animateTransform attributeName="transform"'
            f' type="translate" from="0 0" to="0 {H}" dur="2.2s"'
            ' repeatCount="indefinite"/></g>')


def build_html() -> str:
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
    background:url("hero/street.jpg") center 78% / cover no-repeat;
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
  .kicker.dim {{ color:#7b8ca4; }}
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
  .foot {{ position:absolute; left:96px; bottom:44px;
      font:400 13px/1.5 ui-monospace,Menlo,Consolas,monospace; color:#7b8ca4; }}
  .foot.right {{ left:auto; right:96px; text-align:right; }}
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
    {_figure_markup()}
    {_rain_markup()}
    <rect width="{W}" height="{H}" fill="url(#sheen)"/>
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
  <div class="foot right">9,265 parameters &middot; numpy only</div>
</div></body></html>
"""


def main() -> None:
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    OUT_JPG.parent.mkdir(parents=True, exist_ok=True)
    chrome = _chrome()
    with tempfile.TemporaryDirectory() as tmp:
        png = pathlib.Path(tmp) / "hero.png"
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             f"--force-device-scale-factor={SCALE}",
             f"--window-size={W},{H}",
             f"--screenshot={png}",
             OUT_HTML.as_uri()],
            check=True, capture_output=True,
        )
        image = Image.open(png).convert("RGB")
        if image.width > 2000:
            image = image.resize((2000, round(image.height * 2000 / image.width)),
                                 Image.LANCZOS)
        image.save(OUT_JPG, "JPEG", quality=QUALITY, optimize=True,
                   progressive=True)
    size = OUT_JPG.stat().st_size
    print(f"wrote {OUT_HTML.relative_to(REPO)}")
    print(f"wrote {OUT_JPG.relative_to(REPO)}  {image.width}x{image.height}  "
          f"{size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
