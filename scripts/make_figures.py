"""
make_figures.py -- regenerate the README and site figures from real runs.

Every mark in these SVGs comes from the lab's own code on the data bundled in
tools/data_cache/: the worldlines, light cones and black-hole wells are the
output of MinkowskiClassifier and BlackHoleDetector (lib/srfm_core.py) via
examples/bh_quickstart.py, and the terminal figure is the captured stdout of
that script. Nothing is typed in by hand.

Run from the repo root (needs requirements-core.txt):
    python scripts/make_figures.py            # writes assets/*.svg
"""

from __future__ import annotations

import math
import subprocess
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import bh_quickstart as q  # noqa: E402

OUT = ROOT / "assets"

SERIF = "'Iowan Old Style','Palatino Linotype',Palatino,'Book Antiqua',Georgia,serif"
MONO = "ui-monospace,SFMono-Regular,'Cascadia Mono',Consolas,'DejaVu Sans Mono',Menlo,monospace"

THEMES = {
    "light": {
        "bg": "#F5F2EA", "card": "#FBF9F4", "ink": "#1A1D24", "ink2": "#4A4B50",
        "muted": "#7A776F", "grid": "#DDD7C9", "rule": "#C9C2B1",
        "time": "#00809A", "space": "#D0402A", "well": "#C9A200",
        "cone": "#00809A", "cone_a": 0.09, "well_a": 0.16, "term": "#16191F",
    },
    "dark": {
        "bg": "#0E1116", "card": "#141820", "ink": "#ECE7DC", "ink2": "#B9B4A9",
        "muted": "#8A867D", "grid": "#232831", "rule": "#343A45",
        "time": "#35A9C6", "space": "#EE5E3F", "well": "#E0B43A",
        "cone": "#35A9C6", "cone_a": 0.12, "well_a": 0.18, "term": "#0A0C10",
    },
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def bh_frame(sym: str) -> pd.DataFrame:
    p = q.PARAMS[sym]
    return q.run_bh(q.load_daily_close(sym), p["cf"], p["bh_form"])


def fmt(v: float, nd: int = 1) -> str:
    return f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"


def svg_open(w: int, h: int, t: dict, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-labelledby="t d">',
        f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>',
        f'<rect width="{w}" height="{h}" fill="{t["bg"]}"/>',
    ]


def text(x, y, s, t, size=14, fill=None, family=None, anchor="start", weight=400,
         italic=False, spacing=None, opacity=None) -> str:
    extra = ""
    if italic:
        extra += ' font-style="italic"'
    if spacing is not None:
        extra += f' letter-spacing="{spacing}"'
    if opacity is not None:
        extra += f' opacity="{opacity}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family or SERIF}" font-size="{size}" '
            f'fill="{fill or t["ink"]}" text-anchor="{anchor}" font-weight="{weight}"{extra}>'
            f'{escape(s)}</text>')


# --------------------------------------------------------------------------
# worldline diagram (shared by the hero and the site)
# --------------------------------------------------------------------------

def worldline(t: dict, res: pd.DataFrame, cf: float, x0: float, y_base: float, px: float,
              xmin: float, xmax: float, annotate: bool = True) -> tuple[list[str], dict]:
    """Draw price as a worldline: time runs left to right (1 day = px), displacement
    x = ln(P/P0)/c runs up (1 unit = px), so the light cone |dx| = c dt is at 45 degrees."""
    close = res["close"].to_numpy(float)
    xs = np.log(close / close[0]) / cf
    n = len(xs)
    X = lambda i: x0 + i * px  # noqa: E731
    Y = lambda v: y_base - (v - xmin) * px  # noqa: E731
    top, bot = Y(xmax), Y(xmin)
    g: list[str] = []

    # grid: horizontal every 2 units, vertical every 5 trading days
    for v in range(math.ceil(xmin), math.floor(xmax) + 1):
        if v % 2 == 0:
            g.append(f'<line x1="{X(0):.1f}" x2="{X(n - 1):.1f}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                     f'stroke="{t["grid"]}" stroke-width="1"/>')
    for i in range(0, n, 5):
        g.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{top:.1f}" y2="{bot:.1f}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')

    # black-hole wells: bands over the days the detector reports active
    act = res["active"].to_numpy(bool)
    i = 0
    wells = []
    while i < n:
        if act[i]:
            j = i
            while j + 1 < n and act[j + 1]:
                j += 1
            wells.append((i, j))
            a, b = X(i) - px * 0.5, X(j) + px * 0.5
            g.append(f'<rect x="{a:.1f}" y="{top:.1f}" width="{b - a:.1f}" height="{bot - top:.1f}" '
                     f'fill="{t["well"]}" opacity="{t["well_a"]}"/>')
            g.append(f'<line x1="{a:.1f}" x2="{b:.1f}" y1="{top:.1f}" y2="{top:.1f}" '
                     f'stroke="{t["well"]}" stroke-width="3"/>')
            i = j + 1
        else:
            i += 1

    # future light cone at every event: the region a timelike next bar must land in
    for k in range(n - 1):
        cx, cy = X(k), Y(xs[k])
        g.append(f'<path d="M{cx:.1f},{cy:.1f} L{cx + px:.1f},{cy - px:.1f} L{cx + px:.1f},{cy + px:.1f} Z" '
                 f'fill="{t["cone"]}" fill-opacity="{t["cone_a"]}" stroke="{t["cone"]}" '
                 f'stroke-opacity="0.22" stroke-width="0.8"/>')

    # the worldline itself, one segment per bar, coloured by the classifier
    bits = res["bit"].to_numpy()
    for k in range(1, n):
        x1, y1, x2, y2 = X(k - 1), Y(xs[k - 1]), X(k), Y(xs[k])
        if bits[k] == "TIMELIKE":
            g.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{t["time"]}" stroke-width="2.6" stroke-linecap="round"/>')
        else:
            g.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                     f'stroke="{t["space"]}" stroke-width="2.6" stroke-linecap="round" '
                     f'stroke-dasharray="5 4"/>')
    for k in range(n):
        r = 3.4 if act[k] else 2.6
        fill = t["well"] if act[k] else t["ink"]
        g.append(f'<circle cx="{X(k):.1f}" cy="{Y(xs[k]):.1f}" r="{r}" fill="{fill}" '
                 f'stroke="{t["bg"]}" stroke-width="1.6"/>')

    info = {"xs": xs, "n": n, "X": X, "Y": Y, "top": top, "bot": bot, "wells": wells}
    return g, info


def date_axis(t, res, info, y, every=5, size=11) -> list[str]:
    g = []
    for i in range(0, info["n"], every):
        d = res.index[i]
        g.append(text(info["X"](i), y, d.strftime("%b %d"), t, size=size, fill=t["muted"],
                      family=MONO, anchor="middle"))
    return g


def mass_strip(t, res, info, y0, h, bh_form, px) -> list[str]:
    """BH mass per bar as thin bars on the shared time axis, with the formation line."""
    mass = res["mass"].to_numpy(float)
    act = res["active"].to_numpy(bool)
    top_m = max(4.0, float(np.ceil(mass.max())))
    g = [f'<line x1="{info["X"](0) - px * 0.5:.1f}" x2="{info["X"](info["n"] - 1) + px * 0.5:.1f}" '
         f'y1="{y0 + h:.1f}" y2="{y0 + h:.1f}" stroke="{t["rule"]}" stroke-width="1"/>']
    bw = max(3.0, px * 0.42)
    for k, m in enumerate(mass):
        bh = h * m / top_m
        col = t["well"] if act[k] else t["ink2"]
        op = 1 if act[k] else 0.55
        g.append(f'<rect x="{info["X"](k) - bw / 2:.1f}" y="{y0 + h - bh:.1f}" width="{bw:.1f}" '
                 f'height="{bh:.1f}" rx="1.5" fill="{col}" opacity="{op}"/>')
    yf = y0 + h - h * bh_form / top_m
    g.append(f'<line x1="{info["X"](0) - px * 0.5:.1f}" x2="{info["X"](info["n"] - 1) + px * 0.5:.1f}" '
             f'y1="{yf:.1f}" y2="{yf:.1f}" stroke="{t["well"]}" stroke-width="1.4" stroke-dasharray="3 3"/>')
    return g


# --------------------------------------------------------------------------
# 1. hero
# --------------------------------------------------------------------------

HERO_SYM, HERO_FROM, HERO_TO = "ES", "2025-08-20", "2025-10-24"


def hero(theme: str, social: bool = False) -> str:
    t = THEMES[theme]
    W, H = 1280, 640
    full = bh_frame(HERO_SYM)
    res = full.loc[HERO_FROM:HERO_TO]
    p = q.PARAMS[HERO_SYM]
    cf, form = p["cf"], p["bh_form"]
    px = 19.0
    xmin, xmax = -2.0, 12.5
    x0 = 100.0
    y_base = 505.0
    s = svg_open(W, H, t, "SRFM Lab: price paths as worldlines",
                 f"{p['proxy']} daily closes {HERO_FROM} to {HERO_TO} drawn as a spacetime worldline "
                 f"with a light cone at every bar, computed by lib/srfm_core.py.")
    body, info = worldline(t, res, cf, x0, y_base, px, xmin, xmax)

    # headline
    s.append(text(64, 64, "SPECIAL RELATIVITY IN FINANCIAL MODELING", t, size=13, fill=t["muted"],
                  family=MONO, spacing=2.4))
    s.append(text(60, 132, "SRFM Lab", t, size=72, weight=600, spacing=-1))
    s.append(text(64, 176, "Price paths drawn as worldlines. Every bar is tested against", t, size=21,
                  fill=t["ink2"]))
    s.append(text(64, 204, "a light cone; runs of ordered bars build mass until a well forms.", t,
                  size=21, fill=t["ink2"]))
    s += body

    X, Y, xs = info["X"], info["Y"], info["xs"]
    # y axis label + ticks
    for v in range(0, int(xmax) + 1, 4):
        s.append(text(x0 - 16, Y(v) + 4, f"{v:+d}" if v else "0", t, size=11, fill=t["muted"],
                      family=MONO, anchor="end"))
    s.append(f'<g transform="translate({x0 - 52:.1f},{Y(5.5):.1f}) rotate(-90)">'
             + text(0, 0, "x = ln(P / P₀) / c", t, size=12, fill=t["muted"], family=MONO,
                    anchor="middle") + "</g>")
    s += date_axis(t, res, info, y_base + 20)

    # annotations, from the data
    idx = list(res.index)
    for a, b in info["wells"]:
        m = res["mass"].iloc[a]
        lx = X(a) - px * 0.5
        s.append(text(lx, info["top"] - 10, f"well forms {idx[a]:%b %d}  mass {m:.2f} > {form}", t,
                      size=11.5, fill=t["ink2"], family=MONO))
        break
    # largest spacelike bar in the window
    ret = res["close"].pct_change()
    k = int(np.nanargmax(np.abs(ret.to_numpy())))
    beta = abs(ret.iloc[k]) / cf
    sx, sy = X(k), Y(xs[k])
    s.append(f'<line x1="{sx + 8:.1f}" y1="{sy + 6:.1f}" x2="{sx + 34:.1f}" y2="{sy + 34:.1f}" '
             f'stroke="{t["space"]}" stroke-width="1"/>')
    s.append(text(sx + 38, sy + 40, f"{idx[k]:%b %d}  {ret.iloc[k] * 100:+.1f}% in a day", t, size=12,
                  fill=t["ink"], family=MONO))
    s.append(text(sx + 38, sy + 56, f"β = {beta:.1f}  spacelike", t, size=12, fill=t["space"],
                  family=MONO))

    # mass strip
    s += mass_strip(t, res, info, 542, 40, form, px)
    s.append(text(x0 - 16, 570, "mass", t, size=11, fill=t["muted"], family=MONO, anchor="end"))

    # legend, right column
    lx, ly = 1060, 300
    s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly - 12}" stroke="{t["time"]}" stroke-width="2.6" stroke-linecap="round"/>')
    s.append(text(lx + 38, ly - 2, "timelike", t, size=14))
    s.append(text(lx + 38, ly + 14, "|r| < c", t, size=11, fill=t["muted"], family=MONO))
    ly += 50
    s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 28}" y2="{ly - 22}" stroke="{t["space"]}" stroke-width="2.6" stroke-dasharray="5 4" stroke-linecap="round"/>')
    s.append(text(lx + 38, ly - 6, "spacelike", t, size=14))
    s.append(text(lx + 38, ly + 10, "|r| ≥ c", t, size=11, fill=t["muted"], family=MONO))
    ly += 44
    s.append(f'<rect x="{lx}" y="{ly - 16}" width="28" height="20" fill="{t["well"]}" opacity="{t["well_a"] * 2}"/>')
    s.append(f'<line x1="{lx}" x2="{lx + 28}" y1="{ly - 16}" y2="{ly - 16}" stroke="{t["well"]}" stroke-width="3"/>')
    s.append(text(lx + 38, ly - 2, "well active", t, size=14))
    s.append(text(lx + 38, ly + 14, f"mass > {form}", t, size=11, fill=t["muted"], family=MONO))
    ly += 44
    s.append(f'<path d="M{lx},{ly - 6} L{lx + 22},{ly - 28} L{lx + 22},{ly + 16} Z" fill="{t["cone"]}" fill-opacity="{t["cone_a"] * 1.6}" stroke="{t["cone"]}" stroke-opacity="0.5"/>')
    s.append(text(lx + 38, ly - 6, "light cone", t, size=14))
    s.append(text(lx + 38, ly + 10, "|dx| = c dt", t, size=11, fill=t["muted"], family=MONO))

    caption = (f"{p['proxy']} daily closes, {res.index[0]:%Y-%m-%d} to {res.index[-1]:%Y-%m-%d}.  "
               f"c = {cf * 100:g}% per day, bh_form = {form}.")
    if not social:
        caption += "  Drawn from MinkowskiClassifier and BlackHoleDetector in lib/srfm_core.py."
    s.append(text(64, 620, caption, t, size=11.5, fill=t["muted"], family=MONO))
    if social:
        s.append(text(1216, 620, "github.com/Mattbusel/srfm-lab", t, size=13, fill=t["ink2"],
                      family=MONO, anchor="end"))
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# 2. the quick start chart, restyled: three instruments, last 750 days
# --------------------------------------------------------------------------

def quick_chart(theme: str, last: int = 750) -> str:
    t = THEMES[theme]
    W = 1200
    row_h, gap = 190, 64
    syms = ["ES", "NQ", "YM"]
    H = 86 + len(syms) * (row_h + gap) + 14
    s = svg_open(W, H, t, "BH wells on SPY, QQQ and DIA",
                 f"Daily closes for the last {last} days of the bundled data with black-hole active "
                 "periods shaded and BH mass below each price panel, as computed by examples/bh_quickstart.py.")
    s.append(text(40, 40, "python examples/bh_quickstart.py", t, size=13, fill=t["muted"], family=MONO))
    x0, x1 = 90, W - 40
    y = 86
    for sym in syms:
        res = bh_frame(sym).iloc[-last:]
        p = q.PARAMS[sym]
        n = len(res)
        X = lambda i: x0 + (x1 - x0) * i / (n - 1)  # noqa: E731
        c = res["close"].to_numpy(float)
        lo, hi = c.min(), c.max()
        pad = (hi - lo) * 0.06
        lo, hi = lo - pad, hi + pad
        ph = row_h - 60
        Yp = lambda v: y + ph - (v - lo) / (hi - lo) * ph  # noqa: E731
        s.append(f'<rect x="{x0 - 60}" y="{y - 36}" width="{x1 - x0 + 80}" height="{row_h + 50}" rx="6" '
                 f'fill="{t["card"]}" stroke="{t["grid"]}"/>')
        s.append(text(x0 - 44, y - 12, sym, t, size=16, weight=600))
        s.append(text(x0 - 12, y - 12, f"{p['proxy']} daily  ·  c = {p['cf'] * 100:g}%  ·  bh_form = {p['bh_form']}",
                      t, size=11.5, fill=t["muted"], family=MONO))
        act = res["active"].to_numpy(bool)
        k = 0
        while k < n:
            if act[k]:
                j = k
                while j + 1 < n and act[j + 1]:
                    j += 1
                a, b = X(max(k - 0.5, 0)), X(min(j + 0.5, n - 1))
                s.append(f'<rect x="{a:.1f}" y="{y + 6:.1f}" width="{max(b - a, 2.5):.1f}" height="{row_h - 44:.1f}" '
                         f'fill="{t["well"]}" opacity="{t["well_a"] * 2.2}"/>')
                k = j + 1
            else:
                k += 1
        for v in np.linspace(lo + pad, hi - pad, 3):
            s.append(f'<line x1="{x0}" x2="{x1}" y1="{Yp(v):.1f}" y2="{Yp(v):.1f}" stroke="{t["grid"]}"/>')
            s.append(text(x0 - 8, Yp(v) + 4, f"{v:.0f}", t, size=10.5, fill=t["muted"], family=MONO, anchor="end"))
        pts = " ".join(f"{X(i):.1f},{Yp(v):.1f}" for i, v in enumerate(c))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{t["time"]}" stroke-width="1.6" stroke-linejoin="round"/>')
        # mass
        m = res["mass"].to_numpy(float)
        my0, mh = y + ph + 14, 34
        top_m = max(4.0, float(np.ceil(m.max())))
        Ym = lambda v: my0 + mh - v / top_m * mh  # noqa: E731
        mpts = " ".join(f"{X(i):.1f},{Ym(v):.1f}" for i, v in enumerate(m))
        s.append(f'<polyline points="{mpts}" fill="none" stroke="{t["ink2"]}" stroke-width="1" opacity="0.8"/>')
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{Ym(p["bh_form"]):.1f}" y2="{Ym(p["bh_form"]):.1f}" '
                 f'stroke="{t["well"]}" stroke-width="1.2" stroke-dasharray="3 3"/>')
        s.append(text(x0 - 8, Ym(p["bh_form"]) + 4, "mass", t, size=10.5, fill=t["muted"], family=MONO, anchor="end"))
        for i in np.linspace(0, n - 1, 5).astype(int):
            s.append(text(X(i), my0 + mh + 16, res.index[i].strftime("%Y-%m-%d"), t, size=10.5,
                          fill=t["muted"], family=MONO, anchor="middle"))
        y += row_h + gap
    s.append(text(40, H - 18, "Shaded: BlackHoleDetector reports an active well.  Dashed: formation threshold.  "
                  "Descriptive, one historical sample, not a backtest.", t, size=11.5, fill=t["muted"], family=MONO))
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# 3. terminal capture of the real quick start run
# --------------------------------------------------------------------------

def capture_quickstart() -> list[str]:
    cmd = [sys.executable, "examples/bh_quickstart.py", "--out", "examples/output/bh_quickstart.png"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out.rstrip("\n").split("\n")


def terminal(theme: str, lines: list[str]) -> str:
    t = THEMES[theme]
    cw, lh, fs = 7.25, 20, 12
    width_chars = max(len(x) for x in lines + ["$ python examples/bh_quickstart.py"])
    W = int(max(900, width_chars * cw + 64))
    H = 58 + (len(lines) + 1) * lh + 28
    fg, dim = "#E6E1D6", "#8C877D"
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">',
         '<title id="t">Output of python examples/bh_quickstart.py</title>',
         f'<rect width="{W}" height="{H}" rx="10" fill="{t["term"]}"/>',
         f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="none" stroke="{t["rule"]}"/>']
    for i, col in enumerate(["#5B5F68", "#5B5F68", "#5B5F68"]):
        s.append(f'<circle cx="{22 + i * 18}" cy="20" r="5.5" fill="{col}"/>')
    s.append(text(W / 2, 25, "srfm-lab  ·  bash", t, size=12, fill=dim, family=MONO, anchor="middle"))
    y = 64
    s.append(f'<text x="28" y="{y}" font-family="{MONO}" font-size="{fs}" xml:space="preserve">'
             f'<tspan fill="{THEMES["dark"]["time"]}">$</tspan><tspan fill="{fg}"> python examples/bh_quickstart.py</tspan></text>')
    for ln in lines:
        y += lh
        col = fg
        if ln.startswith("Descriptive"):
            col = THEMES["dark"]["well"]
        elif ln.startswith("chart saved"):
            col = dim
        elif ln.strip().startswith("proxy") or ln.strip() == "symbol" or ln.startswith("symbol"):
            col = dim
        s.append(f'<text x="28" y="{y}" font-family="{MONO}" font-size="{fs}" fill="{col}" '
                 f'xml:space="preserve">{escape(ln)}</text>')
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# 4. the SRFM family: the paper at the origin, implementations in its future cone
# --------------------------------------------------------------------------

FAMILY = [
    # key, name, eyebrow, description lines, card centre
    ("cpp", "Special-Relativity-in-", "C++20 CORE",
     ["Financial-Modeling", "Lorentz factors, interval labels,", "Christoffel symbols, geodesic deviation"], (415, 262)),
    ("py", "srfm-python", "PYTHON SDK",
     [None, "pandas df.srfm accessor and a Polars", "wrapper for the Lorentz-factor pipeline"], (785, 262)),
    ("lab", "srfm-lab", "RESEARCH LAB  ·  YOU ARE HERE",
     [None, "BH signal, backtests with Monte Carlo,", "idea engine, paper-trading research"], (600, 78)),
    ("paper", "srfm-paper-impl", "THE PAPER",
     [None, "Preprint PDF, scripts that regenerate its", "figures, a Rust reference of the formulas"], (600, 478)),
]
CARD_W = 340


def _card(t, key, name, eyebrow, desc, cx, cy) -> list[str]:
    here = key == "lab"
    two_line_name = desc[0] is not None
    h = 112 if two_line_name else 96
    x, y = cx - CARD_W / 2, cy - h / 2
    g = [f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{h}" rx="8" fill="{t["card"]}" '
         f'stroke="{t["well"] if here else t["rule"]}" stroke-width="{2 if here else 1}"/>',
         text(x + 18, y + 24, eyebrow, t, size=10.5, fill=t["well"] if here else t["muted"], family=MONO,
              spacing=1.2),
         text(x + 18, y + 48, name, t, size=18, weight=600)]
    yy = y + 48
    if two_line_name:
        yy += 22
        g.append(text(x + 18, yy, desc[0], t, size=18, weight=600))
    g.append(text(x + 18, yy + 22, desc[1], t, size=13, fill=t["ink2"]))
    g.append(text(x + 18, yy + 39, desc[2], t, size=13, fill=t["ink2"]))
    return g


def family(theme: str) -> str:
    t = THEMES[theme]
    W, H = 1200, 600
    s = svg_open(W, H, t, "The SRFM family of repositories",
                 "srfm-paper-impl at the origin; the C++ core, the Python SDK and this lab lie in its future light cone.")
    ox, oy = 600, 430  # the event: top edge of the paper card
    k = 1.75  # cone opening, drawn wide so the cards sit inside it
    span = oy + 40
    s.append(f'<path d="M{ox},{oy} L{ox - k * span},{oy - span} L{ox + k * span},{oy - span} Z" '
             f'fill="{t["cone"]}" fill-opacity="{t["cone_a"] * 0.7}"/>')
    for sgn in (-1, 1):
        s.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + sgn * k * span}" y2="{oy - span}" stroke="{t["cone"]}" '
                 f'stroke-opacity="0.55" stroke-width="1.2" stroke-dasharray="6 5"/>')
    s.append(text(1010, 240, "future light cone", t, size=12, fill=t["muted"], family=MONO, italic=True))
    s.append(text(1010, 256, "of the paper", t, size=12, fill=t["muted"], family=MONO, italic=True))
    # worldlines from the paper to each implementation
    for cx in (415, 785):
        s.append(f'<path d="M{ox},{oy} C{ox},{oy - 70} {cx},{380} {cx},{318}" fill="none" '
                 f'stroke="{t["time"]}" stroke-width="2.2"/>')
    s.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{126}" stroke="{t["time"]}" stroke-width="2.2"/>')
    for cx in (415, 785):
        s.append(f'<path d="M{cx},{206} C{cx},{160} {ox},{190} {ox + (cx - ox) * 0.15:.0f},{126}" fill="none" '
                 f'stroke="{t["time"]}" stroke-width="1.4" stroke-dasharray="4 4"/>')
    s.append(f'<circle cx="{ox}" cy="{oy}" r="5" fill="{t["ink"]}" stroke="{t["bg"]}" stroke-width="2"/>')
    for key, name, eyebrow, desc, (cx, cy) in FAMILY:
        s += _card(t, key, name, eyebrow, desc, cx, cy)
    s.append(text(40, H - 18, "Related: fin-stream, a Rust crate with a streaming lorentz module built on the same transform.",
                  t, size=12, fill=t["muted"], family=MONO))
    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------
# 5. how the lab fits together
# --------------------------------------------------------------------------

PIPE = [
    ("Bars", ["tools/data_cache", "Alpaca, Binance feeds"]),
    ("Minkowski", ["timelike or spacelike", "lib/srfm_core.py"]),
    ("Black hole", ["mass, wells, direction", "lib/srfm_core.py"]),
    ("Signal stack", ["GARCH, OU, geodesic,", "Hawking, regime"]),
    ("Backtest", ["costs, Monte Carlo", "tools/, backtest/"]),
    ("Paper trader", ["Alpaca paper account", "live_trader_alpaca.py"]),
]


def pipeline(theme: str) -> str:
    t = THEMES[theme]
    W, H = 1200, 300
    s = svg_open(W, H, t, "How srfm-lab fits together",
                 "Bars flow through the Minkowski classifier and black-hole detector into the signal stack, "
                 "backtests and the paper trader; the idea engine feeds tuned parameters back.")
    n = len(PIPE)
    x0, x1, y = 40, W - 40, 96
    gap = 22
    w = (x1 - x0 - gap * (n - 1)) / n
    h = 84
    for i, (name, lines) in enumerate(PIPE):
        x = x0 + i * (w + gap)
        core = name in ("Minkowski", "Black hole")
        s.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="8" fill="{t["card"]}" '
                 f'stroke="{t["time"] if core else t["rule"]}" stroke-width="{1.8 if core else 1}"/>')
        s.append(text(x + 14, y + 28, name, t, size=16, weight=600))
        s.append(text(x + 14, y + 50, lines[0], t, size=12, fill=t["ink2"]))
        s.append(text(x + 14, y + 67, lines[1], t, size=11, fill=t["muted"], family=MONO))
        if i < n - 1:
            ax = x + w + 3
            s.append(f'<path d="M{ax:.1f},{y + h / 2} l{gap - 8},0 m-5,-4 l5,4 l-5,4" fill="none" '
                     f'stroke="{t["ink2"]}" stroke-width="1.4"/>')
    s.append(text(x0 + w + gap / 2, y - 22, "the SRFM core", t, size=12, fill=t["time"], family=MONO))
    s.append(f'<line x1="{x0 + w + gap:.1f}" x2="{x0 + 3 * w + 2 * gap:.1f}" y1="{y - 12}" y2="{y - 12}" '
             f'stroke="{t["time"]}" stroke-width="1.4"/>')
    # feedback loop
    bx = x0 + 4 * (w + gap) + w / 2
    ex = x0 + 3 * (w + gap) + w / 2
    fy = y + h + 62
    s.append(f'<path d="M{bx:.1f},{y + h} L{bx:.1f},{fy} L{ex:.1f},{fy} L{ex:.1f},{y + h + 6}" fill="none" '
             f'stroke="{t["well"]}" stroke-width="1.6"/>')
    s.append(f'<path d="M{ex - 5:.1f},{y + h + 12} l5,-6 l5,6" fill="none" stroke="{t["well"]}" stroke-width="1.6"/>')
    lx = (bx + ex) / 2
    s.append(f'<rect x="{lx - 150:.1f}" y="{fy - 16}" width="300" height="32" rx="16" fill="{t["bg"]}" stroke="{t["well"]}"/>')
    s.append(text(lx, fy + 5, "Idea engine: tunes parameters", t, size=13, anchor="middle"))
    s.append(text(x0, H - 20, "Most stages need network or keys; the core (outlined) runs offline in examples/bh_quickstart.py.",
                  t, size=11.5, fill=t["muted"], family=MONO))
    s.append("</svg>")
    return "\n".join(s)


def site_data(lines: list[str]) -> None:
    """Per-bar output of the classifier and detector for the interactive site diagram."""
    import json
    site = ROOT / "site" / "data"
    site.mkdir(parents=True, exist_ok=True)
    data = {"hero": {"sym": HERO_SYM, "from": HERO_FROM, "to": HERO_TO}, "instruments": {}}
    for sym, p in q.PARAMS.items():
        res = bh_frame(sym)
        data["instruments"][sym] = {
            "proxy": p["proxy"], "cf": p["cf"], "bh_form": p["bh_form"],
            "d": [d.strftime("%Y-%m-%d") for d in res.index],
            "c": [round(float(v), 2) for v in res["close"]],
            "t": "".join("T" if b == "TIMELIKE" else ("S" if b == "SPACELIKE" else "U") for b in res["bit"]),
            "m": [round(float(v), 3) for v in res["mass"]],
            "a": "".join("1" if v else "0" for v in res["active"]),
            "dir": [int(v) for v in res["dir"]],
        }
    (site / "worldlines.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    (site / "quickstart.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    lines = capture_quickstart()
    for theme in THEMES:
        (OUT / f"hero-{theme}.svg").write_text(hero(theme), encoding="utf-8")
        (OUT / f"wells-{theme}.svg").write_text(quick_chart(theme), encoding="utf-8")
        (OUT / f"family-{theme}.svg").write_text(family(theme), encoding="utf-8")
        (OUT / f"pipeline-{theme}.svg").write_text(pipeline(theme), encoding="utf-8")
    (OUT / "quickstart-terminal.svg").write_text(terminal("dark", lines), encoding="utf-8")
    site_data(lines)
    (OUT / "social-card.svg").write_text(hero("dark", social=True), encoding="utf-8")
    print(f"wrote figures to {OUT.relative_to(ROOT)}/ and site data to site/data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
