"""Build the README's physics explainer figures (assets/*-light.svg and *-dark.svg).

  physics-explainer   the four steps from a price bar to a black hole, as cards
  bh-formation        a real run of SPY daily bars: speed, mass and the well

Nothing is recomputed by hand: the chart feeds the bundled bars through the
real classes in lib/srfm_core.py (MinkowskiClassifier, BlackHoleDetector) with
the ES parameters from examples/bh_quickstart.py, exactly as that example does.

    python scripts/figures/make_physics_explainer.py
    python scripts/figures/make_physics_explainer.py --start 2025-12-10 --end 2026-01-28
"""
from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "examples"))

from bh_quickstart import PARAMS, load_daily_close  # noqa: E402
from srfm_core import BlackHoleDetector, MinkowskiClassifier  # noqa: E402

THEMES = {
    "light": dict(bg="#F5F2EA", card="#FBF9F4", ink="#1A1D24", ink2="#4A4B50", muted="#6F6C64",
                  grid="#DDD7C9", rule="#C9C2B1", time="#00809A", space="#D0402A", well="#C9A200",
                  mass="#6A4C93", cone_a=0.19),
    "dark": dict(bg="#0E1116", card="#141820", ink="#ECE7DC", ink2="#B9B4A9", muted="#8F8B82",
                 grid="#232831", rule="#343A45", time="#35A9C6", space="#EE5E3F", well="#E0B43A",
                 mass="#B39DDB", cone_a=0.25),
}
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'Cascadia Mono',Consolas,'DejaVu Sans Mono',Menlo,monospace"
W = 900


def text(x, y, s, t, size=16, fill=None, family=SANS, anchor="start", weight=400, pre=False, halo=False):
    sp = ' xml:space="preserve"' if pre else ""
    sp += f' paint-order="stroke" stroke="{t["bg"]}" stroke-width="5" stroke-linejoin="round"' if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'fill="{fill or t["ink"]}" text-anchor="{anchor}" font-weight="{weight}"{sp}>{escape(s)}</text>')


def rich(x, y, parts, t, size=16):
    spans = "".join(f'<tspan fill="{c or t["ink"]}" font-weight="{w}">{escape(s)}</tspan>' for s, c, w in parts)
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" xml:space="preserve">{spans}</text>'


def open_svg(H, t, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-labelledby="t d">',
            f'<title id="t">{escape(title)}</title>', f'<desc id="d">{escape(desc)}</desc>',
            f'<rect width="{W}" height="{H}" rx="14" fill="{t["bg"]}"/>']


def run(symbol: str):
    """The same loop as examples/bh_quickstart.run_bh, also keeping beta and the streak."""
    p = PARAMS[symbol]
    close = load_daily_close(symbol)
    clf, det = MinkowskiClassifier(cf=p["cf"]), BlackHoleDetector(bh_form=p["bh_form"])
    rows, prev = [], None
    for ts, price in close.items():
        bit = clf.update(float(price))
        if prev is not None:
            det.update(bit, float(price), prev)
        rows.append(dict(date=ts.date().isoformat(), close=float(price), bit=bit,
                         beta=clf.beta if prev is not None else None, mass=det.bh_mass,
                         active=det.bh_active, ctl=det.ctl))
        prev = float(price)
    return rows, p


def fig_steps(theme: str, p: dict) -> str:
    t = THEMES[theme]
    cw, ch, gx, gy, x0, y0 = 396, 236, 20, 20, 44, 118
    ky = y0 + 2 * ch + gy + 44
    H = ky + 70
    cf_pct = p["cf"] * 100
    s = open_svg(H, t, "From a price bar to a black hole, in four steps",
                 "Each bar gets a speed beta = |return| / c. Under 1 it is timelike, over it spacelike. "
                 "Runs of timelike bars add mass; when mass crosses the formation threshold after five "
                 "timelike bars in a row, a black hole (a well) forms.")
    s.append(text(x0, 58, "From a price bar to a black hole", t, size=28, weight=700))
    s.append(text(x0, 90, "What lib/srfm_core.py does to every bar. No physics needed.", t, size=18, fill=t["ink2"]))
    cards = [
        ("1", "Give the bar a speed", [f"c is the market's speed limit:", f"{cf_pct:g}% a day for SPY."],
         ["β = |return| ÷ c"], "MinkowskiClassifier"),
        ("2", "Inside or outside the cone?", ["β < 1: timelike, an ordinary move.", "β ≥ 1: spacelike, a jump."],
         ["ds² = c²dt² − dx²"], "MinkowskiClassifier.update"),
        ("3", "Ordered bars add mass", ["Timelike: mass grows with the move,", "boosted by the streak (x1 to x2).",
                                        "Spacelike: mass × 0.7, streak = 0."],
         ["mass = mass×0.95 + |r|×100×boost"], "BlackHoleDetector.update"),
        ("4", "A black hole forms", [f"mass > {p['bh_form']:g} and 5+ timelike", "bars in a row: a well is active.",
                                     "It stays while mass > 1.0 and", "the streak holds."], [], ""),
    ]
    for k, (num, head, body, code, fn) in enumerate(cards):
        x, y = x0 + (k % 2) * (cw + gx), y0 + (k // 2) * (ch + gy)
        s.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="10" fill="{t["card"]}" stroke="{t["rule"]}"/>')
        s.append(f'<circle cx="{x + 32}" cy="{y + 34}" r="17" fill="{t["time"] if k < 3 else t["well"]}"/>')
        s.append(text(x + 32, y + 40, num, t, size=18, fill=t["card"], anchor="middle", weight=700))
        s.append(text(x + 60, y + 41, head, t, size=21, weight=700))
        for j, ln in enumerate(body):
            s.append(text(x + 20, y + 80 + j * 24, ln, t, size=17, fill=t["ink2"]))
        for j, ln in enumerate(code):
            s.append(text(x + 20, y + 80 + len(body) * 24 + 16 + j * 24, ln, t, size=17, family=MONO, pre=True))
        if fn:
            s.append(text(x + 20, y + ch - 18, fn, t, size=15, family=MONO, fill=t["muted"]))
    # card 4: a well drawn as a funnel with bars falling in
    x, y = x0 + cw + gx, y0 + ch + gy
    cx, top = x + 300, y + 160
    s.append(f'<path d="M{cx - 80},{top} C{cx - 30},{top + 6} {cx - 14},{top + 30} {cx - 8},{top + 52} '
             f'L{cx + 8},{top + 52} C{cx + 14},{top + 30} {cx + 30},{top + 6} {cx + 80},{top} Z" '
             f'fill="{t["well"]}" fill-opacity="0.28" stroke="{t["well"]}" stroke-width="2"/>')
    for i, dx in enumerate((-62, -40, -20)):
        s.append(f'<circle cx="{cx + dx}" cy="{top - 16 + i * 6}" r="5" fill="{t["time"]}"/>')
    s.append(text(x + 20, y + ch - 18, "BlackHoleDetector.bh_active", t, size=15, family=MONO, fill=t["muted"]))
    for k in range(2):
        ay = y0 + k * (ch + gy) + 34
        s.append(f'<path d="M{x0 + cw + 3},{ay} L{x0 + cw + gx - 3},{ay} M{x0 + cw + gx - 9},{ay - 5} '
                 f'L{x0 + cw + gx - 3},{ay} L{x0 + cw + gx - 9},{ay + 5}" stroke="{t["ink2"]}" stroke-width="2" fill="none"/>')
    s.append(rich(x0, ky, [("Lorentz factor", t["ink"], 700),
                           (" γ = 1/√(1 − v²), v = |r| ÷ 1% (max 0.99): each bar adds 1/γ of proper time.",
                            t["ink2"], 400)], t, size=16))
    s.append(text(x0, ky + 26, "Direction of the well: the sign of the timelike run's cumulative return.",
                  t, size=16, fill=t["ink2"]))
    s.append("</svg>")
    return "\n".join(s)


def fig_run(theme: str, rows, p, start: str, end: str) -> str:
    t = THEMES[theme]
    win = [r for r in rows if start <= r["date"] <= end and r["beta"] is not None]
    n = len(win)
    px0, px1 = 96, W - 40
    b0, b1 = 150, 300          # beta panel
    m0, m1 = 372, 540          # mass panel
    H = m1 + 110
    s = open_svg(H, t, "A black hole forming on real SPY bars",
                 f"SPY daily bars {win[0]['date']} to {win[-1]['date']}: each bar's speed beta against the "
                 "limit of 1, then the black-hole mass against the formation threshold, with the days a "
                 "well was active shaded.")
    s.append(text(44, 50, "Real data: a black hole forming on SPY", t, size=24, weight=700))
    s.append(text(44, 78, f"Daily bars {win[0]['date']} to {win[-1]['date']}, run through lib/srfm_core.py "
                  f"(c = {p['cf'] * 100:g}%, bh_form = {p['bh_form']:g}).", t, size=17, fill=t["ink2"]))
    bw = (px1 - px0) / n
    X = lambda i: px0 + i * bw  # noqa: E731
    # shaded active days across both panels
    for i, r in enumerate(win):
        if r["active"]:
            s.append(f'<rect x="{X(i):.1f}" y="{b0 - 10}" width="{bw:.1f}" height="{m1 - b0 + 10}" '
                     f'fill="{t["well"]}" fill-opacity="0.22"/>')
    # beta panel
    bmax = max(3.0, min(6.0, max(r["beta"] for r in win)))
    YB = lambda v: b1 - min(v, bmax) / bmax * (b1 - b0)  # noqa: E731
    s.append(text(44, b0 - 22, "Speed β of each bar", t, size=17, weight=700))
    s.append(f'<rect x="230" y="{b0 - 36}" width="15" height="15" rx="2" fill="{t["time"]}"/>')
    s.append(text(251, b0 - 23, "timelike", t, size=15))
    s.append(f'<rect x="332" y="{b0 - 36}" width="15" height="15" rx="2" fill="{t["space"]}"/>')
    s.append(text(353, b0 - 23, "spacelike", t, size=15))
    s.append(f'<rect x="446" y="{b0 - 36}" width="15" height="15" rx="2" fill="{t["well"]}" fill-opacity="0.5"/>')
    s.append(text(467, b0 - 23, "black hole active", t, size=15))
    for v in range(0, int(bmax) + 1):
        s.append(f'<line x1="{px0}" x2="{px1}" y1="{YB(v):.1f}" y2="{YB(v):.1f}" stroke="{t["grid"]}"/>')
        s.append(text(px0 - 10, YB(v) + 5, f"{v}", t, size=14, family=MONO, fill=t["muted"], anchor="end"))
    for i, r in enumerate(win):
        col = t["time"] if r["bit"] == "TIMELIKE" else t["space"]
        h = b1 - YB(r["beta"])
        s.append(f'<rect x="{X(i) + bw * 0.18:.1f}" y="{b1 - max(h, 1):.1f}" width="{bw * 0.64:.1f}" '
                 f'height="{max(h, 1):.1f}" rx="1.5" fill="{col}"><title>{r["date"]}: beta {r["beta"]:.2f}</title></rect>')
    s.append(f'<line x1="{px0}" x2="{px1}" y1="{YB(1):.1f}" y2="{YB(1):.1f}" stroke="{t["ink"]}" '
             f'stroke-width="2" stroke-dasharray="7 5"/>')
    s.append(text(px0 + 4, YB(1) - 6, "limit β = 1", t, size=14, family=MONO, weight=700, halo=True))
    # streak numbers
    for i, r in enumerate(win):
        if r["ctl"]:
            s.append(text(X(i) + bw / 2, b1 + 20, str(r["ctl"]), t, size=13, family=MONO,
                          fill=t["time"], anchor="middle"))
    s.append(text(px0 - 10, b1 + 20, "run", t, size=13, family=MONO, fill=t["muted"], anchor="end"))
    # mass panel
    mmax = max(2.5, max(r["mass"] for r in win) * 1.1)
    YM = lambda v: m1 - v / mmax * (m1 - m0)  # noqa: E731
    s.append(text(44, m0 - 12, "Black-hole mass", t, size=17, weight=700))
    for v in (0, 1, 2):
        s.append(f'<line x1="{px0}" x2="{px1}" y1="{YM(v):.1f}" y2="{YM(v):.1f}" stroke="{t["grid"]}"/>')
        s.append(text(px0 - 10, YM(v) + 5, f"{v}", t, size=14, family=MONO, fill=t["muted"], anchor="end"))
    s.append(f'<line x1="{px0}" x2="{px1}" y1="{YM(p["bh_form"]):.1f}" y2="{YM(p["bh_form"]):.1f}" '
             f'stroke="{t["well"]}" stroke-width="2" stroke-dasharray="7 5"/>')
    s.append(text(px1, YM(p["bh_form"]) - 6, f"forms above {p['bh_form']:g}", t, size=14, family=MONO,
                  fill=t["ink"], anchor="end", weight=700, halo=True))
    pts = " ".join(f"{X(i) + bw / 2:.1f},{YM(r['mass']):.1f}" for i, r in enumerate(win))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{t["mass"]}" stroke-width="3" stroke-linejoin="round"/>')
    for i, r in enumerate(win):
        s.append(f'<circle cx="{X(i) + bw / 2:.1f}" cy="{YM(r["mass"]):.1f}" r="3.5" fill="{t["mass"]}"/>')
    for i in range(0, n, max(1, n // 5)):
        s.append(text(X(i) + bw / 2, m1 + 24, win[i]["date"], t, size=14, family=MONO,
                      fill=t["muted"], anchor="middle"))
    ons = [i for i, r in enumerate(win) if r["active"] and (i == 0 or not win[i - 1]["active"])]
    s.append(text(44, m1 + 60, f"{len(ons)} formations in this window: " +
                  ", ".join(win[i]["date"] for i in ons) + ". Each one needed 5+ timelike bars in a row.",
                  t, size=15, fill=t["ink2"]))
    s.append(text(44, m1 + 84, "Spacelike bars cut the mass by 30% and reset the run; a well that just "
                  "collapsed gets half its mass back (reform memory).", t, size=15, fill=t["muted"]))
    s.append("</svg>")
    return "\n".join(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="ES", choices=sorted(PARAMS))
    ap.add_argument("--start", default="2025-12-10")
    ap.add_argument("--end", default="2026-01-28")
    ap.add_argument("--out", type=Path, default=ROOT / "assets")
    a = ap.parse_args()
    rows, p = run(a.symbol)
    a.out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (a.out / f"physics-explainer-{theme}.svg").write_text(fig_steps(theme, p), encoding="utf-8")
        (a.out / f"bh-formation-{theme}.svg").write_text(fig_run(theme, rows, p, a.start, a.end), encoding="utf-8")
    win = [r for r in rows if a.start <= r["date"] <= a.end]
    print(f"{a.symbol} ({p['proxy']}): {len(win)} bars {win[0]['date']} to {win[-1]['date']}, "
          f"{sum(r['bit'] == 'TIMELIKE' for r in win)} timelike, {sum(r['active'] for r in win)} days active")
    print(f"wrote physics-explainer and bh-formation (-light/-dark).svg to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
