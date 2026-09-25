"""
bh_quickstart.py -- run the SRFM black-hole (BH) signal on bundled data.

No API keys, no network. Uses the hourly bars cached in tools/data_cache/
(SPY, QQQ and DIA as stand-ins for the ES, NQ and YM futures), resampled to
daily closes, and the core physics in lib/srfm_core.py:

  1. MinkowskiClassifier marks each bar TIMELIKE (|return| < cf) or SPACELIKE.
  2. BlackHoleDetector accumulates "mass" on runs of timelike bars and
     declares a well (BH active) when mass crosses bh_form.

It prints a per-instrument table and saves a chart to
examples/output/bh_quickstart.png.

The forward-return columns are descriptive statistics on one historical
sample, not a backtest and not evidence of a tradable edge.

Run from the repo root:
    python examples/bh_quickstart.py
    python examples/bh_quickstart.py --symbols ES --horizon 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from srfm_core import BlackHoleDetector, MinkowskiClassifier  # noqa: E402

DATA_DIR = ROOT / "tools" / "data_cache"
OUT_DIR = Path(__file__).resolve().parent / "output"

# Daily "speed of light" (cf_1d) and formation threshold per instrument, as
# used by tools/local_backtest.py for these ETF proxies.
PARAMS = {
    "ES": {"proxy": "SPY", "cf": 0.005, "bh_form": 1.5},
    "NQ": {"proxy": "QQQ", "cf": 0.006, "bh_form": 1.5},
    "YM": {"proxy": "DIA", "cf": 0.004, "bh_form": 1.5},
}


def load_daily_close(symbol: str) -> pd.Series:
    path = DATA_DIR / f"{symbol}_1h.csv"
    df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index()
    return df["Close"].resample("1D").last().dropna()


def run_bh(close: pd.Series, cf: float, bh_form: float) -> pd.DataFrame:
    clf = MinkowskiClassifier(cf=cf)
    det = BlackHoleDetector(bh_form=bh_form)
    bits, mass, active, direction = [], [], [], []
    prev = None
    for price in close.to_numpy(dtype=float):
        bit = clf.update(price)
        if prev is not None:
            det.update(bit, price, prev)
        bits.append(bit)
        mass.append(det.bh_mass)
        active.append(det.bh_active)
        direction.append(det.bh_dir)
        prev = price
    return pd.DataFrame(
        {"close": close, "bit": bits, "mass": mass, "active": active, "dir": direction},
        index=close.index,
    )


def summarize(symbol: str, res: pd.DataFrame, horizon: int) -> dict:
    fwd = res["close"].shift(-horizon) / res["close"] - 1.0
    onsets = res["active"] & ~res["active"].shift(1, fill_value=False)
    # Signed by the well's direction, so positive means "moved the way the well pointed".
    onset_fwd = (fwd * res["dir"])[onsets].dropna()
    return {
        "symbol": symbol,
        "proxy": PARAMS[symbol]["proxy"],
        "bars": len(res),
        "from": res.index[0].date().isoformat(),
        "to": res.index[-1].date().isoformat(),
        "timelike %": 100.0 * (res["bit"] == "TIMELIKE").mean(),
        "BH active %": 100.0 * res["active"].mean(),
        "onsets": int(onsets.sum()),
        f"all days |ret {horizon}d| %": 100.0 * fwd.abs().mean(),
        f"onset signed ret {horizon}d %": 100.0 * onset_fwd.mean() if len(onset_fwd) else np.nan,
        "onset hit rate %": 100.0 * (onset_fwd > 0).mean() if len(onset_fwd) else np.nan,
    }


def plot(results: dict[str, pd.DataFrame], path: Path, last_bars: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(2 * n, 1, figsize=(11, 3.2 * n), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1] * n})
    axes = np.atleast_1d(axes)
    for i, (symbol, res) in enumerate(results.items()):
        tail = res.iloc[-last_bars:]
        x = np.arange(len(tail))
        ax_p, ax_m = axes[2 * i], axes[2 * i + 1]
        ax_p.plot(x, tail["close"], color="#1f4e79", lw=1.0)
        ax_p.fill_between(x, tail["close"].min(), tail["close"].max(),
                          where=tail["active"].to_numpy(), color="#f4a261", alpha=0.35,
                          step="mid", label="BH active")
        ax_p.set_title(f"{symbol} ({PARAMS[symbol]['proxy']} daily), last {len(tail)} days")
        ax_p.set_ylabel("close")
        ax_p.legend(loc="upper left", frameon=False)
        ax_m.plot(x, tail["mass"], color="#6a4c93", lw=0.9)
        ax_m.axhline(PARAMS[symbol]["bh_form"], color="#999999", ls="--", lw=0.8)
        ax_m.set_ylabel("BH mass")
        ticks = np.linspace(0, len(tail) - 1, 5).astype(int)
        ax_m.set_xticks(ticks)
        ax_m.set_xticklabels([tail.index[t].strftime("%Y-%m-%d") for t in ticks])
        ax_p.set_xticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--symbols", default="ES,NQ,YM", help="comma list from ES,NQ,YM")
    ap.add_argument("--horizon", type=int, default=5, help="forward return horizon in days")
    ap.add_argument("--plot-bars", type=int, default=750, help="days shown in the chart")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "bh_quickstart.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    unknown = [s for s in symbols if s not in PARAMS]
    if unknown:
        ap.error(f"unknown symbols {unknown}; choose from {sorted(PARAMS)}")

    results = {}
    rows = []
    for sym in symbols:
        close = load_daily_close(sym)
        res = run_bh(close, PARAMS[sym]["cf"], PARAMS[sym]["bh_form"])
        results[sym] = res
        rows.append(summarize(sym, res, args.horizon))

    table = pd.DataFrame(rows).set_index("symbol")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.2f}".format):
        print(table.to_string())
    print(
        "\nDescriptive statistics on one historical sample; not a backtest, "
        "no costs, not financial advice."
    )

    if not args.no_plot:
        plot(results, args.out, args.plot_bars)
        try:
            shown = args.out.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            shown = args.out.resolve().as_posix()
        print(f"chart saved to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
