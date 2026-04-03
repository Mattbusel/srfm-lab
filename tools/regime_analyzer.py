"""
regime_analyzer.py — Analyze LARSA market regimes across historical price data.

Computes EMA12/26/50/200, ATR, and ADX from raw OHLCV data, then runs
RegimeDetector (exact LARSA detect_regime() logic).

Usage:
    python tools/regime_analyzer.py --csv data/synthetic_ES_hourly.csv --ticker ES
    python tools/regime_analyzer.py --csv data/ES.csv --ticker ES --plot --save-csv
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, deque
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from srfm_core import MarketRegime
from regime import RegimeDetector


# --- Data loading ------------------------------------------------------------

def load_ohlcv(path: str) -> List[dict]:
    bars = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            def g(keys):
                for k in keys:
                    v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
                    if v:
                        return float(v)
                return None
            close  = g(["close", "Close", "CLOSE"])
            high   = g(["high",  "High",  "HIGH"])  or close
            low    = g(["low",   "Low",   "LOW"])   or close
            date   = (row.get("date") or row.get("Date") or
                      row.get("DATE") or row.get("time") or "")
            if close and close > 0:
                bars.append({"date": date, "high": high, "low": low, "close": close})
    return bars


# --- Indicator helpers -------------------------------------------------------

class EMA:
    def __init__(self, period: int):
        self.k    = 2.0 / (period + 1)
        self.val: Optional[float] = None

    def update(self, x: float) -> float:
        if self.val is None:
            self.val = x
        else:
            self.val = x * self.k + self.val * (1.0 - self.k)
        return self.val


class ATR:
    """Average True Range (Wilder smoothing, period=14)."""
    def __init__(self, period: int = 14):
        self.period    = period
        self.prev_close: Optional[float] = None
        self.val: Optional[float] = None
        self._buf: List[float] = []

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(high - low,
                     abs(high - self.prev_close),
                     abs(low  - self.prev_close))
        self.prev_close = close

        if self.val is None:
            self._buf.append(tr)
            if len(self._buf) >= self.period:
                self.val = sum(self._buf) / len(self._buf)
        else:
            self.val = (self.val * (self.period - 1) + tr) / self.period
        return self.val or tr


class ADX:
    """ADX (Wilder, period=14)."""
    def __init__(self, period: int = 14):
        self.period = period
        self.atr_   = ATR(period)
        self.prev_high: Optional[float] = None
        self.prev_low:  Optional[float] = None
        self._pos_dm_smooth = 0.0
        self._neg_dm_smooth = 0.0
        self._dx_buf: List[float] = []
        self.val: float = 0.0
        self._ready = False

    def update(self, high: float, low: float, close: float) -> float:
        atr = self.atr_.update(high, low, close)

        if self.prev_high is None:
            self.prev_high = high
            self.prev_low  = low
            return 0.0

        pos_dm = high - self.prev_high
        neg_dm = self.prev_low - low
        if pos_dm < 0: pos_dm = 0.0
        if neg_dm < 0: neg_dm = 0.0
        if pos_dm > neg_dm: neg_dm = 0.0
        elif neg_dm > pos_dm: pos_dm = 0.0
        else: pos_dm = neg_dm = 0.0

        k = 2.0 / (self.period + 1)
        self._pos_dm_smooth = pos_dm * k + self._pos_dm_smooth * (1.0 - k)
        self._neg_dm_smooth = neg_dm * k + self._neg_dm_smooth * (1.0 - k)

        if atr > 0:
            pdi = 100 * self._pos_dm_smooth / atr
            ndi = 100 * self._neg_dm_smooth / atr
            denom = pdi + ndi
            dx = 100 * abs(pdi - ndi) / denom if denom > 0 else 0.0
        else:
            dx = 0.0

        self._dx_buf.append(dx)
        if len(self._dx_buf) >= self.period:
            self.val = sum(self._dx_buf[-self.period:]) / self.period

        self.prev_high = high
        self.prev_low  = low
        return self.val


# --- Analysis ----------------------------------------------------------------

def analyze(bars: List[dict]) -> List[Tuple[str, MarketRegime, float]]:
    ema12  = EMA(12)
    ema26  = EMA(26)
    ema50  = EMA(50)
    ema200 = EMA(200)
    atr_   = ATR(14)
    adx_   = ADX(14)
    det    = RegimeDetector(atr_window=50)

    records = []
    for bar in bars:
        close = bar["close"]
        high  = bar["high"]
        low   = bar["low"]
        date  = bar["date"]

        e12  = ema12.update(close)
        e26  = ema26.update(close)
        e50  = ema50.update(close)
        e200 = ema200.update(close)
        atr  = atr_.update(high, low, close)
        adx  = adx_.update(high, low, close)

        regime, conf = det.update(close, e12, e26, e50, e200, adx, atr)
        records.append((date, regime, conf))

    return records


def print_summary(records: List[Tuple[str, MarketRegime, float]], ticker: str):
    regimes = [r for _, r, _ in records]
    counts  = Counter(regimes)
    total   = len(regimes)

    print(f"\n{'-'*52}")
    print(f"  Regime Summary -- {ticker}  ({total} bars)")
    print(f"{'-'*52}")
    for regime in MarketRegime:
        n   = counts.get(regime, 0)
        pct = 100.0 * n / total if total else 0.0
        bar = "#" * int(pct / 2)
        print(f"  {regime.name:<16} {n:>6} bars  {pct:>5.1f}%  {bar}")
    print(f"{'-'*52}\n")

    # Transitions
    transitions = []
    prev_r = None
    for date, regime, _ in records:
        if regime != prev_r:
            transitions.append((date, regime))
            prev_r = regime

    print(f"  Regime transitions ({len(transitions)} total):")
    for date, regime in transitions[:20]:
        print(f"    {date:<22} -> {regime.name}")
    if len(transitions) > 20:
        print(f"    ... and {len(transitions) - 20} more")
    print()


def save_csv(records: List[Tuple[str, MarketRegime, float]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "regime", "confidence"])
        for date, regime, conf in records:
            writer.writerow([date, regime.name, f"{conf:.4f}"])
    print(f"CSV saved -> {path}")


REGIME_COLORS = {
    MarketRegime.BULL:            "green",
    MarketRegime.BEAR:            "red",
    MarketRegime.SIDEWAYS:        "gray",
    MarketRegime.HIGH_VOLATILITY: "orange",
}


def plot_regimes(bars: List[dict], records: List[Tuple[str, MarketRegime, float]], ticker: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] matplotlib not installed.")
        return

    closes = [b["close"] for b in bars]
    n      = len(closes)
    xs     = list(range(n))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax1.plot(xs, closes, color="black", linewidth=0.7, label="Close")

    # Shade by regime
    prev_r, seg_start = None, 0
    for i, (_, regime, _) in enumerate(records):
        if regime != prev_r or i == n - 1:
            if prev_r is not None:
                ax1.axvspan(seg_start, i, alpha=0.15,
                            color=REGIME_COLORS.get(prev_r, "white"))
            seg_start = i
            prev_r = regime

    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r.name)
               for r in MarketRegime]
    ax1.legend(handles=patches, loc="upper left")
    ax1.set_title(f"Regime Analysis -- {ticker}", fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.2)

    regime_map = {
        MarketRegime.BULL: 3, MarketRegime.BEAR: 1,
        MarketRegime.SIDEWAYS: 2, MarketRegime.HIGH_VOLATILITY: 0,
    }
    rv = [regime_map.get(r, 2) for _, r, _ in records]
    ax2.fill_between(xs, rv, alpha=0.5, color="steelblue")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["HI_VOL", "BEAR", "SIDEWAYS", "BULL"])
    ax2.set_xlabel("Bar index")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out = f"results/regimes_{ticker}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Plot saved -> {out}")
    plt.show()


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRFM regime analyzer")
    parser.add_argument("--csv",      required=True)
    parser.add_argument("--ticker",   default="UNKNOWN")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.csv}...", end=" ")
    bars = load_ohlcv(args.csv)
    print(f"{len(bars)} bars")

    records = analyze(bars)
    print_summary(records, args.ticker)

    if args.save_csv:
        save_csv(records, f"results/regimes_{args.ticker}.csv")

    if args.plot:
        plot_regimes(bars, records, args.ticker)


if __name__ == "__main__":
    main()
