"""
kelly_sizer.py — Kelly criterion and fractional Kelly position sizing.

The Kelly criterion gives the mathematically optimal bet fraction to
maximize long-run geometric growth. For trading: f* = (p*b - q) / b
where p=win rate, q=1-p, b=avg win / avg loss.

Usage:
    python tools/kelly_sizer.py  # reads research/trade_analysis_data.json
    echo "0.55 1.34" | python tools/kelly_sizer.py --pipe  # p=0.55, ratio=1.34
    python tools/kelly_sizer.py --by-type  # kelly for solo vs convergence separately
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

TRADE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "research", "trade_analysis_data.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---------------------------------------------------------------------------
# LARSA hardcoded forensic stats (from forensics analysis)
# Used when trade_analysis_data.json is not available
# ---------------------------------------------------------------------------
LARSA_STATS = {
    "all": {
        "n_trades":     263,
        "win_rate":     0.544,
        "avg_win":      70463.0,
        "avg_loss":    -59785.0,
    },
    "convergence": {
        "n_trades":     47,
        "win_rate":     0.745,
        "avg_win":      112450.0,
        "avg_loss":    -38200.0,
    },
    "solo": {
        "n_trades":     216,
        "win_rate":     0.500,
        "avg_win":      58300.0,
        "avg_loss":    -61400.0,
    },
}


# ---------------------------------------------------------------------------
# Kelly calculation
# ---------------------------------------------------------------------------

def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """
    Kelly: f* = (p*b - q) / b
    p = win_rate, q = 1-p, b = win/loss ratio
    """
    p = win_rate
    q = 1.0 - p
    b = win_loss_ratio
    if b <= 0:
        return 0.0
    f = (p * b - q) / b
    return max(0.0, f)


def simulate_kelly_equity(trades: List[float], kelly_f: float,
                          initial_equity: float = 1_000_000.0) -> List[float]:
    """
    Simulate equity curve using Kelly sizing.
    trades: list of returns as fractions (positive=win, negative=loss)
    kelly_f: fraction of portfolio per trade
    """
    equity = [initial_equity]
    pv = initial_equity
    for ret in trades:
        if ret > 0:
            pv *= (1.0 + kelly_f * ret)
        else:
            pv *= (1.0 + kelly_f * ret)  # ret is negative, reduces equity
        equity.append(pv)
    return equity


def theoretical_cagr(win_rate: float, win_loss_ratio: float,
                     kelly_f: float, n_per_year: float = 52.0) -> float:
    """
    Geometric growth rate per trade at fraction f.
    G = p * ln(1 + f*b) + q * ln(1 - f)
    Annual CAGR = exp(G * n_per_year) - 1
    """
    p = win_rate
    q = 1.0 - p
    b = win_loss_ratio
    if kelly_f <= 0 or kelly_f >= 1.0:
        return 0.0
    try:
        g_per_trade = p * math.log(1.0 + kelly_f * b) + q * math.log(1.0 - kelly_f)
    except ValueError:
        return 0.0
    return math.exp(g_per_trade * n_per_year) - 1.0


def compute_actual_cagr(equity_series: List[float], n_per_year: float = 52.0,
                        n_trades: int = 0) -> float:
    if len(equity_series) < 2 or equity_series[0] <= 0:
        return 0.0
    total_ret = equity_series[-1] / equity_series[0]
    n = n_trades if n_trades > 0 else len(equity_series) - 1
    years = n / n_per_year
    if years <= 0:
        return 0.0
    return total_ret ** (1.0 / years) - 1.0


def max_drawdown(equity_series: List[float]) -> float:
    peak = equity_series[0]
    mdd = 0.0
    for v in equity_series:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-9)
        if dd > mdd:
            mdd = dd
    return mdd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trade_data() -> Optional[Dict]:
    if os.path.exists(TRADE_DATA_PATH):
        with open(TRADE_DATA_PATH) as f:
            return json.load(f)
    return None


def stats_from_data(trade_data: Dict, trade_type: Optional[str] = None) -> Dict:
    """Extract win rate and avg win/loss from trade data."""
    trades = trade_data.get("trades", [])
    if trade_type:
        trades = [t for t in trades if t.get("type", "").lower() == trade_type.lower()]

    wins = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
    losses = [t["pnl"] for t in trades if t.get("pnl", 0) <= 0]
    n = len(trades)
    if n == 0:
        return {}

    return {
        "n_trades":  n,
        "win_rate":  len(wins) / n,
        "avg_win":   float(np.mean(wins)) if wins else 0.0,
        "avg_loss":  float(np.mean(losses)) if losses else 0.0,
    }


# ---------------------------------------------------------------------------
# Kelly analysis section
# ---------------------------------------------------------------------------

def print_kelly_section(label: str, stats: Dict, portfolio_value: float = 1_000_000.0,
                        actual_cagr: Optional[float] = None) -> None:
    n = stats["n_trades"]
    wr = stats["win_rate"]
    avg_w = stats["avg_win"]
    avg_l = abs(stats["avg_loss"])

    if avg_l < 1e-6:
        print(f"\n{label}: insufficient loss data")
        return

    wl_ratio = avg_w / avg_l
    full_k = kelly_fraction(wr, wl_ratio)
    half_k = full_k / 2.0
    quarter_k = full_k / 4.0

    n_per_year = 52.0
    full_cagr = theoretical_cagr(wr, wl_ratio, full_k, n_per_year)
    half_cagr = theoretical_cagr(wr, wl_ratio, half_k, n_per_year)

    print(f"\n{label} ({n} wells):")
    print(f"  Win Rate:           {wr*100:.1f}%")
    print(f"  Avg Win:           ${avg_w:,.0f}")
    print(f"  Avg Loss:         -${avg_l:,.0f}")
    print(f"  Win/Loss Ratio:     {wl_ratio:.3f}")
    print()
    print(f"  Full Kelly:         {full_k*100:5.1f}%  of portfolio per trade")
    print(f"  Half Kelly:         {half_k*100:5.1f}%  (recommended — reduces variance)")
    print(f"  Quarter Kelly:      {quarter_k*100:5.1f}%  (conservative)")
    print()
    print(f"  Theoretical max CAGR at Full Kelly: {full_cagr*100:.1f}%")
    print(f"  Theoretical CAGR at Half Kelly:     {half_cagr*100:.1f}%")
    if actual_cagr is not None:
        print(f"  Actual CAGR achieved: {actual_cagr*100:.1f}%")
        if full_cagr > 0:
            efficiency = actual_cagr / full_cagr
            print(f"  Efficiency: {efficiency*100:.1f}% of Kelly-optimal")

    return full_k, half_k


def run_kelly_analysis(by_type: bool = False) -> None:
    trade_data = load_trade_data()

    # Decide which stats to use
    if trade_data and "trades" in trade_data:
        all_stats = stats_from_data(trade_data)
        conv_stats = stats_from_data(trade_data, "convergence")
        solo_stats = stats_from_data(trade_data, "solo")
        # Fall back to hardcoded if empty
        if not all_stats:
            all_stats = LARSA_STATS["all"]
        if not conv_stats:
            conv_stats = LARSA_STATS["convergence"]
        if not solo_stats:
            solo_stats = LARSA_STATS["solo"]
    else:
        all_stats = LARSA_STATS["all"]
        conv_stats = LARSA_STATS["convergence"]
        solo_stats = LARSA_STATS["solo"]
        if not trade_data:
            print(f"Note: {TRADE_DATA_PATH} not found — using LARSA forensic constants.")

    print("KELLY CRITERION ANALYSIS — LARSA v1")
    print("=" * 38)

    # Estimated actual CAGR from equity context
    actual_cagr = 0.214  # 21.4% — from RESULTS.md / known backtest

    result_all = print_kelly_section("ALL TRADES", all_stats, actual_cagr=actual_cagr)
    all_full_k = result_all[0] if isinstance(result_all, tuple) else 0.0

    print()
    print("CONVERGENCE WELLS ({n}):".format(n=conv_stats["n_trades"]))
    conv_wr = conv_stats["win_rate"]
    conv_wl = abs(conv_stats["avg_win"]) / abs(conv_stats["avg_loss"])
    conv_full_k = kelly_fraction(conv_wr, conv_wl)
    conv_half_k = conv_full_k / 2.0
    print(f"  Win Rate:           {conv_wr*100:.1f}%")
    print(f"  Full Kelly:         {conv_full_k*100:5.1f}%  <- much higher! Size up massively on convergence")
    print(f"  Half Kelly:         {conv_half_k*100:5.1f}%")

    print()
    print("SOLO WELLS ({n}):".format(n=solo_stats["n_trades"]))
    solo_wr = solo_stats["win_rate"]
    solo_wl = abs(solo_stats["avg_win"]) / abs(solo_stats["avg_loss"])
    solo_full_k = kelly_fraction(solo_wr, solo_wl)
    solo_half_k = solo_full_k / 2.0
    print(f"  Win Rate:           {solo_wr*100:.1f}%  <- coin flip")
    print(f"  Full Kelly:         {solo_full_k*100:5.1f}%  <- almost zero! Don't size up on solo")
    print(f"  Half Kelly:         {solo_half_k*100:5.1f}%")

    # Key insight
    print()
    if solo_full_k > 1e-6:
        ratio = conv_full_k / solo_full_k
        print(f">>> KEY INSIGHT: Kelly says convergence should be {ratio:.1f}x larger than solo!")
    else:
        print(">>> KEY INSIGHT: Kelly says solo BH has near-zero edge (50% WR).")
    print("    Current strategy uses ~3x ratio (0.65 conv / 0.20 solo in v6).")
    print("    Kelly-optimal ratio would be much more extreme.")

    # Equity simulation comparison
    print()
    print("EQUITY SIMULATION (Kelly vs Actual Sizing):")
    _simulate_and_compare(all_stats, conv_stats, solo_stats)


def _simulate_and_compare(all_stats: Dict, conv_stats: Dict, solo_stats: Dict) -> None:
    """Simulate equity curves at different Kelly fractions."""
    rng = np.random.default_rng(42)
    n = all_stats["n_trades"]
    wr = all_stats["win_rate"]
    avg_w = all_stats["avg_win"]
    avg_l = abs(all_stats["avg_loss"])
    wl_ratio = avg_w / avg_l
    full_k = kelly_fraction(wr, wl_ratio)
    actual_f = 0.043  # approximate actual average sizing used

    # Simulate trade outcomes
    wins = rng.random(n) < wr
    trade_returns = np.where(wins, avg_w / 1_000_000.0, -avg_l / 1_000_000.0)

    # Three scenarios
    eq_full = [1.0]
    eq_half = [1.0]
    eq_actual = [1.0]
    pf, ph, pa = 1.0, 1.0, 1.0
    for ret in trade_returns:
        sign = 1 if ret > 0 else -1
        magnitude = abs(ret)
        pf *= (1.0 + sign * full_k * magnitude * pf)  # reinvest
        ph *= (1.0 + sign * (full_k / 2.0) * magnitude)
        pa *= (1.0 + sign * actual_f * magnitude)
        eq_full.append(pf)
        eq_half.append(ph)
        eq_actual.append(pa)

    def _dd(eq):
        pk = eq[0]; md = 0.0
        for v in eq:
            if v > pk: pk = v
            dd = (pk - v) / (pk + 1e-9)
            if dd > md: md = dd
        return md

    print(f"  Full Kelly  ({full_k*100:.1f}%): final={eq_full[-1]:.2f}x  "
          f"maxDD={_dd(eq_full)*100:.1f}%")
    print(f"  Half Kelly  ({full_k/2*100:.1f}%): final={eq_half[-1]:.2f}x  "
          f"maxDD={_dd(eq_half)*100:.1f}%")
    print(f"  Actual size ({actual_f*100:.1f}%): final={eq_actual[-1]:.2f}x  "
          f"maxDD={_dd(eq_actual)*100:.1f}%")
    print()
    print("  Note: Full Kelly maximizes growth but drawdowns are severe.")
    print("  Half Kelly is the practitioner's standard — half the variance.")


# ---------------------------------------------------------------------------
# Pipe mode: read p and ratio from stdin
# ---------------------------------------------------------------------------

def run_pipe_mode() -> None:
    line = sys.stdin.readline().strip()
    parts = line.split()
    if len(parts) < 2:
        print("Usage: echo '<win_rate> <win_loss_ratio>' | kelly_sizer.py --pipe")
        sys.exit(1)
    p = float(parts[0])
    b = float(parts[1])
    f_full = kelly_fraction(p, b)
    f_half = f_full / 2.0
    f_qtr = f_full / 4.0
    cagr = theoretical_cagr(p, b, f_full)
    print(f"Kelly Analysis: p={p:.3f}  b={b:.3f}")
    print(f"  Full Kelly:    {f_full*100:.2f}%")
    print(f"  Half Kelly:    {f_half*100:.2f}%")
    print(f"  Quarter Kelly: {f_qtr*100:.2f}%")
    print(f"  Theoretical CAGR (Full, 52 trades/yr): {cagr*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kelly criterion position sizing for LARSA")
    parser.add_argument("--pipe", action="store_true",
                        help="Read 'p ratio' from stdin and compute Kelly")
    parser.add_argument("--by-type", action="store_true",
                        help="Break down Kelly by solo vs convergence")
    args = parser.parse_args()

    if args.pipe:
        run_pipe_mode()
        return

    run_kelly_analysis(by_type=args.by_type)


if __name__ == "__main__":
    main()
