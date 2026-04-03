"""
monte_carlo.py — Monte Carlo analysis of LARSA trade distribution.

Takes a list of trade returns (from a LEAN backtest result JSON) and
resamples them 10,000 times to build the outcome distribution:
  - Median terminal return
  - 5th / 95th percentile
  - Probability of ruin (equity < 50% of start)
  - Probability of > 100% return
  - Probability of > 200% return

Usage:
    python tools/monte_carlo.py results/larsa-v1/20240101/result.json
    python tools/monte_carlo.py results/larsa-v1/20240101/result.json --sims 50000 --ruin-threshold 0.5

Output:
    Console summary
    results/larsa-v1/mc/mc_distribution.png
    results/larsa-v1/mc/mc_summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


# --- Trade extraction ---------------------------------------------------------

def extract_trade_returns(result_json: str) -> List[float]:
    """Extract per-trade returns from LEAN result JSON."""
    with open(result_json) as f:
        data = json.load(f)

    trades = data.get('TotalPerformance', {}).get('ClosedTrades', [])
    if not trades:
        # Try alternate structure
        trades = data.get('AlgorithmConfiguration', {}).get('Trades', [])

    returns = []
    for t in trades:
        profit = t.get('ProfitLoss') or t.get('NetProfit') or t.get('profit_loss') or 0.0
        entry  = t.get('EntryPrice') or t.get('entry_price') or 1.0
        if entry and entry > 0:
            returns.append(float(profit) / float(entry))

    # If no closed trades found, try extracting from equity curve
    if not returns:
        charts = data.get('Charts', {})
        eq     = charts.get('Strategy Equity', {}).get('Series', {}).get('Equity', {}).get('Values', [])
        if len(eq) >= 2:
            vals = [p['y'] for p in eq if 'y' in p]
            # Use daily returns as proxy
            returns = [
                (vals[i] - vals[i-1]) / vals[i-1]
                for i in range(1, len(vals))
                if vals[i-1] > 0
            ]

    return returns


# --- Monte Carlo engine -------------------------------------------------------

def run_monte_carlo(
    returns:         List[float],
    n_sims:          int   = 10_000,
    ruin_threshold:  float = 0.50,
    initial_equity:  float = 1.0,
    rng_seed:        int   = 42,
) -> dict:
    """
    Bootstrap resample the trade sequence N times.
    Returns dict of statistics.
    """
    if not returns:
        raise ValueError("No trade returns provided.")

    rng       = np.random.default_rng(rng_seed)
    n_trades  = len(returns)
    arr       = np.array(returns, dtype=float)

    terminal  = np.zeros(n_sims)
    max_dds   = np.zeros(n_sims)

    for i in range(n_sims):
        # Resample with replacement
        sample     = rng.choice(arr, size=n_trades, replace=True)
        equity     = np.cumprod(1.0 + sample) * initial_equity
        terminal[i] = equity[-1]

        # Max drawdown for this sim
        peak  = np.maximum.accumulate(np.concatenate([[initial_equity], equity]))
        dds   = (peak[1:] - equity) / peak[1:]
        max_dds[i] = dds.max()

    p_ruin   = float(np.mean(terminal < initial_equity * ruin_threshold))
    p_100    = float(np.mean(terminal > initial_equity * 2.0))
    p_200    = float(np.mean(terminal > initial_equity * 3.0))
    p_274    = float(np.mean(terminal > initial_equity * 3.74))   # beat baseline

    return {
        'n_sims':           n_sims,
        'n_trades':         n_trades,
        'median_terminal':  float(np.median(terminal)),
        'mean_terminal':    float(np.mean(terminal)),
        'p5_terminal':      float(np.percentile(terminal, 5)),
        'p25_terminal':     float(np.percentile(terminal, 25)),
        'p75_terminal':     float(np.percentile(terminal, 75)),
        'p95_terminal':     float(np.percentile(terminal, 95)),
        'p_ruin':           p_ruin,
        'p_100pct_return':  p_100,
        'p_200pct_return':  p_200,
        'p_274pct_return':  p_274,
        'avg_max_dd':       float(np.mean(max_dds)),
        'p95_max_dd':       float(np.percentile(max_dds, 95)),
        'terminal_array':   terminal.tolist(),
    }


# --- Output -------------------------------------------------------------------

def print_summary(stats: dict):
    sep = "-" * 55
    print(f"\n{sep}")
    print(f"  Monte Carlo Summary  ({stats['n_sims']:,} sims, {stats['n_trades']} trades)")
    print(sep)
    print(f"  Median terminal return : {stats['median_terminal'] - 1.0:+.1%}")
    print(f"  Mean terminal return   : {stats['mean_terminal']   - 1.0:+.1%}")
    print(f"  5th  percentile        : {stats['p5_terminal']     - 1.0:+.1%}")
    print(f"  95th percentile        : {stats['p95_terminal']    - 1.0:+.1%}")
    print(sep)
    print(f"  P(ruin  < 50%)         : {stats['p_ruin']          :.1%}")
    print(f"  P(return > 100%)       : {stats['p_100pct_return'] :.1%}")
    print(f"  P(return > 200%)       : {stats['p_200pct_return'] :.1%}")
    print(f"  P(return > 274%)       : {stats['p_274pct_return'] :.1%}  <- beat baseline")
    print(f"  {sep}")
    print(f"  Avg max drawdown       : {stats['avg_max_dd']      :.1%}")
    print(f"  P95 max drawdown       : {stats['p95_max_dd']      :.1%}")
    print(sep)


def plot_distribution(stats: dict, path: str, strategy_name: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[WARN] matplotlib not installed.")
        return

    terminal = np.array(stats['terminal_array'])
    returns  = terminal - 1.0   # convert to return %

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(returns, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax1.axvline(stats['median_terminal'] - 1.0, color='orange', linewidth=2, label=f"Median {stats['median_terminal']-1:.1%}")
    ax1.axvline(stats['p5_terminal']    - 1.0, color='red',    linewidth=1.5, linestyle='--', label=f"P5 {stats['p5_terminal']-1:.1%}")
    ax1.axvline(stats['p95_terminal']   - 1.0, color='green',  linewidth=1.5, linestyle='--', label=f"P95 {stats['p95_terminal']-1:.1%}")
    ax1.axvline(0, color='black', linewidth=1.0)
    ax1.set_title(f'Return Distribution — {strategy_name}', fontweight='bold')
    ax1.set_xlabel('Terminal Return')
    ax1.set_ylabel('Frequency')
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # CDF
    sorted_r = np.sort(returns)
    cdf      = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax2.plot(sorted_r, cdf, color='steelblue', linewidth=1.5)
    ax2.axvline(0, color='black', linewidth=1.0)
    ax2.axhline(stats['p_ruin'],         color='red',   linestyle='--', alpha=0.6, label=f"P(ruin)={stats['p_ruin']:.1%}")
    ax2.axhline(stats['p_100pct_return'],color='green', linestyle='--', alpha=0.6, label=f"P(>100%)={stats['p_100pct_return']:.1%}")
    ax2.set_title('CDF of Terminal Returns', fontweight='bold')
    ax2.set_xlabel('Terminal Return')
    ax2.set_ylabel('Cumulative Probability')
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"PNG -> {path}")


# --- Synthetic fallback (when no result JSON available) -----------------------

def synthetic_trade_returns(
    n_trades: int = 200,
    win_rate: float = 0.58,
    avg_win: float = 0.018,
    avg_loss: float = -0.009,
    seed: int = 42,
) -> List[float]:
    """
    Generate synthetic trade returns matching LARSA's approximate statistics.
    Used when running without a completed backtest.
    win_rate=0.58, avg_win≈1.8%, avg_loss≈-0.9% -> profit factor ≈ 2.3
    """
    rng     = np.random.default_rng(seed)
    wins    = rng.normal(avg_win,  abs(avg_win)  * 0.5, int(n_trades * win_rate))
    losses  = rng.normal(avg_loss, abs(avg_loss) * 0.5, n_trades - len(wins))
    trades  = np.concatenate([wins, losses])
    rng.shuffle(trades)
    return trades.tolist()


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo analysis')
    parser.add_argument('result_json', nargs='?', help='LEAN result JSON (optional)')
    parser.add_argument('--sims',           type=int,   default=10_000)
    parser.add_argument('--ruin-threshold', type=float, default=0.50)
    parser.add_argument('--synthetic',      action='store_true',
                        help='Use synthetic LARSA-like trade distribution instead of real data')
    parser.add_argument('--no-plot',        action='store_true')
    args = parser.parse_args()

    if args.synthetic or not args.result_json:
        print("[INFO] Using synthetic LARSA-like trade distribution.")
        returns  = synthetic_trade_returns(n_trades=200, win_rate=0.58)
        out_base = os.path.join('results', 'larsa-v1', 'mc')
        name     = 'larsa-v1 (synthetic)'
    else:
        returns  = extract_trade_returns(args.result_json)
        strategy = Path(args.result_json).parts
        name     = strategy[1] if len(strategy) > 1 else 'strategy'
        out_base = os.path.join('results', name, 'mc')

    if not returns:
        print("[ERROR] No trades found. Use --synthetic for demo run.")
        sys.exit(1)

    print(f"Loaded {len(returns)} trade returns. Running {args.sims:,} simulations...")
    stats = run_monte_carlo(returns, n_sims=args.sims, ruin_threshold=args.ruin_threshold)
    print_summary(stats)

    # Save JSON (exclude large array)
    os.makedirs(out_base, exist_ok=True)
    save_stats = {k: v for k, v in stats.items() if k != 'terminal_array'}
    json_path  = os.path.join(out_base, 'mc_summary.json')
    with open(json_path, 'w') as f:
        json.dump(save_stats, f, indent=2)
    print(f"JSON -> {json_path}")

    if not args.no_plot:
        plot_distribution(stats, os.path.join(out_base, 'mc_distribution.png'), name)


if __name__ == '__main__':
    main()
