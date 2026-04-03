"""
walk_forward.py — Walk-forward validation for SRFM strategies.

Splits a date range into N (train, test) windows, runs lean backtest on each,
and reports in-sample vs out-of-sample performance.  This is the real overfitting
test — QC's built-in detector is helpful but WFA on non-overlapping OOS windows
is authoritative.

Usage:
    python tools/walk_forward.py strategies/larsa-v1 --start 2018-01-01 --end 2024-12-31 --windows 6
    python tools/walk_forward.py strategies/larsa-v1 --train-months 12 --test-months 6

Output:
    results/larsa-v1/wfa/summary.csv
    results/larsa-v1/wfa/summary.md
    results/larsa-v1/wfa/wfa_chart.png
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# --- Date utilities -----------------------------------------------------------

def generate_windows(
    start: date,
    end:   date,
    train_months: int,
    test_months:  int,
) -> List[Tuple[date, date, date, date]]:
    """
    Returns list of (train_start, train_end, test_start, test_end) tuples.
    Walk-forward: each window slides by one test period.
    """
    windows = []
    train_start = start
    while True:
        train_end  = train_start + relativedelta(months=train_months) - relativedelta(days=1)
        test_start = train_end  + relativedelta(days=1)
        test_end   = test_start + relativedelta(months=test_months)  - relativedelta(days=1)
        if test_end > end:
            break
        windows.append((train_start, train_end, test_start, test_end))
        train_start = test_start   # walk forward by one test period
    return windows


# --- LEAN parameter injection -------------------------------------------------

def patch_dates(src: str, dst: str, start: date, end: date):
    """Patch SetStartDate / SetEndDate in a LEAN main.py."""
    with open(src) as f:
        code = f.read()
    code = re.sub(
        r'self\.set_start_date\([^)]+\)',
        f'self.set_start_date({start.year}, {start.month}, {start.day})',
        code,
    )
    code = re.sub(
        r'self\.set_end_date\([^)]+\)',
        f'self.set_end_date({end.year}, {end.month}, {end.day})',
        code,
    )
    with open(dst, 'w') as f:
        f.write(code)


def run_backtest(strategy_dir: str, output_dir: str) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ['lean', 'backtest', strategy_dir, '--output', output_dir],
        capture_output=True, text=True, timeout=1200,
    )
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[:300]}", file=sys.stderr)
        return None
    for name in ['result.json', 'backtest-results.json']:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return p
    jsons = list(Path(output_dir).glob('*.json'))
    return str(jsons[0]) if jsons else None


def extract_metrics(result_json: str) -> Dict:
    try:
        with open(result_json) as f:
            data = json.load(f)
        stats  = data.get('TotalPerformance', {}).get('PortfolioStatistics', {})
        trades = data.get('TotalPerformance', {}).get('TradeStatistics', {})
        def g(d, k):
            v = d.get(k)
            if isinstance(v, str):
                v = v.replace('%', '').strip()
                try: return float(v)
                except: return None
            return v
        return {
            'total_return': g(stats, 'TotalNetProfit'),
            'cagr':         g(stats, 'CompoundingAnnualReturn'),
            'sharpe':       g(stats, 'SharpeRatio'),
            'sortino':      g(stats, 'SortinoRatio'),
            'max_dd':       g(stats, 'Drawdown'),
            'win_rate':     g(stats, 'WinRate'),
            'n_trades':     g(trades, 'NumberOfTrades'),
        }
    except Exception as e:
        print(f"  [WARN] metrics extraction failed: {e}", file=sys.stderr)
        return {}


# --- Main ---------------------------------------------------------------------

def main():
    # Optional dependency check
    try:
        from dateutil.relativedelta import relativedelta as _
    except ImportError:
        print("pip install python-dateutil")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Walk-forward analysis')
    parser.add_argument('strategy',       help='Strategy directory')
    parser.add_argument('--start',        default='2018-01-01')
    parser.add_argument('--end',          default='2024-12-31')
    parser.add_argument('--train-months', type=int, default=12)
    parser.add_argument('--test-months',  type=int, default=6)
    parser.add_argument('--windows',      type=int, default=None,
                        help='Override: total windows to use (auto-computes train/test split)')
    parser.add_argument('--no-plot',      action='store_true')
    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').date()
    end   = datetime.strptime(args.end,   '%Y-%m-%d').date()

    windows = generate_windows(start, end, args.train_months, args.test_months)
    if args.windows:
        windows = windows[:args.windows]

    strategy_name = Path(args.strategy).name
    out_base      = os.path.join('results', strategy_name, 'wfa')
    src_main      = os.path.join(args.strategy, 'main.py')

    print(f"\n=== Walk-Forward Analysis: {strategy_name} ===")
    print(f"Windows: {len(windows)} × ({args.train_months}m train / {args.test_months}m test)")
    print()

    rows = []
    for i, (ts, te, os_s, os_e) in enumerate(windows):
        print(f"Window {i+1}/{len(windows)}: train {ts}->{te}  test {os_s}->{os_e}")

        with tempfile.TemporaryDirectory() as tmp:
            # In-sample
            is_dir = os.path.join(tmp, 'is')
            shutil.copytree(args.strategy, is_dir)
            patch_dates(src_main, os.path.join(is_dir, 'main.py'), ts, te)
            is_out = os.path.join(out_base, f'w{i+1}_is')
            is_result = run_backtest(is_dir, is_out)
            is_m = extract_metrics(is_result) if is_result else {}

            # Out-of-sample
            oos_dir = os.path.join(tmp, 'oos')
            shutil.copytree(args.strategy, oos_dir)
            patch_dates(src_main, os.path.join(oos_dir, 'main.py'), os_s, os_e)
            oos_out = os.path.join(out_base, f'w{i+1}_oos')
            oos_result = run_backtest(oos_dir, oos_out)
            oos_m = extract_metrics(oos_result) if oos_result else {}

        row = {
            'window':        i + 1,
            'train_start':   str(ts),  'train_end': str(te),
            'test_start':    str(os_s), 'test_end':  str(os_e),
            'is_return':     is_m.get('total_return'),
            'is_sharpe':     is_m.get('sharpe'),
            'oos_return':    oos_m.get('total_return'),
            'oos_sharpe':    oos_m.get('sharpe'),
            'oos_max_dd':    oos_m.get('max_dd'),
        }
        rows.append(row)
        status = 'OK' if oos_result else 'FAILED'
        oos_r  = f"{oos_m.get('total_return', 0.0):.1%}" if oos_m.get('total_return') else 'N/A'
        is_r   = f"{is_m.get('total_return', 0.0):.1%}"  if is_m.get('total_return')  else 'N/A'
        print(f"  [{status}] IS={is_r}  OOS={oos_r}")

    # Save CSV
    os.makedirs(out_base, exist_ok=True)
    csv_path = os.path.join(out_base, 'summary.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV -> {csv_path}")

    # Markdown summary
    _write_md_summary(rows, os.path.join(out_base, 'summary.md'), strategy_name)

    # Plot
    if not args.no_plot:
        _plot_wfa(rows, os.path.join(out_base, 'wfa_chart.png'), strategy_name)


def _write_md_summary(rows: list, path: str, name: str):
    lines = [
        f"# Walk-Forward Analysis — {name}\n",
        "| Window | IS Return | IS Sharpe | OOS Return | OOS Sharpe | OOS MaxDD |",
        "|--------|-----------|-----------|------------|------------|-----------|",
    ]
    for r in rows:
        def fmt(v): return f"{v:.2f}" if isinstance(v, float) else "—"
        lines.append(
            f"| {r['window']} | {fmt(r.get('is_return'))} | {fmt(r.get('is_sharpe'))} "
            f"| {fmt(r.get('oos_return'))} | {fmt(r.get('oos_sharpe'))} | {fmt(r.get('oos_max_dd'))} |"
        )
    oos_returns = [r['oos_return'] for r in rows if isinstance(r.get('oos_return'), float)]
    if oos_returns:
        avg = sum(oos_returns) / len(oos_returns)
        pos = sum(1 for r in oos_returns if r > 0)
        lines += [
            "",
            f"**OOS Summary:** avg return = {avg:.2f}  |  "
            f"positive windows = {pos}/{len(oos_returns)} ({pos/len(oos_returns):.0%})",
        ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"MD  -> {path}")


def _plot_wfa(rows: list, path: str, name: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    xs      = [r['window'] for r in rows]
    is_ret  = [r.get('is_return')  or 0 for r in rows]
    oos_ret = [r.get('oos_return') or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([x - 0.2 for x in xs], is_ret,  width=0.4, label='In-Sample',     color='steelblue', alpha=0.8)
    ax.bar([x + 0.2 for x in xs], oos_ret, width=0.4, label='Out-of-Sample', color='tomato',    alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Window')
    ax.set_ylabel('Return')
    ax.set_title(f'Walk-Forward Analysis — {name}', fontweight='bold')
    ax.set_xticks(xs)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"PNG -> {path}")


if __name__ == '__main__':
    main()
