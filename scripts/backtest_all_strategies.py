#!/usr/bin/env python3
"""
Batch backtest all registered strategies and rank by Sharpe ratio.

Usage:
    python backtest_all_strategies.py --synthetic --n 500
    python backtest_all_strategies.py --data prices.csv --save-results ranking.csv
    python backtest_all_strategies.py --synthetic --category momentum --verbose
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_all_strategies")


# ---------------------------------------------------------------------------
# Full strategy registry with default parameters
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    # ---- Momentum / Trend Following ----
    "DualMovingAverage": {
        "module": "strategies.momentum.trend_following",
        "class": "DualMovingAverage",
        "kwargs": {"fast": 20, "slow": 50},
        "category": "momentum",
    },
    "TripleMovingAverage": {
        "module": "strategies.momentum.trend_following",
        "class": "TripleMovingAverage",
        "kwargs": {"fast": 10, "medium": 30, "slow": 60},
        "category": "momentum",
    },
    "TurtleSystem": {
        "module": "strategies.momentum.trend_following",
        "class": "TurtleSystem",
        "kwargs": {"entry_period": 20, "exit_period": 10, "atr_period": 14},
        "category": "momentum",
    },
    "KeltnerBreakout": {
        "module": "strategies.momentum.trend_following",
        "class": "KeltnerBreakout",
        "kwargs": {"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
        "category": "momentum",
    },
    "DonchianBreakout": {
        "module": "strategies.momentum.trend_following",
        "class": "DonchianBreakout",
        "kwargs": {"entry_period": 20, "exit_period": 10},
        "category": "momentum",
    },
    "AdaptiveTrendFollowing": {
        "module": "strategies.momentum.trend_following",
        "class": "AdaptiveTrendFollowing",
        "kwargs": {"fast_min": 5, "fast_max": 30, "slow_min": 30, "slow_max": 100},
        "category": "momentum",
    },
    "TimeSeriesMomentum": {
        "module": "strategies.momentum.momentum",
        "class": "TimeSeriesMomentum",
        "kwargs": {"lookback": 252, "holding_period": 21},
        "category": "momentum",
    },
    "DualMomentum": {
        "module": "strategies.momentum.momentum",
        "class": "DualMomentum",
        "kwargs": {"lookback": 126},
        "category": "momentum",
    },
    "RiskAdjustedMomentum": {
        "module": "strategies.momentum.momentum",
        "class": "RiskAdjustedMomentum",
        "kwargs": {"lookback": 126, "vol_window": 21},
        "category": "momentum",
    },
    "SkewnessAdjustedMomentum": {
        "module": "strategies.momentum.momentum",
        "class": "SkewnessAdjustedMomentum",
        "kwargs": {"lookback": 126, "skew_window": 63},
        "category": "momentum",
    },
    "ForwardRateCarry": {
        "module": "strategies.momentum.carry",
        "class": "ForwardRateCarry",
        "kwargs": {},
        "category": "carry",
    },
    "RollYieldCarry": {
        "module": "strategies.momentum.carry",
        "class": "RollYieldCarry",
        "kwargs": {},
        "category": "carry",
    },
    "TermStructureCarry": {
        "module": "strategies.momentum.carry",
        "class": "TermStructureCarry",
        "kwargs": {},
        "category": "carry",
    },
    # ---- Mean Reversion ----
    "PairsTrading": {
        "module": "strategies.mean_reversion.stat_arb",
        "class": "PairsTrading",
        "kwargs": {"entry_z": 2.0, "exit_z": 0.5, "lookback": 63},
        "category": "mean_reversion",
    },
    "KalmanPairsTrading": {
        "module": "strategies.mean_reversion.stat_arb",
        "class": "KalmanPairsTrading",
        "kwargs": {"entry_z": 2.0, "exit_z": 0.5},
        "category": "mean_reversion",
    },
    "OUMeanReversion": {
        "module": "strategies.mean_reversion.stat_arb",
        "class": "OUMeanReversion",
        "kwargs": {"entry_z": 1.5, "exit_z": 0.3, "lookback": 63},
        "category": "mean_reversion",
    },
    "VIXMeanReversion": {
        "module": "strategies.mean_reversion.volatility_mean_reversion",
        "class": "VIXMeanReversion",
        "kwargs": {"lookback": 63, "entry_z": 1.5, "exit_z": 0.5},
        "category": "mean_reversion",
    },
    "VolatilityArbitrage": {
        "module": "strategies.mean_reversion.volatility_mean_reversion",
        "class": "VolatilityArbitrage",
        "kwargs": {},
        "category": "mean_reversion",
    },
    # ---- Volatility ----
    "VolatilityTargeting": {
        "module": "strategies.volatility.vol_targeting",
        "class": "VolatilityTargeting",
        "kwargs": {"target_vol": 0.10, "vol_window": 21},
        "category": "volatility",
    },
    "RiskParity": {
        "module": "strategies.volatility.vol_targeting",
        "class": "RiskParity",
        "kwargs": {"target_vol": 0.10},
        "category": "volatility",
    },
    "MaxDiversification": {
        "module": "strategies.volatility.vol_targeting",
        "class": "MaxDiversification",
        "kwargs": {},
        "category": "volatility",
    },
    "VariancePremiumCapture": {
        "module": "strategies.volatility.variance_premium",
        "class": "VariancePremiumCapture",
        "kwargs": {"rv_window": 21, "iv_lookback": 21},
        "category": "volatility",
    },
    "GARCHVolTrading": {
        "module": "strategies.volatility.regime_vol",
        "class": "GARCHVolTrading",
        "kwargs": {"p": 1, "q": 1},
        "category": "volatility",
    },
    # ---- Crypto ----
    "FundingRateArbitrage": {
        "module": "strategies.crypto.funding_rate",
        "class": "FundingRateArbitrage",
        "kwargs": {"threshold": 0.0001, "holding_period": 8},
        "category": "crypto",
    },
    "NVTRatio": {
        "module": "strategies.crypto.onchain",
        "class": "NVTRatio",
        "kwargs": {},
        "category": "crypto",
    },
    "MayerMultiple": {
        "module": "strategies.crypto.onchain",
        "class": "MayerMultiple",
        "kwargs": {},
        "category": "crypto",
    },
    # ---- Event Driven ----
    "EarningsSurprise": {
        "module": "strategies.event_driven.earnings",
        "class": "EarningsSurprise",
        "kwargs": {"lookback": 4, "n_long": 5, "n_short": 5},
        "category": "event_driven",
    },
    "EarningsMomentum": {
        "module": "strategies.event_driven.earnings",
        "class": "EarningsMomentum",
        "kwargs": {"min_streak": 2, "max_streak": 6},
        "category": "event_driven",
    },
    "FOMCDrift": {
        "module": "strategies.event_driven.macro_events",
        "class": "FOMCDrift",
        "kwargs": {"drift_period": 5, "entry_delay": 1},
        "category": "event_driven",
    },
    "NFPMomentum": {
        "module": "strategies.event_driven.macro_events",
        "class": "NFPMomentum",
        "kwargs": {"holding_period": 5, "z_threshold": 1.0},
        "category": "event_driven",
    },
    "CPIRegime": {
        "module": "strategies.event_driven.macro_events",
        "class": "CPIRegime",
        "kwargs": {"low_threshold": 2.0, "high_threshold": 4.0},
        "category": "event_driven",
    },
}

CATEGORIES = sorted(set(v["category"] for v in STRATEGY_REGISTRY.values()))


def add_srfm_lab_to_path():
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))


def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate rich synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    log_rets = 0.0003 + 0.015 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * rng.uniform(0.995, 1.005, n),
        "high": close * rng.uniform(1.000, 1.015, n),
        "low": close * rng.uniform(0.985, 1.000, n),
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        "funding_rate": rng.uniform(-0.0001, 0.0003, n),
        "vix": 15 + 5 * np.abs(rng.standard_normal(n)),
        "open_interest": rng.integers(10_000, 500_000, n).astype(float),
    }, index=idx)
    return df


def run_single_strategy(
    name: str,
    config: Dict,
    data: pd.DataFrame,
    timeout_sec: float = 60.0,
) -> Dict:
    """Run a single strategy backtest and return result dict."""
    import importlib
    import inspect

    record = {
        "strategy": name,
        "category": config["category"],
        "status": "failed",
        "error": None,
        "sharpe": np.nan,
        "total_return": np.nan,
        "cagr": np.nan,
        "max_drawdown": np.nan,
        "sortino": np.nan,
        "calmar": np.nan,
        "win_rate": np.nan,
        "n_trades": np.nan,
        "elapsed_sec": np.nan,
    }

    t0 = time.time()
    try:
        module = importlib.import_module(config["module"])
        cls = getattr(module, config["class"])

        # Filter kwargs to valid params
        sig = inspect.signature(cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in config["kwargs"].items() if k in valid_params}

        strategy = cls(**filtered)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = strategy.backtest(data)
            except TypeError:
                result = strategy.backtest(data["close"])

        for attr in ["sharpe", "total_return", "cagr", "max_drawdown",
                     "sortino", "calmar", "win_rate", "n_trades"]:
            val = getattr(result, attr, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                record[attr] = float(val)

        record["status"] = "ok"
    except Exception as e:
        record["error"] = str(e)[:120]
        logger.debug(f"{name} failed: {e}")

    record["elapsed_sec"] = time.time() - t0
    return record


def print_ranking(df: pd.DataFrame, top_n: int = 20):
    """Print strategy ranking table."""
    ok = df[df["status"] == "ok"].copy()
    failed = df[df["status"] != "ok"]

    ok_sorted = ok.sort_values("sharpe", ascending=False)

    print("\n" + "=" * 90)
    print(f"STRATEGY RANKING BY SHARPE RATIO  (top {min(top_n, len(ok_sorted))} of {len(ok_sorted)} succeeded)")
    print("=" * 90)
    display_cols = ["strategy", "category", "sharpe", "cagr", "max_drawdown",
                    "sortino", "calmar", "win_rate", "n_trades", "elapsed_sec"]
    display_cols = [c for c in display_cols if c in ok_sorted.columns]
    top = ok_sorted[display_cols].head(top_n)

    # Format numeric columns
    fmt_pct = ["cagr", "max_drawdown", "win_rate"]
    fmt_dec = ["sharpe", "sortino", "calmar"]
    for col in fmt_pct:
        if col in top.columns:
            top[col] = top[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    for col in fmt_dec:
        if col in top.columns:
            top[col] = top[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    if "n_trades" in top.columns:
        top["n_trades"] = top["n_trades"].map(lambda x: f"{int(x)}" if pd.notna(x) else "")
    if "elapsed_sec" in top.columns:
        top["elapsed_sec"] = top["elapsed_sec"].map(lambda x: f"{x:.2f}s" if pd.notna(x) else "")

    print(top.to_string(index=False))

    # Category summary
    print("\n" + "=" * 60)
    print("CATEGORY SUMMARY")
    print("=" * 60)
    cat_ok = df[df["status"] == "ok"].groupby("category")["sharpe"]
    cat_summary = cat_ok.agg(["count", "mean", "max"]).rename(
        columns={"count": "n_ok", "mean": "avg_sharpe", "max": "best_sharpe"}
    )
    print(cat_summary.round(4).to_string())

    if len(failed) > 0:
        print(f"\n{'=' * 60}")
        print(f"FAILED STRATEGIES ({len(failed)})")
        print("=" * 60)
        for _, row in failed.iterrows():
            print(f"  {row['strategy']:<40} {row.get('error', 'unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch backtest all SRFM lab strategies and rank by Sharpe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true",
                       help="Use synthetic OHLCV data.")
    group.add_argument("--data", type=str, help="Path to OHLCV CSV.")

    parser.add_argument("--n", type=int, default=500, help="Number of synthetic periods.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category", type=str, default="all",
                        choices=["all"] + CATEGORIES,
                        help="Filter to a specific strategy category.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top strategies to display.")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Save full results table to CSV.")
    parser.add_argument("--save-top", type=str, default=None,
                        help="Save top-N results to CSV.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    add_srfm_lab_to_path()

    # Load data
    if args.synthetic:
        logger.info(f"Generating synthetic data: n={args.n}")
        data = generate_synthetic_data(args.n, args.seed)
    else:
        data = pd.read_csv(args.data, index_col=0, parse_dates=True)
        logger.info(f"Loaded data: {data.shape}")

    # Filter strategies
    strategies_to_run = {
        name: cfg for name, cfg in STRATEGY_REGISTRY.items()
        if args.category == "all" or cfg["category"] == args.category
    }
    logger.info(f"Running {len(strategies_to_run)} strategies...")

    # Run all strategies
    records = []
    for i, (name, config) in enumerate(strategies_to_run.items()):
        logger.info(f"[{i+1}/{len(strategies_to_run)}] {name}")
        record = run_single_strategy(name, config, data)
        records.append(record)
        status = "OK" if record["status"] == "ok" else "FAIL"
        sharpe_str = f"Sharpe={record['sharpe']:.3f}" if pd.notna(record["sharpe"]) else ""
        logger.info(f"  [{status}] {sharpe_str} ({record['elapsed_sec']:.2f}s)")

    results_df = pd.DataFrame(records)

    # Print ranking
    print_ranking(results_df, args.top_n)

    n_ok = (results_df["status"] == "ok").sum()
    n_fail = (results_df["status"] != "ok").sum()
    print(f"\nTotal: {n_ok} succeeded, {n_fail} failed out of {len(results_df)}")
    elapsed_total = results_df["elapsed_sec"].sum()
    print(f"Total elapsed: {elapsed_total:.1f}s")

    # Save outputs
    if args.save_results:
        results_df.to_csv(args.save_results, index=False)
        logger.info(f"Full results saved to: {args.save_results}")

    if args.save_top:
        ok_df = results_df[results_df["status"] == "ok"].sort_values("sharpe", ascending=False)
        ok_df.head(args.top_n).to_csv(args.save_top, index=False)
        logger.info(f"Top-{args.top_n} results saved to: {args.save_top}")

    logger.info("Batch backtest complete.")


if __name__ == "__main__":
    main()
