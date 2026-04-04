#!/usr/bin/env python3
"""
CLI script to run any registered strategy on historical price data.

Usage:
    python run_strategy.py --strategy DualMovingAverage --fast 20 --slow 50
    python run_strategy.py --strategy TurtleSystem --atr 14 --entry 20
    python run_strategy.py --strategy TimeSeriesMomentum --lookback 252 --data prices.csv

Outputs:
    - Performance summary table to stdout
    - Optional equity curve CSV/PNG
"""

import argparse
import importlib
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_strategy")

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    # Momentum
    "DualMovingAverage": ("strategies.momentum.trend_following", "DualMovingAverage"),
    "TripleMovingAverage": ("strategies.momentum.trend_following", "TripleMovingAverage"),
    "TurtleSystem": ("strategies.momentum.trend_following", "TurtleSystem"),
    "KeltnerBreakout": ("strategies.momentum.trend_following", "KeltnerBreakout"),
    "DonchianBreakout": ("strategies.momentum.trend_following", "DonchianBreakout"),
    "AdaptiveTrendFollowing": ("strategies.momentum.trend_following", "AdaptiveTrendFollowing"),
    "TimeSeriesMomentum": ("strategies.momentum.momentum", "TimeSeriesMomentum"),
    "DualMomentum": ("strategies.momentum.momentum", "DualMomentum"),
    "RiskAdjustedMomentum": ("strategies.momentum.momentum", "RiskAdjustedMomentum"),
    "SkewnessAdjustedMomentum": ("strategies.momentum.momentum", "SkewnessAdjustedMomentum"),
    "ForwardRateCarry": ("strategies.momentum.carry", "ForwardRateCarry"),
    "RollYieldCarry": ("strategies.momentum.carry", "RollYieldCarry"),
    "TermStructureCarry": ("strategies.momentum.carry", "TermStructureCarry"),
    # Mean Reversion
    "PairsTrading": ("strategies.mean_reversion.stat_arb", "PairsTrading"),
    "KalmanPairsTrading": ("strategies.mean_reversion.stat_arb", "KalmanPairsTrading"),
    "OUMeanReversion": ("strategies.mean_reversion.stat_arb", "OUMeanReversion"),
    "VIXMeanReversion": ("strategies.mean_reversion.volatility_mean_reversion", "VIXMeanReversion"),
    "VolatilityArbitrage": ("strategies.mean_reversion.volatility_mean_reversion", "VolatilityArbitrage"),
    # Volatility
    "VolatilityTargeting": ("strategies.volatility.vol_targeting", "VolatilityTargeting"),
    "VariancePremiumCapture": ("strategies.volatility.variance_premium", "VariancePremiumCapture"),
    "GARCHVolTrading": ("strategies.volatility.regime_vol", "GARCHVolTrading"),
    # Crypto
    "FundingRateArbitrage": ("strategies.crypto.funding_rate", "FundingRateArbitrage"),
    "NVTRatio": ("strategies.crypto.onchain", "NVTRatio"),
    "MayerMultiple": ("strategies.crypto.onchain", "MayerMultiple"),
    # Event driven
    "EarningsSurprise": ("strategies.event_driven.earnings", "EarningsSurprise"),
    "EarningsMomentum": ("strategies.event_driven.earnings", "EarningsMomentum"),
    "FOMCDrift": ("strategies.event_driven.macro_events", "FOMCDrift"),
    "NFPMomentum": ("strategies.event_driven.macro_events", "NFPMomentum"),
    "CPIRegime": ("strategies.event_driven.macro_events", "CPIRegime"),
}


def add_srfm_lab_to_path():
    """Add srfm-lab directory to sys.path."""
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))


def load_strategy_class(strategy_name: str):
    """Dynamically load strategy class from registry."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_name}\n"
            f"Available strategies: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    module_path, class_name = STRATEGY_REGISTRY[strategy_name]
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except ImportError as e:
        logger.error(f"Failed to import {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Class {class_name} not found in {module_path}: {e}")
        raise


def load_price_data(path: str) -> pd.DataFrame:
    """Load price data from CSV file."""
    logger.info(f"Loading price data from: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    logger.info(f"Generating synthetic data: n={n}, seed={seed}")
    rng = np.random.default_rng(seed)
    log_rets = 0.0005 + 0.015 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "open": close * rng.uniform(0.995, 1.005, n),
        "high": close * rng.uniform(1.000, 1.015, n),
        "low": close * rng.uniform(0.985, 1.000, n),
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        "funding_rate": rng.uniform(-0.0001, 0.0003, n),
        "vix": 15 + 5 * np.abs(rng.standard_normal(n)),
    }, index=idx)
    return df


def format_result(result: Any) -> pd.DataFrame:
    """Format a BacktestResult into a display DataFrame."""
    rows = []
    fields = [
        ("Total Return", "total_return", "{:.2%}"),
        ("CAGR", "cagr", "{:.2%}"),
        ("Sharpe Ratio", "sharpe", "{:.3f}"),
        ("Sortino Ratio", "sortino", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar Ratio", "calmar", "{:.3f}"),
        ("Win Rate", "win_rate", "{:.2%}"),
        ("Profit Factor", "profit_factor", "{:.3f}"),
        ("N Trades", "n_trades", "{:.0f}"),
    ]
    for display_name, attr, fmt in fields:
        val = getattr(result, attr, None)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            rows.append({"Metric": display_name, "Value": fmt.format(val)})
    return pd.DataFrame(rows).set_index("Metric")


def run_strategy(
    strategy_name: str,
    strategy_kwargs: Dict,
    data: pd.DataFrame,
    save_equity: Optional[str] = None,
) -> Any:
    """Load, instantiate, and run a strategy."""
    logger.info(f"Loading strategy: {strategy_name}")
    cls = load_strategy_class(strategy_name)

    # Filter kwargs to those accepted by the strategy
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in strategy_kwargs.items() if k in valid_params}
    ignored = {k: v for k, v in strategy_kwargs.items() if k not in valid_params}
    if ignored:
        logger.warning(f"Ignored kwargs (not accepted by {strategy_name}): {ignored}")

    logger.info(f"Instantiating {strategy_name} with kwargs: {filtered_kwargs}")
    strategy = cls(**filtered_kwargs)

    # Run backtest
    logger.info("Running backtest...")
    try:
        result = strategy.backtest(data)
    except TypeError:
        # Some strategies need specific data format
        if hasattr(data, "squeeze"):
            logger.info("Retrying with price series (squeeze)...")
            result = strategy.backtest(data.squeeze())
        else:
            raise

    return result


def print_summary(strategy_name: str, result: Any):
    """Print a formatted summary of the backtest result."""
    print("\n" + "=" * 60)
    print(f"Strategy: {strategy_name}")
    print("=" * 60)
    df = format_result(result)
    print(df.to_string())
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run a strategy from the SRFM lab strategy suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=sorted(STRATEGY_REGISTRY.keys()),
        help="Strategy name to run.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available strategies and exit.",
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV with OHLCV data (index=date). If not provided, synthetic data is used.",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=500,
        help="Number of periods for synthetic data generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data.",
    )

    # Strategy parameters (passed as --param name=value)
    parser.add_argument(
        "--param",
        action="append",
        metavar="NAME=VALUE",
        default=[],
        help="Strategy parameters as name=value. Can be specified multiple times.",
    )

    # Output
    parser.add_argument(
        "--save-equity",
        type=str,
        default=None,
        help="Path to save equity curve CSV.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save results summary CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List strategies
    if args.list_strategies:
        print("\nAvailable strategies:")
        print("-" * 60)
        for name, (module, cls) in sorted(STRATEGY_REGISTRY.items()):
            print(f"  {name:<40} ({module})")
        print()
        return

    add_srfm_lab_to_path()

    # Parse strategy kwargs
    strategy_kwargs = {}
    for param_str in args.param:
        if "=" not in param_str:
            logger.warning(f"Ignoring malformed param (expected name=value): {param_str}")
            continue
        name, value_str = param_str.split("=", 1)
        # Attempt type conversion
        try:
            value = int(value_str)
        except ValueError:
            try:
                value = float(value_str)
            except ValueError:
                if value_str.lower() == "true":
                    value = True
                elif value_str.lower() == "false":
                    value = False
                else:
                    value = value_str
        strategy_kwargs[name] = value

    # Load data
    if args.data:
        data = load_price_data(args.data)
    else:
        data = generate_synthetic_data(n=args.n_synthetic, seed=args.seed)

    # Run
    try:
        result = run_strategy(args.strategy, strategy_kwargs, data, args.save_equity)
    except Exception as e:
        logger.error(f"Strategy run failed: {e}", exc_info=True)
        sys.exit(1)

    # Print summary
    print_summary(args.strategy, result)

    # Save outputs
    if args.save_equity and hasattr(result, "equity_curve"):
        eq = result.equity_curve
        if isinstance(eq, pd.Series):
            eq.to_csv(args.save_equity, header=True)
            logger.info(f"Equity curve saved to: {args.save_equity}")

    if args.save_results:
        df = format_result(result)
        df.to_csv(args.save_results)
        logger.info(f"Results saved to: {args.save_results}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
