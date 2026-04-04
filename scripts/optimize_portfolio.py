#!/usr/bin/env python3
"""
Portfolio optimization CLI.

Supports: min-variance, risk-parity, max-Sharpe, Kelly, equal-weight, max-diversification.

Usage:
    python optimize_portfolio.py --synthetic --n-assets 10 --method min_var
    python optimize_portfolio.py --prices prices.csv --method all --save-weights weights.csv
    python optimize_portfolio.py --synthetic --method risk_parity --target-vol 0.10
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("optimize_portfolio")

METHOD_CHOICES = [
    "min_var", "risk_parity", "max_sharpe", "kelly",
    "equal_weight", "max_div", "all"
]


def add_srfm_lab_to_path():
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))


def generate_synthetic_universe(n_assets: int = 10, n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    prices = {}
    for i in range(n_assets):
        drift = rng.uniform(-0.0002, 0.0008)
        vol = rng.uniform(0.008, 0.030)
        prices[f"asset_{i:02d}"] = 100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    return pd.DataFrame(prices, index=idx)


# ---------------------------------------------------------------------------
# Optimization methods
# ---------------------------------------------------------------------------

def equal_weight(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mu)
    return np.ones(n) / n


def min_variance(mu: np.ndarray, cov: np.ndarray, allow_short: bool = False) -> np.ndarray:
    n = len(mu)
    bounds = ((-1, 1) if allow_short else (0, 1),) * n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    result = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result.x / result.x.sum()


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0,
               allow_short: bool = False) -> np.ndarray:
    n = len(mu)
    bounds = ((-1, 1) if allow_short else (0, 1),) * n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w + 1e-12)
        return -(ret - rf) / vol

    result = minimize(
        neg_sharpe,
        np.ones(n) / n,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = result.x
    w = np.where(w < 0, 0, w)
    return w / (w.sum() + 1e-12)


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Equal risk contribution via SLSQP."""
    n = len(cov)
    w0 = np.ones(n) / n

    def risk_concentration(w):
        port_vol = np.sqrt(w @ cov @ w + 1e-12)
        rc = w * (cov @ w) / port_vol
        target = port_vol / n
        return np.sum((rc - target) ** 2)

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = ((1e-6, 1.0),) * n
    result = minimize(
        risk_concentration,
        w0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    w = np.maximum(result.x, 0)
    return w / (w.sum() + 1e-12)


def kelly_weights(mu: np.ndarray, cov: np.ndarray, fraction: float = 0.5,
                  leverage_limit: float = 2.0) -> np.ndarray:
    """Multi-asset Kelly: w* = fraction * Sigma^{-1} mu, capped at leverage_limit."""
    try:
        w = fraction * np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        w = fraction * np.linalg.lstsq(cov, mu, rcond=None)[0]

    # Long-only version: floor at 0 and cap leverage
    w_long = np.maximum(w, 0)
    if w_long.sum() > leverage_limit:
        w_long = w_long / w_long.sum() * leverage_limit
    # If all negative, fall back to equal weight
    if w_long.sum() < 1e-10:
        return np.ones(len(mu)) / len(mu)
    return w_long / w_long.sum()


def max_diversification_weights(cov: np.ndarray) -> np.ndarray:
    """Maximize diversification ratio: sum(w_i * sigma_i) / sqrt(w' Cov w)."""
    n = len(cov)
    sigmas = np.sqrt(np.diag(cov))

    def neg_div_ratio(w):
        port_vol = np.sqrt(w @ cov @ w + 1e-12)
        weighted_vols = w @ sigmas
        return -weighted_vols / port_vol

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = ((0, 1),) * n
    result = minimize(
        neg_div_ratio,
        np.ones(n) / n,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = np.maximum(result.x, 0)
    return w / (w.sum() + 1e-12)


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_metrics(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    annualize: int = 252,
) -> Dict:
    port_ret = float(weights @ mu) * annualize
    port_var = float(weights @ cov @ weights)
    port_vol = float(np.sqrt(port_var)) * np.sqrt(annualize)
    sharpe = (port_ret - rf) / (port_vol + 1e-12)
    div_ratio = float(weights @ np.sqrt(np.diag(cov))) / (float(np.sqrt(port_var)) + 1e-12)
    port_vol_d = float(np.sqrt(port_var))
    rc = weights * (cov @ weights) / (port_vol_d + 1e-12)
    rc_pct = rc / (rc.sum() + 1e-12)
    max_rc = float(rc_pct.max())
    hhi = float(np.sum(weights ** 2))

    return {
        "ann_return": port_ret,
        "ann_vol": port_vol,
        "sharpe": sharpe,
        "diversification_ratio": div_ratio,
        "max_risk_contribution": max_rc,
        "herfindahl": hhi,
    }


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> pd.Series:
    port_vol = float(np.sqrt(weights @ cov @ weights + 1e-12))
    rc = weights * (cov @ weights) / port_vol
    return pd.Series(rc / rc.sum(), name="risk_contribution_%")


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 50) -> pd.DataFrame:
    """Trace the efficient frontier (long-only)."""
    target_returns = np.linspace(mu.min() * 252, mu.max() * 252, n_points)
    frontier = []
    n = len(mu)
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu * 252 - t},
        ]
        bounds = ((0, 1),) * n
        result = minimize(
            lambda w: w @ cov @ w,
            np.ones(n) / n,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if result.success:
            vol = float(np.sqrt(result.fun)) * np.sqrt(252)
            frontier.append({"target_return": target, "volatility": vol,
                             "sharpe": target / (vol + 1e-12)})
    return pd.DataFrame(frontier)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_weights_table(weights_dict: Dict[str, np.ndarray], asset_names: List[str]):
    df = pd.DataFrame(
        {method: w for method, w in weights_dict.items()},
        index=asset_names,
    )
    print("\n" + "=" * 80)
    print("PORTFOLIO WEIGHTS")
    print("=" * 80)
    print(df.round(4).to_string())


def print_metrics_table(metrics_dict: Dict[str, Dict], weights_dict: Dict[str, np.ndarray],
                        mu: np.ndarray, cov: np.ndarray, rf: float):
    rows = []
    for method, weights in weights_dict.items():
        m = portfolio_metrics(weights, mu, cov, rf)
        m["method"] = method
        rows.append(m)
    df = pd.DataFrame(rows).set_index("method")
    print("\n" + "=" * 80)
    print("PORTFOLIO METRICS")
    print("=" * 80)
    print(df.round(4).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio optimization across multiple methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--prices", type=str, help="Path to prices CSV.")

    parser.add_argument("--returns", type=str, default=None,
                        help="Path to returns CSV (overrides prices).")
    parser.add_argument("--n-assets", type=int, default=10,
                        help="Number of synthetic assets.")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of periods (synthetic).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="all",
                        choices=METHOD_CHOICES,
                        help="Optimization method.")
    parser.add_argument("--rf", type=float, default=0.0,
                        help="Risk-free rate (annualized).")
    parser.add_argument("--target-vol", type=float, default=0.10,
                        help="Target portfolio volatility (for vol-scaled outputs).")
    parser.add_argument("--kelly-fraction", type=float, default=0.5,
                        help="Kelly fraction (0.5 = half-Kelly).")
    parser.add_argument("--leverage-limit", type=float, default=2.0,
                        help="Maximum leverage for Kelly.")
    parser.add_argument("--allow-short", action="store_true",
                        help="Allow short positions.")
    parser.add_argument("--show-risk-contributions", action="store_true",
                        help="Show risk contribution table.")
    parser.add_argument("--show-frontier", action="store_true",
                        help="Compute and display efficient frontier.")
    parser.add_argument("--frontier-points", type=int, default=30,
                        help="Number of points on efficient frontier.")
    parser.add_argument("--save-weights", type=str, default=None,
                        help="Save weights to CSV.")
    parser.add_argument("--save-metrics", type=str, default=None,
                        help="Save metrics to CSV.")
    parser.add_argument("--save-frontier", type=str, default=None,
                        help="Save efficient frontier to CSV.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    add_srfm_lab_to_path()

    # Load data
    if args.returns:
        returns_df = pd.read_csv(args.returns, index_col=0, parse_dates=True)
        logger.info(f"Loaded returns: {returns_df.shape}")
    elif args.synthetic:
        prices_df = generate_synthetic_universe(args.n_assets, args.n, args.seed)
        returns_df = prices_df.pct_change().dropna()
        logger.info(f"Generated {len(returns_df)} returns × {returns_df.shape[1]} assets")
    else:
        prices_df = pd.read_csv(args.prices, index_col=0, parse_dates=True)
        returns_df = prices_df.pct_change().dropna()
        logger.info(f"Loaded prices → returns: {returns_df.shape}")

    mu = returns_df.mean().values
    cov = returns_df.cov().values
    asset_names = list(returns_df.columns)
    n_assets = len(asset_names)

    logger.info(f"Assets: {n_assets}, Periods: {len(returns_df)}")

    # Determine which methods to run
    if args.method == "all":
        methods = ["equal_weight", "min_var", "risk_parity", "max_sharpe", "kelly", "max_div"]
    else:
        methods = [args.method]

    weights_dict = {}

    for method in methods:
        logger.info(f"Computing {method} portfolio...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if method == "equal_weight":
                    w = equal_weight(mu, cov)
                elif method == "min_var":
                    w = min_variance(mu, cov, args.allow_short)
                elif method == "max_sharpe":
                    w = max_sharpe(mu, cov, args.rf / 252, args.allow_short)
                elif method == "risk_parity":
                    w = risk_parity_weights(cov)
                elif method == "kelly":
                    w = kelly_weights(mu, cov, args.kelly_fraction, args.leverage_limit)
                elif method == "max_div":
                    w = max_diversification_weights(cov)
                else:
                    continue
            weights_dict[method] = w
        except Exception as e:
            logger.warning(f"{method} failed: {e}")

    if not weights_dict:
        logger.error("No optimizations succeeded.")
        sys.exit(1)

    # Print results
    print_weights_table(weights_dict, asset_names)
    print_metrics_table({}, weights_dict, mu, cov, args.rf / 252)

    if args.show_risk_contributions:
        print("\n" + "=" * 60)
        print("RISK CONTRIBUTIONS (%)")
        print("=" * 60)
        rc_df = pd.DataFrame(
            {method: risk_contributions(w, cov).values for method, w in weights_dict.items()},
            index=asset_names,
        )
        print(rc_df.round(4).to_string())

    if args.show_frontier:
        logger.info("Computing efficient frontier...")
        try:
            frontier_df = efficient_frontier(mu, cov, args.frontier_points)
            print("\n" + "=" * 60)
            print("EFFICIENT FRONTIER (sample points)")
            print("=" * 60)
            print(frontier_df.round(4).head(10).to_string())
            if args.save_frontier:
                frontier_df.to_csv(args.save_frontier, index=False)
                logger.info(f"Frontier saved to: {args.save_frontier}")
        except Exception as e:
            logger.warning(f"Efficient frontier failed: {e}")

    # Save outputs
    if args.save_weights:
        weights_df = pd.DataFrame(
            {method: w for method, w in weights_dict.items()},
            index=asset_names,
        )
        weights_df.to_csv(args.save_weights)
        logger.info(f"Weights saved to: {args.save_weights}")

    if args.save_metrics:
        rows = []
        for method, w in weights_dict.items():
            m = portfolio_metrics(w, mu, cov, args.rf / 252)
            m["method"] = method
            rows.append(m)
        pd.DataFrame(rows).set_index("method").to_csv(args.save_metrics)
        logger.info(f"Metrics saved to: {args.save_metrics}")

    logger.info("Portfolio optimization complete.")


if __name__ == "__main__":
    main()
