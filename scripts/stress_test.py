#!/usr/bin/env python3
"""
Stress test runner for strategy portfolios.

Simulates historical-style crash scenarios, fat-tail shocks, correlation
breakdowns, regime shifts, and custom stress scenarios. Reports P&L
distributions, VaR/CVaR, max drawdown, and recovery statistics under stress.

Usage:
    python stress_test.py --synthetic --n-assets 10 --scenarios all
    python stress_test.py --returns returns.csv --scenarios crash,fat_tail
    python stress_test.py --synthetic --n-paths 5000 --save-results stress.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stress_test")

SCENARIO_CHOICES = [
    "crash", "fat_tail", "correlation_breakdown",
    "regime_shift", "liquidity_crisis", "rate_shock",
    "custom", "all"
]


def add_srfm_lab_to_path():
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))


def generate_synthetic_returns(
    n_assets: int = 10,
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    # Correlated returns via factor model
    n_factors = 3
    factor_loadings = rng.uniform(-0.5, 1.0, (n_assets, n_factors))
    factor_rets = rng.standard_normal((n, n_factors)) * 0.01
    idio = rng.standard_normal((n, n_assets)) * rng.uniform(0.005, 0.015, n_assets)
    rets = factor_rets @ factor_loadings.T + idio
    # Add a small drift
    rets += rng.uniform(-0.0002, 0.0005, n_assets)
    return pd.DataFrame(rets, index=idx, columns=[f"asset_{i:02d}" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# Stress scenario generators
# ---------------------------------------------------------------------------

def scenario_crash(
    returns: pd.DataFrame,
    shock_magnitude: float = -0.20,
    shock_duration: int = 10,
    recovery_vol_mult: float = 3.0,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Market crash: sudden large drawdown followed by elevated volatility.
    Returns n_paths × (n_assets) array of cumulative P&L over shock window.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    mu = returns.mean().values
    sigma = returns.std().values
    corr = returns.corr().values

    # Stress corr: increases toward 1 during crash
    stress_corr = 0.3 * corr + 0.7 * np.ones_like(corr)
    np.fill_diagonal(stress_corr, 1.0)
    stress_sigma = sigma * recovery_vol_mult
    try:
        L = np.linalg.cholesky(stress_corr)
    except np.linalg.LinAlgError:
        stress_corr = (stress_corr + stress_corr.T) / 2
        stress_corr += 1e-6 * np.eye(n_assets)
        L = np.linalg.cholesky(stress_corr)

    # Shock day: uniform crash across assets proportional to beta-to-market
    shock_day = shock_magnitude * rng.uniform(0.5, 1.5, (n_paths, n_assets))
    # Post-shock days: high-vol recovery
    Z = rng.standard_normal((n_paths, shock_duration, n_assets))
    post_shock = np.einsum("tij,...j->...ti", np.tile(L, (shock_duration, 1, 1)), Z)
    post_shock = post_shock * stress_sigma + mu
    # Combine: day 0 = shock, then recovery period
    all_rets = np.concatenate([shock_day[:, np.newaxis, :], post_shock], axis=1)
    cum_pnl = np.cumsum(all_rets, axis=1)
    # Final cumulative P&L per path per asset
    return pd.DataFrame(
        cum_pnl[:, -1, :],
        columns=returns.columns,
    )


def scenario_fat_tail(
    returns: pd.DataFrame,
    df_t: float = 3.0,
    scale_mult: float = 2.0,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Fat-tail simulation: draws from Student-t with heavy tails.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    sigma = returns.std().values * scale_mult
    mu = returns.mean().values

    # Multivariate t via chi-squared mixing
    z = rng.standard_normal((n_paths, n_assets))
    chi2 = rng.chisquare(df_t, n_paths) / df_t
    t_draws = z / np.sqrt(chi2[:, np.newaxis])
    fat_rets = mu + sigma * t_draws
    return pd.DataFrame(fat_rets, columns=returns.columns)


def scenario_correlation_breakdown(
    returns: pd.DataFrame,
    target_corr: float = 0.90,
    vol_mult: float = 2.5,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Correlation breakdown: all assets become highly correlated.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    mu = returns.mean().values
    sigma = returns.std().values * vol_mult

    # Stress correlation matrix
    stress_corr = target_corr * np.ones((n_assets, n_assets))
    np.fill_diagonal(stress_corr, 1.0)
    try:
        L = np.linalg.cholesky(stress_corr)
    except np.linalg.LinAlgError:
        stress_corr += 1e-6 * np.eye(n_assets)
        L = np.linalg.cholesky(stress_corr)

    Z = rng.standard_normal((n_paths, n_assets))
    corr_rets = (Z @ L.T) * sigma + mu
    return pd.DataFrame(corr_rets, columns=returns.columns)


def scenario_regime_shift(
    returns: pd.DataFrame,
    bear_mu_mult: float = -3.0,
    bear_vol_mult: float = 2.0,
    regime_duration: int = 63,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Regime shift: transition from bull to bear and back.
    Simulates regime_duration-day cumulative P&L.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    mu = returns.mean().values
    sigma = returns.std().values

    # Bear regime parameters
    bear_mu = mu * bear_mu_mult
    bear_sigma = sigma * bear_vol_mult

    Z = rng.standard_normal((n_paths, regime_duration, n_assets))
    bear_rets = bear_mu + bear_sigma * Z
    cum_pnl = np.cumsum(bear_rets, axis=1)
    return pd.DataFrame(cum_pnl[:, -1, :], columns=returns.columns)


def scenario_liquidity_crisis(
    returns: pd.DataFrame,
    spread_bps: float = 200,
    forced_sell_pct: float = 0.30,
    vol_mult: float = 3.0,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Liquidity crisis: forced selling, wide bid-ask spreads.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    sigma = returns.std().values * vol_mult

    # Transaction cost from spread
    spread_cost = spread_bps / 10000 * forced_sell_pct

    # Random returns under stress
    Z = rng.standard_normal((n_paths, n_assets))
    rets = -np.abs(sigma * Z) - spread_cost  # Net negative bias from selling pressure
    return pd.DataFrame(rets, columns=returns.columns)


def scenario_rate_shock(
    returns: pd.DataFrame,
    rate_shock_bps: float = 200,
    duration_years: float = 5.0,
    equity_beta_to_rates: float = -0.5,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rate shock: sudden rise in interest rates.
    Duration impact + equity repricing.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]
    sigma = returns.std().values

    # Bond P&L: -duration * rate_shock
    rate_shock_pct = rate_shock_bps / 10000
    bond_impact = -duration_years * rate_shock_pct

    # Equity impact: proportional to rate shock with noise
    equity_impact = equity_beta_to_rates * rate_shock_pct

    # Total shock per asset (simple linear factor model)
    systematic_shock = np.full(n_assets, equity_impact)
    # Add idiosyncratic noise
    Z = rng.standard_normal((n_paths, n_assets))
    rets = systematic_shock + sigma * 2.0 * Z + bond_impact * rng.uniform(0, 1, n_assets)
    return pd.DataFrame(rets, columns=returns.columns)


def scenario_custom(
    returns: pd.DataFrame,
    asset_shocks: Dict[str, float],
    vol_mult: float = 2.0,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Custom scenario: user-defined per-asset shocks plus elevated vol.
    asset_shocks: dict mapping asset name → daily return shock (e.g. -0.10)
    """
    rng = np.random.default_rng(seed)
    sigma = returns.std().values * vol_mult
    mu = np.array([asset_shocks.get(col, 0.0) for col in returns.columns])
    Z = rng.standard_normal((n_paths, returns.shape[1]))
    rets = mu + sigma * Z
    return pd.DataFrame(rets, columns=returns.columns)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def compute_var(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Value at Risk at given confidence level."""
    return float(-np.quantile(pnl, 1 - confidence))


def compute_cvar(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall)."""
    threshold = np.quantile(pnl, 1 - confidence)
    tail = pnl[pnl <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def compute_max_drawdown_from_paths(paths_cum_pnl: np.ndarray) -> float:
    """Approximate max drawdown: min of cumulative P&L paths (1-day path end)."""
    return float(-np.percentile(paths_cum_pnl, 5))


def portfolio_stress_metrics(
    scenario_df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    confidence: float = 0.95,
) -> Dict:
    """Compute stress metrics for a portfolio given scenario returns."""
    n_assets = scenario_df.shape[1]
    if weights is None:
        weights = np.ones(n_assets) / n_assets

    port_pnl = scenario_df.values @ weights

    return {
        "mean_pnl": float(port_pnl.mean()),
        "std_pnl": float(port_pnl.std()),
        "min_pnl": float(port_pnl.min()),
        "max_pnl": float(port_pnl.max()),
        f"var_{int(confidence*100)}": compute_var(port_pnl, confidence),
        f"cvar_{int(confidence*100)}": compute_cvar(port_pnl, confidence),
        "pct_positive": float((port_pnl > 0).mean()),
        "skewness": float(stats.skew(port_pnl)),
        "kurtosis": float(stats.kurtosis(port_pnl)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_stress_report(results: Dict[str, Dict]):
    print("\n" + "=" * 90)
    print("STRESS TEST RESULTS")
    print("=" * 90)
    df = pd.DataFrame(results).T
    # Order columns
    preferred_order = [
        "mean_pnl", "std_pnl", "min_pnl", "var_95", "cvar_95",
        "pct_positive", "skewness", "kurtosis"
    ]
    cols = [c for c in preferred_order if c in df.columns] + \
           [c for c in df.columns if c not in preferred_order]
    print(df[cols].round(4).to_string())
    print()

    # Highlight worst scenarios
    if "var_95" in df.columns:
        worst = df["var_95"].idxmax()
        print(f"Worst scenario by VaR-95: {worst} ({df.loc[worst, 'var_95']:.4f})")
    if "cvar_95" in df.columns:
        worst_cvar = df["cvar_95"].idxmax()
        print(f"Worst scenario by CVaR-95: {worst_cvar} ({df.loc[worst_cvar, 'cvar_95']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Run stress scenarios on a portfolio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true")
    group.add_argument("--returns", type=str, help="Path to returns CSV.")

    parser.add_argument("--n-assets", type=int, default=10,
                        help="Number of synthetic assets.")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of historical periods.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-paths", type=int, default=2000,
                        help="Number of Monte Carlo paths per scenario.")
    parser.add_argument("--scenarios", type=str, default="all",
                        help=f"Comma-separated list: {', '.join(SCENARIO_CHOICES)}")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for VaR/CVaR.")
    parser.add_argument("--weights", type=str, default=None,
                        help="CSV of portfolio weights (index=asset, value=weight).")
    parser.add_argument("--crash-shock", type=float, default=-0.15,
                        help="Crash scenario shock magnitude.")
    parser.add_argument("--crash-duration", type=int, default=10,
                        help="Crash scenario duration (days).")
    parser.add_argument("--fat-tail-df", type=float, default=3.0,
                        help="Student-t degrees of freedom for fat tail scenario.")
    parser.add_argument("--stress-corr", type=float, default=0.90,
                        help="Target correlation in correlation-breakdown scenario.")
    parser.add_argument("--rate-shock-bps", type=float, default=200,
                        help="Rate shock in basis points.")
    parser.add_argument("--regime-duration", type=int, default=63,
                        help="Duration of bear regime (days).")
    parser.add_argument("--custom-shocks", type=str, default=None,
                        help="Custom per-asset shocks as 'asset=shock,...' e.g. 'asset_00=-0.10'.")
    parser.add_argument("--show-asset-breakdown", action="store_true",
                        help="Show per-asset P&L under each scenario.")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Save stress results to CSV.")
    parser.add_argument("--save-paths", type=str, default=None,
                        help="Save raw P&L paths for one scenario.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    add_srfm_lab_to_path()

    # Load returns
    if args.synthetic:
        returns_df = generate_synthetic_returns(args.n_assets, args.n, args.seed)
        logger.info(f"Generated synthetic returns: {returns_df.shape}")
    else:
        returns_df = pd.read_csv(args.returns, index_col=0, parse_dates=True)
        logger.info(f"Loaded returns: {returns_df.shape}")

    returns_df = returns_df.dropna()

    # Load or compute weights
    if args.weights:
        w_series = pd.read_csv(args.weights, index_col=0).iloc[:, 0]
        w_series = w_series.reindex(returns_df.columns).fillna(0)
        weights = w_series.values / w_series.values.sum()
        logger.info(f"Loaded portfolio weights: {dict(zip(returns_df.columns, weights.round(4)))}")
    else:
        n_assets = returns_df.shape[1]
        weights = np.ones(n_assets) / n_assets
        logger.info("Using equal-weight portfolio.")

    # Parse scenarios
    if args.scenarios == "all":
        scenarios_to_run = [s for s in SCENARIO_CHOICES if s not in ("all", "custom")]
        if args.custom_shocks:
            scenarios_to_run.append("custom")
    else:
        scenarios_to_run = [s.strip() for s in args.scenarios.split(",")]

    # Parse custom shocks
    custom_shocks = {}
    if args.custom_shocks:
        for item in args.custom_shocks.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                custom_shocks[k.strip()] = float(v.strip())
        logger.info(f"Custom shocks: {custom_shocks}")

    # Run scenarios
    scenario_results = {}
    raw_pnl_cache = {}

    scenario_funcs = {
        "crash": lambda: scenario_crash(
            returns_df,
            shock_magnitude=args.crash_shock,
            shock_duration=args.crash_duration,
            n_paths=args.n_paths,
            seed=args.seed,
        ),
        "fat_tail": lambda: scenario_fat_tail(
            returns_df,
            df_t=args.fat_tail_df,
            n_paths=args.n_paths,
            seed=args.seed,
        ),
        "correlation_breakdown": lambda: scenario_correlation_breakdown(
            returns_df,
            target_corr=args.stress_corr,
            n_paths=args.n_paths,
            seed=args.seed,
        ),
        "regime_shift": lambda: scenario_regime_shift(
            returns_df,
            regime_duration=args.regime_duration,
            n_paths=args.n_paths,
            seed=args.seed,
        ),
        "liquidity_crisis": lambda: scenario_liquidity_crisis(
            returns_df,
            n_paths=args.n_paths,
            seed=args.seed,
        ),
        "rate_shock": lambda: scenario_rate_shock(
            returns_df,
            rate_shock_bps=args.rate_shock_bps,
            n_paths=args.n_paths,
            seed=args.seed,
        ),
        "custom": lambda: scenario_custom(
            returns_df,
            asset_shocks=custom_shocks,
            n_paths=args.n_paths,
            seed=args.seed,
        ),
    }

    for scenario_name in scenarios_to_run:
        if scenario_name not in scenario_funcs:
            logger.warning(f"Unknown scenario: {scenario_name}, skipping.")
            continue

        logger.info(f"Running scenario: {scenario_name}...")
        try:
            scenario_df = scenario_funcs[scenario_name]()
            metrics = portfolio_stress_metrics(scenario_df, weights, args.confidence)
            scenario_results[scenario_name] = metrics
            raw_pnl_cache[scenario_name] = scenario_df

            logger.info(
                f"  VaR-{int(args.confidence*100)}: {metrics[f'var_{int(args.confidence*100)}']:.4f}, "
                f"CVaR: {metrics[f'cvar_{int(args.confidence*100)}']:.4f}, "
                f"Mean: {metrics['mean_pnl']:.4f}"
            )
        except Exception as e:
            logger.warning(f"Scenario {scenario_name} failed: {e}", exc_info=args.verbose)

    if not scenario_results:
        logger.error("No scenarios succeeded.")
        sys.exit(1)

    print_stress_report(scenario_results)

    # Asset-level breakdown
    if args.show_asset_breakdown:
        print("\n" + "=" * 80)
        print("PER-ASSET MEAN P&L UNDER EACH SCENARIO")
        print("=" * 80)
        asset_df = pd.DataFrame(
            {name: df.mean() for name, df in raw_pnl_cache.items()},
            index=returns_df.columns,
        )
        print(asset_df.round(4).to_string())

    # Save raw P&L paths
    if args.save_paths and raw_pnl_cache:
        first_scenario = list(raw_pnl_cache.keys())[0]
        raw_pnl_cache[first_scenario].to_csv(args.save_paths, index=False)
        logger.info(f"P&L paths ({first_scenario}) saved to: {args.save_paths}")

    # Save summary
    if args.save_results:
        pd.DataFrame(scenario_results).T.to_csv(args.save_results)
        logger.info(f"Stress results saved to: {args.save_results}")

    logger.info("Stress test complete.")


if __name__ == "__main__":
    main()
