#!/usr/bin/env python3
"""
Factor model analysis CLI script.

Runs Fama-French, WML, QMJ, BAB, and PCA factor construction and
reports factor statistics, correlations, and time-series regression results.

Usage:
    python generate_factor_report.py --synthetic --n-assets 30 --n 500
    python generate_factor_report.py --prices prices.csv --save-report report.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_factor_report")


def add_srfm_lab_to_path():
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))


def generate_synthetic_universe(n_assets=30, n=500, seed=42):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:03d}" for i in range(n_assets)]
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    prices = {}
    for t in tickers:
        drift = rng.uniform(-0.0002, 0.001)
        vol = rng.uniform(0.01, 0.025)
        prices[t] = 100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    return pd.DataFrame(prices, index=idx)


def print_section(title: str, df=None, text: str = None):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    if df is not None:
        print(df.to_string())
    if text:
        print(text)


def main():
    parser = argparse.ArgumentParser(
        description="Generate factor model analysis report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true",
                       help="Use synthetically generated universe.")
    group.add_argument("--prices", type=str, help="Path to prices CSV.")

    parser.add_argument("--n-assets", type=int, default=30, help="Number of synthetic assets.")
    parser.add_argument("--n", type=int, default=500, help="Number of time periods.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-factors", type=int, default=5, help="PCA factors to extract.")
    parser.add_argument("--save-report", type=str, default=None, help="Save summary to CSV.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    add_srfm_lab_to_path()

    # Load data
    if args.synthetic:
        prices_df = generate_synthetic_universe(args.n_assets, args.n, args.seed)
    else:
        prices_df = pd.read_csv(args.prices, index_col=0, parse_dates=True)

    logger.info(f"Universe: {prices_df.shape[1]} assets × {len(prices_df)} periods")
    returns = prices_df.pct_change().dropna()

    all_results = {}

    # 1. WML momentum factor
    logger.info("Computing WML factor...")
    try:
        from research.factor_model.factors import WMLFactor
        wml = WMLFactor(formation_period=min(126, len(returns) // 4),
                        holding_period=21)
        wml_series = wml.compute(prices_df)
        stats_wml = wml.momentum_statistics(prices_df)
        print_section("WML Factor Statistics", text=str(stats_wml))
        all_results["wml"] = stats_wml
    except Exception as e:
        logger.warning(f"WML failed: {e}")

    # 2. PCA factor model
    logger.info("Fitting PCA factor model...")
    try:
        from research.factor_model.pca import StatisticalFactorModel
        pca = StatisticalFactorModel(n_factors=args.n_factors, use_correlation=True)
        pca_result = pca.fit(returns)
        summary = pca.factor_summary(pca_result)
        print_section("PCA Factor Summary", summary)
        all_results["pca_summary"] = summary
    except Exception as e:
        logger.warning(f"PCA failed: {e}")

    # 3. Time-series regression of equal-weight portfolio on first PC
    logger.info("Running time-series regression...")
    try:
        from research.factor_model.regression import TimeSeriesRegression
        from research.factor_model.pca import StatisticalFactorModel

        pca = StatisticalFactorModel(n_factors=min(3, args.n_assets - 1))
        pca_result = pca.fit(returns)
        factor_rets = pca_result.factor_returns

        ew_port = returns.mean(axis=1)
        ts_reg = TimeSeriesRegression()
        reg_result = ts_reg.fit(ew_port, factor_rets)

        print_section(
            "Time-Series Regression (EW Portfolio ~ PCA Factors)",
            text=(
                f"Alpha (annualized): {reg_result.alpha:.4%}\n"
                f"Alpha t-stat:       {reg_result.alpha_tstat:.4f}\n"
                f"R-squared:          {reg_result.r_squared:.4f}\n"
                f"Betas:              {reg_result.betas}\n"
                f"Tracking Error:     {reg_result.tracking_error:.4%}"
            ),
        )
        all_results["ts_regression"] = {
            "alpha": reg_result.alpha,
            "alpha_tstat": reg_result.alpha_tstat,
            "r_squared": reg_result.r_squared,
        }
    except Exception as e:
        logger.warning(f"Regression failed: {e}")

    # 4. Factor correlations
    logger.info("Computing factor correlations...")
    try:
        if "pca_result" in dir() or True:
            pca2 = StatisticalFactorModel(n_factors=min(5, args.n_assets - 1))
            pca_result2 = pca2.fit(returns)
            factor_corr = pca2.factor_correlation(pca_result2)
            print_section("PCA Factor Correlation Matrix", factor_corr)
    except Exception as e:
        logger.warning(f"Factor correlation failed: {e}")

    # 5. Scree data
    logger.info("Computing scree data...")
    try:
        pca3 = StatisticalFactorModel(n_factors=2)
        scree = pca3.scree_data(returns, max_factors=min(15, args.n_assets - 1))
        print_section("Scree Data (Explained Variance)", scree.head(10))
    except Exception as e:
        logger.warning(f"Scree failed: {e}")

    # Save report
    if args.save_report and all_results:
        rows = []
        for key, val in all_results.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    rows.append({"section": key, "metric": k, "value": str(v)})
            elif isinstance(val, pd.DataFrame):
                for row_idx in val.index:
                    for col in val.columns:
                        rows.append({"section": key, "metric": f"{row_idx}/{col}",
                                      "value": str(val.loc[row_idx, col])})
        if rows:
            pd.DataFrame(rows).to_csv(args.save_report, index=False)
            logger.info(f"Report saved to: {args.save_report}")

    logger.info("Factor report generation complete.")


if __name__ == "__main__":
    main()
