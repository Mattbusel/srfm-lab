#!/usr/bin/env python3
"""
CLI script to run regime detection and analysis.

Usage:
    python run_regime_analysis.py --synthetic --method hmm --n-states 3
    python run_regime_analysis.py --prices prices.csv --method pelt --save-regimes regimes.csv
    python run_regime_analysis.py --synthetic --method all --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_regime_analysis")

METHOD_CHOICES = ["hmm", "cusum", "pelt", "bocpd", "all"]


def add_srfm_lab_to_path():
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))


def generate_synthetic_regime_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with 3 clear regimes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")

    # Regime parameters: (mu_daily, sigma_daily)
    regimes = [
        (0.0008, 0.008),   # bull: trending up, low vol
        (-0.001, 0.025),   # bear: trending down, high vol
        (0.0001, 0.012),   # sideways: flat, medium vol
    ]

    # True regime changes
    breakpoints = [0, n // 3, 2 * n // 3, n]
    regime_seq = [0, 1, 2]

    returns = np.zeros(n)
    for i, reg_idx in enumerate(regime_seq):
        start, end = breakpoints[i], breakpoints[i + 1]
        mu, sigma = regimes[reg_idx]
        returns[start:end] = mu + sigma * rng.standard_normal(end - start)

    prices = 100 * np.exp(np.cumsum(returns))
    vix_proxy = pd.Series(returns).rolling(21).std().fillna(0.01) * np.sqrt(252) * 100

    df = pd.DataFrame({
        "close": prices,
        "returns": returns,
        "realized_vol": vix_proxy.values,
        "volume": rng.integers(500_000, 5_000_000, n).astype(float),
    }, index=idx)
    return df


def run_hmm(returns_series: pd.Series, n_states: int, verbose: bool = False):
    """Run Gaussian HMM regime detection."""
    from research.regime_analysis.hmm_regime import RegimeDetector

    logger.info(f"Fitting HMM with {n_states} states...")
    detector = RegimeDetector(
        n_states=n_states,
        features=["returns", "vol"],
        vol_window=21,
    )
    result = detector.fit(returns_series)
    regime_series = detector.get_regime_series(result)
    stats = detector.regime_statistics(result, returns_series)

    return result, regime_series, stats


def run_cusum(returns_series: pd.Series, threshold: float = 3.0):
    """Run CUSUM change point detection."""
    from research.regime_analysis.change_point import CUSUM

    logger.info(f"Running CUSUM (threshold={threshold})...")
    cusum = CUSUM(threshold=threshold, drift=0.5)
    cp_result = cusum.detect(returns_series)
    stats = cusum.cusum_statistics()
    return cp_result, stats


def run_pelt(returns_series: pd.Series, penalty: Optional[float] = None):
    """Run PELT change point detection."""
    from research.regime_analysis.change_point import PELT

    logger.info("Running PELT...")
    pelt = PELT(model="normal", penalty=penalty)
    cp_result = pelt.detect(returns_series)
    return cp_result


def run_bocpd(returns_series: pd.Series, hazard_rate: float = 1.0 / 252):
    """Run Bayesian Online Change Point Detection."""
    from research.regime_analysis.change_point import BayesianOnlineCP

    logger.info(f"Running BOCPD (hazard_rate={hazard_rate:.4f})...")
    bocpd = BayesianOnlineCP(hazard_rate=hazard_rate, threshold=0.5)
    cp_result = bocpd.detect(returns_series)
    return cp_result


def print_hmm_results(result, regime_series, stats):
    print("\n" + "=" * 70)
    print("HMM REGIME DETECTION RESULTS")
    print("=" * 70)
    print(f"\nLog-Likelihood: {result.log_likelihood:.2f}")
    print(f"BIC:            {result.BIC:.2f}")
    print(f"AIC:            {result.AIC:.2f}")
    print(f"Iterations:     {result.n_iter}")
    print(f"\nTransition Matrix:")
    tm = pd.DataFrame(
        result.transition_matrix,
        index=[f"State {i}" for i in range(len(result.transition_matrix))],
        columns=[f"State {i}" for i in range(len(result.transition_matrix))],
    )
    print(tm.round(4).to_string())
    print(f"\nState Means: {np.round(result.means, 6)}")
    print(f"\nRegime Distribution:")
    counts = regime_series.value_counts().sort_index()
    for state, count in counts.items():
        pct = count / len(regime_series) * 100
        print(f"  State {state}: {count:4d} periods ({pct:.1f}%)")
    if stats is not None:
        print(f"\nRegime Statistics:")
        print(stats.to_string())


def print_cp_results(cp_result, method: str):
    print(f"\n{'=' * 70}")
    print(f"{method.upper()} CHANGE POINT DETECTION RESULTS")
    print("=" * 70)
    print(f"\nAlgorithm:       {cp_result.algorithm}")
    print(f"N Segments:      {cp_result.n_segments}")
    print(f"Change Points:   {cp_result.n_segments - 1}")

    if cp_result.change_points:
        print(f"\nChange Point Indices: {cp_result.change_points}")
        if cp_result.change_point_dates is not None:
            print(f"Change Point Dates:   {[str(d.date()) for d in cp_result.change_point_dates]}")

    if cp_result.segment_stats:
        print(f"\nSegment Statistics:")
        for i, seg in enumerate(cp_result.segment_stats):
            print(f"  Segment {i + 1}: {seg}")


def compare_all_methods(returns_series: pd.Series, n_states: int):
    """Run all detection methods and compare."""
    from research.regime_analysis.change_point import compare_detectors
    logger.info("Running all change point detectors for comparison...")
    comparison = compare_detectors(returns_series)
    print("\n" + "=" * 70)
    print("CHANGE POINT DETECTOR COMPARISON")
    print("=" * 70)
    for method, result in comparison.items():
        print(f"\n{method}: {result.n_segments - 1} change points detected")
        if result.change_points:
            print(f"  Indices: {result.change_points[:10]}")


def main():
    parser = argparse.ArgumentParser(
        description="Run regime detection and analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true",
                       help="Use synthetically generated regime data.")
    group.add_argument("--prices", type=str, help="Path to prices CSV.")

    parser.add_argument("--returns", type=str, default=None,
                        help="Path to returns CSV (alternative to prices).")
    parser.add_argument("--method", type=str, default="hmm", choices=METHOD_CHOICES,
                        help="Regime detection method.")
    parser.add_argument("--n-states", type=int, default=3,
                        help="Number of HMM states.")
    parser.add_argument("--n-synthetic", type=int, default=600,
                        help="Number of periods for synthetic data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cusum-threshold", type=float, default=3.0,
                        help="CUSUM threshold (in std devs).")
    parser.add_argument("--pelt-penalty", type=float, default=None,
                        help="PELT penalty (None = auto BIC).")
    parser.add_argument("--hazard-rate", type=float, default=1.0 / 252,
                        help="BOCPD hazard rate.")
    parser.add_argument("--select-n-states", action="store_true",
                        help="Run BIC-based model selection for HMM.")
    parser.add_argument("--max-states", type=int, default=6,
                        help="Maximum states to try in model selection.")
    parser.add_argument("--save-regimes", type=str, default=None,
                        help="Save regime series to CSV.")
    parser.add_argument("--save-stats", type=str, default=None,
                        help="Save regime statistics to CSV.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    add_srfm_lab_to_path()

    # Load or generate data
    if args.synthetic:
        logger.info(f"Generating synthetic regime data: n={args.n_synthetic}")
        data_df = generate_synthetic_regime_data(args.n_synthetic, args.seed)
        returns_series = pd.Series(data_df["returns"].values, index=data_df.index, name="returns")
        logger.info(f"Generated {len(data_df)} periods with 3 embedded regimes.")
    elif args.returns:
        returns_df = pd.read_csv(args.returns, index_col=0, parse_dates=True)
        returns_series = returns_df.iloc[:, 0]
        logger.info(f"Loaded returns: {returns_series.shape}")
    else:
        prices_df = pd.read_csv(args.prices, index_col=0, parse_dates=True)
        if prices_df.shape[1] > 1:
            logger.info(f"Multiple columns found; using first: {prices_df.columns[0]}")
        prices_series = prices_df.iloc[:, 0]
        returns_series = prices_series.pct_change().dropna()
        logger.info(f"Loaded prices → returns: {returns_series.shape}")

    logger.info(
        f"Returns summary: mean={returns_series.mean():.4%}, "
        f"std={returns_series.std():.4%}, n={len(returns_series)}"
    )

    regime_series = None
    regime_stats = None

    if args.method in ("hmm", "all"):
        if args.select_n_states:
            logger.info("Running HMM model selection (BIC)...")
            try:
                from research.regime_analysis.hmm_regime import RegimeDetector
                detector = RegimeDetector(n_states=2)
                best_n, bic_scores = detector.select_n_states(
                    returns_series, max_states=args.max_states
                )
                print(f"\nHMM Model Selection (BIC):")
                print(f"  Best n_states: {best_n}")
                bic_df = pd.DataFrame(bic_scores, columns=["n_states", "BIC"]).set_index("n_states")
                print(bic_df.to_string())
                args.n_states = best_n
            except Exception as e:
                logger.warning(f"Model selection failed: {e}")

        try:
            result, regime_series, regime_stats = run_hmm(
                returns_series, args.n_states, args.verbose
            )
            print_hmm_results(result, regime_series, regime_stats)
        except Exception as e:
            logger.warning(f"HMM failed: {e}", exc_info=args.verbose)

    if args.method in ("cusum", "all"):
        try:
            cp_result, cusum_stats = run_cusum(returns_series, args.cusum_threshold)
            print_cp_results(cp_result, "CUSUM")
            if cusum_stats is not None:
                print(f"\nCUSUM Diagnostics:")
                print(f"  Max positive:  {cusum_stats.get('max_cusum_pos', 'N/A'):.4f}")
                print(f"  Max negative:  {cusum_stats.get('max_cusum_neg', 'N/A'):.4f}")
                print(f"  N detections:  {cusum_stats.get('n_detections', 0)}")
        except Exception as e:
            logger.warning(f"CUSUM failed: {e}", exc_info=args.verbose)

    if args.method in ("pelt", "all"):
        try:
            cp_result = run_pelt(returns_series, args.pelt_penalty)
            print_cp_results(cp_result, "PELT")
        except Exception as e:
            logger.warning(f"PELT failed: {e}", exc_info=args.verbose)

    if args.method in ("bocpd", "all"):
        try:
            cp_result = run_bocpd(returns_series, args.hazard_rate)
            print_cp_results(cp_result, "BOCPD")
        except Exception as e:
            logger.warning(f"BOCPD failed: {e}", exc_info=args.verbose)

    if args.method == "all":
        try:
            compare_all_methods(returns_series, args.n_states)
        except Exception as e:
            logger.warning(f"Comparison failed: {e}", exc_info=args.verbose)

    # Regime-conditional backtest
    if regime_series is not None:
        try:
            from research.regime_analysis.regime_backtest import RegimeBacktest
            rb = RegimeBacktest()
            perf = rb.conditional_performance(returns_series, regime_series)
            print("\n" + "=" * 70)
            print("REGIME-CONDITIONAL PERFORMANCE")
            print("=" * 70)
            print(perf.to_string())
            if regime_stats is not None:
                regime_stats = perf
        except Exception as e:
            logger.warning(f"Regime backtest failed: {e}", exc_info=args.verbose)

    # Save outputs
    if args.save_regimes and regime_series is not None:
        regime_series.to_csv(args.save_regimes, header=True)
        logger.info(f"Regime series saved to: {args.save_regimes}")

    if args.save_stats and regime_stats is not None:
        if isinstance(regime_stats, pd.DataFrame):
            regime_stats.to_csv(args.save_stats)
            logger.info(f"Regime stats saved to: {args.save_stats}")

    logger.info("Regime analysis complete.")


if __name__ == "__main__":
    main()
