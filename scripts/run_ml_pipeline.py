#!/usr/bin/env python3
"""
CLI script to run the ML alpha pipeline.

Usage:
    python run_ml_pipeline.py --prices prices.csv --features features.csv
    python run_ml_pipeline.py --synthetic --n 500 --model lgbm --n-splits 5
    python run_ml_pipeline.py --synthetic --model ensemble --save-results ml_results.csv

Outputs:
    - IC, ICIR, decile returns
    - Feature importance table
    - Rolling prediction CSV
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_ml_pipeline")

MODEL_CHOICES = ["lgbm", "rf", "xgb", "linear", "neural", "ensemble"]


def add_srfm_lab_to_path():
    script_dir = Path(__file__).parent
    lab_dir = script_dir.parent
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))


def generate_synthetic_data(
    n: int = 500,
    n_assets: int = 20,
    n_features: int = 30,
    seed: int = 42,
) -> tuple:
    """Generate synthetic prices and features for pipeline testing."""
    logger.info(f"Generating synthetic data: n={n}, n_assets={n_assets}, n_features={n_features}")
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    tickers = [f"ticker_{i:03d}" for i in range(n_assets)]

    # Price matrix
    prices = {}
    for t in tickers:
        drift = rng.uniform(-0.0003, 0.001)
        vol = rng.uniform(0.01, 0.025)
        log_rets = drift + vol * rng.standard_normal(n)
        prices[t] = 100 * np.exp(np.cumsum(log_rets))
    prices_df = pd.DataFrame(prices, index=idx)

    # Feature matrix
    feature_names = [f"feat_{i:02d}" for i in range(n_features)]
    features = rng.standard_normal((n, n_features))
    features_df = pd.DataFrame(features, index=idx, columns=feature_names)

    return prices_df, features_df


def load_csv_with_dates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {df.shape} from {path}")
    return df


def build_model(model_name: str, model_kwargs: Dict):
    """Instantiate the requested model."""
    if model_name == "lgbm":
        from strategies.ml_alpha.models import LightGBMAlpha
        return LightGBMAlpha(**model_kwargs)
    elif model_name == "rf":
        from strategies.ml_alpha.models import RandomForestAlpha
        return RandomForestAlpha(**model_kwargs)
    elif model_name == "xgb":
        from strategies.ml_alpha.models import XGBoostAlpha
        return XGBoostAlpha(**model_kwargs)
    elif model_name == "linear":
        from strategies.ml_alpha.models import LinearAlpha
        return LinearAlpha(**model_kwargs)
    elif model_name == "neural":
        from strategies.ml_alpha.models import NeuralNetAlpha
        return NeuralNetAlpha(**model_kwargs)
    elif model_name == "ensemble":
        from strategies.ml_alpha.models import (
            LightGBMAlpha, RandomForestAlpha, LinearAlpha, EnsembleAlpha
        )
        members = [LightGBMAlpha(), RandomForestAlpha(), LinearAlpha()]
        return EnsembleAlpha(models=members)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def print_pipeline_results(result):
    """Print ML pipeline results."""
    print("\n" + "=" * 70)
    print("ML ALPHA PIPELINE RESULTS")
    print("=" * 70)

    print(f"\nIC Mean:    {result.icir:.4f}" if hasattr(result, "icir") else "")
    if hasattr(result, "ic_series") and result.ic_series is not None:
        ic_clean = result.ic_series.dropna()
        print(f"Rolling IC  mean:   {ic_clean.mean():.4f}")
        print(f"Rolling IC  std:    {ic_clean.std():.4f}")
        print(f"ICIR:               {ic_clean.mean() / (ic_clean.std() + 1e-12):.4f}")
        print(f"IC > 0 fraction:    {(ic_clean > 0).mean():.2%}")

    if hasattr(result, "decile_returns") and result.decile_returns is not None:
        print("\nDecile Returns (mean):")
        print("-" * 40)
        print(result.decile_returns.to_string())

    if hasattr(result, "feature_importance") and result.feature_importance is not None:
        print("\nTop 10 Features by Importance:")
        print("-" * 40)
        fi = result.feature_importance
        if isinstance(fi, pd.Series):
            print(fi.nlargest(10).to_string())
        elif isinstance(fi, pd.DataFrame):
            print(fi.head(10).to_string())

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the SRFM ML Alpha pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetically generated price and feature data.",
    )
    group.add_argument(
        "--prices",
        type=str,
        help="Path to prices CSV (dates x tickers).",
    )

    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to features CSV (dates x features). Required when --prices is used.",
    )

    # Synthetic data options
    parser.add_argument("--n", type=int, default=500, help="Number of periods.")
    parser.add_argument("--n-assets", type=int, default=20, help="Number of synthetic assets.")
    parser.add_argument("--n-features", type=int, default=30, help="Number of synthetic features.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Pipeline options
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=MODEL_CHOICES,
        help="ML model to use.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--target-horizon", type=int, default=5, help="Forward return horizon (days).")
    parser.add_argument("--ic-window", type=int, default=63, help="Rolling IC window.")
    parser.add_argument(
        "--purge-pct",
        type=float,
        default=0.02,
        help="Fraction of data to purge around each fold boundary.",
    )
    parser.add_argument(
        "--embargo-pct",
        type=float,
        default=0.01,
        help="Fraction of data to embargo after each fold.",
    )

    # Model-specific
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees (tree models).")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")

    # Rolling prediction
    parser.add_argument(
        "--rolling-pred",
        action="store_true",
        help="Also run rolling (expanding window) prediction.",
    )
    parser.add_argument(
        "--retrain-freq",
        type=int,
        default=63,
        help="Retraining frequency for rolling prediction.",
    )

    # Output
    parser.add_argument("--save-results", type=str, default=None, help="Save IC/results to CSV.")
    parser.add_argument("--save-predictions", type=str, default=None, help="Save predictions CSV.")
    parser.add_argument("--save-importance", type=str, default=None, help="Save feature importance CSV.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    add_srfm_lab_to_path()

    # Load or generate data
    if args.synthetic:
        prices_df, features_df = generate_synthetic_data(
            n=args.n, n_assets=args.n_assets,
            n_features=args.n_features, seed=args.seed,
        )
    else:
        prices_df = load_csv_with_dates(args.prices)
        if args.features is None:
            logger.error("--features required when --prices is specified.")
            sys.exit(1)
        features_df = load_csv_with_dates(args.features)

    logger.info(f"Prices shape: {prices_df.shape}")
    logger.info(f"Features shape: {features_df.shape}")

    # Build model
    model_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }
    # Filter kwargs per model
    try:
        model = build_model(args.model, model_kwargs)
        logger.info(f"Model: {type(model).__name__}")
    except Exception as e:
        logger.warning(f"Could not build model {args.model} with kwargs: {e}. Trying no kwargs.")
        model = build_model(args.model, {})

    # Run pipeline
    try:
        from strategies.ml_alpha.pipeline import MLPipeline, PurgedKFold
        pipeline = MLPipeline(
            model=model,
            purge_pct=args.purge_pct,
            embargo_pct=args.embargo_pct,
        )
        logger.info("Running ML pipeline...")
        result = pipeline.run(
            prices_df=prices_df,
            feature_df=features_df,
            target_horizon=args.target_horizon,
            n_splits=args.n_splits,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    print_pipeline_results(result)

    # Rolling prediction
    if args.rolling_pred:
        logger.info("Running rolling predictions...")
        try:
            rolling_pred = pipeline.rolling_prediction(
                prices_df=prices_df,
                feature_df=features_df,
                retrain_freq=args.retrain_freq,
                min_train_size=int(len(prices_df) * 0.3),
            )
            if args.save_predictions and rolling_pred is not None:
                if isinstance(rolling_pred, pd.Series):
                    rolling_pred.to_csv(args.save_predictions, header=True)
                else:
                    rolling_pred.to_csv(args.save_predictions)
                logger.info(f"Predictions saved to: {args.save_predictions}")
        except Exception as e:
            logger.warning(f"Rolling prediction failed: {e}")

    # Save outputs
    if args.save_results and hasattr(result, "ic_series"):
        ic = result.ic_series
        if ic is not None:
            ic.to_csv(args.save_results, header=True)
            logger.info(f"IC series saved to: {args.save_results}")

    if args.save_importance and hasattr(result, "feature_importance"):
        fi = result.feature_importance
        if fi is not None:
            if isinstance(fi, pd.Series):
                fi.to_csv(args.save_importance, header=True)
            else:
                fi.to_csv(args.save_importance)
            logger.info(f"Feature importance saved to: {args.save_importance}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
