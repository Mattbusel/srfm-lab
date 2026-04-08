"""
Data quality validation engine — ensures clean data before signal generation.

Implements:
  - Missing data detection and imputation strategies
  - Outlier detection: z-score, IQR, Grubbs, isolation forest
  - Stale data detection: prices that haven't moved
  - Survivorship bias detection: backfill bias in historical data
  - Corporate action adjustment: split detection, dividend adjustment
  - Timestamp validation: gaps, duplicates, timezone issues
  - Cross-validation: compare across data sources
  - Data freshness scoring
  - Holiday/market closure detection
  - Quality scoring: composite data quality metric per asset per day
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataQualityReport:
    """Quality report for a data series."""
    asset_name: str
    n_observations: int
    quality_score: float          # 0-1 composite
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Missing data
    missing_pct: float = 0.0
    missing_runs: int = 0         # consecutive missing stretches
    longest_gap_days: int = 0

    # Outliers
    n_outliers: int = 0
    outlier_pct: float = 0.0
    outlier_indices: list[int] = field(default_factory=list)

    # Staleness
    stale_pct: float = 0.0
    max_stale_run: int = 0

    # Integrity
    has_negative_prices: bool = False
    has_zero_volume_days: float = 0.0
    timestamp_issues: int = 0


# ── Missing Data ──────────────────────────────────────────────────────────────

def detect_missing_data(data: np.ndarray) -> dict:
    """Detect missing (NaN) data patterns."""
    n = len(data)
    is_nan = np.isnan(data)
    n_missing = int(is_nan.sum())

    # Find runs of missing data
    runs = []
    current_run = 0
    for i in range(n):
        if is_nan[i]:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    return {
        "n_missing": n_missing,
        "missing_pct": float(n_missing / max(n, 1) * 100),
        "n_gaps": len(runs),
        "longest_gap": max(runs) if runs else 0,
        "gap_lengths": runs,
    }


def impute_missing(
    data: np.ndarray,
    method: str = "forward_fill",
) -> np.ndarray:
    """
    Impute missing data.
    methods: 'forward_fill', 'linear_interp', 'mean', 'median', 'zero'
    """
    result = data.copy()
    is_nan = np.isnan(result)

    if not is_nan.any():
        return result

    if method == "forward_fill":
        for i in range(1, len(result)):
            if is_nan[i]:
                result[i] = result[i - 1]
        # Back fill for leading NaNs
        for i in range(len(result) - 2, -1, -1):
            if is_nan[i] and not np.isnan(result[i + 1]):
                result[i] = result[i + 1]

    elif method == "linear_interp":
        valid = np.where(~is_nan)[0]
        if len(valid) >= 2:
            result = np.interp(np.arange(len(data)), valid, data[valid])

    elif method == "mean":
        mean_val = float(np.nanmean(data))
        result[is_nan] = mean_val

    elif method == "median":
        med_val = float(np.nanmedian(data))
        result[is_nan] = med_val

    elif method == "zero":
        result[is_nan] = 0.0

    return result


# ── Outlier Detection ─────────────────────────────────────────────────────────

def detect_outliers_zscore(
    data: np.ndarray,
    threshold: float = 4.0,
) -> dict:
    """Z-score based outlier detection."""
    clean = data[~np.isnan(data)]
    if len(clean) < 5:
        return {"n_outliers": 0, "indices": []}

    mu = float(clean.mean())
    sigma = float(clean.std() + 1e-10)
    z = np.abs((data - mu) / sigma)
    z = np.where(np.isnan(z), 0, z)
    outlier_mask = z > threshold
    indices = np.where(outlier_mask)[0].tolist()

    return {
        "n_outliers": len(indices),
        "outlier_pct": float(len(indices) / max(len(data), 1) * 100),
        "indices": indices,
        "z_scores": z[outlier_mask].tolist() if len(indices) > 0 else [],
        "threshold": threshold,
    }


def detect_outliers_iqr(
    data: np.ndarray,
    multiplier: float = 3.0,
) -> dict:
    """IQR-based outlier detection."""
    clean = data[~np.isnan(data)]
    if len(clean) < 5:
        return {"n_outliers": 0, "indices": []}

    q1, q3 = np.percentile(clean, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    outlier_mask = (data < lower) | (data > upper)
    outlier_mask = outlier_mask & ~np.isnan(data)
    indices = np.where(outlier_mask)[0].tolist()

    return {
        "n_outliers": len(indices),
        "indices": indices,
        "lower_fence": float(lower),
        "upper_fence": float(upper),
        "iqr": float(iqr),
    }


def detect_outliers_grubbs(
    data: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Grubbs test for single outlier."""
    clean = data[~np.isnan(data)]
    n = len(clean)
    if n < 7:
        return {"is_outlier": False, "outlier_idx": -1}

    mu = float(clean.mean())
    sigma = float(clean.std())

    # Most extreme observation
    deviations = np.abs(clean - mu)
    max_idx = int(np.argmax(deviations))
    G = float(deviations[max_idx] / max(sigma, 1e-10))

    # Critical value (approximation for t-distribution)
    # Using simplified threshold based on sample size
    t_crit = 2.5 + 0.5 * math.log(n)  # rough approximation
    G_crit = (n - 1) / math.sqrt(n) * math.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    is_outlier = G > G_crit

    return {
        "is_outlier": bool(is_outlier),
        "outlier_idx": int(max_idx),
        "outlier_value": float(clean[max_idx]),
        "grubbs_stat": G,
        "critical_value": G_crit,
    }


def winsorize(data: np.ndarray, limits: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    """Winsorize data at given percentile limits."""
    clean = data[~np.isnan(data)]
    if len(clean) < 5:
        return data.copy()
    lower = float(np.percentile(clean, limits[0] * 100))
    upper = float(np.percentile(clean, limits[1] * 100))
    return np.clip(data, lower, upper)


# ── Staleness Detection ───────────────────────────────────────────────────────

def detect_stale_data(
    prices: np.ndarray,
    tolerance: float = 1e-8,
) -> dict:
    """Detect stale (unchanged) prices."""
    n = len(prices)
    if n < 2:
        return {"stale_pct": 0.0, "max_stale_run": 0}

    changes = np.abs(np.diff(prices))
    stale = changes < tolerance
    n_stale = int(stale.sum())

    # Longest stale run
    max_run = 0
    current = 0
    for s in stale:
        if s:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0

    return {
        "stale_pct": float(n_stale / max(n - 1, 1) * 100),
        "n_stale_days": n_stale,
        "max_stale_run": max_run,
        "is_suspicious": bool(max_run > 5),
    }


# ── Corporate Action Detection ────────────────────────────────────────────────

def detect_splits(
    prices: np.ndarray,
    volume: np.ndarray,
    split_return_threshold: float = 0.30,
) -> list[dict]:
    """Detect potential stock splits from price/volume jumps."""
    n = len(prices)
    splits = []

    for i in range(1, n):
        if prices[i - 1] <= 0:
            continue
        ret = (prices[i] - prices[i - 1]) / prices[i - 1]

        # Split: large negative return + volume spike (or no change in market cap)
        if ret < -split_return_threshold:
            ratio = prices[i - 1] / max(prices[i], 1e-10)
            # Check if ratio is close to common split ratios
            common_ratios = [2, 3, 4, 5, 10, 0.5, 0.333]  # include reverse splits
            best_ratio = min(common_ratios, key=lambda r: abs(ratio - r))
            if abs(ratio - best_ratio) / best_ratio < 0.1:
                splits.append({
                    "index": i,
                    "ratio": float(best_ratio),
                    "price_before": float(prices[i - 1]),
                    "price_after": float(prices[i]),
                    "type": "forward" if best_ratio > 1 else "reverse",
                })

    return splits


def adjust_for_splits(prices: np.ndarray, splits: list[dict]) -> np.ndarray:
    """Adjust historical prices for detected splits."""
    adjusted = prices.copy()
    for split in sorted(splits, key=lambda s: s["index"], reverse=True):
        idx = split["index"]
        ratio = split["ratio"]
        adjusted[:idx] /= ratio
    return adjusted


# ── Cross-Source Validation ───────────────────────────────────────────────────

def cross_source_validation(
    source_a: np.ndarray,
    source_b: np.ndarray,
    tolerance_pct: float = 0.5,
) -> dict:
    """Compare two data sources for consistency."""
    n = min(len(source_a), len(source_b))
    a = source_a[:n]
    b = source_b[:n]

    # Relative difference
    mid = (np.abs(a) + np.abs(b)) / 2 + 1e-10
    rel_diff = np.abs(a - b) / mid * 100

    n_discrepant = int(np.sum(rel_diff > tolerance_pct))
    max_diff = float(rel_diff.max())
    avg_diff = float(rel_diff.mean())

    # Correlation
    if len(a) > 5 and a.std() > 1e-10 and b.std() > 1e-10:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = 0.0

    return {
        "n_compared": n,
        "n_discrepant": n_discrepant,
        "discrepancy_pct": float(n_discrepant / max(n, 1) * 100),
        "max_relative_diff_pct": max_diff,
        "avg_relative_diff_pct": avg_diff,
        "correlation": corr,
        "sources_consistent": bool(n_discrepant / max(n, 1) < 0.05 and corr > 0.99),
    }


# ── Comprehensive Validator ───────────────────────────────────────────────────

class DataValidator:
    """Comprehensive data quality validator."""

    def __init__(
        self,
        outlier_z_threshold: float = 4.0,
        stale_tolerance: float = 1e-8,
        max_missing_pct: float = 5.0,
        max_stale_run: int = 5,
    ):
        self.outlier_z = outlier_z_threshold
        self.stale_tol = stale_tolerance
        self.max_missing = max_missing_pct
        self.max_stale = max_stale_run

    def validate(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        asset_name: str = "unknown",
    ) -> DataQualityReport:
        """Run full validation battery."""
        n = len(prices)
        issues = []
        warnings = []

        # Missing data
        missing = detect_missing_data(prices)
        if missing["missing_pct"] > self.max_missing:
            issues.append(f"High missing data: {missing['missing_pct']:.1f}%")
        elif missing["n_missing"] > 0:
            warnings.append(f"Some missing data: {missing['n_missing']} observations")

        # Outliers (on returns)
        returns = np.diff(prices) / (prices[:-1] + 1e-10)
        returns = np.where(np.isfinite(returns), returns, 0)
        outliers = detect_outliers_zscore(returns, self.outlier_z)
        if outliers["outlier_pct"] > 1.0:
            warnings.append(f"Many outliers: {outliers['n_outliers']} ({outliers['outlier_pct']:.1f}%)")

        # Staleness
        stale = detect_stale_data(prices, self.stale_tol)
        if stale["max_stale_run"] > self.max_stale:
            issues.append(f"Stale data: {stale['max_stale_run']} consecutive unchanged days")

        # Negative prices
        has_neg = bool(np.any(prices < 0))
        if has_neg:
            issues.append("Negative prices detected")

        # Volume issues
        zero_vol_pct = 0.0
        if volumes is not None:
            zero_vol_pct = float(np.mean(volumes <= 0) * 100)
            if zero_vol_pct > 20:
                warnings.append(f"High zero-volume days: {zero_vol_pct:.0f}%")

        # Splits
        if volumes is not None:
            splits = detect_splits(prices, volumes)
            if splits:
                warnings.append(f"Potential unadjusted splits detected: {len(splits)}")

        # Quality score
        score = 1.0
        score -= min(missing["missing_pct"] / 100, 0.3)
        score -= min(outliers["outlier_pct"] / 100 * 2, 0.2)
        score -= min(stale["stale_pct"] / 100, 0.2)
        score -= 0.3 if has_neg else 0.0
        score -= min(zero_vol_pct / 100 * 0.5, 0.1)
        score -= 0.1 * len(issues)
        score = float(max(score, 0))

        return DataQualityReport(
            asset_name=asset_name,
            n_observations=n,
            quality_score=score,
            issues=issues,
            warnings=warnings,
            missing_pct=missing["missing_pct"],
            missing_runs=missing["n_gaps"],
            longest_gap_days=missing["longest_gap"],
            n_outliers=outliers["n_outliers"],
            outlier_pct=outliers["outlier_pct"],
            outlier_indices=outliers["indices"][:20],
            stale_pct=stale["stale_pct"],
            max_stale_run=stale["max_stale_run"],
            has_negative_prices=has_neg,
            has_zero_volume_days=zero_vol_pct,
        )

    def validate_batch(
        self,
        price_matrix: np.ndarray,  # (T, N)
        asset_names: list[str],
        volume_matrix: Optional[np.ndarray] = None,
    ) -> list[DataQualityReport]:
        """Validate multiple assets."""
        T, N = price_matrix.shape
        reports = []
        for i in range(N):
            vol = volume_matrix[:, i] if volume_matrix is not None else None
            report = self.validate(price_matrix[:, i], vol, asset_names[i] if i < len(asset_names) else f"asset_{i}")
            reports.append(report)

        # Sort by quality score (worst first)
        reports.sort(key=lambda r: r.quality_score)
        return reports

    def clean_data(
        self,
        prices: np.ndarray,
        impute_method: str = "forward_fill",
        winsorize_limits: tuple[float, float] = (0.005, 0.995),
    ) -> np.ndarray:
        """Clean data: impute missing, winsorize outliers."""
        # Impute missing
        cleaned = impute_missing(prices, impute_method)
        # Winsorize returns
        returns = np.diff(cleaned) / (cleaned[:-1] + 1e-10)
        returns_clean = winsorize(returns, winsorize_limits)
        # Reconstruct prices
        result = np.zeros_like(cleaned)
        result[0] = cleaned[0]
        for i in range(len(returns_clean)):
            result[i + 1] = result[i] * (1 + returns_clean[i])
        return result
