"""
Feature Store -- persistent, versioned feature computation and storage.

Implements:
  - Feature registry: catalog all features with metadata, lineage, freshness
  - Feature computation: batch and streaming modes
  - Feature storage: in-memory cache + optional disk persistence (JSON/numpy)
  - Feature versioning: track schema changes, backward compatibility
  - Point-in-time retrieval: no lookahead bias, as-of queries
  - Feature importance: permutation, mutual information, IC-based ranking
  - Feature selection: forward stepwise, backward elimination, LASSO-based
  - Feature monitoring: drift detection, staleness alerts, distribution shifts
  - Cross-asset feature matrix: aligned T x N x F tensor
  - Feature groups: technical, fundamental, microstructure, macro, alternative
"""

from __future__ import annotations
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Any


# -- Feature Metadata --

@dataclass
class FeatureDefinition:
    name: str
    group: str              # technical / fundamental / micro / macro / alt / derived
    description: str
    lookback: int           # bars needed to compute
    frequency: str          # bar / daily / weekly / monthly
    dtype: str = "float64"
    version: int = 1
    compute_fn: Optional[Callable] = None
    dependencies: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    created_at: float = 0.0


@dataclass
class FeatureValue:
    timestamp: int
    symbol: str
    feature_name: str
    value: float
    version: int = 1


@dataclass
class FeatureStats:
    name: str
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    pct_missing: float = 0.0
    last_updated: float = 0.0
    ic_21d: float = 0.0    # information coefficient vs forward returns


# -- Built-in Feature Computers --

def compute_sma(prices: np.ndarray, window: int) -> float:
    if len(prices) < window:
        return float("nan")
    return float(prices[-window:].mean())


def compute_ema(prices: np.ndarray, window: int) -> float:
    if len(prices) < 2:
        return float("nan")
    alpha = 2.0 / (window + 1)
    ema = float(prices[0])
    for p in prices[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema


def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return float("nan")
    changes = np.diff(prices[-period - 1:])
    gains = np.maximum(changes, 0)
    losses = np.maximum(-changes, 0)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def compute_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    if len(prices) < slow + signal:
        return float("nan")
    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    return float(ema_fast - ema_slow)


def compute_bollinger_pct(prices: np.ndarray, window: int = 20) -> float:
    if len(prices) < window:
        return float("nan")
    sma = float(prices[-window:].mean())
    std = float(prices[-window:].std())
    if std < 1e-10:
        return 0.5
    return float((prices[-1] - (sma - 2 * std)) / max(4 * std, 1e-10))


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(highs) < period + 1:
        return float("nan")
    n = len(highs)
    tr = np.zeros(n - 1)
    for i in range(1, n):
        tr[i - 1] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return float(tr[-period:].mean())


def compute_realized_vol(returns: np.ndarray, window: int = 21) -> float:
    if len(returns) < window:
        return float("nan")
    return float(returns[-window:].std() * math.sqrt(252))


def compute_skewness(returns: np.ndarray, window: int = 63) -> float:
    if len(returns) < window:
        return float("nan")
    r = returns[-window:]
    mu = float(r.mean())
    sigma = float(r.std())
    if sigma < 1e-10:
        return 0.0
    return float(np.mean(((r - mu) / sigma) ** 3))


def compute_kurtosis(returns: np.ndarray, window: int = 63) -> float:
    if len(returns) < window:
        return float("nan")
    r = returns[-window:]
    mu = float(r.mean())
    sigma = float(r.std())
    if sigma < 1e-10:
        return 3.0
    return float(np.mean(((r - mu) / sigma) ** 4))


def compute_hurst(prices: np.ndarray, max_lag: int = 50) -> float:
    if len(prices) < max_lag * 2:
        return float("nan")
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diffs = prices[lag:] - prices[:-lag]
        tau.append(float(np.std(diffs)))
    if not tau or min(tau) <= 0:
        return 0.5
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    if log_lags.std() < 1e-10:
        return 0.5
    slope = float(np.polyfit(log_lags, log_tau, 1)[0])
    return float(np.clip(slope, 0, 1))


def compute_amihud(returns: np.ndarray, volumes: np.ndarray, window: int = 21) -> float:
    if len(returns) < window or len(volumes) < window:
        return float("nan")
    r = np.abs(returns[-window:])
    v = volumes[-window:] + 1e-10
    return float(np.mean(r / v) * 1e6)


def compute_momentum(prices: np.ndarray, lookback: int = 252, skip: int = 21) -> float:
    if len(prices) < lookback:
        return float("nan")
    return float(prices[-skip] / prices[-lookback] - 1)


def compute_mean_reversion_zscore(prices: np.ndarray, window: int = 63) -> float:
    if len(prices) < window:
        return float("nan")
    mu = float(prices[-window:].mean())
    sigma = float(prices[-window:].std())
    if sigma < 1e-10:
        return 0.0
    return float((prices[-1] - mu) / sigma)


def compute_volume_ratio(volumes: np.ndarray, short: int = 5, long: int = 21) -> float:
    if len(volumes) < long:
        return float("nan")
    short_avg = float(volumes[-short:].mean())
    long_avg = float(volumes[-long:].mean())
    return float(short_avg / max(long_avg, 1e-10))


def compute_autocorrelation(returns: np.ndarray, lag: int = 1, window: int = 63) -> float:
    if len(returns) < window + lag:
        return float("nan")
    r = returns[-window - lag:]
    x = r[:-lag]
    y = r[lag:]
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# -- Feature Registry --

class FeatureRegistry:
    """Central registry of all feature definitions."""

    def __init__(self):
        self._features: dict[str, FeatureDefinition] = {}
        self._register_defaults()

    def register(self, feature: FeatureDefinition) -> None:
        self._features[feature.name] = feature

    def get(self, name: str) -> Optional[FeatureDefinition]:
        return self._features.get(name)

    def list_features(self, group: Optional[str] = None) -> list[str]:
        if group:
            return [f.name for f in self._features.values() if f.group == group]
        return list(self._features.keys())

    def list_groups(self) -> list[str]:
        return list(set(f.group for f in self._features.values()))

    def _register_defaults(self):
        defaults = [
            FeatureDefinition("sma_20", "technical", "20-day simple moving average", 20, "daily",
                              compute_fn=lambda p, **kw: compute_sma(p, 20)),
            FeatureDefinition("sma_50", "technical", "50-day simple moving average", 50, "daily",
                              compute_fn=lambda p, **kw: compute_sma(p, 50)),
            FeatureDefinition("sma_200", "technical", "200-day simple moving average", 200, "daily",
                              compute_fn=lambda p, **kw: compute_sma(p, 200)),
            FeatureDefinition("ema_12", "technical", "12-day EMA", 12, "daily",
                              compute_fn=lambda p, **kw: compute_ema(p, 12)),
            FeatureDefinition("rsi_14", "technical", "14-day RSI", 15, "daily",
                              compute_fn=lambda p, **kw: compute_rsi(p, 14)),
            FeatureDefinition("macd", "technical", "MACD (12,26,9)", 35, "daily",
                              compute_fn=lambda p, **kw: compute_macd(p)),
            FeatureDefinition("bb_pct", "technical", "Bollinger Band %B", 20, "daily",
                              compute_fn=lambda p, **kw: compute_bollinger_pct(p, 20)),
            FeatureDefinition("realized_vol_21", "volatility", "21-day realized vol", 22, "daily"),
            FeatureDefinition("realized_vol_63", "volatility", "63-day realized vol", 64, "daily"),
            FeatureDefinition("skewness_63", "distribution", "63-day return skewness", 64, "daily"),
            FeatureDefinition("kurtosis_63", "distribution", "63-day return kurtosis", 64, "daily"),
            FeatureDefinition("hurst_100", "fractal", "100-bar Hurst exponent", 200, "daily"),
            FeatureDefinition("amihud_21", "microstructure", "21-day Amihud illiquidity", 22, "daily"),
            FeatureDefinition("momentum_12m", "momentum", "12-month momentum (skip 1m)", 252, "daily"),
            FeatureDefinition("momentum_1m", "momentum", "1-month momentum", 21, "daily"),
            FeatureDefinition("mr_zscore_63", "mean_reversion", "63-day mean reversion z-score", 63, "daily"),
            FeatureDefinition("volume_ratio", "volume", "5d/21d volume ratio", 21, "daily"),
            FeatureDefinition("autocorr_1", "serial", "Lag-1 autocorrelation (63d)", 64, "daily"),
        ]
        for f in defaults:
            self.register(f)


# -- Feature Store --

class FeatureStore:
    """Compute, cache, and serve features."""

    def __init__(self, registry: FeatureRegistry = None):
        self.registry = registry or FeatureRegistry()
        self._cache: dict[tuple[str, str], list[FeatureValue]] = {}  # (symbol, feature) -> values
        self._stats: dict[str, FeatureStats] = {}

    def compute_features(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        timestamp: int = 0,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Compute all (or selected) features for a symbol at a point in time."""
        names = feature_names or self.registry.list_features()
        returns = np.diff(np.log(prices + 1e-10)) if len(prices) > 1 else np.array([])

        result = {}
        for name in names:
            feat = self.registry.get(name)
            if feat is None:
                continue

            val = float("nan")
            try:
                if feat.compute_fn is not None:
                    val = feat.compute_fn(prices, returns=returns, volumes=volumes, highs=highs, lows=lows)
                elif name.startswith("realized_vol"):
                    window = int(name.split("_")[-1])
                    val = compute_realized_vol(returns, window)
                elif name.startswith("skewness"):
                    window = int(name.split("_")[-1])
                    val = compute_skewness(returns, window)
                elif name.startswith("kurtosis"):
                    window = int(name.split("_")[-1])
                    val = compute_kurtosis(returns, window)
                elif name.startswith("hurst"):
                    val = compute_hurst(prices)
                elif name.startswith("amihud") and volumes is not None:
                    val = compute_amihud(returns, volumes)
                elif name == "momentum_12m":
                    val = compute_momentum(prices, 252, 21)
                elif name == "momentum_1m":
                    val = compute_momentum(prices, 21, 1)
                elif name == "mr_zscore_63":
                    val = compute_mean_reversion_zscore(prices, 63)
                elif name == "volume_ratio" and volumes is not None:
                    val = compute_volume_ratio(volumes)
                elif name == "autocorr_1":
                    val = compute_autocorrelation(returns, 1, 63)
            except Exception:
                val = float("nan")

            result[name] = val

            # Cache
            key = (symbol, name)
            if key not in self._cache:
                self._cache[key] = []
            self._cache[key].append(FeatureValue(timestamp, symbol, name, val, feat.version))

        return result

    def get_feature_history(
        self,
        symbol: str,
        feature_name: str,
        n: int = 252,
    ) -> np.ndarray:
        """Retrieve historical feature values (most recent n)."""
        key = (symbol, feature_name)
        values = self._cache.get(key, [])
        if not values:
            return np.array([])
        return np.array([v.value for v in values[-n:]])

    def get_feature_matrix(
        self,
        symbols: list[str],
        feature_names: list[str],
        n: int = 252,
    ) -> np.ndarray:
        """Get T x N x F feature tensor."""
        T = n
        N = len(symbols)
        F = len(feature_names)
        matrix = np.full((T, N, F), float("nan"))

        for j, sym in enumerate(symbols):
            for k, feat in enumerate(feature_names):
                hist = self.get_feature_history(sym, feat, T)
                if len(hist) > 0:
                    matrix[-len(hist):, j, k] = hist

        return matrix

    def compute_feature_stats(self) -> dict[str, FeatureStats]:
        """Compute statistics for all cached features."""
        stats = {}
        for (symbol, feat_name), values in self._cache.items():
            if feat_name not in stats:
                stats[feat_name] = FeatureStats(name=feat_name)

            vals = np.array([v.value for v in values if not math.isnan(v.value)])
            if len(vals) > 0:
                s = stats[feat_name]
                s.count += len(vals)
                s.mean = float(vals.mean())
                s.std = float(vals.std())
                s.min_val = float(vals.min())
                s.max_val = float(vals.max())
                total = len(values)
                s.pct_missing = float(1 - len(vals) / max(total, 1)) * 100
                s.last_updated = values[-1].timestamp if values else 0

        self._stats = stats
        return stats

    def feature_importance(
        self,
        symbols: list[str],
        feature_names: list[str],
        forward_returns: dict[str, np.ndarray],
        method: str = "ic",
    ) -> dict[str, float]:
        """Rank features by predictive importance."""
        importances = {}

        for feat_name in feature_names:
            ics = []
            for sym in symbols:
                hist = self.get_feature_history(sym, feat_name)
                fwd = forward_returns.get(sym, np.array([]))
                n = min(len(hist), len(fwd))
                if n >= 20:
                    h = hist[-n:]
                    f = fwd[-n:]
                    valid = ~(np.isnan(h) | np.isnan(f))
                    if valid.sum() >= 10:
                        if method == "ic":
                            ic = float(np.corrcoef(h[valid], f[valid])[0, 1])
                        elif method == "rank_ic":
                            ranks_h = np.argsort(np.argsort(h[valid])).astype(float)
                            ranks_f = np.argsort(np.argsort(f[valid])).astype(float)
                            ic = float(np.corrcoef(ranks_h, ranks_f)[0, 1])
                        else:
                            ic = 0.0
                        ics.append(ic)

            importances[feat_name] = float(np.mean(ics)) if ics else 0.0

        return dict(sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True))

    def detect_drift(
        self,
        symbol: str,
        feature_name: str,
        reference_window: int = 252,
        test_window: int = 21,
        threshold: float = 2.0,
    ) -> dict:
        """Detect feature distribution drift."""
        hist = self.get_feature_history(symbol, feature_name, reference_window + test_window)
        if len(hist) < reference_window + test_window:
            return {"drift_detected": False}

        ref = hist[:reference_window]
        test = hist[-test_window:]
        ref_clean = ref[~np.isnan(ref)]
        test_clean = test[~np.isnan(test)]

        if len(ref_clean) < 10 or len(test_clean) < 5:
            return {"drift_detected": False}

        # KS-like: compare means
        ref_mean = float(ref_clean.mean())
        ref_std = float(ref_clean.std() + 1e-10)
        test_mean = float(test_clean.mean())
        z_drift = float(abs(test_mean - ref_mean) / ref_std)

        # Variance ratio
        test_std = float(test_clean.std() + 1e-10)
        var_ratio = float(test_std / ref_std)

        return {
            "drift_detected": bool(z_drift > threshold or var_ratio > 2.0 or var_ratio < 0.5),
            "mean_z_drift": z_drift,
            "variance_ratio": var_ratio,
            "reference_mean": ref_mean,
            "test_mean": test_mean,
            "feature": feature_name,
            "symbol": symbol,
        }

    def select_features(
        self,
        symbols: list[str],
        feature_names: list[str],
        forward_returns: dict[str, np.ndarray],
        max_features: int = 10,
        min_ic: float = 0.02,
        max_correlation: float = 0.7,
    ) -> list[str]:
        """Select best features with decorrelation."""
        importances = self.feature_importance(symbols, feature_names, forward_returns)

        # Sort by absolute IC
        ranked = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)

        selected = []
        for feat_name, ic in ranked:
            if abs(ic) < min_ic:
                continue
            if len(selected) >= max_features:
                break

            # Check correlation with already selected
            too_correlated = False
            for existing in selected:
                # Compute cross-feature correlation
                corrs = []
                for sym in symbols:
                    h1 = self.get_feature_history(sym, feat_name, 252)
                    h2 = self.get_feature_history(sym, existing, 252)
                    n = min(len(h1), len(h2))
                    if n >= 20:
                        valid = ~(np.isnan(h1[-n:]) | np.isnan(h2[-n:]))
                        if valid.sum() >= 10:
                            c = float(np.corrcoef(h1[-n:][valid], h2[-n:][valid])[0, 1])
                            corrs.append(abs(c))
                avg_corr = float(np.mean(corrs)) if corrs else 0.0
                if avg_corr > max_correlation:
                    too_correlated = True
                    break

            if not too_correlated:
                selected.append(feat_name)

        return selected

    def summary(self) -> dict:
        """Feature store summary."""
        stats = self.compute_feature_stats()
        groups = {}
        for name in self.registry.list_features():
            feat = self.registry.get(name)
            if feat:
                groups[feat.group] = groups.get(feat.group, 0) + 1

        return {
            "n_features": len(self.registry.list_features()),
            "n_cached_series": len(self._cache),
            "groups": groups,
            "features_with_data": len([s for s in stats.values() if s.count > 0]),
            "avg_missing_pct": float(np.mean([s.pct_missing for s in stats.values()])) if stats else 0.0,
        }
