"""
Regime Oracle — ensemble regime detection and forecasting.

Combines multiple regime detection methods into a consensus view:
  - Hidden Markov Model regimes (volatility states)
  - Trend/momentum regime (directional bias)
  - Correlation regime (herding vs dispersed)
  - Liquidity regime (normal vs stressed)
  - Macro regime (expansion/contraction)
  - Volatility regime (low/normal/high/crisis)
  - Ensemble consensus with confidence
  - Regime transition probability forecasting
  - Historical regime analytics: duration, frequency, characteristics
  - Regime-conditional strategy recommendations
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Regime Definitions ────────────────────────────────────────────────────────

REGIME_NAMES = [
    "risk_on",          # low vol, positive drift, low corr
    "risk_off",         # rising vol, negative drift, rising corr
    "trending_up",      # positive momentum, moderate vol
    "trending_down",    # negative momentum, moderate vol
    "mean_reverting",   # oscillating, low autocorrelation
    "high_volatility",  # high vol, uncertain direction
    "crisis",           # extreme vol, high corr, negative drift
    "recovery",         # vol declining, positive drift from lows
]


@dataclass
class RegimeState:
    """Current regime assessment."""
    primary_regime: str
    confidence: float               # 0-1
    regime_probabilities: dict[str, float]
    secondary_regime: str
    transition_probability: float   # P(regime change in next 5 days)

    # Sub-regime details
    vol_regime: str                 # low/normal/high/crisis
    trend_regime: str               # up/down/flat
    correlation_regime: str         # dispersed/normal/herding
    liquidity_regime: str           # normal/thinning/stressed

    # Duration
    days_in_current_regime: int
    avg_regime_duration: float

    # Characteristics
    expected_return_annual: float
    expected_vol_annual: float
    expected_sharpe: float


# ── Individual Regime Detectors ───────────────────────────────────────────────

class VolatilityRegimeDetector:
    """Classify volatility regime from return data."""

    def __init__(
        self,
        low_threshold: float = 0.10,
        high_threshold: float = 0.25,
        crisis_threshold: float = 0.40,
    ):
        self.low_t = low_threshold
        self.high_t = high_threshold
        self.crisis_t = crisis_threshold

    def detect(self, returns: np.ndarray, window: int = 21) -> dict:
        if len(returns) < window:
            return {"regime": "normal", "vol_annualized": 0.15, "vol_percentile": 0.5}

        recent_vol = float(returns[-window:].std() * math.sqrt(252))

        # Historical percentile
        if len(returns) >= 252:
            rolling_vols = []
            for t in range(window, len(returns)):
                v = float(returns[t - window:t].std() * math.sqrt(252))
                rolling_vols.append(v)
            percentile = float(np.mean(np.array(rolling_vols) <= recent_vol))
        else:
            percentile = 0.5

        if recent_vol > self.crisis_t:
            regime = "crisis"
        elif recent_vol > self.high_t:
            regime = "high"
        elif recent_vol < self.low_t:
            regime = "low"
        else:
            regime = "normal"

        return {
            "regime": regime,
            "vol_annualized": recent_vol,
            "vol_percentile": percentile,
        }


class TrendRegimeDetector:
    """Classify trend regime from price data."""

    def __init__(self, ma_fast: int = 20, ma_slow: int = 60, adx_period: int = 14):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.adx_period = adx_period

    def detect(self, prices: np.ndarray) -> dict:
        n = len(prices)
        if n < self.ma_slow + 5:
            return {"regime": "flat", "trend_strength": 0.0, "direction": 0.0}

        # Moving averages
        ma_f = float(prices[-self.ma_fast:].mean())
        ma_s = float(prices[-self.ma_slow:].mean())

        # Trend direction
        if ma_f > ma_s * 1.01:
            direction = 1.0
        elif ma_f < ma_s * 0.99:
            direction = -1.0
        else:
            direction = 0.0

        # Trend strength: R-squared of linear regression on recent prices
        recent = prices[-self.ma_fast:]
        t = np.arange(len(recent))
        if recent.std() > 1e-10:
            slope, intercept = np.polyfit(t, recent, 1)
            fitted = slope * t + intercept
            ss_res = float(np.sum((recent - fitted)**2))
            ss_tot = float(np.sum((recent - recent.mean())**2))
            r2 = float(1 - ss_res / max(ss_tot, 1e-10))
            r2 = max(r2, 0)
        else:
            r2 = 0.0
            slope = 0.0

        # ADX-like: absolute slope normalized
        norm_slope = float(abs(slope) / max(abs(prices[-1]), 1e-10) * 252)

        if r2 > 0.6 and norm_slope > 0.1:
            regime = "up" if direction > 0 else "down"
        elif r2 < 0.2:
            regime = "choppy"
        else:
            regime = "flat"

        return {
            "regime": regime,
            "trend_strength": float(r2),
            "direction": float(direction),
            "slope_annualized": float(norm_slope),
        }


class CorrelationRegimeDetector:
    """Classify correlation regime from multi-asset returns."""

    def __init__(self, herding_threshold: float = 0.55, dispersed_threshold: float = 0.20):
        self.herding_t = herding_threshold
        self.dispersed_t = dispersed_threshold

    def detect(self, returns: np.ndarray, window: int = 21) -> dict:
        """returns: (T, N) multi-asset returns."""
        T, N = returns.shape
        if T < window or N < 2:
            return {"regime": "normal", "avg_correlation": 0.3}

        recent = returns[-window:]
        corr = np.corrcoef(recent.T)
        upper = corr[np.triu_indices(N, k=1)]
        avg_corr = float(upper.mean())

        # Historical comparison
        if T >= 126:
            hist_corrs = []
            for t in range(window, T, 5):
                c = np.corrcoef(returns[t-window:t].T)
                hist_corrs.append(float(c[np.triu_indices(N, k=1)].mean()))
            percentile = float(np.mean(np.array(hist_corrs) <= avg_corr))
        else:
            percentile = 0.5

        if avg_corr > self.herding_t:
            regime = "herding"
        elif avg_corr < self.dispersed_t:
            regime = "dispersed"
        else:
            regime = "normal"

        return {
            "regime": regime,
            "avg_correlation": avg_corr,
            "percentile": percentile,
            "max_eigenvalue_ratio": float(np.linalg.eigvalsh(corr)[-1] / N),
        }


class MacroRegimeDetector:
    """Classify macro regime from yield curve, credit spreads, etc."""

    def detect(
        self,
        yield_curve_slope: float,    # 2s10s in bps
        credit_spread_z: float,      # IG OAS z-score
        pmi_level: float,            # ISM PMI
        inflation_surprise: float = 0.0,
    ) -> dict:
        # Simple rule-based classification
        if pmi_level > 52 and yield_curve_slope > 0 and credit_spread_z < 1:
            regime = "expansion"
        elif pmi_level > 50 and credit_spread_z > 1:
            regime = "late_cycle"
        elif pmi_level < 48 and credit_spread_z > 1.5:
            regime = "contraction"
        elif pmi_level < 50 and yield_curve_slope < -20:
            regime = "recession_risk"
        elif pmi_level > 48 and pmi_level < 52:
            regime = "slowdown"
        else:
            regime = "transition"

        # Stagflation check
        if inflation_surprise > 1.0 and pmi_level < 50:
            regime = "stagflation"

        return {
            "regime": regime,
            "yield_curve_slope_bps": float(yield_curve_slope),
            "credit_stress": float(credit_spread_z),
            "growth_indicator": float(pmi_level),
        }


# ── Regime Transition Model ──────────────────────────────────────────────────

class RegimeTransitionModel:
    """Estimate and forecast regime transitions."""

    def __init__(self, n_regimes: int = 8):
        self.n_regimes = n_regimes
        self._transition_counts = np.ones((n_regimes, n_regimes)) * 0.1  # Dirichlet prior
        self._regime_history: list[int] = []

    def update(self, regime_idx: int) -> None:
        self._regime_history.append(regime_idx)
        if len(self._regime_history) >= 2:
            prev = self._regime_history[-2]
            self._transition_counts[prev, regime_idx] += 1

    @property
    def transition_matrix(self) -> np.ndarray:
        row_sums = self._transition_counts.sum(axis=1, keepdims=True)
        return self._transition_counts / (row_sums + 1e-10)

    def predict_next(self, current_regime: int, horizon: int = 1) -> np.ndarray:
        """Predict regime distribution at horizon steps ahead."""
        P = self.transition_matrix
        dist = np.zeros(self.n_regimes)
        dist[current_regime] = 1.0
        for _ in range(horizon):
            dist = dist @ P
        return dist

    def expected_duration(self, regime_idx: int) -> float:
        """Expected duration (days) in a regime."""
        P = self.transition_matrix
        p_stay = P[regime_idx, regime_idx]
        if p_stay >= 1.0:
            return float("inf")
        return float(1 / max(1 - p_stay, 1e-6))

    def stationary_distribution(self) -> np.ndarray:
        """Long-run regime probabilities."""
        P = self.transition_matrix
        n = self.n_regimes
        # Solve pi @ P = pi, sum(pi) = 1
        A = (P.T - np.eye(n))
        A[-1] = 1.0
        b = np.zeros(n)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.maximum(pi, 0)
            pi /= pi.sum()
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n
        return pi

    def regime_change_probability(self, current: int, horizon: int = 5) -> float:
        """Probability of leaving current regime within horizon days."""
        dist = self.predict_next(current, horizon)
        return float(1 - dist[current])


# ── Regime Ensemble ───────────────────────────────────────────────────────────

class RegimeOracle:
    """
    Ensemble regime detector: combines multiple signals into consensus regime.
    """

    def __init__(self):
        self.vol_detector = VolatilityRegimeDetector()
        self.trend_detector = TrendRegimeDetector()
        self.corr_detector = CorrelationRegimeDetector()
        self.macro_detector = MacroRegimeDetector()
        self.transition_model = RegimeTransitionModel()

        self._current_regime: str = "risk_on"
        self._days_in_regime: int = 0
        self._regime_history: list[str] = []

        # Regime characteristics (empirical averages)
        self._regime_stats: dict[str, dict] = {
            "risk_on":         {"ret": 0.15, "vol": 0.12, "sharpe": 1.25},
            "risk_off":        {"ret": -0.05, "vol": 0.22, "sharpe": -0.23},
            "trending_up":     {"ret": 0.20, "vol": 0.15, "sharpe": 1.33},
            "trending_down":   {"ret": -0.15, "vol": 0.20, "sharpe": -0.75},
            "mean_reverting":  {"ret": 0.05, "vol": 0.14, "sharpe": 0.36},
            "high_volatility": {"ret": 0.0, "vol": 0.30, "sharpe": 0.0},
            "crisis":          {"ret": -0.30, "vol": 0.50, "sharpe": -0.60},
            "recovery":        {"ret": 0.25, "vol": 0.25, "sharpe": 1.00},
        }

    def detect(
        self,
        returns: np.ndarray,           # single asset or portfolio returns
        prices: Optional[np.ndarray] = None,
        multi_asset_returns: Optional[np.ndarray] = None,
        yield_curve_slope: float = 50.0,
        credit_spread_z: float = 0.0,
        pmi_level: float = 52.0,
    ) -> RegimeState:
        """Run all detectors and build consensus regime."""

        # --- Individual detectors ---
        vol_result = self.vol_detector.detect(returns)
        trend_result = self.trend_detector.detect(prices if prices is not None else np.cumprod(1 + returns))

        if multi_asset_returns is not None and multi_asset_returns.ndim == 2:
            corr_result = self.corr_detector.detect(multi_asset_returns)
        else:
            corr_result = {"regime": "normal", "avg_correlation": 0.3}

        macro_result = self.macro_detector.detect(yield_curve_slope, credit_spread_z, pmi_level)

        # --- Consensus mapping ---
        regime_scores = {r: 0.0 for r in REGIME_NAMES}

        # Vol regime mapping
        vol_r = vol_result["regime"]
        if vol_r == "crisis":
            regime_scores["crisis"] += 3.0
            regime_scores["high_volatility"] += 1.0
        elif vol_r == "high":
            regime_scores["high_volatility"] += 2.0
            regime_scores["risk_off"] += 1.0
        elif vol_r == "low":
            regime_scores["risk_on"] += 2.0
            regime_scores["mean_reverting"] += 0.5

        # Trend regime mapping
        trend_r = trend_result["regime"]
        trend_strength = trend_result["trend_strength"]
        if trend_r == "up":
            regime_scores["trending_up"] += 2.0 * trend_strength
            regime_scores["risk_on"] += 1.0
        elif trend_r == "down":
            regime_scores["trending_down"] += 2.0 * trend_strength
            regime_scores["risk_off"] += 1.0
        elif trend_r == "choppy":
            regime_scores["mean_reverting"] += 2.0

        # Correlation regime mapping
        corr_r = corr_result["regime"]
        if corr_r == "herding":
            regime_scores["crisis"] += 1.5
            regime_scores["risk_off"] += 1.0
        elif corr_r == "dispersed":
            regime_scores["risk_on"] += 1.0
            regime_scores["mean_reverting"] += 0.5

        # Macro regime mapping
        macro_r = macro_result["regime"]
        if macro_r in ("expansion",):
            regime_scores["risk_on"] += 1.5
            regime_scores["trending_up"] += 0.5
        elif macro_r in ("contraction", "recession_risk"):
            regime_scores["risk_off"] += 1.5
            regime_scores["trending_down"] += 0.5
        elif macro_r == "stagflation":
            regime_scores["high_volatility"] += 1.0
            regime_scores["risk_off"] += 1.0
        elif macro_r == "late_cycle":
            regime_scores["high_volatility"] += 0.5

        # Recovery detection: was crisis, now vol declining + positive returns
        if self._current_regime in ("crisis", "risk_off") and vol_r in ("normal", "low") and trend_r == "up":
            regime_scores["recovery"] += 3.0

        # --- Select primary regime ---
        total_score = sum(regime_scores.values()) + 1e-10
        regime_probs = {r: s / total_score for r, s in regime_scores.items()}
        primary = max(regime_scores, key=regime_scores.get)
        confidence = float(regime_scores[primary] / total_score)

        # Secondary regime
        sorted_regimes = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
        secondary = sorted_regimes[1][0] if len(sorted_regimes) > 1 else primary

        # --- Update state ---
        if primary == self._current_regime:
            self._days_in_regime += 1
        else:
            self._days_in_regime = 1
            self._current_regime = primary

        self._regime_history.append(primary)
        regime_idx = REGIME_NAMES.index(primary) if primary in REGIME_NAMES else 0
        self.transition_model.update(regime_idx)

        # Transition probability
        trans_prob = self.transition_model.regime_change_probability(regime_idx, horizon=5)

        # Average duration
        avg_dur = self.transition_model.expected_duration(regime_idx)

        # Regime characteristics
        stats = self._regime_stats.get(primary, {"ret": 0, "vol": 0.2, "sharpe": 0})

        return RegimeState(
            primary_regime=primary,
            confidence=confidence,
            regime_probabilities=regime_probs,
            secondary_regime=secondary,
            transition_probability=trans_prob,
            vol_regime=vol_result["regime"],
            trend_regime=trend_result["regime"],
            correlation_regime=corr_result["regime"],
            liquidity_regime="normal",  # would come from liquidity oracle
            days_in_current_regime=self._days_in_regime,
            avg_regime_duration=avg_dur,
            expected_return_annual=stats["ret"],
            expected_vol_annual=stats["vol"],
            expected_sharpe=stats["sharpe"],
        )

    def regime_strategy_recommendations(self, state: RegimeState) -> dict:
        """Recommend strategy adjustments for current regime."""
        r = state.primary_regime

        recommendations = {
            "risk_on": {
                "equity_bias": "long", "vol_position": "short_vol",
                "strategy_preference": ["momentum", "trend_following", "carry"],
                "leverage": 1.0, "hedge_ratio": 0.0,
            },
            "risk_off": {
                "equity_bias": "underweight", "vol_position": "long_vol",
                "strategy_preference": ["defensive", "quality", "low_vol"],
                "leverage": 0.7, "hedge_ratio": 0.3,
            },
            "trending_up": {
                "equity_bias": "long", "vol_position": "neutral",
                "strategy_preference": ["momentum", "breakout", "trend_following"],
                "leverage": 1.0, "hedge_ratio": 0.1,
            },
            "trending_down": {
                "equity_bias": "short", "vol_position": "long_vol",
                "strategy_preference": ["short_momentum", "put_spreads", "defensive"],
                "leverage": 0.8, "hedge_ratio": 0.4,
            },
            "mean_reverting": {
                "equity_bias": "neutral", "vol_position": "short_vol",
                "strategy_preference": ["mean_reversion", "pairs", "stat_arb"],
                "leverage": 0.9, "hedge_ratio": 0.1,
            },
            "high_volatility": {
                "equity_bias": "reduced", "vol_position": "long_vol",
                "strategy_preference": ["vol_arb", "straddles", "tail_hedge"],
                "leverage": 0.5, "hedge_ratio": 0.5,
            },
            "crisis": {
                "equity_bias": "minimal", "vol_position": "max_long_vol",
                "strategy_preference": ["cash", "treasuries", "tail_hedge", "gold"],
                "leverage": 0.3, "hedge_ratio": 0.7,
            },
            "recovery": {
                "equity_bias": "aggressive_long", "vol_position": "short_vol",
                "strategy_preference": ["value", "high_beta", "momentum"],
                "leverage": 1.2, "hedge_ratio": 0.05,
            },
        }

        rec = recommendations.get(r, recommendations["risk_on"])
        rec["regime"] = r
        rec["confidence"] = state.confidence
        rec["transition_risk"] = state.transition_probability
        return rec

    def regime_analytics(self) -> dict:
        """Historical regime analytics."""
        if not self._regime_history:
            return {"n_observations": 0}

        regimes = self._regime_history
        n = len(regimes)

        # Frequency
        freq = {}
        for r in REGIME_NAMES:
            freq[r] = float(regimes.count(r) / n)

        # Average duration per regime
        durations: dict[str, list] = {r: [] for r in REGIME_NAMES}
        current = regimes[0]
        count = 1
        for i in range(1, n):
            if regimes[i] == current:
                count += 1
            else:
                durations[current].append(count)
                current = regimes[i]
                count = 1
        durations[current].append(count)

        avg_dur = {r: float(np.mean(d)) if d else 0.0 for r, d in durations.items()}

        return {
            "n_observations": n,
            "frequency": freq,
            "avg_duration_days": avg_dur,
            "current_regime": self._current_regime,
            "days_in_current": self._days_in_regime,
            "transition_matrix": self.transition_model.transition_matrix.tolist(),
            "stationary_distribution": self.transition_model.stationary_distribution().tolist(),
        }
