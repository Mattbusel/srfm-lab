"""
macro_regime.py -- Macro regime classification from multi-source signals.

Classifies the current macroeconomic environment into discrete regimes that
drive crypto allocation multipliers. Inputs include yield curve spreads, VIX,
DXY, credit spreads, and crypto-specific indicators (BTC dominance).

Regime taxonomy:
  RISK_ON        -- Broad risk appetite, equities and crypto both bid
  RISK_OFF       -- Stress, VIX elevated, credit wide, risk assets sold
  STAGFLATIONARY -- Inflation high, growth weak, hardest macro backdrop
  DEFLATIONARY   -- Falling prices, weak demand, growth collapsing
  GOLDILOCKS     -- Low volatility, healthy growth, tight spreads
  NEUTRAL        -- No dominant regime signal
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MacroRegime(Enum):
    """Discrete macro regime labels used for allocation decisions."""
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    STAGFLATIONARY = "STAGFLATIONARY"
    DEFLATIONARY = "DEFLATIONARY"
    GOLDILOCKS = "GOLDILOCKS"
    NEUTRAL = "NEUTRAL"


class Direction(Enum):
    """Directional trend of a macro indicator."""
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


# ---------------------------------------------------------------------------
# MacroIndicator dataclass
# ---------------------------------------------------------------------------

@dataclass
class MacroIndicator:
    """
    Snapshot of a single macro indicator at a point in time.

    Attributes
    ----------
    name    : Human-readable indicator label (e.g., "VIX", "2y10y_spread").
    value   : Current raw numeric value.
    z_score : Standardized score vs the trailing 252-day (1-year) history.
              Positive z-score means above average; negative below average.
    direction : Trend direction relative to recent observations.
    source  : Data provider tag (e.g., "FRED", "Bloomberg", "CoinGlass").
    """
    name: str
    value: float
    z_score: float
    direction: Direction
    source: str

    def is_extreme(self, threshold: float = 2.0) -> bool:
        """Return True if |z_score| >= threshold -- identifies outlier readings."""
        return abs(self.z_score) >= threshold

    def __repr__(self) -> str:
        return (
            f"MacroIndicator(name={self.name!r}, value={self.value:.4f}, "
            f"z_score={self.z_score:.2f}, direction={self.direction.value}, "
            f"source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Helper: compute z-score from a history list
# ---------------------------------------------------------------------------

def _compute_z_score(value: float, history: List[float]) -> float:
    """
    Compute the z-score of `value` relative to `history`.

    Returns 0.0 when history is too short to compute a meaningful standard
    deviation (avoids division by zero).
    """
    if len(history) < 2:
        return 0.0
    mean = statistics.mean(history)
    stdev = statistics.stdev(history)
    if stdev < 1e-10:
        return 0.0
    return (value - mean) / stdev


def _compute_direction(value: float, history: List[float], lookback: int = 5) -> Direction:
    """
    Determine whether `value` is trending UP, DOWN, or FLAT vs its recent
    history using a simple linear slope over the last `lookback` samples.
    """
    if len(history) < lookback:
        return Direction.FLAT
    recent = history[-lookback:]
    n = len(recent)
    x_mean = (n - 1) / 2.0
    y_mean = statistics.mean(recent)
    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator < 1e-12:
        return Direction.FLAT
    slope = numerator / denominator
    # Normalize slope by mean to get a relative change per period
    rel_slope = slope / max(abs(y_mean), 1e-6)
    if rel_slope > 0.005:
        return Direction.UP
    if rel_slope < -0.005:
        return Direction.DOWN
    return Direction.FLAT


# ---------------------------------------------------------------------------
# MacroRegimeClassifier
# ---------------------------------------------------------------------------

class MacroRegimeClassifier:
    """
    Classifies the macro regime from a basket of risk indicators.

    Parameters
    ----------
    majority_window : int
        Rolling window length (in periods) for the majority-vote filter that
        prevents rapid regime whipsawing. Default 20 (trading days).
    history_window  : int
        Number of periods of each indicator to retain for z-score computation.
        Default 252 (approximately 1 trading year).

    Usage
    -----
    classifier = MacroRegimeClassifier()
    regime = classifier.classify({
        "vix": 28.5,
        "yield_curve_2y10y": -0.25,
        "credit_spread_hy": 350,
        "inflation_yoy": 3.2,
        "gdp_growth_yoy": 2.1,
        "equity_momentum_20d": 0.04,
        "crypto_momentum_20d": 0.08,
        "btc_dominance": 52.0,
        "dxy": 104.5,
    })
    multiplier = classifier.get_crypto_allocation_multiplier(regime)
    """

    # Regime classification thresholds
    VIX_STRESS_LEVEL        = 30.0
    VIX_CALM_LEVEL          = 20.0
    VIX_GOLDILOCKS_MAX      = 15.0
    CREDIT_SPREAD_STRESS_BP = 200.0   # basis points (HY OAS)
    YIELD_CURVE_GOLDILOCKS  = 0.005   # 0.5% in decimal
    INFLATION_STAGFLATION   = 0.04    # 4% YoY
    GDP_STAGFLATION_MAX     = 0.01    # 1% real GDP growth
    GDP_GOLDILOCKS_MIN      = 0.015   # 1.5% real GDP growth

    # Crypto allocation multipliers by regime
    ALLOCATION_MULTIPLIERS: Dict[MacroRegime, float] = {
        MacroRegime.RISK_ON:        1.2,
        MacroRegime.GOLDILOCKS:     1.1,
        MacroRegime.NEUTRAL:        1.0,
        MacroRegime.STAGFLATIONARY: 0.7,
        MacroRegime.DEFLATIONARY:   0.6,
        MacroRegime.RISK_OFF:       0.5,
    }

    def __init__(
        self,
        majority_window: int = 20,
        history_window: int = 252,
    ) -> None:
        self.majority_window = majority_window
        self.history_window  = history_window
        # Rolling history per indicator key
        self._histories: Dict[str, Deque[float]] = {}
        # Recent regime labels for majority-vote smoothing
        self._regime_buffer: Deque[MacroRegime] = deque(maxlen=majority_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, indicators: Dict[str, float]) -> MacroRegime:
        """
        Classify the current macro regime from a snapshot of indicator values.

        Parameters
        ----------
        indicators : dict with the following optional keys
            vix                 -- CBOE VIX index level
            yield_curve_2y10y   -- 10y minus 2y US Treasury spread (decimal, e.g. 0.005 = 50bp)
            credit_spread_hy    -- High-yield OAS in basis points
            credit_spread_ig    -- Investment grade OAS in basis points
            inflation_yoy       -- CPI YoY as decimal (e.g. 0.035 = 3.5%)
            gdp_growth_yoy      -- Real GDP YoY as decimal
            equity_momentum_20d -- 20-day return of broad equity index (decimal)
            crypto_momentum_20d -- 20-day return of BTC (decimal)
            btc_dominance       -- BTC dominance percentage (0-100)
            dxy                 -- DXY USD index level

        Returns
        -------
        MacroRegime (majority-vote smoothed over last `majority_window` periods)
        """
        self._update_histories(indicators)
        raw_regime = self._classify_raw(indicators)
        self._regime_buffer.append(raw_regime)
        return self._majority_vote()

    def get_crypto_allocation_multiplier(self, regime: MacroRegime) -> float:
        """
        Return the position-size multiplier for crypto given the current regime.

        Values:
          RISK_ON        -> 1.2x (increase exposure)
          GOLDILOCKS     -> 1.1x (slight overweight)
          NEUTRAL        -> 1.0x (neutral sizing)
          STAGFLATIONARY -> 0.7x (reduce exposure)
          DEFLATIONARY   -> 0.6x (significant reduction)
          RISK_OFF       -> 0.5x (half normal exposure)
        """
        return self.ALLOCATION_MULTIPLIERS[regime]

    def get_indicators_snapshot(
        self, indicators: Dict[str, float]
    ) -> Dict[str, MacroIndicator]:
        """
        Build a dictionary of MacroIndicator objects enriched with z-scores
        and directions for the provided indicator snapshot. Useful for
        downstream logging and dashboard display.
        """
        snapshot: Dict[str, MacroIndicator] = {}
        source_map = {
            "vix":                 "CBOE",
            "yield_curve_2y10y":   "FRED",
            "credit_spread_hy":    "ICE/BofA",
            "credit_spread_ig":    "ICE/BofA",
            "inflation_yoy":       "BLS",
            "gdp_growth_yoy":      "BEA",
            "equity_momentum_20d": "Exchange",
            "crypto_momentum_20d": "CoinGlass",
            "btc_dominance":       "CoinMarketCap",
            "dxy":                 "ICE",
        }
        for key, value in indicators.items():
            history = list(self._histories.get(key, []))
            z = _compute_z_score(value, history)
            d = _compute_direction(value, history)
            snapshot[key] = MacroIndicator(
                name=key,
                value=value,
                z_score=z,
                direction=d,
                source=source_map.get(key, "unknown"),
            )
        return snapshot

    def regime_history(self) -> List[MacroRegime]:
        """Return the current regime buffer contents (oldest to newest)."""
        return list(self._regime_buffer)

    def reset(self) -> None:
        """Clear all stored history and regime buffers."""
        self._histories.clear()
        self._regime_buffer.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_histories(self, indicators: Dict[str, float]) -> None:
        for key, value in indicators.items():
            if key not in self._histories:
                self._histories[key] = deque(maxlen=self.history_window)
            self._histories[key].append(value)

    def _classify_raw(self, indicators: Dict[str, float]) -> MacroRegime:
        """
        Apply hard-threshold classification rules to raw indicator values.
        Rules are evaluated in priority order -- most severe regimes first.
        """
        vix       = indicators.get("vix", 20.0)
        hy_spread = indicators.get("credit_spread_hy", 300.0)
        inflation = indicators.get("inflation_yoy", 0.02)
        gdp       = indicators.get("gdp_growth_yoy", 0.02)
        curve     = indicators.get("yield_curve_2y10y", 0.005)
        eq_mom    = indicators.get("equity_momentum_20d", 0.0)
        cr_mom    = indicators.get("crypto_momentum_20d", 0.0)

        # ------------------------------------------------------------------
        # Rule 1 -- RISK_OFF: high VIX + wide credit spreads
        # ------------------------------------------------------------------
        if vix > self.VIX_STRESS_LEVEL and hy_spread > self.CREDIT_SPREAD_STRESS_BP:
            return MacroRegime.RISK_OFF

        # ------------------------------------------------------------------
        # Rule 2 -- STAGFLATIONARY: high inflation + weak growth
        # ------------------------------------------------------------------
        if inflation > self.INFLATION_STAGFLATION and gdp < self.GDP_STAGFLATION_MAX:
            return MacroRegime.STAGFLATIONARY

        # ------------------------------------------------------------------
        # Rule 3 -- DEFLATIONARY: falling prices + contracting growth
        # (uses z-scores of inflation history if available)
        # ------------------------------------------------------------------
        infl_hist = list(self._histories.get("inflation_yoy", []))
        gdp_hist  = list(self._histories.get("gdp_growth_yoy", []))
        infl_z    = _compute_z_score(inflation, infl_hist)
        gdp_z     = _compute_z_score(gdp, gdp_hist)
        if inflation < 0.0 and gdp < 0.0:
            return MacroRegime.DEFLATIONARY
        # Also catch deflationary momentum: both strongly below their histories
        if infl_z < -2.0 and gdp_z < -1.5 and inflation < 0.015:
            return MacroRegime.DEFLATIONARY

        # ------------------------------------------------------------------
        # Rule 4 -- GOLDILOCKS: calm VIX + positive curve + decent growth
        # ------------------------------------------------------------------
        if (
            vix < self.VIX_GOLDILOCKS_MAX
            and curve > self.YIELD_CURVE_GOLDILOCKS
            and gdp > self.GDP_GOLDILOCKS_MIN
        ):
            return MacroRegime.GOLDILOCKS

        # ------------------------------------------------------------------
        # Rule 5 -- RISK_ON: moderate VIX + positive risk momentum
        # ------------------------------------------------------------------
        crypto_rally = cr_mom > 0.05   # >5% 20-day BTC gain
        equity_rally = eq_mom > 0.02   # >2% 20-day equity gain
        if vix < self.VIX_CALM_LEVEL and equity_rally and crypto_rally:
            return MacroRegime.RISK_ON

        # ------------------------------------------------------------------
        # Default
        # ------------------------------------------------------------------
        return MacroRegime.NEUTRAL

    def _majority_vote(self) -> MacroRegime:
        """
        Return the most common regime in the rolling buffer.
        Ties are broken by the most recently seen regime.
        """
        if not self._regime_buffer:
            return MacroRegime.NEUTRAL
        counts: Dict[MacroRegime, int] = {}
        for r in self._regime_buffer:
            counts[r] = counts.get(r, 0) + 1
        # Sort by count descending, then by insertion order for tie-break
        max_count = max(counts.values())
        # Among all regimes with the max count, return the most recent one
        for regime in reversed(list(self._regime_buffer)):
            if counts[regime] == max_count:
                return regime
        return MacroRegime.NEUTRAL

    def _score_regime_confidence(self) -> Dict[MacroRegime, float]:
        """
        Return a probability-like confidence score for each regime based on
        the current rolling buffer. Useful for soft allocation blending.
        """
        if not self._regime_buffer:
            return {r: 1.0 / len(MacroRegime) for r in MacroRegime}
        counts: Dict[MacroRegime, int] = {}
        for r in self._regime_buffer:
            counts[r] = counts.get(r, 0) + 1
        total = sum(counts.values())
        return {r: counts.get(r, 0) / total for r in MacroRegime}


# ---------------------------------------------------------------------------
# YieldCurveMonitor
# ---------------------------------------------------------------------------

class YieldCurveMonitor:
    """
    Monitors the US Treasury yield curve across key maturities and computes
    curve slope, inversion status, and duration of any active inversion.

    Sustained inversion (>30 days) is treated as a recession warning signal
    that informs the macro regime classifier with a hawkish bias.

    Parameters
    ----------
    recession_warning_days : int
        Number of consecutive inverted periods before issuing a warning.
        Default 30 (trading days -- approximately 6 calendar weeks).
    """

    RECESSION_WARNING_DAYS = 30

    def __init__(self, recession_warning_days: int = 30) -> None:
        self.recession_warning_days = recession_warning_days
        # Yield levels for each key maturity (percent, e.g. 4.5 = 4.5%)
        self._y2: Optional[float] = None
        self._y5: Optional[float] = None
        self._y10: Optional[float] = None
        self._y30: Optional[float] = None
        # Rolling inversion tracker: True if inverted on each stored period
        self._inversion_streak: int = 0
        self._inversion_start_period: Optional[int] = None
        self._period_counter: int = 0
        # History for z-score computations
        self._slope_history: Deque[float] = deque(maxlen=252)

    def update(
        self,
        y2: float,
        y5: float,
        y10: float,
        y30: float,
    ) -> None:
        """
        Update the yield curve with new observations.

        Parameters
        ----------
        y2, y5, y10, y30 : Treasury yields in percent (e.g. 4.75 = 4.75%).
        """
        self._y2  = y2
        self._y5  = y5
        self._y10 = y10
        self._y30 = y30
        self._period_counter += 1

        slope = self.get_slope()
        self._slope_history.append(slope)

        # Update inversion streak
        if self.is_inverted():
            self._inversion_streak += 1
            if self._inversion_start_period is None:
                self._inversion_start_period = self._period_counter
        else:
            self._inversion_streak = 0
            self._inversion_start_period = None

    def get_slope(self) -> float:
        """
        Return the 10y minus 2y spread in percent.

        Positive -> upward-sloping (normal / healthy).
        Negative -> inverted (historically precedes recessions by 12-24 months).
        """
        if self._y10 is None or self._y2 is None:
            raise ValueError("Yield curve not yet populated -- call update() first.")
        return self._y10 - self._y2

    def get_5y30y_slope(self) -> float:
        """Return the 30y minus 5y spread as an alternative curve measure."""
        if self._y30 is None or self._y5 is None:
            raise ValueError("Yield curve not yet populated -- call update() first.")
        return self._y30 - self._y5

    def is_inverted(self) -> bool:
        """Return True if the 2y yield exceeds the 10y yield."""
        return self.get_slope() < 0.0

    def inversion_duration_days(self) -> int:
        """
        Return the number of consecutive periods the curve has been inverted.
        Returns 0 if the curve is currently upward-sloping.
        """
        return self._inversion_streak

    def is_recession_warning(self) -> bool:
        """
        Return True if inversion has been sustained for at least
        `recession_warning_days` consecutive periods.
        """
        return self._inversion_streak >= self.recession_warning_days

    def curve_z_score(self) -> float:
        """
        Return the z-score of the current slope vs its 252-period history.
        A very negative z-score indicates historically deep inversion.
        """
        hist = list(self._slope_history)
        if len(hist) < 2:
            return 0.0
        current = self.get_slope()
        return _compute_z_score(current, hist)

    def get_curve_regime(self) -> str:
        """
        Classify the curve shape into a descriptive regime string.

        Returns one of:
          STEEP_NORMAL    -- slope > 1.5%
          NORMAL          -- 0.5% < slope <= 1.5%
          FLAT            -- -0.25% < slope <= 0.5%
          INVERTED        -- slope <= -0.25%
          DEEPLY_INVERTED -- slope <= -1.0%
        """
        slope = self.get_slope()
        if slope > 1.5:
            return "STEEP_NORMAL"
        if slope > 0.5:
            return "NORMAL"
        if slope > -0.25:
            return "FLAT"
        if slope > -1.0:
            return "INVERTED"
        return "DEEPLY_INVERTED"

    def as_dict(self) -> Dict[str, float]:
        """Return current yield levels and derived metrics as a plain dict."""
        result: Dict[str, float] = {}
        if self._y2 is not None:
            result["y2"]        = self._y2
            result["y5"]        = self._y5 or float("nan")
            result["y10"]       = self._y10 or float("nan")
            result["y30"]       = self._y30 or float("nan")
            result["slope_2y10y"] = self.get_slope()
            result["inversion_days"] = float(self.inversion_duration_days())
        return result


# ---------------------------------------------------------------------------
# CreditSpreadMonitor
# ---------------------------------------------------------------------------

class CreditSpreadMonitor:
    """
    Monitors investment-grade (IG) and high-yield (HY) option-adjusted spreads
    (OAS) and generates a composite risk signal scaled to [-1, +1].

    Spread data is typically sourced from ICE BofA indices available via FRED.

    Parameters
    ----------
    history_window : int
        Number of periods to retain for z-score normalization.
    """

    # Stress thresholds in basis points
    HY_STRESS_BP = 600
    HY_TIGHT_BP  = 250
    IG_STRESS_BP = 200
    IG_TIGHT_BP  = 80

    def __init__(self, history_window: int = 252) -> None:
        self.history_window = history_window
        self._ig_history: Deque[float] = deque(maxlen=history_window)
        self._hy_history: Deque[float] = deque(maxlen=history_window)
        self._ig_current: Optional[float] = None
        self._hy_current: Optional[float] = None

    def update(self, ig_oas_bp: float, hy_oas_bp: float) -> None:
        """
        Record new spread observations.

        Parameters
        ----------
        ig_oas_bp : Investment grade OAS in basis points (e.g. 120 = 1.20%).
        hy_oas_bp : High yield OAS in basis points (e.g. 350 = 3.50%).
        """
        self._ig_current = ig_oas_bp
        self._hy_current = hy_oas_bp
        self._ig_history.append(ig_oas_bp)
        self._hy_history.append(hy_oas_bp)

    def get_ig_spread(self) -> float:
        """Return the most recent IG OAS in basis points."""
        if self._ig_current is None:
            raise ValueError("No IG spread data -- call update() first.")
        return self._ig_current

    def get_hy_spread(self) -> float:
        """Return the most recent HY OAS in basis points."""
        if self._hy_current is None:
            raise ValueError("No HY spread data -- call update() first.")
        return self._hy_current

    def ig_z_score(self) -> float:
        """Return z-score of current IG spread vs stored history."""
        return _compute_z_score(self.get_ig_spread(), list(self._ig_history))

    def hy_z_score(self) -> float:
        """Return z-score of current HY spread vs stored history."""
        return _compute_z_score(self.get_hy_spread(), list(self._hy_history))

    def risk_signal(self) -> float:
        """
        Compute a composite credit risk signal scaled to [-1, +1].

        Interpretation:
          +1.0 -> credit conditions extremely easy (tight spreads, low z-score)
          0.0  -> neutral / average credit conditions
         -1.0  -> credit conditions under severe stress (wide spreads, high z-score)

        Methodology:
          1. Compute z-scores for both IG and HY spreads.
          2. Average them (HY weighted 60%, IG 40%).
          3. Flip sign: widening spreads = negative signal.
          4. Clamp to [-1, +1].
        """
        ig_z = self.ig_z_score()
        hy_z = self.hy_z_score()
        weighted_z = 0.4 * ig_z + 0.6 * hy_z
        # Flip: higher z (wider spreads) -> negative credit signal
        raw = -weighted_z
        return max(-1.0, min(1.0, raw / 3.0))  # /3 to map ~3-sigma range to [-1,1]

    def is_credit_stress(self) -> bool:
        """Return True if HY spreads exceed the stress threshold."""
        return self.get_hy_spread() > self.HY_STRESS_BP

    def is_credit_easy(self) -> bool:
        """Return True if HY spreads are historically tight."""
        return self.get_hy_spread() < self.HY_TIGHT_BP

    def spread_trend(self) -> Direction:
        """Return the near-term trend of HY spreads (last 5 periods)."""
        hist = list(self._hy_history)
        return _compute_direction(self._hy_current or 300.0, hist)

    def regime_contribution(self) -> Dict[str, float]:
        """
        Return a dict summarizing credit spread contributions to regime scoring.
        Useful for diagnostics -- shows raw values, z-scores, and combined signal.
        """
        return {
            "ig_oas_bp":     self._ig_current or float("nan"),
            "hy_oas_bp":     self._hy_current or float("nan"),
            "ig_z_score":    self.ig_z_score(),
            "hy_z_score":    self.hy_z_score(),
            "risk_signal":   self.risk_signal(),
            "credit_stress": float(self.is_credit_stress()),
            "credit_easy":   float(self.is_credit_easy()),
        }

    def blended_spread_bp(self, ig_weight: float = 0.3, hy_weight: float = 0.7) -> float:
        """
        Return a weighted blend of IG and HY spreads as a single credit
        tightness gauge. Weights should sum to 1.0.
        """
        ig = self.get_ig_spread()
        hy = self.get_hy_spread()
        return ig_weight * ig + hy_weight * hy

    def history_percentile(self, spread_type: str = "hy") -> float:
        """
        Return the current spread's percentile rank within stored history (0-100).

        Parameters
        ----------
        spread_type : "hy" or "ig"
        """
        if spread_type == "hy":
            current = self._hy_current
            history = list(self._hy_history)
        else:
            current = self._ig_current
            history = list(self._ig_history)

        if current is None or len(history) < 2:
            return 50.0
        below = sum(1 for h in history if h <= current)
        return 100.0 * below / len(history)


# ---------------------------------------------------------------------------
# RegimeTransitionMatrix
# ---------------------------------------------------------------------------

class RegimeTransitionMatrix:
    """
    Tracks regime transition frequencies and estimates transition probabilities.

    This is useful for:
      - Estimating how likely a regime change is given the current state
      - Detecting when the current regime is historically unstable
      - Providing Markov-chain-style regime forecasts

    Parameters
    ----------
    regimes : list of MacroRegime values to track (default: all)
    """

    def __init__(self, regimes: Optional[List[MacroRegime]] = None) -> None:
        self.regimes = regimes or list(MacroRegime)
        n = len(self.regimes)
        self._idx = {r: i for i, r in enumerate(self.regimes)}
        # Transition count matrix: counts[i][j] = transitions from regimes[i] to regimes[j]
        self._counts: List[List[int]] = [[0] * n for _ in range(n)]
        self._last_regime: Optional[MacroRegime] = None

    def record(self, regime: MacroRegime) -> None:
        """Record an observed regime. Updates the transition count matrix."""
        if self._last_regime is not None:
            i = self._idx[self._last_regime]
            j = self._idx[regime]
            self._counts[i][j] += 1
        self._last_regime = regime

    def transition_probability(
        self, from_regime: MacroRegime, to_regime: MacroRegime
    ) -> float:
        """Return P(to_regime | from_regime) from historical frequencies."""
        i = self._idx[from_regime]
        row_total = sum(self._counts[i])
        if row_total == 0:
            return 1.0 / len(self.regimes)
        j = self._idx[to_regime]
        return self._counts[i][j] / row_total

    def most_likely_next(self, current_regime: MacroRegime) -> MacroRegime:
        """Return the historically most likely next regime from the current one."""
        probs = {
            r: self.transition_probability(current_regime, r)
            for r in self.regimes
        }
        return max(probs, key=lambda r: probs[r])

    def stability_score(self, regime: MacroRegime) -> float:
        """
        Return the self-transition probability P(same | same).
        High score -> regime tends to persist; low score -> frequently transitions out.
        """
        return self.transition_probability(regime, regime)

    def to_matrix(self) -> List[List[float]]:
        """Return the full normalized transition probability matrix."""
        n = len(self.regimes)
        matrix = []
        for i in range(n):
            row_total = sum(self._counts[i])
            if row_total == 0:
                matrix.append([1.0 / n] * n)
            else:
                matrix.append([self._counts[i][j] / row_total for j in range(n)])
        return matrix
