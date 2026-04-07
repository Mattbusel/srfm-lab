"""
regime_adaptive_strategy.py # Regime-adaptive master strategy for SRFM.

Classifies current market regime and delegates to the appropriate sub-strategy:
  TRENDING_BULL / TRENDING_BEAR  => Wave4Detector + BH mass momentum
  RANGING                        => MeanReversionEnsemble
  HIGH_VOL                       => VolatilityBreakoutStrategy (tight stops)
  CRISIS                         => reduce sizes 50%, exits only

Regime detection uses Hurst, EMA200 comparison, BH mass, and realized vol.
Hysteresis: requires 5 consecutive bars of agreement before switching regime.
"""

from __future__ import annotations

import math
import sqlite3
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from .wave4_strategy import Wave4Detector, Wave4StrategyAdapter, hurst_rs
from .mean_reversion_strategy import MeanReversionEnsemble
from .volatility_strategy import VolatilityBreakoutStrategy, GARCHVolForecast


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BH_MASS_THRESH        = 1.92
HURST_TRENDING_MIN    = 0.58
HURST_MR_MAX          = 0.42
REGIME_SWITCH_BARS    = 5     # hysteresis: must see consistent signal for N bars
EMA200_PERIOD         = 200
HIGH_VOL_MULT         = 2.0   # realized_vol > 2x median => HIGH_VOL
CRISIS_VOL_MULT       = 4.0   # realized_vol > 4x median => CRISIS
CRISIS_DD_THRESH      = 0.10  # max drawdown > 10% => CRISIS
CRISIS_SIZE_REDUCTION = 0.50  # trade at 50% normal size in CRISIS


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    """Market regime classifications."""
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING       = "RANGING"
    HIGH_VOL      = "HIGH_VOL"
    CRISIS        = "CRISIS"
    UNKNOWN       = "UNKNOWN"


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Classifies market regime from bar data, Hurst exponent, and vol ratio.

    Hysteresis mechanism: the candidate regime must persist for REGIME_SWITCH_BARS
    consecutive bars before the confirmed regime changes.  This prevents rapid
    flapping between regime states on noisy data.
    """

    def __init__(
        self,
        switch_bars:     int   = REGIME_SWITCH_BARS,
        ema200_period:   int   = EMA200_PERIOD,
        hurst_trend_min: float = HURST_TRENDING_MIN,
        hurst_mr_max:    float = HURST_MR_MAX,
        bh_mass_min:     float = BH_MASS_THRESH,
        high_vol_mult:   float = HIGH_VOL_MULT,
        crisis_vol_mult: float = CRISIS_VOL_MULT,
        crisis_dd:       float = CRISIS_DD_THRESH,
        vol_history_bars: int  = 60,
    ):
        self._switch_bars      = switch_bars
        self._hurst_trend_min  = hurst_trend_min
        self._hurst_mr_max     = hurst_mr_max
        self._bh_mass_min      = bh_mass_min
        self._high_vol_mult    = high_vol_mult
        self._crisis_vol_mult  = crisis_vol_mult
        self._crisis_dd        = crisis_dd

        # EMA200 computation
        self._ema200_period = ema200_period
        self._ema200:   Optional[float] = None
        self._ema200_k: float = 2.0 / (ema200_period + 1)

        # Vol history for ratio computation
        self._vols: Deque[float] = deque(maxlen=vol_history_bars)

        # Drawdown tracking
        self._peak_price: float = 0.0

        # Hysteresis state
        self._confirmed_regime: Regime = Regime.UNKNOWN
        self._candidate_regime: Regime = Regime.UNKNOWN
        self._candidate_count:  int    = 0

        # Price history for Hurst estimation
        self._prices: Deque[float] = deque(maxlen=120)

    # ------------------------------------------------------------------

    def classify(
        self,
        bars:      List[dict],
        hurst:     float,
        vol_ratio: float,
    ) -> Regime:
        """
        Classify current regime.

        Parameters
        #--------
        bars:
            Recent bars (newest last), each with keys: close, bh_mass.
        hurst:
            Current Hurst exponent (0-1).
        vol_ratio:
            Current realized vol / historical median vol.

        Returns
        #-----
        Confirmed Regime after applying hysteresis.
        """
        if not bars:
            return self._confirmed_regime

        bar   = bars[-1]
        close = float(bar.get("close", 0.0))
        mass  = float(bar.get("bh_mass", 0.0))

        # Update EMA200
        self._update_ema200(close)
        self._prices.append(close)

        # Update peak price for drawdown calc
        self._peak_price = max(self._peak_price, close)
        drawdown = (self._peak_price - close) / (self._peak_price + 1e-9)

        ema200 = self._ema200 if self._ema200 is not None else close

        # Classify candidate regime
        candidate = self._raw_classify(
            close, hurst, vol_ratio, ema200, mass, drawdown
        )

        # Apply hysteresis
        return self._apply_hysteresis(candidate)

    def _raw_classify(
        self,
        close:    float,
        hurst:    float,
        vol_ratio: float,
        ema200:   float,
        mass:     float,
        drawdown: float,
    ) -> Regime:
        """
        Raw regime classification without hysteresis.
        Priority order: CRISIS > HIGH_VOL > TRENDING > RANGING.
        """
        # CRISIS: extreme vol or severe drawdown
        if vol_ratio > self._crisis_vol_mult or drawdown > self._crisis_dd:
            return Regime.CRISIS

        # HIGH_VOL: elevated realized vol
        if vol_ratio > self._high_vol_mult:
            return Regime.HIGH_VOL

        # TRENDING: Hurst trending AND meaningful BH mass
        if hurst > self._hurst_trend_min and mass > self._bh_mass_min * 0.75:
            if close > ema200:
                return Regime.TRENDING_BULL
            else:
                return Regime.TRENDING_BEAR

        # RANGING: Hurst in mean-reverting zone
        if hurst < self._hurst_mr_max:
            return Regime.RANGING

        # Default: Hurst is in the middle zone
        return Regime.RANGING

    def _apply_hysteresis(self, candidate: Regime) -> Regime:
        """
        Require candidate regime to persist for _switch_bars consecutive
        bars before updating the confirmed regime.
        """
        if candidate == self._candidate_regime:
            self._candidate_count += 1
        else:
            self._candidate_regime = candidate
            self._candidate_count  = 1

        if self._candidate_count >= self._switch_bars:
            self._confirmed_regime = self._candidate_regime

        if self._confirmed_regime == Regime.UNKNOWN:
            # Bootstrap: use candidate directly for the first assignment
            self._confirmed_regime = candidate

        return self._confirmed_regime

    def _update_ema200(self, price: float) -> None:
        if self._ema200 is None:
            self._ema200 = price
        else:
            self._ema200 = (price - self._ema200) * self._ema200_k + self._ema200

    def get_confirmed_regime(self) -> Regime:
        return self._confirmed_regime

    def get_candidate_regime(self) -> Regime:
        return self._candidate_regime

    def get_candidate_count(self) -> int:
        return self._candidate_count

    def reset(self) -> None:
        """Reset all state."""
        self._ema200            = None
        self._peak_price        = 0.0
        self._confirmed_regime  = Regime.UNKNOWN
        self._candidate_regime  = Regime.UNKNOWN
        self._candidate_count   = 0
        self._prices.clear()
        self._vols.clear()


# ---------------------------------------------------------------------------
# RegimeTransitionLogger
# ---------------------------------------------------------------------------

class RegimeTransitionLogger:
    """
    Logs regime transitions and performance stats to a SQLite database.

    Schema:
      transitions: (id, timestamp, from_regime, to_regime, bar_index, price)
      regime_spans: (id, regime, start_bar, end_bar, duration_bars,
                     start_price, end_price, pnl_pct)
      performance: (id, regime, n_transitions, avg_duration_bars,
                    total_pnl_pct, win_rate)
    """

    def __init__(self, db_path: str = "regime_log.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._current_regime: Regime = Regime.UNKNOWN
        self._regime_start_bar: int   = 0
        self._regime_start_price: float = 0.0
        self._bar_index: int = 0
        self._init_db()

    def _init_db(self) -> None:
        try:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            c = self._conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS transitions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    REAL,
                    from_regime  TEXT,
                    to_regime    TEXT,
                    bar_index    INTEGER,
                    price        REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS regime_spans (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime        TEXT,
                    start_bar     INTEGER,
                    end_bar       INTEGER,
                    duration_bars INTEGER,
                    start_price   REAL,
                    end_price     REAL,
                    pnl_pct       REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime         TEXT UNIQUE,
                    n_transitions  INTEGER,
                    avg_duration   REAL,
                    total_pnl_pct  REAL,
                    win_rate       REAL
                )
            """)
            self._conn.commit()
        except Exception as e:
            self._conn = None

    # ------------------------------------------------------------------

    def log_bar(self, regime: Regime, price: float) -> None:
        """
        Called every bar.  Detects transitions and records spans.
        """
        self._bar_index += 1
        if regime == self._current_regime:
            return

        # Regime changed
        if self._current_regime != Regime.UNKNOWN and self._conn is not None:
            self._close_span(price)
            self._log_transition(self._current_regime, regime, price)

        self._current_regime      = regime
        self._regime_start_bar    = self._bar_index
        self._regime_start_price  = price

    def _log_transition(
        self,
        from_r: Regime,
        to_r:   Regime,
        price:  float,
    ) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """INSERT INTO transitions
                   (timestamp, from_regime, to_regime, bar_index, price)
                   VALUES (?, ?, ?, ?, ?)""",
                (time.time(), from_r.value, to_r.value, self._bar_index, price),
            )
            self._conn.commit()
        except Exception:
            pass

    def _close_span(self, end_price: float) -> None:
        if self._conn is None:
            return
        duration = self._bar_index - self._regime_start_bar
        pnl_pct  = (
            (end_price - self._regime_start_price) / (self._regime_start_price + 1e-9)
            if self._regime_start_price > 0 else 0.0
        )
        try:
            self._conn.execute(
                """INSERT INTO regime_spans
                   (regime, start_bar, end_bar, duration_bars, start_price, end_price, pnl_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    self._current_regime.value,
                    self._regime_start_bar,
                    self._bar_index,
                    duration,
                    self._regime_start_price,
                    end_price,
                    pnl_pct,
                ),
            )
            self._conn.commit()
        except Exception:
            pass

    def get_regime_stats(self, regime: Regime) -> Dict:
        """
        Return performance statistics for a specific regime.

        Returns dict with keys:
          n_spans, avg_duration_bars, total_pnl_pct, win_rate
        """
        if self._conn is None:
            return {}
        try:
            c = self._conn.cursor()
            c.execute(
                """SELECT duration_bars, pnl_pct FROM regime_spans
                   WHERE regime = ?""",
                (regime.value,),
            )
            rows = c.fetchall()
            if not rows:
                return {"n_spans": 0, "avg_duration_bars": 0.0,
                        "total_pnl_pct": 0.0, "win_rate": 0.0}
            durations = [r[0] for r in rows]
            pnls      = [r[1] for r in rows]
            wins      = sum(1 for p in pnls if p > 0)
            return {
                "n_spans":          len(rows),
                "avg_duration_bars": float(np.mean(durations)),
                "total_pnl_pct":    float(np.sum(pnls)),
                "win_rate":         wins / len(rows) if rows else 0.0,
            }
        except Exception:
            return {}

    def get_all_transitions(self) -> List[Dict]:
        """Return all logged transitions as list of dicts."""
        if self._conn is None:
            return []
        try:
            c = self._conn.cursor()
            c.execute("SELECT * FROM transitions ORDER BY bar_index")
            rows = c.fetchall()
            cols = ["id", "timestamp", "from_regime", "to_regime", "bar_index", "price"]
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# RegimeAdaptiveStrategy
# ---------------------------------------------------------------------------

class RegimeAdaptiveStrategy:
    """
    Master strategy that selects and runs sub-strategies based on regime.

    Sub-strategy routing:
      TRENDING_BULL / TRENDING_BEAR => Wave4StrategyAdapter + BH mass momentum
      RANGING                       => MeanReversionEnsemble
      HIGH_VOL                      => VolatilityBreakoutStrategy (tight stops)
      CRISIS                        => 50% size reduction, exits only

    compute_signal() returns float in [-1, 1]:
      Positive = long, Negative = short, 0 = no trade.

    In CRISIS mode the signal is forced to 0 (exits only) and the
    size_multiplier property returns 0.5.
    """

    def __init__(
        self,
        hurst_window:      int   = 100,
        vol_history_bars:  int   = 60,
        db_path:           str   = "regime_log.db",
        log_transitions:   bool  = True,
    ):
        self._hurst_window = hurst_window

        # Sub-strategies
        self._wave4     = Wave4StrategyAdapter()
        self._mr        = MeanReversionEnsemble()
        self._vol_break = VolatilityBreakoutStrategy()

        # Regime detection
        self._regime_det = RegimeDetector()

        # Transition logger
        self._logger: Optional[RegimeTransitionLogger] = None
        if log_transitions:
            try:
                self._logger = RegimeTransitionLogger(db_path=db_path)
            except Exception:
                self._logger = None

        # State
        self._current_regime: Regime = Regime.UNKNOWN
        self._bar_index:      int    = 0
        self._prices:         Deque[float] = deque(maxlen=max(hurst_window, vol_history_bars + 10))
        self._returns:        Deque[float] = deque(maxlen=vol_history_bars + 10)
        self._vol_history:    Deque[float] = deque(maxlen=vol_history_bars)
        self._size_mult:      float  = 1.0

    # ------------------------------------------------------------------
    # Main signal computation
    # ------------------------------------------------------------------

    def compute_signal(self, bars: List[dict]) -> float:
        """
        Process latest bar and return normalized signal [-1, 1].

        Parameters
        #--------
        bars:
            List of bar dicts (newest last).  Each bar must have:
            close, high, low, bh_mass.

        Returns
        #-----
        float: signal strength and direction.
        """
        if not bars:
            return 0.0

        bar   = bars[-1]
        close = float(bar.get("close", 0.0))
        self._bar_index += 1

        # Update price / return history
        self._prices.append(close)
        if len(self._prices) >= 2:
            r = math.log(self._prices[-1] / max(self._prices[-2], 1e-9))
            self._returns.append(r)

        # Compute Hurst exponent
        prices_arr = np.array(list(self._prices), dtype=float)
        hurst = hurst_rs(prices_arr) if len(prices_arr) >= 20 else 0.5

        # Compute realized vol ratio
        vol_ratio = self._compute_vol_ratio()

        # Classify regime (with hysteresis)
        regime = self._regime_det.classify(bars, hurst, vol_ratio)
        self._current_regime = regime
        self._size_mult = CRISIS_SIZE_REDUCTION if regime == Regime.CRISIS else 1.0

        # Log transition
        if self._logger is not None:
            try:
                self._logger.log_bar(regime, close)
            except Exception:
                pass

        # Dispatch to sub-strategy
        return self._dispatch(bars, hurst, regime)

    def _dispatch(
        self,
        bars:   List[dict],
        hurst:  float,
        regime: Regime,
    ) -> float:
        """Route to the appropriate sub-strategy and return its signal."""

        if regime == Regime.CRISIS:
            # Crisis: no new entries, signal = 0
            return 0.0

        elif regime in (Regime.TRENDING_BULL, Regime.TRENDING_BEAR):
            # Trending: use Wave4 + BH mass momentum
            self._wave4.update_hurst(hurst)
            wave4_sig = self._wave4.compute_signal(bars)

            # BH mass momentum: direct from last bar
            mass     = float(bars[-1].get("bh_mass", 0.0))
            mass_sig = self._bh_mass_momentum_signal(mass, regime)

            # Blend 60% Wave4, 40% mass momentum
            combined = 0.60 * wave4_sig + 0.40 * mass_sig
            return float(np.clip(combined, -1.0, 1.0))

        elif regime == Regime.RANGING:
            # Ranging: use mean reversion ensemble
            self._mr.update_hurst(hurst)
            close = float(bars[-1].get("close", 0.0))
            mr_sig = self._mr.compute_signal(close)
            return float(np.clip(mr_sig, -1.0, 1.0))

        elif regime == Regime.HIGH_VOL:
            # High vol: breakout strategy with tighter effective stops
            breakout_sig, _ = self._vol_break.update(bars[-1])
            # Attenuate signal by 50% for risk management in high vol
            return float(np.clip(breakout_sig * 0.5, -1.0, 1.0))

        return 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> Regime:
        return self._current_regime

    @property
    def size_multiplier(self) -> float:
        """Position size multiplier (0.5 in CRISIS, 1.0 otherwise)."""
        return self._size_mult

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_vol_ratio(self) -> float:
        """
        Compute ratio of recent realized vol to historical median vol.
        Returns 1.0 if insufficient data.
        """
        if len(self._returns) < 21:
            return 1.0

        rets = np.array(list(self._returns), dtype=float)
        recent_vol  = float(np.std(rets[-21:], ddof=1)) * math.sqrt(252.0)
        self._vol_history.append(recent_vol)

        if len(self._vol_history) < 10:
            return 1.0

        median_vol = float(np.median(list(self._vol_history)))
        if median_vol < 1e-10:
            return 1.0

        return recent_vol / median_vol

    def _bh_mass_momentum_signal(self, mass: float, regime: Regime) -> float:
        """
        Convert raw BH mass to a directional momentum signal.
        BH mass > BH_MASS_THRESH and regime aligns => +1 or -1.
        """
        norm_mass = min(1.0, max(0.0, (mass - 0.5) / (BH_MASS_THRESH - 0.5)))
        if regime == Regime.TRENDING_BULL:
            return norm_mass
        elif regime == Regime.TRENDING_BEAR:
            return -norm_mass
        return 0.0

    def get_regime_stats(self, regime: Regime) -> Dict:
        """Return performance stats for a regime from the transition log."""
        if self._logger is None:
            return {}
        return self._logger.get_regime_stats(regime)

    def reset(self) -> None:
        """Reset all sub-strategy and internal state."""
        self._wave4.reset()
        self._mr.reset()
        self._vol_break   = VolatilityBreakoutStrategy()
        self._regime_det.reset()
        self._current_regime = Regime.UNKNOWN
        self._bar_index      = 0
        self._size_mult      = 1.0
        self._prices.clear()
        self._returns.clear()
        self._vol_history.clear()

    def close(self) -> None:
        """Clean up resources (closes SQLite connection)."""
        if self._logger is not None:
            self._logger.close()
