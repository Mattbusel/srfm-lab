"""
regime-oracle/classifier.py
────────────────────────────
Multi-signal real-time regime classifier.

Detects one of six market regimes:
  BULL      — sustained uptrend, healthy momentum, low-mid vol
  BEAR      — sustained downtrend, negative momentum
  NEUTRAL   — range-bound, moderate vol, no clear direction
  CRISIS    — sudden vol spike, severe drawdown, panic selling
  RECOVERY  — post-crisis bounce, improving breadth, rising EMAs
  TOPPING   — late-cycle: momentum diverging from price, vol rising

Signal ensemble (six signals, each maps to soft regime probabilities):
  1. Trend signal        : EMA(50)/EMA(200) crossover state
  2. Volatility regime   : GARCH conditional vol vs rolling percentile
  3. Momentum signal     : 20-day return z-score
  4. Breadth signal      : fraction of tracked alts above their 50-EMA
  5. BH mass regime      : mean BH mass across instruments
  6. Drawdown state      : current drawdown from ATH as fraction

The final probabilities are a weighted average of all six signal outputs,
then softmax-normalised.

Usage
-----
    oracle = RegimeOracle(db_path="idea_engine.db")
    state  = oracle.classify(features)      # features from RegimeFeatureBuilder
    print(state.regime)                     # 'BULL'
    print(state.probabilities)              # {'BULL': 0.62, 'BEAR': 0.05, ...}

    history = oracle.classify_history(ohlcv_df)   # pd.Series of regimes
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_builder import RegimeFeatureBuilder

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent
_ENGINE_ROOT = _HERE.parent
_DB_DEFAULT  = _ENGINE_ROOT / "idea_engine.db"


# ── Regime enum ───────────────────────────────────────────────────────────────

class Regime(str, Enum):
    """Six market regimes detected by the Regime Oracle."""
    BULL     = "BULL"
    BEAR     = "BEAR"
    NEUTRAL  = "NEUTRAL"
    CRISIS   = "CRISIS"
    RECOVERY = "RECOVERY"
    TOPPING  = "TOPPING"

    @classmethod
    def all(cls) -> List["Regime"]:
        return [cls.BULL, cls.BEAR, cls.NEUTRAL, cls.CRISIS, cls.RECOVERY, cls.TOPPING]

    @classmethod
    def from_str(cls, s: str) -> "Regime":
        try:
            return cls(s.upper())
        except ValueError:
            return cls.NEUTRAL


# ── RegimeState ───────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """
    Output of a single regime classification.

    Attributes
    ----------
    regime       : most-likely regime label
    probabilities: dict mapping regime name → probability (sums to 1.0)
    bull_prob    : probability of BULL regime
    bear_prob    : probability of BEAR regime
    neutral_prob : probability of NEUTRAL regime
    crisis_prob  : probability of CRISIS regime
    recovery_prob: probability of RECOVERY regime
    topping_prob : probability of TOPPING regime
    confidence   : max probability (how certain the oracle is)
    features     : raw normalised features used for classification
    ts           : timestamp of classification (ISO-8601)
    symbol       : instrument symbol
    """
    regime:        str
    probabilities: Dict[str, float]
    bull_prob:     float = 0.0
    bear_prob:     float = 0.0
    neutral_prob:  float = 0.0
    crisis_prob:   float = 0.0
    recovery_prob: float = 0.0
    topping_prob:  float = 0.0
    confidence:    float = 0.0
    features:      Dict[str, Any]   = field(default_factory=dict)
    ts:            str              = ""
    symbol:        str              = "BTC"

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Fill individual prob fields from probabilities dict
        probs = self.probabilities
        self.bull_prob     = probs.get("BULL",     0.0)
        self.bear_prob     = probs.get("BEAR",     0.0)
        self.neutral_prob  = probs.get("NEUTRAL",  0.0)
        self.crisis_prob   = probs.get("CRISIS",   0.0)
        self.recovery_prob = probs.get("RECOVERY", 0.0)
        self.topping_prob  = probs.get("TOPPING",  0.0)
        self.confidence    = max(probs.values()) if probs else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts":           self.ts,
            "symbol":       self.symbol,
            "regime":       self.regime,
            "bull_prob":    round(self.bull_prob,     4),
            "bear_prob":    round(self.bear_prob,     4),
            "neutral_prob": round(self.neutral_prob,  4),
            "crisis_prob":  round(self.crisis_prob,   4),
            "recovery_prob":round(self.recovery_prob, 4),
            "topping_prob": round(self.topping_prob,  4),
            "confidence":   round(self.confidence,    4),
        }

    def __repr__(self) -> str:
        return (
            f"RegimeState(regime={self.regime!r}, "
            f"conf={self.confidence:.2f}, "
            f"ts={self.ts[:16]!r})"
        )


# ── Signal score functions ────────────────────────────────────────────────────

def _trend_signal(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute regime probability contributions from trend signals.

    Uses ema_ratio_50_200 and ema_ratio_20_50.
    Both ratios are normalised to [0,1] by the feature builder.

    Returns
    -------
    dict mapping regime → contribution score (unnormalised)
    """
    r50_200 = features.get("ema_ratio_50_200", 0.5)
    r20_50  = features.get("ema_ratio_20_50",  0.5)

    # r > 0.5 → price above EMA200 (bullish); < 0.5 → bearish
    # Use the normalised ratio directly (0 = max bearish, 1 = max bullish)
    bull_strength = (r50_200 + r20_50) / 2.0   # 0..1
    bear_strength = 1.0 - bull_strength

    scores = {
        "BULL":     bull_strength ** 1.5,
        "BEAR":     bear_strength ** 1.5,
        "NEUTRAL":  1.0 - abs(bull_strength - 0.5) * 2.0,
        "CRISIS":   bear_strength ** 2.5 * 0.5,
        "RECOVERY": max(0.0, bull_strength - 0.5) * 2.0 * max(0.0, 0.5 - r50_200) * 2.0,
        "TOPPING":  max(0.0, bull_strength - 0.7) * 2.5 * max(0.0, 0.5 - r20_50) * 2.0,
    }
    return scores


def _volatility_signal(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute regime contributions from volatility signals.

    Uses vol_percentile_30d and vol_percentile_90d.
    High vol → CRISIS or BEAR; low vol → BULL; moderate → NEUTRAL.
    """
    vol30 = features.get("vol_percentile_30d", 0.5)
    vol90 = features.get("vol_percentile_90d", 0.5)
    atr_p = features.get("atr_percentile",     0.5)

    vol_level = (vol30 * 0.5 + vol90 * 0.3 + atr_p * 0.2)   # 0..1

    scores = {
        "BULL":     max(0.0, 1.0 - vol_level * 1.5),
        "BEAR":     vol_level * 0.6,
        "NEUTRAL":  1.0 - abs(vol_level - 0.35) * 2.0,
        "CRISIS":   vol_level ** 2.0 * 2.0,
        "RECOVERY": (1.0 - vol_level) * 0.4,
        "TOPPING":  max(0.0, vol_level - 0.5) * 1.2,
    }
    return scores


def _momentum_signal(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute regime contributions from momentum signals.

    Uses momentum_5d, momentum_20d, momentum_60d (normalised via sigmoid).
    """
    m5  = features.get("momentum_5d",  0.5)
    m20 = features.get("momentum_20d", 0.5)
    m60 = features.get("momentum_60d", 0.5)

    # Weighted composite: short < mid < long
    mom = 0.2 * m5 + 0.4 * m20 + 0.4 * m60   # 0..1

    # Divergence: short momentum lagging long (topping signal)
    divergence = max(0.0, m60 - m5)   # long > short → momentum fading

    scores = {
        "BULL":     mom ** 1.2,
        "BEAR":     (1.0 - mom) ** 1.2,
        "NEUTRAL":  1.0 - abs(mom - 0.5) * 2.0,
        "CRISIS":   (1.0 - mom) ** 2.5,
        "RECOVERY": max(0.0, mom - 0.5) * 2.0 * max(0.0, 0.5 - m60) * 2.0,
        "TOPPING":  divergence * 1.5,
    }
    return scores


def _breadth_signal(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute regime contributions from market breadth.

    Uses breadth_50d (fraction of assets above 50-EMA).
    """
    breadth = features.get("breadth_50d", 0.5)

    scores = {
        "BULL":     breadth ** 1.2,
        "BEAR":     (1.0 - breadth) ** 1.2,
        "NEUTRAL":  1.0 - abs(breadth - 0.5) * 1.5,
        "CRISIS":   (1.0 - breadth) ** 2.0,
        "RECOVERY": max(0.0, breadth - 0.3) * max(0.0, 0.5 - breadth) * 4.0,
        "TOPPING":  max(0.0, breadth - 0.7) * max(0.0, 0.5 - breadth + 0.2) * 3.0,
    }
    return scores


def _bh_mass_signal(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute regime contributions from Black-Hole mass regime.

    bh_mass_mean normalised to [0, 1] (0.5 ≈ mass = 2.0).
    High mass → strong directional force.
    """
    mass_mean = features.get("bh_mass_mean", 0.5)
    mass_max  = features.get("bh_mass_max",  0.5)

    # High mass can indicate either strong bull or strong bear
    # Combine with trend to disambiguate
    ema_ratio = features.get("ema_ratio_50_200", 0.5)
    directional = ema_ratio  # >0.5 = bullish context
    mass_force  = (mass_mean + mass_max) / 2.0

    scores = {
        "BULL":     mass_force * directional * 1.5,
        "BEAR":     mass_force * (1.0 - directional) * 1.5,
        "NEUTRAL":  (1.0 - mass_force) * 1.2,
        "CRISIS":   mass_force * (1.0 - directional) ** 2.0 * 1.5,
        "RECOVERY": mass_force * 0.3 * directional,
        "TOPPING":  mass_max * (1.0 - mass_mean) * directional,
    }
    return scores


def _drawdown_signal(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute regime contributions from drawdown state.

    drawdown_from_ath is normalised to [0, 1] (0 = at ATH, 1 = maximum drawdown).
    """
    dd = features.get("drawdown_from_ath", 0.0)   # 0..1, higher = more DD

    scores = {
        "BULL":     (1.0 - dd) ** 1.5,
        "BEAR":     dd ** 1.2,
        "NEUTRAL":  1.0 - abs(dd - 0.1) * 2.0,
        "CRISIS":   dd ** 2.0 * 2.5,
        "RECOVERY": max(0.0, dd - 0.2) * max(0.0, 1.0 - dd - 0.3) * 4.0,
        "TOPPING":  (1.0 - dd) ** 2.5 * max(0.0, dd - 0.05),
    }
    return scores


def _skew_kurtosis_signal(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute regime contributions from return distribution shape.

    High kurtosis → fat tails → crisis risk.
    Negative skewness → crash risk.
    """
    skew = features.get("skewness_20d", 0.5)    # 0..1, low = negative skew (bearish)
    kurt = features.get("kurtosis_20d", 0.5)    # 0..1, high = fat tails

    fat_tail = kurt
    neg_skew  = 1.0 - skew   # higher = more negative skew

    scores = {
        "BULL":     (1.0 - fat_tail) * skew,
        "BEAR":     neg_skew * 0.6,
        "NEUTRAL":  (1.0 - fat_tail) * (1.0 - abs(skew - 0.5) * 2.0),
        "CRISIS":   fat_tail * neg_skew * 1.5,
        "RECOVERY": (1.0 - neg_skew) * (1.0 - fat_tail) * 0.5,
        "TOPPING":  fat_tail * skew * 0.8,
    }
    return scores


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Apply softmax normalisation to a score dict."""
    keys = list(scores.keys())
    vals = np.array([scores[k] for k in keys], dtype=float)
    vals = np.nan_to_num(vals, nan=0.0)
    # Shift for numerical stability
    vals = vals - vals.max()
    exp_vals = np.exp(np.clip(vals, -20, 20))
    total = exp_vals.sum()
    if total < 1e-10:
        probs = np.ones(len(keys)) / len(keys)
    else:
        probs = exp_vals / total
    return {k: float(p) for k, p in zip(keys, probs)}


# ── RegimeOracle ─────────────────────────────────────────────────────────────

class RegimeOracle:
    """
    Multi-signal regime classifier for the Idea Automation Engine.

    Parameters
    ----------
    db_path        : path to idea_engine.db
    feature_builder: RegimeFeatureBuilder instance (created if None)
    signal_weights : dict mapping signal name → weight (defaults to equal weights)
    """

    # Default signal weights (must sum to ≈ 1)
    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "trend":         0.25,
        "volatility":    0.20,
        "momentum":      0.20,
        "breadth":       0.10,
        "bh_mass":       0.10,
        "drawdown":      0.10,
        "skew_kurtosis": 0.05,
    }

    def __init__(
        self,
        db_path:         Path | str                      = _DB_DEFAULT,
        feature_builder: Optional[RegimeFeatureBuilder] = None,
        signal_weights:  Optional[Dict[str, float]]     = None,
    ) -> None:
        self.db_path         = Path(db_path)
        self.feature_builder = feature_builder or RegimeFeatureBuilder()
        self.signal_weights  = signal_weights or dict(self._DEFAULT_WEIGHTS)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None or not self._is_alive():
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def _is_alive(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _ensure_schema(self) -> None:
        schema_path = _HERE / "schema_extension.sql"
        conn = self._connect()
        if schema_path.exists():
            try:
                conn.executescript(schema_path.read_text(encoding="utf-8"))
                conn.commit()
            except sqlite3.Error as exc:
                logger.warning("Schema extension failed: %s", exc)
        else:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS regime_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              TEXT    NOT NULL,
                    symbol          TEXT    NOT NULL DEFAULT 'BTC',
                    regime          TEXT    NOT NULL,
                    bull_prob       REAL,
                    bear_prob       REAL,
                    neutral_prob    REAL,
                    crisis_prob     REAL,
                    recovery_prob   REAL,
                    topping_prob    REAL,
                    features_json   TEXT,
                    created_at      TEXT    NOT NULL
                        DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
                    UNIQUE(ts, symbol)
                );
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Primary classification API
    # ------------------------------------------------------------------

    def classify(
        self,
        features: Dict[str, Any],
        store:    bool = False,
    ) -> RegimeState:
        """
        Classify market regime from a pre-built feature dict.

        Parameters
        ----------
        features : normalised feature dict from RegimeFeatureBuilder
        store    : if True, persist result to regime_history

        Returns
        -------
        RegimeState with regime probabilities for all six regimes
        """
        # Normalise if not already done
        fb = self.feature_builder
        normed = fb.normalize_features(features)

        # Compute signal contributions
        signal_scores: Dict[str, Dict[str, float]] = {
            "trend":         _trend_signal(normed),
            "volatility":    _volatility_signal(normed),
            "momentum":      _momentum_signal(normed),
            "breadth":       _breadth_signal(normed),
            "bh_mass":       _bh_mass_signal(normed),
            "drawdown":      _drawdown_signal(normed),
            "skew_kurtosis": _skew_kurtosis_signal(normed),
        }

        # Weighted aggregation
        regimes = [r.value for r in Regime.all()]
        composite: Dict[str, float] = {r: 0.0 for r in regimes}

        for sig_name, sig_scores in signal_scores.items():
            w = self.signal_weights.get(sig_name, 0.0)
            for r in regimes:
                composite[r] += w * max(0.0, sig_scores.get(r, 0.0))

        # Apply regime plausibility constraints
        composite = self._apply_constraints(composite, normed)

        # Softmax normalisation
        probs = _softmax(composite)

        # Most-likely regime
        best_regime = max(probs, key=lambda k: probs[k])

        symbol = str(features.get("symbol", "BTC"))
        ts     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        state = RegimeState(
            regime        = best_regime,
            probabilities = probs,
            features      = normed,
            ts            = ts,
            symbol        = symbol,
        )

        if store:
            self._store_regime(state)

        logger.debug(
            "Regime classified: %s (conf=%.2f)  probs=%s",
            best_regime, state.confidence,
            {k: f"{v:.2f}" for k, v in probs.items()},
        )
        return state

    def classify_bar(
        self,
        ohlcv_bar:      Dict[str, Any],
        history_df:     Optional[pd.DataFrame] = None,
        symbol:         str = "BTC",
        store:          bool = False,
    ) -> RegimeState:
        """
        Classify regime for a single OHLCV bar.

        Requires historical context to compute rolling features.
        If history_df is None, uses stored regime history to estimate context.

        Parameters
        ----------
        ohlcv_bar  : dict with keys open, high, low, close, volume
        history_df : recent OHLCV history for rolling feature computation
        symbol     : instrument symbol
        store      : if True, persist result to DB

        Returns
        -------
        RegimeState
        """
        if history_df is None or len(history_df) < 50:
            # Use a minimal single-bar feature estimation
            features = self._single_bar_features(ohlcv_bar, symbol)
        else:
            bar_df = history_df.copy()
            # Append the new bar if not already present
            new_row = pd.DataFrame([ohlcv_bar])
            if isinstance(bar_df.index, pd.DatetimeIndex):
                bar_df = pd.concat([bar_df, new_row]).iloc[-5000:]
            else:
                bar_df = pd.concat([bar_df, new_row]).iloc[-5000:]
            features = self.feature_builder.build_features(bar_df, symbol=symbol)

        return self.classify(features, store=store)

    def classify_history(
        self,
        df:     pd.DataFrame,
        symbol: str = "BTC",
        store:  bool = False,
    ) -> pd.Series:
        """
        Classify regime for every bar in a historical OHLCV DataFrame.

        Parameters
        ----------
        df     : OHLCV DataFrame with DatetimeIndex
        symbol : instrument symbol
        store  : if True, persist all regimes to regime_history

        Returns
        -------
        pd.Series of regime label strings, indexed like df
        """
        feature_df = self.feature_builder.build_features_rolling(df, symbol=symbol)
        normed_df  = self.feature_builder.normalize_features_df(feature_df)

        labels: List[str] = []
        states: List[RegimeState] = []

        for idx, row in normed_df.iterrows():
            feat_dict = row.to_dict()
            feat_dict["symbol"] = symbol
            state = self.classify(feat_dict)
            state.ts = str(idx)
            labels.append(state.regime)
            states.append(state)

        if store:
            for s in states:
                try:
                    self._store_regime(s)
                except Exception as exc:
                    logger.debug("Failed to store regime at %s: %s", s.ts, exc)

        series = pd.Series(labels, index=df.index, name="regime")
        logger.info(
            "Classified %d bars  value_counts: %s",
            len(series), series.value_counts().to_dict(),
        )
        return series

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def transition_matrix(
        self,
        regime_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute empirical regime transition probability matrix.

        P[i, j] = P(next regime = j | current regime = i)

        Parameters
        ----------
        regime_series : pd.Series of regime label strings

        Returns
        -------
        pd.DataFrame with regimes as both index and columns (probabilities)
        """
        regimes = [r.value for r in Regime.all()]
        counts = pd.DataFrame(0, index=regimes, columns=regimes)

        series_clean = regime_series.dropna().astype(str)
        for i in range(len(series_clean) - 1):
            curr = series_clean.iloc[i]
            nxt  = series_clean.iloc[i + 1]
            if curr in regimes and nxt in regimes:
                counts.loc[curr, nxt] += 1

        # Normalise rows to probabilities
        row_sums = counts.sum(axis=1)
        probs = counts.div(row_sums.replace(0, 1), axis=0)
        return probs

    def expected_duration(
        self,
        regime:            str,
        transition_matrix: pd.DataFrame,
    ) -> float:
        """
        Compute expected number of bars remaining in the current regime.

        Uses the diagonal element of the transition matrix.
        If P(stay) = p, expected duration = 1 / (1 - p).

        Parameters
        ----------
        regime            : regime name (e.g. 'BULL')
        transition_matrix : empirical transition matrix from self.transition_matrix()

        Returns
        -------
        float — expected bars remaining (inf if regime has probability 1.0 of staying)
        """
        regime = regime.upper()
        try:
            p_stay = float(transition_matrix.loc[regime, regime])
        except (KeyError, TypeError):
            return 1.0

        if p_stay >= 1.0:
            return float("inf")
        if p_stay <= 0.0:
            return 1.0

        return float(1.0 / (1.0 - p_stay))

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _store_regime(self, state: RegimeState) -> None:
        """Insert a RegimeState into regime_history."""
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO regime_history
                    (ts, symbol, regime,
                     bull_prob, bear_prob, neutral_prob,
                     crisis_prob, recovery_prob, topping_prob,
                     features_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    state.ts,
                    state.symbol,
                    state.regime,
                    round(state.bull_prob,     4),
                    round(state.bear_prob,     4),
                    round(state.neutral_prob,  4),
                    round(state.crisis_prob,   4),
                    round(state.recovery_prob, 4),
                    round(state.topping_prob,  4),
                    json.dumps({k: round(float(v), 4) for k, v in state.features.items()
                                if isinstance(v, (int, float))}),
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:
            logger.warning("Failed to store regime: %s", exc)

    def get_latest_regime(self, symbol: str = "BTC") -> Optional[RegimeState]:
        """
        Return the most-recently stored RegimeState for a symbol.

        Parameters
        ----------
        symbol : instrument symbol

        Returns
        -------
        RegimeState or None
        """
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM regime_history
                WHERE symbol = ?
                ORDER BY ts DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None

        if row is None:
            return None

        row = dict(row)
        probs = {
            "BULL":     row.get("bull_prob", 0.0)     or 0.0,
            "BEAR":     row.get("bear_prob", 0.0)     or 0.0,
            "NEUTRAL":  row.get("neutral_prob", 0.0)  or 0.0,
            "CRISIS":   row.get("crisis_prob", 0.0)   or 0.0,
            "RECOVERY": row.get("recovery_prob", 0.0) or 0.0,
            "TOPPING":  row.get("topping_prob", 0.0)  or 0.0,
        }
        feat = {}
        if row.get("features_json"):
            try:
                feat = json.loads(row["features_json"])
            except json.JSONDecodeError:
                pass

        return RegimeState(
            regime        = row["regime"],
            probabilities = probs,
            features      = feat,
            ts            = row.get("ts", ""),
            symbol        = row.get("symbol", symbol),
        )

    def regime_history_df(
        self,
        symbol: str = "BTC",
        limit:  int = 10_000,
    ) -> pd.DataFrame:
        """
        Return regime history as a DataFrame.

        Parameters
        ----------
        symbol : instrument symbol
        limit  : maximum number of rows

        Returns
        -------
        pd.DataFrame ordered by ts descending
        """
        conn = self._connect()
        try:
            return pd.read_sql(
                "SELECT * FROM regime_history WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                conn,
                params=(symbol, limit),
            )
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Regime plausibility constraints
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_constraints(
        composite: Dict[str, float],
        normed:    Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply domain-knowledge constraints to prevent physically implausible
        regime assignments.

        Rules:
          - CRISIS requires high vol AND large drawdown
          - RECOVERY requires recent large drawdown but improving trend
          - TOPPING requires bull context but momentum divergence
          - BULL and BEAR are mutually suppressed

        Parameters
        ----------
        composite : raw composite scores before softmax
        normed    : normalised features

        Returns
        -------
        dict with adjusted scores
        """
        scores = dict(composite)

        vol   = normed.get("vol_percentile_30d", 0.5)
        dd    = normed.get("drawdown_from_ath",  0.0)
        trend = normed.get("ema_ratio_50_200",   0.5)
        mom20 = normed.get("momentum_20d",       0.5)

        # CRISIS needs high vol + significant drawdown
        crisis_gate = vol * dd
        scores["CRISIS"] *= crisis_gate * 2.0

        # RECOVERY requires some drawdown history but upward trend
        recovery_gate = dd * trend * max(0.0, 1.0 - vol)
        scores["RECOVERY"] *= recovery_gate * 2.0

        # TOPPING requires bull trend but fading short momentum
        topping_gate = trend * max(0.0, 0.5 - mom20) * 2.0
        scores["TOPPING"] *= topping_gate * 2.0

        # BULL suppressed during high vol
        if vol > 0.75:
            scores["BULL"] *= 0.5

        # BEAR suppressed during low vol
        if vol < 0.25:
            scores["BEAR"] *= 0.6

        # Mutual suppression BULL ↔ BEAR
        if scores["BULL"] > scores["BEAR"] * 2:
            scores["BEAR"] *= 0.7
        elif scores["BEAR"] > scores["BULL"] * 2:
            scores["BULL"] *= 0.7

        # Floor: prevent any regime from being completely impossible
        floor = 0.001
        for k in scores:
            scores[k] = max(scores[k], floor)

        return scores

    # ------------------------------------------------------------------
    # Single bar estimation (no history)
    # ------------------------------------------------------------------

    def _single_bar_features(
        self,
        bar:    Dict[str, Any],
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Build a minimal feature set from a single bar (no rolling context).

        Used as a fallback when history is unavailable.
        All rolling features default to neutral (0.5).
        """
        close = float(bar.get("close") or bar.get("Close") or 0.0)
        features: Dict[str, Any] = {
            "symbol":             symbol,
            "n_bars":             1,
            "ema_ratio_50_200":   1.0,   # assume neutral
            "ema_ratio_20_50":    1.0,
            "vol_percentile_30d": 0.5,
            "vol_percentile_90d": 0.5,
            "momentum_5d":        0.0,
            "momentum_20d":       0.0,
            "momentum_60d":       0.0,
            "breadth_50d":        0.5,
            "drawdown_from_ath":  0.0,
            "bh_mass_mean":       float("nan"),
            "bh_mass_max":        float("nan"),
            "atr_percentile":     0.5,
            "skewness_20d":       0.0,
            "kurtosis_20d":       3.0,
        }
        return features

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "RegimeOracle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RegimeOracle(db={self.db_path.name!r})"


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Regime Oracle — classify current regime")
    parser.add_argument("--db",     default=str(_DB_DEFAULT))
    parser.add_argument("--symbol", default="BTC")
    args = parser.parse_args()

    oracle = RegimeOracle(db_path=args.db)
    state  = oracle.get_latest_regime(args.symbol)
    if state:
        print(json.dumps(state.to_dict(), indent=2))
    else:
        print(f"No regime history found for {args.symbol}.")
    oracle.close()
