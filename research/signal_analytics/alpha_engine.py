"""
research/signal_analytics/alpha_engine.py
==========================================
Systematic IC/ICIR tracking and signal lifecycle management for SRFM-Lab.

Provides:
- Signal and SignalRecord dataclasses
- ICCalculator: rolling IC/ICIR, decay by horizon, sign-flip detection
- AlphaDecayAnalyzer: Fama-MacBeth regression, exponential decay, half-life
- SignalCombiner: equal-weight, IC-weight, max-diversification, redundancy detection
- SignalRetirementEngine: auto-retirement on low ICIR, SQLite logging
- AlphaEngine: orchestrates all above, SQLite persistence, daily scoring, weekly reports
- AlphaReport: top/bottom signals, new signals, combined ICIR, factor correlations

Dependencies: numpy, pandas, scipy, sqlite3, logging
"""

from __future__ import annotations

import logging
import math
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETIREMENT_ICIR_THRESHOLD = 0.2
RETIREMENT_CONSECUTIVE_DAYS = 30
RETIREMENT_LOOKBACK = 60
REDUNDANCY_CORR_THRESHOLD = 0.8
NEW_SIGNAL_DAYS = 30
HALF_LIFE_STALE_DAYS = 5

VALID_CATEGORIES = frozenset(
    ["momentum", "mean_reversion", "volatility", "macro", "microstructure", "physics", "technical"]
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """Descriptor for a tradeable signal."""

    name: str
    description: str
    category: str  # momentum / mean_reversion / volatility / macro / microstructure / physics / technical
    compute: Callable  # callable(prices, volume=None, **params) -> pd.Series
    parameters: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    retirement_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"Signal category '{self.category}' not in {VALID_CATEGORIES}"
            )

    def compute_signal(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Run the compute function with stored parameters."""
        return self.compute(prices, volume=volume, **self.parameters)


@dataclass
class SignalRecord:
    """One observation row: signal value + forward returns + IC estimates."""

    signal_name: str
    symbol: str
    timestamp: datetime
    value: float
    forward_return_1d: Optional[float] = None
    forward_return_5d: Optional[float] = None
    forward_return_20d: Optional[float] = None
    ic_1d: Optional[float] = None
    ic_5d: Optional[float] = None
    ic_20d: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "signal_name": self.signal_name,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "value": self.value,
            "forward_return_1d": self.forward_return_1d,
            "forward_return_5d": self.forward_return_5d,
            "forward_return_20d": self.forward_return_20d,
            "ic_1d": self.ic_1d,
            "ic_5d": self.ic_5d,
            "ic_20d": self.ic_20d,
        }


@dataclass
class AlphaReport:
    """Weekly alpha engine summary report."""

    generated_at: datetime
    top_signals: List[Dict]          # top 10 by ICIR
    bottom_signals: List[Dict]       # bottom 5 (retirement candidates)
    new_signals: List[str]           # signals < 30 days old
    combined_icir: float             # ICIR of the combined signal
    factor_corr_summary: Dict        # mean/max cross-signal correlation by category
    total_active: int
    total_retired: int
    retirement_candidates: List[str]


# ---------------------------------------------------------------------------
# ICCalculator
# ---------------------------------------------------------------------------


class ICCalculator:
    """
    Information coefficient calculation utilities.

    IC = Spearman rank correlation between signal values and forward returns.
    ICIR = rolling IC mean / rolling IC std.
    """

    def __init__(self, min_obs: int = 20) -> None:
        self.min_obs = min_obs

    # ------------------------------------------------------------------
    # Core IC
    # ------------------------------------------------------------------

    def compute_ic(
        self,
        signal: pd.Series,
        forward_return: pd.Series,
        method: str = "spearman",
    ) -> float:
        """Compute IC between a signal and its forward return series."""
        valid = signal.dropna().index.intersection(forward_return.dropna().index)
        if len(valid) < self.min_obs:
            return float("nan")
        s = signal.loc[valid]
        r = forward_return.loc[valid]
        if method == "spearman":
            rho, _ = stats.spearmanr(s, r)
        elif method == "pearson":
            rho, _ = stats.pearsonr(s, r)
        else:
            rho, _ = stats.kendalltau(s, r)
        return float(rho)

    # ------------------------------------------------------------------
    # Rolling IC
    # ------------------------------------------------------------------

    def rolling_ic(
        self,
        signal: pd.Series,
        forward_return: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """Compute rolling window IC between signal and forward return."""
        combined = pd.concat([signal.rename("sig"), forward_return.rename("fwd")], axis=1).dropna()
        if len(combined) < window:
            return pd.Series(dtype=float)

        ics: List[float] = []
        dates: List = []
        for i in range(window, len(combined) + 1):
            chunk = combined.iloc[i - window : i]
            if len(chunk) < self.min_obs:
                ics.append(float("nan"))
            else:
                rho, _ = stats.spearmanr(chunk["sig"], chunk["fwd"])
                ics.append(float(rho))
            dates.append(combined.index[i - 1])
        return pd.Series(ics, index=dates, name="rolling_ic")

    # ------------------------------------------------------------------
    # ICIR
    # ------------------------------------------------------------------

    def rolling_icir(
        self,
        signal: pd.Series,
        forward_return: pd.Series,
        window: int = 60,
        icir_window: int = 60,
    ) -> pd.Series:
        """Rolling ICIR = rolling_ic_mean / rolling_ic_std."""
        ic_ts = self.rolling_ic(signal, forward_return, window=window)
        if ic_ts.empty:
            return pd.Series(dtype=float)
        roll_mean = ic_ts.rolling(icir_window, min_periods=max(10, icir_window // 4)).mean()
        roll_std = ic_ts.rolling(icir_window, min_periods=max(10, icir_window // 4)).std()
        icir = roll_mean / roll_std.replace(0, float("nan"))
        icir.name = "icir"
        return icir

    def compute_icir(
        self,
        ic_series: pd.Series,
        window: Optional[int] = None,
    ) -> float:
        """ICIR from a pre-computed IC time series."""
        if window is not None:
            ic_series = ic_series.iloc[-window:]
        s = ic_series.dropna()
        if len(s) < 5:
            return float("nan")
        std = s.std()
        if std == 0:
            return float("nan")
        return float(s.mean() / std)

    # ------------------------------------------------------------------
    # IC decay by horizon
    # ------------------------------------------------------------------

    def ic_decay(
        self,
        signal: pd.Series,
        prices: pd.Series,
        horizons: List[int] = (1, 5, 10, 20, 40, 60),
    ) -> Dict[int, float]:
        """
        Compute IC at multiple forward-return horizons.
        Returns dict mapping horizon (days) -> IC value.
        """
        log_prices = np.log(prices.replace(0, float("nan")))
        result: Dict[int, float] = {}
        for h in horizons:
            fwd = log_prices.shift(-h) - log_prices
            ic = self.compute_ic(signal, fwd)
            result[int(h)] = ic
        return result

    # ------------------------------------------------------------------
    # Sign-flip detection
    # ------------------------------------------------------------------

    def detect_sign_flip(
        self,
        ic_series: pd.Series,
        window: int = 30,
        threshold: float = 0.1,
    ) -> bool:
        """
        Return True if the signal IC has inverted direction recently.

        Compares early-period IC mean to recent-period IC mean.
        A flip is detected when both halves exceed *threshold* in magnitude
        but have opposite signs.
        """
        s = ic_series.dropna()
        if len(s) < 2 * window:
            return False
        early = s.iloc[-(2 * window) : -window].mean()
        recent = s.iloc[-window:].mean()
        return (
            abs(early) > threshold
            and abs(recent) > threshold
            and np.sign(early) != np.sign(recent)
        )

    # ------------------------------------------------------------------
    # Batch IC computation across horizons with t-stats
    # ------------------------------------------------------------------

    def full_ic_summary(
        self,
        signal: pd.Series,
        prices: pd.Series,
        horizons: List[int] = (1, 5, 20, 60),
    ) -> pd.DataFrame:
        """Return DataFrame with IC, t-stat, p-value per horizon."""
        log_prices = np.log(prices.replace(0, float("nan")))
        rows = []
        for h in horizons:
            fwd = log_prices.shift(-h) - log_prices
            valid = signal.dropna().index.intersection(fwd.dropna().index)
            if len(valid) < self.min_obs:
                rows.append({"horizon": h, "ic": float("nan"), "t_stat": float("nan"), "p_value": float("nan"), "n_obs": 0})
                continue
            s = signal.loc[valid]
            r = fwd.loc[valid]
            rho, p = stats.spearmanr(s, r)
            n = len(valid)
            # t-stat = rho * sqrt((n-2)/(1-rho^2))
            denom = max(1 - rho ** 2, 1e-10)
            t = rho * math.sqrt(max(n - 2, 0) / denom)
            rows.append({"horizon": h, "ic": float(rho), "t_stat": float(t), "p_value": float(p), "n_obs": n})
        return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# AlphaDecayAnalyzer
# ---------------------------------------------------------------------------


class AlphaDecayAnalyzer:
    """
    Fama-MacBeth regression and exponential decay fitting for signal half-life.

    The decay model: IC(h) = IC0 * exp(-lambda * h)
    Half-life = ln(2) / lambda
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Exponential decay fit
    # ------------------------------------------------------------------

    @staticmethod
    def _exp_decay(h: np.ndarray, ic0: float, lam: float) -> np.ndarray:
        return ic0 * np.exp(-lam * np.asarray(h))

    def fit_decay(
        self,
        ic_decay_dict: Dict[int, float],
    ) -> Dict:
        """
        Fit exponential decay to IC-horizon dict.
        Returns dict with ic0, lambda_, half_life, r_squared.
        """
        horizons = np.array(sorted(ic_decay_dict.keys()), dtype=float)
        ic_vals = np.array([ic_decay_dict[int(h)] for h in horizons], dtype=float)

        # Remove NaN
        mask = np.isfinite(ic_vals)
        if mask.sum() < 3:
            return {"ic0": float("nan"), "lambda_": float("nan"), "half_life": float("nan"), "r_squared": float("nan")}

        h_clean = horizons[mask]
        ic_clean = ic_vals[mask]

        ic0_guess = ic_clean[0] if abs(ic_clean[0]) > 1e-8 else 0.05
        lam_guess = 0.02

        try:
            popt, _ = curve_fit(
                self._exp_decay,
                h_clean,
                ic_clean,
                p0=[ic0_guess, lam_guess],
                maxfev=2000,
                bounds=([-1, 1e-6], [1, 10]),
            )
            ic0, lam = popt
            fitted = self._exp_decay(h_clean, ic0, lam)
            ss_res = np.sum((ic_clean - fitted) ** 2)
            ss_tot = np.sum((ic_clean - ic_clean.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
            half_life = math.log(2) / max(lam, 1e-10)
        except Exception:
            ic0, lam, half_life, r2 = float("nan"), float("nan"), float("nan"), float("nan")

        return {
            "ic0": float(ic0),
            "lambda_": float(lam),
            "half_life": float(half_life),
            "r_squared": float(r2),
        }

    # ------------------------------------------------------------------
    # Fama-MacBeth regression of IC on lag
    # ------------------------------------------------------------------

    def fama_macbeth_ic_regression(
        self,
        ic_panel: pd.DataFrame,
    ) -> Dict:
        """
        Fama-MacBeth regression.

        ic_panel: DataFrame indexed by date, columns are signals.
        Returns average cross-sectional slope (beta) and t-stat.
        """
        betas: List[float] = []
        for _, row in ic_panel.iterrows():
            y = row.dropna().values
            if len(y) < 5:
                continue
            x = np.arange(len(y), dtype=float)
            if np.std(x) < 1e-10:
                continue
            slope, _, _, _, _ = stats.linregress(x, y)
            betas.append(float(slope))

        if len(betas) < 5:
            return {"beta": float("nan"), "t_stat": float("nan"), "n_periods": len(betas)}

        betas_arr = np.array(betas)
        t_stat = betas_arr.mean() / (betas_arr.std(ddof=1) / math.sqrt(len(betas_arr)))
        return {
            "beta": float(betas_arr.mean()),
            "t_stat": float(t_stat),
            "n_periods": len(betas_arr),
        }

    # ------------------------------------------------------------------
    # Is signal stale?
    # ------------------------------------------------------------------

    def is_stale(self, ic_decay_dict: Dict[int, float]) -> bool:
        """Return True if fitted half-life < HALF_LIFE_STALE_DAYS."""
        result = self.fit_decay(ic_decay_dict)
        hl = result.get("half_life", float("nan"))
        if not math.isfinite(hl):
            return False
        return hl < HALF_LIFE_STALE_DAYS


# ---------------------------------------------------------------------------
# SignalCombiner
# ---------------------------------------------------------------------------


class SignalCombiner:
    """
    Combines multiple signals into a composite.

    Methods:
    - equal_weight: simple average
    - ic_weight: weight by recent IC
    - max_diversification: weight to maximize diversification ratio
    """

    def __init__(
        self,
        redundancy_threshold: float = REDUNDANCY_CORR_THRESHOLD,
    ) -> None:
        self.redundancy_threshold = redundancy_threshold

    # ------------------------------------------------------------------
    # Correlation and redundancy
    # ------------------------------------------------------------------

    def signal_correlation_matrix(
        self, signal_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Spearman rank correlation matrix of all signals."""
        ranked = signal_df.rank()
        return ranked.corr(method="spearman")

    def deduplicate(
        self, signal_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove redundant signals (|corr| > threshold).
        Returns (deduplicated_df, list_of_removed_names).
        """
        corr = self.signal_correlation_matrix(signal_df)
        cols = list(corr.columns)
        keep: List[str] = []
        removed: List[str] = []

        for col in cols:
            redundant = False
            for k in keep:
                if abs(corr.loc[col, k]) > self.redundancy_threshold:
                    redundant = True
                    break
            if redundant:
                removed.append(col)
            else:
                keep.append(col)

        return signal_df[keep], removed

    # ------------------------------------------------------------------
    # Equal weight
    # ------------------------------------------------------------------

    def equal_weight(self, signal_df: pd.DataFrame) -> pd.Series:
        """Z-score each signal, then average."""
        z_scores = (signal_df - signal_df.mean()) / signal_df.std().replace(0, float("nan"))
        combined = z_scores.mean(axis=1)
        combined.name = "combined_equal_weight"
        return combined

    # ------------------------------------------------------------------
    # IC weight
    # ------------------------------------------------------------------

    def ic_weight(
        self,
        signal_df: pd.DataFrame,
        ic_dict: Dict[str, float],
    ) -> pd.Series:
        """
        Weight each signal by its IC value.
        Only uses positive-IC signals; falls back to equal weight if none.
        """
        weights: Dict[str, float] = {}
        for col in signal_df.columns:
            ic = ic_dict.get(col, 0.0)
            if math.isfinite(ic):
                weights[col] = max(ic, 0.0)

        total_weight = sum(weights.values())
        if total_weight < 1e-10:
            return self.equal_weight(signal_df)

        z_scores = (signal_df - signal_df.mean()) / signal_df.std().replace(0, float("nan"))
        combined = pd.Series(0.0, index=signal_df.index)
        for col, w in weights.items():
            if col in z_scores.columns:
                combined += z_scores[col].fillna(0.0) * (w / total_weight)
        combined.name = "combined_ic_weight"
        return combined

    # ------------------------------------------------------------------
    # Max diversification
    # ------------------------------------------------------------------

    def max_diversification(self, signal_df: pd.DataFrame) -> pd.Series:
        """
        Approximate max-diversification combination.
        Weights are proportional to each signal's individual std divided
        by the portfolio std (greedy approach).
        """
        z_scores = (signal_df - signal_df.mean()) / signal_df.std().replace(0, float("nan"))
        z_clean = z_scores.fillna(0.0)
        corr = self.signal_correlation_matrix(signal_df.dropna())

        # Greedy: weight proportional to 1/(avg_abs_corr_with_others + eps)
        n = len(corr.columns)
        weights = np.ones(n)
        for i, col in enumerate(corr.columns):
            others = [c for c in corr.columns if c != col]
            if others:
                avg_corr = corr.loc[col, others].abs().mean()
                weights[i] = 1.0 / (avg_corr + 0.1)

        weights = weights / weights.sum()
        combined = z_clean.values @ weights
        result = pd.Series(combined, index=signal_df.index, name="combined_max_div")
        return result

    # ------------------------------------------------------------------
    # Active signal count after dedup
    # ------------------------------------------------------------------

    def active_count_after_dedup(self, signal_df: pd.DataFrame) -> int:
        """Number of signals remaining after redundancy removal."""
        deduped, _ = self.deduplicate(signal_df)
        return len(deduped.columns)


# ---------------------------------------------------------------------------
# SignalRetirementEngine
# ---------------------------------------------------------------------------


class SignalRetirementEngine:
    """
    Monitors ICIR over rolling window and auto-retires stale signals.

    Retirement condition: ICIR < 0.2 for 30 consecutive trading days
    over a 60-day rolling lookback.
    """

    def __init__(
        self,
        db_path: Path,
        icir_threshold: float = RETIREMENT_ICIR_THRESHOLD,
        consecutive_days: int = RETIREMENT_CONSECUTIVE_DAYS,
        lookback: int = RETIREMENT_LOOKBACK,
    ) -> None:
        self.db_path = Path(db_path)
        self.icir_threshold = icir_threshold
        self.consecutive_days = consecutive_days
        self.lookback = lookback
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_retirements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_name TEXT NOT NULL,
                    retired_at TEXT NOT NULL,
                    reason TEXT,
                    final_icir REAL,
                    consecutive_days_below INTEGER
                )
                """
            )

    def check_and_retire(
        self,
        signal_name: str,
        ic_series: pd.Series,
        signals_registry: Dict[str, Signal],
    ) -> bool:
        """
        Check if a signal should be retired.
        Returns True if the signal was retired.
        """
        recent = ic_series.dropna().iloc[-self.lookback :]
        if len(recent) < self.consecutive_days:
            return False

        icir_calc = ICCalculator()
        icir_val = icir_calc.compute_icir(recent)

        # Count consecutive days below threshold at end of series
        below = (recent.abs() < self.icir_threshold).values
        consecutive = 0
        for flag in reversed(below):
            if flag:
                consecutive += 1
            else:
                break

        if consecutive >= self.consecutive_days:
            self._retire_signal(signal_name, icir_val, consecutive, signals_registry)
            return True
        return False

    def _retire_signal(
        self,
        signal_name: str,
        final_icir: float,
        consecutive: int,
        signals_registry: Dict[str, Signal],
    ) -> None:
        reason = (
            f"ICIR {final_icir:.4f} below threshold {self.icir_threshold} "
            f"for {consecutive} consecutive days"
        )
        logger.warning(
            "Signal '%s' auto-retired. %s", signal_name, reason
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO signal_retirements
                (signal_name, retired_at, reason, final_icir, consecutive_days_below)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    signal_name,
                    datetime.utcnow().isoformat(),
                    reason,
                    float(final_icir) if math.isfinite(float(final_icir)) else None,
                    int(consecutive),
                ),
            )
        if signal_name in signals_registry:
            signals_registry[signal_name].is_active = False
            signals_registry[signal_name].retirement_reason = reason

    def get_retirement_log(self) -> pd.DataFrame:
        """Return full retirement history from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("SELECT * FROM signal_retirements ORDER BY retired_at DESC", conn)
        return df


# ---------------------------------------------------------------------------
# AlphaEngine
# ---------------------------------------------------------------------------


class AlphaEngine:
    """
    Master orchestrator: wraps signals, records to SQLite, scores daily,
    generates weekly reports.
    """

    def __init__(
        self,
        db_path: Path = Path("data/alpha_engine.db"),
        ic_window: int = 60,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ic_window = ic_window

        self.signals: Dict[str, Signal] = {}
        self.ic_calc = ICCalculator()
        self.decay_analyzer = AlphaDecayAnalyzer()
        self.combiner = SignalCombiner()
        self.retirement_engine = SignalRetirementEngine(db_path=self.db_path)

        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alpha_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    value REAL,
                    forward_return_1d REAL,
                    forward_return_5d REAL,
                    forward_return_20d REAL,
                    ic_1d REAL,
                    ic_5d REAL,
                    ic_20d REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_signal_symbol ON alpha_records(signal_name, symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON alpha_records(timestamp)"
            )

    # ------------------------------------------------------------------
    # Signal registration
    # ------------------------------------------------------------------

    def register(self, signal: Signal) -> None:
        """Register a signal with the engine."""
        if signal.name in self.signals:
            logger.warning("Signal '%s' already registered; overwriting.", signal.name)
        self.signals[signal.name] = signal
        logger.debug("Registered signal '%s' (category=%s)", signal.name, signal.category)

    def register_many(self, signals: List[Signal]) -> None:
        for s in signals:
            self.register(s)

    def get_active_signals(self) -> Dict[str, Signal]:
        return {k: v for k, v in self.signals.items() if v.is_active}

    # ------------------------------------------------------------------
    # Daily scoring
    # ------------------------------------------------------------------

    def score_daily(
        self,
        symbol: str,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Compute all active signals for a symbol on a given date.
        Returns dict of signal_name -> value.
        Also persists records to SQLite.
        """
        if date is None:
            date = datetime.utcnow()

        scores: Dict[str, float] = {}
        log_prices = np.log(prices.replace(0, float("nan")))
        fwd_1d = (log_prices.shift(-1) - log_prices).iloc[-1] if len(prices) > 1 else float("nan")
        fwd_5d = (log_prices.shift(-5) - log_prices).iloc[-1] if len(prices) > 5 else float("nan")
        fwd_20d = (log_prices.shift(-20) - log_prices).iloc[-1] if len(prices) > 20 else float("nan")

        records_to_insert = []
        for name, signal in self.get_active_signals().items():
            try:
                sig_series = signal.compute_signal(prices, volume=volume)
                if sig_series is None or sig_series.empty:
                    continue
                latest_val = float(sig_series.dropna().iloc[-1]) if not sig_series.dropna().empty else float("nan")
                scores[name] = latest_val

                record = {
                    "signal_name": name,
                    "symbol": symbol,
                    "timestamp": date.isoformat(),
                    "value": latest_val if math.isfinite(latest_val) else None,
                    "forward_return_1d": float(fwd_1d) if math.isfinite(float(fwd_1d)) else None,
                    "forward_return_5d": float(fwd_5d) if math.isfinite(float(fwd_5d)) else None,
                    "forward_return_20d": float(fwd_20d) if math.isfinite(float(fwd_20d)) else None,
                    "ic_1d": None,
                    "ic_5d": None,
                    "ic_20d": None,
                }
                records_to_insert.append(record)
            except Exception as exc:
                logger.debug("Signal '%s' failed for '%s': %s", name, symbol, exc)

        if records_to_insert:
            self._bulk_insert(records_to_insert)

        return scores

    def _bulk_insert(self, records: List[Dict]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO alpha_records
                (signal_name, symbol, timestamp, value,
                 forward_return_1d, forward_return_5d, forward_return_20d,
                 ic_1d, ic_5d, ic_20d)
                VALUES
                (:signal_name, :symbol, :timestamp, :value,
                 :forward_return_1d, :forward_return_5d, :forward_return_20d,
                 :ic_1d, :ic_5d, :ic_20d)
                """,
                records,
            )

    # ------------------------------------------------------------------
    # IC computation from stored records
    # ------------------------------------------------------------------

    def compute_rolling_ics(
        self,
        signal_name: str,
        symbol: str,
        horizon: int = 1,
    ) -> pd.Series:
        """Load stored records and compute rolling IC time series."""
        col_map = {1: "forward_return_1d", 5: "forward_return_5d", 20: "forward_return_20d"}
        fwd_col = col_map.get(horizon, "forward_return_1d")

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                f"SELECT timestamp, value, {fwd_col} FROM alpha_records "
                f"WHERE signal_name=? AND symbol=? ORDER BY timestamp",
                conn,
                params=(signal_name, symbol),
            )
        if df.empty:
            return pd.Series(dtype=float)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").dropna()
        return self.ic_calc.rolling_ic(df["value"], df[fwd_col], window=self.ic_window)

    def get_icir_summary(
        self,
        symbol: str,
        window: int = 60,
    ) -> pd.DataFrame:
        """Return DataFrame of signal_name -> ICIR over *window* recent IC observations."""
        rows = []
        for name in self.get_active_signals():
            for h in [1, 5, 20]:
                ic_ts = self.compute_rolling_ics(name, symbol, horizon=h)
                icir = self.ic_calc.compute_icir(ic_ts, window=window)
                rows.append({"signal_name": name, "horizon": h, "icir": icir})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Retirement checks
    # ------------------------------------------------------------------

    def run_retirement_check(
        self,
        symbol: str,
    ) -> List[str]:
        """Run retirement check for all active signals; return retired names."""
        retired = []
        for name in list(self.get_active_signals().keys()):
            ic_ts = self.compute_rolling_ics(name, symbol, horizon=1)
            if ic_ts.empty:
                continue
            was_retired = self.retirement_engine.check_and_retire(
                name, ic_ts, self.signals
            )
            if was_retired:
                retired.append(name)
        return retired

    # ------------------------------------------------------------------
    # Signal combination
    # ------------------------------------------------------------------

    def build_combined_signal(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        method: str = "ic_weight",
    ) -> pd.Series:
        """
        Compute all active signals and combine them.

        method: 'equal_weight' | 'ic_weight' | 'max_diversification'
        """
        sig_series: Dict[str, pd.Series] = {}
        for name, signal in self.get_active_signals().items():
            try:
                s = signal.compute_signal(prices, volume=volume)
                if s is not None and not s.empty:
                    sig_series[name] = s
            except Exception as exc:
                logger.debug("Signal '%s' failed during combination: %s", name, exc)

        if not sig_series:
            return pd.Series(dtype=float)

        # Align all series to common index
        signal_df = pd.DataFrame(sig_series).dropna(how="all")
        if signal_df.empty:
            return pd.Series(dtype=float)

        if method == "equal_weight":
            return self.combiner.equal_weight(signal_df)
        elif method == "max_diversification":
            return self.combiner.max_diversification(signal_df)
        else:
            # ic_weight: need IC estimates per signal
            log_prices = np.log(prices.replace(0, float("nan")))
            fwd_1d = log_prices.shift(-1) - log_prices
            ic_dict: Dict[str, float] = {}
            for col in signal_df.columns:
                ic_dict[col] = self.ic_calc.compute_ic(signal_df[col], fwd_1d)
            return self.combiner.ic_weight(signal_df, ic_dict)

    # ------------------------------------------------------------------
    # Weekly report
    # ------------------------------------------------------------------

    def generate_weekly_report(
        self,
        symbol: str,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> AlphaReport:
        """Generate the weekly AlphaReport."""
        active = self.get_active_signals()
        total_active = len(active)
        total_retired = len(self.signals) - total_active

        # Build signal matrix for correlation
        sig_series: Dict[str, pd.Series] = {}
        for name, signal in active.items():
            try:
                s = signal.compute_signal(prices, volume=volume)
                if s is not None and not s.empty:
                    sig_series[name] = s
            except Exception:
                pass

        signal_df = pd.DataFrame(sig_series).dropna(how="all")

        # Per-signal ICIR
        log_prices = np.log(prices.replace(0, float("nan")))
        signal_stats: List[Dict] = []
        for name in sig_series:
            fwd_1d = log_prices.shift(-1) - log_prices
            ic_ts = self.ic_calc.rolling_ic(signal_df[name].dropna(), fwd_1d.reindex(signal_df.index), window=min(60, len(signal_df)))
            icir = self.ic_calc.compute_icir(ic_ts)
            decay_dict = self.ic_calc.ic_decay(signal_df[name], prices) if len(prices) > 60 else {}
            hl_info = self.decay_analyzer.fit_decay(decay_dict) if decay_dict else {}
            signal_stats.append({
                "name": name,
                "category": active[name].category,
                "icir": icir if math.isfinite(icir) else float("nan"),
                "half_life": hl_info.get("half_life", float("nan")),
                "created_at": active[name].created_at,
            })

        signal_stats_sorted = sorted(
            [s for s in signal_stats if math.isfinite(s["icir"])],
            key=lambda x: x["icir"],
            reverse=True,
        )

        top_signals = signal_stats_sorted[:10]
        bottom_signals = signal_stats_sorted[-5:] if len(signal_stats_sorted) >= 5 else signal_stats_sorted[::-1]

        # New signals (< 30 days old)
        cutoff = datetime.utcnow() - timedelta(days=NEW_SIGNAL_DAYS)
        new_signals = [
            s["name"]
            for s in signal_stats
            if isinstance(s["created_at"], datetime) and s["created_at"] > cutoff
        ]

        # Combined signal ICIR
        combined_icir = float("nan")
        retirement_candidates: List[str] = []
        if not signal_df.empty:
            try:
                combined = self.combiner.equal_weight(signal_df)
                fwd_1d = log_prices.shift(-1) - log_prices
                ic_ts = self.ic_calc.rolling_ic(combined, fwd_1d.reindex(signal_df.index), window=min(60, len(signal_df)))
                combined_icir = self.ic_calc.compute_icir(ic_ts)
            except Exception:
                pass

            # Retirement candidates: ICIR < threshold
            retirement_candidates = [
                s["name"]
                for s in signal_stats
                if math.isfinite(s["icir"]) and s["icir"] < RETIREMENT_ICIR_THRESHOLD
            ]

        # Factor correlation summary
        factor_corr_summary: Dict = {}
        if not signal_df.empty and len(signal_df.columns) > 1:
            corr_mat = self.combiner.signal_correlation_matrix(signal_df)
            np.fill_diagonal(corr_mat.values, float("nan"))
            factor_corr_summary = {
                "mean_abs_corr": float(corr_mat.abs().stack().mean()),
                "max_abs_corr": float(corr_mat.abs().stack().max()),
                "n_redundant_pairs": int(
                    (corr_mat.abs().stack() > REDUNDANCY_CORR_THRESHOLD).sum() // 2
                ),
            }

        return AlphaReport(
            generated_at=datetime.utcnow(),
            top_signals=top_signals,
            bottom_signals=bottom_signals,
            new_signals=new_signals,
            combined_icir=float(combined_icir) if math.isfinite(float(combined_icir) if math.isfinite(combined_icir) else 0) else float("nan"),
            factor_corr_summary=factor_corr_summary,
            total_active=total_active,
            total_retired=total_retired,
            retirement_candidates=retirement_candidates,
        )

    # ------------------------------------------------------------------
    # Persist a single SignalRecord
    # ------------------------------------------------------------------

    def insert_record(self, rec: SignalRecord) -> None:
        """Insert a single SignalRecord into the database."""
        d = rec.to_dict()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO alpha_records
                (signal_name, symbol, timestamp, value,
                 forward_return_1d, forward_return_5d, forward_return_20d,
                 ic_1d, ic_5d, ic_20d)
                VALUES
                (:signal_name, :symbol, :timestamp, :value,
                 :forward_return_1d, :forward_return_5d, :forward_return_20d,
                 :ic_1d, :ic_5d, :ic_20d)
                """,
                d,
            )

    # ------------------------------------------------------------------
    # Load records
    # ------------------------------------------------------------------

    def load_records(
        self,
        signal_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """Load records from SQLite with optional filters."""
        clauses = []
        params: List = []
        if signal_name:
            clauses.append("signal_name = ?")
            params.append(signal_name)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"SELECT * FROM alpha_records {where} ORDER BY timestamp DESC LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(query, conn, params=params)


# ---------------------------------------------------------------------------
# Utility: build SignalRecord from live data
# ---------------------------------------------------------------------------


def make_signal_record(
    signal_name: str,
    symbol: str,
    timestamp: datetime,
    signal_value: float,
    prices: pd.Series,
) -> SignalRecord:
    """Convenience constructor for a SignalRecord with forward returns."""
    log_prices = np.log(prices.replace(0, float("nan")))
    idx = prices.index.get_loc(prices.index[-1]) if hasattr(prices.index, "get_loc") else -1

    def safe_fwd(shift: int) -> Optional[float]:
        try:
            lp = log_prices
            val = (lp.shift(-shift) - lp).iloc[-1]
            return float(val) if math.isfinite(float(val)) else None
        except Exception:
            return None

    return SignalRecord(
        signal_name=signal_name,
        symbol=symbol,
        timestamp=timestamp,
        value=signal_value,
        forward_return_1d=safe_fwd(1),
        forward_return_5d=safe_fwd(5),
        forward_return_20d=safe_fwd(20),
    )


# ---------------------------------------------------------------------------
# PortfolioSignalScorer
# ---------------------------------------------------------------------------


class PortfolioSignalScorer:
    """
    Score a universe of symbols simultaneously.

    Wraps AlphaEngine.score_daily for multiple symbols and returns a
    cross-sectional score matrix.
    """

    def __init__(self, engine: AlphaEngine) -> None:
        self.engine = engine

    def score_universe(
        self,
        symbol_price_map: Dict[str, pd.Series],
        symbol_volume_map: Optional[Dict[str, pd.Series]] = None,
        date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Score all symbols and return a DataFrame of shape
        (n_symbols, n_signals).

        symbol_price_map: {symbol: price_series}
        symbol_volume_map: optional {symbol: volume_series}
        """
        if date is None:
            date = datetime.utcnow()

        rows: Dict[str, Dict[str, float]] = {}
        for symbol, prices in symbol_price_map.items():
            volume = None
            if symbol_volume_map:
                volume = symbol_volume_map.get(symbol)
            try:
                scores = self.engine.score_daily(symbol, prices, volume=volume, date=date)
                rows[symbol] = scores
            except Exception as exc:
                logger.error("PortfolioSignalScorer failed for %s: %s", symbol, exc)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).T

    def rank_universe(
        self,
        symbol_price_map: Dict[str, pd.Series],
        symbol_volume_map: Optional[Dict[str, pd.Series]] = None,
        date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Return cross-sectional rank (0-1) of each signal across symbols.
        """
        scores = self.score_universe(symbol_price_map, symbol_volume_map, date)
        if scores.empty:
            return scores
        return scores.rank(pct=True)

    def composite_score(
        self,
        symbol_price_map: Dict[str, pd.Series],
        symbol_volume_map: Optional[Dict[str, pd.Series]] = None,
        method: str = "equal_weight",
    ) -> pd.Series:
        """
        Compute a single composite score per symbol.
        Returns pd.Series of symbol -> composite_score.
        """
        scores = self.score_universe(symbol_price_map, symbol_volume_map)
        if scores.empty:
            return pd.Series(dtype=float)

        combiner = SignalCombiner()
        if method == "rank":
            ranked = scores.rank(pct=True)
            return ranked.mean(axis=1)
        else:
            z_scores = (scores - scores.mean()) / scores.std().replace(0.0, float("nan"))
            return z_scores.mean(axis=1)


# ---------------------------------------------------------------------------
# ICPanel: time-series IC tracking for an entire signal library
# ---------------------------------------------------------------------------


class ICPanel:
    """
    Maintains a rolling panel of IC observations across all signals.

    Stores IC(signal, fwd_1d), IC(signal, fwd_5d), IC(signal, fwd_20d)
    per date, enabling cross-signal IC correlation and factor model analysis.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._calc = ICCalculator()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ic_panel (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    signal_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    ic_1d REAL,
                    ic_5d REAL,
                    ic_20d REAL,
                    UNIQUE(date, signal_name, symbol)
                )
                """
            )

    def record(
        self,
        date: datetime,
        signal_name: str,
        symbol: str,
        signal_values: pd.Series,
        prices: pd.Series,
    ) -> None:
        """Record IC values for one signal/symbol/date combination."""
        log_prices = np.log(prices.replace(0, float("nan")))
        ic_vals = {}
        for h, col in [(1, "ic_1d"), (5, "ic_5d"), (20, "ic_20d")]:
            fwd = log_prices.shift(-h) - log_prices
            ic = self._calc.compute_ic(signal_values, fwd)
            ic_vals[col] = float(ic) if math.isfinite(ic) else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ic_panel
                (date, signal_name, symbol, ic_1d, ic_5d, ic_20d)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    date.date().isoformat(),
                    signal_name,
                    symbol,
                    ic_vals["ic_1d"],
                    ic_vals["ic_5d"],
                    ic_vals["ic_20d"],
                ),
            )

    def load_ic_matrix(
        self,
        horizon: int = 1,
        signal_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load IC panel as a date x signal matrix.
        """
        col = {1: "ic_1d", 5: "ic_5d", 20: "ic_20d"}.get(horizon, "ic_1d")
        if signal_names:
            placeholders = ",".join("?" * len(signal_names))
            query = f"SELECT date, signal_name, {col} FROM ic_panel WHERE signal_name IN ({placeholders}) ORDER BY date"
            params = signal_names
        else:
            query = f"SELECT date, signal_name, {col} FROM ic_panel ORDER BY date"
            params = []

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            return pd.DataFrame()

        return df.pivot(index="date", columns="signal_name", values=col)

    def ic_correlation_matrix(
        self,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Cross-signal IC correlation matrix."""
        panel = self.load_ic_matrix(horizon=horizon)
        if panel.empty:
            return pd.DataFrame()
        return panel.corr(method="spearman")

    def top_signals_by_mean_ic(
        self,
        horizon: int = 1,
        n: int = 10,
    ) -> pd.Series:
        """Return top-N signals ranked by mean IC."""
        panel = self.load_ic_matrix(horizon=horizon)
        if panel.empty:
            return pd.Series(dtype=float)
        return panel.mean().sort_values(ascending=False).head(n)

    def icir_table(
        self,
        horizon: int = 1,
    ) -> pd.DataFrame:
        """Return DataFrame of signal -> (mean_ic, std_ic, icir, n_obs)."""
        panel = self.load_ic_matrix(horizon=horizon)
        if panel.empty:
            return pd.DataFrame()

        rows = []
        for col in panel.columns:
            s = panel[col].dropna()
            if len(s) < 5:
                continue
            mean_ic = float(s.mean())
            std_ic = float(s.std(ddof=1))
            icir = mean_ic / std_ic if std_ic > 1e-10 else float("nan")
            rows.append({
                "signal_name": col,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": icir,
                "n_obs": len(s),
            })
        return pd.DataFrame(rows).sort_values("icir", ascending=False)


# ---------------------------------------------------------------------------
# SignalDriftMonitor
# ---------------------------------------------------------------------------


class SignalDriftMonitor:
    """
    Monitors signals for statistical drift over time.

    Detects:
    - Mean shift (signal distribution has shifted)
    - Variance change (signal became more or less noisy)
    - IC sign flip (as in ICCalculator.detect_sign_flip)
    """

    def __init__(self, window: int = 60, alpha: float = 0.05) -> None:
        self.window = window
        self.alpha = alpha
        self._ic_calc = ICCalculator()

    def check_mean_shift(
        self,
        signal_ts: pd.Series,
        reference_window: int = 120,
        test_window: int = 30,
    ) -> Dict:
        """
        Two-sample t-test comparing recent vs reference window means.
        Returns dict with t_stat, p_value, detected (bool).
        """
        s = signal_ts.dropna()
        if len(s) < reference_window + test_window:
            return {"t_stat": float("nan"), "p_value": float("nan"), "detected": False}

        reference = s.iloc[-(reference_window + test_window) : -test_window]
        recent = s.iloc[-test_window:]

        t_stat, p_value = stats.ttest_ind(reference, recent, equal_var=False)
        detected = float(p_value) < self.alpha

        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "detected": bool(detected),
            "reference_mean": float(reference.mean()),
            "recent_mean": float(recent.mean()),
        }

    def check_variance_change(
        self,
        signal_ts: pd.Series,
        reference_window: int = 120,
        test_window: int = 30,
    ) -> Dict:
        """
        Levene's test for variance change between reference and recent windows.
        """
        s = signal_ts.dropna()
        if len(s) < reference_window + test_window:
            return {"stat": float("nan"), "p_value": float("nan"), "detected": False}

        reference = s.iloc[-(reference_window + test_window) : -test_window]
        recent = s.iloc[-test_window:]

        stat, p_value = stats.levene(reference, recent)
        detected = float(p_value) < self.alpha

        return {
            "stat": float(stat),
            "p_value": float(p_value),
            "detected": bool(detected),
            "reference_var": float(reference.var()),
            "recent_var": float(recent.var()),
        }

    def check_ic_sign_flip(
        self,
        ic_ts: pd.Series,
        window: int = 30,
        threshold: float = 0.1,
    ) -> bool:
        """Delegate to ICCalculator.detect_sign_flip."""
        return self._ic_calc.detect_sign_flip(ic_ts, window=window, threshold=threshold)

    def full_drift_report(
        self,
        signal_ts: pd.Series,
        ic_ts: Optional[pd.Series] = None,
    ) -> Dict:
        """Run all drift checks and return a summary dict."""
        mean_check = self.check_mean_shift(signal_ts)
        var_check = self.check_variance_change(signal_ts)
        sign_flip = False
        if ic_ts is not None and not ic_ts.empty:
            sign_flip = self.check_ic_sign_flip(ic_ts)

        return {
            "mean_shift": mean_check,
            "variance_change": var_check,
            "ic_sign_flip": sign_flip,
            "any_drift": bool(mean_check.get("detected") or var_check.get("detected") or sign_flip),
        }


# ---------------------------------------------------------------------------
# SignalBacktester
# ---------------------------------------------------------------------------


class SignalBacktester:
    """
    Lightweight signal-level backtester.

    Computes quintile returns, information ratio, hit rate, and
    turnover for a single signal.
    """

    def __init__(
        self,
        transaction_cost: float = 0.0005,
        n_quantiles: int = 5,
    ) -> None:
        self.transaction_cost = transaction_cost
        self.n_quantiles = n_quantiles

    def quintile_returns(
        self,
        signal: pd.Series,
        forward_return: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute mean forward return per signal quintile.
        Returns DataFrame with columns: quantile, mean_return, count.
        """
        combined = pd.concat([signal.rename("sig"), forward_return.rename("fwd")], axis=1).dropna()
        combined["quintile"] = pd.qcut(combined["sig"], self.n_quantiles, labels=False, duplicates="drop")
        grouped = combined.groupby("quintile")["fwd"].agg(["mean", "count"]).reset_index()
        grouped.columns = ["quintile", "mean_return", "count"]
        return grouped

    def hit_rate(
        self,
        signal: pd.Series,
        forward_return: pd.Series,
    ) -> float:
        """Fraction of periods where signal direction matches return direction."""
        combined = pd.concat([signal.rename("sig"), forward_return.rename("fwd")], axis=1).dropna()
        if combined.empty:
            return float("nan")
        same_sign = (np.sign(combined["sig"]) == np.sign(combined["fwd"]))
        return float(same_sign.mean())

    def annualized_ic(
        self,
        signal: pd.Series,
        forward_return: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """Annualized information ratio from IC and ICIR."""
        calc = ICCalculator()
        ic_ts = calc.rolling_ic(signal, forward_return, window=60)
        icir = calc.compute_icir(ic_ts)
        return icir * math.sqrt(periods_per_year / 60)

    def turnover(self, signal: pd.Series, normalize: bool = True) -> float:
        """
        Mean absolute daily change in signal rank.
        High turnover = expensive to trade.
        """
        ranked = signal.rank(pct=True)
        daily_change = ranked.diff().abs()
        if normalize:
            return float(daily_change.mean())
        return float(daily_change.sum())

    def net_ic(
        self,
        signal: pd.Series,
        forward_return: pd.Series,
        holding_period: int = 1,
    ) -> float:
        """IC adjusted for transaction costs."""
        calc = ICCalculator()
        gross_ic = calc.compute_ic(signal, forward_return)
        cost_drag = self.transaction_cost * 2 / holding_period
        return float(gross_ic) - cost_drag if math.isfinite(gross_ic) else float("nan")

    def full_backtest(
        self,
        signal: pd.Series,
        prices: pd.Series,
        holding_period: int = 1,
    ) -> Dict:
        """Run full signal backtest and return summary stats."""
        log_prices = np.log(prices.replace(0, float("nan")))
        fwd = log_prices.shift(-holding_period) - log_prices

        calc = ICCalculator()
        ic_1d = calc.compute_ic(signal, fwd)
        ic_ts = calc.rolling_ic(signal, fwd, window=60)
        icir = calc.compute_icir(ic_ts)
        hr = self.hit_rate(signal, fwd)
        to = self.turnover(signal)
        net = self.net_ic(signal, fwd, holding_period)
        quint = self.quintile_returns(signal, fwd)

        return {
            "ic": ic_1d,
            "icir": icir,
            "hit_rate": hr,
            "turnover": to,
            "net_ic": net,
            "quintile_returns": quint,
        }
