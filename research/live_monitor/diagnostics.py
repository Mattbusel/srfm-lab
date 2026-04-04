"""
diagnostics.py — Live Trading Diagnostics
==========================================

Deep diagnostic tools for analysing live trading quality:
  - Order failure classification
  - Signal quality (IC, hit rate)
  - Regime exposure analysis
  - Ensemble model usage diagnostics
  - Position concentration (HHI)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FailureReport:
    """Order failure analysis."""
    n_total_failures: int
    failure_counts_by_reason: dict[str, int]
    failure_rate_by_reason: dict[str, float]
    notional_violation_count: int
    notional_violation_frequency: float      # fraction of all orders
    most_common_reason: str
    recent_failures: list[dict[str, Any]]    # last 20
    first_failure_at: datetime | None
    last_failure_at: datetime | None


@dataclass
class SignalDiagnostic:
    """BH signal quality in live trading."""
    n_signal_trades: int
    information_coefficient: float           # Pearson corr(signal, return)
    ic_p_value: float
    hit_rate: float                          # fraction of correct direction predictions
    mean_return_with_signal: float           # bps
    mean_return_without_signal: float        # bps
    signal_decay_minutes: float              # how long IC stays positive
    is_predictive: bool                      # IC > 0 and p < 0.05
    by_sym: dict[str, float]                 # sym → IC


@dataclass
class RegimeDiagnostic:
    """Regime exposure analysis."""
    regime_trade_counts: dict[str, int]
    regime_pnl: dict[str, float]
    regime_win_rates: dict[str, float]
    current_regime: str
    high_vol_fraction: float                 # fraction of trades in HIGH_VOL
    high_vol_pnl_per_trade: float
    low_vol_pnl_per_trade: float
    is_over_exposed_high_vol: bool           # > 50% in HIGH_VOL
    recommendations: list[str]


@dataclass
class EnsembleDiagnostic:
    """Ensemble model usage and performance."""
    n_ensemble_trades: int
    n_single_model_trades: int
    ensemble_win_rate: float
    single_model_win_rate: float
    ensemble_mean_pnl: float
    single_model_mean_pnl: float
    best_threshold: float                    # confidence threshold maximising Sharpe
    ensemble_sharpe: float
    single_model_sharpe: float
    ensemble_is_better: bool
    threshold_analysis: pd.DataFrame         # threshold vs Sharpe


@dataclass
class ConcentrationDiagnostic:
    """Portfolio concentration analysis."""
    current_hhi: float                       # Herfindahl-Hirschman Index (0–1)
    max_hhi: float
    mean_hhi: float
    current_max_position_frac: float
    over_concentrated_at: list[datetime]
    concentration_by_sym: dict[str, float]   # sym → current fraction
    alerts: list[str]
    delta_max_frac: float                    # configured maximum


# ---------------------------------------------------------------------------
# Diagnostics class
# ---------------------------------------------------------------------------

class LiveDiagnostics:
    """
    Deep diagnostics for the live Alpaca trader.

    Parameters
    ----------
    db_path : str | Path
        Path to live_trades.db SQLite.
    delta_max_frac : float
        Maximum allowed single-position fraction of portfolio.
    """

    def __init__(
        self,
        db_path: str | Path,
        delta_max_frac: float = 0.20,
    ) -> None:
        self.db_path = Path(db_path)
        self.delta_max_frac = delta_max_frac

    # -----------------------------------------------------------------------
    # DB helpers
    # -----------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        try:
            conn = self._connect()
            df = pd.read_sql_query(sql, conn, params=params)
            conn.close()
            return df
        except Exception as exc:
            logger.error("Query failed: %s — %s", sql[:80], exc)
            return pd.DataFrame()

    def _load_trades(self, days: int = 90) -> pd.DataFrame:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self._query(
            "SELECT * FROM trades WHERE DATE(fill_time) >= ? ORDER BY fill_time",
            (cutoff,),
        )

    # -----------------------------------------------------------------------
    # Order failure diagnostics
    # -----------------------------------------------------------------------

    def diagnose_order_failures(
        self,
        live_db_path: str | Path | None = None,
    ) -> FailureReport:
        """
        Parse order failure logs and classify by reason.

        Failure reasons recognised:
          - notional_limit: "exceeded maximum notional" / "$200K"
          - insufficient_funds: "insufficient buying power"
          - api_error: HTTP 4xx/5xx from Alpaca
          - market_closed: order submitted outside hours
          - unknown: anything else

        Parameters
        ----------
        live_db_path : str | Path, optional
            Override DB path.

        Returns
        -------
        FailureReport
        """
        db = Path(live_db_path) if live_db_path else self.db_path
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row

        try:
            fail_df = pd.read_sql_query(
                "SELECT * FROM order_failures ORDER BY ts DESC", conn
            )
            total_df = pd.read_sql_query("SELECT COUNT(*) as n FROM trades", conn)
        except Exception as exc:
            logger.warning("order_failures table missing or error: %s", exc)
            conn.close()
            return FailureReport(
                n_total_failures=0,
                failure_counts_by_reason={},
                failure_rate_by_reason={},
                notional_violation_count=0,
                notional_violation_frequency=0.0,
                most_common_reason="n/a",
                recent_failures=[],
                first_failure_at=None,
                last_failure_at=None,
            )
        finally:
            conn.close()

        n_total_orders = int(total_df.iloc[0]["n"]) if not total_df.empty else 1
        n_failures = len(fail_df)

        # Classify each failure
        reason_map: dict[str, str] = {}

        def classify(reason: str) -> str:
            r = str(reason).lower()
            if "200k" in r or "200,000" in r or "notional" in r or "maximum order" in r:
                return "notional_limit"
            elif "buying power" in r or "insufficient" in r or "margin" in r:
                return "insufficient_funds"
            elif "market closed" in r or "outside" in r or "after hours" in r:
                return "market_closed"
            elif "timeout" in r or "connection" in r or "network" in r:
                return "connectivity"
            elif any(c in r for c in ("400", "403", "422", "429", "500", "502", "503")):
                return "api_error"
            else:
                return "unknown"

        if not fail_df.empty and "reason" in fail_df.columns:
            fail_df["classified_reason"] = fail_df["reason"].apply(classify)
            counts: dict[str, int] = fail_df["classified_reason"].value_counts().to_dict()
        else:
            counts = {}

        notional_count = counts.get("notional_limit", 0)
        total_ops = n_total_orders + n_failures

        failure_rate_by_reason = {
            reason: cnt / max(total_ops, 1)
            for reason, cnt in counts.items()
        }

        most_common = max(counts, key=counts.get) if counts else "n/a"

        recent_failures: list[dict[str, Any]] = []
        if not fail_df.empty:
            recent_failures = fail_df.head(20).to_dict(orient="records")

        first_at: datetime | None = None
        last_at: datetime | None = None
        if not fail_df.empty and "ts" in fail_df.columns:
            ts_series = pd.to_datetime(fail_df["ts"])
            first_at = ts_series.min().to_pydatetime()
            last_at = ts_series.max().to_pydatetime()

        return FailureReport(
            n_total_failures=n_failures,
            failure_counts_by_reason=counts,
            failure_rate_by_reason=failure_rate_by_reason,
            notional_violation_count=notional_count,
            notional_violation_frequency=notional_count / max(total_ops, 1),
            most_common_reason=most_common,
            recent_failures=recent_failures,
            first_failure_at=first_at,
            last_failure_at=last_at,
        )

    def notional_violation_frequency(self) -> float:
        """
        How often does the $200K notional limit cause order failures?

        Returns
        -------
        float
            Fraction of all order attempts that fail due to notional limit.
        """
        report = self.diagnose_order_failures()
        return report.notional_violation_frequency

    # -----------------------------------------------------------------------
    # Signal quality
    # -----------------------------------------------------------------------

    def diagnose_signal_quality(
        self,
        live_trades: pd.DataFrame | None = None,
        signal_col: str = "bh_signal",
        return_col: str = "forward_return",
        days: int = 90,
    ) -> SignalDiagnostic:
        """
        Measure predictive quality of the BH signal in live trading.

        Parameters
        ----------
        live_trades : pd.DataFrame, optional
            Pre-loaded trades with 'bh_signal' and 'forward_return' columns.
        signal_col : str
        return_col : str
        days : int

        Returns
        -------
        SignalDiagnostic
        """
        from scipy import stats as scipy_stats

        if live_trades is None:
            live_trades = self._load_trades(days)

        if live_trades.empty:
            return self._empty_signal_diagnostic()

        # Need signal and return columns
        available_signal = signal_col in live_trades.columns
        available_return = return_col in live_trades.columns

        if not available_signal or not available_return:
            # Try to compute forward return from fill prices
            if "fill_price" in live_trades.columns:
                live_trades = live_trades.sort_values("fill_time")
                live_trades["forward_return"] = (
                    live_trades.groupby("sym")["fill_price"]
                    .pct_change()
                    .shift(-1)
                )
                return_col = "forward_return"
            else:
                logger.warning("Cannot compute signal quality: missing signal/return columns")
                return self._empty_signal_diagnostic()

        if signal_col not in live_trades.columns:
            logger.warning("No signal column '%s' found", signal_col)
            return self._empty_signal_diagnostic()

        df = live_trades.dropna(subset=[signal_col, return_col]).copy()
        df[signal_col] = df[signal_col].astype(float)
        df[return_col] = df[return_col].astype(float)

        if len(df) < 10:
            logger.warning("Too few signal observations: %d", len(df))
            return self._empty_signal_diagnostic()

        sig = df[signal_col].values
        ret = df[return_col].values

        # Information coefficient
        ic, p_val = scipy_stats.pearsonr(sig, ret)

        # Hit rate: sign(signal) == sign(return)
        correct = np.sign(sig) == np.sign(ret)
        hit_rate = float(correct.mean())

        # Mean return with vs without signal
        signal_mask = df[signal_col].abs() > df[signal_col].abs().median()
        mean_with = float(df.loc[signal_mask, return_col].mean() * 10_000)   # to bps
        mean_without = float(df.loc[~signal_mask, return_col].mean() * 10_000)

        # Signal decay: IC over time bins
        signal_decay_minutes = self._estimate_signal_decay(df, signal_col, return_col)

        # Per-symbol IC
        by_sym: dict[str, float] = {}
        if "sym" in df.columns:
            for sym, grp in df.groupby("sym"):
                if len(grp) >= 5:
                    ic_sym, _ = scipy_stats.pearsonr(
                        grp[signal_col].values, grp[return_col].values
                    )
                    by_sym[str(sym)] = float(ic_sym)

        return SignalDiagnostic(
            n_signal_trades=len(df),
            information_coefficient=float(ic),
            ic_p_value=float(p_val),
            hit_rate=hit_rate,
            mean_return_with_signal=mean_with,
            mean_return_without_signal=mean_without,
            signal_decay_minutes=signal_decay_minutes,
            is_predictive=float(ic) > 0 and float(p_val) < 0.05,
            by_sym=by_sym,
        )

    @staticmethod
    def _empty_signal_diagnostic() -> SignalDiagnostic:
        return SignalDiagnostic(
            n_signal_trades=0,
            information_coefficient=0.0,
            ic_p_value=1.0,
            hit_rate=0.5,
            mean_return_with_signal=0.0,
            mean_return_without_signal=0.0,
            signal_decay_minutes=0.0,
            is_predictive=False,
            by_sym={},
        )

    @staticmethod
    def _estimate_signal_decay(
        df: pd.DataFrame,
        signal_col: str,
        return_col: str,
    ) -> float:
        """
        Estimate how long the signal remains informative (in minutes).

        We look at IC across forward return horizons.
        (Simplified: returns a rough estimate based on autocorrelation.)
        """
        if "fill_time" not in df.columns:
            return 60.0  # default

        from scipy import stats as scipy_stats

        df = df.copy()
        df["fill_time"] = pd.to_datetime(df["fill_time"])
        df = df.sort_values("fill_time")

        # Estimate signal autocorrelation decay
        sig = df[signal_col].values.astype(float)
        if len(sig) < 20:
            return 30.0

        # Find lag at which autocorrelation drops below 0.1
        max_lag = min(len(sig) - 1, 50)
        for lag in range(1, max_lag):
            if len(sig) > lag:
                corr = float(np.corrcoef(sig[:-lag], sig[lag:])[0, 1])
                if abs(corr) < 0.1:
                    # Approximate: each observation ≈ 5 min apart
                    return float(lag * 5)

        return float(max_lag * 5)

    # -----------------------------------------------------------------------
    # Regime exposure
    # -----------------------------------------------------------------------

    def diagnose_regime_exposure(
        self,
        live_trades: pd.DataFrame | None = None,
        regime_series: pd.Series | None = None,
        regime_col: str = "regime",
        days: int = 90,
    ) -> RegimeDiagnostic:
        """
        Analyse regime exposure and performance.

        Parameters
        ----------
        live_trades : pd.DataFrame, optional
        regime_series : pd.Series, optional
            Time-indexed Series of regime labels.
        regime_col : str
            Column in live_trades containing regime label.
        days : int

        Returns
        -------
        RegimeDiagnostic
        """
        if live_trades is None:
            live_trades = self._load_trades(days)

        if live_trades.empty:
            return self._empty_regime_diagnostic()

        df = live_trades.copy()
        if "fill_time" in df.columns:
            df["fill_time"] = pd.to_datetime(df["fill_time"])

        # Join regime if provided as external series
        if regime_series is not None and regime_col not in df.columns:
            regime_df = regime_series.rename("regime").to_frame()
            regime_df.index = pd.to_datetime(regime_df.index)
            if "fill_time" in df.columns:
                df = df.set_index("fill_time")
                df = df.merge(
                    regime_df,
                    left_index=True,
                    right_index=True,
                    how="left",
                ).reset_index()
                df.rename(columns={"index": "fill_time"}, inplace=True)
                regime_col = "regime"

        if regime_col not in df.columns:
            logger.warning("No regime column found — using 'UNKNOWN'")
            df[regime_col] = "UNKNOWN"

        # Current regime (latest trade)
        current_regime = "UNKNOWN"
        if not df.empty:
            current_regime = str(df[regime_col].iloc[-1])

        # Stats by regime
        regime_counts: dict[str, int] = {}
        regime_pnl: dict[str, float] = {}
        regime_win_rates: dict[str, float] = {}

        if "pnl" not in df.columns:
            df["pnl"] = 0.0

        for regime, grp in df.groupby(regime_col):
            regime_counts[str(regime)] = len(grp)
            regime_pnl[str(regime)] = float(grp["pnl"].sum())
            regime_win_rates[str(regime)] = float((grp["pnl"] > 0).mean())

        total_trades = sum(regime_counts.values())
        high_vol_count = sum(
            v for k, v in regime_counts.items()
            if "HIGH_VOL" in k.upper() or "VOLATILE" in k.upper()
        )
        high_vol_frac = high_vol_count / max(total_trades, 1)

        high_vol_trades = df[df[regime_col].str.upper().str.contains("HIGH_VOL|VOLATILE", na=False)]
        low_vol_trades = df[~df[regime_col].str.upper().str.contains("HIGH_VOL|VOLATILE", na=False)]

        high_vol_ppt = float(high_vol_trades["pnl"].mean()) if not high_vol_trades.empty else 0.0
        low_vol_ppt = float(low_vol_trades["pnl"].mean()) if not low_vol_trades.empty else 0.0

        recommendations: list[str] = []
        if high_vol_frac > 0.5:
            recommendations.append(
                f"Over-exposed to HIGH_VOL regime ({high_vol_frac*100:.0f}% of trades). "
                "Consider reducing position size during high-vol periods."
            )
        if high_vol_ppt < 0 and low_vol_ppt > 0:
            recommendations.append(
                "HIGH_VOL regime is loss-making while LOW_VOL is profitable. "
                "Consider regime filter on signals."
            )

        return RegimeDiagnostic(
            regime_trade_counts=regime_counts,
            regime_pnl=regime_pnl,
            regime_win_rates=regime_win_rates,
            current_regime=current_regime,
            high_vol_fraction=high_vol_frac,
            high_vol_pnl_per_trade=high_vol_ppt,
            low_vol_pnl_per_trade=low_vol_ppt,
            is_over_exposed_high_vol=high_vol_frac > 0.5,
            recommendations=recommendations,
        )

    @staticmethod
    def _empty_regime_diagnostic() -> RegimeDiagnostic:
        return RegimeDiagnostic(
            regime_trade_counts={},
            regime_pnl={},
            regime_win_rates={},
            current_regime="UNKNOWN",
            high_vol_fraction=0.0,
            high_vol_pnl_per_trade=0.0,
            low_vol_pnl_per_trade=0.0,
            is_over_exposed_high_vol=False,
            recommendations=["Insufficient data"],
        )

    # -----------------------------------------------------------------------
    # Ensemble usage
    # -----------------------------------------------------------------------

    def diagnose_ensemble_usage(
        self,
        live_trades: pd.DataFrame | None = None,
        ensemble_col: str = "used_ensemble",
        confidence_col: str = "ensemble_confidence",
        days: int = 90,
    ) -> EnsembleDiagnostic:
        """
        Analyse whether the ensemble signal improves performance.

        Parameters
        ----------
        live_trades : pd.DataFrame, optional
        ensemble_col : str
            Boolean column: was ensemble used?
        confidence_col : str
            Float column: ensemble confidence score.
        days : int

        Returns
        -------
        EnsembleDiagnostic
        """
        if live_trades is None:
            live_trades = self._load_trades(days)

        df = live_trades.copy()
        if df.empty or ensemble_col not in df.columns:
            return self._empty_ensemble_diagnostic()

        if "pnl" not in df.columns:
            df["pnl"] = 0.0

        ensemble_df = df[df[ensemble_col].astype(bool)]
        single_df = df[~df[ensemble_col].astype(bool)]

        def _sharpe(pnls: pd.Series) -> float:
            if len(pnls) < 2 or pnls.std() == 0:
                return 0.0
            return float(pnls.mean() / pnls.std() * math.sqrt(252))

        ens_wr = float((ensemble_df["pnl"] > 0).mean()) if len(ensemble_df) > 0 else 0.0
        sgl_wr = float((single_df["pnl"] > 0).mean()) if len(single_df) > 0 else 0.0
        ens_mean = float(ensemble_df["pnl"].mean()) if len(ensemble_df) > 0 else 0.0
        sgl_mean = float(single_df["pnl"].mean()) if len(single_df) > 0 else 0.0
        ens_sharpe = _sharpe(ensemble_df["pnl"])
        sgl_sharpe = _sharpe(single_df["pnl"])

        # Find best confidence threshold by scanning values
        best_threshold = 0.5
        threshold_rows = []

        if confidence_col in df.columns:
            thresholds = np.arange(0.3, 0.9, 0.05)
            for thresh in thresholds:
                high_conf = df[df[confidence_col] >= thresh]
                if len(high_conf) >= 5:
                    s = _sharpe(high_conf["pnl"])
                    threshold_rows.append({"threshold": thresh, "n_trades": len(high_conf), "sharpe": s})
            if threshold_rows:
                thresh_df = pd.DataFrame(threshold_rows)
                best_row = thresh_df.loc[thresh_df["sharpe"].idxmax()]
                best_threshold = float(best_row["threshold"])
        else:
            thresh_df = pd.DataFrame(columns=["threshold", "n_trades", "sharpe"])

        return EnsembleDiagnostic(
            n_ensemble_trades=len(ensemble_df),
            n_single_model_trades=len(single_df),
            ensemble_win_rate=ens_wr,
            single_model_win_rate=sgl_wr,
            ensemble_mean_pnl=ens_mean,
            single_model_mean_pnl=sgl_mean,
            best_threshold=best_threshold,
            ensemble_sharpe=ens_sharpe,
            single_model_sharpe=sgl_sharpe,
            ensemble_is_better=ens_sharpe > sgl_sharpe,
            threshold_analysis=thresh_df if threshold_rows else pd.DataFrame(),
        )

    @staticmethod
    def _empty_ensemble_diagnostic() -> EnsembleDiagnostic:
        return EnsembleDiagnostic(
            n_ensemble_trades=0,
            n_single_model_trades=0,
            ensemble_win_rate=0.0,
            single_model_win_rate=0.0,
            ensemble_mean_pnl=0.0,
            single_model_mean_pnl=0.0,
            best_threshold=0.5,
            ensemble_sharpe=0.0,
            single_model_sharpe=0.0,
            ensemble_is_better=False,
            threshold_analysis=pd.DataFrame(),
        )

    # -----------------------------------------------------------------------
    # Concentration
    # -----------------------------------------------------------------------

    def diagnose_concentration(
        self,
        live_trades: pd.DataFrame | None = None,
        days: int = 30,
    ) -> ConcentrationDiagnostic:
        """
        Analyse position concentration using HHI.

        HHI = sum of squared position weight fractions.
        HHI = 1.0 means 100% in one asset (fully concentrated).
        HHI = 1/N means perfectly equal distribution.

        Parameters
        ----------
        live_trades : pd.DataFrame, optional
        days : int

        Returns
        -------
        ConcentrationDiagnostic
        """
        if live_trades is None:
            live_trades = self._load_trades(days)

        df = live_trades.copy()
        alerts: list[str] = []
        over_concentrated_at: list[datetime] = []

        if df.empty or "notional" not in df.columns:
            # Try to compute notional from fills
            if "fill_price" in df.columns and "qty" in df.columns:
                df["notional"] = df["fill_price"].astype(float) * df["qty"].abs().astype(float)
            else:
                return ConcentrationDiagnostic(
                    current_hhi=0.0,
                    max_hhi=0.0,
                    mean_hhi=0.0,
                    current_max_position_frac=0.0,
                    over_concentrated_at=[],
                    concentration_by_sym={},
                    alerts=["Insufficient data"],
                    delta_max_frac=self.delta_max_frac,
                )

        # Current concentration (using latest snapshot)
        if "fill_time" in df.columns:
            df["fill_time"] = pd.to_datetime(df["fill_time"])
            # Use last 1-day snapshot as proxy for current positions
            cutoff = df["fill_time"].max() - timedelta(days=1)
            current_df = df[df["fill_time"] >= cutoff]
        else:
            current_df = df

        current_hhi = self._compute_hhi(current_df)
        conc_by_sym = self._concentration_by_sym(current_df)

        # Check per-symbol fraction
        for sym, frac in conc_by_sym.items():
            if frac > self.delta_max_frac:
                alerts.append(
                    f"CRITICAL: {sym} concentration {frac*100:.1f}% > DELTA_MAX_FRAC "
                    f"{self.delta_max_frac*100:.0f}%"
                )

        # Historical HHI — daily snapshots
        hhi_series: list[float] = []
        if "fill_time" in df.columns:
            for date, daily_df in df.groupby(df["fill_time"].dt.date):
                hhi = self._compute_hhi(daily_df)
                hhi_series.append(hhi)
                if hhi > 0.5:  # > 50% concentration
                    over_concentrated_at.append(
                        datetime.combine(date, datetime.min.time())
                    )

        max_hhi = max(hhi_series) if hhi_series else current_hhi
        mean_hhi = float(np.mean(hhi_series)) if hhi_series else current_hhi
        max_frac = max(conc_by_sym.values()) if conc_by_sym else 0.0

        return ConcentrationDiagnostic(
            current_hhi=current_hhi,
            max_hhi=max_hhi,
            mean_hhi=mean_hhi,
            current_max_position_frac=max_frac,
            over_concentrated_at=over_concentrated_at,
            concentration_by_sym=conc_by_sym,
            alerts=alerts if alerts else ["Concentration within limits"],
            delta_max_frac=self.delta_max_frac,
        )

    @staticmethod
    def _compute_hhi(df: pd.DataFrame) -> float:
        """Compute Herfindahl-Hirschman Index from notional column."""
        if df.empty or "notional" not in df.columns:
            return 0.0
        notional = df.groupby("sym")["notional"].sum() if "sym" in df.columns else df["notional"]
        total = float(notional.sum())
        if total <= 0:
            return 0.0
        weights = notional / total
        return float((weights ** 2).sum())

    @staticmethod
    def _concentration_by_sym(df: pd.DataFrame) -> dict[str, float]:
        """Compute per-symbol fraction of total notional."""
        if df.empty or "notional" not in df.columns or "sym" not in df.columns:
            return {}
        by_sym = df.groupby("sym")["notional"].sum()
        total = float(by_sym.sum())
        if total <= 0:
            return {}
        return {str(sym): float(n / total) for sym, n in by_sym.items()}

    # -----------------------------------------------------------------------
    # Summary report (all diagnostics)
    # -----------------------------------------------------------------------

    def full_diagnostic_report(self, days: int = 90) -> dict[str, Any]:
        """
        Run all diagnostics and return a structured report.

        Returns
        -------
        dict with keys: failures, signal, regime, ensemble, concentration
        """
        live_trades = self._load_trades(days)

        report: dict[str, Any] = {}

        try:
            fail = self.diagnose_order_failures()
            report["failures"] = {
                "n_total": fail.n_total_failures,
                "by_reason": fail.failure_counts_by_reason,
                "notional_violation_rate": fail.notional_violation_frequency,
                "most_common": fail.most_common_reason,
            }
        except Exception as exc:
            report["failures"] = {"error": str(exc)}

        try:
            sig = self.diagnose_signal_quality(live_trades)
            report["signal"] = {
                "ic": sig.information_coefficient,
                "ic_p_value": sig.ic_p_value,
                "hit_rate": sig.hit_rate,
                "is_predictive": sig.is_predictive,
                "n_trades": sig.n_signal_trades,
            }
        except Exception as exc:
            report["signal"] = {"error": str(exc)}

        try:
            reg = self.diagnose_regime_exposure(live_trades)
            report["regime"] = {
                "current": reg.current_regime,
                "high_vol_fraction": reg.high_vol_fraction,
                "is_over_exposed": reg.is_over_exposed_high_vol,
                "recommendations": reg.recommendations,
            }
        except Exception as exc:
            report["regime"] = {"error": str(exc)}

        try:
            ens = self.diagnose_ensemble_usage(live_trades)
            report["ensemble"] = {
                "ensemble_is_better": ens.ensemble_is_better,
                "ensemble_sharpe": ens.ensemble_sharpe,
                "single_model_sharpe": ens.single_model_sharpe,
                "best_threshold": ens.best_threshold,
            }
        except Exception as exc:
            report["ensemble"] = {"error": str(exc)}

        try:
            conc = self.diagnose_concentration(live_trades)
            report["concentration"] = {
                "current_hhi": conc.current_hhi,
                "max_position_frac": conc.current_max_position_frac,
                "alerts": conc.alerts,
            }
        except Exception as exc:
            report["concentration"] = {"error": str(exc)}

        return report
