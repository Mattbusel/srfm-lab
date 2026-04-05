"""
live-feedback/attribution.py
============================
Trade attribution engine.

Every live trade is attributed to the hypothesis (or signal, regime, or set
of signals) most likely responsible for the position.  Attribution is used
downstream by the Bayesian scorer and the drift detector to correctly credit
or debit each hypothesis with live P&L.

Attribution rules (applied in priority order)
---------------------------------------------
1. **Exact symbol + side** within the hypothesis's active window
   (``created_at`` → now or until next retest).
2. **Parameter match**: the trade's ``params_json`` overlaps with the
   hypothesis's ``parameters`` JSON by >= PARAM_MATCH_THRESHOLD.
3. **Temporal proximity**: trade within N bars (``TEMPORAL_WINDOW_HOURS``)
   of the hypothesis adoption timestamp.

If multiple rules match, the highest-confidence match wins.
Unattributed trades are stored with ``hypothesis_id = NULL``.

The ``trade_attributions`` table schema lives in ``schema_extension.sql``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_MATCH_THRESHOLD = 0.40          # fraction of hypothesis params found in trade params
TEMPORAL_WINDOW_HOURS = 4             # max hours between adoption and trade entry
CONFIDENCE_EXACT = 0.95               # rule 1
CONFIDENCE_PARAM = 0.70               # rule 2
CONFIDENCE_TEMPORAL = 0.50            # rule 3
MIN_CONFIDENCE = 0.30                 # below this we treat as unattributed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Attribution:
    """Result of attributing a single trade to a cause."""

    trade_id: str
    hypothesis_id: int | None
    signal_name: str | None
    regime: str | None
    confidence: float
    rule_used: str        # 'exact_symbol', 'param_match', 'temporal', 'unattributed'
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TradeAttributor
# ---------------------------------------------------------------------------

class TradeAttributor:
    """
    Maps live trades to the hypotheses (and signals / regimes) responsible
    for them.

    Parameters
    ----------
    iae_conn : sqlite3.Connection
        Open connection to ``idea_engine.db``.
    temporal_window_hours : float
        Maximum hours between hypothesis adoption and trade entry for a
        temporal-proximity match.
    param_match_threshold : float
        Minimum fraction of hypothesis params that must appear in the
        trade's ``params_json`` for a parameter-match attribution.
    """

    def __init__(
        self,
        iae_conn: sqlite3.Connection,
        *,
        temporal_window_hours: float = TEMPORAL_WINDOW_HOURS,
        param_match_threshold: float = PARAM_MATCH_THRESHOLD,
    ) -> None:
        self._conn = iae_conn
        self._temporal_window = timedelta(hours=temporal_window_hours)
        self._param_threshold = param_match_threshold

    # ------------------------------------------------------------------
    # Primary entry-point
    # ------------------------------------------------------------------

    def build_attribution_map(
        self,
        hypotheses_df: pd.DataFrame,
        trades_df: pd.DataFrame,
    ) -> dict[str, int | None]:
        """
        Attribute every trade in ``trades_df`` to a hypothesis in
        ``hypotheses_df``.

        Parameters
        ----------
        hypotheses_df : pd.DataFrame
            Active hypotheses with columns: id, type, parameters, created_at.
        trades_df : pd.DataFrame
            Live trades with columns: trade_id, symbol, side, opened_at,
            params_json.

        Returns
        -------
        dict  {trade_id: hypothesis_id | None}
        """
        if trades_df.empty or hypotheses_df.empty:
            return {tid: None for tid in trades_df.get("trade_id", pd.Series()).tolist()}

        # Pre-parse hypothesis parameters once
        hyp_params: dict[int, dict[str, Any]] = {}
        hyp_adoption: dict[int, datetime] = {}
        for _, hrow in hypotheses_df.iterrows():
            hid = int(hrow["id"])
            raw = hrow.get("parameters", "{}")
            hyp_params[hid] = json.loads(raw) if isinstance(raw, str) else (raw or {})
            ts_str = hrow.get("created_at", "2000-01-01T00:00:00+00:00")
            hyp_adoption[hid] = _parse_utc(ts_str)

        result: dict[str, int | None] = {}
        for _, trade in trades_df.iterrows():
            attribution = self._attribute_single_trade(
                trade, hypotheses_df, hyp_params, hyp_adoption
            )
            result[str(trade["trade_id"])] = attribution.hypothesis_id

        return result

    # ------------------------------------------------------------------
    # P&L Attribution by dimension
    # ------------------------------------------------------------------

    def pnl_attribution_by_hypothesis(
        self,
        trades_df: pd.DataFrame,
        attribution_map: dict[str, int | None],
    ) -> pd.DataFrame:
        """
        Aggregate P&L by hypothesis_id.

        Parameters
        ----------
        trades_df       : pd.DataFrame  — must have columns: trade_id, pnl, pnl_pct
        attribution_map : dict          — trade_id → hypothesis_id

        Returns
        -------
        pd.DataFrame with columns:
            hypothesis_id, total_pnl, avg_pnl, trade_count, win_rate
        """
        if trades_df.empty:
            return pd.DataFrame(
                columns=["hypothesis_id", "total_pnl", "avg_pnl", "trade_count", "win_rate"]
            )

        df = trades_df.copy()
        df["hypothesis_id"] = df["trade_id"].astype(str).map(attribution_map)

        agg = (
            df.groupby("hypothesis_id")["pnl"]
            .agg(
                total_pnl="sum",
                avg_pnl="mean",
                trade_count="count",
            )
            .reset_index()
        )

        # Win rate (requires pnl column)
        win_rates = (
            df.groupby("hypothesis_id")
            .apply(lambda g: (g["pnl"] > 0).mean(), include_groups=False)
            .rename("win_rate")
            .reset_index()
        )
        agg = agg.merge(win_rates, on="hypothesis_id", how="left")
        return agg

    def pnl_attribution_by_regime(
        self,
        trades_df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> pd.DataFrame:
        """
        Aggregate P&L by market regime.

        Parameters
        ----------
        trades_df     : pd.DataFrame — must have columns: trade_id, pnl, opened_at
        regime_labels : pd.Series   — index = timestamp, value = regime string
            Should cover the trades' ``opened_at`` range.

        Returns
        -------
        pd.DataFrame with columns:
            regime, total_pnl, avg_pnl, trade_count, win_rate
        """
        if trades_df.empty:
            return pd.DataFrame(
                columns=["regime", "total_pnl", "avg_pnl", "trade_count", "win_rate"]
            )

        df = trades_df.copy()
        df["opened_at_ts"] = pd.to_datetime(df["opened_at"], utc=True, errors="coerce")
        df["regime"] = df["opened_at_ts"].apply(
            lambda ts: _nearest_regime(ts, regime_labels)
        )

        agg = (
            df.groupby("regime")["pnl"]
            .agg(total_pnl="sum", avg_pnl="mean", trade_count="count")
            .reset_index()
        )
        win_rates = (
            df.groupby("regime")
            .apply(lambda g: (g["pnl"] > 0).mean(), include_groups=False)
            .rename("win_rate")
            .reset_index()
        )
        agg = agg.merge(win_rates, on="regime", how="left")
        return agg

    def pnl_attribution_by_signal(
        self,
        trades_df: pd.DataFrame,
        signal_values_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attribute P&L to the strongest signal active at trade entry.

        Parameters
        ----------
        trades_df        : pd.DataFrame — columns: trade_id, pnl, opened_at
        signal_values_df : pd.DataFrame — columns: ts, signal_name, value
            Each row represents a signal value at a given timestamp.

        Returns
        -------
        pd.DataFrame with columns:
            signal_name, total_pnl, avg_pnl, trade_count, win_rate
        """
        if trades_df.empty or signal_values_df.empty:
            return pd.DataFrame(
                columns=["signal_name", "total_pnl", "avg_pnl", "trade_count", "win_rate"]
            )

        df = trades_df.copy()
        df["opened_at_ts"] = pd.to_datetime(df["opened_at"], utc=True, errors="coerce")

        signal_values_df = signal_values_df.copy()
        signal_values_df["ts"] = pd.to_datetime(
            signal_values_df["ts"], utc=True, errors="coerce"
        )
        signal_values_df.sort_values("ts", inplace=True)

        def _dominant_signal(trade_ts: pd.Timestamp) -> str:
            """Return the signal with the highest absolute value at trade_ts."""
            nearby = signal_values_df[
                (signal_values_df["ts"] <= trade_ts)
                & (signal_values_df["ts"] >= trade_ts - pd.Timedelta(hours=1))
            ]
            if nearby.empty:
                return "unknown"
            idx = nearby["value"].abs().idxmax()
            return str(nearby.loc[idx, "signal_name"])

        df["signal_name"] = df["opened_at_ts"].apply(_dominant_signal)

        agg = (
            df.groupby("signal_name")["pnl"]
            .agg(total_pnl="sum", avg_pnl="mean", trade_count="count")
            .reset_index()
        )
        win_rates = (
            df.groupby("signal_name")
            .apply(lambda g: (g["pnl"] > 0).mean(), include_groups=False)
            .rename("win_rate")
            .reset_index()
        )
        agg = agg.merge(win_rates, on="signal_name", how="left")
        return agg

    def unexplained_pnl(
        self,
        total_pnl: float,
        attributed_pnl: float,
    ) -> float:
        """
        Compute the residual (unexplained) alpha.

        Parameters
        ----------
        total_pnl      : float — sum of all live trade P&L
        attributed_pnl : float — sum of P&L successfully attributed to hypotheses

        Returns
        -------
        float — residual P&L not explained by any hypothesis
        """
        return total_pnl - attributed_pnl

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_attributions(
        self,
        attribution_map: dict[str, int | None],
        trades_df: pd.DataFrame,
    ) -> int:
        """
        Write attribution records to ``trade_attributions`` in idea_engine.db.

        Parameters
        ----------
        attribution_map : dict  trade_id → hypothesis_id
        trades_df       : pd.DataFrame  — original trades (for confidence etc.)

        Returns
        -------
        int — number of rows inserted / replaced.
        """
        rows_written = 0
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for trade_id, hyp_id in attribution_map.items():
            # Look up trade row for confidence / regime info
            trade_row = (
                trades_df[trades_df["trade_id"].astype(str) == str(trade_id)]
                if not trades_df.empty
                else pd.DataFrame()
            )
            confidence = (
                float(trade_row.iloc[0].get("_attribution_confidence", CONFIDENCE_EXACT))
                if not trade_row.empty and "_attribution_confidence" in trade_row.columns
                else (CONFIDENCE_EXACT if hyp_id is not None else 0.0)
            )

            try:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO trade_attributions
                        (trade_id, hypothesis_id, signal_name, regime,
                         attribution_confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(trade_id),
                        hyp_id,
                        None,   # signal_name: filled later by pnl_attribution_by_signal
                        None,   # regime: filled later by pnl_attribution_by_regime
                        confidence,
                        now_str,
                    ),
                )
                rows_written += 1
            except sqlite3.Error as exc:
                logger.warning("Could not persist attribution for %s: %s", trade_id, exc)

        self._conn.commit()
        logger.debug("Persisted %d attribution records.", rows_written)
        return rows_written

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _attribute_single_trade(
        self,
        trade: pd.Series,
        hypotheses_df: pd.DataFrame,
        hyp_params: dict[int, dict[str, Any]],
        hyp_adoption: dict[int, datetime],
    ) -> Attribution:
        """
        Apply the three attribution rules to a single trade and return
        the best Attribution found.
        """
        trade_id = str(trade.get("trade_id", ""))
        symbol = str(trade.get("symbol", ""))
        side = str(trade.get("side", ""))
        opened_at = _parse_utc(str(trade.get("opened_at", "")))
        trade_params_raw = trade.get("params_json", "{}")
        trade_params: dict[str, Any] = (
            json.loads(trade_params_raw)
            if isinstance(trade_params_raw, str)
            else (trade_params_raw or {})
        )

        best: Attribution = Attribution(
            trade_id=trade_id,
            hypothesis_id=None,
            signal_name=None,
            regime=None,
            confidence=0.0,
            rule_used="unattributed",
        )

        for _, hrow in hypotheses_df.iterrows():
            hid = int(hrow["id"])
            h_params = hyp_params[hid]
            h_adoption = hyp_adoption[hid]

            # -- Rule 1: exact symbol + side within active window ------
            hyp_symbols = _extract_symbols(h_params)
            hyp_side = h_params.get("side", "")
            if (
                symbol in hyp_symbols or not hyp_symbols
            ) and (
                hyp_side in ("", side)
            ) and (
                opened_at >= h_adoption
            ):
                candidate = Attribution(
                    trade_id=trade_id,
                    hypothesis_id=hid,
                    signal_name=None,
                    regime=None,
                    confidence=CONFIDENCE_EXACT,
                    rule_used="exact_symbol",
                )
                if candidate.confidence > best.confidence:
                    best = candidate
                continue  # skip lower-priority rules for this hyp

            # -- Rule 2: parameter overlap --------------------------------
            param_score = _param_overlap_score(h_params, trade_params)
            if param_score >= self._param_threshold:
                confidence = CONFIDENCE_PARAM * param_score
                if confidence > best.confidence:
                    best = Attribution(
                        trade_id=trade_id,
                        hypothesis_id=hid,
                        signal_name=None,
                        regime=None,
                        confidence=confidence,
                        rule_used="param_match",
                    )
                continue

            # -- Rule 3: temporal proximity --------------------------------
            time_diff = abs(opened_at - h_adoption)
            if time_diff <= self._temporal_window:
                decay = 1.0 - time_diff.total_seconds() / self._temporal_window.total_seconds()
                confidence = CONFIDENCE_TEMPORAL * decay
                if confidence > best.confidence:
                    best = Attribution(
                        trade_id=trade_id,
                        hypothesis_id=hid,
                        signal_name=None,
                        regime=None,
                        confidence=confidence,
                        rule_used="temporal",
                    )

        if best.confidence < MIN_CONFIDENCE:
            best.hypothesis_id = None
            best.rule_used = "unattributed"

        return best


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_utc(ts_str: str) -> datetime:
    """Parse an ISO-8601 string to an aware UTC datetime; falls back to epoch."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def _extract_symbols(params: dict[str, Any]) -> list[str]:
    """
    Pull any symbol / instrument / ticker values from a params dict.
    Supports common key names used in the IAE codebase.
    """
    candidates: list[str] = []
    for key in ("symbol", "symbols", "instrument", "instruments", "ticker", "tickers"):
        val = params.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            candidates.extend(str(v) for v in val)
        else:
            candidates.append(str(val))
    return [s.upper() for s in candidates if s]


def _param_overlap_score(
    hyp_params: dict[str, Any],
    trade_params: dict[str, Any],
) -> float:
    """
    Compute the fraction of hypothesis parameter keys whose values are
    consistent with (i.e. present and equal in) the trade params dict.

    Returns a float in [0, 1].  If hyp_params is empty, returns 0.
    """
    if not hyp_params:
        return 0.0

    matches = 0
    for key, hyp_val in hyp_params.items():
        if key in trade_params:
            trade_val = trade_params[key]
            # Numeric near-equality (within 1 %)
            try:
                if abs(float(hyp_val) - float(trade_val)) / (abs(float(hyp_val)) + 1e-9) < 0.01:
                    matches += 1
                    continue
            except (TypeError, ValueError):
                pass
            # Exact string match
            if str(hyp_val).lower() == str(trade_val).lower():
                matches += 1

    return matches / len(hyp_params)


def _nearest_regime(
    trade_ts: pd.Timestamp | None,
    regime_labels: pd.Series,
) -> str:
    """
    Find the regime label closest in time to ``trade_ts``.

    Parameters
    ----------
    trade_ts      : pd.Timestamp | None
    regime_labels : pd.Series  — index = pd.DatetimeTZDtype, values = regime str

    Returns
    -------
    str — regime label, or 'unknown' if the series is empty / ts is None.
    """
    if trade_ts is None or regime_labels.empty:
        return "unknown"
    try:
        idx = regime_labels.index.get_indexer([trade_ts], method="nearest")[0]
        return str(regime_labels.iloc[idx])
    except Exception:
        return "unknown"
