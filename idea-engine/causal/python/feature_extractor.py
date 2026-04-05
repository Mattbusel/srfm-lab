"""
causal/python/feature_extractor.py

CausalFeatureExtractor: builds a feature matrix aligned to trade timestamps
from trade data + regime logs stored in idea_engine.db or provided as DataFrames.

Features produced:
    bh_mass_d          — daily BH mass
    bh_mass_h          — hourly BH mass
    bh_mass_15m        — 15-minute BH mass
    tf_score           — timeframe alignment score (0-10 int)
    atr                — ATR (normalised by price)
    garch_vol          — GARCH(1,1) estimated conditional volatility
    ou_zscore          — Ornstein-Uhlenbeck z-score of price deviation
    hour_of_day        — UTC hour (0-23)
    day_of_week        — 0=Monday … 6=Sunday
    btc_dominance_proxy — BTC's share of total portfolio mass (proxy for dominance)
    cross_asset_momentum — rolling z-score of mean cross-asset return
"""

from __future__ import annotations

import logging
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")

# Window sizes
ATR_PERIOD = 14
GARCH_VOL_WINDOW = 60        # rolling std as GARCH proxy
OU_WINDOW = 30               # OU mean-reversion z-score window
CROSS_ASSET_MOMENTUM_WINDOW = 24  # bars


# ---------------------------------------------------------------------------
# GARCH(1,1) approximation via rolling EWMA variance
# ---------------------------------------------------------------------------

def _garch_vol_proxy(returns: pd.Series, alpha: float = 0.1, beta: float = 0.85) -> pd.Series:
    """
    GARCH(1,1) conditional variance via recursive EWMA.
    h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
    omega is set so unconditional variance = long-run variance of the series.
    """
    r2 = returns ** 2
    long_run_var = float(r2.mean()) if len(r2) > 1 else 1e-8
    omega = long_run_var * (1 - alpha - beta)
    omega = max(omega, 1e-10)

    h = np.full(len(returns), long_run_var)
    vals = returns.values
    for t in range(1, len(vals)):
        h[t] = omega + alpha * vals[t - 1] ** 2 + beta * h[t - 1]

    return pd.Series(np.sqrt(h), index=returns.index)


# ---------------------------------------------------------------------------
# OU z-score
# ---------------------------------------------------------------------------

def _ou_zscore(prices: pd.Series, window: int = OU_WINDOW) -> pd.Series:
    """
    Ornstein-Uhlenbeck z-score: (price - rolling_mean) / rolling_std.
    Captures mean-reversion pressure.
    """
    roll_mean = prices.rolling(window, min_periods=max(window // 2, 2)).mean()
    roll_std = prices.rolling(window, min_periods=max(window // 2, 2)).std()
    z = (prices - roll_mean) / roll_std.replace(0, np.nan)
    return z.fillna(0.0)


# ---------------------------------------------------------------------------
# ATR (normalised)
# ---------------------------------------------------------------------------

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder's ATR, normalised by closing price."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_raw = tr.ewm(span=period, min_periods=1).mean()
    return (atr_raw / close.replace(0, np.nan)).fillna(0.0)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class CausalFeatureExtractor:
    """
    Builds a pd.DataFrame with causal candidate features aligned to trade timestamps.

    Can be constructed from:
      1. A path to idea_engine.db (reads from trade_log and regime_log tables if present)
      2. Raw DataFrames passed directly (for testing / offline use)

    Parameters
    ----------
    db_path        : path to idea_engine.db
    trade_df       : optional pre-loaded trade DataFrame
    price_df       : optional OHLCV price DataFrame (index=timestamp, cols=open/high/low/close/volume)
    regime_df      : optional regime DataFrame with columns like tf_score, bh_mass_*
    btc_price_df   : optional BTC price series for dominance proxy
    """

    FEATURE_COLUMNS = [
        "bh_mass_d",
        "bh_mass_h",
        "bh_mass_15m",
        "tf_score",
        "atr",
        "garch_vol",
        "ou_zscore",
        "hour_of_day",
        "day_of_week",
        "btc_dominance_proxy",
        "cross_asset_momentum",
    ]

    def __init__(
        self,
        db_path: Path | str = DB_PATH,
        trade_df: pd.DataFrame | None = None,
        price_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
        btc_price_df: pd.DataFrame | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._trade_df = trade_df
        self._price_df = price_df
        self._regime_df = regime_df
        self._btc_price_df = btc_price_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        instrument: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Build and return the feature matrix.
        Rows = trade timestamps (or all OHLCV bars if no trades provided).
        Columns = FEATURE_COLUMNS plus trade outcome columns if available.

        Parameters
        ----------
        instrument : filter to a specific instrument
        start      : ISO datetime string, filter start
        end        : ISO datetime string, filter end
        """
        price_df = self._load_price(instrument, start, end)
        regime_df = self._load_regime(instrument, start, end)
        trade_df = self._load_trades(instrument, start, end)

        if price_df is None or price_df.empty:
            log.warning("No price data available; cannot extract causal features")
            return pd.DataFrame(columns=self.FEATURE_COLUMNS)

        # Ensure DatetimeIndex
        price_df = self._ensure_datetime_index(price_df)
        feat = pd.DataFrame(index=price_df.index)

        # -- Price-derived features --
        feat["atr"] = self._compute_atr(price_df)
        feat["garch_vol"] = self._compute_garch_vol(price_df)
        feat["ou_zscore"] = self._compute_ou_zscore(price_df)

        # -- Calendar features --
        feat["hour_of_day"] = feat.index.hour.astype(float)
        feat["day_of_week"] = feat.index.dayofweek.astype(float)

        # -- Regime features (bh_mass_*, tf_score) --
        feat = self._merge_regime_features(feat, regime_df)

        # -- BTC dominance proxy --
        feat["btc_dominance_proxy"] = self._compute_btc_dominance(price_df)

        # -- Cross-asset momentum --
        feat["cross_asset_momentum"] = self._compute_cross_asset_momentum(price_df)

        # -- Trade outcomes (if available) --
        if trade_df is not None and not trade_df.empty:
            feat = self._merge_trade_outcomes(feat, trade_df)

        # -- Handle missing values --
        feat = self._handle_missing(feat)

        log.info(
            "Feature matrix: %d rows × %d cols", len(feat), len(feat.columns)
        )
        return feat

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_price(
        self, instrument: str | None, start: str | None, end: str | None
    ) -> pd.DataFrame | None:
        if self._price_df is not None:
            df = self._price_df.copy()
            return self._filter_time(df, start, end)

        try:
            conn = sqlite3.connect(str(self.db_path))
            query = "SELECT * FROM ohlcv WHERE 1=1"
            params: list[Any] = []
            if instrument:
                query += " AND instrument = ?"
                params.append(instrument)
            if start:
                query += " AND timestamp >= ?"
                params.append(start)
            if end:
                query += " AND timestamp <= ?"
                params.append(end)
            query += " ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df if not df.empty else None
        except Exception as exc:
            log.debug("Could not load OHLCV from DB: %s", exc)
            return None

    def _load_regime(
        self, instrument: str | None, start: str | None, end: str | None
    ) -> pd.DataFrame | None:
        if self._regime_df is not None:
            df = self._regime_df.copy()
            return self._filter_time(df, start, end)

        try:
            conn = sqlite3.connect(str(self.db_path))
            query = "SELECT * FROM regime_log WHERE 1=1"
            params: list[Any] = []
            if instrument:
                query += " AND instrument = ?"
                params.append(instrument)
            if start:
                query += " AND timestamp >= ?"
                params.append(start)
            if end:
                query += " AND timestamp <= ?"
                params.append(end)
            query += " ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df if not df.empty else None
        except Exception as exc:
            log.debug("Could not load regime_log from DB: %s", exc)
            return None

    def _load_trades(
        self, instrument: str | None, start: str | None, end: str | None
    ) -> pd.DataFrame | None:
        if self._trade_df is not None:
            df = self._trade_df.copy()
            return self._filter_time(df, start, end)

        try:
            conn = sqlite3.connect(str(self.db_path))
            query = "SELECT * FROM trade_log WHERE 1=1"
            params: list[Any] = []
            if instrument:
                query += " AND instrument = ?"
                params.append(instrument)
            if start:
                query += " AND entry_time >= ?"
                params.append(start)
            if end:
                query += " AND entry_time <= ?"
                params.append(end)
            query += " ORDER BY entry_time ASC"
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df if not df.empty else None
        except Exception as exc:
            log.debug("Could not load trade_log from DB: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Feature computers
    # ------------------------------------------------------------------

    def _compute_atr(self, price_df: pd.DataFrame) -> pd.Series:
        needed = {"high", "low", "close"}
        cols = set(price_df.columns.str.lower())
        if not needed.issubset(cols):
            # Fallback: rolling std of close as ATR proxy
            close = self._get_close(price_df)
            return close.pct_change().rolling(ATR_PERIOD, min_periods=1).std().fillna(0.0)

        high = price_df[self._col(price_df, "high")]
        low = price_df[self._col(price_df, "low")]
        close = price_df[self._col(price_df, "close")]
        return _atr(high, low, close)

    def _compute_garch_vol(self, price_df: pd.DataFrame) -> pd.Series:
        close = self._get_close(price_df)
        returns = close.pct_change().fillna(0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _garch_vol_proxy(returns)

    def _compute_ou_zscore(self, price_df: pd.DataFrame) -> pd.Series:
        close = self._get_close(price_df)
        return _ou_zscore(close)

    def _compute_btc_dominance(self, price_df: pd.DataFrame) -> pd.Series:
        """
        If BTC price data is available separately, compute ratio of BTC close
        to total portfolio close as a dominance proxy. Otherwise return zeros.
        """
        if self._btc_price_df is not None:
            try:
                btc = self._ensure_datetime_index(self._btc_price_df.copy())
                btc_close = self._get_close(btc).reindex(price_df.index, method="ffill")
                inst_close = self._get_close(price_df)
                total = (btc_close + inst_close).replace(0, np.nan)
                proxy = (btc_close / total).fillna(0.5)
                return proxy
            except Exception as exc:
                log.debug("BTC dominance proxy failed: %s", exc)
        return pd.Series(0.0, index=price_df.index)

    def _compute_cross_asset_momentum(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Rolling z-score of the instrument's return over CROSS_ASSET_MOMENTUM_WINDOW bars.
        In a multi-asset context this would be the mean across assets; here it's a proxy.
        """
        close = self._get_close(price_df)
        returns = close.pct_change().fillna(0.0)
        w = CROSS_ASSET_MOMENTUM_WINDOW
        roll_mean = returns.rolling(w, min_periods=max(w // 4, 2)).mean()
        roll_std = returns.rolling(w, min_periods=max(w // 4, 2)).std()
        z = (returns - roll_mean) / roll_std.replace(0, np.nan)
        return z.fillna(0.0)

    # ------------------------------------------------------------------
    # Regime merging
    # ------------------------------------------------------------------

    def _merge_regime_features(
        self, feat: pd.DataFrame, regime_df: pd.DataFrame | None
    ) -> pd.DataFrame:
        bh_mass_cols = {"bh_mass_d": 0.5, "bh_mass_h": 0.5, "bh_mass_15m": 0.5, "tf_score": 5.0}

        if regime_df is None or regime_df.empty:
            for col, default in bh_mass_cols.items():
                feat[col] = default
            return feat

        regime_df = self._ensure_datetime_index(regime_df)

        for col, default in bh_mass_cols.items():
            if col in regime_df.columns:
                aligned = regime_df[col].reindex(feat.index, method="ffill").fillna(default)
                feat[col] = aligned.values
            else:
                feat[col] = default

        return feat

    def _merge_trade_outcomes(
        self, feat: pd.DataFrame, trade_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Align trade outcomes (pnl, win/loss) to the feature index."""
        ts_col = None
        for c in ("entry_time", "timestamp", "time", "open_time"):
            if c in trade_df.columns:
                ts_col = c
                break
        if ts_col is None:
            return feat

        trade_df = trade_df.copy()
        trade_df[ts_col] = pd.to_datetime(trade_df[ts_col], utc=True, errors="coerce")
        trade_df = trade_df.dropna(subset=[ts_col])
        trade_df = trade_df.set_index(ts_col).sort_index()

        if "pnl" in trade_df.columns:
            pnl = trade_df["pnl"].reindex(feat.index).fillna(0.0)
            feat["trade_pnl"] = pnl.values

        if "win" in trade_df.columns:
            win = trade_df["win"].reindex(feat.index).fillna(np.nan)
            feat["trade_win"] = win.values
        elif "pnl" in trade_df.columns:
            feat["trade_win"] = (feat["trade_pnl"] > 0).astype(float)

        return feat

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("timestamp", "time", "open_time", "datetime"):
                if col in df.columns:
                    df = df.set_index(col)
                    break
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()].sort_index()
        return df

    @staticmethod
    def _get_close(df: pd.DataFrame) -> pd.Series:
        for col in ("close", "Close", "CLOSE", "price", "last"):
            if col in df.columns:
                return df[col].astype(float)
        # fallback: last numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            return df[numeric_cols[-1]].astype(float)
        raise ValueError("Cannot determine close price column in DataFrame")

    @staticmethod
    def _col(df: pd.DataFrame, name: str) -> str:
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        raise KeyError(f"Column '{name}' not found in DataFrame")

    @staticmethod
    def _filter_time(
        df: pd.DataFrame, start: str | None, end: str | None
    ) -> pd.DataFrame:
        if start or end:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
            if start:
                df = df[idx >= pd.Timestamp(start, tz="UTC")]
            if end:
                df = df[idx <= pd.Timestamp(end, tz="UTC")]
        return df

    @staticmethod
    def _handle_missing(feat: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill then backward-fill then fill with column median.
        This order preserves causal structure: we don't look ahead.
        """
        feat = feat.ffill().bfill()
        for col in feat.columns:
            if feat[col].isna().any():
                median = feat[col].median()
                feat[col] = feat[col].fillna(median if pd.notna(median) else 0.0)
        return feat
