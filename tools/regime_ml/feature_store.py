"""
feature_store.py — Centralized DuckDB-backed feature store.

Feature groups
    momentum       — 1h / 4h / 1d returns
    volatility     — GARCH vol, realised vol, ATR ratio
    microstructure — bid-ask spread proxy, depth imbalance proxy
    regime         — Hurst exponent, permutation entropy, BH state
    cross_asset    — BTC correlation, SPY beta (rolling OLS)

Point-in-time correctness
    All features are written with an *as_of* timestamp equal to the
    bar-close time of the observation.  Queries filter with
    feature_ts <= as_of so there is never any lookahead.

DuckDB schema
    CREATE TABLE features (
        symbol       VARCHAR NOT NULL,
        feature_ts   TIMESTAMP NOT NULL,   -- bar-close time
        feature_name VARCHAR NOT NULL,
        value        DOUBLE  NOT NULL,
        PRIMARY KEY (symbol, feature_ts, feature_name)
    )

Feature drift monitor
    Maintains a rolling baseline distribution (mean, std) per feature.
    Alerts when |new_value - baseline_mean| > 2 * baseline_std.

REST API
    Thin FastAPI wrapper: GET /features/{symbol}?as_of=2024-01-01T12:00:00
    Run with:  uvicorn tools.regime_ml.feature_store:app --reload
"""

from __future__ import annotations

import math
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import duckdb  # type: ignore

    _DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DUCKDB_AVAILABLE = False
    warnings.warn("duckdb not installed — FeatureStore will use in-memory dict fallback.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "features.duckdb"
_DRIFT_WINDOW    = 200   # bars for baseline distribution
_DRIFT_THRESHOLD = 2.0   # sigma
_ATR_WINDOW      = 14
_REALISED_VOL_WINDOW = 20
_CORR_WINDOW     = 60
_HURST_WINDOW    = 100


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftAlert:
    symbol: str
    feature_name: str
    ts: float
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float


@dataclass
class FeatureRow:
    symbol: str
    feature_ts: datetime
    feature_name: str
    value: float


# ---------------------------------------------------------------------------
# Rolling statistics helper
# ---------------------------------------------------------------------------


class _RollingStats:
    """Maintains rolling mean and std for drift detection."""

    def __init__(self, window: int = _DRIFT_WINDOW) -> None:
        self._buf: Deque[float] = deque(maxlen=window)

    def update(self, x: float) -> Tuple[float, float]:
        """Add x and return (mean, std) of current window."""
        self._buf.append(x)
        arr = np.array(self._buf)
        return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0

    def z_score(self, x: float) -> float:
        if len(self._buf) < 5:
            return 0.0
        arr = np.array(self._buf)
        mean = float(arr.mean())
        std  = float(arr.std())
        return (x - mean) / std if std > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# GARCH(1,1) inline (minimal, no arch dependency)
# ---------------------------------------------------------------------------


def _garch11_vol(returns: np.ndarray, omega: float = 1e-6,
                 alpha: float = 0.1, beta: float = 0.85) -> float:
    """One-step-ahead GARCH(1,1) volatility forecast."""
    if len(returns) < 5:
        return float(np.std(returns)) if len(returns) > 0 else 0.0
    h = float(np.var(returns))
    for r in returns:
        h = omega + alpha * r ** 2 + beta * h
    return math.sqrt(max(0.0, h))


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------


def _atr(highs: Sequence[float], lows: Sequence[float],
         closes: Sequence[float], window: int = _ATR_WINDOW) -> float:
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    trs_arr = np.array(trs[-window:])
    return float(trs_arr.mean()) if len(trs_arr) > 0 else 0.0


# ---------------------------------------------------------------------------
# Rolling OLS beta
# ---------------------------------------------------------------------------


def _rolling_beta(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope of y ~ x (no intercept)."""
    if len(x) < 10 or len(y) < 10:
        return 0.0
    n = min(len(x), len(y))
    x_s, y_s = x[-n:], y[-n:]
    xTx = float(np.dot(x_s, x_s))
    if xTx < 1e-12:
        return 0.0
    return float(np.dot(x_s, y_s)) / xTx


# ---------------------------------------------------------------------------
# Rolling Hurst (R/S, fast version)
# ---------------------------------------------------------------------------


def _fast_hurst_rs(series: np.ndarray) -> float:
    n = len(series)
    if n < 20:
        return 0.5
    # Simple two-scale R/S
    half = n // 2
    results = []
    for chunk in [series[:half], series[half:], series]:
        if len(chunk) < 10:
            continue
        mean = chunk.mean()
        devs = np.cumsum(chunk - mean)
        rng  = devs.max() - devs.min()
        std  = chunk.std(ddof=1)
        if std > 1e-12:
            results.append(rng / std / math.sqrt(len(chunk)))
    if not results:
        return 0.5
    return max(0.0, min(1.0, float(np.mean(results))))


# ---------------------------------------------------------------------------
# Per-symbol OHLCV buffer
# ---------------------------------------------------------------------------


@dataclass
class _SymbolBuffer:
    symbol: str
    opens:   Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    highs:   Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    lows:    Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    closes:  Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    volumes: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    timestamps: Deque[datetime] = field(default_factory=lambda: deque(maxlen=300))
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=300))
    bar_count: int = 0
    # External state injected by caller
    bh_mass: float = 0.0
    hurst: float = 0.5
    entropy: float = 0.7

    # Drift trackers per feature
    drift_stats: Dict[str, _RollingStats] = field(default_factory=dict)

    def push(
        self,
        o: float, h: float, l: float, c: float, v: float,
        ts: datetime,
    ) -> None:
        if self.closes:
            self.returns.append(math.log(c / self.closes[-1]) if self.closes[-1] > 0 else 0.0)
        else:
            self.returns.append(0.0)
        self.opens.append(o)
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)
        self.volumes.append(v)
        self.timestamps.append(ts)
        self.bar_count += 1


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------


class _DuckDBBackend:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                symbol       VARCHAR      NOT NULL,
                feature_ts   TIMESTAMP    NOT NULL,
                feature_name VARCHAR      NOT NULL,
                value        DOUBLE       NOT NULL,
                PRIMARY KEY (symbol, feature_ts, feature_name)
            )
            """
        )

    def upsert_many(self, rows: List[FeatureRow]) -> None:
        if not rows:
            return
        data = [
            (r.symbol, r.feature_ts.isoformat(), r.feature_name, r.value)
            for r in rows
        ]
        self._con.executemany(
            """
            INSERT OR REPLACE INTO features (symbol, feature_ts, feature_name, value)
            VALUES (?, ?, ?, ?)
            """,
            data,
        )

    def query(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        ts_filter = f"AND feature_ts <= '{as_of.isoformat()}'" if as_of else ""
        name_filter = ""
        if feature_names:
            quoted = ", ".join(f"'{n}'" for n in feature_names)
            name_filter = f"AND feature_name IN ({quoted})"
        sql = f"""
            SELECT feature_name, value
            FROM (
                SELECT feature_name, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY feature_name
                           ORDER BY feature_ts DESC
                       ) AS rn
                FROM features
                WHERE symbol = '{symbol}'
                {ts_filter}
                {name_filter}
            ) sub
            WHERE rn = 1
        """
        rows = self._con.execute(sql).fetchall()
        return {r[0]: float(r[1]) for r in rows}

    def close(self) -> None:
        self._con.close()


class _DictBackend:
    """In-memory fallback when DuckDB is not available."""

    def __init__(self) -> None:
        # Dict[symbol -> Dict[feature_name -> (ts, value)]]
        self._store: Dict[str, Dict[str, Tuple[datetime, float]]] = defaultdict(dict)

    def upsert_many(self, rows: List[FeatureRow]) -> None:
        for r in rows:
            prev = self._store[r.symbol].get(r.feature_name)
            if prev is None or r.feature_ts >= prev[0]:
                self._store[r.symbol][r.feature_name] = (r.feature_ts, r.value)

    def query(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        result = {}
        for fname, (ts, val) in self._store.get(symbol, {}).items():
            if as_of and ts > as_of:
                continue
            if feature_names and fname not in feature_names:
                continue
            result[fname] = val
        return result

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------


class FeatureStore:
    """
    Centralized, point-in-time-correct feature store backed by DuckDB.

    Parameters
    ----------
    db_path : str | Path
        DuckDB file path.
    drift_threshold : float
        Number of standard deviations to trigger a feature drift alert.
    btc_symbol : str
        Symbol used as the cross-asset reference (transfer entropy / correlation).
    spy_symbol : str
        Symbol used as equity beta reference.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        drift_threshold: float = _DRIFT_THRESHOLD,
        btc_symbol: str = "BTC",
        spy_symbol: str = "SPY",
    ) -> None:
        self.drift_threshold = drift_threshold
        self.btc_symbol = btc_symbol
        self.spy_symbol = spy_symbol

        if _DUCKDB_AVAILABLE:
            self._backend: _DuckDBBackend | _DictBackend = _DuckDBBackend(Path(db_path))
        else:
            self._backend = _DictBackend()

        self._buffers: Dict[str, _SymbolBuffer] = {}
        self._drift_alerts: List[FeatureDriftAlert] = []

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        ts: Optional[datetime] = None,
        *,
        bh_mass: float = 0.0,
        hurst: float = 0.5,
        entropy: float = 0.7,
    ) -> Dict[str, float]:
        """
        Ingest a new OHLCV bar and compute/store all features.

        Returns the computed feature dict for this bar.
        """
        if ts is None:
            ts = datetime.now(tz=timezone.utc)

        buf = self._get_buffer(symbol)
        buf.bh_mass = bh_mass
        buf.hurst   = hurst
        buf.entropy = entropy
        buf.push(open_, high, low, close, volume, ts)

        features = self._compute_features(symbol, buf, ts)
        rows = [
            FeatureRow(symbol, ts, name, val)
            for name, val in features.items()
            if math.isfinite(val)
        ]
        self._backend.upsert_many(rows)

        # Drift monitoring
        alerts = self._check_drift(symbol, buf, ts, features)
        self._drift_alerts.extend(alerts)

        return features

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_features(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Retrieve the latest features for *symbol* as of *as_of*.

        Point-in-time correct: only features with feature_ts <= as_of
        are returned, preventing lookahead.
        """
        return self._backend.query(symbol, as_of, feature_names)

    def get_feature_vector(
        self,
        symbol: str,
        as_of: Optional[datetime] = None,
    ) -> np.ndarray:
        """Return a flat numpy feature vector in a canonical order."""
        d = self.get_features(symbol, as_of)
        keys = sorted(d.keys())
        return np.array([d[k] for k in keys], dtype=float)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        symbol: str,
        buf: _SymbolBuffer,
        ts: datetime,
    ) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        rets = np.array(list(buf.returns), dtype=float)
        closes = np.array(list(buf.closes), dtype=float)
        highs  = list(buf.highs)
        lows   = list(buf.lows)
        vols   = np.array(list(buf.volumes), dtype=float)

        # ---- Momentum ----
        feats.update(self._momentum_features(closes, rets))

        # ---- Volatility ----
        feats.update(self._volatility_features(rets, closes, highs, lows))

        # ---- Microstructure ----
        feats.update(self._microstructure_features(closes, vols))

        # ---- Regime ----
        feats["regime_hurst"]   = buf.hurst
        feats["regime_entropy"] = buf.entropy
        feats["regime_bh_mass"] = buf.bh_mass

        # ---- Cross-asset ----
        feats.update(self._cross_asset_features(symbol, rets))

        return feats

    def _momentum_features(
        self, closes: np.ndarray, rets: np.ndarray
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}
        n = len(closes)
        # 1h = 1 bar, 4h = 4 bars, 1d ≈ 6.5 bars (hourly)
        for label, k in [("1h", 1), ("4h", 4), ("1d", 7)]:
            if n > k:
                f[f"mom_{label}"] = math.log(closes[-1] / closes[-(k + 1)])
            else:
                f[f"mom_{label}"] = 0.0
        # Rate of change vs rolling 20-bar mean
        if n >= 20:
            ma20 = float(closes[-20:].mean())
            f["price_vs_ma20"] = (closes[-1] - ma20) / ma20 if ma20 > 0 else 0.0
        else:
            f["price_vs_ma20"] = 0.0
        return f

    def _volatility_features(
        self,
        rets: np.ndarray,
        closes: np.ndarray,
        highs: List[float],
        lows: List[float],
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}
        if len(rets) >= _REALISED_VOL_WINDOW:
            f["realised_vol"] = float(rets[-_REALISED_VOL_WINDOW:].std())
        else:
            f["realised_vol"] = float(rets.std()) if len(rets) > 1 else 0.0

        f["garch_vol"] = _garch11_vol(rets[-50:] if len(rets) >= 50 else rets)

        if len(closes) >= _ATR_WINDOW + 1:
            atr = _atr(highs, lows, closes.tolist(), _ATR_WINDOW)
            f["atr"]       = atr
            f["atr_ratio"] = atr / closes[-1] if closes[-1] > 0 else 0.0
        else:
            f["atr"] = 0.0
            f["atr_ratio"] = 0.0

        # Vol regime: ratio of short-term to long-term vol
        if len(rets) >= 50:
            short_vol = float(rets[-10:].std()) if len(rets) >= 10 else 0.0
            long_vol  = float(rets[-50:].std())
            f["vol_regime_ratio"] = short_vol / long_vol if long_vol > 1e-10 else 1.0
        else:
            f["vol_regime_ratio"] = 1.0

        return f

    def _microstructure_features(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}
        n = len(closes)
        if n < 5:
            f["spread_proxy"] = 0.0
            f["depth_imbalance"] = 0.0
            f["volume_ratio"]    = 1.0
            return f

        # Spread proxy: Corwin-Schultz 2-day high/low spread estimator (simplified)
        if n >= 2:
            # Using close-to-close as a minimal proxy
            ret2 = abs(float(closes[-1]) - float(closes[-2])) / float(closes[-2]) if closes[-2] > 0 else 0.0
            f["spread_proxy"] = ret2

        # Depth imbalance proxy: volume-weighted directional pressure
        # Up bars vs down bars in last 10
        if n >= 10:
            up_vol   = float(sum(volumes[i] for i in range(n - 10, n) if closes[i] > closes[i - 1]))
            down_vol = float(sum(volumes[i] for i in range(n - 10, n) if closes[i] < closes[i - 1]))
            total_vol = up_vol + down_vol
            f["depth_imbalance"] = (up_vol - down_vol) / total_vol if total_vol > 0 else 0.0
        else:
            f["depth_imbalance"] = 0.0

        # Volume ratio: current vs 20-bar mean
        if n >= 20:
            avg_vol = float(volumes[-20:].mean())
            f["volume_ratio"] = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1.0
        else:
            f["volume_ratio"] = 1.0

        return f

    def _cross_asset_features(
        self,
        symbol: str,
        rets: np.ndarray,
    ) -> Dict[str, float]:
        f: Dict[str, float] = {}
        n = len(rets)

        # BTC correlation
        btc_buf = self._buffers.get(self.btc_symbol)
        if btc_buf and symbol != self.btc_symbol and n >= _CORR_WINDOW:
            btc_rets = np.array(list(btc_buf.returns)[-_CORR_WINDOW:], dtype=float)
            this_rets = rets[-_CORR_WINDOW:]
            if len(btc_rets) >= _CORR_WINDOW:
                m = min(len(btc_rets), len(this_rets))
                corr = float(np.corrcoef(btc_rets[-m:], this_rets[-m:])[0, 1])
                f["btc_corr"] = corr if math.isfinite(corr) else 0.0
            else:
                f["btc_corr"] = 0.0
        else:
            f["btc_corr"] = 0.0

        # SPY beta (rolling OLS)
        spy_buf = self._buffers.get(self.spy_symbol)
        if spy_buf and symbol != self.spy_symbol and n >= _CORR_WINDOW:
            spy_rets = np.array(list(spy_buf.returns)[-_CORR_WINDOW:], dtype=float)
            this_rets = rets[-_CORR_WINDOW:]
            m = min(len(spy_rets), len(this_rets))
            if m >= 10:
                f["spy_beta"] = _rolling_beta(this_rets[-m:], spy_rets[-m:])
            else:
                f["spy_beta"] = 0.0
        else:
            f["spy_beta"] = 0.0

        return f

    # ------------------------------------------------------------------
    # Drift monitoring
    # ------------------------------------------------------------------

    def _check_drift(
        self,
        symbol: str,
        buf: _SymbolBuffer,
        ts: datetime,
        features: Dict[str, float],
    ) -> List[FeatureDriftAlert]:
        alerts = []
        for fname, val in features.items():
            if not math.isfinite(val):
                continue
            key = (symbol, fname)
            if fname not in buf.drift_stats:
                buf.drift_stats[fname] = _RollingStats(window=_DRIFT_WINDOW)
            stats = buf.drift_stats[fname]
            z = stats.z_score(val)
            mean, std = stats.update(val)
            if abs(z) > self.drift_threshold and len(stats._buf) > 50:
                alerts.append(FeatureDriftAlert(
                    symbol=symbol,
                    feature_name=fname,
                    ts=ts.timestamp(),
                    value=val,
                    baseline_mean=mean,
                    baseline_std=std,
                    z_score=z,
                ))
        return alerts

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _get_buffer(self, symbol: str) -> _SymbolBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = _SymbolBuffer(symbol=symbol)
        return self._buffers[symbol]

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------

    def build_app(self):
        """
        Build a FastAPI app exposing GET /features/{symbol}.

        Usage:
            app = store.build_app()
            uvicorn.run(app, host="0.0.0.0", port=8000)
        """
        try:
            from fastapi import FastAPI, HTTPException, Query  # type: ignore
        except ImportError as e:
            raise ImportError("fastapi is required for the REST API: pip install fastapi uvicorn") from e

        app = FastAPI(title="FeatureStore API", version="1.0")

        store_ref = self  # capture self in closure

        @app.get("/features/{symbol}")
        def get_features(
            symbol: str,
            as_of: Optional[str] = Query(None, description="ISO timestamp, e.g. 2024-01-01T12:00:00"),
            feature_names: Optional[str] = Query(None, description="Comma-separated feature names"),
        ) -> Dict[str, Any]:
            as_of_dt: Optional[datetime] = None
            if as_of:
                try:
                    as_of_dt = datetime.fromisoformat(as_of)
                    if as_of_dt.tzinfo is None:
                        as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid as_of format: {as_of}")

            names: Optional[List[str]] = None
            if feature_names:
                names = [n.strip() for n in feature_names.split(",") if n.strip()]

            result = store_ref.get_features(symbol, as_of_dt, names)
            if not result:
                raise HTTPException(status_code=404, detail=f"No features found for {symbol}")
            return {
                "symbol": symbol,
                "as_of": as_of or "latest",
                "features": result,
                "count": len(result),
            }

        @app.get("/symbols")
        def list_symbols() -> Dict[str, Any]:
            return {"symbols": sorted(store_ref._buffers.keys())}

        @app.get("/drift_alerts")
        def get_drift_alerts(limit: int = 50) -> Dict[str, Any]:
            alerts = store_ref._drift_alerts[-limit:]
            return {
                "alerts": [
                    {
                        "symbol": a.symbol,
                        "feature": a.feature_name,
                        "value": a.value,
                        "z_score": a.z_score,
                        "baseline_mean": a.baseline_mean,
                        "baseline_std": a.baseline_std,
                    }
                    for a in alerts
                ]
            }

        return app

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"FeatureStore — {len(self._buffers)} symbols"]
        for sym, buf in sorted(self._buffers.items()):
            lines.append(f"  {sym:10s}  bars={buf.bar_count}")
        n_alerts = len(self._drift_alerts)
        lines.append(f"  Drift alerts total: {n_alerts}")
        if self._drift_alerts:
            recent = self._drift_alerts[-3:]
            for a in recent:
                lines.append(
                    f"    DRIFT {a.symbol}/{a.feature_name}  z={a.z_score:.1f}  "
                    f"val={a.value:.4f}  mean={a.baseline_mean:.4f}"
                )
        return "\n".join(lines)

    def close(self) -> None:
        self._backend.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import csv
    from pathlib import Path

    db_path = Path("/tmp/features_demo.duckdb")
    store = FeatureStore(db_path=db_path)

    csv_path = Path(__file__).parent.parent.parent / "data" / "NDX_hourly_poly.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path)))

        def _g(row: dict, *keys: str) -> float:
            for k in keys:
                v = row.get(k)
                if v:
                    try:
                        return float(v)
                    except Exception:
                        pass
            return 0.0

        for i, row in enumerate(rows[:500]):
            ts_str = row.get("date", row.get("Date", ""))
            try:
                ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(tz=timezone.utc)
            c = _g(row, "close", "Close")
            o = _g(row, "open", "Open") or c
            h = _g(row, "high", "High") or c
            l = _g(row, "low", "Low") or c
            v = _g(row, "volume", "Volume") or 1000.0
            feats = store.update("NDX", o, h, l, c, v, ts)
    else:
        rng = np.random.default_rng(5)
        closes = 15000.0 + np.cumsum(rng.normal(0, 30, 500))
        for i, c in enumerate(closes):
            ts = datetime.utcfromtimestamp(float(i * 3600)).replace(tzinfo=timezone.utc)
            feats = store.update("SIM", c * 0.999, c * 1.001, c * 0.997, c, 1000.0, ts)

    print(store.summary())

    latest = store.get_features("NDX" if csv_path.exists() else "SIM")
    print(f"\nFeatures ({len(latest)}):")
    for k, v in sorted(latest.items()):
        print(f"  {k:25s} = {v:.6f}")

    store.close()


# app factory for uvicorn (module-level)
def _make_app():
    store = FeatureStore()
    return store.build_app()


# Support:  uvicorn tools.regime_ml.feature_store:app
try:
    app = _make_app()
except Exception:
    app = None  # type: ignore


if __name__ == "__main__":
    _demo()
