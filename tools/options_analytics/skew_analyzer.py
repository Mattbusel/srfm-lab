"""
skew_analyzer.py — Volatility skew analysis and term-structure monitoring.

Features
--------
* 25-delta risk reversal (RR25) and 10-delta butterfly (BF10) per expiry
* SKEW-index proxy computed from option chain
* ATM vol term structure: 30 / 60 / 90 / 120 DTE
* Realized vs implied vol spread (vol risk premium)
* Historical skew percentile (1-year lookback from skew_history.db)
* Skew signal: extreme put skew → mean-reversion alert
* SQLite persistence of skew_history.db for backtesting

Usage
-----
    python skew_analyzer.py --symbol SPY
    python skew_analyzer.py --symbol SPY --plot
    python skew_analyzer.py --symbol NVDA --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm, percentileofscore

RISK_FREE_RATE = 0.0525
INSTRUMENTS = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA", "MSFT"]
DB_PATH = os.path.join(os.path.dirname(__file__), "skew_history.db")
REALIZED_VOL_WINDOW = 21  # trading days for HV calc


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SkewSnapshot:
    symbol: str
    as_of: str                # ISO date
    # Risk reversals (call IV - put IV at same delta)
    rr_25d: Optional[float]   # 25-delta risk reversal (nearest expiry ~30d)
    rr_10d: Optional[float]   # 10-delta risk reversal
    # Butterflies
    bf_25d: Optional[float]   # 25-delta butterfly
    bf_10d: Optional[float]   # 10-delta butterfly
    # ATM vol term structure
    atm_30d: Optional[float]
    atm_60d: Optional[float]
    atm_90d: Optional[float]
    atm_120d: Optional[float]
    # Skew metrics
    skew_index_proxy: Optional[float]   # approximated CBOE SKEW
    put_call_skew: Optional[float]      # OTM put IV / OTM call IV
    # Vol risk premium
    realized_vol_21d: Optional[float]
    vrp: Optional[float]                # ATM IV - realized vol
    # Percentiles vs history
    rr25_percentile: Optional[float]    # 0-100
    atm30_percentile: Optional[float]
    # Signal
    skew_signal: Optional[str]          # "extreme_put_skew" | "normal" | "call_skew"


@dataclass
class ExpirySkew:
    expiry: date
    dte: int
    atm_iv: float
    rr_25d: Optional[float]
    rr_10d: Optional[float]
    bf_25d: Optional[float]
    bf_10d: Optional[float]
    strikes: List[float] = field(default_factory=list)
    ivs: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BS helpers
# ---------------------------------------------------------------------------

def _bs_price(S: float, K: float, T: float, r: float, sig: float, otype: str) -> float:
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0) if otype == "call" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    disc = math.exp(-r * T)
    if otype == "call":
        return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S: float, K: float, T: float, r: float, sig: float, otype: str) -> float:
    if T <= 0 or sig <= 0:
        return 1.0 if (otype == "call" and S > K) else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
    return norm.cdf(d1) if otype == "call" else norm.cdf(d1) - 1.0


def _bs_iv(mkt: float, S: float, K: float, T: float, r: float, otype: str) -> Optional[float]:
    if mkt <= 0 or T <= 0:
        return None
    try:
        def obj(sig: float) -> float:
            return _bs_price(S, K, T, r, sig, otype) - mkt
        if obj(0.001) * obj(10.0) > 0:
            return None
        return float(brentq(obj, 0.001, 10.0, xtol=1e-7))
    except Exception:
        return None


def _strike_at_delta(
    S: float, T: float, r: float, target_delta: float, otype: str, iv_guess: float = 0.25
) -> float:
    """
    Find the strike K such that BS delta(S,K,T,r,iv_guess) == target_delta.
    Uses Newton's method on log-moneyness.
    """
    d_sign = 1.0 if otype == "call" else -1.0
    # d1 = N_inv(delta) for calls, N_inv(delta+1) for puts
    if otype == "call":
        d1_target = norm.ppf(target_delta)
    else:
        d1_target = norm.ppf(target_delta + 1.0)
    log_k = math.log(S) + (r + 0.5 * iv_guess**2) * T - d1_target * iv_guess * math.sqrt(T)
    return math.exp(log_k)


# ---------------------------------------------------------------------------
# Alpaca client (minimal)
# ---------------------------------------------------------------------------

class _AlpacaClient:
    DATA_URL = "https://data.alpaca.markets/v2"

    def __init__(self) -> None:
        self.key = os.environ.get("ALPACA_API_KEY", "")
        self.secret = os.environ.get("ALPACA_SECRET_KEY", "")
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
        })

    def latest_price(self, sym: str) -> float:
        try:
            r = self._session.get(f"{self.DATA_URL}/stocks/{sym}/quotes/latest", timeout=8)
            r.raise_for_status()
            q = r.json().get("quote", {})
            return (q.get("bp", 0) + q.get("ap", 0)) / 2.0
        except Exception:
            return 0.0

    def option_chain(self, sym: str) -> List[dict]:
        """Fetch all option snapshots, paginated."""
        url = f"{self.DATA_URL}/options/snapshots/{sym}"
        results = []
        params: dict = {"feed": "indicative", "limit": 1000}
        while True:
            try:
                r = self._session.get(url, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                warnings.warn(f"Chain fetch failed: {e}")
                break
            for sym_key, snap in data.get("snapshots", {}).items():
                snap["_sym"] = sym_key
                results.append(snap)
            pt = data.get("next_page_token")
            if not pt:
                break
            params["page_token"] = pt
        return results

    def bars(self, sym: str, start: str, end: str, timeframe: str = "1Day") -> List[dict]:
        """Fetch OHLCV bars for realized vol computation."""
        try:
            r = self._session.get(
                f"{self.DATA_URL}/stocks/{sym}/bars",
                params={"start": start, "end": end, "timeframe": timeframe, "limit": 500},
                timeout=10,
            )
            r.raise_for_status()
            return r.json().get("bars", [])
        except Exception:
            return []


# ---------------------------------------------------------------------------
# OCC symbol parser
# ---------------------------------------------------------------------------

def _parse_occ(sym: str) -> Optional[Tuple[date, str, float]]:
    try:
        for i in range(len(sym) - 1, -1, -1):
            if sym[i] in ("C", "P") and i >= 3:
                dt = datetime.strptime(sym[i - 6: i], "%y%m%d").date()
                otype = "call" if sym[i] == "C" else "put"
                strike = float(sym[i + 1:]) / 1000.0
                return dt, otype, strike
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# SQLite history store
# ---------------------------------------------------------------------------

class SkewHistoryDB:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skew_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    rr_25d REAL,
                    rr_10d REAL,
                    bf_25d REAL,
                    bf_10d REAL,
                    atm_30d REAL,
                    atm_60d REAL,
                    atm_90d REAL,
                    atm_120d REAL,
                    skew_index_proxy REAL,
                    put_call_skew REAL,
                    realized_vol_21d REAL,
                    vrp REAL,
                    UNIQUE(symbol, as_of)
                )
            """)
            conn.commit()

    def insert(self, snap: SkewSnapshot) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO skew_history
                (symbol, as_of, rr_25d, rr_10d, bf_25d, bf_10d,
                 atm_30d, atm_60d, atm_90d, atm_120d,
                 skew_index_proxy, put_call_skew, realized_vol_21d, vrp)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                snap.symbol, snap.as_of, snap.rr_25d, snap.rr_10d,
                snap.bf_25d, snap.bf_10d, snap.atm_30d, snap.atm_60d,
                snap.atm_90d, snap.atm_120d, snap.skew_index_proxy,
                snap.put_call_skew, snap.realized_vol_21d, snap.vrp,
            ))
            conn.commit()

    def fetch_history(self, symbol: str, days: int = 252) -> List[dict]:
        since = (date.today() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM skew_history WHERE symbol=? AND as_of>=? ORDER BY as_of",
                (symbol, since),
            ).fetchall()
        return [dict(r) for r in rows]

    def percentile(self, symbol: str, field_name: str, current_val: float, days: int = 252) -> Optional[float]:
        hist = self.fetch_history(symbol, days)
        vals = [r[field_name] for r in hist if r[field_name] is not None]
        if not vals:
            return None
        return float(percentileofscore(vals, current_val, kind="rank"))


# ---------------------------------------------------------------------------
# Main SkewAnalyzer
# ---------------------------------------------------------------------------

class SkewAnalyzer:
    """
    Compute comprehensive skew metrics from option chain data.
    """

    def __init__(
        self,
        symbol: str,
        r: float = RISK_FREE_RATE,
        db_path: str = DB_PATH,
    ):
        self.symbol = symbol.upper()
        self.r = r
        self.client = _AlpacaClient()
        self.db = SkewHistoryDB(db_path)
        self.underlying_price: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self) -> SkewSnapshot:
        """Full skew analysis pipeline."""
        print(f"[SkewAnalyzer] Fetching data for {self.symbol}...")
        self.underlying_price = self.client.latest_price(self.symbol)
        chain_raw = self.client.option_chain(self.symbol)
        print(f"[SkewAnalyzer] Raw contracts: {len(chain_raw)}, S={self.underlying_price:.2f}")

        expiry_slices = self._build_slices(chain_raw)
        print(f"[SkewAnalyzer] Expiry slices: {len(expiry_slices)}")

        realized_vol = self._realized_vol()
        atm_ivs = self._term_structure(expiry_slices)
        rr_bf = self._risk_reversals(expiry_slices)

        atm_30d = atm_ivs.get(30)
        vrp = (atm_30d - realized_vol) if (atm_30d and realized_vol) else None
        skew_index = self._skew_index_proxy(expiry_slices)
        pc_skew = self._put_call_skew(expiry_slices)

        snap = SkewSnapshot(
            symbol=self.symbol,
            as_of=date.today().isoformat(),
            rr_25d=rr_bf.get("rr_25d"),
            rr_10d=rr_bf.get("rr_10d"),
            bf_25d=rr_bf.get("bf_25d"),
            bf_10d=rr_bf.get("bf_10d"),
            atm_30d=atm_ivs.get(30),
            atm_60d=atm_ivs.get(60),
            atm_90d=atm_ivs.get(90),
            atm_120d=atm_ivs.get(120),
            skew_index_proxy=skew_index,
            put_call_skew=pc_skew,
            realized_vol_21d=realized_vol,
            vrp=vrp,
            rr25_percentile=None,
            atm30_percentile=None,
            skew_signal=None,
        )

        # Historical percentiles
        snap.rr25_percentile = (
            self.db.percentile(self.symbol, "rr_25d", snap.rr_25d)
            if snap.rr_25d is not None else None
        )
        snap.atm30_percentile = (
            self.db.percentile(self.symbol, "atm_30d", snap.atm_30d)
            if snap.atm_30d is not None else None
        )

        # Skew signal
        snap.skew_signal = self._skew_signal(snap)

        # Persist
        self.db.insert(snap)

        return snap

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_slices(self, chain_raw: List[dict]) -> List[ExpirySkew]:
        """Parse raw chain → per-expiry IV slices."""
        today = date.today()
        S = self.underlying_price
        expiry_map: Dict[date, Dict[str, List[Tuple[float, float]]]] = {}
        # {expiry: {"call": [(strike, iv), ...], "put": [(strike, iv), ...]}}

        for snap in chain_raw:
            sym = snap.get("_sym", "")
            parsed = _parse_occ(sym)
            if parsed is None:
                continue
            expiry, otype, strike = parsed
            dte = (expiry - today).days
            if dte < 5 or dte > 200:
                continue

            q = snap.get("latestQuote", {})
            bid = float(q.get("bp", 0) or 0)
            ask = float(q.get("ap", 0) or 0)
            mid = (bid + ask) / 2.0
            if mid < 0.01:
                continue

            T = max(dte / 365.0, 1e-6)
            iv = _bs_iv(mid, S, strike, T, self.r, otype)
            if iv is None or iv < 0.01 or iv > 5.0:
                continue

            expiry_map.setdefault(expiry, {"call": [], "put": []})
            expiry_map[expiry][otype].append((strike, iv))

        slices: List[ExpirySkew] = []
        for expiry, otype_data in sorted(expiry_map.items()):
            dte = (expiry - today).days
            T = max(dte / 365.0, 1e-6)

            # ATM IV: nearest strike to S using calls
            calls = sorted(otype_data["call"], key=lambda x: x[0])
            puts = sorted(otype_data["put"], key=lambda x: x[0])

            if len(calls) < 2 and len(puts) < 2:
                continue

            # Use put-call parity region for ATM
            all_strikes = sorted({s for s, _ in calls + puts})
            if not all_strikes:
                continue

            atm_strike = min(all_strikes, key=lambda k: abs(k - S))
            # ATM IV: average of call and put at ATM strike (put-call parity)
            atm_call_iv = next((iv for k, iv in calls if abs(k - atm_strike) < 0.01 * S), None)
            atm_put_iv = next((iv for k, iv in puts if abs(k - atm_strike) < 0.01 * S), None)
            atm_iv = np.mean([x for x in [atm_call_iv, atm_put_iv] if x is not None]) if (atm_call_iv or atm_put_iv) else 0.0
            if atm_iv <= 0:
                continue

            # Build unified strikes/ivs using calls above ATM, puts below ATM
            combined: Dict[float, float] = {}
            for k, iv in puts:
                if k <= atm_strike:
                    combined[k] = iv
            for k, iv in calls:
                if k >= atm_strike:
                    combined[k] = iv

            sl = ExpirySkew(
                expiry=expiry,
                dte=dte,
                atm_iv=float(atm_iv),
                rr_25d=None,
                rr_10d=None,
                bf_25d=None,
                bf_10d=None,
                strikes=sorted(combined.keys()),
                ivs=[combined[k] for k in sorted(combined.keys())],
            )
            slices.append(sl)

        return slices

    def _term_structure(self, slices: List[ExpirySkew]) -> Dict[int, float]:
        """Interpolate ATM IV at target DTEs."""
        if not slices:
            return {}
        dtes = np.array([sl.dte for sl in slices], dtype=float)
        ivs = np.array([sl.atm_iv for sl in slices])
        f = interp1d(dtes, ivs, kind="linear", bounds_error=False, fill_value=(ivs[0], ivs[-1]))
        result = {}
        for target in [30, 60, 90, 120]:
            if dtes[0] <= target <= dtes[-1]:
                result[target] = float(f(target))
            elif target < dtes[0]:
                result[target] = float(ivs[0])
            else:
                result[target] = float(ivs[-1])
        return result

    def _iv_at_delta(
        self, sl: ExpirySkew, target_delta: float, otype: str
    ) -> Optional[float]:
        """
        Find IV at the strike corresponding to target_delta for the given expiry slice.
        Uses cubic interpolation of the IV smile.
        """
        if len(sl.strikes) < 2:
            return None

        S = self.underlying_price
        T = max(sl.dte / 365.0, 1e-6)
        strikes = np.array(sl.strikes)
        ivs = np.array(sl.ivs)

        # Compute delta for each strike
        deltas = np.array([
            _bs_delta(S, k, T, self.r, iv, otype)
            for k, iv in zip(strikes, ivs)
        ])

        # Sort by delta for interpolation
        if otype == "put":
            # Puts: delta is negative; sort ascending by abs(delta)
            idx = np.argsort(np.abs(deltas))
            abs_deltas = np.abs(deltas[idx])
            sorted_ivs = ivs[idx]
            target = abs(target_delta)
        else:
            idx = np.argsort(deltas)[::-1]
            abs_deltas = deltas[idx]
            sorted_ivs = ivs[idx]
            target = abs(target_delta)

        # Interpolate
        if target < abs_deltas.min() or target > abs_deltas.max():
            return None
        try:
            f = interp1d(abs_deltas, sorted_ivs, kind="linear", bounds_error=True)
            return float(f(target))
        except Exception:
            return None

    def _risk_reversals(self, slices: List[ExpirySkew]) -> Dict[str, Optional[float]]:
        """
        Compute 25d and 10d risk reversals and butterflies.
        RR = call_iv(delta) - put_iv(delta)
        BF = (call_iv(delta) + put_iv(delta)) / 2 - atm_iv
        Uses the nearest-to-30DTE slice.
        """
        result: Dict[str, Optional[float]] = {
            "rr_25d": None, "rr_10d": None, "bf_25d": None, "bf_10d": None
        }

        # Find slice closest to 30 DTE
        target_slices = [sl for sl in slices if 15 <= sl.dte <= 60]
        if not target_slices:
            target_slices = slices
        if not target_slices:
            return result

        sl = min(target_slices, key=lambda s: abs(s.dte - 30))

        # 25-delta
        c25 = self._iv_at_delta(sl, 0.25, "call")
        p25 = self._iv_at_delta(sl, -0.25, "put")
        if c25 and p25:
            result["rr_25d"] = round(c25 - p25, 6)
            result["bf_25d"] = round((c25 + p25) / 2.0 - sl.atm_iv, 6)

        # 10-delta
        c10 = self._iv_at_delta(sl, 0.10, "call")
        p10 = self._iv_at_delta(sl, -0.10, "put")
        if c10 and p10:
            result["rr_10d"] = round(c10 - p10, 6)
            result["bf_10d"] = round((c10 + p10) / 2.0 - sl.atm_iv, 6)

        return result

    def _skew_index_proxy(self, slices: List[ExpirySkew]) -> Optional[float]:
        """
        CBOE SKEW proxy: 100 - 10 * (mean of log-strike-weighted put IV - ATM IV).
        Approximated as: 100 - 10*(p10_iv - atm_iv) for the front-month slice.
        CBOE SKEW > 130 indicates tail risk concerns.
        """
        target = [sl for sl in slices if 20 <= sl.dte <= 50]
        if not target:
            return None
        sl = min(target, key=lambda s: abs(s.dte - 30))
        p10 = self._iv_at_delta(sl, -0.10, "put")
        if p10 is None:
            return None
        # SKEW proxy based on 10d put premium
        skew = 100.0 - 10.0 * (p10 - sl.atm_iv) * 100.0
        return round(float(skew), 2)

    def _put_call_skew(self, slices: List[ExpirySkew]) -> Optional[float]:
        """Ratio of 25d put IV to 25d call IV (skew ratio > 1.2 = pronounced put skew)."""
        target = [sl for sl in slices if 20 <= sl.dte <= 50]
        if not target:
            return None
        sl = min(target, key=lambda s: abs(s.dte - 30))
        c25 = self._iv_at_delta(sl, 0.25, "call")
        p25 = self._iv_at_delta(sl, -0.25, "put")
        if c25 and p25 and c25 > 0:
            return round(p25 / c25, 4)
        return None

    def _realized_vol(self) -> Optional[float]:
        """Compute 21-day historical realized vol from daily closes."""
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=40)).isoformat()
        bars = self.client.bars(self.symbol, start, end)
        if len(bars) < 5:
            return None
        closes = np.array([b["c"] for b in bars if "c" in b])
        if len(closes) < 2:
            return None
        log_rets = np.diff(np.log(closes))
        window = min(REALIZED_VOL_WINDOW, len(log_rets))
        rv = float(np.std(log_rets[-window:]) * math.sqrt(252))
        return round(rv, 6)

    def _skew_signal(self, snap: SkewSnapshot) -> str:
        """Generate skew signal from current metrics."""
        # Extreme put skew: RR25 very negative, percentile > 85
        if (
            snap.rr_25d is not None
            and snap.rr_25d < -0.05
            and snap.rr25_percentile is not None
            and snap.rr25_percentile > 85
        ):
            return "extreme_put_skew"
        if (
            snap.rr_25d is not None
            and snap.rr_25d > 0.02
        ):
            return "call_skew"
        if snap.put_call_skew is not None and snap.put_call_skew > 1.25:
            return "elevated_put_skew"
        return "normal"

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, snap: SkewSnapshot) -> None:
        """Print a formatted skew report."""
        print(f"\n{'='*60}")
        print(f"  SKEW REPORT — {snap.symbol} — {snap.as_of}")
        print(f"{'='*60}")
        print(f"  Underlying:        {self.underlying_price:.2f}")
        print()
        print("  RISK REVERSALS")
        print(f"    25d RR:           {_fmt(snap.rr_25d, pct=True)}")
        print(f"    10d RR:           {_fmt(snap.rr_10d, pct=True)}")
        print()
        print("  BUTTERFLIES")
        print(f"    25d BF:           {_fmt(snap.bf_25d, pct=True)}")
        print(f"    10d BF:           {_fmt(snap.bf_10d, pct=True)}")
        print()
        print("  ATM TERM STRUCTURE")
        print(f"    30d ATM IV:       {_fmt(snap.atm_30d, pct=True)}")
        print(f"    60d ATM IV:       {_fmt(snap.atm_60d, pct=True)}")
        print(f"    90d ATM IV:       {_fmt(snap.atm_90d, pct=True)}")
        print(f"   120d ATM IV:       {_fmt(snap.atm_120d, pct=True)}")
        print()
        print("  VOL RISK PREMIUM")
        print(f"    21d Realized Vol: {_fmt(snap.realized_vol_21d, pct=True)}")
        print(f"    VRP (IV-RV):      {_fmt(snap.vrp, pct=True)}")
        print()
        print("  SKEW INDICES")
        print(f"    SKEW proxy:       {_fmt(snap.skew_index_proxy)}")
        print(f"    Put/Call Skew:    {_fmt(snap.put_call_skew)}")
        print()
        print("  HISTORICAL PERCENTILES (1yr)")
        print(f"    RR25 pct:         {_fmt(snap.rr25_percentile)}%")
        print(f"    ATM30 pct:        {_fmt(snap.atm30_percentile)}%")
        print()
        signal_color = "** " if snap.skew_signal != "normal" else "   "
        print(f"  SIGNAL:  {signal_color}{snap.skew_signal or 'normal'}")
        print(f"{'='*60}\n")

    def plot_term_structure(self, slices: List[ExpirySkew]) -> None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[SkewAnalyzer] plotly not installed.")
            return
        if not slices:
            return
        dtes = [sl.dte for sl in slices]
        atm_ivs = [sl.atm_iv * 100 for sl in slices]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dtes, y=atm_ivs, mode="lines+markers", name="ATM IV"))
        fig.update_layout(
            title=f"{self.symbol} ATM Vol Term Structure",
            xaxis_title="DTE",
            yaxis_title="IV (%)",
            template="plotly_dark",
        )
        fig.show()

    def plot_smile(self, slices: List[ExpirySkew]) -> None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[SkewAnalyzer] plotly not installed.")
            return
        fig = go.Figure()
        for sl in slices[:6]:
            fig.add_trace(go.Scatter(
                x=sl.strikes,
                y=[iv * 100 for iv in sl.ivs],
                mode="lines+markers",
                name=f"{sl.expiry} (DTE={sl.dte})",
            ))
        fig.add_vline(x=self.underlying_price, line_dash="dash", line_color="white")
        fig.update_layout(
            title=f"{self.symbol} Volatility Smile",
            xaxis_title="Strike",
            yaxis_title="IV (%)",
            template="plotly_dark",
        )
        fig.show()


def _fmt(val: Optional[float], pct: bool = False) -> str:
    if val is None:
        return "N/A"
    if pct:
        return f"{val*100:.3f}%"
    return f"{val:.4f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Volatility skew analyzer")
    parser.add_argument("--symbol", default="SPY", choices=INSTRUMENTS)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--report", action="store_true", default=True)
    parser.add_argument("--export", default=None, help="Export JSON snapshot")
    parser.add_argument("--db", default=DB_PATH, help="SQLite DB path")
    args = parser.parse_args()

    analyzer = SkewAnalyzer(args.symbol, db_path=args.db)
    snap = analyzer.analyze()

    if args.report:
        analyzer.report(snap)

    if args.export:
        import dataclasses
        with open(args.export, "w") as f:
            json.dump(dataclasses.asdict(snap), f, indent=2, default=str)
        print(f"[SkewAnalyzer] Exported to {args.export}")

    if args.plot:
        chain_raw = analyzer.client.option_chain(args.symbol)
        slices = analyzer._build_slices(chain_raw)
        analyzer.plot_term_structure(slices)
        analyzer.plot_smile(slices)


if __name__ == "__main__":
    _cli()
