"""
changepoint.py — Real-time changepoint detection for market regime transitions.

Implements three complementary approaches and combines them into an ensemble:

1.  ruptures PELT   — exact offline segmentation on buffered windows
2.  ruptures BinSeg — approximate binary segmentation, fast on long buffers
3.  ruptures Window — sliding-window cost comparison
4.  BOCPD           — Bayesian Online Changepoint Detection (Adams & MacKay 2007)
                       implemented from scratch, no external dependency

The ensemble flags a changepoint event when >=2 methods agree within a
configurable time-tolerance.

Tracks four series per symbol
    - returns          (log-return series)
    - volatility       (rolling absolute returns as a vol proxy)
    - bh_mass          (external BH/pressure mass, injected by caller)
    - correlation      (rolling pairwise correlation supplied by caller)

Persistence
    Changepoints are stored in SQLite → changepoints.db
    Schema: symbol TEXT, ts REAL, method TEXT, confidence REAL, series TEXT

Stream interface
    detector.update(symbol, price, ts, bh_mass?, corr?) → Optional[ChangepointEvent]
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional ruptures import — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import ruptures as rpt  # type: ignore

    _RUPTURES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RUPTURES_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "changepoints.db"
_MIN_BUFFER = 40          # minimum bars before any detection fires
_ENSEMBLE_AGREE = 2       # votes needed to raise an event
_TIME_TOLERANCE = 5       # bars — methods must agree within this window


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChangepointEvent:
    """A confirmed regime-transition signal from the ensemble."""
    symbol: str
    ts: float               # UNIX timestamp
    bar_index: int
    methods: List[str]      # which detectors voted
    confidence: float       # 0–1, fraction of detectors that agreed
    series: str             # which series triggered (returns/vol/bh_mass/corr)
    magnitude: float        # abs change in mean before/after


@dataclass
class _SingleDetection:
    """Internal: a raw changepoint from one detector."""
    bar_index: int
    method: str
    series: str
    confidence: float
    magnitude: float


# ---------------------------------------------------------------------------
# BOCPD implementation (Adams & MacKay 2007)
# ---------------------------------------------------------------------------

class BOCPDDetector:
    """
    Bayesian Online Changepoint Detection with a Gaussian observation model
    and a conjugate Normal-Inverse-Gamma prior.

    Parameters
    ----------
    hazard_lambda : float
        Expected run length between changepoints.  Smaller = more sensitive.
    mu0, kappa0, alpha0, beta0 : float
        Normal-Inverse-Gamma prior hyper-parameters.
    threshold : float
        Posterior probability P(run_length=0) threshold to declare a CP.
    """

    def __init__(
        self,
        hazard_lambda: float = 200.0,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        threshold: float = 0.35,
    ) -> None:
        self.hazard = 1.0 / hazard_lambda
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.threshold = threshold

        # Sufficient statistics for each run-length hypothesis
        # index k = run length k (0 = new segment)
        self._R: np.ndarray = np.array([1.0])   # run-length distribution
        self._mus: np.ndarray = np.array([mu0])
        self._kappas: np.ndarray = np.array([kappa0])
        self._alphas: np.ndarray = np.array([alpha0])
        self._betas: np.ndarray = np.array([beta0])
        self._t: int = 0

    # ------------------------------------------------------------------
    def _predictive_prob(self, x: float) -> np.ndarray:
        """Student-t predictive probability for each run-length hypothesis."""
        df = 2.0 * self._alphas
        scale = np.sqrt(self._betas * (self._kappas + 1.0) / (self._alphas * self._kappas))
        # log of Student-t pdf
        log_norm = (
            math.lgamma((df[0] + 1.0) / 2.0) - math.lgamma(df[0] / 2.0)
            - 0.5 * np.log(df * math.pi)
            - np.log(scale)
        )
        # vectorise
        log_p = np.zeros(len(df))
        for i in range(len(df)):
            lg1 = math.lgamma((df[i] + 1.0) / 2.0)
            lg2 = math.lgamma(df[i] / 2.0)
            log_p[i] = (
                lg1 - lg2
                - 0.5 * math.log(df[i] * math.pi)
                - math.log(scale[i])
                - ((df[i] + 1.0) / 2.0) * math.log(1.0 + ((x - self._mus[i]) / scale[i]) ** 2 / df[i])
            )
        return np.exp(log_p - log_p.max())  # numerically stable

    def update(self, x: float) -> Tuple[bool, float]:
        """
        Consume one observation.

        Returns
        -------
        (is_changepoint, cp_probability)
        """
        if not math.isfinite(x):
            return False, 0.0

        pp = self._predictive_prob(x)

        # Growth probabilities (existing run lengths grow by 1)
        R_grow = self._R * pp * (1.0 - self.hazard)

        # Changepoint probability (all mass collapses to run-length 0)
        R_cp = np.sum(self._R * pp * self.hazard)

        # New run-length distribution: prepend CP prob
        R_new = np.concatenate(([R_cp], R_grow))

        # Normalise
        total = R_new.sum()
        if total < 1e-300:
            R_new = np.array([1.0])
            self._mus = np.array([self.mu0])
            self._kappas = np.array([self.kappa0])
            self._alphas = np.array([self.alpha0])
            self._betas = np.array([self.beta0])
        else:
            R_new /= total
            # Update NIG sufficient statistics for each hypothesis
            kappas_new = np.concatenate(([self.kappa0], self._kappas + 1.0))
            mus_new = np.concatenate((
                [self.mu0],
                (self._kappas * self._mus + x) / (self._kappas + 1.0),
            ))
            alphas_new = np.concatenate(([self.alpha0], self._alphas + 0.5))
            betas_new = np.concatenate((
                [self.beta0],
                self._betas
                + (self._kappas * (x - self._mus) ** 2) / (2.0 * (self._kappas + 1.0)),
            ))
            self._R = R_new
            self._mus = mus_new
            self._kappas = kappas_new
            self._alphas = alphas_new
            self._betas = betas_new

        self._t += 1
        cp_prob = float(self._R[0])
        return cp_prob >= self.threshold, cp_prob

    def reset(self) -> None:
        """Hard reset (e.g., after confirmed changepoint)."""
        self._R = np.array([1.0])
        self._mus = np.array([self.mu0])
        self._kappas = np.array([self.kappa0])
        self._alphas = np.array([self.alpha0])
        self._betas = np.array([self.beta0])
        self._t = 0


# ---------------------------------------------------------------------------
# Ruptures-based offline detectors (applied to rolling buffers)
# ---------------------------------------------------------------------------

def _ruptures_detect(
    signal: np.ndarray,
    method: str,
    pen: float,
    min_size: int = 10,
) -> List[int]:
    """
    Run ruptures PELT / BinSeg / Window on *signal* and return breakpoint
    bar indices (0-based, relative to the signal array).
    """
    if not _RUPTURES_AVAILABLE or len(signal) < 2 * min_size:
        return []
    try:
        if method == "pelt":
            algo = rpt.Pelt(model="rbf", min_size=min_size, jump=1).fit(signal)
            bkps = algo.predict(pen=pen)
        elif method == "binseg":
            algo = rpt.Binseg(model="rbf", min_size=min_size, jump=2).fit(signal)
            bkps = algo.predict(n_bkps=max(1, len(signal) // 50))
        elif method == "window":
            w = max(min_size, len(signal) // 10)
            algo = rpt.Window(width=w, model="rbf").fit(signal)
            bkps = algo.predict(n_bkps=max(1, len(signal) // 50))
        else:
            return []
        # ruptures returns 1-based inclusive endpoints; exclude the last sentinel
        return [b - 1 for b in bkps if b < len(signal)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Per-symbol state
# ---------------------------------------------------------------------------

@dataclass
class _SymbolState:
    symbol: str
    buffer_size: int
    # Series buffers
    prices: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    returns: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    vol: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    bh_mass: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    corr: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=600))
    bar_count: int = 0

    # BOCPD detectors — one per series
    bocpd_returns: BOCPDDetector = field(default_factory=lambda: BOCPDDetector(hazard_lambda=150))
    bocpd_vol: BOCPDDetector = field(default_factory=lambda: BOCPDDetector(hazard_lambda=150))
    bocpd_bh: BOCPDDetector = field(default_factory=lambda: BOCPDDetector(hazard_lambda=150))
    bocpd_corr: BOCPDDetector = field(default_factory=lambda: BOCPDDetector(hazard_lambda=150))

    # Recent raw detections waiting to be consolidated into ensemble events
    pending: List[_SingleDetection] = field(default_factory=list)
    # Bar index of last emitted ensemble event (to avoid duplicates)
    last_event_bar: int = -999


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class ChangepointDetector:
    """
    Real-time changepoint detection ensemble.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database where events are persisted.
    buffer_size : int
        Number of bars to keep in the rolling buffer for ruptures methods.
    sensitivity : float
        Controls ruptures penalty (lower = more changepoints detected).
        Range 0.1 – 10.0, default 1.0.
    ensemble_agree : int
        How many methods must agree for an event to be emitted.
    vol_window : int
        Rolling window for volatility proxy.
    """

    def __init__(
        self,
        db_path: str | Path = _DEFAULT_DB,
        buffer_size: int = 200,
        sensitivity: float = 1.0,
        ensemble_agree: int = _ENSEMBLE_AGREE,
        vol_window: int = 20,
    ) -> None:
        self.db_path = Path(db_path)
        self.buffer_size = buffer_size
        self.sensitivity = max(0.1, float(sensitivity))
        self.ensemble_agree = ensemble_agree
        self.vol_window = vol_window

        self._states: Dict[str, _SymbolState] = {}
        self._db_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS changepoints (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol  TEXT    NOT NULL,
                ts      REAL    NOT NULL,
                bar_idx INTEGER NOT NULL,
                method  TEXT    NOT NULL,
                confidence REAL NOT NULL,
                series  TEXT    NOT NULL,
                magnitude REAL  NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS cp_sym_ts ON changepoints (symbol, ts)")
        conn.commit()
        self._db_conn = conn

    def _persist(self, symbol: str, ts: float, bar_idx: int, methods: List[str],
                 confidence: float, series: str, magnitude: float) -> None:
        assert self._db_conn is not None
        self._db_conn.execute(
            "INSERT INTO changepoints (symbol, ts, bar_idx, method, confidence, series, magnitude) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (symbol, ts, bar_idx, ",".join(methods), confidence, series, magnitude),
        )
        self._db_conn.commit()

    # ------------------------------------------------------------------
    # Symbol state
    # ------------------------------------------------------------------

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(
                symbol=symbol,
                buffer_size=self.buffer_size,
            )
        return self._states[symbol]

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        symbol: str,
        price: float,
        ts: Optional[float] = None,
        bh_mass: Optional[float] = None,
        corr: Optional[float] = None,
    ) -> Optional[ChangepointEvent]:
        """
        Ingest a new price observation.

        Parameters
        ----------
        symbol  : ticker string
        price   : last trade price
        ts      : UNIX timestamp; defaults to time.time()
        bh_mass : external BH pressure mass (0–1), optional
        corr    : rolling pairwise correlation scalar, optional

        Returns
        -------
        ChangepointEvent if the ensemble agrees, else None.
        """
        if ts is None:
            ts = time.time()
        st = self._get_state(symbol)

        # ---- compute series values ----
        if len(st.prices) > 0:
            ret = math.log(price / st.prices[-1]) if st.prices[-1] > 0 else 0.0
        else:
            ret = 0.0

        st.prices.append(price)
        st.returns.append(ret)
        st.timestamps.append(ts)
        st.bar_count += 1

        # volatility proxy: rolling abs-return mean
        _vol_buf = list(st.returns)[-self.vol_window:]
        vol_val = float(np.mean(np.abs(_vol_buf))) if _vol_buf else 0.0
        st.vol.append(vol_val)

        bh_val = bh_mass if bh_mass is not None else 0.0
        corr_val = corr if corr is not None else 0.0
        st.bh_mass.append(bh_val)
        st.corr.append(corr_val)

        n = st.bar_count
        if n < _MIN_BUFFER:
            return None

        detections: List[_SingleDetection] = []

        # ---- 1. BOCPD on each series ----
        bocpd_pairs = [
            (st.bocpd_returns, ret, "returns"),
            (st.bocpd_vol, vol_val, "vol"),
        ]
        if bh_mass is not None:
            bocpd_pairs.append((st.bocpd_bh, bh_val, "bh_mass"))
        if corr is not None:
            bocpd_pairs.append((st.bocpd_corr, corr_val, "corr"))

        for bocpd, val, series_name in bocpd_pairs:
            is_cp, cp_prob = bocpd.update(val)
            if is_cp:
                mag = self._compute_magnitude(st, series_name)
                detections.append(_SingleDetection(
                    bar_index=n,
                    method="bocpd",
                    series=series_name,
                    confidence=cp_prob,
                    magnitude=mag,
                ))

        # ---- 2. Ruptures (every N bars to avoid O(n^2) overhead) ----
        if n % 10 == 0 and _RUPTURES_AVAILABLE:
            detections.extend(self._run_ruptures(st, n))

        # ---- Ensemble: accumulate pending detections ----
        st.pending.extend(detections)
        # Prune old pending entries
        st.pending = [d for d in st.pending if n - d.bar_index <= _TIME_TOLERANCE]

        event = self._check_ensemble(st, ts, n)
        return event

    # ------------------------------------------------------------------
    # Ruptures sweep
    # ------------------------------------------------------------------

    def _run_ruptures(self, st: _SymbolState, n: int) -> List[_SingleDetection]:
        pen = 5.0 * self.sensitivity
        results: List[_SingleDetection] = []
        series_map = {
            "returns": list(st.returns),
            "vol": list(st.vol),
            "bh_mass": list(st.bh_mass),
        }
        for series_name, data in series_map.items():
            arr = np.array(data, dtype=float)
            if len(arr) < _MIN_BUFFER:
                continue
            for method in ("pelt", "binseg", "window"):
                bkps = _ruptures_detect(arr.reshape(-1, 1), method, pen=pen)
                # Only care about recent breakpoints (last 10 bars relative to buffer)
                for b in bkps:
                    relative_bar = n - (len(arr) - 1 - b)
                    if abs(relative_bar - n) <= _TIME_TOLERANCE:
                        mag = self._compute_magnitude_at(arr, b)
                        results.append(_SingleDetection(
                            bar_index=relative_bar,
                            method=f"ruptures_{method}",
                            series=series_name,
                            confidence=0.7,
                            magnitude=mag,
                        ))
        return results

    # ------------------------------------------------------------------
    # Ensemble logic
    # ------------------------------------------------------------------

    def _check_ensemble(
        self, st: _SymbolState, ts: float, n: int
    ) -> Optional[ChangepointEvent]:
        if n - st.last_event_bar < 15:
            # Suppress events too close together
            return None

        # Group by series to vote independently per series
        by_series: Dict[str, List[_SingleDetection]] = defaultdict(list)
        for d in st.pending:
            by_series[d.series].append(d)

        for series_name, ds in by_series.items():
            unique_methods = {d.method for d in ds}
            if len(unique_methods) >= self.ensemble_agree:
                # Emit event
                methods = sorted(unique_methods)
                confidence = min(1.0, len(unique_methods) / 4.0)
                magnitude = float(np.mean([d.magnitude for d in ds]))
                event = ChangepointEvent(
                    symbol=st.symbol,
                    ts=ts,
                    bar_index=n,
                    methods=methods,
                    confidence=confidence,
                    series=series_name,
                    magnitude=magnitude,
                )
                self._persist(
                    st.symbol, ts, n, methods, confidence, series_name, magnitude
                )
                st.pending.clear()
                st.last_event_bar = n
                return event

        return None

    # ------------------------------------------------------------------
    # Magnitude helpers
    # ------------------------------------------------------------------

    def _compute_magnitude(self, st: _SymbolState, series_name: str) -> float:
        buf_map = {
            "returns": st.returns,
            "vol": st.vol,
            "bh_mass": st.bh_mass,
            "corr": st.corr,
        }
        buf = list(buf_map.get(series_name, st.returns))
        if len(buf) < 20:
            return 0.0
        half = len(buf) // 2
        return float(abs(np.mean(buf[-half:]) - np.mean(buf[:half])))

    @staticmethod
    def _compute_magnitude_at(arr: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(arr) - 1:
            return 0.0
        before = arr[:idx]
        after = arr[idx:]
        if len(before) == 0 or len(after) == 0:
            return 0.0
        return float(abs(np.mean(after) - np.mean(before)))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_history(self, symbol: str, limit: int = 100) -> List[dict]:
        """Return most recent changepoints for a symbol from the DB."""
        assert self._db_conn is not None
        rows = self._db_conn.execute(
            "SELECT ts, bar_idx, method, confidence, series, magnitude "
            "FROM changepoints WHERE symbol=? ORDER BY ts DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [
            {
                "ts": r[0],
                "bar_idx": r[1],
                "method": r[2],
                "confidence": r[3],
                "series": r[4],
                "magnitude": r[5],
            }
            for r in rows
        ]

    def close(self) -> None:
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def _demo() -> None:
    import csv
    import sys
    from pathlib import Path

    csv_path = Path(__file__).parent.parent.parent / "data" / "NDX_hourly_poly.csv"
    if not csv_path.exists():
        print("Demo CSV not found — generating synthetic prices")
        rng = np.random.default_rng(42)
        prices = 15000.0 + np.cumsum(rng.normal(0, 50, 500))
        # inject a regime shift at bar 250
        prices[250:] += 500
    else:
        rows = list(csv.DictReader(open(csv_path)))
        prices = np.array([float(r.get("close", r.get("Close", 0))) for r in rows[:500]])

    db_path = Path("/tmp/cp_demo.db")
    detector = ChangepointDetector(db_path=db_path, sensitivity=1.0)
    events = []
    for i, p in enumerate(prices):
        ev = detector.update("NDX", float(p), ts=float(i))
        if ev:
            events.append(ev)
            print(
                f"  bar={ev.bar_index:4d}  series={ev.series:<10s}  "
                f"methods={ev.methods}  conf={ev.confidence:.2f}  mag={ev.magnitude:.6f}"
            )

    print(f"\nTotal changepoint events: {len(events)}")
    detector.close()


if __name__ == "__main__":
    _demo()
