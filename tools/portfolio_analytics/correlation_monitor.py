# ============================================================
# correlation_monitor.py
# Real-time correlation monitoring for quantitative trading
# ============================================================

from __future__ import annotations

import json
import logging
import sqlite3
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# ---- Configuration -------------------------------------------------------

CROWDING_ALERT_THRESHOLD = 0.70   # avg corr > this → crowding risk
STRESS_SPIKE_THRESHOLD = 0.85     # corr > this → risk-off event
DEFAULT_EWMA_HALFLIFE = 20        # bars
DEFAULT_ROLLING_WINDOW = 60       # bars
N_REGIMES = 3                      # K-means clusters
DB_NAME = "correlation_history.db"


# ---- Data classes ---------------------------------------------------------

@dataclass
class CorrelationSnapshot:
    timestamp: datetime
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    kendall: pd.DataFrame
    ewm_corr: pd.DataFrame
    avg_pearson: float
    avg_spearman: float
    regime: int
    is_stress: bool
    centrality: dict[str, float]
    mst_edges: list[tuple[str, str, float]]


@dataclass
class CorrelationAlert:
    timestamp: datetime
    alert_type: str          # 'crowding' | 'stress' | 'regime_change'
    severity: str            # 'info' | 'warning' | 'critical'
    avg_corr: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---- DB helper ------------------------------------------------------------

class _CorrelationDB:
    """Thin SQLite wrapper for persisting correlation snapshots."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS corr_snapshots (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                avg_pearson REAL,
                avg_spearman REAL,
                regime     INTEGER,
                is_stress  INTEGER,
                payload    TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS corr_alerts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         TEXT    NOT NULL,
                alert_type TEXT,
                severity   TEXT,
                avg_corr   REAL,
                message    TEXT
            )
        """)
        self._conn.commit()

    def save_snapshot(self, snap: CorrelationSnapshot) -> None:
        payload = {
            "pearson": snap.pearson.to_dict(),
            "centrality": snap.centrality,
            "mst_edges": snap.mst_edges,
        }
        self._conn.execute(
            "INSERT INTO corr_snapshots (ts, avg_pearson, avg_spearman, regime, is_stress, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                snap.timestamp.isoformat(),
                snap.avg_pearson,
                snap.avg_spearman,
                snap.regime,
                int(snap.is_stress),
                json.dumps(payload),
            ),
        )
        self._conn.commit()

    def save_alert(self, alert: CorrelationAlert) -> None:
        self._conn.execute(
            "INSERT INTO corr_alerts (ts, alert_type, severity, avg_corr, message) VALUES (?, ?, ?, ?, ?)",
            (alert.timestamp.isoformat(), alert.alert_type, alert.severity, alert.avg_corr, alert.message),
        )
        self._conn.commit()

    def load_avg_pearson_series(self, limit: int = 500) -> pd.Series:
        rows = self._conn.execute(
            "SELECT ts, avg_pearson FROM corr_snapshots ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows, columns=["ts", "avg_pearson"])
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")["avg_pearson"].sort_index()

    def close(self) -> None:
        self._conn.close()


# ---- Main class -----------------------------------------------------------

class CorrelationMonitor:
    """
    Real-time rolling correlation monitor.

    Supports:
    - Rolling Pearson, Spearman, Kendall correlations
    - DCC-GARCH proxy via exponentially weighted correlation
    - Regime detection via K-means on the upper-triangle of the corr matrix
    - Stress detection (risk-off signal)
    - NetworkX graph with centrality and MST
    - SQLite persistence
    - Plotly heatmap export
    """

    def __init__(
        self,
        symbols: list[str],
        window: int = DEFAULT_ROLLING_WINDOW,
        ewm_halflife: int = DEFAULT_EWMA_HALFLIFE,
        n_regimes: int = N_REGIMES,
        corr_threshold: float = 0.4,
        db_path: str | Path = DB_NAME,
    ):
        self.symbols = list(symbols)
        self.n = len(symbols)
        self.window = window
        self.ewm_halflife = ewm_halflife
        self.n_regimes = n_regimes
        self.corr_threshold = corr_threshold

        self._returns: deque[pd.Series] = deque(maxlen=window + 50)
        self._log_vol: dict[str, deque] = {s: deque(maxlen=window) for s in symbols}
        self._regime_history: list[int] = []
        self._alerts: list[CorrelationAlert] = []

        self._db = _CorrelationDB(Path(db_path))
        self._kmeans: Optional[KMeans] = None
        self._last_snapshot: Optional[CorrelationSnapshot] = None

        logger.info("CorrelationMonitor initialised for %d symbols, window=%d", self.n, window)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, bar_returns: dict[str, float]) -> CorrelationSnapshot:
        """
        Feed one bar of returns for all symbols and recompute correlations.

        Parameters
        ----------
        bar_returns : {symbol: return_value}

        Returns
        -------
        CorrelationSnapshot
        """
        row = pd.Series({s: bar_returns.get(s, np.nan) for s in self.symbols})
        self._returns.append(row)

        # Update vol estimates for DCC proxy
        for s in self.symbols:
            r = bar_returns.get(s, np.nan)
            if not np.isnan(r):
                self._log_vol[s].append(abs(r))

        if len(self._returns) < 5:
            # Not enough data yet, return empty snapshot
            return self._empty_snapshot()

        ret_df = pd.DataFrame(list(self._returns))

        pearson = self._rolling_pearson(ret_df)
        spearman = self._rolling_spearman(ret_df)
        kendall = self._rolling_kendall(ret_df)
        ewm_corr = self._ewm_correlation(ret_df)

        avg_pearson = self._average_correlation(pearson)
        avg_spearman = self._average_correlation(spearman)

        regime = self._detect_regime(pearson)
        is_stress = self._detect_stress(pearson)

        centrality: dict[str, float] = {}
        mst_edges: list[tuple[str, str, float]] = []
        if HAS_NX:
            G = self._build_graph(pearson)
            centrality = self._compute_centrality(G)
            mst_edges = self._minimum_spanning_tree(pearson)

        snap = CorrelationSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            pearson=pearson,
            spearman=spearman,
            kendall=kendall,
            ewm_corr=ewm_corr,
            avg_pearson=avg_pearson,
            avg_spearman=avg_spearman,
            regime=regime,
            is_stress=is_stress,
            centrality=centrality,
            mst_edges=mst_edges,
        )

        self._last_snapshot = snap
        self._db.save_snapshot(snap)
        self._check_alerts(snap)
        self._regime_history.append(regime)

        return snap

    def get_last_snapshot(self) -> Optional[CorrelationSnapshot]:
        return self._last_snapshot

    def get_alerts(self, n: int = 20) -> list[CorrelationAlert]:
        return self._alerts[-n:]

    def export_heatmap(self, output_path: str = "correlation_heatmap.html") -> None:
        """Export a Plotly interactive heatmap of the latest Pearson matrix."""
        if not HAS_PLOTLY:
            logger.warning("plotly not installed; cannot export heatmap")
            return
        if self._last_snapshot is None:
            logger.warning("No snapshot available yet")
            return

        corr = self._last_snapshot.pearson
        z = corr.values.tolist()
        labels = list(corr.columns)

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in z],
                texttemplate="%{text}",
                hovertemplate="<b>%{x} / %{y}</b><br>corr = %{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Correlation Matrix — {self._last_snapshot.timestamp.strftime('%Y-%m-%d %H:%M')}",
            width=800,
            height=800,
        )
        fig.write_html(output_path)
        logger.info("Heatmap written to %s", output_path)

    def export_regime_heatmap(self, output_path: str = "regime_heatmap.html") -> None:
        """Export a dendrogram-ordered heatmap grouped by correlation regime."""
        if not HAS_PLOTLY:
            return
        if self._last_snapshot is None:
            return

        corr = self._last_snapshot.pearson
        dist = 1 - corr.abs()
        np.fill_diagonal(dist.values, 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = ff.create_dendrogram(dist.values, labels=list(corr.columns), orientation="left")

        fig.update_layout(title="Correlation Dendrogram")
        fig.write_html(output_path)
        logger.info("Regime heatmap written to %s", output_path)

    # ------------------------------------------------------------------
    # Correlation computation methods
    # ------------------------------------------------------------------

    def _rolling_pearson(self, ret_df: pd.DataFrame) -> pd.DataFrame:
        tail = ret_df.tail(self.window).dropna(how="all")
        if len(tail) < 5:
            return pd.DataFrame(np.eye(self.n), index=self.symbols, columns=self.symbols)
        return tail.corr(method="pearson").fillna(0)

    def _rolling_spearman(self, ret_df: pd.DataFrame) -> pd.DataFrame:
        tail = ret_df.tail(self.window).dropna(how="all")
        if len(tail) < 5:
            return pd.DataFrame(np.eye(self.n), index=self.symbols, columns=self.symbols)
        return tail.corr(method="spearman").fillna(0)

    def _rolling_kendall(self, ret_df: pd.DataFrame) -> pd.DataFrame:
        tail = ret_df.tail(self.window).dropna(how="all")
        if len(tail) < 5:
            return pd.DataFrame(np.eye(self.n), index=self.symbols, columns=self.symbols)
        return tail.corr(method="kendall").fillna(0)

    def _ewm_correlation(self, ret_df: pd.DataFrame) -> pd.DataFrame:
        """
        DCC-GARCH proxy: compute EWMA correlation with vol-adjusted (standardised) returns.
        r_std[t] = r[t] / ewm_vol[t]
        Then apply EWM correlation to standardised returns.
        """
        alpha = 1 - np.exp(-np.log(2) / self.ewm_halflife)
        tail = ret_df.tail(self.window * 2).copy()
        # Vol-adjust each series
        for col in tail.columns:
            ewm_vol = tail[col].ewm(halflife=self.ewm_halflife).std()
            tail[col] = tail[col] / ewm_vol.replace(0, np.nan)
        # EWM correlation on vol-adjusted returns
        ewm_cov = tail.ewm(halflife=self.ewm_halflife).corr().iloc[-self.n :]
        # ewm_cov is MultiIndex — extract last slice
        try:
            last = ewm_cov.groupby(level=1).last()
            return last.reindex(index=self.symbols, columns=self.symbols).fillna(0)
        except Exception:
            return pd.DataFrame(np.eye(self.n), index=self.symbols, columns=self.symbols)

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(self, corr: pd.DataFrame) -> int:
        """
        K-means clustering on flattened upper-triangle of corr matrix.
        Returns cluster label 0..n_regimes-1.
        """
        triu = corr.values[np.triu_indices(self.n, k=1)]
        if np.all(np.isnan(triu)):
            return 0

        # Accumulate history of upper-triangles
        if not hasattr(self, "_regime_vectors"):
            self._regime_vectors: deque = deque(maxlen=500)
        self._regime_vectors.append(triu)

        if len(self._regime_vectors) < self.n_regimes * 10:
            # Not enough history for reliable clustering
            return 0

        X = np.array(self._regime_vectors)
        # Fit K-means if not yet fitted or periodically refit
        if self._kmeans is None or len(self._regime_vectors) % 50 == 0:
            self._kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            self._kmeans.fit(X)

        label = int(self._kmeans.predict(triu.reshape(1, -1))[0])
        return label

    # ------------------------------------------------------------------
    # Stress detection
    # ------------------------------------------------------------------

    def _detect_stress(self, corr: pd.DataFrame) -> bool:
        """
        Risk-off event: average off-diagonal correlation spikes toward 1.
        """
        avg = self._average_correlation(corr)
        return bool(avg > STRESS_SPIKE_THRESHOLD)

    def _average_correlation(self, corr: pd.DataFrame) -> float:
        """Return mean of upper-triangle (off-diagonal) correlations."""
        vals = corr.values
        mask = np.triu(np.ones_like(vals, dtype=bool), k=1)
        upper = vals[mask]
        if len(upper) == 0:
            return 0.0
        return float(np.nanmean(np.abs(upper)))

    # ------------------------------------------------------------------
    # Graph / network methods
    # ------------------------------------------------------------------

    def _build_graph(self, corr: pd.DataFrame) -> "nx.Graph":
        G = nx.Graph()
        G.add_nodes_from(self.symbols)
        for i, s1 in enumerate(self.symbols):
            for j, s2 in enumerate(self.symbols):
                if j <= i:
                    continue
                w = float(corr.iloc[i, j])
                if abs(w) >= self.corr_threshold:
                    G.add_edge(s1, s2, weight=w)
        return G

    def _compute_centrality(self, G: "nx.Graph") -> dict[str, float]:
        """Degree centrality weighted by edge correlation."""
        if G.number_of_edges() == 0:
            return {s: 0.0 for s in self.symbols}
        try:
            centrality = nx.betweenness_centrality(G, weight="weight")
        except Exception:
            centrality = nx.degree_centrality(G)
        return {k: round(v, 4) for k, v in centrality.items()}

    def _minimum_spanning_tree(
        self, corr: pd.DataFrame
    ) -> list[tuple[str, str, float]]:
        """
        Compute the MST of the distance matrix d = sqrt(2*(1-corr)).
        Returns list of (sym1, sym2, distance) edges.
        """
        n = self.n
        dist = np.sqrt(2 * (1 - corr.values.clip(-1, 1)))
        np.fill_diagonal(dist, 0)
        G = nx.from_numpy_array(dist)
        mapping = {i: self.symbols[i] for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        mst = nx.minimum_spanning_tree(G, weight="weight")
        edges = [
            (u, v, round(d["weight"], 4))
            for u, v, d in mst.edges(data=True)
        ]
        return edges

    # ------------------------------------------------------------------
    # Alert generation
    # ------------------------------------------------------------------

    def _check_alerts(self, snap: CorrelationSnapshot) -> None:
        now = snap.timestamp

        # Crowding risk
        if snap.avg_pearson > CROWDING_ALERT_THRESHOLD:
            alert = CorrelationAlert(
                timestamp=now,
                alert_type="crowding",
                severity="warning",
                avg_corr=snap.avg_pearson,
                message=(
                    f"Crowding risk: avg Pearson corr = {snap.avg_pearson:.3f} "
                    f"> {CROWDING_ALERT_THRESHOLD}"
                ),
                details={"regime": snap.regime},
            )
            self._alerts.append(alert)
            self._db.save_alert(alert)
            logger.warning("CROWDING ALERT: avg_corr=%.3f", snap.avg_pearson)

        # Stress / risk-off
        if snap.is_stress:
            alert = CorrelationAlert(
                timestamp=now,
                alert_type="stress",
                severity="critical",
                avg_corr=snap.avg_pearson,
                message=(
                    f"Risk-off stress event: avg corr = {snap.avg_pearson:.3f} "
                    f"> {STRESS_SPIKE_THRESHOLD}"
                ),
                details={"centrality": snap.centrality},
            )
            self._alerts.append(alert)
            self._db.save_alert(alert)
            logger.critical("STRESS EVENT: avg_corr=%.3f", snap.avg_pearson)

        # Regime change
        if len(self._regime_history) >= 2:
            prev = self._regime_history[-1]
            if prev != snap.regime:
                alert = CorrelationAlert(
                    timestamp=now,
                    alert_type="regime_change",
                    severity="info",
                    avg_corr=snap.avg_pearson,
                    message=f"Correlation regime changed: {prev} → {snap.regime}",
                )
                self._alerts.append(alert)
                self._db.save_alert(alert)
                logger.info("Regime change: %d → %d", prev, snap.regime)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _empty_snapshot(self) -> CorrelationSnapshot:
        eye = pd.DataFrame(np.eye(self.n), index=self.symbols, columns=self.symbols)
        return CorrelationSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            pearson=eye,
            spearman=eye,
            kendall=eye,
            ewm_corr=eye,
            avg_pearson=0.0,
            avg_spearman=0.0,
            regime=0,
            is_stress=False,
            centrality={s: 0.0 for s in self.symbols},
            mst_edges=[],
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the current state."""
        if self._last_snapshot is None:
            return {"status": "no data"}
        snap = self._last_snapshot
        return {
            "timestamp": snap.timestamp.isoformat(),
            "avg_pearson": snap.avg_pearson,
            "avg_spearman": snap.avg_spearman,
            "regime": snap.regime,
            "is_stress": snap.is_stress,
            "crowding_risk": snap.avg_pearson > CROWDING_ALERT_THRESHOLD,
            "top_connected": sorted(
                snap.centrality.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "mst_edge_count": len(snap.mst_edges),
            "n_alerts": len(self._alerts),
        }

    def close(self) -> None:
        self._db.close()


# ---- Standalone usage example --------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    symbols = ["BTC", "ETH", "SOL", "BNB", "AVAX", "LINK", "UNI", "AAVE"]
    monitor = CorrelationMonitor(symbols, window=60, db_path="corr_test.db")

    rng = np.random.default_rng(42)
    for bar in range(200):
        # Simulate correlated returns
        cov = np.full((len(symbols), len(symbols)), 0.4)
        np.fill_diagonal(cov, 1.0)
        rets = rng.multivariate_normal(np.zeros(len(symbols)), cov * 0.0001)
        bar_data = dict(zip(symbols, rets))
        snap = monitor.update(bar_data)
        if bar % 50 == 49:
            print(f"Bar {bar+1}: avg_pearson={snap.avg_pearson:.3f}, regime={snap.regime}, stress={snap.is_stress}")

    monitor.export_heatmap("test_heatmap.html")
    print(json.dumps(monitor.summary(), indent=2))
    monitor.close()
