"""
alternative_data/pipeline.py
================================
Orchestrates all alternative data fetchers, caches results, and emits
signals to the IAE hypothesis store.

Architecture
------------
  GoogleTrendsFetcher    ─┐
  GitHubActivityFetcher  ─┤
  FuturesOIFetcher       ─┤──→ AltDataPipeline.run_cycle()
  FundingRateFetcher     ─┤         │
  ExchangeFlowSimulator  ─┤         ├──→ DerivativesSignalBuilder
  LiquidationSimulator   ─┘         │
                                    ├──→ persist to alt_data_signals table
                                    └──→ emit hypothesis rows to hypotheses table

Signal-to-hypothesis thresholds
---------------------------------
  DerivativesSignal squeeze_setup + confidence > 0.55   → bullish hypothesis
  DerivativesSignal fade_setup    + confidence > 0.55   → bearish hypothesis
  ExchangeFlow bearish_distribution + signal_strength > 0.6 → bearish hypothesis
  ExchangeFlow bullish_accumulation + signal_strength > 0.6 → bullish hypothesis
  GoogleTrends "crypto buy" bullish_demand + acceleration > 3 → bullish hypothesis

Caching
-------
Each fetcher result is cached for CACHE_TTL_S seconds to avoid redundant
API calls when run_cycle() is called more frequently than necessary.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .google_trends      import GoogleTrendsFetcher, TrendSignal
from .github_activity    import GitHubActivityFetcher, RepoActivity
from .futures_oi         import FuturesOIFetcher, OISnapshot
from .funding_rates      import FundingRateFetcher, FundingRateSnapshot
from .exchange_flows     import ExchangeFlowSimulator, ExchangeFlowSnapshot
from .liquidations       import LiquidationSimulator, LiquidationSnapshot
from .derivatives_signal import DerivativesSignalBuilder, DerivativesSignal

logger = logging.getLogger(__name__)

_HERE      = Path(__file__).parent.parent
DEFAULT_DB = _HERE / "idea_engine.db"

CACHE_TTL_S: int = 900   # 15 minutes

# Hypothesis generation thresholds
DERIV_SIGNAL_CONF_THRESH:  float = 0.50
FLOW_SIGNAL_THRESH:        float = 0.55
TREND_ACCEL_THRESH:        float = 2.0

# DDL for alt_data_signals table
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS alt_data_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    symbol          TEXT    NOT NULL,
    source          TEXT    NOT NULL,    -- 'derivatives' | 'exchange_flow' | 'google_trends' | 'github'
    signal_label    TEXT    NOT NULL,
    composite_score REAL,
    confidence      REAL,
    payload_json    TEXT,
    hypothesis_id   INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_altdata_sym ON alt_data_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_altdata_ts  ON alt_data_signals(ts);
"""


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """
    Summary of one AltDataPipeline.run_cycle() execution.

    Attributes
    ----------
    derivatives_signals : List of DerivativesSignal
    exchange_flows      : List of ExchangeFlowSnapshot
    trend_signals       : List of TrendSignal
    github_activities   : List of RepoActivity
    hypotheses_created  : Count of hypothesis rows inserted
    duration_s          : Wall-clock time of cycle in seconds
    timestamp           : Cycle start time (UTC ISO)
    """
    derivatives_signals: list[DerivativesSignal]
    exchange_flows:      list[ExchangeFlowSnapshot]
    trend_signals:       list[TrendSignal]
    github_activities:   list[RepoActivity]
    hypotheses_created:  int
    duration_s:          float
    timestamp:           str

    def summary(self) -> dict:
        return {
            "derivatives_signals": len(self.derivatives_signals),
            "exchange_flows":      len(self.exchange_flows),
            "trend_signals":       len(self.trend_signals),
            "github_activities":   len(self.github_activities),
            "hypotheses_created":  self.hypotheses_created,
            "duration_s":          round(self.duration_s, 2),
            "timestamp":           self.timestamp,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AltDataPipeline:
    """
    Orchestrates all alternative data sources into a unified signal cycle.

    Parameters
    ----------
    db_path        : Path to idea_engine.db
    enable_github  : Whether to include GitHub activity (slow API calls)
    enable_trends  : Whether to include Google Trends (requires pytrends)
    """

    def __init__(
        self,
        db_path:        Path | str = DEFAULT_DB,
        enable_github:  bool       = True,
        enable_trends:  bool       = True,
    ) -> None:
        self.db_path       = Path(db_path)
        self.enable_github = enable_github
        self.enable_trends = enable_trends

        # Fetchers
        self._trends_fetcher    = GoogleTrendsFetcher()   if enable_trends else None
        self._github_fetcher    = GitHubActivityFetcher() if enable_github else None
        self._oi_fetcher        = FuturesOIFetcher()
        self._funding_fetcher   = FundingRateFetcher()
        self._flow_simulator    = ExchangeFlowSimulator()
        self._liq_simulator     = LiquidationSimulator()
        self._deriv_builder     = DerivativesSignalBuilder()

        # Cache store: {key: (data, cache_ts)}
        self._cache: dict[str, tuple[Any, float]] = {}

        self._ensure_schema()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run_cycle(self) -> CycleResult:
        """
        Execute a full alt-data collection cycle:
          1. Fetch all alt-data sources (with caching)
          2. Build composite derivatives signal
          3. Emit hypothesis rows for qualifying signals
          4. Persist to alt_data_signals table

        Returns
        -------
        CycleResult with all collected signals and metadata.
        """
        ts_start = time.monotonic()
        ts_now   = datetime.now(timezone.utc).isoformat()
        logger.info("AltDataPipeline: starting cycle …")

        # ── 1. Fetch ─────────────────────────────────────────────────
        oi_snaps      = self._cached("oi",      self._oi_fetcher.fetch_all)
        funding_snaps = self._cached("funding", self._funding_fetcher.fetch_all)
        flow_snaps    = self._cached("flows",   self._flow_simulator.fetch_all)
        liq_snaps     = self._cached("liq",     self._liq_simulator.fetch_all)

        trend_signals: list[TrendSignal] = []
        if self.enable_trends and self._trends_fetcher:
            trend_signals = self._cached("trends", self._trends_fetcher.fetch_all)

        github_activities: list[RepoActivity] = []
        if self.enable_github and self._github_fetcher:
            github_activities = self._cached("github", self._github_fetcher.fetch_all)

        # ── 2. Derivatives composite signals ─────────────────────────
        deriv_signals = self._deriv_builder.build_all(oi_snaps, funding_snaps, liq_snaps)

        # ── 3. Generate hypotheses ───────────────────────────────────
        n_hyps = self._emit_hypotheses(
            deriv_signals, flow_snaps, trend_signals, github_activities
        )

        # ── 4. Persist alt_data_signals rows ─────────────────────────
        self._persist_signals(deriv_signals, flow_snaps, trend_signals)

        duration = time.monotonic() - ts_start
        logger.info(
            "AltDataPipeline: cycle complete in %.1fs — %d hypotheses created.",
            duration, n_hyps,
        )

        return CycleResult(
            derivatives_signals=deriv_signals,
            exchange_flows=flow_snaps,
            trend_signals=trend_signals,
            github_activities=github_activities,
            hypotheses_created=n_hyps,
            duration_s=duration,
            timestamp=ts_now,
        )

    def query_latest_signals(self, symbol: str = None, limit: int = 200) -> list[dict]:
        """Query recent alt_data_signals from the DB."""
        conn = self._get_conn()
        try:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM alt_data_signals WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                    (symbol.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alt_data_signals ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Hypothesis emission                                                  #
    # ------------------------------------------------------------------ #

    def _emit_hypotheses(
        self,
        deriv:    list[DerivativesSignal],
        flows:    list[ExchangeFlowSnapshot],
        trends:   list[TrendSignal],
        github:   list[RepoActivity],
    ) -> int:
        """Generate hypothesis rows for qualifying signals."""
        rows = []

        # Derivatives signals
        for sig in deriv:
            if sig.confidence < DERIV_SIGNAL_CONF_THRESH:
                continue
            if sig.signal_label in ("squeeze_setup", "capitulation_bottom"):
                rows.append(self._make_deriv_hypothesis(sig, bullish=True))
            elif sig.signal_label in ("fade_setup",):
                rows.append(self._make_deriv_hypothesis(sig, bullish=False))

        # Exchange flow signals
        for flow in flows:
            if flow.signal_strength >= FLOW_SIGNAL_THRESH:
                rows.append(self._make_flow_hypothesis(flow))

        # Google Trends
        for ts in trends:
            if ts.signal_type == "bullish_demand" and ts.acceleration > TREND_ACCEL_THRESH:
                rows.append(self._make_trend_hypothesis(ts))

        if rows:
            conn = self._get_conn()
            try:
                for title, body, rationale, prior, priority, tags in rows:
                    conn.execute(
                        """
                        INSERT INTO hypotheses
                            (title, body, rationale, prior_prob, status, priority, tags)
                        VALUES (?, ?, ?, ?, 'open', ?, ?)
                        """,
                        (title, body, rationale, prior, priority, tags),
                    )
                conn.commit()
            except Exception as exc:
                logger.error("AltDataPipeline: hypothesis insert failed: %s", exc)
                conn.rollback()
            finally:
                conn.close()

        return len(rows)

    # ------------------------------------------------------------------ #
    # Hypothesis formatters                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_deriv_hypothesis(sig: DerivativesSignal, bullish: bool) -> tuple:
        direction = "Bullish" if bullish else "Bearish"
        prior     = min(0.75, 0.50 + sig.confidence * 0.30)
        title     = (
            f"[AltData-Deriv] {direction} derivatives signal for {sig.symbol} "
            f"— {sig.signal_label} (conf={sig.confidence:.2f})"
        )
        body = (
            f"Composite derivatives signal for {sig.symbol}: "
            f"label={sig.signal_label}, score={sig.composite_score:+.3f}, "
            f"confidence={sig.confidence:.2f}. "
            f"Components: funding_rate={sig.funding_rate:+.5f} ({sig.oi_regime} OI), "
            f"liquidation_cascade={sig.liq_cascade}, dominant_liq_side={sig.dominant_liq_side}, "
            f"all-component confluence={sig.confluence}."
        )
        rationale = (
            "Derivatives market structure (funding + OI + liquidations) provides a "
            "cleaner signal of institutional positioning than price alone.  "
            f"The {sig.signal_label} pattern has historically preceded "
            f"{'bullish squeezes' if bullish else 'bearish unwinds'} within 1-6 hours."
        )
        return (title, body, rationale, round(prior, 3), 3 if bullish else 2, f"alt_data,derivatives,{sig.signal_label}")

    @staticmethod
    def _make_flow_hypothesis(flow: ExchangeFlowSnapshot) -> tuple:
        bullish   = flow.is_bullish
        direction = "Bullish accumulation" if bullish else "Bearish distribution"
        prior     = min(0.72, 0.50 + flow.signal_strength * 0.25)
        title = (
            f"[AltData-Flow] {direction} signal for {flow.symbol} "
            f"— strength={flow.signal_strength:.2f}"
        )
        body = (
            f"Exchange flow analysis for {flow.symbol}: regime={flow.flow_regime}, "
            f"sell_pressure_ratio={flow.sell_pressure_ratio:.3f}, "
            f"inflow/outflow={flow.inflow_usd_1h:.1f}M / {flow.outflow_usd_1h:.1f}M USD/hr, "
            f"24h reserve change={flow.reserve_change_24h:+.1f}M USD."
        )
        rationale = (
            "Exchange inflow/outflow data is one of the most reliable on-chain metrics. "
            "Sustained net outflows (accumulation) historically precede price appreciation "
            "as supply leaves exchanges.  Sustained inflows (distribution) signal "
            "impending sell pressure."
        )
        return (title, body, rationale, round(prior, 3), 3, f"alt_data,exchange_flow,{'accumulation' if bullish else 'distribution'}")

    @staticmethod
    def _make_trend_hypothesis(ts: TrendSignal) -> tuple:
        title = (
            f"[AltData-Trends] Rising retail demand detected "
            f"— '{ts.keyword}' acceleration={ts.acceleration:+.1f}"
        )
        body = (
            f"Google Trends for '{ts.keyword}': latest_value={ts.latest_value:.1f}/100, "
            f"delta_1w={ts.delta_1w:+.1f}, acceleration={ts.acceleration:+.1f}. "
            f"Signal: {ts.signal_type}. "
            f"Rising 'crypto buy' searches indicate growing retail interest that "
            f"historically precedes retail inflows within 1-4 weeks."
        )
        rationale = (
            "Google search volume acceleration for buy-intent terms is a documented "
            "leading indicator of retail capital inflows into crypto assets.  "
            "The 2nd derivative (acceleration) captures emerging trends before they "
            "plateau, providing earlier entry signals than the level alone."
        )
        return (title, body, rationale, 0.57, 4, "alt_data,google_trends,retail_demand")

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _persist_signals(
        self,
        deriv:  list[DerivativesSignal],
        flows:  list[ExchangeFlowSnapshot],
        trends: list[TrendSignal],
    ) -> None:
        """Write all signals to alt_data_signals table."""
        conn = self._get_conn()
        rows = []

        for sig in deriv:
            rows.append((
                sig.symbol, "derivatives", sig.signal_label,
                sig.composite_score, sig.confidence, json.dumps(sig.to_dict()),
            ))
        for flow in flows:
            rows.append((
                flow.symbol, "exchange_flow", flow.signal_type,
                None, flow.signal_strength,
                json.dumps({"regime": flow.flow_regime, "sell_ratio": flow.sell_pressure_ratio}),
            ))
        for ts in trends:
            rows.append((
                "MARKET", "google_trends", ts.signal_type,
                ts.acceleration, None,
                json.dumps({"keyword": ts.keyword, "latest": ts.latest_value, "delta_1w": ts.delta_1w}),
            ))

        try:
            conn.executemany(
                """
                INSERT INTO alt_data_signals
                    (symbol, source, signal_label, composite_score, confidence, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            logger.debug("AltDataPipeline: persisted %d alt_data rows.", len(rows))
        except Exception as exc:
            logger.error("AltDataPipeline: persist failed: %s", exc)
            conn.rollback()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Caching                                                              #
    # ------------------------------------------------------------------ #

    def _cached(self, key: str, fetcher_fn) -> Any:
        """Return cached data if fresh, otherwise call fetcher_fn and cache."""
        cached_val, cached_ts = self._cache.get(key, (None, 0.0))
        if cached_val is not None and (time.monotonic() - cached_ts) < CACHE_TTL_S:
            logger.debug("AltDataPipeline: cache hit for '%s'.", key)
            return cached_val
        try:
            result = fetcher_fn()
            self._cache[key] = (result, time.monotonic())
            return result
        except Exception as exc:
            logger.error("AltDataPipeline: fetcher '%s' failed: %s", key, exc)
            return cached_val or []

    # ------------------------------------------------------------------ #
    # DB helpers                                                           #
    # ------------------------------------------------------------------ #

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript(_CREATE_SQL)
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run one AltDataPipeline cycle")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--no-github", action="store_true")
    parser.add_argument("--no-trends", action="store_true")
    args = parser.parse_args()

    pipeline = AltDataPipeline(
        db_path=args.db,
        enable_github=not args.no_github,
        enable_trends=not args.no_trends,
    )
    result = pipeline.run_cycle()
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    _main()
