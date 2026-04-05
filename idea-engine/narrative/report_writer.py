"""
report_writer.py — Narrative Report Writer
===========================================
Pulls data from idea_engine.db and writes human-readable research
narratives in Markdown format.  Uses the hand-rolled TemplateEngine
(no Jinja2 dependency).

Four report types
-----------------
1. ``weekly_report()``     — Full weekly research report in Markdown.
2. ``idea_card()``         — One-page Markdown card for a single hypothesis.
3. ``genome_biography()``  — Life story of a genome: parent, mutations,
                              performance arc.
4. ``regime_narrative()``  — Description of the current market regime
                              and what strategies are working.

All reports are stored in the ``narrative_reports`` table.

Usage
-----
    writer = NarrativeWriter(db_path="idea_engine.db")
    report = writer.weekly_report()
    print(report)

    card = writer.idea_card(hypothesis_id=42)
    bio  = writer.genome_biography(genome_id=7)
    reg  = writer.regime_narrative(regime="trending")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .template_engine import TemplateEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime descriptions
# ---------------------------------------------------------------------------

REGIME_DESCRIPTIONS: Dict[str, str] = {
    "trending": (
        "A trending regime is characterised by sustained directional price "
        "movement, high ADX (>25), and low mean-reversion tendency. Momentum "
        "strategies thrive; mean-reversion approaches typically underperform. "
        "Volatility is moderate and expanding. Funding rates often trend in "
        "one direction for extended periods."
    ),
    "ranging": (
        "A ranging (mean-reverting) regime features bounded price oscillation, "
        "low ADX (<20), and high predictability of reversion to a mean. "
        "Mean-reversion and stat-arb strategies excel here. Momentum strategies "
        "generate false breakout signals and accumulate losses."
    ),
    "crisis": (
        "A crisis regime is marked by extreme volatility, correlation breakdown, "
        "and fat-tailed return distributions. Liquidity evaporates, spreads widen "
        "dramatically, and most systematic strategies see drawdowns simultaneously. "
        "Capital preservation and defensive sizing are paramount."
    ),
    "accumulation": (
        "An accumulation regime shows low volatility, declining volume, and "
        "compressed ranges. Large players are accumulating positions quietly. "
        "Breakout strategies are poised to profit but timing is uncertain. "
        "Signal quality is low during this phase."
    ),
    "distribution": (
        "A distribution regime features elevated volume on up-moves reversing "
        "to lower prices, signalling institutional selling. Upside momentum is "
        "weak and fading. Short bias or volatility strategies tend to outperform."
    ),
    "unknown": (
        "The current regime is unclassified or in transition. Multiple regime "
        "indicators are contradicting each other. Reduced position sizing and "
        "filter-heavy entry conditions are recommended until regime clarifies."
    ),
}

REGIME_RECOMMENDATIONS: Dict[str, List[str]] = {
    "trending": [
        "Increase momentum signal weight by 20-30%.",
        "Relax mean-reversion entry threshold (wider z-score band).",
        "Use trailing stops rather than fixed exits.",
        "Consider moderate leverage increase (1.2–1.5×) via Kelly fraction.",
    ],
    "ranging": [
        "Increase OU z-score signal weight; target tighter bands.",
        "Reduce or pause pure momentum signals.",
        "Shorten holding period: reversion happens faster in ranges.",
        "Reduce leverage to avoid whipsaws.",
    ],
    "crisis": [
        "Activate the drawdown_gate filter immediately.",
        "Reduce all positions to 25% of normal size.",
        "Increase stop-loss width to avoid being stopped by noise.",
        "Consider moving to cash until crisis regime ends.",
        "Monitor for reversal signal: entropy drop + vol contraction.",
    ],
    "accumulation": [
        "Build a watch-list of breakout trigger levels.",
        "Use vol_filter_low to avoid trading in low-energy periods.",
        "Prepare breakout strategy variants for next regime.",
        "Reduce position size 30% until direction clarifies.",
    ],
    "distribution": [
        "Bias entry signals toward short-side opportunities.",
        "Tighten trailing stops on long positions.",
        "Activate correlation_filter to avoid sector beta exposure.",
        "Monitor funding rate for flip signal.",
    ],
    "unknown": [
        "Apply all available regime filters simultaneously.",
        "Reduce position size to 50% of normal.",
        "Run shadow comparison daily to detect regime shift.",
        "Review HMM state probability distributions.",
    ],
}


# ---------------------------------------------------------------------------
# NarrativeWriter
# ---------------------------------------------------------------------------

class NarrativeWriter:
    """
    Generates human-readable Markdown research narratives from DB data.

    Parameters
    ----------
    db_path : str
        SQLite database path.
    version : str
        IAE version string embedded in reports.
    """

    def __init__(
        self,
        db_path: str = "idea_engine.db",
        version: str = "0.1.0",
    ) -> None:
        self.db_path = db_path
        self.version = version
        self._db: Optional[sqlite3.Connection] = None
        self._engine = TemplateEngine()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS narrative_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT    NOT NULL,
                subject_id  TEXT,
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
            CREATE TABLE IF NOT EXISTS narrative_alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type   TEXT    NOT NULL,
                severity     TEXT    NOT NULL DEFAULT 'info',
                message      TEXT    NOT NULL,
                data_json    TEXT,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                created_at   TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        conn.commit()

    def _table_exists(self, table: str) -> bool:
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return row is not None

    def _safe_query(self, sql: str, params: tuple = ()) -> List[dict]:
        """Execute a query and return rows as dicts; return [] on error."""
        if not all(t in sql or True for t in []):  # always try
            pass
        try:
            conn = self._connect()
            cur = conn.execute(sql, params)
            return [dict(r) for r in cur]
        except sqlite3.OperationalError as exc:
            logger.debug("Query skipped (table absent?): %s", exc)
            return []

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    def _fetch_top_genome(self) -> Optional[dict]:
        """Return the top genome from hall_of_fame or genomes table."""
        rows = self._safe_query(
            """
            SELECT genome_id, fitness, generation, description, strategy
            FROM hall_of_fame
            ORDER BY fitness DESC
            LIMIT 1
            """
        )
        if not rows:
            rows = self._safe_query(
                """
                SELECT id AS genome_id, fitness, generation,
                       '' AS description, params AS strategy
                FROM genomes
                WHERE created_at >= datetime('now', '-7 days')
                ORDER BY fitness DESC
                LIMIT 1
                """
            )
        return rows[0] if rows else None

    def _fetch_best_counterfactual(self) -> Optional[dict]:
        """Return the best counterfactual from the last 7 days."""
        rows = self._safe_query(
            """
            SELECT id, parameter, original_value, cf_value,
                   improvement_pct, summary, created_at
            FROM counterfactual_results
            WHERE created_at >= datetime('now', '-7 days')
            ORDER BY improvement_pct DESC
            LIMIT 1
            """
        )
        return rows[0] if rows else None

    def _fetch_shadow_results(self) -> List[dict]:
        """Return shadow runner results from the last 7 days."""
        rows = self._safe_query(
            """
            SELECT shadow_id,
                   SUM(live_pnl)   AS live_pnl,
                   SUM(shadow_pnl) AS shadow_pnl,
                   SUM(shadow_pnl - live_pnl) AS delta,
                   COUNT(*) AS days
            FROM shadow_runs
            WHERE run_date >= date('now', '-7 days')
            GROUP BY shadow_id
            ORDER BY delta DESC
            """
        )
        for row in rows:
            row["beating"] = (row.get("delta", 0) or 0) > 0
        return rows

    def _fetch_top_hypotheses(self, n: int = 5) -> List[dict]:
        """Return the top-N hypotheses from this week."""
        return self._safe_query(
            """
            SELECT id, hypothesis_text, mapped_component,
                   param_suggestions, confidence, status, created_at
            FROM hypothesis_candidates
            WHERE created_at >= datetime('now', '-7 days')
            ORDER BY confidence DESC
            LIMIT ?
            """,
            (n,),
        )

    def _fetch_papers_this_week(self) -> List[dict]:
        """Return academic papers mined in the last 7 days."""
        return self._safe_query(
            """
            SELECT source, paper_id, title, relevance_score, url, mined_at
            FROM academic_papers
            WHERE mined_at >= datetime('now', '-7 days')
            ORDER BY relevance_score DESC
            LIMIT 20
            """
        )

    def _fetch_serendipity_ideas(self, n: int = 5) -> List[dict]:
        """Return top serendipity ideas from this week."""
        return self._safe_query(
            """
            SELECT technique, domain, idea_text, complexity, score, created_at
            FROM serendipity_ideas
            WHERE created_at >= datetime('now', '-7 days')
            ORDER BY score DESC
            LIMIT ?
            """,
            (n,),
        )

    def _fetch_causal_links(self) -> List[dict]:
        """Return new Granger causality links from this week."""
        return self._safe_query(
            """
            SELECT cause, effect, pvalue, lag, discovered_at
            FROM granger_links
            WHERE discovered_at >= datetime('now', '-7 days')
            ORDER BY pvalue ASC
            LIMIT 10
            """
        )

    # ------------------------------------------------------------------
    # Executive summary generator
    # ------------------------------------------------------------------

    def _build_executive_summary(
        self,
        top_genome: Optional[dict],
        best_cf: Optional[dict],
        papers: List[dict],
        hypotheses: List[dict],
        shadow_results: List[dict],
        ideas: List[dict],
    ) -> str:
        """
        Build a concise executive summary paragraph.

        Parameters
        ----------
        (all data fetched from DB)

        Returns
        -------
        str
        """
        lines: List[str] = []
        now = datetime.now(timezone.utc)
        week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")

        lines.append(
            f"Research activity for the week ending {now.strftime('%Y-%m-%d')}. "
        )

        # Genome
        if top_genome:
            lines.append(
                f"**Top genome** this week: `{top_genome.get('genome_id', 'N/A')}` "
                f"with fitness {top_genome.get('fitness', 0):.4f}. "
            )
        else:
            lines.append("No new genome data available this week. ")

        # Papers
        if papers:
            high_relevance = [p for p in papers if (p.get("relevance_score") or 0) >= 0.7]
            lines.append(
                f"**{len(papers)}** academic papers mined "
                f"({len(high_relevance)} with relevance ≥ 0.70). "
            )
        else:
            lines.append("No academic papers mined this week. ")

        # Hypotheses
        if hypotheses:
            lines.append(
                f"**{len(hypotheses)}** new hypothesis candidates generated "
                f"(top confidence: {max(h.get('confidence',0) or 0 for h in hypotheses):.3f}). "
            )

        # Counterfactual
        if best_cf:
            imp = best_cf.get("improvement_pct", 0) or 0
            lines.append(
                f"**Best counterfactual**: {imp:.1f}% improvement found by changing "
                f"'{best_cf.get('parameter', '?')}'. "
            )

        # Shadow
        if shadow_results:
            beating = [s for s in shadow_results if s.get("beating")]
            if beating:
                lines.append(
                    f"**{len(beating)}** shadow strategy(ies) beating live this week. "
                    f"Review for potential promotion. "
                )
            else:
                lines.append("No shadow strategies outperforming live this week. ")

        # Serendipity
        if ideas:
            lines.append(
                f"**{len(ideas)}** serendipity ideas generated "
                f"(top score: {max(i.get('score',0) or 0 for i in ideas):.3f}). "
            )

        return "".join(lines)

    # ------------------------------------------------------------------
    # Public report methods
    # ------------------------------------------------------------------

    def weekly_report(self) -> str:
        """
        Generate a full weekly research report in Markdown.

        Pulls live data from the DB and renders via the 'weekly_report'
        template. The report is stored in ``narrative_reports``.

        Returns
        -------
        str
            Formatted Markdown report.
        """
        now       = datetime.now(timezone.utc)
        week_label = now.strftime("Week of %Y-%m-%d")
        generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fetch all data
        top_genome       = self._fetch_top_genome()
        best_cf          = self._fetch_best_counterfactual()
        shadow_results   = self._fetch_shadow_results()
        hypotheses       = self._fetch_top_hypotheses(5)
        papers           = self._fetch_papers_this_week()
        ideas            = self._fetch_serendipity_ideas(5)
        causal_links     = self._fetch_causal_links()

        exec_summary = self._build_executive_summary(
            top_genome, best_cf, papers, hypotheses, shadow_results, ideas
        )

        context = {
            "week_label":         week_label,
            "generated_at":       generated_at,
            "executive_summary":  exec_summary,
            "top_genome":         top_genome,
            "best_counterfactual": best_cf,
            "shadow_results":     shadow_results,
            "hypotheses":         hypotheses,
            "papers":             papers,
            "serendipity_ideas":  ideas,
            "causal_links":       causal_links,
            "version":            self.version,
        }
        report = self._engine.render("weekly_report", context)
        self._store_report("weekly", None, report)
        logger.info("Weekly report generated (%d chars).", len(report))
        return report

    def idea_card(self, hypothesis_id: int) -> str:
        """
        Generate a one-page Markdown card for a single hypothesis.

        Parameters
        ----------
        hypothesis_id : int
            Row id in hypothesis_candidates.

        Returns
        -------
        str
        """
        rows = self._safe_query(
            """
            SELECT hc.id, hc.hypothesis_text, hc.mapped_component,
                   hc.param_suggestions, hc.confidence, hc.status, hc.created_at,
                   ap.title AS paper_title, ap.source AS paper_source,
                   ap.relevance_score AS paper_score
            FROM hypothesis_candidates hc
            LEFT JOIN academic_papers ap ON ap.id = hc.source_paper_id
            WHERE hc.id = ?
            """,
            (hypothesis_id,),
        )
        if not rows:
            return f"# Error\n\nHypothesis ID {hypothesis_id} not found in database.\n"

        row = rows[0]

        # Parse param_suggestions
        param_suggestions: dict = {}
        raw_params = row.get("param_suggestions")
        if raw_params:
            try:
                param_suggestions = json.loads(raw_params)
            except (json.JSONDecodeError, TypeError):
                param_suggestions = {"raw": str(raw_params)}

        # Build experiment plan
        component = row.get("mapped_component", "entry_signal")
        experiment_plan = self._build_experiment_plan(
            component, param_suggestions, row.get("confidence", 0.5)
        )

        context = {
            "hypothesis_id":   hypothesis_id,
            "created_at":      row.get("created_at", ""),
            "status":          row.get("status", "pending"),
            "confidence":      row.get("confidence", 0.0) or 0.0,
            "hypothesis_text": row.get("hypothesis_text", ""),
            "mapped_component": component,
            "param_suggestions": param_suggestions,
            "paper_title":     row.get("paper_title"),
            "paper_source":    row.get("paper_source", ""),
            "paper_score":     row.get("paper_score"),
            "experiment_plan": experiment_plan,
        }
        report = self._engine.render("idea_card", context)
        self._store_report("idea_card", str(hypothesis_id), report)
        return report

    def genome_biography(self, genome_id: int) -> str:
        """
        Generate the life story of a genome: parent, mutations, arc.

        Parameters
        ----------
        genome_id : int

        Returns
        -------
        str
        """
        # Fetch the genome
        rows = self._safe_query(
            "SELECT * FROM genomes WHERE id=?", (genome_id,)
        )
        if not rows:
            return f"# Error\n\nGenome ID {genome_id} not found.\n"

        genome = rows[0]
        parent_id = genome.get("parent_id")
        parent_fitness = None
        mutations: List[str] = []

        # Fetch parent
        if parent_id:
            parent_rows = self._safe_query(
                "SELECT id, fitness FROM genomes WHERE id=?", (parent_id,)
            )
            if parent_rows:
                parent_fitness = parent_rows[0].get("fitness")

        # Fetch mutation history
        mut_rows = self._safe_query(
            "SELECT description FROM genome_mutations WHERE genome_id=? ORDER BY applied_at",
            (genome_id,),
        )
        mutations = [r["description"] for r in mut_rows]

        # Build performance arc (previous generations in lineage)
        arc = self._build_performance_arc(genome_id, genome)

        # Parse strategy params
        params: dict = {}
        raw_params = genome.get("params") or genome.get("strategy")
        if raw_params:
            try:
                params = json.loads(raw_params)
            except (json.JSONDecodeError, TypeError):
                params = {"raw": str(raw_params)}

        # Notable events
        events = self._fetch_genome_events(genome_id)

        context = {
            "genome_id":          genome_id,
            "birth_generation":   genome.get("generation", 0),
            "current_generation": genome.get("generation", 0),
            "status":             genome.get("status", "active"),
            "parent_id":          parent_id,
            "parent_fitness":     parent_fitness,
            "mutations":          mutations,
            "performance_arc":    arc,
            "params":             params,
            "events":             events,
        }
        report = self._engine.render("genome_bio", context)
        self._store_report("genome_bio", str(genome_id), report)
        return report

    def regime_narrative(self, regime: str) -> str:
        """
        Describe the current market regime and what strategies are working.

        Parameters
        ----------
        regime : str
            One of: trending, ranging, crisis, accumulation, distribution, unknown.

        Returns
        -------
        str
        """
        regime_l = regime.lower()
        if regime_l not in REGIME_DESCRIPTIONS:
            regime_l = "unknown"

        description     = REGIME_DESCRIPTIONS[regime_l]
        recommendations = REGIME_RECOMMENDATIONS.get(regime_l, [])

        # Fetch strategy performance data if available
        working   = self._fetch_regime_strategies(regime_l, working=True)
        failing   = self._fetch_regime_strategies(regime_l, working=False)

        # Determine period
        now = datetime.now(timezone.utc)
        period_end   = now.strftime("%Y-%m-%d")
        period_start = (now - timedelta(days=30)).strftime("%Y-%m-%d")

        context = {
            "regime_name":        regime_l,
            "period_start":       period_start,
            "period_end":         period_end,
            "duration_days":      30,
            "confidence":         self._estimate_regime_confidence(regime_l),
            "regime_description": description,
            "working_strategies": working,
            "failing_strategies": failing,
            "recommendations":    recommendations,
        }
        report = self._engine.render("regime_summary", context)
        self._store_report("regime", regime_l, report)
        return report

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _build_experiment_plan(
        self,
        component: str,
        params: dict,
        confidence: float,
    ) -> str:
        """Build a plain-English experiment plan for a hypothesis card."""
        base_plans: Dict[str, str] = {
            "entry_signal": (
                "1. Implement the proposed entry signal modification.\n"
                "2. Run walk-forward backtest: 6-month in-sample, 3-month OOS.\n"
                "3. Compare vs. baseline: Sharpe ratio, win rate, avg trade.\n"
                "4. If OOS Sharpe > baseline by >10%, promote to shadow runner."
            ),
            "exit_rule": (
                "1. Implement the proposed exit rule change.\n"
                "2. Run backtest on same historical data as baseline.\n"
                "3. Focus on: avg holding period, max adverse excursion, win%.\n"
                "4. Validate that net profit/trade improves net of costs."
            ),
            "position_sizing": (
                "1. Apply proposed sizing to live paper-trading account.\n"
                "2. Compare daily PnL distribution vs. fixed-fraction baseline.\n"
                "3. Monitor: maximum drawdown, Kelly ratio compliance.\n"
                "4. If 30-day Sharpe improves, update production sizing."
            ),
            "risk_management": (
                "1. Backtest with modified risk parameters.\n"
                "2. Stress-test against 5 historical crisis episodes.\n"
                "3. Confirm max drawdown stays within mandate (<20%).\n"
                "4. Review correlation with existing risk limits."
            ),
            "regime_filter": (
                "1. Implement regime detection logic.\n"
                "2. Run historical regime labelling over 2-year lookback.\n"
                "3. Compare conditional Sharpe: filtered vs. unfiltered.\n"
                "4. Check trade count reduction (should be 20–50% fewer trades)."
            ),
        }
        plan = base_plans.get(component, base_plans["entry_signal"])
        if params:
            param_text = "\n\nSuggested parameters to test:\n" + "\n".join(
                f"- `{k}`: {v}" for k, v in list(params.items())[:5]
            )
            plan += param_text
        if confidence < 0.6:
            plan += (
                "\n\n**Note:** Low confidence (<0.60) — run as shadow only before "
                "committing to live parameter changes."
            )
        return plan

    def _build_performance_arc(self, genome_id: int, genome: dict) -> List[dict]:
        """Build a performance arc table for the genome biography."""
        # Try fetching lineage fitness data
        rows = self._safe_query(
            """
            WITH RECURSIVE lineage(id, generation, fitness, parent_id) AS (
                SELECT id, generation, fitness, parent_id
                FROM genomes WHERE id = ?
                UNION ALL
                SELECT g.id, g.generation, g.fitness, g.parent_id
                FROM genomes g
                JOIN lineage l ON g.id = l.parent_id
                LIMIT 20
            )
            SELECT id, generation, fitness FROM lineage ORDER BY generation
            """,
            (genome_id,),
        )
        arc: List[dict] = []
        prev_fitness = None
        for row in rows:
            note = ""
            if prev_fitness is not None:
                delta = row.get("fitness", 0) - prev_fitness
                if delta > 0:
                    note = f"+{delta:.4f}"
                else:
                    note = f"{delta:.4f}"
            arc.append({
                "generation": row.get("generation", 0),
                "fitness":    row.get("fitness", 0.0) or 0.0,
                "sharpe":     None,
                "maxdd":      None,
                "note":       note,
            })
            prev_fitness = row.get("fitness", 0)

        if not arc:
            # Fallback: just the current genome
            arc = [{
                "generation": genome.get("generation", 0),
                "fitness":    genome.get("fitness", 0.0) or 0.0,
                "sharpe":     None,
                "maxdd":      None,
                "note":       "current",
            }]
        return arc

    def _fetch_genome_events(self, genome_id: int) -> List[dict]:
        """Fetch notable events for a genome."""
        return self._safe_query(
            """
            SELECT generation, description, event_type, created_at
            FROM genome_events
            WHERE genome_id = ?
            ORDER BY generation
            """,
            (genome_id,),
        )

    def _fetch_regime_strategies(
        self,
        regime: str,
        working: bool,
    ) -> List[dict]:
        """Fetch strategies labelled as working/failing in a given regime."""
        op = ">" if working else "<"
        return self._safe_query(
            f"""
            SELECT strategy_id AS name, sharpe_ratio AS sharpe, win_rate
            FROM regime_strategy_performance
            WHERE regime = ?
              AND sharpe_ratio {op} 0.5
            ORDER BY sharpe_ratio {"DESC" if working else "ASC"}
            LIMIT 5
            """,
            (regime,),
        )

    def _estimate_regime_confidence(self, regime: str) -> float:
        """
        Estimate confidence in the current regime label (0–1).

        Queries HMM state probabilities if available; otherwise returns 0.6.
        """
        rows = self._safe_query(
            """
            SELECT confidence FROM regime_states
            WHERE regime = ?
            ORDER BY detected_at DESC
            LIMIT 1
            """,
            (regime,),
        )
        if rows:
            return float(rows[0].get("confidence", 0.6) or 0.6)
        return 0.6

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store_report(
        self,
        report_type: str,
        subject_id: Optional[str],
        content: str,
    ) -> int:
        """Store a rendered report in ``narrative_reports``."""
        conn = self._connect()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur = conn.execute(
            """
            INSERT INTO narrative_reports
                (report_type, subject_id, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (report_type, subject_id, content, now),
        )
        conn.commit()
        return cur.lastrowid or 0

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def recent_reports(self, report_type: Optional[str] = None, limit: int = 10) -> List[dict]:
        """
        Return recently generated reports.

        Parameters
        ----------
        report_type : str or None
            Filter by type ('weekly', 'idea_card', 'genome_bio', 'regime').
        limit : int

        Returns
        -------
        List[dict]
        """
        if report_type:
            return self._safe_query(
                """
                SELECT id, report_type, subject_id, created_at,
                       substr(content, 1, 200) AS preview
                FROM narrative_reports
                WHERE report_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (report_type, limit),
            )
        return self._safe_query(
            """
            SELECT id, report_type, subject_id, created_at,
                   substr(content, 1, 200) AS preview
            FROM narrative_reports
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    def get_report(self, report_id: int) -> Optional[str]:
        """
        Return the full content of a stored report.

        Parameters
        ----------
        report_id : int

        Returns
        -------
        str or None
        """
        rows = self._safe_query(
            "SELECT content FROM narrative_reports WHERE id=?",
            (report_id,),
        )
        return rows[0]["content"] if rows else None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def __repr__(self) -> str:
        return f"NarrativeWriter(db={self.db_path!r})"

    def __enter__(self) -> "NarrativeWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    db = sys.argv[1] if len(sys.argv) > 1 else "idea_engine.db"

    with NarrativeWriter(db_path=db) as writer:
        print("Generating weekly report …")
        report = writer.weekly_report()
        print(report[:3000])
        print("\n--- [truncated] ---\n")

        print("Generating regime narrative (trending) …")
        regime = writer.regime_narrative("trending")
        print(regime)
