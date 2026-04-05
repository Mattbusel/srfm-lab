"""
generator.py — Serendipity Generator
=====================================
Produces unexpected, creative trading strategy ideas by combining signals
from disparate sources and applying creative mutations.  The generator
uses five techniques to introduce novelty into the BH strategy:

    1. Random signal combination  — picks 2-3 unrelated indicators
    2. Domain borrowing           — maps physics/ecology/etc. to trading
    3. Inversion                  — inverts key current assumptions
    4. Extremization              — pushes parameters to limits
    5. Analogy engine             — structural cross-domain analogies

All ideas are stored in the ``serendipity_ideas`` table for review.

Usage
-----
    gen   = SerendipityGenerator(db_path="idea_engine.db")
    ideas = gen.generate_wild_ideas(n=5)
    for idea in ideas:
        print(f"[{idea.technique}] {idea.description}")
        print(f"  Complexity: {idea.estimated_complexity}")
        print(f"  Experiment: {idea.suggested_experiment}")
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .analogy_engine  import AnalogyEngine, Analogy
from .mutation_engine import StrategyMutator, MutatedStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TECHNIQUES: List[str] = [
    "random_combination",
    "domain_borrow",
    "inversion",
    "extremization",
    "analogy",
]

SUPPORTED_DOMAINS: List[str] = [
    "thermodynamics",
    "fluid_dynamics",
    "ecology",
    "game_theory",
    "information_theory",
    "network_theory",
]

# Base strategy template for mutation / inversion
BASE_STRATEGY: dict = {
    "signals":   ["momentum_20", "ou_zscore"],
    "filters":   ["regime_hmm", "vol_filter_high"],
    "timeframe": "1h",
    "direction": "long_short",
    "params": {
        "entry_threshold": 1.5,
        "lookback":        20,
        "stop_loss_atr":   2.0,
        "kelly_fraction":  0.25,
        "regime_lookback": 50,
        "vol_lookback":    30,
    },
}

# BH strategy current assumptions — inverted by the inversion technique
CURRENT_ASSUMPTIONS: List[Tuple[str, str, str]] = [
    (
        "high_frequency_signals",
        "We use 1h bars as primary resolution.",
        "Use daily bars for signals and 1h bars only for execution.",
    ),
    (
        "trend_following_primary",
        "Momentum (trend-following) is the primary alpha source.",
        "Mean-reversion is the primary alpha source; momentum is a filter.",
    ),
    (
        "continuous_market",
        "We trade continuously, 24/7.",
        "Trade only during high-volume sessions (8:00–16:00 UTC).",
    ),
    (
        "single_asset",
        "The strategy trades BTC as the primary asset.",
        "Run the strategy simultaneously on 10 altcoins; aggregate signal.",
    ),
    (
        "fixed_position_size",
        "Position size is fixed as a fraction of equity.",
        "Position size scales dynamically with signal confidence.",
    ),
    (
        "symmetric_long_short",
        "Long and short positions are treated symmetrically.",
        "Long-only mode; use cash as the 'short' (avoid negative funding).",
    ),
    (
        "stop_loss_atr",
        "Stops are placed at ATR multiples.",
        "Stops are placed at fixed dollar amounts (volatility-agnostic).",
    ),
    (
        "no_macro_filter",
        "Strategy ignores macro regime data.",
        "Add BTC dominance and total market cap as macro regime filters.",
    ),
]

# Parameter extremes for extremization
PARAMETER_EXTREMES: Dict[str, Tuple[Any, Any]] = {
    "entry_threshold": (0.1,  10.0),
    "lookback":        (2,    500),
    "stop_loss_atr":   (0.1,  10.0),
    "kelly_fraction":  (0.01, 1.0),
    "regime_lookback": (5,    500),
    "vol_lookback":    (2,    200),
    "lag_bars":        (0,    50),
    "max_positions":   (1,    100),
}

# Random signal combination ideas
SIGNAL_PAIRS: List[Tuple[str, str, str]] = [
    ("funding_rate",       "momentum_20",        "Funding-momentum hybrid"),
    ("hurst_exponent",     "ou_zscore",           "Fractal + mean-reversion"),
    ("entropy_signal",     "vol_breakout",        "Low-entropy vol breakout"),
    ("order_flow_imbalance","ema_crossover_9_21", "Flow-confirmed trend"),
    ("rsi_14",             "bollinger_squeeze",   "Oversold squeeze expansion"),
    ("returns_skew",       "momentum_50",         "Skew-adjusted momentum"),
    ("macd_signal",        "regime_hmm",          "MACD gated by regime"),
    ("funding_rate",       "rsi_14",              "Funding-divergence signal"),
    ("hurst_exponent",     "momentum_20",         "Persistent momentum filter"),
    ("entropy_signal",     "ou_zscore",           "Low-entropy mean reversion"),
    ("order_flow_imbalance","ou_zscore",          "Flow-confirmed reversion"),
    ("returns_skew",       "bollinger_squeeze",   "Skewed squeeze breakout"),
]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class WildIdea:
    """
    A single creative / serendipitous strategy idea.

    Attributes
    ----------
    technique : str
        Generation technique used.
    domain : str
        Source domain (for domain_borrow/analogy; else 'strategy').
    description : str
        Full human-readable description.
    rationale : str
        Why this idea might work.
    estimated_complexity : str
        Implementation complexity: 'low', 'medium', 'high'.
    suggested_experiment : str
        Concrete experiment to test the idea.
    experiment_json : dict
        Structured experiment spec.
    score : float
        Initial scoring (updated after testing).
    db_id : int or None
        Row id after DB storage.
    """

    technique:              str
    domain:                 str                = "strategy"
    description:            str                = ""
    rationale:              str                = ""
    estimated_complexity:   str                = "medium"
    suggested_experiment:   str                = ""
    experiment_json:        dict               = field(default_factory=dict)
    score:                  float              = 0.0
    db_id:                  Optional[int]      = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("db_id", None)
        return d

    def __repr__(self) -> str:
        return (
            f"WildIdea(technique={self.technique!r}, "
            f"complexity={self.estimated_complexity!r}, "
            f"desc={self.description[:60]!r})"
        )


# ---------------------------------------------------------------------------
# SerendipityGenerator
# ---------------------------------------------------------------------------

class SerendipityGenerator:
    """
    Generates unexpected, creative strategy ideas using five techniques.

    Parameters
    ----------
    db_path : str
        SQLite database path.
    seed : int or None
        Random seed for reproducibility.
    base_strategy : dict or None
        Current strategy parameters (defaults to BASE_STRATEGY).
    """

    def __init__(
        self,
        db_path: str = "idea_engine.db",
        seed: Optional[int] = None,
        base_strategy: Optional[dict] = None,
    ) -> None:
        self.db_path         = db_path
        self._rng            = random.Random(seed)
        self.base_strategy   = base_strategy or dict(BASE_STRATEGY)
        self._db: Optional[sqlite3.Connection] = None
        self._analogy_engine = AnalogyEngine()
        self._mutator        = StrategyMutator(seed=seed)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_wild_ideas(self, n: int = 10) -> List[WildIdea]:
        """
        Generate *n* wild ideas using a mix of all five techniques.

        The ideas are diverse: each technique is called in rotation so
        no single technique dominates the output.

        Parameters
        ----------
        n : int
            Number of ideas to generate.

        Returns
        -------
        List[WildIdea]
            Ideas sorted by estimated novelty (score desc).
        """
        ideas: List[WildIdea] = []
        technique_cycle = [
            "random_combination",
            "domain_borrow",
            "inversion",
            "extremization",
            "analogy",
        ]
        for i in range(n):
            technique = technique_cycle[i % len(technique_cycle)]
            idea = self._generate_by_technique(technique)
            if idea:
                ideas.append(idea)

        # Score ideas by novelty heuristic
        for idea in ideas:
            idea.score = self._novelty_score(idea)

        ideas.sort(key=lambda x: x.score, reverse=True)
        self.store_ideas(ideas)
        logger.info("Generated %d wild ideas.", len(ideas))
        return ideas

    def generate_by_technique(self, technique: str, n: int = 1) -> List[WildIdea]:
        """
        Generate *n* ideas using a specific technique.

        Parameters
        ----------
        technique : str
            One of TECHNIQUES.
        n : int

        Returns
        -------
        List[WildIdea]
        """
        if technique not in TECHNIQUES:
            raise ValueError(f"Unknown technique {technique!r}. Valid: {TECHNIQUES}")
        ideas = []
        for _ in range(n):
            idea = self._generate_by_technique(technique)
            if idea:
                idea.score = self._novelty_score(idea)
                ideas.append(idea)
        return ideas

    # ------------------------------------------------------------------
    # Technique dispatch
    # ------------------------------------------------------------------

    def _generate_by_technique(self, technique: str) -> Optional[WildIdea]:
        dispatch = {
            "random_combination": self._random_combination,
            "domain_borrow":      self._domain_borrow,
            "inversion":          self._inversion,
            "extremization":      self._extremization,
            "analogy":            self._analogy,
        }
        try:
            return dispatch[technique]()
        except Exception as exc:
            logger.warning("Technique %r failed: %s", technique, exc)
            return None

    # ------------------------------------------------------------------
    # Technique 1: Random signal combination
    # ------------------------------------------------------------------

    def _random_combination(self) -> WildIdea:
        """
        Pick 2–3 unrelated indicators and hypothesize their interaction.

        Draws from SIGNAL_PAIRS plus random three-way combinations.
        """
        if self._rng.random() < 0.6:
            # Use a curated pair
            pair = self._rng.choice(SIGNAL_PAIRS)
            sig_a, sig_b, combo_name = pair
            logic = self._rng.choice(["AND", "AND (threshold)", "weighted average"])
            description = (
                f"Combine {sig_a} and {sig_b} into a composite entry: "
                f"'{combo_name}' using {logic} logic."
            )
            rationale = (
                f"'{sig_a}' captures {self._signal_desc(sig_a)} while "
                f"'{sig_b}' captures {self._signal_desc(sig_b)}. "
                f"Their combination may filter noise from each individual signal, "
                f"resulting in higher-precision entries."
            )
            experiment = (
                f"Backtest: enter only when BOTH {sig_a} AND {sig_b} agree. "
                f"Compare Sharpe vs. each signal alone over 6-month OOS window."
            )
            exp_json = {
                "type":      "signal_combination",
                "signal_a":  sig_a,
                "signal_b":  sig_b,
                "logic":     logic,
                "metric":    "sharpe_ratio",
            }
            complexity = "low"
        else:
            # Three-way random combination
            from .mutation_engine import ENTRY_SIGNALS
            three = self._rng.sample(ENTRY_SIGNALS, 3)
            logic = self._rng.choice(["majority vote", "weighted sum", "AND"])
            description = (
                f"Three-signal ensemble: {three[0]} + {three[1]} + {three[2]} "
                f"using {logic}. Enter when ensemble agrees."
            )
            rationale = (
                f"Three-signal ensembles with {logic} voting reduce noise via "
                f"redundancy. If the three signals are weakly correlated, the "
                f"ensemble should have superior risk-adjusted returns."
            )
            experiment = (
                f"Backtest ensemble: {three[0]}, {three[1]}, {three[2]}. "
                f"Measure IC of ensemble vs. individual. Compare drawdown profiles."
            )
            exp_json = {
                "type":    "ensemble",
                "signals": three,
                "logic":   logic,
                "metric":  "information_coefficient",
            }
            complexity = "medium"

        return WildIdea(
            technique              = "random_combination",
            domain                 = "strategy",
            description            = description,
            rationale              = rationale,
            estimated_complexity   = complexity,
            suggested_experiment   = experiment,
            experiment_json        = exp_json,
        )

    # ------------------------------------------------------------------
    # Technique 2: Domain borrowing
    # ------------------------------------------------------------------

    def domain_borrow(self, domain: str) -> WildIdea:
        """
        Borrow a concept from *domain* and map it to a trading idea.

        Parameters
        ----------
        domain : str
            One of SUPPORTED_DOMAINS.

        Returns
        -------
        WildIdea
        """
        if domain not in SUPPORTED_DOMAINS:
            raise ValueError(f"Unknown domain {domain!r}")

        analogies = self._analogy_engine.find_by_domain(domain)
        if not analogies:
            raise ValueError(f"No analogies for domain {domain!r}")

        analogy  = self._rng.choice(analogies)
        exp_dict = self._analogy_engine.analogize_to_experiment(analogy)

        description = (
            f"[{domain.upper()}] {analogy.source_concept} → Trading: "
            f"{analogy.target_concept}. "
            f"{analogy.description[:200]}"
        )
        complexity = "high" if len(analogy.description) > 300 else "medium"
        return WildIdea(
            technique              = "domain_borrow",
            domain                 = domain,
            description            = description,
            rationale              = analogy.description,
            estimated_complexity   = complexity,
            suggested_experiment   = analogy.experiment_hint,
            experiment_json        = exp_dict,
            score                  = analogy.confidence,
        )

    def _domain_borrow(self) -> WildIdea:
        """Random domain borrowing."""
        domain = self._rng.choice(SUPPORTED_DOMAINS)
        return self.domain_borrow(domain)

    # ------------------------------------------------------------------
    # Technique 3: Inversion
    # ------------------------------------------------------------------

    def _inversion(self) -> WildIdea:
        """
        Invert a current BH strategy assumption.

        Picks one of CURRENT_ASSUMPTIONS and generates an idea that
        challenges or reverses that assumption.
        """
        key, current, inverted = self._rng.choice(CURRENT_ASSUMPTIONS)

        description = (
            f"INVERSION of '{key}': Instead of \"{current}\" → "
            f"\"{inverted}\""
        )
        rationale = (
            f"The current assumption '{current}' may be a local maximum "
            f"that prevents exploring a superior strategy structure. "
            f"Inverting to '{inverted}' tests whether the assumption is "
            f"essential or merely conventional. "
            f"Inversion is a powerful creative tool: many breakout strategies "
            f"were discovered by inverting a prior approach."
        )
        experiment = (
            f"Implement: {inverted} "
            f"Run parallel backtest vs. current setup over same period. "
            f"Compare: Sharpe, MaxDD, win_rate, avg_trade."
        )
        exp_json = {
            "type":           "inversion",
            "assumption_key": key,
            "current":        current,
            "inverted":       inverted,
        }
        complexity = "high" if "simultaneously" in inverted or "dynamically" in inverted else "medium"

        return WildIdea(
            technique              = "inversion",
            domain                 = "strategy",
            description            = description,
            rationale              = rationale,
            estimated_complexity   = complexity,
            suggested_experiment   = experiment,
            experiment_json        = exp_json,
        )

    # ------------------------------------------------------------------
    # Technique 4: Extremization
    # ------------------------------------------------------------------

    def _extremization(self) -> WildIdea:
        """
        Push one or two parameters to their extremes.

        Tests the boundary behaviour of the strategy — what happens at
        the limits of the parameter space reveals structural properties.
        """
        # Pick 1 or 2 parameters to extremize
        n_params  = self._rng.choice([1, 2])
        param_keys = self._rng.sample(list(PARAMETER_EXTREMES.keys()), n_params)

        extremes: Dict[str, Any] = {}
        for key in param_keys:
            lo, hi = PARAMETER_EXTREMES[key]
            # Randomly pick low or high extreme
            extreme_val = lo if self._rng.random() < 0.5 else hi
            extremes[key] = extreme_val

        param_strs = [f"{k}={v}" for k, v in extremes.items()]
        description = (
            f"Extremize {n_params} parameter(s): {', '.join(param_strs)}. "
            f"Test strategy behaviour at the edge of the parameter space."
        )
        rationale = (
            f"Extremization reveals structural properties of the strategy: "
            f"does performance degrade gracefully or catastrophically? "
            f"Extreme values often uncover hidden regime-dependencies or "
            f"signal decay patterns that mild parameter changes miss. "
            f"Parameters: {extremes}"
        )
        extreme_notes = []
        for key, val in extremes.items():
            lo, hi = PARAMETER_EXTREMES[key]
            side   = "minimum" if val == lo else "maximum"
            extreme_notes.append(f"{key}={val} ({side} extreme)")

        experiment = (
            f"Set {'; '.join(extreme_notes)}. "
            f"Run full backtest + forward test. "
            f"Expected: extreme degradation or unexpected robustness reveals signal nature."
        )
        exp_json = {
            "type":        "extremization",
            "param_delta": extremes,
            "extremes":    {k: {"lo": PARAMETER_EXTREMES[k][0], "hi": PARAMETER_EXTREMES[k][1]}
                            for k in param_keys},
        }
        return WildIdea(
            technique              = "extremization",
            domain                 = "strategy",
            description            = description,
            rationale              = rationale,
            estimated_complexity   = "low",
            suggested_experiment   = experiment,
            experiment_json        = exp_json,
        )

    # ------------------------------------------------------------------
    # Technique 5: Analogy engine
    # ------------------------------------------------------------------

    def _analogy(self) -> WildIdea:
        """
        Draw a random analogy from the analogy library.

        Returns a WildIdea wrapping the cross-domain analogy.
        """
        analogy  = self._analogy_engine.random_analogy()
        exp_dict = self._analogy_engine.analogize_to_experiment(analogy)

        description = (
            f"[ANALOGY: {analogy.domain.upper()}] "
            f"'{analogy.source_concept}' maps to '{analogy.target_concept}'. "
            f"{analogy.description}"
        )
        return WildIdea(
            technique              = "analogy",
            domain                 = analogy.domain,
            description            = description,
            rationale              = analogy.description,
            estimated_complexity   = "medium",
            suggested_experiment   = analogy.experiment_hint,
            experiment_json        = exp_dict,
            score                  = analogy.confidence,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _novelty_score(self, idea: WildIdea) -> float:
        """
        Heuristic novelty / actionability score for a wild idea.

        Higher scores for:
        - Domain-borrowed ideas with high confidence
        - Ideas referencing concrete metrics (%, Sharpe, etc.)
        - Ideas that touch multiple strategy components
        - Non-trivial complexity
        - Ideas with detailed experiment_json

        Parameters
        ----------
        idea : WildIdea

        Returns
        -------
        float
            Score in [0, 1].
        """
        score = 0.0

        # Base from idea.score if set
        score += idea.score * 0.4

        # Technique novelty bonus
        novelty_bonus = {
            "analogy":            0.25,
            "domain_borrow":      0.22,
            "inversion":          0.20,
            "random_combination": 0.15,
            "extremization":      0.10,
        }
        score += novelty_bonus.get(idea.technique, 0.10)

        # Concreteness: experiment mentions numbers
        import re
        if re.search(r"\d+", idea.suggested_experiment):
            score += 0.08
        if re.search(r"sharpe|drawdown|backtest|oos", idea.suggested_experiment, re.I):
            score += 0.05

        # Complexity bonus for non-trivial ideas
        complexity_bonus = {"high": 0.10, "medium": 0.05, "low": 0.0}
        score += complexity_bonus.get(idea.estimated_complexity, 0)

        # exp_json depth
        if len(idea.experiment_json) >= 4:
            score += 0.05

        return round(min(score, 1.0), 4)

    # ------------------------------------------------------------------
    # Signal description helper
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_desc(signal: str) -> str:
        """Return a brief description of a signal name."""
        descriptions = {
            "momentum_20":          "20-bar normalised price momentum",
            "momentum_50":          "50-bar normalised price momentum",
            "ou_zscore":            "mean-reversion z-score (OU process)",
            "ema_crossover_9_21":   "short-term EMA crossover (9/21)",
            "ema_crossover_21_55":  "medium-term EMA crossover (21/55)",
            "rsi_14":               "RSI(14) overbought/oversold",
            "rsi_7":                "RSI(7) fast overbought/oversold",
            "macd_signal":          "MACD histogram sign change",
            "bollinger_squeeze":    "Bollinger Band squeeze breakout",
            "vol_breakout":         "realised volatility breakout",
            "order_flow_imbalance": "bid/ask volume imbalance (order flow)",
            "funding_rate":         "perpetual funding rate divergence",
            "hurst_exponent":       "Hurst exponent (persistence/anti-persistence)",
            "returns_skew":         "rolling 20-bar return skewness",
            "entropy_signal":       "Shannon entropy of return distribution",
        }
        return descriptions.get(signal, signal.replace("_", " "))

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS serendipity_ideas (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                technique       TEXT    NOT NULL,
                domain          TEXT,
                idea_text       TEXT    NOT NULL,
                rationale       TEXT,
                complexity      TEXT,
                experiment_json TEXT,
                score           REAL    DEFAULT 0.0,
                status          TEXT    NOT NULL DEFAULT 'new',
                created_at      TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        conn.commit()

    def store_ideas(self, ideas: List[WildIdea]) -> int:
        """
        Insert WildIdea objects into ``serendipity_ideas``.

        Parameters
        ----------
        ideas : List[WildIdea]

        Returns
        -------
        int
            Rows inserted.
        """
        conn = self._connect()
        inserted = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for idea in ideas:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO serendipity_ideas
                        (technique, domain, idea_text, rationale,
                         complexity, experiment_json, score, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?)
                    """,
                    (
                        idea.technique,
                        idea.domain,
                        idea.description,
                        idea.rationale,
                        idea.estimated_complexity,
                        json.dumps(idea.experiment_json),
                        idea.score,
                        now,
                    ),
                )
                idea.db_id = cur.lastrowid
                inserted += 1
            except sqlite3.Error as exc:
                logger.warning("DB insert failed for wild idea: %s", exc)
        conn.commit()
        logger.info("Stored %d/%d wild ideas.", inserted, len(ideas))
        return inserted

    def top_ideas(self, n: int = 10, status: str = "new") -> List[dict]:
        """
        Return top-*n* stored ideas by score.

        Parameters
        ----------
        n : int
        status : str

        Returns
        -------
        List[dict]
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT * FROM serendipity_ideas
            WHERE status = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (status, n),
        )
        return [dict(r) for r in cur]

    def ideas_this_week(self) -> List[dict]:
        """Return all ideas created in the last 7 days."""
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT * FROM serendipity_ideas
            WHERE created_at >= datetime('now', '-7 days')
            ORDER BY score DESC
            """
        )
        return [dict(r) for r in cur]

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def __repr__(self) -> str:
        return f"SerendipityGenerator(db={self.db_path!r})"

    def __enter__(self) -> "SerendipityGenerator":
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    with SerendipityGenerator(seed=42) as gen:
        ideas = gen.generate_wild_ideas(n=n)
        print(f"\nGenerated {len(ideas)} wild ideas:\n")
        for i, idea in enumerate(ideas, 1):
            print(f"  {i:2d}. [{idea.technique:20s}] [{idea.estimated_complexity:6s}] "
                  f"score={idea.score:.3f}")
            print(f"      {idea.description[:90]}")
            print(f"      -> {idea.suggested_experiment[:80]}")
            print()
