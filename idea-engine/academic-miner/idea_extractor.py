"""
idea_extractor.py — NLP Idea Extraction Engine
===============================================
Extracts actionable hypotheses from academic paper abstracts using a
rich pattern library (regex + keyword scoring).  No heavy ML dependencies
are required — the engine relies entirely on the Python standard library.

Architecture
------------
1. ``IdeaExtractor`` parses raw abstract text through a pattern library
   organised into semantic groups (claims, comparisons, proposals, etc.).
2. Each matched pattern yields an ``IdeaCandidate`` with a confidence
   score, a raw excerpt, and a mapping to a BH strategy component.
3. ``generate_experiment`` converts an ``IdeaCandidate`` into a
   ``HypothesisTemplate`` — a dict of parameter changes ready for the
   hypothesis generator to consume.
4. Results are stored in the ``hypothesis_candidates`` table.

Usage
-----
    extractor = IdeaExtractor(db_path="idea_engine.db")
    candidates = extractor.extract_hypothesis(abstract_text)
    for c in candidates:
        print(c.text, "->", c.mapped_component)
        template = extractor.generate_experiment(c)
        extractor.store_candidates([c], source_paper_id=42)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BH Strategy component taxonomy
# ---------------------------------------------------------------------------

STRATEGY_COMPONENTS: List[str] = [
    "entry_signal",
    "exit_rule",
    "position_sizing",
    "risk_management",
    "regime_filter",
]

# Keyword sets that signal which strategy component an idea applies to
COMPONENT_KEYWORDS: Dict[str, List[str]] = {
    "entry_signal": [
        "entry", "signal", "indicator", "trigger", "momentum", "breakout",
        "moving average", "rsi", "macd", "crossover", "threshold", "z-score",
        "predictor", "forecast", "predict", "lead", "precede",
    ],
    "exit_rule": [
        "exit", "stop loss", "take profit", "trailing stop", "close position",
        "holding period", "time stop", "target", "reversion", "profit taking",
        "drawdown limit", "max loss",
    ],
    "position_sizing": [
        "position size", "kelly", "fraction", "allocation", "bet size",
        "optimal f", "weight", "leverage", "exposure", "concentration",
        "portfolio", "risk parity", "volatility scaling",
    ],
    "risk_management": [
        "risk", "drawdown", "var", "cvar", "volatility", "hedge", "stop",
        "loss limit", "margin", "liquidation", "tail risk", "stress test",
        "correlation", "diversification", "beta", "delta",
    ],
    "regime_filter": [
        "regime", "market state", "bull", "bear", "trending", "ranging",
        "crisis", "transition", "hidden markov", "hmm", "cluster",
        "classification", "filter", "condition", "environment", "phase",
    ],
}

# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

# Each entry is (pattern_name, regex, confidence_base, component_hint or None)
# Groups:
#   group(1) = subject / method X
#   group(2) = outcome / target Y
#   group(3) = quantity / magnitude Z (optional)

PATTERN_LIBRARY: List[Tuple[str, str, float, Optional[str]]] = [
    # -------------------------------------------------------------------
    # Performance claims
    # -------------------------------------------------------------------
    (
        "improves_by_pct",
        r"(?P<x>[A-Z][^.]{3,60}?)\s+(?:improves?|increases?|boosts?|enhances?)\s+"
        r"(?P<y>[^.]{3,60}?)\s+by\s+(?P<z>\d+(?:\.\d+)?(?:\s*%|\s*percent|\s*basis points?))",
        0.85,
        None,
    ),
    (
        "outperforms",
        r"(?P<x>[A-Z][^.]{3,60}?)\s+(?:outperforms?|beats?|surpasses?|exceeds?)\s+"
        r"(?P<y>[^.]{5,80}?)(?:\s+(?:in|on|for|by)[^.]{0,60})?[.\n]",
        0.80,
        None,
    ),
    (
        "achieves_sharpe",
        r"(?:achiev(?:es?|ing)|yields?|produces?|delivers?)\s+"
        r"(?:an?\s+)?(?:annualized\s+)?[Ss]harpe\s+ratio\s+of\s+(?P<z>\d+(?:\.\d+)?)",
        0.75,
        "position_sizing",
    ),
    (
        "reduces_drawdown",
        r"(?:reduc(?:es?|ing)|lower(?:s|ing)?|decreas(?:es?|ing))\s+"
        r"(?:maximum\s+)?drawdown\s+(?:by\s+)?(?P<z>\d+(?:\.\d+)?(?:\s*%|\s*percent)?)",
        0.80,
        "risk_management",
    ),
    # -------------------------------------------------------------------
    # Significant predictor claims
    # -------------------------------------------------------------------
    (
        "significant_predictor",
        r"(?P<x>[A-Z][^.]{3,60}?)\s+(?:is|are)\s+(?:a\s+)?(?:significant|strong|robust|reliable)\s+"
        r"predictor[s]?\s+of\s+(?P<y>[^.]{5,80}?)[.\n,]",
        0.80,
        None,
    ),
    (
        "granger_causes",
        r"(?P<x>[A-Z][^.]{3,60}?)\s+Granger[\s-]causes?\s+(?P<y>[^.]{5,60}?)[.\n,]",
        0.85,
        "entry_signal",
    ),
    # -------------------------------------------------------------------
    # Method / proposal claims
    # -------------------------------------------------------------------
    (
        "we_propose",
        r"[Ww]e\s+(?:propose|present|introduce|develop)\s+(?P<x>[^.]{10,120}?)\.",
        0.70,
        None,
    ),
    (
        "we_find",
        r"[Ww]e\s+find\s+(?:that\s+)?(?P<x>[^.]{10,120}?)\.",
        0.72,
        None,
    ),
    (
        "combining_improves",
        r"[Cc]ombining\s+(?P<x>[^.]{5,60}?)\s+and\s+(?P<y>[^.]{5,60}?)\s+"
        r"(?:yields?|produces?|achieves?|results?\s+in)\s+(?P<z>[^.]{5,80}?)[.\n]",
        0.78,
        None,
    ),
    (
        "novel_approach",
        r"[Aa]\s+(?:novel|new|improved)\s+(?P<x>[^.]{5,80}?)\s+"
        r"(?:for|to)\s+(?P<y>[^.]{5,80}?)[.\n]",
        0.65,
        None,
    ),
    # -------------------------------------------------------------------
    # Regime / condition claims
    # -------------------------------------------------------------------
    (
        "regime_dependent",
        r"(?P<x>[A-Z][^.]{3,60}?)\s+(?:is|are|performs?)\s+(?:better|superior|more effective)\s+"
        r"(?:during|in|under)\s+(?P<y>[^.]{5,60}?)\s+(?:regimes?|conditions?|periods?)[.\n]",
        0.78,
        "regime_filter",
    ),
    (
        "market_state",
        r"(?:in|during|under)\s+(?P<y>[A-Za-z][^.]{3,50}?)\s+"
        r"(?:market\s+)?(?:regimes?|states?|conditions?),\s+(?P<x>[^.]{10,120}?)[.\n]",
        0.70,
        "regime_filter",
    ),
    # -------------------------------------------------------------------
    # Parameter / threshold claims
    # -------------------------------------------------------------------
    (
        "optimal_parameter",
        r"(?:optimal|best|ideal)\s+(?P<x>[^.]{3,50}?)\s+"
        r"(?:is|was|are|were)\s+(?:found\s+to\s+be\s+)?(?P<z>\d+(?:\.\d+)?(?:\s*%|\s*days?|\s*hours?)?)",
        0.72,
        None,
    ),
    (
        "threshold_claim",
        r"(?:when|if|once)\s+(?P<x>[^.]{5,60}?)\s+(?:exceeds?|crosses?|reaches?|falls?\s+below)\s+"
        r"(?P<z>\d+(?:\.\d+)?(?:\s*%|\s*sigma)?)",
        0.68,
        "entry_signal",
    ),
    # -------------------------------------------------------------------
    # Failure / limitation claims (useful for inverse ideas)
    # -------------------------------------------------------------------
    (
        "fails_in_regime",
        r"(?P<x>[A-Z][^.]{3,60}?)\s+(?:fails?|deteriorates?|underperforms?|breaks?\s+down)\s+"
        r"(?:during|in|under|when)\s+(?P<y>[^.]{5,80}?)[.\n]",
        0.65,
        "regime_filter",
    ),
]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class IdeaCandidate:
    """
    A single hypothesis candidate extracted from an abstract.

    Attributes
    ----------
    pattern_name : str
        Name of the regex pattern that matched.
    text : str
        The extracted hypothesis text (brief narrative).
    excerpt : str
        The raw matched substring from the abstract.
    mapped_component : str
        Which BH strategy component this idea applies to.
    confidence : float
        Base confidence from the pattern, modified by keyword scoring.
    param_suggestions : dict
        Suggested parameter changes for the hypothesis generator.
    source_paper_id : int or None
        FK to academic_papers.id (set after DB insert).
    db_id : int or None
        Row ID in hypothesis_candidates (set after DB insert).
    """

    pattern_name:      str
    text:              str
    excerpt:           str
    mapped_component:  str   = "entry_signal"
    confidence:        float = 0.5
    param_suggestions: dict  = field(default_factory=dict)
    source_paper_id:   Optional[int] = None
    db_id:             Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("db_id", None)
        return d

    def __repr__(self) -> str:
        return (f"IdeaCandidate(component={self.mapped_component!r}, "
                f"conf={self.confidence:.2f}, text={self.text[:60]!r})")


@dataclass
class HypothesisTemplate:
    """
    A ready-to-test experiment template derived from an IdeaCandidate.

    Fields
    ------
    idea_text : str
        Human-readable description of the hypothesis.
    component : str
        Which strategy component to modify.
    param_delta : dict
        Parameter key → suggested new value (or relative change).
    rationale : str
        Why this experiment is worth running.
    priority : float
        Experiment priority in [0, 1].
    experiment_type : str
        One of: 'parameter_sweep', 'signal_addition', 'regime_gate',
                'signal_removal', 'sizing_change'.
    """

    idea_text:       str
    component:       str
    param_delta:     dict
    rationale:       str       = ""
    priority:        float     = 0.5
    experiment_type: str       = "parameter_sweep"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# IdeaExtractor
# ---------------------------------------------------------------------------

class IdeaExtractor:
    """
    Extracts actionable hypotheses from academic abstracts and maps them
    to BH strategy components.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    min_confidence : float
        Only keep candidates with confidence >= this threshold.
    """

    def __init__(
        self,
        db_path: str = "idea_engine.db",
        min_confidence: float = 0.50,
    ) -> None:
        self.db_path        = db_path
        self.min_confidence = min_confidence
        self._db: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_hypothesis(self, abstract: str) -> List[IdeaCandidate]:
        """
        Run the full pattern library against *abstract* and return
        a list of IdeaCandidate objects.

        Processing pipeline:
        1. Clean and normalise the abstract text.
        2. Apply every pattern in PATTERN_LIBRARY.
        3. For each match, infer the strategy component and adjust
           confidence using keyword scoring.
        4. Deduplicate by excerpt similarity.
        5. Filter by min_confidence.
        6. Sort by confidence descending.

        Parameters
        ----------
        abstract : str

        Returns
        -------
        List[IdeaCandidate]
        """
        if not abstract or not abstract.strip():
            return []

        text = self._normalise_text(abstract)
        raw_candidates: List[IdeaCandidate] = []

        for pat_name, pattern, base_conf, component_hint in PATTERN_LIBRARY:
            for m in re.finditer(pattern, text, re.DOTALL):
                excerpt = m.group(0).strip()
                idea_text = self._build_idea_text(pat_name, m)
                if not idea_text or len(idea_text) < 15:
                    continue

                component = (
                    component_hint
                    if component_hint
                    else self._infer_component(idea_text + " " + excerpt)
                )
                confidence = self._adjust_confidence(base_conf, idea_text, abstract)

                candidate = IdeaCandidate(
                    pattern_name      = pat_name,
                    text              = idea_text,
                    excerpt           = excerpt[:300],
                    mapped_component  = component,
                    confidence        = confidence,
                    param_suggestions = self._suggest_params(pat_name, m, component),
                )
                raw_candidates.append(candidate)

        deduped   = self._deduplicate(raw_candidates)
        filtered  = [c for c in deduped if c.confidence >= self.min_confidence]
        filtered.sort(key=lambda c: c.confidence, reverse=True)
        return filtered[:10]  # cap at 10 per abstract

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Strip excess whitespace and normalise line breaks."""
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _build_idea_text(self, pat_name: str, match: re.Match) -> str:
        """Construct a human-readable idea string from a pattern match."""
        gd = match.groupdict()
        x  = gd.get("x", "").strip()
        y  = gd.get("y", "").strip()
        z  = gd.get("z", "").strip()

        templates: Dict[str, str] = {
            "improves_by_pct":      f"{x} improves {y} by {z}" if z else f"{x} improves {y}",
            "outperforms":          f"{x} outperforms {y}" if y else f"{x} outperforms baseline",
            "achieves_sharpe":      f"Strategy achieves Sharpe ratio of {z}",
            "reduces_drawdown":     f"Approach reduces drawdown by {z}" if z else "Approach reduces drawdown",
            "significant_predictor": f"{x} is a significant predictor of {y}" if y else f"{x} is predictive",
            "granger_causes":       f"{x} Granger-causes {y}",
            "we_propose":           x,
            "we_find":              x,
            "combining_improves":   f"Combining {x} and {y} yields {z}" if z else f"Combining {x} and {y}",
            "novel_approach":       f"Novel {x} for {y}" if y else f"Novel approach: {x}",
            "regime_dependent":     f"{x} performs better in {y} regimes" if y else f"{x} is regime-dependent",
            "market_state":         f"In {y} conditions: {x}" if y else x,
            "optimal_parameter":    f"Optimal {x} is {z}" if z else f"Optimal {x} found",
            "threshold_claim":      f"When {x} exceeds {z}, signal fires" if z else f"Threshold on {x}",
            "fails_in_regime":      f"{x} fails/underperforms in {y} regime" if y else f"{x} has regime limitation",
        }
        return templates.get(pat_name, match.group(0)[:120].strip())

    # ------------------------------------------------------------------
    # Component inference
    # ------------------------------------------------------------------

    def _infer_component(self, text: str) -> str:
        """
        Map extracted idea text to the most relevant BH strategy component.

        Uses keyword density scoring across COMPONENT_KEYWORDS.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
            One of STRATEGY_COMPONENTS.
        """
        text_l = text.lower()
        scores: Dict[str, float] = {c: 0.0 for c in STRATEGY_COMPONENTS}
        for component, keywords in COMPONENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_l:
                    scores[component] += 1.0
        best = max(scores, key=lambda c: scores[c])
        if scores[best] == 0:
            return "entry_signal"  # default
        return best

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def _adjust_confidence(self, base: float, idea_text: str, abstract: str) -> float:
        """
        Refine the base confidence using abstract-level quality signals.

        Boosts
        ------
        - Quantitative claim (number present): +0.05
        - Abstract mentions evaluation / backtest: +0.05
        - Abstract mentions crypto / BH strategy keywords: +0.08
        - Idea is specific (>50 chars): +0.02

        Penalties
        ---------
        - Abstract is very short (<100 chars): -0.10
        - Idea text is vague (<20 chars): -0.10

        Parameters
        ----------
        base : float
        idea_text : str
        abstract : str

        Returns
        -------
        float
            Adjusted confidence, clamped to [0.1, 1.0].
        """
        conf = base
        combined = (idea_text + " " + abstract).lower()

        if re.search(r"\d+(?:\.\d+)?(?:\s*%|\s*percent|\s*x\b)", idea_text):
            conf += 0.05
        if any(w in combined for w in ["backtest", "out-of-sample", "empirical", "experiment"]):
            conf += 0.05
        if any(w in combined for w in ["crypto", "bitcoin", "ethereum", "blockchain"]):
            conf += 0.08
        if len(idea_text) > 50:
            conf += 0.02
        if len(abstract) < 100:
            conf -= 0.10
        if len(idea_text) < 20:
            conf -= 0.10

        return round(max(0.10, min(conf, 1.0)), 4)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(candidates: List[IdeaCandidate]) -> List[IdeaCandidate]:
        """
        Remove near-duplicate candidates using Jaccard similarity on word sets.

        Keeps the higher-confidence candidate when two ideas are >70% similar.

        Parameters
        ----------
        candidates : List[IdeaCandidate]

        Returns
        -------
        List[IdeaCandidate]
        """
        def word_set(text: str) -> set:
            return set(re.findall(r"[a-z]+", text.lower()))

        def jaccard(a: str, b: str) -> float:
            sa, sb = word_set(a), word_set(b)
            union = sa | sb
            if not union:
                return 0.0
            return len(sa & sb) / len(union)

        kept: List[IdeaCandidate] = []
        for cand in sorted(candidates, key=lambda c: c.confidence, reverse=True):
            if not any(jaccard(cand.text, k.text) > 0.70 for k in kept):
                kept.append(cand)
        return kept

    # ------------------------------------------------------------------
    # Component mapping
    # ------------------------------------------------------------------

    def map_to_strategy_components(self, idea: IdeaCandidate) -> str:
        """
        Re-infer the strategy component for an idea and update it in-place.

        This is the public facing method; internally calls _infer_component.

        Parameters
        ----------
        idea : IdeaCandidate

        Returns
        -------
        str
            The assigned component name.
        """
        component = self._infer_component(idea.text)
        idea.mapped_component = component
        return component

    # ------------------------------------------------------------------
    # Experiment generation
    # ------------------------------------------------------------------

    def generate_experiment(self, idea_candidate: IdeaCandidate) -> HypothesisTemplate:
        """
        Convert an IdeaCandidate into a HypothesisTemplate ready for the
        hypothesis generator to consume.

        The template contains:
        - param_delta: concrete parameter changes based on the pattern type
        - experiment_type: what kind of experiment to run
        - rationale: brief reasoning

        Parameters
        ----------
        idea_candidate : IdeaCandidate

        Returns
        -------
        HypothesisTemplate
        """
        pat  = idea_candidate.pattern_name
        comp = idea_candidate.mapped_component
        text = idea_candidate.text
        sugg = idea_candidate.param_suggestions or {}

        # Determine experiment type from pattern
        exp_type_map: Dict[str, str] = {
            "improves_by_pct":       "parameter_sweep",
            "outperforms":           "signal_addition",
            "achieves_sharpe":       "sizing_change",
            "reduces_drawdown":      "sizing_change",
            "significant_predictor": "signal_addition",
            "granger_causes":        "signal_addition",
            "we_propose":            "parameter_sweep",
            "we_find":               "parameter_sweep",
            "combining_improves":    "signal_addition",
            "novel_approach":        "parameter_sweep",
            "regime_dependent":      "regime_gate",
            "market_state":          "regime_gate",
            "optimal_parameter":     "parameter_sweep",
            "threshold_claim":       "parameter_sweep",
            "fails_in_regime":       "signal_removal",
        }
        exp_type = exp_type_map.get(pat, "parameter_sweep")

        # Build param_delta from suggestions + component defaults
        param_delta = dict(sugg)
        if not param_delta:
            param_delta = self._default_param_delta(comp, text)

        rationale = (
            f"Pattern '{pat}' in abstract suggests {comp} modification. "
            f"Confidence: {idea_candidate.confidence:.2f}."
        )

        return HypothesisTemplate(
            idea_text       = text,
            component       = comp,
            param_delta     = param_delta,
            rationale       = rationale,
            priority        = idea_candidate.confidence,
            experiment_type = exp_type,
        )

    def _default_param_delta(self, component: str, text: str) -> dict:
        """
        Generate sensible default parameter delta for a component type.

        Parameters
        ----------
        component : str
        text : str

        Returns
        -------
        dict
        """
        defaults: Dict[str, dict] = {
            "entry_signal":     {"signal_lookback": "sweep(5,50,5)", "entry_threshold": "sweep(0.5,2.5,0.25)"},
            "exit_rule":        {"exit_lookback": "sweep(5,20,5)", "stop_loss_atr": "sweep(1.0,3.0,0.5)"},
            "position_sizing":  {"kelly_fraction": "sweep(0.1,0.5,0.05)", "max_position_pct": "sweep(0.05,0.25,0.05)"},
            "risk_management":  {"max_drawdown_pct": "sweep(0.05,0.20,0.02)", "var_confidence": "sweep(0.95,0.99,0.01)"},
            "regime_filter":    {"regime_lookback": "sweep(20,100,10)", "regime_threshold": "sweep(0.3,0.7,0.05)"},
        }

        # Try to extract a specific number from the text
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        base = defaults.get(component, {"param": "sweep(auto)"}).copy()
        if nums:
            base["suggested_value"] = float(nums[0])
        return base

    def _suggest_params(
        self,
        pat_name: str,
        match: re.Match,
        component: str,
    ) -> dict:
        """
        Extract parameter suggestions from a regex match's named groups.

        Parameters
        ----------
        pat_name : str
        match : re.Match
        component : str

        Returns
        -------
        dict
        """
        gd = match.groupdict()
        z  = gd.get("z", "").strip()
        sugg: dict = {}

        if z:
            # Try to parse a numeric value
            num_m = re.search(r"(\d+(?:\.\d+)?)", z)
            if num_m:
                val = float(num_m.group(1))
                if "%" in z or "percent" in z:
                    val = val / 100.0
                    sugg["param_pct"] = val
                elif "days" in z or "day" in z:
                    sugg["lookback_days"] = int(val)
                elif "hours" in z or "hour" in z:
                    sugg["lookback_hours"] = int(val)
                else:
                    sugg["param_value"] = val
        return sugg

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
            CREATE TABLE IF NOT EXISTS academic_papers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source          TEXT    NOT NULL,
                paper_id        TEXT    UNIQUE,
                title           TEXT    NOT NULL,
                authors         TEXT,
                abstract        TEXT,
                relevance_score REAL,
                categories      TEXT,
                url             TEXT,
                mined_at        TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
            CREATE TABLE IF NOT EXISTS hypothesis_candidates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id INTEGER REFERENCES academic_papers(id),
                hypothesis_text TEXT    NOT NULL,
                mapped_component TEXT,
                param_suggestions TEXT,
                confidence      REAL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                created_at      TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        conn.commit()

    def store_candidates(
        self,
        candidates: List[IdeaCandidate],
        source_paper_id: Optional[int] = None,
    ) -> int:
        """
        Insert IdeaCandidate objects into ``hypothesis_candidates``.

        Parameters
        ----------
        candidates : List[IdeaCandidate]
        source_paper_id : int or None

        Returns
        -------
        int
            Rows inserted.
        """
        conn = self._connect()
        inserted = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for c in candidates:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO hypothesis_candidates
                        (source_paper_id, hypothesis_text, mapped_component,
                         param_suggestions, confidence, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        source_paper_id or c.source_paper_id,
                        c.text,
                        c.mapped_component,
                        json.dumps(c.param_suggestions),
                        c.confidence,
                        now,
                    ),
                )
                c.db_id = cur.lastrowid
                inserted += 1
            except sqlite3.Error as exc:
                logger.warning("DB insert failed for candidate: %s", exc)
        conn.commit()
        logger.info("Stored %d/%d hypothesis candidates.", inserted, len(candidates))
        return inserted

    def pending_candidates(self, limit: int = 50) -> List[dict]:
        """
        Fetch pending hypothesis candidates from the DB.

        Parameters
        ----------
        limit : int

        Returns
        -------
        List[dict]
        """
        conn = self._connect()
        cur = conn.execute(
            """
            SELECT hc.*, ap.title AS paper_title, ap.source AS paper_source
            FROM   hypothesis_candidates hc
            LEFT JOIN academic_papers ap ON ap.id = hc.source_paper_id
            WHERE  hc.status = 'pending'
            ORDER  BY hc.confidence DESC
            LIMIT  ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur]

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def __repr__(self) -> str:
        return f"IdeaExtractor(db={self.db_path!r}, min_conf={self.min_confidence})"

    def __enter__(self) -> "IdeaExtractor":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    sample = (
        "We propose a novel momentum-based trading strategy for cryptocurrency markets. "
        "Our approach combines Ornstein-Uhlenbeck mean reversion with a GARCH volatility "
        "filter to detect regime changes. The strategy achieves a Sharpe ratio of 2.3 and "
        "reduces maximum drawdown by 35% compared to a naive momentum baseline. "
        "We find that BTC order flow is a significant predictor of short-term price momentum. "
        "Combining the momentum signal and the vol filter yields superior risk-adjusted returns "
        "out-of-sample. The optimal lookback period is 14 days."
    )
    with IdeaExtractor() as ext:
        candidates = ext.extract_hypothesis(sample)
        print(f"\nExtracted {len(candidates)} candidates:\n")
        for c in candidates:
            print(f"  [{c.confidence:.2f}] [{c.mapped_component:18s}] {c.text}")
            tmpl = ext.generate_experiment(c)
            print(f"           -> {tmpl.experiment_type}  params={tmpl.param_delta}")
        print()
