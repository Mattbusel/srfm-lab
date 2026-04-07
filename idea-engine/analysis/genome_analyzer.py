"""
genome_analyzer.py -- Analyze IAE genome evolution from SQLite history.

The Go IAE writes genome snapshots to a SQLite database after each adaptation
cycle.  This module reads that data, computes fitness landscapes, detects
breakthroughs, and produces diagnostic plots.
"""

from __future__ import annotations

import sqlite3
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- never show windows
import matplotlib.pyplot as plt
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FitnessLandscape:
    """Summary of the current fitness landscape sampled from genome history."""

    best_fitness: float
    mean_fitness: float
    diversity: float  # mean pairwise Euclidean distance across parameter vectors

    # {param_name: (min_explored, max_explored)}
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # generation index where diversity first dropped below 10 % of initial diversity
    convergence_generation: Optional[int] = None

    def summary(self) -> str:
        lines = [
            f"Best fitness     : {self.best_fitness:.6f}",
            f"Mean fitness     : {self.mean_fitness:.6f}",
            f"Diversity        : {self.diversity:.6f}",
            f"Convergence gen  : {self.convergence_generation}",
            "Parameter ranges :",
        ]
        for k, (lo, hi) in self.parameter_ranges.items():
            lines.append(f"  {k:30s}  [{lo:.4f}, {hi:.4f}]")
        return "\n".join(lines)


@dataclass
class BreakthroughEvent:
    """A generation where fitness improved significantly over the recent baseline."""

    generation: int
    param_name: str       # parameter that changed the most in this generation
    old_value: float      # value of that parameter in the prior generation
    new_value: float      # value of that parameter in this generation
    fitness_before: float # 10-gen rolling average just before this generation
    fitness_after: float  # fitness at this generation
    improvement_pct: float = 0.0  # (fitness_after - fitness_before) / abs(fitness_before)

    def __post_init__(self) -> None:
        if self.fitness_before != 0:
            self.improvement_pct = (
                (self.fitness_after - self.fitness_before) / abs(self.fitness_before)
            )


# ---------------------------------------------------------------------------
# GenomeDatabase -- low-level SQLite access
# ---------------------------------------------------------------------------

# Expected schema written by the Go IAE service:
#
#   CREATE TABLE genome_history (
#       id            TEXT PRIMARY KEY,
#       parent_id     TEXT,
#       generation    INTEGER NOT NULL,
#       fitness       REAL NOT NULL,
#       params_json   TEXT NOT NULL,   -- JSON object of parameter name -> float
#       created_at    TEXT NOT NULL    -- ISO-8601 timestamp
#   );

class GenomeDatabase:
    """
    Read-only interface to the SQLite genome_history table created by Go IAE.
    All methods return plain Python structures or DataFrames -- no mutations.
    """

    TABLE = "genome_history"

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._check_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _check_schema(self) -> None:
        """Verify the expected table exists; log a warning if not."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (self.TABLE,),
                )
                if cur.fetchone() is None:
                    logger.warning(
                        "Table '%s' not found in %s -- queries will return empty results",
                        self.TABLE,
                        self.db_path,
                    )
        except sqlite3.Error as exc:
            logger.error("Cannot open genome DB %s: %s", self.db_path, exc)

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        import json
        d = dict(row)
        if "params_json" in d and isinstance(d["params_json"], str):
            d["params"] = json.loads(d["params_json"])
        return d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_rows(self) -> List[Dict]:
        """Return every row in the genome_history table, ordered by generation."""
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT * FROM {self.TABLE} ORDER BY generation ASC, created_at ASC"
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_elite_genomes(self, n: int = 10) -> List[Dict]:
        """Return the top-N genomes by fitness (highest fitness = best)."""
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT * FROM {self.TABLE} ORDER BY fitness DESC LIMIT ?",
                (n,),
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_lineage(self, genome_id: str) -> List[Dict]:
        """
        Walk the parent_id chain from genome_id back to the root genome.
        Returns list ordered oldest first (root at index 0).
        """
        chain: List[Dict] = []
        current_id: Optional[str] = genome_id
        visited = set()

        with self._connect() as conn:
            while current_id is not None and current_id not in visited:
                visited.add(current_id)
                cur = conn.execute(
                    f"SELECT * FROM {self.TABLE} WHERE id = ?",
                    (current_id,),
                )
                row = cur.fetchone()
                if row is None:
                    break
                d = self._row_to_dict(row)
                chain.append(d)
                current_id = d.get("parent_id")

        chain.reverse()
        return chain

    def diversity_over_time(self) -> pd.Series:
        """
        Compute per-generation genome diversity as the mean pairwise Euclidean
        distance between all parameter vectors in that generation.

        Returns a pd.Series indexed by generation with diversity values.
        """
        rows = self.get_all_rows()
        if not rows:
            return pd.Series(dtype=float)

        # Group by generation
        by_gen: Dict[int, List[np.ndarray]] = {}
        for r in rows:
            gen = r["generation"]
            params = r.get("params", {})
            if params:
                vec = np.array(list(params.values()), dtype=float)
                by_gen.setdefault(gen, []).append(vec)

        result: Dict[int, float] = {}
        for gen, vecs in sorted(by_gen.items()):
            if len(vecs) < 2:
                result[gen] = 0.0
            else:
                mat = np.stack(vecs)  # (n, d)
                dists = []
                n = mat.shape[0]
                for i in range(n):
                    for j in range(i + 1, n):
                        dists.append(float(np.linalg.norm(mat[i] - mat[j])))
                result[gen] = float(np.mean(dists))

        return pd.Series(result, name="diversity")


# ---------------------------------------------------------------------------
# GenomeAnalyzer -- higher-level analysis
# ---------------------------------------------------------------------------

class GenomeAnalyzer:
    """
    Analyze genome evolution data produced by the Go IAE service.

    Typical workflow:
        analyzer = GenomeAnalyzer()
        history  = analyzer.load_history("path/to/iae.db")
        landscape = analyzer.compute_fitness_landscape(history)
        fig       = analyzer.plot_convergence(history)
        breaks    = analyzer.identify_breakthroughs(history)
    """

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_history(db_path: str) -> pd.DataFrame:
        """
        Load the full genome evolution history from the Go IAE SQLite database.

        Returns a DataFrame with columns:
            id, parent_id, generation, fitness, params_json, created_at,
            plus one column per parameter (prefixed 'p_').
        """
        import json

        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(
                "SELECT * FROM genome_history ORDER BY generation ASC, created_at ASC",
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            logger.warning("genome_history is empty in %s", db_path)
            return df

        # Expand params_json into individual columns for easier analysis
        def _parse(js: str) -> Dict:
            try:
                return json.loads(js)
            except (json.JSONDecodeError, TypeError):
                return {}

        param_dicts = df["params_json"].apply(_parse)
        param_df = pd.json_normalize(param_dicts)
        param_df.columns = [f"p_{c}" for c in param_df.columns]
        param_df.index = df.index

        df = pd.concat([df, param_df], axis=1)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        df["generation"] = df["generation"].astype(int)
        df["fitness"] = df["fitness"].astype(float)

        logger.info(
            "Loaded %d genome rows spanning generations %d..%d",
            len(df),
            df["generation"].min(),
            df["generation"].max(),
        )
        return df

    # ------------------------------------------------------------------
    # Fitness landscape
    # ------------------------------------------------------------------

    @staticmethod
    def compute_fitness_landscape(history: pd.DataFrame) -> FitnessLandscape:
        """
        Compute a FitnessLandscape summary from the full genome history.

        Diversity is the mean pairwise Euclidean distance across all parameter
        vectors present in the LATEST generation only.
        """
        if history.empty:
            return FitnessLandscape(
                best_fitness=0.0,
                mean_fitness=0.0,
                diversity=0.0,
            )

        best_fitness = float(history["fitness"].max())
        mean_fitness = float(history["fitness"].mean())

        param_cols = [c for c in history.columns if c.startswith("p_")]

        # Diversity from the last generation
        last_gen = int(history["generation"].max())
        last_gen_df = history[history["generation"] == last_gen]

        diversity = 0.0
        if param_cols and len(last_gen_df) >= 2:
            mat = last_gen_df[param_cols].dropna().values.astype(float)
            if mat.shape[0] >= 2:
                dists = []
                for i in range(mat.shape[0]):
                    for j in range(i + 1, mat.shape[0]):
                        dists.append(float(np.linalg.norm(mat[i] - mat[j])))
                diversity = float(np.mean(dists)) if dists else 0.0

        # Parameter ranges across entire history
        parameter_ranges: Dict[str, Tuple[float, float]] = {}
        for col in param_cols:
            param_name = col[2:]  # strip 'p_' prefix
            series = history[col].dropna()
            if not series.empty:
                parameter_ranges[param_name] = (float(series.min()), float(series.max()))

        # Convergence generation -- first gen where per-gen diversity < 10% of initial
        convergence_generation: Optional[int] = None
        if param_cols:
            per_gen_div: Dict[int, float] = {}
            for gen, grp in history.groupby("generation"):
                mat = grp[param_cols].dropna().values.astype(float)
                if mat.shape[0] >= 2:
                    dists = [
                        float(np.linalg.norm(mat[i] - mat[j]))
                        for i in range(mat.shape[0])
                        for j in range(i + 1, mat.shape[0])
                    ]
                    per_gen_div[int(gen)] = float(np.mean(dists))
                else:
                    per_gen_div[int(gen)] = 0.0

            if per_gen_div:
                gens_sorted = sorted(per_gen_div.keys())
                initial_div = per_gen_div[gens_sorted[0]] if gens_sorted else 0.0
                threshold = 0.10 * initial_div
                for gen in gens_sorted:
                    if per_gen_div[gen] < threshold and initial_div > 0:
                        convergence_generation = gen
                        break

        return FitnessLandscape(
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            diversity=diversity,
            parameter_ranges=parameter_ranges,
            convergence_generation=convergence_generation,
        )

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------

    @staticmethod
    def plot_convergence(history: pd.DataFrame) -> plt.Figure:
        """
        Return a matplotlib Figure showing fitness over generations.

        Subplots:
            Top    -- best and mean fitness per generation
            Bottom -- genome diversity (mean pairwise distance) per generation
        """
        if history.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            return fig

        param_cols = [c for c in history.columns if c.startswith("p_")]

        # Per-generation stats
        gen_stats = (
            history.groupby("generation")["fitness"]
            .agg(best="max", mean="mean")
            .reset_index()
            .sort_values("generation")
        )

        # Diversity per generation
        div_records = []
        for gen, grp in history.groupby("generation"):
            mat = grp[param_cols].dropna().values.astype(float) if param_cols else np.empty((0, 0))
            if mat.shape[0] >= 2:
                dists = [
                    float(np.linalg.norm(mat[i] - mat[j]))
                    for i in range(mat.shape[0])
                    for j in range(i + 1, mat.shape[0])
                ]
                div_records.append({"generation": int(gen), "diversity": float(np.mean(dists))})
            else:
                div_records.append({"generation": int(gen), "diversity": 0.0})
        div_df = pd.DataFrame(div_records).sort_values("generation")

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("IAE Genome Convergence", fontsize=14, fontweight="bold")

        # Top panel -- fitness
        ax_fit = axes[0]
        ax_fit.plot(
            gen_stats["generation"], gen_stats["best"],
            label="Best fitness", color="#1f77b4", linewidth=2,
        )
        ax_fit.plot(
            gen_stats["generation"], gen_stats["mean"],
            label="Mean fitness", color="#ff7f0e", linewidth=1.5, linestyle="--",
        )
        ax_fit.set_ylabel("Fitness")
        ax_fit.legend(loc="lower right")
        ax_fit.grid(True, alpha=0.3)
        ax_fit.set_title("Fitness over Generations")

        # Bottom panel -- diversity
        ax_div = axes[1]
        ax_div.plot(
            div_df["generation"], div_df["diversity"],
            color="#2ca02c", linewidth=1.5,
        )
        ax_div.set_ylabel("Diversity (mean pairwise dist)")
        ax_div.set_xlabel("Generation")
        ax_div.grid(True, alpha=0.3)
        ax_div.set_title("Genome Diversity over Generations")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Parameter evolution
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_evolution(history: pd.DataFrame, param_name: str) -> pd.Series:
        """
        Return a pd.Series of the BEST-fitness genome's parameter value per
        generation, indexed by generation number.

        Falls back to mean if no 'p_{param_name}' column is present.
        """
        col = f"p_{param_name}"
        if col not in history.columns:
            logger.warning("Parameter '%s' not found in genome history", param_name)
            return pd.Series(dtype=float, name=param_name)

        # For each generation, take the value from the best-fitness genome
        records = []
        for gen, grp in history.groupby("generation"):
            best_row = grp.loc[grp["fitness"].idxmax()]
            records.append({"generation": int(gen), "value": float(best_row[col])})

        df_evo = pd.DataFrame(records).sort_values("generation").set_index("generation")
        return df_evo["value"].rename(param_name)

    # ------------------------------------------------------------------
    # Breakthrough detection
    # ------------------------------------------------------------------

    @staticmethod
    def identify_breakthroughs(
        history: pd.DataFrame,
        threshold_pct: float = 0.10,
    ) -> List[BreakthroughEvent]:
        """
        Identify generations where fitness improved by more than threshold_pct
        relative to the rolling 10-generation average prior to that generation.

        A BreakthroughEvent records the parameter that changed the most between
        the breakthrough generation and the previous generation's best genome.
        """
        if history.empty:
            return []

        param_cols = [c for c in history.columns if c.startswith("p_")]

        # Best fitness per generation
        gen_best = (
            history.groupby("generation")
            .apply(lambda g: g.loc[g["fitness"].idxmax()])
            .sort_values("generation")
            .reset_index(drop=True)
        )

        if len(gen_best) < 2:
            return []

        generations = gen_best["generation"].tolist()
        fitnesses = gen_best["fitness"].tolist()

        events: List[BreakthroughEvent] = []
        window = 10

        for idx in range(1, len(generations)):
            gen = generations[idx]
            fit = fitnesses[idx]

            # Rolling mean of the window prior to this generation
            start = max(0, idx - window)
            prior_mean = float(np.mean(fitnesses[start:idx]))

            if prior_mean == 0.0:
                continue

            improvement = (fit - prior_mean) / abs(prior_mean)
            if improvement <= threshold_pct:
                continue

            # Find param that changed the most vs previous best genome
            prev_row = gen_best.iloc[idx - 1]
            curr_row = gen_best.iloc[idx]

            param_name = "unknown"
            old_val = 0.0
            new_val = 0.0
            max_delta = -1.0

            for col in param_cols:
                pname = col[2:]
                try:
                    ov = float(prev_row[col])
                    nv = float(curr_row[col])
                    delta = abs(nv - ov)
                    if delta > max_delta:
                        max_delta = delta
                        param_name = pname
                        old_val = ov
                        new_val = nv
                except (TypeError, ValueError):
                    continue

            events.append(
                BreakthroughEvent(
                    generation=int(gen),
                    param_name=param_name,
                    old_value=old_val,
                    new_value=new_val,
                    fitness_before=prior_mean,
                    fitness_after=fit,
                )
            )

        logger.info(
            "Identified %d breakthrough events (threshold=%.1f%%)",
            len(events),
            threshold_pct * 100,
        )
        return events

    # ------------------------------------------------------------------
    # Convenience -- plot parameter evolution with breakthrough markers
    # ------------------------------------------------------------------

    def plot_parameter_with_breakthroughs(
        self,
        history: pd.DataFrame,
        param_name: str,
        threshold_pct: float = 0.10,
    ) -> plt.Figure:
        """
        Plot the evolution of a single parameter over generations and mark
        breakthrough events on the fitness axis.
        """
        evo = self.parameter_evolution(history, param_name)
        breaks = self.identify_breakthroughs(history, threshold_pct)
        breakthrough_gens = {b.generation for b in breaks}

        gen_best_fit = (
            history.groupby("generation")["fitness"]
            .max()
            .sort_index()
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(f"Parameter Evolution -- {param_name}", fontsize=13)

        ax_param = axes[0]
        ax_param.plot(evo.index, evo.values, color="#9467bd", linewidth=1.8)
        for gen in breakthrough_gens:
            if gen in evo.index:
                ax_param.axvline(gen, color="red", alpha=0.4, linestyle=":")
        ax_param.set_ylabel(param_name)
        ax_param.grid(True, alpha=0.3)

        ax_fit = axes[1]
        ax_fit.plot(
            gen_best_fit.index, gen_best_fit.values,
            color="#1f77b4", linewidth=1.8,
        )
        for gen in breakthrough_gens:
            ax_fit.axvline(gen, color="red", alpha=0.4, linestyle=":", label="Breakthrough" if gen == min(breakthrough_gens) else "")
        if breakthrough_gens:
            ax_fit.legend(loc="lower right")
        ax_fit.set_ylabel("Best Fitness")
        ax_fit.set_xlabel("Generation")
        ax_fit.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Heatmap of parameter correlations vs fitness
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_fitness_correlations(history: pd.DataFrame) -> pd.Series:
        """
        Return Pearson correlations between each parameter and fitness,
        sorted descending by absolute correlation.
        """
        if history.empty:
            return pd.Series(dtype=float)

        param_cols = [c for c in history.columns if c.startswith("p_")]
        if not param_cols:
            return pd.Series(dtype=float)

        corrs = {}
        for col in param_cols:
            pname = col[2:]
            valid = history[[col, "fitness"]].dropna()
            if len(valid) >= 3:
                corr = float(valid[col].corr(valid["fitness"]))
                corrs[pname] = corr

        result = pd.Series(corrs, name="pearson_r")
        return result.reindex(result.abs().sort_values(ascending=False).index)

    # ------------------------------------------------------------------
    # Stat summary table
    # ------------------------------------------------------------------

    @staticmethod
    def generation_summary(history: pd.DataFrame) -> pd.DataFrame:
        """
        Return a per-generation summary DataFrame with columns:
        generation, best_fitness, mean_fitness, std_fitness, n_genomes.
        """
        if history.empty:
            return pd.DataFrame(
                columns=["generation", "best_fitness", "mean_fitness", "std_fitness", "n_genomes"]
            )

        agg = (
            history.groupby("generation")["fitness"]
            .agg(
                best_fitness="max",
                mean_fitness="mean",
                std_fitness="std",
                n_genomes="count",
            )
            .reset_index()
            .sort_values("generation")
        )
        agg["std_fitness"] = agg["std_fitness"].fillna(0.0)
        return agg
