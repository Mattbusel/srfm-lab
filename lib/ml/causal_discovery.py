"""
Causal Discovery via PC Algorithm (T4-5)
Builds a causal Directed Acyclic Graph (DAG) of signal relationships.

Uses simplified Peter-Clark (PC) algorithm with conditional independence tests.
Identifies causally redundant signals for pruning and genuine causal lead-lag relationships.

Usage:
    discovery = CausalDiscovery()
    discovery.add_observations({"bh_mass": 2.1, "garch_vol": 0.025, "hurst": 0.63, ...})
    # After 500+ observations:
    dag = discovery.run_pc_algorithm()
    prunable = discovery.get_redundant_signals(dag)
    lead_lags = discovery.get_lead_lag_pairs(dag)
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

log = logging.getLogger(__name__)

@dataclass
class CausalEdge:
    cause: str
    effect: str
    strength: float    # partial correlation coefficient
    lag: int = 0       # lag at which causal relationship holds

@dataclass
class CausalDAG:
    edges: list[CausalEdge]
    variables: list[str]
    skeleton: dict[tuple[str, str], float]  # undirected edges with strength

    def get_parents(self, variable: str) -> list[str]:
        return [e.cause for e in self.edges if e.effect == variable]

    def get_children(self, variable: str) -> list[str]:
        return [e.effect for e in self.edges if e.cause == variable]

    def is_redundant(self, v1: str, v2: str) -> bool:
        """True if v1 and v2 have the same parents (redundant signal)."""
        p1 = set(self.get_parents(v1))
        p2 = set(self.get_parents(v2))
        if not p1 and not p2:
            return False
        overlap = len(p1 & p2) / max(len(p1 | p2), 1)
        return overlap > 0.7

@dataclass
class PCConfig:
    significance_level: float = 0.05   # alpha for independence tests
    max_conditioning_set: int = 3       # max |S| in conditional independence test
    min_observations: int = 500         # minimum obs before running PC
    rerun_every: int = 1000             # rerun algorithm every N new observations
    max_lag: int = 5                    # maximum lag for cross-signal causal test

class CausalDiscovery:
    """
    PC algorithm for causal structure learning in the signal space.

    Variables tracked: all major SRFM signals
    Conditional independence test: partial correlation (Gaussian assumption)

    Output: Causal DAG indicating which signals causally drive others.
    This helps:
      1. Prune redundant signals (same causal parents)
      2. Identify genuine lead-lag relationships
      3. Provide principled basis for signal weights
    """

    SIGNAL_VARIABLES = [
        "bh_mass", "beta", "cf", "garch_vol", "hurst", "ou_zscore",
        "ml_score", "spin_rate", "geodesic_dev", "spacetime_curvature",
        "hawking_temp", "price_return",
    ]

    def __init__(self, cfg: PCConfig = None, variables: list[str] = None):
        self.cfg = cfg or PCConfig()
        self.variables = variables or self.SIGNAL_VARIABLES
        self._obs: dict[str, list[float]] = {v: [] for v in self.variables}
        self._n_obs: int = 0
        self._last_run: int = 0
        self._current_dag: Optional[CausalDAG] = None

    def add_observations(self, obs_dict: dict[str, float]):
        """Add one bar's worth of signal observations."""
        for var in self.variables:
            if var in obs_dict:
                self._obs[var].append(float(obs_dict[var]))
                if len(self._obs[var]) > 5000:
                    self._obs[var] = self._obs[var][-5000:]
        self._n_obs += 1

        if (self._n_obs >= self.cfg.min_observations and
                self._n_obs - self._last_run >= self.cfg.rerun_every):
            self._current_dag = self.run_pc_algorithm()
            self._last_run = self._n_obs

    def run_pc_algorithm(self) -> Optional[CausalDAG]:
        """
        Run simplified PC algorithm on current observations.
        Returns CausalDAG or None if insufficient data.
        """
        # Get variables with sufficient data
        vars_with_data = [v for v in self.variables if len(self._obs[v]) >= 100]
        if len(vars_with_data) < 3:
            return None

        log.info("CausalDiscovery: running PC algorithm on %d variables, %d obs",
                 len(vars_with_data), self._n_obs)

        # Step 1: Start with complete undirected graph
        skeleton = {}
        for i, v1 in enumerate(vars_with_data):
            for v2 in vars_with_data[i+1:]:
                corr = self._partial_correlation(v1, v2, [])
                skeleton[(v1, v2)] = corr
                skeleton[(v2, v1)] = corr

        # Step 2: Remove edges via conditional independence tests
        # For each pair (X, Y), test independence given subsets of other variables
        edges_to_remove = set()

        for i, v1 in enumerate(vars_with_data):
            for v2 in vars_with_data[i+1:]:
                pair = (v1, v2)

                # Test unconditional independence
                corr = abs(self._partial_correlation(v1, v2, []))
                if not self._is_correlated(corr, len(self._obs[v1])):
                    edges_to_remove.add(pair)
                    continue

                # Test conditional independence given subsets
                conditioning_vars = [v for v in vars_with_data if v != v1 and v != v2]
                found_independent = False

                for size in range(1, min(self.cfg.max_conditioning_set + 1, len(conditioning_vars) + 1)):
                    if found_independent:
                        break
                    # Sample conditioning sets (too many to enumerate all)
                    n_trials = min(10, len(conditioning_vars))
                    for _ in range(n_trials):
                        import random
                        cond_set = random.sample(conditioning_vars, min(size, len(conditioning_vars)))
                        pcorr = abs(self._partial_correlation(v1, v2, cond_set))
                        if not self._is_correlated(pcorr, len(self._obs[v1])):
                            edges_to_remove.add(pair)
                            found_independent = True
                            break

        # Remove conditionally independent pairs
        skeleton = {k: v for k, v in skeleton.items()
                   if (k[0], k[1]) not in edges_to_remove and (k[1], k[0]) not in edges_to_remove}

        # Step 3: Orient edges using v-structures (simplified Meek rules)
        # For triplets X - Z - Y where X and Y not directly connected:
        # Orient as X → Z ← Y if Z not in separating set of X,Y
        directed_edges = []
        skeleton_pairs = set((min(k), max(k)) for k in skeleton.keys())

        for i, v1 in enumerate(vars_with_data):
            for v3 in vars_with_data:
                if v1 == v3:
                    continue
                for v2 in vars_with_data[i+1:]:
                    if v2 == v3:
                        continue
                    # Check if v1-v3 and v2-v3 edges exist but v1-v2 edge doesn't
                    pair13 = (min(v1, v3), max(v1, v3))
                    pair23 = (min(v2, v3), max(v2, v3))
                    pair12 = (min(v1, v2), max(v1, v2))

                    if pair13 in skeleton_pairs and pair23 in skeleton_pairs and pair12 not in skeleton_pairs:
                        # v-structure: v1 → v3 ← v2
                        strength13 = abs(skeleton.get((v1, v3), skeleton.get((v3, v1), 0)))
                        strength23 = abs(skeleton.get((v2, v3), skeleton.get((v3, v2), 0)))
                        directed_edges.append(CausalEdge(v1, v3, strength13))
                        directed_edges.append(CausalEdge(v2, v3, strength23))

        # Add remaining undirected as bidirectional (ambiguous orientation)
        oriented_pairs = set((e.cause, e.effect) for e in directed_edges)
        for (v1, v2), strength in skeleton.items():
            if (v1, v2) not in oriented_pairs and (v2, v1) not in oriented_pairs:
                # Undirected: use temporal ordering heuristic (earlier in list = more likely cause)
                if vars_with_data.index(v1) < vars_with_data.index(v2):
                    directed_edges.append(CausalEdge(v1, v2, abs(strength)))
                else:
                    directed_edges.append(CausalEdge(v2, v1, abs(strength)))

        dag = CausalDAG(
            edges=directed_edges,
            variables=vars_with_data,
            skeleton=skeleton,
        )

        log.info("CausalDiscovery: DAG has %d edges among %d variables",
                 len(directed_edges), len(vars_with_data))
        return dag

    def get_redundant_signals(self, dag: CausalDAG = None) -> list[tuple[str, str]]:
        """Return pairs of signals that appear causally redundant."""
        dag = dag or self._current_dag
        if dag is None:
            return []

        redundant = []
        for i, v1 in enumerate(dag.variables):
            for v2 in dag.variables[i+1:]:
                if dag.is_redundant(v1, v2):
                    redundant.append((v1, v2))
        return redundant

    def get_lead_lag_pairs(self, dag: CausalDAG = None) -> list[CausalEdge]:
        """Return direct causal edges (lead-lag relationships)."""
        dag = dag or self._current_dag
        if dag is None:
            return []
        return [e for e in dag.edges if e.strength > 0.2]

    def _partial_correlation(self, v1: str, v2: str, conditioning: list[str]) -> float:
        """Compute partial correlation of v1, v2 given conditioning set."""
        n = min(len(self._obs[v1]), len(self._obs[v2]))
        if n < 10:
            return 0.0

        x = self._obs[v1][-n:]
        y = self._obs[v2][-n:]

        if not conditioning:
            # Simple correlation
            return self._correlation(x, y)

        # Partial correlation via residual regression
        # Regress x and y on conditioning set, correlate residuals
        x_res = self._regress_out(x, conditioning, n)
        y_res = self._regress_out(y, conditioning, n)
        return self._correlation(x_res, y_res)

    def _regress_out(self, y: list[float], predictors: list[str], n: int) -> list[float]:
        """Remove linear effects of predictors from y. Returns residuals."""
        if not predictors:
            return y

        # Single predictor OLS (simplified for speed)
        pred_var = predictors[0]
        if pred_var not in self._obs or len(self._obs[pred_var]) < n:
            return y

        x = self._obs[pred_var][-n:]
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        var_x = sum((xi - x_mean)**2 for xi in x)

        if var_x < 1e-12:
            return y

        beta = cov / var_x
        residuals = [y[i] - beta * (x[i] - x_mean) for i in range(n)]
        return residuals

    def _correlation(self, x: list[float], y: list[float]) -> float:
        """Pearson correlation."""
        n = min(len(x), len(y))
        if n < 3:
            return 0.0
        x_mean = sum(x[:n]) / n
        y_mean = sum(y[:n]) / n
        cov = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        std_x = (sum((xi - x_mean)**2 for xi in x[:n])) ** 0.5
        std_y = (sum((yi - y_mean)**2 for yi in y[:n])) ** 0.5
        if std_x < 1e-12 or std_y < 1e-12:
            return 0.0
        return cov / (std_x * std_y)

    def _is_correlated(self, corr: float, n: int) -> bool:
        """Test if correlation is statistically significant (t-test proxy)."""
        if abs(corr) >= 1.0:
            return True
        t_stat = corr * math.sqrt(n - 2) / math.sqrt(1 - corr**2 + 1e-12)
        # Critical t for alpha=0.05, two-tailed ~ 2.0 for large n
        critical_t = 1.96 + 0.5 / math.sqrt(n)  # approximate
        return abs(t_stat) > critical_t
