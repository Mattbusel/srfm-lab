"""
causal_hypothesis_engine.py
===========================
Causal hypothesis generation engine for the idea-engine.

Implements Granger causality, transfer entropy, intervention analysis,
instrumental variable identification, causal DAG construction (PC algorithm),
counterfactual estimation, mediation analysis, time-lagged cross-correlation,
and a master CausalHypothesisEngine that produces trading hypotheses.
"""

from __future__ import annotations

import itertools
import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class CausalDirection(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    AMBIGUOUS = "ambiguous"


@dataclass
class CausalHypothesis:
    """A single causal hypothesis linking two signals."""

    hypothesis_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    cause: str = ""
    effect: str = ""
    mechanism: str = ""
    direction: CausalDirection = CausalDirection.AMBIGUOUS
    lag: int = 0
    evidence_strength: float = 0.0
    p_value: float = 1.0
    method: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        return (
            f"[{self.method}] {self.cause} -> {self.effect} "
            f"(dir={self.direction.value}, lag={self.lag}, "
            f"strength={self.evidence_strength:.4f}, p={self.p_value:.4f})"
        )


@dataclass
class CausalDAGEdge:
    source: str
    target: str
    weight: float = 0.0
    lag: int = 0


@dataclass
class CausalDAG:
    nodes: List[str] = field(default_factory=list)
    edges: List[CausalDAGEdge] = field(default_factory=list)

    def adjacency_matrix(self) -> np.ndarray:
        n = len(self.nodes)
        idx = {name: i for i, name in enumerate(self.nodes)}
        mat = np.zeros((n, n))
        for e in self.edges:
            i, j = idx.get(e.source), idx.get(e.target)
            if i is not None and j is not None:
                mat[i, j] = e.weight if e.weight != 0 else 1.0
        return mat

    def parents(self, node: str) -> List[str]:
        return [e.source for e in self.edges if e.target == node]

    def children(self, node: str) -> List[str]:
        return [e.target for e in self.edges if e.source == node]


# ---------------------------------------------------------------------------
# Helper: embed-safe stats
# ---------------------------------------------------------------------------

def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals via normal equations."""
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ beta
    except np.linalg.LinAlgError:
        return y.copy()


def _lag_matrix(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Build a matrix of lagged versions of *x*."""
    n = len(x)
    out = np.zeros((n - max_lag, max_lag))
    for lag in range(1, max_lag + 1):
        out[:, lag - 1] = x[max_lag - lag: n - lag]
    return out


def _f_test_p_value(rss_r: float, rss_u: float, df1: int, df2: int) -> float:
    """Approximate F-test p-value without scipy."""
    if rss_u <= 0 or df2 <= 0 or df1 <= 0:
        return 1.0
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    if f_stat <= 0:
        return 1.0
    # Beta-distribution approximation (Abramowitz & Stegun)
    x = df2 / (df2 + df1 * f_stat)
    # rough p from x via logistic approx
    p = x ** (df2 / 2) * (1 - x) ** (df1 / 2)
    return min(max(p, 0.0), 1.0)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    a_m = a - np.mean(a)
    b_m = b - np.mean(b)
    denom = np.sqrt(np.sum(a_m ** 2) * np.sum(b_m ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(a_m * b_m) / denom)


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

class GrangerCausalityTest:
    """Test whether signal X Granger-causes signal Y."""

    def __init__(self, max_lag: int = 5, significance: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance

    def test(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Return dict with f_stat, p_value, optimal_lag, is_causal."""
        best: Dict[str, Any] = {"f_stat": 0, "p_value": 1.0, "optimal_lag": 1, "is_causal": False}
        n = len(y)
        for lag in range(1, self.max_lag + 1):
            if n - lag < 2 * lag + 2:
                continue
            y_dep = y[lag:]
            y_lags = _lag_matrix(y, lag)
            x_lags = _lag_matrix(x, lag)
            # restricted model: y on own lags
            X_r = np.column_stack([np.ones(len(y_dep)), y_lags])
            res_r = _ols_residuals(y_dep, X_r)
            rss_r = float(np.sum(res_r ** 2))
            # unrestricted: y on own lags + x lags
            X_u = np.column_stack([X_r, x_lags])
            res_u = _ols_residuals(y_dep, X_u)
            rss_u = float(np.sum(res_u ** 2))
            df1 = lag
            df2 = len(y_dep) - 2 * lag - 1
            p = _f_test_p_value(rss_r, rss_u, df1, df2)
            f_stat = ((rss_r - rss_u) / max(df1, 1)) / (rss_u / max(df2, 1)) if rss_u > 0 and df2 > 0 else 0
            if p < best["p_value"]:
                best = {"f_stat": f_stat, "p_value": p, "optimal_lag": lag, "is_causal": p < self.significance}
        return best

    def bidirectional(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        xy = self.test(x, y)
        yx = self.test(y, x)
        return {"x_causes_y": xy, "y_causes_x": yx}


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------

class TransferEntropy:
    """Information-theoretic causation via transfer entropy."""

    def __init__(self, n_bins: int = 10, lag: int = 1):
        self.n_bins = n_bins
        self.lag = lag

    @staticmethod
    def _entropy(counts: np.ndarray) -> float:
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    def _digitize(self, x: np.ndarray) -> np.ndarray:
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-15:
            return np.zeros(len(x), dtype=int)
        edges = np.linspace(mn, mx, self.n_bins + 1)
        d = np.digitize(x, edges[1:-1])
        return d

    def compute(self, source: np.ndarray, target: np.ndarray) -> float:
        """Compute TE from source -> target."""
        n = min(len(source), len(target))
        if n <= self.lag + 1:
            return 0.0
        src = self._digitize(source[:n])
        tgt = self._digitize(target[:n])
        lag = self.lag
        # Bins: target_future, target_past, source_past
        tf = tgt[lag:]
        tp = tgt[:n - lag]
        sp = src[:n - lag]
        # Joint counts for H(tf, tp, sp), H(tp, sp), H(tf, tp), H(tp)
        nb = self.n_bins
        joint_3 = np.zeros((nb, nb, nb))
        joint_tp_sp = np.zeros((nb, nb))
        joint_tf_tp = np.zeros((nb, nb))
        count_tp = np.zeros(nb)
        m = len(tf)
        for i in range(m):
            a, b, c = int(tf[i]) % nb, int(tp[i]) % nb, int(sp[i]) % nb
            joint_3[a, b, c] += 1
            joint_tp_sp[b, c] += 1
            joint_tf_tp[a, b] += 1
            count_tp[b] += 1

        h_3 = self._entropy(joint_3.ravel())
        h_tp_sp = self._entropy(joint_tp_sp.ravel())
        h_tf_tp = self._entropy(joint_tf_tp.ravel())
        h_tp = self._entropy(count_tp)
        te = h_tf_tp + h_tp_sp - h_3 - h_tp
        return max(te, 0.0)

    def net_transfer(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        te_xy = self.compute(x, y)
        te_yx = self.compute(y, x)
        return {"te_x_to_y": te_xy, "te_y_to_x": te_yx, "net": te_xy - te_yx}


# ---------------------------------------------------------------------------
# Intervention analysis
# ---------------------------------------------------------------------------

class InterventionAnalysis:
    """Analyse the impact on Y when X has a shock."""

    def __init__(self, shock_std: float = 2.0, window_after: int = 10):
        self.shock_std = shock_std
        self.window_after = window_after

    def detect_shocks(self, x: np.ndarray) -> List[int]:
        if len(x) < 20:
            return []
        roll_mean = np.convolve(x, np.ones(20) / 20, mode="same")
        roll_std = np.array([
            np.std(x[max(0, i - 20):i + 1]) if i >= 1 else 1.0
            for i in range(len(x))
        ])
        roll_std[roll_std < 1e-10] = 1.0
        z = (x - roll_mean) / roll_std
        shocks = [i for i in range(len(z)) if abs(z[i]) > self.shock_std]
        return shocks

    def measure_impact(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        shocks = self.detect_shocks(x)
        if not shocks:
            return {"n_shocks": 0, "avg_impact": 0.0, "impacts": []}
        impacts: List[float] = []
        for idx in shocks:
            if idx + self.window_after >= len(y):
                continue
            pre_mean = float(np.mean(y[max(0, idx - self.window_after):idx])) if idx >= self.window_after else float(np.mean(y[:idx + 1]))
            post_mean = float(np.mean(y[idx:idx + self.window_after]))
            impacts.append(post_mean - pre_mean)
        avg = float(np.mean(impacts)) if impacts else 0.0
        return {"n_shocks": len(shocks), "avg_impact": avg, "impacts": impacts}

    def impulse_response(self, x: np.ndarray, y: np.ndarray, max_horizon: int = 20) -> np.ndarray:
        """Simple cross-correlation based impulse-response proxy."""
        n = min(len(x), len(y))
        responses = np.zeros(max_horizon)
        x_centered = x[:n] - np.mean(x[:n])
        y_centered = y[:n] - np.mean(y[:n])
        x_std = np.std(x_centered)
        if x_std < 1e-15:
            return responses
        for h in range(max_horizon):
            if h >= n:
                break
            responses[h] = _correlation(x_centered[:n - h], y_centered[h:n])
        return responses


# ---------------------------------------------------------------------------
# Instrumental variable identification
# ---------------------------------------------------------------------------

class InstrumentalVariableFinder:
    """Identify valid instrumental variables from a pool of candidates."""

    def __init__(self, relevance_threshold: float = 0.15, exogeneity_threshold: float = 0.10):
        self.relevance_threshold = relevance_threshold
        self.exogeneity_threshold = exogeneity_threshold

    def evaluate_candidate(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        residuals_y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Check relevance (corr(z,x)) and exogeneity (corr(z,residual_y))."""
        n = min(len(z), len(x), len(y))
        z, x, y = z[:n], x[:n], y[:n]
        corr_zx = abs(_correlation(z, x))
        if residuals_y is None:
            X_mat = np.column_stack([np.ones(n), x])
            residuals_y = _ols_residuals(y, X_mat)
        corr_ze = abs(_correlation(z, residuals_y[:n]))
        is_valid = corr_zx > self.relevance_threshold and corr_ze < self.exogeneity_threshold
        return {
            "relevance": corr_zx,
            "exogeneity_violation": corr_ze,
            "is_valid": is_valid,
        }

    def find_best(
        self,
        candidates: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        n = min(len(x), len(y))
        X_mat = np.column_stack([np.ones(n), x[:n]])
        res_y = _ols_residuals(y[:n], X_mat)
        results: List[Tuple[str, Dict[str, Any]]] = []
        for name, z in candidates.items():
            info = self.evaluate_candidate(z, x, y, res_y)
            if info["is_valid"]:
                results.append((name, info))
        results.sort(key=lambda t: t[1]["relevance"], reverse=True)
        return results

    def two_stage_ls(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Two-stage least squares estimate."""
        n = min(len(z), len(x), len(y))
        z, x, y = z[:n], x[:n], y[:n]
        # Stage 1: regress x on z
        Z = np.column_stack([np.ones(n), z])
        beta1 = np.linalg.lstsq(Z, x, rcond=None)[0]
        x_hat = Z @ beta1
        # Stage 2: regress y on x_hat
        X_hat = np.column_stack([np.ones(n), x_hat])
        beta2 = np.linalg.lstsq(X_hat, y, rcond=None)[0]
        return {"intercept": float(beta2[0]), "causal_effect": float(beta2[1])}


# ---------------------------------------------------------------------------
# Simplified PC algorithm for DAG construction
# ---------------------------------------------------------------------------

class PCAlgorithm:
    """Simplified constraint-based causal discovery (PC algorithm)."""

    def __init__(self, alpha: float = 0.05, max_cond_set: int = 2):
        self.alpha = alpha
        self.max_cond_set = max_cond_set

    @staticmethod
    def _partial_corr(
        x: np.ndarray, y: np.ndarray, z_set: List[np.ndarray]
    ) -> float:
        if not z_set:
            return abs(_correlation(x, y))
        Z = np.column_stack(z_set)
        Z_aug = np.column_stack([np.ones(len(x)), Z])
        rx = _ols_residuals(x, Z_aug)
        ry = _ols_residuals(y, Z_aug)
        return abs(_correlation(rx, ry))

    @staticmethod
    def _fisher_z_p(r: float, n: int) -> float:
        """P-value for partial correlation via Fisher-z transform."""
        if n <= 4:
            return 1.0
        if abs(r) >= 1.0:
            return 0.0
        z = 0.5 * math.log((1 + r) / (1 - r + 1e-15))
        se = 1.0 / math.sqrt(n - 3)
        z_stat = abs(z) / se
        # Approximate standard normal tail
        p = math.exp(-0.5 * z_stat ** 2) / (z_stat * math.sqrt(2 * math.pi) + 1e-15)
        return min(p * 2, 1.0)

    def build_dag(self, data: Dict[str, np.ndarray]) -> CausalDAG:
        names = sorted(data.keys())
        n_obs = min(len(v) for v in data.values())
        arrays = {k: data[k][:n_obs] for k in names}
        # Start with complete undirected skeleton
        edges = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                edges.add((a, b))

        # Remove edges based on conditional independence
        for cond_size in range(self.max_cond_set + 1):
            removals: List[Tuple[str, str]] = []
            for a, b in list(edges):
                neighbours_a = [n for n in names if n != a and n != b and ((min(a, n), max(a, n)) in edges or (min(b, n), max(b, n)) in edges)]
                for cond in itertools.combinations(neighbours_a, cond_size):
                    z_set = [arrays[c] for c in cond]
                    pc = self._partial_corr(arrays[a], arrays[b], z_set)
                    pv = self._fisher_z_p(pc, n_obs)
                    if pv > self.alpha:
                        removals.append((a, b))
                        break
            for pair in removals:
                edges.discard(pair)

        # Orient edges heuristically using asymmetric correlation at lag-1
        dag_edges: List[CausalDAGEdge] = []
        for a, b in edges:
            corr_ab = abs(_correlation(arrays[a][:-1], arrays[b][1:]))
            corr_ba = abs(_correlation(arrays[b][:-1], arrays[a][1:]))
            if corr_ab >= corr_ba:
                dag_edges.append(CausalDAGEdge(source=a, target=b, weight=corr_ab))
            else:
                dag_edges.append(CausalDAGEdge(source=b, target=a, weight=corr_ba))

        return CausalDAG(nodes=names, edges=dag_edges)


# ---------------------------------------------------------------------------
# Counterfactual causal effect
# ---------------------------------------------------------------------------

class CounterfactualEstimator:
    """Estimate: what would Y be if X were different?"""

    def __init__(self, n_bootstrap: int = 200):
        self.n_bootstrap = n_bootstrap

    def estimate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_counterfactual: np.ndarray,
    ) -> Dict[str, Any]:
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        X_mat = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        y_factual = X_mat @ beta
        n_cf = len(x_counterfactual)
        X_cf = np.column_stack([np.ones(n_cf), x_counterfactual])
        y_cf = X_cf @ beta
        ate = float(np.mean(y_cf)) - float(np.mean(y_factual))
        # bootstrap CI
        ates: List[float] = []
        rng = np.random.default_rng(42)
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Xb = np.column_stack([np.ones(n), x[idx]])
            yb = y[idx]
            bb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            y_f = Xb @ bb
            X_cfb = np.column_stack([np.ones(n_cf), x_counterfactual])
            y_cfb = X_cfb @ bb
            ates.append(float(np.mean(y_cfb) - np.mean(y_f)))
        ci_lo = float(np.percentile(ates, 2.5))
        ci_hi = float(np.percentile(ates, 97.5))
        return {
            "ate": ate,
            "ci_95": (ci_lo, ci_hi),
            "beta": beta.tolist(),
            "y_counterfactual_mean": float(np.mean(y_cf)),
        }

    def treatment_effect_at_quantiles(
        self, x: np.ndarray, y: np.ndarray, quantiles: Sequence[float] = (0.25, 0.5, 0.75)
    ) -> Dict[float, float]:
        """Estimate treatment effect when X is set at given quantiles."""
        results: Dict[float, float] = {}
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        X_mat = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
        baseline = float(beta[0] + beta[1] * np.median(x))
        for q in quantiles:
            x_q = float(np.percentile(x, q * 100))
            y_q = float(beta[0] + beta[1] * x_q)
            results[q] = y_q - baseline
        return results


# ---------------------------------------------------------------------------
# Mediation analysis
# ---------------------------------------------------------------------------

class MediationAnalysis:
    """Does X affect Y through mediator Z?"""

    def __init__(self, n_bootstrap: int = 500):
        self.n_bootstrap = n_bootstrap

    def analyze(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Dict[str, Any]:
        n = min(len(x), len(y), len(z))
        x, y, z = x[:n], y[:n], z[:n]
        ones = np.ones(n)
        # Total effect: Y = c*X
        Xm = np.column_stack([ones, x])
        c = np.linalg.lstsq(Xm, y, rcond=None)[0][1]
        # X -> Z: Z = a*X
        a = np.linalg.lstsq(Xm, z, rcond=None)[0][1]
        # X,Z -> Y: Y = c'*X + b*Z
        Xmz = np.column_stack([ones, x, z])
        coefs = np.linalg.lstsq(Xmz, y, rcond=None)[0]
        c_prime = coefs[1]
        b = coefs[2]
        indirect = a * b
        proportion_mediated = indirect / c if abs(c) > 1e-10 else 0.0
        # Bootstrap CI for indirect effect
        rng = np.random.default_rng(42)
        ab_boots: List[float] = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            xb, yb, zb = x[idx], y[idx], z[idx]
            ob = np.ones(n)
            Xb = np.column_stack([ob, xb])
            a_b = np.linalg.lstsq(Xb, zb, rcond=None)[0][1]
            Xbz = np.column_stack([ob, xb, zb])
            b_b = np.linalg.lstsq(Xbz, yb, rcond=None)[0][2]
            ab_boots.append(a_b * b_b)
        ci_lo = float(np.percentile(ab_boots, 2.5))
        ci_hi = float(np.percentile(ab_boots, 97.5))
        return {
            "total_effect": float(c),
            "direct_effect": float(c_prime),
            "indirect_effect": float(indirect),
            "proportion_mediated": float(proportion_mediated),
            "indirect_ci_95": (ci_lo, ci_hi),
            "path_a": float(a),
            "path_b": float(b),
        }


# ---------------------------------------------------------------------------
# Time-lagged cross-correlation map
# ---------------------------------------------------------------------------

class CrossCorrelationMap:
    """Compute lead-lag relationships across a universe of signals."""

    def __init__(self, max_lag: int = 20):
        self.max_lag = max_lag

    def pairwise(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        lags = range(-self.max_lag, self.max_lag + 1)
        corrs: List[float] = []
        for lag in lags:
            if lag >= 0:
                a = x[:n - lag] if lag > 0 else x
                b = y[lag:] if lag > 0 else y
            else:
                a = x[-lag:]
                b = y[:n + lag]
            corrs.append(_correlation(a, b))
        best_lag = int(list(lags)[int(np.argmax(np.abs(corrs)))])
        best_corr = corrs[int(np.argmax(np.abs(corrs)))]
        return {
            "lags": list(lags),
            "correlations": corrs,
            "best_lag": best_lag,
            "best_corr": best_corr,
            "leader": "x" if best_lag > 0 else "y" if best_lag < 0 else "simultaneous",
        }

    def full_map(self, signals: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        names = sorted(signals.keys())
        results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                results[(a, b)] = self.pairwise(signals[a], signals[b])
        return results

    def lead_lag_matrix(self, signals: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
        names = sorted(signals.keys())
        n = len(names)
        mat = np.zeros((n, n))
        full = self.full_map(signals)
        idx = {name: i for i, name in enumerate(names)}
        for (a, b), info in full.items():
            mat[idx[a], idx[b]] = info["best_lag"]
            mat[idx[b], idx[a]] = -info["best_lag"]
        return names, mat


# ---------------------------------------------------------------------------
# Master engine
# ---------------------------------------------------------------------------

class CausalHypothesisEngine:
    """Generate trading hypotheses from discovered causal relationships."""

    def __init__(
        self,
        granger_max_lag: int = 10,
        te_bins: int = 10,
        te_lag: int = 1,
        pc_alpha: float = 0.05,
        significance: float = 0.05,
    ):
        self.granger = GrangerCausalityTest(max_lag=granger_max_lag, significance=significance)
        self.te = TransferEntropy(n_bins=te_bins, lag=te_lag)
        self.intervention = InterventionAnalysis()
        self.iv_finder = InstrumentalVariableFinder()
        self.pc = PCAlgorithm(alpha=pc_alpha)
        self.counterfactual = CounterfactualEstimator()
        self.mediation = MediationAnalysis()
        self.xcorr = CrossCorrelationMap()
        self.significance = significance
        self.hypotheses: List[CausalHypothesis] = []

    def _add(self, h: CausalHypothesis) -> None:
        self.hypotheses.append(h)
        logger.info("Hypothesis added: %s", h.summary())

    # ------------------------------------------------------------------
    def run_granger_scan(self, signals: Dict[str, np.ndarray]) -> List[CausalHypothesis]:
        """Pairwise Granger causality scan."""
        names = sorted(signals.keys())
        found: List[CausalHypothesis] = []
        for i, a in enumerate(names):
            for b in names:
                if a == b:
                    continue
                res = self.granger.test(signals[a], signals[b])
                if res["is_causal"]:
                    h = CausalHypothesis(
                        cause=a,
                        effect=b,
                        mechanism="granger_causality",
                        direction=CausalDirection.AMBIGUOUS,
                        lag=res["optimal_lag"],
                        evidence_strength=1.0 - res["p_value"],
                        p_value=res["p_value"],
                        method="granger",
                    )
                    self._add(h)
                    found.append(h)
        return found

    def run_transfer_entropy_scan(self, signals: Dict[str, np.ndarray]) -> List[CausalHypothesis]:
        names = sorted(signals.keys())
        found: List[CausalHypothesis] = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                info = self.te.net_transfer(signals[a], signals[b])
                if abs(info["net"]) > 0.01:
                    cause, effect = (a, b) if info["net"] > 0 else (b, a)
                    h = CausalHypothesis(
                        cause=cause,
                        effect=effect,
                        mechanism="transfer_entropy",
                        direction=CausalDirection.POSITIVE,
                        lag=self.te.lag,
                        evidence_strength=abs(info["net"]),
                        p_value=0.0,
                        method="transfer_entropy",
                        metadata=info,
                    )
                    self._add(h)
                    found.append(h)
        return found

    def run_dag_discovery(self, signals: Dict[str, np.ndarray]) -> CausalDAG:
        dag = self.pc.build_dag(signals)
        for edge in dag.edges:
            h = CausalHypothesis(
                cause=edge.source,
                effect=edge.target,
                mechanism="pc_algorithm_dag",
                direction=CausalDirection.AMBIGUOUS,
                lag=edge.lag,
                evidence_strength=edge.weight,
                p_value=0.0,
                method="pc_dag",
            )
            self._add(h)
        return dag

    def run_lead_lag_scan(self, signals: Dict[str, np.ndarray]) -> List[CausalHypothesis]:
        full = self.xcorr.full_map(signals)
        found: List[CausalHypothesis] = []
        for (a, b), info in full.items():
            if abs(info["best_corr"]) > 0.2 and info["best_lag"] != 0:
                cause = a if info["best_lag"] > 0 else b
                effect = b if info["best_lag"] > 0 else a
                direction = CausalDirection.POSITIVE if info["best_corr"] > 0 else CausalDirection.NEGATIVE
                h = CausalHypothesis(
                    cause=cause,
                    effect=effect,
                    mechanism="lead_lag_xcorr",
                    direction=direction,
                    lag=abs(info["best_lag"]),
                    evidence_strength=abs(info["best_corr"]),
                    p_value=0.0,
                    method="xcorr",
                    metadata=info,
                )
                self._add(h)
                found.append(h)
        return found

    # ------------------------------------------------------------------
    def generate_trading_hypotheses(
        self, signals: Dict[str, np.ndarray]
    ) -> List[CausalHypothesis]:
        """Run all causal discovery methods and return consolidated hypotheses."""
        self.hypotheses.clear()
        self.run_granger_scan(signals)
        self.run_transfer_entropy_scan(signals)
        self.run_dag_discovery(signals)
        self.run_lead_lag_scan(signals)
        # Deduplicate: keep strongest per (cause, effect) pair
        best: Dict[Tuple[str, str], CausalHypothesis] = {}
        for h in self.hypotheses:
            key = (h.cause, h.effect)
            if key not in best or h.evidence_strength > best[key].evidence_strength:
                best[key] = h
        deduped = sorted(best.values(), key=lambda h: h.evidence_strength, reverse=True)
        logger.info("Generated %d deduplicated causal trading hypotheses", len(deduped))
        return deduped

    def rank_hypotheses(self) -> List[CausalHypothesis]:
        return sorted(self.hypotheses, key=lambda h: h.evidence_strength, reverse=True)

    def filter_significant(self) -> List[CausalHypothesis]:
        return [h for h in self.hypotheses if h.is_significant]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for h in self.hypotheses:
            out.append({
                "id": h.hypothesis_id,
                "cause": h.cause,
                "effect": h.effect,
                "mechanism": h.mechanism,
                "direction": h.direction.value,
                "lag": h.lag,
                "strength": h.evidence_strength,
                "p_value": h.p_value,
                "method": h.method,
            })
        return out
