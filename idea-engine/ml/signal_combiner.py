"""
signal_combiner.py
------------------
Meta-learning signal combination engine for the idea-engine.

Learns optimal weights across signals using multiple methods, updates online,
detects correlated signals, applies crisis regime overrides, and validates
weight stability via walk-forward cross-validation.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

CRISIS_VOL_THRESHOLD   = 0.30   # annualised vol above which crisis mode activates
CORRELATION_PENALTY    = 0.5    # factor by which correlated signal weights are penalised
TURNOVER_LAMBDA        = 0.10   # turnover penalty coefficient
MIN_WEIGHT             = 0.0    # floor on individual weights
MAX_WEIGHT_CONCENTRATION = 0.6  # single-signal weight cap (except equal-weight)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _mean_vec(matrix: List[List[float]]) -> List[float]:
    n, d = len(matrix), len(matrix[0])
    return [sum(matrix[i][j] for i in range(n)) / n for j in range(d)]


def _cov_matrix(X: List[List[float]]) -> List[List[float]]:
    """Sample covariance matrix."""
    n, d = len(X), len(X[0])
    mu = _mean_vec(X)
    cov = [[0.0] * d for _ in range(d)]
    for row in X:
        for i in range(d):
            for j in range(d):
                cov[i][j] += (row[i] - mu[i]) * (row[j] - mu[j])
    for i in range(d):
        for j in range(d):
            cov[i][j] /= max(n - 1, 1)
    return cov


def _corr_matrix(X: List[List[float]]) -> List[List[float]]:
    """Correlation matrix from signal return matrix (T x N)."""
    cov = _cov_matrix(X)
    n = len(cov)
    stds = [math.sqrt(max(cov[i][i], 1e-12)) for i in range(n)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            corr[i][j] = cov[i][j] / max(stds[i] * stds[j], 1e-12)
    return corr


def _project_simplex(v: List[float]) -> List[float]:
    """
    Project a vector onto the probability simplex (weights sum to 1, >= 0).
    Uses the O(n log n) algorithm of Duchi et al. (2008).
    """
    n = len(v)
    u = sorted(v, reverse=True)
    cssv = 0.0
    rho = 0
    for i, ui in enumerate(u):
        cssv += ui
        if ui - (cssv - 1.0) / (i + 1) > 0:
            rho = i
    theta = (sum(u[:rho + 1]) - 1.0) / (rho + 1)
    return [max(vi - theta, 0.0) for vi in v]


def _ridge_solve(
    X: List[List[float]],   # (T, N) signal matrix
    y: List[float],          # (T,) target returns
    alpha: float = 1.0,      # ridge regularisation
) -> List[float]:
    """
    Closed-form ridge regression: w = (X^T X + alpha I)^{-1} X^T y.

    Uses gradient descent for simplicity (avoids numpy dependency).
    """
    T, N = len(X), len(X[0])
    # Initialise weights
    w = [1.0 / N] * N
    lr = 0.01
    for _ in range(500):
        # Gradient of ||Xw - y||^2 + alpha ||w||^2
        pred = [_dot(X[t], w) for t in range(T)]
        grad = [0.0] * N
        for t in range(T):
            err = pred[t] - y[t]
            for j in range(N):
                grad[j] += 2.0 * err * X[t][j] / T
        for j in range(N):
            grad[j] += 2.0 * alpha * w[j]
        for j in range(N):
            w[j] -= lr * grad[j]
    return w


def _elastic_net_solve(
    X: List[List[float]],
    y: List[float],
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    lr: float = 0.005,
) -> List[float]:
    """
    Elastic-net via proximal gradient descent.
    min ||Xw - y||^2 + alpha*l1_ratio*||w||_1 + 0.5*alpha*(1-l1_ratio)*||w||^2
    """
    T, N = len(X), len(X[0])
    w = [1.0 / N] * N
    l1 = alpha * l1_ratio
    l2 = alpha * (1.0 - l1_ratio)

    for _ in range(max_iter):
        pred = [_dot(X[t], w) for t in range(T)]
        grad = [0.0] * N
        for t in range(T):
            err = pred[t] - y[t]
            for j in range(N):
                grad[j] += 2.0 * err * X[t][j] / T
        for j in range(N):
            grad[j] += 2.0 * l2 * w[j]
            # Gradient step
            w[j] -= lr * grad[j]
            # Proximal step (soft-threshold for L1)
            w[j] = math.copysign(max(abs(w[j]) - lr * l1, 0.0), w[j])

    return w


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalMeta:
    signal_id: str
    name: str
    sharpe_history: List[float] = field(default_factory=list)
    portfolio_weight: float = 0.0
    correlation_penalty: float = 1.0   # multiplicative weight reduction
    active: bool = True


@dataclass
class WeightSolution:
    method: str
    weights: Dict[str, float]           # signal_id -> weight
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def as_list(self, signal_ids: List[str]) -> List[float]:
        return [self.weights.get(sid, 0.0) for sid in signal_ids]


@dataclass
class WalkForwardResult:
    method: str
    mean_oos_sharpe: float
    std_oos_sharpe: float
    fold_sharpes: List[float]
    mean_turnover: float
    weight_stability: float     # 1 = perfectly stable, 0 = random


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class SignalCombiner:
    """
    Combines multiple alpha signals into a single portfolio weight vector.

    Methods
    -------
    equal_weight          : 1/N across active signals
    sharpe_weight         : proportional to rolling Sharpe ratio
    min_correlation       : minimise average pairwise correlation
    ridge                 : regularised OLS regression on returns
    elastic_net           : L1+L2 regularised regression
    stacking              : meta-learner (ridge on out-of-sample predictions)

    Online updates with exponential forgetting.
    Walk-forward cross-validation for weight stability assessment.
    Crisis regime override: switch to defensive weights.
    Turnover penalty: regularise weight changes.
    """

    def __init__(
        self,
        forgetting_factor: float = 0.97,
        ridge_alpha: float = 1.0,
        elastic_alpha: float = 0.5,
        elastic_l1_ratio: float = 0.5,
        turnover_penalty: float = TURNOVER_LAMBDA,
        crisis_vol_threshold: float = CRISIS_VOL_THRESHOLD,
        max_weight: float = MAX_WEIGHT_CONCENTRATION,
    ):
        self.forgetting_factor = forgetting_factor
        self.ridge_alpha = ridge_alpha
        self.elastic_alpha = elastic_alpha
        self.elastic_l1_ratio = elastic_l1_ratio
        self.turnover_penalty = turnover_penalty
        self.crisis_vol_threshold = crisis_vol_threshold
        self.max_weight = max_weight

        self._signals: Dict[str, SignalMeta] = {}
        # Rolling signal returns buffer: signal_id -> deque of returns
        self._return_buffer: Dict[str, List[float]] = {}
        self._target_buffer: List[float] = []  # portfolio / benchmark returns
        self._current_weights: Optional[WeightSolution] = None
        self._weight_history: List[WeightSolution] = []
        self._in_crisis: bool = False
        self._realized_vol: float = 0.0

    # ------------------------------------------------------------------
    # Signal registration
    # ------------------------------------------------------------------

    def register_signal(self, signal_id: str, name: str) -> None:
        if signal_id in self._signals:
            return
        self._signals[signal_id] = SignalMeta(signal_id=signal_id, name=name)
        self._return_buffer[signal_id] = []

    def deactivate_signal(self, signal_id: str) -> None:
        if signal_id in self._signals:
            self._signals[signal_id].active = False

    def activate_signal(self, signal_id: str) -> None:
        if signal_id in self._signals:
            self._signals[signal_id].active = True

    @property
    def active_signals(self) -> List[str]:
        return [sid for sid, s in self._signals.items() if s.active]

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def push_returns(
        self,
        signal_returns: Dict[str, float],   # signal_id -> return this period
        target_return: float = 0.0,
    ) -> None:
        """
        Push one period of signal returns for online updating.

        signal_returns : dict mapping signal_id -> signal return (or prediction)
        target_return  : actual portfolio/benchmark return for supervision
        """
        for sid, ret in signal_returns.items():
            if sid in self._return_buffer:
                self._return_buffer[sid].append(ret)
        self._target_buffer.append(target_return)

        # Update rolling vol for crisis detection
        if len(self._target_buffer) >= 20:
            recent = self._target_buffer[-20:]
            self._realized_vol = statistics.stdev(recent) * math.sqrt(252)
        self._in_crisis = self._realized_vol > self.crisis_vol_threshold

        # Update Sharpe estimates per signal
        for sid, meta in self._signals.items():
            buf = self._return_buffer[sid]
            if len(buf) >= 20:
                recent = buf[-20:]
                mu = statistics.mean(recent)
                sd = statistics.stdev(recent) + 1e-9
                sharpe = mu / sd * math.sqrt(252)
                meta.sharpe_history.append(sharpe)

    def push_return_matrix(
        self,
        signal_returns: List[Dict[str, float]],
        target_returns: Optional[List[float]] = None,
    ) -> None:
        """Push a batch of historical return observations."""
        targets = target_returns or [0.0] * len(signal_returns)
        for sr, tr in zip(signal_returns, targets):
            self.push_returns(sr, tr)

    # ------------------------------------------------------------------
    # Weight computation methods
    # ------------------------------------------------------------------

    def equal_weight(self) -> WeightSolution:
        """1/N equal weighting across active signals."""
        ids = self.active_signals
        n = len(ids)
        w = {sid: 1.0 / n for sid in ids} if n > 0 else {}
        return WeightSolution(method="equal_weight", weights=w)

    def sharpe_weight(self, min_periods: int = 20) -> WeightSolution:
        """
        Weights proportional to rolling Sharpe ratio (positive Sharpes only).
        Signals with negative Sharpe receive zero weight.
        """
        ids = self.active_signals
        raw = {}
        for sid in ids:
            meta = self._signals[sid]
            if meta.sharpe_history:
                # Exponentially weighted average Sharpe
                sharpes = meta.sharpe_history[-min_periods:]
                ewa = 0.0
                factor = self.forgetting_factor
                weight_sum = 0.0
                for i, s in enumerate(reversed(sharpes)):
                    w = factor ** i
                    ewa += w * s
                    weight_sum += w
                ewa /= max(weight_sum, 1e-9)
                raw[sid] = max(0.0, ewa)
            else:
                raw[sid] = 0.0

        total = sum(raw.values())
        if total < 1e-9:
            return self.equal_weight()

        w = {sid: raw[sid] / total for sid in ids}
        w = self._apply_weight_cap(w)
        w = self._apply_turnover_penalty(w)
        return WeightSolution(
            method="sharpe_weight",
            weights=w,
            metadata={"raw_sharpes": {k: round(v, 4) for k, v in raw.items()}},
        )

    def min_correlation_weight(self) -> WeightSolution:
        """
        Minimise average pairwise correlation of the combined signal.

        Signals with high average correlation to others are down-weighted.
        Heuristic: w_i ∝ 1 / sum_j |corr(i, j)|.
        """
        ids = self.active_signals
        if len(ids) < 2:
            return self.equal_weight()

        # Build return matrix
        X = self._build_return_matrix(ids)
        if len(X) < 5:
            return self.equal_weight()

        corr = _corr_matrix(X)   # shape (N, N) but X is (T, N) → transpose
        # corr_matrix operates on columns as signals
        N = len(ids)
        avg_corr = []
        for i in range(N):
            ac = sum(abs(corr[j][i]) for j in range(N) if j != i) / max(N - 1, 1)
            avg_corr.append(ac)

        # Update correlation penalties on signal metadata
        for i, sid in enumerate(ids):
            self._signals[sid].correlation_penalty = 1.0 / max(avg_corr[i], 0.05)

        raw = {sid: self._signals[sid].correlation_penalty for sid in ids}
        total = sum(raw.values())
        w = {sid: raw[sid] / max(total, 1e-9) for sid in ids}
        w = self._apply_weight_cap(w)
        w = self._apply_turnover_penalty(w)
        return WeightSolution(
            method="min_correlation",
            weights=w,
            metadata={"avg_corr": {sid: round(ac, 4) for sid, ac in zip(ids, avg_corr)}},
        )

    def ridge_weight(self) -> WeightSolution:
        """
        Ridge regression: regress target returns on signal returns.
        Projects resulting coefficients onto the simplex.
        """
        ids = self.active_signals
        X_T = self._build_return_matrix(ids)
        if len(X_T) < 10:
            return self.equal_weight()
        y = self._target_buffer[-len(X_T):]
        # Exponentially weight observations
        X_w = self._ew_scale_matrix(X_T)
        raw_w = _ridge_solve(X_w, y, alpha=self.ridge_alpha)
        # Project onto simplex
        w_simplex = _project_simplex(raw_w)
        w = dict(zip(ids, w_simplex))
        w = self._apply_weight_cap(w)
        w = self._apply_turnover_penalty(w)
        return WeightSolution(method="ridge", weights=w)

    def elastic_net_weight(self) -> WeightSolution:
        """Elastic-net regression with L1+L2 penalty."""
        ids = self.active_signals
        X_T = self._build_return_matrix(ids)
        if len(X_T) < 10:
            return self.equal_weight()
        y = self._target_buffer[-len(X_T):]
        X_w = self._ew_scale_matrix(X_T)
        raw_w = _elastic_net_solve(
            X_w, y,
            alpha=self.elastic_alpha,
            l1_ratio=self.elastic_l1_ratio,
        )
        w_simplex = _project_simplex(raw_w)
        w = dict(zip(ids, w_simplex))
        w = self._apply_weight_cap(w)
        w = self._apply_turnover_penalty(w)
        return WeightSolution(method="elastic_net", weights=w)

    def stacking_weight(self, n_folds: int = 3) -> WeightSolution:
        """
        Stacking: generate OOS signal predictions via time-series CV,
        then meta-learn weights via ridge regression on OOS predictions.
        """
        ids = self.active_signals
        X_T = self._build_return_matrix(ids)
        T = len(X_T)
        if T < n_folds * 5:
            return self.ridge_weight()

        y = self._target_buffer[-T:]
        fold_size = T // n_folds
        oos_preds: List[List[float]] = []   # (T_oos, N)
        oos_targets: List[float] = []

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end   = val_start + fold_size
            train_X   = X_T[:val_start] if val_start > 0 else X_T[val_end:]
            train_y   = y[:val_start] if val_start > 0 else y[val_end:]
            val_X     = X_T[val_start:val_end]
            val_y     = y[val_start:val_end]
            if len(train_X) < 5:
                continue
            fold_w = _ridge_solve(train_X, train_y, alpha=self.ridge_alpha)
            oos_preds.extend(val_X)
            oos_targets.extend(val_y)

        if not oos_preds:
            return self.ridge_weight()

        meta_w = _ridge_solve(oos_preds, oos_targets, alpha=self.ridge_alpha * 0.5)
        w_simplex = _project_simplex(meta_w)
        w = dict(zip(ids, w_simplex))
        w = self._apply_weight_cap(w)
        w = self._apply_turnover_penalty(w)
        return WeightSolution(method="stacking", weights=w)

    # ------------------------------------------------------------------
    # Main combine() entry point
    # ------------------------------------------------------------------

    def combine(
        self,
        method: str = "sharpe_weight",
        crisis_override: bool = True,
    ) -> WeightSolution:
        """
        Compute combined weights.

        If in crisis regime and crisis_override=True, applies defensive
        weight scheme: equal-weight to low-vol signals, or full cash (empty).
        """
        if crisis_override and self._in_crisis:
            solution = self._crisis_weights()
            solution.metadata["crisis_override"] = True
            solution.metadata["realized_vol"] = round(self._realized_vol, 4)
        else:
            if method == "equal_weight":
                solution = self.equal_weight()
            elif method == "sharpe_weight":
                solution = self.sharpe_weight()
            elif method == "min_correlation":
                solution = self.min_correlation_weight()
            elif method == "ridge":
                solution = self.ridge_weight()
            elif method == "elastic_net":
                solution = self.elastic_net_weight()
            elif method == "stacking":
                solution = self.stacking_weight()
            else:
                raise ValueError(f"Unknown method: {method}")

        self._current_weights = solution
        self._weight_history.append(solution)
        return solution

    def _crisis_weights(self) -> WeightSolution:
        """
        Crisis regime: equal-weight among signals with positive recent Sharpe.
        If none qualify, return cash (zero weights).
        """
        ids = self.active_signals
        defensive = []
        for sid in ids:
            meta = self._signals[sid]
            if meta.sharpe_history and meta.sharpe_history[-1] > 0.0:
                defensive.append(sid)
        if not defensive:
            # Full cash
            return WeightSolution(
                method="crisis_cash",
                weights={sid: 0.0 for sid in ids},
            )
        n = len(defensive)
        w = {sid: (1.0 / n if sid in defensive else 0.0) for sid in ids}
        return WeightSolution(method="crisis_equal", weights=w)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def online_update(self, method: str = "sharpe_weight") -> WeightSolution:
        """Recompute weights after each new return observation."""
        return self.combine(method=method, crisis_override=True)

    # ------------------------------------------------------------------
    # Walk-forward cross-validation
    # ------------------------------------------------------------------

    def walk_forward_cv(
        self,
        method: str = "sharpe_weight",
        n_folds: int = 5,
        min_train: int = 40,
    ) -> WalkForwardResult:
        """
        Walk-forward cross-validation to assess weight stability and OOS Sharpe.

        Splits the historical return buffer into expanding windows:
          Train on [0..t], test on [t..t+step].
        """
        ids = self.active_signals
        X_T = self._build_return_matrix(ids)
        T = len(X_T)
        y = self._target_buffer[-T:]

        if T < min_train + n_folds:
            return WalkForwardResult(
                method=method,
                mean_oos_sharpe=float("nan"),
                std_oos_sharpe=float("nan"),
                fold_sharpes=[],
                mean_turnover=float("nan"),
                weight_stability=float("nan"),
            )

        step = (T - min_train) // n_folds
        fold_sharpes = []
        prev_w: Optional[List[float]] = None
        turnovers = []
        weight_snapshots: List[List[float]] = []

        for fold in range(n_folds):
            train_end = min_train + fold * step
            test_end = min(train_end + step, T)
            if test_end <= train_end:
                break

            train_X = X_T[:train_end]
            train_y = y[:train_end]
            test_X  = X_T[train_end:test_end]
            test_y  = y[train_end:test_end]

            # Fit weights on training window
            if method == "ridge":
                raw_w = _ridge_solve(train_X, train_y, alpha=self.ridge_alpha)
                fold_w = _project_simplex(raw_w)
            elif method == "elastic_net":
                raw_w = _elastic_net_solve(
                    train_X, train_y,
                    alpha=self.elastic_alpha,
                    l1_ratio=self.elastic_l1_ratio,
                )
                fold_w = _project_simplex(raw_w)
            else:
                # Sharpe weights: compute from training Sharpes
                col_sharpes = []
                for j in range(len(ids)):
                    col = [train_X[t][j] for t in range(len(train_X))]
                    mu = statistics.mean(col)
                    sd = statistics.stdev(col) + 1e-9
                    col_sharpes.append(max(0.0, mu / sd * math.sqrt(252)))
                s_total = sum(col_sharpes)
                if s_total > 0:
                    fold_w = [s / s_total for s in col_sharpes]
                else:
                    fold_w = [1.0 / len(ids)] * len(ids)

            weight_snapshots.append(fold_w)

            # OOS performance
            oos_port_rets = [_dot(test_X[t], fold_w) for t in range(len(test_X))]
            if len(oos_port_rets) >= 2:
                mu_oos = statistics.mean(oos_port_rets)
                sd_oos = statistics.stdev(oos_port_rets) + 1e-9
                fold_sharpes.append(mu_oos / sd_oos * math.sqrt(252))

            # Turnover
            if prev_w is not None:
                turnover = sum(abs(a - b) for a, b in zip(fold_w, prev_w))
                turnovers.append(turnover)
            prev_w = fold_w

        # Weight stability: 1 - mean pairwise weight distance / max possible
        if len(weight_snapshots) >= 2:
            dists = []
            for i in range(len(weight_snapshots) - 1):
                d = sum(
                    abs(weight_snapshots[i][j] - weight_snapshots[i + 1][j])
                    for j in range(len(weight_snapshots[0]))
                )
                dists.append(d)
            mean_dist = statistics.mean(dists)
            stability = max(0.0, 1.0 - mean_dist)
        else:
            stability = float("nan")

        return WalkForwardResult(
            method=method,
            mean_oos_sharpe=statistics.mean(fold_sharpes) if fold_sharpes else float("nan"),
            std_oos_sharpe=statistics.stdev(fold_sharpes) if len(fold_sharpes) > 1 else float("nan"),
            fold_sharpes=[round(s, 4) for s in fold_sharpes],
            mean_turnover=statistics.mean(turnovers) if turnovers else float("nan"),
            weight_stability=round(stability, 4) if not math.isnan(stability) else float("nan"),
        )

    # ------------------------------------------------------------------
    # Correlation monitoring
    # ------------------------------------------------------------------

    def correlation_report(self) -> Dict:
        """Return pairwise correlation matrix for active signals."""
        ids = self.active_signals
        X = self._build_return_matrix(ids)
        if len(X) < 5:
            return {"error": "Insufficient data."}
        corr = _corr_matrix(X)
        result = {}
        for i, sid_i in enumerate(ids):
            result[sid_i] = {
                ids[j]: round(corr[j][i], 4)
                for j in range(len(ids))
                if j != i
            }
        return result

    def penalize_correlated_signals(
        self, correlation_threshold: float = 0.70
    ) -> List[str]:
        """
        Identify and down-weight signals that are highly correlated.
        Returns list of penalised signal IDs.
        """
        ids = self.active_signals
        X = self._build_return_matrix(ids)
        if len(X) < 5:
            return []
        corr = _corr_matrix(X)
        penalised = []
        N = len(ids)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(corr[j][i]) > correlation_threshold:
                    # Down-weight the signal with lower Sharpe
                    meta_i = self._signals[ids[i]]
                    meta_j = self._signals[ids[j]]
                    sharpe_i = meta_i.sharpe_history[-1] if meta_i.sharpe_history else 0.0
                    sharpe_j = meta_j.sharpe_history[-1] if meta_j.sharpe_history else 0.0
                    if sharpe_i < sharpe_j:
                        meta_i.correlation_penalty *= CORRELATION_PENALTY
                        penalised.append(ids[i])
                    else:
                        meta_j.correlation_penalty *= CORRELATION_PENALTY
                        penalised.append(ids[j])
        return list(set(penalised))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_return_matrix(self, ids: List[str]) -> List[List[float]]:
        """
        Build aligned return matrix (T x N) from internal buffers.
        Uses the minimum available length across signals.
        """
        if not ids:
            return []
        lengths = [len(self._return_buffer[sid]) for sid in ids if sid in self._return_buffer]
        if not lengths:
            return []
        T = min(lengths)
        return [
            [self._return_buffer[sid][-T + t] for sid in ids]
            for t in range(T)
        ]

    def _ew_scale_matrix(self, X: List[List[float]]) -> List[List[float]]:
        """Apply exponential weighting to rows (recent = higher weight)."""
        T = len(X)
        ff = self.forgetting_factor
        weights = [ff ** (T - 1 - t) for t in range(T)]
        w_sum = sum(weights)
        result = []
        for t, row in enumerate(X):
            scale = math.sqrt(weights[t] / w_sum * T)
            result.append([x * scale for x in row])
        return result

    def _apply_weight_cap(self, w: Dict[str, float]) -> Dict[str, float]:
        """Enforce per-signal weight cap with redistribution."""
        cap = self.max_weight
        excess = 0.0
        capped = {}
        for sid, wi in w.items():
            if wi > cap:
                excess += wi - cap
                capped[sid] = cap
            else:
                capped[sid] = wi
        # Redistribute excess proportionally to uncapped signals
        if excess > 1e-9:
            uncapped = [sid for sid, wi in capped.items() if wi < cap]
            if uncapped:
                per_signal = excess / len(uncapped)
                for sid in uncapped:
                    capped[sid] = min(capped[sid] + per_signal, cap)
        return capped

    def _apply_turnover_penalty(self, new_w: Dict[str, float]) -> Dict[str, float]:
        """
        Penalise deviations from current weights (turnover regularisation).
        New weight = argmin [ score_loss + lambda * ||w_new - w_old||_1 ]
        Approximated by linear interpolation toward current weights.
        """
        if self._current_weights is None:
            return new_w
        lam = self.turnover_penalty
        blended = {}
        for sid, w_new in new_w.items():
            w_old = self._current_weights.weights.get(sid, w_new)
            blended[sid] = (1.0 - lam) * w_new + lam * w_old
        # Renormalise
        total = sum(blended.values())
        if total > 1e-9:
            blended = {sid: v / total for sid, v in blended.items()}
        return blended

    def summary(self) -> Dict:
        """Return current state summary."""
        return {
            "n_signals": len(self._signals),
            "active_signals": self.active_signals,
            "in_crisis": self._in_crisis,
            "realized_vol": round(self._realized_vol, 4),
            "current_method": (
                self._current_weights.method if self._current_weights else None
            ),
            "current_weights": (
                {k: round(v, 4) for k, v in self._current_weights.weights.items()}
                if self._current_weights else {}
            ),
        }

    def __repr__(self) -> str:
        return (
            f"SignalCombiner("
            f"signals={len(self._signals)}, "
            f"active={len(self.active_signals)}, "
            f"crisis={self._in_crisis})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(7)

    combiner = SignalCombiner(
        forgetting_factor=0.95,
        ridge_alpha=0.5,
        turnover_penalty=0.05,
    )

    signals = {
        "momentum": "12-1 month momentum",
        "value":    "Book-to-market factor",
        "quality":  "ROE quality factor",
        "reversal": "Short-term reversal",
    }
    for sid, name in signals.items():
        combiner.register_signal(sid, name)

    # Generate synthetic return history
    T = 120
    base_ret = [random.gauss(0.0003, 0.01) for _ in range(T)]
    correlated_noise = [random.gauss(0, 0.005) for _ in range(T)]

    for t in range(T):
        ret_dict = {
            "momentum": base_ret[t] * 0.8 + correlated_noise[t] + random.gauss(0, 0.003),
            "value":    base_ret[t] * 0.3 + random.gauss(0, 0.008),
            "quality":  base_ret[t] * 0.5 + random.gauss(0, 0.006),
            "reversal": -base_ret[t] * 0.4 + random.gauss(0, 0.007),
        }
        combiner.push_returns(ret_dict, target_return=base_ret[t])

    print("=== Summary ===")
    print(combiner.summary())

    print("\n=== Equal Weight ===")
    print(combiner.combine("equal_weight", crisis_override=False))

    print("\n=== Sharpe Weight ===")
    w = combiner.combine("sharpe_weight", crisis_override=False)
    print(w.weights)

    print("\n=== Min Correlation Weight ===")
    w = combiner.combine("min_correlation", crisis_override=False)
    print(w.weights)

    print("\n=== Ridge Weight ===")
    w = combiner.combine("ridge", crisis_override=False)
    print(w.weights)

    print("\n=== Elastic Net Weight ===")
    w = combiner.combine("elastic_net", crisis_override=False)
    print(w.weights)

    print("\n=== Stacking Weight ===")
    w = combiner.combine("stacking", crisis_override=False)
    print(w.weights)

    print("\n=== Walk-Forward CV (ridge) ===")
    wf = combiner.walk_forward_cv(method="ridge", n_folds=4)
    print(wf)

    print("\n=== Correlation Report ===")
    print(combiner.correlation_report())

    print("\n=== Penalise Correlated (threshold=0.5) ===")
    print("Penalised:", combiner.penalize_correlated_signals(correlation_threshold=0.5))
