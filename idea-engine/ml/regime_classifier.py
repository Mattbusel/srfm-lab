"""
regime_classifier.py
--------------------
ML-based market regime classifier for the idea-engine.

Implements a simplified Hidden Markov Model (EM / Baum-Welch) from scratch,
a Gaussian Mixture Model, and threshold-based rules. Combines them into an
ensemble classifier for four market regimes.

Regimes
-------
  0: trending_bull
  1: trending_bear
  2: mean_reverting
  3: chaotic
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

REGIME_LABELS = ["trending_bull", "trending_bear", "mean_reverting", "chaotic"]
N_REGIMES = len(REGIME_LABELS)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _log_sum_exp(log_vals: List[float]) -> float:
    if not log_vals:
        return float("-inf")
    mx = max(log_vals)
    if mx == float("-inf"):
        return float("-inf")
    return mx + math.log(sum(math.exp(v - mx) for v in log_vals))


def _mvn_log_pdf(
    x: List[float], mean: List[float], var: List[float]
) -> float:
    """Diagonal-covariance multivariate Gaussian log PDF."""
    d = len(x)
    ll = -0.5 * d * math.log(2 * math.pi)
    for xi, mu, vi in zip(x, mean, var):
        if vi < 1e-10:
            vi = 1e-10
        ll -= 0.5 * (math.log(vi) + (xi - mu) ** 2 / vi)
    return ll


def _softmax(logits: List[float]) -> List[float]:
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _normalize_rows(matrix: List[List[float]]) -> List[List[float]]:
    result = []
    for row in matrix:
        s = sum(row)
        if s < 1e-12:
            result.append([1.0 / len(row)] * len(row))
        else:
            result.append([v / s for v in row])
    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class MarketFeatures:
    """Feature vector for a single time-step."""
    volatility: float          # realized vol (annualised)
    skewness: float            # return skewness over window
    autocorrelation: float     # lag-1 autocorrelation of returns
    volume_ratio: float        # current vol / MA vol
    cross_asset_corr: float    # avg cross-asset correlation
    trend_strength: float      # |mean return| / vol
    kurtosis: float            # excess kurtosis of returns

    def to_list(self) -> List[float]:
        return [
            self.volatility,
            self.skewness,
            self.autocorrelation,
            self.volume_ratio,
            self.cross_asset_corr,
            self.trend_strength,
            self.kurtosis,
        ]

    @staticmethod
    def n_features() -> int:
        return 7


def extract_features(
    returns: List[float],
    volume: Optional[List[float]] = None,
    cross_returns: Optional[List[List[float]]] = None,
    window: int = 20,
) -> List[MarketFeatures]:
    """
    Extract MarketFeatures from a returns series using a rolling window.

    Parameters
    ----------
    returns : list of float
        Daily return series (e.g. 0.01 = 1%).
    volume : list of float, optional
        Daily volume series (same length as returns).
    cross_returns : list of list of float, optional
        Returns for other assets (each inner list = one asset's returns).
    window : int
        Rolling window for feature computation.
    """
    n = len(returns)
    features: List[MarketFeatures] = []

    for i in range(window, n + 1):
        window_rets = returns[i - window: i]

        # Volatility (annualised)
        if len(window_rets) > 1:
            std = statistics.stdev(window_rets)
        else:
            std = 1e-6
        vol = std * math.sqrt(252)

        # Skewness
        mean_r = statistics.mean(window_rets)
        m3 = sum((r - mean_r) ** 3 for r in window_rets) / window
        m2 = sum((r - mean_r) ** 2 for r in window_rets) / window
        skew = m3 / (m2 ** 1.5 + 1e-12)

        # Kurtosis (excess)
        m4 = sum((r - mean_r) ** 4 for r in window_rets) / window
        kurt = m4 / (m2 ** 2 + 1e-12) - 3.0

        # Autocorrelation (lag-1)
        if len(window_rets) > 2:
            pairs = list(zip(window_rets[:-1], window_rets[1:]))
            mx = statistics.mean(window_rets[:-1])
            my = statistics.mean(window_rets[1:])
            num = sum((a - mx) * (b - my) for a, b in pairs)
            den = math.sqrt(
                sum((a - mx) ** 2 for a, _ in pairs)
                * sum((b - my) ** 2 for _, b in pairs)
            )
            acf1 = num / max(den, 1e-12)
        else:
            acf1 = 0.0

        # Volume ratio
        if volume is not None:
            vol_window = volume[max(0, i - window): i]
            vol_ma = statistics.mean(vol_window) if vol_window else 1.0
            vol_ratio = volume[i - 1] / max(vol_ma, 1e-9)
        else:
            vol_ratio = 1.0

        # Cross-asset correlation
        if cross_returns is not None:
            corrs = []
            for asset_rets in cross_returns:
                w_asset = asset_rets[i - window: i]
                if len(w_asset) < window:
                    continue
                ma, mb = statistics.mean(window_rets), statistics.mean(w_asset)
                num = sum((a - ma) * (b - mb)
                          for a, b in zip(window_rets, w_asset))
                dA = math.sqrt(sum((a - ma) ** 2 for a in window_rets))
                dB = math.sqrt(sum((b - mb) ** 2 for b in w_asset))
                corrs.append(num / max(dA * dB, 1e-12))
            cross_corr = statistics.mean(corrs) if corrs else 0.0
        else:
            cross_corr = 0.0

        # Trend strength
        trend = abs(mean_r) / max(std, 1e-9) * math.sqrt(window)

        features.append(
            MarketFeatures(
                volatility=vol,
                skewness=skew,
                autocorrelation=acf1,
                volume_ratio=vol_ratio,
                cross_asset_corr=cross_corr,
                trend_strength=trend,
                kurtosis=kurt,
            )
        )
    return features


# ---------------------------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------------------------

class GaussianMixtureModel:
    """
    EM-based Gaussian Mixture Model with diagonal covariance.

    Fitted independently per component (regime).
    """

    def __init__(self, n_components: int = N_REGIMES, max_iter: int = 100, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means_: List[List[float]] = []
        self.vars_: List[List[float]] = []
        self.weights_: List[float] = []
        self.fitted = False

    def fit(self, X: List[List[float]]) -> "GaussianMixtureModel":
        n, d = len(X), len(X[0])
        k = self.n_components

        # Initialise: split data into k roughly equal groups
        sorted_X = sorted(X, key=lambda row: row[0])
        chunk = max(1, n // k)
        self.means_ = [
            [statistics.mean(row[j] for row in sorted_X[i * chunk: (i + 1) * chunk])
             for j in range(d)]
            for i in range(k)
        ]
        self.vars_ = [[1.0] * d for _ in range(k)]
        self.weights_ = [1.0 / k] * k

        log_likelihood = float("-inf")
        for _ in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            # M-step
            self._m_step(X, responsibilities)
            # Convergence check
            new_ll = self._log_likelihood(X)
            if abs(new_ll - log_likelihood) < self.tol:
                break
            log_likelihood = new_ll

        self.fitted = True
        return self

    def _e_step(self, X: List[List[float]]) -> List[List[float]]:
        n = len(X)
        k = self.n_components
        resp = []
        for xi in X:
            log_probs = [
                math.log(max(self.weights_[j], 1e-12))
                + _mvn_log_pdf(xi, self.means_[j], self.vars_[j])
                for j in range(k)
            ]
            log_sum = _log_sum_exp(log_probs)
            resp.append([math.exp(lp - log_sum) for lp in log_probs])
        return resp

    def _m_step(
        self, X: List[List[float]], responsibilities: List[List[float]]
    ) -> None:
        n, d = len(X), len(X[0])
        k = self.n_components
        for j in range(k):
            r_j = [responsibilities[i][j] for i in range(n)]
            n_j = sum(r_j) + 1e-12
            self.weights_[j] = n_j / n
            self.means_[j] = [
                sum(r_j[i] * X[i][f] for i in range(n)) / n_j
                for f in range(d)
            ]
            self.vars_[j] = [
                max(
                    1e-6,
                    sum(r_j[i] * (X[i][f] - self.means_[j][f]) ** 2
                        for i in range(n)) / n_j
                )
                for f in range(d)
            ]

    def _log_likelihood(self, X: List[List[float]]) -> float:
        k = self.n_components
        ll = 0.0
        for xi in X:
            log_probs = [
                math.log(max(self.weights_[j], 1e-12))
                + _mvn_log_pdf(xi, self.means_[j], self.vars_[j])
                for j in range(k)
            ]
            ll += _log_sum_exp(log_probs)
        return ll

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        return self._e_step(X)

    def predict(self, X: List[List[float]]) -> List[int]:
        proba = self.predict_proba(X)
        return [max(range(self.n_components), key=lambda j: p[j]) for p in proba]


# ---------------------------------------------------------------------------
# Simplified HMM (Baum-Welch EM)
# ---------------------------------------------------------------------------

class HiddenMarkovModel:
    """
    Simplified HMM with diagonal Gaussian emissions.

    Implements:
      - Baum-Welch EM for parameter estimation
      - Viterbi decoding for most-likely state sequence
      - Forward algorithm for probability computation
    """

    def __init__(
        self,
        n_states: int = N_REGIMES,
        n_iter: int = 30,
        tol: float = 1e-4,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        # Parameters
        self.pi_: List[float] = [1.0 / n_states] * n_states
        self.A_: List[List[float]] = [
            [1.0 / n_states] * n_states for _ in range(n_states)
        ]
        self.means_: List[List[float]] = []
        self.vars_: List[List[float]] = []
        self.fitted = False

    def _init_params(self, X: List[List[float]]) -> None:
        n, d = len(X), len(X[0])
        k = self.n_states
        chunk = max(1, n // k)
        self.means_ = [
            [statistics.mean(X[i][f] for i in range(j * chunk, min((j + 1) * chunk, n)))
             for f in range(d)]
            for j in range(k)
        ]
        self.vars_ = [[1.0] * d for _ in range(k)]

    def _emission_log_prob(self, x: List[float], state: int) -> float:
        return _mvn_log_pdf(x, self.means_[state], self.vars_[state])

    def fit(self, X: List[List[float]]) -> "HiddenMarkovModel":
        self._init_params(X)
        T = len(X)
        k = self.n_states
        log_likelihood = float("-inf")

        for iteration in range(self.n_iter):
            # --- Forward pass ---
            log_alpha = [[float("-inf")] * k for _ in range(T)]
            for j in range(k):
                log_alpha[0][j] = (
                    math.log(max(self.pi_[j], 1e-12))
                    + self._emission_log_prob(X[0], j)
                )
            for t in range(1, T):
                for j in range(k):
                    log_sum = _log_sum_exp([
                        log_alpha[t - 1][i] + math.log(max(self.A_[i][j], 1e-12))
                        for i in range(k)
                    ])
                    log_alpha[t][j] = (
                        log_sum + self._emission_log_prob(X[t], j)
                    )

            # --- Backward pass ---
            log_beta = [[0.0] * k for _ in range(T)]
            for t in range(T - 2, -1, -1):
                for i in range(k):
                    log_beta[t][i] = _log_sum_exp([
                        math.log(max(self.A_[i][j], 1e-12))
                        + self._emission_log_prob(X[t + 1], j)
                        + log_beta[t + 1][j]
                        for j in range(k)
                    ])

            # --- E-step: compute gamma and xi ---
            log_gamma = []
            for t in range(T):
                log_g = [log_alpha[t][j] + log_beta[t][j] for j in range(k)]
                log_norm = _log_sum_exp(log_g)
                log_gamma.append([lg - log_norm for lg in log_g])

            log_xi: List[List[List[float]]] = []
            for t in range(T - 1):
                log_xi_t = []
                for i in range(k):
                    row = []
                    for j in range(k):
                        row.append(
                            log_alpha[t][i]
                            + math.log(max(self.A_[i][j], 1e-12))
                            + self._emission_log_prob(X[t + 1], j)
                            + log_beta[t + 1][j]
                        )
                    log_xi_t.append(row)
                log_norm_xi = _log_sum_exp(
                    [log_xi_t[i][j] for i in range(k) for j in range(k)]
                )
                log_xi.append(
                    [[log_xi_t[i][j] - log_norm_xi for j in range(k)]
                     for i in range(k)]
                )

            # --- M-step ---
            # Update pi
            self.pi_ = [math.exp(log_gamma[0][j]) for j in range(k)]

            # Update A
            for i in range(k):
                for j in range(k):
                    num = _log_sum_exp(
                        [log_xi[t][i][j] for t in range(T - 1)]
                    )
                    denom = _log_sum_exp(
                        [log_gamma[t][i] for t in range(T - 1)]
                    )
                    self.A_[i][j] = math.exp(num - denom) if denom != float("-inf") else 1e-6

            self.A_ = _normalize_rows(self.A_)

            # Update emission params
            d = len(X[0])
            for j in range(k):
                gamma_j = [math.exp(log_gamma[t][j]) for t in range(T)]
                sum_gamma = sum(gamma_j) + 1e-12
                self.means_[j] = [
                    sum(gamma_j[t] * X[t][f] for t in range(T)) / sum_gamma
                    for f in range(d)
                ]
                self.vars_[j] = [
                    max(
                        1e-6,
                        sum(gamma_j[t] * (X[t][f] - self.means_[j][f]) ** 2
                            for t in range(T)) / sum_gamma
                    )
                    for f in range(d)
                ]

            new_ll = _log_sum_exp(log_alpha[-1])
            if abs(new_ll - log_likelihood) < self.tol:
                break
            log_likelihood = new_ll

        self.fitted = True
        return self

    def predict(self, X: List[List[float]]) -> List[int]:
        """Viterbi decoding."""
        T, k = len(X), self.n_states
        dp = [[float("-inf")] * k for _ in range(T)]
        bp = [[-1] * k for _ in range(T)]

        for j in range(k):
            dp[0][j] = (
                math.log(max(self.pi_[j], 1e-12))
                + self._emission_log_prob(X[0], j)
            )

        for t in range(1, T):
            for j in range(k):
                best_score = float("-inf")
                best_prev = 0
                for i in range(k):
                    s = dp[t - 1][i] + math.log(max(self.A_[i][j], 1e-12))
                    if s > best_score:
                        best_score = s
                        best_prev = i
                dp[t][j] = best_score + self._emission_log_prob(X[t], j)
                bp[t][j] = best_prev

        # Backtrack
        path = [-1] * T
        path[T - 1] = max(range(k), key=lambda j: dp[T - 1][j])
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1][path[t + 1]]
        return path

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        """Return smoothed state probabilities via forward-backward."""
        T, k = len(X), self.n_states
        log_alpha = [[float("-inf")] * k for _ in range(T)]
        for j in range(k):
            log_alpha[0][j] = (
                math.log(max(self.pi_[j], 1e-12))
                + self._emission_log_prob(X[0], j)
            )
        for t in range(1, T):
            for j in range(k):
                log_sum = _log_sum_exp([
                    log_alpha[t - 1][i] + math.log(max(self.A_[i][j], 1e-12))
                    for i in range(k)
                ])
                log_alpha[t][j] = log_sum + self._emission_log_prob(X[t], j)

        log_beta = [[0.0] * k for _ in range(T)]
        for t in range(T - 2, -1, -1):
            for i in range(k):
                log_beta[t][i] = _log_sum_exp([
                    math.log(max(self.A_[i][j], 1e-12))
                    + self._emission_log_prob(X[t + 1], j)
                    + log_beta[t + 1][j]
                    for j in range(k)
                ])

        proba = []
        for t in range(T):
            log_g = [log_alpha[t][j] + log_beta[t][j] for j in range(k)]
            proba.append(_softmax(log_g))
        return proba


# ---------------------------------------------------------------------------
# Threshold-based rules
# ---------------------------------------------------------------------------

def _threshold_regime_proba(feat: MarketFeatures) -> List[float]:
    """
    Hard-rule heuristics converted to soft probabilities.

    Returns [p_bull, p_bear, p_mr, p_chaotic].
    """
    logits = [0.0, 0.0, 0.0, 0.0]
    # Trending bull: high trend strength, positive autocorrelation, moderate vol
    logits[0] += 2.0 * feat.trend_strength
    logits[0] += 1.5 * max(feat.autocorrelation, 0.0)
    logits[0] -= feat.volatility * 2.0   # penalise high vol
    logits[0] -= max(-feat.skewness, 0.0)  # penalise negative skew

    # Trending bear: trend strength but negative skew, high vol
    logits[1] += 1.5 * feat.trend_strength
    logits[1] += 1.5 * max(feat.autocorrelation, 0.0)
    logits[1] += max(-feat.skewness, 0.0) * 2.0
    logits[1] += feat.volatility

    # Mean reverting: negative autocorrelation, moderate vol
    logits[2] += 3.0 * max(-feat.autocorrelation, 0.0)
    logits[2] -= feat.volatility
    logits[2] -= 0.5 * feat.trend_strength

    # Chaotic: high vol, high kurtosis, high cross-asset correlation (contagion)
    logits[3] += feat.volatility * 2.0
    logits[3] += max(feat.kurtosis, 0.0) * 0.3
    logits[3] += feat.cross_asset_corr * 2.0
    logits[3] -= feat.trend_strength

    return _softmax(logits)


# ---------------------------------------------------------------------------
# Regime statistics helpers
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    label: str
    index: int
    frequency: float
    mean_duration_days: float
    max_duration_days: float
    transition_probs: List[float] = field(default_factory=list)


def _compute_regime_stats(
    state_sequence: List[int],
    transition_matrix: Optional[List[List[float]]] = None,
) -> List[RegimeStats]:
    n = len(state_sequence)
    stats = []
    for k in range(N_REGIMES):
        # Frequency
        freq = state_sequence.count(k) / max(n, 1)

        # Duration runs
        durations = []
        run = 0
        for s in state_sequence:
            if s == k:
                run += 1
            else:
                if run > 0:
                    durations.append(run)
                    run = 0
        if run > 0:
            durations.append(run)

        mean_dur = statistics.mean(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0

        trans = transition_matrix[k] if transition_matrix else [0.0] * N_REGIMES

        stats.append(RegimeStats(
            label=REGIME_LABELS[k],
            index=k,
            frequency=round(freq, 4),
            mean_duration_days=round(mean_dur, 2),
            max_duration_days=float(max_dur),
            transition_probs=[round(p, 4) for p in trans],
        ))
    return stats


# ---------------------------------------------------------------------------
# Ensemble classifier
# ---------------------------------------------------------------------------

class RegimeClassifier:
    """
    Ensemble market regime classifier combining:
      1. HMM (Baum-Welch EM)
      2. Gaussian Mixture Model
      3. Threshold-based rule system

    Regimes: trending_bull (0), trending_bear (1), mean_reverting (2), chaotic (3)

    Usage
    -----
    clf = RegimeClassifier()
    clf.fit(returns, volume, cross_returns)
    labels = clf.predict(returns_test)
    proba  = clf.predict_proba(returns_test)
    """

    def __init__(
        self,
        hmm_weight: float = 0.40,
        gmm_weight: float = 0.35,
        rule_weight: float = 0.25,
        window: int = 20,
        hmm_n_iter: int = 30,
        gmm_n_iter: int = 100,
    ):
        self.hmm_weight = hmm_weight
        self.gmm_weight = gmm_weight
        self.rule_weight = rule_weight
        self.window = window
        self._hmm = HiddenMarkovModel(n_states=N_REGIMES, n_iter=hmm_n_iter)
        self._gmm = GaussianMixtureModel(n_components=N_REGIMES, max_iter=gmm_n_iter)
        self._fitted = False
        self._transition_matrix: Optional[List[List[float]]] = None
        self._regime_stats: Optional[List[RegimeStats]] = None

    def fit(
        self,
        returns: List[float],
        volume: Optional[List[float]] = None,
        cross_returns: Optional[List[List[float]]] = None,
    ) -> "RegimeClassifier":
        """Fit HMM and GMM on the feature matrix derived from returns."""
        features = extract_features(returns, volume, cross_returns, self.window)
        if not features:
            raise ValueError("Not enough data to extract features.")
        X = [f.to_list() for f in features]

        self._hmm.fit(X)
        self._gmm.fit(X)
        self._fitted = True

        # Compute transition matrix from HMM state sequence
        hmm_states = self._hmm.predict(X)
        self._transition_matrix = self._estimate_transition_matrix(hmm_states)
        self._regime_stats = _compute_regime_stats(hmm_states, self._transition_matrix)
        return self

    def predict(
        self,
        returns: List[float],
        volume: Optional[List[float]] = None,
        cross_returns: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Return most likely regime label for each time-step."""
        proba = self.predict_proba(returns, volume, cross_returns)
        return [REGIME_LABELS[max(range(N_REGIMES), key=lambda j: p[j])] for p in proba]

    def predict_proba(
        self,
        returns: List[float],
        volume: Optional[List[float]] = None,
        cross_returns: Optional[List[List[float]]] = None,
    ) -> List[List[float]]:
        """Return ensemble probability distribution over regimes per time-step."""
        if not self._fitted:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        features = extract_features(returns, volume, cross_returns, self.window)
        if not features:
            return []
        X = [f.to_list() for f in features]

        hmm_proba = self._hmm.predict_proba(X)
        gmm_proba = self._gmm.predict_proba(X)
        rule_proba = [_threshold_regime_proba(f) for f in features]

        ensemble = []
        for h, g, r in zip(hmm_proba, gmm_proba, rule_proba):
            combined = [
                self.hmm_weight * h[j]
                + self.gmm_weight * g[j]
                + self.rule_weight * r[j]
                for j in range(N_REGIMES)
            ]
            s = sum(combined)
            ensemble.append([c / max(s, 1e-12) for c in combined])
        return ensemble

    def predict_current_regime(
        self,
        recent_returns: List[float],
        volume: Optional[List[float]] = None,
        cross_returns: Optional[List[List[float]]] = None,
    ) -> Tuple[str, List[float]]:
        """
        Classify the most recent regime given at least `window` return observations.
        Returns (label, probabilities).
        """
        proba_series = self.predict_proba(recent_returns, volume, cross_returns)
        if not proba_series:
            raise ValueError("Insufficient data.")
        last_proba = proba_series[-1]
        label = REGIME_LABELS[max(range(N_REGIMES), key=lambda j: last_proba[j])]
        return label, last_proba

    def get_transition_matrix(self) -> Optional[List[List[float]]]:
        return self._transition_matrix

    def get_regime_stats(self) -> Optional[List[RegimeStats]]:
        return self._regime_stats

    @staticmethod
    def _estimate_transition_matrix(states: List[int]) -> List[List[float]]:
        counts = [[0.0] * N_REGIMES for _ in range(N_REGIMES)]
        for t in range(len(states) - 1):
            counts[states[t]][states[t + 1]] += 1.0
        return _normalize_rows(counts)

    def __repr__(self) -> str:
        return (
            f"RegimeClassifier(fitted={self._fitted}, "
            f"hmm_w={self.hmm_weight}, gmm_w={self.gmm_weight}, "
            f"rule_w={self.rule_weight})"
        )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(0)

    # Synthetic returns: 4 regime blocks
    def _gen_returns(n: int, mu: float, sigma: float) -> List[float]:
        return [random.gauss(mu, sigma) for _ in range(n)]

    returns = (
        _gen_returns(60, 0.0008, 0.008)   # bull
        + _gen_returns(40, -0.001, 0.015)  # bear
        + _gen_returns(50, 0.0001, 0.006)  # mean reverting
        + _gen_returns(30, -0.0005, 0.025) # chaotic
    )

    clf = RegimeClassifier(window=15, hmm_n_iter=20, gmm_n_iter=50)
    clf.fit(returns)

    labels = clf.predict(returns)
    # Print regime counts
    from collections import Counter
    print("Regime counts:", Counter(labels))

    current_label, proba = clf.predict_current_regime(returns[-30:])
    print(f"Current regime: {current_label}")
    print(f"Probabilities: {[round(p, 3) for p in proba]}")

    print("\nTransition matrix:")
    tm = clf.get_transition_matrix()
    for i, row in enumerate(tm):
        print(f"  {REGIME_LABELS[i]}: {[round(p, 3) for p in row]}")

    print("\nRegime stats:")
    for rs in clf.get_regime_stats():
        print(f"  {rs.label}: freq={rs.frequency}, "
              f"mean_dur={rs.mean_duration_days}d, max_dur={rs.max_duration_days}d")
