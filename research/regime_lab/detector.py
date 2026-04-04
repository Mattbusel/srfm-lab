"""
research/regime_lab/detector.py
================================
Multiple regime-detection methods for financial time-series.

Classes
-------
HMMRegimeDetector          — Gaussian HMM with EM (pure numpy fallback)
RollingVolRegimeDetector   — Rolling-volatility + trend percentile method
TrendRegimeDetector        — EMA / ATR threshold-based method
ChangePointDetector        — PELT dynamic-programming segmentation
EnsembleRegimeDetector     — Voting ensemble over all detectors

Functions
---------
regime_agreement_score(detector_outputs) -> float
regime_detection_metrics(predicted, true_regimes) -> dict
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime constants
# ---------------------------------------------------------------------------
BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"
REGIMES  = (BULL, BEAR, SIDEWAYS, HIGH_VOL)


# ===========================================================================
# Helper maths
# ===========================================================================

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average via pandas for numerical stability."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()


def _rolling_std(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series)
    return s.rolling(window=window, min_periods=1).std(ddof=1).to_numpy()


def _rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(series)
    return s.rolling(window=window, min_periods=1).mean().to_numpy()


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    prev_close = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows,
         np.maximum(np.abs(highs - prev_close),
                    np.abs(lows  - prev_close)))
    return tr


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
          period: int = 14) -> np.ndarray:
    tr = _true_range(highs, lows, closes)
    return _ema(tr, period)


# ===========================================================================
# 1. HMMRegimeDetector — pure-numpy Gaussian HMM with EM
# ===========================================================================

class _GaussianHMM:
    """
    Minimal Gaussian HMM (full covariance) implemented in pure NumPy.

    Each hidden state k emits a multivariate Gaussian N(mu_k, Sigma_k).
    Parameters are estimated via the Baum-Welch EM algorithm.
    """

    def __init__(self, n_states: int = 4, covariance_type: str = "full",
                 n_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = None):
        self.n_states        = n_states
        self.covariance_type = covariance_type
        self.n_iter          = n_iter
        self.tol             = tol
        self.rng             = np.random.default_rng(random_state)

        # Model parameters (set after fit)
        self.startprob_: Optional[np.ndarray] = None   # (K,)
        self.transmat_:  Optional[np.ndarray] = None   # (K, K)
        self.means_:     Optional[np.ndarray] = None   # (K, D)
        self.covars_:    Optional[np.ndarray] = None   # (K, D, D)  full
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #

    def _init_params(self, X: np.ndarray) -> None:
        K, D = self.n_states, X.shape[1]

        self.startprob_ = np.full(K, 1.0 / K)

        trans = self.rng.dirichlet(np.ones(K) * 10, size=K)
        self.transmat_ = trans / trans.sum(axis=1, keepdims=True)

        # K-means++ style centre selection
        idx = self.rng.integers(0, len(X))
        centres = [X[idx]]
        for _ in range(K - 1):
            dists = np.array([min(np.sum((x - c) ** 2) for c in centres) for x in X])
            probs = dists / dists.sum()
            idx   = self.rng.choice(len(X), p=probs)
            centres.append(X[idx])
        self.means_ = np.array(centres)

        cov_base = np.cov(X.T) + np.eye(D) * 1e-3
        self.covars_ = np.stack([cov_base.copy() for _ in range(K)])

    # ------------------------------------------------------------------ #
    # Log-probability helpers
    # ------------------------------------------------------------------ #

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """Return (T, K) log-emission matrix."""
        T = len(X)
        K = self.n_states
        log_b = np.zeros((T, K))
        for k in range(K):
            mu  = self.means_[k]
            cov = self.covars_[k] + np.eye(mu.shape[0]) * 1e-6
            try:
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    cov = cov + np.eye(mu.shape[0]) * 1e-4
                    _, logdet = np.linalg.slogdet(cov)
                inv_cov = np.linalg.inv(cov)
                diff    = X - mu
                mahal   = np.einsum('ti,ij,tj->t', diff, inv_cov, diff)
                log_b[:, k] = -0.5 * (mahal + logdet + mu.shape[0] * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                log_b[:, k] = -1e30
        return log_b

    # ------------------------------------------------------------------ #
    # Forward-backward
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_sum_exp(a: np.ndarray) -> float:
        m = np.max(a)
        return m + np.log(np.sum(np.exp(a - m)))

    def _forward(self, log_b: np.ndarray) -> Tuple[np.ndarray, float]:
        T, K = log_b.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        log_trans = np.log(self.transmat_ + 1e-300)

        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = (self._log_sum_exp(log_alpha[t-1] + log_trans[:, j])
                                   + log_b[t, j])
        log_likelihood = self._log_sum_exp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        T, K = log_b.shape
        log_beta = np.zeros((T, K))
        log_trans = np.log(self.transmat_ + 1e-300)

        for t in range(T - 2, -1, -1):
            for i in range(K):
                vals = log_trans[i] + log_b[t+1] + log_beta[t+1]
                log_beta[t, i] = self._log_sum_exp(vals)
        return log_beta

    # ------------------------------------------------------------------ #
    # EM steps
    # ------------------------------------------------------------------ #

    def _e_step(self, X: np.ndarray, log_b: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, float]:
        log_alpha, log_lik = self._forward(log_b)
        log_beta            = self._backward(log_b)
        log_trans           = np.log(self.transmat_ + 1e-300)
        T, K                = log_b.shape

        # gamma: (T, K)
        log_gamma = log_alpha + log_beta
        log_gamma -= log_gamma.max(axis=1, keepdims=True)
        gamma      = np.exp(log_gamma)
        gamma     /= gamma.sum(axis=1, keepdims=True)

        # xi: (T-1, K, K)
        log_xi = np.zeros((T-1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    log_xi[t, i, j] = (log_alpha[t, i] + log_trans[i, j]
                                       + log_b[t+1, j] + log_beta[t+1, j])
            log_xi[t] -= self._log_sum_exp(log_xi[t].ravel())
        xi = np.exp(log_xi)

        return gamma, xi, log_lik

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        K, D = self.n_states, X.shape[1]

        self.startprob_ = gamma[0] / (gamma[0].sum() + 1e-300)

        xi_sum = xi.sum(axis=0)
        self.transmat_ = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-300)

        gamma_sum = gamma.sum(axis=0)  # (K,)
        self.means_ = (gamma.T @ X) / (gamma_sum[:, None] + 1e-300)

        for k in range(K):
            diff = X - self.means_[k]
            w    = gamma[:, k]
            if self.covariance_type == "full":
                cov = (w[:, None] * diff).T @ diff
                self.covars_[k] = cov / (gamma_sum[k] + 1e-300) + np.eye(D) * 1e-4
            else:
                var = (w[:, None] * diff ** 2).sum(axis=0) / (gamma_sum[k] + 1e-300) + 1e-6
                self.covars_[k] = np.diag(var)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray) -> "_GaussianHMM":
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._init_params(X)
        prev_ll = -np.inf
        for i in range(self.n_iter):
            log_b        = self._log_emission(X)
            gamma, xi, ll = self._e_step(X, log_b)
            self._m_step(X, gamma, xi)
            if abs(ll - prev_ll) < self.tol:
                logger.debug("HMM EM converged at iteration %d  ll=%.4f", i, ll)
                break
            prev_ll = ll
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding → integer state sequence."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, K    = len(X), self.n_states
        log_b   = self._log_emission(X)
        log_t   = np.log(self.transmat_ + 1e-300)
        delta   = np.zeros((T, K))
        psi     = np.zeros((T, K), dtype=int)

        delta[0] = np.log(self.startprob_ + 1e-300) + log_b[0]
        for t in range(1, T):
            trans_scores = delta[t-1, :, None] + log_t
            psi[t]   = trans_scores.argmax(axis=0)
            delta[t] = trans_scores.max(axis=0) + log_b[t]

        path    = np.zeros(T, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Smoothed state probabilities (forward-backward)."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        log_b   = self._log_emission(X)
        log_alpha, _ = self._forward(log_b)
        log_beta     = self._backward(log_b)
        log_gamma    = log_alpha + log_beta
        log_gamma   -= log_gamma.max(axis=1, keepdims=True)
        gamma        = np.exp(log_gamma)
        gamma       /= gamma.sum(axis=1, keepdims=True)
        return gamma


# ---------------------------------------------------------------------------
# Public detector wrapper
# ---------------------------------------------------------------------------

class HMMRegimeDetector:
    """
    Gaussian HMM regime detector.

    Tries to use hmmlearn if installed; falls back to the pure-numpy
    implementation above.

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 4).
    covariance_type : str
        'full' or 'diag' (default 'full').
    n_iter : int
        Maximum EM iterations (default 100).
    random_state : int or None
        Reproducibility seed.
    """

    def __init__(self, n_states: int = 4, covariance_type: str = "full",
                 n_iter: int = 100, tol: float = 1e-4,
                 random_state: Optional[int] = 42):
        self.n_states        = n_states
        self.covariance_type = covariance_type
        self.n_iter          = n_iter
        self.tol             = tol
        self.random_state    = random_state
        self._model: Any     = None
        self._use_hmmlearn   = False
        self._state_map: Dict[int, str] = {}

        # Attempt hmmlearn
        try:
            import hmmlearn.hmm as _hmm  # type: ignore
            self._HMMClass   = _hmm.GaussianHMM
            self._use_hmmlearn = True
            logger.debug("HMMRegimeDetector: using hmmlearn backend")
        except ImportError:
            self._HMMClass   = _GaussianHMM
            logger.debug("HMMRegimeDetector: using pure-numpy backend")

    # ------------------------------------------------------------------ #

    def fit(self, returns: np.ndarray | pd.Series) -> "HMMRegimeDetector":
        """
        Fit the HMM to a 1-D returns series.

        Parameters
        ----------
        returns : array-like of shape (T,)

        Returns
        -------
        self
        """
        X = np.asarray(returns, dtype=float).reshape(-1, 1)

        if self._use_hmmlearn:
            self._model = self._HMMClass(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
        else:
            self._model = self._HMMClass(
                n_states=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                tol=self.tol,
                random_state=self.random_state,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X)

        self._build_state_map()
        return self

    # ------------------------------------------------------------------ #

    def _build_state_map(self) -> None:
        """
        Heuristically assign semantic regime labels to HMM state indices
        based on the fitted emission means and variances.

        Strategy:
          1. Sort states by mean return.
          2. Identify the highest-variance state → HIGH_VOL.
          3. Among remaining: highest mean → BULL, lowest → BEAR, middle → SIDEWAYS.
        """
        if self._use_hmmlearn:
            means  = self._model.means_.ravel()
            if self._model.covariance_type == "full":
                vars_  = np.array([self._model.covars_[k][0, 0]
                                   for k in range(self.n_states)])
            else:
                vars_  = self._model.covars_.ravel()
        else:
            means = self._model.means_.ravel()
            vars_ = np.array([self._model.covars_[k][0, 0]
                               for k in range(self.n_states)])

        high_vol_state = int(np.argmax(vars_))
        remaining      = [i for i in range(self.n_states) if i != high_vol_state]
        remaining_sorted = sorted(remaining, key=lambda i: means[i])

        self._state_map = {high_vol_state: HIGH_VOL}
        if len(remaining_sorted) == 1:
            self._state_map[remaining_sorted[0]] = BULL
        elif len(remaining_sorted) == 2:
            self._state_map[remaining_sorted[0]] = BEAR
            self._state_map[remaining_sorted[1]] = BULL
        else:
            self._state_map[remaining_sorted[0]] = BEAR
            self._state_map[remaining_sorted[-1]] = BULL
            for mid in remaining_sorted[1:-1]:
                self._state_map[mid] = SIDEWAYS

        logger.debug("HMM state map: %s", self._state_map)

    # ------------------------------------------------------------------ #

    def predict(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """
        Viterbi-decode regime labels.

        Returns
        -------
        np.ndarray of dtype int, shape (T,)
            Integer state indices.
        """
        X = np.asarray(returns, dtype=float).reshape(-1, 1)
        if self._use_hmmlearn:
            _, states = self._model.decode(X)
        else:
            states = self._model.predict(X)
        return states

    def predict_proba(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """
        Return smoothed state-occupation probabilities.

        Returns
        -------
        np.ndarray of shape (T, n_states)
        """
        X = np.asarray(returns, dtype=float).reshape(-1, 1)
        if self._use_hmmlearn:
            return self._model.predict_proba(X)
        else:
            return self._model.predict_proba(X)

    def decode_states(self, states: np.ndarray) -> np.ndarray:
        """
        Map integer state indices to regime name strings.

        Parameters
        ----------
        states : array of int

        Returns
        -------
        np.ndarray of str  (same length as states)
        """
        decoder = np.vectorize(lambda s: self._state_map.get(int(s), SIDEWAYS))
        return decoder(states)

    def fit_predict(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """Convenience: fit + Viterbi-decode → regime name array."""
        self.fit(returns)
        return self.decode_states(self.predict(returns))

    @property
    def transition_matrix(self) -> np.ndarray:
        if not self._model:
            raise RuntimeError("Call fit() first.")
        if self._use_hmmlearn:
            return self._model.transmat_
        return self._model.transmat_

    @property
    def emission_means(self) -> np.ndarray:
        if self._use_hmmlearn:
            return self._model.means_.ravel()
        return self._model.means_.ravel()


# ===========================================================================
# 2. RollingVolRegimeDetector
# ===========================================================================

class RollingVolRegimeDetector:
    """
    Simple regime detector based on rolling volatility percentile and trend.

    Rules
    -----
    1. Compute rolling standard deviation of log-returns over *vol_window*.
    2. Compute rolling percentile of vol series over trailing *pct_lookback* bars.
    3. If current vol > *high_vol_pct*-th percentile → HIGH_VOL.
    4. Else compute rolling trend: close vs. SMA(*trend_window*).
       - Close > SMA + eps → BULL
       - Close < SMA - eps → BEAR
       - Otherwise         → SIDEWAYS

    Parameters
    ----------
    vol_window      : int  (default 20)
    trend_window    : int  (default 50)
    high_vol_pct    : float (default 80.0)
    pct_lookback    : int  (default 252)  — window to compute vol percentile
    sideways_band   : float (default 0.01) — ±1 % of SMA is "sideways"
    """

    def __init__(self, vol_window: int = 20, trend_window: int = 50,
                 high_vol_pct: float = 80.0, pct_lookback: int = 252,
                 sideways_band: float = 0.01):
        self.vol_window    = vol_window
        self.trend_window  = trend_window
        self.high_vol_pct  = high_vol_pct
        self.pct_lookback  = pct_lookback
        self.sideways_band = sideways_band

    def detect(self, prices: np.ndarray | pd.Series) -> np.ndarray:
        """
        Classify each bar into a regime.

        Parameters
        ----------
        prices : array-like of shape (T,) — closing prices

        Returns
        -------
        np.ndarray of str, shape (T,)
        """
        prices = np.asarray(prices, dtype=float)
        T      = len(prices)

        log_ret = np.diff(np.log(np.where(prices > 0, prices, 1e-10)))
        log_ret = np.concatenate([[0.0], log_ret])

        rolling_vol = _rolling_std(log_ret, self.vol_window)

        # Rolling percentile rank of current vol
        vol_pct_rank = np.zeros(T)
        for t in range(T):
            lo = max(0, t - self.pct_lookback + 1)
            window_vols = rolling_vol[lo : t + 1]
            if len(window_vols) < 2:
                vol_pct_rank[t] = 50.0
            else:
                vol_pct_rank[t] = float(np.sum(window_vols <= rolling_vol[t]) /
                                         len(window_vols) * 100)

        sma = _rolling_mean(prices, self.trend_window)

        regimes = np.empty(T, dtype=object)
        for t in range(T):
            if vol_pct_rank[t] >= self.high_vol_pct:
                regimes[t] = HIGH_VOL
            else:
                dev = (prices[t] - sma[t]) / (sma[t] + 1e-10)
                if dev > self.sideways_band:
                    regimes[t] = BULL
                elif dev < -self.sideways_band:
                    regimes[t] = BEAR
                else:
                    regimes[t] = SIDEWAYS

        return regimes

    def detect_series(self, prices: pd.Series) -> pd.Series:
        """Return a pd.Series of regime labels with the same index as *prices*."""
        labels = self.detect(prices.to_numpy())
        return pd.Series(labels, index=prices.index, name="regime")


# ===========================================================================
# 3. TrendRegimeDetector
# ===========================================================================

class TrendRegimeDetector:
    """
    EMA200 / EMA50 trend-following regime detector with ATR-based HIGH_VOL overlay.

    Rules (applied in order)
    ------------------------
    1. HIGH_VOL  : ATR(14) / close  > *atr_vol_threshold*
    2. BULL      : close > EMA200   AND  EMA50 > EMA200
    3. BEAR      : close < EMA200   AND  EMA50 < EMA200
    4. SIDEWAYS  : |close - EMA200| / EMA200 < *sideways_pct*
    5. Default   : BEAR  (below EMA200 but not clearly trending)

    Parameters
    ----------
    ema_fast    : int (default 50)
    ema_slow    : int (default 200)
    atr_period  : int (default 14)
    sideways_pct: float (default 0.03)  — within ±3 % of EMA200 = SIDEWAYS
    atr_vol_threshold : float (default 0.025) — ATR/close > 2.5 % → HIGH_VOL
    """

    def __init__(self, ema_fast: int = 50, ema_slow: int = 200,
                 atr_period: int = 14, sideways_pct: float = 0.03,
                 atr_vol_threshold: float = 0.025):
        self.ema_fast           = ema_fast
        self.ema_slow           = ema_slow
        self.atr_period         = atr_period
        self.sideways_pct       = sideways_pct
        self.atr_vol_threshold  = atr_vol_threshold

    def detect(self, prices: Optional[np.ndarray] = None,
               highs: Optional[np.ndarray] = None,
               lows: Optional[np.ndarray] = None,
               closes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect regimes from OHLC data.

        If only *prices* is provided, it is used as both closes/highs/lows
        (ATR will be approximate — just based on close-to-close moves).

        Parameters
        ----------
        prices : optional 1-D array of close prices
        highs, lows, closes : optional 1-D arrays (override *prices*)

        Returns
        -------
        np.ndarray of str, shape (T,)
        """
        if closes is None:
            if prices is None:
                raise ValueError("Provide either prices or closes.")
            closes = np.asarray(prices, dtype=float)
        else:
            closes = np.asarray(closes, dtype=float)

        if highs is None:
            highs = closes.copy()
        else:
            highs = np.asarray(highs, dtype=float)

        if lows is None:
            lows = closes.copy()
        else:
            lows = np.asarray(lows, dtype=float)

        T       = len(closes)
        ema50   = _ema(closes, self.ema_fast)
        ema200  = _ema(closes, self.ema_slow)
        atr     = _atr(highs, lows, closes, self.atr_period)
        atr_pct = atr / (closes + 1e-10)

        regimes = np.empty(T, dtype=object)
        for t in range(T):
            if atr_pct[t] > self.atr_vol_threshold:
                regimes[t] = HIGH_VOL
            elif closes[t] > ema200[t] and ema50[t] > ema200[t]:
                regimes[t] = BULL
            elif closes[t] < ema200[t] and ema50[t] < ema200[t]:
                regimes[t] = BEAR
            else:
                dev = abs(closes[t] - ema200[t]) / (ema200[t] + 1e-10)
                if dev < self.sideways_pct:
                    regimes[t] = SIDEWAYS
                else:
                    regimes[t] = BEAR

        return regimes

    def detect_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Convenience wrapper for a OHLC DataFrame with columns
        high/High, low/Low, close/Close (case-insensitive).
        """
        col = {c.lower(): c for c in df.columns}
        h = df[col.get("high",  col.get("h", list(col.values())[0]))].to_numpy()
        l = df[col.get("low",   col.get("l", list(col.values())[0]))].to_numpy()
        c = df[col.get("close", col.get("c", list(col.values())[0]))].to_numpy()
        labels = self.detect(closes=c, highs=h, lows=l)
        return pd.Series(labels, index=df.index, name="regime")


# ===========================================================================
# 4. ChangePointDetector — PELT
# ===========================================================================

@dataclass
class SegmentStats:
    start: int
    end: int
    mean: float
    std: float
    regime: str = SIDEWAYS


class ChangePointDetector:
    """
    PELT (Pruned Exact Linear Time) changepoint detector.

    Uses a Normal log-likelihood cost function with BIC or AIC penalty.

    References
    ----------
    Killick, Fearnhead & Eckley (2012), "Optimal Detection of Changepoints
    with a Linear Computational Cost", JASA.

    Parameters
    ----------
    penalty : str  — 'bic', 'aic', or 'manual'
    manual_penalty : float — penalty value when penalty='manual'
    min_size : int — minimum segment length (default 20)
    """

    def __init__(self, penalty: str = "bic", manual_penalty: float = 10.0,
                 min_size: int = 20):
        self.penalty        = penalty
        self.manual_penalty = manual_penalty
        self.min_size       = min_size

    # ------------------------------------------------------------------ #
    # Cost function (Normal log-likelihood)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cost(x: np.ndarray) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        var = float(np.var(x, ddof=1))
        if var <= 0:
            return 0.0
        return n * (np.log(2 * np.pi * var) + 1)

    # ------------------------------------------------------------------ #

    def _penalty_value(self, n: int) -> float:
        if self.penalty == "bic":
            return 2 * np.log(n)      # 2 params per segment (mean, var)
        elif self.penalty == "aic":
            return 2 * 2.0
        else:
            return float(self.manual_penalty)

    # ------------------------------------------------------------------ #

    def detect_changepoints(self, series: np.ndarray | pd.Series,
                            penalty: Optional[str] = None,
                            min_size: Optional[int] = None) -> List[int]:
        """
        Run PELT and return a list of changepoint indices (exclusive end of segment).

        Parameters
        ----------
        series : 1-D array of floats
        penalty : override instance penalty setting
        min_size : override instance min_size setting

        Returns
        -------
        List[int] — sorted indices where regime changes occur.
                    Index i means a change occurs *before* bar i.
        """
        x   = np.asarray(series, dtype=float)
        n   = len(x)
        pen = self._penalty_value(n) if penalty is None else self._penalty_value(n)
        ms  = self.min_size if min_size is None else min_size

        if penalty and penalty != self.penalty:
            old = self.penalty
            self.penalty = penalty
            pen = self._penalty_value(n)
            self.penalty = old

        # Build prefix-cost array for O(1) segment cost queries
        # cost(i,j) = _cost(x[i:j])
        # We use a cumulative approach: precompute prefix sums for mean/var
        cs      = np.cumsum(x)
        cs2     = np.cumsum(x ** 2)

        def seg_cost(i: int, j: int) -> float:
            """O(1) Normal cost for x[i:j]."""
            nn = j - i
            if nn < 2:
                return 0.0
            s  = cs[j-1]  - (cs[i-1]  if i > 0 else 0.0)
            s2 = cs2[j-1] - (cs2[i-1] if i > 0 else 0.0)
            var = max((s2 - s**2 / nn) / nn, 1e-12)
            return nn * (np.log(2 * np.pi * var) + 1)

        # PELT dynamic programming
        f     = np.full(n + 1, np.inf)
        f[0]  = -pen
        last  = [-1] * (n + 1)
        admissible = [0]

        for t in range(ms, n + 1):
            new_admissible = []
            best_cost = np.inf
            best_s    = 0
            for s in admissible:
                cost = f[s] + seg_cost(s, t) + pen
                if cost < best_cost:
                    best_cost = cost
                    best_s    = s
            f[t]    = best_cost
            last[t] = best_s
            # Pruning: keep s if f[s] + cost(s,t) <= f[t]
            for s in admissible:
                if f[s] + seg_cost(s, t) <= f[t]:
                    new_admissible.append(s)
            if t >= ms:
                new_admissible.append(t)
            admissible = new_admissible

        # Reconstruct
        cps   = []
        t     = n
        while last[t] != -1:
            s = last[t]
            if s > 0:
                cps.append(s)
            t = s
        cps.sort()
        return cps

    def segment_regimes(self, series: np.ndarray | pd.Series,
                        changepoints: Optional[List[int]] = None) -> np.ndarray:
        """
        Assign regime labels to each point based on changepoints.

        Strategy: within each segment, classify by mean return and std.

        Parameters
        ----------
        series       : 1-D returns array
        changepoints : if None, calls detect_changepoints(series)

        Returns
        -------
        np.ndarray of str, shape (T,)
        """
        x   = np.asarray(series, dtype=float)
        n   = len(x)
        if changepoints is None:
            changepoints = self.detect_changepoints(x)

        boundaries = [0] + changepoints + [n]
        segments: List[SegmentStats] = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i+1]
            seg  = x[s:e]
            segments.append(SegmentStats(
                start=s, end=e,
                mean=float(np.mean(seg)),
                std=float(np.std(seg, ddof=1) if len(seg) > 1 else 0.0),
            ))

        # Classify each segment
        all_means = np.array([sg.mean for sg in segments])
        all_stds  = np.array([sg.std  for sg in segments])
        mean_p50  = float(np.median(all_means))
        std_p75   = float(np.percentile(all_stds, 75)) if len(all_stds) else 1.0

        regimes = np.empty(n, dtype=object)
        for sg in segments:
            if sg.std > std_p75:
                label = HIGH_VOL
            elif sg.mean > mean_p50 * 0.5:
                label = BULL
            elif sg.mean < -abs(mean_p50) * 0.5:
                label = BEAR
            else:
                label = SIDEWAYS
            regimes[sg.start : sg.end] = label

        return regimes

    def get_segment_stats(self, series: np.ndarray,
                          changepoints: Optional[List[int]] = None
                          ) -> List[SegmentStats]:
        x   = np.asarray(series, dtype=float)
        n   = len(x)
        if changepoints is None:
            changepoints = self.detect_changepoints(x)
        boundaries = [0] + changepoints + [n]
        result = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i+1]
            seg  = x[s:e]
            result.append(SegmentStats(
                start=s, end=e,
                mean=float(np.mean(seg)),
                std=float(np.std(seg, ddof=1) if len(seg) > 1 else 0.0),
            ))
        return result


# ===========================================================================
# 5. EnsembleRegimeDetector
# ===========================================================================

class EnsembleRegimeDetector:
    """
    Voting ensemble over HMM, RollingVol, Trend, and ChangePoint detectors.

    Detectors are weighted (default equal weights).  The mode of weighted
    votes is used as the final regime label.

    Parameters
    ----------
    detectors : list of (name, detector, weight) triples.
                If None, defaults to all four detectors with equal weights.
    require_ohlc : bool — if True, expects full OHLC data
    """

    def __init__(self,
                 detectors: Optional[List[Tuple[str, Any, float]]] = None,
                 require_ohlc: bool = False):
        if detectors is None:
            self._detectors = [
                ("hmm",         HMMRegimeDetector(),         1.0),
                ("rolling_vol", RollingVolRegimeDetector(),  1.0),
                ("changepoint", ChangePointDetector(),       1.0),
            ]
        else:
            self._detectors = detectors
        self.require_ohlc = require_ohlc

    def fit(self, returns: np.ndarray | pd.Series,
            prices: Optional[np.ndarray] = None) -> "EnsembleRegimeDetector":
        """Fit all detectors that have a fit() method."""
        for name, det, _ in self._detectors:
            if hasattr(det, "fit"):
                try:
                    det.fit(returns)
                except Exception as exc:
                    logger.warning("EnsembleRegimeDetector: %s fit failed: %s", name, exc)
        return self

    def predict(self, returns: np.ndarray | pd.Series,
                prices: Optional[np.ndarray] = None,
                highs:  Optional[np.ndarray] = None,
                lows:   Optional[np.ndarray] = None,
                closes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run all detectors and return weighted-vote regime labels.

        Parameters
        ----------
        returns : 1-D log-return array
        prices  : 1-D price array  (for rolling-vol & trend detectors)
        highs, lows, closes : for TrendRegimeDetector

        Returns
        -------
        np.ndarray of str
        """
        returns = np.asarray(returns, dtype=float)
        T       = len(returns)

        all_labels: List[Tuple[np.ndarray, float]] = []

        for name, det, weight in self._detectors:
            try:
                if isinstance(det, HMMRegimeDetector):
                    det.fit(returns)
                    labels = det.decode_states(det.predict(returns))
                elif isinstance(det, RollingVolRegimeDetector):
                    p = prices if prices is not None else np.exp(np.cumsum(returns))
                    labels = det.detect(p)
                elif isinstance(det, TrendRegimeDetector):
                    p = closes if closes is not None else (
                        prices if prices is not None else np.exp(np.cumsum(returns)))
                    labels = det.detect(closes=p, highs=highs, lows=lows)
                elif isinstance(det, ChangePointDetector):
                    labels = det.segment_regimes(returns)
                else:
                    labels = det.detect(returns)
                all_labels.append((labels, weight))
            except Exception as exc:
                logger.warning("EnsembleRegimeDetector: %s predict failed: %s", name, exc)

        if not all_labels:
            return np.full(T, SIDEWAYS, dtype=object)

        # Weighted voting
        regime_votes: np.ndarray = np.zeros((T, len(REGIMES)))
        regime_idx = {r: i for i, r in enumerate(REGIMES)}

        for labels, w in all_labels:
            for t in range(T):
                r = str(labels[t]) if labels[t] is not None else SIDEWAYS
                idx = regime_idx.get(r, 2)
                regime_votes[t, idx] += w

        winner_idx = regime_votes.argmax(axis=1)
        result = np.array([REGIMES[i] for i in winner_idx])
        return result


# ===========================================================================
# 6. Agreement score and evaluation metrics
# ===========================================================================

def regime_agreement_score(detector_outputs: List[np.ndarray]) -> float:
    """
    Compute a scalar [0, 1] score measuring how much multiple detectors agree.

    Score = mean fraction of bars where all detectors give the same label.

    Parameters
    ----------
    detector_outputs : list of 1-D arrays of regime strings (same length)

    Returns
    -------
    float in [0, 1]
    """
    if not detector_outputs:
        return 0.0
    if len(detector_outputs) == 1:
        return 1.0

    matrix = np.column_stack([np.asarray(d, dtype=str) for d in detector_outputs])
    T      = matrix.shape[0]
    agree  = np.sum(np.all(matrix == matrix[:, :1], axis=1))
    return float(agree / T)


def regime_detection_metrics(predicted: np.ndarray | Sequence[str],
                              true_regimes: np.ndarray | Sequence[str]
                              ) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1 per regime.

    Parameters
    ----------
    predicted    : 1-D array of predicted regime labels
    true_regimes : 1-D array of ground-truth regime labels

    Returns
    -------
    Dict mapping regime → {precision, recall, f1, support}
    Also includes 'macro_avg' key.
    """
    pred = np.asarray(predicted, dtype=str)
    true = np.asarray(true_regimes, dtype=str)

    all_labels = sorted(set(true) | set(pred))
    metrics: Dict[str, Dict[str, float]] = {}

    precisions, recalls, f1s = [], [], []

    for label in all_labels:
        tp = int(np.sum((pred == label) & (true == label)))
        fp = int(np.sum((pred == label) & (true != label)))
        fn = int(np.sum((pred != label) & (true == label)))
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        metrics[label] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics["macro_avg"] = {
        "precision": round(float(np.mean(precisions)), 4),
        "recall":    round(float(np.mean(recalls)), 4),
        "f1":        round(float(np.mean(f1s)), 4),
        "support":   int(len(true)),
    }

    # Accuracy
    metrics["accuracy"] = {"value": round(float(np.mean(pred == true)), 4),
                            "n": int(len(true))}

    return metrics


# ===========================================================================
# 7. Utility: smooth noisy regime series
# ===========================================================================

def smooth_regime_series(regimes: np.ndarray, min_duration: int = 5) -> np.ndarray:
    """
    Remove very short regime spikes by replacing them with the surrounding regime.

    A segment shorter than *min_duration* bars is replaced with the previous
    regime (or next regime if at the start).

    Parameters
    ----------
    regimes      : 1-D array of regime strings
    min_duration : minimum allowable segment length

    Returns
    -------
    np.ndarray of str (same length)
    """
    regimes = np.asarray(regimes, dtype=object).copy()
    n       = len(regimes)
    if n == 0:
        return regimes

    # Find run-length encoding
    def rle(arr: np.ndarray) -> List[Tuple[Any, int, int]]:
        runs = []
        start = 0
        for i in range(1, len(arr)):
            if arr[i] != arr[i - 1]:
                runs.append((arr[start], start, i))
                start = i
        runs.append((arr[start], start, len(arr)))
        return runs

    changed = True
    while changed:
        changed = False
        runs    = rle(regimes)
        for label, s, e in runs:
            if (e - s) < min_duration:
                replacement = regimes[s - 1] if s > 0 else (regimes[e] if e < n else label)
                regimes[s:e] = replacement
                changed = True
                break  # restart after modification

    return regimes


# ===========================================================================
# 8. Detector factory
# ===========================================================================

def build_detector(method: str = "ensemble", **kwargs: Any) -> Any:
    """
    Factory function to instantiate a regime detector by name.

    Parameters
    ----------
    method : str — one of 'hmm', 'rolling_vol', 'trend', 'changepoint', 'ensemble'
    **kwargs : passed to detector constructor

    Returns
    -------
    detector instance
    """
    mapping = {
        "hmm":         HMMRegimeDetector,
        "rolling_vol": RollingVolRegimeDetector,
        "trend":       TrendRegimeDetector,
        "changepoint": ChangePointDetector,
        "ensemble":    EnsembleRegimeDetector,
    }
    cls = mapping.get(method.lower())
    if cls is None:
        raise ValueError(f"Unknown detector method '{method}'. "
                         f"Choose from: {list(mapping)}")
    return cls(**kwargs)


# ===========================================================================
# 9. Regime label helpers
# ===========================================================================

def regime_to_int(regime: str) -> int:
    """Map regime string to integer (for array storage)."""
    return {BULL: 0, BEAR: 1, SIDEWAYS: 2, HIGH_VOL: 3}.get(regime, 2)


def int_to_regime(i: int) -> str:
    """Map integer back to regime string."""
    return [BULL, BEAR, SIDEWAYS, HIGH_VOL][i % 4]


def encode_regimes(regimes: np.ndarray) -> np.ndarray:
    """Convert string regime array to integer array."""
    return np.vectorize(regime_to_int)(regimes)


def decode_regimes(encoded: np.ndarray) -> np.ndarray:
    """Convert integer array back to string regime array."""
    return np.vectorize(int_to_regime)(encoded)
