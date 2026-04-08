"""
information_theory.py — Information-theoretic measures for financial markets.

Functions:
  shannon_entropy          – KDE-based differential entropy of returns
  relative_entropy         – KL divergence D(P||Q)
  jensen_shannon_divergence– symmetric KL (bounded, metric)
  mutual_information       – I(X;Y) via KDE
  transfer_entropy         – TE(X→Y) directional information flow
  renyi_entropy            – Rényi entropy of order alpha
  tsallis_entropy          – Tsallis q-entropy
  permutation_entropy      – ordinal pattern complexity
  sample_entropy           – SampEn regularity measure
  approximate_entropy      – ApEn regularity measure
  market_efficiency_ratio  – entropy ratio vs maximum entropy
  information_ratio        – signal/noise decomposition of return series
  mdl_model_selection      – Minimum Description Length model selection
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kde_density(x: np.ndarray, bw: Optional[float] = None) -> stats.gaussian_kde:
    """Fit a Gaussian KDE; if bw is None, use Scott's rule."""
    kde = stats.gaussian_kde(x, bw_method=bw)
    return kde


def _entropy_from_kde(kde: stats.gaussian_kde, x: np.ndarray,
                      n_eval: int = 500) -> float:
    """Numerically integrate -p(x) log p(x) dx over the support."""
    lo, hi = float(x.min()), float(x.max())
    margin = (hi - lo) * 0.2
    xgrid = np.linspace(lo - margin, hi + margin, n_eval)
    p = kde.evaluate(xgrid)
    dx = xgrid[1] - xgrid[0]
    p = np.maximum(p, 1e-300)
    return float(-np.sum(p * np.log(p)) * dx)


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

def shannon_entropy(
    x: np.ndarray,
    bw: Optional[float] = None,
    n_eval: int = 500,
    base: float = math.e,
) -> float:
    """
    Differential Shannon entropy H(X) = -∫ p(x) ln p(x) dx
    estimated via kernel density estimation.

    Parameters
    ----------
    x    : 1-D array of observations (e.g. log-returns)
    bw   : KDE bandwidth (None → Scott's rule)
    base : logarithm base (e = nats, 2 = bits)

    Returns
    -------
    H in nats (or bits if base=2)
    """
    kde = _kde_density(x, bw)
    h_nats = _entropy_from_kde(kde, x, n_eval)
    return h_nats / math.log(base) if base != math.e else h_nats


# ---------------------------------------------------------------------------
# Relative entropy (KL divergence)
# ---------------------------------------------------------------------------

def relative_entropy(
    x: np.ndarray,
    y: np.ndarray,
    bw: Optional[float] = None,
    n_eval: int = 500,
) -> float:
    """
    KL divergence D(P||Q) = ∫ p(x) ln(p(x)/q(x)) dx.
    P is estimated from x, Q from y.
    Estimated on a common grid over the support of P.
    """
    kde_p = _kde_density(x, bw)
    kde_q = _kde_density(y, bw)
    lo, hi = float(x.min()), float(x.max())
    margin = (hi - lo) * 0.2
    xgrid = np.linspace(lo - margin, hi + margin, n_eval)
    p = np.maximum(kde_p.evaluate(xgrid), 1e-300)
    q = np.maximum(kde_q.evaluate(xgrid), 1e-300)
    dx = xgrid[1] - xgrid[0]
    kl = float(np.sum(p * np.log(p / q)) * dx)
    return max(kl, 0.0)   # numerical KL can be slightly negative


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(
    x: np.ndarray,
    y: np.ndarray,
    bw: Optional[float] = None,
    n_eval: int = 500,
) -> float:
    """
    JS divergence: JSD(P||Q) = 0.5 D(P||M) + 0.5 D(Q||M)  where M = (P+Q)/2.
    Bounded in [0, ln 2], symmetric.
    """
    kde_p = _kde_density(x, bw)
    kde_q = _kde_density(y, bw)
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    margin = (hi - lo) * 0.2
    xgrid = np.linspace(lo - margin, hi + margin, n_eval)
    p = np.maximum(kde_p.evaluate(xgrid), 1e-300)
    q = np.maximum(kde_q.evaluate(xgrid), 1e-300)
    m = 0.5 * (p + q)
    dx = xgrid[1] - xgrid[0]
    jsd = float(0.5 * np.sum(p * np.log(p / m) + q * np.log(q / m)) * dx)
    return max(jsd, 0.0)


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    n_eval: int = 50,
) -> float:
    """
    Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)
    estimated via bivariate KDE on a grid.
    """
    kde_xy = stats.gaussian_kde(np.vstack([x, y]))
    kde_x = _kde_density(x)
    kde_y = _kde_density(y)

    # Grid over joint support
    xg = np.linspace(float(x.min()), float(x.max()), n_eval)
    yg = np.linspace(float(y.min()), float(y.max()), n_eval)
    XX, YY = np.meshgrid(xg, yg)
    xy_pts = np.vstack([XX.ravel(), YY.ravel()])

    p_xy = np.maximum(kde_xy(xy_pts).reshape(n_eval, n_eval), 1e-300)
    p_x = np.maximum(kde_x.evaluate(xg), 1e-300)
    p_y = np.maximum(kde_y.evaluate(yg), 1e-300)
    p_xpy = np.outer(p_y, p_x)   # independent product

    dx = xg[1] - xg[0]
    dy = yg[1] - yg[0]
    mi = float(np.sum(p_xy * np.log(p_xy / np.maximum(p_xpy, 1e-300))) * dx * dy)
    return max(mi, 0.0)


# ---------------------------------------------------------------------------
# Transfer entropy
# ---------------------------------------------------------------------------

def transfer_entropy(
    X: np.ndarray,
    Y: np.ndarray,
    lag: int = 1,
    k: int = 1,
    n_bins: int = 20,
) -> float:
    """
    Transfer entropy TE(X→Y) measuring directional information flow
    from X to Y beyond Y's own history.

    TE(X→Y) = H(Y_t | Y_{t-1..k}) - H(Y_t | Y_{t-1..k}, X_{t-lag..k})

    Implemented via binning (histogram) estimator for speed.

    Parameters
    ----------
    X, Y   : equal-length 1-D return series
    lag    : time lag for X (default 1)
    k      : history length (default 1)

    Returns
    -------
    TE in nats (≥ 0)
    """
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    T = n - max(k, lag)

    # Build state vectors
    y_now = Y[max(k, lag):]
    y_past = np.column_stack([Y[max(k, lag) - i - 1: n - i - 1] for i in range(k)])
    x_past = np.column_stack([X[max(k, lag) - lag - i: n - lag - i] for i in range(k)])

    def _hist_entropy(*arrays: np.ndarray) -> float:
        """Joint entropy via histogram for up to 3 variables."""
        data = np.column_stack(arrays)
        if data.shape[1] == 1:
            counts, _ = np.histogram(data[:, 0], bins=n_bins)
        elif data.shape[1] == 2:
            counts, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=n_bins)
        else:
            # 3-D histogram via binning
            edges = [np.linspace(data[:, j].min(), data[:, j].max(), n_bins + 1)
                     for j in range(data.shape[1])]
            counts, _ = np.histogramdd(data, bins=edges)
        probs = counts.ravel() / (counts.sum() + 1e-300)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    # TE = H(Y_t, Y_past) + H(Y_past, X_past) - H(Y_t, Y_past, X_past) - H(Y_past)
    h_yt_ypast = _hist_entropy(y_now.reshape(-1, 1), y_past)
    h_ypast_xpast = _hist_entropy(y_past, x_past)
    h_yt_ypast_xpast = _hist_entropy(y_now.reshape(-1, 1), y_past, x_past)
    h_ypast = _hist_entropy(y_past)

    te = h_yt_ypast + h_ypast_xpast - h_yt_ypast_xpast - h_ypast
    return max(float(te), 0.0)


# ---------------------------------------------------------------------------
# Rényi entropy
# ---------------------------------------------------------------------------

def renyi_entropy(
    x: np.ndarray,
    alpha: float = 2.0,
    bw: Optional[float] = None,
    n_eval: int = 500,
) -> float:
    """
    Rényi entropy of order alpha: H_α(X) = (1/(1-α)) log(∫ p(x)^α dx).
    alpha → 1 recovers Shannon entropy.
    """
    if abs(alpha - 1.0) < 1e-6:
        return shannon_entropy(x, bw=bw, n_eval=n_eval)
    kde = _kde_density(x, bw)
    lo, hi = float(x.min()), float(x.max())
    margin = (hi - lo) * 0.2
    xgrid = np.linspace(lo - margin, hi + margin, n_eval)
    p = np.maximum(kde.evaluate(xgrid), 1e-300)
    dx = xgrid[1] - xgrid[0]
    integral = float(np.sum(p ** alpha) * dx)
    return math.log(max(integral, 1e-300)) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Tsallis entropy
# ---------------------------------------------------------------------------

def tsallis_entropy(
    x: np.ndarray,
    q: float = 1.5,
    bw: Optional[float] = None,
    n_eval: int = 500,
) -> float:
    """
    Tsallis (non-extensive) entropy: S_q = (1 - ∫ p(x)^q dx) / (q - 1).
    q → 1 recovers Shannon entropy.
    """
    if abs(q - 1.0) < 1e-6:
        return shannon_entropy(x, bw=bw, n_eval=n_eval)
    kde = _kde_density(x, bw)
    lo, hi = float(x.min()), float(x.max())
    margin = (hi - lo) * 0.2
    xgrid = np.linspace(lo - margin, hi + margin, n_eval)
    p = np.maximum(kde.evaluate(xgrid), 1e-300)
    dx = xgrid[1] - xgrid[0]
    integral = float(np.sum(p ** q) * dx)
    return (1.0 - integral) / (q - 1.0)


# ---------------------------------------------------------------------------
# Permutation entropy
# ---------------------------------------------------------------------------

def permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Permutation entropy (Bandt & Pompe 2002).
    Measures complexity via relative frequencies of ordinal patterns.

    Parameters
    ----------
    x         : 1-D time series
    order     : embedding dimension (pattern length)
    delay     : time delay
    normalize : if True, divide by log(order!)

    Returns H_perm in nats (or bits if you change log base).
    """
    n = len(x)
    patterns: dict = {}
    total = 0

    for i in range(n - (order - 1) * delay):
        segment = x[i: i + order * delay: delay]
        perm = tuple(np.argsort(segment))
        patterns[perm] = patterns.get(perm, 0) + 1
        total += 1

    probs = np.array(list(patterns.values()), dtype=float) / total
    h = float(-np.sum(probs * np.log(np.maximum(probs, 1e-300))))
    if normalize:
        h_max = math.log(math.factorial(order))
        h = h / h_max if h_max > 0 else h
    return h


# ---------------------------------------------------------------------------
# Sample entropy
# ---------------------------------------------------------------------------

def sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
) -> float:
    """
    Sample entropy SampEn(m, r): negative log of conditional probability
    that two subsequences similar for m points remain similar at m+1.

    Parameters
    ----------
    m : template length
    r : tolerance (default 0.2 * std(x))

    Returns SampEn in nats.
    """
    n = len(x)
    if r is None:
        r = 0.2 * float(np.std(x))

    def _count_matches(template_len: int) -> int:
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(x[i: i + template_len] - x[j: j + template_len])) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return float("inf")
    return float(-math.log(max(A, 1) / max(B, 1)))


# ---------------------------------------------------------------------------
# Approximate entropy (ApEn)
# ---------------------------------------------------------------------------

def approximate_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
) -> float:
    """
    Approximate entropy ApEn(m, r) (Pincus 1991).
    Lower ApEn = more regularity, higher = more randomness/unpredictability.

    Parameters
    ----------
    m : template length
    r : tolerance (default 0.2 * std(x))
    """
    n = len(x)
    if r is None:
        r = 0.2 * float(np.std(x))

    def _phi(template_len: int) -> float:
        count = np.zeros(n - template_len + 1)
        for i in range(n - template_len + 1):
            template = x[i: i + template_len]
            for j in range(n - template_len + 1):
                if np.max(np.abs(x[j: j + template_len] - template)) <= r:
                    count[i] += 1
        count = np.maximum(count / (n - template_len + 1), 1e-300)
        return float(np.sum(np.log(count)) / (n - template_len + 1))

    return float(_phi(m) - _phi(m + 1))


# ---------------------------------------------------------------------------
# Market efficiency measure
# ---------------------------------------------------------------------------

def market_efficiency_ratio(
    returns: np.ndarray,
    bw: Optional[float] = None,
    n_eval: int = 500,
) -> float:
    """
    Entropy efficiency ratio = H(X) / H_max,
    where H_max = ln(range / bin_width) is the maximum entropy under a
    uniform distribution over the same support.

    A ratio near 1 indicates high entropy (efficient / random market).
    A ratio near 0 indicates low entropy (predictable / inefficient market).
    """
    h = shannon_entropy(returns, bw=bw, n_eval=n_eval)
    lo, hi = float(returns.min()), float(returns.max())
    support = hi - lo
    h_max = math.log(max(support, 1e-12))
    return float(h / h_max) if h_max > 0 else 0.0


# ---------------------------------------------------------------------------
# Information ratio of returns: signal vs noise decomposition
# ---------------------------------------------------------------------------

def information_ratio(
    returns: np.ndarray,
    signal_lag: int = 1,
) -> float:
    """
    Information ratio as the ratio of mutual information between lagged
    returns (signal) to the total entropy (signal + noise).

    IR = I(r_t; r_{t-lag}) / H(r_t)

    A higher IR implies more predictable (lower noise) return dynamics.
    """
    h = shannon_entropy(returns)
    if len(returns) < 2 * signal_lag + 10:
        return float("nan")
    x_lag = returns[:-signal_lag]
    y_cur = returns[signal_lag:]
    mi = mutual_information(x_lag, y_cur)
    return float(mi / max(h, 1e-10))


# ---------------------------------------------------------------------------
# Minimum Description Length (MDL) model selection
# ---------------------------------------------------------------------------

def mdl_model_selection(
    returns: np.ndarray,
    models: List[Tuple[str, Callable[[np.ndarray], float], int]],
) -> List[Tuple[str, float, float]]:
    """
    MDL model selection: choose the model that minimises description length.

    MDL(model) = -log L(model | data) + (k/2) * log(n)
    (equivalent to BIC; the model with lowest MDL is preferred)

    Parameters
    ----------
    returns : 1-D return series
    models  : list of (name, log_likelihood_fn, n_params)
              where log_likelihood_fn(returns) → float

    Returns
    -------
    List of (name, log_likelihood, MDL_score) sorted by MDL ascending.
    """
    n = len(returns)
    results = []
    for name, ll_fn, k in models:
        try:
            ll = float(ll_fn(returns))
            mdl = -ll + 0.5 * k * math.log(n)
            results.append((name, ll, mdl))
        except Exception:
            results.append((name, float("nan"), float("inf")))
    results.sort(key=lambda row: row[2])
    return results
