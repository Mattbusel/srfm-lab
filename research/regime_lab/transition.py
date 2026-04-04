"""
research/regime_lab/transition.py
===================================
Regime transition analysis utilities.

Functions
---------
compute_transition_matrix(regime_series) -> np.ndarray
stationary_distribution(transition_matrix) -> np.ndarray
regime_duration_stats(regime_series) -> Dict[str, DurationStats]
regime_clustering(price_features, n_regimes=4) -> np.ndarray
forward_regime_probabilities(current_regime, transition_matrix, horizon=10)
conditional_strategy_performance(trades, transitions) -> pd.DataFrame
plot_transition_matrix(matrix, save_path)
plot_regime_timeline(regime_series, prices, save_path)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"
REGIMES  = (BULL, BEAR, SIDEWAYS, HIGH_VOL)

REGIME_COLORS: Dict[str, str] = {
    BULL:     "#2196F3",   # blue
    BEAR:     "#F44336",   # red
    SIDEWAYS: "#FF9800",   # orange
    HIGH_VOL: "#9C27B0",   # purple
}


# ===========================================================================
# 1. DurationStats dataclass
# ===========================================================================

@dataclass
class DurationStats:
    regime:      str
    count:       int         # number of distinct regime episodes
    mean_dur:    float       # mean episode length (bars)
    median_dur:  float
    max_dur:     int
    min_dur:     int
    std_dur:     float
    total_bars:  int         # total bars spent in this regime
    pct_time:    float       # fraction of total bars
    durations:   List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime":     self.regime,
            "count":      self.count,
            "mean_dur":   round(self.mean_dur, 2),
            "median_dur": round(self.median_dur, 2),
            "max_dur":    self.max_dur,
            "min_dur":    self.min_dur,
            "std_dur":    round(self.std_dur, 2),
            "total_bars": self.total_bars,
            "pct_time":   round(self.pct_time * 100, 2),
        }


# ===========================================================================
# 2. compute_transition_matrix
# ===========================================================================

def compute_transition_matrix(regime_series: np.ndarray | Sequence[str],
                                regimes: Optional[Tuple[str, ...]] = None,
                                smoothing: float = 0.0) -> np.ndarray:
    """
    Estimate the MLE Markov transition matrix from an observed regime sequence.

    Parameters
    ----------
    regime_series : 1-D array of regime label strings
    regimes       : ordered tuple of all possible regime labels (default: REGIMES)
    smoothing     : Laplace smoothing count added to each cell (default 0)

    Returns
    -------
    np.ndarray of shape (K, K)  — row i sums to 1
    P[i, j] = P(next = j | current = i)
    """
    if regimes is None:
        regimes = REGIMES
    K     = len(regimes)
    r2i   = {r: i for i, r in enumerate(regimes)}
    series = [str(r) for r in regime_series]

    counts = np.full((K, K), smoothing, dtype=float)
    for t in range(1, len(series)):
        i = r2i.get(series[t - 1])
        j = r2i.get(series[t])
        if i is not None and j is not None:
            counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    return counts / row_sums


# ===========================================================================
# 3. stationary_distribution
# ===========================================================================

def stationary_distribution(transition_matrix: np.ndarray,
                              method: str = "eigen") -> np.ndarray:
    """
    Compute the stationary distribution π of a Markov transition matrix.

    π satisfies: π P = π  and  sum(π) = 1

    Parameters
    ----------
    transition_matrix : (K, K) row-stochastic matrix
    method            : 'eigen' (default) or 'power' (power iteration)

    Returns
    -------
    np.ndarray of shape (K,)
    """
    P = np.asarray(transition_matrix, dtype=float)
    K = P.shape[0]

    if method == "power":
        pi = np.ones(K) / K
        for _ in range(10_000):
            pi_new = pi @ P
            if np.max(np.abs(pi_new - pi)) < 1e-12:
                break
            pi = pi_new
        return pi / pi.sum()

    # Eigen method: find left eigenvector for eigenvalue 1
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi  = eigvecs[:, idx].real
    pi  = np.abs(pi) / np.abs(pi).sum()
    return pi


# ===========================================================================
# 4. regime_duration_stats
# ===========================================================================

def regime_duration_stats(regime_series: np.ndarray | Sequence[str],
                           regimes: Optional[Tuple[str, ...]] = None
                           ) -> Dict[str, DurationStats]:
    """
    Compute statistics on the duration of each regime episode.

    Parameters
    ----------
    regime_series : 1-D sequence of regime labels
    regimes       : ordered tuple of all possible regime labels

    Returns
    -------
    Dict mapping regime → DurationStats
    """
    if regimes is None:
        regimes = REGIMES

    series = [str(r) for r in regime_series]
    T      = len(series)

    # Run-length encoding
    runs: Dict[str, List[int]] = {r: [] for r in regimes}
    if T == 0:
        return {r: DurationStats(r, 0, 0.0, 0.0, 0, 0, 0.0, 0, 0.0) for r in regimes}

    cur_regime = series[0]
    cur_len    = 1
    for t in range(1, T):
        if series[t] == cur_regime:
            cur_len += 1
        else:
            if cur_regime in runs:
                runs[cur_regime].append(cur_len)
            cur_regime = series[t]
            cur_len    = 1
    if cur_regime in runs:
        runs[cur_regime].append(cur_len)

    result: Dict[str, DurationStats] = {}
    for r in regimes:
        durs = runs[r]
        if not durs:
            result[r] = DurationStats(r, 0, 0.0, 0.0, 0, 0, 0.0, 0, 0.0, [])
            continue
        arr        = np.array(durs, dtype=int)
        total_bars = int(arr.sum())
        result[r]  = DurationStats(
            regime=r,
            count=len(durs),
            mean_dur=float(np.mean(arr)),
            median_dur=float(np.median(arr)),
            max_dur=int(arr.max()),
            min_dur=int(arr.min()),
            std_dur=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            total_bars=total_bars,
            pct_time=total_bars / T,
            durations=list(arr),
        )
    return result


def duration_stats_dataframe(regime_series: np.ndarray | Sequence[str]) -> pd.DataFrame:
    """Return regime_duration_stats as a formatted DataFrame."""
    stats = regime_duration_stats(regime_series)
    return pd.DataFrame([v.to_dict() for v in stats.values()])


# ===========================================================================
# 5. regime_clustering — KMeans / GMM
# ===========================================================================

def regime_clustering(price_features: np.ndarray | pd.DataFrame,
                       n_regimes: int = 4,
                       method: str = "kmeans",
                       random_state: int = 42) -> np.ndarray:
    """
    Cluster price features into *n_regimes* clusters using KMeans or GMM.

    Features should be columns such as returns, rolling vol, momentum, etc.

    Parameters
    ----------
    price_features : (T, D) feature matrix
    n_regimes      : number of clusters (default 4)
    method         : 'kmeans' or 'gmm'
    random_state   : seed

    Returns
    -------
    np.ndarray of str regime labels, shape (T,)
    """
    if isinstance(price_features, pd.DataFrame):
        X = price_features.to_numpy()
    else:
        X = np.asarray(price_features, dtype=float)

    # Replace NaN with column means
    col_means = np.nanmean(X, axis=0)
    inds      = np.where(np.isnan(X))
    X[inds]   = np.take(col_means, inds[1])

    labels: np.ndarray

    if method == "gmm":
        try:
            from sklearn.mixture import GaussianMixture  # type: ignore
            gmm    = GaussianMixture(n_components=n_regimes, random_state=random_state,
                                     covariance_type="full", n_init=5)
            labels = gmm.fit_predict(X)
        except ImportError:
            logger.warning("sklearn not available; falling back to KMeans.")
            method = "kmeans"

    if method == "kmeans":
        try:
            from sklearn.cluster import KMeans  # type: ignore
            km     = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
            labels = km.fit_predict(X)
        except ImportError:
            labels = _numpy_kmeans(X, n_regimes, random_state)

    # Map integer cluster labels to regime names
    # Heuristic: sort clusters by first feature (assumed to be mean return)
    cluster_means = np.array([X[labels == k, 0].mean() if (labels == k).any() else 0.0
                               for k in range(n_regimes)])
    sorted_clusters = np.argsort(cluster_means)  # ascending mean return

    regime_names = list(REGIMES[:n_regimes])
    # BEAR = lowest mean, BULL = highest, SIDEWAYS and HIGH_VOL in between
    ordered = [BEAR, SIDEWAYS, HIGH_VOL, BULL][:n_regimes]
    cluster_to_regime: Dict[int, str] = {}
    for i, c in enumerate(sorted_clusters):
        cluster_to_regime[int(c)] = ordered[i] if i < len(ordered) else SIDEWAYS

    return np.array([cluster_to_regime.get(int(l), SIDEWAYS) for l in labels])


def _numpy_kmeans(X: np.ndarray, k: int, seed: int = 42,
                   n_iter: int = 300) -> np.ndarray:
    """Minimal NumPy KMeans (Lloyd's algorithm)."""
    rng     = np.random.default_rng(seed)
    n       = len(X)
    indices = rng.choice(n, size=k, replace=False)
    centres = X[indices].copy()

    labels  = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # Assignment
        dists  = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
        new_lbl = dists.argmin(axis=1)
        if np.all(new_lbl == labels):
            break
        labels = new_lbl
        # Update
        for ki in range(k):
            mask = labels == ki
            if mask.any():
                centres[ki] = X[mask].mean(axis=0)

    return labels


def build_feature_matrix(prices: np.ndarray | pd.Series,
                          highs: Optional[np.ndarray] = None,
                          lows:  Optional[np.ndarray] = None,
                          windows: Tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """
    Construct a feature matrix suitable for regime_clustering.

    Features:
      - log_return (1-bar)
      - rolling vol (multiple windows)
      - rolling mean return (momentum)
      - ATR/price ratio if highs/lows provided

    Parameters
    ----------
    prices  : 1-D price series
    highs   : 1-D high series (optional)
    lows    : 1-D low series (optional)
    windows : rolling windows for vol/momentum

    Returns
    -------
    pd.DataFrame of shape (T, n_features)
    """
    prices_arr = np.asarray(prices, dtype=float)
    T          = len(prices_arr)
    log_ret    = np.diff(np.log(np.where(prices_arr > 0, prices_arr, 1e-10)))
    log_ret    = np.concatenate([[0.0], log_ret])

    features: Dict[str, np.ndarray] = {"log_ret": log_ret}

    for w in windows:
        s = pd.Series(log_ret)
        features[f"vol_{w}"]  = s.rolling(w, min_periods=1).std(ddof=1).to_numpy()
        features[f"mom_{w}"]  = s.rolling(w, min_periods=1).mean().to_numpy()

    if highs is not None and lows is not None:
        h = np.asarray(highs, dtype=float)
        l = np.asarray(lows,  dtype=float)
        prev_c = np.concatenate([[prices_arr[0]], prices_arr[:-1]])
        tr     = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
        atr14  = pd.Series(tr).ewm(span=14, adjust=False).mean().to_numpy()
        features["atr_pct"] = atr14 / (prices_arr + 1e-10)

    return pd.DataFrame(features)


# ===========================================================================
# 6. forward_regime_probabilities
# ===========================================================================

def forward_regime_probabilities(current_regime: str,
                                   transition_matrix: np.ndarray,
                                   horizon: int = 10,
                                   regimes: Optional[Tuple[str, ...]] = None
                                   ) -> pd.DataFrame:
    """
    Compute marginal regime-occupation probabilities over a forward horizon.

    Starts from *current_regime* (probability = 1) and iterates the
    transition matrix forward *horizon* steps.

    Parameters
    ----------
    current_regime    : starting regime label
    transition_matrix : (K, K) row-stochastic matrix
    horizon           : number of forward steps
    regimes           : ordered regime tuple (default REGIMES)

    Returns
    -------
    pd.DataFrame of shape (horizon + 1, K)
    index = step 0..horizon, columns = regime names
    """
    if regimes is None:
        regimes = REGIMES
    K   = len(regimes)
    P   = np.asarray(transition_matrix, dtype=float)
    r2i = {r: i for i, r in enumerate(regimes)}

    pi = np.zeros(K)
    idx = r2i.get(current_regime, 0)
    pi[idx] = 1.0

    rows = [pi.copy()]
    for _ in range(horizon):
        pi = pi @ P
        rows.append(pi.copy())

    df = pd.DataFrame(rows, columns=list(regimes))
    df.index.name = "step"
    return df


# ===========================================================================
# 7. conditional_strategy_performance
# ===========================================================================

def conditional_strategy_performance(
        trades: Any,
        transitions: Optional[np.ndarray] = None,
        regime_series: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Compute trade performance statistics conditional on regime transitions.

    For each observed regime→regime transition, computes:
      - mean P&L of trades entered in the *new* regime within the next N bars
      - win rate, Sharpe, count

    Parameters
    ----------
    trades        : list of trade dicts or pd.DataFrame
    transitions   : (T-1, 2) int array of (from_regime_idx, to_regime_idx) transitions.
                    If None, derived from regime_series.
    regime_series : 1-D string array of regimes (required if transitions is None)

    Returns
    -------
    pd.DataFrame with columns:
        from_regime, to_regime, count, mean_pnl, win_rate, sharpe
    """
    from research.regime_lab.stress import _extract_trades, _trade_pnl, _trade_regime

    trade_list = _extract_trades(trades)
    if not trade_list:
        return pd.DataFrame()

    # Group trades by regime
    regime_pnls: Dict[str, List[float]] = {r: [] for r in REGIMES}
    for trade in trade_list:
        r   = _trade_regime(trade)
        pnl = _trade_pnl(trade)
        if r in regime_pnls:
            regime_pnls[r].append(pnl)

    # Build transition pairs
    if regime_series is not None:
        series = [str(r) for r in regime_series]
        pairs  = [(series[t-1], series[t]) for t in range(1, len(series))
                  if series[t] != series[t-1]]   # only actual transitions
    else:
        pairs = []

    if not pairs:
        # No transition data: just report per-regime stats
        rows = []
        for r in REGIMES:
            pnls = regime_pnls.get(r, [])
            if pnls:
                arr = np.array(pnls, dtype=float)
                rows.append({
                    "from_regime": r,
                    "to_regime":   r,
                    "count":       len(arr),
                    "mean_pnl":    round(float(np.mean(arr)), 4),
                    "win_rate":    round(float(np.mean(arr > 0)), 4),
                    "sharpe":      round(_sharpe(arr), 4),
                })
        return pd.DataFrame(rows)

    # For each unique (from, to) transition, report P&L of trades
    # that land in the *to* regime
    transition_counts: Dict[Tuple[str, str], List[float]] = {}
    for fr, to in pairs:
        key = (fr, to)
        if key not in transition_counts:
            transition_counts[key] = []
        transition_counts[key].extend(regime_pnls.get(to, []))

    rows = []
    for (fr, to), pnls in sorted(transition_counts.items()):
        arr = np.array(pnls, dtype=float) if pnls else np.zeros(0)
        rows.append({
            "from_regime": fr,
            "to_regime":   to,
            "count":       len(arr),
            "mean_pnl":    round(float(np.mean(arr)) if len(arr) else 0.0, 4),
            "win_rate":    round(float(np.mean(arr > 0)) if len(arr) else 0.0, 4),
            "sharpe":      round(_sharpe(arr) if len(arr) > 1 else 0.0, 4),
        })

    return pd.DataFrame(rows).sort_values(["from_regime", "to_regime"])


def _sharpe(arr: np.ndarray, annualise: bool = False) -> float:
    if len(arr) < 2:
        return 0.0
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    if s == 0:
        return 0.0
    raw = m / s
    return raw * (252 ** 0.5) if annualise else raw


# ===========================================================================
# 8. plot_transition_matrix — heatmap
# ===========================================================================

def plot_transition_matrix(matrix: np.ndarray,
                             save_path: Optional[str] = None,
                             regimes: Optional[Tuple[str, ...]] = None,
                             title: str = "Regime Transition Matrix",
                             figsize: Tuple[int, int] = (7, 6)) -> Any:
    """
    Render the transition matrix as an annotated heatmap.

    Parameters
    ----------
    matrix    : (K, K) row-stochastic ndarray
    save_path : optional save path (PNG)
    regimes   : regime label tuple
    title     : plot title
    figsize   : matplotlib figsize

    Returns
    -------
    matplotlib Figure or None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        logger.warning("matplotlib not installed; cannot plot transition matrix.")
        return None

    if regimes is None:
        regimes = REGIMES
    K  = len(regimes)
    P  = np.asarray(matrix, dtype=float)[:K, :K]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(regimes, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(regimes, fontsize=10)
    ax.set_xlabel("To Regime",   fontsize=11)
    ax.set_ylabel("From Regime", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(K):
        for j in range(K):
            val  = P[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Transition Probability")
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Transition matrix plot saved to %s", save_path)

    return fig


# ===========================================================================
# 9. plot_regime_timeline
# ===========================================================================

def plot_regime_timeline(regime_series: np.ndarray | Sequence[str],
                          prices: np.ndarray | pd.Series,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (14, 6)) -> Any:
    """
    Two-panel plot:
      Top : price series with background shading by regime
      Bottom: regime integer encoding

    Parameters
    ----------
    regime_series : 1-D array of regime labels
    prices        : 1-D price array (same length)
    save_path     : optional PNG save path
    figsize       : matplotlib figsize

    Returns
    -------
    matplotlib Figure or None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed; cannot plot regime timeline.")
        return None

    regimes_arr = np.asarray([str(r) for r in regime_series])
    prices_arr  = np.asarray(prices, dtype=float)
    T           = min(len(regimes_arr), len(prices_arr))
    regimes_arr = regimes_arr[:T]
    prices_arr  = prices_arr[:T]
    x           = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

    # Price with shaded regime backgrounds
    ax1.plot(x, prices_arr, color="black", linewidth=0.8, zorder=3)
    ax1.set_ylabel("Price", fontsize=10)
    ax1.set_title("Price Series with Regime Background", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, T - 1)

    prev_regime = regimes_arr[0]
    start_idx   = 0
    for t in range(1, T):
        if regimes_arr[t] != prev_regime or t == T - 1:
            end_idx = t if regimes_arr[t] != prev_regime else T
            color   = REGIME_COLORS.get(prev_regime, "#AAAAAA")
            ax1.axvspan(start_idx, end_idx, alpha=0.15, color=color, zorder=1)
            prev_regime = regimes_arr[t]
            start_idx   = t

    # Regime int encoding
    regime_int = np.array([{BULL: 0, BEAR: 1, SIDEWAYS: 2, HIGH_VOL: 3}.get(r, 2)
                             for r in regimes_arr])
    colors_bar = [REGIME_COLORS.get(r, "#AAAAAA") for r in regimes_arr]

    ax2.bar(x, np.ones(T), color=colors_bar, width=1.0, align="edge")
    ax2.set_yticks([])
    ax2.set_xlabel("Bar Index", fontsize=10)
    ax2.set_title("Regime Timeline", fontsize=10)
    ax2.set_xlim(0, T)

    # Legend
    patches = [mpatches.Patch(color=REGIME_COLORS[r], label=r) for r in REGIMES]
    ax1.legend(handles=patches, loc="upper left", fontsize=8)

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Regime timeline plot saved to %s", save_path)

    return fig


# ===========================================================================
# 10. Regime transition frequency table
# ===========================================================================

def transition_frequency_table(regime_series: np.ndarray | Sequence[str],
                                 regimes: Optional[Tuple[str, ...]] = None
                                 ) -> pd.DataFrame:
    """
    Count the number of observed transitions between each pair of regimes.

    Returns
    -------
    pd.DataFrame with raw counts (not probabilities)
    """
    if regimes is None:
        regimes = REGIMES
    K   = len(regimes)
    r2i = {r: i for i, r in enumerate(regimes)}
    series = [str(r) for r in regime_series]

    counts = np.zeros((K, K), dtype=int)
    for t in range(1, len(series)):
        i = r2i.get(series[t-1])
        j = r2i.get(series[t])
        if i is not None and j is not None:
            counts[i, j] += 1

    return pd.DataFrame(counts, index=list(regimes), columns=list(regimes))


def regime_persistence(transition_matrix: np.ndarray,
                        regimes: Optional[Tuple[str, ...]] = None) -> pd.Series:
    """
    Compute the expected time to leave each regime (= 1 / (1 - p_ii)).

    Returns
    -------
    pd.Series indexed by regime name
    """
    if regimes is None:
        regimes = REGIMES
    P    = np.asarray(transition_matrix, dtype=float)
    diag = np.diag(P)[:len(regimes)]
    exit_rate   = 1.0 - diag
    exit_rate[exit_rate == 0] = 1e-6
    expected_duration = 1.0 / exit_rate
    return pd.Series(expected_duration, index=list(regimes[:len(diag)]),
                     name="expected_duration_bars")
