"""
alpha_decay.py — Alpha decay analysis for SRFM strategy signals.

Measures how quickly a trading signal's predictive power decays over time.
Computes IC decay curves, half-life estimation, and signal freshness metrics.
"""

from __future__ import annotations

import json
import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit, minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DecayResult:
    """Alpha/IC decay analysis result for a single signal."""
    signal_name: str
    horizons: List[int]
    ic_values: List[float]
    ic_std: List[float]
    cumulative_ic: List[float]
    half_life_days: float
    decay_model: str           # "exponential", "power_law", "linear", "none"
    decay_params: Dict[str, float]
    r_squared: float
    n_periods: int

    def to_dict(self) -> dict:
        return {
            "signal_name": self.signal_name,
            "horizons": self.horizons,
            "ic_values": [round(v, 6) for v in self.ic_values],
            "ic_std": [round(v, 6) for v in self.ic_std],
            "cumulative_ic": [round(v, 6) for v in self.cumulative_ic],
            "half_life_days": round(self.half_life_days, 2)
            if not np.isinf(self.half_life_days) else None,
            "decay_model": self.decay_model,
            "decay_params": {k: round(v, 6) for k, v in self.decay_params.items()},
            "r_squared": round(self.r_squared, 4),
            "n_periods": self.n_periods,
        }

    @property
    def is_decaying(self) -> bool:
        """True if IC is meaningfully positive and decaying."""
        if len(self.ic_values) < 2:
            return False
        return self.ic_values[0] > 0.02 and self.ic_values[-1] < self.ic_values[0] * 0.5

    @property
    def information_ratio(self) -> float:
        """IR = mean IC / std IC (using first-horizon IC)."""
        if not self.ic_std or self.ic_std[0] < 1e-10:
            return 0.0
        return self.ic_values[0] / self.ic_std[0]


@dataclass
class SignalComparisonResult:
    """Comparison of multiple signal decay profiles."""
    signals: List[str]
    half_lives: Dict[str, float]
    irs: Dict[str, float]
    peak_ics: Dict[str, float]
    best_signal: str
    most_persistent: str
    fastest_decay: str


# ---------------------------------------------------------------------------
# Core decay computation
# ---------------------------------------------------------------------------

def compute_alpha_decay(
    signal: pd.Series,
    returns: pd.Series,
    horizons: Optional[List[int]] = None,
    min_periods: int = 30,
    use_rank: bool = True,
) -> DecayResult:
    """
    Compute alpha decay: IC (Spearman or Pearson) between signal and
    forward returns at multiple horizons.

    Parameters
    ----------
    signal : pd.Series
        Signal values indexed by date (e.g., BH mass, tf_score).
    returns : pd.Series
        Daily return series indexed by date.
    horizons : list of int
        Forward return horizons in days. Default: [1, 2, 5, 10, 20, 40, 60].
    min_periods : int
        Minimum number of valid pairs required to compute IC.
    use_rank : bool
        If True, use Spearman rank correlation (robust to outliers).
        If False, use Pearson correlation.

    Returns
    -------
    DecayResult with IC values, half-life, and decay model.
    """
    if horizons is None:
        horizons = [1, 2, 5, 10, 20, 40, 60]

    # Ensure datetime index
    if not isinstance(signal.index, pd.DatetimeIndex):
        signal = signal.copy()
        signal.index = pd.to_datetime(signal.index)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)

    signal_name = signal.name or "signal"
    ic_values = []
    ic_stds = []
    rolling_ics_list = []

    for h in horizons:
        # Forward h-day cumulative return
        fwd_ret = returns.rolling(h).sum().shift(-h)

        # Align
        common = signal.index.intersection(fwd_ret.index)
        s = signal.loc[common].dropna()
        r = fwd_ret.loc[s.index].dropna()
        common2 = s.index.intersection(r.index)

        if len(common2) < min_periods:
            ic_values.append(0.0)
            ic_stds.append(0.0)
            rolling_ics_list.append([])
            continue

        s_v = s.loc[common2].values
        r_v = r.loc[common2].values

        # Rolling IC with window = max(30, h*3)
        window = max(min_periods, h * 3)
        rolling_ics = []
        for i in range(window, len(common2)):
            sw = s_v[i - window:i]
            rw = r_v[i - window:i]
            if np.std(sw) < 1e-10 or np.std(rw) < 1e-10:
                continue
            if use_rank:
                rho, _ = stats.spearmanr(sw, rw)
            else:
                rho, _ = stats.pearsonr(sw, rw)
            if np.isfinite(rho):
                rolling_ics.append(float(rho))

        rolling_ics_list.append(rolling_ics)

        if rolling_ics:
            ic_values.append(float(np.mean(rolling_ics)))
            ic_stds.append(float(np.std(rolling_ics, ddof=1)) if len(rolling_ics) > 1 else 0.0)
        else:
            ic_values.append(0.0)
            ic_stds.append(0.0)

    # Cumulative IC (cumulative sum normalized by sqrt(horizon))
    cumulative_ic = []
    running = 0.0
    for i, (ic, h) in enumerate(zip(ic_values, horizons)):
        running += ic / np.sqrt(h)
        cumulative_ic.append(running)

    n_periods = max(len(x) for x in rolling_ics_list) if rolling_ics_list else 0

    # Fit decay model
    half_life, model, params, r_sq = _fit_decay_model(horizons, ic_values)

    return DecayResult(
        signal_name=str(signal_name),
        horizons=list(horizons),
        ic_values=ic_values,
        ic_std=ic_stds,
        cumulative_ic=cumulative_ic,
        half_life_days=half_life,
        decay_model=model,
        decay_params=params,
        r_squared=r_sq,
        n_periods=n_periods,
    )


# ---------------------------------------------------------------------------
# Decay model fitting
# ---------------------------------------------------------------------------

def _exp_decay(x: np.ndarray, a: float, lam: float) -> np.ndarray:
    return a * np.exp(-lam * x)


def _power_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * np.power(x + 1e-9, -b)


def _linear_decay(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a - b * x


def _fit_decay_model(
    horizons: List[int],
    ic_values: List[float],
) -> Tuple[float, str, Dict[str, float], float]:
    """
    Fit exponential, power law, and linear decay models.
    Returns (half_life, best_model_name, params, r_squared).
    """
    h = np.array(horizons, dtype=float)
    ic = np.array(ic_values, dtype=float)

    # Only fit if we have a meaningful positive IC at short horizon
    if ic[0] <= 0.005 or len(h) < 3:
        return np.inf, "none", {}, 0.0

    best_r2 = -np.inf
    best_model = "none"
    best_half_life = np.inf
    best_params: Dict[str, float] = {}

    # --- Exponential decay ---
    try:
        p0 = [ic[0], 0.05]
        bounds = ([0, 1e-6], [1.0, 10.0])
        popt, _ = curve_fit(_exp_decay, h, ic, p0=p0, bounds=bounds, maxfev=1000)
        ic_hat = _exp_decay(h, *popt)
        r2 = _r_squared(ic, ic_hat)
        if r2 > best_r2:
            best_r2 = r2
            best_model = "exponential"
            best_half_life = float(np.log(2) / popt[1]) if popt[1] > 1e-10 else np.inf
            best_params = {"a": float(popt[0]), "lambda": float(popt[1])}
    except (RuntimeError, ValueError, OverflowError):
        pass

    # --- Power law decay ---
    try:
        p0 = [ic[0], 0.5]
        bounds = ([0, 1e-6], [1.0, 5.0])
        popt, _ = curve_fit(_power_decay, h, ic, p0=p0, bounds=bounds, maxfev=1000)
        ic_hat = _power_decay(h, *popt)
        r2 = _r_squared(ic, ic_hat)
        if r2 > best_r2:
            best_r2 = r2
            best_model = "power_law"
            # half-life: when a * h^(-b) = a/2 → h^b = 2 → h = 2^(1/b)
            best_half_life = float(2 ** (1 / popt[1])) if popt[1] > 1e-10 else np.inf
            best_params = {"a": float(popt[0]), "b": float(popt[1])}
    except (RuntimeError, ValueError, OverflowError):
        pass

    # --- Linear decay ---
    try:
        p0 = [ic[0], ic[0] / (h[-1] + 1)]
        popt, _ = curve_fit(_linear_decay, h, ic, p0=p0, maxfev=1000)
        ic_hat = _linear_decay(h, *popt)
        r2 = _r_squared(ic, ic_hat)
        if r2 > best_r2:
            best_r2 = r2
            best_model = "linear"
            # half-life: a - b*h = a/2 → h = a/(2b)
            best_half_life = float(popt[0] / (2 * popt[1])) if popt[1] > 1e-10 else np.inf
            best_params = {"a": float(popt[0]), "b": float(popt[1])}
    except (RuntimeError, ValueError, OverflowError):
        pass

    return best_half_life, best_model, best_params, float(best_r2)


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0


# ---------------------------------------------------------------------------
# Half-life estimation
# ---------------------------------------------------------------------------

def estimate_half_life(
    ic_series: pd.Series,
    method: str = "ar1",
) -> float:
    """
    Estimate IC series half-life using AR(1) mean reversion or direct fitting.

    Parameters
    ----------
    ic_series : pd.Series
        Time series of rolling IC values.
    method : str
        "ar1" — Ornstein-Uhlenbeck half-life via AR(1) regression.
        "direct" — direct exponential fit to IC decay curve.

    Returns
    -------
    Estimated half-life in days. Returns np.inf if no mean reversion detected.
    """
    ic = ic_series.dropna().values

    if len(ic) < 10:
        return np.inf

    if method == "ar1":
        # OU process: d(IC) = -kappa * IC * dt + sigma * dW
        # Discretized: IC[t] = phi * IC[t-1] + epsilon
        # kappa = -log(phi), half-life = log(2) / kappa
        ic_lag = ic[:-1]
        ic_curr = ic[1:]

        if np.std(ic_lag) < 1e-10:
            return np.inf

        # OLS regression IC[t] = phi * IC[t-1]
        try:
            X = ic_lag.reshape(-1, 1)
            y = ic_curr
            phi = float(np.linalg.lstsq(X, y, rcond=None)[0][0])
        except (np.linalg.LinAlgError, ValueError):
            return np.inf

        # Stability check
        if phi <= 0 or phi >= 1.0:
            return np.inf

        kappa = -np.log(phi)
        if kappa <= 1e-10:
            return np.inf

        return float(np.log(2) / kappa)

    elif method == "direct":
        # Fit exponential to IC(t) = IC(0) * exp(-kappa * t)
        t = np.arange(len(ic), dtype=float)
        if ic[0] <= 0:
            return np.inf
        try:
            ic_norm = np.clip(ic / ic[0], 1e-6, 10.0)
            valid = ic_norm > 0.01
            if valid.sum() < 3:
                return np.inf
            slope, _, r, _, _ = stats.linregress(t[valid], np.log(ic_norm[valid]))
            kappa = -slope
            if kappa <= 1e-10:
                return np.inf
            return float(np.log(2) / kappa)
        except (ValueError, ZeroDivisionError):
            return np.inf

    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Multi-signal comparison
# ---------------------------------------------------------------------------

def compare_signal_decay(
    signals: Dict[str, pd.Series],
    returns: pd.Series,
    horizons: Optional[List[int]] = None,
) -> SignalComparisonResult:
    """
    Compare alpha decay profiles across multiple signals.

    Parameters
    ----------
    signals : dict of {name: pd.Series}
        Dictionary of signal series to compare.
    returns : pd.Series
        Return series for computing forward returns.
    horizons : list of int
        Forward horizons in days.

    Returns
    -------
    SignalComparisonResult with cross-signal comparisons.
    """
    if horizons is None:
        horizons = [1, 2, 5, 10, 20, 40]

    decay_results = {}
    for name, sig in signals.items():
        sig.name = name
        decay_results[name] = compute_alpha_decay(sig, returns, horizons)

    half_lives = {n: r.half_life_days for n, r in decay_results.items()}
    irs = {n: r.information_ratio for n, r in decay_results.items()}
    peak_ics = {n: r.ic_values[0] if r.ic_values else 0.0 for n, r in decay_results.items()}

    # Best signal: highest IR
    best = max(irs, key=lambda k: irs[k]) if irs else list(signals.keys())[0]

    # Most persistent: longest half-life (finite)
    finite_hls = {k: v for k, v in half_lives.items() if np.isfinite(v)}
    most_persistent = (
        max(finite_hls, key=lambda k: finite_hls[k])
        if finite_hls
        else list(signals.keys())[0]
    )

    # Fastest decay: shortest half-life (finite)
    fastest = (
        min(finite_hls, key=lambda k: finite_hls[k])
        if finite_hls
        else list(signals.keys())[0]
    )

    return SignalComparisonResult(
        signals=list(signals.keys()),
        half_lives=half_lives,
        irs=irs,
        peak_ics=peak_ics,
        best_signal=best,
        most_persistent=most_persistent,
        fastest_decay=fastest,
    )


# ---------------------------------------------------------------------------
# Signal freshness
# ---------------------------------------------------------------------------

def signal_freshness_score(
    signal: pd.Series,
    decay_result: DecayResult,
    current_date: Optional[pd.Timestamp] = None,
    max_staleness_days: int = 5,
) -> pd.Series:
    """
    Compute a freshness-adjusted signal score.
    Applies exponential decay to signal value based on how old it is.

    If decay half-life is known, score = signal_value * exp(-age / half_life).
    Returns a Series of freshness-adjusted scores.
    """
    if current_date is None:
        current_date = pd.Timestamp.now()

    hl = decay_result.half_life_days
    if np.isinf(hl) or hl <= 0:
        return signal.copy()

    # Compute age in days at each point
    ages = [(current_date - t).days if hasattr(t, "days") else 0
            for t in (current_date - signal.index)]
    ages = np.clip(ages, 0, max_staleness_days * 3)
    decay_factors = np.exp(-np.log(2) * ages / hl)

    return signal * decay_factors


def optimal_rebalance_frequency(
    decay_result: DecayResult,
    transaction_cost_bps: float = 2.0,
    target_ic_fraction: float = 0.5,
) -> float:
    """
    Estimate optimal rebalancing frequency given signal decay and transaction costs.

    Rebalance when the benefit of fresh signal (increased IC) equals the
    transaction cost. Simplified model: rebalance at half-life adjusted by costs.

    Parameters
    ----------
    decay_result : DecayResult
        Computed decay result for the signal.
    transaction_cost_bps : float
        Round-trip transaction cost in basis points.
    target_ic_fraction : float
        Rebalance when IC falls to this fraction of peak (default 0.5 = half-life).

    Returns
    -------
    Optimal rebalancing frequency in days.
    """
    hl = decay_result.half_life_days
    if np.isinf(hl):
        return 252.0  # Annual if no decay

    # Adjust half-life for transaction costs
    # If costs are high, it's worth holding longer (wait for IC to decay more)
    peak_ic = decay_result.ic_values[0] if decay_result.ic_values else 0.0
    if peak_ic < 1e-6:
        return 252.0

    # Cost threshold: IC * some_vol = transaction_cost
    # When adjusted IC drops below cost threshold, don't rebalance
    cost_fraction = transaction_cost_bps / 10_000 / (peak_ic + 1e-9)
    cost_fraction = min(cost_fraction, 0.95)

    # Rebalance horizon: when IC drops to max(target_ic_fraction, cost_fraction)
    threshold = max(target_ic_fraction, cost_fraction)

    # Exponential: fraction = exp(-lambda * h) → h = -log(fraction) / lambda
    if decay_result.decay_model == "exponential" and "lambda" in decay_result.decay_params:
        lam = decay_result.decay_params["lambda"]
        if lam > 0:
            return float(-np.log(threshold) / lam)

    # Default: use half-life
    return float(hl * (-np.log(threshold) / np.log(2)))


# ---------------------------------------------------------------------------
# IC ICIR surface
# ---------------------------------------------------------------------------

def compute_ic_surface(
    signal: pd.Series,
    returns: pd.Series,
    horizons: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute IC across multiple evaluation windows and forward horizons.
    Returns a DataFrame: rows = windows, columns = horizons, values = mean IC.
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]
    if windows is None:
        windows = [20, 40, 60, 120]

    if not isinstance(signal.index, pd.DatetimeIndex):
        signal = signal.copy()
        signal.index = pd.to_datetime(signal.index)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)

    surface = pd.DataFrame(index=windows, columns=horizons, dtype=float)

    for h in horizons:
        fwd_ret = returns.rolling(h).sum().shift(-h)
        common = signal.index.intersection(fwd_ret.index)
        s = signal.loc[common].dropna()
        r = fwd_ret.loc[s.index].dropna()
        common2 = s.index.intersection(r.index)

        if len(common2) < 20:
            for w in windows:
                surface.loc[w, h] = 0.0
            continue

        s_v = s.loc[common2].values
        r_v = r.loc[common2].values

        for w in windows:
            if w > len(common2):
                surface.loc[w, h] = 0.0
                continue

            rolling_ics = []
            for i in range(w, len(common2)):
                sw = s_v[i - w:i]
                rw = r_v[i - w:i]
                if np.std(sw) < 1e-10 or np.std(rw) < 1e-10:
                    continue
                rho, _ = stats.spearmanr(sw, rw)
                if np.isfinite(rho):
                    rolling_ics.append(rho)

            surface.loc[w, h] = float(np.mean(rolling_ics)) if rolling_ics else 0.0

    return surface


# ---------------------------------------------------------------------------
# Plot alpha decay
# ---------------------------------------------------------------------------

def plot_alpha_decay(
    decay_results: List[DecayResult],
    output_path: Optional[str] = None,
    show: bool = False,
    title: str = "SRFM Signal Alpha Decay Analysis",
) -> None:
    """
    Plot alpha decay curves for one or multiple signals.

    Parameters
    ----------
    decay_results : list of DecayResult
        One or more decay results to plot.
    output_path : str, optional
        Save plot to this path.
    show : bool
        Display interactively if True.
    title : str
        Plot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    n_signals = len(decay_results)
    fig = plt.figure(figsize=(18, 5 * max(2, (n_signals + 1) // 2 + 1)))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    n_rows = max(2, (n_signals + 1) // 2 + 1)
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = ["steelblue", "darkorange", "green", "purple", "red",
              "brown", "pink", "gray", "olive", "cyan"]

    # ---- Top row: all signals overlaid ----
    ax_all = fig.add_subplot(gs[0, :])
    for i, dr in enumerate(decay_results):
        h = np.array(dr.horizons, dtype=float)
        ic = np.array(dr.ic_values)
        ic_std = np.array(dr.ic_std)
        color = colors[i % len(colors)]

        ax_all.plot(h, ic, "o-", color=color, linewidth=2,
                    label=f"{dr.signal_name} (HL={dr.half_life_days:.1f}d)"
                    if np.isfinite(dr.half_life_days)
                    else f"{dr.signal_name} (no decay)")
        ax_all.fill_between(h, ic - ic_std, ic + ic_std, alpha=0.15, color=color)

        # Plot fitted decay model
        if dr.decay_model != "none" and dr.decay_params:
            h_fine = np.linspace(h[0], h[-1], 100)
            if dr.decay_model == "exponential":
                ic_fit = _exp_decay(h_fine, dr.decay_params["a"], dr.decay_params["lambda"])
            elif dr.decay_model == "power_law":
                ic_fit = _power_decay(h_fine, dr.decay_params["a"], dr.decay_params["b"])
            elif dr.decay_model == "linear":
                ic_fit = _linear_decay(h_fine, dr.decay_params["a"], dr.decay_params["b"])
            else:
                ic_fit = None

            if ic_fit is not None:
                ax_all.plot(h_fine, ic_fit, "--", color=color, alpha=0.5, linewidth=1.5)

    ax_all.axhline(y=0, color="k", linewidth=0.8, linestyle="--")
    ax_all.set_title("IC Decay Comparison (solid=data, dashed=fitted model)", fontweight="bold")
    ax_all.set_xlabel("Forward Horizon (days)")
    ax_all.set_ylabel("Mean IC (Spearman)")
    ax_all.legend(fontsize=9)

    # ---- Individual signal panels ----
    for i, dr in enumerate(decay_results):
        row = i // 2 + 1
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        color = colors[i % len(colors)]

        h = np.array(dr.horizons, dtype=float)
        ic = np.array(dr.ic_values)
        ic_std = np.array(dr.ic_std)

        ax.bar(h, ic, width=h * 0.3, color=color, alpha=0.6)
        ax.errorbar(h, ic, yerr=ic_std, fmt="none", color="k", capsize=4)
        ax.axhline(y=0, color="k", linewidth=0.8)

        # Half-life vertical line
        if np.isfinite(dr.half_life_days) and h[0] <= dr.half_life_days <= h[-1] * 2:
            ax.axvline(x=dr.half_life_days, color="red", linestyle="--",
                       label=f"Half-life: {dr.half_life_days:.1f}d")
            ax.legend(fontsize=8)

        info_lines = [
            f"Model: {dr.decay_model}",
            f"R²: {dr.r_squared:.3f}",
            f"IR: {dr.information_ratio:.3f}",
            f"n={dr.n_periods}",
        ]
        ax.text(0.97, 0.97, "\n".join(info_lines),
                transform=ax.transAxes, fontsize=7,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_title(f"Signal: {dr.signal_name}", fontweight="bold")
        ax.set_xlabel("Forward Horizon (days)")
        ax.set_ylabel("Mean IC")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Alpha decay plot saved to {output_path}")
    elif show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generate synthetic signals and returns for testing
# ---------------------------------------------------------------------------

def generate_test_signals(
    n_days: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """
    Generate synthetic signals and returns for testing decay analysis.

    Signal types:
    - fast_signal: decays in ~5 days (high-frequency alpha)
    - medium_signal: decays in ~20 days (swing alpha)
    - slow_signal: decays in ~60 days (trend-following alpha)
    - noisy_signal: weak IC, fast noise
    - bh_mass: BH-style signal correlated with medium_signal

    Returns (signals_dict, returns_series).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="1D")

    # Generate returns with some structure
    base_returns = rng.normal(0.0003, 0.012, n_days)

    # Fast signal: AR(1) with phi=0.8 (half-life ~3 days), correlated with 1-day forward
    fast_raw = np.zeros(n_days)
    phi_fast = 0.80
    for t in range(1, n_days):
        fast_raw[t] = phi_fast * fast_raw[t - 1] + rng.normal(0, 1)
    # Correlate with future 1d return
    fast_signal = 0.08 * np.roll(base_returns, -1) + 0.92 * fast_raw / fast_raw.std()
    fast_signal[-1] = 0.0

    # Medium signal: AR(1) with phi=0.97 (half-life ~23 days)
    medium_raw = np.zeros(n_days)
    phi_med = 0.97
    for t in range(1, n_days):
        medium_raw[t] = phi_med * medium_raw[t - 1] + rng.normal(0, 1)
    fwd_5d = np.convolve(base_returns, np.ones(5), mode="same") / 5
    medium_signal = 0.10 * np.roll(fwd_5d, -5) + 0.90 * medium_raw / medium_raw.std()
    medium_signal[-5:] = 0.0

    # Slow signal: AR(1) with phi=0.989 (half-life ~63 days)
    slow_raw = np.zeros(n_days)
    phi_slow = 0.989
    for t in range(1, n_days):
        slow_raw[t] = phi_slow * slow_raw[t - 1] + rng.normal(0, 1)
    fwd_20d = np.convolve(base_returns, np.ones(20), mode="same") / 20
    slow_signal = 0.12 * np.roll(fwd_20d, -20) + 0.88 * slow_raw / slow_raw.std()
    slow_signal[-20:] = 0.0

    # Noisy signal: pure noise with tiny IC
    noisy_signal = rng.normal(0, 1, n_days) + 0.02 * np.roll(base_returns, -1)
    noisy_signal[-1] = 0.0

    # BH mass proxy: positive with occasional resets
    bh_mass = np.zeros(n_days)
    for t in range(1, n_days):
        if rng.random() < 0.03:  # BH collapses 3% of time
            bh_mass[t] = 0.0
        else:
            increment = 0.1 if base_returns[t - 1] > 0 else -0.05
            bh_mass[t] = max(0, bh_mass[t - 1] + increment + rng.normal(0, 0.05))
    # BH mass > 1.5 is bullish signal
    bh_signal = bh_mass - 1.5

    signals = {
        "fast_signal": pd.Series(fast_signal, index=dates, name="fast_signal"),
        "medium_signal": pd.Series(medium_signal, index=dates, name="medium_signal"),
        "slow_signal": pd.Series(slow_signal, index=dates, name="slow_signal"),
        "noisy_signal": pd.Series(noisy_signal, index=dates, name="noisy_signal"),
        "bh_mass_signal": pd.Series(bh_signal, index=dates, name="bh_mass_signal"),
    }

    returns = pd.Series(base_returns, index=dates, name="returns")

    return signals, returns


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_decay_results(
    results: List[DecayResult],
    output_path: str,
) -> dict:
    """Export decay results to JSON."""
    data = {
        "n_signals": len(results),
        "signals": [r.to_dict() for r in results],
        "summary": {
            r.signal_name: {
                "half_life_days": r.half_life_days if np.isfinite(r.half_life_days) else None,
                "peak_ic": r.ic_values[0] if r.ic_values else 0.0,
                "ir": r.information_ratio,
                "decay_model": r.decay_model,
                "r_squared": r.r_squared,
            }
            for r in results
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Decay results saved to {output_path}")
    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SRFM Alpha Decay Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alpha_decay.py --output alpha_decay.png
  python alpha_decay.py --n-days 2000 --horizons 1 2 5 10 20 40 60 --output decay.png
  python alpha_decay.py --export decay_results.json
""",
    )
    parser.add_argument("--n-days", type=int, default=1000,
                        help="Number of synthetic data days")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[1, 2, 5, 10, 20, 40, 60],
                        help="Forward horizons in days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="alpha_decay.png",
                        help="Output plot path")
    parser.add_argument("--export", type=str, help="Export JSON results to path")
    parser.add_argument("--signal", type=str,
                        help="Single signal to analyze (default: all)")
    args = parser.parse_args()

    print(f"Generating {args.n_days} days of synthetic signals and returns...")
    signals, returns = generate_test_signals(args.n_days, args.seed)

    if args.signal:
        if args.signal not in signals:
            print(f"Signal '{args.signal}' not found. Available: {list(signals.keys())}")
            return
        signals = {args.signal: signals[args.signal]}

    print(f"\nComputing alpha decay for {len(signals)} signal(s)...")
    print(f"Horizons: {args.horizons}")

    all_results = []
    for name, sig in signals.items():
        print(f"\n  Analyzing: {name}...")
        dr = compute_alpha_decay(sig, returns, horizons=args.horizons)
        all_results.append(dr)

        hl_str = f"{dr.half_life_days:.1f} days" if np.isfinite(dr.half_life_days) else "∞ (persistent)"
        print(f"    Half-life:    {hl_str}")
        print(f"    Decay model:  {dr.decay_model}  (R²={dr.r_squared:.3f})")
        print(f"    Peak IC:      {dr.ic_values[0]:.4f}" if dr.ic_values else "    Peak IC: N/A")
        print(f"    Info ratio:   {dr.information_ratio:.3f}")
        print(f"    IC by horizon: " +
              "  ".join(f"h{h}={ic:.3f}" for h, ic in zip(dr.horizons[:6], dr.ic_values[:6])))

    # Cross-signal comparison
    if len(all_results) > 1:
        comp = compare_signal_decay(signals, returns, horizons=args.horizons)
        print(f"\n--- Signal Comparison ---")
        print(f"  Best signal (highest IR):    {comp.best_signal}")
        print(f"  Most persistent:             {comp.most_persistent}")
        print(f"  Fastest decay:               {comp.fastest_decay}")

        print(f"\n  Signal           | Half-life  | Peak IC | IR")
        print(f"  " + "-" * 50)
        for name in comp.signals:
            hl = comp.half_lives[name]
            hl_str = f"{hl:.1f}d" if np.isfinite(hl) else "  ∞  "
            print(f"  {name:<18} | {hl_str:<10} | "
                  f"{comp.peak_ics[name]:6.4f}  | {comp.irs[name]:.3f}")

    # Rebalance frequency
    print(f"\n--- Optimal Rebalance Frequencies ---")
    for dr in all_results:
        for tc_bps in [1.0, 2.0, 5.0]:
            freq = optimal_rebalance_frequency(dr, transaction_cost_bps=tc_bps)
            print(f"  {dr.signal_name:<20}  tc={tc_bps}bps → rebalance every {freq:.1f} days")

    print(f"\nGenerating alpha decay plot → {args.output}")
    plot_alpha_decay(all_results, output_path=args.output, title="SRFM Alpha Decay Analysis")

    if args.export:
        export_decay_results(all_results, args.export)


if __name__ == "__main__":
    main()
