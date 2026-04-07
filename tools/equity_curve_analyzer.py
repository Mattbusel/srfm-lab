"""
equity_curve_analyzer.py -- In-depth equity curve analysis for LARSA v18.

Provides:
- HP-filter trend/cycle decomposition
- Baxter-King bandpass filter for cyclical component
- CUSUM-based regime change detection
- Recovery metrics (drawdown periods, avg recovery time, recovery factor)
- Bootstrap Sharpe CI
- Markov-chain drawdown probability prediction
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class EquityDecomposition:
    """
    Result of decomposing an equity curve into structural components.
    All lists are the same length as the input equity curve.
    """
    original: List[float]
    trend: List[float]       # HP filter long-run trend
    cyclical: List[float]    # Baxter-King bandpass (medium frequency)
    noise: List[float]       # residual = original - trend - cyclical
    log_curve: List[float]   # log(original)

    @property
    def n(self) -> int:
        return len(self.original)

    def trend_slope(self) -> float:
        """Average slope of trend component per bar."""
        if len(self.trend) < 2:
            return 0.0
        return (self.trend[-1] - self.trend[0]) / (len(self.trend) - 1)

    def cyclical_amplitude(self) -> float:
        """Peak-to-trough amplitude of cyclical component."""
        if not self.cyclical:
            return 0.0
        return max(self.cyclical) - min(self.cyclical)

    def noise_std(self) -> float:
        if not self.noise:
            return 0.0
        mean = sum(self.noise) / len(self.noise)
        var = sum((x - mean) ** 2 for x in self.noise) / len(self.noise)
        return math.sqrt(var)

    def signal_to_noise(self) -> float:
        """Trend slope / noise_std ratio."""
        ns = self.noise_std()
        return abs(self.trend_slope()) / ns if ns > 0 else 0.0


@dataclass
class RegimeBreakpoint:
    """A detected structural break in the equity curve."""
    index: int               # bar index of breakpoint
    cusum_value: float       # CUSUM statistic at detection
    direction: str           # "up_to_down" or "down_to_up"
    pre_mean: float          # mean return before breakpoint
    post_mean: float         # mean return after breakpoint (estimated)
    confidence: float        # 0-1 confidence based on CUSUM magnitude


@dataclass
class RecoveryMetrics:
    """Comprehensive recovery and drawdown metrics."""
    num_drawdown_periods: int
    avg_drawdown_depth: float       # fraction
    max_drawdown_depth: float       # fraction
    avg_recovery_bars: Optional[float]
    max_recovery_bars: Optional[int]
    recovery_factor: float          # total_return / max_dd
    calmar_ratio: float
    underwater_fraction: float      # fraction of time spent in drawdown
    pain_index: float               # avg drawdown across all bars
    ulcer_index: float              # RMS of drawdown series


# ---------------------------------------------------------------------------
# HP filter (Hodrick-Prescott)
# ---------------------------------------------------------------------------


def _hp_filter(series: List[float], lam: float = 1600.0) -> Tuple[List[float], List[float]]:
    """
    Hodrick-Prescott filter.
    Returns (trend, cycle) where cycle = series - trend.
    lam=1600 is standard for quarterly data; use lam=6.25 for annual,
    lam=129600 for monthly, lam=1e7 for daily.
    """
    n = len(series)
    if n < 4:
        return list(series), [0.0] * n

    # Build the penalty matrix T (second difference) and solve
    # (I + lam * T'T) * trend = series
    # Using a tridiagonal solver for efficiency
    y = series[:]

    # Build T'T as a pentadiagonal matrix
    # diagonal[0], diagonal[1], diagonal[2] = sub2, sub1, main, super1, super2
    # For efficiency, use direct computation via numpy if available, else fall back
    try:
        import numpy as np
        yn = np.array(y, dtype=float)
        I = np.eye(n)
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i] = 1
            D[i, i + 1] = -2
            D[i, i + 2] = 1
        trend_arr = np.linalg.solve(I + lam * D.T @ D, yn)
        trend = trend_arr.tolist()
    except ImportError:
        # Pure Python fallback: simple moving average approximation
        window = max(3, int(math.sqrt(lam)))
        trend = _moving_average(y, window)

    cycle = [y[i] - trend[i] for i in range(n)]
    return trend, cycle


def _moving_average(series: List[float], window: int) -> List[float]:
    """Simple centred moving average with edge padding."""
    n = len(series)
    half = window // 2
    result = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(series[lo:hi]) / (hi - lo))
    return result


# ---------------------------------------------------------------------------
# Baxter-King bandpass filter (approximate)
# ---------------------------------------------------------------------------


def _baxter_king_bandpass(
    series: List[float],
    low_period: int = 6,
    high_period: int = 32,
    K: int = 12,
) -> List[float]:
    """
    Approximate Baxter-King bandpass filter.
    Extracts cyclical components between low_period and high_period bars.
    K is the number of leads/lags.

    Returns filtered series (same length, edges padded with 0).
    """
    n = len(series)
    if n < 2 * K + 1:
        return [0.0] * n

    # Compute ideal bandpass filter weights
    omega_l = 2 * math.pi / high_period
    omega_h = 2 * math.pi / low_period

    weights = []
    for k in range(-K, K + 1):
        if k == 0:
            w = (omega_h - omega_l) / math.pi
        else:
            w = (math.sin(omega_h * k) - math.sin(omega_l * k)) / (math.pi * k)
        weights.append(w)

    # Remove sum (enforce zero-frequency = 0)
    total_w = sum(weights)
    adj = total_w / len(weights)
    weights = [w - adj for w in weights]

    # Apply filter
    filtered = [0.0] * n
    for i in range(K, n - K):
        val = 0.0
        for j, w in enumerate(weights):
            val += w * series[i + j - K]
        filtered[i] = val

    return filtered


# ---------------------------------------------------------------------------
# CUSUM regime change detection
# ---------------------------------------------------------------------------


def _cusum_detect(
    returns: List[float],
    threshold_multiplier: float = 4.0,
    min_segment: int = 10,
) -> List[RegimeBreakpoint]:
    """
    CUSUM (cumulative sum) test for structural breaks in a return series.

    Parameters
    ----------
    returns : List[float]
        Per-bar or per-trade returns.
    threshold_multiplier : float
        CUSUM threshold = threshold_multiplier * std(returns).
    min_segment : int
        Minimum number of bars between breakpoints.

    Returns
    -------
    List[RegimeBreakpoint]
    """
    n = len(returns)
    if n < min_segment * 2:
        return []

    mean_r = sum(returns) / n
    var_r = sum((r - mean_r) ** 2 for r in returns) / max(n - 1, 1)
    std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
    threshold = threshold_multiplier * std_r

    breakpoints: List[RegimeBreakpoint] = []
    cusum_pos = 0.0
    cusum_neg = 0.0
    last_break = 0

    running_sum = 0.0
    running_count = 0
    running_mean = mean_r

    for i, r in enumerate(returns):
        cusum_pos = max(0.0, cusum_pos + r - mean_r - std_r * 0.5)
        cusum_neg = min(0.0, cusum_neg + r - mean_r + std_r * 0.5)

        if i - last_break < min_segment:
            running_sum += r
            running_count += 1
            continue

        if cusum_pos > threshold or abs(cusum_neg) > threshold:
            # Detected breakpoint
            direction = "down_to_up" if cusum_pos > threshold else "up_to_down"
            pre_mean = running_mean
            # Estimate post-mean from remaining
            if i + 1 < n:
                remaining = returns[i + 1: min(i + 1 + min_segment, n)]
                post_mean = sum(remaining) / len(remaining) if remaining else mean_r
            else:
                post_mean = mean_r

            cusum_val = cusum_pos if cusum_pos > threshold else abs(cusum_neg)
            confidence = min(1.0, cusum_val / (threshold * 2))

            bp = RegimeBreakpoint(
                index=i,
                cusum_value=cusum_val,
                direction=direction,
                pre_mean=pre_mean,
                post_mean=post_mean,
                confidence=confidence,
            )
            breakpoints.append(bp)
            # Reset
            cusum_pos = 0.0
            cusum_neg = 0.0
            last_break = i
            running_mean = sum(returns[i:]) / max(len(returns[i:]), 1)

        running_sum += r
        running_count += 1

    return breakpoints


# ---------------------------------------------------------------------------
# Bootstrap Sharpe
# ---------------------------------------------------------------------------


def _bootstrap_sharpe(
    returns: List[float],
    n_boot: int = 1000,
    seed: int = 42,
    rf: float = 0.0,
) -> Tuple[float, float]:
    """
    Bootstrap Sharpe ratio with 95% confidence interval.

    Returns
    -------
    (sharpe_point_estimate, half_width_95_ci)
    """
    if len(returns) < 4:
        return 0.0, 0.0

    rng = random.Random(seed)
    n = len(returns)

    def sharpe_of(r: List[float]) -> float:
        if len(r) < 2:
            return 0.0
        excess = [x - rf for x in r]
        mean_e = sum(excess) / len(excess)
        var_e = sum((x - mean_e) ** 2 for x in excess) / (len(excess) - 1)
        std_e = math.sqrt(var_e) if var_e > 0 else 0.0
        return mean_e / std_e if std_e > 0 else 0.0

    point_estimate = sharpe_of(returns)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = [rng.choice(returns) for _ in range(n)]
        boot_sharpes.append(sharpe_of(sample))

    boot_sharpes.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot)
    ci_lo = boot_sharpes[lo_idx]
    ci_hi = boot_sharpes[hi_idx]
    half_width = (ci_hi - ci_lo) / 2.0

    return point_estimate, half_width


# ---------------------------------------------------------------------------
# Markov chain drawdown prediction
# ---------------------------------------------------------------------------


def _build_markov_matrix(
    states: List[int],
) -> Dict[int, Dict[int, float]]:
    """
    Build a first-order Markov transition matrix from a sequence of states.
    States: 0 = above_watermark (not in DD), 1 = in drawdown.
    """
    counts: Dict[int, Dict[int, int]] = {0: {0: 0, 1: 0}, 1: {0: 0, 1: 0}}
    for i in range(len(states) - 1):
        s0 = states[i]
        s1 = states[i + 1]
        counts[s0][s1] = counts[s0].get(s1, 0) + 1

    matrix: Dict[int, Dict[int, float]] = {}
    for s, trans in counts.items():
        total = sum(trans.values())
        matrix[s] = {
            t: (c / total if total > 0 else 0.5)
            for t, c in trans.items()
        }
    return matrix


def _predict_drawdown_probability(
    equity_curve: List[float],
    horizon: int = 1,
) -> float:
    """
    Use a first-order Markov chain to estimate the probability that the
    equity curve enters a drawdown within the next `horizon` bars.

    Returns a probability in [0, 1].
    """
    if len(equity_curve) < 10:
        return 0.5

    # Build state sequence
    peak = equity_curve[0]
    states: List[int] = []
    for val in equity_curve:
        if val > peak:
            peak = val
        state = 0 if val >= peak else 1
        states.append(state)

    matrix = _build_markov_matrix(states)
    current_state = states[-1]

    # Forward propagate for `horizon` steps
    probs = {0: 0.0, 1: 0.0}
    probs[current_state] = 1.0

    for _ in range(horizon):
        new_probs = {0: 0.0, 1: 0.0}
        for s in (0, 1):
            for s_next, p_trans in matrix.get(s, {}).items():
                new_probs[s_next] = new_probs.get(s_next, 0.0) + probs[s] * p_trans
        probs = new_probs

    return probs.get(1, 0.0)


# ---------------------------------------------------------------------------
# EquityCurveAnalyzer
# ---------------------------------------------------------------------------


class EquityCurveAnalyzer:
    """
    Comprehensive equity curve analysis engine for LARSA v18.

    All methods accept a list of NAV values (equity curve).
    """

    def __init__(
        self,
        hp_lambda: float = 1600.0,
        bk_low_period: int = 6,
        bk_high_period: int = 32,
        bk_K: int = 12,
        cusum_threshold: float = 4.0,
        cusum_min_segment: int = 10,
    ):
        self.hp_lambda = hp_lambda
        self.bk_low_period = bk_low_period
        self.bk_high_period = bk_high_period
        self.bk_K = bk_K
        self.cusum_threshold = cusum_threshold
        self.cusum_min_segment = cusum_min_segment

    # -- Decomposition --

    def decompose(self, equity_curve: List[float]) -> EquityDecomposition:
        """
        Decompose equity curve into trend, cyclical, and noise components.

        1. Compute log curve
        2. HP filter -> trend + cycle
        3. Apply BK bandpass to cycle -> cyclical
        4. noise = cycle - cyclical
        """
        n = len(equity_curve)
        if n < 4:
            zero = [0.0] * n
            return EquityDecomposition(
                original=list(equity_curve),
                trend=list(equity_curve),
                cyclical=zero,
                noise=zero,
                log_curve=zero,
            )

        # Log transform (protect against <= 0)
        log_curve = [math.log(max(v, 1e-9)) for v in equity_curve]

        # HP filter on log curve
        trend_log, cycle_log = _hp_filter(log_curve, lam=self.hp_lambda)

        # Back-transform trend to levels
        trend_levels = [math.exp(t) for t in trend_log]

        # BK bandpass on the cycle component
        cyclical = _baxter_king_bandpass(
            cycle_log,
            low_period=self.bk_low_period,
            high_period=self.bk_high_period,
            K=self.bk_K,
        )

        # Noise = cycle - BK_cycle
        noise = [cycle_log[i] - cyclical[i] for i in range(n)]

        # Convert cyclical and noise back to level scale (approx: trend * exp(component))
        cyclical_levels = [trend_levels[i] * (math.exp(cyclical[i]) - 1) for i in range(n)]
        noise_levels = [trend_levels[i] * (math.exp(noise[i]) - 1) for i in range(n)]

        return EquityDecomposition(
            original=list(equity_curve),
            trend=trend_levels,
            cyclical=cyclical_levels,
            noise=noise_levels,
            log_curve=log_curve,
        )

    # -- Regime changes --

    def detect_regime_changes(
        self, equity_curve: List[float]
    ) -> List[RegimeBreakpoint]:
        """
        Detect structural regime changes using CUSUM test on equity returns.
        """
        if len(equity_curve) < 2:
            return []
        returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            if equity_curve[i - 1] != 0 else 0.0
            for i in range(1, len(equity_curve))
        ]
        return _cusum_detect(
            returns,
            threshold_multiplier=self.cusum_threshold,
            min_segment=self.cusum_min_segment,
        )

    # -- Recovery metrics --

    def compute_recovery_metrics(
        self, equity_curve: List[float]
    ) -> RecoveryMetrics:
        """
        Compute comprehensive recovery and drawdown metrics.
        """
        from performance_report import _find_all_drawdowns

        n = len(equity_curve)
        if n < 2:
            return RecoveryMetrics(
                num_drawdown_periods=0,
                avg_drawdown_depth=0.0,
                max_drawdown_depth=0.0,
                avg_recovery_bars=None,
                max_recovery_bars=None,
                recovery_factor=0.0,
                calmar_ratio=0.0,
                underwater_fraction=0.0,
                pain_index=0.0,
                ulcer_index=0.0,
            )

        periods = _find_all_drawdowns(equity_curve)
        total_return = (
            (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            if equity_curve[0] > 0 else 0.0
        )

        # Max drawdown
        max_dd = max((p.depth for p in periods), default=0.0)
        avg_dd = sum(p.depth for p in periods) / len(periods) if periods else 0.0

        # Recovery times
        recovered = [p for p in periods if p.recovery_bars is not None]
        avg_rec = (
            sum(p.recovery_bars for p in recovered) / len(recovered)
            if recovered else None
        )
        max_rec = max(
            (p.recovery_bars for p in recovered), default=None
        )

        # Recovery factor
        recovery_factor = total_return / max_dd if max_dd > 0 else float("inf")

        # Calmar (annualised return / max_dd) -- using bars as proxy for time
        # Assume 252 trading days per year, 6.5 hours per day, 1 bar = 1h
        bars_per_year = 252 * 6
        years = n / bars_per_year
        cagr = (1.0 + total_return) ** (1.0 / max(years, 1 / 12)) - 1.0
        calmar = cagr / max_dd if max_dd > 0 else 0.0

        # Underwater fraction: bars in drawdown / total bars
        peak_running = equity_curve[0]
        underwater_bars = 0
        dd_series = []
        for val in equity_curve:
            if val > peak_running:
                peak_running = val
            dd = (peak_running - val) / peak_running if peak_running > 0 else 0.0
            dd_series.append(dd)
            if dd > 0:
                underwater_bars += 1

        underwater_fraction = underwater_bars / n

        # Pain index = avg dd across all bars
        pain_index = sum(dd_series) / n

        # Ulcer index = sqrt(mean(dd^2))
        ulcer_index = math.sqrt(sum(d ** 2 for d in dd_series) / n)

        return RecoveryMetrics(
            num_drawdown_periods=len(periods),
            avg_drawdown_depth=avg_dd,
            max_drawdown_depth=max_dd,
            avg_recovery_bars=avg_rec,
            max_recovery_bars=max_rec,
            recovery_factor=recovery_factor,
            calmar_ratio=calmar,
            underwater_fraction=underwater_fraction,
            pain_index=pain_index,
            ulcer_index=ulcer_index,
        )

    # -- Bootstrap Sharpe --

    def bootstrap_sharpe(
        self,
        equity_curve: List[float],
        n_boot: int = 1000,
        rf: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Bootstrap Sharpe with 95% CI.

        Parameters
        ----------
        equity_curve : List[float]
            NAV values.
        n_boot : int
            Number of bootstrap replications.
        rf : float
            Risk-free rate per bar.

        Returns
        -------
        (point_estimate, half_width_95_ci)
        """
        if len(equity_curve) < 4:
            return 0.0, 0.0
        returns = [
            (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
            if equity_curve[i - 1] != 0 else 0.0
            for i in range(1, len(equity_curve))
        ]
        return _bootstrap_sharpe(returns, n_boot=n_boot, rf=rf)

    # -- Drawdown prediction --

    def predict_next_drawdown(
        self, equity_curve: List[float], horizon: int = 1
    ) -> float:
        """
        Markov-chain probability of entering a drawdown in the next `horizon` bars.

        Returns float in [0, 1].
        """
        return _predict_drawdown_probability(equity_curve, horizon=horizon)

    # -- Convenience: full analysis --

    def full_analysis(
        self,
        equity_curve: List[float],
        n_boot: int = 500,
    ) -> Dict:
        """
        Run all analyses and return a combined result dict.
        """
        decomp = self.decompose(equity_curve)
        breakpoints = self.detect_regime_changes(equity_curve)
        recovery = self.compute_recovery_metrics(equity_curve)
        sharpe_pt, sharpe_ci = self.bootstrap_sharpe(equity_curve, n_boot=n_boot)
        dd_prob = self.predict_next_drawdown(equity_curve)

        return {
            "decomposition": {
                "trend_slope": decomp.trend_slope(),
                "cyclical_amplitude": decomp.cyclical_amplitude(),
                "noise_std": decomp.noise_std(),
                "signal_to_noise": decomp.signal_to_noise(),
                "trend": decomp.trend,
                "cyclical": decomp.cyclical,
                "noise": decomp.noise,
            },
            "regime_breakpoints": [
                {
                    "index": bp.index,
                    "cusum_value": bp.cusum_value,
                    "direction": bp.direction,
                    "pre_mean": bp.pre_mean,
                    "post_mean": bp.post_mean,
                    "confidence": bp.confidence,
                }
                for bp in breakpoints
            ],
            "recovery": {
                "num_drawdowns": recovery.num_drawdown_periods,
                "avg_depth": recovery.avg_drawdown_depth,
                "max_depth": recovery.max_drawdown_depth,
                "avg_recovery_bars": recovery.avg_recovery_bars,
                "recovery_factor": recovery.recovery_factor,
                "calmar": recovery.calmar_ratio,
                "underwater_fraction": recovery.underwater_fraction,
                "pain_index": recovery.pain_index,
                "ulcer_index": recovery.ulcer_index,
            },
            "bootstrap_sharpe": {
                "point_estimate": sharpe_pt,
                "ci_half_width_95": sharpe_ci,
                "ci_lo": sharpe_pt - sharpe_ci,
                "ci_hi": sharpe_pt + sharpe_ci,
            },
            "drawdown_prediction": {
                "next_bar_dd_probability": dd_prob,
            },
        }


# ---------------------------------------------------------------------------
# Plotting (optional, requires matplotlib)
# ---------------------------------------------------------------------------


def plot_decomposition(
    decomp: EquityDecomposition,
    title: str = "Equity Curve Decomposition",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the 4-panel decomposition chart using matplotlib.
    Silently skips if matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        return

    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    fig.suptitle(title, color="#c9d1d9", fontsize=14)
    gs = gridspec.GridSpec(4, 1, hspace=0.4)

    axes_configs = [
        (decomp.original, "Equity Curve", "#58a6ff"),
        (decomp.trend, "Trend (HP Filter)", "#3fb950"),
        (decomp.cyclical, "Cyclical (BK Bandpass)", "#f0883e"),
        (decomp.noise, "Noise (Residual)", "#f85149"),
    ]

    for i, (data, label, color) in enumerate(axes_configs):
        ax = fig.add_subplot(gs[i])
        ax.plot(data, color=color, linewidth=1.2)
        ax.set_ylabel(label, color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e")
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


def plot_regime_breakpoints(
    equity_curve: List[float],
    breakpoints: List[RegimeBreakpoint],
    title: str = "Regime Breakpoints (CUSUM)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot equity curve with CUSUM breakpoints overlaid.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(equity_curve, color="#58a6ff", linewidth=1.5, label="Equity")

    for bp in breakpoints:
        color = "#3fb950" if bp.direction == "down_to_up" else "#f85149"
        ax.axvline(x=bp.index, color=color, linestyle="--", alpha=0.7,
                   label=f"{bp.direction} @ {bp.index}")

    ax.set_title(title, color="#c9d1d9")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random as _rng

    rng = _rng.Random(0)
    # Simulate a bumpy equity curve
    nav = 100_000.0
    curve = [nav]
    for i in range(500):
        ret = rng.gauss(0.0005, 0.015)
        # Inject regime change at bar 200
        if 200 <= i < 300:
            ret = rng.gauss(-0.001, 0.02)
        nav *= (1.0 + ret)
        curve.append(nav)

    analyzer = EquityCurveAnalyzer(hp_lambda=1600.0, bk_K=8)
    result = analyzer.full_analysis(curve, n_boot=200)

    print(f"Trend slope: {result['decomposition']['trend_slope']:.4f}")
    print(f"Noise std: {result['decomposition']['noise_std']:.2f}")
    print(f"Signal-to-noise: {result['decomposition']['signal_to_noise']:.3f}")
    print(f"Regime breakpoints: {len(result['regime_breakpoints'])}")
    for bp in result["regime_breakpoints"][:3]:
        print(f"  bar {bp['index']:4d} | {bp['direction']} | conf={bp['confidence']:.2f}")
    print(f"Max drawdown: {result['recovery']['max_depth']:.2%}")
    print(f"Recovery factor: {result['recovery']['recovery_factor']:.2f}")
    print(f"Calmar: {result['recovery']['calmar']:.3f}")
    print(f"Sharpe: {result['bootstrap_sharpe']['point_estimate']:.3f} "
          f"+/- {result['bootstrap_sharpe']['ci_half_width_95']:.3f}")
    print(f"P(next bar DD): {result['drawdown_prediction']['next_bar_dd_probability']:.3f}")
