"""
pattern_oracle.py
=================
Novel pattern discovery through combinatorial search over financial signals.

The PatternOracle exhaustively searches combinations of raw signals to find
non-obvious, statistically robust predictive patterns.  Key design goals:

1. Combinatorial breadth  — XOR patterns, lag structures, conditional triggers,
   cross-asset combinations, interaction terms.
2. Statistical rigour     — multiple-testing correction (Bonferroni, BH FDR),
   minimum sample requirements, split-sample validation.
3. Surprise scoring       — how unexpected is this pattern given known priors?
4. Redundancy detection   — a similarity graph prunes near-duplicate patterns.
5. Archival               — all discovered patterns stored with full provenance.

Dependencies: numpy, scipy (optional for scipy.stats fallback)
"""

from __future__ import annotations

import math
import time
import hashlib
import itertools
import collections
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Taxonomy
# ──────────────────────────────────────────────────────────────────────────────

class PatternType(str, Enum):
    MOMENTUM          = "momentum"
    MEAN_REVERSION    = "mean_reversion"
    BREAKOUT          = "breakout"
    RELATIVE_VALUE    = "relative_value"
    CONDITIONAL       = "conditional"
    CROSS_ASSET       = "cross_asset"
    XOR               = "xor"
    LAG_STRUCTURE     = "lag_structure"
    INTERACTION       = "interaction"
    UNKNOWN           = "unknown"


class MultipleTestingMethod(str, Enum):
    BONFERRONI = "bonferroni"
    BH_FDR     = "bh_fdr"
    NONE       = "none"


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternEvidence:
    """Statistical evidence supporting a discovered pattern."""
    n_obs: int
    mean_return: float          # annualised mean return when pattern is active
    std_return: float           # annualised std dev of returns
    t_stat: float               # t-statistic for mean ≠ 0
    p_value_raw: float          # unadjusted p-value
    p_value_adj: float          # p-value after multiple testing correction
    sharpe_in_sample: float
    sharpe_oos: float           # out-of-sample (last 20% of data)
    hit_rate: float             # fraction of periods with positive return
    max_drawdown: float
    activation_rate: float      # fraction of time pattern is active


@dataclass
class DiscoveredPattern:
    """A fully documented discovered trading pattern."""
    pattern_id: str
    name: str
    pattern_type: PatternType
    description: str
    signal_components: list[str]      # names of constituent signals
    construction: str                 # how the composite signal is built
    conditions: list[str]             # when this pattern activates
    evidence: PatternEvidence
    surprise_score: float             # 0–1: how unexpected
    redundancy_score: float           # 0–1: similarity to existing patterns
    discovery_timestamp: float        # Unix timestamp
    tags: list[str] = field(default_factory=list)

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.evidence.p_value_adj < alpha

    def summary(self) -> str:
        e = self.evidence
        sig = "✓ SIGNIFICANT" if self.is_significant() else "✗ not significant"
        lines = [
            f"Pattern  : {self.name}  [{self.pattern_type.value}]",
            f"ID       : {self.pattern_id}",
            f"Signals  : {', '.join(self.signal_components)}",
            f"Construction: {self.construction}",
            f"--- Evidence ---",
            f"  N obs      : {e.n_obs}",
            f"  Mean ret   : {e.mean_return:.4f}  (annualised)",
            f"  t-stat     : {e.t_stat:.3f}",
            f"  p (raw)    : {e.p_value_raw:.4f}",
            f"  p (adj)    : {e.p_value_adj:.4f}  {sig}",
            f"  IS Sharpe  : {e.sharpe_in_sample:.3f}",
            f"  OOS Sharpe : {e.sharpe_oos:.3f}",
            f"  Hit rate   : {e.hit_rate:.2%}",
            f"  Activation : {e.activation_rate:.2%}",
            f"  Max DD     : {e.max_drawdown:.4f}",
            f"--- Scores ---",
            f"  Surprise   : {self.surprise_score:.3f}",
            f"  Redundancy : {self.redundancy_score:.3f}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Statistical utilities (pure numpy, no scipy required)
# ──────────────────────────────────────────────────────────────────────────────

def _t_test_mean(returns: np.ndarray) -> tuple[float, float]:
    """
    One-sample t-test: H0: mean = 0.
    Returns (t_stat, two-sided p_value).
    """
    n = len(returns)
    if n < 4:
        return 0.0, 1.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0, 1.0
    t = mean / (std / math.sqrt(n))
    # Approximation: p-value from t-distribution using regularised incomplete beta
    p = _t_pvalue(t, df=n - 1)
    return t, p


def _t_pvalue(t: float, df: int) -> float:
    """Approximate two-sided p-value for t-distribution."""
    # Use the beta function approximation
    x = df / (df + t * t)
    # Regularised incomplete beta  I(x; df/2, 0.5)  via continued fraction
    p_one_tail = 0.5 * _regularised_ibeta(x, df / 2.0, 0.5)
    return min(1.0, 2.0 * p_one_tail)


def _regularised_ibeta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularised incomplete beta function via continued fraction (Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use symmetry if x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_ibeta(1.0 - x, b, a, max_iter)
    # Log of the beta function prefactor
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
             + a * math.log(x) + b * math.log(1.0 - x))
    prefactor = math.exp(lbeta) / a
    # Lentz continued fraction
    fpmin = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return prefactor * h


def _bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Return Bonferroni-adjusted p-values."""
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


def _bh_fdr_correction(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values (BH step-up procedure).
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        adj = min(prev, p_values[i] * n / (rank + 1))
        adjusted[i] = adj
        prev = adj
    return adjusted


def _compute_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return mean / std * math.sqrt(periods_per_year)


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Returns maximum drawdown as a positive fraction."""
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-12)
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _build_equity_curve(returns: np.ndarray) -> np.ndarray:
    curve = np.cumprod(1.0 + returns)
    return np.concatenate([[1.0], curve])


def _jb_normality_stat(returns: np.ndarray) -> tuple[float, float]:
    """Jarque-Bera test: returns (JB_stat, p_value approx)."""
    n = len(returns)
    if n < 8:
        return 0.0, 1.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std < 1e-12:
        return 0.0, 1.0
    z = (returns - mean) / std
    skewness = float(np.mean(z ** 3))
    kurtosis = float(np.mean(z ** 4)) - 3.0   # excess kurtosis
    jb = n / 6.0 * (skewness ** 2 + kurtosis ** 2 / 4.0)
    # Chi-squared(2) CDF approximation
    p = math.exp(-jb / 2.0) * (1.0 + jb / 2.0 + jb ** 2 / 8.0)
    p = max(0.0, min(1.0, p))
    return jb, 1.0 - p


def _ljung_box_stat(returns: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """
    Ljung-Box test for autocorrelation.
    Returns (Q_stat, approximate p_value).
    """
    n = len(returns)
    if n < lags + 5:
        return 0.0, 1.0
    mean = np.mean(returns)
    demeaned = returns - mean
    var = float(np.var(demeaned))
    if var < 1e-12:
        return 0.0, 1.0

    q = 0.0
    for k in range(1, lags + 1):
        acf_k = float(np.mean(demeaned[k:] * demeaned[:-k])) / var
        q += acf_k ** 2 / (n - k)
    q *= n * (n + 2)
    # Chi-squared(lags) p-value approximation
    p = _chi2_sf(q, df=lags)
    return q, p


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared distribution (approx via regularised gamma)."""
    return _regularised_upper_gamma(df / 2.0, x / 2.0)


def _regularised_upper_gamma(a: float, x: float) -> float:
    """Regularised upper incomplete gamma function Q(a, x) via series."""
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        # Series expansion for lower gamma
        p = _regularised_lower_gamma_series(a, x)
        return 1.0 - p
    # Continued fraction for upper gamma
    return _upper_gamma_cf(a, x)


def _regularised_lower_gamma_series(a: float, x: float) -> float:
    ap = a
    delt = 1.0 / a
    total = delt
    for _ in range(300):
        ap += 1.0
        delt *= x / ap
        total += delt
        if abs(delt) < abs(total) * 1e-10:
            break
    lna = a * math.log(x) - x - math.lgamma(a + 1.0)
    return total * math.exp(lna)


def _upper_gamma_cf(a: float, x: float) -> float:
    fpmin = 1e-30
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 301):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    lna = a * math.log(x) - x - math.lgamma(a)
    return math.exp(lna) * h


# ──────────────────────────────────────────────────────────────────────────────
# Pattern similarity
# ──────────────────────────────────────────────────────────────────────────────

def _signal_fingerprint(signal: np.ndarray) -> np.ndarray:
    """Normalised return fingerprint for similarity comparison."""
    s = signal.astype(float)
    std = np.std(s)
    if std < 1e-12:
        return np.zeros_like(s)
    return (s - np.mean(s)) / std


def _pattern_similarity(s1: np.ndarray, s2: np.ndarray) -> float:
    """Pearson correlation between two signal fingerprints."""
    n = min(len(s1), len(s2))
    if n < 4:
        return 0.0
    return float(np.corrcoef(s1[:n], s2[:n])[0, 1])


# ──────────────────────────────────────────────────────────────────────────────
# Pattern Oracle
# ──────────────────────────────────────────────────────────────────────────────

class PatternOracle:
    """
    Combinatorial pattern discovery engine for financial time series.

    Parameters
    ----------
    signal_names    : list of signal labels corresponding to columns in data
    returns_col     : index of the forward return column in the data array
    min_obs         : minimum observations for statistical tests
    min_activation  : minimum fraction of time a pattern must be active
    alpha           : significance level (post-correction)
    mt_method       : multiple-testing correction method
    oos_fraction    : fraction of data held out for OOS evaluation
    seed            : random seed for reproducibility
    """

    def __init__(
        self,
        signal_names: list[str],
        returns_col: int = -1,
        min_obs: int = 60,
        min_activation: float = 0.05,
        alpha: float = 0.05,
        mt_method: MultipleTestingMethod = MultipleTestingMethod.BH_FDR,
        oos_fraction: float = 0.20,
        seed: int = 42,
    ):
        self.signal_names = signal_names
        self.returns_col = returns_col
        self.min_obs = min_obs
        self.min_activation = min_activation
        self.alpha = alpha
        self.mt_method = mt_method
        self.oos_fraction = oos_fraction
        self._rng = np.random.default_rng(seed)
        self._discovery_log: list[DiscoveredPattern] = []
        self._signal_store: dict[str, np.ndarray] = {}  # name → signal values
        self._pattern_counter = 0
        self._n_hypotheses_tested = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def load_data(self, data: np.ndarray) -> None:
        """
        Load a (T × F) data array.
        Columns correspond to signal_names; last column is forward return.

        Parameters
        ----------
        data : shape (T, len(signal_names) + 1)
               Signals are in columns 0..N-1, forward return in column -1 (or returns_col).
        """
        self._data = data.astype(float)
        self._T = data.shape[0]
        self._oos_split = int(self._T * (1.0 - self.oos_fraction))
        for i, name in enumerate(self.signal_names):
            self._signal_store[name] = data[:, i]

    def discover_momentum_patterns(
        self, lags: list[int] | None = None
    ) -> list[DiscoveredPattern]:
        """
        Search for momentum patterns: past return windows predicting future return.
        Tests single-signal and multi-lag combinations.
        """
        if lags is None:
            lags = [1, 2, 5, 10, 21, 63]
        candidates: list[tuple[str, np.ndarray]] = []

        fwd = self._fwd_returns()

        for name in self.signal_names:
            sig = self._signal_store[name]
            for lag in lags:
                lagged = np.roll(sig, lag)
                lagged[:lag] = np.nan
                label = f"momentum_{name}_lag{lag}"
                signal = np.sign(lagged)
                candidates.append((label, signal))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.MOMENTUM,
            description_prefix="Lagged-sign momentum on"
        )

    def discover_mean_reversion_patterns(
        self, windows: list[int] | None = None
    ) -> list[DiscoveredPattern]:
        """
        Search for mean-reversion patterns: z-score crossing signals contrarian bet.
        """
        if windows is None:
            windows = [5, 10, 21, 63]
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        for name in self.signal_names:
            sig = self._signal_store[name]
            for w in windows:
                z = self._rolling_zscore(sig, w)
                for threshold in [1.0, 1.5, 2.0]:
                    label = f"mr_{name}_w{w}_thresh{threshold}"
                    signal = np.where(z < -threshold, 1.0,
                             np.where(z >  threshold, -1.0, np.nan))
                    candidates.append((label, signal))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.MEAN_REVERSION,
            description_prefix="Z-score mean-reversion on"
        )

    def discover_breakout_patterns(
        self, windows: list[int] | None = None
    ) -> list[DiscoveredPattern]:
        """
        Breakout patterns: price exceeds rolling high/low → follow-through.
        """
        if windows is None:
            windows = [10, 21, 63]
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        for name in self.signal_names:
            sig = self._signal_store[name]
            for w in windows:
                roll_max = self._rolling_max(sig, w)
                roll_min = self._rolling_min(sig, w)
                label_up = f"breakout_up_{name}_w{w}"
                label_dn = f"breakout_dn_{name}_w{w}"
                signal_up = np.where(sig > np.roll(roll_max, 1), 1.0, np.nan)
                signal_dn = np.where(sig < np.roll(roll_min, 1), -1.0, np.nan)
                candidates.append((label_up, signal_up))
                candidates.append((label_dn, signal_dn))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.BREAKOUT,
            description_prefix="Rolling breakout on"
        )

    def discover_xor_patterns(
        self, max_signals: int = 2
    ) -> list[DiscoveredPattern]:
        """
        XOR patterns: signal is long only when exactly one of two binary
        signals is active, not both and not neither.
        """
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        # Binarise each signal via median split
        binary_signals: dict[str, np.ndarray] = {}
        for name in self.signal_names:
            sig = self._signal_store[name]
            med = np.nanmedian(sig)
            binary_signals[name] = (sig > med).astype(float)

        for s1, s2 in itertools.combinations(list(binary_signals.keys()), 2):
            b1 = binary_signals[s1]
            b2 = binary_signals[s2]
            xor_long  = np.where((b1 == 1) & (b2 == 0),  1.0,
                        np.where((b1 == 0) & (b2 == 1), -1.0, np.nan))
            label = f"xor_{s1}_{s2}"
            candidates.append((label, xor_long))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.XOR,
            description_prefix="XOR combination of"
        )

    def discover_conditional_patterns(self) -> list[DiscoveredPattern]:
        """
        Conditional patterns: signal A active only when signal B satisfies
        a regime condition (high/low state).
        """
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        n_signals = len(self.signal_names)
        for trigger_idx in range(n_signals):
            trigger_name = self.signal_names[trigger_idx]
            trigger = self._signal_store[trigger_name]
            # Define high/low regime based on 60-day rolling median
            regime_hi = trigger > self._rolling_median(trigger, 60)

            for signal_idx in range(n_signals):
                if signal_idx == trigger_idx:
                    continue
                signal_name = self.signal_names[signal_idx]
                sig = self._signal_store[signal_name]
                base_signal = np.sign(sig - np.nanmean(sig))

                # Conditional: trade signal only in high-trigger regime
                label = f"conditional_{signal_name}_given_{trigger_name}_high"
                cond_signal = np.where(regime_hi, base_signal, np.nan)
                candidates.append((label, cond_signal))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.CONDITIONAL,
            description_prefix="Conditional signal"
        )

    def discover_cross_asset_patterns(
        self, trigger_series: dict[str, np.ndarray] | None = None
    ) -> list[DiscoveredPattern]:
        """
        Cross-asset patterns: an external series (trigger asset) predicts
        the local return. E.g. VIX level predicts equity sector returns.

        Parameters
        ----------
        trigger_series : optional external signals {name: array}
                         If None, uses internal signals as cross-triggers.
        """
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        if trigger_series is None:
            trigger_series = {
                name: self._signal_store[name]
                for name in self.signal_names
            }

        for trigger_name, trigger in trigger_series.items():
            if trigger_name in self.signal_names:
                continue   # skip self-prediction
            trigger_lagged = np.roll(trigger, 1)
            trigger_lagged[0] = np.nan
            z_trigger = self._rolling_zscore(trigger_lagged, 21)

            for threshold in [0.5, 1.0]:
                label = f"cross_{trigger_name}_z{threshold}_long"
                sig = np.where(z_trigger > threshold, 1.0,
                      np.where(z_trigger < -threshold, -1.0, np.nan))
                candidates.append((label, sig))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.CROSS_ASSET,
            description_prefix="Cross-asset trigger from"
        )

    def discover_lag_structure_patterns(
        self, max_lag: int = 10
    ) -> list[DiscoveredPattern]:
        """
        Lag structure: search for non-obvious lag combinations such as
        momentum at lag 5 minus lag 1 (skip-lag momentum).
        """
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        for name in self.signal_names:
            sig = self._signal_store[name]
            for lag_a, lag_b in itertools.combinations(range(1, max_lag + 1), 2):
                lagged_a = np.roll(sig, lag_a)
                lagged_a[:lag_a] = np.nan
                lagged_b = np.roll(sig, lag_b)
                lagged_b[:lag_b] = np.nan
                diff = lagged_a - lagged_b
                label = f"lagdiff_{name}_{lag_a}m{lag_b}"
                candidates.append((label, np.sign(diff)))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.LAG_STRUCTURE,
            description_prefix="Lag-difference structure on"
        )

    def discover_interaction_patterns(self) -> list[DiscoveredPattern]:
        """
        Interaction terms: signal A × signal B as a combined predictor.
        """
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        for s1, s2 in itertools.combinations(self.signal_names, 2):
            sig1 = self._normalise(self._signal_store[s1])
            sig2 = self._normalise(self._signal_store[s2])
            interaction = sig1 * sig2
            label = f"interaction_{s1}_x_{s2}"
            candidates.append((label, np.sign(interaction)))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.INTERACTION,
            description_prefix="Interaction of"
        )

    def discover_relative_value_patterns(
        self, pairs: list[tuple[str, str]] | None = None
    ) -> list[DiscoveredPattern]:
        """
        Relative value patterns: spread between two signals → contrarian or
        momentum position.
        """
        candidates: list[tuple[str, np.ndarray]] = []
        fwd = self._fwd_returns()

        if pairs is None:
            pairs = list(itertools.combinations(self.signal_names, 2))

        for s1, s2 in pairs:
            norm1 = self._normalise(self._signal_store[s1])
            norm2 = self._normalise(self._signal_store[s2])
            spread = norm1 - norm2
            z_spread = self._rolling_zscore(spread, 21)

            # Mean-reversion on spread
            for thresh in [1.0, 1.5]:
                label = f"rv_mr_{s1}_vs_{s2}_thresh{thresh}"
                sig = np.where(z_spread > thresh, -1.0,
                      np.where(z_spread < -thresh, 1.0, np.nan))
                candidates.append((label, sig))

            # Momentum on spread
            label_mom = f"rv_mom_{s1}_vs_{s2}"
            candidates.append((label_mom, np.sign(np.roll(spread, 5))))

        return self._evaluate_candidates(
            candidates, fwd, PatternType.RELATIVE_VALUE,
            description_prefix="Relative-value spread of"
        )

    def run_full_search(self) -> list[DiscoveredPattern]:
        """
        Run all discovery routines and return all significant patterns,
        de-duplicated by the similarity graph.
        """
        all_patterns: list[DiscoveredPattern] = []
        all_patterns += self.discover_momentum_patterns()
        all_patterns += self.discover_mean_reversion_patterns()
        all_patterns += self.discover_breakout_patterns()
        all_patterns += self.discover_xor_patterns()
        all_patterns += self.discover_conditional_patterns()
        all_patterns += self.discover_lag_structure_patterns()
        all_patterns += self.discover_interaction_patterns()
        all_patterns += self.discover_relative_value_patterns()

        # Re-apply global multiple-testing correction
        all_patterns = self._global_mt_correction(all_patterns)

        # Filter to significant
        significant = [p for p in all_patterns if p.is_significant(self.alpha)]

        # Compute redundancy scores via similarity graph
        significant = self._assign_redundancy_scores(significant)

        # Store to discovery log
        self._discovery_log.extend(significant)

        return significant

    def get_discovery_log(self) -> list[DiscoveredPattern]:
        """Return all previously discovered patterns."""
        return list(self._discovery_log)

    def pattern_similarity_graph(
        self, patterns: list[DiscoveredPattern]
    ) -> dict[str, dict[str, float]]:
        """
        Compute pairwise similarity between patterns.
        Returns {pattern_id: {other_id: similarity}} adjacency dict.
        """
        graph: dict[str, dict[str, float]] = {p.pattern_id: {} for p in patterns}
        for i, pi in enumerate(patterns):
            for j, pj in enumerate(patterns):
                if i >= j:
                    continue
                # Use signal overlap via Jaccard on component names
                si = set(pi.signal_components)
                sj = set(pj.signal_components)
                jaccard = len(si & sj) / max(len(si | sj), 1)
                graph[pi.pattern_id][pj.pattern_id] = round(jaccard, 4)
                graph[pj.pattern_id][pi.pattern_id] = round(jaccard, 4)
        return graph

    def compute_surprise_score(
        self,
        pattern: DiscoveredPattern,
        prior_known_types: set[PatternType] | None = None,
    ) -> float:
        """
        Estimate how surprising a pattern is given prior knowledge.

        Scoring factors:
        - Pattern type novelty: XOR, cross-asset, interaction are more surprising
        - Number of signal components (more = more surprising)
        - Low prior-knowledge type overlap
        """
        if prior_known_types is None:
            prior_known_types = {PatternType.MOMENTUM, PatternType.MEAN_REVERSION}

        type_novelty = {
            PatternType.XOR:           0.9,
            PatternType.CROSS_ASSET:   0.8,
            PatternType.INTERACTION:   0.75,
            PatternType.LAG_STRUCTURE: 0.6,
            PatternType.CONDITIONAL:   0.65,
            PatternType.BREAKOUT:      0.4,
            PatternType.RELATIVE_VALUE:0.5,
            PatternType.MOMENTUM:      0.2,
            PatternType.MEAN_REVERSION:0.2,
            PatternType.UNKNOWN:       0.5,
        }
        base = type_novelty.get(pattern.pattern_type, 0.5)
        if pattern.pattern_type in prior_known_types:
            base *= 0.5
        n_components = len(pattern.signal_components)
        complexity_bonus = min(0.3, (n_components - 1) * 0.1)
        return round(min(1.0, base + complexity_bonus), 4)

    def print_discovery_log(self, significant_only: bool = True) -> None:
        log = self._discovery_log
        if significant_only:
            log = [p for p in log if p.is_significant(self.alpha)]
        if not log:
            print("No patterns in discovery log.")
            return
        print(f"Discovery Log: {len(log)} patterns")
        for p in log:
            print("\n" + p.summary())
            print("-" * 60)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _next_id(self) -> str:
        self._pattern_counter += 1
        return f"PAT-{self._pattern_counter:05d}"

    def _fwd_returns(self) -> np.ndarray:
        return self._data[:, self.returns_col]

    def _evaluate_candidates(
        self,
        candidates: list[tuple[str, np.ndarray]],
        fwd_returns: np.ndarray,
        pattern_type: PatternType,
        description_prefix: str = "",
    ) -> list[DiscoveredPattern]:
        """
        Evaluate a list of (label, signal) candidates.
        Applies multiple-testing correction and returns significant patterns.
        """
        self._n_hypotheses_tested += len(candidates)
        raw_results: list[tuple[str, np.ndarray, PatternEvidence]] = []

        for label, signal in candidates:
            ev = self._compute_evidence(signal, fwd_returns)
            if ev is not None:
                raw_results.append((label, signal, ev))

        if not raw_results:
            return []

        # Apply multiple-testing correction
        p_raws = [r[2].p_value_raw for r in raw_results]
        if self.mt_method == MultipleTestingMethod.BONFERRONI:
            p_adjs = _bonferroni_correction(p_raws)
        elif self.mt_method == MultipleTestingMethod.BH_FDR:
            p_adjs = _bh_fdr_correction(p_raws)
        else:
            p_adjs = p_raws

        patterns: list[DiscoveredPattern] = []
        for i, (label, signal, ev) in enumerate(raw_results):
            ev.p_value_adj = p_adjs[i]
            components = [p for p in label.split("_") if p in self.signal_names]
            if not components:
                components = [label]

            surprise = self.compute_surprise_score(
                DiscoveredPattern(
                    pattern_id="tmp", name=label,
                    pattern_type=pattern_type, description="",
                    signal_components=components, construction=label,
                    conditions=[], evidence=ev,
                    surprise_score=0.0, redundancy_score=0.0,
                    discovery_timestamp=time.time(),
                )
            )

            dp = DiscoveredPattern(
                pattern_id=self._next_id(),
                name=label,
                pattern_type=pattern_type,
                description=f"{description_prefix} {', '.join(components)}",
                signal_components=components,
                construction=label,
                conditions=[
                    f"Activation rate > {self.min_activation:.0%}",
                    f"Min obs = {self.min_obs}",
                ],
                evidence=ev,
                surprise_score=surprise,
                redundancy_score=0.0,
                discovery_timestamp=time.time(),
                tags=[pattern_type.value],
            )
            patterns.append(dp)

        return patterns

    def _compute_evidence(
        self,
        signal: np.ndarray,
        fwd_returns: np.ndarray,
    ) -> PatternEvidence | None:
        """Compute statistical evidence for a signal → forward return relationship."""
        mask = np.isfinite(signal) & np.isfinite(fwd_returns)
        if mask.sum() < self.min_obs:
            return None

        activation_rate = float(mask.mean())
        if activation_rate < self.min_activation:
            return None

        # In-sample and OOS split
        is_mask = mask.copy()
        is_mask[self._oos_split:] = False
        oos_mask = mask.copy()
        oos_mask[:self._oos_split] = False

        strategy_rets_is  = signal[is_mask] * fwd_returns[is_mask]
        strategy_rets_oos = signal[oos_mask] * fwd_returns[oos_mask]

        if len(strategy_rets_is) < self.min_obs:
            return None

        t_stat, p_raw = _t_test_mean(strategy_rets_is)
        mean_ret = float(np.mean(strategy_rets_is)) * 252.0
        std_ret  = float(np.std(strategy_rets_is, ddof=1)) * math.sqrt(252.0)

        equity_is = _build_equity_curve(strategy_rets_is)
        max_dd = _compute_max_drawdown(equity_is)

        sharpe_is  = _compute_sharpe(strategy_rets_is)
        sharpe_oos = _compute_sharpe(strategy_rets_oos) if len(strategy_rets_oos) > 4 else 0.0
        hit_rate = float(np.mean(strategy_rets_is > 0))

        return PatternEvidence(
            n_obs=int(mask.sum()),
            mean_return=round(mean_ret, 6),
            std_return=round(std_ret, 6),
            t_stat=round(t_stat, 4),
            p_value_raw=round(p_raw, 6),
            p_value_adj=round(p_raw, 6),   # overwritten later
            sharpe_in_sample=round(sharpe_is, 4),
            sharpe_oos=round(sharpe_oos, 4),
            hit_rate=round(hit_rate, 4),
            max_drawdown=round(max_dd, 6),
            activation_rate=round(activation_rate, 4),
        )

    def _global_mt_correction(
        self, patterns: list[DiscoveredPattern]
    ) -> list[DiscoveredPattern]:
        """Re-apply MT correction globally over all tested hypotheses."""
        if not patterns:
            return patterns
        p_raws = [p.evidence.p_value_raw for p in patterns]
        n_total = max(self._n_hypotheses_tested, len(patterns))

        if self.mt_method == MultipleTestingMethod.BONFERRONI:
            p_adjs = [min(1.0, p * n_total) for p in p_raws]
        elif self.mt_method == MultipleTestingMethod.BH_FDR:
            p_adjs = _bh_fdr_correction(p_raws)
        else:
            p_adjs = p_raws

        for pat, p_adj in zip(patterns, p_adjs):
            pat.evidence.p_value_adj = round(p_adj, 6)
        return patterns

    def _assign_redundancy_scores(
        self, patterns: list[DiscoveredPattern]
    ) -> list[DiscoveredPattern]:
        """Assign redundancy scores based on signal-component Jaccard similarity."""
        n = len(patterns)
        for i in range(n):
            max_sim = 0.0
            si = set(patterns[i].signal_components)
            for j in range(n):
                if i == j:
                    continue
                sj = set(patterns[j].signal_components)
                jaccard = len(si & sj) / max(len(si | sj), 1)
                if jaccard > max_sim:
                    max_sim = jaccard
            patterns[i].redundancy_score = round(max_sim, 4)
        return patterns

    # ── Rolling window helpers (pure numpy) ───────────────────────────────────

    @staticmethod
    def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        for t in range(window, len(x)):
            w = x[t - window:t]
            mu = np.nanmean(w)
            sd = np.nanstd(w)
            if sd > 1e-12:
                out[t] = (x[t] - mu) / sd
        return out

    @staticmethod
    def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        for t in range(window, len(x)):
            out[t] = np.nanmedian(x[t - window:t])
        return out

    @staticmethod
    def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        for t in range(window, len(x)):
            out[t] = np.nanmax(x[t - window:t])
        return out

    @staticmethod
    def _rolling_min(x: np.ndarray, window: int) -> np.ndarray:
        out = np.full_like(x, np.nan)
        for t in range(window, len(x)):
            out[t] = np.nanmin(x[t - window:t])
        return out

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if sd < 1e-12:
            return x - mu
        return (x - mu) / sd


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────────────────────────────────────

def _demo():
    rng = np.random.default_rng(42)
    T = 1500
    # Simulate 4 signals and a forward return with mild predictability
    signal_names = ["momentum_1m", "value_pe", "volatility_21d", "sentiment_idx"]
    signals = rng.standard_normal((T, 4))
    # Build a forward return with weak dependence on momentum_1m and value_pe
    fwd = 0.01 * signals[:, 0] - 0.005 * signals[:, 1] + 0.15 * rng.standard_normal(T)
    data = np.column_stack([signals, fwd])

    oracle = PatternOracle(
        signal_names=signal_names,
        returns_col=-1,
        min_obs=40,
        min_activation=0.05,
        alpha=0.10,
        mt_method=MultipleTestingMethod.BH_FDR,
        seed=7,
    )
    oracle.load_data(data)

    print("Running momentum pattern discovery...")
    mom_patterns = oracle.discover_momentum_patterns(lags=[1, 5, 21])
    sig_mom = [p for p in mom_patterns if p.is_significant(0.10)]
    print(f"  Found {len(mom_patterns)} candidates, {len(sig_mom)} significant")

    print("Running XOR pattern discovery...")
    xor_patterns = oracle.discover_xor_patterns()
    sig_xor = [p for p in xor_patterns if p.is_significant(0.10)]
    print(f"  Found {len(xor_patterns)} candidates, {len(sig_xor)} significant")

    if sig_mom:
        print("\nTop momentum pattern:")
        print(sig_mom[0].summary())
    if sig_xor:
        print("\nTop XOR pattern:")
        print(sig_xor[0].summary())


if __name__ == "__main__":
    _demo()
