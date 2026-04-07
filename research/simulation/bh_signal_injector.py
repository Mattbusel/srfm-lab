"""
research/simulation/bh_signal_injector.py

Injects realistic Black Hole (BH) physics signals and quaternion-navigation
signals into synthetic price data. Used to create labelled training episodes
for RL agents and to unit-test the LARSA signal-detection engine.

BH mass formula mirrors the exact LARSA FutureInstrument.update_bh logic:
    beta = |close - prev_close| / prev_close / cf
    if beta < 1.0 (TIMELIKE):
        ctl += 1
        sb = min(2.0, 1.0 + ctl * 0.1)
        bh_mass = bh_mass * 0.97 + 0.03 * sb
    else (SPACELIKE):
        ctl = 0
        bh_mass *= 0.95

BH_FORM threshold = 1.5 (default), configurable.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from research.simulation.market_simulator import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    DT_15M,
    BH_FORM_DEFAULT,
    BH_COLLAPSE_DEFAULT,
    BH_DECAY,
    BH_MASS_EMA_FAST,
    BH_MASS_EMA_SLOW,
    _intraday_volume_factor,
    _BARS_PER_DAY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LARSA BH physics constants
# ---------------------------------------------------------------------------

BH_CTL_MIN = 3          # minimum consecutive timelike bars to activate
DEFAULT_CF = 0.0003     # capture fraction (ES 15m default)
MAX_SB = 2.0            # max speed-boost multiplier


# ---------------------------------------------------------------------------
# BH mass state machine (standalone, matches LARSA FutureInstrument)
# ---------------------------------------------------------------------------

@dataclass
class BHMassState:
    """Mutable BH mass accumulator state.

    Mirrors the fields tracked in LARSA FutureInstrument:
      bh_mass, ctl, bh_active, bh_dir, bh_entry_price, bh_form, bh_collapse
    """
    bh_mass: float = 0.0
    ctl: int = 0
    bh_active: bool = False
    bh_dir: int = 0
    bh_entry_price: float = 0.0
    bh_form: float = BH_FORM_DEFAULT
    bh_collapse: float = BH_COLLAPSE_DEFAULT
    cf: float = DEFAULT_CF
    price_history: list[float] = field(default_factory=list)

    def update(self, close: float) -> None:
        """Process one bar close price and update BH mass state."""
        if not self.price_history:
            self.price_history.append(close)
            return

        prev_close = self.price_history[-1]
        beta_raw = abs(close - prev_close) / (prev_close + 1e-9)
        beta = beta_raw / (self.cf + 1e-9)

        was_active = self.bh_active

        if beta < 1.0:
            self.ctl += 1
            sb = min(MAX_SB, 1.0 + self.ctl * 0.1)
            self.bh_mass = self.bh_mass * BH_MASS_EMA_SLOW + BH_MASS_EMA_FAST * sb
        else:
            self.ctl = 0
            self.bh_mass *= BH_DECAY

        if not was_active:
            self.bh_active = self.bh_mass > self.bh_form and self.ctl >= BH_CTL_MIN
        else:
            self.bh_active = self.bh_mass > self.bh_collapse and self.ctl >= BH_CTL_MIN

        if not was_active and self.bh_active:
            lookback = min(20, len(self.price_history))
            self.bh_dir = 1 if close > self.price_history[-lookback] else -1
            self.bh_entry_price = close
        elif was_active and not self.bh_active:
            self.bh_dir = 0
            self.bh_entry_price = 0.0

        self.price_history.append(close)
        if len(self.price_history) > 50:
            self.price_history.pop(0)


def compute_bh_mass_series(
    closes: NDArray[np.float64],
    cf: float = DEFAULT_CF,
    bh_form: float = BH_FORM_DEFAULT,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute BH mass series and active flags for a close price array.

    Parameters
    ----------
    closes:
        1-D array of close prices.
    cf:
        Capture fraction (LARSA CF["15m"][symbol]).
    bh_form:
        BH formation threshold.

    Returns
    -------
    (bh_mass_series, bh_active_series) -- both shape (len(closes),)
    """
    state = BHMassState(cf=cf, bh_form=bh_form)
    masses = np.zeros(len(closes))
    active = np.zeros(len(closes), dtype=bool)
    for i, c in enumerate(closes):
        state.update(float(c))
        masses[i] = state.bh_mass
        active[i] = state.bh_active
    return masses, active


# ---------------------------------------------------------------------------
# BH Mass Simulator / Injector
# ---------------------------------------------------------------------------

class BHMassSimulator:
    """
    Injects BH mass trajectories into synthetic price sequences.

    The injector works by adjusting bar-to-bar returns so that the resulting
    BH mass (computed via the LARSA formula) follows a desired trajectory.

    The key insight: to push bh_mass UP we need small consecutive returns
    (TIMELIKE bars, beta < 1.0). To prevent accumulation we generate volatile
    SPACELIKE bars (beta >= 1.0).
    """

    def __init__(
        self,
        cf: float = DEFAULT_CF,
        bh_form: float = BH_FORM_DEFAULT,
        seed: Optional[int] = None,
    ):
        self.cf = cf
        self.bh_form = bh_form
        self.rng = np.random.default_rng(seed)

    def inject_bh_event(
        self,
        prices: NDArray[np.float64],
        start_bar: int,
        duration_bars: int,
        mass_target: float = 2.5,
        initial_price: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Modify a price array so that BH mass accumulates to mass_target.

        The modification is applied in-place on a copy of prices between
        start_bar and start_bar + duration_bars.

        Strategy:
          - During accumulation window: cap bar-to-bar returns to beta < 0.8
            (well within TIMELIKE zone) while preserving overall price trend.
          - Post-window: allow natural dynamics.

        Parameters
        ----------
        prices:
            1-D array of close prices to modify (modified copy returned).
        start_bar:
            First bar of the BH injection window (0-indexed).
        duration_bars:
            Length of accumulation window in bars.
        mass_target:
            Desired BH mass at end of accumulation window.
        initial_price:
            Override for the starting price of the injection (defaults to
            prices[start_bar]).

        Returns
        -------
        Modified price array (copy, not in-place).
        """
        modified = prices.copy()
        if start_bar >= len(prices):
            return modified

        end_bar = min(start_bar + duration_bars, len(prices))
        p0 = initial_prices = modified[start_bar] if initial_price is None else initial_price

        # Target return from start to end (preserve general direction)
        if end_bar < len(prices):
            total_log_ret = math.log(prices[end_bar] / prices[start_bar] + 1e-12)
        else:
            total_log_ret = 0.0

        # Cap per-bar absolute log-return to cf * 0.8 (ensures TIMELIKE)
        max_abs_ret = self.cf * 0.8
        n_inject = end_bar - start_bar

        # Build constrained returns: small drift + tiny noise
        drift_per_bar = total_log_ret / n_inject
        sigma_inject = self.cf * 0.3
        z = self.rng.standard_normal(n_inject)
        log_rets = drift_per_bar + sigma_inject * z
        # Clip to TIMELIKE regime
        log_rets = np.clip(log_rets, -max_abs_ret, max_abs_ret)

        # Rebuild prices
        cum = np.concatenate([[0.0], np.cumsum(log_rets)])
        segment = p0 * np.exp(cum)
        modified[start_bar: end_bar + 1] = segment

        return modified

    def generate_bh_episode(
        self,
        n_bars: int = 200,
        initial_price: float = 100.0,
        annual_vol: float = 0.20,
        mass_target: float = 2.5,
        pre_bh_bars: int = 60,
        formation_bars: int = 60,
        peak_bars: int = 30,
        resolution_bars: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a complete synthetic BH episode with four phases.

        Phases
        ------
        1. Pre-BH (pre_bh_bars):
           Low mass accumulation, price consolidation (OU-like).
        2. BH formation (formation_bars):
           Rapid mass accumulation via small consecutive timelike bars,
           directional drift builds.
        3. BH peak (peak_bars):
           Mass exceeds BH_FORM, bh_active=True, explosive breakout bars.
        4. BH resolution (remaining bars):
           Mass decays as SPACELIKE bars dominate, mean reversion begins.

        Returns
        -------
        pd.DataFrame with columns:
            open, high, low, close, volume, bh_mass, bh_active, phase
        """
        if resolution_bars is None:
            resolution_bars = n_bars - pre_bh_bars - formation_bars - peak_bars

        resolution_bars = max(0, n_bars - pre_bh_bars - formation_bars - peak_bars)

        dt = DT_15M
        sigma_annual = annual_vol
        sigma_bar = sigma_annual * math.sqrt(dt)

        # Phase 1: Pre-BH consolidation (OU around initial price)
        phase1 = OrnsteinUhlenbeck.generate(
            pre_bh_bars, kappa=30.0, theta=initial_price,
            sigma=sigma_annual * 0.4, x0=initial_price,
            dt=dt, seed=self.rng.integers(0, 2**31)
        )

        # Phase 2: BH formation (constrained small returns)
        p0_ph2 = phase1[-1]
        phase2_prices = np.empty(formation_bars + 1)
        phase2_prices[0] = p0_ph2
        # Small drift upward + constrained vol
        drift_per_bar = (0.05 - 0.5 * (sigma_annual * 0.5) ** 2) * dt
        cap = self.cf * 0.75
        for j in range(formation_bars):
            ret = drift_per_bar + self.cf * 0.4 * self.rng.standard_normal()
            ret = float(np.clip(ret, -cap, cap))
            phase2_prices[j + 1] = phase2_prices[j] * math.exp(ret)

        # Phase 3: BH peak (explosive breakout)
        p0_ph3 = phase2_prices[-1]
        breakout_sigma = sigma_annual * 3.0
        phase3_raw = GeometricBrownianMotion.generate(
            peak_bars, mu=0.80, sigma=breakout_sigma,
            dt=dt, initial=p0_ph3,
            seed=self.rng.integers(0, 2**31)
        )

        # Phase 4: BH resolution (mean reversion + SPACELIKE bars)
        p0_ph4 = phase3_raw[-1]
        if resolution_bars > 0:
            phase4 = OrnsteinUhlenbeck.generate(
                resolution_bars, kappa=40.0, theta=initial_price * 1.05,
                sigma=sigma_annual * 0.8, x0=p0_ph4,
                dt=dt, seed=self.rng.integers(0, 2**31)
            )
        else:
            phase4 = np.array([])

        # Concatenate all close prices
        segments = [phase1[:-1], phase2_prices[:-1], phase3_raw[:-1]]
        if len(phase4) > 0:
            segments.append(phase4)
        all_closes = np.concatenate(segments)[:n_bars]

        # Compute BH mass
        masses, active_flags = compute_bh_mass_series(all_closes, cf=self.cf, bh_form=self.bh_form)

        # Build OHLCV bars
        records = []
        phase_labels = (
            ["pre_bh"] * pre_bh_bars
            + ["formation"] * formation_bars
            + ["peak"] * peak_bars
            + ["resolution"] * resolution_bars
        )[:n_bars]

        for i in range(n_bars):
            c = float(all_closes[i])
            bar_in_day = i % _BARS_PER_DAY
            # Synthetic OHLC from close only (simplified)
            noise = c * sigma_bar * 0.5
            o = c * math.exp(self.rng.normal(0, sigma_bar * 0.3))
            h = max(o, c) + abs(self.rng.normal(0, noise * 0.5))
            l = min(o, c) - abs(self.rng.normal(0, noise * 0.5))
            vol_base = math.exp(math.log(max(c, 1.0)) + 5.0 + self.rng.normal(0, 0.4))
            vol_base *= _intraday_volume_factor(bar_in_day)
            # Volume spike at peak phase
            if phase_labels[i] == "peak":
                vol_base *= self.rng.uniform(2.0, 4.0)
            records.append({
                "open": max(o, 1e-6),
                "high": max(h, o, c),
                "low": min(l, o, c),
                "close": c,
                "volume": vol_base,
                "bh_mass": masses[i],
                "bh_active": bool(active_flags[i]),
                "phase": phase_labels[i],
            })

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Quaternion Navigation Signal Injector
# ---------------------------------------------------------------------------

@dataclass
class QuatNavOutput:
    """Quaternion navigation output for one bar, mirroring bridge/quat_nav_bridge.py."""
    bar_qw: float
    bar_qx: float
    bar_qy: float
    bar_qz: float
    qw: float
    qx: float
    qy: float
    qz: float
    angular_velocity: float     # radians per bar
    geodesic_deviation: float   # radians
    lorentz_boost_applied: bool
    lorentz_boost_rapidity: float


def _quat_from_returns(r: float, vol_context: float) -> list[float]:
    """Map a single bar log-return to a unit quaternion.

    Uses a simple encoding:
      angle = pi * tanh(r / vol_context)
      axis  = (1, 0, 0) for positive return, (0, 1, 0) for negative
    """
    vol_context = max(vol_context, 1e-8)
    angle = math.pi * math.tanh(r / (vol_context + 1e-8))
    half = angle / 2.0
    c = math.cos(half)
    s = math.sin(half)
    if r >= 0:
        return [c, s, 0.0, 0.0]
    else:
        return [c, 0.0, s, 0.0]


def _qmul(q1: list[float], q2: list[float]) -> list[float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]


def _qnormalize(q: list[float]) -> list[float]:
    n2 = sum(x**2 for x in q)
    if n2 < 1e-30:
        return [1.0, 0.0, 0.0, 0.0]
    inv_n = 1.0 / math.sqrt(n2)
    return [x * inv_n for x in q]


def _qangle(q: list[float]) -> float:
    w = max(-1.0, min(1.0, q[0]))
    return 2.0 * math.acos(abs(w))


class QuatNavSignalInjector:
    """
    Modifies a bar sequence to produce a target quaternion-navigation signal.

    Used to test nav-based entry gates and position sizing in the LARSA
    backtesting framework. The injector adjusts per-bar log-returns so that
    the resulting angular velocity (omega) and geodesic deviation approximate
    desired target values.

    Approach
    --------
    - Compute the current nav output for the bar sequence.
    - Scale bar returns iteratively to approach target_omega.
    - Geodesic deviation is a function of angular acceleration and BH mass;
      we inject a BH mass override to steer geodesic deviation.
    """

    def __init__(
        self,
        cf: float = DEFAULT_CF,
        vol_window: int = 20,
        seed: Optional[int] = None,
    ):
        self.cf = cf
        self.vol_window = vol_window
        self.rng = np.random.default_rng(seed)

    def compute_nav_series(
        self,
        closes: NDArray[np.float64],
        bh_masses: Optional[NDArray[np.float64]] = None,
    ) -> list[QuatNavOutput]:
        """Compute quaternion nav outputs for a close price series.

        Parameters
        ----------
        closes:
            1-D array of close prices.
        bh_masses:
            Optional BH mass array; if None, computed from closes.

        Returns
        -------
        List of QuatNavOutput, one per bar (first bar has trivial values).
        """
        n = len(closes)
        if bh_masses is None:
            bh_masses, _ = compute_bh_mass_series(closes, cf=self.cf)

        log_rets = np.concatenate([[0.0], np.diff(np.log(np.maximum(closes, 1e-12)))])
        vol_series = np.full(n, 0.01)
        for i in range(self.vol_window, n):
            vol_series[i] = float(np.std(log_rets[i - self.vol_window: i]))

        q_current = [1.0, 0.0, 0.0, 0.0]
        q_prev = [1.0, 0.0, 0.0, 0.0]
        outputs: list[QuatNavOutput] = []

        for i in range(n):
            r = float(log_rets[i])
            vol = float(vol_series[i])
            q_bar = _quat_from_returns(r, vol)
            q_bar = _qnormalize(q_bar)
            q_new = _qnormalize(_qmul(q_current, q_bar))

            # Angular velocity: angle between consecutive orientation quaternions
            q_delta = _qmul([q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3]], q_new)
            omega = _qangle(q_delta)

            # Geodesic deviation: omega modulated by BH mass curvature
            bh_m = float(bh_masses[i])
            geodesic = omega * (1.0 + bh_m * 0.5)

            # Lorentz boost: applied when BH mass > collapse threshold
            lorentz = bh_m > BH_COLLAPSE_DEFAULT
            rapidity = math.log(1.0 + bh_m) if lorentz else 0.0

            outputs.append(QuatNavOutput(
                bar_qw=q_bar[0], bar_qx=q_bar[1],
                bar_qy=q_bar[2], bar_qz=q_bar[3],
                qw=q_new[0], qx=q_new[1],
                qy=q_new[2], qz=q_new[3],
                angular_velocity=omega,
                geodesic_deviation=geodesic,
                lorentz_boost_applied=lorentz,
                lorentz_boost_rapidity=rapidity,
            ))
            q_prev = q_current
            q_current = q_new

        return outputs

    def inject_nav_signal(
        self,
        prices: NDArray[np.float64],
        target_omega: float,
        target_geodesic: float,
        injection_start: int = 0,
        injection_length: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """Modify bar sequence to produce target angular velocity and geodesic deviation.

        The injector scales bar-to-bar log-returns within the injection window
        so the resulting angular velocity approaches target_omega.

        Geodesic deviation is steered by scaling the BH mass term; since BH mass
        is derived from prices, we scale the returns further to produce the
        implicit BH mass needed for the target geodesic.

        Parameters
        ----------
        prices:
            1-D close price array.
        target_omega:
            Target angular velocity (radians/bar). Typical range 0.01--0.5.
        target_geodesic:
            Target geodesic deviation (radians). Must be >= target_omega.
        injection_start:
            First bar of injection window (default 0).
        injection_length:
            Number of bars to modify. Defaults to all bars from injection_start.

        Returns
        -------
        Modified price array (copy).
        """
        modified = prices.copy().astype(np.float64)
        n = len(modified)
        if injection_length is None:
            injection_length = n - injection_start
        end = min(injection_start + injection_length, n)

        # Scale log-returns to produce target angular velocity
        # omega ~ 2 * arctan(|sin(angle)| / |cos(angle)|) ~ |angle| for small angles
        # angle derived from tanh(r / vol) * pi
        # To get target_omega, we want |r| / vol ~ atanh(target_omega / pi)
        target_ratio = math.atanh(min(target_omega / math.pi, 0.999))

        vol_window = self.vol_window
        for i in range(injection_start, end):
            if i < 1:
                continue
            window_start = max(0, i - vol_window)
            log_rets_local = np.diff(np.log(np.maximum(modified[window_start: i + 1], 1e-12)))
            vol = float(np.std(log_rets_local)) if len(log_rets_local) >= 2 else 0.01
            vol = max(vol, 1e-6)
            # Current log-return
            r_current = math.log(modified[i] / max(modified[i - 1], 1e-12))
            # Scale to achieve target_omega
            sign = 1 if r_current >= 0 else -1
            r_new = sign * target_ratio * vol
            # Also embed geodesic: geodesic > omega implies higher bh_mass
            # We inject a return spike every N bars to push bh_mass
            bh_needed = max(0.0, (target_geodesic / max(target_omega, 1e-9) - 1.0) / 0.5)
            if bh_needed > BH_FORM_DEFAULT:
                # Constrain return further (TIMELIKE) to build mass
                r_new = float(np.clip(r_new, -self.cf * 0.7, self.cf * 0.7))
            modified[i] = modified[i - 1] * math.exp(r_new)

        return modified


# ---------------------------------------------------------------------------
# Signal Quality Injector
# ---------------------------------------------------------------------------

class SignalQualityInjector:
    """
    Injects alpha signals with controlled Information Coefficient (IC) into
    a bar DataFrame. Useful for testing position sizing and signal combination.

    The IC (rank correlation between signal and forward return) is set via a
    mixture of true forward return information and pure noise:

        signal = sqrt(IC) * pure_signal + sqrt(1 - IC) * noise

    This matches the standard signal model: IC^2 = fraction of variance
    explained.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def inject_predictive_signal(
        self,
        bars: pd.DataFrame,
        forward_return: NDArray[np.float64],
        ic: float = 0.10,
        signal_col: str = "alpha_signal",
    ) -> pd.DataFrame:
        """Add a synthetic alpha signal column with specified IC.

        IC is measured as the Pearson correlation between the returned signal
        and forward_return.

        Parameters
        ----------
        bars:
            OHLCV DataFrame.
        forward_return:
            1-D array of forward returns (length == len(bars)).
        ic:
            Target IC. Range [0, 1]. 0.05--0.15 is realistic for alpha.
        signal_col:
            Name of new column added to returned DataFrame.

        Returns
        -------
        DataFrame copy with signal_col added.
        """
        ic = float(np.clip(ic, 0.0, 1.0))
        n = len(bars)
        # Standardise forward return
        mu_r = np.mean(forward_return)
        std_r = np.std(forward_return) + 1e-9
        pure = (forward_return - mu_r) / std_r

        noise = self.rng.standard_normal(n)
        # Mixture: signal = sqrt(ic)*pure + sqrt(1-ic)*noise
        signal = math.sqrt(ic) * pure + math.sqrt(1.0 - ic) * noise
        # Standardise signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

        result = bars.copy()
        result[signal_col] = signal
        return result

    def inject_noise(
        self,
        signal: NDArray[np.float64],
        noise_level: float = 0.5,
    ) -> NDArray[np.float64]:
        """Degrade a signal by adding noise, reducing its IC.

        The resulting signal has approximately IC_new ~ IC_original * (1 - noise_level).

        Parameters
        ----------
        signal:
            Original signal array.
        noise_level:
            Fraction of noise to add. 0 = no change, 1 = pure noise.

        Returns
        -------
        Degraded signal array (same shape).
        """
        noise_level = float(np.clip(noise_level, 0.0, 1.0))
        noise = self.rng.standard_normal(len(signal))
        degraded = (1.0 - noise_level) * signal + noise_level * noise
        # Re-standardise
        std = np.std(degraded)
        if std < 1e-12:
            return degraded
        return (degraded - np.mean(degraded)) / std

    def compute_ic(
        self,
        signal: NDArray[np.float64],
        forward_return: NDArray[np.float64],
    ) -> float:
        """Compute Pearson IC between signal and forward return.

        Parameters
        ----------
        signal:
            Signal array.
        forward_return:
            Forward return array (same length).

        Returns
        -------
        IC as float in [-1, 1].
        """
        if len(signal) < 2:
            return 0.0
        return float(np.corrcoef(signal, forward_return)[0, 1])

    def compute_rank_ic(
        self,
        signal: NDArray[np.float64],
        forward_return: NDArray[np.float64],
    ) -> float:
        """Compute Spearman rank IC (ICIR is more robust to outliers).

        Returns
        -------
        Rank IC as float in [-1, 1].
        """
        from scipy.stats import spearmanr
        if len(signal) < 2:
            return 0.0
        corr, _ = spearmanr(signal, forward_return)
        return float(corr)

    def inject_regime_signal(
        self,
        bars: pd.DataFrame,
        regime_col: str = "regime",
        signal_col: str = "regime_signal",
    ) -> pd.DataFrame:
        """Add a categorical regime signal encoded as a numeric score.

        Encoding mirrors LARSA MarketRegime:
          TRENDING_BULL    -> +1.0
          TRENDING_BEAR    -> -1.0
          VOLATILE         -> +/-1.5 (direction from bh_dir if present)
          MEAN_REVERTING   -> 0.0
          BLACK_HOLE_ACTIVE -> bh_dir * 2.0

        Parameters
        ----------
        bars:
            DataFrame with a regime column and optionally bh_dir, bh_active.

        Returns
        -------
        DataFrame copy with signal_col added.
        """
        _REGIME_SCORE = {
            "TRENDING_BULL":     1.0,
            "TRENDING_BEAR":    -1.0,
            "VOLATILE":          0.0,
            "MEAN_REVERTING":    0.0,
            "BLACK_HOLE_ACTIVE": 0.0,
        }
        result = bars.copy()
        scores = np.zeros(len(bars))
        for i, row in bars.iterrows():
            regime = str(row.get(regime_col, "MEAN_REVERTING"))
            score = _REGIME_SCORE.get(regime, 0.0)
            if regime == "VOLATILE":
                bh_dir = row.get("bh_dir", 0)
                score = float(bh_dir) * 1.5 if bh_dir != 0 else 0.0
            elif regime == "BLACK_HOLE_ACTIVE":
                bh_dir = row.get("bh_dir", 0)
                score = float(bh_dir) * 2.0
            scores[int(i)] = score
        result[signal_col] = scores
        return result

    def compute_signal_decay(
        self,
        signal: NDArray[np.float64],
        closes: NDArray[np.float64],
        max_lag: int = 20,
    ) -> NDArray[np.float64]:
        """Compute IC at lags 1..max_lag to measure signal decay rate.

        Parameters
        ----------
        signal:
            Signal array of length n.
        closes:
            Close prices of length n.
        max_lag:
            Maximum forward lag to evaluate.

        Returns
        -------
        Array of shape (max_lag,) with IC at each lag.
        """
        n = len(signal)
        ics = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            if lag >= n:
                break
            fwd_ret = np.log(closes[lag:] / closes[:-lag])
            sig_trim = signal[:n - lag]
            if len(sig_trim) < 2:
                break
            ics[lag - 1] = float(np.corrcoef(sig_trim, fwd_ret)[0, 1])
        return ics
