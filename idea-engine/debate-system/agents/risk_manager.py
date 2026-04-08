"""
debate-system/agents/risk_manager.py

RiskManager — comprehensive risk assessment debate agent.

Evaluates hypotheses across seven risk dimensions:
  1. Position sizing  : Kelly criterion + vol targeting
  2. VaR / CVaR       : Historical simulation + parametric
  3. Correlation risk : Portfolio correlation contribution
  4. Tail risk        : EVT / GPD fit to losses
  5. Drawdown risk    : Expected max drawdown from vol + drift
  6. Liquidity risk   : Position size vs average daily volume
  7. Regime-specific  : Risk multipliers by market regime

Has hard-veto power: if projected MDD increase > 20pp or EVT 99% CVaR
exceeds portfolio hard limit, the agent issues a near-certain AGAINST.

evaluate() returns a full RiskAssessment dataclass.
analyze() wraps evaluate() into an AnalystVerdict for the DebateChamber.

Expected market_data keys
-------------------------
returns              : np.ndarray — historical return series (daily, decimal)
portfolio_returns    : np.ndarray — current portfolio return series
position_size_usd    : float — proposed position in USD
portfolio_nav        : float — total portfolio value
adv_usd              : float — average daily volume in USD
win_rate             : float — estimated win rate [0, 1]
avg_win              : float — average winning return
avg_loss             : float — average losing return (positive magnitude)
vol_target           : float — portfolio vol target (annualized, e.g. 0.20)
current_regime       : str  — "bull" | "bear" | "vol_spike" | "trending" | "ranging"
portfolio_corr_matrix: np.ndarray — (N, N) correlation matrix of current positions
asset_correlations   : np.ndarray — correlations of new asset with each position
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar

from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote
from hypothesis.types import Hypothesis


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LiquidityRisk(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    SEVERE = "severe"


class TailRisk(str, Enum):
    LIGHT  = "light"     # GPD shape < 0.1
    MEDIUM = "medium"    # shape 0.1–0.3
    HEAVY  = "heavy"     # shape > 0.3 (heavy tail)
    EXTREME = "extreme"  # shape > 0.5


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """
    Comprehensive risk evaluation result.

    risk_score : 0.0 (very low risk) to 1.0 (very high / veto-level risk)
    """

    # --- Position sizing ---
    kelly_fraction: float               # raw Kelly (may be > 1, apply fractional)
    fractional_kelly: float             # 0.25 * kelly (conservative)
    vol_target_size: float              # position size from vol targeting (% NAV)
    recommended_size_pct: float         # min(fractional_kelly, vol_target)

    # --- VaR / CVaR ---
    var_95_hist: float                  # 95% VaR, historical simulation
    var_99_hist: float                  # 99% VaR, historical simulation
    var_95_param: float                 # 95% VaR, parametric (normal)
    cvar_95: float                      # 95% CVaR / Expected Shortfall
    cvar_99: float                      # 99% CVaR / Expected Shortfall

    # --- Correlation risk ---
    marginal_portfolio_corr: float      # how much this trade increases avg portfolio corr
    portfolio_corr_increase: float      # absolute increase in portfolio correlation
    diversification_ratio: float        # 1.0 = fully diversifying, 0.0 = duplicate

    # --- Tail risk (EVT) ---
    gpd_shape: float                    # Generalized Pareto shape parameter (ξ)
    gpd_scale: float                    # GPD scale parameter
    tail_risk: TailRisk
    evt_var_99: float                   # EVT-implied 99% VaR
    evt_cvar_99: float                  # EVT-implied 99% CVaR

    # --- Drawdown risk ---
    expected_max_drawdown: float        # analytical estimate from vol + drift
    drawdown_risk_score: float          # 0-1

    # --- Liquidity risk ---
    adv_fraction: float                 # position_size_usd / adv_usd
    liquidity_risk: LiquidityRisk
    days_to_exit: float                 # estimated days to unwind at 10% ADV

    # --- Regime risk multiplier ---
    regime_multiplier: float            # 0.5 (low risk regime) to 2.0 (high risk)
    current_regime: str

    # --- Composite ---
    risk_score: float                   # 0 (safe) to 1.0 (veto-level)
    conviction_adjusted: float          # conviction * (1 - risk_score)
    hard_veto: bool                     # True = should be rejected regardless

    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager(BaseAnalyst):
    """
    Portfolio risk and tail-event specialist.

    Provides Kelly sizing, VaR/CVaR, EVT tail risk, correlation risk,
    liquidity risk, and drawdown estimation.  Has hard-veto authority.
    """

    # Hard veto thresholds
    MAX_MDD_INCREASE   = 0.20   # 20pp portfolio MDD increase
    MAX_CVAR_99        = 0.25   # 25% portfolio CVaR 99% (existential risk)
    MAX_ADV_FRACTION   = 0.10   # >10% of ADV = severe liquidity concern

    # Soft concern thresholds
    SOFT_ADV_FRACTION  = 0.03   # >3% of ADV = flag
    MIN_DIVERSIFICATION = 0.30  # below = correlated / redundant

    # Regime risk multipliers (applied to position sizing)
    REGIME_MULTIPLIERS: Dict[str, float] = {
        "bull":        0.8,
        "trending":    0.9,
        "ranging":     1.0,
        "bear":        1.4,
        "vol_spike":   1.8,
        "crisis":      2.5,
        "unknown":     1.2,
    }

    def __init__(self) -> None:
        super().__init__(
            name="RiskManager",
            specialization=(
                "Kelly sizing, VaR/CVaR, EVT tail risk, drawdown estimation, "
                "correlation risk, liquidity risk, regime-specific adjustments"
            ),
        )

    # ------------------------------------------------------------------
    # Primary rich output: RiskAssessment
    # ------------------------------------------------------------------

    def evaluate(
        self,
        hypothesis: Hypothesis,
        market_data: Dict[str, Any],
    ) -> RiskAssessment:
        """Full risk assessment returning a structured RiskAssessment."""
        returns:       np.ndarray = np.asarray(market_data.get("returns", []), dtype=np.float64)
        port_returns:  np.ndarray = np.asarray(market_data.get("portfolio_returns", []), dtype=np.float64)
        pos_size_usd:  float      = float(market_data.get("position_size_usd", 10_000.0))
        nav:           float      = float(market_data.get("portfolio_nav",      100_000.0))
        adv_usd:       float      = float(market_data.get("adv_usd",            1_000_000.0))
        win_rate:      float      = float(market_data.get("win_rate",           0.50))
        avg_win:       float      = abs(float(market_data.get("avg_win",        0.02)))
        avg_loss:      float      = abs(float(market_data.get("avg_loss",       0.015)))
        vol_target:    float      = float(market_data.get("vol_target",         0.20))
        regime:        str        = str(market_data.get("current_regime",       "unknown")).lower()
        conviction:    float      = float(market_data.get("conviction",         0.5))

        port_corr_matrix: Optional[np.ndarray] = market_data.get("portfolio_corr_matrix")
        asset_corrs:      Optional[np.ndarray] = market_data.get("asset_correlations")

        warnings: List[str] = []

        # Ensure at least minimal return series
        if len(returns) < 20:
            returns = np.random.default_rng(0).normal(0, 0.02, 100)
        if len(port_returns) < 20:
            port_returns = np.random.default_rng(1).normal(0, 0.015, 100)

        # --- Kelly sizing ---
        kelly, frac_kelly, vol_size, rec_size = self._kelly_and_vol_target(
            win_rate, avg_win, avg_loss, returns, vol_target, nav, pos_size_usd
        )

        # --- VaR / CVaR ---
        v95h, v99h, v95p, c95, c99 = self._var_cvar(returns)

        # --- Correlation risk ---
        marg_corr, corr_increase, div_ratio = self._correlation_risk(
            port_returns, returns, port_corr_matrix, asset_corrs
        )
        if div_ratio < self.MIN_DIVERSIFICATION:
            warnings.append(
                f"Low diversification ratio ({div_ratio:.2f}) — trade is correlated "
                f"with existing book, adds concentration not alpha"
            )

        # --- EVT tail risk ---
        gpd_xi, gpd_sigma, tail_risk, evt_v99, evt_c99 = self._evt_tail_risk(returns)
        if tail_risk in (TailRisk.HEAVY, TailRisk.EXTREME):
            warnings.append(
                f"Heavy tail detected (GPD ξ={gpd_xi:.3f}) — parametric VaR "
                f"significantly underestimates true tail risk"
            )

        # --- Drawdown risk ---
        ann_vol   = float(np.std(returns) * math.sqrt(252))
        ann_drift = float(np.mean(returns) * 252)
        exp_mdd, dd_score = self._expected_max_drawdown(ann_drift, ann_vol)
        if exp_mdd > 0.30:
            warnings.append(f"Expected max drawdown {exp_mdd:.1%} — substantial drawdown risk")

        # --- Liquidity risk ---
        adv_frac, liq_risk, days_exit = self._liquidity_risk(pos_size_usd, adv_usd)
        if liq_risk in (LiquidityRisk.HIGH, LiquidityRisk.SEVERE):
            warnings.append(
                f"Liquidity concern: position is {adv_frac:.1%} of ADV "
                f"(~{days_exit:.1f} days to exit at 10% ADV)"
            )

        # --- Regime multiplier ---
        reg_mult = self.REGIME_MULTIPLIERS.get(regime, 1.2)
        if reg_mult >= 1.5:
            warnings.append(f"High-risk regime '{regime}' — apply {reg_mult:.1f}x risk multiplier")

        # --- Composite risk score ---
        # Normalize each component to [0, 1] danger level
        mdd_danger   = float(np.clip(exp_mdd / 0.50, 0, 1))
        cvar_danger  = float(np.clip(evt_c99 / 0.20, 0, 1))
        corr_danger  = float(np.clip(1.0 - div_ratio, 0, 1))
        liq_danger   = float(np.clip(adv_frac / self.MAX_ADV_FRACTION, 0, 1))
        tail_danger  = {TailRisk.LIGHT: 0.1, TailRisk.MEDIUM: 0.35,
                        TailRisk.HEAVY: 0.65, TailRisk.EXTREME: 0.90}[tail_risk]

        risk_score = float(np.clip(
            0.30 * mdd_danger +
            0.25 * cvar_danger +
            0.20 * corr_danger +
            0.15 * liq_danger +
            0.10 * tail_danger,
            0.0, 1.0,
        ))

        # Apply regime multiplier to risk score
        risk_score = float(np.clip(risk_score * (reg_mult / 1.0) * 0.5 + risk_score * 0.5, 0, 1))

        # Hard veto check
        hard_veto = (evt_c99 > self.MAX_CVAR_99) or (exp_mdd > 0.50)
        if hard_veto:
            warnings.insert(0, f"HARD VETO: EVT CVaR99={evt_c99:.1%} or MDD={exp_mdd:.1%} exceeds hard limit")

        conviction_adj = float(np.clip(conviction * (1.0 - risk_score), 0.0, 1.0))

        return RiskAssessment(
            kelly_fraction=kelly,
            fractional_kelly=frac_kelly,
            vol_target_size=vol_size,
            recommended_size_pct=rec_size,
            var_95_hist=v95h,
            var_99_hist=v99h,
            var_95_param=v95p,
            cvar_95=c95,
            cvar_99=c99,
            marginal_portfolio_corr=marg_corr,
            portfolio_corr_increase=corr_increase,
            diversification_ratio=div_ratio,
            gpd_shape=gpd_xi,
            gpd_scale=gpd_sigma,
            tail_risk=tail_risk,
            evt_var_99=evt_v99,
            evt_cvar_99=evt_c99,
            expected_max_drawdown=exp_mdd,
            drawdown_risk_score=dd_score,
            adv_fraction=adv_frac,
            liquidity_risk=liq_risk,
            days_to_exit=days_exit,
            regime_multiplier=reg_mult,
            current_regime=regime,
            risk_score=risk_score,
            conviction_adjusted=conviction_adj,
            hard_veto=hard_veto,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # BaseAnalyst.analyze() — debate chamber interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        hypothesis: Hypothesis,
        market_data: Dict[str, Any],
    ) -> AnalystVerdict:
        assessment = self.evaluate(hypothesis, market_data)
        rs = assessment.risk_score

        if assessment.hard_veto:
            return self._make_verdict(
                Vote.AGAINST,
                confidence=0.97,
                reasoning=(
                    f"HARD RISK VETO. EVT CVaR99={assessment.evt_cvar_99:.1%}, "
                    f"expected MDD={assessment.expected_max_drawdown:.1%}. "
                    f"Risk exceeds hard portfolio limits."
                ),
                key_concerns=assessment.warnings,
            )

        if rs < 0.25:
            vote, confidence = Vote.FOR, 0.80
            reasoning = (
                f"Risk profile acceptable. Score={rs:.3f}. "
                f"Rec size={assessment.recommended_size_pct:.1%} NAV, "
                f"EVT CVaR99={assessment.evt_cvar_99:.1%}, "
                f"div_ratio={assessment.diversification_ratio:.2f}."
            )
        elif rs < 0.50:
            vote, confidence = Vote.FOR, 0.55
            reasoning = (
                f"Moderate risk — proceed with reduced sizing. Score={rs:.3f}. "
                f"Apply {assessment.regime_multiplier:.1f}x regime multiplier. "
                f"Rec size={assessment.recommended_size_pct:.1%} NAV."
            )
        elif rs < 0.70:
            vote, confidence = Vote.AGAINST, 0.65
            reasoning = (
                f"Elevated risk — not recommended at current conditions. "
                f"Score={rs:.3f}. "
                f"Key issues: {'; '.join(assessment.warnings[:2])}."
            )
        else:
            vote, confidence = Vote.AGAINST, 0.85
            reasoning = (
                f"High risk profile. Score={rs:.3f}. "
                f"EVT CVaR99={assessment.evt_cvar_99:.1%}, "
                f"MDD={assessment.expected_max_drawdown:.1%}, "
                f"liquidity={assessment.liquidity_risk.value}."
            )

        return self._make_verdict(
            vote=vote,
            confidence=confidence,
            reasoning=reasoning,
            key_concerns=assessment.warnings,
        )

    # ------------------------------------------------------------------
    # Private: Kelly + vol targeting
    # ------------------------------------------------------------------

    def _kelly_and_vol_target(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        returns: np.ndarray,
        vol_target: float,
        nav: float,
        pos_size_usd: float,
    ) -> tuple[float, float, float, float]:
        """
        Kelly fraction: f* = (p/b) - (q/1) = p - q/b  where b = avg_win/avg_loss.
        Vol targeting: size = vol_target / position_vol / sqrt(252).
        Returns (kelly, frac_kelly, vol_size_pct, rec_size_pct).
        """
        if avg_loss < 1e-9:
            avg_loss = 1e-9
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        kelly = float(np.clip(win_rate - q / b, 0.0, 1.0))
        frac_kelly = kelly * 0.25  # quarter-Kelly for safety

        pos_vol = float(np.std(returns) * math.sqrt(252))
        if pos_vol > 1e-9:
            vol_size = float(np.clip(vol_target / pos_vol, 0.0, 0.50))
        else:
            vol_size = 0.05

        rec_size = float(min(frac_kelly, vol_size))
        return kelly, frac_kelly, vol_size, rec_size

    # ------------------------------------------------------------------
    # Private: VaR / CVaR
    # ------------------------------------------------------------------

    def _var_cvar(
        self,
        returns: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """
        Historical and parametric VaR + CVaR.
        Returns (var95h, var99h, var95p, cvar95, cvar99) as positive magnitudes.
        """
        losses = -returns  # flip sign: losses are positive

        # Historical simulation
        v95h = float(np.percentile(losses, 95))
        v99h = float(np.percentile(losses, 99))
        cvar95 = float(np.mean(losses[losses >= v95h]))
        cvar99 = float(np.mean(losses[losses >= v99h])) if np.sum(losses >= v99h) > 0 else v99h

        # Parametric (normal)
        mu  = float(np.mean(losses))
        sig = float(np.std(losses)) + 1e-12
        v95p = float(mu + sig * sp_stats.norm.ppf(0.95))

        return v95h, v99h, v95p, cvar95, cvar99

    # ------------------------------------------------------------------
    # Private: Correlation risk
    # ------------------------------------------------------------------

    def _correlation_risk(
        self,
        port_returns: np.ndarray,
        asset_returns: np.ndarray,
        corr_matrix: Optional[np.ndarray],
        asset_corrs: Optional[np.ndarray],
    ) -> tuple[float, float, float]:
        """
        Estimate marginal correlation increase from adding this asset.

        Returns (marginal_corr, corr_increase, diversification_ratio).
        diversification_ratio: 0 = fully correlated, 1 = uncorrelated.
        """
        n = min(len(port_returns), len(asset_returns))
        if n < 10:
            return 0.0, 0.0, 0.5

        p = port_returns[-n:]
        a = asset_returns[-n:]

        if np.std(p) < 1e-9 or np.std(a) < 1e-9:
            return 0.0, 0.0, 0.5

        corr, _ = sp_stats.pearsonr(p, a)
        corr = float(np.clip(corr, -1.0, 1.0))
        abs_corr = abs(corr)

        # If we have the full correlation matrix, compute average current correlation
        if corr_matrix is not None and corr_matrix.ndim == 2:
            n_assets = corr_matrix.shape[0]
            if n_assets > 1:
                upper_tri = corr_matrix[np.triu_indices(n_assets, k=1)]
                avg_current = float(np.mean(np.abs(upper_tri)))
            else:
                avg_current = 0.0
        else:
            avg_current = 0.3  # assume moderate existing correlation

        # Marginal impact on average portfolio correlation
        if asset_corrs is not None:
            marginal = float(np.mean(np.abs(asset_corrs)))
        else:
            marginal = abs_corr

        corr_increase = max(0.0, marginal - avg_current)
        div_ratio = float(np.clip(1.0 - abs_corr, 0.0, 1.0))

        return marginal, corr_increase, div_ratio

    # ------------------------------------------------------------------
    # Private: EVT / GPD tail risk
    # ------------------------------------------------------------------

    def _evt_tail_risk(
        self,
        returns: np.ndarray,
        threshold_pct: float = 90.0,
    ) -> tuple[float, float, TailRisk, float, float]:
        """
        Fit a Generalized Pareto Distribution (GPD) to exceedances above
        a high threshold to estimate tail risk.

        Returns (xi, sigma, tail_risk_label, evt_var99, evt_cvar99).
        xi > 0 → heavy tail (Frechet / Pareto), xi = 0 → exponential, xi < 0 → bounded.
        """
        losses = -returns
        threshold = float(np.percentile(losses, threshold_pct))
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 10:
            # Fall back to parametric
            sig = float(np.std(losses))
            return 0.0, sig, TailRisk.LIGHT, float(np.percentile(losses, 99)), float(np.mean(losses[losses > np.percentile(losses, 99)]))

        # Method of Moments for GPD: xi = 0.5*(1 - m^2/s^2), sigma = 0.5*m*(1 + m^2/s^2)
        m = float(np.mean(exceedances))
        s2 = float(np.var(exceedances))
        if s2 < 1e-12:
            xi, sigma = 0.0, m
        else:
            xi    = float(0.5 * (1.0 - m ** 2 / s2))
            sigma = float(0.5 * m * (1.0 + m ** 2 / s2))
            sigma = max(sigma, 1e-9)

        # Classify tail heaviness
        if xi < 0.1:
            tail_risk = TailRisk.LIGHT
        elif xi < 0.3:
            tail_risk = TailRisk.MEDIUM
        elif xi < 0.5:
            tail_risk = TailRisk.HEAVY
        else:
            tail_risk = TailRisk.EXTREME

        # EVT VaR and CVaR at 99%
        n_total = len(losses)
        n_exceed = len(exceedances)
        p_exceed = n_exceed / n_total

        # GPD quantile function: VaR_q = u + (sigma/xi)*((n/Nu*(1-q))^(-xi) - 1)
        q = 0.99
        try:
            if abs(xi) < 1e-6:
                evt_var99 = threshold + sigma * math.log(p_exceed / (1.0 - q))
            else:
                evt_var99 = threshold + (sigma / xi) * (
                    math.pow(p_exceed / (1.0 - q), xi) - 1.0
                )
            evt_var99 = float(max(evt_var99, float(np.percentile(losses, 99))))
        except (ValueError, OverflowError):
            evt_var99 = float(np.percentile(losses, 99))

        # CVaR from GPD: E[L | L > VaR] = (VaR + sigma - xi*threshold) / (1 - xi)
        try:
            if xi < 1.0:
                evt_cvar99 = (evt_var99 + sigma - xi * threshold) / (1.0 - xi)
            else:
                evt_cvar99 = evt_var99 * 2.0  # infinite mean if xi >= 1
        except (ZeroDivisionError, ValueError):
            evt_cvar99 = evt_var99 * 1.5

        return xi, sigma, tail_risk, float(evt_var99), float(evt_cvar99)

    # ------------------------------------------------------------------
    # Private: Drawdown risk
    # ------------------------------------------------------------------

    def _expected_max_drawdown(
        self,
        drift: float,
        vol: float,
        horizon_days: int = 252,
    ) -> tuple[float, float]:
        """
        Approximate expected maximum drawdown using the Magdon-Ismail formula.

        E[MDD] ≈ vol * sqrt(T) * f(drift / (vol * sqrt(T)))
        where f is a correction for the drift.

        For simplicity we use: E[MDD] ≈ vol * sqrt(T) * sqrt(2 * ln(T))
        (standard Brownian bridge approximation for a driftless process),
        adjusted by max(0, -drift) for negative drift.

        Returns (expected_mdd, risk_score 0-1).
        """
        T = horizon_days
        if vol < 1e-9:
            return 0.0, 0.0

        # Brownian bridge maximum: E[max] ≈ vol*sqrt(T*2*ln(T))
        bm_max = vol / math.sqrt(252) * math.sqrt(T) * math.sqrt(2.0 * math.log(max(T, 2)))
        # Drift adjustment: negative drift worsens drawdown
        drift_adj = max(0.0, -drift / 252.0 * T)
        exp_mdd = float(bm_max + drift_adj)
        exp_mdd = float(np.clip(exp_mdd, 0.0, 1.0))
        dd_score = float(np.clip(exp_mdd / 0.5, 0.0, 1.0))
        return exp_mdd, dd_score

    # ------------------------------------------------------------------
    # Private: Liquidity risk
    # ------------------------------------------------------------------

    def _liquidity_risk(
        self,
        pos_size_usd: float,
        adv_usd: float,
    ) -> tuple[float, LiquidityRisk, float]:
        """
        Classify liquidity risk based on position as fraction of ADV.

        Assume unwinding at 10% of ADV per day.
        Returns (adv_fraction, LiquidityRisk, days_to_exit).
        """
        if adv_usd < 1.0:
            adv_usd = 1.0
        adv_frac = float(pos_size_usd / adv_usd)
        days_exit = float(adv_frac / 0.10)  # exit at 10% ADV

        if adv_frac < 0.01:
            liq_risk = LiquidityRisk.LOW
        elif adv_frac < self.SOFT_ADV_FRACTION:
            liq_risk = LiquidityRisk.MEDIUM
        elif adv_frac < self.MAX_ADV_FRACTION:
            liq_risk = LiquidityRisk.HIGH
        else:
            liq_risk = LiquidityRisk.SEVERE

        return adv_frac, liq_risk, days_exit


# ---------------------------------------------------------------------------
# Backward-compatible alias (original name used elsewhere in codebase)
# ---------------------------------------------------------------------------

RiskManagementAnalyst = RiskManager


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uuid
    from hypothesis.types import Hypothesis, HypothesisType, HypothesisStatus

    hyp = Hypothesis(
        hypothesis_id=str(uuid.uuid4()),
        type=HypothesisType.ENTRY_TIMING,
        parent_pattern_id="test",
        parameters={},
        predicted_sharpe_delta=0.3,
        predicted_dd_delta=-0.05,
        novelty_score=0.7,
        priority_rank=1,
        status=HypothesisStatus.PENDING,
        created_at="2026-01-01T00:00:00+00:00",
        description="Momentum entry on BTC breakout",
    )

    rng = np.random.default_rng(42)
    # Simulate asset with moderate vol and slight negative drift
    asset_returns    = rng.normal(-0.0005, 0.025, 500)
    portfolio_returns = rng.normal(0.0003, 0.012, 500)

    md: Dict[str, Any] = {
        "returns":          asset_returns,
        "portfolio_returns": portfolio_returns,
        "position_size_usd": 50_000,
        "portfolio_nav":     500_000,
        "adv_usd":           2_000_000,
        "win_rate":          0.52,
        "avg_win":           0.022,
        "avg_loss":          0.018,
        "vol_target":        0.20,
        "current_regime":    "ranging",
        "conviction":        0.65,
    }

    agent = RiskManager()
    assessment = agent.evaluate(hyp, md)
    verdict    = agent.analyze(hyp, md)

    print(f"Vote              : {verdict.vote.value}")
    print(f"Confidence        : {verdict.confidence:.3f}")
    print(f"Risk score        : {assessment.risk_score:.3f}")
    print(f"Hard veto         : {assessment.hard_veto}")
    print(f"Kelly             : {assessment.kelly_fraction:.3f} → fractional={assessment.fractional_kelly:.3f}")
    print(f"Rec size          : {assessment.recommended_size_pct:.2%} NAV")
    print(f"VaR95/99 (hist)   : {assessment.var_95_hist:.2%} / {assessment.var_99_hist:.2%}")
    print(f"CVaR95/99         : {assessment.cvar_95:.2%} / {assessment.cvar_99:.2%}")
    print(f"EVT CVaR99        : {assessment.evt_cvar_99:.2%} (ξ={assessment.gpd_shape:.3f})")
    print(f"Tail risk         : {assessment.tail_risk.value}")
    print(f"Expected MDD      : {assessment.expected_max_drawdown:.2%}")
    print(f"Liquidity risk    : {assessment.liquidity_risk.value} ({assessment.adv_fraction:.2%} ADV)")
    print(f"Diversif. ratio   : {assessment.diversification_ratio:.3f}")
    print(f"Conviction adj.   : {assessment.conviction_adjusted:.3f}")
    if assessment.warnings:
        print("Warnings:")
        for w in assessment.warnings:
            print(f"  - {w}")
