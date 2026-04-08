"""
debate-system/agents/macro_analyst.py

MacroAnalyst — evaluates trading hypotheses through a macro lens.

Macro factors analysed:
  - Yield curve (2s10s spread, 3m10y spread)
  - Credit spreads (IG, HY)
  - DXY (US dollar index)
  - Commodities complex
  - Risk-on / risk-off / transition regime
  - Rate regime (rising / falling / flat)
  - Dollar regime (strong / weak)
  - Liquidity conditions (tight / loose / normal)
  - Inter-market correlations

evaluate() returns a MacroAnalysis dataclass with a macro score, regime label,
factor-level breakdown, and a list of human-readable warnings.

The analyze() method wraps evaluate() into an AnalystVerdict for the DebateChamber.

Expected market_data keys (passed to both methods)
---------------------------------------------------
yield_2y              : float  — 2-year Treasury yield (%)
yield_10y             : float  — 10-year Treasury yield (%)
yield_3m              : float  — 3-month T-bill yield (%)
ig_spread             : float  — IG credit spread (bps)
hy_spread             : float  — HY credit spread (bps)
dxy                   : float  — Dollar Index level
dxy_history           : np.ndarray, optional
asset_prices          : np.ndarray, optional
commodity_index       : float, optional
hypothesis_direction  : str  — "long" | "short" | "neutral"
asset_class           : str  — "equity" | "crypto" | "fx" | "commodity" | "rates"
macro_score_history   : np.ndarray, optional

Historical baselines (all optional):
hist_ig_spreads, hist_hy_spreads, hist_2s10s, hist_3m10y, hist_dxy : np.ndarray
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from debate_system.agents.base_agent import BaseAnalyst, AnalystVerdict, Vote
from hypothesis.types import Hypothesis


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MacroRegime(str, Enum):
    RISK_ON    = "risk_on"
    RISK_OFF   = "risk_off"
    TRANSITION = "transition"


class RateRegime(str, Enum):
    RISING  = "rising"
    FALLING = "falling"
    FLAT    = "flat"


class DollarRegime(str, Enum):
    STRONG  = "strong"
    WEAK    = "weak"
    NEUTRAL = "neutral"


class LiquidityCondition(str, Enum):
    LOOSE  = "loose"
    NORMAL = "normal"
    TIGHT  = "tight"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MacroAnalysis:
    """
    Structured macro evaluation result.

    macro_score  : -1.0 (strongly bearish for hypothesis) to +1.0 (strongly supportive)
    regime       : RISK_ON / RISK_OFF / TRANSITION
    factor_scores: per-factor breakdown, each in [-1, +1]
    warnings     : list of human-readable macro risks
    """

    macro_score: float                       # -1 to +1
    regime: MacroRegime
    rate_regime: RateRegime
    dollar_regime: DollarRegime
    liquidity: LiquidityCondition

    # Per-factor signals, each [-1, +1]
    factor_scores: Dict[str, float]

    # Raw macro state
    yield_curve_2s10s: float                 # bps
    yield_curve_3m10y: float                 # bps
    ig_spread: float                         # bps
    hy_spread: float                         # bps
    dxy_level: float
    dxy_trend: float                         # normalized slope

    # Inter-market context
    asset_macro_correlation: float           # corr(asset_rets, macro_score_changes)
    correlation_regime_match: bool           # True if correlation aligns with hypothesis

    # Conviction scaling
    macro_conviction_multiplier: float       # 0.5 (headwind) to 1.5 (tailwind)
    directional_alignment: float             # how much macro supports hypothesis direction

    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MacroAnalyst
# ---------------------------------------------------------------------------

class MacroAnalyst(BaseAnalyst):
    """
    Evaluates hypotheses through a macro lens.

    Inherits Bayesian credibility tracking from BaseAnalyst so that
    the DebateChamber can weight votes by empirical track record.
    """

    # Factor weights (sum = 1.0)
    FACTOR_WEIGHTS: Dict[str, float] = {
        "yield_curve":    0.25,
        "credit_spreads": 0.25,
        "dollar":         0.20,
        "liquidity":      0.15,
        "commodities":    0.15,
    }

    # Asset-class sensitivity to macro factors
    # +1.0 = fully positively exposed, -1.0 = fully inversely exposed
    ASSET_MACRO_SENSITIVITY: Dict[str, Dict[str, float]] = {
        "equity":    {"yield_curve":  0.8, "credit_spreads": -0.9, "dollar": -0.3, "liquidity":  0.7, "commodities":  0.4},
        "crypto":    {"yield_curve":  0.6, "credit_spreads": -0.8, "dollar": -0.7, "liquidity":  0.9, "commodities":  0.3},
        "fx":        {"yield_curve":  0.5, "credit_spreads": -0.4, "dollar":  1.0, "liquidity":  0.5, "commodities":  0.2},
        "commodity": {"yield_curve":  0.3, "credit_spreads": -0.5, "dollar": -0.8, "liquidity":  0.4, "commodities":  1.0},
        "rates":     {"yield_curve":  1.0, "credit_spreads": -0.6, "dollar":  0.2, "liquidity":  0.8, "commodities": -0.3},
    }

    # Credit spread regime thresholds (bps)
    IG_TIGHT = 80
    IG_WIDE  = 180
    HY_TIGHT = 300
    HY_WIDE  = 600

    def __init__(self) -> None:
        super().__init__(
            name="MacroAnalyst",
            specialization=(
                "Yield curve analysis, credit spread regimes, DXY dynamics, "
                "liquidity conditions, rate regimes, inter-market correlations, "
                "macro risk-on/off regime classification"
            ),
        )

    # ------------------------------------------------------------------
    # Primary rich output: MacroAnalysis
    # ------------------------------------------------------------------

    def evaluate(
        self,
        hypothesis: Hypothesis,
        market_data: Dict[str, Any],
    ) -> MacroAnalysis:
        """
        Full macro evaluation returning a structured MacroAnalysis.

        This is the rich entry point.  analyze() wraps it into an AnalystVerdict
        for the debate chamber.
        """
        y2   = float(market_data.get("yield_2y",   3.0))
        y10  = float(market_data.get("yield_10y",  4.0))
        y3m  = float(market_data.get("yield_3m",   5.0))
        ig   = float(market_data.get("ig_spread",  120.0))
        hy   = float(market_data.get("hy_spread",  350.0))
        dxy  = float(market_data.get("dxy",        103.0))

        direction   = str(market_data.get("hypothesis_direction", "long"))
        asset_class = str(market_data.get("asset_class", "crypto")).lower()
        if asset_class not in self.ASSET_MACRO_SENSITIVITY:
            asset_class = "crypto"

        # Optional data
        dxy_history  = market_data.get("dxy_history")
        asset_prices = market_data.get("asset_prices")
        macro_hist   = market_data.get("macro_score_history")
        hist_ig      = market_data.get("hist_ig_spreads")
        hist_hy      = market_data.get("hist_hy_spreads")
        hist_2s10s   = market_data.get("hist_2s10s")
        hist_3m10y   = market_data.get("hist_3m10y")
        commodity_idx = market_data.get("commodity_index")

        warnings: List[str] = []

        # --- yield curve ---
        spread_2s10s = (y10 - y2) * 100.0
        spread_3m10y = (y10 - y3m) * 100.0
        yc_signal, yc_warns = self._yield_curve_signal(
            spread_2s10s, spread_3m10y, hist_2s10s, hist_3m10y
        )
        warnings.extend(yc_warns)

        # --- credit spreads ---
        credit_signal, credit_warns = self._credit_signal(ig, hy, hist_ig, hist_hy)
        warnings.extend(credit_warns)

        # --- dollar ---
        dxy_trend   = self._compute_dxy_trend(dxy_history, dxy)
        dollar_sig, dollar_regime = self._dollar_signal(dxy, dxy_trend)
        if dollar_regime == DollarRegime.STRONG and asset_class == "crypto":
            warnings.append("Strong dollar regime — historically a headwind for crypto assets")

        # --- rate regime ---
        rate_regime = self._rate_regime(y2, y10, hist_2s10s)
        if rate_regime == RateRegime.RISING and asset_class in ("crypto", "equity"):
            warnings.append(f"Rising rate regime — elevated discount rates pressure {asset_class}")

        # --- liquidity ---
        liq_signal, liq_condition = self._liquidity_signal(ig, hy, y3m)
        if liq_condition == LiquidityCondition.TIGHT:
            warnings.append(f"Tight liquidity (IG={ig:.0f}bps, HY={hy:.0f}bps, y3m={y3m:.2f}%)")

        # --- commodities ---
        commodity_sig = self._commodity_signal(commodity_idx, asset_class)

        # --- asset-class adjusted factor scores ---
        sens = self.ASSET_MACRO_SENSITIVITY[asset_class]
        factor_scores: Dict[str, float] = {
            "yield_curve":    float(np.clip(yc_signal     * sens["yield_curve"],    -1.0, 1.0)),
            "credit_spreads": float(np.clip(credit_signal * sens["credit_spreads"], -1.0, 1.0)),
            "dollar":         float(np.clip(dollar_sig    * sens["dollar"],         -1.0, 1.0)),
            "liquidity":      float(np.clip(liq_signal    * sens["liquidity"],      -1.0, 1.0)),
            "commodities":    float(np.clip(commodity_sig * sens["commodities"],    -1.0, 1.0)),
        }

        macro_score = float(np.clip(
            sum(factor_scores[k] * self.FACTOR_WEIGHTS[k] for k in self.FACTOR_WEIGHTS),
            -1.0, 1.0,
        ))

        regime = self._classify_regime(macro_score, credit_signal, yc_signal)

        # --- inter-market correlation ---
        asset_macro_corr = 0.0
        if asset_prices is not None and macro_hist is not None:
            asset_macro_corr = self._intermarket_correlation(
                np.asarray(asset_prices, dtype=np.float64),
                np.asarray(macro_hist, dtype=np.float64),
            )

        # --- directional alignment: how much does macro support this trade? ---
        d_sign = 1.0 if direction == "long" else -1.0
        directional_alignment = float(np.clip(macro_score * d_sign, -1.0, 1.0))
        conviction_mult = float(np.clip(1.0 + directional_alignment * 0.5, 0.5, 1.5))

        corr_match = (
            (asset_macro_corr > 0.3 and direction == "long") or
            (asset_macro_corr < -0.3 and direction == "short")
        )

        return MacroAnalysis(
            macro_score=macro_score,
            regime=regime,
            rate_regime=rate_regime,
            dollar_regime=dollar_regime,
            liquidity=liq_condition,
            factor_scores=factor_scores,
            yield_curve_2s10s=spread_2s10s,
            yield_curve_3m10y=spread_3m10y,
            ig_spread=ig,
            hy_spread=hy,
            dxy_level=dxy,
            dxy_trend=dxy_trend,
            asset_macro_correlation=asset_macro_corr,
            correlation_regime_match=corr_match,
            macro_conviction_multiplier=conviction_mult,
            directional_alignment=directional_alignment,
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
        analysis = self.evaluate(hypothesis, market_data)
        direction = str(market_data.get("hypothesis_direction", "long"))
        alignment = analysis.directional_alignment

        if alignment > 0.3:
            vote = Vote.FOR
            confidence = float(np.clip(0.5 + alignment * 0.5, 0.0, 1.0))
            reasoning = (
                f"Macro backdrop is supportive. "
                f"Regime: {analysis.regime.value}, composite macro score={analysis.macro_score:+.3f}. "
                f"Tailwinds: {self._top_factors(analysis.factor_scores, direction, top_n=2)}. "
                f"Conviction multiplier: {analysis.macro_conviction_multiplier:.2f}x."
            )
        elif alignment < -0.3:
            vote = Vote.AGAINST
            confidence = float(np.clip(0.5 - alignment * 0.5, 0.0, 1.0))
            reasoning = (
                f"Macro backdrop is a headwind. "
                f"Regime: {analysis.regime.value}, macro score={analysis.macro_score:+.3f}. "
                f"Headwinds: {self._top_factors(analysis.factor_scores, direction, reverse=True, top_n=2)}."
            )
        else:
            vote = Vote.ABSTAIN
            confidence = 0.35
            reasoning = (
                f"Macro signals are mixed / neutral. "
                f"Regime: {analysis.regime.value} (transition), score={analysis.macro_score:+.3f}. "
                f"Insufficient macro conviction to vote FOR or AGAINST."
            )

        return self._make_verdict(
            vote=vote,
            confidence=confidence,
            reasoning=reasoning,
            key_concerns=analysis.warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _yield_curve_signal(
        self,
        spread_2s10s: float,
        spread_3m10y: float,
        hist_2s10s: Optional[np.ndarray],
        hist_3m10y: Optional[np.ndarray],
    ) -> tuple[float, List[str]]:
        """
        Positive = steep curve (risk-on / early cycle).
        Negative = flat / inverted (risk-off / late cycle).
        """
        warnings: List[str] = []
        if spread_3m10y < -50:
            warnings.append(f"3m10y deeply inverted ({spread_3m10y:.0f}bps) — recession signal")
        elif spread_3m10y < 0:
            warnings.append(f"3m10y inverted ({spread_3m10y:.0f}bps)")
        if spread_2s10s < -30:
            warnings.append(f"2s10s inverted ({spread_2s10s:.0f}bps)")

        composite = (spread_2s10s + spread_3m10y) / 2.0
        if hist_2s10s is not None and len(hist_2s10s) > 20:
            hist = np.concatenate([hist_2s10s, hist_3m10y if hist_3m10y is not None else hist_2s10s])
            mu  = float(np.mean(hist))
            sig = float(np.std(hist)) + 1e-9
            signal = float(np.clip((composite - mu) / (2.0 * sig), -1.0, 1.0))
        else:
            signal = float(np.clip(composite / 200.0, -1.0, 1.0))
        return signal, warnings

    def _credit_signal(
        self,
        ig: float,
        hy: float,
        hist_ig: Optional[np.ndarray],
        hist_hy: Optional[np.ndarray],
    ) -> tuple[float, List[str]]:
        """
        Positive = tight spreads (risk-on / loose financial conditions).
        Negative = wide spreads (risk-off / credit stress).
        """
        warnings: List[str] = []
        if hy > self.HY_WIDE:
            warnings.append(f"HY spreads extremely wide ({hy:.0f}bps) — credit crisis signal")
        elif hy > self.HY_WIDE * 0.7:
            warnings.append(f"HY spreads elevated ({hy:.0f}bps)")
        if ig > self.IG_WIDE:
            warnings.append(f"IG spreads wide ({ig:.0f}bps)")

        if hist_ig is not None and len(hist_ig) > 20:
            ig_pct = float(sp_stats.percentileofscore(hist_ig, ig)) / 100.0
        else:
            ig_pct = float(np.clip(ig / self.IG_WIDE, 0.0, 1.0))

        if hist_hy is not None and len(hist_hy) > 20:
            hy_pct = float(sp_stats.percentileofscore(hist_hy, hy)) / 100.0
        else:
            hy_pct = float(np.clip(hy / self.HY_WIDE, 0.0, 1.0))

        credit_risk = 0.4 * ig_pct + 0.6 * hy_pct  # high = wide = bearish
        signal = float(np.clip(1.0 - 2.0 * credit_risk, -1.0, 1.0))
        return signal, warnings

    def _compute_dxy_trend(
        self,
        dxy_history: Optional[Any],
        current_dxy: float,
    ) -> float:
        """Normalized 20-bar DXY trend. Positive = strengthening dollar."""
        if dxy_history is None:
            return 0.0
        arr = np.asarray(dxy_history, dtype=np.float64)
        if len(arr) < 10:
            return 0.0
        recent = arr[-20:]
        x = np.arange(len(recent), dtype=np.float64)
        slope, *_ = sp_stats.linregress(x, recent)
        dxy_std = float(np.std(recent)) + 1e-9
        return float(np.clip(slope * 10.0 / dxy_std, -1.0, 1.0))

    def _dollar_signal(
        self,
        dxy: float,
        dxy_trend: float,
    ) -> tuple[float, DollarRegime]:
        """
        Positive = dollar weakening (good for risk assets).
        Negative = dollar strengthening (headwind for risk assets).
        DXY long-run mean ~100.
        """
        level_sig = float(np.clip(-(dxy - 100.0) / 15.0, -1.0, 1.0))
        combined  = float(np.clip(0.4 * level_sig + 0.6 * (-dxy_trend), -1.0, 1.0))

        if combined < -0.3:
            regime = DollarRegime.STRONG
        elif combined > 0.3:
            regime = DollarRegime.WEAK
        else:
            regime = DollarRegime.NEUTRAL
        return combined, regime

    def _rate_regime(
        self,
        y2: float,
        y10: float,
        hist_2s10s: Optional[np.ndarray],
    ) -> RateRegime:
        """Classify rate regime from yield level and recent trend."""
        if hist_2s10s is not None and len(hist_2s10s) >= 10:
            slope = float(np.polyfit(np.arange(10), hist_2s10s[-10:], 1)[0])
            if slope > 5:
                return RateRegime.RISING
            elif slope < -5:
                return RateRegime.FALLING
        if y2 > 5.0:
            return RateRegime.RISING
        elif y2 < 2.5:
            return RateRegime.FALLING
        return RateRegime.FLAT

    def _liquidity_signal(
        self,
        ig: float,
        hy: float,
        y3m: float,
    ) -> tuple[float, LiquidityCondition]:
        """
        Positive = loose liquidity (low spreads, low short rates).
        Negative = tight liquidity (wide spreads, high short rates).
        """
        credit_stress = ig > self.IG_WIDE or hy > self.HY_WIDE
        rate_elevated = y3m > 5.0

        if credit_stress and rate_elevated:
            return -1.0, LiquidityCondition.TIGHT
        elif credit_stress or rate_elevated:
            return -0.4, LiquidityCondition.TIGHT
        elif ig < self.IG_TIGHT and hy < self.HY_TIGHT:
            return 1.0, LiquidityCondition.LOOSE
        return 0.0, LiquidityCondition.NORMAL

    def _commodity_signal(
        self,
        commodity_idx: Optional[float],
        asset_class: str,
    ) -> float:
        """Rising commodities proxy for inflationary / reflationary environment."""
        if commodity_idx is None:
            return 0.0
        return float(np.clip((commodity_idx - 275.0) / 75.0, -1.0, 1.0))

    def _classify_regime(
        self,
        macro_score: float,
        credit_signal: float,
        yc_signal: float,
    ) -> MacroRegime:
        if macro_score > 0.25 and credit_signal > 0.0 and yc_signal > -0.2:
            return MacroRegime.RISK_ON
        elif macro_score < -0.25 or (credit_signal < -0.4 and yc_signal < -0.3):
            return MacroRegime.RISK_OFF
        return MacroRegime.TRANSITION

    def _intermarket_correlation(
        self,
        asset_prices: np.ndarray,
        macro_scores: np.ndarray,
    ) -> float:
        """Pearson correlation between asset log-returns and macro score changes."""
        n = min(len(asset_prices) - 1, len(macro_scores) - 1, 60)
        if n < 10:
            return 0.0
        a_rets = np.diff(np.log(np.abs(asset_prices[-n - 1:]) + 1e-12))
        m_chg  = np.diff(macro_scores[-n - 1:])
        min_len = min(len(a_rets), len(m_chg))
        a_rets  = a_rets[-min_len:]
        m_chg   = m_chg[-min_len:]
        if np.std(a_rets) < 1e-9 or np.std(m_chg) < 1e-9:
            return 0.0
        corr, _ = sp_stats.pearsonr(a_rets, m_chg)
        return float(np.clip(corr, -1.0, 1.0))

    def _top_factors(
        self,
        factor_scores: Dict[str, float],
        direction: str,
        reverse: bool = False,
        top_n: int = 2,
    ) -> str:
        """Human-readable list of the top supporting or opposing factors."""
        d_sign = 1.0 if direction == "long" else -1.0
        sign   = -1 if reverse else 1
        ranked = sorted(factor_scores.items(), key=lambda x: sign * d_sign * x[1], reverse=True)
        return ", ".join(f"{k}={v:+.2f}" for k, v in ranked[:top_n])


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
        predicted_sharpe_delta=0.2,
        predicted_dd_delta=-0.05,
        novelty_score=0.6,
        priority_rank=1,
        status=HypothesisStatus.PENDING,
        created_at="2026-01-01T00:00:00+00:00",
        description="Long BTC on rate pivot signal",
    )

    analyst = MacroAnalyst()
    rng = np.random.default_rng(7)
    md: Dict[str, Any] = {
        "yield_2y":  4.8,
        "yield_10y": 4.2,
        "yield_3m":  5.2,
        "ig_spread": 110.0,
        "hy_spread": 380.0,
        "dxy":       104.5,
        "dxy_history": 104.5 + np.cumsum(rng.normal(0, 0.2, 60)),
        "asset_prices": np.cumprod(1 + rng.normal(0.001, 0.02, 60)) * 50000,
        "hypothesis_direction": "long",
        "asset_class": "crypto",
        "hist_ig_spreads": rng.normal(120, 25, 500),
        "hist_hy_spreads": rng.normal(350, 80, 500),
    }

    verdict  = analyst.analyze(hyp, md)
    analysis = analyst.evaluate(hyp, md)

    print(f"Vote       : {verdict.vote.value}")
    print(f"Confidence : {verdict.confidence:.3f}")
    print(f"Regime     : {analysis.regime.value}")
    print(f"Macro score: {analysis.macro_score:+.3f}")
    print(f"Alignment  : {analysis.directional_alignment:+.3f}")
    print(f"Conv. mult : {analysis.macro_conviction_multiplier:.3f}x")
    print("Factors    :", {k: f"{v:+.3f}" for k, v in analysis.factor_scores.items()})
    if analysis.warnings:
        print("Warnings:")
        for w in analysis.warnings:
            print(f"  - {w}")
    print(f"Reasoning  : {verdict.reasoning}")
