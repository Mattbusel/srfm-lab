"""
hypothesis_emitter.py
=====================
Convert Bayesian parameter drift findings into structured IAE hypotheses.

The IAE hypothesis system expects structured records with a *delta* dict
(parameter changes to test), a *rationale*, and a *confidence* score.
This module bridges the gap between raw posterior estimates and that
expected format.

Confidence scoring
------------------
Confidence is derived from the posterior credible interval width relative
to the prior interval width:

    confidence = 1 - (CI_post_width / CI_prior_width)

A narrow posterior (small CI_post_width) means the data have been
informative and we can assign high confidence to the recommendation.
Confidence is clamped to [0.05, 0.95] to avoid degenerate extremes.

Hypothesis types
----------------
1. **PARAM_DRIFT**
   A single parameter has drifted significantly from its prior.
   Delta: {param_name: posterior_mean}.

2. **WIN_RATE_DETERIORATION**
   The posterior win-rate proxy suggests holding longer would help.
   Delta: {min_hold_bars: new_value}.

3. **VOL_REGIME_SHIFT**
   The GARCH vol posterior has drifted.
   Delta: {garch_target_vol: new_value}.

4. **REGIME_CHANGE**
   A DriftAlert has been received -- recommend full re-estimation.
   Delta: {} (no parameter change recommended until re-estimation).

All hypotheses are formatted to be directly insertable into the IAE
hypothesis database via the bus/queue_manager interface.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .posterior import PosteriorEstimate, ParameterPosteriors
from .priors import build_default_priors
from .drift_monitor import DriftAlert
from .updater import DriftFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IAEHypothesis dataclass
# ---------------------------------------------------------------------------

@dataclass
class IAEHypothesis:
    """
    A structured hypothesis ready for insertion into the IAE system.

    Attributes
    ----------
    hypothesis_id  : unique UUID string.
    hypothesis_type: e.g. "PARAM_DRIFT", "WIN_RATE_DETERIORATION".
    title          : short human-readable title (< 80 chars).
    rationale      : detailed explanation of why this hypothesis was raised.
    delta          : dict of {param_name: new_value} to test.
    confidence     : float in [0, 1].
    source         : always "bayesian_updater".
    tags           : list of string tags for filtering.
    """

    hypothesis_id:   str
    hypothesis_type: str
    title:           str
    rationale:       str
    delta:           Dict[str, float]
    confidence:      float
    source:          str = "bayesian_updater"
    tags:            List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":             self.hypothesis_id,
            "type":           self.hypothesis_type,
            "title":          self.title,
            "rationale":      self.rationale,
            "delta":          self.delta,
            "confidence":     round(self.confidence, 4),
            "source":         self.source,
            "tags":           self.tags,
        }


# ---------------------------------------------------------------------------
# HypothesisEmitter
# ---------------------------------------------------------------------------

class HypothesisEmitter:
    """
    Converts posterior estimates and drift flags into IAE hypotheses.

    Parameters
    ----------
    priors          : dict of prior objects (for CI width comparison).
    min_confidence  : minimum confidence for a hypothesis to be emitted
                      (hypotheses below this threshold are suppressed).
    """

    # Minimum trades required before emitting high-confidence hypotheses
    MIN_TRADES_FOR_HIGH_CONFIDENCE = 50

    def __init__(
        self,
        priors: Optional[dict] = None,
        min_confidence: float = 0.20,
    ):
        self.priors         = priors or build_default_priors()
        self.min_confidence = min_confidence
        self._emitted:      List[IAEHypothesis] = []

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def emit_from_drift_flags(
        self,
        flags: List[DriftFlag],
        posteriors: Optional[ParameterPosteriors] = None,
        n_trades: int = 0,
    ) -> List[IAEHypothesis]:
        """
        Generate hypotheses from a list of DriftFlag objects.

        Parameters
        ----------
        flags     : list from BayesianUpdater.detect_drift().
        posteriors: current posteriors (used for CI confidence scoring).
        n_trades  : total trades seen (used for confidence calibration).

        Returns
        -------
        List of IAEHypothesis objects.
        """
        hypotheses: List[IAEHypothesis] = []

        for flag in flags:
            hyp = self._flag_to_hypothesis(flag, posteriors, n_trades)
            if hyp and hyp.confidence >= self.min_confidence:
                hypotheses.append(hyp)
                self._emitted.append(hyp)
                logger.info(
                    "Emitted hypothesis [%s] %s (confidence=%.2f)",
                    hyp.hypothesis_type, hyp.title, hyp.confidence,
                )

        return hypotheses

    def emit_from_regime_alert(
        self,
        alert: DriftAlert,
        n_trades: int = 0,
    ) -> IAEHypothesis:
        """
        Generate a REGIME_CHANGE hypothesis from a DriftAlert.

        Parameters
        ----------
        alert    : DriftAlert from DriftMonitor.
        n_trades : total trades seen for context.

        Returns
        -------
        IAEHypothesis with type "REGIME_CHANGE".
        """
        severity_conf = {"CRITICAL": 0.90, "WARNING": 0.65}.get(alert.severity, 0.50)
        confidence    = min(severity_conf, 0.95)

        rationale_parts = [alert.message]
        if alert.ks_pvalue is not None:
            rationale_parts.append(
                f"KS p-value={alert.ks_pvalue:.4f} is below threshold {self.priors.get('ks_alpha', 0.01)}."
            )
        rationale_parts.append(
            "All Bayesian posteriors should be reset to priors and "
            "parameter estimation should restart using only data from "
            "the new regime."
        )

        hyp = IAEHypothesis(
            hypothesis_id=str(uuid.uuid4()),
            hypothesis_type="REGIME_CHANGE",
            title=f"Regime change detected ({alert.alert_type}) -- reset posteriors",
            rationale=" ".join(rationale_parts),
            delta={},
            confidence=confidence,
            tags=["regime_change", alert.alert_type.lower(), alert.severity.lower()],
        )
        self._emitted.append(hyp)
        logger.warning("Emitted REGIME_CHANGE hypothesis (confidence=%.2f)", confidence)
        return hyp

    def emit_from_posteriors(
        self,
        posteriors: ParameterPosteriors,
        n_trades: int = 0,
    ) -> List[IAEHypothesis]:
        """
        Scan all posteriors and emit hypotheses for notable deviations.

        Unlike drift-flag-based emission (which requires a 2-sigma threshold),
        this method emits for any posterior that has meaningfully narrowed
        relative to the prior, indicating the data are informative.

        Parameters
        ----------
        posteriors : current ParameterPosteriors.
        n_trades   : total trades seen.

        Returns
        -------
        List of IAEHypothesis objects.
        """
        hypotheses: List[IAEHypothesis] = []

        for name, est in posteriors.estimates.items():
            prior = self.priors.get(name)
            if prior is None:
                continue

            conf = self._compute_confidence(est, prior)
            if conf < self.min_confidence:
                continue

            hyp = self._posterior_to_hypothesis(name, est, prior, conf, n_trades)
            if hyp:
                hypotheses.append(hyp)
                self._emitted.append(hyp)

        return hypotheses

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _flag_to_hypothesis(
        self,
        flag: DriftFlag,
        posteriors: Optional[ParameterPosteriors],
        n_trades: int,
    ) -> Optional[IAEHypothesis]:
        """Build a hypothesis for a single DriftFlag."""
        prior = self.priors.get(flag.param_name)
        if prior is None:
            return None

        # Confidence from posterior CI width if available
        if posteriors and flag.param_name in posteriors:
            est  = posteriors[flag.param_name]
            conf = self._compute_confidence(est, prior)
        else:
            # Fall back to severity heuristic
            conf = min(0.50 + 0.10 * abs(flag.z_score), 0.90)

        # Scale confidence down for small trade counts
        if n_trades < self.MIN_TRADES_FOR_HIGH_CONFIDENCE:
            conf *= (n_trades / self.MIN_TRADES_FOR_HIGH_CONFIDENCE)

        conf = max(0.05, min(conf, 0.95))

        # Choose hypothesis type and message
        hyp_type, title, rationale = self._build_drift_narrative(flag)

        return IAEHypothesis(
            hypothesis_id=str(uuid.uuid4()),
            hypothesis_type=hyp_type,
            title=title,
            rationale=rationale,
            delta={flag.param_name: flag.posterior_mean},
            confidence=conf,
            tags=["param_drift", flag.param_name, flag.severity],
        )

    def _build_drift_narrative(
        self, flag: DriftFlag
    ):
        """Return (hyp_type, title, rationale) for a DriftFlag."""
        name   = flag.param_name
        prior  = round(flag.prior_mean, 5)
        post   = round(flag.posterior_mean, 5)
        z      = round(flag.z_score, 2)
        dir_   = flag.direction

        # --- min_hold_bars ---
        if name == "min_hold_bars":
            action = "raising" if dir_ == "up" else "lowering"
            return (
                "WIN_RATE_DETERIORATION" if dir_ == "up" else "PARAM_DRIFT",
                f"min_hold_bars posterior drifted {dir_}: {prior:.1f} -> {post:.1f} bars",
                (
                    f"Bayesian posterior for min_hold_bars has shifted {dir_} by {z} sigma "
                    f"from the prior mean of {prior:.1f} bars to {post:.1f} bars. "
                    f"This suggests the market is rewarding {'longer' if dir_=='up' else 'shorter'} "
                    f"hold periods. Hypothesis: test {action} min_hold_bars to {post:.1f}."
                ),
            )

        # --- garch_target_vol ---
        if name == "garch_target_vol":
            regime = "lower-vol" if dir_ == "down" else "higher-vol"
            return (
                "VOL_REGIME_SHIFT",
                f"GARCH vol target drifted {dir_}: {prior:.3f} -> {post:.3f}",
                (
                    f"GARCH target vol posterior has moved from {prior:.3f} to {post:.3f} "
                    f"({z:+.1f} sigma). Strategy may be entering a {regime} regime. "
                    f"Recommend adjusting garch_target_vol to {post:.3f} to right-size "
                    f"position volatility."
                ),
            )

        # --- stale_15m_move ---
        if name == "stale_15m_move":
            return (
                "PARAM_DRIFT",
                f"stale_15m_move threshold drifted {dir_}: {prior:.5f} -> {post:.5f}",
                (
                    f"The staleness threshold for 15m moves has shifted from {prior:.5f} "
                    f"to {post:.5f} ({z:+.1f} sigma). "
                    f"{'More' if dir_=='up' else 'Fewer'} signals are being accepted at the "
                    f"current setting, suggesting the threshold should be adjusted."
                ),
            )

        # --- winner_protection_pct ---
        if name == "winner_protection_pct":
            return (
                "PARAM_DRIFT",
                f"winner_protection_pct drifted {dir_}: {prior:.5f} -> {post:.5f}",
                (
                    f"Winner protection level posterior has drifted {dir_} from {prior:.5f} "
                    f"to {post:.5f} ({z:+.1f} sigma). "
                    f"{'Tighten' if dir_=='up' else 'Loosen'} the protection threshold to "
                    f"lock in {'more' if dir_=='up' else 'fewer'} profitable exits."
                ),
            )

        # --- hour_boost_multiplier ---
        if name == "hour_boost_multiplier":
            return (
                "PARAM_DRIFT",
                f"hour_boost_multiplier drifted {dir_}: {prior:.3f} -> {post:.3f}",
                (
                    f"Hour boost multiplier has shifted {dir_} from {prior:.3f} to {post:.3f} "
                    f"({z:+.1f} sigma). "
                    f"Peak-hour edge appears to be {'strengthening' if dir_=='up' else 'weakening'}."
                ),
            )

        # --- generic fallback ---
        return (
            "PARAM_DRIFT",
            f"{name} drifted {dir_}: {prior} -> {post}",
            (
                f"Parameter {name} posterior has moved from {prior} to {post} "
                f"({z:+.1f} sigma from prior). Consider testing the updated value."
            ),
        )

    def _posterior_to_hypothesis(
        self,
        name: str,
        est: PosteriorEstimate,
        prior,
        conf: float,
        n_trades: int,
    ) -> Optional[IAEHypothesis]:
        """
        Build a hypothesis directly from a PosteriorEstimate when
        the posterior mean differs meaningfully from the prior mean.
        """
        delta_frac = abs(est.mean - prior.mean) / max(prior.std, 1e-10)
        if delta_frac < 0.5:
            return None  # Not enough deviation to be interesting

        ci_lo, ci_hi = est.credible_interval_95
        return IAEHypothesis(
            hypothesis_id=str(uuid.uuid4()),
            hypothesis_type="POSTERIOR_UPDATE",
            title=f"{name} posterior estimate: {est.mean:.4f} (CI: [{ci_lo:.4f}, {ci_hi:.4f}])",
            rationale=(
                f"After {n_trades} live trades, the Bayesian posterior for {name} "
                f"has settled at {est.mean:.4f} (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]). "
                f"The prior mean was {prior.mean:.4f}. "
                f"Posterior std={est.std:.4f} vs prior std={prior.std:.4f}."
            ),
            delta={name: est.mean},
            confidence=conf,
            tags=["posterior_update", name],
        )

    def _compute_confidence(self, est: PosteriorEstimate, prior) -> float:
        """
        Confidence = 1 - (posterior CI width / prior CI width).

        Clamped to [0.05, 0.95].
        """
        prior_ci  = prior.credible_interval()
        prior_w   = max(prior_ci[1] - prior_ci[0], 1e-10)

        post_lo, post_hi = est.credible_interval_95
        post_w           = max(post_hi - post_lo, 1e-10)

        raw_conf = 1.0 - (post_w / prior_w)
        return max(0.05, min(raw_conf, 0.95))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def emitted_hypotheses(self) -> List[IAEHypothesis]:
        """Return all hypotheses ever emitted in this session."""
        return list(self._emitted)

    def clear_history(self) -> None:
        """Clear the emitted hypothesis history."""
        self._emitted.clear()
