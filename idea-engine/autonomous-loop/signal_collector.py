"""
idea-engine/autonomous-loop/signal_collector.py

SignalCollector: collects signals from all IAE data sources in parallel.

Uses asyncio.gather so every data source runs concurrently. Each call is
capped at 30 seconds. Results are cached for 15 minutes so rapid
re-execution does not hammer external APIs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_FETCH_TIMEOUT = 30.0          # seconds per data source
_CACHE_TTL = 900.0             # 15 minutes


@dataclass
class SystemSignal:
    """
    Composite view of all IAE signal layers for a single cycle.

    Scores are normalised to [-1, 1] where -1 is maximally bearish/risky
    and +1 is maximally bullish/safe.
    """

    macro_regime: str                    # e.g. "RISK_ON", "RISK_OFF", "CRISIS"
    onchain_score: float                 # composite on-chain health score
    sentiment_score: float               # fear & greed normalised
    derivatives_signal: str             # "BULLISH", "NEUTRAL", "BEARISH"
    liquidation_risk: float             # 0-1, probability of cascade
    composite_score: float              # weighted blend
    fetched_at: str                     # ISO 8601
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def is_crisis(self) -> bool:
        return self.macro_regime == "CRISIS"

    @property
    def is_extreme_fear(self) -> bool:
        return self.sentiment_score < -0.6   # fear & greed < 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_regime": self.macro_regime,
            "onchain_score": self.onchain_score,
            "sentiment_score": self.sentiment_score,
            "derivatives_signal": self.derivatives_signal,
            "liquidation_risk": self.liquidation_risk,
            "composite_score": self.composite_score,
            "fetched_at": self.fetched_at,
            "errors": self.errors,
        }


class SignalCollector:
    """
    Fetch all IAE data sources in parallel with per-source timeouts.

    Returns a SystemSignal with a composite score and per-source details.
    Cache prevents redundant network calls within a 15-minute window.
    """

    def __init__(self, cache_ttl: float = _CACHE_TTL) -> None:
        self._cache_ttl = cache_ttl
        self._cached_signal: SystemSignal | None = None
        self._cache_timestamp: float = 0.0

    async def collect(self) -> SystemSignal:
        """Collect all signals, returning cached results if still fresh."""
        now = time.monotonic()
        if self._cached_signal and (now - self._cache_timestamp) < self._cache_ttl:
            logger.debug("SignalCollector: returning cached signal (%.0fs old)",
                         now - self._cache_timestamp)
            return self._cached_signal

        signal = await self._fetch_all()
        self._cached_signal = signal
        self._cache_timestamp = time.monotonic()
        return signal

    async def _fetch_all(self) -> SystemSignal:
        """Run all data sources in parallel."""
        logger.info("SignalCollector: fetching from all sources …")
        results = await asyncio.gather(
            self._fetch_macro(),
            self._fetch_onchain(),
            self._fetch_sentiment(),
            self._fetch_derivatives(),
            self._fetch_liquidations(),
            return_exceptions=True,
        )

        macro_res, onchain_res, sentiment_res, derivatives_res, liq_res = results
        errors: dict[str, str] = {}

        # ---- macro regime ------------------------------------------------
        macro_regime = "RISK_NEUTRAL"
        if isinstance(macro_res, Exception):
            errors["macro"] = str(macro_res)
            logger.warning("Macro fetch failed: %s", macro_res)
        elif macro_res is not None:
            try:
                macro_regime = str(getattr(macro_res.regime, "value", macro_res.regime)
                                   if hasattr(macro_res, "regime") else "RISK_NEUTRAL").upper()
            except Exception as exc:
                errors["macro_parse"] = str(exc)

        # ---- on-chain score ----------------------------------------------
        onchain_score = 0.0
        if isinstance(onchain_res, Exception):
            errors["onchain"] = str(onchain_res)
            logger.warning("OnChain fetch failed: %s", onchain_res)
        elif onchain_res is not None:
            try:
                onchain_score = float(getattr(onchain_res, "composite_score", 0.0))
            except Exception as exc:
                errors["onchain_parse"] = str(exc)

        # ---- sentiment ---------------------------------------------------
        sentiment_score = 0.0
        if isinstance(sentiment_res, Exception):
            errors["sentiment"] = str(sentiment_res)
            logger.warning("Sentiment fetch failed: %s", sentiment_res)
        elif sentiment_res is not None:
            try:
                raw = float(getattr(sentiment_res, "value", 50))
                # Fear & Greed 0-100  → normalise to [-1, 1]
                sentiment_score = (raw - 50.0) / 50.0
            except Exception as exc:
                errors["sentiment_parse"] = str(exc)

        # ---- derivatives -------------------------------------------------
        derivatives_signal = "NEUTRAL"
        if isinstance(derivatives_res, Exception):
            errors["derivatives"] = str(derivatives_res)
            logger.warning("Derivatives fetch failed: %s", derivatives_res)
        elif derivatives_res is not None:
            try:
                derivatives_signal = str(
                    getattr(derivatives_res, "signal", "NEUTRAL")
                ).upper()
            except Exception as exc:
                errors["derivatives_parse"] = str(exc)

        # ---- liquidation risk --------------------------------------------
        liquidation_risk = 0.0
        if isinstance(liq_res, Exception):
            errors["liquidations"] = str(liq_res)
            logger.warning("Liquidation fetch failed: %s", liq_res)
        elif liq_res is not None:
            try:
                liquidation_risk = float(getattr(liq_res, "risk_score", 0.0))
            except Exception as exc:
                errors["liquidations_parse"] = str(exc)

        # ---- composite ---------------------------------------------------
        composite_score = self._compute_composite(
            macro_regime, onchain_score, sentiment_score,
            derivatives_signal, liquidation_risk
        )

        return SystemSignal(
            macro_regime=macro_regime,
            onchain_score=onchain_score,
            sentiment_score=sentiment_score,
            derivatives_signal=derivatives_signal,
            liquidation_risk=liquidation_risk,
            composite_score=composite_score,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Per-source fetchers — each wraps the real IAE module with a timeout
    # ------------------------------------------------------------------

    async def _fetch_macro(self) -> Any:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self._run_macro_pipeline),
            timeout=_FETCH_TIMEOUT,
        )

    def _run_macro_pipeline(self) -> Any:
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
            from macro_factor.pipeline import MacroFactorPipeline
            return MacroFactorPipeline().run()
        except Exception as exc:
            logger.debug("MacroFactorPipeline import/run error: %s", exc)
            raise

    async def _fetch_onchain(self) -> Any:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self._run_onchain_engine),
            timeout=_FETCH_TIMEOUT,
        )

    def _run_onchain_engine(self) -> Any:
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
            from onchain.composite_signal import OnChainEngine
            return OnChainEngine().run("BTC")
        except Exception as exc:
            logger.debug("OnChainEngine import/run error: %s", exc)
            raise

    async def _fetch_sentiment(self) -> Any:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self._run_fear_greed),
            timeout=_FETCH_TIMEOUT,
        )

    def _run_fear_greed(self) -> Any:
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
            from sentiment_engine.scrapers.fear_greed import FearGreedClient
            return FearGreedClient().get_current()
        except Exception as exc:
            logger.debug("FearGreedClient import/run error: %s", exc)
            raise

    async def _fetch_derivatives(self) -> Any:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self._run_futures_oi),
            timeout=_FETCH_TIMEOUT,
        )

    def _run_futures_oi(self) -> Any:
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
            from alternative_data.futures_oi import FuturesOIFetcher
            return FuturesOIFetcher().fetch_all()
        except Exception as exc:
            logger.debug("FuturesOIFetcher import/run error: %s", exc)
            raise

    async def _fetch_liquidations(self) -> Any:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, self._run_liquidations),
            timeout=_FETCH_TIMEOUT,
        )

    def _run_liquidations(self) -> Any:
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
            from alternative_data.liquidations import LiquidationSimulator
            return LiquidationSimulator().fetch_all()
        except Exception as exc:
            logger.debug("LiquidationSimulator import/run error: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def _compute_composite(
        self,
        macro_regime: str,
        onchain_score: float,
        sentiment_score: float,
        derivatives_signal: str,
        liquidation_risk: float,
    ) -> float:
        """
        Weighted blend of all signal layers.

        Returns a score in [-1, 1]:
          positive = bullish / safe conditions
          negative = bearish / risky conditions
        """
        macro_num = {"RISK_ON": 0.5, "RISK_NEUTRAL": 0.0,
                     "RISK_OFF": -0.5, "CRISIS": -1.0}.get(macro_regime, 0.0)
        deriv_num = {"BULLISH": 0.5, "NEUTRAL": 0.0, "BEARISH": -0.5}.get(
            derivatives_signal, 0.0
        )
        liq_penalty = -liquidation_risk  # high risk → drag on composite

        composite = (
            0.35 * macro_num
            + 0.25 * onchain_score
            + 0.20 * sentiment_score
            + 0.15 * deriv_num
            + 0.05 * liq_penalty
        )
        return max(-1.0, min(1.0, composite))
