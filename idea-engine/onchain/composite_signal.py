"""
onchain/composite_signal.py
────────────────────────────
On-Chain Composite Score — Aggregate all on-chain signals into a single
directional score per symbol.

Weighting scheme
────────────────
  MVRV           30% — best cycle-top/bottom indicator historically
  NVT            20% — valuation relative to network utility
  SOPR           20% — short-term seller profit/loss (capitulation detector)
  Exchange Res.  20% — supply flow direction
  Whale Flow     10% — large-holder sentiment

Output: OnChainScore in [-1, +1].
  > +0.5  → strong accumulation zone
  +0.2 to +0.5 → mild bullish on-chain conditions
  -0.2 to +0.2 → neutral
  -0.5 to -0.2 → mild distribution warning
  < -0.5  → overheated / strong distribution zone

Note: HODL Waves and Hash Rate are computed but stored as supplementary signals;
they are not included in the weighted composite to keep the weighting clean and
prevent double-counting (hash rate is only available for BTC).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .data_store import OnChainDataStore
from .metrics.mvrv import compute_mvrv, MVRVResult
from .metrics.nvt import compute_nvt, NVTResult
from .metrics.sopr import compute_sopr, SOPRResult
from .metrics.exchange_reserves import compute_exchange_reserves, ExchangeReserveResult
from .metrics.whale_tracker import compute_whale_tracker, WhaleTrackerResult
from .metrics.hodl_waves import compute_hodl_waves, HODLWaveResult
from .metrics.hash_rate import compute_hash_rate, HashRateResult

logger = logging.getLogger(__name__)

# Composite weighting (must sum to 1.0)
_WEIGHTS: Dict[str, float] = {
    "mvrv":              0.30,
    "nvt":               0.20,
    "sopr":              0.20,
    "exchange_reserves": 0.20,
    "whale":             0.10,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Minimum price bars required for a valid run
_MIN_BARS = 200

# yfinance lookback for price history
_PRICE_LOOKBACK_DAYS = 800


@dataclass
class OnChainResult:
    """Full result of a single on-chain engine run."""
    symbol: str
    composite_score: float          # [-1, +1] weighted composite
    component_signals: Dict[str, float]
    mvrv: Optional[MVRVResult]      = None
    nvt: Optional[NVTResult]        = None
    sopr: Optional[SOPRResult]      = None
    exchange_reserves: Optional[ExchangeReserveResult] = None
    whale: Optional[WhaleTrackerResult]                = None
    hodl_waves: Optional[HODLWaveResult]               = None
    hash_rate: Optional[HashRateResult]                = None
    errors: Dict[str, str]          = field(default_factory=dict)
    computed_at: str                = ""

    @property
    def regime_label(self) -> str:
        s = self.composite_score
        if s >  0.6:  return "STRONG_ACCUMULATION"
        if s >  0.2:  return "MILD_ACCUMULATION"
        if s > -0.2:  return "NEUTRAL"
        if s > -0.6:  return "MILD_DISTRIBUTION"
        return "STRONG_DISTRIBUTION"


def _fetch_price_history(symbol: str, days: int = _PRICE_LOOKBACK_DAYS) -> pd.Series:
    """Download daily close prices via yfinance."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=f"{days}d", interval="1d", auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    return hist["Close"].dropna()


class OnChainEngine:
    """Orchestrates all on-chain metrics and produces a composite signal.

    Parameters
    ----------
    db_path:
        Path to SQLite database for storing results.
    """

    def __init__(
        self,
        db_path: Path | str = "C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db",
    ) -> None:
        self.store = OnChainDataStore(db_path=db_path)
        logger.info("OnChainEngine initialised (db=%s)", db_path)

    def run(self, symbol: str = "BTC-USD") -> OnChainResult:
        """Run all on-chain metrics for the given symbol.

        Downloads price history, computes every metric, weights them into a
        composite score, persists to SQLite, and returns the full result.

        Parameters
        ----------
        symbol:
            Yahoo Finance ticker symbol.  BTC-USD is fully supported.
            ETH-USD is supported where metrics are applicable.
        """
        logger.info("OnChainEngine.run(%s) started", symbol)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # --- Fetch price history ---
        try:
            prices = _fetch_price_history(symbol)
        except Exception as exc:
            logger.error("Price fetch failed for %s: %s", symbol, exc)
            return OnChainResult(
                symbol=symbol,
                composite_score=0.0,
                component_signals={},
                errors={"price_fetch": str(exc)},
                computed_at=datetime.now(timezone.utc).isoformat(),
            )

        if len(prices) < _MIN_BARS:
            return OnChainResult(
                symbol=symbol,
                composite_score=0.0,
                component_signals={},
                errors={"insufficient_data": f"Only {len(prices)} bars available, need {_MIN_BARS}"},
                computed_at=datetime.now(timezone.utc).isoformat(),
            )

        supply = 19_700_000.0 if "BTC" in symbol.upper() else 120_000_000.0

        component_signals: Dict[str, float] = {}
        errors: Dict[str, str] = {}
        results: Dict[str, Any] = {}

        # --- MVRV ---
        try:
            mvrv = compute_mvrv(prices, circulating_supply=supply, symbol=symbol)
            component_signals["mvrv"] = mvrv.signal
            results["mvrv"] = mvrv
            self.store.upsert_metric(symbol, "mvrv", today, mvrv.signal, mvrv.mvrv_ratio, mvrv.source, _dataclass_to_dict(mvrv))
        except Exception as exc:
            logger.warning("MVRV failed: %s", exc)
            errors["mvrv"] = str(exc)
            component_signals["mvrv"] = 0.0

        # --- NVT ---
        try:
            nvt = compute_nvt(prices, circulating_supply=supply, symbol=symbol)
            component_signals["nvt"] = nvt.signal
            results["nvt"] = nvt
            self.store.upsert_metric(symbol, "nvt", today, nvt.signal, nvt.nvt_signal, nvt.source, _dataclass_to_dict(nvt))
        except Exception as exc:
            logger.warning("NVT failed: %s", exc)
            errors["nvt"] = str(exc)
            component_signals["nvt"] = 0.0

        # --- SOPR ---
        try:
            sopr = compute_sopr(prices, symbol=symbol)
            component_signals["sopr"] = sopr.signal
            results["sopr"] = sopr
            self.store.upsert_metric(symbol, "sopr", today, sopr.signal, sopr.sopr_smooth, sopr.source, _dataclass_to_dict(sopr))
        except Exception as exc:
            logger.warning("SOPR failed: %s", exc)
            errors["sopr"] = str(exc)
            component_signals["sopr"] = 0.0

        # --- Exchange Reserves ---
        try:
            exr = compute_exchange_reserves(prices, circulating_supply=supply, symbol=symbol)
            component_signals["exchange_reserves"] = exr.signal
            results["exchange_reserves"] = exr
            self.store.upsert_metric(symbol, "exchange_reserves", today, exr.signal, exr.roc_30d, exr.source, _dataclass_to_dict(exr))
        except Exception as exc:
            logger.warning("ExchangeReserves failed: %s", exc)
            errors["exchange_reserves"] = str(exc)
            component_signals["exchange_reserves"] = 0.0

        # --- Whale Tracker ---
        try:
            whale = compute_whale_tracker(prices, symbol=symbol)
            component_signals["whale"] = whale.signal
            results["whale"] = whale
            self.store.upsert_metric(symbol, "whale", today, whale.signal, whale.net_flow_30d, "simulated", _dataclass_to_dict(whale))
        except Exception as exc:
            logger.warning("WhaleTracker failed: %s", exc)
            errors["whale"] = str(exc)
            component_signals["whale"] = 0.0

        # --- Supplementary: HODL Waves (no weight in composite) ---
        hodl_result: Optional[HODLWaveResult] = None
        try:
            hodl_result = compute_hodl_waves(prices, symbol=symbol)
            self.store.upsert_metric(symbol, "hodl_waves", today, hodl_result.signal, hodl_result.lth_pct, "simulated", _dataclass_to_dict(hodl_result))
        except Exception as exc:
            logger.warning("HODLWaves failed: %s", exc)
            errors["hodl_waves"] = str(exc)

        # --- Supplementary: Hash Rate (BTC only) ---
        hr_result: Optional[HashRateResult] = None
        if "BTC" in symbol.upper():
            try:
                hr_result = compute_hash_rate(prices, symbol=symbol)
                self.store.upsert_metric(symbol, "hash_rate", today, hr_result.signal, hr_result.ribbon_ratio, hr_result.source, _dataclass_to_dict(hr_result))
            except Exception as exc:
                logger.warning("HashRate failed: %s", exc)
                errors["hash_rate"] = str(exc)

        # --- Composite Score ---
        composite = _compute_weighted_composite(component_signals, _WEIGHTS)

        # Persist composite
        self.store.upsert_composite(
            symbol=symbol,
            date=today,
            composite_score=composite,
            component_signals=component_signals,
            payload={
                "component_signals": component_signals,
                "weights": _WEIGHTS,
                "errors": errors,
            },
        )

        result = OnChainResult(
            symbol=symbol,
            composite_score=round(composite, 4),
            component_signals={k: round(v, 4) for k, v in component_signals.items()},
            mvrv=results.get("mvrv"),
            nvt=results.get("nvt"),
            sopr=results.get("sopr"),
            exchange_reserves=results.get("exchange_reserves"),
            whale=results.get("whale"),
            hodl_waves=hodl_result,
            hash_rate=hr_result,
            errors=errors,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "OnChainEngine.run(%s) complete — composite=%.3f regime=%s",
            symbol, composite, result.regime_label,
        )
        return result


def _compute_weighted_composite(
    signals: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Compute a weighted average of component signals.

    Missing signals (errors) default to 0.0 (neutral) so they reduce
    conviction without biasing direction.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for metric, weight in weights.items():
        sig = signals.get(metric, 0.0)
        weighted_sum  += sig * weight
        total_weight  += weight
    if total_weight == 0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, -1.0, 1.0))


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a plain dict (shallow, no nested objects)."""
    import dataclasses
    if dataclasses.is_dataclass(obj):
        result = {}
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            if dataclasses.is_dataclass(val):
                result[f.name] = _dataclass_to_dict(val)
            elif isinstance(val, list):
                result[f.name] = [_dataclass_to_dict(v) if dataclasses.is_dataclass(v) else v for v in val]
            else:
                result[f.name] = val
        return result
    return {"value": str(obj)}
