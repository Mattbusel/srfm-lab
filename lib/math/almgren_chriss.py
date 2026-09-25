"""
Almgren-Chriss Implementation Shortfall Minimizer (T2-10)
Minimizes trading cost using the Almgren-Chriss (2000) market impact model.

Market impact: MI = sigma * sqrt(V_order / ADV)
Optimal execution: VWAP/TWAP schedule weighted by intraday volume profile.

Also maintains a TCA database for post-trade slippage measurement and
per-instrument market impact model calibration.
"""
import math
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import deque

log = logging.getLogger(__name__)

@dataclass
class ExecutionConfig:
    participation_rate: float = 0.15    # target 15% of market volume per slice
    max_slices: int = 10                # max child orders per parent
    min_slice_frac: float = 0.05        # minimum slice size as fraction of total
    tca_window: int = 500               # bars for TCA calibration
    impact_model_alpha: float = 0.05    # EMA weight for impact model update

@dataclass
class ExecutionSlice:
    quantity: float
    side: str  # "buy" or "sell"
    urgency: float  # 0-1: higher = more aggressive participation rate
    arrival_price: float
    filled_price: Optional[float] = None
    filled_at: Optional[float] = None

@dataclass
class TCARecord:
    symbol: str
    side: str
    quantity: float
    arrival_price: float
    fill_price: float
    slippage_bps: float   # (fill - arrival) / arrival * 10000
    market_impact_bps: float  # estimated impact component
    timestamp: float = field(default_factory=time.time)

class AlmgrenChrissExecutor:
    """
    Optimal order scheduling using Almgren-Chriss market impact model.

    Key methods:
      schedule_order() -- returns list of child order sizes and timing
      record_fill()    -- records actual fill for TCA
      get_impact_estimate() -- estimates market impact before trading

    Usage:
        executor = AlmgrenChrissExecutor()
        slices = executor.schedule_order("BTC", qty=0.5, side="buy", urgency=0.7)
        for slice_ in slices:
            # execute slice_.quantity at next bar
            executor.record_fill("BTC", slice_, fill_price=50100.0)
    """

    def __init__(self, cfg: ExecutionConfig = None, tca_path: str = None):
        self.cfg = cfg or ExecutionConfig()
        self._tca_path = Path(tca_path or "data/tca_records.jsonl")

        # Per-symbol calibrated impact parameters
        # impact_est = impact_coeff * sigma * sqrt(qty / adv)
        self._impact_coeffs: dict[str, float] = {}
        self._adv_ema: dict[str, float] = {}    # average daily volume EMA
        self._vol_ema: dict[str, float] = {}    # price volatility EMA
        self._tca_records: dict[str, list[TCARecord]] = {}

        # Intraday volume profile: bar_of_day -> volume fraction
        # Default: U-shaped (higher at open/close, lower midday)
        self._volume_profile = self._default_volume_profile()

    def schedule_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        arrival_price: float,
        urgency: float = 0.5,
        max_slices: int = None,
    ) -> list[ExecutionSlice]:
        """
        Generate optimal child order schedule.

        urgency: 0 = VWAP (patient, minimize impact), 1 = immediate (minimize timing risk)
        Returns list of ExecutionSlice objects ordered by execution sequence.
        """
        max_n = max_slices or self.cfg.max_slices

        # Estimate total market impact
        impact_bps = self.get_impact_estimate(symbol, qty, arrival_price)

        # Number of slices: fewer for urgent orders, more for patient
        n_slices = max(1, min(max_n, round(max_n * (1 - urgency * 0.7))))

        # Almgren-Chriss optimal trajectory: balance impact vs timing risk
        # Simplified: use a power law distribution based on urgency
        if urgency > 0.8:
            # Front-load: execute most qty immediately
            weights = [urgency ** i for i in range(n_slices)]
        else:
            # Back-load toward VWAP: more even distribution
            weights = [1.0 - 0.3 * (i / max(n_slices - 1, 1)) for i in range(n_slices)]

        total_w = sum(weights)
        slice_qtys = [qty * w / total_w for w in weights]

        # Clamp minimum slice
        min_qty = qty * self.cfg.min_slice_frac
        slice_qtys = [max(q, min_qty) for q in slice_qtys]

        # Renormalize to maintain total quantity
        total_q = sum(slice_qtys)
        if total_q > 0:
            slice_qtys = [q * qty / total_q for q in slice_qtys]

        slices = [
            ExecutionSlice(
                quantity=q,
                side=side,
                urgency=urgency,
                arrival_price=arrival_price,
            )
            for q in slice_qtys
        ]

        log.info(
            "AlmgrenChriss: %s %s %.4f @ %.2f -> %d slices, impact_est=%.1f bps",
            side, symbol, qty, arrival_price, n_slices, impact_bps
        )
        return slices

    def get_impact_estimate(self, symbol: str, qty: float, price: float) -> float:
        """
        Estimate market impact in basis points using Almgren-Chriss formula.
        MI_bps = impact_coeff * sigma * sqrt(qty_usd / ADV_usd) * 10000
        """
        sigma = self._vol_ema.get(symbol, 0.02)  # default 2% daily vol
        adv = self._adv_ema.get(symbol, 1e6)     # default $1M ADV
        coeff = self._impact_coeffs.get(symbol, 0.3)  # default coefficient

        qty_usd = abs(qty) * price
        impact = coeff * sigma * math.sqrt(qty_usd / (adv + 1e-10))
        return impact * 10000  # convert to bps

    def record_fill(self, symbol: str, slice_: ExecutionSlice, fill_price: float):
        """Record actual fill for TCA and model calibration."""
        slice_.filled_price = fill_price
        slice_.filled_at = time.time()

        slippage_bps = (fill_price - slice_.arrival_price) / slice_.arrival_price * 10000
        if slice_.side == "sell":
            slippage_bps = -slippage_bps  # positive slippage = got worse price for sell

        impact_bps = self.get_impact_estimate(symbol, slice_.quantity, fill_price)

        record = TCARecord(
            symbol=symbol,
            side=slice_.side,
            quantity=slice_.quantity,
            arrival_price=slice_.arrival_price,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            market_impact_bps=impact_bps,
        )

        if symbol not in self._tca_records:
            self._tca_records[symbol] = []
        self._tca_records[symbol].append(record)

        # Update impact coefficient via EMA
        if impact_bps > 0:
            measured_coeff = abs(slippage_bps) / (impact_bps / self._impact_coeffs.get(symbol, 0.3) + 1e-10)
            current = self._impact_coeffs.get(symbol, 0.3)
            self._impact_coeffs[symbol] = (
                (1 - self.cfg.impact_model_alpha) * current +
                self.cfg.impact_model_alpha * measured_coeff
            )

        # Persist
        self._append_tca_record(record)

        if abs(slippage_bps) > 20:
            log.warning("%s fill: slippage=%.1f bps (impact_est=%.1f bps)", symbol, slippage_bps, impact_bps)

    def update_market_data(self, symbol: str, volume: float, price_ret: float):
        """Update ADV and vol estimates. Call on every bar."""
        # ADV: EMA of daily volume (96 bars/day at 15m)
        bars_per_day = 96
        adv_alpha = 2.0 / (20 * bars_per_day + 1)  # 20-day EMA equivalent
        vol_alpha = 2.0 / (20 * bars_per_day + 1)

        prev_adv = self._adv_ema.get(symbol, volume)
        self._adv_ema[symbol] = adv_alpha * volume + (1 - adv_alpha) * prev_adv

        prev_vol = self._vol_ema.get(symbol, abs(price_ret))
        self._vol_ema[symbol] = vol_alpha * abs(price_ret) + (1 - vol_alpha) * prev_vol

    def get_tca_summary(self, symbol: str) -> dict:
        """Return TCA summary stats for a symbol."""
        records = self._tca_records.get(symbol, [])
        if not records:
            return {}

        slippages = [r.slippage_bps for r in records]
        n = len(slippages)
        mean_slip = sum(slippages) / n
        std_slip = (sum((s - mean_slip)**2 for s in slippages) / n) ** 0.5

        return {
            "symbol": symbol,
            "n_fills": n,
            "mean_slippage_bps": mean_slip,
            "std_slippage_bps": std_slip,
            "impact_coeff": self._impact_coeffs.get(symbol, 0.3),
        }

    def _default_volume_profile(self) -> list[float]:
        """U-shaped intraday volume profile (96 bars for 24h crypto)."""
        profile = []
        for i in range(96):
            t = i / 96.0  # 0 to 1
            # Higher volume at start/end of "trading day" (UTC midnight)
            vol = 0.7 + 0.3 * (math.cos(2 * math.pi * (t - 0.5)) ** 2)
            profile.append(vol)
        total = sum(profile)
        return [v / total for v in profile]

    def _append_tca_record(self, record: TCARecord):
        try:
            self._tca_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._tca_path, "a") as f:
                f.write(json.dumps({
                    "sym": record.symbol,
                    "side": record.side,
                    "qty": record.quantity,
                    "arrival": record.arrival_price,
                    "fill": record.fill_price,
                    "slip_bps": record.slippage_bps,
                    "ts": record.timestamp,
                }) + "\n")
        except Exception as e:
            log.debug("TCA persist failed: %s", e)
