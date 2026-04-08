"""
Options Overlay — Systematic Delta Hedging for ETF Instruments (T2-6)
Extends options overlay to all ETF/equity instruments.

When BH confidence is high on equity instruments:
  - High confidence: buy ATM call/put (7-14 DTE) instead of direct equity
  - Medium confidence: use direct equity + covered call overlay
  - Low confidence: skip or small direct equity

Covered call overlay on existing long positions generates theta income
between BH signals.
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

@dataclass
class OptionsOverlayConfig:
    # BH confidence thresholds for options vs equity
    high_confidence_threshold: float = 0.70   # use options above this
    medium_confidence_threshold: float = 0.40  # hybrid above this

    # Options parameters
    target_dte_min: int = 7
    target_dte_max: int = 21
    target_delta: float = 0.50      # ATM options (delta ~0.5)
    max_premium_pct: float = 0.025  # max premium as % of underlying

    # Covered call parameters
    covered_call_delta: float = 0.25  # sell slightly OTM calls
    covered_call_dte: int = 14
    covered_call_min_yield: float = 0.005  # min premium to bother (0.5%)

    # Instruments eligible for options overlay
    options_eligible: tuple = ("SPY", "QQQ", "IWM", "GLD", "TLT", "GLD", "NVDA", "AAPL", "TSLA", "MSFT")

@dataclass
class OptionsPosition:
    symbol: str
    option_type: str        # "call" or "put"
    strike: float
    expiry: datetime
    premium_paid: float
    quantity: float
    underlying_entry: float
    bh_confidence_at_entry: float

    @property
    def is_expired(self) -> bool:
        return datetime.now() >= self.expiry

    def intrinsic_value(self, current_price: float) -> float:
        if self.option_type == "call":
            return max(0.0, current_price - self.strike)
        else:
            return max(0.0, self.strike - current_price)

class OptionsOverlayEngine:
    """
    Manages the options overlay for equity instruments.

    In live trading, actual options orders require Alpaca's options API.
    This engine generates the intent (what to trade) and tracks positions.
    Execution is delegated to the broker adapter.

    Usage:
        engine = OptionsOverlayEngine()

        # On BH formation for equity:
        intent = engine.on_bh_entry("SPY",
            current_price=500.0,
            bh_confidence=0.75,
            direction="long")

        # intent.use_options = True → execute options order
        # intent.use_options = False → execute direct equity order
    """

    def __init__(self, cfg: OptionsOverlayConfig = None):
        self.cfg = cfg or OptionsOverlayConfig()
        self._open_options: list[OptionsPosition] = []
        self._covered_calls: dict[str, OptionsPosition] = {}  # sym → covered call

    def on_bh_entry(
        self,
        symbol: str,
        current_price: float,
        bh_confidence: float,   # [0, 1] signal confidence
        direction: str,         # "long" or "short"
    ) -> dict:
        """
        Generate options overlay intent for a BH entry signal.

        Returns dict with:
          use_options: bool — True = use options, False = direct equity
          option_type: str — "call" or "put"
          target_strike: float
          target_expiry_days: int
          max_premium: float — max $ amount to spend on premium
          rationale: str
        """
        if symbol not in self.cfg.options_eligible:
            return {"use_options": False, "rationale": "not options eligible"}

        if bh_confidence >= self.cfg.high_confidence_threshold:
            # High confidence: use options for defined-risk entry
            option_type = "call" if direction == "long" else "put"

            # ATM strike (nearest round number)
            strike_spacing = max(1.0, round(current_price * 0.01))  # 1% of price
            target_strike = round(current_price / strike_spacing) * strike_spacing

            max_premium = current_price * self.cfg.max_premium_pct

            log.info(
                "OptionsOverlay: HIGH confidence %s BH → %s %s %.0f DTE, strike=%.2f, max_prem=%.2f",
                symbol, option_type, symbol, self.cfg.target_dte_min, target_strike, max_premium
            )

            return {
                "use_options": True,
                "option_type": option_type,
                "target_strike": target_strike,
                "target_expiry_days": self.cfg.target_dte_min,
                "max_premium": max_premium,
                "rationale": f"BH confidence={bh_confidence:.2f} > {self.cfg.high_confidence_threshold}",
            }

        elif bh_confidence >= self.cfg.medium_confidence_threshold:
            # Medium confidence: direct equity + plan covered call overlay
            return {
                "use_options": False,
                "plan_covered_call": True,
                "covered_call_delta": self.cfg.covered_call_delta,
                "covered_call_dte": self.cfg.covered_call_dte,
                "rationale": f"BH confidence={bh_confidence:.2f} — hybrid approach",
            }

        else:
            return {
                "use_options": False,
                "rationale": f"BH confidence={bh_confidence:.2f} below threshold",
            }

    def on_bh_exit(self, symbol: str, current_price: float) -> dict:
        """Close any open options positions for symbol."""
        actions = []

        # Close directional options
        to_close = [p for p in self._open_options if p.symbol == symbol]
        for pos in to_close:
            intrinsic = pos.intrinsic_value(current_price)
            pnl = intrinsic - pos.premium_paid
            actions.append({
                "action": "close_option",
                "symbol": symbol,
                "option_type": pos.option_type,
                "strike": pos.strike,
                "pnl_estimate": pnl,
            })
            self._open_options.remove(pos)

        # Close covered call if exists
        if symbol in self._covered_calls:
            cc = self._covered_calls[symbol]
            actions.append({
                "action": "close_covered_call",
                "symbol": symbol,
                "strike": cc.strike,
                "premium_received": cc.premium_paid,
            })
            del self._covered_calls[symbol]

        return {"actions": actions}

    def on_bar(self, symbol: str, current_price: float) -> list[dict]:
        """
        Process one bar. Check for expiring options, covered call opportunities.
        Returns list of recommended actions.
        """
        actions = []

        # Remove expired options
        expired = [p for p in self._open_options if p.symbol == symbol and p.is_expired]
        for p in expired:
            log.info("OptionsOverlay: %s %s expired (intrinsic=%.2f, paid=%.2f)",
                     symbol, p.option_type, p.intrinsic_value(current_price), p.premium_paid)
            self._open_options.remove(p)

        return actions

    def record_option_fill(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiry_days: int,
        premium: float,
        quantity: float,
        current_price: float,
        bh_confidence: float,
    ):
        """Record a new options position after fill."""
        expiry = datetime.now() + timedelta(days=expiry_days)
        pos = OptionsPosition(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            premium_paid=premium,
            quantity=quantity,
            underlying_entry=current_price,
            bh_confidence_at_entry=bh_confidence,
        )
        self._open_options.append(pos)

    def get_theta_pnl(self) -> float:
        """Estimate total theta P&L from covered calls (premium collected)."""
        return sum(cc.premium_paid for cc in self._covered_calls.values())

    def summary(self) -> dict:
        return {
            "open_options": len(self._open_options),
            "covered_calls": len(self._covered_calls),
            "estimated_theta_pnl": self.get_theta_pnl(),
        }
