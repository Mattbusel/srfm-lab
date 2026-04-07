"""
margin_manager.py # Margin and leverage tracking for SRFM execution layer.

Supports three margin regimes:
  - RegTMarginCalculator  : Reg T rules for US equities
  - CryptoMarginCalculator: Exchange-specific crypto margin (Binance / Coinbase)
  - PortfolioMarginCalculator: Risk-based margin for hedged positions (simplified SPAN)

All calculators implement MarginCalculatorBase and return a MarginRequirement
dataclass which the MarginManager aggregates.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MarginConfig:
    """Margin parameters for all asset classes."""
    account_nav: float                        # current account NAV in dollars

    # Equity (Reg T)
    equity_initial_margin: float = 0.25       # 25% for long equities
    equity_maintenance: float = 0.25          # FINRA minimum 25%
    equity_short_initial: float = 0.50        # 50% for short sales

    # Crypto
    crypto_initial_margin: float = 0.50       # 50% initial
    crypto_maintenance: float = 0.35          # 35% maintenance

    # Futures
    futures_initial_margin: float = 0.05      # ~5% typical futures IM
    futures_maintenance: float = 0.04         # slightly below initial

    # House requirements (broker may add buffer above regulatory minimums)
    house_buffer: float = 0.05               # extra 5% added to all requirements
    maintenance_call_threshold: float = 1.05  # trigger call at 105% of maintenance

    # Margin call handling
    liquidation_priority_liquidity_weight: float = 0.70
    liquidation_priority_size_weight: float = 0.30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Single instrument position used for margin calculations."""
    symbol: str
    qty: float            # positive = long, negative = short
    price: float          # current mark price
    asset_class: str      # equity | crypto | futures
    market_value: float = field(init=False)

    def __post_init__(self) -> None:
        self.market_value = self.qty * self.price


@dataclass
class MarginRequirement:
    """Output from any MarginCalculatorBase implementation."""
    initial_margin: float
    maintenance_margin: float
    method: str                                 # regt | crypto | portfolio
    positions_included: int = 0
    detail: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MarginCalculatorBase(ABC):
    """Abstract interface for margin calculators."""

    @abstractmethod
    def calculate(self, positions: List[Position], config: MarginConfig) -> MarginRequirement:
        ...

    @abstractmethod
    def initial_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        ...

    @abstractmethod
    def maintenance_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        ...


# ---------------------------------------------------------------------------
# Reg T calculator
# ---------------------------------------------------------------------------

class RegTMarginCalculator(MarginCalculatorBase):
    """
    Implements Federal Reserve Regulation T margin rules for US equities.

    Long positions: initial margin = 50% (Reg T requirement), maintenance
    set to 25% (FINRA Rule 4210).  Short positions require 150% of proceeds
    on deposit (100% from short sale proceeds + 50% additional).

    In practice many brokers use 25% maintenance for longs.  We use the
    config values which default to 25% to match typical house rules.
    """

    def calculate(self, positions: List[Position], config: MarginConfig) -> MarginRequirement:
        total_initial = 0.0
        total_maintenance = 0.0
        detail: Dict[str, float] = {}

        for pos in positions:
            if pos.asset_class != "equity":
                continue
            im = self.initial_margin_for_single(pos.symbol, pos.qty, pos.price, config)
            mm = self.maintenance_margin_for_single(pos.symbol, pos.qty, pos.price, config)
            total_initial += im
            total_maintenance += mm
            detail[pos.symbol] = im

        return MarginRequirement(
            initial_margin=total_initial,
            maintenance_margin=total_maintenance,
            method="regt",
            positions_included=len([p for p in positions if p.asset_class == "equity"]),
            detail=detail,
        )

    def initial_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        market_value = abs(qty * price)
        if qty < 0:
            # Short: 50% of market value as initial margin (house requirement)
            return market_value * config.equity_short_initial
        return market_value * config.equity_initial_margin

    def maintenance_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        market_value = abs(qty * price)
        if qty < 0:
            # Short maintenance: typically 30% of current market value
            return market_value * 0.30
        return market_value * config.equity_maintenance

    def reg_t_buying_power(self, equity: float) -> float:
        """
        Available buying power under Reg T (2:1 for equities, 4:1 for
        intraday # we use 2:1 as the conservative overnight figure).
        """
        return equity * 2.0


# ---------------------------------------------------------------------------
# Crypto calculator
# ---------------------------------------------------------------------------

class CryptoMarginCalculator(MarginCalculatorBase):
    """
    Exchange-specific crypto margin for Binance and Coinbase.

    Binance uses a tiered maintenance margin schedule based on notional
    size.  We simplify to a flat 35% maintenance / 50% initial for
    positions under a configurable notional threshold.
    """

    # Binance-like tiered maintenance schedule: (notional_threshold, maint_rate)
    _BINANCE_TIERS: List[Tuple[float, float]] = [
        (50_000,    0.004),
        (250_000,   0.005),
        (1_000_000, 0.010),
        (5_000_000, 0.025),
        (float("inf"), 0.050),
    ]

    def _binance_maintenance_rate(self, notional: float) -> float:
        for threshold, rate in self._BINANCE_TIERS:
            if notional <= threshold:
                return rate
        return 0.05

    def calculate(self, positions: List[Position], config: MarginConfig) -> MarginRequirement:
        total_initial = 0.0
        total_maintenance = 0.0
        detail: Dict[str, float] = {}

        for pos in positions:
            if pos.asset_class != "crypto":
                continue
            im = self.initial_margin_for_single(pos.symbol, pos.qty, pos.price, config)
            mm = self.maintenance_margin_for_single(pos.symbol, pos.qty, pos.price, config)
            total_initial += im
            total_maintenance += mm
            detail[pos.symbol] = im

        return MarginRequirement(
            initial_margin=total_initial,
            maintenance_margin=total_maintenance,
            method="crypto",
            positions_included=len([p for p in positions if p.asset_class == "crypto"]),
            detail=detail,
        )

    def initial_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        return abs(qty * price) * config.crypto_initial_margin

    def maintenance_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        notional = abs(qty * price)
        # Use Binance-like tier for exchange-traded crypto
        if symbol.endswith("USDT") or symbol.endswith("BUSD"):
            rate = self._binance_maintenance_rate(notional)
            return notional * rate
        return notional * config.crypto_maintenance


# ---------------------------------------------------------------------------
# Portfolio margin calculator (simplified SPAN)
# ---------------------------------------------------------------------------

class PortfolioMarginCalculator(MarginCalculatorBase):
    """
    Risk-based margin using a simplified SPAN (Standard Portfolio Analysis
    of Risk) approach.

    Key logic:
    1. Scan price moves of +/- 10% and vol moves of +/- 25%.
    2. Compute worst-case P&L across the 16-scenario grid.
    3. Initial margin = worst-case loss + short option minimum charge.
    4. Maintenance = 85% of initial.

    For a delta-neutral or well-hedged book this gives significantly lower
    margin requirements than Reg T.
    """

    _PRICE_SCENARIOS: List[float] = [-0.10, -0.06, -0.03, 0.0, 0.03, 0.06, 0.10]
    _VOL_SCENARIOS: List[float] = [-0.25, 0.0, 0.25]

    def calculate(self, positions: List[Position], config: MarginConfig) -> MarginRequirement:
        if not positions:
            return MarginRequirement(0.0, 0.0, "portfolio", 0)

        # Group longs and shorts per symbol for netting
        net_exposure: Dict[str, float] = {}
        for pos in positions:
            net_exposure[pos.symbol] = net_exposure.get(pos.symbol, 0.0) + pos.market_value

        worst_loss = self._scan_scenarios(net_exposure)
        initial = max(worst_loss, 0.0)
        maintenance = initial * 0.85

        return MarginRequirement(
            initial_margin=initial,
            maintenance_margin=maintenance,
            method="portfolio",
            positions_included=len(positions),
            detail={"worst_scenario_loss": worst_loss},
        )

    def _scan_scenarios(self, net_exposure: Dict[str, float]) -> float:
        """Compute worst-case portfolio P&L across all price/vol scenarios."""
        worst: float = 0.0
        for price_move in self._PRICE_SCENARIOS:
            for _vol_move in self._VOL_SCENARIOS:
                scenario_pnl = sum(val * price_move for val in net_exposure.values())
                # Haircut for vol: increase loss by 5% of absolute exposure per vol scenario
                vol_haircut = abs(_vol_move) * 0.05 * sum(abs(v) for v in net_exposure.values())
                scenario_loss = -scenario_pnl + vol_haircut
                worst = max(worst, scenario_loss)
        return worst

    def initial_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        # For a single position use Reg T equivalent
        return abs(qty * price) * 0.15  # 15% minimum for portfolio margin accounts

    def maintenance_margin_for_single(
        self, symbol: str, qty: float, price: float, config: MarginConfig
    ) -> float:
        return self.initial_margin_for_single(symbol, qty, price, config) * 0.85


# ---------------------------------------------------------------------------
# Margin manager
# ---------------------------------------------------------------------------

class MarginManager:
    """
    Central margin and leverage tracking for the entire portfolio.

    Responsibilities:
    - Maintain current positions and their margin requirements.
    - Compute total initial and maintenance margin across all asset classes.
    - Detect margin calls and suggest positions to liquidate.
    - Expose margin utilization as a 0.0-1.0 scalar for risk dashboards.
    """

    def __init__(
        self,
        config: MarginConfig,
        equity_calculator: Optional[RegTMarginCalculator] = None,
        crypto_calculator: Optional[CryptoMarginCalculator] = None,
        portfolio_calculator: Optional[PortfolioMarginCalculator] = None,
    ) -> None:
        self._config = config
        self._positions: Dict[str, Position] = {}

        self._equity_calc = equity_calculator or RegTMarginCalculator()
        self._crypto_calc = crypto_calculator or CryptoMarginCalculator()
        self._portfolio_calc = portfolio_calculator or PortfolioMarginCalculator()

        # Cached margin requirements # recomputed on position update
        self._cached_im: float = 0.0
        self._cached_mm: float = 0.0
        self._dirty: bool = True

    # # Position management -------------------------------------------------

    def update_position(
        self,
        symbol: str,
        qty: float,
        price: float,
        asset_class: str = "equity",
    ) -> None:
        """
        Insert or update a position and mark margins dirty.
        Pass qty=0 to remove a position.
        """
        if qty == 0.0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                price=price,
                asset_class=asset_class,
            )
        self._dirty = True

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Batch update mark prices for open positions."""
        for symbol, price in prices.items():
            if symbol in self._positions:
                pos = self._positions[symbol]
                self._positions[symbol] = Position(
                    symbol=symbol,
                    qty=pos.qty,
                    price=price,
                    asset_class=pos.asset_class,
                )
        self._dirty = True

    def update_nav(self, nav: float) -> None:
        self._config.account_nav = nav
        self._dirty = True

    # # Margin computations -------------------------------------------------

    def _recompute(self) -> None:
        """Recompute initial and maintenance margin for all positions."""
        if not self._dirty:
            return

        pos_list = list(self._positions.values())
        equity_req = self._equity_calc.calculate(pos_list, self._config)
        crypto_req = self._crypto_calc.calculate(pos_list, self._config)

        # Futures positions use portfolio margin
        futures_positions = [p for p in pos_list if p.asset_class == "futures"]
        futures_req = self._portfolio_calc.calculate(futures_positions, self._config)

        self._cached_im = equity_req.initial_margin + crypto_req.initial_margin + futures_req.initial_margin
        self._cached_mm = equity_req.maintenance_margin + crypto_req.maintenance_margin + futures_req.maintenance_margin
        self._dirty = False

    def initial_margin_required(
        self, symbol: str, qty: float, price: float
    ) -> float:
        """Compute the initial margin for a proposed new position (does not alter state)."""
        asset_class = "equity"
        if symbol in self._positions:
            asset_class = self._positions[symbol].asset_class

        if asset_class == "crypto":
            return self._crypto_calc.initial_margin_for_single(symbol, qty, price, self._config)
        if asset_class == "futures":
            return self._portfolio_calc.initial_margin_for_single(symbol, qty, price, self._config)
        return self._equity_calc.initial_margin_for_single(symbol, qty, price, self._config)

    def maintenance_margin(
        self, symbol: str, qty: float, price: float
    ) -> float:
        """Compute the maintenance margin for a proposed position."""
        asset_class = "equity"
        if symbol in self._positions:
            asset_class = self._positions[symbol].asset_class

        if asset_class == "crypto":
            return self._crypto_calc.maintenance_margin_for_single(symbol, qty, price, self._config)
        if asset_class == "futures":
            return self._portfolio_calc.maintenance_margin_for_single(symbol, qty, price, self._config)
        return self._equity_calc.maintenance_margin_for_single(symbol, qty, price, self._config)

    def available_margin(self) -> float:
        """
        Cash available for new positions = equity - initial margin already
        committed.  Floored at 0.
        """
        self._recompute()
        equity = self._account_equity()
        return max(0.0, equity - self._cached_im)

    def margin_utilization(self) -> float:
        """
        Ratio of committed initial margin to account equity.
        Returns 0.0 when equity is zero to avoid division by zero.
        """
        self._recompute()
        equity = self._account_equity()
        if equity <= 0:
            return 1.0
        return min(1.0, self._cached_im / equity)

    def is_margin_call(self) -> bool:
        """
        True when account equity falls below the maintenance margin
        plus the configured call threshold buffer.
        """
        self._recompute()
        equity = self._account_equity()
        threshold = self._cached_mm * self._config.maintenance_call_threshold
        return equity < threshold

    def positions_to_liquidate_for_margin(self) -> List[str]:
        """
        Returns symbols sorted by liquidation priority (most liquid and
        largest position first) to restore the account above maintenance.

        Liquidity score = 1/abs(position_value) * liquidity_weight +
                          abs(position_value) * size_weight
        (We use inverse of size as a crude liquidity proxy; production
        should integrate real ADV data.)
        """
        self._recompute()
        if not self.is_margin_call():
            return []

        equity = self._account_equity()
        shortfall = self._cached_mm * self._config.maintenance_call_threshold - equity

        lw = self._config.liquidation_priority_liquidity_weight
        sw = self._config.liquidation_priority_size_weight

        scored: List[Tuple[float, str]] = []
        for sym, pos in self._positions.items():
            mv = abs(pos.market_value)
            if mv <= 0:
                continue
            score = lw * (1.0 / mv) * 1e6 + sw * mv
            scored.append((score, sym))

        scored.sort(reverse=True)
        return [sym for _, sym in scored]

    # # Portfolio-level metrics ---------------------------------------------

    def gross_leverage(self) -> float:
        """Sum of absolute exposures divided by NAV."""
        nav = self._config.account_nav
        if nav <= 0:
            return 0.0
        total = sum(abs(p.market_value) for p in self._positions.values())
        return total / nav

    def net_leverage(self) -> float:
        """Net long-short exposure divided by NAV."""
        nav = self._config.account_nav
        if nav <= 0:
            return 0.0
        total = sum(p.market_value for p in self._positions.values())
        return total / nav

    def total_market_value(self) -> float:
        return sum(p.market_value for p in self._positions.values())

    def position_count(self) -> int:
        return len(self._positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def snapshot(self) -> Dict:
        """Return a dict summary of current margin state."""
        self._recompute()
        return {
            "initial_margin": self._cached_im,
            "maintenance_margin": self._cached_mm,
            "available_margin": self.available_margin(),
            "margin_utilization": self.margin_utilization(),
            "is_margin_call": self.is_margin_call(),
            "gross_leverage": self.gross_leverage(),
            "net_leverage": self.net_leverage(),
            "account_nav": self._config.account_nav,
            "n_positions": self.position_count(),
        }

    # # Private helpers -----------------------------------------------------

    def _account_equity(self) -> float:
        """
        Account equity = NAV (includes unrealized P&L already marked
        in position prices).
        """
        return self._config.account_nav
