"""
Shadow Trading Infrastructure (A2)
Runs a complete shadow instance of the live system with IAE-proposed parameters.

Compares shadow performance vs live performance on ongoing basis.
When shadow Sharpe exceeds live Sharpe by >0.3 over 30 days, triggers parameter migration.
"""
import logging
import time
import copy
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class ShadowConfig:
    comparison_window_bars: int = 2880   # 30 days of 15m bars
    migration_sharpe_gap: float = 0.30   # shadow must beat live by this much
    min_trades_for_comparison: int = 30  # need enough trades to compare
    max_position_frac: float = 0.05      # shadow uses smaller positions

@dataclass
class ShadowTrade:
    symbol: str
    entry_price: float
    entry_bar: int
    exit_price: float = 0.0
    exit_bar: int = 0
    qty_frac: float = 0.0
    pnl_pct: float = 0.0
    is_closed: bool = False
    params_version: str = "shadow"

class ShadowTrader:
    """
    Maintains a parallel paper-trading instance using IAE-proposed parameters.
    Tracks performance separately from live trading.

    Usage:
        shadow = ShadowTrader(iae_params={"BH_FORM": 2.10, ...})

        # On each bar:
        shadow.on_bar(symbol="BTC", close=50000.0, bar_seq=100,
                      shadow_signal=0.15)  # signal from IAE-proposed params

        # Check migration recommendation:
        rec = shadow.get_migration_recommendation()
        if rec["recommend_migration"]:
            # Apply IAE params to live trader
    """

    def __init__(self, shadow_params: dict = None, cfg: ShadowConfig = None):
        self.cfg = cfg or ShadowConfig()
        self._shadow_params = shadow_params or {}
        self._shadow_positions: dict[str, float] = {}  # sym -> frac held
        self._shadow_entry_prices: dict[str, float] = {}
        self._shadow_entry_bars: dict[str, int] = {}
        self._shadow_trades: list[ShadowTrade] = []
        self._shadow_returns: deque[float] = deque(maxlen=self.cfg.comparison_window_bars)
        self._live_returns: deque[float] = deque(maxlen=self.cfg.comparison_window_bars)
        self._bar_count: int = 0
        self._last_report: int = 0

    def on_bar(
        self,
        symbol: str,
        close: float,
        bar_seq: int,
        shadow_signal: float,  # proposed target fraction from IAE params
        live_signal: float,    # actual live signal
        live_pnl_pct: float = 0.0,   # live trade P&L (if any closed this bar)
    ):
        """Process one bar for both shadow and live tracking."""
        self._bar_count += 1

        # Record live returns
        if live_pnl_pct != 0.0:
            self._live_returns.append(live_pnl_pct)

        # Shadow position management
        current_pos = self._shadow_positions.get(symbol, 0.0)
        target = min(self.cfg.max_position_frac, abs(shadow_signal)) * (1 if shadow_signal >= 0 else -1)

        # Shadow entry
        if abs(current_pos) < 0.01 and abs(target) > 0.01:
            self._shadow_positions[symbol] = target
            self._shadow_entry_prices[symbol] = close
            self._shadow_entry_bars[symbol] = bar_seq

        # Shadow exit
        elif abs(current_pos) > 0.01 and abs(target) < 0.005:
            entry_price = self._shadow_entry_prices.get(symbol, close)
            pnl_pct = (close - entry_price) / entry_price * current_pos
            self._shadow_returns.append(pnl_pct)

            trade = ShadowTrade(
                symbol=symbol,
                entry_price=entry_price,
                entry_bar=self._shadow_entry_bars.get(symbol, 0),
                exit_price=close,
                exit_bar=bar_seq,
                qty_frac=current_pos,
                pnl_pct=pnl_pct,
                is_closed=True,
            )
            self._shadow_trades.append(trade)
            self._shadow_positions[symbol] = 0.0

        # Periodic comparison report
        if self._bar_count - self._last_report >= 500:
            self._last_report = self._bar_count
            self._log_comparison()

    def get_migration_recommendation(self) -> dict:
        """
        Returns migration recommendation.
        recommend_migration=True if shadow significantly outperforms live.
        """
        shadow_sharpe = self._compute_sharpe(list(self._shadow_returns))
        live_sharpe = self._compute_sharpe(list(self._live_returns))

        shadow_trades = len(self._shadow_trades)
        live_trades_est = len(self._live_returns)

        recommend = (
            shadow_sharpe - live_sharpe > self.cfg.migration_sharpe_gap and
            shadow_trades >= self.cfg.min_trades_for_comparison
        )

        return {
            "recommend_migration": recommend,
            "shadow_sharpe": shadow_sharpe,
            "live_sharpe": live_sharpe,
            "gap": shadow_sharpe - live_sharpe,
            "shadow_trades": shadow_trades,
            "shadow_params": self._shadow_params,
        }

    def update_shadow_params(self, new_params: dict):
        """Update shadow parameters (from IAE genome evolution output)."""
        self._shadow_params = copy.deepcopy(new_params)
        log.info("ShadowTrader: params updated with %d new values", len(new_params))

    def _compute_sharpe(self, returns: list[float]) -> float:
        if len(returns) < 5:
            return 0.0
        n = len(returns)
        mean_r = sum(returns) / n
        std_r = (sum((r - mean_r)**2 for r in returns) / n) ** 0.5
        if std_r < 1e-9:
            return 0.0
        # Annualized (252 trading days * 96 bars/day = 24192 bars/year for 15m)
        return mean_r / std_r * math.sqrt(24192)

    def _log_comparison(self):
        shadow_sharpe = self._compute_sharpe(list(self._shadow_returns))
        live_sharpe = self._compute_sharpe(list(self._live_returns))
        gap = shadow_sharpe - live_sharpe

        log.info(
            "ShadowTrader: shadow_sharpe=%.3f, live_sharpe=%.3f, gap=%+.3f (%s)",
            shadow_sharpe, live_sharpe, gap,
            "-> RECOMMEND MIGRATION" if gap > self.cfg.migration_sharpe_gap else "monitoring"
        )
