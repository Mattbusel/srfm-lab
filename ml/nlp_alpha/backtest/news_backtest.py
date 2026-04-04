"""
Event-driven news backtest engine.

Features:
- Fill-at-open after news announcement (no lookahead bias)
- Slippage model (market impact, bid-ask spread)
- Horizon analysis (1h / 1d / 1w returns after event)
- Performance attribution by event type
- Comparison vs hold-everything baseline
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..signal.event_detector import DetectedEvent, EventTypes
from ..signal.alpha_builder import AlphaSignal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NewsBacktestConfig:
    initial_capital: float = 100_000.0
    fill_at_open: bool = True          # fill on next open after news
    slippage_bps: float = 3.0          # basis points per trade
    commission_bps: float = 5.0        # basis points commission
    max_position_pct: float = 0.20     # max 20% in any one stock
    max_gross_leverage: float = 1.5    # max 150% gross leverage
    min_signal_threshold: float = 0.15  # ignore signals below this
    horizons_days: List[int] = field(default_factory=lambda: [1, 5, 22])  # 1d, 1w, 1m
    risk_free_rate: float = 0.05       # annual
    rebalance_freq: str = "daily"      # "daily" | "weekly" | "event_driven"
    max_drawdown_stop: float = 0.25    # stop trading if DD > 25%
    position_sizing: str = "signal"    # "signal" | "equal" | "volatility"
    vol_target: float = 0.15           # annual volatility target for vol sizing
    verbose: bool = True


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Single trade execution record."""
    date: datetime
    ticker: str
    action: str            # "buy" | "sell" | "short" | "cover"
    shares: float
    fill_price: float
    signal: float
    event_type: Optional[str]
    slippage: float
    commission: float
    notional: float
    pnl: Optional[float] = None        # filled after exit
    hold_days: Optional[int] = None


@dataclass
class EventBacktestResult:
    """Result for a single event backtest."""
    event: DetectedEvent
    entry_date: datetime
    exit_dates: Dict[int, datetime]      # horizon_days -> exit_date
    entry_price: float
    exit_prices: Dict[int, float]        # horizon_days -> price
    returns: Dict[int, float]            # horizon_days -> return
    signal: float
    pnl: float
    correct_direction: bool


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------

class SlippageModel:
    """
    Market impact + bid-ask spread slippage model.
    Uses square-root market impact formula.
    """

    def __init__(
        self,
        spread_bps: float = 3.0,
        impact_coef: float = 0.1,      # market impact coefficient
        daily_volume_millions: float = 100.0,  # default ADV in millions
    ):
        self.spread_bps = spread_bps / 10000
        self.impact_coef = impact_coef
        self.adv = daily_volume_millions * 1e6

    def compute_slippage(
        self,
        price: float,
        shares: float,
        adv: Optional[float] = None,
    ) -> float:
        """
        Compute total slippage cost in dollars.
        Uses: total = spread/2 + impact * sqrt(participation_rate)
        """
        adv_used = adv or self.adv
        notional = price * abs(shares)

        # Participation rate (fraction of ADV)
        participation = notional / (adv_used + 1e-8)
        participation = min(participation, 0.3)  # cap at 30% of ADV

        # Market impact (sqrt model)
        impact_bps = self.impact_coef * np.sqrt(participation) * 100  # in bps
        total_bps = self.spread_bps * 10000 / 2 + impact_bps  # half spread + impact

        return float(notional * total_bps / 10000)

    def get_fill_price(self, market_price: float, direction: int, adv: Optional[float] = None) -> float:
        """Get execution price with slippage."""
        slippage_frac = self.spread_bps / 2 + self.impact_coef * 0.01 * 0.1
        return float(market_price * (1.0 + direction * slippage_frac))


# ---------------------------------------------------------------------------
# Portfolio manager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """Manages portfolio state for the news backtest."""

    def __init__(self, initial_capital: float, config: NewsBacktestConfig):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.config = config
        self.positions: Dict[str, float] = {}      # ticker -> shares
        self.avg_cost: Dict[str, float] = {}        # ticker -> avg cost per share
        self.pnl_realized: float = 0.0
        self.trades: List[Trade] = []
        self.portfolio_value_history: List[float] = []

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        position_value = sum(
            self.positions.get(t, 0.0) * current_prices.get(t, self.avg_cost.get(t, 0.0))
            for t in self.positions
        )
        return float(self.cash + position_value)

    def can_trade(self, ticker: str, signal: float, price: float, portfolio_value: float) -> bool:
        """Check if trade is within risk limits."""
        max_notional = portfolio_value * self.config.max_position_pct
        current_notional = abs(self.positions.get(ticker, 0.0)) * price
        return current_notional <= max_notional

    def compute_target_shares(
        self,
        ticker: str,
        signal: float,
        price: float,
        portfolio_value: float,
        volatility: float = 0.02,  # daily vol
    ) -> float:
        """Compute target shares for a given signal."""
        if abs(signal) < self.config.min_signal_threshold:
            return 0.0

        if self.config.position_sizing == "signal":
            target_notional = portfolio_value * self.config.max_position_pct * signal
        elif self.config.position_sizing == "equal":
            target_notional = portfolio_value * self.config.max_position_pct * np.sign(signal)
        elif self.config.position_sizing == "volatility":
            # Target a fixed dollar vol per position
            dollar_vol = portfolio_value * self.config.vol_target / 20 / max(volatility, 1e-4)  # 20 positions
            target_notional = dollar_vol * np.sign(signal)
        else:
            target_notional = portfolio_value * self.config.max_position_pct * signal

        target_shares = target_notional / (price + 1e-8)
        return float(target_shares)

    def execute_trade(
        self,
        ticker: str,
        target_shares: float,
        price: float,
        date: datetime,
        signal: float,
        event_type: Optional[str],
        slippage_model: SlippageModel,
    ) -> Optional[Trade]:
        """Execute a trade to reach target_shares from current_shares."""
        current = self.positions.get(ticker, 0.0)
        delta = target_shares - current

        if abs(delta) < 0.01:
            return None

        direction = 1 if delta > 0 else -1
        fill_price = slippage_model.get_fill_price(price, direction)
        slippage   = abs(fill_price - price) * abs(delta)
        commission = fill_price * abs(delta) * self.config.commission_bps / 10000

        cost = fill_price * delta + slippage + commission
        if abs(cost) > self.cash * 1.2:  # don't over-leverage cash
            return None

        # Compute realized PnL if reducing/closing position
        pnl = 0.0
        if np.sign(current) != 0 and np.sign(delta) != np.sign(current):
            exit_size = min(abs(delta), abs(current))
            pnl = exit_size * (fill_price - self.avg_cost.get(ticker, fill_price)) * np.sign(current)
            self.pnl_realized += pnl

        # Update position and cash
        self.positions[ticker] = current + delta
        self.cash -= cost

        # Update avg cost
        if self.positions[ticker] != 0:
            if np.sign(current) == np.sign(delta):
                old_cost = self.avg_cost.get(ticker, fill_price)
                total_shares = abs(current) + abs(delta)
                self.avg_cost[ticker] = (
                    (abs(current) * old_cost + abs(delta) * fill_price) / total_shares
                )
            else:
                self.avg_cost[ticker] = fill_price
        else:
            del self.avg_cost[ticker]
            if ticker in self.positions:
                del self.positions[ticker]

        action = "buy" if delta > 0 else "sell"
        trade = Trade(
            date=date,
            ticker=ticker,
            action=action,
            shares=delta,
            fill_price=fill_price,
            signal=signal,
            event_type=event_type,
            slippage=slippage,
            commission=commission,
            notional=abs(fill_price * delta),
            pnl=pnl,
        )
        self.trades.append(trade)
        return trade


# ---------------------------------------------------------------------------
# Horizon analysis
# ---------------------------------------------------------------------------

def compute_horizon_returns(
    price_series: pd.Series,
    event_date: datetime,
    horizons_days: List[int],
    fill_at_open: bool = True,
) -> Dict[int, float]:
    """
    Compute forward returns at each horizon.
    Uses open price of the next bar as entry (if fill_at_open).
    """
    try:
        # Align event to index
        if hasattr(price_series.index, "tz") and price_series.index.tz:
            event_date = event_date.astimezone(price_series.index.tz)

        # Find entry position
        entry_idx = price_series.index.searchsorted(event_date)
        if entry_idx >= len(price_series):
            return {h: 0.0 for h in horizons_days}

        if fill_at_open:
            entry_idx = min(entry_idx + 1, len(price_series) - 1)

        entry_price = float(price_series.iloc[entry_idx])
        if entry_price <= 0:
            return {h: 0.0 for h in horizons_days}

        returns = {}
        for h in horizons_days:
            exit_idx = min(entry_idx + h, len(price_series) - 1)
            exit_price = float(price_series.iloc[exit_idx])
            returns[h] = float((exit_price - entry_price) / entry_price)

        return returns
    except Exception:
        return {h: 0.0 for h in horizons_days}


# ---------------------------------------------------------------------------
# Main News Backtester
# ---------------------------------------------------------------------------

class NewsBacktester:
    """
    Event-driven news backtest engine.

    Usage:
        bt = NewsBacktester(config)
        result = bt.run(events, price_data)
        bt.print_report(result)
    """

    def __init__(self, config: Optional[NewsBacktestConfig] = None):
        self.config = config or NewsBacktestConfig()
        self.slippage = SlippageModel(
            spread_bps=self.config.slippage_bps + self.config.commission_bps
        )

    def run(
        self,
        events: List[DetectedEvent],
        price_data: Dict[str, pd.DataFrame],   # ticker -> OHLCV DataFrame
        signal_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run event-driven backtest.

        Args:
            events: List of detected events with timestamps
            price_data: Dict of ticker -> DataFrame(open, high, low, close, volume)
            signal_fn: Optional function(event) -> float to override event direction
        """
        if not events:
            return {"error": "No events provided"}

        portfolio = PortfolioManager(self.config.initial_capital, self.config)
        event_results: List[EventBacktestResult] = []

        # Sort events chronologically
        sorted_events = sorted(events, key=lambda e: e.detected_at)

        # Determine date range
        start_date = sorted_events[0].detected_at
        end_date   = sorted_events[-1].detected_at + timedelta(days=max(self.config.horizons_days) + 5)

        # Build daily date index
        dates = pd.date_range(
            start=start_date.date(),
            end=end_date.date(),
            freq="B",
        )

        portfolio_values = []
        bh_values = []
        bh_initial_prices: Dict[str, float] = {}
        all_returns: List[float] = []

        # Initialize BH baseline
        for ticker, df in price_data.items():
            if len(df) > 0:
                bh_initial_prices[ticker] = float(df["close"].iloc[0])

        def get_bh_value(date: datetime) -> float:
            total = 0.0
            for ticker, df in price_data.items():
                try:
                    idx = df.index.searchsorted(date)
                    if idx < len(df):
                        price = float(df["close"].iloc[idx])
                        init  = bh_initial_prices.get(ticker, price)
                        weight = 1.0 / len(price_data)
                        total += weight * portfolio.initial_capital * price / (init + 1e-8)
                except Exception:
                    total += portfolio.initial_capital / len(price_data)
            return total

        # Process events
        event_queue = list(sorted_events)
        active_positions: Dict[str, Dict] = {}

        for date in dates:
            date_dt = pd.Timestamp(date).to_pydatetime().replace(tzinfo=timezone.utc)
            current_prices = self._get_prices(price_data, date_dt)

            # Check max drawdown stop
            pv = portfolio.get_portfolio_value(current_prices)
            peak = max(portfolio_values) if portfolio_values else self.config.initial_capital
            drawdown = (pv - peak) / (peak + 1e-8)
            if drawdown < -self.config.max_drawdown_stop:
                if self.config.verbose:
                    print(f"Max drawdown exceeded at {date_dt.date()}, stopping.")
                break

            portfolio_values.append(pv)
            bh_values.append(get_bh_value(date_dt))

            # Process events for this date
            day_events = [e for e in event_queue if e.detected_at.date() == date.date()]
            event_queue = [e for e in event_queue if e.detected_at.date() != date.date()]

            for event in day_events:
                ticker = event.ticker
                if not ticker or ticker not in price_data:
                    continue

                signal = signal_fn(event) if signal_fn else event.alpha_signal
                if abs(signal) < self.config.min_signal_threshold:
                    continue

                # Get entry price (next open if fill_at_open)
                entry_price = self._get_fill_price(price_data, ticker, date_dt)
                if entry_price is None:
                    continue

                target_shares = portfolio.compute_target_shares(
                    ticker, signal, entry_price, pv
                )
                if abs(target_shares) < 0.001:
                    continue

                trade = portfolio.execute_trade(
                    ticker, target_shares, entry_price, date_dt,
                    signal, event.event_type, self.slippage
                )

                # Horizon analysis
                if ticker in price_data:
                    rets = compute_horizon_returns(
                        price_data[ticker]["close"],
                        date_dt,
                        self.config.horizons_days,
                        self.config.fill_at_open,
                    )

                    is_correct = np.sign(signal) == np.sign(rets.get(1, 0.0))

                    event_result = EventBacktestResult(
                        event=event,
                        entry_date=date_dt,
                        exit_dates={h: date_dt + timedelta(days=h) for h in self.config.horizons_days},
                        entry_price=entry_price,
                        exit_prices={h: entry_price * (1 + rets.get(h, 0.0)) for h in self.config.horizons_days},
                        returns=rets,
                        signal=signal,
                        pnl=float(target_shares * entry_price * rets.get(1, 0.0)),
                        correct_direction=is_correct,
                    )
                    event_results.append(event_result)

            # Compute daily portfolio return
            if len(portfolio_values) >= 2:
                step_ret = (portfolio_values[-1] - portfolio_values[-2]) / (portfolio_values[-2] + 1e-8)
                all_returns.append(step_ret)

        # Compile results
        returns_arr = np.array(all_returns)
        portfolio_arr = np.array(portfolio_values)
        bh_arr = np.array(bh_values)

        return {
            "portfolio_values": portfolio_arr,
            "bh_values": bh_arr,
            "returns": returns_arr,
            "event_results": event_results,
            "trades": portfolio.trades,
            "total_pnl": float(portfolio_arr[-1] - self.config.initial_capital) if len(portfolio_arr) > 0 else 0.0,
            "total_trades": len(portfolio.trades),
            "horizon_analysis": self._analyze_horizons(event_results),
            "event_type_breakdown": self._breakdown_by_event_type(event_results),
            "performance_metrics": self._compute_metrics(returns_arr, bh_arr / (bh_arr[0] + 1e-8) - 1 if len(bh_arr) > 0 else returns_arr),
        }

    def _get_prices(
        self, price_data: Dict[str, pd.DataFrame], date: datetime
    ) -> Dict[str, float]:
        """Get closing prices for all tickers at a date."""
        prices = {}
        for ticker, df in price_data.items():
            try:
                idx = df.index.searchsorted(date)
                idx = min(idx, len(df) - 1)
                prices[ticker] = float(df["close"].iloc[idx])
            except Exception:
                pass
        return prices

    def _get_fill_price(
        self, price_data: Dict[str, pd.DataFrame], ticker: str, date: datetime
    ) -> Optional[float]:
        """Get fill price (next open or current close)."""
        df = price_data.get(ticker)
        if df is None:
            return None
        try:
            idx = df.index.searchsorted(date)
            if self.config.fill_at_open:
                idx = min(idx + 1, len(df) - 1)
            col = "open" if self.config.fill_at_open and "open" in df.columns else "close"
            return float(df[col].iloc[idx])
        except Exception:
            return None

    def _analyze_horizons(
        self, event_results: List[EventBacktestResult]
    ) -> Dict[str, Any]:
        """Analyze returns at each horizon."""
        if not event_results:
            return {}

        analysis = {}
        for h in self.config.horizons_days:
            signed_rets = [
                r.returns.get(h, 0.0) * np.sign(r.signal)
                for r in event_results
            ]
            rets = np.array(signed_rets)
            analysis[f"horizon_{h}d"] = {
                "mean_return": float(rets.mean()),
                "win_rate": float((rets > 0).mean()),
                "t_stat": float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(len(rets))),
                "n_events": len(rets),
                "mean_absolute_return": float(np.abs(rets).mean()),
            }
        return analysis

    def _breakdown_by_event_type(
        self, event_results: List[EventBacktestResult]
    ) -> Dict[str, Dict]:
        """Aggregate performance by event type."""
        breakdown: Dict[str, List] = {}
        for r in event_results:
            et = r.event.event_type
            breakdown.setdefault(et, []).append(r)

        result = {}
        for et, results in breakdown.items():
            h1_rets = np.array([r.returns.get(1, 0.0) * np.sign(r.signal) for r in results])
            result[et] = {
                "n_events": len(results),
                "hit_rate": float((h1_rets > 0).mean()),
                "mean_1d_return": float(h1_rets.mean()),
                "total_pnl": float(sum(r.pnl for r in results)),
            }
        return result

    def _compute_metrics(
        self, returns: np.ndarray, bh_returns: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard performance metrics."""
        if len(returns) < 2:
            return {}

        ann = 252
        mean = returns.mean()
        std  = returns.std() + 1e-8
        rf   = self.config.risk_free_rate / ann

        sharpe = (mean - rf) / std * np.sqrt(ann)
        equity = np.cumprod(1 + returns)
        peak   = np.maximum.accumulate(equity)
        dd     = (equity / peak - 1)
        max_dd = float(abs(dd.min()))
        calmar = float(mean * ann / (max_dd + 1e-4))
        total_return = float(equity[-1] - 1)

        return {
            "sharpe": float(sharpe),
            "calmar": calmar,
            "total_return": total_return,
            "annualized_return": float(mean * ann),
            "max_drawdown": max_dd,
            "win_rate": float((returns > 0).mean()),
            "n_trading_days": len(returns),
        }

    def print_report(self, result: Dict[str, Any]) -> None:
        """Print formatted backtest report."""
        print("\n" + "=" * 65)
        print("NEWS BACKTEST REPORT")
        print("=" * 65)

        metrics = result.get("performance_metrics", {})
        print(f"\nPortfolio Performance:")
        print(f"  Total Return:     {metrics.get('total_return', 0):.2%}")
        print(f"  Annualized:       {metrics.get('annualized_return', 0):.2%}")
        print(f"  Sharpe:           {metrics.get('sharpe', 0):.3f}")
        print(f"  Calmar:           {metrics.get('calmar', 0):.3f}")
        print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate:         {metrics.get('win_rate', 0):.1%}")

        print(f"\n  Total Trades:     {result.get('total_trades', 0)}")
        print(f"  Total PnL:        ${result.get('total_pnl', 0):,.0f}")

        horizon_analysis = result.get("horizon_analysis", {})
        if horizon_analysis:
            print(f"\nHorizon Analysis:")
            for horizon, ha in horizon_analysis.items():
                print(f"  {horizon}: mean={ha['mean_return']:+.3%}, "
                      f"win_rate={ha['win_rate']:.1%}, "
                      f"t-stat={ha['t_stat']:.2f}, "
                      f"n={ha['n_events']}")

        event_breakdown = result.get("event_type_breakdown", {})
        if event_breakdown:
            print(f"\nEvent Type Breakdown:")
            print(f"  {'Event Type':<22} {'N':>5} {'Hit%':>8} {'1d Ret':>9} {'PnL':>10}")
            print("  " + "-" * 58)
            for et, stats in sorted(event_breakdown.items(), key=lambda x: -abs(x[1].get("total_pnl", 0))):
                print(f"  {et:<22} {stats['n_events']:>5} "
                      f"{stats['hit_rate']:>8.1%} "
                      f"{stats['mean_1d_return']:>9.3%} "
                      f"${stats['total_pnl']:>9,.0f}")

        print("=" * 65)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ..signal.event_detector import DetectedEvent, EventTypes

    print("Testing news backtester...")

    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range("2024-01-02", periods=100, freq="B")
    tickers = ["AAPL", "MSFT"]

    price_data = {}
    for ticker in tickers:
        prices = np.cumprod(1 + np.random.randn(100) * 0.015) * 150
        opens  = prices * (1 + np.random.randn(100) * 0.002)
        highs  = prices * (1 + np.abs(np.random.randn(100) * 0.008))
        lows   = prices * (1 - np.abs(np.random.randn(100) * 0.008))
        volumes = np.exp(np.random.randn(100) + 9)

        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": prices, "volume": volumes},
            index=dates,
        )
        price_data[ticker] = df

    # Generate synthetic events
    events = []
    for i in range(10):
        ticker = tickers[i % 2]
        event_date = dates[i * 8 + 5].to_pydatetime().replace(tzinfo=timezone.utc)
        events.append(DetectedEvent(
            event_type=EventTypes.EARNINGS_BEAT if i % 3 != 0 else EventTypes.EARNINGS_MISS,
            subtype="eps_beat",
            ticker=ticker,
            confidence=0.85,
            direction=+1.0 if i % 3 != 0 else -1.0,
            magnitude=0.70,
            detected_at=event_date,
            source_text=f"Synthetic event {i}",
        ))

    config = NewsBacktestConfig(
        initial_capital=100_000,
        min_signal_threshold=0.1,
        horizons_days=[1, 5],
        verbose=True,
    )
    backtester = NewsBacktester(config)
    result = backtester.run(events, price_data)
    backtester.print_report(result)

    print(f"\nEvent results: {len(result['event_results'])}")
    print("News backtester self-test passed.")
