# ============================================================
# performance_attribution.py
# Comprehensive performance attribution for the trading system
# ============================================================

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

# ---- Configuration --------------------------------------------------------

TIMEFRAMES = ["15m", "1h", "4h"]
ASSET_CLASSES = ["crypto", "equity", "other"]
SESSIONS = ["asian", "london", "new_york", "after_hours"]
BH_TIMEFRAMES = ["15m", "1h", "4h"]


# ---- Data classes ---------------------------------------------------------

@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    asset_class: str          # 'crypto' | 'equity' | 'other'
    timeframe: str            # '15m' | '1h' | '4h'
    side: str                 # 'long' | 'short'
    entry_price: float
    exit_price: float
    qty: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    session: str              # 'asian' | 'london' | 'new_york' | 'after_hours'
    signal_source: str        # e.g. 'BH_15m', 'BH_1h', 'manual'
    fee: float = 0.0


@dataclass
class BHBAttribution:
    """Brinson-Hood-Beebower attribution: allocation + selection + interaction."""
    period: str
    category: str             # e.g. symbol or asset class
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float    # (wp - wb) × rb
    selection_effect: float     # wb × (rp - rb)
    interaction_effect: float   # (wp - wb) × (rp - rb)
    total_active_return: float


@dataclass
class TimeAttribution:
    """P&L attribution by time period."""
    session: str
    total_pnl: float
    trade_count: int
    win_rate: float
    avg_pnl: float
    pnl_pct_of_total: float


@dataclass
class InstrumentAttribution:
    symbol: str
    asset_class: str
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    trade_count: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    pnl_pct_of_total: float


@dataclass
class SignalAttribution:
    """P&L attributed to each signal timeframe / source."""
    signal_source: str
    timeframe: str
    total_pnl: float
    trade_count: int
    win_rate: float
    avg_pnl: float
    total_return_contribution: float


@dataclass
class RollingAttributionWindow:
    window: str               # '1W' | '1M' | '3M' | 'YTD'
    start_date: datetime
    end_date: datetime
    total_return: float
    alpha: float
    beta: float
    sharpe: float
    sortino: float
    max_drawdown: float
    instrument_attribution: list[InstrumentAttribution]
    signal_attribution: list[SignalAttribution]


@dataclass
class PerformanceTearsheet:
    generated_at: datetime
    total_return: float
    annualised_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_trade: TradeRecord | None
    worst_trade: TradeRecord | None
    bhb_attribution: list[BHBAttribution]
    instrument_attribution: list[InstrumentAttribution]
    signal_attribution: list[SignalAttribution]
    time_attribution: list[TimeAttribution]
    rolling_windows: list[RollingAttributionWindow]


# ---- BHB Attribution engine ----------------------------------------------

def _brinson_attribution(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
    portfolio_returns: dict[str, float],
    benchmark_returns: dict[str, float],
    period: str = "period",
) -> list[BHBAttribution]:
    """Compute Brinson-Hood-Beebower attribution for each category."""
    all_cats = set(portfolio_weights) | set(benchmark_weights)
    results = []

    for cat in all_cats:
        wp = portfolio_weights.get(cat, 0.0)
        wb = benchmark_weights.get(cat, 0.0)
        rp = portfolio_returns.get(cat, 0.0)
        rb = benchmark_returns.get(cat, 0.0)

        alloc = (wp - wb) * rb
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)
        total = alloc + selection + interaction

        results.append(BHBAttribution(
            period=period,
            category=cat,
            portfolio_weight=wp,
            benchmark_weight=wb,
            portfolio_return=rp,
            benchmark_return=rb,
            allocation_effect=alloc,
            selection_effect=selection,
            interaction_effect=interaction,
            total_active_return=total,
        ))

    return sorted(results, key=lambda x: abs(x.total_active_return), reverse=True)


# ---- Performance metrics helpers -----------------------------------------

def _sharpe(returns: np.ndarray, ann: int = 252) -> float:
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(ann))


def _sortino(returns: np.ndarray, ann: int = 252) -> float:
    neg = returns[returns < 0]
    if len(neg) < 2 or np.std(neg) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(neg) * np.sqrt(ann))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(np.min(dd))


def _profit_factor(pnls: list[float]) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else float("inf")


def _session_from_time(dt: datetime) -> str:
    h = dt.hour
    if 0 <= h < 8:
        return "asian"
    elif 8 <= h < 13:
        return "london"
    elif 13 <= h < 21:
        return "new_york"
    else:
        return "after_hours"


# ---- Main engine ---------------------------------------------------------

class PerformanceAttributionEngine:
    """
    Comprehensive performance attribution system.

    Features:
    - Brinson-Hood-Beebower (BHB): allocation / selection / interaction
    - Holdings-based attribution: daily return decomposition
    - Trade attribution: which trades drove P&L
    - BH signal attribution: P&L per timeframe (15m/1h/4h)
    - Instrument attribution: crypto vs equity
    - Time attribution: session, hour-of-day
    - Rolling attribution windows: 1W, 1M, 3M, YTD
    - Full tearsheet (PDF via matplotlib)
    """

    def __init__(
        self,
        symbols: list[str],
        asset_class_map: dict[str, str] | None = None,
        benchmark_weights: dict[str, float] | None = None,
        portfolio_value: float = 1_000_000.0,
    ):
        self.symbols = list(symbols)
        self.asset_class_map = asset_class_map or {s: "crypto" for s in symbols}
        self.benchmark_weights = benchmark_weights or {s: 1.0 / len(symbols) for s in symbols}
        self.portfolio_value = portfolio_value

        self._trades: list[TradeRecord] = []
        self._daily_returns: list[tuple[datetime, float]] = []  # (ts, portfolio_return)
        self._position_returns: list[tuple[datetime, dict[str, float]]] = []  # (ts, {sym: ret})
        self._equity_curve: list[float] = [portfolio_value]

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)

    def update_returns(
        self,
        timestamp: datetime,
        portfolio_return: float,
        symbol_returns: dict[str, float],
        current_weights: dict[str, float] | None = None,
    ) -> None:
        """Feed daily/bar return data."""
        self._daily_returns.append((timestamp, portfolio_return))
        self._position_returns.append((timestamp, symbol_returns))
        last_equity = self._equity_curve[-1]
        self._equity_curve.append(last_equity * (1 + portfolio_return))

    # ------------------------------------------------------------------
    # Attribution methods
    # ------------------------------------------------------------------

    def trade_attribution(self) -> list[tuple[TradeRecord, float]]:
        """Sort trades by P&L contribution, return list of (trade, contribution_pct)."""
        if not self._trades:
            return []
        total_pnl = sum(t.pnl for t in self._trades)
        if total_pnl == 0:
            return [(t, 0.0) for t in self._trades]
        result = [(t, t.pnl / total_pnl * 100) for t in self._trades]
        return sorted(result, key=lambda x: x[1], reverse=True)

    def instrument_attribution(
        self, trades: list[TradeRecord] | None = None
    ) -> list[InstrumentAttribution]:
        """Per-symbol P&L attribution."""
        trades = trades or self._trades
        by_sym: dict[str, list[TradeRecord]] = defaultdict(list)
        for t in trades:
            by_sym[t.symbol].append(t)

        total_pnl = sum(t.pnl for t in trades)
        results = []

        for sym, sym_trades in by_sym.items():
            pnls = [t.pnl for t in sym_trades]
            returns = [t.pnl_pct for t in sym_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            total = sum(pnls)

            arr = np.array(returns)
            sharpe = _sharpe(arr, ann=252) if len(arr) > 1 else 0.0

            results.append(InstrumentAttribution(
                symbol=sym,
                asset_class=self.asset_class_map.get(sym, "other"),
                total_pnl=total,
                realized_pnl=total,
                unrealized_pnl=0.0,
                trade_count=len(sym_trades),
                win_rate=len(wins) / max(len(pnls), 1),
                avg_return=float(np.mean(returns)) if returns else 0.0,
                sharpe_ratio=sharpe,
                pnl_pct_of_total=total / total_pnl * 100 if total_pnl != 0 else 0.0,
            ))

        return sorted(results, key=lambda x: x.total_pnl, reverse=True)

    def signal_attribution(
        self, trades: list[TradeRecord] | None = None
    ) -> list[SignalAttribution]:
        """P&L attribution by signal source / timeframe."""
        trades = trades or self._trades
        by_signal: dict[str, list[TradeRecord]] = defaultdict(list)
        for t in trades:
            key = t.signal_source
            by_signal[key].append(t)

        total_pnl = sum(t.pnl for t in trades)
        results = []

        for sig, sig_trades in by_signal.items():
            pnls = [t.pnl for t in sig_trades]
            total = sum(pnls)
            wins = [p for p in pnls if p > 0]
            tf = sig_trades[0].timeframe if sig_trades else "unknown"

            results.append(SignalAttribution(
                signal_source=sig,
                timeframe=tf,
                total_pnl=total,
                trade_count=len(sig_trades),
                win_rate=len(wins) / max(len(pnls), 1),
                avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
                total_return_contribution=total / total_pnl * 100 if total_pnl != 0 else 0.0,
            ))

        return sorted(results, key=lambda x: x.total_pnl, reverse=True)

    def time_attribution(
        self, trades: list[TradeRecord] | None = None
    ) -> list[TimeAttribution]:
        """P&L attribution by trading session."""
        trades = trades or self._trades
        by_session: dict[str, list[TradeRecord]] = defaultdict(list)
        for t in trades:
            by_session[t.session].append(t)

        total_pnl = sum(t.pnl for t in trades)
        results = []

        for session in SESSIONS:
            sess_trades = by_session.get(session, [])
            if not sess_trades:
                continue
            pnls = [t.pnl for t in sess_trades]
            wins = [p for p in pnls if p > 0]
            total = sum(pnls)
            results.append(TimeAttribution(
                session=session,
                total_pnl=total,
                trade_count=len(sess_trades),
                win_rate=len(wins) / max(len(pnls), 1),
                avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
                pnl_pct_of_total=total / total_pnl * 100 if total_pnl != 0 else 0.0,
            ))

        return sorted(results, key=lambda x: x.total_pnl, reverse=True)

    def bhb_attribution(
        self,
        period: str = "since_inception",
    ) -> list[BHBAttribution]:
        """Brinson-Hood-Beebower attribution by symbol."""
        if not self._position_returns:
            return []

        # Aggregate portfolio and benchmark returns by symbol
        portfolio_returns_by_sym: dict[str, list[float]] = defaultdict(list)
        for ts, sym_rets in self._position_returns:
            for s, r in sym_rets.items():
                portfolio_returns_by_sym[s].append(r)

        pf_return = {s: float(np.mean(v)) for s, v in portfolio_returns_by_sym.items()}
        bm_return = {s: pf_return.get(s, 0.0) * 0.9 for s in self.symbols}  # simple BM proxy

        # Equal portfolio weight
        pf_weight = {s: 1.0 / len(self.symbols) for s in self.symbols}

        return _brinson_attribution(pf_weight, self.benchmark_weights, pf_return, bm_return, period)

    def rolling_attribution(self, window: str = "1M") -> RollingAttributionWindow:
        """Compute attribution over a rolling window."""
        now = datetime.now(tz=timezone.utc)
        if window == "1W":
            cutoff = now - pd.Timedelta(days=7)
        elif window == "1M":
            cutoff = now - pd.Timedelta(days=30)
        elif window == "3M":
            cutoff = now - pd.Timedelta(days=90)
        else:  # YTD
            cutoff = now.replace(month=1, day=1, hour=0, minute=0, second=0)

        cutoff = cutoff.replace(tzinfo=timezone.utc) if cutoff.tzinfo is None else cutoff

        filtered_trades = [
            t for t in self._trades
            if t.exit_time >= cutoff
        ]
        filtered_returns = [
            (ts, r) for ts, r in self._daily_returns
            if ts >= cutoff
        ]

        rets = np.array([r for _, r in filtered_returns])
        total_return = float(np.prod(1 + rets) - 1) if len(rets) > 0 else 0.0
        sharpe = _sharpe(rets)
        sortino = _sortino(rets)
        eq_curve = np.cumprod(1 + rets) if len(rets) > 0 else np.array([1.0])
        mdd = _max_drawdown(eq_curve)
        alpha = float(np.mean(rets) * 252) if len(rets) > 0 else 0.0
        beta = 1.0  # placeholder; would need benchmark series

        start_date = cutoff
        end_date = now

        return RollingAttributionWindow(
            window=window,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            alpha=alpha,
            beta=beta,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=mdd,
            instrument_attribution=self.instrument_attribution(filtered_trades),
            signal_attribution=self.signal_attribution(filtered_trades),
        )

    # ------------------------------------------------------------------
    # Full tearsheet
    # ------------------------------------------------------------------

    def generate_tearsheet(self) -> PerformanceTearsheet:
        """Compute all attribution metrics for the full tearsheet."""
        pnls = [t.pnl for t in self._trades]
        returns = np.array([r for _, r in self._daily_returns])
        equity = np.array(self._equity_curve)

        total_return = float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0
        n_years = max(len(returns) / 252, 1e-4)
        ann_return = float((1 + total_return) ** (1 / n_years) - 1)
        sharpe = _sharpe(returns)
        sortino = _sortino(returns)
        mdd = _max_drawdown(equity)
        calmar = ann_return / abs(mdd) if mdd != 0 else 0.0

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / max(len(pnls), 1)
        pf = _profit_factor(pnls)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        best = max(self._trades, key=lambda t: t.pnl, default=None)
        worst = min(self._trades, key=lambda t: t.pnl, default=None)

        bhb = self.bhb_attribution()
        instr = self.instrument_attribution()
        signal = self.signal_attribution()
        time_attr = self.time_attribution()
        rolling = [self.rolling_attribution(w) for w in ["1W", "1M", "3M", "YTD"]]

        return PerformanceTearsheet(
            generated_at=datetime.now(tz=timezone.utc),
            total_return=total_return,
            annualised_return=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=mdd,
            win_rate=win_rate,
            profit_factor=pf,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best,
            worst_trade=worst,
            bhb_attribution=bhb,
            instrument_attribution=instr,
            signal_attribution=signal,
            time_attribution=time_attr,
            rolling_windows=rolling,
        )

    # ------------------------------------------------------------------
    # PDF tearsheet export
    # ------------------------------------------------------------------

    def export_pdf_tearsheet(
        self,
        output_path: str = "tearsheet.pdf",
        tearsheet: PerformanceTearsheet | None = None,
    ) -> None:
        """Export multi-page PDF tearsheet via matplotlib."""
        if not HAS_MPL:
            logger.warning("matplotlib not available; cannot export PDF")
            return

        if tearsheet is None:
            tearsheet = self.generate_tearsheet()

        with PdfPages(output_path) as pdf:
            # --- Page 1: Summary metrics ---
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle("Performance Attribution Tearsheet", fontsize=16, fontweight="bold")
            gs = gridspec.GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            metrics = {
                "Total Return": f"{tearsheet.total_return * 100:.2f}%",
                "Ann. Return": f"{tearsheet.annualised_return * 100:.2f}%",
                "Sharpe": f"{tearsheet.sharpe_ratio:.3f}",
                "Sortino": f"{tearsheet.sortino_ratio:.3f}",
                "Calmar": f"{tearsheet.calmar_ratio:.3f}",
                "Max DD": f"{tearsheet.max_drawdown * 100:.2f}%",
                "Win Rate": f"{tearsheet.win_rate * 100:.1f}%",
                "Profit Factor": f"{tearsheet.profit_factor:.2f}",
            }
            ax1.axis("off")
            table_data = list(metrics.items())
            ax1.table(cellText=table_data, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
            ax1.set_title("Summary Statistics")

            # Equity curve
            ax2 = fig.add_subplot(gs[0, 1])
            if len(self._equity_curve) > 1:
                ax2.plot(self._equity_curve, color="#2196F3", linewidth=1.5)
                ax2.fill_between(range(len(self._equity_curve)), self._equity_curve,
                                 self._equity_curve[0], alpha=0.1, color="#2196F3")
            ax2.set_title("Equity Curve")
            ax2.set_xlabel("Bars")
            ax2.set_ylabel("Portfolio Value ($)")
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax2.grid(alpha=0.3)

            # Instrument attribution
            ax3 = fig.add_subplot(gs[1, 0])
            if tearsheet.instrument_attribution:
                ia = tearsheet.instrument_attribution[:10]
                syms = [x.symbol for x in ia]
                pnls = [x.total_pnl for x in ia]
                colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
                ax3.barh(syms, pnls, color=colors)
                ax3.axvline(0, color="black", linewidth=0.5)
                ax3.set_title("P&L by Instrument")
                ax3.set_xlabel("P&L ($)")
                ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

            # Signal attribution
            ax4 = fig.add_subplot(gs[1, 1])
            if tearsheet.signal_attribution:
                sa = tearsheet.signal_attribution[:8]
                sigs = [x.signal_source for x in sa]
                spnls = [x.total_pnl for x in sa]
                scolors = ["#4CAF50" if p > 0 else "#F44336" for p in spnls]
                ax4.barh(sigs, spnls, color=scolors)
                ax4.axvline(0, color="black", linewidth=0.5)
                ax4.set_title("P&L by Signal Source")
                ax4.set_xlabel("P&L ($)")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # --- Page 2: Session attribution + BHB ---
            fig2 = plt.figure(figsize=(11, 8.5))
            gs2 = gridspec.GridSpec(2, 2, figure=fig2)

            ax5 = fig2.add_subplot(gs2[0, 0])
            if tearsheet.time_attribution:
                sessions = [x.session for x in tearsheet.time_attribution]
                session_pnls = [x.total_pnl for x in tearsheet.time_attribution]
                ax5.pie(
                    [abs(p) for p in session_pnls],
                    labels=sessions,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax5.set_title("P&L by Trading Session")

            ax6 = fig2.add_subplot(gs2[0, 1])
            if tearsheet.bhb_attribution:
                bhb_items = tearsheet.bhb_attribution[:8]
                cats = [x.category for x in bhb_items]
                alloc_eff = [x.allocation_effect * 100 for x in bhb_items]
                sel_eff = [x.selection_effect * 100 for x in bhb_items]
                inter_eff = [x.interaction_effect * 100 for x in bhb_items]
                x = np.arange(len(cats))
                w = 0.25
                ax6.bar(x - w, alloc_eff, w, label="Allocation", color="#2196F3")
                ax6.bar(x, sel_eff, w, label="Selection", color="#4CAF50")
                ax6.bar(x + w, inter_eff, w, label="Interaction", color="#FF9800")
                ax6.set_xticks(x)
                ax6.set_xticklabels(cats, rotation=45, ha="right")
                ax6.set_title("BHB Attribution")
                ax6.legend()
                ax6.set_ylabel("Active Return (%)")

            # Rolling Sharpe
            ax7 = fig2.add_subplot(gs2[1, :])
            roll_windows = tearsheet.rolling_windows
            if roll_windows:
                labels = [rw.window for rw in roll_windows]
                sharpes = [rw.sharpe for rw in roll_windows]
                sorinos = [rw.sortino for rw in roll_windows]
                x = np.arange(len(labels))
                ax7.bar(x - 0.2, sharpes, 0.4, label="Sharpe", color="#2196F3")
                ax7.bar(x + 0.2, sorinos, 0.4, label="Sortino", color="#4CAF50")
                ax7.set_xticks(x)
                ax7.set_xticklabels(labels)
                ax7.axhline(1.0, color="orange", linestyle="--", label="Target Sharpe=1")
                ax7.set_title("Rolling Risk-Adjusted Performance")
                ax7.legend()
                ax7.set_ylabel("Ratio")

            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        logger.info("Tearsheet exported to %s", output_path)

    def summary(self) -> dict[str, Any]:
        pnls = [t.pnl for t in self._trades]
        return {
            "n_trades": len(self._trades),
            "total_pnl": sum(pnls),
            "win_rate": len([p for p in pnls if p > 0]) / max(len(pnls), 1),
            "profit_factor": _profit_factor(pnls),
            "n_daily_returns": len(self._daily_returns),
        }


# ---- Standalone test -------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)

    symbols = ["BTC", "ETH", "SOL", "BNB", "AVAX"]
    engine = PerformanceAttributionEngine(
        symbols,
        asset_class_map={s: "crypto" for s in symbols},
    )

    now = datetime.now(tz=timezone.utc)

    # Simulate 200 trades across timeframes and symbols
    for i in range(200):
        sym = rng.choice(symbols)
        tf = rng.choice(BH_TIMEFRAMES)
        side = rng.choice(["long", "short"])
        entry = float(rng.uniform(100, 60000))
        exit_p = entry * (1 + float(rng.normal(0.002, 0.03)))
        qty = float(rng.uniform(0.1, 2.0))
        pnl = (exit_p - entry) * qty if side == "long" else (entry - exit_p) * qty
        dt = now - pd.Timedelta(days=float(rng.uniform(0, 90)))

        engine.add_trade(TradeRecord(
            trade_id=f"T{i:04d}",
            symbol=sym,
            asset_class="crypto",
            timeframe=tf,
            side=side,
            entry_price=entry,
            exit_price=exit_p,
            qty=qty,
            entry_time=dt,
            exit_time=dt + pd.Timedelta(hours=float(rng.uniform(1, 24))),
            pnl=pnl,
            pnl_pct=(exit_p / entry - 1) * 100,
            session=_session_from_time(dt),
            signal_source=f"BH_{tf}",
        ))

    # Simulate daily returns
    for day in range(90):
        ts = now - pd.Timedelta(days=90 - day)
        r = float(rng.normal(0.0008, 0.015))
        sym_rets = {s: float(rng.normal(0.0005, 0.02)) for s in symbols}
        engine.update_returns(ts, r, sym_rets)

    ts = engine.generate_tearsheet()
    print(f"Total return: {ts.total_return*100:.2f}%  Sharpe: {ts.sharpe_ratio:.3f}  MDD: {ts.max_drawdown*100:.2f}%")
    print("Top 3 instruments:", [(ia.symbol, f"${ia.total_pnl:.0f}") for ia in ts.instrument_attribution[:3]])
    print("Signal attribution:", [(sa.signal_source, f"${sa.total_pnl:.0f}") for sa in ts.signal_attribution])
    engine.export_pdf_tearsheet("test_tearsheet.pdf", ts)
    print("PDF tearsheet exported.")
