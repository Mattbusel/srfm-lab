"""
backtest_v12.py  —  QC-Equivalent local backtest of LARSA v12

Matches QuantConnect behavior:
  - SimulatedBroker with IB margin model (ES $13,200 / NQ $19,500 / YM $10,500 maint)
  - set_holdings(sym, pct): pct = fraction of total portfolio value
    num_contracts = floor(pct × equity / (price × multiplier))
  - $150 round-trip fee per contract ($75 each way)
  - Margin call if margin_used > 0.5 × equity  → liquidate to 50%
  - Full QC stats: Total Return, Annual Return, Sharpe, Sortino, MaxDD,
    Win Rate (trade-level), Total Trades, Total Fees, Profit Factor, Avg Win, Avg Loss
  - BULL REGIME DEEP DIVE section
  - Saves results/backtest_v12.json  +  results/backtest_v12.png

Usage:
  python tools/backtest_v12.py [--strategy larsa-v12]
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT    = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA    = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Strategy constants (identical to main.py) ────────────────────────────────
CF_1H  = {"ES": 0.001,  "NQ": 0.0012, "YM": 0.0008}
CF_15M = {"ES": 0.0003, "NQ": 0.0004, "YM": 0.00025}
CF_1D  = {"ES": 0.005,  "NQ": 0.006,  "YM": 0.004}
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
MIN_HOLD      = 4
N_INST        = 3
INST_CORR     = 0.90
PORT_RISK     = 0.01
CORR_FACTOR   = math.sqrt(N_INST + N_INST * (N_INST - 1) * INST_CORR)  # 2.898
PER_INST_RISK = PORT_RISK / CORR_FACTOR                                  # ≈ 0.003450
START_EQUITY  = 1_000_000.0
SYMS          = ["ES", "NQ", "YM"]
BULL_CF_SCALE = 3.0

# ── IB margin model ───────────────────────────────────────────────────────────
MAINT_MARGIN = {"ES": 13_200, "NQ": 19_500, "YM": 10_500}   # $ per contract
MULTIPLIER   = {"ES": 50,     "NQ": 20,     "YM": 5}         # $ per point

FEE_ONE_WAY  = 75   # $ per contract one-way  ($150 round-trip)


# ═══════════════════════════════════════════════════════════════════════════════
# Indicator helpers
# ═══════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, period):
        self.k = 2.0 / (period + 1)
        self.v = None

    def update(self, x):
        self.v = x if self.v is None else self.v + self.k * (x - self.v)
        return self.v or 0.0

    @property
    def value(self):
        return self.v or 0.0


class WilderATR:
    def __init__(self, period=14):
        self.k = 1.0 / period
        self.v = None
        self._pc = None

    def update(self, h, l, c):
        if self._pc is None:
            self._pc = c
            return 0.0
        tr = max(h - l, abs(h - self._pc), abs(l - self._pc))
        self.v = tr if self.v is None else self.v + self.k * (tr - self.v)
        self._pc = c
        return self.v or 0.0

    @property
    def value(self):
        return self.v or 0.0


class ADX:
    def __init__(self, p=14):
        self.k = 1.0 / p
        self._pDM = self._nDM = self._TR = self._ADX = 0.0
        self._ph = self._pl = self._pc = None

    def update(self, h, l, c):
        if self._pc is None:
            self._ph, self._pl, self._pc = h, l, c
            return 0.0
        ph, pl, pc = self._ph, self._pl, self._pc
        tr  = max(h - l, abs(h - pc), abs(l - pc))
        pdm = max(h - ph, 0) if (h - ph) > (pl - l) else 0.0
        ndm = max(pl - l, 0) if (pl - l) > (h - ph) else 0.0
        self._pDM = self._pDM * (1 - self.k) + pdm * self.k
        self._nDM = self._nDM * (1 - self.k) + ndm * self.k
        self._TR  = self._TR  * (1 - self.k) + tr  * self.k
        if self._TR > 0:
            pDI = 100 * self._pDM / self._TR
            nDI = 100 * self._nDM / self._TR
            d   = pDI + nDI
            dx  = 100 * abs(pDI - nDI) / d if d > 0 else 0.0
            self._ADX = self._ADX * (1 - self.k) + dx * self.k
        self._ph, self._pl, self._pc = h, l, c
        return self._ADX

    @property
    def value(self):
        return self._ADX


# ═══════════════════════════════════════════════════════════════════════════════
# Black-hole accumulator
# ═══════════════════════════════════════════════════════════════════════════════
class BH:
    def __init__(self, cf, warmup_bars):
        self.cf       = cf
        self.wu       = warmup_bars
        self.mass     = 0.0
        self.ctl      = 0
        self.active   = False
        self.direction= 0
        self.prices   = []
        self.bc       = 0
        self.bh_decay = 0.95

    def update(self, c, cf_scale=1.0):
        self.bc += 1
        self.prices.append(c)
        if len(self.prices) < 2:
            return
        eff_cf = self.cf * cf_scale
        beta   = abs(c - self.prices[-2]) / (self.prices[-2] + 1e-9) / (eff_cf + 1e-9)
        was    = self.active
        if beta < 1.0:
            self.ctl += 1
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.mass = self.mass * 0.97 + 0.03 * 1.0 * sb
        else:
            self.ctl = 0
            self.mass *= self.bh_decay

        form_thresh  = 1.0 if was else 1.5
        self.active  = self.mass > form_thresh and self.ctl >= 3

        if not was and self.active:
            lb = min(20, len(self.prices) - 1)
            self.direction = 1 if c > self.prices[-1 - lb] else -1
        elif was and not self.active:
            self.direction = 0

        if self.bc < self.wu:
            self.active    = False
            self.direction = 0

    def get_direction(self):
        if self.direction != 0:
            return self.direction
        if len(self.prices) >= 5:
            return 1 if self.prices[-1] > self.prices[-5] else -1
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Regime detector (mirrors main.py FutureInstrument.detect_regime)
# ═══════════════════════════════════════════════════════════════════════════════
REGIME_BULL     = "BULL"
REGIME_BEAR     = "BEAR"
REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_HIGH_VOL = "HIGH_VOL"


class RegimeDetector:
    def __init__(self):
        self.e12  = EMA(12);  self.e26 = EMA(26)
        self.e50  = EMA(50);  self.e200 = EMA(200)
        self.adx  = ADX(14)
        self.atr  = WilderATR(14)
        self.atr_hist = []
        self.regime   = REGIME_SIDEWAYS
        self.cf_scale = 1.0
        self.rhb      = 0     # bars since regime change
        self.bc       = 0

    def update(self, o, h, l, c):
        self.bc += 1
        e12  = self.e12.update(c)
        e26  = self.e26.update(c)
        e50  = self.e50.update(c)
        e200 = self.e200.update(c)
        adx  = self.adx.update(h, l, c)
        atr  = self.atr.update(h, l, c)

        self.atr_hist.append(atr)
        if len(self.atr_hist) > 50:
            self.atr_hist.pop(0)

        if self.bc < 200:   # EMA200 needs history
            return

        atr_ratio = 1.0
        if len(self.atr_hist) >= 20:
            mean_atr  = float(np.mean(self.atr_hist))
            atr_ratio = atr / mean_atr if mean_atr > 0 else 1.0

        self.rhb += 1
        if atr_ratio >= 1.5:
            nr = REGIME_HIGH_VOL
        elif c > e200 and e12 > e26:
            full_stack = e12 > e26 > e50 > e200
            nr = REGIME_BULL if adx > (14 if full_stack else 18) else REGIME_SIDEWAYS
        elif c < e200 and e12 < e26:
            full_stack = e200 > e50 > e26 > e12
            nr = REGIME_BEAR if adx > (14 if full_stack else 18) else REGIME_SIDEWAYS
        else:
            nr = REGIME_SIDEWAYS

        if nr != self.regime:
            self.rhb   = 0
            self.regime = nr

        # v12: CF×3 in BULL so typical bull moves are TIMELIKE
        self.cf_scale = BULL_CF_SCALE if self.regime == REGIME_BULL else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Per-instrument state
# ═══════════════════════════════════════════════════════════════════════════════
class Instrument:
    def __init__(self, sym):
        self.sym        = sym
        self.bh_15m     = BH(CF_15M[sym], warmup_bars=400)
        self.bh_1h      = BH(CF_1H[sym],  warmup_bars=120)
        self.bh_1d      = BH(CF_1D[sym],  warmup_bars=30)
        self.regime     = RegimeDetector()
        self.atr_1h     = WilderATR(14)
        self.last_target= 0.0      # fraction of equity (QC set_holdings pct)
        self.bars_held  = 0

    def tf_score(self):
        return (4 * int(self.bh_1d.active) +
                2 * int(self.bh_1h.active) +
                int(self.bh_15m.active))

    def direction(self):
        if self.bh_1d.active:  return self.bh_1d.get_direction()
        if self.bh_1h.active:  return self.bh_1h.get_direction()
        if self.bh_15m.active: return self.bh_15m.get_direction()
        return 0

    def tf_label(self):
        if self.bh_1d.active:  return "1d"
        if self.bh_1h.active:  return "1h"
        if self.bh_15m.active: return "15m"
        return "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Simulated Broker  (IB margin model, $75/contract one-way)
# ═══════════════════════════════════════════════════════════════════════════════
class SimulatedBroker:
    """
    Futures broker with daily mark-to-market settlement.

    In futures accounting:
      - Positions are settled to cash each bar (MTM P&L goes straight into cash).
      - equity() == cash  (there is no separate unrealised balance after settlement).
      - Margin used = sum(|contracts| × maintenance_margin_per_contract)
      - Notional = contracts × price × multiplier  (for sizing calcs only)

    set_holdings(sym, pct):
      pct is a fraction of total equity.
      num_contracts = floor(|pct| × equity / (price × multiplier)), sign from pct.
    """

    def __init__(self, start_cash):
        self.cash       = start_cash          # settled equity
        self.contracts  = {s: 0  for s in SYMS}
        self.prices     = {s: 0.0 for s in SYMS}
        self.total_fees = 0.0

    # ── mark-to-market (called BEFORE execution each bar) ────────────────────
    def mark_to_market(self, new_prices: dict) -> float:
        """Settle price changes into cash; return bar P&L in $."""
        pnl = 0.0
        for s in SYMS:
            old_p = self.prices[s]
            new_p = new_prices[s]
            if old_p > 0 and self.contracts[s] != 0:
                pnl += self.contracts[s] * (new_p - old_p) * MULTIPLIER[s]
            self.prices[s] = new_p
        self.cash += pnl
        return pnl

    def equity(self) -> float:
        """After MTM settlement, equity == cash."""
        return self.cash

    def margin_used(self) -> float:
        return sum(abs(self.contracts[s]) * MAINT_MARGIN[s] for s in SYMS)

    def excess_liquidity(self) -> float:
        return self.equity() - self.margin_used()

    # ── set_holdings (QC semantics) ───────────────────────────────────────────
    def set_holdings(self, sym: str, pct: float, equity_override: float = None) -> float:
        """
        Target pct of equity in sym.  Returns fee charged ($).
        """
        eq    = equity_override if equity_override is not None else self.equity()
        price = self.prices[sym]
        if price <= 0:
            return 0.0
        mult             = MULTIPLIER[sym]
        target_contracts = (
            int(math.copysign(math.floor(abs(pct) * eq / (price * mult)), pct))
            if pct != 0 else 0
        )
        delta = abs(target_contracts - self.contracts[sym])
        if delta == 0:
            return 0.0
        fee               = delta * FEE_ONE_WAY
        self.cash        -= fee
        self.total_fees  += fee
        self.contracts[sym] = target_contracts
        return fee

    # ── margin call ───────────────────────────────────────────────────────────
    def check_margin_call(self):
        """Liquidate if margin_used > 50% of equity."""
        eq = self.equity()
        if eq <= 0:
            return
        mu = self.margin_used()
        if mu <= 0.5 * eq:
            return
        # Reduce each position proportionally
        target_ratio = 0.5 * eq / mu   # scale so total margin = 50% equity
        for s in SYMS:
            old = self.contracts[s]
            if old == 0:
                continue
            new_abs = int(abs(old) * target_ratio)
            new_c   = int(math.copysign(new_abs, old))
            delta   = abs(new_c - old)
            if delta == 0:
                continue
            fee              = delta * FEE_ONE_WAY
            self.cash       -= fee
            self.total_fees += fee
            self.contracts[s] = new_c


# ═══════════════════════════════════════════════════════════════════════════════
# Trade tracker
# ═══════════════════════════════════════════════════════════════════════════════
class TradeRecord:
    __slots__ = ["sym", "direction", "tf_label", "regime",
                 "open_bar", "close_bar", "open_price", "close_price",
                 "contracts", "pnl", "hold_bars"]

    def __init__(self, sym, direction, tf_label, regime, bar_idx, price, contracts):
        self.sym        = sym
        self.direction  = direction   # +1 / -1
        self.tf_label   = tf_label
        self.regime     = regime
        self.open_bar   = bar_idx
        self.close_bar  = None
        self.open_price = price
        self.close_price= None
        self.contracts  = contracts
        self.pnl        = None
        self.hold_bars  = None


# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
def load_data():
    dfs = {}
    for sym in SYMS:
        path = os.path.join(DATA, f"{sym}_hourly_real.csv")
        df   = pd.read_csv(path, parse_dates=["date"])
        df   = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        dfs[sym] = df
    common = set(dfs["ES"]["date"])
    for sym in SYMS:
        common &= set(dfs[sym]["date"])
    common = sorted(common)
    aligned = {}
    for sym in SYMS:
        aligned[sym] = dfs[sym].set_index("date").loc[common].reset_index()
    return aligned, common


# ═══════════════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════════════
def annual_return(total_ret, n_bars, bars_per_year=6.5 * 252):
    """CAGR from total return fraction."""
    years = n_bars / bars_per_year
    if years <= 0:
        return 0.0
    return (1 + total_ret) ** (1 / years) - 1


def sharpe(daily_rets):
    dr = np.array(daily_rets)
    if len(dr) < 2 or dr.std() == 0:
        return 0.0
    return float(dr.mean() / dr.std() * math.sqrt(252))


def sortino(daily_rets):
    dr  = np.array(daily_rets)
    neg = dr[dr < 0]
    if len(neg) < 2 or neg.std() == 0:
        return 0.0
    return float(dr.mean() / neg.std() * math.sqrt(252))


def max_drawdown(equity_arr):
    pk  = equity_arr[0]
    mdd = 0.0
    for v in equity_arr:
        pk  = max(pk, v)
        mdd = max(mdd, (pk - v) / pk)
    return mdd


# ═══════════════════════════════════════════════════════════════════════════════
# Main backtest
# ═══════════════════════════════════════════════════════════════════════════════
def run(strategy="larsa-v12", start_equity=None):
    equity0 = float(start_equity) if start_equity is not None else START_EQUITY

    banner = "═" * 65
    print(banner)
    print(f"  LARSA v12 — QC-Equivalent Local Backtest  (equity=${equity0:,.0f})")
    print("  Data: Nov 2023 → Apr 2026  (ES/NQ/YM hourly)")
    print(banner)

    data, timestamps = load_data()
    n = len(timestamps)
    print(f"\n  Loaded {n:,} aligned hourly bars per symbol")
    print(f"  Range: {timestamps[0].strftime('%Y-%m-%d')} → {timestamps[-1].strftime('%Y-%m-%d')}\n")

    instruments = {sym: Instrument(sym) for sym in SYMS}
    broker      = SimulatedBroker(equity0)

    equity_curve   = []
    daily_returns  = []
    last_day_eq    = equity0
    last_day_bar   = 0
    BARS_PER_DAY   = 6.5   # approximate

    regime_counts  = defaultdict(int)
    regime_pnl     = defaultdict(float)

    completed_trades: list[TradeRecord] = []
    open_trades: dict[str, TradeRecord] = {}   # sym -> open TradeRecord

    peak = equity0

    for i, ts in enumerate(timestamps):
        bars = {sym: data[sym].iloc[i] for sym in SYMS}

        # ── update prices first, then mark-to-market ──────────────────────────
        new_prices = {sym: float(bars[sym]["close"]) for sym in SYMS}
        bar_pnl_usd = broker.mark_to_market(new_prices)

        eq = broker.equity()
        if eq > peak:
            peak = eq

        # ── update instruments ────────────────────────────────────────────────
        for sym in SYMS:
            b    = bars[sym]
            o    = float(b["open"]); h = float(b["high"])
            l    = float(b["low"]);  c = float(b["close"])
            inst = instruments[sym]

            # 15m sub-bars (4 synthetic bars per OHLC hour)
            for sub in [o, (o + h) / 2, (l + c) / 2, c]:
                inst.bh_15m.update(sub)

            cf_scale_1h = inst.regime.cf_scale   # updated LAST bar's regime
            inst.bh_1h.update(c, cf_scale=cf_scale_1h)
            inst.atr_1h.update(h, l, c)
            inst.regime.update(o, h, l, c)       # now update regime for THIS bar

            # Daily bar: every 6 hourly bars
            if inst.bh_1h.bc % 6 == 0:
                inst.bh_1d.update(c)

        # ── execution (one bar = one hour, implicit gate) ─────────────────────
        for sym in SYMS:
            if instruments[sym].bars_held > 0 and abs(instruments[sym].last_target) > 0.02:
                instruments[sym].bars_held += 1

        raw_targets = {}
        eq = broker.equity()

        for sym in SYMS:
            inst = instruments[sym]
            tfs  = inst.tf_score()
            ceil = TF_CAP[tfs]

            if tfs == 1 and abs(inst.last_target) < 0.01:
                ceil = 0.0

            if ceil == 0.0:
                tgt = 0.0
            else:
                d = inst.direction()
                if d == 0:
                    tgt = 0.0
                else:
                    atr = inst.atr_1h.value
                    p   = broker.prices[sym]
                    if atr > 0 and p > 0:
                        dv  = (atr / p) * math.sqrt(6.5)
                        raw = PER_INST_RISK / (dv + 1e-9)
                        cap = min(raw, ceil)
                    else:
                        cap = ceil
                    tgt = cap * d

                    reg = inst.regime.regime
                    # BEAR gate: suppress longs in sustained bear
                    if reg == REGIME_BEAR and tgt > 0 and inst.regime.rhb > 5:
                        tgt = 0.0
                    # BULL gate: suppress shorts in sustained bull (v12)
                    if reg == REGIME_BULL and tgt < 0 and inst.regime.rhb > 5:
                        tgt = 0.0

            # 4-bar minimum hold gate
            is_reversal = (
                abs(inst.last_target) > 0.01 and
                abs(tgt) > 0.01 and
                math.copysign(1, tgt) != math.copysign(1, inst.last_target)
            )
            if is_reversal and inst.bars_held < MIN_HOLD:
                tgt = inst.last_target

            raw_targets[sym] = tgt

        # Portfolio exposure cap
        total_exp = sum(abs(v) for v in raw_targets.values())
        scale     = 1.0 / total_exp if total_exp > 1.0 else 1.0

        for sym in SYMS:
            inst = instruments[sym]
            tgt  = float(raw_targets[sym] * scale)

            if abs(tgt - inst.last_target) > 0.02:
                old_contracts = broker.contracts[sym]

                # Execute
                broker.set_holdings(sym, tgt, equity_override=eq)
                new_contracts = broker.contracts[sym]

                # Trade tracking
                old_dir = int(math.copysign(1, old_contracts)) if old_contracts != 0 else 0
                new_dir = int(math.copysign(1, new_contracts)) if new_contracts != 0 else 0

                # Close existing trade
                if sym in open_trades and (new_contracts == 0 or new_dir != old_dir):
                    ot = open_trades.pop(sym)
                    ot.close_bar   = i
                    ot.close_price = broker.prices[sym]
                    ot.hold_bars   = i - ot.open_bar
                    ot.pnl         = (ot.close_price - ot.open_price) * ot.direction * abs(ot.contracts) * MULTIPLIER[sym]
                    completed_trades.append(ot)

                # Open new trade
                if new_contracts != 0:
                    open_trades[sym] = TradeRecord(
                        sym       = sym,
                        direction = new_dir,
                        tf_label  = inst.tf_label(),
                        regime    = inst.regime.regime,
                        bar_idx   = i,
                        price     = broker.prices[sym],
                        contracts = new_contracts,
                    )

                # Update bars_held
                if abs(tgt) < 0.01:
                    inst.bars_held = 0
                elif new_dir != old_dir:
                    inst.bars_held = 0
                else:
                    inst.bars_held = 0  # resize but same direction — reset for simplicity

                inst.last_target = tgt

        # Margin call check
        broker.check_margin_call()

        # ── daily return tracking ─────────────────────────────────────────────
        eq = broker.equity()
        equity_curve.append(eq)

        # Approximate daily: every ~6-7 bars
        bars_since_day = i - last_day_bar
        if bars_since_day >= 7:
            if last_day_eq > 0:
                daily_returns.append((eq - last_day_eq) / last_day_eq)
            last_day_eq  = eq
            last_day_bar = i

        # ── regime stats (ES as proxy) ────────────────────────────────────────
        reg = instruments["ES"].regime.regime
        regime_counts[reg] += 1
        regime_pnl[reg]    += bar_pnl_usd / equity0

    # Close any open trades at last price
    for sym, ot in open_trades.items():
        ot.close_bar   = n - 1
        ot.close_price = broker.prices[sym]
        ot.hold_bars   = (n - 1) - ot.open_bar
        ot.pnl         = (ot.close_price - ot.open_price) * ot.direction * abs(ot.contracts) * MULTIPLIER[sym]
        completed_trades.append(ot)

    # ═══════════════════════════════════════════════════════════════════════════
    # Compute stats
    # ═══════════════════════════════════════════════════════════════════════════
    eq_arr      = np.array(equity_curve)
    final_eq    = float(eq_arr[-1])
    total_ret   = (final_eq - equity0) / equity0
    ann_ret     = annual_return(total_ret, n)
    mdd         = max_drawdown(eq_arr)
    sh          = sharpe(daily_returns)
    so          = sortino(daily_returns)

    # Trade-level stats
    trade_pnls   = [t.pnl for t in completed_trades if t.pnl is not None]
    wins         = [p for p in trade_pnls if p > 0]
    losses       = [p for p in trade_pnls if p <= 0]
    win_rate     = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win      = float(np.mean(wins))   if wins   else 0.0
    avg_loss     = float(np.mean(losses)) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor= gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_trades = len(completed_trades)
    total_fees   = broker.total_fees

    # Per-year
    year_eq = {}
    for idx, eq_v in enumerate(equity_curve):
        y = timestamps[idx].year
        if y not in year_eq:
            year_eq[y] = [eq_v, eq_v]
        year_eq[y][1] = eq_v

    # ═══════════════════════════════════════════════════════════════════════════
    # Print stats (QC format)
    # ═══════════════════════════════════════════════════════════════════════════
    div = "─" * 65
    print(div)
    print("  PERFORMANCE SUMMARY  (QC-Equivalent)")
    print(div)
    print(f"  Total Return:     {total_ret:>+7.1%}")
    print(f"  Annual Return:    {ann_ret:>+7.1%}")
    print(f"  Sharpe Ratio:     {sh:>7.2f}")
    print(f"  Sortino Ratio:    {so:>7.2f}")
    print(f"  Max Drawdown:     {mdd:>7.1%}")
    print(f"  Win Rate:         {win_rate:>7.1%}     (trade-level)")
    print(f"  Total Trades:     {total_trades:>7,}")
    print(f"  Total Fees:       ${total_fees:>10,.0f}")
    print(f"  Profit Factor:    {profit_factor:>7.2f}")
    print(f"  Avg Win:          ${avg_win:>10,.0f}")
    print(f"  Avg Loss:         ${avg_loss:>10,.0f}")
    print(f"  Final Equity:     ${final_eq:>12,.0f}")
    print(f"  Excess Liquidity: ${broker.excess_liquidity():>12,.0f}")

    # Regime breakdown
    print(f"\n{div}")
    print("  REGIME PERFORMANCE")
    print(div)
    print(f"  {'Regime':<12}  {'Bars':>6}  {'% Time':>7}  {'Cum PnL':>9}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*7}  {'─'*9}")
    for reg in [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL]:
        cnt = regime_counts[reg]
        pct = cnt / n * 100
        pnl = regime_pnl[reg] * 100
        print(f"  {reg:<12}  {cnt:>6}  {pct:>6.1f}%  {pnl:>+8.2f}%")

    # Per-year
    print(f"\n{div}")
    print("  PER-YEAR RETURNS")
    print(div)
    print(f"  {'Year':>4}  {'Return':>8}")
    for y in sorted(year_eq):
        s_eq, e_eq = year_eq[y]
        ret = (e_eq - s_eq) / s_eq
        print(f"  {y:>4}  {ret:>+7.1%}")

    # ═══════════════════════════════════════════════════════════════════════════
    # BULL REGIME DEEP DIVE
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{div}")
    print("  BULL REGIME DEEP DIVE")
    print(div)

    bull_trades  = [t for t in completed_trades if t.regime == REGIME_BULL]
    bull_long    = [t for t in bull_trades if t.direction > 0]
    bull_short   = [t for t in bull_trades if t.direction < 0]

    def _wr(lst):
        if not lst: return float("nan")
        return len([t for t in lst if (t.pnl or 0) > 0]) / len(lst)

    def _avg_hold(lst):
        if not lst: return float("nan")
        return float(np.mean([t.hold_bars for t in lst]))

    def _sum_pnl(lst):
        return sum((t.pnl or 0) for t in lst)

    bull_long_wr  = _wr(bull_long)
    bull_short_wr = _wr(bull_short)
    bull_edge     = _sum_pnl(bull_trades)

    print(f"  Bull trades total:  {len(bull_trades)}")
    print(f"  Bull LONG  trades:  {len(bull_long)}  |  Win rate: {bull_long_wr:>5.1%}  |  Total PnL: ${_sum_pnl(bull_long):>+,.0f}")
    print(f"  Bull SHORT trades:  {len(bull_short)}  |  Win rate: {bull_short_wr:>5.1%}  |  Total PnL: ${_sum_pnl(bull_short):>+,.0f}")
    print(f"  Avg hold (LONG):    {_avg_hold(bull_long):.1f} bars")
    print(f"  Avg hold (SHORT):   {_avg_hold(bull_short):.1f} bars")
    print(f"  Bull TOTAL edge:    ${bull_edge:>+,.0f}")

    # TF breakdown in BULL
    print(f"\n  TF Trigger breakdown (BULL regime):")
    for tf in ["1d", "1h", "15m"]:
        tf_t = [t for t in bull_trades if t.tf_label == tf]
        print(f"    {tf:>3}: {len(tf_t):>3} trades  |  PnL ${_sum_pnl(tf_t):>+,.0f}")

    # Print worst BULL trades (top 3 losses)
    bull_losses_sorted = sorted(bull_trades, key=lambda t: t.pnl or 0)
    print(f"\n  Top-3 BULL losses:")
    for t in bull_losses_sorted[:3]:
        d_str = "LONG" if t.direction > 0 else "SHORT"
        print(f"    {t.sym} {d_str:5s}  TF={t.tf_label:3s}  hold={t.hold_bars:4d}h  PnL=${t.pnl:>+,.0f}")

    # Diagnose if edge is negative
    if bull_edge < 0:
        print(f"\n  WARNING BULL EDGE NEGATIVE — diagnose:")
        # Category 1: shorts still getting through
        short_loss = _sum_pnl(bull_short)
        if short_loss < 0:
            print(f"    1. Bull SHORT losses: ${short_loss:>+,.0f}  — BULL gate may not be blocking all shorts")
        # Category 2: wrong-direction BH
        wrong_dir  = [t for t in bull_long if (t.pnl or 0) < 0]
        if wrong_dir:
            avg_loss_l = float(np.mean([t.pnl for t in wrong_dir]))
            print(f"    2. Bull LONG losses ({len(wrong_dir)} trades, avg ${avg_loss_l:,.0f}) — BH direction may be wrong (lookback issue?)")
        # Category 3: premature BH formation
        short_hold = [t for t in bull_trades if t.hold_bars < 4 and (t.pnl or 0) < 0]
        if short_hold:
            print(f"    3. {len(short_hold)} sub-4-bar losers — min-hold gate not fully working or BH collapsing fast")
    elif bull_edge > 0:
        print(f"\n  BULL edge is POSITIVE. Strategy performs well in bull markets.")

    # Non-bull regimes summary
    print(f"\n  Non-BULL regime breakdown:")
    for reg in [REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL]:
        reg_t = [t for t in completed_trades if t.regime == reg]
        if not reg_t: continue
        print(f"    {reg:<10}: {len(reg_t):>3} trades  |  Win rate: {_wr(reg_t):>5.1%}  |  PnL ${_sum_pnl(reg_t):>+,.0f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save JSON
    # ═══════════════════════════════════════════════════════════════════════════
    out = {
        "strategy":        strategy,
        "start_equity":    equity0,
        "final_equity":    final_eq,
        "total_return_pct":total_ret * 100,
        "annual_return_pct":ann_ret * 100,
        "max_drawdown_pct":mdd * 100,
        "sharpe":          sh,
        "sortino":         so,
        "win_rate_trade":  win_rate,
        "total_trades":    total_trades,
        "total_fees":      total_fees,
        "profit_factor":   profit_factor if profit_factor != float("inf") else 999.0,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "n_bars":          n,
        "regime_counts":   dict(regime_counts),
        "regime_pnl_pct":  {k: float(v * 100) for k, v in regime_pnl.items()},
        "per_year":        {str(y): float((year_eq[y][1] - year_eq[y][0]) / year_eq[y][0] * 100)
                            for y in sorted(year_eq)},
        "bull_edge":       bull_edge,
        "bull_long_wr":    bull_long_wr,
        "bull_short_wr":   bull_short_wr,
    }
    json_path = os.path.join(RESULTS, "backtest_v12.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # Chart
    # ═══════════════════════════════════════════════════════════════════════════
    img_path = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("LARSA v12 — QC-Equivalent Local Backtest", fontsize=13, fontweight="bold")
        ts_plot = [t.replace(tzinfo=None) if hasattr(t, "tzinfo") else t for t in timestamps]

        # Equity
        ax = axes[0, 0]
        ax.plot(ts_plot, eq_arr / equity0, color="#4c78a8", lw=1.5)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title("Equity Curve (normalized)")
        ax.set_ylabel("Multiple of Start")

        # Drawdown
        ax = axes[0, 1]
        pk_arr = np.maximum.accumulate(eq_arr)
        dd_arr = (pk_arr - eq_arr) / pk_arr * 100
        ax.fill_between(ts_plot, -dd_arr, 0, color="#e45756", alpha=0.7)
        ax.set_title("Drawdown %")

        # Regime pie
        ax = axes[1, 0]
        labels = [REGIME_BULL, REGIME_BEAR, REGIME_SIDEWAYS, REGIME_HIGH_VOL]
        colors = ["#54a24b", "#e45756", "#79706e", "#f58518"]
        vals   = [regime_counts[r] for r in labels]
        ax.pie(vals, labels=labels, colors=colors, autopct="%1.0f%%", startangle=140)
        ax.set_title("Time in Each Regime")

        # Regime PnL bars
        ax = axes[1, 1]
        pnls = [regime_pnl[r] * 100 for r in labels]
        bars = ax.bar(labels, pnls, color=colors, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Cumulative PnL by Regime (%)")
        for bar, val in zip(bars, pnls):
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.05 if val >= 0 else -0.05),
                    f"{val:+.1f}%", ha="center", va=va, fontsize=9)

        plt.tight_layout()
        img_path = os.path.join(RESULTS, "backtest_v12.png")
        plt.savefig(img_path, dpi=130, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"\n  Chart skipped: {e}")

    print(f"\n  Saved → {json_path}")
    if img_path:
        print(f"  Chart  → {img_path}")
    print(f"\n{'═'*65}")
    print("  Done.")
    print(f"{'═'*65}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="larsa-v12")
    parser.add_argument("--equity", type=float, default=None,
                        help="Override starting equity (default: 1_000_000)")
    args = parser.parse_args()
    run(strategy=args.strategy, start_equity=args.equity)
