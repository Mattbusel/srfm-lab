"""
SRFM Strategy Template — Minimal LEAN algorithm using srfm_core.

Copy this directory to strategies/<experiment-name>/ and modify one thing.
The template is already wired with:
  - ES futures subscription
  - Minkowski classification
  - BlackHole detection
  - ProperTime gating
  - AgentEnsemble signal
  - RiskManager stops + circuit breaker

Quick start:
    make new name=my-experiment
    # edit strategies/my-experiment/main.py
    make backtest s=my-experiment
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))

from AlgorithmImports import *

from srfm_core import (
    MinkowskiClassifier,
    BlackHoleDetector,
    GeodesicAnalyzer,
    HawkingMonitor,
    GravitationalLens,
    ProperTimeClock,
    Causal,
    BHState,
)
from agents import AgentEnsemble
from regime import RegimeDetector
from risk import RiskManager


# ─── Configuration ────────────────────────────────────────────────────────────
# Modify these constants to experiment.  One knob per experiment is the rule.

TICKER          = "ES"           # Futures ticker
RESOLUTION      = Resolution.Hour
CF              = 1.2            # Minkowski speed-of-light scaling
BH_FORM         = 1.5            # Black hole formation threshold
BH_COLLAPSE     = 0.4            # Black hole collapse threshold
MASS_DECAY      = 0.92           # Per-bar mass decay
PROPER_TIME_MIN = 5.0            # Minimum proper time between entries
MAX_DAILY_LOSS  = 0.02           # 2% daily loss limit
MAX_DRAWDOWN    = 0.10           # 10% max drawdown
STOP_LOSS_PCT   = 0.008          # 0.8% stop loss
TAKE_PROFIT_PCT = 0.025          # 2.5% take profit


# ─── Algorithm ────────────────────────────────────────────────────────────────

class SRFMTemplate(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100_000)

        # Subscribe to futures
        future = self.AddFuture(TICKER, RESOLUTION)
        future.SetFilter(lambda u: u.StandardsOnly().FrontMonth())
        self._symbol = future.Symbol

        # SRFM physics
        self._mink    = MinkowskiClassifier(cf=CF)
        self._bh      = BlackHoleDetector(bh_form_threshold=BH_FORM, bh_collapse_threshold=BH_COLLAPSE, mass_decay=MASS_DECAY)
        self._geo     = GeodesicAnalyzer()
        self._hawking = HawkingMonitor()
        self._lens    = GravitationalLens()
        self._clock   = ProperTimeClock(min_proper_time=PROPER_TIME_MIN)

        # Agents
        self._ensemble = AgentEnsemble(learning_rate=0.01, gamma=0.95)

        # Regime + risk
        self._regime = RegimeDetector()
        self._risk   = RiskManager(
            max_daily_loss_pct=MAX_DAILY_LOSS,
            max_drawdown_pct=MAX_DRAWDOWN,
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
        )

        self._prev_price: float = 0.0
        self._invested: bool = False
        self._direction: int = 0

    # ------------------------------------------------------------------
    def OnData(self, data: Slice):
        # Get front-month contract
        chain = data.FutureChains.get(self._symbol)
        if not chain:
            return
        contract = sorted(chain, key=lambda c: c.Expiry)[0]
        price = contract.LastPrice
        if price <= 0 or self._prev_price <= 0:
            self._prev_price = price
            return

        ret = (price - self._prev_price) / self._prev_price
        self._prev_price = price

        # ── Physics update ──────────────────────────────────────────
        causal  = self._mink.update(ret)
        bh_state = self._bh.update(causal, ret)
        self._geo.update(ret, causal)
        self._clock.tick(ret)
        self._regime.update(ret, causal)

        # ── Risk update ─────────────────────────────────────────────
        equity = self.Portfolio.TotalPortfolioValue
        self._risk.update_equity(equity)

        # Stop / take-profit on open position
        if self._invested:
            if self._risk.should_stop(price) or self._risk.should_take_profit(price):
                self.Liquidate(contract.Symbol)
                self._risk.on_exit(hit_stop=self._risk.should_stop(price))
                self._invested = False
                self._direction = 0
                return

        # ── Entry logic ─────────────────────────────────────────────
        if not self._invested:
            if not self._risk.check(equity):
                return
            if not self._clock.gate_passed():
                return
            if not self._bh.is_active:
                return
            if self._regime.is_crisis:
                return

            # Build state vector for agents
            state = self._build_state(ret, causal, bh_state)
            signal = self._ensemble.signal(state)

            if signal == 0.0:
                return

            direction = int(signal)   # +1 or -1
            scalar = self._hawking.size_scalar(self._bh.mass)
            qty = self._risk.position_size(equity, price, hawking_scalar=scalar)

            if qty <= 0:
                return

            if direction == 1:
                self.MarketOrder(contract.Symbol, qty)
            else:
                self.MarketOrder(contract.Symbol, -qty)

            self._risk.on_entry(price, direction)
            self._clock.reset_gate()
            self._invested = True
            self._direction = direction

        # ── Learning update ─────────────────────────────────────────
        reward = ret * self._direction if self._invested else 0.0
        state = self._build_state(ret, causal, bh_state)
        self._ensemble.remember(state, max(0, self._direction), reward, state, False)
        self._ensemble.learn()

    # ------------------------------------------------------------------
    def _build_state(self, ret: float, causal: Causal, bh_state: BHState) -> tuple:
        """Discretise continuous features into a state tuple for the Q-agents."""
        def bucket(v: float, lo: float, hi: float, bins: int = 5) -> int:
            v = max(lo, min(hi, v))
            return int((v - lo) / (hi - lo) * (bins - 1))

        return (
            int(causal.value == "TIMELIKE"),
            int(bh_state.value == "ACTIVE"),
            bucket(self._bh.mass, 0, 5),
            bucket(self._geo.causal_fraction, 0, 1),
            bucket(self._geo.geodesic_deviation, 0, 0.03),
            bucket(ret, -0.02, 0.02),
            int(self._direction),
        )

    def OnEndOfDay(self, symbol):
        equity = self.Portfolio.TotalPortfolioValue
        self._risk.start_new_day(equity)
