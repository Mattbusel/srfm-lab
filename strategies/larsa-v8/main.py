# LARSA v8: Multi-Resolution SRFM — fractal timeframe alignment
# Three independent BH detectors per instrument: 15min, 1H, 1D
# Position size = f(tf_score): 0→flat, 1→0.15, 2→0.25, 3→0.30, 4→0.35, 5→0.45, 6→0.55, 7→0.65
# Direction from highest active TF. Cross-instrument convergence bonus preserved.
# Expected: 5000+ trades vs v7's ~350. Triple-aligned events = highest conviction.

# region imports
from AlgorithmImports import *
import numpy as np
from enum import IntEnum
# endregion

class MarketRegime(IntEnum):
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    HIGH_VOLATILITY = 3

# CF calibration per resolution
CF = {
    Resolution.FIFTEEN_MINUTE: {"ES": 0.0003, "NQ": 0.0004, "YM": 0.00025},
    Resolution.HOUR:           {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008},
    Resolution.DAILY:          {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004},
}

# tf_score → position cap
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}

class FutureInstrument:
    def __init__(self, algo, future_ticker, resolution, cf, label):
        self.algo = algo
        self.label = label
        self.cf = cf
        self.resolution = resolution

        # res_label for keying
        res_map = {
            Resolution.FIFTEEN_MINUTE: "15m",
            Resolution.HOUR: "1h",
            Resolution.DAILY: "1d",
        }
        self.res_label = res_map.get(resolution, "??")

        self.future = algo.add_future(
            future_ticker, resolution,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0,
        )
        self.future.set_filter(timedelta(0), timedelta(182))
        self.sym = self.future.symbol

        # Rolling windows — all resolutions need these
        self.cw = RollingWindow[float](50)   # close prices
        self.bw = RollingWindow[float](20)   # beta values

        # BH physics state
        self.bh_mass = 0.0
        self.bh_form = 1.5
        self.bh_collapse = 1.0
        self.bh_decay = 0.95
        self.bh_active = False
        self.bh_dir = 0           # +1 long, -1 short
        self.bh_entry_price = 0.0
        self.ctl = 0              # TIMELIKE bar count
        self.bit = "UNKNOWN"      # TIMELIKE or SPACELIKE

        # bar count (warmup gate)
        self.bc = 0

        # last target (used only on hourly instruments but kept on all for simplicity)
        self.last_target = 0.0

        # Hourly-only state — indicators, regime, pos_floor
        self._is_hourly = (resolution == Resolution.HOUR)
        if self._is_hourly:
            h = resolution
            self.e12  = algo.ema(self.sym, 12,  h)
            self.e26  = algo.ema(self.sym, 26,  h)
            self.e50  = algo.ema(self.sym, 50,  h)
            self.e200 = algo.ema(self.sym, 200, h)
            self.rsi  = algo.rsi(self.sym, 14, MovingAverageType.WILDERS, h)
            self.macd = algo.macd(self.sym, 12, 26, 9, MovingAverageType.EXPONENTIAL, h)
            self.atr  = algo.atr(self.sym, 14, MovingAverageType.WILDERS, h)
            self.adx  = algo.adx(self.sym, 14, h)
            self.aw   = RollingWindow[float](50)   # ATR history
            self.regime = MarketRegime.SIDEWAYS
            self.rhb = 0       # regime history bars (bear gate)
            self.rc  = 0.5
            self.pos_floor = 0.0
            for ind in [self.e12, self.e26, self.e50, self.e200,
                        self.rsi, self.macd, self.atr, self.adx]:
                algo.warm_up_indicator(self.sym, ind, h)

    # ------------------------------------------------------------------
    # Regime detection — HOURLY instruments only
    # ------------------------------------------------------------------
    def _ind_ready(self):
        return all(i.is_ready for i in [
            self.e12, self.e26, self.e50, self.e200,
            self.rsi, self.macd, self.atr, self.adx,
        ])

    def detect_regime(self):
        if not self._is_hourly: return
        if not self._ind_ready() or not self.cw.is_ready: return
        price = self.cw[0]
        atr   = self.atr.current.value
        adx   = self.adx.current.value
        e12   = self.e12.current.value
        e26   = self.e26.current.value
        e200  = self.e200.current.value
        atr_ratio = 1.0
        if self.aw.is_ready:
            arr = np.array([self.aw[i] for i in range(int(self.aw.count))])
            atr_ratio = atr / arr.mean() if arr.mean() > 0 else 1.0
        self.rhb += 1
        if atr_ratio >= 1.5:
            nr = MarketRegime.HIGH_VOLATILITY
            nc = min(0.9, 0.5 + (atr_ratio - 1.5) * 0.4)
        elif price > e200 and e12 > e26:
            full_stack = e12 > e26 > self.e50.current.value > e200
            if adx > (14 if full_stack else 18):
                nr = MarketRegime.BULL
                nc = min(0.95, 0.5 + (adx - 14) / 60)
            else:
                nr = MarketRegime.SIDEWAYS
                nc = max(0.3, 0.7 - adx / 80)
        elif price < e200 and e12 < e26:
            full_stack = e200 > self.e50.current.value > e26 > e12
            if adx > (14 if full_stack else 18):
                nr = MarketRegime.BEAR
                nc = min(0.95, 0.5 + (adx - 14) / 60)
            else:
                nr = MarketRegime.SIDEWAYS
                nc = max(0.3, 0.7 - adx / 80)
        else:
            nr = MarketRegime.SIDEWAYS
            nc = max(0.3, 0.7 - adx / 80)
        if nr != self.regime:
            self.rhb = 0
            self.regime = nr
        self.rc = nc

    # ------------------------------------------------------------------
    # BH physics update — all resolutions
    # ------------------------------------------------------------------
    def update_bh(self):
        if self.cw.count < 2: return
        beta = abs(self.cw[0] - self.cw[1]) / (self.cw[1] + 1e-9) / (self.cf + 1e-9)
        self.bw.add(beta)
        was_active = self.bh_active

        if beta < 1.0:
            # TIMELIKE — accrete
            self.bit = "TIMELIKE"
            self.ctl += 1
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.bh_mass = self.bh_mass * 0.97 + 0.03 * 1.0 * sb
        else:
            # SPACELIKE — decay
            self.bit = "SPACELIKE"
            self.ctl = 0
            self.bh_mass *= self.bh_decay

        # Activation / collapse
        if not was_active:
            self.bh_active = self.bh_mass > self.bh_form and self.ctl >= 3
        else:
            self.bh_active = self.bh_mass > self.bh_collapse and self.ctl >= 3

        # Direction tracking
        bh_now_active = self.bh_active
        if not was_active and bh_now_active:
            # BH just activated — set direction from recent price move
            lookback = min(20, self.cw.count - 1)
            self.bh_dir = 1 if self.cw[0] > self.cw[lookback] else -1
            self.bh_entry_price = self.cw[0]
        elif was_active and not bh_now_active:
            self.bh_dir = 0
            self.bh_entry_price = 0.0


class LarsaV8(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        # Three resolution tiers
        self.instr_15m = {
            "ES": FutureInstrument(self, Futures.Indices.SP_500_E_MINI,    Resolution.FIFTEEN_MINUTE, CF[Resolution.FIFTEEN_MINUTE]["ES"], "ES"),
            "NQ": FutureInstrument(self, Futures.Indices.NASDAQ_100_E_MINI, Resolution.FIFTEEN_MINUTE, CF[Resolution.FIFTEEN_MINUTE]["NQ"], "NQ"),
            "YM": FutureInstrument(self, Futures.Indices.DOW_30_E_MINI,    Resolution.FIFTEEN_MINUTE, CF[Resolution.FIFTEEN_MINUTE]["YM"], "YM"),
        }
        self.instr_1h = {
            "ES": FutureInstrument(self, Futures.Indices.SP_500_E_MINI,    Resolution.HOUR, CF[Resolution.HOUR]["ES"], "ES"),
            "NQ": FutureInstrument(self, Futures.Indices.NASDAQ_100_E_MINI, Resolution.HOUR, CF[Resolution.HOUR]["NQ"], "NQ"),
            "YM": FutureInstrument(self, Futures.Indices.DOW_30_E_MINI,    Resolution.HOUR, CF[Resolution.HOUR]["YM"], "YM"),
        }
        self.instr_1d = {
            "ES": FutureInstrument(self, Futures.Indices.SP_500_E_MINI,    Resolution.DAILY, CF[Resolution.DAILY]["ES"], "ES"),
            "NQ": FutureInstrument(self, Futures.Indices.NASDAQ_100_E_MINI, Resolution.DAILY, CF[Resolution.DAILY]["NQ"], "NQ"),
            "YM": FutureInstrument(self, Futures.Indices.DOW_30_E_MINI,    Resolution.DAILY, CF[Resolution.DAILY]["YM"], "YM"),
        }

        # Flat dict for _process_instrument iteration
        self.all_instruments = {}
        for d in [self.instr_15m, self.instr_1h, self.instr_1d]:
            for k, v in d.items():
                self.all_instruments[f"{k}_{v.res_label}"] = v

        self.peak = 1_000_000.0
        self.ramp_back = 0

        # Charts
        rc = Chart("Regime")
        for k in ["ES", "NQ", "YM"]:
            rc.add_series(Series(k, SeriesType.LINE, 0))
        self.add_chart(rc)
        dc = Chart("Risk")
        dc.add_series(Series("Drawdown%", SeriesType.LINE, 0))
        self.add_chart(dc)

        # Warmup — daily 200-bar EMA needs ~200 trading days; add buffer
        self.set_warm_up(timedelta(days=400))

    # ------------------------------------------------------------------
    # Data handler
    # ------------------------------------------------------------------
    def on_data(self, data):
        if self.is_warming_up: return

        # Update portfolio peak and check circuit-breaker
        pv = self.portfolio.total_portfolio_value
        if pv > self.peak:
            self.peak = pv
        dd = (self.peak - pv) / (self.peak + 1e-9)
        if dd >= 0.12:
            self.liquidate()
            for inst in self.all_instruments.values():
                inst.last_target = 0.0
                if inst._is_hourly:
                    inst.pos_floor = 0.0
            self.ramp_back = 5
            self.peak = self.portfolio.total_portfolio_value
            return

        if self.ramp_back > 0:
            self.ramp_back -= 1

        # Step 1: process all instruments (state update only — no set_holdings here)
        for key, inst in self.all_instruments.items():
            self._process_instrument(data, inst)

        # Step 2: compute tf_score and execute per underlying
        for sym in ["ES", "NQ", "YM"]:
            i15 = self.instr_15m[sym]
            i1h = self.instr_1h[sym]
            i1d = self.instr_1d[sym]

            # Use 15min mapped contract — most current
            mapped = i15.future.mapped
            if mapped is None: continue
            if mapped not in self.securities: continue
            if not self.securities[mapped].exchange.exchange_open: continue

            # tf_score
            tf_score = 0
            if i1d.bh_active: tf_score += 4
            if i1h.bh_active: tf_score += 2
            if i15.bh_active: tf_score += 1

            cap = TF_CAP[tf_score]

            if cap == 0.0:
                tgt = 0.0
            else:
                direction = self._get_direction(i15, i1h, i1d)
                if direction == 0:
                    tgt = 0.0
                else:
                    tgt = cap * direction

                    # BEAR gate — block longs in sustained BEAR (hourly regime)
                    bear_rhb_thresh = 3 if sym == "YM" else 5
                    if i1h.regime == MarketRegime.BEAR and tgt > 0 and i1h.rhb > bear_rhb_thresh:
                        tgt = 0.0

                    # NQ notional cap in non-BULL
                    if sym == "NQ" and i1h.regime != MarketRegime.BULL:
                        nq_cap = 400000.0 / (self.portfolio.total_portfolio_value + 1e-9)
                        if abs(tgt) > nq_cap:
                            tgt = float(np.sign(tgt) * nq_cap)

            # Cross-instrument convergence bonus
            conv_count = sum(1 for s in ["ES", "NQ", "YM"] if self.instr_1h[s].bh_active)
            if conv_count >= 2 and tf_score >= 2 and not np.isclose(tgt, 0.0):
                tgt = float(np.clip(tgt * 1.1, -0.65, 0.65))

            # pos_floor — high-conviction locking
            if (tf_score >= 6 and not np.isclose(tgt, 0.0)
                    and abs(tgt) > 0.4 and i1h.ctl >= 5):
                i1h.pos_floor = max(i1h.pos_floor, 0.70 * abs(tgt))
            if (i1h.pos_floor > 0.0 and tf_score >= 4
                    and not np.isclose(i1h.last_target, 0.0)):
                tgt = float(np.sign(i1h.last_target) * max(abs(tgt), i1h.pos_floor))
                i1h.pos_floor *= 0.95
            if tf_score < 4 or np.isclose(tgt, 0.0):
                i1h.pos_floor = 0.0

            # Reset pos_floor when both daily and hourly BH are off
            if not i1d.bh_active and not i1h.bh_active:
                i1h.pos_floor = 0.0

            # Ramp-back scaling after circuit-breaker
            if self.ramp_back > 0:
                tgt = float(np.clip(tgt, -0.35, 0.35))

            # Execute if target moved enough
            if abs(tgt - i1h.last_target) > 0.02:
                i1h.last_target = tgt
                self.set_holdings(mapped, tgt)

        # Periodic charts
        any_bc = any(i.bc % 24 == 0 for i in self.instr_1h.values())
        if any_bc:
            pv2 = self.portfolio.total_portfolio_value
            self.plot("Risk", "Drawdown%", (self.peak - pv2) / (self.peak + 1e-9) * 100)
            for key, inst in self.instr_1h.items():
                self.plot("Regime", key, int(inst.regime))

    # ------------------------------------------------------------------
    # Direction helper
    # ------------------------------------------------------------------
    def _get_direction(self, i15, i1h, i1d):
        """Return +1, -1, or 0 from the highest active TF's BH direction."""
        def _resolve(inst):
            if inst.bh_dir != 0:
                return inst.bh_dir
            if inst.cw.count >= 2:
                return 1 if inst.cw[0] > inst.cw[min(4, inst.cw.count - 1)] else -1
            return 0

        if i1d.bh_active:
            return _resolve(i1d)
        if i1h.bh_active:
            return _resolve(i1h)
        if i15.bh_active:
            return _resolve(i15)
        return 0

    # ------------------------------------------------------------------
    # Instrument state update — no set_holdings
    # ------------------------------------------------------------------
    def _process_instrument(self, data, inst):
        mapped = inst.future.mapped
        if mapped is None: return

        trade_bar = data.bars.get(mapped)
        quote_bar = data.quote_bars.get(mapped)
        if trade_bar is not None:
            bar = trade_bar
        elif quote_bar is not None:
            bar = quote_bar
        else:
            return

        inst.bc += 1
        inst.cw.add(float(bar.close))

        # BH physics (also adds to bw internally)
        inst.update_bh()

        # Hourly extras
        if inst._is_hourly:
            if inst.atr.is_ready:
                inst.aw.add(inst.atr.current.value)
            inst.detect_regime()

        # Warmup gate — suppress BH activity until enough bars seen
        bc_thresh = {
            Resolution.FIFTEEN_MINUTE: 400,
            Resolution.HOUR: 120,
            Resolution.DAILY: 30,
        }.get(inst.resolution, 120)
        if inst.bc < bc_thresh:
            inst.bh_active = False
            inst.bh_dir = 0

    # ------------------------------------------------------------------
    # End of algorithm logging
    # ------------------------------------------------------------------
    def on_end_of_algorithm(self):
        pv = self.portfolio.total_portfolio_value
        dd = (self.peak - pv) / (self.peak + 1e-9)
        for sym in ["ES", "NQ", "YM"]:
            i15 = self.instr_15m[sym]
            i1h = self.instr_1h[sym]
            i1d = self.instr_1d[sym]
            self.log(
                f"[{sym}] END "
                f"regime_1h={i1h.regime.name} "
                f"bc_15m={i15.bc} bc_1h={i1h.bc} bc_1d={i1d.bc} "
                f"bh_1d={i1d.bh_active} bh_1h={i1h.bh_active} bh_15m={i15.bh_active} "
                f"tf_score={4*int(i1d.bh_active)+2*int(i1h.bh_active)+int(i15.bh_active)}"
            )
        self.log(f"PORTFOLIO peak={self.peak:.0f} final={pv:.0f} dd={dd:.1%}")
