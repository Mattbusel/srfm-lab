# LARSA v9: v8 + minimum 4-bar hold before direction reversal
#
# Single change from v8: bars_held counter per instrument.
# Reducing size: always allowed.
# Going flat:    always allowed.
# REVERSAL (long→short or short→long): blocked until bars_held >= 4.
#
# Forensic basis (deathloop_detective.py on QC backtest):
#   Trades held <1h:  30-36% WR, -$5.1M net  ← eliminated
#   Trades held >4h:  53% WR,    +$4.9M net  ← kept
#   Trades held >1d:  62% WR,    +$8.3M net  ← kept
#   505 hyperactive hours, 42 direction flips <4min → $1.9M destroyed
#
# QC NOTE: Resolution.FIFTEEN_MINUTE does not exist in QC.
#          15min instruments subscribe at Resolution.Minute and consolidate via TradeBarConsolidator.

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

# CF calibration per resolution (keyed by string label)
CF = {
    "15m":  {"ES": 0.0003, "NQ": 0.0004,  "YM": 0.00025},
    "1h":   {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008},
    "1d":   {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004},
}

# tf_score → position cap
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}

MIN_HOLD_BARS = 4   # minimum hourly bars before a direction reversal is allowed

class FutureInstrument:
    def __init__(self, algo, future_ticker, res_label, cf, label):
        self.algo = algo
        self.label = label
        self.cf = cf
        self.res_label = res_label

        qc_res_map = {
            "15m": Resolution.MINUTE,
            "1h":  Resolution.HOUR,
            "1d":  Resolution.DAILY,
        }
        self._qc_res = qc_res_map[res_label]
        self._is_15m = (res_label == "15m")
        self._is_hourly = (res_label == "1h")

        self.future = algo.add_future(
            future_ticker, self._qc_res,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0,
        )
        self.future.set_filter(timedelta(0), timedelta(182))
        self.sym = self.future.symbol

        self.cw = RollingWindow[float](50)
        self.bw = RollingWindow[float](20)

        self.bh_mass = 0.0
        self.bh_form = 1.5
        self.bh_collapse = 1.0
        self.bh_decay = 0.95
        self.bh_active = False
        self.bh_dir = 0
        self.bh_entry_price = 0.0
        self.ctl = 0
        self.bit = "UNKNOWN"
        self.bc = 0
        self.last_target = 0.0
        self.bars_held = 0   # v9: hourly bars held in current direction

        if self._is_hourly:
            h = Resolution.HOUR
            self.e12  = algo.ema(self.sym, 12,  h)
            self.e26  = algo.ema(self.sym, 26,  h)
            self.e50  = algo.ema(self.sym, 50,  h)
            self.e200 = algo.ema(self.sym, 200, h)
            self.rsi  = algo.rsi(self.sym, 14, MovingAverageType.WILDERS, h)
            self.macd = algo.macd(self.sym, 12, 26, 9, MovingAverageType.EXPONENTIAL, h)
            self.atr  = algo.atr(self.sym, 14, MovingAverageType.WILDERS, h)
            self.adx  = algo.adx(self.sym, 14, h)
            self.aw   = RollingWindow[float](50)
            self.regime = MarketRegime.SIDEWAYS
            self.rhb = 0
            self.rc  = 0.5
            self.pos_floor = 0.0
            for ind in [self.e12, self.e26, self.e50, self.e200,
                        self.rsi, self.macd, self.atr, self.adx]:
                algo.warm_up_indicator(self.sym, ind, h)
        else:
            self.pos_floor = 0.0
            self.regime = MarketRegime.SIDEWAYS
            self.rhb = 0

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

    def update_bh(self):
        if self.cw.count < 2: return
        beta = abs(self.cw[0] - self.cw[1]) / (self.cw[1] + 1e-9) / (self.cf + 1e-9)
        self.bw.add(beta)
        was_active = self.bh_active

        if beta < 1.0:
            self.bit = "TIMELIKE"
            self.ctl += 1
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.bh_mass = self.bh_mass * 0.97 + 0.03 * 1.0 * sb
        else:
            self.bit = "SPACELIKE"
            self.ctl = 0
            self.bh_mass *= self.bh_decay

        if not was_active:
            self.bh_active = self.bh_mass > self.bh_form and self.ctl >= 3
        else:
            self.bh_active = self.bh_mass > self.bh_collapse and self.ctl >= 3

        bh_now_active = self.bh_active
        if not was_active and bh_now_active:
            lookback = min(20, self.cw.count - 1)
            self.bh_dir = 1 if self.cw[0] > self.cw[lookback] else -1
            self.bh_entry_price = self.cw[0]
        elif was_active and not bh_now_active:
            self.bh_dir = 0
            self.bh_entry_price = 0.0

    def apply_warmup_gate(self):
        bc_thresh = {"15m": 400, "1h": 120, "1d": 30}.get(self.res_label, 120)
        if self.bc < bc_thresh:
            self.bh_active = False
            self.bh_dir = 0


class LarsaV9(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        self.instr_15m = {
            "ES": FutureInstrument(self, Futures.Indices.SP_500_E_MINI,     "15m", CF["15m"]["ES"], "ES"),
            "NQ": FutureInstrument(self, Futures.Indices.NASDAQ_100_E_MINI, "15m", CF["15m"]["NQ"], "NQ"),
            "YM": FutureInstrument(self, Futures.Indices.DOW_30_E_MINI,     "15m", CF["15m"]["YM"], "YM"),
        }
        self.instr_1h = {
            "ES": FutureInstrument(self, Futures.Indices.SP_500_E_MINI,     "1h", CF["1h"]["ES"], "ES"),
            "NQ": FutureInstrument(self, Futures.Indices.NASDAQ_100_E_MINI, "1h", CF["1h"]["NQ"], "NQ"),
            "YM": FutureInstrument(self, Futures.Indices.DOW_30_E_MINI,     "1h", CF["1h"]["YM"], "YM"),
        }
        self.instr_1d = {
            "ES": FutureInstrument(self, Futures.Indices.SP_500_E_MINI,     "1d", CF["1d"]["ES"], "ES"),
            "NQ": FutureInstrument(self, Futures.Indices.NASDAQ_100_E_MINI, "1d", CF["1d"]["NQ"], "NQ"),
            "YM": FutureInstrument(self, Futures.Indices.DOW_30_E_MINI,     "1d", CF["1d"]["YM"], "YM"),
        }

        for inst in self.instr_15m.values():
            self.consolidate(inst.sym, timedelta(minutes=15),
                             lambda bar, i=inst: self._on_15m_bar(i, bar))

        self.non_15m_instruments = {}
        for k, v in self.instr_1h.items():
            self.non_15m_instruments[f"{k}_1h"] = v
        for k, v in self.instr_1d.items():
            self.non_15m_instruments[f"{k}_1d"] = v

        self.peak = 1_000_000.0
        self._last_exec_hour = None

        rc = Chart("Regime")
        for k in ["ES", "NQ", "YM"]:
            rc.add_series(Series(k, SeriesType.LINE, 0))
        self.add_chart(rc)
        dc = Chart("Risk")
        dc.add_series(Series("Drawdown%", SeriesType.LINE, 0))
        self.add_chart(dc)

        self.set_warm_up(timedelta(days=400))

    def _on_15m_bar(self, inst, bar):
        if self.is_warming_up: return
        inst.bc += 1
        inst.cw.add(float(bar.close))
        inst.update_bh()
        inst.apply_warmup_gate()

    def on_data(self, data):
        if self.is_warming_up: return

        pv = self.portfolio.total_portfolio_value
        if pv > self.peak:
            self.peak = pv

        for inst in self.non_15m_instruments.values():
            self._process_instrument(data, inst)

        current_hour = self.time.replace(minute=0, second=0, microsecond=0)
        if current_hour == self._last_exec_hour:
            return
        self._last_exec_hour = current_hour

        # Increment bars_held for any instrument currently in a position
        for i1h in self.instr_1h.values():
            if abs(i1h.last_target) > 0.02:
                i1h.bars_held += 1

        # ── Step 1: raw targets ──────────────────────────────────────────
        raw_targets = {}

        for sym in ["ES", "NQ", "YM"]:
            i15 = self.instr_15m[sym]
            i1h = self.instr_1h[sym]
            i1d = self.instr_1d[sym]

            mapped = i1h.future.mapped
            if mapped is None:
                raw_targets[sym] = (None, 0.0)
                continue
            if mapped not in self.securities or not self.securities[mapped].exchange.exchange_open:
                raw_targets[sym] = (None, 0.0)
                continue

            tf_score = 0
            if i1d.bh_active: tf_score += 4
            if i1h.bh_active: tf_score += 2
            if i15.bh_active: tf_score += 1

            cap = TF_CAP[tf_score]

            currently_flat = np.isclose(i1h.last_target, 0.0)
            if tf_score == 1 and currently_flat:
                cap = 0.0

            if cap == 0.0:
                tgt = 0.0
            else:
                direction = self._get_direction(i15, i1h, i1d)
                if direction == 0:
                    tgt = 0.0
                else:
                    tgt = cap * direction

                    if i1h.regime == MarketRegime.BEAR and tgt > 0 and i1h.rhb > (3 if sym == "YM" else 5):
                        tgt = 0.0

                    if sym == "NQ" and i1h.regime != MarketRegime.BULL:
                        nq_cap = 400000.0 / (self.portfolio.total_portfolio_value + 1e-9)
                        tgt = float(np.sign(tgt) * min(abs(tgt), nq_cap))

            # pos_floor
            if (tf_score >= 6 and not np.isclose(tgt, 0.0)
                    and abs(tgt) > 0.15 and i1h.ctl >= 5):
                i1h.pos_floor = max(i1h.pos_floor, 0.70 * abs(tgt))
            if (i1h.pos_floor > 0.0 and tf_score >= 4
                    and not np.isclose(i1h.last_target, 0.0)):
                tgt = float(np.sign(i1h.last_target) * max(abs(tgt), i1h.pos_floor))
                i1h.pos_floor *= 0.95
            if tf_score < 4 or np.isclose(tgt, 0.0):
                i1h.pos_floor = 0.0
            if not i1d.bh_active and not i1h.bh_active:
                i1h.pos_floor = 0.0

            # ── v9: minimum hold gate ────────────────────────────────────
            # A reversal is: currently long (last_target > 0) → new target < 0
            #             or: currently short (last_target < 0) → new target > 0
            # Reducing size or going flat: always allowed.
            # Reversal: blocked until bars_held >= MIN_HOLD_BARS.
            is_reversal = (
                not np.isclose(i1h.last_target, 0.0) and
                not np.isclose(tgt, 0.0) and
                np.sign(tgt) != np.sign(i1h.last_target)
            )
            if is_reversal and i1h.bars_held < MIN_HOLD_BARS:
                tgt = i1h.last_target   # hold current position, skip reversal

            raw_targets[sym] = (mapped, tgt)

        # ── Step 2: portfolio exposure cap at 1.0 ───────────────────────
        total_exposure = sum(abs(tgt) for _, tgt in raw_targets.values())
        scale = 1.0 / total_exposure if total_exposure > 1.0 else 1.0

        # ── Step 3: execute ──────────────────────────────────────────────
        for sym in ["ES", "NQ", "YM"]:
            i1h = self.instr_1h[sym]
            mapped, tgt = raw_targets[sym]
            if mapped is None:
                continue
            tgt = float(tgt * scale)

            if abs(tgt - i1h.last_target) > 0.02:
                # Reset bars_held when going flat; preserve when adjusting size
                if np.isclose(tgt, 0.0):
                    i1h.bars_held = 0
                elif np.sign(tgt) != np.sign(i1h.last_target):
                    i1h.bars_held = 0   # direction change granted — reset clock
                i1h.last_target = tgt
                self.set_holdings(mapped, tgt)

        # Periodic charts
        if any(i.bc % 24 == 0 for i in self.instr_1h.values()):
            pv2 = self.portfolio.total_portfolio_value
            self.plot("Risk", "Drawdown%", (self.peak - pv2) / (self.peak + 1e-9) * 100)
            for key, inst in self.instr_1h.items():
                self.plot("Regime", key, int(inst.regime))

    def _get_direction(self, i15, i1h, i1d):
        def _resolve(inst):
            if inst.bh_dir != 0:
                return inst.bh_dir
            if inst.cw.count >= 2:
                return 1 if inst.cw[0] > inst.cw[min(4, inst.cw.count - 1)] else -1
            return 0
        if i1d.bh_active: return _resolve(i1d)
        if i1h.bh_active: return _resolve(i1h)
        if i15.bh_active: return _resolve(i15)
        return 0

    def _process_instrument(self, data, inst):
        mapped = inst.future.mapped
        if mapped is None: return
        trade_bar = data.bars.get(mapped)
        quote_bar = data.quote_bars.get(mapped)
        bar = trade_bar if trade_bar is not None else quote_bar
        if bar is None: return

        inst.bc += 1
        inst.cw.add(float(bar.close))
        inst.update_bh()

        if inst._is_hourly:
            if inst.atr.is_ready:
                inst.aw.add(inst.atr.current.value)
            inst.detect_regime()

        inst.apply_warmup_gate()

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
                f"bars_held={i1h.bars_held} "
                f"bc_15m={i15.bc} bc_1h={i1h.bc} bc_1d={i1d.bc} "
                f"bh_1d={i1d.bh_active} bh_1h={i1h.bh_active} bh_15m={i15.bh_active} "
                f"tf_score={4*int(i1d.bh_active)+2*int(i1h.bh_active)+int(i15.bh_active)}"
            )
        self.log(f"PORTFOLIO peak={self.peak:.0f} final={pv:.0f} dd={dd:.1%}")
