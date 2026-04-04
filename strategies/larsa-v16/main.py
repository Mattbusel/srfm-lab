# LARSA v16: v14 OG + expanded instrument universe
#
# GEAR 1 — TAIL CAPTURE (fixed $3M bucket):
#   BH physics across 8 instruments: ES, NQ, YM, CL, GC, ZB, NG, VX
#   Same engine, uncorrelated events. More BH convergence points per year.
#   Less bleed-to-event ratio because something is always near a regime break.
#
# GEAR 2 — HARVEST MODE (everything above $3M):
#   Z-score mean reversion on all 8 instruments during SIDEWAYS regime.
#
# INSTRUMENT GROUPS (structural correlation):
#   Equity (ES/NQ/YM): corr ~0.90 — fire together
#   Energy (CL/NG):    corr ~0.50 — semi-independent
#   Safe haven (GC/ZB): corr ~0.30 — often counter-equity
#   Vol (VX):          corr ~0.0  — structurally uncorrelated

# region imports
from AlgorithmImports import *
import numpy as np
import math
from enum import IntEnum
# endregion

class MarketRegime(IntEnum):
    BULL = 0
    BEAR = 1
    SIDEWAYS = 2
    HIGH_VOLATILITY = 3

CF = {
    "15m": {"ES": 0.0003, "NQ": 0.0004,  "YM": 0.00025,
            "CL": 0.0015, "GC": 0.0008,  "ZB": 0.0005,
            "NG": 0.0020, "VX": 0.0030},
    "1h":  {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008,
            "CL": 0.004,  "GC": 0.0025,  "ZB": 0.0015,
            "NG": 0.006,  "VX": 0.008},
    "1d":  {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004,
            "CL": 0.015,  "GC": 0.008,   "ZB": 0.005,
            "NG": 0.020,  "VX": 0.025},
}

TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}

MIN_HOLD_BARS = 4

# 8 instruments across uncorrelated asset classes — blended correlation ~0.35
N_INSTRUMENTS    = 8
INST_CORRELATION = 0.35
PORTFOLIO_DAILY_RISK = 0.01
_CORR_FACTOR = math.sqrt(N_INSTRUMENTS + N_INSTRUMENTS * (N_INSTRUMENTS - 1) * INST_CORRELATION)
PER_INST_RISK = PORTFOLIO_DAILY_RISK / _CORR_FACTOR   # ≈ 0.00181

TAIL_FIXED_CAPITAL    = 3_000_000.0
HARVEST_RISK_PER_INST = 0.02
HARVEST_Z_ENTRY       = 1.5
HARVEST_Z_EXIT        = 0.3
HARVEST_Z_STOP        = 2.8
HARVEST_LOOKBACK      = 20

# Thin-instrument position caps (notional / portfolio value)
INST_CAPS = {
    "NQ": 400_000,
    "NG": 200_000,
    "VX": 150_000,
}

# Higher BH formation threshold for noisy instruments — require stronger signal before entry
# CL/NG/VX are prone to false formations; GC/ZB trend cleanly once they break
BH_FORM_OVERRIDE = {
    "CL": 1.8,
    "NG": 1.8,
    "VX": 1.8,
}

ALL_SYMS = ["ES", "NQ", "YM", "CL", "GC", "ZB", "NG", "VX"]


class FutureInstrument:
    def __init__(self, algo, future_ticker, res_label, cf, label):
        self.algo = algo
        self.label = label
        self.cf = cf
        self.res_label = res_label

        qc_res_map = {"15m": Resolution.MINUTE, "1h": Resolution.HOUR, "1d": Resolution.DAILY}
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

        self.cf_scale = 1.0
        self.bh_mass = 0.0
        self.bh_form = BH_FORM_OVERRIDE.get(label, 1.5)
        self.bh_collapse = 1.0
        self.bh_decay = 0.95
        self.bh_active = False
        self.bh_dir = 0
        self.bh_entry_price = 0.0
        self.ctl = 0
        self.bit = "UNKNOWN"
        self.bc = 0
        self.last_target = 0.0
        self.bars_held = 0

        self.harv_pos = 0.0
        self.harv_entry_z = 0.0

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

    def z_score(self):
        if self.cw.count < HARVEST_LOOKBACK:
            return 0.0
        prices = np.array([self.cw[i] for i in range(HARVEST_LOOKBACK)])
        mu  = prices.mean()
        std = prices.std()
        if std < 1e-9:
            return 0.0
        return (self.cw[0] - mu) / std

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
        self.cf_scale = 3.0 if self.regime == MarketRegime.BULL else 1.0

    def update_bh(self):
        if self.cw.count < 2: return
        effective_cf = self.cf * self.cf_scale
        beta = abs(self.cw[0] - self.cw[1]) / (self.cw[1] + 1e-9) / (effective_cf + 1e-9)
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

        if not was_active and self.bh_active:
            lookback = min(20, self.cw.count - 1)
            self.bh_dir = 1 if self.cw[0] > self.cw[lookback] else -1
            self.bh_entry_price = self.cw[0]
        elif was_active and not self.bh_active:
            self.bh_dir = 0
            self.bh_entry_price = 0.0

    def apply_warmup_gate(self):
        bc_thresh = {"15m": 400, "1h": 120, "1d": 30}.get(self.res_label, 120)
        if self.bc < bc_thresh:
            self.bh_active = False
            self.bh_dir = 0


class LarsaV16(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0

        tickers = {
            "ES": Futures.Indices.SP_500_E_MINI,
            "NQ": Futures.Indices.NASDAQ_100_E_MINI,
            "YM": Futures.Indices.DOW_30_E_MINI,
            "CL": "CL",
            "GC": "GC",
            "ZB": Futures.Financials.Y_30_TREASURY_BOND,
            "NG": "NG",
            "VX": "VX",
        }

        self.instr_15m = {
            sym: FutureInstrument(self, tickers[sym], "15m", CF["15m"][sym], sym)
            for sym in ALL_SYMS
        }
        self.instr_1h = {
            sym: FutureInstrument(self, tickers[sym], "1h", CF["1h"][sym], sym)
            for sym in ALL_SYMS
        }
        self.instr_1d = {
            sym: FutureInstrument(self, tickers[sym], "1d", CF["1d"][sym], sym)
            for sym in ALL_SYMS
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
        self._last_harvest_day = None

        self.log(
            f"[v16 PARAMS] "
            f"instruments={ALL_SYMS} "
            f"tail_capital=${TAIL_FIXED_CAPITAL:,.0f} "
            f"per_inst_risk={PER_INST_RISK:.4%} "
            f"harvest_risk_per_inst={HARVEST_RISK_PER_INST:.1%}"
        )

        rc = Chart("Regime")
        for k in ALL_SYMS:
            rc.add_series(Series(k, SeriesType.LINE, 0))
        self.add_chart(rc)

        dc = Chart("Risk")
        dc.add_series(Series("Drawdown%", SeriesType.LINE, 0))
        self.add_chart(dc)

        ac = Chart("Allocation")
        ac.add_series(Series("TailFrac%",    SeriesType.LINE, 0))
        ac.add_series(Series("HarvestFrac%", SeriesType.LINE, 0))
        self.add_chart(ac)

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

        tail_frac    = min(TAIL_FIXED_CAPITAL, pv) / pv
        harvest_frac = max(0.0, pv - TAIL_FIXED_CAPITAL) / pv
        harvest_equity = pv * harvest_frac

        # ── Gear 1: TAIL CAPTURE ─────────────────────────────────────────────
        for i1h in self.instr_1h.values():
            if abs(i1h.last_target) > 0.02:
                i1h.bars_held += 1

        raw_targets = {}

        for sym in ALL_SYMS:
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

            ceiling = TF_CAP[tf_score]

            if tf_score == 1 and np.isclose(i1h.last_target, 0.0):
                ceiling = 0.0

            if ceiling == 0.0:
                tail_tgt = 0.0
            else:
                direction = self._get_direction(i15, i1h, i1d)
                if direction == 0:
                    tail_tgt = 0.0
                else:
                    if i1h.atr.is_ready and i1h.atr.current.value > 0:
                        price = self.securities[mapped].price
                        if price > 0:
                            hourly_vol_pct = i1h.atr.current.value / price
                            daily_vol_pct  = hourly_vol_pct * math.sqrt(6.5)
                            raw_size = PER_INST_RISK / (daily_vol_pct + 1e-9)
                            cap = min(raw_size, ceiling)
                        else:
                            cap = ceiling
                    else:
                        cap = ceiling

                    tail_tgt = cap * direction

                    if i1h.regime == MarketRegime.BEAR and tail_tgt > 0 and i1h.rhb > (3 if sym == "YM" else 5):
                        tail_tgt = 0.0
                    if i1h.regime == MarketRegime.BULL and tail_tgt < 0 and i1h.rhb > 5:
                        tail_tgt = 0.0

                    # Thin-instrument caps
                    if sym in INST_CAPS:
                        inst_cap = INST_CAPS[sym] / (pv + 1e-9)
                        tail_tgt = float(np.sign(tail_tgt) * min(abs(tail_tgt), inst_cap))

            # pos_floor
            if (tf_score >= 6 and not np.isclose(tail_tgt, 0.0)
                    and abs(tail_tgt) > 0.15 and i1h.ctl >= 5):
                i1h.pos_floor = max(i1h.pos_floor, 0.70 * abs(tail_tgt))
            if (i1h.pos_floor > 0.0 and tf_score >= 4
                    and not np.isclose(i1h.last_target, 0.0)):
                tail_tgt = float(np.sign(i1h.last_target) * max(abs(tail_tgt), i1h.pos_floor))
                i1h.pos_floor *= 0.95
            if tf_score < 4 or np.isclose(tail_tgt, 0.0):
                i1h.pos_floor = 0.0
            if not i1d.bh_active and not i1h.bh_active:
                i1h.pos_floor = 0.0

            # Minimum hold gate
            is_reversal = (
                not np.isclose(i1h.last_target, 0.0) and
                not np.isclose(tail_tgt, 0.0) and
                np.sign(tail_tgt) != np.sign(i1h.last_target)
            )
            if is_reversal and i1h.bars_held < MIN_HOLD_BARS:
                tail_tgt = i1h.last_target

            raw_targets[sym] = (mapped, tail_tgt)

        total_tail_exposure = sum(abs(tgt) for _, tgt in raw_targets.values())
        tail_scale = 1.0 / total_tail_exposure if total_tail_exposure > 1.0 else 1.0

        for sym in ALL_SYMS:
            i1h = self.instr_1h[sym]
            mapped, tail_tgt = raw_targets[sym]
            if mapped is None:
                continue
            final_tail = float(tail_tgt * tail_scale * tail_frac)

            if abs(final_tail - i1h.last_target) > 0.02:
                if np.isclose(final_tail, 0.0):
                    i1h.bars_held = 0
                elif np.sign(final_tail) != np.sign(i1h.last_target):
                    i1h.bars_held = 0
                i1h.last_target = final_tail
                self.set_holdings(mapped, final_tail)

        # ── Gear 2: HARVEST MODE ─────────────────────────────────────────────
        today = self.time.date()
        if harvest_equity > 10_000 and today != self._last_harvest_day:
            self._last_harvest_day = today
            self._run_harvest(data, harvest_frac)

        # ── Diagnostics ──────────────────────────────────────────────────────
        if any(i.bc % 24 == 0 for i in self.instr_1h.values()):
            pv2 = self.portfolio.total_portfolio_value
            self.plot("Risk", "Drawdown%", (self.peak - pv2) / (self.peak + 1e-9) * 100)
            for key, inst in self.instr_1h.items():
                self.plot("Regime", key, int(inst.regime))
            self.plot("Allocation", "TailFrac%",    tail_frac * 100)
            self.plot("Allocation", "HarvestFrac%", harvest_frac * 100)

    def _run_harvest(self, data, harvest_frac):
        for sym in ALL_SYMS:
            i1h = self.instr_1h[sym]
            mapped = i1h.future.mapped
            if mapped is None: continue
            if mapped not in self.securities: continue
            if not self.securities[mapped].exchange.exchange_open: continue

            if i1h.regime != MarketRegime.SIDEWAYS:
                if not np.isclose(i1h.harv_pos, 0.0):
                    self.set_holdings(mapped, i1h.last_target)
                    i1h.harv_pos = 0.0
                continue

            z = i1h.z_score()
            harv_size = HARVEST_RISK_PER_INST * harvest_frac

            if np.isclose(i1h.harv_pos, 0.0):
                if z > HARVEST_Z_ENTRY:
                    i1h.harv_pos = -harv_size
                    i1h.harv_entry_z = z
                    self.set_holdings(mapped, i1h.last_target + i1h.harv_pos)
                elif z < -HARVEST_Z_ENTRY:
                    i1h.harv_pos = harv_size
                    i1h.harv_entry_z = z
                    self.set_holdings(mapped, i1h.last_target + i1h.harv_pos)
            else:
                in_short = i1h.harv_pos < 0
                exit_trade = False

                if in_short and z < HARVEST_Z_EXIT:
                    exit_trade = True
                elif not in_short and z > -HARVEST_Z_EXIT:
                    exit_trade = True

                if in_short and z > HARVEST_Z_STOP:
                    exit_trade = True
                elif not in_short and z < -HARVEST_Z_STOP:
                    exit_trade = True

                if exit_trade:
                    i1h.harv_pos = 0.0
                    self.set_holdings(mapped, i1h.last_target)

    def _get_direction(self, i15, i1h, i1d):
        def _resolve(inst):
            if inst.bh_dir != 0: return inst.bh_dir
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
        tail_frac = min(TAIL_FIXED_CAPITAL, pv) / pv
        harvest_frac = max(0.0, pv - TAIL_FIXED_CAPITAL) / pv
        for sym in ALL_SYMS:
            i15 = self.instr_15m[sym]
            i1h = self.instr_1h[sym]
            i1d = self.instr_1d[sym]
            self.log(
                f"[{sym}] END "
                f"regime_1h={i1h.regime.name} "
                f"bh_1d={i1d.bh_active} bh_1h={i1h.bh_active} bh_15m={i15.bh_active} "
                f"tf_score={4*int(i1d.bh_active)+2*int(i1h.bh_active)+int(i15.bh_active)} "
                f"harv_pos={i1h.harv_pos:.3f}"
            )
        self.log(
            f"PORTFOLIO peak={self.peak:.0f} final={pv:.0f} dd={dd:.1%} "
            f"tail_frac={tail_frac:.1%} harvest_frac={harvest_frac:.1%}"
        )
