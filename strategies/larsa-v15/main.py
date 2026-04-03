# LARSA v15: Four-Gear Multi-Strategy Architecture
#
# GEAR 1 — TAIL CAPTURE ($3M fixed bucket)
#   LARSA v12 BH physics. Unchanged. The $1M→$22M engine.
#   Catches vol regime breaks. HIGH_VOL and strong BULL/BEAR trends.
#   Always allocated first. Never touches harvest capital.
#
# GEAR 2 — HARVEST (everything above $6M)
#   Mean reversion. Z-score fade on hourly bars.
#   Only active in SIDEWAYS regime — exactly when Gear 1 is flat.
#   Clips 2-5 point oscillations, 3-4x per week.
#
# GEAR 3 — TREND FOLLOW ($2M fixed bucket, activates at $5M total equity)
#   Daily 50/200 EMA crossover. Holds positions for days to weeks.
#   Catches smooth trending years (2023 grind-up) that Gear 1 misses.
#   Gear 1 needs vol spikes to form BH. Gear 3 needs only direction.
#   Gate: inactive during HIGH_VOLATILITY (Gear 1 owns that regime).
#
# GEAR 4 — STAT ARB ($1M fixed bucket, activates at $6M total equity)
#   ES vs NQ spread trading. Correlation = 0.90 but they diverge.
#   NQ leads on tech moves, YM lags. Fade the spread back to convergence.
#   Market-neutral: long the laggard, short the leader.
#   Works in ALL regimes — completely uncorrelated to other gears.
#
# CAPITAL DEPLOYMENT (sequential as equity grows):
#   equity < $3M  → 100% Gear 1
#   $3M–$5M       → $3M Gear 1 + $2M Gear 3
#   $5M–$6M       → $3M Gear 1 + $2M Gear 3 + $1M Gear 4
#   $6M+          → $3M G1 + $2M G3 + $1M G4 + remainder G2
#
# All gears combined into single set_holdings call per instrument.
# Order budget: ~5,000 orders over 7 years — well within QC 10k cap.

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
    "15m":  {"ES": 0.0003, "NQ": 0.0004,  "YM": 0.00025},
    "1h":   {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008},
    "1d":   {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004},
}

TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
MIN_HOLD_BARS = 4

N_INSTRUMENTS    = 3
INST_CORRELATION = 0.90
PORTFOLIO_DAILY_RISK = 0.01
_CORR_FACTOR = math.sqrt(N_INSTRUMENTS + N_INSTRUMENTS * (N_INSTRUMENTS - 1) * INST_CORRELATION)
PER_INST_RISK = PORTFOLIO_DAILY_RISK / _CORR_FACTOR

# Capital buckets
TAIL_CAP  = 1_000_000   # Gear 1: optimal from 80-combo Monte Carlo sweep
TREND_CAP = 1_000_000   # Gear 3: activates at $2M total equity
ARB_CAP   = 0           # Gear 4: disabled — sweep showed no edge over harvest

# Gear 2: Harvest
HARVEST_Z_ENTRY  = 1.5
HARVEST_Z_EXIT   = 0.3
HARVEST_Z_STOP   = 2.8
HARVEST_LOOKBACK = 20
HARVEST_RISK     = 0.02   # 2% of harvest allocation per instrument

# Gear 3: Trend Follow
TREND_RISK       = 0.04   # 4% of trend allocation per instrument per signal
TREND_ADX_MIN    = 18     # minimum ADX to enter trend position

# Gear 4: Stat Arb
ARB_Z_ENTRY      = 2.0    # z-score to enter ES/NQ spread trade
ARB_Z_EXIT       = 0.5
ARB_Z_STOP       = 3.5
ARB_LOOKBACK     = 30     # bars for spread z-score
ARB_RISK         = 0.04   # 4% of arb allocation per leg


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
        self._is_daily = (res_label == "1d")

        self.future = algo.add_future(
            future_ticker, self._qc_res,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0,
        )
        self.future.set_filter(timedelta(0), timedelta(182))
        self.sym = self.future.symbol

        self.cw  = RollingWindow[float](50)
        self.bw  = RollingWindow[float](20)
        self.ret_w = RollingWindow[float](ARB_LOOKBACK + 5)  # for stat arb returns

        self.cf_scale = 1.0
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
        self.bars_held = 0

        # Gear 2 harvest state
        self.harv_pos     = 0.0
        self.harv_entry_z = 0.0

        # Gear 3 trend state
        self.trend_pos    = 0.0

        if self._is_hourly:
            h = Resolution.HOUR
            d = Resolution.DAILY
            self.e12  = algo.ema(self.sym, 12,  h)
            self.e26  = algo.ema(self.sym, 26,  h)
            self.e50  = algo.ema(self.sym, 50,  h)
            self.e200 = algo.ema(self.sym, 200, h)
            self.rsi  = algo.rsi(self.sym, 14, MovingAverageType.WILDERS, h)
            self.macd = algo.macd(self.sym, 12, 26, 9, MovingAverageType.EXPONENTIAL, h)
            self.atr  = algo.atr(self.sym, 14, MovingAverageType.WILDERS, h)
            self.adx  = algo.adx(self.sym, 14, h)
            self.aw   = RollingWindow[float](50)
            # Gear 3: daily trend indicators
            self.d50  = algo.ema(self.sym, 50,  d)
            self.d200 = algo.ema(self.sym, 200, d)
            self.d_adx = algo.adx(self.sym, 14, d)
            self.regime = MarketRegime.SIDEWAYS
            self.rhb = 0
            self.rc  = 0.5
            self.pos_floor = 0.0
            for ind in [self.e12, self.e26, self.e50, self.e200,
                        self.rsi, self.macd, self.atr, self.adx,
                        self.d50, self.d200, self.d_adx]:
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

    def z_score(self, window):
        if window.count < HARVEST_LOOKBACK:
            return 0.0
        prices = np.array([window[i] for i in range(HARVEST_LOOKBACK)])
        mu, std = prices.mean(), prices.std()
        return (window[0] - mu) / std if std > 1e-9 else 0.0

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
            nr = MarketRegime.BULL if adx > (14 if full_stack else 18) else MarketRegime.SIDEWAYS
            nc = min(0.95, 0.5 + (adx - 14) / 60) if nr == MarketRegime.BULL else max(0.3, 0.7 - adx / 80)
        elif price < e200 and e12 < e26:
            full_stack = e200 > self.e50.current.value > e26 > e12
            nr = MarketRegime.BEAR if adx > (14 if full_stack else 18) else MarketRegime.SIDEWAYS
            nc = min(0.95, 0.5 + (adx - 14) / 60) if nr == MarketRegime.BEAR else max(0.3, 0.7 - adx / 80)
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


class LarsaV15(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0

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
        self._last_harvest_day = None
        self._last_trend_day   = None
        self._last_arb_day     = None

        # Gear 4 stat arb state: spread return history ES-NQ, ES-YM
        self._arb_spread_w = RollingWindow[float](ARB_LOOKBACK + 5)
        self._arb_pos = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}

        self.log(
            f"[v15] tail=${TAIL_CAP:,.0f} trend=${TREND_CAP:,.0f} "
            f"arb=${ARB_CAP:,.0f} per_inst_risk={PER_INST_RISK:.4%}"
        )

        ac = Chart("Allocation")
        for g in ["Tail%", "Trend%", "Arb%", "Harvest%"]:
            ac.add_series(Series(g, SeriesType.LINE, 0))
        self.add_chart(ac)

        rc = Chart("Regime")
        for k in ["ES", "NQ", "YM"]:
            rc.add_series(Series(k, SeriesType.LINE, 0))
        self.add_chart(rc)

        dc = Chart("Risk")
        dc.add_series(Series("Drawdown%", SeriesType.LINE, 0))
        self.add_chart(dc)

        self.set_warm_up(timedelta(days=400))

    # ── 15m consolidator ─────────────────────────────────────────────────────
    def _on_15m_bar(self, inst, bar):
        if self.is_warming_up: return
        inst.bc += 1
        inst.cw.add(float(bar.close))
        inst.update_bh()
        inst.apply_warmup_gate()

    # ── Main loop ─────────────────────────────────────────────────────────────
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

        # ── Capital bucket fractions ─────────────────────────────────────────
        tail_alloc    = min(TAIL_CAP, pv)
        trend_alloc   = min(TREND_CAP, max(0.0, pv - TAIL_CAP))
        arb_alloc     = min(ARB_CAP,   max(0.0, pv - TAIL_CAP - TREND_CAP))
        harvest_alloc = max(0.0, pv - TAIL_CAP - TREND_CAP - ARB_CAP)

        tail_frac    = tail_alloc    / pv
        trend_frac   = trend_alloc   / pv
        arb_frac     = arb_alloc     / pv
        harvest_frac = harvest_alloc / pv

        # ── Compute all gear targets (as fraction of TOTAL portfolio) ─────────
        g1 = self._gear1_targets(pv, tail_frac)
        g3 = self._gear3_targets(trend_frac)
        g4 = self._gear4_targets(data, arb_frac, pv)
        g2 = self._gear2_targets(harvest_frac)

        # ── Combine and execute ───────────────────────────────────────────────
        # Scale combined targets so total portfolio exposure never exceeds 1.0
        raw_combined = {sym: g1[sym] + g3[sym] + g4[sym] + g2[sym] for sym in ["ES","NQ","YM"]}
        total_exp = sum(abs(v) for v in raw_combined.values())
        port_scale = 1.0 / total_exp if total_exp > 1.0 else 1.0

        for sym in ["ES", "NQ", "YM"]:
            i1h = self.instr_1h[sym]
            mapped = i1h.future.mapped
            if mapped is None: continue
            if mapped not in self.securities: continue
            if not self.securities[mapped].exchange.exchange_open: continue

            combined = float(raw_combined[sym] * port_scale)

            if abs(combined - i1h.last_target) > 0.02:
                if np.isclose(combined, 0.0):
                    i1h.bars_held = 0
                elif np.sign(combined) != np.sign(i1h.last_target):
                    i1h.bars_held = 0
                i1h.last_target = combined
                self.set_holdings(mapped, combined)

        # ── Diagnostics ───────────────────────────────────────────────────────
        if any(i.bc % 24 == 0 for i in self.instr_1h.values()):
            pv2 = self.portfolio.total_portfolio_value
            self.plot("Risk", "Drawdown%", (self.peak - pv2) / (self.peak + 1e-9) * 100)
            for k, inst in self.instr_1h.items():
                self.plot("Regime", k, int(inst.regime))
            self.plot("Allocation", "Tail%",    tail_frac    * 100)
            self.plot("Allocation", "Trend%",   trend_frac   * 100)
            self.plot("Allocation", "Arb%",     arb_frac     * 100)
            self.plot("Allocation", "Harvest%", harvest_frac * 100)

    # ── GEAR 1: Tail Capture (v12 BH physics, unchanged) ─────────────────────
    def _gear1_targets(self, pv, tail_frac):
        targets = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}

        for i1h in self.instr_1h.values():
            if abs(i1h.last_target) > 0.02:
                i1h.bars_held += 1

        raw = {}
        for sym in ["ES", "NQ", "YM"]:
            i15 = self.instr_15m[sym]
            i1h = self.instr_1h[sym]
            i1d = self.instr_1d[sym]
            mapped = i1h.future.mapped
            if mapped is None or mapped not in self.securities:
                raw[sym] = 0.0; continue

            tf_score = (4 * int(i1d.bh_active) + 2 * int(i1h.bh_active) + int(i15.bh_active))
            ceiling  = TF_CAP[tf_score]
            if tf_score == 1 and np.isclose(i1h.last_target, 0.0):
                ceiling = 0.0

            if ceiling == 0.0:
                raw[sym] = 0.0; continue

            direction = self._get_direction(i15, i1h, i1d)
            if direction == 0:
                raw[sym] = 0.0; continue

            if i1h.atr.is_ready and i1h.atr.current.value > 0:
                price = self.securities[mapped].price
                if price > 0:
                    hv  = i1h.atr.current.value / price
                    dv  = hv * math.sqrt(6.5)
                    cap = min(PER_INST_RISK / (dv + 1e-9), ceiling)
                else:
                    cap = ceiling
            else:
                cap = ceiling

            tgt = cap * direction

            if i1h.regime == MarketRegime.BEAR and tgt > 0 and i1h.rhb > (3 if sym == "YM" else 5):
                tgt = 0.0
            if i1h.regime == MarketRegime.BULL and tgt < 0 and i1h.rhb > 5:
                tgt = 0.0
            if sym == "NQ" and i1h.regime != MarketRegime.BULL:
                nq_cap = 400000.0 / (self.portfolio.total_portfolio_value + 1e-9)
                tgt = float(np.sign(tgt) * min(abs(tgt), nq_cap))

            # pos_floor
            if tf_score >= 6 and not np.isclose(tgt, 0.0) and abs(tgt) > 0.15 and i1h.ctl >= 5:
                i1h.pos_floor = max(i1h.pos_floor, 0.70 * abs(tgt))
            if i1h.pos_floor > 0.0 and tf_score >= 4 and not np.isclose(i1h.last_target, 0.0):
                tgt = float(np.sign(i1h.last_target) * max(abs(tgt), i1h.pos_floor))
                i1h.pos_floor *= 0.95
            if tf_score < 4 or np.isclose(tgt, 0.0):
                i1h.pos_floor = 0.0
            if not i1d.bh_active and not i1h.bh_active:
                i1h.pos_floor = 0.0

            is_reversal = (not np.isclose(i1h.last_target, 0.0) and
                           not np.isclose(tgt, 0.0) and
                           np.sign(tgt) != np.sign(i1h.last_target))
            if is_reversal and i1h.bars_held < MIN_HOLD_BARS:
                tgt = i1h.last_target

            raw[sym] = tgt

        total_exp = sum(abs(v) for v in raw.values())
        scale = 1.0 / total_exp if total_exp > 1.0 else 1.0
        for sym in ["ES", "NQ", "YM"]:
            targets[sym] = raw[sym] * scale * tail_frac
        return targets

    # ── GEAR 2: Harvest / Mean Reversion ─────────────────────────────────────
    def _gear2_targets(self, harvest_frac):
        targets = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
        if harvest_frac < 0.01:
            return targets

        today = self.time.date()
        if today == self._last_harvest_day:
            # Return existing harvest positions
            for sym in ["ES", "NQ", "YM"]:
                targets[sym] = self.instr_1h[sym].harv_pos
            return targets
        self._last_harvest_day = today

        harv_size = HARVEST_RISK * harvest_frac

        for sym in ["ES", "NQ", "YM"]:
            i1h = self.instr_1h[sym]
            if i1h.regime != MarketRegime.SIDEWAYS:
                i1h.harv_pos = 0.0
                targets[sym] = 0.0
                continue

            z = i1h.z_score(i1h.cw)

            if np.isclose(i1h.harv_pos, 0.0):
                if z > HARVEST_Z_ENTRY:
                    i1h.harv_pos = -harv_size
                elif z < -HARVEST_Z_ENTRY:
                    i1h.harv_pos = harv_size
            else:
                in_short = i1h.harv_pos < 0
                exit_now = False
                if in_short  and z < HARVEST_Z_EXIT:   exit_now = True
                if not in_short and z > -HARVEST_Z_EXIT: exit_now = True
                if in_short  and z > HARVEST_Z_STOP:   exit_now = True
                if not in_short and z < -HARVEST_Z_STOP: exit_now = True
                if exit_now:
                    i1h.harv_pos = 0.0

            targets[sym] = i1h.harv_pos
        return targets

    # ── GEAR 3: Trend Follow (daily 50/200 EMA) ───────────────────────────────
    def _gear3_targets(self, trend_frac):
        targets = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
        if trend_frac < 0.01:
            return targets

        today = self.time.date()
        if today == self._last_trend_day:
            for sym in ["ES", "NQ", "YM"]:
                targets[sym] = self.instr_1h[sym].trend_pos
            return targets
        self._last_trend_day = today

        for sym in ["ES", "NQ", "YM"]:
            i1h = self.instr_1h[sym]

            # Skip during HIGH_VOLATILITY — Gear 1 owns that regime
            if i1h.regime == MarketRegime.HIGH_VOLATILITY:
                i1h.trend_pos = 0.0
                targets[sym]  = 0.0
                continue

            if not (i1h.d50.is_ready and i1h.d200.is_ready and i1h.d_adx.is_ready):
                targets[sym] = i1h.trend_pos
                continue

            d50  = i1h.d50.current.value
            d200 = i1h.d200.current.value
            adx  = i1h.d_adx.current.value

            trend_size = TREND_RISK * trend_frac

            if d50 > d200 and adx > TREND_ADX_MIN:
                # Golden cross with sufficient trend strength → LONG
                i1h.trend_pos = trend_size
            elif d50 < d200 and adx > TREND_ADX_MIN:
                # Death cross → SHORT
                i1h.trend_pos = -trend_size
            else:
                # No clear trend — flat
                i1h.trend_pos = 0.0

            targets[sym] = i1h.trend_pos
        return targets

    # ── GEAR 4: Stat Arb (ES vs NQ spread) ───────────────────────────────────
    def _gear4_targets(self, data, arb_frac, pv):
        targets = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
        if arb_frac < 0.01:
            return targets

        today = self.time.date()
        if today == self._last_arb_day:
            return dict(self._arb_pos)
        self._last_arb_day = today

        i_es = self.instr_1h["ES"]
        i_nq = self.instr_1h["NQ"]

        if i_es.cw.count < 2 or i_nq.cw.count < 2:
            return targets

        # Spread: ES hourly return minus NQ hourly return
        es_ret = (i_es.cw[0] - i_es.cw[1]) / (i_es.cw[1] + 1e-9)
        nq_ret = (i_nq.cw[0] - i_nq.cw[1]) / (i_nq.cw[1] + 1e-9)
        spread = es_ret - nq_ret
        self._arb_spread_w.add(spread)

        if self._arb_spread_w.count < ARB_LOOKBACK:
            return targets

        spreads = np.array([self._arb_spread_w[i] for i in range(ARB_LOOKBACK)])
        mu, std = spreads.mean(), spreads.std()
        if std < 1e-9:
            return targets
        z = (spread - mu) / std

        arb_size = ARB_RISK * arb_frac
        cur_es = self._arb_pos["ES"]
        cur_nq = self._arb_pos["NQ"]

        if np.isclose(cur_es, 0.0) and np.isclose(cur_nq, 0.0):
            # No position — look for entry
            if z > ARB_Z_ENTRY:
                # ES outperformed NQ → ES will revert down, NQ up
                # Short ES, Long NQ
                self._arb_pos["ES"] = -arb_size
                self._arb_pos["NQ"] =  arb_size
            elif z < -ARB_Z_ENTRY:
                # NQ outperformed ES → Long ES, Short NQ
                self._arb_pos["ES"] =  arb_size
                self._arb_pos["NQ"] = -arb_size
        else:
            # In position — check exit
            in_short_es = cur_es < 0
            exit_now = False
            if in_short_es  and z < ARB_Z_EXIT:    exit_now = True
            if not in_short_es and z > -ARB_Z_EXIT: exit_now = True
            if abs(z) > ARB_Z_STOP:                 exit_now = True  # stop
            if exit_now:
                self._arb_pos["ES"] = 0.0
                self._arb_pos["NQ"] = 0.0

        for sym in ["ES", "NQ", "YM"]:
            targets[sym] = self._arb_pos[sym]
        return targets

    # ── Helpers ───────────────────────────────────────────────────────────────
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
        if inst.cw.count >= 2:
            ret = (inst.cw[0] - inst.cw[1]) / (inst.cw[1] + 1e-9)
            inst.ret_w.add(ret)
        inst.update_bh()
        if inst._is_hourly:
            if inst.atr.is_ready:
                inst.aw.add(inst.atr.current.value)
            inst.detect_regime()
        inst.apply_warmup_gate()

    def on_end_of_algorithm(self):
        pv = self.portfolio.total_portfolio_value
        dd = (self.peak - pv) / (self.peak + 1e-9)
        tail_frac    = min(TAIL_CAP, pv) / pv
        trend_frac   = min(TREND_CAP, max(0.0, pv - TAIL_CAP)) / pv
        arb_frac     = min(ARB_CAP,   max(0.0, pv - TAIL_CAP - TREND_CAP)) / pv
        harvest_frac = max(0.0, pv - TAIL_CAP - TREND_CAP - ARB_CAP) / pv
        self.log(
            f"PORTFOLIO peak={self.peak:.0f} final={pv:.0f} dd={dd:.1%} | "
            f"tail={tail_frac:.1%} trend={trend_frac:.1%} "
            f"arb={arb_frac:.1%} harvest={harvest_frac:.1%}"
        )
        for sym in ["ES", "NQ", "YM"]:
            i1h = self.instr_1h[sym]
            self.log(
                f"[{sym}] regime={i1h.regime.name} rhb={i1h.rhb} "
                f"bh_active={i1h.bh_active} trend_pos={i1h.trend_pos:.3f} "
                f"harv_pos={i1h.harv_pos:.3f} arb_pos={self._arb_pos[sym]:.3f}"
            )
