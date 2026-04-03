# LARSA v12: v11 + regime-aware BH formation (BULL regime fix)
#
# THE BUG IN v11:
#   CF threshold calibrated for crash moves (0.1%/hr → beta=1)
#   Bull market moves are 0.3%/hr → beta=3 (SPACELIKE) → BH never forms in bull trends
#   When BH does form in bull consolidations, 20-bar lookback shows DOWN → direction=SHORT
#   Result: -8.6% edge in BULL, strategy misses entire bull-market upside
#
# THE FIX:
#   1. Regime-scaled CF: in BULL regime, CF×3 so 0.3%/hr is TIMELIKE (beta=1)
#      BH can now form during sustained bull trends → strategy goes LONG in bull markets
#   2. Symmetric direction gate: suppress shorts in BULL (mirrors existing BEAR long-suppression)
#
# INHERITED FROM v11 (below comment block retained for reference):
# LARSA v11: v10 + correlation-adjusted PORTFOLIO-level risk budget
#
# THE BUG IN v10:
#   TARGET_DAILY_RISK = 1% per instrument
#   ES/NQ/YM correlation = 0.90 → they move together
#   Effective portfolio risk = 1% × sqrt(3 + 3×2×0.90) = 1% × 2.898 = 2.9% per day
#   At $19.5M peak → $585k/day expected loss in a correlated crash
#   5 bad days = -14% → margin calls → cascade
#
# THE FIX:
#   PORTFOLIO_DAILY_RISK = 1% TOTAL (not per instrument)
#   Per-instrument allocation = PORTFOLIO_DAILY_RISK / corr_factor
#   corr_factor = sqrt(N + N×(N-1)×CORR) = sqrt(3 + 6×0.90) = 2.898
#   Per-instrument risk = 1% / 2.898 = 0.345%
#
# WHAT THIS MEANS:
#   At $19.5M, normal vol:  position per inst ~0.338 (was 0.55 in v10)
#   Worst correlated day:   $195k loss (was $585k in v10)
#   After 5 bad days:       -4.9% (was -14.1% in v10)
#   In Volmageddon (6× vol): position per inst ~0.057 (was ~0.092 in v10)
#   Signal uncapped in calm: still hits ceiling when vol is low and signal is strong
#
# All other v9/v10 logic preserved: BH physics, 4-bar hold, hourly gate, portfolio cap.
#
# QC NOTE: Resolution.FIFTEEN_MINUTE does not exist in QC.

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

# v11: PORTFOLIO-level 1% daily risk budget, correlation-adjusted
# ES/NQ/YM average pairwise correlation (empirical): 0.90
# corr_factor = sqrt(N + N*(N-1)*CORR) where N=3, CORR=0.90
# = sqrt(3 + 6*0.90) = sqrt(8.4) = 2.898
# Per-instrument risk = 0.01 / 2.898 = 0.3450%
N_INSTRUMENTS    = 3
INST_CORRELATION = 0.90
PORTFOLIO_DAILY_RISK = 0.01
_CORR_FACTOR = math.sqrt(N_INSTRUMENTS + N_INSTRUMENTS * (N_INSTRUMENTS - 1) * INST_CORRELATION)
PER_INST_RISK = PORTFOLIO_DAILY_RISK / _CORR_FACTOR   # ≈ 0.003450


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

        self.cf_scale = 1.0   # v12: regime-scaled CF multiplier (3.0 in BULL)
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
        # v12: scale CF up in BULL so typical bull moves (0.3%/hr) are TIMELIKE
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


class LarsaV12(QCAlgorithm):
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

        # Log the correlation-adjusted risk parameters at startup
        self.log(
            f"[v12 RISK PARAMS] "
            f"portfolio_daily_risk={PORTFOLIO_DAILY_RISK:.2%} "
            f"corr_factor={_CORR_FACTOR:.4f} "
            f"per_inst_risk={PER_INST_RISK:.4%}"
        )

        for c in [
            Chart("Regime"), Chart("Risk"), Chart("VolSizing"), Chart("PortfolioRisk")
        ]:
            pass

        rc = Chart("Regime")
        for k in ["ES", "NQ", "YM"]:
            rc.add_series(Series(k, SeriesType.LINE, 0))
        self.add_chart(rc)

        dc = Chart("Risk")
        dc.add_series(Series("Drawdown%", SeriesType.LINE, 0))
        self.add_chart(dc)

        vc = Chart("VolSizing")
        for k in ["ES_size", "NQ_size", "YM_size"]:
            vc.add_series(Series(k, SeriesType.LINE, 0))
        self.add_chart(vc)

        pr = Chart("PortfolioRisk")
        pr.add_series(Series("DailyRisk%", SeriesType.LINE, 0))
        pr.add_series(Series("Target%",   SeriesType.LINE, 0))
        self.add_chart(pr)

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

        for i1h in self.instr_1h.values():
            if abs(i1h.last_target) > 0.02:
                i1h.bars_held += 1

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

            ceiling = TF_CAP[tf_score]

            if tf_score == 1 and np.isclose(i1h.last_target, 0.0):
                ceiling = 0.0

            if ceiling == 0.0:
                tgt = 0.0
            else:
                direction = self._get_direction(i15, i1h, i1d)
                if direction == 0:
                    tgt = 0.0
                else:
                    # ── v11: correlation-adjusted portfolio risk sizing ───
                    # Each instrument gets PER_INST_RISK of equity as daily risk budget.
                    # PER_INST_RISK = PORTFOLIO_DAILY_RISK / corr_factor
                    # corr_factor accounts for the fact that ES/NQ/YM move together.
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
                    # ─────────────────────────────────────────────────────

                    tgt = cap * direction

                    if i1h.regime == MarketRegime.BEAR and tgt > 0 and i1h.rhb > (3 if sym == "YM" else 5):
                        tgt = 0.0
                    # v12: symmetric gate — suppress shorts in sustained bull markets
                    if i1h.regime == MarketRegime.BULL and tgt < 0 and i1h.rhb > 5:
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

            # Minimum hold gate
            is_reversal = (
                not np.isclose(i1h.last_target, 0.0) and
                not np.isclose(tgt, 0.0) and
                np.sign(tgt) != np.sign(i1h.last_target)
            )
            if is_reversal and i1h.bars_held < MIN_HOLD_BARS:
                tgt = i1h.last_target

            raw_targets[sym] = (mapped, tgt)

        # Portfolio exposure cap at 1.0
        total_exposure = sum(abs(tgt) for _, tgt in raw_targets.values())
        scale = 1.0 / total_exposure if total_exposure > 1.0 else 1.0

        for sym in ["ES", "NQ", "YM"]:
            i1h = self.instr_1h[sym]
            mapped, tgt = raw_targets[sym]
            if mapped is None:
                continue
            tgt = float(tgt * scale)

            if abs(tgt - i1h.last_target) > 0.02:
                if np.isclose(tgt, 0.0):
                    i1h.bars_held = 0
                elif np.sign(tgt) != np.sign(i1h.last_target):
                    i1h.bars_held = 0
                i1h.last_target = tgt
                self.set_holdings(mapped, tgt)

        if any(i.bc % 24 == 0 for i in self.instr_1h.values()):
            pv2 = self.portfolio.total_portfolio_value
            self.plot("Risk", "Drawdown%", (self.peak - pv2) / (self.peak + 1e-9) * 100)
            for key, inst in self.instr_1h.items():
                self.plot("Regime", key, int(inst.regime))

            # VolSizing chart: actual position sizes
            total_risk_est = 0.0
            for sym in ["ES", "NQ", "YM"]:
                i1h = self.instr_1h[sym]
                mapped = i1h.future.mapped
                if mapped and i1h.atr.is_ready and i1h.atr.current.value > 0:
                    try:
                        price = self.securities[mapped].price
                        if price > 0:
                            hv = i1h.atr.current.value / price
                            dv = hv * math.sqrt(6.5)
                            sz = min(PER_INST_RISK / (dv + 1e-9), TF_CAP.get(7, 0.65))
                            self.plot("VolSizing", f"{sym}_size", sz)
                            total_risk_est += abs(i1h.last_target) * dv
                    except Exception:
                        pass
            self.plot("PortfolioRisk", "DailyRisk%", total_risk_est * 100)
            self.plot("PortfolioRisk", "Target%", PORTFOLIO_DAILY_RISK * 100)

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
        for sym in ["ES", "NQ", "YM"]:
            i15 = self.instr_15m[sym]
            i1h = self.instr_1h[sym]
            i1d = self.instr_1d[sym]
            vol_info = "atr_not_ready"
            mapped = i1h.future.mapped
            if mapped and i1h.atr.is_ready and i1h.atr.current.value > 0:
                try:
                    price = self.securities[mapped].price
                    if price > 0:
                        hv = i1h.atr.current.value / price
                        dv = hv * math.sqrt(6.5)
                        sz = PER_INST_RISK / (dv + 1e-9)
                        vol_info = (f"atr={i1h.atr.current.value:.1f} "
                                    f"daily_vol={dv:.4f} "
                                    f"vol_size={sz:.3f} "
                                    f"per_inst_risk={PER_INST_RISK:.4%}")
                except Exception:
                    pass
            self.log(
                f"[{sym}] END "
                f"regime_1h={i1h.regime.name} "
                f"bars_held={i1h.bars_held} "
                f"bc_15m={i15.bc} bc_1h={i1h.bc} bc_1d={i1d.bc} "
                f"bh_1d={i1d.bh_active} bh_1h={i1h.bh_active} bh_15m={i15.bh_active} "
                f"tf_score={4*int(i1d.bh_active)+2*int(i1h.bh_active)+int(i15.bh_active)} "
                f"vol_sizing=[{vol_info}]"
            )
        self.log(
            f"PORTFOLIO peak={self.peak:.0f} final={pv:.0f} dd={dd:.1%} "
            f"portfolio_risk_target={PORTFOLIO_DAILY_RISK:.2%} "
            f"per_inst_risk={PER_INST_RISK:.4%} "
            f"corr_factor={_CORR_FACTOR:.4f}"
        )
