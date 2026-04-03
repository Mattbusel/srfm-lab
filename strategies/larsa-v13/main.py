# LARSA v13: v12 + concave equity scaling + trailing anchor + circuit breaker
#
# THE DEATH SPIRAL DIAGNOSIS (from QC order logs, 2018-2024):
#   The vol-targeting IS working correctly — contracts barely grow as equity grows.
#   At $1M: ~25x leverage. At $5.86M peak: ~10x leverage. Sizing scales down. ✓
#
#   The ACTUAL bug: single-session 18-24% equity wipes from whipsaw at regime transitions.
#   Pattern: strategy closes position at cycle peak, immediately re-enters opposite direction
#   at the worst price, gets caught by V-shaped reversal. ATR lag window ~14 bars.
#   At $1M: 14 bars of max-size wrong-direction costs ~$150K (recoverable).
#   At $22M: same 14 bars costs ~$3.3M per session → death spiral.
#
#   Root cause: sizing is LINEAR with equity. The ATR lag window cost scales 1:1 with
#   account size. The strategy's edge (signal quality) does NOT scale with account size.
#   So marginal risk grows linearly while marginal edge stays constant → Kelly says
#   you are overbetting at large equity.
#
# THE FIX — three layers:
#
# 1. CONCAVE EQUITY SCALING (fractional Kelly tiers):
#    equity < $2M   → base_risk = 0.010  (full aggression — same as v10)
#    $2M–$5M        → base_risk = 0.007
#    $5M–$10M       → base_risk = 0.005
#    $10M+          → base_risk = 0.003
#    Sizing function goes from linear → concave. Strategy can still compound to $15M+
#    but position growth decelerates instead of going vertical then collapsing.
#
# 2. TRAILING EQUITY ANCHOR (20-bar EMA of portfolio value):
#    Size off EMA(equity) not spot equity.
#    - Equity spikes $5M→$10M in one week → anchor still at $6M → no sudden 2x position ramp
#    - Equity drops $10M→$7M → anchor still at $9M → no panic collapse of sizing
#    - Gives positions time to adjust gradually in both directions
#    - Prevents the strategy from going max-size right at a cycle peak
#
# 3. CIRCUIT BREAKER (hard backstop):
#    If portfolio_value < peak × 0.85 (15% drawdown from ATH):
#    → Liquidate all positions immediately
#    → Go flat for CIRCUIT_BREAKER_BARS bars (48h cooldown)
#    → Resume at 50% normal sizing for another 48h, then full
#    Stops the death spiral at -15% instead of -99%.
#
# INHERITED: all v12 logic (BH physics, regime-scaled CF, BULL/BEAR gates, 4-bar hold)
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

# v13: Concave equity scaling — fractional Kelly tiers
# corr_factor still used to distribute portfolio risk across 3 correlated instruments
N_INSTRUMENTS    = 3
INST_CORRELATION = 0.90
_CORR_FACTOR = math.sqrt(N_INSTRUMENTS + N_INSTRUMENTS * (N_INSTRUMENTS - 1) * INST_CORRELATION)

# Equity tiers: base daily risk budget decays as equity grows
# Below $2M: full aggression (same as original v10) — this is where the big runs come from
# Above $10M: conservative — ATR lag cost at this scale is structurally lethal
_EQUITY_TIERS = [
    (2_000_000,  0.010),   # < $2M  → 1.0% portfolio daily risk
    (5_000_000,  0.007),   # < $5M  → 0.7%
    (10_000_000, 0.005),   # < $10M → 0.5%
    (float("inf"), 0.003), # $10M+  → 0.3%
]

def _get_portfolio_risk(equity_anchor: float) -> float:
    for threshold, risk in _EQUITY_TIERS:
        if equity_anchor < threshold:
            return risk
    return 0.003

# Trailing anchor: 20-bar EMA of portfolio value (set in initialize)
_EQUITY_EMA_PERIOD = 20
_EQUITY_EMA_K      = 2.0 / (_EQUITY_EMA_PERIOD + 1)

# Circuit breaker
CIRCUIT_BREAKER_DD   = 0.15   # go flat if equity < peak × (1 - 0.15)
CIRCUIT_BREAKER_BARS = 48     # cooldown bars before resuming
CIRCUIT_BREAKER_HALF = 48     # bars at half-size before returning to full


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


class LarsaV13(QCAlgorithm):
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

        # v13: trailing equity anchor (20-bar EMA of portfolio value)
        self._equity_ema   = 1_000_000.0

        # v13: circuit breaker state
        self._cb_flat_bars = 0    # bars remaining in full-flat cooldown
        self._cb_half_bars = 0    # bars remaining in half-size recovery

        self.log(
            f"[v13 RISK PARAMS] "
            f"corr_factor={_CORR_FACTOR:.4f} "
            f"tiers={_EQUITY_TIERS} "
            f"circuit_breaker_dd={CIRCUIT_BREAKER_DD:.0%} "
            f"ema_period={_EQUITY_EMA_PERIOD}"
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

        # ── v13: update trailing equity EMA anchor ───────────────────────────
        self._equity_ema = (self._equity_ema * (1 - _EQUITY_EMA_K)
                            + pv * _EQUITY_EMA_K)

        # ── v13: circuit breaker check ────────────────────────────────────────
        if self._cb_flat_bars > 0:
            self._cb_flat_bars -= 1
            for inst in self.non_15m_instruments.values():
                self._process_instrument(data, inst)
            return  # flat — no orders
        if pv < self.peak * (1.0 - CIRCUIT_BREAKER_DD):
            self.log(f"[v13 CIRCUIT BREAKER] pv={pv:.0f} peak={self.peak:.0f} "
                     f"dd={(self.peak-pv)/self.peak:.1%} — LIQUIDATING ALL")
            for sym in ["ES", "NQ", "YM"]:
                i1h = self.instr_1h[sym]
                mapped = i1h.future.mapped
                if mapped:
                    self.liquidate(mapped)
                    i1h.last_target = 0.0
                    i1h.bars_held   = 0
                    i1h.pos_floor   = 0.0
            self._cb_flat_bars = CIRCUIT_BREAKER_BARS
            self._cb_half_bars = CIRCUIT_BREAKER_BARS + CIRCUIT_BREAKER_HALF
            for inst in self.non_15m_instruments.values():
                self._process_instrument(data, inst)
            return

        # Half-size recovery window after circuit breaker
        cb_size_scale = 0.5 if self._cb_half_bars > 0 else 1.0
        if self._cb_half_bars > 0:
            self._cb_half_bars -= 1

        for inst in self.non_15m_instruments.values():
            self._process_instrument(data, inst)

        current_hour = self.time.replace(minute=0, second=0, microsecond=0)
        if current_hour == self._last_exec_hour:
            return
        self._last_exec_hour = current_hour

        # ── v13: compute per-instrument risk from equity tier + EMA anchor ───
        equity_anchor  = self._equity_ema
        port_risk      = _get_portfolio_risk(equity_anchor)
        per_inst_risk  = port_risk / _CORR_FACTOR

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
                    # ── v13: concave equity scaling via EMA anchor + tier ────
                    # per_inst_risk decays as equity_anchor grows (fractional Kelly)
                    # equity_anchor = 20-bar EMA of portfolio value — prevents
                    # ramping to max size right at a cycle peak or panicking at trough
                    if i1h.atr.is_ready and i1h.atr.current.value > 0:
                        price = self.securities[mapped].price
                        if price > 0:
                            hourly_vol_pct = i1h.atr.current.value / price
                            daily_vol_pct  = hourly_vol_pct * math.sqrt(6.5)
                            raw_size = per_inst_risk / (daily_vol_pct + 1e-9)
                            cap = min(raw_size, ceiling)
                        else:
                            cap = ceiling
                    else:
                        cap = ceiling
                    cap *= cb_size_scale  # apply half-size during CB recovery
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
                            cur_per_inst = _get_portfolio_risk(self._equity_ema) / _CORR_FACTOR
                            sz = min(cur_per_inst / (dv + 1e-9), TF_CAP.get(7, 0.65))
                            self.plot("VolSizing", f"{sym}_size", sz)
                            total_risk_est += abs(i1h.last_target) * dv
                    except Exception:
                        pass
            cur_port_risk = _get_portfolio_risk(self._equity_ema)
            self.plot("PortfolioRisk", "DailyRisk%", total_risk_est * 100)
            self.plot("PortfolioRisk", "Target%",    cur_port_risk * 100)
            self.plot("PortfolioRisk", "EquityEMA",  self._equity_ema)

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
                        fin_per_inst = _get_portfolio_risk(self._equity_ema) / _CORR_FACTOR
                        sz = fin_per_inst / (dv + 1e-9)
                        vol_info = (f"atr={i1h.atr.current.value:.1f} "
                                    f"daily_vol={dv:.4f} "
                                    f"vol_size={sz:.3f} "
                                    f"per_inst_risk={fin_per_inst:.4%}")
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
        fin_port_risk = _get_portfolio_risk(self._equity_ema)
        self.log(
            f"PORTFOLIO peak={self.peak:.0f} final={pv:.0f} dd={dd:.1%} "
            f"equity_ema={self._equity_ema:.0f} "
            f"final_port_risk={fin_port_risk:.2%} "
            f"corr_factor={_CORR_FACTOR:.4f} "
            f"cb_state: flat_bars_left={self._cb_flat_bars} half_bars_left={self._cb_half_bars}"
        )
