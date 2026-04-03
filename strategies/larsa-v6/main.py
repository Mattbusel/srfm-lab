# LARSA v6: convergence-first sizing | solo cap 0.15-0.25 | pos_floor conv-only | SIDEWAYS solo gate
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

class FutureInstrument:
    def __init__(self, algo, future_ticker, resolution, cf, label):
        self.algo = algo
        self.label = label
        self.cf = cf
        self.future = algo.add_future(future_ticker, resolution,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO, contract_depth_offset=0)
        self.future.set_filter(timedelta(0), timedelta(182))
        self.sym = self.future.symbol
        h = resolution
        self.e12 = algo.ema(self.sym, 12, h)
        self.e26 = algo.ema(self.sym, 26, h)
        self.e50 = algo.ema(self.sym, 50, h)
        self.e200 = algo.ema(self.sym, 200, h)
        self.rsi = algo.rsi(self.sym, 14, MovingAverageType.WILDERS, h)
        self.macd = algo.macd(self.sym, 12, 26, 9, MovingAverageType.EXPONENTIAL, h)
        self.mom = algo.mom(self.sym, 10, h)
        self.roc = algo.roc(self.sym, 14, h)
        self.atr = algo.atr(self.sym, 14, MovingAverageType.WILDERS, h)
        self.bb = algo.bb(self.sym, 20, 2, MovingAverageType.SIMPLE, h)
        self.std = algo.std(self.sym, 20, h)
        self.adx = algo.adx(self.sym, 14, h)
        self.aw = RollingWindow[float](50)
        self.ow = RollingWindow[float](20)
        self.rw = RollingWindow[float](14)
        self.cw = RollingWindow[float](50)
        self.vw = RollingWindow[float](20)
        self.tlw = RollingWindow[float](50)
        self.bw = RollingWindow[float](20)
        self.regime = MarketRegime.SIDEWAYS
        self.rc = 0.5
        self.rhb = 0
        self.rweights = {
            MarketRegime.BULL: np.array([0.40, 0.35, 0.25]),
            MarketRegime.BEAR: np.array([0.25, 0.40, 0.35]),
            MarketRegime.SIDEWAYS: np.array([0.30, 0.30, 0.40]),
            MarketRegime.HIGH_VOLATILITY: np.array([0.25, 0.35, 0.40]),
        }
        self.bit = "UNKNOWN"
        self.pz = 0.0
        self.ht = 0.0
        self.bc = 0
        self.bh_mass = 0.0
        self.bh_form = 1.5
        self.bh_collapse = 1.0
        self.bh_decay = 0.95
        self.bh_active = False
        self.ctl = 0
        self.cum_disp = 0.0
        self.bh_dir = 0
        self.bear_bounce = False
        self.bear_neg_ctl = 0
        self.tl_confirm = 0
        self.proper_time = 0.0
        self.pt_threshold = 0.5
        self.max_vol = 0.01
        self.geo_dev = 0.0
        self.geo_slope = 0.0
        self.causal_frac = 1.0
        self.rapidity = 0.0
        self.mu = 1.0
        self.ht_exit = False
        self.weak_bars = 0
        self.pos_floor = 0.0
        self.trade_entry_bar = 0
        self.trade_entry_pv = 0.0
        self.agent_signals = (0.0, 0.0, 0.0)
        self.last_target = 0.0
        self.beta_hist = []
        self.pat_ret = {}
        self.sig_win = 5
        self.pat_lb = 500
        self.mapped_sym = None
        self.well_entry_pv = 0.0
        self.well_gain = 0.0
        self.prev_bh_mass = 0.0
        self.reform_bars = 0
        for ind in [self.e12, self.e26, self.e50, self.e200, self.rsi, self.macd,
                    self.mom, self.roc, self.atr, self.bb, self.std, self.adx]:
            algo.warm_up_indicator(self.sym, ind, h)

    def ind_ready(self):
        return all(i.is_ready for i in [self.e12, self.e26, self.e50, self.e200,
            self.rsi, self.macd, self.mom, self.roc, self.atr, self.bb, self.std, self.adx])

    def detect_regime(self):
        if not self.ind_ready() or not self.cw.is_ready: return
        price = self.cw[0]
        atr = self.atr.current.value
        adx = self.adx.current.value
        e12 = self.e12.current.value
        e26 = self.e26.current.value
        e200 = self.e200.current.value
        atr_ratio = 1.0
        if self.aw.is_ready:
            arr = np.array([self.aw[i] for i in range(int(self.aw.count))])
            atr_ratio = atr / arr.mean() if arr.mean() > 0 else 1.0
        prev = self.regime
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

    def compute_features(self):
        if not self.ind_ready() or not self.cw.is_ready: return None
        price = self.cw[0]
        cw = 1.0 if self.bit == "TIMELIKE" else 0.3
        rsi = self.rsi.current.value / 100.0
        mv = self.macd.current.value
        ms = self.macd.signal.current.value
        mh = self.macd.histogram.current.value
        mn = np.tanh(mv / (price * 0.01 + 1e-9)) * cw
        msn = np.tanh(ms / (price * 0.01 + 1e-9)) * cw
        mhn = np.tanh(mh / (price * 0.01 + 1e-9)) * cw
        momnorm = np.tanh(self.mom.current.value / (price * 0.05 + 1e-9)) * cw
        rocnorm = np.tanh(self.roc.current.value / 10.0) * cw
        atr = self.atr.current.value
        atr_pct = atr / price
        bbu = self.bb.upper_band.current.value
        bbl = self.bb.lower_band.current.value
        bbm = self.bb.middle_band.current.value
        bbw = (bbu - bbl) / (bbm + 1e-9)
        bbp = (price - bbl) / (bbu - bbl + 1e-9)
        bbd = np.tanh((price - bbm) / (atr + 1e-9))
        stn = np.tanh(self.std.current.value / (price * 0.02 + 1e-9))
        e12v = self.e12.current.value; e26v = self.e26.current.value
        e50v = self.e50.current.value; e200v = self.e200.current.value
        d12 = np.tanh((price - e12v) / (atr + 1e-9))
        d26 = np.tanh((price - e26v) / (atr + 1e-9))
        d50 = np.tanh((price - e50v) / (atr + 1e-9))
        d200 = np.tanh((price - e200v) / (atr + 1e-9))
        ex = np.tanh((e12v - e26v) / (atr + 1e-9))
        ex2 = np.tanh((e12v - e50v) / (atr + 1e-9))
        adxn = self.adx.current.value / 100.0
        volr = 0.0
        if self.vw.is_ready:
            va = np.array([self.vw[i] for i in range(int(min(20, self.vw.count)))])
            volr = np.tanh((va[0] / (va.mean() + 1e-9)) - 1.0)
        obs = 0.0
        if self.ow.count >= 5:
            ro = np.array([self.ow[i] for i in range(5)])
            obs = np.tanh((ro[0] - ro[4]) / (abs(ro[4]) + 1e-9))
        lr = np.tanh(np.log(self.cw[0] / (self.cw[1] + 1e-9)) * 100) if self.cw.count >= 2 else 0.0
        r3 = np.tanh(np.log(self.cw[0] / (self.cw[2] + 1e-9)) * 50) if self.cw.count >= 3 else 0.0
        r10 = np.tanh(np.log(self.cw[0] / (self.cw[9] + 1e-9)) * 20) if self.cw.count >= 10 else 0.0
        roh = np.zeros(4); roh[int(self.regime)] = 1.0
        bn = float(np.clip(self.bw[0] / 3.0, 0, 1)) if self.bw.count > 0 else 0.5
        tlf = 0.5
        if self.tlw.count > 0:
            tla = [self.tlw[i] for i in range(int(self.tlw.count))]
            tlf = float(sum(tla) / len(tla))
        hn = float(np.tanh(self.ht / 4.0))
        f = np.array([rsi, mn, msn, mhn, momnorm, rocnorm,
            atr_pct * 10, bbw, np.clip(bbp, 0, 1), bbd, stn,
            d12, d26, d50, d200, ex, ex2, adxn,
            volr, obs, 0.0, lr, r3, r10, *roh, bn, tlf, hn], dtype=np.float32)
        return np.clip(f, -3.0, 3.0)

    def agent_d3qn(self, f):
        s = f[15]*0.25 + f[11]*0.15 + f[14]*0.20 + f[1]*0.15 + f[4]*0.10
        if f[0] < 0.30: s += 0.10
        elif f[0] > 0.70: s -= 0.10
        s *= max(0.3, 1.0 - f[6]*2)
        s = np.tanh(s * self.mu)
        return float(s), float(np.clip(abs(s) * self.rc, 0, 1))

    def agent_ddqn(self, f):
        aln = sum([np.sign(f[1]), np.sign(f[3]), np.sign(f[4]), np.sign(f[5])])
        s = aln*0.12 + f[12]*0.12 + f[15]*0.10
        s *= 1.0 + np.clip(f[19], -0.5, 0.5)
        s += f[20]*0.10 + f[21]*0.10 + f[22]*0.08 + f[23]*0.08
        if f[0] < 0.25: s += 0.08
        elif f[0] > 0.75: s -= 0.08
        s = np.tanh(s * self.mu)
        return float(s), float(np.clip(abs(s) * self.rc, 0, 1))

    def agent_td3qn(self, f):
        s = -(f[8]-0.5)*0.40
        if f[7] < 0.02: s += f[15]*0.20
        elif f[7] > 0.08: s -= f[9]*0.15
        s += (0.5-f[0])*0.25
        if f[6] > 0.03: s *= 0.60
        s += f[14]*0.08
        if self.ht > 1.5: s -= 0.15
        elif self.ht < -1.5: s += 0.10
        sc = 1.0 - np.clip(f[10], 0, 0.8)
        s = np.tanh(s * self.mu)
        return float(s), float(np.clip(abs(s) * self.rc * sc, 0, 1))

    def ensemble(self, f):
        beta = self.bw[0] if self.bw.count > 0 else 1.0
        s1, c1 = self.agent_d3qn(f)
        s2, c2 = self.agent_ddqn(f)
        s3, c3 = self.agent_td3qn(f)
        self.agent_signals = (s1, s2, s3)
        if beta < 1.0:
            g = min(1.0 / np.sqrt(1 - beta*beta + 1e-9), 2.0)
            s1 *= g; s2 *= g; s3 *= g
        else:
            d = 1.0 / (beta + 1e-9)
            s1 *= d; s2 *= d; s3 *= d
        if self.geo_slope > 0 and self.geo_dev < 0: s1 += 0.10
        s3 += self.geo_dev * -0.3
        if np.sign(self.geo_slope) != np.sign(s2): s2 *= 0.5
        s1 += self.rapidity * 0.10
        sigs = np.array([s1, s2, s3])
        cons = np.array([c1, c2, c3])
        w = self.rweights[self.regime].copy() * (cons + 0.1)
        w /= w.sum()
        return float(np.dot(w, sigs)), float(np.dot(w, cons))

    def size(self, f, action, conf, rm):
        if rm == 0.0: return 0.0
        sign = float(np.sign(action)) if action != 0 else 0.0
        mag = abs(action)
        if self.regime == MarketRegime.BULL:
            rw = mag * conf * rm * 1.5
            if self.tlw.count > 0:
                tla = [self.tlw[i] for i in range(int(self.tlw.count))]
                rw *= max(0.3, min(1.5, sum(tla)/len(tla)/0.7))
            return sign * rw
        if self.regime == MarketRegime.SIDEWAYS:
            s3, c3 = self.agent_td3qn(f)
            return sign * max(abs(s3)*c3*rm*1.25, mag*conf*rm*1.0)
        if self.regime == MarketRegime.HIGH_VOLATILITY:
            rw = mag * conf * rm * 1.25
            return sign * rw
        if self.regime == MarketRegime.BEAR:
            rw = mag * conf * rm * 1.5
            if self.tlw.count > 0:
                tla = [self.tlw[i] for i in range(int(self.tlw.count))]
                rw *= max(0.3, min(1.5, sum(tla)/len(tla)/0.7))
            return sign * rw
        return 0.0

    def update_bh(self):
        if self.cw.count < 2: return
        br = (self.cw[0]-self.cw[1])/(self.cw[1]+1e-9)
        if self.bit == "TIMELIKE":
            self.ctl += 1; self.cum_disp += br
            sb = min(2.0, 1.0 + self.ctl*0.1)
            self.bh_mass = self.bh_mass*self.bh_decay + abs(br)*100*sb
        else:
            self.ctl = 0; self.bh_mass *= 0.7
        if self.cum_disp > 0: self.bh_dir = 1
        elif self.cum_disp < 0: self.bh_dir = -1
        prev = self.bh_active
        if not prev:
            if self.reform_bars > 0 and self.reform_bars < 15:
                self.bh_mass += self.prev_bh_mass * 0.5
                self.prev_bh_mass = 0.0
            self.bh_active = self.bh_mass > self.bh_form and self.ctl >= 5
        else:
            self.bh_active = self.bh_mass > self.bh_collapse and self.ctl >= 5
        if self.bh_active and not prev:
            self.well_entry_pv = self.algo.portfolio.total_portfolio_value
            self.well_gain = 0.0; self.reform_bars = 0
        elif not self.bh_active and prev:
            self.prev_bh_mass = self.bh_mass; self.reform_bars = 1
        if not self.bh_active:
            if self.reform_bars > 0: self.reform_bars += 1
            if self.ctl == 0: self.cum_disp = 0.0
        if self.bh_active and self.well_entry_pv > 0:
            pv = self.algo.portfolio.total_portfolio_value
            self.well_gain = (pv - self.well_entry_pv)/(self.well_entry_pv + 1e-9)

    def update_pattern(self, cur_ret):
        n = self.sig_win
        if len(self.beta_hist) < n+1: return 0.0
        past_sig = self._sig_encode(self.beta_hist[-(n+1):-1])
        if past_sig not in self.pat_ret: self.pat_ret[past_sig] = []
        self.pat_ret[past_sig].append(cur_ret)
        if len(self.pat_ret[past_sig]) > self.pat_lb:
            self.pat_ret[past_sig] = self.pat_ret[past_sig][-self.pat_lb:]
        if len(self.beta_hist) < n: return 0.0
        cur_sig = self._sig_encode(self.beta_hist[-n:])
        if cur_sig in self.pat_ret:
            hist = self.pat_ret[cur_sig]
            if len(hist) >= 5:
                wr = sum(1 for r in hist if r > 0)/len(hist)
                if wr > 0.6 or wr < 0.4: return (sum(hist)/len(hist))*100
        return 0.0

    def _sig_encode(self, hist):
        s = ""
        for _, r, tl in hist:
            if tl and r >= 0: s += "A"
            elif tl and r < 0: s += "B"
            elif not tl and r >= 0: s += "C"
            else: s += "D"
        return s

class LarsaMultiFuturesAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.instruments = {
            "ES": FutureInstrument(self, Futures.Indices.SP_500_E_MINI, Resolution.HOUR, 0.001, "ES"),
            "NQ": FutureInstrument(self, Futures.Indices.NASDAQ_100_E_MINI, Resolution.HOUR, 0.0012, "NQ"),
            "YM": FutureInstrument(self, Futures.Indices.DOW_30_E_MINI, Resolution.HOUR, 0.0008, "YM"),
        }
        self.peak = self.portfolio.total_portfolio_value
        self.daystart = self.portfolio.total_portfolio_value
        self.cb = False
        self.initial_equity = 1_000_000.0
        self.hwm_cooldown = 0
        self.trail_flat_bars = 0
        self.ramp_back = 0
        self.convergence = 1.0
        self.grace_period = 0
        self.maxdd = 0.12
        self.dlim = 0.02
        rc = Chart("Regime")
        for k in self.instruments: rc.add_series(Series(k, SeriesType.LINE, 0))
        self.add_chart(rc)
        dc = Chart("Risk")
        dc.add_series(Series("Drawdown%", SeriesType.LINE, 0))
        self.add_chart(dc)
        self.schedule.on(self.date_rules.every_day(), self.time_rules.at(0, 0), self._daily_reset)
        self.set_warm_up(timedelta(days=30))

    def _daily_reset(self):
        self.daystart = self.portfolio.total_portfolio_value
        self.cb = False

    def on_securities_changed(self, changes):
        for s in changes.added_securities:
            if s.symbol.security_type == SecurityType.FUTURE:
                sym_str = str(s.symbol.id)
                for key, inst in self.instruments.items():
                    canon_str = str(inst.future.symbol.id)
                    if sym_str.startswith(canon_str[:2]):
                        inst.mapped_sym = s.symbol; break

    def _portfolio_risk(self):
        pv = self.portfolio.total_portfolio_value
        if pv > self.peak: self.peak = pv
        dd = (self.peak - pv)/(self.peak + 1e-9)
        if dd >= self.maxdd:
            if not self.cb: self.cb = True
            return 0.0
        if (pv - self.daystart)/(self.daystart + 1e-9) <= -self.dlim: return 0.0
        if dd >= self.maxdd * 0.70: return 0.50
        return 1.0

    def _inverse_vol_weights(self):
        inv_vols = {}
        for key, inst in self.instruments.items():
            if inst.atr.is_ready and inst.cw.count > 0:
                inv_vols[key] = 1.0/(inst.atr.current.value/(inst.cw[0]+1e-9) + 1e-9)
            else: inv_vols[key] = 1.0
        total = sum(inv_vols.values())
        return {k: v/total for k, v in inv_vols.items()}

    def on_data(self, data):
        if self.is_warming_up: return
        pv_now = self.portfolio.total_portfolio_value
        if pv_now > self.peak: self.peak = pv_now
        if self.hwm_cooldown > 0:
            self.hwm_cooldown -= 1
            if self.hwm_cooldown == 0: self.peak = self.portfolio.total_portfolio_value
            return
        elif (self.peak - pv_now)/(self.peak + 1e-9) >= 0.12:
            self.hwm_cooldown = 3; self.ramp_back = 5; self.grace_period = 50
            self.liquidate()
            for inst in self.instruments.values(): inst.last_target = 0.0; inst.pos_floor = 0.0
            return
        pv = self.portfolio.total_portfolio_value
        peak_gain = (self.peak - self.initial_equity)/self.initial_equity
        if peak_gain >= 1.50: trail_pct = 0.10
        elif peak_gain >= 1.00: trail_pct = 0.12
        elif peak_gain >= 0.50: trail_pct = 0.15
        elif peak_gain >= 0.20: trail_pct = 0.18
        else: trail_pct = 0.20
        profit_floor = self.initial_equity * (1.0 + peak_gain * 0.50)
        if pv < profit_floor and peak_gain > 1.00 and self.grace_period == 0:
            self.trail_flat_bars = 5; self.ramp_back = 5; self.grace_period = 50
            self.liquidate()
            for inst in self.instruments.values(): inst.last_target = 0.0
            return
        if self.trail_flat_bars > 0:
            self.trail_flat_bars -= 1
            if self.trail_flat_bars == 0: self.peak = self.portfolio.total_portfolio_value
            return
        elif self.grace_period == 0 and (self.peak - pv)/(self.peak + 1e-9) >= trail_pct:
            self.trail_flat_bars = 5; self.ramp_back = 5; self.grace_period = 50
            self.liquidate()
            for inst in self.instruments.values(): inst.last_target = 0.0
            return
        if self.grace_period > 0: self.grace_period -= 1
        if self.ramp_back > 0: self.ramp_back -= 1
        rm = self._portfolio_risk()
        vol_weights = self._inverse_vol_weights()
        tl_count = sum(1 for i in self.instruments.values() if i.bit == "TIMELIKE" and i.tl_confirm >= 3)
        bh_count = sum(1 for i in self.instruments.values() if i.bh_active)
        if bh_count >= 3: self.convergence = 2.5
        elif bh_count >= 2 and tl_count >= 3: self.convergence = 2.0
        elif bh_count >= 2: self.convergence = 1.7
        elif tl_count >= 3: self.convergence = 1.4
        elif tl_count >= 2: self.convergence = 1.2
        else: self.convergence = 1.0

        for key, inst in self.instruments.items():
            t = self._process_instrument(data, inst, rm, vol_weights.get(key, 0.33))
            if t is None: continue
            bh_active_count = sum(1 for i in self.instruments.values() if i.bh_active)
            # v6: convergence-first — solo BH is near break-even (50% WR per forensics).
            # Convergence (2+ BH active simultaneously) = 74.5% WR, 24.5x more profitable/well.
            # Kelly criterion says convergence should be 31x larger than solo.
            if bh_active_count >= 2:
                cap = 0.65   # full size: convergence edge confirmed
            elif inst.regime == MarketRegime.BULL:
                cap = 0.25   # solo in BULL: some trend following value
            else:
                cap = 0.15   # solo in SIDEWAYS/BEAR: minimal, barely break-even
            tgt = float(np.clip(t, -cap, cap))
            # v5: regime-aware NQ cap — full size in BULL (2020 rally), capped in SIDEWAYS/BEAR
            # v4 flat cap hurt 2020 COVID recovery (-$289k). In BULL we WANT the full NQ exposure.
            if key == "NQ" and inst.mapped_sym is not None and inst.regime != MarketRegime.BULL:
                nq_cap_fraction = 400000.0 / (self.portfolio.total_portfolio_value + 1e-9)
                if abs(tgt) > nq_cap_fraction:
                    tgt = float(np.sign(tgt) * nq_cap_fraction)
            if abs(tgt - inst.last_target) > 0.02:
                mapped = inst.future.mapped
                if mapped not in self.securities or not self.securities[mapped].exchange.exchange_open: continue
                inst.last_target = tgt
                self.set_holdings(mapped, tgt)

        if any(i.bc % 4 == 0 for i in self.instruments.values()):
            pv = self.portfolio.total_portfolio_value
            self.plot("Risk", "Drawdown%", (self.peak - pv)/(self.peak + 1e-9)*100)
            for key, inst in self.instruments.items():
                self.plot("Regime", key, int(inst.regime))

    def _process_instrument(self, data, inst, rm, weight):
        mapped = inst.future.mapped
        if mapped is None: return
        trade_bar = data.bars.get(mapped)
        quote_bar = data.quote_bars.get(mapped)
        if trade_bar is not None: bar = trade_bar; vol = float(trade_bar.volume)
        elif quote_bar is not None: bar = quote_bar; vol = 0.0
        else: return
        inst.bc += 1; inst.cw.add(bar.close); inst.vw.add(vol)
        if inst.cw.count >= 2:
            beta = abs(inst.cw[0]-inst.cw[1])/(inst.cw[1]+1e-9)/inst.cf
            inst.bw.add(beta)
            tl = 1.0 if beta < 1.0 else 0.0
            inst.tlw.add(tl)
            inst.bit = "TIMELIKE" if beta < 1.0 else "SPACELIKE"
            if inst.bit == "TIMELIKE": inst.tl_confirm = min(inst.tl_confirm+1, 3)
            else: inst.tl_confirm = 0
            hv = abs(inst.cw[0]-inst.cw[1])/(inst.cw[1]+1e-9)
            v = min(0.99, hv/inst.max_vol)
            gamma = 1.0/np.sqrt(1-v*v+1e-9)
            inst.proper_time += 1.0/gamma
        if inst.atr.is_ready: inst.aw.add(inst.atr.current.value)
        if inst.rsi.is_ready: inst.rw.add(inst.rsi.current.value)
        if inst.cw.count >= 20:
            prices = [inst.cw[i] for i in range(20)]; prices.reverse()
            lp = np.log(np.array(prices)+1e-9)
            x = np.arange(20, dtype=float); n = 20.0
            sx = x.sum(); slp = lp.sum(); sxx = np.dot(x,x); sxlp = np.dot(x,lp)
            slope = (n*sxlp-sx*slp)/(n*sxx-sx*sx+1e-9)
            intercept = (slp-slope*sx)/n
            geo_p = np.exp(slope*19+intercept)
            inst.geo_dev = float(np.tanh((inst.cw[0]-geo_p)/(inst.atr.current.value+1e-9)))
            inst.geo_slope = float(slope*100)
            cp = inst.cw[0]; cc = 0
            for k in range(1, 20):
                pp = inst.cw[k]; dp = abs(cp-pp)/(pp+1e-9)
                ds2 = -(inst.cf*k)**2 + dp*dp
                if ds2 < 0: cc += 1
            inst.causal_frac = cc/19.0
            E = (inst.cw[0]-inst.cw[19])/(inst.cw[19]+1e-9); px = E
            denom = E-px+1e-9; num = E+px
            if abs(denom) > 1e-9 and num/denom > 0:
                inst.rapidity = float(np.tanh(0.5*np.log(num/denom+1e-9)))
            else: inst.rapidity = 0.0
        if not inst.ind_ready(): return
        inst.detect_regime()
        if inst.bb.is_ready and inst.std.is_ready:
            bs = inst.std.current.value
            if bs > 0:
                z = (bar.close-inst.bb.middle_band.current.value)/bs
                inst.ht = z*(z-inst.pz); inst.pz = z
        M = float(inst.ctl + (1 if inst.bit == "TIMELIKE" else 0))
        R_E = np.sqrt(M+1e-9)
        if M < 2.0: inst.mu = max(0.3, M/3.0)
        elif inst.atr.is_ready and inst.cw.count >= 2 and inst.tlw.count >= 2:
            ni = int(min(inst.cw.count, inst.tlw.count, inst.vw.count))
            lp2, lv = [], []
            for i in range(ni):
                if inst.tlw[i] == 1.0: lp2.append(inst.cw[i]); lv.append(inst.vw[i])
            if lp2:
                sv = sum(lv)
                vwap = sum(p*v for p,v in zip(lp2,lv))/sv if sv > 0 else sum(lp2)/len(lp2)
                r = abs(inst.cw[0]-vwap)/(inst.atr.current.value+1e-9)
                inst.mu = float(1.0+R_E/(r+R_E))
            else: inst.mu = 1.0
        else: inst.mu = 1.0
        f = inst.compute_features()
        if f is None: return
        action, conf = inst.ensemble(f)
        tgt = inst.size(f, action, conf, rm)
        if (inst.e12.current.value > inst.e26.current.value > inst.e50.current.value > inst.e200.current.value
                and inst.adx.current.value > 25 and rm > 0.0):
            adx = inst.adx.current.value
            tgt = max(tgt, min(2.5, 1.0+(adx-25)*0.04))
        if (inst.e200.current.value > inst.e50.current.value > inst.e26.current.value > inst.e12.current.value
                and inst.adx.current.value > 25 and rm > 0.0):
            adx = inst.adx.current.value
            bear_floor = -min(2.5, 1.0+(adx-25)*0.04)
            tgt = min(tgt, bear_floor)
        inst.update_bh()
        if inst.bh_active and inst.bh_dir > 0 and inst.ht < 1.8 and rm > 0.0:
            tgt = max(tgt, min(2.5, 1.0+inst.bh_mass*0.30))
        if inst.bh_active and inst.bh_dir < 0 and inst.ht < 1.8 and rm > 0.0:
            tgt = min(tgt, -min(2.5, 1.0+inst.bh_mass*0.30))
        if inst.cw.count >= 2:
            br = (inst.cw[0]-inst.cw[1])/(inst.cw[1]+1e-9)
            bv = abs(br)/(inst.cf+1e-9)
            inst.beta_hist.append((bv, br, inst.bit == "TIMELIKE"))
            if len(inst.beta_hist) > 1000: inst.beta_hist = inst.beta_hist[-1000:]
            pat_sig = inst.update_pattern(br)
            if pat_sig > 0.5:
                if (inst.regime in (MarketRegime.BULL, MarketRegime.BEAR)) and inst.bh_active: tgt *= 1.5
                else: tgt = min(tgt*1.3, 2.0)
            elif pat_sig < -0.5: tgt *= 0.5
        if not ((inst.regime == MarketRegime.BULL or inst.regime == MarketRegime.BEAR) and inst.bh_active): tgt *= inst.causal_frac
        pt_gate = inst.pt_threshold/2.0 if inst.regime in (MarketRegime.BULL, MarketRegime.BEAR) else inst.pt_threshold
        if inst.proper_time < pt_gate: return
        inst.proper_time = 0.0
        if inst.mu > 1.8: max_lev = 2.5
        elif inst.mu > 1.5 and inst.ctl >= 5: max_lev = 2.0
        elif inst.mu > 1.2: max_lev = 1.5
        else: max_lev = 1.0
        max_lev *= self.convergence
        tgt = float(np.clip(tgt, -max_lev, max_lev))
        killed = False
        geo_raw = float(np.arctanh(np.clip(abs(inst.geo_dev), 0.0, 0.9999)))
        if geo_raw > 2.0: tgt = 0.0; killed = True
        if inst.bc < 120: tgt = 0.0; killed = True
        tl_req = 1 if inst.regime == MarketRegime.HIGH_VOLATILITY else 3
        if inst.tl_confirm < tl_req: tgt = 0.0; killed = True
        elif inst.bit == "SPACELIKE":
            tgt *= 0.50 if inst.regime == MarketRegime.HIGH_VOLATILITY else 0.15
        # v5: BEAR regime long gate — instrument-specific thresholds
        # BEAR is 96.6% persistent (avg 29.5 bars). Longs in sustained BEAR = low edge.
        # YM 2018-2019: 39% WR in BEAR — needs tighter gate (rhb>3 vs rhb>5 for ES/NQ)
        # v4 rhb>5 was not catching YM's extended 2018-2019 bear phases.
        bear_rhb_thresh = 3 if inst.label == "YM" else 5
        if inst.regime == MarketRegime.BEAR and tgt > 0 and inst.rhb > bear_rhb_thresh:
            tgt = 0.0; killed = True  # v5: instrument-aware BEAR gate
        # v6: SIDEWAYS solo gate — solo trades in SIDEWAYS need very strong confirmation
        bh_conv = sum(1 for i in self.instruments.values() if i.bh_active) >= 2
        if inst.regime == MarketRegime.SIDEWAYS and not bh_conv and abs(tgt) > 0:
            if inst.ctl < 8:
                tgt = 0.0; killed = True
        if 0.0 < abs(tgt) < 0.03: tgt = 0.0
        wb_thresh = 6 if inst.regime == MarketRegime.HIGH_VOLATILITY else 3
        if abs(tgt) < 0.30:
            inst.weak_bars += 1
            if inst.weak_bars >= wb_thresh: tgt = 0.0; killed = True
        else: inst.weak_bars = 0
        # v6: pos_floor only during convergence — prevents solo BH locking in losing positions
        bh_conv = sum(1 for i in self.instruments.values() if i.bh_active) >= 2
        if not killed and abs(tgt) > 0.5 and inst.ctl >= 5 and bh_conv:
            inst.pos_floor = max(inst.pos_floor, 0.70*abs(tgt))
        if not killed and inst.pos_floor > 0.0 and inst.last_target != 0.0 and bh_conv:
            tgt = float(np.sign(inst.last_target)*max(abs(tgt), inst.pos_floor))
            inst.pos_floor *= 0.95
        if geo_raw > 1.5 or killed or not bh_conv: inst.pos_floor = 0.0
        if self.ramp_back > 0: tgt = float(np.clip(tgt, -0.5, 0.5))
        if abs(tgt) < 0.02:
            if abs(inst.last_target) > 0.02: return 0.0
            return None
        return tgt

    def on_end_of_algorithm(self):
        pv = self.portfolio.total_portfolio_value
        dd = (self.peak-pv)/(self.peak+1e-9)
        for key, inst in self.instruments.items():
            self.log(f"[{key}] END regime={inst.regime.name} bc={inst.bc}")
        self.log(f"PORTFOLIO peak={self.peak:.0f} dd={dd:.1%}")
