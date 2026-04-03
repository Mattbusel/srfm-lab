"""
arena_v2.py — LARSA v2 experiment arena.

V2 modifications (from trade log forensics):
  A. ATR_SCALE:    position * min(1, 1.5/atr_ratio) when atr_ratio > 1.5
  B. STOP_LOSS:    cut position if per-instrument PnL < -3% of portfolio
  C. BEAR_FAST:    tl_req=2 in BEAR regime (was 3, reduces 2022-style flat periods)
  D. BH_BOOST:     scale position by min(1.5, bh_mass/1.5) when bh active (conviction sizing)

Usage:
    python tools/arena_v2.py --csv data/NDX_hourly_poly.csv --cf 0.005 --exp ABCD
    python tools/arena_v2.py --synthetic --n-bars 20000 --exp A
    python tools/arena_v2.py --csv data/NDX_hourly_poly.csv --cf 0.005  # baseline (no exp)

Experiment flags: any combination of letters A B C D (e.g. --exp AB, --exp ABCD)
"""

import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from srfm_core import (
    MinkowskiClassifier, BlackHoleDetector,
    GeodesicAnalyzer, GravitationalLens, HawkingMonitor,
    MarketRegime,
)
from agents import ensemble, size_position
from regime import RegimeDetector
from risk import KillConditions, PortfolioRiskManager
from broker import SimulatedBroker


DEFAULT_CFG = {"cf": 0.001, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
CONFIGS = {
    "ES": {"cf": 0.001,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "NQ": {"cf": 0.0012, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "YM": {"cf": 0.0008, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
}


def load_ohlcv(path: str) -> List[dict]:
    bars = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            def g(*keys):
                for k in keys:
                    v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
                    if v not in (None, "", "null", "None"):
                        try: return float(v)
                        except: pass
                return None
            c = g("close", "Close")
            if c and c > 0:
                bars.append({
                    "date":   row.get("date") or row.get("Date") or "",
                    "open":   g("open","Open") or c,
                    "high":   g("high","High") or c,
                    "low":    g("low","Low") or c,
                    "close":  c,
                    "volume": g("volume","Volume") or 1000.0,
                })
    return bars


def generate_synthetic(n_bars: int = 20000, seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    REGIMES = {
        "bull":     (0.0002, 0.0008),
        "bear":     (-0.0003, 0.0014),
        "sideways": (0.00005, 0.0006),
        "crisis":   (-0.001, 0.003),
    }
    price = 4500.0
    bars = []
    regime_seq = list(rng.choice(["bull","bear","sideways","crisis"], size=n_bars,
                                  p=[0.55, 0.15, 0.25, 0.05]))
    for i in range(n_bars):
        mu, sig = REGIMES[regime_seq[i]]
        ret   = max(-0.05, min(0.05, mu + sig * float(rng.standard_normal())))
        close = price * (1 + ret)
        bars.append({
            "date": f"bar_{i:06d}", "open": price,
            "high": close * (1 + abs(float(rng.standard_normal())) * sig * 0.3),
            "low":  close * (1 - abs(float(rng.standard_normal())) * sig * 0.3),
            "close": close, "volume": 50000.0,
        })
        price = close
    return bars


# --- Inline indicators -------------------------------------------------------
class _EMA:
    def __init__(self, p):
        self.k = 2.0/(p+1); self.v = None
    def update(self, x):
        self.v = x if self.v is None else x*self.k + self.v*(1-self.k)
        return self.v

class _ATR:
    def __init__(self, p=14):
        self.p=p; self._prev=None; self._buf=[]; self.v=None
    def update(self, h, lo, c):
        tr = (h-lo) if self._prev is None else max(h-lo, abs(h-self._prev), abs(lo-self._prev))
        self._prev = c
        if self.v is None:
            self._buf.append(tr)
            if len(self._buf) >= self.p: self.v = sum(self._buf)/len(self._buf)
        else: self.v = (self.v*(self.p-1)+tr)/self.p
        return self.v or tr

class _ADX:
    def __init__(self, p=14):
        self.p=p; self._atr=_ATR(p); self._ph=self._pl=None
        self._pdm=self._ndm=0.0; self.v=0.0
    def update(self, h, lo, c):
        atr = self._atr.update(h, lo, c)
        if self._ph is None: self._ph, self._pl = h, lo; return 0.0
        pdm=max(0.0,h-self._ph); ndm=max(0.0,self._pl-lo)
        if pdm>ndm: ndm=0.0
        elif ndm>pdm: pdm=0.0
        else: pdm=ndm=0.0
        k=2.0/(self.p+1)
        self._pdm=pdm*k+self._pdm*(1-k); self._ndm=ndm*k+self._ndm*(1-k)
        if atr>0:
            pdi,ndi=100*self._pdm/atr,100*self._ndm/atr; d=pdi+ndi
            dx=100*abs(pdi-ndi)/d if d>0 else 0.0
        else: dx=0.0
        self.v=self.v*(self.p-1)/self.p + dx/self.p
        self._ph, self._pl = h, lo
        return self.v

class _RSI:
    def __init__(self, p=14):
        self.p=p; self._prev=None; self._gains=[]; self._losses=[]
        self._avg_gain=self._avg_loss=None; self.v=50.0
    def update(self, c):
        if self._prev is None: self._prev=c; return 50.0
        chg=c-self._prev; self._prev=c
        if self._avg_gain is None:
            self._gains.append(max(0,chg)); self._losses.append(max(0,-chg))
            if len(self._gains)>=self.p:
                self._avg_gain=sum(self._gains)/self.p
                self._avg_loss=sum(self._losses)/self.p
                self.v=100.0 if self._avg_loss==0 else 100-100/(1+self._avg_gain/self._avg_loss)
        else:
            self._avg_gain=(self._avg_gain*(self.p-1)+max(0,chg))/self.p
            self._avg_loss=(self._avg_loss*(self.p-1)+max(0,-chg))/self.p
            self.v=100.0 if self._avg_loss==0 else 100-100/(1+self._avg_gain/self._avg_loss)
        return self.v

class _BB:
    def __init__(self, p=20):
        self.p=p; self._buf=[]
    def update(self, c):
        self._buf.append(c)
        if len(self._buf)>self.p: self._buf.pop(0)
        mid=sum(self._buf)/len(self._buf)
        std=(sum((x-mid)**2 for x in self._buf)/len(self._buf))**0.5
        return mid, std

class _ATRAvg:
    """50-bar running average of ATR for atr_ratio computation."""
    def __init__(self, p=50):
        self._buf=[]; self.p=p; self.avg=None
    def update(self, atr_val):
        self._buf.append(atr_val)
        if len(self._buf)>self.p: self._buf.pop(0)
        if len(self._buf)>=self.p: self.avg=sum(self._buf)/len(self._buf)
        return self.avg


# --- V2 strategy runner -------------------------------------------------------

def run_v2(
    bars: List[dict],
    cfg: dict,
    max_leverage: float = 0.65,
    exp_flags: str = "",       # subset of "ABCD"
    verbose: bool = False,
) -> Tuple["SimulatedBroker", list]:
    """
    Run LARSA v2 strategy.

    exp_flags controls which v2 improvements are active:
      A = ATR_SCALE   (position * min(1, 1.5/atr_ratio) when vol spikes)
      B = STOP_LOSS   (per-instrument -3% portfolio stop)
      C = BEAR_FAST   (tl_req=2 in BEAR, was 3)
      D = BH_BOOST    (scale position by min(1.5, bh_mass/1.5) when BH active)
    """
    USE_A = "A" in exp_flags.upper()
    USE_B = "B" in exp_flags.upper()
    USE_C = "C" in exp_flags.upper()
    USE_D = "D" in exp_flags.upper()

    # Physics
    mc  = MinkowskiClassifier(cf=cfg["cf"])
    bh  = BlackHoleDetector(cfg["bh_form"], cfg["bh_collapse"], cfg["bh_decay"])
    geo = GeodesicAnalyzer()
    gl  = GravitationalLens()
    hw  = HawkingMonitor()
    reg = RegimeDetector(atr_window=50)

    # Indicators
    e12=_EMA(12); e26=_EMA(26); e50=_EMA(50); e200=_EMA(200)
    atr_=_ATR(14); adx_=_ADX(14); rsi_=_RSI(14); bb_=_BB(20)
    atr_avg = _ATRAvg(50)   # for atr_ratio (EXP A)

    broker = SimulatedBroker(
        cash=1_000_000, fee_per_trade=50, slippage_pct=0.0002,
        max_leverage=max_leverage,
    )
    kc = KillConditions()

    # Per-instrument stop-loss tracking (EXP B)
    entry_equity_for_stop = None

    bar_log = []
    prev_close: Optional[float] = None
    bar_count = 0

    for bar in bars:
        close  = bar["close"]
        high   = bar["high"]
        low    = bar["low"]
        volume = bar["volume"]
        date   = bar["date"]

        if prev_close is None or prev_close <= 0:
            prev_close = close
            mc.update(close); e12.update(close); e26.update(close)
            e50.update(close); e200.update(close)
            atr_v0 = atr_.update(high, low, close)
            atr_avg.update(atr_v0)
            adx_.update(high, low, close); rsi_.update(close); bb_.update(close)
            broker.update(0.0)
            continue

        bar_count += 1
        bar_return = (close - prev_close) / prev_close

        # Physics
        bit    = mc.update(close)
        active = bh.update(bit, close, prev_close)

        # Indicators
        ema12=e12.update(close); ema26=e26.update(close)
        ema50=e50.update(close); ema200=e200.update(close)
        atr_v = atr_.update(high, low, close)
        atr_avg.update(atr_v)
        adx_v = adx_.update(high, low, close)
        rsi_v = rsi_.update(close)
        bb_mid, bb_std = bb_.update(close)

        # ATR ratio for EXP A
        atr_ratio = 1.0
        if atr_avg.avg and atr_avg.avg > 0:
            atr_ratio = atr_v / atr_avg.avg

        # Geodesic + lensing + Hawking
        geo_dev, geo_slope, causal_frac, rapidity = geo.update(close, atr_v)
        gl.update(close, volume, bit, mc.tl_confirm, atr_v)
        mu = gl.mu
        ht = hw.update(close, bb_mid, bb_std)

        # Regime
        regime, conf = reg.update(close, ema12, ema26, ema50, ema200, adx_v, atr_v)

        # Feature vector
        f = np.zeros(31, dtype=np.float32)
        macd_val = (ema12 - ema26) / (close + 1e-9)
        f[0]  = np.clip((rsi_v - 50) / 50, -3, 3)
        f[1]  = np.clip(macd_val * 1000, -3, 3)
        f[4]  = np.clip((close - ema50) / (atr_v + 1e-9), -3, 3)
        f[11] = np.clip(adx_v / 50, -3, 3)
        f[14] = np.clip(bh.bh_mass / 3.0, -3, 3)
        f[15] = float(bh.bh_dir)
        f[16] = np.clip(geo_dev, -3, 3)
        f[17] = np.clip(geo_slope, -3, 3)
        f[22] = np.clip(mu, -3, 3)
        f[23] = np.clip(atr_v / (close + 1e-9) * 1000, -3, 3)
        bb_pct = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)
        f[24] = np.clip(bb_pct - 0.5, -3, 3)
        f[25] = np.clip(bb_std / (close + 1e-9) * 1000, -3, 3)
        f[30] = np.clip(ht, -3, 3)
        regime_idx = int(regime)
        f[27 + min(regime_idx, 3)] = 1.0

        # Ensemble
        action, conf_sig, _ = ensemble(
            f, mu, causal_frac, ht, mc.beta, regime, geo_slope, geo_dev, rapidity,
        )
        rm = min(1.0, broker.equity / broker.initial_equity)
        tl_window = [1.0 if mc.tl_confirm > 0 else 0.0]
        target_pos = size_position(f, action, conf_sig, rm, regime, mu, causal_frac, ht, tl_window)
        target_pos = float(np.clip(target_pos, -max_leverage, max_leverage))

        # EXP D: BH mass conviction boost
        if USE_D and active and bh.bh_mass > 1.5:
            bh_boost = min(1.5, bh.bh_mass / 1.5)
            target_pos = float(np.clip(target_pos * bh_boost, -max_leverage, max_leverage))

        # EXP A: ATR scaling (reduce position when volatility spikes)
        if USE_A and atr_ratio > 1.5:
            atr_scale = max(0.3, 1.5 / atr_ratio)
            target_pos *= atr_scale

        # Kill conditions (v2: EXP C — faster BEAR entry)
        from srfm_core import MarketRegime as MR
        geo_raw = float(np.arctanh(np.clip(abs(geo_dev), 0.0, 0.9999)))

        killed = False
        if geo_raw > 2.0:
            target_pos = 0.0; killed = True
        if bar_count < 120:
            target_pos = 0.0; killed = True

        # EXP C: tl_req=2 in BEAR (was 3 in v1)
        if USE_C:
            if regime == MR.HIGH_VOLATILITY:
                tl_req = 1
            elif regime == MR.BEAR:
                tl_req = 2   # v2 change: was 3
            else:
                tl_req = 3
        else:
            tl_req = 1 if regime == MR.HIGH_VOLATILITY else 3

        if mc.tl_confirm < tl_req:
            target_pos = 0.0; killed = True
        elif bit == "SPACELIKE":
            target_pos *= 0.50 if regime == MR.HIGH_VOLATILITY else 0.15

        if 0.0 < abs(target_pos) < 0.03:
            target_pos = 0.0

        if abs(target_pos) < 0.30:
            kc.weak_bars += 1
            wb_thresh = 6 if regime == MR.HIGH_VOLATILITY else 3
            if kc.weak_bars >= wb_thresh:
                target_pos = 0.0; killed = True
        else:
            kc.weak_bars = 0

        if not killed and abs(target_pos) > 0.5 and bh.ctl >= 3:
            kc.pos_floor = max(kc.pos_floor, 0.90 * abs(target_pos))
        if not killed and kc.pos_floor > 0.0 and broker.position != 0.0:
            target_pos = float(np.sign(broker.position) * max(abs(target_pos), kc.pos_floor))
        if geo_raw > 1.5 or killed:
            kc.pos_floor = 0.0

        # EXP B: per-trade stop loss (-3% portfolio)
        if USE_B:
            if abs(broker.position) > 0.02:
                # Track entry equity
                if entry_equity_for_stop is None:
                    entry_equity_for_stop = broker.equity
                else:
                    trade_loss = (broker.equity - entry_equity_for_stop) / (entry_equity_for_stop + 1e-9)
                    if trade_loss < -0.03:
                        target_pos = 0.0
                        killed = True
                        entry_equity_for_stop = None
                        kc.pos_floor = 0.0
            else:
                # Position is flat (or near flat) — reset entry tracking
                if abs(target_pos) > 0.02:
                    entry_equity_for_stop = broker.equity
                else:
                    entry_equity_for_stop = None

        broker.update(bar_return)
        broker.set_position(target_pos)

        if verbose:
            bar_log.append({
                "date": date, "close": close, "regime": regime.name,
                "bh_active": active, "bh_mass": round(bh.bh_mass, 4),
                "atr_ratio": round(atr_ratio, 3),
                "position": round(target_pos, 4), "equity": round(broker.equity, 0),
            })

        prev_close = close

    return broker, bar_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",          help="Price CSV")
    parser.add_argument("--ticker",       default="NDX")
    parser.add_argument("--cf",           type=float, default=0.005)
    parser.add_argument("--bh-form",      type=float, default=1.5)
    parser.add_argument("--bh-decay",     type=float, default=0.95)
    parser.add_argument("--max-leverage", type=float, default=0.65)
    parser.add_argument("--synthetic",    action="store_true")
    parser.add_argument("--n-bars",       type=int, default=20000)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--exp",          default="", help="Experiment flags: ABCD")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        bars = generate_synthetic(args.n_bars, args.seed)
    elif args.csv:
        bars = load_ohlcv(args.csv)
        print(f"  {len(bars)} bars from {args.csv}")
    else:
        parser.error("Provide --csv or --synthetic")

    cfg = {"cf": args.cf, "bh_form": args.bh_form, "bh_collapse": 1.0, "bh_decay": args.bh_decay}
    exp_name = args.exp.upper() or "BASELINE"
    print(f"Running arena_v2 [{exp_name}] cf={args.cf} max_lev={args.max_leverage}...")

    broker, log = run_v2(bars, cfg, args.max_leverage, args.exp, verbose=args.verbose)
    s = broker.stats()

    print(f"\n{'='*55}")
    print(f"  Arena v2 [{exp_name}] -- {args.ticker}")
    print(f"{'='*55}")
    print(f"  Return  : {s['total_return_pct']:+.2f}%")
    print(f"  MaxDD   : {s['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe  : {s['sharpe']:.3f}")
    print(f"  Trades  : {s['trade_count']}")
    print(f"  Win     : {s['win_rate']:.1%}")
    print(f"  Equity  : ${s['final_equity']:,.0f}")
    print(f"{'='*55}\n")

    if args.verbose and log:
        for r in log[-10:]:
            print(f"  {r['date']:<22}  pos={r['position']:+.3f}  eq=${r['equity']:>10,.0f}  "
                  f"atr_r={r['atr_ratio']:.2f}  bh={r['bh_active']}")

    os.makedirs("results", exist_ok=True)
    import json
    out = f"results/arena_v2_{exp_name}_{args.ticker}.json"
    with open(out, "w") as f:
        json.dump({"exp": exp_name, "ticker": args.ticker, **s}, f, indent=2)
    return s


if __name__ == "__main__":
    main()
