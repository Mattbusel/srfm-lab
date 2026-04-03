"""
tournament.py — SRFM parameter tournament. No LEAN, no Docker, no QC.

Sweeps CF, BH_FORM, BH_DECAY across a grid and runs the full SRFM strategy
on synthetic data for each combination. Outputs a ranked leaderboard.

At ~100-500 bars/second per run, 1000 parameter sets x 20k bars = minutes.

Usage:
    python tools/tournament.py --n-bars 20000 --n-sims 500
    python tools/tournament.py --n-bars 50000 --n-sims 1000 --workers 4
    python tools/tournament.py --csv data/ES_hourly_real.csv --n-sims 200
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))


# --- Parameter grid ----------------------------------------------------------

def build_param_grid(n_sims: int, seed: int = 0) -> List[dict]:
    """Sample n_sims parameter combinations from the search space."""
    rng = np.random.default_rng(seed)

    cf_range        = (0.0005, 0.0025)
    bh_form_range   = (0.8,    2.5)
    bh_collapse_range = (0.5,  1.5)
    bh_decay_range  = (0.88,   0.99)
    max_lev_range   = (0.5,    2.5)

    params = []
    for i in range(n_sims):
        params.append({
            "id":           i,
            "cf":           float(rng.uniform(*cf_range)),
            "bh_form":      float(rng.uniform(*bh_form_range)),
            "bh_collapse":  float(rng.uniform(*bh_collapse_range)),
            "bh_decay":     float(rng.uniform(*bh_decay_range)),
            "max_leverage": float(rng.uniform(*max_lev_range)),
        })
    return params


# --- Single backtest (importable for multiprocessing) -----------------------

def _run_one(args: Tuple) -> dict:
    """Run one parameter set. Designed for ProcessPoolExecutor."""
    params, bars, seed = args

    # Import inside worker (needed for multiprocessing on Windows)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
    from srfm_core import (
        MinkowskiClassifier, BlackHoleDetector,
        GeodesicAnalyzer, GravitationalLens, HawkingMonitor,
        MarketRegime,
    )
    from agents import ensemble, size_position
    from regime import RegimeDetector
    from risk import KillConditions
    from broker import SimulatedBroker
    import numpy as np

    cfg = {
        "cf":          params["cf"],
        "bh_form":     params["bh_form"],
        "bh_collapse": params["bh_collapse"],
        "bh_decay":    params["bh_decay"],
    }

    # --- Inline indicator helpers (no external deps) ---
    class EMA:
        def __init__(self, p): self.k = 2/(p+1); self.v = None
        def update(self, x):
            self.v = x if self.v is None else x*self.k + self.v*(1-self.k)
            return self.v
    class ATR_:
        def __init__(self,p=14): self.p=p;self._prev=None;self._buf=[];self.v=None
        def update(self,h,lo,c):
            tr=(h-lo) if self._prev is None else max(h-lo,abs(h-self._prev),abs(lo-self._prev))
            self._prev=c
            if self.v is None:
                self._buf.append(tr)
                if len(self._buf)>=self.p: self.v=sum(self._buf)/len(self._buf)
            else: self.v=(self.v*(self.p-1)+tr)/self.p
            return self.v or tr
    class ADX_:
        def __init__(self,p=14):
            self.p=p;self._atr=ATR_(p);self._ph=self._pl=None
            self._pdm=self._ndm=0.0;self._dx=[];self.v=0.0
        def update(self,h,lo,c):
            atr=self._atr.update(h,lo,c)
            if self._ph is None: self._ph,self._pl=h,lo; return 0.0
            pdm=max(0.0,h-self._ph); ndm=max(0.0,self._pl-lo)
            if pdm>ndm: ndm=0.0
            elif ndm>pdm: pdm=0.0
            else: pdm=ndm=0.0
            k=2/(self.p+1)
            self._pdm=pdm*k+self._pdm*(1-k); self._ndm=ndm*k+self._ndm*(1-k)
            if atr>0:
                pdi,ndi=100*self._pdm/atr,100*self._ndm/atr
                d=pdi+ndi; dx=100*abs(pdi-ndi)/d if d>0 else 0.0
            else: dx=0.0
            self._dx.append(dx)
            if len(self._dx)>=self.p: self.v=sum(self._dx[-self.p:])/self.p
            self._ph,self._pl=h,lo; return self.v
    class RSI_:
        def __init__(self,p=14): self.p=p;self._prev=None;self._g=[];self._l=[];self._ag=self._al=None;self.v=50.0
        def update(self,c):
            if self._prev is None: self._prev=c; return 50.0
            chg=c-self._prev; self._prev=c
            if self._ag is None:
                self._g.append(max(0,chg)); self._l.append(max(0,-chg))
                if len(self._g)>=self.p:
                    self._ag=sum(self._g)/self.p; self._al=sum(self._l)/self.p
                    self.v=100-100/(1+self._ag/self._al) if self._al else 100.0
            else:
                self._ag=(self._ag*(self.p-1)+max(0,chg))/self.p
                self._al=(self._al*(self.p-1)+max(0,-chg))/self.p
                self.v=100-100/(1+self._ag/self._al) if self._al else 100.0
            return self.v
    class BB_:
        def __init__(self,p=20): self.p=p;self._buf=[]
        def update(self,c):
            self._buf.append(c)
            if len(self._buf)>self.p: self._buf.pop(0)
            mid=sum(self._buf)/len(self._buf)
            std=(sum((x-mid)**2 for x in self._buf)/len(self._buf))**0.5
            return mid,std

    mc  = MinkowskiClassifier(cf=cfg["cf"])
    bh  = BlackHoleDetector(cfg["bh_form"], cfg["bh_collapse"], cfg["bh_decay"])
    geo = GeodesicAnalyzer()
    gl  = GravitationalLens()
    hw  = HawkingMonitor()
    reg = RegimeDetector(atr_window=50)
    kc  = KillConditions()

    e12=EMA(12);e26=EMA(26);e50=EMA(50);e200=EMA(200)
    atr_=ATR_();adx_=ADX_();rsi_=RSI_();bb_=BB_()

    broker = SimulatedBroker(
        cash=1_000_000,
        fee_per_trade=50,
        slippage_pct=0.0002,
        max_leverage=params["max_leverage"],
    )

    prev_close = None
    bar_count  = 0

    for bar in bars:
        close  = bar["close"]
        high   = bar["high"]
        low    = bar["low"]
        volume = bar["volume"]

        if prev_close is None or prev_close <= 0:
            prev_close = close
            mc.update(close)
            e12.update(close);e26.update(close);e50.update(close);e200.update(close)
            atr_.update(high,low,close);adx_.update(high,low,close)
            rsi_.update(close);bb_.update(close)
            broker.update(0.0)
            continue

        bar_count  += 1
        bar_return  = (close - prev_close) / prev_close

        bit    = mc.update(close)
        bh.update(bit, close, prev_close)

        ema12=e12.update(close);ema26=e26.update(close)
        ema50=e50.update(close);ema200=e200.update(close)
        atr_v=atr_.update(high,low,close)
        adx_v=adx_.update(high,low,close)
        rsi_v=rsi_.update(close)
        bb_mid,bb_std=bb_.update(close)

        geo_dev,geo_slope,causal_frac,rapidity = geo.update(close, atr_v)
        gl.update(close, volume, bit, mc.tl_confirm, atr_v)
        mu = gl.mu
        ht = hw.update(close, bb_mid, bb_std)

        regime, conf = reg.update(close, ema12, ema26, ema50, ema200, adx_v, atr_v)

        f = np.zeros(31, dtype=np.float32)
        f[0]  = np.clip((rsi_v-50)/50, -3, 3)
        f[1]  = np.clip((ema12-ema26)/(close+1e-9)*1000, -3, 3)
        f[4]  = np.clip((close-ema50)/(atr_v+1e-9), -3, 3)
        f[11] = np.clip(adx_v/50, -3, 3)
        f[14] = np.clip(bh.bh_mass/3.0, -3, 3)
        f[15] = float(bh.bh_dir)
        f[16] = np.clip(geo_dev, -3, 3)
        f[17] = np.clip(geo_slope, -3, 3)
        f[22] = np.clip(mu, -3, 3)
        f[23] = np.clip(atr_v/(close+1e-9)*1000, -3, 3)
        bb_pct = (close-(bb_mid-2*bb_std))/(4*bb_std+1e-9)
        f[24] = np.clip(bb_pct-0.5, -3, 3)
        f[25] = np.clip(bb_std/(close+1e-9)*1000, -3, 3)
        f[30] = np.clip(ht, -3, 3)
        f[27+min(int(regime),3)] = 1.0

        action, conf, _ = ensemble(f, mu, causal_frac, ht, mc.beta, regime, geo_slope, geo_dev, rapidity)

        rm  = min(1.0, broker.equity / broker.initial_equity)
        tl_window = [1.0 if mc.tl_confirm > 0 else 0.0]
        tgt = size_position(f, action, conf, rm, regime, mu, causal_frac, ht, tl_window)
        tgt = float(np.clip(tgt, -params["max_leverage"], params["max_leverage"]))

        killed, tgt = kc.apply(
            tgt=tgt, geo_dev=abs(geo_dev), bc=bar_count,
            tl_confirm=mc.tl_confirm, bit=bit, regime=regime,
            ctl=bh.ctl, last_target=broker.position, ramp_back=0,
        )
        if killed:
            tgt = 0.0

        broker.update(bar_return)
        broker.set_position(tgt)
        prev_close = close

    stats = broker.stats()
    return {
        "id":           params["id"],
        "cf":           round(params["cf"], 6),
        "bh_form":      round(params["bh_form"], 4),
        "bh_collapse":  round(params["bh_collapse"], 4),
        "bh_decay":     round(params["bh_decay"], 4),
        "max_leverage": round(params["max_leverage"], 3),
        **stats,
    }


# --- Data loading (shared) ---------------------------------------------------

def load_bars(path: str) -> List[dict]:
    bars = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            def g(*keys):
                for k in keys:
                    for kk in [k, k.lower(), k.upper()]:
                        v = row.get(kk)
                        if v not in (None, "", "null", "None"):
                            try: return float(v)
                            except: pass
                return None
            c = g("close","Close")
            h = g("high","High") or c
            lo = g("low","Low") or c
            v = g("volume","Volume") or 1000.0
            if c and c > 0:
                bars.append({"close":c,"high":h,"low":lo,"volume":v})
    return bars


def generate_bars(n_bars: int, seed: int = 42) -> List[dict]:
    rng = np.random.default_rng(seed)
    REGIMES = {
        "bull":     (0.00015, 0.00070),
        "bear":     (-0.0002, 0.00110),
        "sideways": (0.00005, 0.00050),
        "crisis":   (-0.0008, 0.00280),
    }
    TRANS = {
        "bull":     {"bull":0.60,"sideways":0.30,"bear":0.05,"crisis":0.05},
        "bear":     {"bear":0.50,"sideways":0.30,"bull":0.10,"crisis":0.10},
        "sideways": {"sideways":0.40,"bull":0.35,"bear":0.20,"crisis":0.05},
        "crisis":   {"bear":0.50,"sideways":0.35,"bull":0.10,"crisis":0.05},
    }
    price   = 4500.0
    regime  = "bull"
    dur_rem = int(rng.integers(200, 500))
    bars    = []
    for _ in range(n_bars):
        mu, sig = REGIMES[regime]
        ret     = max(-0.05, min(0.05, mu + sig * float(rng.standard_normal())))
        close   = price * (1 + ret)
        h = close * (1 + abs(float(rng.standard_normal())) * sig * 0.4)
        lo = close * (1 - abs(float(rng.standard_normal())) * sig * 0.4)
        bars.append({"close":close,"high":h,"low":lo,"volume":50000.0})
        price   = close
        dur_rem -= 1
        if dur_rem <= 0:
            probs = TRANS[regime]
            regime  = str(rng.choice(list(probs.keys()), p=list(probs.values())))
            dur_rem = int(rng.integers(100, 400))
    return bars


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRFM parameter tournament")
    parser.add_argument("--csv",     help="Real price CSV (uses synthetic if omitted)")
    parser.add_argument("--n-bars",  type=int, default=20000, help="Synthetic bars if no CSV")
    parser.add_argument("--n-sims",  type=int, default=200,   help="Parameter combinations to test")
    parser.add_argument("--workers", type=int, default=1,     help="Parallel workers (1 = serial)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--top",     type=int, default=20,    help="Show top N results")
    args = parser.parse_args()

    # Load / generate price data
    if args.csv:
        print(f"Loading {args.csv}...")
        bars = load_bars(args.csv)
        print(f"  {len(bars)} bars")
    else:
        print(f"Generating {args.n_bars} synthetic bars (seed={args.seed})...")
        bars = generate_bars(args.n_bars, args.seed)

    # Build parameter grid
    params = build_param_grid(args.n_sims, seed=args.seed)
    print(f"Running {len(params)} parameter combinations "
          f"({'parallel x' + str(args.workers) if args.workers > 1 else 'serial'})...")

    t0 = time.time()
    results = []

    if args.workers > 1:
        work = [(p, bars, args.seed) for p in params]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, w): w[0]["id"] for w in work}
            done = 0
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"  {done}/{len(params)}  ({rate:.1f} runs/s)")
    else:
        for i, p in enumerate(params):
            results.append(_run_one((p, bars, args.seed)))
            if (i+1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                print(f"  {i+1}/{len(params)}  ({rate:.1f} runs/s)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} runs in {elapsed:.1f}s ({len(results)/elapsed:.1f} runs/s)")

    # Sort by Sharpe
    results.sort(key=lambda r: r.get("sharpe", -99), reverse=True)

    # Print leaderboard
    print(f"\n{'='*90}")
    print(f"  TOURNAMENT LEADERBOARD (top {args.top} by Sharpe)")
    print(f"{'='*90}")
    hdr = (f"{'#':>3} {'Sharpe':>7} {'Return%':>8} {'MaxDD%':>7} "
           f"{'Trades':>7} {'CF':>8} {'BHForm':>7} {'BHDecay':>8} {'MaxLev':>7}")
    print(hdr)
    print("-" * 90)
    for rank, r in enumerate(results[:args.top], 1):
        print(f"{rank:>3} {r['sharpe']:>7.3f} {r['total_return_pct']:>+8.2f}% "
              f"{r['max_drawdown_pct']:>6.2f}% {r['trade_count']:>7} "
              f"{r['cf']:>8.5f} {r['bh_form']:>7.3f} {r['bh_decay']:>8.4f} "
              f"{r['max_leverage']:>7.3f}")
    print("=" * 90)

    # LARSA baseline
    larsa = {"cf":0.001,"bh_form":1.5,"bh_collapse":1.0,"bh_decay":0.95,"max_leverage":1.0}
    larsa_result = _run_one(({"id":-1, **larsa}, bars, args.seed))
    print(f"\nLARSA baseline: Sharpe={larsa_result['sharpe']:.3f}  "
          f"Return={larsa_result['total_return_pct']:+.2f}%  "
          f"MaxDD={larsa_result['max_drawdown_pct']:.2f}%  "
          f"Trades={larsa_result['trade_count']}")

    # Save leaderboard
    os.makedirs("results/tournament", exist_ok=True)
    csv_path = "results/tournament/leaderboard.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\nCSV -> {csv_path}")

    with open("results/tournament/summary.json", "w") as f:
        json.dump({
            "n_runs": len(results),
            "n_bars": len(bars),
            "elapsed_s": round(elapsed, 1),
            "runs_per_sec": round(len(results)/elapsed, 1),
            "larsa_baseline": larsa_result,
            "top_1": results[0] if results else {},
        }, f, indent=2)
    print("JSON -> results/tournament/summary.json")


if __name__ == "__main__":
    main()
