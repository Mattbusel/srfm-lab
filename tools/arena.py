"""
arena.py — Pure-Python LARSA backtest arena. No LEAN, no Docker, no QC.

Runs the full SRFM strategy stack (from lib/) on real or synthetic price data
using SimulatedBroker. Thousands of backtests per minute.

Usage:
    python tools/arena.py --csv data/ES_hourly_real.csv --ticker ES
    python tools/arena.py --synthetic --n-bars 20000 --seed 42
    python tools/arena.py --csv data/ES_hourly_real.csv --ticker ES --plot

The strategy signal pipeline (mirrors LARSA logic):
  1. MinkowskiClassifier  -> TIMELIKE/SPACELIKE
  2. BlackHoleDetector    -> bh_active, bh_dir, bh_mass
  3. GeodesicAnalyzer     -> geo_dev, geo_slope
  4. GravitationalLens    -> mu
  5. HawkingMonitor       -> ht
  6. RegimeDetector       -> regime, confidence
  7. KillConditions       -> position signal (or 0)
  8. size_position()      -> final position size
  9. SimulatedBroker      -> P&L tracking
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


# --- Asset configs (mirror LARSA) -------------------------------------------

CONFIGS = {
    "ES": {"cf": 0.001,  "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "NQ": {"cf": 0.0012, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
    "YM": {"cf": 0.0008, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95},
}
DEFAULT_CFG = {"cf": 0.001, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}


# --- Data loading ------------------------------------------------------------

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
            h = g("high",  "High")  or c
            lo = g("low",  "Low")   or c
            v = g("volume","Volume") or 1000.0
            d = (row.get("date") or row.get("Date") or
                 row.get("Datetime") or row.get("datetime") or "")
            if c and c > 0:
                bars.append({"date": d, "open": g("open","Open") or c,
                             "high": h, "low": lo, "close": c, "volume": v})
    return bars


def generate_synthetic(n_bars: int = 20000, seed: int = 42, regime: str = "mixed") -> List[dict]:
    rng = np.random.default_rng(seed)
    REGIMES = {
        "bull":     (0.0002, 0.0008),
        "bear":     (-0.0003, 0.0014),
        "sideways": (0.00005, 0.0006),
        "crisis":   (-0.001, 0.003),
    }
    price = 4500.0
    bars = []
    if regime == "mixed":
        regime_seq = list(rng.choice(["bull","bear","sideways","crisis"],
                                     size=n_bars,
                                     p=[0.55,0.15,0.25,0.05]))
    else:
        regime_seq = [regime] * n_bars

    for i in range(n_bars):
        mu, sig = REGIMES[regime_seq[i]]
        ret   = mu + sig * float(rng.standard_normal())
        ret   = max(-0.05, min(0.05, ret))
        close = price * (1 + ret)
        bars.append({
            "date": f"bar_{i:06d}", "open": price,
            "high": close * (1 + abs(float(rng.standard_normal())) * sig * 0.3),
            "low":  close * (1 - abs(float(rng.standard_normal())) * sig * 0.3),
            "close": close, "volume": 50000.0,
        })
        price = close
    return bars


# --- Indicator helpers -------------------------------------------------------

class _EMA:
    def __init__(self, p):
        self.k = 2.0 / (p + 1); self.v = None
    def update(self, x):
        self.v = x if self.v is None else x * self.k + self.v * (1 - self.k)
        return self.v

class _ATR:
    def __init__(self, p=14):
        self.p = p; self._prev = None; self._buf = []; self.v = None
    def update(self, h, lo, c):
        tr = (h - lo) if self._prev is None else max(h-lo,abs(h-self._prev),abs(lo-self._prev))
        self._prev = c
        if self.v is None:
            self._buf.append(tr)
            if len(self._buf) >= self.p:
                self.v = sum(self._buf) / len(self._buf)
        else:
            self.v = (self.v * (self.p-1) + tr) / self.p
        return self.v or tr

class _ADX:
    def __init__(self, p=14):
        self.p = p; self._atr = _ATR(p)
        self._ph = self._pl = None
        self._pdm = self._ndm = 0.0
        self._dx_buf = []; self.v = 0.0
    def update(self, h, lo, c):
        atr = self._atr.update(h, lo, c)
        if self._ph is None:
            self._ph, self._pl = h, lo; return 0.0
        pdm = max(0.0, h - self._ph); ndm = max(0.0, self._pl - lo)
        if pdm > ndm: ndm = 0.0
        elif ndm > pdm: pdm = 0.0
        else: pdm = ndm = 0.0
        k = 2.0 / (self.p + 1)
        self._pdm = pdm * k + self._pdm * (1-k)
        self._ndm = ndm * k + self._ndm * (1-k)
        if atr > 0:
            pdi, ndi = 100*self._pdm/atr, 100*self._ndm/atr
            d = pdi + ndi
            dx = 100*abs(pdi-ndi)/d if d > 0 else 0.0
        else:
            dx = 0.0
        self._dx_buf.append(dx)
        if len(self._dx_buf) >= self.p:
            self.v = sum(self._dx_buf[-self.p:]) / self.p
        self._ph, self._pl = h, lo
        return self.v

class _RSI:
    def __init__(self, p=14):
        self.p = p; self._prev = None; self._gains = []; self._losses = []
        self._avg_gain = self._avg_loss = None; self.v = 50.0
    def update(self, c):
        if self._prev is None: self._prev = c; return 50.0
        chg = c - self._prev; self._prev = c
        if self._avg_gain is None:
            self._gains.append(max(0, chg)); self._losses.append(max(0, -chg))
            if len(self._gains) >= self.p:
                self._avg_gain = sum(self._gains) / self.p
                self._avg_loss = sum(self._losses) / self.p
                if self._avg_loss == 0: self.v = 100.0
                else: self.v = 100 - 100/(1+self._avg_gain/self._avg_loss)
        else:
            self._avg_gain = (self._avg_gain*(self.p-1) + max(0,chg)) / self.p
            self._avg_loss = (self._avg_loss*(self.p-1) + max(0,-chg)) / self.p
            if self._avg_loss == 0: self.v = 100.0
            else: self.v = 100 - 100/(1+self._avg_gain/self._avg_loss)
        return self.v

class _BB:
    def __init__(self, p=20):
        self.p = p; self._buf = []
    def update(self, c):
        self._buf.append(c)
        if len(self._buf) > self.p: self._buf.pop(0)
        mid = sum(self._buf)/len(self._buf)
        std = (sum((x-mid)**2 for x in self._buf)/len(self._buf))**0.5
        return mid, std


# --- Strategy runner ---------------------------------------------------------

def run_strategy(
    bars: List[dict],
    cfg: dict,
    max_leverage: float = 1.0,
    verbose: bool = False,
) -> Tuple[SimulatedBroker, List[dict]]:

    # Physics
    mc  = MinkowskiClassifier(cf=cfg["cf"])
    bh  = BlackHoleDetector(cfg["bh_form"], cfg["bh_collapse"], cfg["bh_decay"])
    geo = GeodesicAnalyzer()
    gl  = GravitationalLens()
    hw  = HawkingMonitor()
    reg = RegimeDetector(atr_window=50)

    # Indicators
    e12  = _EMA(12);  e26  = _EMA(26)
    e50  = _EMA(50);  e200 = _EMA(200)
    atr_ = _ATR(14);  adx_ = _ADX(14)
    rsi_ = _RSI(14);  bb_  = _BB(20)

    broker = SimulatedBroker(
        cash=1_000_000,
        fee_per_trade=50,
        slippage_pct=0.0002,
        max_leverage=max_leverage,
    )

    kc     = KillConditions()
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
            atr_.update(high, low, close); adx_.update(high, low, close)
            rsi_.update(close); bb_.update(close)
            broker.update(0.0)
            continue

        bar_count += 1
        bar_return = (close - prev_close) / prev_close

        # Physics
        bit    = mc.update(close)
        active = bh.update(bit, close, prev_close)

        # Indicators
        ema12  = e12.update(close);  ema26  = e26.update(close)
        ema50  = e50.update(close);  ema200 = e200.update(close)
        atr_v  = atr_.update(high, low, close)
        adx_v  = adx_.update(high, low, close)
        rsi_v  = rsi_.update(close)
        bb_mid, bb_std = bb_.update(close)

        # Geodesic + lensing + Hawking
        geo_dev, geo_slope, causal_frac, rapidity = geo.update(close, atr_v)
        gl.update(close, volume, bit, mc.tl_confirm, atr_v)
        mu = gl.mu
        ht = hw.update(close, bb_mid, bb_std)

        # Regime
        regime, conf = reg.update(close, ema12, ema26, ema50, ema200, adx_v, atr_v)

        # Feature vector (simplified — key fields only, enough for ensemble)
        # Build a minimal feature array matching the 31-element spec from features.py
        import numpy as np
        f = np.zeros(31, dtype=np.float32)
        macd_val = (ema12 - ema26) / (close + 1e-9)
        f[0]  = np.clip((rsi_v - 50) / 50, -3, 3)           # F_RSI
        f[1]  = np.clip(macd_val * 1000, -3, 3)              # F_MACD
        f[4]  = np.clip((close - ema50) / (atr_v + 1e-9), -3, 3)   # F_MOM
        f[11] = np.clip(adx_v / 50, -3, 3)                   # F_ADX
        f[14] = np.clip(bh.bh_mass / 3.0, -3, 3)             # F_BHMASS
        f[15] = float(bh.bh_dir)                              # F_BHDIR
        f[16] = np.clip(geo_dev, -3, 3)                       # F_GEODEV
        f[17] = np.clip(geo_slope, -3, 3)                     # F_GEOSLOPE
        f[22] = np.clip(mu, -3, 3)                            # F_MU
        f[23] = np.clip(atr_v / (close + 1e-9) * 1000, -3, 3) # F_ATR
        bb_pct = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)
        f[24] = np.clip(bb_pct - 0.5, -3, 3)                 # F_BBP
        f[25] = np.clip(bb_std / (close + 1e-9) * 1000, -3, 3) # F_BBW
        f[30] = np.clip(ht, -3, 3)                            # F_HT
        # Regime one-hot (indices 27-30 per features.py, but we'll use a simplified slot)
        regime_idx = int(regime)
        f[27 + min(regime_idx, 3)] = 1.0

        # Ensemble runs every bar (BH active is a feature, not a hard gate)
        action, conf, _ = ensemble(
            f, mu, causal_frac, ht,
            mc.beta, regime, geo_slope, geo_dev, rapidity,
        )
        rm = broker.equity / broker.initial_equity   # simplified risk multiplier
        rm = min(1.0, rm)
        tl_window = [1.0 if mc.tl_confirm > 0 else 0.0]  # simplified
        target_pos = size_position(
            f, action, conf, rm, regime, mu, causal_frac, ht, tl_window,
        )
        target_pos = float(np.clip(target_pos, -max_leverage, max_leverage))

        # Kill conditions gate
        killed, target_pos = kc.apply(
            tgt=target_pos,
            geo_dev=abs(geo_dev),
            bc=bar_count,
            tl_confirm=mc.tl_confirm,
            bit=bit,
            regime=regime,
            ctl=bh.ctl,
            last_target=broker.position,
            ramp_back=0,
        )
        if killed:
            target_pos = 0.0

        broker.update(bar_return)
        broker.set_position(target_pos)

        if verbose:
            bar_log.append({
                "date": date, "close": close, "regime": regime.name,
                "bh_active": active, "bh_mass": round(bh.bh_mass, 4),
                "mu": round(mu, 4), "ht": round(ht, 4),
                "position": target_pos, "equity": round(broker.equity, 0),
            })

        prev_close = close

    return broker, bar_log


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRFM arena — no LEAN needed")
    parser.add_argument("--csv",         help="Price CSV (date,open,high,low,close,volume)")
    parser.add_argument("--ticker",      default="ES")
    parser.add_argument("--synthetic",   action="store_true", help="Use synthetic data")
    parser.add_argument("--n-bars",      type=int, default=20000)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--regime",      default="mixed",
                        choices=["mixed","bull","bear","sideways","crisis"])
    parser.add_argument("--max-leverage",type=float, default=1.0)
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--plot",        action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        print(f"Generating {args.n_bars} synthetic {args.regime} bars (seed={args.seed})...")
        bars = generate_synthetic(args.n_bars, args.seed, args.regime)
    elif args.csv:
        print(f"Loading {args.csv}...")
        bars = load_ohlcv(args.csv)
        print(f"  {len(bars)} bars")
    else:
        parser.error("Provide --csv or --synthetic")

    cfg = CONFIGS.get(args.ticker, DEFAULT_CFG)
    print(f"Running strategy (ticker={args.ticker}, cf={cfg['cf']}, max_lev={args.max_leverage})...")

    broker, log = run_strategy(bars, cfg, args.max_leverage, verbose=args.verbose or args.plot)
    stats = broker.stats()

    print(f"\n{'='*50}")
    print(f"  SRFM Arena Results -- {args.ticker}")
    print(f"{'='*50}")
    print(f"  Total return  : {stats['total_return_pct']:+.2f}%")
    print(f"  Max drawdown  : {stats['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe (ann.) : {stats['sharpe']:.3f}")
    print(f"  Trades        : {stats['trade_count']}")
    print(f"  Win rate      : {stats['win_rate']:.1%}")
    print(f"  Final equity  : ${stats['final_equity']:,.0f}")
    print(f"{'='*50}\n")

    if args.verbose and log:
        print("\nBar log (last 20):")
        for row in log[-20:]:
            print(f"  {row['date']:<22} close={row['close']:>9.2f}  "
                  f"regime={row['regime']:<16}  BH={row['bh_active']}  "
                  f"pos={row['position']:>+5.2f}  eq=${row['equity']:>12,.0f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            curve = np.array(broker.equity_curve)
            bh_active = [r["bh_active"] for r in log] if log else []

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)

            ax1.plot(curve, color="steelblue", linewidth=1.2, label="Equity")
            ax1.axhline(broker.initial_equity, color="gray", linestyle="--", alpha=0.5)
            ax1.set_title(f"SRFM Arena -- {args.ticker}  |  Return: {stats['total_return_pct']:+.2f}%  "
                          f"MaxDD: {stats['max_drawdown_pct']:.2f}%  Sharpe: {stats['sharpe']:.2f}",
                          fontweight="bold")
            ax1.set_ylabel("Equity ($)")
            ax1.legend(); ax1.grid(alpha=0.3)

            # Price
            closes = [b["close"] for b in bars]
            ax2.plot(closes, color="black", linewidth=0.7)
            # Shade BH-active bars
            if bh_active:
                for i, a in enumerate(bh_active):
                    if a:
                        ax2.axvspan(i, i+1, alpha=0.2, color="green")
            ax2.set_ylabel("Price"); ax2.set_xlabel("Bar")
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            out = f"results/arena_{args.ticker}.png"
            os.makedirs("results", exist_ok=True)
            plt.savefig(out, dpi=150)
            print(f"Plot saved -> {out}")
            plt.show()
        except ImportError:
            print("[WARN] matplotlib not installed; skipping plot")

    # Save summary
    os.makedirs("results", exist_ok=True)
    summary_path = f"results/arena_{args.ticker}_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump({"ticker": args.ticker, "bars": len(bars), **stats}, f, indent=2)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
