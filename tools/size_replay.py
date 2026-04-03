"""
size_replay.py — Per-bar position sizing state logger for LARSA v11.

Runs v11 on a 4000-bar synthetic world with a vol spike at bar 2000.
Logs every sizing decision, ATR, TF score, notional exposure, and rolling
correlations.  Outputs a rich CSV log and a 5-panel chart.

Outputs:
    results/size_replay.csv   — full per-bar log
    results/size_replay.png   — 5-panel chart

Usage:
    python tools/size_replay.py
    python tools/size_replay.py --n-bars 4000 --spike-at 2000
"""

import argparse
import csv
import math
import os
import sys
from collections import deque

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
CF = {
    "15m": {"ES": 0.0003, "NQ": 0.0004,  "YM": 0.00025},
    "1h":  {"ES": 0.001,  "NQ": 0.0012,  "YM": 0.0008},
    "1d":  {"ES": 0.005,  "NQ": 0.006,   "YM": 0.004},
}
TF_CAP        = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
MIN_HOLD      = 4
N_INST        = 3
BASE_CORR     = 0.90
PORTFOLIO_RISK = 0.01
SYMS          = ["ES", "NQ", "YM"]
START_EQUITY  = 1_000_000.0
DAILY_VOL_THRESH = 0.013   # 1.3 %/day


# ── BH Instrument ────────────────────────────────────────────────────────────
class BHI:
    def __init__(self, sym, cf, res):
        self.sym = sym; self.cf = cf; self.res = res
        self.bh_mass = 0.0; self.bh_active = False; self.bh_dir = 0
        self.ctl = 0; self.bc = 0; self.prices = []
        self._wu = {"15m": 400, "1h": 120, "1d": 30}.get(res, 120)

    def update(self, c):
        self.bc += 1; self.prices.append(c)
        if len(self.prices) < 2:
            return
        beta = abs(c - self.prices[-2]) / (self.prices[-2] + 1e-9) / (self.cf + 1e-9)
        was = self.bh_active
        if beta < 1.0:
            self.ctl += 1
            self.bh_mass = self.bh_mass * 0.97 + 0.03 * min(2.0, 1 + self.ctl * 0.1)
        else:
            self.ctl = 0; self.bh_mass *= 0.95
        self.bh_active = (self.bh_mass > (1.5 if not was else 1.0)) and self.ctl >= 3
        if not was and self.bh_active:
            lb = min(20, len(self.prices) - 1)
            self.bh_dir = 1 if c > self.prices[-1 - lb] else -1
        elif was and not self.bh_active:
            self.bh_dir = 0
        if self.bc < self._wu:
            self.bh_active = False; self.bh_dir = 0

    def direction(self):
        if self.bh_dir:
            return self.bh_dir
        if len(self.prices) >= 5:
            return 1 if self.prices[-1] > self.prices[-5] else -1
        return 0


class WilderATR:
    def __init__(self, p=14):
        self.p = p; self._trs = []; self._v = None

    def update(self, h, l, pc):
        tr = max(h - l, abs(h - pc), abs(l - pc))
        self._trs.append(tr)
        if len(self._trs) >= self.p:
            self._v = (np.mean(self._trs[-self.p:]) if self._v is None
                       else (self._v * (self.p - 1) + tr) / self.p)
        return self._v or 0.0

    @property
    def value(self):
        return self._v or 0.0


class MRI:
    def __init__(self, sym):
        self.sym = sym
        self.i15 = BHI(sym, CF["15m"][sym], "15m")
        self.i1h = BHI(sym, CF["1h"][sym],  "1h")
        self.i1d = BHI(sym, CF["1d"][sym],  "1d")
        self.atr = WilderATR(14)
        self.last_target = 0.0; self.bars_held = 0; self.pos_floor = 0.0

    def tf_score(self):
        return 4 * self.i1d.bh_active + 2 * self.i1h.bh_active + self.i15.bh_active

    def direction(self):
        if self.i1d.bh_active: return self.i1d.direction()
        if self.i1h.bh_active: return self.i1h.direction()
        if self.i15.bh_active: return self.i15.direction()
        return 0

    def update(self, o, h, l, c, pc):
        for sub in [o, (o + h) / 2, (l + c) / 2, c]:
            self.i15.update(sub)
        self.i1h.update(c)
        self.atr.update(h, l, pc)
        if self.i1h.bc % 6 == 0:
            self.i1d.update(c)


# ── Sizing ────────────────────────────────────────────────────────────────────
def size_v11(ceiling, direction, atr, price, corr=BASE_CORR):
    if not direction or not ceiling:
        return 0.0
    cf = math.sqrt(N_INST + N_INST * (N_INST - 1) * corr)
    per = PORTFOLIO_RISK / cf
    if atr > 0 and price > 0:
        dv = (atr / price) * math.sqrt(6.5)
        raw = per / (dv + 1e-9)
        return min(raw, ceiling) * direction
    return ceiling * direction


# ── World generator ──────────────────────────────────────────────────────────
def gen_world(n=4000, seed=42, vol_spike_at=2000, drift=0.0003, corr=0.90):
    rng = np.random.default_rng(seed)
    vols = np.array([0.15, 0.20, 0.14]) / math.sqrt(252 * 6.5)
    C = np.array([[1, corr, corr], [corr, 1, corr], [corr, corr, 1]], dtype=float)
    L = np.linalg.cholesky(np.outer(vols, vols) * C)
    starts = {"ES": 4000.0, "NQ": 14000.0, "YM": 33000.0}
    ps = {s: [v] for s, v in starts.items()}
    for i in range(n):
        base_sig = 1.0
        if vol_spike_at is not None and abs(i - vol_spike_at) < 30:
            base_sig = 6.0
        z = L @ rng.standard_normal(3)
        for j, s in enumerate(SYMS):
            ret = drift + base_sig * vols[j] * z[j] * math.sqrt(252 * 6.5)
            ps[s].append(ps[s][-1] * (1 + ret))
    result = {s: [] for s in SYMS}
    for s in SYMS:
        v = ps[s]
        for i in range(1, len(v)):
            hi = max(v[i], v[i - 1]) * (1 + abs(rng.normal(0, 0.002)))
            lo = min(v[i], v[i - 1]) * (1 - abs(rng.normal(0, 0.002)))
            result[s].append({"o": v[i - 1], "h": hi, "l": lo, "c": v[i]})
    return result


# ── Rolling correlation helper ────────────────────────────────────────────────
class RollingCorr:
    def __init__(self, window=20):
        self.w = window
        self._x: deque = deque(maxlen=window)
        self._y: deque = deque(maxlen=window)

    def update(self, x, y):
        self._x.append(x); self._y.append(y)
        if len(self._x) < self.w:
            return float("nan")
        xa = np.array(self._x); ya = np.array(self._y)
        if xa.std() < 1e-9 or ya.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(xa, ya)[0, 1])


# ── Simulation + per-bar logging ──────────────────────────────────────────────
def run_replay(n_bars=4000, spike_at=2000):
    bars = gen_world(n=n_bars, seed=42, vol_spike_at=spike_at)
    insts = {s: MRI(s) for s in SYMS}
    equity = START_EQUITY
    peak   = equity
    positions = {s: 0.0 for s in SYMS}
    pc    = {s: bars[s][0]["c"] for s in SYMS}

    corr_es_nq = RollingCorr(20)
    corr_es_ym = RollingCorr(20)

    # synthetic date: start 2018-01-02, 1 bar ~ 1 hour (6.5 bars/day ~ 6 bars)
    from datetime import date, timedelta
    base_date = date(2018, 1, 2)

    log = []

    for i in range(n_bars):
        # mark-to-market
        for s in SYMS:
            b = bars[s][i]
            pos = positions[s]
            if abs(pos) > 1e-6 and pc[s] > 0:
                equity += pos * (b["c"] - pc[s]) / pc[s] * equity

        # update instruments
        pr = {}
        for s in SYMS:
            b = bars[s][i]
            insts[s].update(b["o"], b["h"], b["l"], b["c"], pc[s])
            pc[s] = b["c"]; pr[s] = b["c"]

        equity = max(equity, 0.0)
        if equity > peak:
            peak = equity
        dd_pct = (peak - equity) / (peak + 1e-9) * 100.0

        # sizing
        raw_sizes   = {}
        capped_sizes = {}
        daily_vols  = {}
        tfs_map     = {}
        atr_map     = {}
        for s in SYMS:
            inst = insts[s]
            tfs = inst.tf_score(); ceiling = TF_CAP[tfs]
            if tfs == 1 and abs(inst.last_target) < 0.01:
                ceiling = 0.0
            d   = inst.direction()
            a   = inst.atr.value
            p   = pr[s]
            dv  = (a / p) * math.sqrt(6.5) if a > 0 and p > 0 else 0.0
            raw = size_v11(ceiling, d, a, p) if ceiling > 0 else 0.0
            raw_sizes[s]    = raw
            capped_sizes[s] = min(abs(raw), ceiling) * (1 if raw >= 0 else -1)
            daily_vols[s]   = dv
            tfs_map[s]      = tfs
            atr_map[s]      = a

        total = sum(abs(v) for v in raw_sizes.values())
        scale = 1.0 / total if total > 1.0 else 1.0
        executed_this_bar = False
        for s in SYMS:
            inst = insts[s]
            tgt = raw_sizes[s] * scale
            old = inst.last_target
            if abs(tgt - old) < 0.02:
                continue
            # hold gate
            is_rev = (abs(old) > 0.001 and abs(tgt) > 0.001 and
                      math.copysign(1, tgt) != math.copysign(1, old))
            if is_rev and inst.bars_held < MIN_HOLD:
                tgt = old
            p = pr[s]
            fee = max(1, int(abs(tgt - old) * equity / (p * 50 + 1e-9))) * 4.0
            equity -= fee
            executed_this_bar = True
            if abs(tgt) < 1e-9:
                inst.bars_held = 0
            elif math.copysign(1, tgt) != math.copysign(1, old):
                inst.bars_held = 0
            else:
                inst.bars_held += 1
            inst.last_target = tgt
            positions[s] = tgt

        # notional exposure
        notional_total = sum(abs(positions[s]) for s in SYMS)
        margin_util    = notional_total  # as fraction of equity (positions are already fractional)

        # rolling returns for correlation
        ret_es = (pr["ES"] - pc.get("ES_prev", pr["ES"])) / (pr["ES"] + 1e-9)
        ret_nq = (pr["NQ"] - pc.get("NQ_prev", pr["NQ"])) / (pr["NQ"] + 1e-9)
        ret_ym = (pr["YM"] - pc.get("YM_prev", pr["YM"])) / (pr["YM"] + 1e-9)
        corr_en = corr_es_nq.update(pr["ES"], pr["NQ"])
        corr_ey = corr_es_ym.update(pr["ES"], pr["YM"])

        # synthetic date
        bar_date = (base_date + timedelta(hours=i)).isoformat()

        row = {
            "bar_idx": i,
            "date": bar_date,
            "equity": round(equity, 2),
            "peak_equity": round(peak, 2),
            "drawdown_pct": round(dd_pct, 4),
            # per instrument
            "ES_atr": round(atr_map["ES"], 4),
            "NQ_atr": round(atr_map["NQ"], 4),
            "YM_atr": round(atr_map["YM"], 4),
            "ES_daily_vol_pct": round(daily_vols["ES"] * 100, 4),
            "NQ_daily_vol_pct": round(daily_vols["NQ"] * 100, 4),
            "YM_daily_vol_pct": round(daily_vols["YM"] * 100, 4),
            "ES_raw_size": round(raw_sizes["ES"], 6),
            "NQ_raw_size": round(raw_sizes["NQ"], 6),
            "YM_raw_size": round(raw_sizes["YM"], 6),
            "ES_capped_size": round(capped_sizes["ES"], 6),
            "NQ_capped_size": round(capped_sizes["NQ"], 6),
            "YM_capped_size": round(capped_sizes["YM"], 6),
            "ES_actual_pos": round(positions["ES"], 6),
            "NQ_actual_pos": round(positions["NQ"], 6),
            "YM_actual_pos": round(positions["YM"], 6),
            "ES_notional_$": round(abs(positions["ES"]) * equity, 2),
            "NQ_notional_$": round(abs(positions["NQ"]) * equity, 2),
            "YM_notional_$": round(abs(positions["YM"]) * equity, 2),
            "ES_tf_score": tfs_map["ES"],
            "NQ_tf_score": tfs_map["NQ"],
            "YM_tf_score": tfs_map["YM"],
            "total_notional": round(notional_total, 6),
            "margin_util_pct": round(margin_util * 100, 4),
            "roll20_corr_ES_NQ": round(corr_en, 4) if not math.isnan(corr_en) else "",
            "roll20_corr_ES_YM": round(corr_ey, 4) if not math.isnan(corr_ey) else "",
            "execution_this_bar": int(executed_this_bar),
        }
        log.append(row)

    return log


# ── CSV output ────────────────────────────────────────────────────────────────
def write_csv(log):
    path = os.path.join(RESULTS, "size_replay.csv")
    fields = list(log[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(log)
    return path


# ── Charts ────────────────────────────────────────────────────────────────────
def make_chart(log, spike_at):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        bars   = [r["bar_idx"]         for r in log]
        equity = np.array([r["equity"] for r in log])
        peak   = np.array([r["peak_equity"] for r in log])
        dd     = np.array([r["drawdown_pct"] for r in log])

        es_pos = np.array([r["ES_actual_pos"] for r in log])
        nq_pos = np.array([r["NQ_actual_pos"] for r in log])
        ym_pos = np.array([r["YM_actual_pos"] for r in log])

        es_dv  = np.array([r["ES_daily_vol_pct"] for r in log])
        nq_dv  = np.array([r["NQ_daily_vol_pct"] for r in log])
        ym_dv  = np.array([r["YM_daily_vol_pct"] for r in log])

        corr_en = []
        corr_ey = []
        corr_bars = []
        for r in log:
            v = r["roll20_corr_ES_NQ"]
            if v != "":
                corr_en.append(float(v)); corr_ey.append(float(r["roll20_corr_ES_YM"])); corr_bars.append(r["bar_idx"])

        notional = np.array([r["total_notional"] for r in log])

        fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
        fig.suptitle("LARSA v11 — Size Replay (4000 bars, vol spike @ bar 2000)",
                     fontsize=13, fontweight="bold")

        # Panel 1: Equity curve + drawdown shading
        ax = axes[0]
        ax.plot(bars, equity / 1e6, color="#4c78a8", lw=1.2, label="Equity ($M)")
        ax.plot(bars, peak / 1e6, color="#888", lw=0.8, ls="--", label="Peak")
        ax.fill_between(bars, equity / 1e6, peak / 1e6, alpha=0.25, color="red", label="Drawdown")
        if spike_at is not None:
            ax.axvline(spike_at, color="orange", ls=":", lw=1.5, label=f"Vol spike @ {spike_at}")
        ax.set_ylabel("Equity ($M)")
        ax.set_title("Equity Curve with Drawdown Shading")
        ax.legend(fontsize=8, loc="upper left")

        # Panel 2: Position sizes
        ax = axes[1]
        ax.plot(bars, es_pos, color="#4c78a8", lw=0.9, label="ES")
        ax.plot(bars, nq_pos, color="#f58518", lw=0.9, label="NQ")
        ax.plot(bars, ym_pos, color="#54a24b", lw=0.9, label="YM")
        ax.axhline(0, color="black", lw=0.5)
        if spike_at is not None:
            ax.axvline(spike_at, color="orange", ls=":", lw=1.5)
        ax.set_ylabel("Position (fraction of equity)")
        ax.set_title("ES/NQ/YM Actual Position Sizes")
        ax.legend(fontsize=8)

        # Panel 3: ATR daily vol %
        ax = axes[2]
        ax.plot(bars, es_dv, color="#4c78a8", lw=0.9, label="ES")
        ax.plot(bars, nq_dv, color="#f58518", lw=0.9, label="NQ")
        ax.plot(bars, ym_dv, color="#54a24b", lw=0.9, label="YM")
        ax.axhline(DAILY_VOL_THRESH * 100, color="red", ls="--", lw=1.2, label="1.3%/day threshold")
        if spike_at is not None:
            ax.axvline(spike_at, color="orange", ls=":", lw=1.5)
        ax.set_ylabel("ATR daily vol (%)")
        ax.set_title("ATR-Derived Daily Volatility vs Threshold")
        ax.legend(fontsize=8)

        # Panel 4: Rolling 20-bar ES-NQ correlation
        ax = axes[3]
        ax.plot(corr_bars, corr_en, color="#4c78a8", lw=1.0, label="ES-NQ")
        ax.plot(corr_bars, corr_ey, color="#e45756", lw=1.0, label="ES-YM")
        ax.axhline(BASE_CORR, color="gray", ls="--", lw=1.0, label=f"baseline {BASE_CORR}")
        if spike_at is not None:
            ax.axvline(spike_at, color="orange", ls=":", lw=1.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Correlation")
        ax.set_title("Rolling 20-Bar Price Correlation")
        ax.legend(fontsize=8)

        # Panel 5: Total notional exposure
        ax = axes[4]
        ax.plot(bars, notional, color="#79706e", lw=1.0, label="Total notional")
        ax.axhline(1.0, color="red", ls="--", lw=1.5, label="1.0× cap")
        if spike_at is not None:
            ax.axvline(spike_at, color="orange", ls=":", lw=1.5, label=f"Vol spike")
        ax.set_ylabel("Notional (× equity)")
        ax.set_xlabel("Bar")
        ax.set_title("Total Notional Exposure as % of Equity")
        ax.legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(RESULTS, "size_replay.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        return path
    except ImportError:
        return None


# ── Terminal summary ──────────────────────────────────────────────────────────
def terminal_summary(log):
    equities = np.array([r["equity"] for r in log])
    dds      = np.array([r["drawdown_pct"] for r in log])
    notional = np.array([r["total_notional"] for r in log])
    executions = sum(r["execution_this_bar"] for r in log)

    es_dv = np.array([r["ES_daily_vol_pct"] for r in log])
    nq_dv = np.array([r["NQ_daily_vol_pct"] for r in log])
    ym_dv = np.array([r["YM_daily_vol_pct"] for r in log])

    high_vol_bars_es = int(np.sum(es_dv > DAILY_VOL_THRESH * 100))
    high_vol_bars_nq = int(np.sum(nq_dv > DAILY_VOL_THRESH * 100))
    high_vol_bars_ym = int(np.sum(ym_dv > DAILY_VOL_THRESH * 100))

    print(f"\n{'═'*65}")
    print("  SIZE REPLAY — Terminal Summary")
    print(f"{'═'*65}")
    print(f"  Bars simulated        : {len(log):,}")
    print(f"  Start equity          : ${equities[0]:,.0f}")
    print(f"  End equity            : ${equities[-1]:,.0f}  ({(equities[-1]/equities[0]-1)*100:+.1f}%)")
    print(f"  Peak equity           : ${np.max(equities):,.0f}")
    print(f"  Max drawdown          : {np.max(dds):.2f}%")
    print(f"  Execution bars        : {executions:,}  ({executions/len(log)*100:.1f}% of bars)")
    print()
    print(f"  Notional exposure")
    print(f"    Mean                : {np.mean(notional):.3f}×")
    print(f"    Median              : {np.median(notional):.3f}×")
    print(f"    Max                 : {np.max(notional):.3f}×")
    print(f"    % bars > 1.0×       : {np.mean(notional>1.0)*100:.1f}%")
    print()
    print(f"  Bars where daily vol > {DAILY_VOL_THRESH*100:.1f}%/day threshold")
    print(f"    ES: {high_vol_bars_es:,}  NQ: {high_vol_bars_nq:,}  YM: {high_vol_bars_ym:,}")
    print()

    # Vol spike region
    spike_region = [r for r in log if 1990 <= r["bar_idx"] <= 2030]
    if spike_region:
        peak_pos_spike = max(abs(r["ES_actual_pos"]) for r in spike_region)
        print(f"  Vol spike region (bars 1990-2030)")
        print(f"    Peak |ES pos| during spike : {peak_pos_spike:.4f}×")
        spike_notional = [r["total_notional"] for r in spike_region]
        print(f"    Peak total notional        : {max(spike_notional):.3f}×")
        print(f"    Min  total notional        : {min(spike_notional):.3f}×")
    print(f"{'═'*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LARSA v11 size replay")
    parser.add_argument("--n-bars",   type=int, default=4000)
    parser.add_argument("--spike-at", type=int, default=2000)
    args = parser.parse_args()

    print(f"\n{'═'*65}")
    print(f"  LARSA v11  —  Size Replay  ({args.n_bars} bars, spike @ {args.spike_at})")
    print(f"{'═'*65}")

    log = run_replay(n_bars=args.n_bars, spike_at=args.spike_at)

    csv_path = write_csv(log)
    print(f"  Saved → {csv_path}")

    png_path = make_chart(log, args.spike_at)
    if png_path:
        print(f"  Saved → {png_path}")
    else:
        print("  matplotlib not available, skipping chart.")

    terminal_summary(log)


if __name__ == "__main__":
    main()
