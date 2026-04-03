import sys
sys.stdout.reconfigure(encoding='utf-8')

import math, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Constants ────────────────────────────────────────────────────────────────
CF = {"15m":{"ES":0.0003,"NQ":0.0004,"YM":0.00025},"1h":{"ES":0.001,"NQ":0.0012,"YM":0.0008},"1d":{"ES":0.005,"NQ":0.006,"YM":0.004}}
TF_CAP = {7:0.65,6:0.55,5:0.45,4:0.35,3:0.30,2:0.25,1:0.15,0:0.0}
PORTFOLIO_DAILY_RISK = 0.01; CORR_FACTOR = math.sqrt(3+6*0.90)
PER_INST_RISK = PORTFOLIO_DAILY_RISK / CORR_FACTOR

# ── BHI + MRI + Broker ───────────────────────────────────────────────────────
class BHI:
    def __init__(self,sym,cf,res):
        self.sym=sym;self.cf=cf;self.res=res;self.bh_mass=0.0;self.bh_active=False
        self.bh_dir=0;self.ctl=0;self.bc=0;self.prices=[]
        self._wu={"15m":400,"1h":120,"1d":30}.get(res,120)
    def update(self,c):
        self.bc+=1;self.prices.append(c)
        if len(self.prices)<2: return
        beta=abs(c-self.prices[-2])/(self.prices[-2]+1e-9)/(self.cf+1e-9)
        was=self.bh_active
        if beta<1.0: self.ctl+=1;self.bh_mass=self.bh_mass*0.97+0.03*min(2.0,1+self.ctl*0.1)
        else: self.ctl=0;self.bh_mass*=0.95
        self.bh_active=(self.bh_mass>(1.5 if not was else 1.0)) and self.ctl>=3
        if not was and self.bh_active:
            lb=min(20,len(self.prices)-1);self.bh_dir=1 if c>self.prices[-1-lb] else -1
        elif was and not self.bh_active: self.bh_dir=0
        if self.bc<self._wu: self.bh_active=False;self.bh_dir=0
    def direction(self):
        if self.bh_dir: return self.bh_dir
        if len(self.prices)>=5: return 1 if self.prices[-1]>self.prices[-5] else -1
        return 0

class MRI:
    def __init__(self,sym):
        self.sym=sym;self.i15=BHI(sym,CF["15m"][sym],"15m");self.i1h=BHI(sym,CF["1h"][sym],"1h")
        self.i1d=BHI(sym,CF["1d"][sym],"1d");self._atr_vals=[];self._atr=None
        self.last_target=0.0;self.bars_held=0;self.pos_floor=0.0
    def tf_score(self): return 4*self.i1d.bh_active+2*self.i1h.bh_active+self.i15.bh_active
    def direction(self):
        if self.i1d.bh_active: return self.i1d.direction()
        if self.i1h.bh_active: return self.i1h.direction()
        if self.i15.bh_active: return self.i15.direction()
        return 0
    def update(self,o,h,l,c,pc):
        for sub in [o,(o+h)/2,(l+c)/2,c]: self.i15.update(sub)
        self.i1h.update(c);tr=max(h-l,abs(h-pc),abs(l-pc));self._atr_vals.append(tr)
        if len(self._atr_vals)>=14:
            self._atr=sum(self._atr_vals[-14:])/14 if self._atr is None else (self._atr*13+tr)/14
        if self.i1h.bc%6==0: self.i1d.update(c)
    @property
    def atr(self): return self._atr or 0.0

def run_sim(bars, equity, version, pdr_override=None, fee_mult=1.0, slip=0.0, delay=0, reject_rate=0.0, rng_seed=None):
    import random
    rng = random.Random(rng_seed)
    insts={s:MRI(s) for s in ["ES","NQ","YM"]}
    eq=equity; peak=equity; curve=[equity]; fees=0.0; trades=0; margin_calls=0
    positions={}; prev_prices={}
    pdr = pdr_override or PORTFOLIO_DAILY_RISK
    cf = CORR_FACTOR; pir = pdr/cf
    syms=["ES","NQ","YM"]; n=min(len(bars[s]) for s in syms)
    pc={s:bars[s][0]["c"] for s in syms}
    target_buffer={}
    # per-instrument equity tracking
    inst_pnl={s:[] for s in syms}
    inst_positions={s:0.0 for s in syms}
    atr_history=[]
    tf_score_history=[]
    for i in range(n):
        pr={};bars_i={s:bars[s][i] for s in syms}
        for s in syms:
            b=bars_i[s]; insts[s].update(b["o"],b["h"],b["l"],b["c"],pc[s])
            pc[s]=b["c"]; pr[s]=b["c"]
        step_pnl={s:0.0 for s in syms}
        if prev_prices:
            for s,pos in inst_positions.items():
                if abs(pos)<0.001: continue
                pp=prev_prices.get(s,0); cp=pr.get(s,pp)
                if pp>0:
                    pnl=pos*(cp-pp)/pp*eq
                    step_pnl[s]+=pnl
                    eq+=pnl
        prev_prices=dict(pr)
        for s in syms: inst_pnl[s].append(step_pnl[s])
        avg_atr = np.mean([insts[s].atr / pr[s] for s in syms if pr[s]>0 and insts[s].atr>0]) if any(pr[s]>0 for s in syms) else 0.0
        atr_history.append(avg_atr)
        tf_score_history.append({s: insts[s].tf_score() for s in syms})
        raw={}
        for s,inst in insts.items():
            tfs=inst.tf_score(); ceil=TF_CAP[tfs]
            if tfs==1 and abs(inst.last_target)<0.01: ceil=0.0
            if ceil==0: raw[s]=0.0; continue
            d=inst.direction(); a=inst.atr; p=pr[s]
            if version=="v9": tgt=ceil*d
            elif version=="v10":
                tgt=(min(0.01/((a/p)*math.sqrt(6.5)+1e-9),ceil)*d if a>0 and p>0 else ceil*d)
            else:
                tgt=(min(pir/((a/p)*math.sqrt(6.5)+1e-9),ceil)*d if a>0 and p>0 else ceil*d)
            if abs(inst.last_target)>0.02: inst.bars_held+=1
            is_rev=(not abs(inst.last_target)<0.001 and not abs(tgt)<0.001 and (tgt*inst.last_target<0))
            if is_rev and inst.bars_held<4: tgt=inst.last_target
            raw[s]=tgt
        if delay>0:
            execute = target_buffer.get(i-delay, raw)
        else:
            execute = raw
        target_buffer[i] = raw
        total=sum(abs(v) for v in execute.values()); scale=1/total if total>1 else 1
        for s,inst in insts.items():
            tgt=float(execute[s]*scale); old=inst.last_target
            if abs(tgt-old)<0.02: continue
            if reject_rate>0 and rng.random()<reject_rate: continue
            p2=pr[s]
            if slip>0: p2*=(1+slip*(1 if tgt>old else -1)*(rng.random()*0.5+0.5))
            fee=max(1,int(abs(tgt-old)*eq/(p2*50+1e-9)))*4.0*fee_mult
            eq-=fee; fees+=fee; trades+=1
            if abs(tgt)<0.001: inst.bars_held=0
            elif tgt*old<0: inst.bars_held=0
            inst.last_target=tgt; inst_positions[s]=tgt; positions[s]=tgt
        curve.append(max(eq,0))
        if eq>peak: peak=eq
        if eq<peak*0.005 and eq>0:
            margin_calls+=1; positions={}; inst_positions={s:0.0 for s in syms}
            for inst in insts.values(): inst.last_target=0.0; inst.bars_held=0
    arr=np.array(curve)
    rets=np.diff(arr)/(arr[:-1]+1e-9)
    sh=rets.mean()/(rets.std()+1e-9)*math.sqrt(252*6.5)
    pk2=arr[0]; mx=0.0
    for v in arr:
        pk2=max(pk2,v); mx=max(mx,(pk2-v)/(pk2+1e-9))
    return {"final":arr[-1],"peak":peak,"ret%":(arr[-1]-arr[0])/arr[0]*100,
            "peak_ret%":(peak-arr[0])/arr[0]*100,"max_dd%":mx*100,
            "sharpe":sh,"trades":trades,"fees":fees,"margin_calls":margin_calls,
            "curve":curve, "inst_pnl":inst_pnl, "atr_history":atr_history,
            "tf_score_history":tf_score_history}

def gen_world(n=3000,seed=42,vol_mult=1.0,drift=0.0003,corr=0.90):
    rng=np.random.default_rng(seed)
    vols=np.array([0.15,0.20,0.14])/math.sqrt(252*6.5)
    C=np.array([[1,corr,corr],[corr,1,corr],[corr,corr,1]],dtype=float)
    L=np.linalg.cholesky(np.outer(vols,vols)*C)
    ps={"ES":[4000.],"NQ":[14000.],"YM":[33000.]}
    for i in range(n):
        vm=vol_mult if not callable(vol_mult) else vol_mult(i)
        z=L@rng.standard_normal(3)
        for j,s in enumerate(["ES","NQ","YM"]):
            ret=drift+vm*vols[j]*z[j]*math.sqrt(252*6.5)
            ps[s].append(ps[s][-1]*(1+ret))
    result={s:[] for s in ["ES","NQ","YM"]}
    for s in ["ES","NQ","YM"]:
        v=ps[s]
        for i in range(1,len(v)):
            hi=max(v[i],v[i-1])*(1+abs(rng.normal(0,.002)))
            lo=min(v[i],v[i-1])*(1-abs(rng.normal(0,.002)))
            result[s].append({"o":v[i-1],"h":hi,"l":lo,"c":v[i]})
    return result

# ── Drawdown decomposition ────────────────────────────────────────────────────
def find_drawdowns(curve, threshold_pct=3.0):
    """Find all drawdowns exceeding threshold_pct."""
    arr = np.array(curve)
    dds = []
    peak_val = arr[0]; peak_idx = 0
    in_dd = False; dd_start = 0; trough_val = arr[0]; trough_idx = 0

    for i, v in enumerate(arr):
        if v > peak_val:
            if in_dd:
                dd_pct = (peak_val - trough_val) / peak_val * 100
                if dd_pct >= threshold_pct:
                    # find recovery
                    rec_idx = None
                    for j in range(trough_idx, len(arr)):
                        if arr[j] >= peak_val:
                            rec_idx = j
                            break
                    dds.append({
                        "peak_idx": peak_idx, "trough_idx": trough_idx,
                        "start_idx": dd_start, "rec_idx": rec_idx,
                        "peak_equity": float(peak_val), "trough_equity": float(trough_val),
                        "dd_pct": float(dd_pct),
                        "duration": trough_idx - peak_idx,
                        "recovery_bars": (rec_idx - trough_idx) if rec_idx else None,
                        "recovered": rec_idx is not None,
                    })
                in_dd = False
            peak_val = v; peak_idx = i; trough_val = v; trough_idx = i
        else:
            if not in_dd:
                in_dd = True; dd_start = i; trough_val = v; trough_idx = i
            if v < trough_val:
                trough_val = v; trough_idx = i

    # Handle open drawdown at end
    if in_dd:
        dd_pct = (peak_val - trough_val) / peak_val * 100
        if dd_pct >= threshold_pct:
            dds.append({
                "peak_idx": peak_idx, "trough_idx": trough_idx,
                "start_idx": dd_start, "rec_idx": None,
                "peak_equity": float(peak_val), "trough_equity": float(trough_val),
                "dd_pct": float(dd_pct),
                "duration": trough_idx - peak_idx,
                "recovery_bars": None,
                "recovered": False,
            })
    return dds

def decompose_drawdown(dd, inst_pnl, atr_history):
    """Add per-instrument P&L breakdown and ATR elevation to a drawdown record."""
    syms = ["ES","NQ","YM"]
    s_idx = dd["peak_idx"]
    e_idx = dd["trough_idx"]
    contrib = {}
    for s in syms:
        pnl_slice = inst_pnl[s][s_idx:e_idx]
        contrib[s] = float(sum(pnl_slice))
    total_loss = sum(contrib.values())

    # ATR: was it elevated?
    atr_slice = atr_history[s_idx:e_idx] if e_idx > s_idx else [0.0]
    baseline_atr = np.mean(atr_history[:s_idx]) if s_idx > 10 else np.mean(atr_history[:10] or [0.001])
    dd_atr_mean = np.mean(atr_slice) if atr_slice else 0.0
    atr_elevated = dd_atr_mean > baseline_atr * 1.3

    worst_sym = min(contrib, key=lambda x: contrib[x])
    dd["inst_contrib"] = contrib
    dd["total_pnl_loss"] = float(total_loss)
    dd["worst_instrument"] = worst_sym
    dd["atr_elevated"] = bool(atr_elevated)
    dd["avg_atr_during"] = float(dd_atr_mean)
    dd["baseline_atr"] = float(baseline_atr)
    return dd

def avg_recovery_time(curve, min_dd_pct=1.0):
    """Average bars to recover from any drawdown >= min_dd_pct."""
    arr = np.array(curve)
    times = []
    peak = arr[0]; i = 0
    while i < len(arr):
        if arr[i] >= peak:
            peak = arr[i]; i += 1; continue
        dd = (peak - arr[i]) / peak * 100
        if dd >= min_dd_pct:
            for j in range(i, len(arr)):
                if arr[j] >= peak:
                    times.append(j - i)
                    i = j
                    break
            else:
                i += 1
        else:
            i += 1
    return float(np.mean(times)) if times else 0.0

def build_monthly_heatmap(curve, n_bars=3000):
    """Return 36-month P&L matrix (3 years × 12 months)."""
    arr = np.array(curve)
    bars_per_month = n_bars // 36
    matrix = np.zeros((3, 12))
    for y in range(3):
        for m in range(12):
            s = (y*12+m) * bars_per_month
            e = min(s + bars_per_month, len(arr)-1)
            if e > s and arr[s] > 0:
                matrix[y, m] = (arr[e] - arr[s]) / arr[s] * 100
    return matrix

def underwater_fraction(curve):
    """Fraction of bars spent below peak."""
    arr = np.array(curve)
    peak = arr[0]; uw = 0
    for v in arr:
        peak = max(peak, v)
        if v < peak: uw += 1
    return uw / len(arr)

def main():
    os.makedirs("results", exist_ok=True)
    EQUITY = 100_000.0
    N_BARS = 3000

    # World with vol spike at bar 1500
    def vol_spike(i):
        return 3.0 if 1490 <= i <= 1550 else 1.0

    bars_v11 = gen_world(n=N_BARS, seed=42, vol_mult=vol_spike)
    bars_v9  = gen_world(n=N_BARS, seed=42, vol_mult=vol_spike)
    bars_v10 = gen_world(n=N_BARS, seed=42, vol_mult=vol_spike)

    print("\nRunning v11 on vol-spike world (spike at bar 1500)...")
    r11 = run_sim(bars_v11, EQUITY, "v11", rng_seed=42)
    r9  = run_sim(bars_v9,  EQUITY, "v9",  rng_seed=42)
    r10 = run_sim(bars_v10, EQUITY, "v10", rng_seed=42)

    curve   = r11["curve"]
    ipnl    = r11["inst_pnl"]
    atr_h   = r11["atr_history"]

    # Find drawdowns > 3%
    dds = find_drawdowns(curve, threshold_pct=3.0)
    for dd in dds:
        decompose_drawdown(dd, ipnl, atr_h)

    # v9/v10 comparison on same drawdown periods
    for dd in dds:
        s = dd["peak_idx"]; e = dd["trough_idx"]
        v9_loss  = sum(sum(r9["inst_pnl"][sym][s:e])  for sym in ["ES","NQ","YM"])
        v10_loss = sum(sum(r10["inst_pnl"][sym][s:e]) for sym in ["ES","NQ","YM"])
        dd["v9_pnl_during"]  = float(v9_loss)
        dd["v10_pnl_during"] = float(v10_loss)
        dd["v11_pnl_during"] = float(dd["total_pnl_loss"])

    heatmap = build_monthly_heatmap(curve, n_bars=N_BARS)
    uw_frac  = underwater_fraction(curve)
    avg_rec  = avg_recovery_time(curve, min_dd_pct=1.0)

    # ── Terminal report ───────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  DRAWDOWN DECOMPOSITION REPORT — LARSA v11")
    print(f"  World: 3000-bar synthetic, vol spike ×3 at bar 1500")
    print(f"{'═'*70}")
    print(f"  v11 final return:  {r11['ret%']:+.1f}%   Sharpe: {r11['sharpe']:.2f}   Max DD: {r11['max_dd%']:.1f}%")
    print(f"  v9  final return:  {r9['ret%']:+.1f}%   Sharpe: {r9['sharpe']:.2f}   Max DD: {r9['max_dd%']:.1f}%")
    print(f"  v10 final return:  {r10['ret%']:+.1f}%   Sharpe: {r10['sharpe']:.2f}   Max DD: {r10['max_dd%']:.1f}%")
    print(f"\n  Time underwater: {uw_frac*100:.1f}% of bars")
    print(f"  Avg recovery (from 1% DD): {avg_rec:.0f} bars")
    print(f"\n  Drawdowns > 3% found: {len(dds)}")
    print(f"{'─'*70}")

    for i, dd in enumerate(dds, 1):
        rec_str = f"{dd['recovery_bars']} bars" if dd["recovered"] else "not recovered"
        print(f"\n  DD #{i}: {dd['dd_pct']:.1f}% | bars {dd['peak_idx']}→{dd['trough_idx']} "
              f"(dur {dd['duration']} bars) | recovery: {rec_str}")
        print(f"    Peak equity: ${dd['peak_equity']:,.0f} → Trough: ${dd['trough_equity']:,.0f}")
        print(f"    Worst instrument: {dd['worst_instrument']}")
        print(f"    Instrument P&L: ES={dd['inst_contrib']['ES']:+,.0f}  "
              f"NQ={dd['inst_contrib']['NQ']:+,.0f}  YM={dd['inst_contrib']['YM']:+,.0f}")
        atr_flag = " [ATR ELEVATED — vol spike period]" if dd["atr_elevated"] else ""
        print(f"    ATR during: {dd['avg_atr_during']:.4f} vs baseline {dd['baseline_atr']:.4f}{atr_flag}")
        print(f"    Version comparison (P&L during drawdown period):")
        print(f"      v9:  {dd['v9_pnl_during']:+,.0f}   v10: {dd['v10_pnl_during']:+,.0f}   v11: {dd['v11_pnl_during']:+,.0f}")
        # Classify cause
        if dd["atr_elevated"]:
            cause = "VOL SPIKE — ATR elevated, sizing should have been reduced"
        elif abs(dd["inst_contrib"][dd["worst_instrument"]]) > abs(dd["total_pnl_loss"]) * 0.6:
            cause = f"CONCENTRATION — {dd['worst_instrument']} drove >60% of loss"
        else:
            cause = "SIGNAL FAILURE — positions held against price move"
        print(f"    Diagnosis: {cause}")

    print(f"\n{'═'*70}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "v11": {"ret%": r11["ret%"], "sharpe": r11["sharpe"], "max_dd%": r11["max_dd%"]},
        "v9":  {"ret%": r9["ret%"],  "sharpe": r9["sharpe"],  "max_dd%": r9["max_dd%"]},
        "v10": {"ret%": r10["ret%"], "sharpe": r10["sharpe"], "max_dd%": r10["max_dd%"]},
        "drawdowns": [{k:v for k,v in dd.items() if k not in ("inst_contrib",)} |
                      {"inst_contrib": dd["inst_contrib"]} for dd in dds],
        "underwater_fraction": uw_frac,
        "avg_recovery_1pct": avg_rec,
        "heatmap_3y_12m": heatmap.tolist(),
    }
    with open("results/dd_decomp.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/dd_decomp.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LARSA v11 — Drawdown Decomposition", fontsize=14, fontweight="bold")

    # Panel 1: Equity curve with DD regions shaded
    ax = axes[0, 0]
    ax.plot(curve, color="steelblue", linewidth=1.2, label="v11 equity")
    ax.axvline(1500, color="orange", linestyle="--", alpha=0.7, label="Vol spike")
    for dd in dds:
        ax.axvspan(dd["peak_idx"], dd["trough_idx"], alpha=0.15, color="red")
    ax.set_title("Equity Curve (red bands = DDs > 3%)")
    ax.set_xlabel("Bar"); ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Underwater curve
    ax = axes[0, 1]
    arr = np.array(curve); peak_arr = np.maximum.accumulate(arr)
    uw_arr = (peak_arr - arr) / (peak_arr + 1e-9) * 100
    ax.fill_between(range(len(uw_arr)), uw_arr, color="tomato", alpha=0.6)
    ax.set_title(f"Underwater Curve ({uw_frac*100:.1f}% of time below peak)")
    ax.set_xlabel("Bar"); ax.set_ylabel("% Below Peak")
    ax.grid(True, alpha=0.3)

    # Panel 3: Monthly heatmap
    ax = axes[1, 0]
    vmax = max(abs(heatmap.min()), abs(heatmap.max()), 1.0)
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(["Y1","Y2","Y3"])
    ax.set_xticks(range(12))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_title("Monthly P&L Heatmap (% return)")
    for y in range(3):
        for m in range(12):
            ax.text(m, y, f"{heatmap[y,m]:.1f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(heatmap[y,m]) < vmax*0.7 else "white")
    plt.colorbar(im, ax=ax, label="Return %")

    # Panel 4: Instrument P&L contribution during DDs
    ax = axes[1, 1]
    if dds:
        x = range(len(dds))
        es_c = [dd["inst_contrib"]["ES"] for dd in dds]
        nq_c = [dd["inst_contrib"]["NQ"] for dd in dds]
        ym_c = [dd["inst_contrib"]["YM"] for dd in dds]
        w = 0.25
        ax.bar([i-w for i in x], es_c, w, label="ES", color="steelblue")
        ax.bar(x,             nq_c, w, label="NQ", color="tomato")
        ax.bar([i+w for i in x], ym_c, w, label="YM", color="forestgreen")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels([f"DD#{i+1}" for i in x])
        ax.set_title("Instrument P&L During Each Drawdown")
        ax.set_ylabel("P&L ($)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No drawdowns > 3% found", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/dd_decomp.png", dpi=150, bbox_inches="tight")
    print("Saved: results/dd_decomp.png")
    plt.close()

if __name__ == "__main__":
    main()
