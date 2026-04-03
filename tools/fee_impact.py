import sys
sys.stdout.reconfigure(encoding='utf-8')

import math, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    for i in range(n):
        pr={};bars_i={s:bars[s][i] for s in syms}
        for s in syms:
            b=bars_i[s]; insts[s].update(b["o"],b["h"],b["l"],b["c"],pc[s])
            pc[s]=b["c"]; pr[s]=b["c"]
        if prev_prices:
            for s,pos in positions.items():
                if abs(pos)<0.001: continue
                pp=prev_prices.get(s,0); cp=pr.get(s,pp)
                if pp>0: eq+=pos*(cp-pp)/pp*eq
        prev_prices=dict(pr)
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
            inst.last_target=tgt; positions[s]=tgt
        curve.append(max(eq,0))
        if eq>peak: peak=eq
        if eq<peak*0.005 and eq>0:
            margin_calls+=1; positions={}
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
            "curve":curve}

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

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    N_WORLDS = 20
    EQUITY   = 100_000.0
    FEE_MULTS = [0.5, 1.0, 2.0, 3.0, 5.0]
    VERSIONS  = ["v9", "v10", "v11"]

    print(f"\nFee Impact Analysis — LARSA v9/v10/v11  ({N_WORLDS} worlds per scenario)\n")

    # ── Per-version at base fees ──────────────────────────────────────────────
    version_rows = []
    for ver in VERSIONS:
        rets, shs, trd, fees_list = [], [], [], []
        for w in range(N_WORLDS):
            bars = gen_world(n=3000, seed=w*77+3)
            r = run_sim(bars, EQUITY, ver, fee_mult=1.0, rng_seed=w)
            rets.append(r["ret%"]); shs.append(r["sharpe"])
            trd.append(r["trades"]); fees_list.append(r["fees"])
        version_rows.append({
            "version": ver,
            "med_ret": float(np.median(rets)),
            "med_sharpe": float(np.median(shs)),
            "med_trades": float(np.median(trd)),
            "med_fees": float(np.median(fees_list)),
        })

    print("Version comparison at base fees (1×):")
    print(f"  {'Ver':<5}  {'MedRet%':>8}  {'Sharpe':>7}  {'Trades':>8}  {'Fees($)':>10}")
    print("  " + "─"*46)
    for vr in version_rows:
        print(f"  {vr['version']:<5}  {vr['med_ret']:+8.1f}  {vr['med_sharpe']:7.2f}  "
              f"{vr['med_trades']:8.0f}  ${vr['med_fees']:>9,.0f}")

    # Count v11 vs v9 trade frequency
    v11_trades = next(v["med_trades"] for v in version_rows if v["version"]=="v11")
    v9_trades  = next(v["med_trades"] for v in version_rows if v["version"]=="v9")
    print(f"\n  v11 trades {v11_trades/v9_trades:.2f}× more than v9 (ATR-dependent sizes change more)")

    # ── Fee sweep for v11 ─────────────────────────────────────────────────────
    print(f"\nFee sweep for v11:")
    print(f"  {'FeeMult':>8}  {'$/trade':>8}  {'MedRet%':>8}  {'GrossRet%':>10}  {'FeeDrag%':>9}  {'Sharpe':>7}  {'MedDD%':>7}")
    print("  " + "─"*65)

    fee_rows = []
    breakeven_mult = None
    underperform_mult = None

    # Compute buy-and-hold return for comparison
    bh_rets = []
    for w in range(N_WORLDS):
        bars = gen_world(n=3000, seed=w*77+3)
        bh_rets.append((bars["ES"][-1]["c"] - bars["ES"][0]["o"]) / bars["ES"][0]["o"] * 100)
    med_bh = float(np.median(bh_rets))

    for fm in FEE_MULTS:
        rets, gross_rets, shs, flist, ddlist = [], [], [], [], []
        for w in range(N_WORLDS):
            bars = gen_world(n=3000, seed=w*77+3)
            r_net   = run_sim(bars, EQUITY, "v11", fee_mult=fm,  rng_seed=w)
            r_gross = run_sim(bars, EQUITY, "v11", fee_mult=0.0, rng_seed=w)
            rets.append(r_net["ret%"])
            gross_rets.append(r_gross["ret%"])
            shs.append(r_net["sharpe"])
            flist.append(r_net["fees"])
            ddlist.append(r_net["max_dd%"])
        med_ret   = float(np.median(rets))
        med_gross = float(np.median(gross_rets))
        med_sh    = float(np.median(shs))
        med_fees  = float(np.median(flist))
        med_dd    = float(np.median(ddlist))
        fee_drag  = med_gross - med_ret

        if breakeven_mult is None and med_sh <= 0:
            breakeven_mult = fm
        if underperform_mult is None and med_ret < med_bh:
            underperform_mult = fm

        row = {
            "fee_mult": fm, "dollar_per_trade": 4.0*fm,
            "med_ret": med_ret, "med_gross_ret": med_gross,
            "fee_drag": fee_drag, "med_sharpe": med_sh,
            "med_dd": med_dd, "med_fees": med_fees,
        }
        fee_rows.append(row)
        print(f"  {fm:>8.1f}×  ${4*fm:>7.2f}  {med_ret:+8.1f}  {med_gross:+10.1f}  {fee_drag:9.1f}  "
              f"{med_sh:7.2f}  {med_dd:7.1f}%")

    print(f"\n  Buy-and-hold (ES) median return: {med_bh:+.1f}%")
    if breakeven_mult:
        print(f"  Sharpe=0 breakeven: fee_mult = {breakeven_mult:.1f}× (${4*breakeven_mult:.2f}/trade)")
    else:
        print(f"  Sharpe=0 breakeven: not reached within sweep range")
    if underperform_mult:
        print(f"  Underperforms B&H at: fee_mult = {underperform_mult:.1f}× (${4*underperform_mult:.2f}/trade)")
    else:
        print(f"  Underperforms B&H: not within sweep range")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "version_comparison": version_rows,
        "fee_sweep_v11": fee_rows,
        "bh_median_ret": med_bh,
        "sharpe_breakeven_mult": breakeven_mult,
        "bh_underperform_mult": underperform_mult,
        "v11_vs_v9_trade_ratio": float(v11_trades / (v9_trades + 1e-9)),
    }
    with open("results/fee_impact.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/fee_impact.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("LARSA — Fee Impact Analysis", fontsize=14, fontweight="bold")

    fmults = [r["fee_mult"] for r in fee_rows]
    dollars = [r["dollar_per_trade"] for r in fee_rows]

    # Panel 1: Net return vs fee mult
    ax = axes[0, 0]
    ax.plot(fmults, [r["med_ret"]       for r in fee_rows], "o-", color="steelblue", label="Net return")
    ax.plot(fmults, [r["med_gross_ret"] for r in fee_rows], "s--", color="gray",     label="Gross return")
    ax.axhline(med_bh, color="orange", linestyle=":", linewidth=1.5, label=f"B&H ({med_bh:+.1f}%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fee Multiplier"); ax.set_ylabel("Median Return %")
    ax.set_title("Net vs Gross Return by Fee Level")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Fee drag
    ax = axes[0, 1]
    ax.bar(fmults, [r["fee_drag"] for r in fee_rows], width=0.3, color="tomato", alpha=0.8)
    ax.set_xlabel("Fee Multiplier"); ax.set_ylabel("Fee Drag (Return % lost)")
    ax.set_title("Fee Drag by Level")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Sharpe vs fee mult
    ax = axes[1, 0]
    ax.plot(fmults, [r["med_sharpe"] for r in fee_rows], "o-", color="forestgreen")
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Sharpe=0 (breakeven)")
    if breakeven_mult:
        ax.axvline(breakeven_mult, color="red", linestyle=":", alpha=0.7,
                   label=f"Breakeven at {breakeven_mult:.1f}×")
    ax.set_xlabel("Fee Multiplier"); ax.set_ylabel("Median Sharpe")
    ax.set_title("Sharpe Ratio vs Fee Level")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4: Version comparison bar chart
    ax = axes[1, 1]
    vers = [v["version"] for v in version_rows]
    x = np.arange(len(vers))
    w = 0.3
    ax.bar(x - w, [v["med_ret"]    for v in version_rows], w, label="Return %", color="steelblue")
    ax.bar(x,     [v["med_sharpe"] for v in version_rows], w, label="Sharpe",   color="forestgreen")
    ax2 = ax.twinx()
    ax2.bar(x + w, [v["med_trades"] for v in version_rows], w, label="Trades", color="tomato", alpha=0.6)
    ax2.set_ylabel("Trade Count", color="tomato")
    ax.set_xticks(x); ax.set_xticklabels(vers)
    ax.set_title("Version Comparison at Base Fees")
    ax.set_ylabel("Return % / Sharpe")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results/fee_impact.png", dpi=150, bbox_inches="tight")
    print("Saved: results/fee_impact.png")
    plt.close()

if __name__ == "__main__":
    main()
