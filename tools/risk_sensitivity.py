import sys
sys.stdout.reconfigure(encoding='utf-8')

import math, os, json, argparse
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
    parser = argparse.ArgumentParser(description="LARSA v11 risk sensitivity sweep")
    parser.add_argument("--quick", action="store_true", help="Fewer worlds per PDR (5 instead of 20)")
    args = parser.parse_args()

    N_WORLDS = 5 if args.quick else 20
    N_STEPS = 25
    PDR_VALUES = np.linspace(0.005, 0.030, N_STEPS)
    EQUITY = 100_000.0

    os.makedirs("results", exist_ok=True)

    print(f"\nRisk Sensitivity Sweep — v11  ({N_WORLDS} worlds per PDR, {N_STEPS} steps)")
    print(f"PDR range: {PDR_VALUES[0]:.4f} → {PDR_VALUES[-1]:.4f}\n")

    rows = []
    for idx, pdr in enumerate(PDR_VALUES):
        ret_list, dd_list, sh_list, fee_list = [], [], [], []
        for w in range(N_WORLDS):
            bars = gen_world(n=3000, seed=w*100+idx)
            r = run_sim(bars, EQUITY, "v11", pdr_override=pdr, rng_seed=w)
            ret_list.append(r["ret%"])
            dd_list.append(r["max_dd%"])
            sh_list.append(r["sharpe"])
            fee_list.append(r["fees"])
        ret_arr = np.array(ret_list)
        dd_arr  = np.array(dd_list)
        sh_arr  = np.array(sh_list)
        row = {
            "pdr": float(pdr),
            "med_ret":    float(np.median(ret_arr)),
            "p10_ret":    float(np.percentile(ret_arr, 10)),
            "p90_ret":    float(np.percentile(ret_arr, 90)),
            "med_dd":     float(np.median(dd_arr)),
            "p10_dd":     float(np.percentile(dd_arr, 10)),
            "p90_dd":     float(np.percentile(dd_arr, 90)),
            "med_sharpe": float(np.median(sh_arr)),
            "med_fees":   float(np.median(fee_list)),
        }
        rows.append(row)
        print(f"  PDR={pdr:.4f}  ret={row['med_ret']:+7.1f}%  DD={row['med_dd']:5.1f}%  "
              f"p90DD={row['p90_dd']:5.1f}%  Sharpe={row['med_sharpe']:5.2f}  "
              f"fees=${row['med_fees']:,.0f}")

    # ── Find efficient frontier & ruin breakeven ──────────────────────────────
    best_idx  = max(range(N_STEPS), key=lambda i: rows[i]["med_sharpe"])
    ruin_pdr  = None
    for r in rows:
        if r["p90_dd"] >= 30.0:
            ruin_pdr = r["pdr"]
            break

    print(f"\n{'─'*70}")
    print(f"  Efficient frontier (max Sharpe):  PDR={rows[best_idx]['pdr']:.4f}  "
          f"Sharpe={rows[best_idx]['med_sharpe']:.2f}  DD={rows[best_idx]['med_dd']:.1f}%")
    if ruin_pdr:
        print(f"  Ruin risk breakeven (p90 DD≥30%): PDR={ruin_pdr:.4f}")
    else:
        print(f"  Ruin risk breakeven (p90 DD≥30%): not reached in sweep range")
    print(f"{'─'*70}\n")

    # ── Terminal table ────────────────────────────────────────────────────────
    hdr = f"{'PDR':>7}  {'MedRet%':>8}  {'p10Ret%':>8}  {'p90Ret%':>8}  {'MedDD%':>7}  {'p90DD%':>7}  {'Sharpe':>7}  {'MedFees':>9}"
    print(hdr)
    print("─" * len(hdr))
    for r in rows:
        flag = " <-- optimal" if r["pdr"] == rows[best_idx]["pdr"] else ""
        flag2 = " <-- ruin" if ruin_pdr and abs(r["pdr"] - ruin_pdr) < 1e-9 else ""
        print(f"  {r['pdr']:.4f}  {r['med_ret']:+8.1f}  {r['p10_ret']:+8.1f}  {r['p90_ret']:+8.1f}  "
              f"{r['med_dd']:7.1f}  {r['p90_dd']:7.1f}  {r['med_sharpe']:7.2f}  ${r['med_fees']:>8,.0f}"
              f"{flag}{flag2}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "sweep": rows,
        "efficient_frontier_pdr": rows[best_idx]["pdr"],
        "efficient_frontier_sharpe": rows[best_idx]["med_sharpe"],
        "ruin_breakeven_pdr": ruin_pdr,
        "n_worlds": N_WORLDS,
    }
    with open("results/risk_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/risk_sensitivity.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    pdrs    = [r["pdr"]        for r in rows]
    med_ret = [r["med_ret"]    for r in rows]
    p10_ret = [r["p10_ret"]    for r in rows]
    p90_ret = [r["p90_ret"]    for r in rows]
    med_dd  = [r["med_dd"]     for r in rows]
    p10_dd  = [r["p10_dd"]     for r in rows]
    p90_dd  = [r["p90_dd"]     for r in rows]
    sharpes = [r["med_sharpe"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("LARSA v11 — Risk Sensitivity Sweep", fontsize=14, fontweight="bold")

    # Panel 1: Return% vs PDR
    ax = axes[0]
    ax.fill_between(pdrs, p10_ret, p90_ret, alpha=0.25, color="steelblue", label="p10–p90 band")
    ax.plot(pdrs, med_ret, "o-", color="steelblue", linewidth=2, markersize=4, label="Median return")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Portfolio Daily Risk (PDR)")
    ax.set_ylabel("Return %")
    ax.set_title("Return% vs PDR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Max DD% vs PDR
    ax = axes[1]
    ax.fill_between(pdrs, p10_dd, p90_dd, alpha=0.25, color="tomato", label="p10–p90 band")
    ax.plot(pdrs, med_dd, "o-", color="tomato", linewidth=2, markersize=4, label="Median max DD")
    ax.axhline(30, color="darkred", linestyle="--", linewidth=1.2, label="30% danger line")
    if ruin_pdr:
        ax.axvline(ruin_pdr, color="darkred", linestyle=":", linewidth=1, label=f"Ruin PDR={ruin_pdr:.3f}")
    ax.set_xlabel("Portfolio Daily Risk (PDR)")
    ax.set_ylabel("Max Drawdown %")
    ax.set_title("Max DD% vs PDR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Sharpe vs PDR
    ax = axes[2]
    ax.plot(pdrs, sharpes, "o-", color="forestgreen", linewidth=2, markersize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    opt_pdr = rows[best_idx]["pdr"]
    opt_sh  = rows[best_idx]["med_sharpe"]
    ax.plot(opt_pdr, opt_sh, "*", color="gold", markersize=16, zorder=5,
            label=f"Optimal PDR={opt_pdr:.3f}\nSharpe={opt_sh:.2f}")
    ax.set_xlabel("Portfolio Daily Risk (PDR)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe vs PDR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/risk_sensitivity.png", dpi=150, bbox_inches="tight")
    print("Saved: results/risk_sensitivity.png")
    plt.close()

if __name__ == "__main__":
    main()
