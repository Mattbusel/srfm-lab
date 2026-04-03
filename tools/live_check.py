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

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = [
    {"name": "Perfect execution", "delay": 0, "slip": 0.0,   "reject_rate": 0.0,  "label": "perfect"},
    {"name": "1-bar delay only",  "delay": 1, "slip": 0.0,   "reject_rate": 0.0,  "label": "delay"},
    {"name": "Slippage only",     "delay": 0, "slip": 0.0003,"reject_rate": 0.0,  "label": "slip"},
    {"name": "Missed fills 5%",   "delay": 0, "slip": 0.0,   "reject_rate": 0.05, "label": "reject"},
    {"name": "Combined",          "delay": 1, "slip": 0.0003,"reject_rate": 0.05, "label": "combined"},
    {"name": "Nightmare",         "delay": 2, "slip": 0.001, "reject_rate": 0.10, "label": "nightmare"},
]

def verdict(ret_degrade_pct):
    if ret_degrade_pct < 25:  return "READY"
    if ret_degrade_pct < 50:  return "CAUTION"
    return "NOT READY"

def verdict_color(v):
    return {"READY": "\033[92m", "CAUTION": "\033[93m", "NOT READY": "\033[91m"}[v]

RESET = "\033[0m"

def main():
    os.makedirs("results", exist_ok=True)
    N_WORLDS = 15
    EQUITY   = 100_000.0

    print(f"\nLive Trading Readiness Simulation — LARSA v11  ({N_WORLDS} worlds per scenario)\n")

    results = {}

    # First run perfect execution baseline
    perf_rets, perf_shs, perf_dds = [], [], []
    for w in range(N_WORLDS):
        bars = gen_world(n=3000, seed=w*53+7)
        r = run_sim(bars, EQUITY, "v11", delay=0, slip=0.0, reject_rate=0.0, rng_seed=w)
        perf_rets.append(r["ret%"]); perf_shs.append(r["sharpe"]); perf_dds.append(r["max_dd%"])
    base_ret = float(np.median(perf_rets))
    base_sh  = float(np.median(perf_shs))
    base_dd  = float(np.median(perf_dds))
    results["perfect"] = {"med_ret": base_ret, "med_sharpe": base_sh, "med_dd": base_dd,
                          "ret_degrade": 0.0, "sh_degrade": 0.0, "dd_increase": 0.0}

    scenario_rows = []
    for sc in SCENARIOS:
        if sc["label"] == "perfect": continue
        rets, shs, dds = [], [], []
        for w in range(N_WORLDS):
            bars = gen_world(n=3000, seed=w*53+7)
            r = run_sim(bars, EQUITY, "v11",
                        delay=sc["delay"], slip=sc["slip"],
                        reject_rate=sc["reject_rate"], rng_seed=w)
            rets.append(r["ret%"]); shs.append(r["sharpe"]); dds.append(r["max_dd%"])
        med_ret = float(np.median(rets))
        med_sh  = float(np.median(shs))
        med_dd  = float(np.median(dds))

        # Degrade %: how much return was lost vs perfect
        if abs(base_ret) > 0.01:
            ret_deg = (base_ret - med_ret) / abs(base_ret) * 100
        else:
            ret_deg = 0.0
        sh_deg   = base_sh - med_sh
        dd_inc   = med_dd - base_dd
        v        = verdict(ret_deg)

        row = {
            "name": sc["name"],
            "label": sc["label"],
            "delay": sc["delay"],
            "slip": sc["slip"],
            "reject_rate": sc["reject_rate"],
            "med_ret": med_ret,
            "med_sharpe": med_sh,
            "med_dd": med_dd,
            "ret_degrade": float(ret_deg),
            "sh_degrade": float(sh_deg),
            "dd_increase": float(dd_inc),
            "verdict": v,
        }
        scenario_rows.append(row)
        results[sc["label"]] = row

    # ── Terminal report ───────────────────────────────────────────────────────
    print(f"  Baseline (perfect execution):  "
          f"ret={base_ret:+.1f}%  Sharpe={base_sh:.2f}  MaxDD={base_dd:.1f}%\n")

    hdr = (f"  {'Scenario':<22}  {'Ret%':>7}  {'Sharpe':>7}  {'MaxDD%':>7}  "
           f"{'RetDeg%':>8}  {'ShDeg':>7}  {'DDInc%':>7}  {'Verdict':<12}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for row in scenario_rows:
        vc = verdict_color(row["verdict"])
        print(f"  {row['name']:<22}  {row['med_ret']:+7.1f}  {row['med_sharpe']:7.2f}  "
              f"{row['med_dd']:7.1f}  {row['ret_degrade']:+8.1f}  {row['sh_degrade']:+7.2f}  "
              f"{row['dd_increase']:+7.1f}  {vc}{row['verdict']:<12}{RESET}")

    print(f"\n  Legend: RetDeg% = % of return lost vs perfect | ShDeg = Sharpe drop | DDInc% = extra drawdown")

    # Overall verdict
    combined = next((r for r in scenario_rows if r["label"] == "combined"), None)
    nightmare = next((r for r in scenario_rows if r["label"] == "nightmare"), None)
    print(f"\n{'═'*60}")
    if combined:
        cv = combined["verdict"]
        print(f"  COMBINED IMPAIRMENTS: {verdict_color(cv)}{cv}{RESET}")
        print(f"    Return degradation: {combined['ret_degrade']:+.1f}%  |  Sharpe drop: {combined['sh_degrade']:+.2f}")
    if nightmare:
        nv = nightmare["verdict"]
        print(f"  NIGHTMARE SCENARIO:  {verdict_color(nv)}{nv}{RESET}")
        print(f"    Return degradation: {nightmare['ret_degrade']:+.1f}%  |  Sharpe drop: {nightmare['sh_degrade']:+.2f}")
    print(f"{'═'*60}\n")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {"baseline": results["perfect"], "scenarios": scenario_rows}
    with open("results/live_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/live_check.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("LARSA v11 — Live Trading Readiness", fontsize=14, fontweight="bold")

    names    = ["Perfect"] + [r["name"] for r in scenario_rows]
    rets_all = [base_ret]  + [r["med_ret"]    for r in scenario_rows]
    shs_all  = [base_sh]   + [r["med_sharpe"] for r in scenario_rows]
    dds_all  = [base_dd]   + [r["med_dd"]     for r in scenario_rows]
    verdicts = ["READY"]   + [r["verdict"]    for r in scenario_rows]

    colors = {"READY": "forestgreen", "CAUTION": "goldenrod", "NOT READY": "tomato"}
    bar_colors = [colors[v] for v in verdicts]
    x = np.arange(len(names))
    short_names = [n.replace(" only","").replace(" fills","") for n in names]

    ax = axes[0]
    ax.bar(x, rets_all, color=bar_colors, alpha=0.85)
    ax.axhline(base_ret, color="steelblue", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Median Return %"); ax.set_title("Return% by Scenario")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(x, shs_all, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Median Sharpe"); ax.set_title("Sharpe by Scenario")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(x, dds_all, color=bar_colors, alpha=0.85)
    ax.axhline(base_dd, color="steelblue", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Median Max DD %"); ax.set_title("Max Drawdown by Scenario")
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[v], label=v) for v in ["READY","CAUTION","NOT READY"]]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("results/live_check.png", dpi=150, bbox_inches="tight")
    print("Saved: results/live_check.png")
    plt.close()

if __name__ == "__main__":
    main()
