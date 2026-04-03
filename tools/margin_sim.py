"""
LARSA v11 — Margin Simulation Tool
Model IB margin requirements for ES/NQ/YM and compute excess liquidity.
Usage:
  python tools/margin_sim.py
  python tools/margin_sim.py --equity 15000000 --daily-loss 0.03
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import math
import json
import os
import argparse

# ── Constants ────────────────────────────────────────────────────────────────
CF = {
    "15m": {"ES": 0.0003, "NQ": 0.0004, "YM": 0.00025},
    "1h":  {"ES": 0.001,  "NQ": 0.0012, "YM": 0.0008},
    "1d":  {"ES": 0.005,  "NQ": 0.006,  "YM": 0.004},
}
TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}
PORTFOLIO_DAILY_RISK = 0.01
CORR_FACTOR = math.sqrt(3 + 6 * 0.90)   # 2.898
PER_INST_RISK = PORTFOLIO_DAILY_RISK / CORR_FACTOR

# IB Margin rates (per contract)
IB_MARGIN = {
    "ES": {"initial": 14_600, "maintenance": 13_200, "multiplier": 50,  "price": 5200.0},
    "NQ": {"initial": 21_500, "maintenance": 19_500, "multiplier": 20,  "price": 18000.0},
    "YM": {"initial": 11_500, "maintenance": 10_500, "multiplier":  5,  "price": 39000.0},
}

# ATR assumptions (% of price) — normal vol
NORMAL_ATR_PCT = {"ES": 0.009, "NQ": 0.011, "YM": 0.010}

# ── Sizing ────────────────────────────────────────────────────────────────────

def v11_fraction(inst: str, atr_pct: float = None, tf_score: int = 5) -> float:
    if atr_pct is None:
        atr_pct = NORMAL_ATR_PCT[inst]
    price = IB_MARGIN[inst]["price"]
    atr   = atr_pct * price
    daily_vol = (atr / price) * math.sqrt(6.5)
    raw  = PER_INST_RISK / daily_vol
    return min(raw, TF_CAP[tf_score])


def v10_fraction(inst: str, atr_pct: float = None) -> float:
    if atr_pct is None:
        atr_pct = NORMAL_ATR_PCT[inst]
    price = IB_MARGIN[inst]["price"]
    atr   = atr_pct * price
    daily_vol = (atr / price) * math.sqrt(6.5)
    return PORTFOLIO_DAILY_RISK / daily_vol


def v9_fraction() -> float:
    return 0.65


# ── Margin calculation ────────────────────────────────────────────────────────

def margin_analysis(equity: float, positions: dict) -> dict:
    """
    positions: {inst: fraction_of_equity}
    Returns margin breakdown per instrument + portfolio totals.
    """
    total_maintenance = 0.0
    detail = {}
    for inst, frac in positions.items():
        m = IB_MARGIN[inst]
        notional_per_contract = m["price"] * m["multiplier"]
        contracts = math.floor(abs(frac) * equity / notional_per_contract)
        maint = contracts * m["maintenance"]
        init  = contracts * m["initial"]
        detail[inst] = {
            "fraction":    frac,
            "contracts":   contracts,
            "notional":    contracts * notional_per_contract,
            "initial_req": init,
            "maint_req":   maint,
        }
        total_maintenance += maint

    excess_liq = equity - total_maintenance
    util_pct   = total_maintenance / equity * 100 if equity > 0 else 0.0
    excess_pct = excess_liq / equity * 100 if equity > 0 else 0.0

    if excess_liq < 0:
        flag = "MARGIN CALL"
    elif excess_pct < 10:
        flag = "CRITICAL"
    elif excess_pct < 20:
        flag = "DANGER"
    else:
        flag = "OK"

    return {
        "equity":          equity,
        "total_maint_req": total_maintenance,
        "excess_liquidity":excess_liq,
        "utilization_pct": util_pct,
        "excess_pct":      excess_pct,
        "flag":            flag,
        "detail":          detail,
    }


def flag_str(flag: str) -> str:
    icons = {"OK": "  OK   ", "DANGER": "DANGER ", "CRITICAL": "CRITCL ", "MARGIN CALL": "MRG_CLL"}
    return icons.get(flag, flag)


# ── Scenario helpers ──────────────────────────────────────────────────────────

def build_positions(version: str, tf_score: int = 5) -> dict:
    """Build a {inst: fraction} dict for a given version."""
    pos = {}
    for inst in ["ES", "NQ", "YM"]:
        if version == "v9":
            pos[inst] = v9_fraction()
        elif version == "v10":
            pos[inst] = v10_fraction(inst)
        else:  # v11
            pos[inst] = v11_fraction(inst, tf_score=tf_score)
    return pos


def print_margin_row(label: str, ma: dict):
    print(f"  {label:<22}  ${ma['equity']:>12,.0f}  "
          f"${ma['total_maint_req']:>12,.0f}  "
          f"${ma['excess_liquidity']:>12,.0f}  "
          f"{ma['excess_pct']:>6.1f}%  "
          f"{flag_str(ma['flag'])}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LARSA v11 — Margin Simulation")
    parser.add_argument("--equity",      type=float, default=15_000_000)
    parser.add_argument("--daily-loss",  type=float, default=0.03)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    output = {}

    # ── Section 1: Equity scaling ($1M → $20M) ───────────────────────────────
    print("=" * 90)
    print("  LARSA v11  —  MARGIN SIMULATION")
    print(f"  IB Rates: ES init={IB_MARGIN['ES']['initial']:,}  NQ init={IB_MARGIN['NQ']['initial']:,}  YM init={IB_MARGIN['YM']['initial']:,}")
    print("=" * 90)
    print()
    print("  SECTION 1 — EQUITY SCALING ($1M → $20M), NORMAL VOL, TF_SCORE=5")
    print()
    hdr = f"  {'Label':<22}  {'Equity':>14}  {'Maint Req':>14}  {'Excess Liq':>14}  {'Excess%':>8}  {'Status'}"
    print(hdr)
    print("  " + "─" * 86)

    scaling_results = {}
    for eq in [1_000_000, 5_000_000, 10_000_000, 15_000_000, 20_000_000]:
        eq_label = f"${eq/1e6:.0f}M"
        scaling_results[eq_label] = {}
        for version in ["v9", "v10", "v11"]:
            pos = build_positions(version)
            ma  = margin_analysis(eq, pos)
            lbl = f"{eq_label} {version}"
            print_margin_row(lbl, ma)
            scaling_results[eq_label][version] = {
                k: v for k, v in ma.items() if k != "detail"
            }
        print()

    output["equity_scaling"] = scaling_results

    # ── Section 2: Detailed contract breakdown at $10M ────────────────────────
    print()
    print("  SECTION 2 — DETAILED BREAKDOWN AT $10M, v11 (TF_SCORE=5)")
    eq10 = 10_000_000
    pos_v11 = build_positions("v11")
    ma10    = margin_analysis(eq10, pos_v11)
    print()
    print(f"  {'Inst':<6} {'Fraction':>10} {'Contracts':>12} {'Notional':>14} {'Maint Req':>12}")
    print("  " + "─" * 56)
    for inst, d in ma10["detail"].items():
        print(f"  {inst:<6} {d['fraction']:>10.4f} {d['contracts']:>12} "
              f"${d['notional']:>12,.0f} ${d['maint_req']:>10,.0f}")
    print(f"  {'TOTAL':<6} {'':>10} {'':>12} {'':>14} ${ma10['total_maint_req']:>10,.0f}")
    print(f"  Excess liquidity: ${ma10['excess_liquidity']:,.0f}  ({ma10['excess_pct']:.1f}%)  [{ma10['flag']}]")

    output["detail_10m_v11"] = {k: v for k, v in ma10.items() if k != "detail"}

    # ── Section 3: Cascade scenario ───────────────────────────────────────────
    cascade_equity = args.equity
    daily_loss_pct = args.daily_loss

    print()
    print(f"  SECTION 3 — CASCADE: Start ${cascade_equity/1e6:.1f}M, -{daily_loss_pct*100:.1f}%/day for 10 days, v9 vs v11")
    print()
    print(f"  {'Day':<5} {'Equity':>14} {'v11 Maint':>12} {'v11 Excess%':>13} {'v11 Flag':<12}  {'v9 Maint':>12} {'v9 Excess%':>12} {'v9 Flag'}")
    print("  " + "─" * 90)

    cascade_results = []
    eq = cascade_equity
    for day in range(11):
        pos_v11 = build_positions("v11")
        pos_v9  = build_positions("v9")
        ma_v11  = margin_analysis(eq, pos_v11)
        ma_v9   = margin_analysis(eq, pos_v9)
        row = {
            "day":         day,
            "equity":      eq,
            "v11_maint":   ma_v11["total_maint_req"],
            "v11_excess%": ma_v11["excess_pct"],
            "v11_flag":    ma_v11["flag"],
            "v9_maint":    ma_v9["total_maint_req"],
            "v9_excess%":  ma_v9["excess_pct"],
            "v9_flag":     ma_v9["flag"],
        }
        cascade_results.append(row)
        print(f"  {day:<5} ${eq:>12,.0f} "
              f"${ma_v11['total_maint_req']:>10,.0f}  "
              f"{ma_v11['excess_pct']:>10.1f}%  "
              f"{flag_str(ma_v11['flag']):<12} "
              f"${ma_v9['total_maint_req']:>10,.0f}  "
              f"{ma_v9['excess_pct']:>10.1f}%  "
              f"{flag_str(ma_v9['flag'])}")
        if eq <= 0:
            break
        eq *= (1 - daily_loss_pct)

    output["cascade"] = cascade_results

    # ── Section 4: Per-instrument position summary ────────────────────────────
    print()
    print("  SECTION 4 — POSITION SIZE COMPARISON (normal vol, TF_SCORE=5)")
    print()
    print(f"  {'Instrument':<12} {'v9 frac':>10} {'v10 frac':>10} {'v11 frac':>10}  {'v9 contracts@$10M':>20} {'v11 contracts@$10M':>20}")
    print("  " + "─" * 86)
    inst_detail = {}
    for inst in ["ES", "NQ", "YM"]:
        fv9  = v9_fraction()
        fv10 = v10_fraction(inst)
        fv11 = v11_fraction(inst)
        m    = IB_MARGIN[inst]
        notional = m["price"] * m["multiplier"]
        c9  = math.floor(fv9  * 10e6 / notional)
        c11 = math.floor(fv11 * 10e6 / notional)
        print(f"  {inst:<12} {fv9:>10.4f} {fv10:>10.4f} {fv11:>10.4f}  {c9:>20} {c11:>20}")
        inst_detail[inst] = {"v9": fv9, "v10": fv10, "v11": fv11, "v9_contracts_10m": c9, "v11_contracts_10m": c11}
    output["instrument_detail"] = inst_detail

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_path = "results/margin_sim.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results written to {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
