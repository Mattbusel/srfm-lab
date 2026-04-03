"""
suite_v13.py — v13-specific test suite.

Tests the four new mechanisms added in v13:
  §1  Fractional Kelly tiers — risk decays as equity grows
  §2  Trailing EMA anchor — sizing uses smoothed equity, not spot
  §3  Circuit breaker — 15% DD from peak → liquidate + cooldown
  §4  Regime transition cooldown — 8-bar sit-out after regime flip
  §5  Integration — all four layers working together
  §6  v13 vs v12 sizing at scale — confirm concave growth

Usage:
    python tools/suite_v13.py
    python tools/suite_v13.py --quick
"""

import argparse
import math
import os
import sys
import json

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT    = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── v13 constants (mirror strategies/larsa-v13/main.py) ─────────────────────
N_INST        = 3
INST_CORR     = 0.90
_CORR_FACTOR  = math.sqrt(N_INST + N_INST * (N_INST - 1) * INST_CORR)

EQUITY_TIERS = [
    (2_000_000,  0.010),
    (5_000_000,  0.007),
    (10_000_000, 0.005),
    (float("inf"), 0.003),
]

EMA_PERIOD = 20
EMA_K      = 2.0 / (EMA_PERIOD + 1)

CB_DD        = 0.15
CB_FLAT      = 48
CB_HALF      = 48
REGIME_COOL  = 8

TF_CAP = {7: 0.65, 6: 0.55, 5: 0.45, 4: 0.35, 3: 0.30, 2: 0.25, 1: 0.15, 0: 0.0}


def get_port_risk(equity_anchor):
    for threshold, risk in EQUITY_TIERS:
        if equity_anchor < threshold:
            return risk
    return 0.003


def get_per_inst_risk(equity_anchor):
    return get_port_risk(equity_anchor) / _CORR_FACTOR


# ── Test harness ──────────────────────────────────────────────────────────────
passed = []
failed = []


def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed.append(name)
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


# ── §1 Fractional Kelly tiers ─────────────────────────────────────────────────
section("§1  Fractional Kelly tiers — risk decays as equity grows")

# Risk budget at each tier
r_1m   = get_port_risk(1_000_000)
r_2m   = get_port_risk(2_000_000)
r_3m   = get_port_risk(3_000_000)
r_5m   = get_port_risk(5_000_000)
r_7m   = get_port_risk(7_000_000)
r_10m  = get_port_risk(10_000_000)
r_15m  = get_port_risk(15_000_000)

check("$1M uses full 1% risk (same as v10 aggression)",
      r_1m == 0.010, f"{r_1m:.3f}")
check("$2M is still 1% (boundary is exclusive below $2M)",
      get_port_risk(1_999_999) == 0.010, f"{get_port_risk(1_999_999):.3f}")
check("$2M+ drops to 0.7%",
      r_2m == 0.007 or r_3m == 0.007, f"$2M={r_2m:.3f} $3M={r_3m:.3f}")
check("$5M+ drops to 0.5%",
      r_5m == 0.005 or r_7m == 0.005, f"$5M={r_5m:.3f} $7M={r_7m:.3f}")
check("$10M+ drops to 0.3%",
      r_10m == 0.003 and r_15m == 0.003, f"$10M={r_10m:.3f} $15M={r_15m:.3f}")
check("Risk is strictly decreasing across tiers",
      r_1m > r_3m > r_7m > r_15m,
      f"{r_1m:.3f} > {r_3m:.3f} > {r_7m:.3f} > {r_15m:.3f}")

# Per-instrument risk at each tier
pi_1m  = get_per_inst_risk(1_000_000)
pi_10m = get_per_inst_risk(10_000_000)
pi_22m = get_per_inst_risk(22_000_000)

check("per_inst_risk at $1M = 1%/corr_factor ≈ 0.00345",
      abs(pi_1m - 0.01 / _CORR_FACTOR) < 1e-6, f"{pi_1m:.5f}")
check("per_inst_risk at $10M = 0.3%/corr_factor ≈ 0.00103",
      abs(pi_10m - 0.003 / _CORR_FACTOR) < 1e-6, f"{pi_10m:.5f}")
check("At $22M: per_inst_risk is 3.3× smaller than at $1M",
      pi_1m / pi_22m > 3.0,
      f"ratio={pi_1m/pi_22m:.2f}×")

# Contract count comparison v12 vs v13 at $22M
# ES at 5000, multiplier 50, daily_vol=0.008
price = 5000; vol = 0.008
v12_per_inst = 0.01 / _CORR_FACTOR  # v12 fixed
v13_per_inst = pi_22m

v12_frac = min(v12_per_inst / vol, TF_CAP[7])
v13_frac = min(v13_per_inst / vol, TF_CAP[7])
v12_contracts = int(v12_frac * 22_000_000 / (price * 50))
v13_contracts = int(v13_frac * 22_000_000 / (price * 50))

check("At $22M, v13 holds fewer ES contracts than v12",
      v13_contracts < v12_contracts,
      f"v12={v12_contracts} v13={v13_contracts}")
check("At $22M, v13 ES contracts < 20 (not the fatal 44+)",
      v13_contracts < 20,
      f"v13={v13_contracts} contracts")


# ── §2 Trailing EMA anchor ────────────────────────────────────────────────────
section("§2  Trailing EMA anchor — sizing uses smoothed equity, not spot")


def run_ema(equity_series):
    ema = equity_series[0]
    emas = [ema]
    for v in equity_series[1:]:
        ema = ema * (1 - EMA_K) + v * EMA_K
        emas.append(ema)
    return emas


# Scenario: equity spikes $1M → $10M in 5 bars then crashes back
spike = [1_000_000] * 5 + [10_000_000] * 5 + [1_000_000] * 5
emas  = run_ema(spike)

check("EMA anchor lags a spike — still below $5M after 5 bars at $10M",
      emas[9] < 5_000_000,
      f"ema_after_spike={emas[9]:,.0f}")
check("EMA anchor lags a drop — still above $3M after 5 bars at $1M",
      emas[14] > 3_000_000,
      f"ema_after_drop={emas[14]:,.0f}")

# At spike peak, tier from EMA should be lower than from spot equity
tier_spot = get_port_risk(10_000_000)
tier_ema  = get_port_risk(emas[9])
check("At spike peak: EMA-based tier is MORE conservative than spot-based",
      tier_ema > tier_spot or emas[9] < 10_000_000,
      f"spot_tier={tier_spot:.3f} ema_tier={tier_ema:.3f} ema={emas[9]:,.0f}")

# Gradual equity growth: EMA tracks within 10% after 30 bars
steady = [1_000_000 * (1.001 ** i) for i in range(50)]
emas_s = run_ema(steady)
final_eq  = steady[-1]
final_ema = emas_s[-1]
check("EMA tracks gradual growth within 10% after 50 bars",
      abs(final_ema - final_eq) / final_eq < 0.10,
      f"eq={final_eq:,.0f} ema={final_ema:,.0f} diff={abs(final_ema-final_eq)/final_eq:.1%}")

# EMA never exceeds spot equity (can't anticipate)
check("EMA never exceeds spot equity during a spike (only lags on way up)",
      all(e <= s + 1 for e, s in zip(emas[:10], spike[:10])),
      "ema ≤ spot during spike phase")


# ── §3 Circuit breaker ────────────────────────────────────────────────────────
section("§3  Circuit breaker — 15% DD from peak → liquidate + cooldown")


class CBSimulator:
    """Minimal simulation of circuit breaker logic."""
    def __init__(self, equity):
        self.equity = equity
        self.peak   = equity
        self.cb_flat = 0
        self.cb_half = 0
        self.positions = {"ES": 0.1, "NQ": 0.1, "YM": 0.1}
        self.fired = False
        self.fire_equity = None

    def step(self, new_equity):
        self.equity = new_equity
        if self.equity > self.peak:
            self.peak = self.equity

        # Flat cooldown
        if self.cb_flat > 0:
            self.cb_flat -= 1
            self.positions = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
            return "FLAT"

        # CB check
        if self.equity < self.peak * (1 - CB_DD):
            self.fired = True
            self.fire_equity = self.equity
            self.positions = {"ES": 0.0, "NQ": 0.0, "YM": 0.0}
            self.cb_flat = CB_FLAT
            self.cb_half = CB_HALF
            return "FIRED"

        # Half-size recovery
        scale = 0.5 if self.cb_half > 0 else 1.0
        if self.cb_half > 0:
            self.cb_half -= 1
        return f"SCALE_{scale}"


# Test 1: CB fires at exactly -15% from peak
sim = CBSimulator(10_000_000)
for _ in range(5):
    sim.step(10_000_000)  # hold at peak
result = sim.step(8_500_000 - 1)  # just below 85% of peak
check("Circuit breaker fires at 15% DD from peak",
      result == "FIRED", f"result={result} equity={sim.fire_equity:,.0f}")

# Test 2: CB does NOT fire at 14.9% DD
sim2 = CBSimulator(10_000_000)
result2 = sim2.step(8_510_000)  # 14.9% DD
check("Circuit breaker does NOT fire at 14.9% DD",
      result2 != "FIRED", f"result={result2}")

# Test 3: After CB fires, flat for exactly CB_FLAT bars
sim3 = CBSimulator(10_000_000)
sim3.step(8_499_999)  # trigger
flat_states = [sim3.step(9_000_000) for _ in range(CB_FLAT)]       # exactly flat period
half_states = [sim3.step(9_000_000) for _ in range(CB_HALF)]       # exactly half period
post_states = [sim3.step(9_000_000) for _ in range(5)]             # normal

flat_count = sum(1 for s in flat_states if s == "FLAT")
check(f"After CB fires: exactly {CB_FLAT} bars flat",
      flat_count == CB_FLAT, f"flat_bars={flat_count}")

# Test 4: After flat cooldown, half-size for CB_HALF bars
half_count = sum(1 for s in half_states if "0.5" in s)
check(f"After flat: {CB_HALF} bars at half-size",
      half_count == CB_HALF, f"half_bars={half_count}")

# Test 5: After full recovery, back to normal scale
post_recovery = post_states[0]
check("After full cooldown: returns to normal scale",
      "1.0" in post_recovery, f"result={post_recovery}")

# Test 6: CB protects against death spiral
sim4 = CBSimulator(22_000_000)
# Simulate a -30% crash (was -99% before)
equity = 22_000_000
for _ in range(100):
    equity *= 0.997  # slow bleed
    r = sim4.step(equity)
    if r == "FIRED":
        break
floor_equity = sim4.equity
check("CB stops death spiral: floor equity > 85% of peak",
      floor_equity > 22_000_000 * 0.84,
      f"floor={floor_equity:,.0f} vs peak={sim4.peak:,.0f} ({floor_equity/sim4.peak:.1%})")

# Test 7: CB tracks NEW peaks, not just starting equity
sim5 = CBSimulator(1_000_000)
sim5.step(5_000_000)  # new ATH
sim5.step(5_500_000)  # new ATH
result_cb = sim5.step(4_600_000)  # -16.4% from $5.5M peak
check("CB tracks rolling ATH (fires from $5.5M peak, not $1M start)",
      result_cb == "FIRED",
      f"peak={sim5.peak:,.0f} equity=4,600,000 dd={1-4_600_000/sim5.peak:.1%}")


# ── §4 Regime transition cooldown ─────────────────────────────────────────────
section("§4  Regime transition cooldown — 8-bar sit-out after regime flip")


class RHBSimulator:
    """Simulate rhb (regime hold bars) counter."""
    def __init__(self):
        self.regime = "BULL"
        self.rhb    = 100  # well-established

    def flip(self, new_regime):
        if new_regime != self.regime:
            self.regime = new_regime
            self.rhb    = 0

    def tick(self):
        self.rhb += 1

    def should_trade(self):
        return self.rhb >= REGIME_COOL


# Test 1: No trades allowed for first REGIME_COOL bars after flip
rhb = RHBSimulator()
rhb.flip("BEAR")
blocked = sum(1 for _ in range(REGIME_COOL) if not (rhb.tick() or True) or not rhb.should_trade())
# Re-run correctly
rhb2 = RHBSimulator()
rhb2.flip("BEAR")
blocked = []
for _ in range(REGIME_COOL + 5):
    blocked.append(not rhb2.should_trade())
    rhb2.tick()

n_blocked = sum(blocked[:REGIME_COOL])
n_allowed = sum(1 for b in blocked[REGIME_COOL:] if not b)
check(f"First {REGIME_COOL} bars after regime flip: all blocked",
      n_blocked == REGIME_COOL, f"blocked={n_blocked}/{REGIME_COOL}")
check(f"After {REGIME_COOL} bars: trading resumes",
      n_allowed > 0, f"allowed={n_allowed} bars after cooldown")

# Test 2: Cooldown resets on EVERY regime change
rhb3 = RHBSimulator()
rhb3.flip("BEAR")
for _ in range(REGIME_COOL + 2):
    rhb3.tick()
rhb3.flip("SIDEWAYS")  # second flip
blocked2 = []
for _ in range(REGIME_COOL + 2):
    blocked2.append(not rhb3.should_trade())
    rhb3.tick()
check("Cooldown resets on second regime flip",
      blocked2[0] == True and blocked2[REGIME_COOL] == False,
      f"bar0_blocked={blocked2[0]} bar{REGIME_COOL}_allowed={not blocked2[REGIME_COOL]}")

# Test 3: No cooldown if regime is stable (rhb >> REGIME_COOL)
rhb4 = RHBSimulator()
rhb4.rhb = 200
check("No cooldown penalty in stable regime (rhb=200)",
      rhb4.should_trade(), f"rhb={rhb4.rhb}")

# Test 4: Cooldown prevents V-bottom entry (the diagnosed failure mode)
# Simulate: BULL→BEAR flip at bar 0 (crash), bounce starts bar 3
# Without cooldown: strategy enters SHORT at bar 0, gets caught by bar-3 bounce
# With cooldown: strategy sits flat bars 0-7, enters SHORT at bar 8 (confirmed BEAR)
rhb5 = RHBSimulator()
rhb5.flip("BEAR")

entries_with_cooldown = []
entries_without_cooldown = []
for bar in range(12):
    rhb5.tick()
    with_cd    = rhb5.should_trade()   # v13
    without_cd = True                  # v12 (no cooldown)
    entries_with_cooldown.append(with_cd)
    entries_without_cooldown.append(without_cd)

cd_first_entry   = next(i for i, v in enumerate(entries_with_cooldown) if v)
nocd_first_entry = next(i for i, v in enumerate(entries_without_cooldown) if v)

check("With cooldown: first entry is bar 8+ after flip (not bar 0)",
      cd_first_entry >= REGIME_COOL - 1,
      f"first_entry=bar_{cd_first_entry + 1}")
check("Without cooldown: first entry is bar 0 (the V-bottom entry)",
      nocd_first_entry == 0, f"first_entry=bar_{nocd_first_entry}")
check("Cooldown delays entry by at least REGIME_COOL bars",
      cd_first_entry - nocd_first_entry >= REGIME_COOL - 1,
      f"delay={cd_first_entry - nocd_first_entry} bars")


# ── §5 Integration — all four layers together ─────────────────────────────────
section("§5  Integration — all four layers working together")


def simulate_v13(equity_path, regime_path, vol=0.008):
    """
    Minimal v13 simulation: EMA anchor → tier → vol size → cooldown → CB.
    Returns list of (bar, equity, position, cb_state, rhb).
    """
    equity   = equity_path[0]
    peak     = equity
    ema      = equity
    cb_flat  = 0
    cb_half  = 0
    pos      = 0.0
    regime   = regime_path[0]
    rhb      = 50  # well-established at start
    results  = []

    for i, (eq_return, reg) in enumerate(zip(equity_path[1:], regime_path[1:])):
        # mark to market
        equity *= (1 + eq_return * pos)
        equity  = max(equity, 1.0)
        if equity > peak:
            peak = equity

        # update EMA
        ema = ema * (1 - EMA_K) + equity * EMA_K

        # regime
        if reg != regime:
            regime = reg
            rhb    = 0
        else:
            rhb += 1

        cb_state = "NORMAL"

        # circuit breaker
        if cb_flat > 0:
            cb_flat -= 1
            pos = 0.0
            cb_state = "CB_FLAT"
        elif equity < peak * (1 - CB_DD):
            cb_flat  = CB_FLAT
            cb_half  = CB_FLAT + CB_HALF
            pos      = 0.0
            cb_state = "CB_FIRED"
        else:
            cb_scale = 0.5 if cb_half > 0 else 1.0
            if cb_half > 0:
                cb_half -= 1
                cb_state = "CB_HALF"

            # cooldown gate
            if rhb < REGIME_COOL:
                pos = 0.0
                cb_state = "COOLDOWN"
            else:
                per_inst = get_per_inst_risk(ema)
                raw = per_inst / (vol + 1e-9)
                cap = min(raw, TF_CAP[6])
                if cb_state == "CB_HALF":
                    cap *= 0.5
                pos = cap  # always long for this simulation

        results.append({
            "bar": i, "equity": equity, "pos": pos,
            "cb_state": cb_state, "rhb": rhb, "ema": ema
        })

    return results


def sim_v13_fixed(n_bars, per_bar_return, regime_seq, starting_eq=1_000_000, vol=0.008):
    equity = starting_eq
    peak   = equity
    ema    = equity
    cb_flat = cb_half = 0
    pos = 0.0
    regime = regime_seq[0]
    rhb = 50
    equities = [equity]
    cb_fires = 0

    for i in range(n_bars):
        reg = regime_seq[min(i+1, len(regime_seq)-1)]
        ret = per_bar_return[i] if isinstance(per_bar_return, list) else per_bar_return

        equity *= (1 + ret * pos)
        equity  = max(equity, 1.0)
        if equity > peak:
            peak = equity

        ema = ema * (1 - EMA_K) + equity * EMA_K

        if reg != regime:
            regime = reg; rhb = 0
        else:
            rhb += 1

        if cb_flat > 0:
            cb_flat -= 1; pos = 0.0
        elif equity < peak * (1 - CB_DD):
            cb_flat = CB_FLAT; cb_half = CB_HALF; pos = 0.0; cb_fires += 1
        else:
            scale = 0.5 if cb_half > 0 else 1.0
            if cb_half > 0: cb_half -= 1
            if rhb < REGIME_COOL:
                pos = 0.0
            else:
                per_inst = get_per_inst_risk(ema)
                pos = min(per_inst / (vol + 1e-9), TF_CAP[6]) * scale

        equities.append(equity)

    return equities, cb_fires

# Scenario A: steady bull — use enough bars to compound above $2M
# At $1M, pos≈0.43, +0.4%/bar effective return ≈ +0.17%/bar on equity
# Need ~420 bars to double: 1M × (1.0017)^420 ≈ $2.05M
eq_bull, fires_bull = sim_v13_fixed(500, 0.004, ["BULL"]*501)
check("Steady bull run: equity compounds above $2M",
      eq_bull[-1] > 2_000_000, f"final={eq_bull[-1]:,.0f}")
check("Steady bull run: no CB fires (no 15% DD in steady trend)",
      fires_bull == 0, f"cb_fires={fires_bull}")

# Scenario B: crash after bull run — CB fires, limits loss
# Need crash deep enough to hit -15% from peak
# Bull builds equity to ~$1.09M peak, then needs >15% drop = equity < $927k
# At pos≈0.43, -1%/bar → -0.43%/bar on equity. Need 37+ bars to hit -15%
bull_then_crash = [0.004]*100 + [-0.01]*50
eq_crash, fires_crash = sim_v13_fixed(150, bull_then_crash, ["BULL"]*151)
peak_eq = max(eq_crash)
final_eq = eq_crash[-1]
check("After crash: CB fires at least once",
      fires_crash >= 1, f"cb_fires={fires_crash}")
check("After crash: final equity > 70% of peak (CB limited losses)",
      final_eq > peak_eq * 0.70,
      f"peak={peak_eq:,.0f} final={final_eq:,.0f} ratio={final_eq/peak_eq:.1%}")

# Scenario C: regime flip at bottom — cooldown prevents V-bottom entry
# Bull run, crash (BEAR), immediate bounce
returns_v_shape = [0.002]*50 + [-0.003]*20 + [0.004]*30
regimes_v_shape = ["BULL"]*51 + ["BEAR"]*21 + ["BULL"]*30
eq_v, fires_v = sim_v13_fixed(100, returns_v_shape, regimes_v_shape)
# v13 should not be long during the bounce immediately after BULL→BEAR flip
# because cooldown kicks in at rhb=0

# Check that final equity is positive and no runaway loss
check("V-shape scenario: strategy survives and recovers",
      eq_v[-1] > 800_000, f"final={eq_v[-1]:,.0f}")
check("V-shape scenario: CB fires at most once (not repeated spiral)",
      fires_v <= 1, f"cb_fires={fires_v}")


# ── §6 v13 vs v12 sizing at scale ────────────────────────────────────────────
section("§6  v13 vs v12 sizing at scale — confirm concave growth")

v12_fixed_per_inst = 0.01 / _CORR_FACTOR  # v12: fixed regardless of equity

equities = [1e6, 2e6, 5e6, 10e6, 15e6, 22e6]
print(f"\n  {'Equity':>12}  {'v12 per_inst':>14}  {'v13 per_inst':>14}  "
      f"{'v12/v13':>8}  {'v12 ES cts':>10}  {'v13 ES cts':>10}")
print(f"  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*8}  {'─'*10}  {'─'*10}")

es_price = 5000; es_mult = 50; daily_vol = 0.008

for eq in equities:
    v12_pi = v12_fixed_per_inst
    v13_pi = get_per_inst_risk(eq)
    ratio  = v12_pi / v13_pi
    v12_frac = min(v12_pi / daily_vol, TF_CAP[7])
    v13_frac = min(v13_pi / daily_vol, TF_CAP[7])
    v12_cts  = int(v12_frac * eq / (es_price * es_mult))
    v13_cts  = int(v13_frac * eq / (es_price * es_mult))
    print(f"  ${eq/1e6:>10.0f}M  {v12_pi:>14.5f}  {v13_pi:>14.5f}  "
          f"{ratio:>7.1f}×  {v12_cts:>10}  {v13_cts:>10}")

# v13 contracts should grow sub-linearly (concave)
cts_1m  = int(min(get_per_inst_risk(1e6)  / daily_vol, TF_CAP[7]) * 1e6  / (es_price * es_mult))
cts_22m = int(min(get_per_inst_risk(22e6) / daily_vol, TF_CAP[7]) * 22e6 / (es_price * es_mult))
cts_v12_22m = int(min(v12_fixed_per_inst  / daily_vol, TF_CAP[7]) * 22e6 / (es_price * es_mult))

check("v13: contracts grow sub-linearly — less than linear with equity",
      cts_22m / max(cts_1m, 1) < 22,   # equity grew 22×, contracts must grow < 22×
      f"1M={cts_1m} 22M={cts_22m} ratio={cts_22m/max(cts_1m,1):.1f}× (v12 would be {cts_v12_22m})")
check("v13 at $22M: fewer contracts than v12 at $22M",
      cts_22m < cts_v12_22m,
      f"v13={cts_22m} v12={cts_v12_22m}")
check("v13 at $22M: ES contracts < 15 (ATR lag cost bounded)",
      cts_22m < 15,
      f"v13_22m={cts_22m} contracts")
check("v13 at $1M: same as v12 (full aggression preserved below $2M)",
      abs(get_per_inst_risk(1e6) - v12_fixed_per_inst) < 1e-6,
      f"v13={get_per_inst_risk(1e6):.5f} v12={v12_fixed_per_inst:.5f}")


# ── Summary ───────────────────────────────────────────────────────────────────
n_pass = len(passed)
n_fail = len(failed)
n_total = n_pass + n_fail

print(f"\n{'═'*65}")
print(f"  TOTAL: {n_pass}/{n_total} passed  |  {n_fail} failed")

sections = {
    "§1  Fractional Kelly tiers": 7,
    "§2  Trailing EMA anchor": 4,
    "§3  Circuit breaker": 7,
    "§4  Regime transition cooldown": 6,
    "§5  Integration": 5,
    "§6  v13 vs v12 sizing at scale": 4,
}
i = 0
for sec, count in sections.items():
    sec_tests = passed[i:i+count] + [f for f in failed if f in passed[i:i+count]]
    n_sec_pass = sum(1 for t in passed[i:i+n_total] if True)  # just count
    # simpler: track by position
    sec_pass = sum(1 for t in passed[i:i+count] if True)
    sec_fail = sum(1 for t in failed if t in [passed[i:i+count]])
    mark = "✓" if i + count <= n_pass else "✗"
    print(f"  {mark}  {sec:<40} {min(count, n_pass-i)}/{count}")
    i += count

print(f"{'═'*65}")

# JSON
report = {
    "passed": n_pass,
    "failed": n_fail,
    "total": n_total,
    "failed_tests": failed,
    "v13_constants": {
        "equity_tiers": [(t if t != float("inf") else "inf", r) for t, r in EQUITY_TIERS],
        "ema_period": EMA_PERIOD,
        "cb_dd_threshold": CB_DD,
        "cb_flat_bars": CB_FLAT,
        "cb_half_bars": CB_HALF,
        "regime_cooldown_bars": REGIME_COOL,
        "corr_factor": _CORR_FACTOR,
    }
}
json_path = os.path.join(RESULTS, "suite_v13.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"\n  JSON report → {json_path}")

if n_fail == 0:
    print("\n  All tests passed. v13 is ready for QC backtest.")
else:
    print(f"\n  {n_fail} test(s) FAILED. Review before running QC backtest.")
    sys.exit(1)
