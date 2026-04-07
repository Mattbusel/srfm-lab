"""
Convert rl-exit-optimizer raw Q-table JSON (u64 StateKey -> [hold_q, exit_q])
to the 5-feature string-key format expected by RLExitPolicy._state_key.

Rust StateKey encodes 12 features (NUM_BINS=5) as a mixed-radix u64:
  key = f0*5^11 + f1*5^10 + ... + f11*5^0

Python RLExitPolicy._state_key uses the first 5 features (indices 0-4):
  f0 = pnl_pct bin         (bin 0=very negative .. 4=very positive)
  f1 = bars_held_norm bin  (bin 0=just entered .. 4=long hold)
  f2 = bh_mass_norm bin    (bin 0=low mass .. 4=high mass)
  f3 = bh_active bin       (0=inactive/dead, 4=active; Python checks 0 vs 4)
  f4 = atr_ratio_norm bin  (bin 0=low vol .. 4=high vol)

After projecting 12-feature Rust keys onto 5 features, the synthetic training
data tends to cluster (fixed pnl, bars, atr bins). We therefore ALSO generate
a full policy-rule table covering all 5^5 = 3125 state combinations, using
domain knowledge to assign Q-values where training data is absent.

Domain rules (matching the heuristic in RLExitPolicy.should_exit):
  - Strong loss (f0 in {0,1}): exit_q >> hold_q
  - Strong gain (f0 in {3,4}) + BH active (f3==4): hold_q > exit_q
  - BH dead (f3==0) + long hold (f1 >= 3): exit_q > hold_q
  - Otherwise: slight hold bias

Training data Q-values override rules where available.
"""

import json
from pathlib import Path
from itertools import product

NUM_BINS = 5
STATE_DIM = 12


def decode_state_key(key: int) -> list[int]:
    """Decode a mixed-radix u64 StateKey into list of bins (MSB first)."""
    bins = []
    for _ in range(STATE_DIM):
        bins.append(key % NUM_BINS)
        key //= NUM_BINS
    return list(reversed(bins))


def rule_q(f0: int, f1: int, f2: int, f3: int, f4: int) -> list[float]:
    """
    Generate heuristic Q-values for a 5-feature state.
    Returns [hold_q, exit_q].

    f1 bins (bars_held_norm):
      0 = bars 0-19    (just entered — MIN_HOLD_MINUTES = 240min / 15m = 16 bars)
      1 = bars 20-39
      2 = bars 40-59
      3 = bars 60-79
      4 = bars 80+     (long hold)

    Key rule: respect MIN_HOLD_MINUTES — never exit in first 16 bars (f1=0)
    unless the position is in serious loss (f0=0, >25% down).
    """
    bh_active  = (f3 >= 3)   # bin 3 or 4 = active
    very_loss  = (f0 == 0)   # extreme loss (>25% down after clip)
    big_loss   = (f0 <= 1)   # bottom 2 bins = losing trade
    big_gain   = (f0 >= 3)   # top 2 bins = profitable trade
    early_hold = (f1 <= 1)   # within first ~40 bars = within min-hold window
    long_hold  = (f1 >= 3)   # held a long time
    bh_mass_hi = (f2 >= 3)   # strong BH signal

    if very_loss:
        # Catastrophic loss — exit even if early (stop-loss)
        return [-0.10, 0.25]
    elif early_hold and not very_loss:
        # Respect MIN_HOLD_MINUTES: hold unless catastrophic
        return [0.05, -0.03]
    elif big_loss:
        # Hard stop territory — exit
        return [-0.05, 0.15]
    elif big_gain and bh_active and bh_mass_hi:
        # BH alive, mass strong, in profit — hold
        return [0.10, -0.05]
    elif not bh_active and long_hold:
        # BH gone, been holding a while — exit
        return [-0.02, 0.08]
    elif big_gain and not bh_active:
        # Profitable but BH dead — exit to lock in gains
        return [-0.01, 0.06]
    else:
        # Default: slight hold bias
        return [0.02, -0.01]


def convert(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        raw = json.load(f)

    table: dict = raw["table"]
    print(f"Input: {len(table)} Rust states")

    # Start with full rule-based table (all 5^5 = 3125 states)
    out: dict[str, list[float]] = {}
    for f0, f1, f2, f3, f4 in product(range(NUM_BINS), repeat=5):
        key = f"{f0},{f1},{f2},{f3},{f4}"
        out[key] = rule_q(f0, f1, f2, f3, f4)

    # Override with trained Q-values where available
    trained_overrides = 0
    for key_str, qs in table.items():
        key_int = int(key_str)
        bins = decode_state_key(key_int)
        f0, f1, f2, f3, f4 = bins[0], bins[1], bins[2], bins[3], bins[4]
        short_key = f"{f0},{f1},{f2},{f3},{f4}"
        hold_q, exit_q = float(qs[0]), float(qs[1])

        existing = out[short_key]
        # Never override early-hold states (f1 <= 1) with trained data —
        # synthetic training data systematically biases toward early exit
        if f1 <= 1:
            continue
        # Override if trained values are more confident (larger magnitude)
        if max(abs(hold_q), abs(exit_q)) > max(abs(existing[0]), abs(existing[1])):
            out[short_key] = [hold_q, exit_q]
            trained_overrides += 1

    print(f"Output: {len(out)} states ({trained_overrides} overridden by training data)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(f"Saved to: {output_path}")

    # Validation: spot-check a few interesting states
    checks = [
        ("0,0,2,0,2", "big loss, BH inactive -> should exit"),
        ("4,1,4,4,2", "big gain, BH active + mass high -> should hold"),
        ("2,3,2,0,2", "neutral, BH dead, long hold -> should exit"),
    ]
    print("Spot checks:")
    for k, desc in checks:
        qs = out.get(k, [None, None])
        action = "EXIT" if qs[1] > qs[0] else "HOLD"
        print(f"  {k} ({desc}): {action}  Q=[{qs[0]:.4f}, {qs[1]:.4f}]")


if __name__ == "__main__":
    repo = Path(__file__).parent.parent
    inp = str(repo / "config" / "rl_exit_qtable_raw.json")
    outp = str(repo / "config" / "rl_exit_qtable.json")
    convert(inp, outp)
