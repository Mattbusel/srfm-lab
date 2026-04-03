"""
autotuner.py — Bayesian optimization (or random search) over LARSA arena_v2 parameters.

Uses optuna if available, falls back to random search over 100 configurations.

Objective: maximize  0.6 * arena_sharpe + 0.3 * synth_sharpe - 0.1 * max(0, arena_dd - 25)

Parameters searched:
  cf        [0.0005, 0.010]  — critical frequency (TIMELIKE/SPACELIKE split)
  bh_form   [1.0, 3.0]       — BH formation threshold
  bh_decay  [0.80, 0.99]     — BH mass decay per SPACELIKE bar
  max_lev   [0.30, 0.80]     — maximum leverage passed to run_v2
  solo_cap  [0.10, 0.40]     — solo BH position cap (FUTURE: limits single-BH sizing in BH_BOOST)
  conv_cap  [0.40, 0.80]     — convergence position cap (FUTURE: limits total pos when geodesic
                                convergence detected, controlling crowding risk)

solo_cap and conv_cap are plumbing for future arena_v3 use; currently passed as exp_flags
string metadata but do not alter arena_v2 outputs.

Usage:
    python tools/autotuner.py --trials 50 --csv data/NDX_hourly_poly.csv
    python tools/autotuner.py --trials 20 --quick   # 3 synth worlds only
"""

import argparse
import json
import os
import sys
import time
import random

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from arena_v2 import run_v2, load_ohlcv, generate_synthetic, CONFIGS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

PARAM_BOUNDS = {
    "cf":       (0.0005, 0.010),
    "bh_form":  (1.0,    3.0),
    "bh_decay": (0.80,   0.99),
    "max_lev":  (0.30,   0.80),
    "solo_cap": (0.10,   0.40),
    "conv_cap": (0.40,   0.80),
}

SYNTH_SEEDS = [42, 137, 999]


def build_cfg(cf, bh_form, bh_decay):
    return {"cf": cf, "bh_form": bh_form, "bh_collapse": 1.0, "bh_decay": bh_decay}


def evaluate(bars_arena, bars_synths, cf, bh_form, bh_decay, max_lev,
             solo_cap, conv_cap):
    """Run arena + synthetic worlds, return objective score and component stats."""
    cfg = build_cfg(cf, bh_form, bh_decay)
    # solo_cap / conv_cap stored for future use; not yet wired into arena_v2
    # When arena_v3 lands: exp_flags will accept S=solo_cap, V=conv_cap values.
    exp_flags = "ABCD"

    def run_one(bars):
        try:
            broker, _ = run_v2(bars, cfg, max_leverage=max_lev, exp_flags=exp_flags)
            s = broker.stats()
            return s.get("sharpe", 0.0), s.get("max_drawdown_pct", 0.0)
        except Exception:
            return 0.0, 100.0

    arena_sh, arena_dd = run_one(bars_arena)

    synth_sharpes = []
    for sb in bars_synths:
        sh, _ = run_one(sb)
        synth_sharpes.append(sh)
    synth_sh = float(np.mean(synth_sharpes)) if synth_sharpes else 0.0

    score = 0.6 * arena_sh + 0.3 * synth_sh - 0.1 * max(0.0, arena_dd - 25.0)
    return score, arena_sh, synth_sh, arena_dd


# ---------------------------------------------------------------------------
# Optuna path
# ---------------------------------------------------------------------------

def run_optuna(bars_arena, bars_synths, n_trials, quiet=False):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []

    def objective(trial):
        cf       = trial.suggest_float("cf",       *PARAM_BOUNDS["cf"])
        bh_form  = trial.suggest_float("bh_form",  *PARAM_BOUNDS["bh_form"])
        bh_decay = trial.suggest_float("bh_decay", *PARAM_BOUNDS["bh_decay"])
        max_lev  = trial.suggest_float("max_lev",  *PARAM_BOUNDS["max_lev"])
        solo_cap = trial.suggest_float("solo_cap", *PARAM_BOUNDS["solo_cap"])
        conv_cap = trial.suggest_float("conv_cap", *PARAM_BOUNDS["conv_cap"])

        score, arena_sh, synth_sh, arena_dd = evaluate(
            bars_arena, bars_synths,
            cf, bh_form, bh_decay, max_lev, solo_cap, conv_cap,
        )
        results.append({
            "trial": trial.number,
            "score": round(score, 4),
            "arena_sharpe": round(arena_sh, 4),
            "synth_sharpe": round(synth_sh, 4),
            "arena_dd": round(arena_dd, 2),
            "cf": cf, "bh_form": bh_form, "bh_decay": bh_decay,
            "max_lev": max_lev, "solo_cap": solo_cap, "conv_cap": conv_cap,
        })
        if not quiet:
            print(f"  Trial {trial.number:3d}  score={score:+.4f}  "
                  f"arena_sh={arena_sh:.3f}  synth_sh={synth_sh:.3f}  "
                  f"dd={arena_dd:.1f}%  cf={cf:.5f}  bh_form={bh_form:.2f}  "
                  f"bh_decay={bh_decay:.3f}  max_lev={max_lev:.3f}")
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return results, study.best_params


# ---------------------------------------------------------------------------
# Random-search fallback
# ---------------------------------------------------------------------------

def run_random(bars_arena, bars_synths, n_trials, seed=0, quiet=False):
    rng = random.Random(seed)
    results = []

    def sample(lo, hi):
        return lo + rng.random() * (hi - lo)

    for i in range(n_trials):
        cf       = sample(*PARAM_BOUNDS["cf"])
        bh_form  = sample(*PARAM_BOUNDS["bh_form"])
        bh_decay = sample(*PARAM_BOUNDS["bh_decay"])
        max_lev  = sample(*PARAM_BOUNDS["max_lev"])
        solo_cap = sample(*PARAM_BOUNDS["solo_cap"])
        conv_cap = sample(*PARAM_BOUNDS["conv_cap"])

        score, arena_sh, synth_sh, arena_dd = evaluate(
            bars_arena, bars_synths,
            cf, bh_form, bh_decay, max_lev, solo_cap, conv_cap,
        )
        results.append({
            "trial": i,
            "score": round(score, 4),
            "arena_sharpe": round(arena_sh, 4),
            "synth_sharpe": round(synth_sh, 4),
            "arena_dd": round(arena_dd, 2),
            "cf": cf, "bh_form": bh_form, "bh_decay": bh_decay,
            "max_lev": max_lev, "solo_cap": solo_cap, "conv_cap": conv_cap,
        })
        if not quiet:
            print(f"  Trial {i:3d}  score={score:+.4f}  "
                  f"arena_sh={arena_sh:.3f}  synth_sh={synth_sh:.3f}  "
                  f"dd={arena_dd:.1f}%  cf={cf:.5f}  bh_form={bh_form:.2f}  "
                  f"bh_decay={bh_decay:.3f}  max_lev={max_lev:.3f}")

    best = max(results, key=lambda r: r["score"])
    best_params = {k: best[k] for k in ("cf", "bh_form", "bh_decay", "max_lev", "solo_cap", "conv_cap")}
    return results, best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian/random optimization over LARSA arena_v2 parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",    default="data/NDX_hourly_poly.csv",
                        help="Path to OHLCV CSV (default: data/NDX_hourly_poly.csv)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of optimization trials (default: 50)")
    parser.add_argument("--quick",  action="store_true",
                        help="Use 3 synthetic worlds only (skip real CSV)")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Random seed for fallback search (default: 0)")
    parser.add_argument("--synth-bars", type=int, default=20000,
                        help="Bars per synthetic world (default: 20000)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    # Load data
    if args.quick:
        print("Quick mode: using synthetic data only.")
        bars_arena = generate_synthetic(args.synth_bars, seed=0)
    else:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)
        if os.path.exists(csv_path):
            print(f"Loading arena data from {csv_path} ...")
            bars_arena = load_ohlcv(csv_path)
            print(f"  {len(bars_arena)} bars loaded.")
        else:
            print(f"WARNING: {csv_path} not found — using synthetic data for arena.")
            bars_arena = generate_synthetic(args.synth_bars, seed=0)

    bars_synths = [generate_synthetic(args.synth_bars, seed=s) for s in SYNTH_SEEDS]

    # Optimizer selection
    try:
        import optuna  # noqa: F401
        backend = "optuna"
        print(f"\nUsing optuna Bayesian optimization ({args.trials} trials).")
    except ImportError:
        n_random = max(args.trials, 100)
        backend = f"random (n={n_random})"
        print(f"\noptuna not installed — falling back to random search ({n_random} configs).")
        args.trials = n_random

    print(f"Arena bars: {len(bars_arena)} | Synth worlds: {len(bars_synths)}\n")

    if backend == "optuna":
        all_results, best_params = run_optuna(bars_arena, bars_synths, args.trials)
    else:
        all_results, best_params = run_random(bars_arena, bars_synths, args.trials, seed=args.seed)

    elapsed = time.time() - t0

    # Sort and display top 10
    ranked = sorted(all_results, key=lambda r: r["score"], reverse=True)
    print("\n" + "=" * 72)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 72)
    hdr = f"{'Rank':>4}  {'Score':>7}  {'Arena_Sh':>8}  {'Synth_Sh':>8}  {'DD%':>6}  {'cf':>8}  {'bh_form':>7}  {'bh_dec':>6}  {'lev':>5}"
    print(hdr)
    print("-" * 72)
    for rank, r in enumerate(ranked[:10], 1):
        print(f"{rank:4d}  {r['score']:>7.4f}  {r['arena_sharpe']:>8.4f}  "
              f"{r['synth_sharpe']:>8.4f}  {r['arena_dd']:>6.2f}  "
              f"{r['cf']:>8.5f}  {r['bh_form']:>7.3f}  {r['bh_decay']:>6.3f}  "
              f"{r['max_lev']:>5.3f}")

    best = ranked[0]
    print(f"\nBest score: {best['score']:.4f}")
    print(f"  cf={best['cf']:.5f}  bh_form={best['bh_form']:.3f}  "
          f"bh_decay={best['bh_decay']:.3f}  max_lev={best['max_lev']:.3f}")
    print(f"  solo_cap={best['solo_cap']:.3f}  conv_cap={best['conv_cap']:.3f}  "
          f"[future arena_v3 params]")
    print(f"\nWall-clock time: {elapsed:.1f}s")

    # Save outputs
    best_config_path = os.path.join(RESULTS_DIR, "best_config.json")
    trials_path      = os.path.join(RESULTS_DIR, "tuner_trials.json")

    best_out = {
        "score":        best["score"],
        "arena_sharpe": best["arena_sharpe"],
        "synth_sharpe": best["synth_sharpe"],
        "arena_dd":     best["arena_dd"],
        "cfg": {
            "cf":       best["cf"],
            "bh_form":  best["bh_form"],
            "bh_collapse": 1.0,
            "bh_decay": best["bh_decay"],
        },
        "run_v2_kwargs": {
            "max_leverage": best["max_lev"],
            "exp_flags":    "ABCD",
        },
        "future_params": {
            "solo_cap": best["solo_cap"],
            "conv_cap": best["conv_cap"],
        },
        "backend": backend,
        "n_trials": len(all_results),
    }
    with open(best_config_path, "w") as f:
        json.dump(best_out, f, indent=2)

    with open(trials_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved best config  → {best_config_path}")
    print(f"Saved all trials   → {trials_path}")


if __name__ == "__main__":
    main()
