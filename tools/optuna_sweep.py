"""
optuna_sweep.py — Bayesian hyperparameter optimization for LARSA arena.

Uses Optuna TPE sampler with MedianPruner for 10x efficiency vs grid search.
Optimizes: cf, bh_form, bh_decay, max_lev over up to 500 trials.

Usage:
    python tools/optuna_sweep.py --trials 200 --csv data/NDX_hourly_poly.csv
    python tools/optuna_sweep.py --trials 50 --quick    # fast mode
    python tools/optuna_sweep.py --resume               # continue existing study
    python tools/optuna_sweep.py --dashboard            # launch optuna-dashboard
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bars(csv_path: str):
    """Load OHLCV bars from CSV."""
    import csv as _csv
    bars = []
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            def g(*keys):
                for k in keys:
                    v = row.get(k) or row.get(k.lower()) or row.get(k.upper())
                    if v not in (None, "", "null", "None"):
                        try:
                            return float(v)
                        except Exception:
                            pass
                return None
            c = g("close", "Close")
            if c and c > 0:
                bars.append({
                    "date":   row.get("date") or row.get("Date") or "",
                    "open":   g("open", "Open") or c,
                    "high":   g("high", "High") or c,
                    "low":    g("low", "Low") or c,
                    "close":  c,
                    "volume": g("volume", "Volume") or 1000.0,
                })
    return bars


def _run_arena(bars, cfg, max_leverage: float):
    """Run arena_v2 and return stats dict. Returns None on error."""
    try:
        from tools.arena_v2 import run_v2
    except ImportError:
        try:
            from arena_v2 import run_v2
        except ImportError:
            return None
    try:
        broker, _ = run_v2(bars, cfg, max_leverage=max_leverage, exp_flags="ABCD")
        return broker.stats()
    except Exception as e:
        print(f"    [arena error] {e}", file=sys.stderr)
        return None


def _generate_synthetic(n_bars: int = 20000, seed: int = 42):
    """Generate synthetic bars matching arena_v2 generator."""
    try:
        from tools.arena_v2 import generate_synthetic
    except ImportError:
        try:
            from arena_v2 import generate_synthetic
        except ImportError:
            return _fallback_synthetic(n_bars, seed)
    return generate_synthetic(n_bars, seed)


def _fallback_synthetic(n_bars: int, seed: int):
    rng = np.random.default_rng(seed)
    price = 4500.0
    bars = []
    for i in range(n_bars):
        mu, sig = 0.0001, 0.001
        ret = max(-0.05, min(0.05, mu + sig * float(rng.standard_normal())))
        close = price * (1 + ret)
        bars.append({
            "date": f"bar_{i:06d}", "open": price,
            "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": 50000.0,
        })
        price = close
    return bars


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def build_objective(bars, quick: bool = False):
    """Return an Optuna objective closure."""
    import optuna

    def objective(trial):
        cf       = trial.suggest_float("cf",       0.001, 0.015, log=True)
        bh_form  = trial.suggest_float("bh_form",  1.0,   3.0)
        bh_decay = trial.suggest_float("bh_decay", 0.80,  0.99)
        max_lev  = trial.suggest_float("max_lev",  0.30,  0.85)

        cfg = {
            "cf":          cf,
            "bh_form":     bh_form,
            "bh_collapse": 1.0,
            "bh_decay":    bh_decay,
        }

        stats = _run_arena(bars, cfg, max_leverage=max_lev)
        if stats is None:
            raise optuna.TrialPruned()

        arena_sharpe = stats.get("sharpe", 0.0)
        arena_dd     = stats.get("max_drawdown_pct", 100.0)

        # Report intermediate for pruner
        trial.report(arena_sharpe, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Synthetic worlds overfitting check (skip in quick mode)
        synth_sharpes = []
        n_synth = 1 if quick else 3
        for seed in range(n_synth):
            synth_bars = _generate_synthetic(20000, seed=seed + 100)
            sb = _run_arena(synth_bars, cfg, max_leverage=max_lev)
            if sb is not None:
                synth_sharpes.append(sb.get("sharpe", 0.0))

        if synth_sharpes:
            synth_sharpe = float(np.median(synth_sharpes))
        else:
            synth_sharpe = 0.0

        score = 0.6 * arena_sharpe + 0.3 * synth_sharpe - 0.1 * max(0, arena_dd - 20)
        return score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna Bayesian sweep for LARSA arena.")
    parser.add_argument("--trials",    type=int, default=200,   help="Number of trials (default 200)")
    parser.add_argument("--csv",       default="data/NDX_hourly_poly.csv", help="Price CSV path")
    parser.add_argument("--synthetic", action="store_true",     help="Use synthetic data instead of CSV")
    parser.add_argument("--n-bars",    type=int, default=20000, help="Bars for synthetic data")
    parser.add_argument("--quick",     action="store_true",     help="Fast mode: 1 synthetic world per trial")
    parser.add_argument("--resume",    action="store_true",     help="Continue existing study from SQLite")
    parser.add_argument("--dashboard", action="store_true",     help="Launch optuna-dashboard after sweep")
    parser.add_argument("--study-name", default="larsa_v6",     help="Study name")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Load bars
    if args.synthetic:
        print(f"Generating {args.n_bars} synthetic bars ...")
        bars = _generate_synthetic(args.n_bars, seed=42)
    else:
        print(f"Loading {args.csv} ...")
        if not os.path.exists(args.csv):
            print(f"  ERROR: {args.csv} not found. Use --synthetic or provide a valid CSV.")
            sys.exit(1)
        bars = _load_bars(args.csv)
        print(f"  {len(bars)} bars loaded.")

    # Check arena_v2 importable
    test_cfg = {"cf": 0.005, "bh_form": 1.5, "bh_collapse": 1.0, "bh_decay": 0.95}
    test = _run_arena(bars[:500], test_cfg, 0.65)
    if test is None:
        print("  WARNING: arena_v2 import failed. Optimization will produce no useful results.")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)

        db_path   = "results/optuna_study.db"
        study_url = f"sqlite:///{db_path}"

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=study_url,
            study_name=args.study_name,
            load_if_exists=args.resume or True,   # always resumable
        )

        n_existing = len(study.trials)
        if n_existing > 0 and args.resume:
            print(f"  Resuming study '{args.study_name}' ({n_existing} existing trials)")
        else:
            print(f"  Starting study '{args.study_name}' — {args.trials} trials")

        objective = build_objective(bars, quick=args.quick)

        study.optimize(
            objective,
            n_trials=args.trials,
            show_progress_bar=True,
        )

        # --- Results ---
        best = study.best_trial
        all_trials    = study.trials
        pruned_count  = sum(1 for t in all_trials if t.state.name == "PRUNED")
        total_count   = len(all_trials)

        print(f"\n{'='*55}")
        print(f"OPTUNA BAYESIAN OPTIMIZATION — {args.trials} trials")
        print(f"{'='*55}")
        print(f"Best trial: #{best.number}  (score={best.value:.3f})")
        for k, v in best.params.items():
            print(f"  {k:<12} {v:.5g}")

        # Re-run best to show stats
        best_cfg = {
            "cf":          best.params["cf"],
            "bh_form":     best.params["bh_form"],
            "bh_collapse": 1.0,
            "bh_decay":    best.params["bh_decay"],
        }
        best_stats = _run_arena(bars, best_cfg, max_leverage=best.params["max_lev"])
        if best_stats:
            print(f"\n  Arena:  Sharpe={best_stats.get('sharpe', 0):.2f}  "
                  f"Return={best_stats.get('total_return_pct', 0):+.1f}%  "
                  f"DD={best_stats.get('max_drawdown_pct', 0):.1f}%")

        synth_bars  = _generate_synthetic(20000, seed=999)
        synth_stats = _run_arena(synth_bars, best_cfg, max_leverage=best.params["max_lev"])
        if synth_stats:
            print(f"  Synth:  Sharpe={synth_stats.get('sharpe', 0):.2f}  "
                  f"Return={synth_stats.get('total_return_pct', 0):+.1f}%")

        print(f"\nPruned trials: {pruned_count}/{total_count} "
              f"({100*pruned_count/max(total_count,1):.1f}% efficiency gain)")

        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\nPARAMETER IMPORTANCE (via optuna.importance):")
            for pname, imp in sorted(importance.items(), key=lambda x: -x[1]):
                bar = "#" * int(imp * 40)
                print(f"  {pname:<12} {imp:.3f}  {bar}")
        except Exception as e:
            print(f"  (param importance unavailable: {e})")

        # Save best config
        best_out = {
            "study_name": args.study_name,
            "best_trial": best.number,
            "score":      best.value,
            "params":     best.params,
        }
        if best_stats:
            best_out["arena_stats"] = best_stats
        if synth_stats:
            best_out["synth_stats"] = synth_stats

        out_path = "results/optuna_best.json"
        with open(out_path, "w") as f:
            json.dump(best_out, f, indent=2)
        print(f"\nSaving best config to {out_path}")
        print(f"Study stored at {db_path} (resumable with --resume)")

        if args.dashboard:
            try:
                import subprocess
                print("\nLaunching optuna-dashboard ...")
                subprocess.Popen(["optuna-dashboard", study_url])
            except Exception as e:
                print(f"  (optuna-dashboard launch failed: {e})")
                print(f"  Run manually: optuna-dashboard {study_url}")

    except ImportError:
        print("  ERROR: optuna not installed. Run: pip install optuna")
        print("  Falling back to simple grid search ...")

        # Minimal grid fallback
        grid = [
            {"cf": 0.003, "bh_form": 1.5, "bh_decay": 0.92, "max_lev": 0.55},
            {"cf": 0.005, "bh_form": 1.5, "bh_decay": 0.92, "max_lev": 0.65},
            {"cf": 0.007, "bh_form": 2.0, "bh_decay": 0.95, "max_lev": 0.65},
            {"cf": 0.010, "bh_form": 2.0, "bh_decay": 0.90, "max_lev": 0.70},
        ]
        best_score = -9e9
        best_params = None
        for p in grid:
            cfg = {"cf": p["cf"], "bh_form": p["bh_form"],
                   "bh_collapse": 1.0, "bh_decay": p["bh_decay"]}
            s = _run_arena(bars, cfg, p["max_lev"])
            if s is not None:
                score = s.get("sharpe", 0.0)
                print(f"  cf={p['cf']:.4f} bh_form={p['bh_form']} "
                      f"bh_decay={p['bh_decay']} max_lev={p['max_lev']} -> sharpe={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_params = p

        if best_params:
            with open("results/optuna_best.json", "w") as f:
                json.dump({"method": "grid_fallback", "params": best_params,
                           "sharpe": best_score}, f, indent=2)
            print(f"\nBest (grid): {best_params}  sharpe={best_score:.3f}")


if __name__ == "__main__":
    main()
