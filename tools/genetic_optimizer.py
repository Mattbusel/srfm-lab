"""
genetic_optimizer.py — Genetic algorithm for LARSA parameter optimization.

Evolves a population of parameter sets. Each individual = one set of
LARSA parameters. Fitness = arena Sharpe + synthetic Sharpe.
Supports crossover, mutation, tournament selection.

Usage:
    python tools/genetic_optimizer.py --generations 50 --pop-size 30
    python tools/genetic_optimizer.py --quick  # 10 generations, 15 individuals
"""

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Arena import
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
try:
    from arena_v2 import run_v2, generate_synthetic, CONFIGS
    ARENA_AVAILABLE = True
except ImportError:
    ARENA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Genome definition
# ---------------------------------------------------------------------------
GENOME: Dict[str, Tuple] = {
    "cf":        (0.001, 0.015),   # continuous
    "bh_form":   (1.0, 3.0),       # continuous
    "bh_decay":  (0.80, 0.99),     # continuous
    "max_lev":   (0.30, 0.85),     # continuous
    "solo_cap":  (0.10, 0.45),     # continuous
    "conv_cap":  (0.40, 0.80),     # continuous
    "ctl_req":   (3, 8),           # integer
    "pt_thresh": (0.2, 1.0),       # continuous
}
INTEGER_GENES = {"ctl_req"}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Individual helpers
# ---------------------------------------------------------------------------

def random_individual(rng: random.Random) -> Dict[str, Any]:
    ind: Dict[str, Any] = {}
    for gene, (lo, hi) in GENOME.items():
        if gene in INTEGER_GENES:
            ind[gene] = rng.randint(int(lo), int(hi))
        else:
            ind[gene] = lo + rng.random() * (hi - lo)
    return ind


def clip_individual(ind: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for gene, (lo, hi) in GENOME.items():
        if gene in INTEGER_GENES:
            out[gene] = int(np.clip(round(ind[gene]), lo, hi))
        else:
            out[gene] = float(np.clip(ind[gene], lo, hi))
    return out


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def _sharpe(returns: List[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float(arr.mean() / std * np.sqrt(252))


def _extract_metrics(broker, bar_log: list) -> Tuple[float, float]:
    """Return (sharpe, max_drawdown_pct)."""
    equity = [b["equity"] for b in bar_log if "equity" in b]
    if not equity:
        equity = [1_000_000.0]
    returns = []
    for i in range(1, len(equity)):
        if equity[i - 1] > 0:
            returns.append((equity[i] - equity[i - 1]) / equity[i - 1])
    sh = _sharpe(returns)
    # drawdown
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / (peak + 1e-9) * 100
        if dd > max_dd:
            max_dd = dd
    return sh, max_dd


def evaluate(ind: Dict[str, Any], synth_bars: list) -> float:
    """
    fitness = arena_sharpe * 0.6 + synth_sharpe * 0.4 - max(0, arena_dd - 20) * 0.1
    """
    if not ARENA_AVAILABLE:
        # Random placeholder when arena not available
        return float(np.random.default_rng().uniform(0.3, 2.5))

    cfg = {
        "cf":           ind["cf"],
        "bh_form":      ind["bh_form"],
        "bh_collapse":  1.0,
        "bh_decay":     ind["bh_decay"],
    }
    max_lev = float(ind["max_lev"])

    try:
        broker_a, log_a = run_v2(synth_bars[:len(synth_bars)//2], cfg, max_leverage=max_lev)
        arena_sh, arena_dd = _extract_metrics(broker_a, log_a)

        broker_s, log_s = run_v2(synth_bars[len(synth_bars)//2:], cfg, max_leverage=max_lev)
        synth_sh, _ = _extract_metrics(broker_s, log_s)
    except Exception:
        return -1.0

    score = arena_sh * 0.6 + synth_sh * 0.4 - max(0.0, arena_dd - 20) * 0.1
    return float(score)


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def tournament_select(population: List[Dict], fitnesses: List[float],
                      k: int = 3, rng: Optional[random.Random] = None) -> Dict:
    if rng is None:
        rng = random.Random()
    candidates = rng.choices(range(len(population)), k=k)
    best = max(candidates, key=lambda i: fitnesses[i])
    return deepcopy(population[best])


def uniform_crossover(p1: Dict, p2: Dict, rng: random.Random) -> Tuple[Dict, Dict]:
    c1, c2 = {}, {}
    for gene in GENOME:
        if rng.random() < 0.5:
            c1[gene] = p1[gene]
            c2[gene] = p2[gene]
        else:
            c1[gene] = p2[gene]
            c2[gene] = p1[gene]
    return c1, c2


def mutate(ind: Dict, rng: random.Random, rate: float = 0.15) -> Dict:
    out = deepcopy(ind)
    for gene, (lo, hi) in GENOME.items():
        if rng.random() < rate:
            if gene in INTEGER_GENES:
                out[gene] = out[gene] + rng.choice([-1, 1])
            else:
                sigma = (hi - lo) * 0.10
                out[gene] = out[gene] + rng.gauss(0, sigma)
    return clip_individual(out)


# ---------------------------------------------------------------------------
# Simple GA (no DEAP)
# ---------------------------------------------------------------------------

def _bar_chart(val: float, max_val: float, width: int = 8) -> str:
    BLOCKS = " ▁▂▃▄▅▆▇█"
    if max_val <= 0:
        return " " * width
    frac = min(1.0, val / max_val)
    filled = int(frac * width)
    partial_idx = int((frac * width - filled) * (len(BLOCKS) - 1))
    bar = BLOCKS[8] * filled
    if filled < width:
        bar += BLOCKS[partial_idx]
        bar += " " * (width - filled - 1)
    return bar


def run_ga(generations: int, pop_size: int, seed: int = 42,
           verbose: bool = True) -> Tuple[Dict, List[float]]:
    rng = random.Random(seed)

    # Generate synthetic data once
    if ARENA_AVAILABLE:
        synth_bars = generate_synthetic(n_bars=20000, seed=seed)
    else:
        synth_bars = []

    # Initialize population
    population = [random_individual(rng) for _ in range(pop_size)]
    fitnesses = [evaluate(ind, synth_bars) for ind in population]

    history: List[float] = []
    all_time_best_fit = max(fitnesses)
    all_time_best_ind = deepcopy(population[np.argmax(fitnesses)])

    for gen in range(1, generations + 1):
        t0 = time.time()

        # Elitism — keep top 2
        sorted_idx = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        new_pop = [deepcopy(population[sorted_idx[0]]), deepcopy(population[sorted_idx[1]])]
        new_fits = [fitnesses[sorted_idx[0]], fitnesses[sorted_idx[1]]]

        while len(new_pop) < pop_size:
            p1 = tournament_select(population, fitnesses, k=3, rng=rng)
            p2 = tournament_select(population, fitnesses, k=3, rng=rng)
            c1, c2 = uniform_crossover(p1, p2, rng)
            c1 = mutate(c1, rng)
            c2 = mutate(c2, rng)
            for child in [c1, c2]:
                if len(new_pop) < pop_size:
                    new_pop.append(child)
                    new_fits.append(evaluate(child, synth_bars))

        population = new_pop
        fitnesses = new_fits

        best_fit = max(fitnesses)
        mean_fit = float(np.mean(fitnesses))
        best_idx = int(np.argmax(fitnesses))
        elapsed = time.time() - t0

        history.append(best_fit)

        if best_fit > all_time_best_fit:
            all_time_best_fit = best_fit
            all_time_best_ind = deepcopy(population[best_idx])

        if verbose:
            top3_idx = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:3]
            top3_str = "  ".join(
                f"[cf={population[i]['cf']:.4f},bh_f={population[i]['bh_form']:.1f},"
                f"lev={population[i]['max_lev']:.2f},sh={fitnesses[i]:.3f}]"
                for i in top3_idx
            )
            print(f"Gen {gen:2d}/{generations}  best={best_fit:.3f}  mean={mean_fit:.3f}"
                  f"  pop={pop_size}  time={elapsed:.0f}s")
            print(f"  Top 3: {top3_str}")

    return all_time_best_ind, history


# ---------------------------------------------------------------------------
# DEAP wrapper (preferred if available)
# ---------------------------------------------------------------------------

def run_ga_deap(generations: int, pop_size: int, seed: int = 42) -> Tuple[Dict, List[float]]:
    from deap import base, creator, tools, algorithms  # type: ignore

    if ARENA_AVAILABLE:
        synth_bars = generate_synthetic(n_bars=20000, seed=seed)
    else:
        synth_bars = []

    rng_py = random.Random(seed)
    gene_keys = list(GENOME.keys())

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def rand_attrs():
        ind_dict = random_individual(rng_py)
        return [ind_dict[k] for k in gene_keys]

    toolbox.register("individual", tools.initIterate, creator.Individual, rand_attrs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_ind(ind_list):
        ind = {k: v for k, v in zip(gene_keys, ind_list)}
        ind = clip_individual(ind)
        return (evaluate(ind, synth_bars),)

    def cx_uniform(ind1, ind2):
        for i in range(len(ind1)):
            if rng_py.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2

    def mut_gaussian(individual):
        for i, key in enumerate(gene_keys):
            if rng_py.random() < 0.15:
                lo, hi = GENOME[key]
                if key in INTEGER_GENES:
                    individual[i] = int(np.clip(individual[i] + rng_py.choice([-1, 1]), lo, hi))
                else:
                    sigma = (hi - lo) * 0.10
                    individual[i] = float(np.clip(individual[i] + rng_py.gauss(0, sigma), lo, hi))
        return (individual,)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", cx_uniform)
    toolbox.register("mutate", mut_gaussian)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("best", max)
    stats.register("mean", np.mean)

    history: List[float] = []

    for gen in range(1, generations + 1):
        t0 = time.time()
        offspring = algorithms.varAnd(toolbox.select(pop, len(pop)), toolbox, cxpb=0.7, mutpb=0.3)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)
        elapsed = time.time() - t0
        best = float(record["best"][0])
        mean = float(record["mean"])
        history.append(best)
        top3 = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)[:3]
        top3_str = "  ".join(
            f"[cf={ind[0]:.4f},bh_f={ind[1]:.1f},lev={ind[3]:.2f},sh={ind.fitness.values[0]:.3f}]"
            for ind in top3
        )
        print(f"Gen {gen:2d}/{generations}  best={best:.3f}  mean={mean:.3f}"
              f"  pop={pop_size}  time={elapsed:.0f}s")
        print(f"  Top 3: {top3_str}")

    best_list = list(hof[0])
    best_ind = {k: v for k, v in zip(gene_keys, best_list)}
    best_ind = clip_individual(best_ind)
    return best_ind, history


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def print_final_report(best: Dict, history: List[float], generations: int,
                       synth_bars: list) -> None:
    print()
    print("GENETIC OPTIMIZER — FINAL RESULTS ({} generations)".format(generations))
    print("=" * 52)
    print(f"WINNER: cf={best['cf']:.4f}  bh_form={best['bh_form']:.2f}  "
          f"bh_decay={best['bh_decay']:.3f}  max_lev={best['max_lev']:.2f}  "
          f"solo_cap={best['solo_cap']:.2f}  conv_cap={best['conv_cap']:.2f}")

    # Re-evaluate winner for detailed stats
    if ARENA_AVAILABLE and synth_bars:
        cfg = {"cf": best["cf"], "bh_form": best["bh_form"],
               "bh_collapse": 1.0, "bh_decay": best["bh_decay"]}
        max_lev = best["max_lev"]
        half = len(synth_bars) // 2
        try:
            broker_a, log_a = run_v2(synth_bars[:half], cfg, max_leverage=max_lev)
            arena_sh, arena_dd = _extract_metrics(broker_a, log_a)
            eq_a = [b["equity"] for b in log_a if "equity" in b]
            arena_ret = (eq_a[-1] / eq_a[0] - 1) * 100 if len(eq_a) >= 2 else 0.0

            broker_s, log_s = run_v2(synth_bars[half:], cfg, max_leverage=max_lev)
            synth_sh, synth_dd = _extract_metrics(broker_s, log_s)
            eq_s = [b["equity"] for b in log_s if "equity" in b]
            synth_ret = (eq_s[-1] / eq_s[0] - 1) * 100 if len(eq_s) >= 2 else 0.0

            score = arena_sh * 0.6 + synth_sh * 0.4 - max(0.0, arena_dd - 20) * 0.1
            print(f"  Arena:  Sharpe={arena_sh:.3f}  Return={arena_ret:+.1f}%  DD={arena_dd:.1f}%")
            print(f"  Synth:  Sharpe={synth_sh:.3f}  Return={synth_ret:+.1f}%  DD={synth_dd:.1f}%")
            print(f"  Score:  {score:.3f}")
        except Exception as e:
            print(f"  (could not re-evaluate: {e})")

    print()
    print("Evolution curve (best fitness per generation):")
    if history:
        max_fit = max(history)
        checkpoints = [1, 5, 10, 20, 50, 100]
        shown = set()
        for i, fit in enumerate(history, 1):
            if i in checkpoints or i == len(history):
                if i not in shown:
                    bar = _bar_chart(fit, max_fit, width=8)
                    print(f"  Gen {i:3d}: {fit:.3f}  {bar}")
                    shown.add(i)

        # Convergence point: first gen reaching 95% of final fitness
        target = max_fit * 0.95
        conv_gen = next((i for i, f in enumerate(history, 1) if f >= target), len(history))
        print()
        print(f"Convergence: reached 95% of final fitness at generation {conv_gen}")


def save_results(best: Dict, history: List[float]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_path = os.path.join(RESULTS_DIR, "genetic_best.json")
    hist_path = os.path.join(RESULTS_DIR, "genetic_history.json")

    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    with open(hist_path, "w") as f:
        json.dump({"history": history, "generations": len(history)}, f, indent=2)

    print(f"\nSaved: {best_path}")
    print(f"Saved: {hist_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Genetic algorithm for LARSA parameter optimization")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="10 generations, 15 individuals")
    parser.add_argument("--no-deap", action="store_true", help="Force simple GA even if DEAP available")
    args = parser.parse_args()

    if args.quick:
        args.generations = 10
        args.pop_size = 15

    print(f"LARSA Genetic Optimizer — {args.generations} generations, "
          f"pop={args.pop_size}, seed={args.seed}")
    if not ARENA_AVAILABLE:
        print("WARNING: arena_v2 not available — using random fitness (demo mode)")

    use_deap = False
    if not args.no_deap:
        try:
            import deap  # noqa: F401
            use_deap = True
            print("Using DEAP library for GA.")
        except ImportError:
            print("DEAP not installed — using built-in GA.")

    t_start = time.time()
    if use_deap:
        best, history = run_ga_deap(args.generations, args.pop_size, args.seed)
    else:
        best, history = run_ga(args.generations, args.pop_size, args.seed)

    elapsed_total = time.time() - t_start
    print(f"\nTotal time: {elapsed_total:.1f}s")

    synth_bars = generate_synthetic(n_bars=20000, seed=args.seed) if ARENA_AVAILABLE else []
    print_final_report(best, history, args.generations, synth_bars)
    save_results(best, history)


if __name__ == "__main__":
    main()
