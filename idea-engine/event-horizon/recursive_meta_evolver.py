"""
Recursive Meta-Evolutionary Architecture (RMEA): the self-improving final boss.

A hierarchical system where each layer evolves the layer below it:

Layer 0: Market Data -> Raw signals (BH physics, technicals, microstructure)
Layer 1: EHS discovers new physics-based signals from concept library
Layer 2: Market Consciousness detects emergent beliefs from agent consensus
Layer 3: Genetic Algorithm evolves signal weights and parameters
Layer 4: Meta-Reward Evolver optimizes the reward function for Layer 3
Layer 5: Hyper-Genome Controller evolves the evolution parameters themselves
Layer 6: Red Queen Engine co-evolves adversarial market regimes
Layer 7: Serendipity Injector mutates the search space from cross-domain analogies

The fitness of each layer is determined by the performance of the layer below it.
This creates a recursive optimization where the system literally evolves
its own ability to evolve.

Integration points:
  - event-horizon/synthesizer.py (EHS)
  - event-horizon/market_consciousness.py
  - event-horizon/spacetime_arbitrage.py
  - ml/genetic/meta_reward_evolution.py
  - ml/genetic/evolution.py
  - ml/genetic/coevolution.py
  - idea-engine/serendipity/cross_domain_mapper.py
  - idea-engine/meta/self_improving_engine.py
"""

from __future__ import annotations
import math
import time
import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hyper-Genome: evolves the parameters of the evolutionary process itself
# ---------------------------------------------------------------------------

@dataclass
class HyperGenome:
    """
    The genome of the evolutionary process.

    Instead of evolving trading parameters directly, this evolves HOW
    the system searches for trading parameters. It's evolution^2.
    """
    genome_id: str = ""
    generation: int = 0

    # Layer 3: GA parameters (how the signal evolution runs)
    population_size: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    tournament_size: int = 5
    elite_fraction: float = 0.1
    sigma_scale: float = 0.1

    # Layer 4: Reward function weights (what the RL agent optimizes for)
    reward_w_pnl: float = 1.0
    reward_w_sharpe: float = 0.5
    reward_w_drawdown: float = 1.0
    reward_w_turnover: float = 0.3
    reward_risk_aversion: float = 2.0

    # Layer 5: Search space parameters (where the GA looks)
    param_bound_expansion: float = 1.0   # how wide to search (1.0 = default, 2.0 = 2x wider)
    novelty_weight: float = 0.3          # how much to reward novelty vs fitness
    diversity_injection_rate: float = 0.05

    # Layer 6: Adversarial parameters (how hard the Red Queen pushes)
    adversarial_intensity: float = 0.5   # 0=no adversary, 1=maximum pressure
    adversarial_mutation_rate: float = 0.2
    regime_diversity_target: int = 5     # how many distinct regimes to co-evolve

    # Layer 7: Serendipity parameters (how often and how far to inject analogies)
    serendipity_interval: int = 10       # generations between serendipity events
    serendipity_magnitude: float = 0.5   # how much analogies perturb the search
    analogy_pool_size: int = 5           # how many cross-domain analogies to draw from

    # Fitness
    fitness: float = 0.0
    child_layer_sharpe: float = 0.0
    child_layer_alpha_decay_rate: float = 0.0
    child_layer_signal_diversity: float = 0.0


# ---------------------------------------------------------------------------
# Red Queen Engine: co-evolve adversarial market regimes
# ---------------------------------------------------------------------------

@dataclass
class SyntheticRegime:
    """A synthetic market regime designed to stress-test signals."""
    regime_id: str = ""
    volatility: float = 0.02           # daily vol
    mean_reversion_speed: float = 0.0  # 0 = random walk, >0 = mean reverting
    trend_strength: float = 0.0        # drift per day
    jump_intensity: float = 0.0        # jumps per day (Poisson)
    jump_size: float = 0.0             # avg jump magnitude
    correlation_structure: str = "normal"  # normal / herding / dispersed
    duration_bars: int = 252

    def generate_returns(self, seed: int = 42) -> np.ndarray:
        """Generate synthetic returns from this regime's parameters."""
        rng = np.random.default_rng(seed)
        n = self.duration_bars
        returns = np.zeros(n)

        for t in range(n):
            # Base: GBM
            base = rng.normal(self.trend_strength / 252, self.volatility / math.sqrt(252))

            # Mean reversion component
            if t > 0 and self.mean_reversion_speed > 0:
                base -= self.mean_reversion_speed * returns[t-1]

            # Jump component
            if rng.random() < self.jump_intensity / 252:
                base += rng.normal(0, self.jump_size)

            returns[t] = base

        return returns


class RedQueenEngine:
    """
    Co-evolves synthetic market regimes designed to break the current
    best-performing signals.

    Named after the Red Queen Hypothesis: "It takes all the running you
    can do, to keep in the same place."

    The adversary's fitness is the INVERSE of the signal's performance.
    This creates an arms race that forces signals to become regime-invariant.
    """

    def __init__(self, n_regimes: int = 10, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._counter = 0
        self.population: List[SyntheticRegime] = []
        self._init_population(n_regimes)

    def _next_id(self) -> str:
        self._counter += 1
        return f"regime_{self._counter:04d}"

    def _init_population(self, n: int) -> None:
        """Initialize diverse regime population."""
        archetypes = [
            {"volatility": 0.01, "trend_strength": 0.10, "mean_reversion_speed": 0.0},   # strong trend
            {"volatility": 0.03, "trend_strength": -0.15, "mean_reversion_speed": 0.0},   # bear market
            {"volatility": 0.02, "trend_strength": 0.0, "mean_reversion_speed": 0.1},     # mean reverting
            {"volatility": 0.05, "trend_strength": 0.0, "mean_reversion_speed": 0.0},     # high vol random
            {"volatility": 0.02, "jump_intensity": 5.0, "jump_size": 0.03},               # jump diffusion
            {"volatility": 0.01, "trend_strength": 0.0, "mean_reversion_speed": 0.0},     # dead market
            {"volatility": 0.08, "trend_strength": -0.30, "mean_reversion_speed": 0.0},   # crisis
        ]

        for i in range(n):
            arch = archetypes[i % len(archetypes)]
            regime = SyntheticRegime(regime_id=self._next_id(), **arch)
            # Add noise
            regime.volatility *= self.rng.uniform(0.7, 1.3)
            regime.trend_strength *= self.rng.uniform(0.5, 1.5)
            self.population.append(regime)

    def evolve_against(
        self,
        signal_evaluator: Callable[[np.ndarray], float],
        n_generations: int = 5,
    ) -> List[SyntheticRegime]:
        """
        Evolve regimes to maximize signal failure.

        signal_evaluator: function that takes returns array and returns signal's Sharpe.
        Lower Sharpe = regime is succeeding at breaking the signal.
        """
        for gen in range(n_generations):
            # Evaluate: regime fitness = inverse of signal performance
            fitnesses = []
            for regime in self.population:
                returns = regime.generate_returns(seed=gen * 1000 + hash(regime.regime_id) % 1000)
                signal_sharpe = signal_evaluator(returns)
                regime_fitness = -signal_sharpe  # adversary wants LOW signal Sharpe
                fitnesses.append(regime_fitness)

            # Sort by fitness (highest = most damaging to signal)
            paired = list(zip(self.population, fitnesses))
            paired.sort(key=lambda x: x[1], reverse=True)
            self.population = [p[0] for p in paired]

            # Evolve: keep top 30%, mutate rest
            elite_n = max(2, len(self.population) // 3)
            new_pop = list(self.population[:elite_n])

            while len(new_pop) < len(self.population):
                parent = self.rng.choice(self.population[:elite_n])
                child = copy.deepcopy(parent)
                child.regime_id = self._next_id()

                # Mutate
                child.volatility *= self.rng.uniform(0.8, 1.2)
                child.trend_strength += self.rng.gauss(0, 0.02)
                child.mean_reversion_speed = max(0, child.mean_reversion_speed + self.rng.gauss(0, 0.02))
                child.jump_intensity = max(0, child.jump_intensity + self.rng.gauss(0, 0.5))
                new_pop.append(child)

            self.population = new_pop

        return self.population

    def get_hardest_regimes(self, n: int = 3) -> List[SyntheticRegime]:
        """Get the regimes that are hardest for the current signal."""
        return self.population[:n]


# ---------------------------------------------------------------------------
# Serendipity Injector: perturb the search space with cross-domain analogies
# ---------------------------------------------------------------------------

ANALOGY_PERTURBATIONS = [
    {
        "name": "Predator-Prey Oscillation",
        "effect": "Increase mean_reversion weight by 2x, decrease momentum weight",
        "param_adjustments": {"novelty_weight": 0.6, "param_bound_expansion": 1.5},
    },
    {
        "name": "Quantum Tunneling",
        "effect": "Allow search to jump over local optima barriers",
        "param_adjustments": {"mutation_rate": 0.4, "sigma_scale": 0.3},
    },
    {
        "name": "Evolutionary Arms Race",
        "effect": "Increase adversarial pressure dramatically",
        "param_adjustments": {"adversarial_intensity": 0.9, "adversarial_mutation_rate": 0.3},
    },
    {
        "name": "Phase Transition (Critical Slowing)",
        "effect": "Slow down evolution, increase precision near optimum",
        "param_adjustments": {"mutation_rate": 0.05, "sigma_scale": 0.03, "elite_fraction": 0.3},
    },
    {
        "name": "Genetic Drift (Small Population)",
        "effect": "Reduce population to increase random exploration",
        "param_adjustments": {"population_size": 20, "diversity_injection_rate": 0.15},
    },
    {
        "name": "Horizontal Gene Transfer",
        "effect": "Cross signals from completely different domains",
        "param_adjustments": {"crossover_rate": 0.95, "novelty_weight": 0.7},
    },
    {
        "name": "Cambrian Explosion",
        "effect": "Massively expand search space and diversity",
        "param_adjustments": {"param_bound_expansion": 3.0, "population_size": 200,
                               "diversity_injection_rate": 0.20},
    },
]


class SerendipityInjector:
    """
    Periodically inject cross-domain analogies into the evolutionary process,
    forcing the search to explore regions it would never reach through
    incremental optimization alone.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._injection_history: List[Dict] = []

    def should_inject(self, generation: int, interval: int = 10,
                       stagnation_detected: bool = False) -> bool:
        """Inject on schedule or when stagnation is detected."""
        if stagnation_detected:
            return True
        return generation > 0 and generation % interval == 0

    def inject(self, hyper_genome: HyperGenome) -> Tuple[HyperGenome, Dict]:
        """
        Apply a random analogy perturbation to the hyper-genome.
        Returns the modified genome and the injection record.
        """
        analogy = self.rng.choice(ANALOGY_PERTURBATIONS)
        modified = copy.deepcopy(hyper_genome)

        for param, value in analogy["param_adjustments"].items():
            if hasattr(modified, param):
                setattr(modified, param, value)

        record = {
            "analogy": analogy["name"],
            "effect": analogy["effect"],
            "adjustments": analogy["param_adjustments"],
            "generation": hyper_genome.generation,
            "timestamp": time.time(),
        }
        self._injection_history.append(record)

        return modified, record


# ---------------------------------------------------------------------------
# Recursive Meta-Evolver: the master orchestrator
# ---------------------------------------------------------------------------

class RecursiveMetaEvolver:
    """
    The self-improving autonomous research system.

    Orchestrates the recursive meta-evolutionary loop:
      1. Hyper-Genome Controller evolves evolution parameters
      2. For each hyper-genome:
         a. Configure the signal evolution engine with these params
         b. Run the EHS to discover new physics signals
         c. Run the Red Queen to co-evolve adversarial regimes
         d. Evaluate signal performance on both real and adversarial data
         e. Use Market Consciousness to detect emergent beliefs
         f. Apply Serendipity Injector if stagnating
      3. Hyper-genome fitness = aggregate performance of everything below
      4. Evolve the hyper-genomes
      5. Repeat

    The system literally evolves its own ability to evolve.
    """

    def __init__(self, n_hyper_genomes: int = 10, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.red_queen = RedQueenEngine(seed=seed)
        self.serendipity = SerendipityInjector(seed=seed)

        # Hyper-genome population
        self._counter = 0
        self.population: List[HyperGenome] = []
        self._init_population(n_hyper_genomes)

        self.generation = 0
        self.history: List[Dict] = []
        self.best_ever: Optional[HyperGenome] = None

    def _next_id(self) -> str:
        self._counter += 1
        return f"hyper_{self._counter:04d}"

    def _init_population(self, n: int) -> None:
        for _ in range(n):
            hg = HyperGenome(genome_id=self._next_id())
            # Randomize
            hg.population_size = self.rng.randint(20, 100)
            hg.mutation_rate = self.rng.uniform(0.05, 0.3)
            hg.crossover_rate = self.rng.uniform(0.5, 0.95)
            hg.reward_w_pnl = self.rng.uniform(0.5, 3.0)
            hg.reward_w_sharpe = self.rng.uniform(0.0, 2.0)
            hg.reward_w_drawdown = self.rng.uniform(0.5, 3.0)
            hg.reward_risk_aversion = self.rng.uniform(1.0, 5.0)
            hg.adversarial_intensity = self.rng.uniform(0.1, 0.8)
            hg.novelty_weight = self.rng.uniform(0.1, 0.5)
            self.population.append(hg)

    def evaluate_hyper_genome(
        self,
        hg: HyperGenome,
        market_returns: np.ndarray,
    ) -> float:
        """
        Evaluate a hyper-genome by running the entire stack below it.

        This is the recursive fitness evaluation:
        hyper_genome_fitness = f(signal_evolution_fitness(hg.params))
        """
        # Simulate signal evolution with this hyper-genome's parameters
        # (simplified: use the hyper-genome's reward weights to evaluate returns)

        # 1. Generate a signal from the hyper-genome's reward weights
        signal = np.zeros_like(market_returns)
        lookback = max(5, int(hg.population_size / 5))
        for t in range(lookback, len(market_returns)):
            window = market_returns[t - lookback:t]
            momentum = window.mean() * hg.reward_w_pnl
            vol = window.std() * hg.reward_w_drawdown
            if vol > 1e-10:
                signal[t] = np.tanh(momentum / vol * hg.reward_w_sharpe)

        # 2. Compute strategy returns
        strategy_returns = signal[:-1] * market_returns[1:]
        cost = np.abs(np.diff(signal, prepend=0))[:-1] * 0.001 * hg.reward_w_turnover
        net_returns = strategy_returns - cost

        # 3. Real data performance
        if len(net_returns) > 20:
            real_sharpe = float(net_returns.mean() / max(net_returns.std(), 1e-10) * math.sqrt(252))
            real_dd = float(np.maximum.accumulate(np.cumprod(1 + net_returns)).max() -
                           np.cumprod(1 + net_returns)[-1]) if len(net_returns) > 0 else 0
        else:
            real_sharpe = 0.0
            real_dd = 0.0

        # 4. Adversarial performance (Red Queen)
        hardest = self.red_queen.get_hardest_regimes(3)
        adversarial_sharpes = []
        for regime in hardest:
            adv_returns = regime.generate_returns()
            adv_signal = np.zeros_like(adv_returns)
            for t in range(lookback, len(adv_returns)):
                w = adv_returns[t - lookback:t]
                m = w.mean() * hg.reward_w_pnl
                v = w.std() * hg.reward_w_drawdown
                if v > 1e-10:
                    adv_signal[t] = np.tanh(m / v * hg.reward_w_sharpe)
            adv_strat = adv_signal[:-1] * adv_returns[1:]
            if len(adv_strat) > 20:
                adversarial_sharpes.append(
                    float(adv_strat.mean() / max(adv_strat.std(), 1e-10) * math.sqrt(252))
                )

        avg_adversarial = float(np.mean(adversarial_sharpes)) if adversarial_sharpes else 0.0

        # 5. Composite fitness: real performance + adversarial robustness + novelty
        fitness = (
            real_sharpe * 0.5 +
            avg_adversarial * 0.3 * hg.adversarial_intensity +
            hg.novelty_weight * 0.2 * self.rng.uniform(0.5, 1.5)  # diversity bonus
        )

        hg.fitness = fitness
        hg.child_layer_sharpe = real_sharpe
        return fitness

    def evolve_generation(self, market_returns: np.ndarray) -> Dict:
        """Run one generation of hyper-evolution."""
        self.generation += 1
        gen_start = time.time()

        # Evaluate all hyper-genomes
        for hg in self.population:
            self.evaluate_hyper_genome(hg, market_returns)

        # Sort by fitness
        self.population.sort(key=lambda hg: hg.fitness, reverse=True)
        best = self.population[0]

        if self.best_ever is None or best.fitness > self.best_ever.fitness:
            self.best_ever = copy.deepcopy(best)

        # Evolve Red Queen against the best signal
        def signal_eval(returns):
            lookback = max(5, int(best.population_size / 5))
            signal = np.zeros_like(returns)
            for t in range(lookback, len(returns)):
                w = returns[t - lookback:t]
                m = w.mean() * best.reward_w_pnl
                v = w.std() * best.reward_w_drawdown
                if v > 1e-10:
                    signal[t] = np.tanh(m / v * best.reward_w_sharpe)
            strat = signal[:-1] * returns[1:]
            return float(strat.mean() / max(strat.std(), 1e-10) * math.sqrt(252)) if len(strat) > 20 else 0.0

        self.red_queen.evolve_against(signal_eval, n_generations=3)

        # Serendipity check
        stagnation = (len(self.history) >= 3 and
                       all(abs(h["best_fitness"] - best.fitness) < 0.01
                           for h in self.history[-3:]))
        serendipity_event = None
        if self.serendipity.should_inject(self.generation, best.serendipity_interval, stagnation):
            self.population[-1], serendipity_event = self.serendipity.inject(self.population[-1])

        # Create next generation
        elite_n = max(2, len(self.population) // 5)
        next_pop = [copy.deepcopy(hg) for hg in self.population[:elite_n]]

        while len(next_pop) < len(self.population):
            parent = self.rng.choice(self.population[:elite_n])
            child = copy.deepcopy(parent)
            child.genome_id = self._next_id()
            child.generation = self.generation

            # Mutate the hyper-genome
            child.mutation_rate = max(0.01, child.mutation_rate + self.rng.gauss(0, 0.03))
            child.crossover_rate = max(0.1, min(0.99, child.crossover_rate + self.rng.gauss(0, 0.05)))
            child.reward_w_pnl = max(0.1, child.reward_w_pnl + self.rng.gauss(0, 0.2))
            child.reward_w_sharpe = max(0, child.reward_w_sharpe + self.rng.gauss(0, 0.2))
            child.reward_w_drawdown = max(0.1, child.reward_w_drawdown + self.rng.gauss(0, 0.2))
            child.reward_risk_aversion = max(0.5, child.reward_risk_aversion + self.rng.gauss(0, 0.3))
            child.adversarial_intensity = max(0, min(1, child.adversarial_intensity + self.rng.gauss(0, 0.1)))
            child.novelty_weight = max(0, min(1, child.novelty_weight + self.rng.gauss(0, 0.05)))
            next_pop.append(child)

        self.population = next_pop

        stats = {
            "generation": self.generation,
            "best_fitness": float(best.fitness),
            "best_sharpe": float(best.child_layer_sharpe),
            "best_mutation_rate": float(best.mutation_rate),
            "best_adversarial_intensity": float(best.adversarial_intensity),
            "best_novelty_weight": float(best.novelty_weight),
            "mean_fitness": float(np.mean([hg.fitness for hg in self.population])),
            "serendipity_event": serendipity_event["analogy"] if serendipity_event else None,
            "red_queen_hardest_vol": float(self.red_queen.population[0].volatility) if self.red_queen.population else 0,
            "elapsed": time.time() - gen_start,
        }
        self.history.append(stats)
        return stats

    def run(self, market_returns: np.ndarray, n_generations: int = 20,
            verbose: bool = True) -> Dict:
        """Run the full recursive meta-evolution."""
        if verbose:
            print("=" * 70)
            print("RECURSIVE META-EVOLUTIONARY ARCHITECTURE (RMEA)")
            print("The system that evolves its own ability to evolve.")
            print("=" * 70)

        for gen in range(n_generations):
            stats = self.evolve_generation(market_returns)

            if verbose:
                serendipity = f" | Serendipity: {stats['serendipity_event']}" if stats['serendipity_event'] else ""
                print(
                    f"  Gen {gen+1:3d} | "
                    f"Fitness: {stats['best_fitness']:+.3f} | "
                    f"Sharpe: {stats['best_sharpe']:.2f} | "
                    f"MutRate: {stats['best_mutation_rate']:.3f} | "
                    f"Adversarial: {stats['best_adversarial_intensity']:.2f} | "
                    f"Novelty: {stats['best_novelty_weight']:.2f}"
                    f"{serendipity}"
                )

        if verbose:
            print("\n" + "=" * 70)
            print("RMEA COMPLETE")
            best = self.best_ever
            print(f"  Best hyper-genome: {best.genome_id}")
            print(f"  Best fitness: {best.fitness:.4f}")
            print(f"  Evolved parameters:")
            print(f"    mutation_rate:        {best.mutation_rate:.4f}")
            print(f"    crossover_rate:       {best.crossover_rate:.4f}")
            print(f"    reward_w_pnl:         {best.reward_w_pnl:.4f}")
            print(f"    reward_w_sharpe:      {best.reward_w_sharpe:.4f}")
            print(f"    reward_w_drawdown:    {best.reward_w_drawdown:.4f}")
            print(f"    adversarial_intensity:{best.adversarial_intensity:.4f}")
            print(f"    novelty_weight:       {best.novelty_weight:.4f}")
            print(f"    serendipity_interval: {best.serendipity_interval}")

        return {
            "best_hyper_genome": {
                "id": self.best_ever.genome_id,
                "fitness": self.best_ever.fitness,
                "mutation_rate": self.best_ever.mutation_rate,
                "crossover_rate": self.best_ever.crossover_rate,
                "reward_weights": {
                    "pnl": self.best_ever.reward_w_pnl,
                    "sharpe": self.best_ever.reward_w_sharpe,
                    "drawdown": self.best_ever.reward_w_drawdown,
                    "turnover": self.best_ever.reward_w_turnover,
                    "risk_aversion": self.best_ever.reward_risk_aversion,
                },
                "adversarial_intensity": self.best_ever.adversarial_intensity,
                "novelty_weight": self.best_ever.novelty_weight,
            },
            "generation_history": self.history,
            "serendipity_events": self.serendipity._injection_history,
            "red_queen_regimes": len(self.red_queen.population),
        }
