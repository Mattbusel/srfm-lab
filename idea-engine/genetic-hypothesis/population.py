"""
Population management for genetic hypothesis evolution — handles the
lifecycle of a population of trading hypotheses through generations.

Capabilities:
  - Population dataclass with fitness tracking
  - Island model: multiple sub-populations with periodic migration
  - Niching / fitness sharing for diversity maintenance
  - Hall of Fame: best-ever individuals preserved across generations
  - Speciation: cluster by similarity, evolve species independently
  - Pareto archive of non-dominated solutions
  - Restart detection + random immigrant injection
  - Population statistics and diversity metrics
  - Genealogy tracking: parent-child lineage
  - Elitism: top N% survive to next generation
"""

from __future__ import annotations
import math
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Gene representation (imports from mutation module concept)
# ---------------------------------------------------------------------------

@dataclass
class HypothesisGene:
    """A single hypothesis genotype."""
    gene_id: str
    parameters: Dict[str, float] = field(default_factory=dict)
    signal_components: List[str] = field(default_factory=list)
    active_regimes: List[str] = field(default_factory=list)
    lookback_windows: Dict[str, int] = field(default_factory=dict)
    entry_threshold: float = 0.6
    exit_threshold: float = 0.4
    direction: int = 1
    fitness: float = 0.0
    fitness_vector: Optional[np.ndarray] = None
    generation: int = 0
    species_id: int = -1
    parent_ids: List[str] = field(default_factory=list)

    def clone(self) -> HypothesisGene:
        g = copy.deepcopy(self)
        return g


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

@dataclass
class Population:
    """A collection of hypothesis genes with associated fitness data."""
    individuals: List[HypothesisGene] = field(default_factory=list)
    generation: int = 0
    population_id: str = "main"

    @property
    def size(self) -> int:
        return len(self.individuals)

    def best(self, n: int = 1) -> List[HypothesisGene]:
        ranked = sorted(self.individuals, key=lambda g: g.fitness, reverse=True)
        return ranked[:n]

    def worst(self, n: int = 1) -> List[HypothesisGene]:
        ranked = sorted(self.individuals, key=lambda g: g.fitness)
        return ranked[:n]

    def mean_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return float(np.mean([g.fitness for g in self.individuals]))

    def fitness_std(self) -> float:
        if len(self.individuals) < 2:
            return 0.0
        return float(np.std([g.fitness for g in self.individuals]))

    def add(self, gene: HypothesisGene) -> None:
        self.individuals.append(gene)

    def remove_worst(self, n: int = 1) -> List[HypothesisGene]:
        self.individuals.sort(key=lambda g: g.fitness)
        removed = self.individuals[:n]
        self.individuals = self.individuals[n:]
        return removed


# ---------------------------------------------------------------------------
# Elitism
# ---------------------------------------------------------------------------

class Elitism:
    """Preserve the top N% of the population unchanged to next generation."""

    def __init__(self, elite_fraction: float = 0.10):
        self.elite_fraction = elite_fraction

    def select_elites(self, population: Population) -> List[HypothesisGene]:
        n_elite = max(1, int(population.size * self.elite_fraction))
        return population.best(n_elite)

    def apply(self, old_pop: Population, new_pop: Population) -> Population:
        """Inject elites from old population into new population."""
        elites = self.select_elites(old_pop)
        # Replace worst in new_pop with elites
        result = Population(
            individuals=list(new_pop.individuals),
            generation=new_pop.generation,
            population_id=new_pop.population_id,
        )
        n_replace = min(len(elites), result.size)
        if n_replace > 0:
            result.remove_worst(n_replace)
            for e in elites[:n_replace]:
                clone = e.clone()
                clone.generation = new_pop.generation
                result.add(clone)
        return result


# ---------------------------------------------------------------------------
# Hall of Fame
# ---------------------------------------------------------------------------

class HallOfFame:
    """Maintain the best-ever individuals across all generations."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.members: List[HypothesisGene] = []
        self._seen_ids: set = set()

    def update(self, population: Population) -> int:
        """Add qualifying individuals. Returns number of new additions."""
        added = 0
        for gene in population.individuals:
            if gene.gene_id in self._seen_ids:
                continue
            if len(self.members) < self.max_size:
                self.members.append(gene.clone())
                self._seen_ids.add(gene.gene_id)
                added += 1
            else:
                worst_idx = min(range(len(self.members)),
                                key=lambda i: self.members[i].fitness)
                if gene.fitness > self.members[worst_idx].fitness:
                    old_id = self.members[worst_idx].gene_id
                    self._seen_ids.discard(old_id)
                    self.members[worst_idx] = gene.clone()
                    self._seen_ids.add(gene.gene_id)
                    added += 1
        self.members.sort(key=lambda g: g.fitness, reverse=True)
        return added

    def best(self, n: int = 1) -> List[HypothesisGene]:
        return self.members[:n]

    @property
    def size(self) -> int:
        return len(self.members)

    def summary(self) -> Dict[str, Any]:
        if not self.members:
            return {"size": 0, "best_fitness": 0.0, "worst_fitness": 0.0}
        return {
            "size": self.size,
            "best_fitness": self.members[0].fitness,
            "worst_fitness": self.members[-1].fitness,
            "mean_fitness": float(np.mean([m.fitness for m in self.members])),
        }


# ---------------------------------------------------------------------------
# Island model
# ---------------------------------------------------------------------------

class IslandModel:
    """
    Multiple sub-populations evolving independently with periodic migration.
    """

    def __init__(self, n_islands: int = 4, migration_rate: float = 0.05,
                 migration_interval: int = 10, topology: str = "ring",
                 seed: int = 42):
        self.n_islands = n_islands
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.topology = topology
        self.rng = np.random.default_rng(seed)
        self.islands: List[Population] = []
        self._generation = 0

    def initialize(self, total_population: Population) -> None:
        """Split a total population into island sub-populations."""
        n = total_population.size
        per_island = n // self.n_islands
        self.islands = []
        individuals = list(total_population.individuals)
        self.rng.shuffle(individuals)

        for i in range(self.n_islands):
            start = i * per_island
            end = start + per_island if i < self.n_islands - 1 else n
            island = Population(
                individuals=individuals[start:end],
                generation=total_population.generation,
                population_id=f"island_{i}",
            )
            self.islands.append(island)

    def _migration_pairs(self) -> List[Tuple[int, int]]:
        """Get migration pairs based on topology."""
        pairs: List[Tuple[int, int]] = []
        if self.topology == "ring":
            for i in range(self.n_islands):
                j = (i + 1) % self.n_islands
                pairs.append((i, j))
        elif self.topology == "fully_connected":
            for i in range(self.n_islands):
                for j in range(self.n_islands):
                    if i != j:
                        pairs.append((i, j))
        elif self.topology == "star":
            # Island 0 is the hub
            for i in range(1, self.n_islands):
                pairs.append((0, i))
                pairs.append((i, 0))
        return pairs

    def migrate(self) -> int:
        """Perform migration between islands. Returns total migrants moved."""
        self._generation += 1
        if self._generation % self.migration_interval != 0:
            return 0

        total_migrated = 0
        pairs = self._migration_pairs()

        for src_idx, dst_idx in pairs:
            src = self.islands[src_idx]
            dst = self.islands[dst_idx]
            n_migrate = max(1, int(src.size * self.migration_rate))

            # Send best from src to dst
            migrants = src.best(n_migrate)
            for m in migrants:
                clone = m.clone()
                clone.generation = self._generation
                dst.add(clone)
            # Remove worst from dst to maintain size
            dst.remove_worst(n_migrate)
            total_migrated += n_migrate

        return total_migrated

    def combined_population(self) -> Population:
        """Merge all islands back into one population."""
        all_individuals: List[HypothesisGene] = []
        for island in self.islands:
            all_individuals.extend(island.individuals)
        return Population(
            individuals=all_individuals,
            generation=self._generation,
            population_id="merged",
        )

    def island_stats(self) -> List[Dict[str, Any]]:
        """Summary statistics per island."""
        stats: List[Dict[str, Any]] = []
        for island in self.islands:
            stats.append({
                "id": island.population_id,
                "size": island.size,
                "mean_fitness": island.mean_fitness(),
                "best_fitness": island.best(1)[0].fitness if island.size > 0 else 0.0,
                "fitness_std": island.fitness_std(),
            })
        return stats


# ---------------------------------------------------------------------------
# Fitness sharing / Niching
# ---------------------------------------------------------------------------

class FitnessSharing:
    """
    Fitness sharing to maintain diversity: reduce fitness of individuals
    in crowded regions of the search space.
    """

    def __init__(self, sigma_share: float = 0.5, alpha: float = 1.0):
        self.sigma_share = sigma_share
        self.alpha = alpha

    def distance(self, a: HypothesisGene, b: HypothesisGene) -> float:
        """Compute genotypic distance between two hypotheses."""
        # Parameter distance
        common_params = set(a.parameters) & set(b.parameters)
        if common_params:
            param_diffs = [(a.parameters[p] - b.parameters.get(p, 0.0)) ** 2
                           for p in common_params]
            param_dist = math.sqrt(sum(param_diffs) / len(common_params))
        else:
            param_dist = 1.0

        # Signal component overlap
        set_a = set(a.signal_components)
        set_b = set(b.signal_components)
        if set_a or set_b:
            jaccard = 1.0 - len(set_a & set_b) / max(len(set_a | set_b), 1)
        else:
            jaccard = 0.0

        # Regime overlap
        reg_a = set(a.active_regimes)
        reg_b = set(b.active_regimes)
        if reg_a or reg_b:
            regime_dist = 1.0 - len(reg_a & reg_b) / max(len(reg_a | reg_b), 1)
        else:
            regime_dist = 0.0

        return 0.5 * param_dist + 0.3 * jaccard + 0.2 * regime_dist

    def sharing_function(self, dist: float) -> float:
        if dist < self.sigma_share:
            return 1.0 - (dist / self.sigma_share) ** self.alpha
        return 0.0

    def apply(self, population: Population) -> None:
        """Modify fitness values in-place via fitness sharing."""
        n = population.size
        for i in range(n):
            niche_count = 0.0
            for j in range(n):
                d = self.distance(population.individuals[i],
                                  population.individuals[j])
                niche_count += self.sharing_function(d)
            if niche_count > 0:
                population.individuals[i].fitness /= niche_count


# ---------------------------------------------------------------------------
# Speciation
# ---------------------------------------------------------------------------

class Speciation:
    """
    Cluster hypotheses by similarity and evolve species independently.
    """

    def __init__(self, compatibility_threshold: float = 0.4,
                 seed: int = 42):
        self.threshold = compatibility_threshold
        self.rng = np.random.default_rng(seed)
        self._sharing = FitnessSharing(sigma_share=self.threshold)
        self.species: Dict[int, List[int]] = {}  # species_id -> member indices
        self._next_species_id = 0

    def speciate(self, population: Population) -> Dict[int, List[int]]:
        """Assign species IDs to all individuals."""
        representatives: Dict[int, HypothesisGene] = {}
        self.species = {}

        for idx, gene in enumerate(population.individuals):
            assigned = False
            for sp_id, rep in representatives.items():
                dist = self._sharing.distance(gene, rep)
                if dist < self.threshold:
                    self.species.setdefault(sp_id, []).append(idx)
                    gene.species_id = sp_id
                    assigned = True
                    break

            if not assigned:
                sp_id = self._next_species_id
                self._next_species_id += 1
                representatives[sp_id] = gene
                self.species[sp_id] = [idx]
                gene.species_id = sp_id

        return self.species

    def species_summary(self, population: Population) -> List[Dict[str, Any]]:
        """Summary of each species."""
        summaries: List[Dict[str, Any]] = []
        for sp_id, indices in self.species.items():
            members = [population.individuals[i] for i in indices]
            fitnesses = [m.fitness for m in members]
            summaries.append({
                "species_id": sp_id,
                "size": len(members),
                "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
                "max_fitness": float(np.max(fitnesses)) if fitnesses else 0.0,
                "signals": list({s for m in members for s in m.signal_components}),
            })
        return summaries

    @property
    def n_species(self) -> int:
        return len(self.species)


# ---------------------------------------------------------------------------
# Pareto archive
# ---------------------------------------------------------------------------

class ParetoArchive:
    """Maintain an archive of non-dominated solutions."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.archive: List[HypothesisGene] = []

    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.all(a >= b) and np.any(a > b))

    def update(self, population: Population) -> int:
        """Add non-dominated individuals to archive. Returns new additions."""
        added = 0
        for gene in population.individuals:
            if gene.fitness_vector is None:
                continue
            vec = gene.fitness_vector
            # Check if dominated by any archive member
            dominated = False
            to_remove: List[int] = []
            for i, member in enumerate(self.archive):
                if member.fitness_vector is None:
                    continue
                if self._dominates(member.fitness_vector, vec):
                    dominated = True
                    break
                if self._dominates(vec, member.fitness_vector):
                    to_remove.append(i)

            if not dominated:
                for i in sorted(to_remove, reverse=True):
                    self.archive.pop(i)
                self.archive.append(gene.clone())
                added += 1

        # Trim if over max_size (remove most crowded)
        while len(self.archive) > self.max_size:
            self._trim_most_crowded()

        return added

    def _trim_most_crowded(self) -> None:
        """Remove the individual with the smallest crowding distance."""
        if len(self.archive) <= 2:
            return
        vectors = [m.fitness_vector for m in self.archive
                   if m.fitness_vector is not None]
        if len(vectors) < 3:
            return

        n = len(vectors)
        n_obj = vectors[0].shape[0]
        distances = np.zeros(n)
        for m in range(n_obj):
            vals = np.array([v[m] for v in vectors])
            idx = np.argsort(vals)
            distances[idx[0]] = float("inf")
            distances[idx[-1]] = float("inf")
            spread = vals[idx[-1]] - vals[idx[0]]
            if spread < 1e-12:
                continue
            for i in range(1, n - 1):
                distances[idx[i]] += (vals[idx[i + 1]] - vals[idx[i - 1]]) / spread

        # Remove the one with smallest finite distance
        finite_mask = np.isfinite(distances)
        if np.any(finite_mask):
            finite_indices = np.where(finite_mask)[0]
            worst = finite_indices[np.argmin(distances[finite_mask])]
            self.archive.pop(int(worst))

    @property
    def size(self) -> int:
        return len(self.archive)

    def best_by_objective(self, obj_index: int) -> Optional[HypothesisGene]:
        """Return archive member with best value for a given objective."""
        best_gene = None
        best_val = -float("inf")
        for m in self.archive:
            if m.fitness_vector is not None and m.fitness_vector[obj_index] > best_val:
                best_val = m.fitness_vector[obj_index]
                best_gene = m
        return best_gene


# ---------------------------------------------------------------------------
# Restart detection + random immigrants
# ---------------------------------------------------------------------------

class RestartDetector:
    """Detect population convergence and inject random immigrants."""

    def __init__(self, diversity_threshold: float = 0.05,
                 stagnation_generations: int = 10,
                 immigrant_fraction: float = 0.20,
                 seed: int = 42):
        self.diversity_threshold = diversity_threshold
        self.stagnation_gens = stagnation_generations
        self.immigrant_fraction = immigrant_fraction
        self.rng = np.random.default_rng(seed)
        self.fitness_history: List[float] = []

    def check(self, population: Population) -> bool:
        """Return True if population has converged and needs restart."""
        self.fitness_history.append(population.mean_fitness())

        # Check fitness diversity
        if population.fitness_std() < self.diversity_threshold:
            return True

        # Check stagnation
        if len(self.fitness_history) >= self.stagnation_gens:
            window = self.fitness_history[-self.stagnation_gens:]
            if abs(max(window) - min(window)) < self.diversity_threshold:
                return True

        return False

    def inject_immigrants(self, population: Population,
                          gene_factory: Callable[[], HypothesisGene]) -> int:
        """Replace worst individuals with random immigrants."""
        n_immigrants = max(1, int(population.size * self.immigrant_fraction))
        population.remove_worst(n_immigrants)

        for _ in range(n_immigrants):
            immigrant = gene_factory()
            immigrant.generation = population.generation
            population.add(immigrant)

        return n_immigrants


# ---------------------------------------------------------------------------
# Population statistics & diversity
# ---------------------------------------------------------------------------

class PopulationStats:
    """Track and compute population statistics across generations."""

    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def record(self, population: Population) -> Dict[str, float]:
        """Record statistics for the current generation."""
        fitnesses = [g.fitness for g in population.individuals]
        if not fitnesses:
            stats = {"generation": population.generation, "size": 0}
            self.history.append(stats)
            return stats

        stats = {
            "generation": float(population.generation),
            "size": float(population.size),
            "mean_fitness": float(np.mean(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "median_fitness": float(np.median(fitnesses)),
            "diversity": self._genotype_diversity(population),
            "signal_diversity": self._signal_diversity(population),
            "regime_diversity": self._regime_diversity(population),
        }
        self.history.append(stats)
        return stats

    def _genotype_diversity(self, population: Population) -> float:
        """Average pairwise parameter distance (sampled)."""
        n = population.size
        if n < 2:
            return 0.0
        sample_size = min(n, 30)
        indices = np.random.choice(n, size=sample_size, replace=False)
        total_dist = 0.0
        count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                a = population.individuals[indices[i]]
                b = population.individuals[indices[j]]
                common = set(a.parameters) & set(b.parameters)
                if common:
                    d = math.sqrt(sum((a.parameters[p] - b.parameters.get(p, 0)) ** 2
                                      for p in common) / len(common))
                    total_dist += d
                    count += 1
        return total_dist / count if count > 0 else 0.0

    def _signal_diversity(self, population: Population) -> float:
        """Number of unique signal component sets / population size."""
        unique = set()
        for g in population.individuals:
            unique.add(frozenset(g.signal_components))
        return len(unique) / max(population.size, 1)

    def _regime_diversity(self, population: Population) -> float:
        """Number of unique regime configurations / population size."""
        unique = set()
        for g in population.individuals:
            unique.add(frozenset(g.active_regimes))
        return len(unique) / max(population.size, 1)

    def convergence_rate(self, window: int = 10) -> float:
        """Rate of fitness improvement over last N generations."""
        if len(self.history) < window:
            return 0.0
        recent = self.history[-window:]
        first = recent[0].get("max_fitness", 0.0)
        last = recent[-1].get("max_fitness", 0.0)
        return (last - first) / window

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"generations": 0}
        latest = self.history[-1]
        return {
            "generations": len(self.history),
            "latest": latest,
            "convergence_rate": self.convergence_rate(),
        }


# ---------------------------------------------------------------------------
# Genealogy tracking
# ---------------------------------------------------------------------------

class Genealogy:
    """Track parent-child relationships across generations."""

    def __init__(self):
        self.parents: Dict[str, List[str]] = {}     # child_id -> parent_ids
        self.children: Dict[str, List[str]] = {}     # parent_id -> child_ids
        self.generation_map: Dict[str, int] = {}     # gene_id -> generation

    def record_birth(self, child: HypothesisGene,
                     parents: Optional[List[HypothesisGene]] = None) -> None:
        """Record a new individual and its parents."""
        parent_ids = [p.gene_id for p in (parents or [])]
        self.parents[child.gene_id] = parent_ids
        self.generation_map[child.gene_id] = child.generation

        for pid in parent_ids:
            if pid not in self.children:
                self.children[pid] = []
            self.children[pid].append(child.gene_id)

    def lineage(self, gene_id: str, max_depth: int = 10) -> List[List[str]]:
        """Trace ancestry back max_depth generations."""
        result: List[List[str]] = [[gene_id]]
        current = [gene_id]
        for _ in range(max_depth):
            ancestors: List[str] = []
            for gid in current:
                ancestors.extend(self.parents.get(gid, []))
            if not ancestors:
                break
            result.append(ancestors)
            current = ancestors
        return result

    def descendants(self, gene_id: str, max_depth: int = 10) -> List[List[str]]:
        """Find all descendants up to max_depth generations."""
        result: List[List[str]] = [[gene_id]]
        current = [gene_id]
        for _ in range(max_depth):
            offspring: List[str] = []
            for gid in current:
                offspring.extend(self.children.get(gid, []))
            if not offspring:
                break
            result.append(offspring)
            current = offspring
        return result

    def common_ancestor(self, id_a: str, id_b: str,
                        max_depth: int = 20) -> Optional[str]:
        """Find most recent common ancestor of two individuals."""
        ancestors_a: set = set()
        current_a = {id_a}
        for _ in range(max_depth):
            ancestors_a.update(current_a)
            next_a: set = set()
            for gid in current_a:
                next_a.update(self.parents.get(gid, []))
            if not next_a:
                break
            current_a = next_a

        current_b = {id_b}
        for _ in range(max_depth):
            for gid in current_b:
                if gid in ancestors_a:
                    return gid
            next_b: set = set()
            for gid in current_b:
                next_b.update(self.parents.get(gid, []))
            if not next_b:
                break
            current_b = next_b

        return None

    def total_individuals(self) -> int:
        return len(self.generation_map)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_individuals": self.total_individuals(),
            "n_with_parents": sum(1 for v in self.parents.values() if v),
            "n_with_children": sum(1 for v in self.children.values() if v),
            "generations_tracked": (max(self.generation_map.values())
                                    if self.generation_map else 0),
        }


# ---------------------------------------------------------------------------
# Convenience: full population management suite
# ---------------------------------------------------------------------------

def create_population_manager(pop_size: int = 100, n_islands: int = 4,
                              seed: int = 42) -> Dict[str, Any]:
    """Create all population management components."""
    return {
        "elitism": Elitism(elite_fraction=0.10),
        "hall_of_fame": HallOfFame(max_size=50),
        "island_model": IslandModel(n_islands=n_islands, seed=seed),
        "fitness_sharing": FitnessSharing(),
        "speciation": Speciation(seed=seed),
        "pareto_archive": ParetoArchive(max_size=100),
        "restart_detector": RestartDetector(seed=seed),
        "stats": PopulationStats(),
        "genealogy": Genealogy(),
    }
