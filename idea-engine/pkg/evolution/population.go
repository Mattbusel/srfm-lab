package evolution

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/google/uuid"
)

// ---------------------------------------------------------------------------
// Population
// ---------------------------------------------------------------------------

// Population holds all individuals for one generation plus summary statistics.
type Population struct {
	// Individuals is the current set of individuals.
	Individuals []Individual
	// Generation is the zero-based generation counter.
	Generation int
	// Best is the individual with the highest WeightedScore in this generation.
	Best Individual
	// Diversity is the normalised average pairwise Euclidean distance.
	// Range [0,1]: 0 = all identical, 1 = maximally spread.
	Diversity float64
}

// ---------------------------------------------------------------------------
// Latin Hypercube Sampling initialisation
// ---------------------------------------------------------------------------

// Initialize creates an initial population of size individuals by sampling
// the search space using Latin Hypercube Sampling (LHS). LHS partitions each
// dimension into size equally spaced strata and ensures exactly one sample
// per stratum per dimension, giving much better coverage than pure random
// initialisation.
//
// bounds[i] = [lo, hi] for the i-th gene. The genome length equals len(bounds).
func Initialize(size int, bounds [][2]float64, generation int) (Population, error) {
	if size <= 0 {
		return Population{}, fmt.Errorf("Initialize: size must be > 0, got %d", size)
	}
	if len(bounds) == 0 {
		return Population{}, fmt.Errorf("Initialize: bounds must not be empty")
	}
	for i, b := range bounds {
		if b[1] <= b[0] {
			return Population{}, fmt.Errorf("Initialize: bounds[%d] has hi <= lo (%g <= %g)", i, b[1], b[0])
		}
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	nDims := len(bounds)

	// Build LHS matrix: lhs[dim][sample] = value in [lo, hi].
	lhs := make([][]float64, nDims)
	for d := 0; d < nDims; d++ {
		lo, hi := bounds[d][0], bounds[d][1]
		step := (hi - lo) / float64(size)
		perm := rng.Perm(size)
		row := make([]float64, size)
		for i, p := range perm {
			// Random point within stratum p.
			row[i] = lo + step*(float64(p)+rng.Float64())
		}
		lhs[d] = row
	}

	individuals := make([]Individual, size)
	for i := 0; i < size; i++ {
		genes := make(Genome, nDims)
		for d := 0; d < nDims; d++ {
			genes[d] = lhs[d][i]
		}
		individuals[i] = Individual{
			ID:         uuid.New().String(),
			Genes:      genes,
			Generation: generation,
		}
	}

	pop := Population{
		Individuals: individuals,
		Generation:  generation,
	}
	pop.Diversity = ComputeDiversity(pop)
	return pop, nil
}

// ---------------------------------------------------------------------------
// Diversity computation
// ---------------------------------------------------------------------------

// ComputeDiversity computes the normalised average pairwise Euclidean distance
// over the population. If the population has fewer than 2 individuals it
// returns 0. Distances are normalised by the square root of genome length so
// that the metric is independent of dimensionality.
func ComputeDiversity(pop Population) float64 {
	n := len(pop.Individuals)
	if n < 2 {
		return 0
	}

	dim := len(pop.Individuals[0].Genes)
	if dim == 0 {
		return 0
	}

	totalDist := 0.0
	count := 0
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			d := euclideanDist(pop.Individuals[i].Genes, pop.Individuals[j].Genes)
			totalDist += d
			count++
		}
	}
	if count == 0 {
		return 0
	}
	avg := totalDist / float64(count)
	// Normalise by sqrt(dim) -- a rough upper bound on pair distance given unit
	// gene ranges. Callers with non-unit ranges should treat diversity as
	// relative, not absolute.
	normalised := avg / math.Sqrt(float64(dim))
	return normalised
}

// euclideanDist computes the L2 distance between two genomes of equal length.
func euclideanDist(a, b Genome) float64 {
	if len(a) != len(b) {
		return 0
	}
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// ---------------------------------------------------------------------------
// Restart-if-stagnant
// ---------------------------------------------------------------------------

// RestartIfStagnant checks whether the population diversity has fallen below
// threshold. If so, it re-initialises the population using the provided
// bounds, preserving the top n_elite individuals from the current population.
//
// Returns the (possibly restarted) population. If diversity >= threshold the
// population is returned unchanged.
func RestartIfStagnant(pop Population, threshold float64, nElite int, bounds [][2]float64) (Population, bool, error) {
	pop.Diversity = ComputeDiversity(pop)
	if pop.Diversity >= threshold {
		return pop, false, nil
	}

	// Preserve elites.
	elites := ElitePreservation(pop.Individuals, nElite)

	// Re-initialise a fresh population.
	newSize := len(pop.Individuals)
	fresh, err := Initialize(newSize-len(elites), bounds, pop.Generation)
	if err != nil {
		return pop, false, fmt.Errorf("RestartIfStagnant re-init: %w", err)
	}

	merged := make([]Individual, 0, newSize)
	merged = append(merged, elites...)
	merged = append(merged, fresh.Individuals...)

	pop.Individuals = merged
	pop.Diversity = ComputeDiversity(pop)
	return pop, true, nil
}

// ---------------------------------------------------------------------------
// Elite archive
// ---------------------------------------------------------------------------

// ArchiveElites merges the top elites from pop into the existing archive,
// de-duplicates by genome ID, and trims to max_size. Returns the updated
// archive sorted by WeightedScore descending.
func ArchiveElites(pop Population, archive []Individual, maxSize int) []Individual {
	// Collect elite candidates from the current population.
	candidates := ElitePreservation(pop.Individuals, len(pop.Individuals))

	// Build a combined set, keyed by genome ID to avoid duplicates.
	seen := make(map[string]bool, len(archive)+len(candidates))
	combined := make([]Individual, 0, len(archive)+len(candidates))
	for _, ind := range archive {
		if !seen[ind.ID] {
			seen[ind.ID] = true
			combined = append(combined, ind)
		}
	}
	for _, ind := range candidates {
		if !seen[ind.ID] {
			seen[ind.ID] = true
			combined = append(combined, ind)
		}
	}

	// Sort by WeightedScore descending.
	sort.Slice(combined, func(i, j int) bool {
		return combined[i].Fitness.WeightedScore > combined[j].Fitness.WeightedScore
	})

	if maxSize > 0 && len(combined) > maxSize {
		combined = combined[:maxSize]
	}
	return combined
}

// ---------------------------------------------------------------------------
// Population helpers
// ---------------------------------------------------------------------------

// UpdateBest scans the population and updates pop.Best to the individual with
// the highest WeightedScore.
func UpdateBest(pop *Population) {
	if len(pop.Individuals) == 0 {
		return
	}
	best := pop.Individuals[0]
	for _, ind := range pop.Individuals[1:] {
		if ind.Fitness.WeightedScore > best.Fitness.WeightedScore {
			best = ind
		}
	}
	pop.Best = best
}

// NewIndividual creates a new Individual with a fresh UUID from the given genes
// and generation number.
func NewIndividual(genes Genome, generation int, parentIDs []string) Individual {
	return Individual{
		ID:         uuid.New().String(),
		Genes:      genes,
		Generation: generation,
		ParentIDs:  parentIDs,
	}
}
