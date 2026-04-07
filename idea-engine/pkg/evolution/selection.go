package evolution

import (
	"math"
	"math/rand"
	"sort"
)

// ---------------------------------------------------------------------------
// Tournament selection
// ---------------------------------------------------------------------------

// TournamentSelection runs a k-way tournament over pop and returns the
// individual with the highest fitness among the k randomly chosen contestants.
//
// k=2 is binary tournament (default); larger k applies more selection pressure.
// If k <= 0 or k > len(pop) it is clamped to len(pop).
func TournamentSelection(pop []Individual, k int) Individual {
	if len(pop) == 0 {
		panic("TournamentSelection: empty population")
	}
	if k <= 0 || k > len(pop) {
		k = len(pop)
	}
	best := pop[rand.Intn(len(pop))]
	for i := 1; i < k; i++ {
		cand := pop[rand.Intn(len(pop))]
		if cand.Fitness.WeightedScore > best.Fitness.WeightedScore {
			best = cand
		}
	}
	return best
}

// ---------------------------------------------------------------------------
// Roulette wheel (fitness-proportional) selection
// ---------------------------------------------------------------------------

// RouletteWheelSelection selects one individual with probability proportional
// to its weighted fitness score. All fitness values are shifted so the minimum
// is zero before computing cumulative probabilities, preventing negative
// selections.
//
// If all individuals have equal fitness one is chosen uniformly at random.
func RouletteWheelSelection(pop []Individual) Individual {
	if len(pop) == 0 {
		panic("RouletteWheelSelection: empty population")
	}

	minFit := pop[0].Fitness.WeightedScore
	for _, ind := range pop {
		if ind.Fitness.WeightedScore < minFit {
			minFit = ind.Fitness.WeightedScore
		}
	}

	total := 0.0
	for _, ind := range pop {
		total += ind.Fitness.WeightedScore - minFit
	}
	if total == 0 {
		return pop[rand.Intn(len(pop))]
	}

	r := rand.Float64() * total
	cumulative := 0.0
	for _, ind := range pop {
		cumulative += ind.Fitness.WeightedScore - minFit
		if r <= cumulative {
			return ind
		}
	}
	return pop[len(pop)-1]
}

// ---------------------------------------------------------------------------
// Rank selection
// ---------------------------------------------------------------------------

// RankSelection implements linear rank-based selection. Individuals are ranked
// by fitness (rank 1 = worst, rank N = best) and selection probabilities are
// assigned as:
//
//	P(rank i) = (1/N) * (pressure - (pressure-1) * 2*(N-i)/(N-1))
//
// pressure is in [1.0, 2.0]:
//   - pressure=1.0 -> uniform selection (no selection pressure)
//   - pressure=2.0 -> maximum linear pressure
func RankSelection(pop []Individual, pressure float64) Individual {
	if len(pop) == 0 {
		panic("RankSelection: empty population")
	}
	if pressure < 1.0 {
		pressure = 1.0
	}
	if pressure > 2.0 {
		pressure = 2.0
	}

	// Sort by ascending fitness (rank 0 = worst).
	sorted := make([]Individual, len(pop))
	copy(sorted, pop)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Fitness.WeightedScore < sorted[j].Fitness.WeightedScore
	})

	n := float64(len(sorted))
	// Build cumulative probability array.
	cumProb := make([]float64, len(sorted))
	sum := 0.0
	for i := range sorted {
		rank := float64(i + 1) // 1-based rank
		p := (1.0 / n) * (pressure - (pressure-1.0)*2.0*(n-rank)/(n-1.0))
		if n == 1 {
			p = 1.0
		}
		sum += p
		cumProb[i] = sum
	}

	r := rand.Float64() * sum
	for i, cp := range cumProb {
		if r <= cp {
			return sorted[i]
		}
	}
	return sorted[len(sorted)-1]
}

// ---------------------------------------------------------------------------
// NSGA-II selection -- non-dominated sorting + crowding distance
// ---------------------------------------------------------------------------

// NSGA2Selection implements the NSGA-II survivor selection algorithm for
// multi-objective optimisation over the three LARSA objectives:
//   - Sharpe ratio (maximise)
//   - Sortino ratio (maximise)
//   - MaxDrawdown (minimise -- stored as positive value)
//
// It returns a subset of pop of the same size after:
//  1. Assigning Pareto fronts via fast non-dominated sorting.
//  2. Within the last admitted front, selecting by crowding distance
//     (higher crowding distance = more isolated = preferred).
//
// Reference: Deb et al. (2002) "A fast and elitist multiobjective genetic
// algorithm: NSGA-II", IEEE Trans. Evolutionary Computation 6(2), 182-197.
func NSGA2Selection(pop []Individual) []Individual {
	if len(pop) == 0 {
		return nil
	}
	targetSize := len(pop) / 2
	if targetSize == 0 {
		targetSize = 1
	}

	fronts := fastNonDominatedSort(pop)
	selected := make([]Individual, 0, targetSize)

	for _, front := range fronts {
		if len(selected)+len(front) <= targetSize {
			selected = append(selected, front...)
			continue
		}
		// Need to fill remainder from this front using crowding distance.
		need := targetSize - len(selected)
		crowdingDistanceAssign(front)
		sort.Slice(front, func(i, j int) bool {
			// Prefer larger crowding distance (more isolated = more diverse).
			return front[i].Fitness.CrowdingDistance > front[j].Fitness.CrowdingDistance
		})
		selected = append(selected, front[:need]...)
		break
	}

	return selected
}

// dominates returns true if a dominates b in the three-objective space
// (Sharpe, Sortino, MaxDD). a dominates b if it is not worse in any objective
// and strictly better in at least one.
func dominates(a, b Individual) bool {
	fa, fb := a.Fitness, b.Fitness
	// All objectives: Sharpe (max), Sortino (max), MaxDD (min -> negate for max).
	aVals := [3]float64{fa.Sharpe, fa.Sortino, -fa.MaxDD}
	bVals := [3]float64{fb.Sharpe, fb.Sortino, -fb.MaxDD}

	atLeastOneBetter := false
	for i := range aVals {
		if aVals[i] < bVals[i] {
			return false // a is worse in at least one objective
		}
		if aVals[i] > bVals[i] {
			atLeastOneBetter = true
		}
	}
	return atLeastOneBetter
}

// fastNonDominatedSort partitions pop into Pareto fronts.
// Front 0 contains all non-dominated individuals; front 1 contains individuals
// dominated only by front 0, and so on.
func fastNonDominatedSort(pop []Individual) [][]Individual {
	n := len(pop)
	dominated := make([][]int, n) // dominated[i] = indices that i dominates
	domCount := make([]int, n)    // domCount[i] = number of individuals that dominate i
	frontIdx := make([]int, n)

	front0 := make([]int, 0)
	for i := 0; i < n; i++ {
		dominated[i] = make([]int, 0)
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if dominates(pop[i], pop[j]) {
				dominated[i] = append(dominated[i], j)
			} else if dominates(pop[j], pop[i]) {
				domCount[i]++
			}
		}
		if domCount[i] == 0 {
			frontIdx[i] = 0
			front0 = append(front0, i)
		}
	}

	allFronts := make([][]Individual, 0)
	currentFront := front0
	for len(currentFront) > 0 {
		frontInds := make([]Individual, len(currentFront))
		for i, idx := range currentFront {
			frontInds[i] = pop[idx]
		}
		allFronts = append(allFronts, frontInds)

		nextFront := make([]int, 0)
		for _, i := range currentFront {
			for _, j := range dominated[i] {
				domCount[j]--
				if domCount[j] == 0 {
					frontIdx[j] = len(allFronts)
					nextFront = append(nextFront, j)
				}
			}
		}
		currentFront = nextFront
	}
	return allFronts
}

// crowdingDistanceAssign assigns the CrowdingDistance field of each individual
// in front based on the three objectives. Boundary individuals receive +Inf.
func crowdingDistanceAssign(front []Individual) {
	n := len(front)
	if n == 0 {
		return
	}
	for i := range front {
		front[i].Fitness.CrowdingDistance = 0
	}

	type objFn func(ind Individual) float64
	objectives := []objFn{
		func(ind Individual) float64 { return ind.Fitness.Sharpe },
		func(ind Individual) float64 { return ind.Fitness.Sortino },
		func(ind Individual) float64 { return -ind.Fitness.MaxDD },
	}

	for _, fn := range objectives {
		// Sort by this objective.
		sort.Slice(front, func(i, j int) bool {
			return fn(front[i]) < fn(front[j])
		})
		// Boundary individuals get infinity.
		front[0].Fitness.CrowdingDistance = math.Inf(1)
		front[n-1].Fitness.CrowdingDistance = math.Inf(1)

		fMin := fn(front[0])
		fMax := fn(front[n-1])
		if fMax-fMin < 1e-12 {
			continue
		}
		for i := 1; i < n-1; i++ {
			front[i].Fitness.CrowdingDistance += (fn(front[i+1]) - fn(front[i-1])) / (fMax - fMin)
		}
	}
}

// ---------------------------------------------------------------------------
// Elite preservation
// ---------------------------------------------------------------------------

// ElitePreservation returns the top n individuals from pop sorted by
// WeightedScore descending. These are guaranteed to survive into the next
// generation unchanged.
func ElitePreservation(pop []Individual, n int) []Individual {
	if n <= 0 {
		return nil
	}
	if n > len(pop) {
		n = len(pop)
	}
	sorted := make([]Individual, len(pop))
	copy(sorted, pop)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Fitness.WeightedScore > sorted[j].Fitness.WeightedScore
	})
	return sorted[:n]
}
