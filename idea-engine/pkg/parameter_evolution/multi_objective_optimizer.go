package parameter_evolution

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"sync"
)

// Objective represents a single optimisation objective.
type Objective struct {
	// Name is a human-readable label, e.g. "Sharpe".
	Name string
	// Value is the objective value for this individual.
	Value float64
	// IsMaximize is true when higher values are preferred.
	IsMaximize bool
}

// Individual is a candidate solution in the multi-objective population.
type Individual struct {
	// ID is a unique identifier.
	ID string
	// Params holds the parameter vector for this candidate.
	Params map[string]float64
	// Objectives holds the evaluated objective values.
	Objectives []Objective
	// FrontRank is the non-domination rank assigned by NonDominatedSort.
	// Rank 0 = Pareto-optimal.
	FrontRank int
	// CrowdingDist is the crowding distance within the individual's front.
	CrowdingDist float64
	// Generation records when this individual was created.
	Generation int
}

// dominates returns true when a dominates b on all objectives.
// An individual dominates another if it is at least as good on every objective
// and strictly better on at least one.
func dominates(a, b Individual) bool {
	if len(a.Objectives) != len(b.Objectives) {
		return false
	}
	atLeastOneBetter := false
	for i := range a.Objectives {
		av := a.Objectives[i].Value
		bv := b.Objectives[i].Value
		if a.Objectives[i].IsMaximize {
			if av < bv {
				return false
			}
			if av > bv {
				atLeastOneBetter = true
			}
		} else {
			if av > bv {
				return false
			}
			if av < bv {
				atLeastOneBetter = true
			}
		}
	}
	return atLeastOneBetter
}

// MultiObjectiveConfig configures the NSGA-II optimizer.
type MultiObjectiveConfig struct {
	// PopulationSize is the number of individuals maintained each generation.
	PopulationSize int
	// TournamentSize is the number of competitors per tournament selection event.
	TournamentSize int
	// CrossoverRate is the probability of performing crossover (vs cloning).
	CrossoverRate float64
	// MutationRate is the per-individual probability of applying mutation.
	MutationRate float64
	// Workers is the number of goroutines used for parallel fitness evaluation.
	Workers int
	// RandSeed seeds the population-level RNG.
	RandSeed int64
}

// DefaultMOConfig returns production-grade NSGA-II defaults.
func DefaultMOConfig() MultiObjectiveConfig {
	return MultiObjectiveConfig{
		PopulationSize: 100,
		TournamentSize: 2,
		CrossoverRate:  0.9,
		MutationRate:   0.1,
		Workers:        4,
		RandSeed:       42,
	}
}

// EvaluatorFunc is a callback that populates the Objectives of an Individual.
type EvaluatorFunc func(ctx context.Context, ind *Individual) error

// MultiObjectiveOptimizer runs an NSGA-II inspired multi-objective GA.
// The four standard objectives for SRFM are Sharpe, -MaxDrawdown, Calmar,
// and IC Stability. The caller supplies an EvaluatorFunc that populates those
// fields on each Individual.
type MultiObjectiveOptimizer struct {
	cfg       MultiObjectiveConfig
	rng       *rand.Rand
	mu        sync.Mutex
	evaluator EvaluatorFunc
	// idCounter is used for unique Individual IDs.
	idCounter int
}

// NewMultiObjectiveOptimizer creates an optimizer with the given config and
// evaluator function.
func NewMultiObjectiveOptimizer(cfg MultiObjectiveConfig, eval EvaluatorFunc) *MultiObjectiveOptimizer {
	return &MultiObjectiveOptimizer{
		cfg:       cfg,
		rng:       rand.New(rand.NewSource(cfg.RandSeed)),
		evaluator: eval,
	}
}

// NonDominatedSort partitions population into non-domination fronts.
// Returns a slice of fronts where front[k] contains the indices of individuals
// in the k-th non-domination layer. front[0] is Pareto-optimal.
func (o *MultiObjectiveOptimizer) NonDominatedSort(population []Individual) [][]int {
	n := len(population)
	// dominationCount[i] = number of individuals that dominate i.
	dominationCount := make([]int, n)
	// dominated[i] = set of individuals that i dominates.
	dominated := make([][]int, n)
	fronts := [][]int{{}}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if dominates(population[i], population[j]) {
				dominated[i] = append(dominated[i], j)
			} else if dominates(population[j], population[i]) {
				dominationCount[i]++
			}
		}
		if dominationCount[i] == 0 {
			population[i].FrontRank = 0
			fronts[0] = append(fronts[0], i)
		}
	}

	k := 0
	for len(fronts[k]) > 0 {
		nextFront := []int{}
		for _, i := range fronts[k] {
			for _, j := range dominated[i] {
				dominationCount[j]--
				if dominationCount[j] == 0 {
					population[j].FrontRank = k + 1
					nextFront = append(nextFront, j)
				}
			}
		}
		fronts = append(fronts, nextFront)
		k++
	}
	// Drop the trailing empty front.
	if len(fronts) > 0 && len(fronts[len(fronts)-1]) == 0 {
		fronts = fronts[:len(fronts)-1]
	}
	return fronts
}

// CrowdingDistance computes the crowding distance for all individuals in a
// single front (given as a slice of Individual values, not indices).
// The function modifies CrowdingDist in-place and also returns the distances.
func CrowdingDistance(front []Individual, nObjectives int) []float64 {
	n := len(front)
	distances := make([]float64, n)
	if n <= 2 {
		for i := range distances {
			distances[i] = math.Inf(1)
			front[i].CrowdingDist = math.Inf(1)
		}
		return distances
	}

	for m := 0; m < nObjectives; m++ {
		// Sort by objective m.
		indices := make([]int, n)
		for i := range indices {
			indices[i] = i
		}
		sort.Slice(indices, func(a, b int) bool {
			va := front[indices[a]].Objectives[m].Value
			vb := front[indices[b]].Objectives[m].Value
			return va < vb
		})

		// Boundary individuals get infinite distance.
		distances[indices[0]] = math.Inf(1)
		distances[indices[n-1]] = math.Inf(1)

		minVal := front[indices[0]].Objectives[m].Value
		maxVal := front[indices[n-1]].Objectives[m].Value
		rangeVal := maxVal - minVal
		if rangeVal == 0 {
			continue
		}

		for k := 1; k < n-1; k++ {
			prev := front[indices[k-1]].Objectives[m].Value
			next := front[indices[k+1]].Objectives[m].Value
			distances[indices[k]] += (next - prev) / rangeVal
		}
	}

	for i := range front {
		front[i].CrowdingDist = distances[i]
	}
	return distances
}

// SelectParents performs binary tournament selection favouring:
//  1. Lower front rank (better non-domination level).
//  2. Higher crowding distance (preserves diversity within a front).
//
// Returns n selected individuals. Population must have FrontRank and
// CrowdingDist populated (call NonDominatedSort + CrowdingDistance first).
func (o *MultiObjectiveOptimizer) SelectParents(population []Individual, n int) []Individual {
	selected := make([]Individual, 0, n)
	size := len(population)
	if size == 0 {
		return selected
	}
	for len(selected) < n {
		// Pick TournamentSize random competitors.
		best := population[o.rng.Intn(size)]
		for k := 1; k < o.cfg.TournamentSize; k++ {
			challenger := population[o.rng.Intn(size)]
			if tournamentWins(challenger, best) {
				best = challenger
			}
		}
		selected = append(selected, best)
	}
	return selected
}

// tournamentWins returns true when a should beat b in tournament selection.
// Lower FrontRank wins; ties broken by higher CrowdingDist.
func tournamentWins(a, b Individual) bool {
	if a.FrontRank != b.FrontRank {
		return a.FrontRank < b.FrontRank
	}
	return a.CrowdingDist > b.CrowdingDist
}

// Evolve runs NSGA-II for the given number of generations, starting from the
// supplied initial population. If initial is nil or empty, an initial
// population is generated by the caller's evaluator after seeding parameter
// values with random perturbations around paramSeed.
//
// Returns the final Pareto front (front rank 0) after all generations.
func (o *MultiObjectiveOptimizer) Evolve(
	ctx context.Context,
	initial []Individual,
	generations int,
) ([]Individual, error) {
	pop := make([]Individual, len(initial))
	copy(pop, initial)

	// Evaluate unevaluated individuals.
	if err := o.evaluatePopulation(ctx, pop); err != nil {
		return nil, err
	}

	for gen := 0; gen < generations; gen++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Assign front ranks and crowding distances.
		fronts := o.NonDominatedSort(pop)
		for _, front := range fronts {
			frontInds := make([]Individual, len(front))
			for k, idx := range front {
				frontInds[k] = pop[idx]
			}
			CrowdingDistance(frontInds, numObjectives(pop))
			for k, idx := range front {
				pop[idx].CrowdingDist = frontInds[k].CrowdingDist
				pop[idx].FrontRank = frontInds[k].FrontRank
			}
		}

		// Generate offspring.
		parents := o.SelectParents(pop, len(pop))
		offspring := o.makeOffspring(parents, gen+1)

		// Evaluate offspring.
		if err := o.evaluatePopulation(ctx, offspring); err != nil {
			return nil, err
		}

		// Combine and select the best popSize individuals.
		combined := append(pop, offspring...)
		pop = o.survivalSelection(combined, len(pop))
	}

	// Extract and return the Pareto front.
	fronts := o.NonDominatedSort(pop)
	if len(fronts) == 0 {
		return pop, nil
	}
	paretoFront := make([]Individual, len(fronts[0]))
	for i, idx := range fronts[0] {
		paretoFront[i] = pop[idx]
	}
	return paretoFront, nil
}

// makeOffspring generates a new population of offspring from the selected
// parents using crossover and mutation.
func (o *MultiObjectiveOptimizer) makeOffspring(parents []Individual, gen int) []Individual {
	offspring := make([]Individual, 0, len(parents))
	for i := 0; i+1 < len(parents); i += 2 {
		p1, p2 := parents[i], parents[i+1]
		var c1, c2 Individual
		if o.rng.Float64() < o.cfg.CrossoverRate {
			c1, c2 = o.uniformCrossover(p1, p2, gen)
		} else {
			c1 = cloneIndividual(p1, gen)
			c2 = cloneIndividual(p2, gen)
		}
		if o.rng.Float64() < o.cfg.MutationRate {
			c1 = o.gaussianMutate(c1)
		}
		if o.rng.Float64() < o.cfg.MutationRate {
			c2 = o.gaussianMutate(c2)
		}
		offspring = append(offspring, c1, c2)
	}
	return offspring
}

// uniformCrossover creates two offspring by randomly swapping parameters
// between two parents.
func (o *MultiObjectiveOptimizer) uniformCrossover(p1, p2 Individual, gen int) (Individual, Individual) {
	o.mu.Lock()
	id1 := o.nextID()
	id2 := o.nextID()
	o.mu.Unlock()

	c1 := Individual{
		ID:         id1,
		Params:     make(map[string]float64),
		Generation: gen,
	}
	c2 := Individual{
		ID:         id2,
		Params:     make(map[string]float64),
		Generation: gen,
	}

	allKeys := unionKeys(p1.Params, p2.Params)
	for _, k := range allKeys {
		v1 := p1.Params[k]
		v2 := p2.Params[k]
		if o.rng.Float64() < 0.5 {
			c1.Params[k] = v1
			c2.Params[k] = v2
		} else {
			c1.Params[k] = v2
			c2.Params[k] = v1
		}
	}
	return c1, c2
}

// gaussianMutate perturbs a random parameter of ind by a small Gaussian noise.
func (o *MultiObjectiveOptimizer) gaussianMutate(ind Individual) Individual {
	if len(ind.Params) == 0 {
		return ind
	}
	keys := sortedKeys(ind.Params)
	k := keys[o.rng.Intn(len(keys))]
	noise := o.rng.NormFloat64() * 0.05
	ind.Params[k] += ind.Params[k] * noise
	return ind
}

// survivalSelection selects the best popSize individuals from combined using
// the NSGA-II selection criterion: prefer lower front rank, break ties by
// higher crowding distance.
func (o *MultiObjectiveOptimizer) survivalSelection(combined []Individual, popSize int) []Individual {
	fronts := o.NonDominatedSort(combined)
	// Assign crowding distances per front.
	for _, front := range fronts {
		frontInds := make([]Individual, len(front))
		for k, idx := range front {
			frontInds[k] = combined[idx]
		}
		CrowdingDistance(frontInds, numObjectives(combined))
		for k, idx := range front {
			combined[idx].CrowdingDist = frontInds[k].CrowdingDist
			combined[idx].FrontRank = frontInds[k].FrontRank
		}
	}

	// Sort combined by (FrontRank asc, CrowdingDist desc).
	sort.Slice(combined, func(i, j int) bool {
		return tournamentWins(combined[i], combined[j])
	})
	if len(combined) > popSize {
		return combined[:popSize]
	}
	return combined
}

// evaluatePopulation evaluates all unevaluated individuals concurrently.
func (o *MultiObjectiveOptimizer) evaluatePopulation(ctx context.Context, pop []Individual) error {
	type job struct {
		idx int
	}
	jobs := make(chan job, len(pop))
	errs := make(chan error, len(pop))

	workers := o.cfg.Workers
	if workers <= 0 {
		workers = 1
	}

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				select {
				case <-ctx.Done():
					errs <- ctx.Err()
					return
				default:
				}
				if err := o.evaluator(ctx, &pop[j.idx]); err != nil {
					errs <- err
				}
			}
		}()
	}

	for i := range pop {
		if len(pop[i].Objectives) == 0 {
			jobs <- job{idx: i}
		}
	}
	close(jobs)
	wg.Wait()
	close(errs)

	for err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// nextID returns the next sequential ID string. Caller must hold o.mu.
func (o *MultiObjectiveOptimizer) nextID() string {
	o.idCounter++
	return idFromInt(o.idCounter)
}

// idFromInt converts an integer counter to a padded ID string.
func idFromInt(n int) string {
	const digits = "0123456789"
	if n == 0 {
		return "ind_000000"
	}
	buf := make([]byte, 6)
	for i := 5; i >= 0; i-- {
		buf[i] = digits[n%10]
		n /= 10
		if n == 0 {
			for j := i - 1; j >= 0; j-- {
				buf[j] = '0'
			}
			break
		}
	}
	return "ind_" + string(buf)
}

// cloneIndividual returns a copy of ind with a new ID and generation stamp.
func cloneIndividual(ind Individual, gen int) Individual {
	cp := Individual{
		ID:         ind.ID + "_clone",
		Params:     copyParams(ind.Params),
		Generation: gen,
	}
	return cp
}

// unionKeys returns the sorted union of keys from two param maps.
func unionKeys(a, b map[string]float64) []string {
	seen := make(map[string]struct{}, len(a)+len(b))
	for k := range a {
		seen[k] = struct{}{}
	}
	for k := range b {
		seen[k] = struct{}{}
	}
	keys := make([]string, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// numObjectives returns the number of objectives for the first evaluated
// individual in pop, or 0 if none are evaluated.
func numObjectives(pop []Individual) int {
	for _, ind := range pop {
		if len(ind.Objectives) > 0 {
			return len(ind.Objectives)
		}
	}
	return 0
}

// StandardObjectives constructs the four SRFM objectives given the raw metrics.
// The four objectives are: Sharpe (max), -MaxDrawdown (max, i.e. min drawdown),
// Calmar (max), and ICStability (max).
func StandardObjectives(sharpe, maxDD, calmar, icStability float64) []Objective {
	return []Objective{
		{Name: "Sharpe", Value: sharpe, IsMaximize: true},
		{Name: "NegMaxDD", Value: -maxDD, IsMaximize: true},
		{Name: "Calmar", Value: calmar, IsMaximize: true},
		{Name: "ICStability", Value: icStability, IsMaximize: true},
	}
}
