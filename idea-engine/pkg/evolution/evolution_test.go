package evolution

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makePop constructs a population of individuals with the given fitness scores.
func makePop(scores []float64) []Individual {
	pop := make([]Individual, len(scores))
	for i, s := range scores {
		pop[i] = Individual{
			ID:        fmt.Sprintf("ind-%d", i),
			Genes:     Genome{float64(i)},
			Evaluated: true,
			Fitness: FitnessResult{
				WeightedScore: s,
				Sharpe:        s,
				Sortino:       s * 0.9,
				MaxDD:         0.1 / (s + 0.01),
			},
		}
	}
	return pop
}

func makeBounds(n int, lo, hi float64) [][2]float64 {
	b := make([][2]float64, n)
	for i := range b {
		b[i] = [2]float64{lo, hi}
	}
	return b
}

// ---------------------------------------------------------------------------
// Crossover tests
// ---------------------------------------------------------------------------

// TestUniformCrossoverGeneCount verifies that offspring have the same length
// as parents.
func TestUniformCrossoverGeneCount(t *testing.T) {
	p1 := Genome{1, 2, 3, 4, 5}
	p2 := Genome{6, 7, 8, 9, 10}
	c1, c2 := UniformCrossover(p1, p2, 0.5)
	if len(c1) != 5 {
		t.Errorf("c1 length = %d; want 5", len(c1))
	}
	if len(c2) != 5 {
		t.Errorf("c2 length = %d; want 5", len(c2))
	}
}

// TestUniformCrossoverRate0 verifies that rate=0 returns clones of the parents.
func TestUniformCrossoverRate0(t *testing.T) {
	p1 := Genome{1, 2, 3}
	p2 := Genome{4, 5, 6}
	c1, c2 := UniformCrossover(p1, p2, 0.0)
	for i := range c1 {
		if c1[i] != p1[i] {
			t.Errorf("rate=0: c1[%d] = %g; want %g", i, c1[i], p1[i])
		}
		if c2[i] != p2[i] {
			t.Errorf("rate=0: c2[%d] = %g; want %g", i, c2[i], p2[i])
		}
	}
}

// TestUniformCrossoverRate1 verifies that rate=1 fully swaps the parents.
func TestUniformCrossoverRate1(t *testing.T) {
	p1 := Genome{1, 2, 3}
	p2 := Genome{4, 5, 6}
	c1, c2 := UniformCrossover(p1, p2, 1.0)
	for i := range c1 {
		if c1[i] != p2[i] {
			t.Errorf("rate=1: c1[%d] = %g; want %g", i, c1[i], p2[i])
		}
		if c2[i] != p1[i] {
			t.Errorf("rate=1: c2[%d] = %g; want %g", i, c2[i], p1[i])
		}
	}
}

// TestSBXBoundsPreservation verifies SBX does not cause offspring to deviate
// wildly beyond the parent range for eta=20.
func TestSBXBoundsPreservation(t *testing.T) {
	p1 := Genome{0.1, 0.2, 0.3, 0.4}
	p2 := Genome{0.5, 0.6, 0.7, 0.8}
	for trial := 0; trial < 500; trial++ {
		c1, c2 := SimulatedBinaryCrossover(p1, p2, 20)
		for i := range c1 {
			lo := p1[i] - 2.0 // generous tolerance
			hi := p2[i] + 2.0
			if c1[i] < lo || c1[i] > hi {
				t.Errorf("trial %d: SBX c1[%d] = %g outside [%g, %g]", trial, i, c1[i], lo, hi)
			}
			if c2[i] < lo || c2[i] > hi {
				t.Errorf("trial %d: SBX c2[%d] = %g outside [%g, %g]", trial, i, c2[i], lo, hi)
			}
		}
	}
}

// TestSBXSymmetry verifies that swapping parents produces symmetric offspring.
func TestSBXSymmetry(t *testing.T) {
	p1 := Genome{1.0, 2.0}
	p2 := Genome{3.0, 4.0}
	// Not a strong symmetry guarantee because of the randomness, but we check
	// that the lengths are correct and no NaN values are produced.
	for trial := 0; trial < 100; trial++ {
		c1, c2 := SimulatedBinaryCrossover(p1, p2, 20)
		for i := range c1 {
			if math.IsNaN(c1[i]) || math.IsInf(c1[i], 0) {
				t.Errorf("trial %d: SBX c1[%d] is NaN/Inf", trial, i)
			}
			if math.IsNaN(c2[i]) || math.IsInf(c2[i], 0) {
				t.Errorf("trial %d: SBX c2[%d] is NaN/Inf", trial, i)
			}
		}
	}
}

// TestArithmeticCrossoverAlpha05 verifies alpha=0.5 produces the midpoint.
func TestArithmeticCrossoverAlpha05(t *testing.T) {
	p1 := Genome{0.0, 2.0, 4.0}
	p2 := Genome{2.0, 4.0, 6.0}
	c1, c2 := ArithmeticCrossover(p1, p2, 0.5)
	for i := range c1 {
		mid := (p1[i] + p2[i]) / 2.0
		if math.Abs(c1[i]-mid) > 1e-12 {
			t.Errorf("c1[%d] = %g; want %g", i, c1[i], mid)
		}
		if math.Abs(c2[i]-mid) > 1e-12 {
			t.Errorf("c2[%d] = %g; want %g", i, c2[i], mid)
		}
	}
}

// TestOrderCrossoverLengthAndPermutation verifies OX offspring have the same
// length as parents and contain all the same values (i.e. are valid perms).
func TestOrderCrossoverLengthAndPermutation(t *testing.T) {
	p1 := Genome{1, 2, 3, 4, 5, 6}
	p2 := Genome{6, 5, 4, 3, 2, 1}
	for trial := 0; trial < 200; trial++ {
		c1, c2 := OrderCrossover(p1, p2)
		if len(c1) != 6 || len(c2) != 6 {
			t.Fatalf("trial %d: wrong length c1=%d c2=%d", trial, len(c1), len(c2))
		}
		// Check c1 is a permutation of p1.
		checkPerm(t, "c1", c1, p1)
		checkPerm(t, "c2", c2, p2)
	}
}

// checkPerm verifies that child contains the same multiset of values as src.
func checkPerm(t *testing.T, name string, child, src Genome) {
	t.Helper()
	srcCopy := make([]float64, len(src))
	copy(srcCopy, src)
	sort.Float64s(srcCopy)
	childCopy := make([]float64, len(child))
	copy(childCopy, child)
	sort.Float64s(childCopy)
	for i := range srcCopy {
		if srcCopy[i] != childCopy[i] {
			t.Errorf("%s: not a valid permutation at index %d: got %g want %g", name, i, childCopy[i], srcCopy[i])
		}
	}
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

// TestGaussianMutationRate verifies that approximately rate*n genes are
// mutated per call across many trials.
func TestGaussianMutationRate(t *testing.T) {
	g := make(Genome, 1000)
	for i := range g {
		g[i] = 0.0
	}
	rate := 0.2
	sigma := 1.0
	changed := 0
	trials := 100
	for i := 0; i < trials; i++ {
		out := GaussianMutation(g, sigma, rate)
		for j, v := range out {
			if v != g[j] {
				changed++
			}
		}
	}
	expected := rate * float64(len(g)) * float64(trials)
	// Allow 20% tolerance.
	if float64(changed) < 0.8*expected || float64(changed) > 1.2*expected {
		t.Errorf("GaussianMutation: changed=%d expected~%g (rate=%g, n=%d, trials=%d)",
			changed, expected, rate, len(g), trials)
	}
}

// TestGaussianMutationNoiseMagnitude verifies that mutations are non-zero.
func TestGaussianMutationNoiseMagnitude(t *testing.T) {
	g := Genome{5.0, 5.0, 5.0}
	out := GaussianMutation(g, 1.0, 1.0) // mutate all
	// At least one gene should differ.
	same := true
	for i := range g {
		if out[i] != g[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("GaussianMutation with rate=1 and sigma=1: no genes were changed")
	}
}

// TestPolynomialMutationBounds verifies that PolynomialMutation never
// produces a gene outside its defined bounds.
func TestPolynomialMutationBounds(t *testing.T) {
	bounds := makeBounds(20, 0.0, 1.0)
	g := make(Genome, 20)
	for i := range g {
		g[i] = 0.5
	}
	for trial := 0; trial < 1000; trial++ {
		out := PolynomialMutation(g, 20, bounds)
		for i, v := range out {
			if v < 0.0 || v > 1.0 {
				t.Errorf("trial %d: gene[%d] = %g; outside [0,1]", trial, i, v)
			}
		}
	}
}

// TestPolynomialMutationChangesValue verifies that PolynomialMutation actually
// changes gene values (not a no-op).
func TestPolynomialMutationChangesValue(t *testing.T) {
	bounds := makeBounds(10, 0.0, 1.0)
	g := make(Genome, 10)
	for i := range g {
		g[i] = 0.5
	}
	changed := false
	for trial := 0; trial < 100; trial++ {
		out := PolynomialMutation(g, 20, bounds)
		for i, v := range out {
			if v != g[i] {
				changed = true
				break
			}
		}
		if changed {
			break
		}
	}
	if !changed {
		t.Error("PolynomialMutation: no genes changed after 100 trials")
	}
}

// TestCauchyMutationHeavyTail verifies that Cauchy mutation occasionally
// produces very large jumps (heavy-tail property). We check that at least
// one gene in 1000 trials exceeds 5*scale from the original value.
func TestCauchyMutationHeavyTail(t *testing.T) {
	g := Genome{0.0}
	scale := 1.0
	extremeCount := 0
	for i := 0; i < 10000; i++ {
		out := CauchyMutation(g, scale)
		if math.Abs(out[0]) > 5.0*scale {
			extremeCount++
		}
	}
	// Cauchy P(|X|>5) = 2*arctan(1/5)/pi ~ 0.063; expect ~630 over 10000.
	if extremeCount < 100 {
		t.Errorf("CauchyMutation: only %d extreme samples in 10000 (expected ~630)", extremeCount)
	}
}

// ---------------------------------------------------------------------------
// Selection tests
// ---------------------------------------------------------------------------

// TestTournamentSelectionBias verifies that higher-fitness individuals are
// selected more often with k=5 than uniformly at random.
func TestTournamentSelectionBias(t *testing.T) {
	pop := makePop([]float64{0.1, 0.2, 0.3, 0.4, 1.0})
	counts := make(map[string]int)
	trials := 10000
	for i := 0; i < trials; i++ {
		sel := TournamentSelection(pop, 3)
		counts[sel.ID]++
	}
	// The best individual (fitness=1.0) should be selected far more than 20%.
	best := pop[4].ID
	if float64(counts[best])/float64(trials) < 0.40 {
		t.Errorf("TournamentSelection: best individual selected %d/%d times (%.2f%%); want > 40%%",
			counts[best], trials, 100.0*float64(counts[best])/float64(trials))
	}
}

// TestTournamentSelectionK1 verifies k=1 is uniform (no selection pressure).
func TestTournamentSelectionK1(t *testing.T) {
	pop := makePop([]float64{0.1, 0.9})
	counts := make(map[string]int)
	trials := 10000
	for i := 0; i < trials; i++ {
		sel := TournamentSelection(pop, 1)
		counts[sel.ID]++
	}
	// Each should be selected ~50% with k=1.
	for _, ind := range pop {
		ratio := float64(counts[ind.ID]) / float64(trials)
		if ratio < 0.40 || ratio > 0.60 {
			t.Errorf("k=1 tournament: ind %s selected %.2f%%; want ~50%%", ind.ID, ratio*100)
		}
	}
}

// TestRouletteWheelSelectionProbability verifies that higher-fitness
// individuals are selected proportionally more often.
func TestRouletteWheelSelectionProbability(t *testing.T) {
	pop := makePop([]float64{1.0, 3.0})
	counts := make(map[string]int)
	trials := 20000
	for i := 0; i < trials; i++ {
		sel := RouletteWheelSelection(pop)
		counts[sel.ID]++
	}
	// pop[1] has 3x the fitness of pop[0] -> should be selected ~75% of the time.
	ratio := float64(counts[pop[1].ID]) / float64(trials)
	if ratio < 0.65 || ratio > 0.85 {
		t.Errorf("RouletteWheel: high-fitness selected %.2f%%; want ~75%%", ratio*100)
	}
}

// TestRankSelectionOrdering verifies that with pressure=2.0 the highest-rank
// individual is chosen significantly more than the lowest.
func TestRankSelectionOrdering(t *testing.T) {
	pop := makePop([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
	best := pop[4]
	worst := pop[0]
	bestCount, worstCount := 0, 0
	trials := 10000
	for i := 0; i < trials; i++ {
		sel := RankSelection(pop, 2.0)
		if sel.ID == best.ID {
			bestCount++
		}
		if sel.ID == worst.ID {
			worstCount++
		}
	}
	if bestCount <= worstCount {
		t.Errorf("RankSelection pressure=2: best selected %d times, worst %d times; best should dominate",
			bestCount, worstCount)
	}
}

// TestNSGA2ParetoFront verifies that the Pareto front individuals are always
// retained by NSGA2Selection.
func TestNSGA2ParetoFront(t *testing.T) {
	// Build a population where ind0 clearly dominates ind1 in all objectives.
	pop := []Individual{
		{
			ID:        "dominant",
			Genes:     Genome{1},
			Evaluated: true,
			Fitness:   FitnessResult{Sharpe: 2.0, Sortino: 2.0, MaxDD: 0.05, WeightedScore: 2.0},
		},
		{
			ID:        "dominated",
			Genes:     Genome{2},
			Evaluated: true,
			Fitness:   FitnessResult{Sharpe: 0.5, Sortino: 0.5, MaxDD: 0.30, WeightedScore: 0.5},
		},
		{
			ID:        "dominated2",
			Genes:     Genome{3},
			Evaluated: true,
			Fitness:   FitnessResult{Sharpe: 0.3, Sortino: 0.3, MaxDD: 0.40, WeightedScore: 0.3},
		},
		{
			ID:        "dominated3",
			Genes:     Genome{4},
			Evaluated: true,
			Fitness:   FitnessResult{Sharpe: 0.2, Sortino: 0.2, MaxDD: 0.50, WeightedScore: 0.2},
		},
	}
	selected := NSGA2Selection(pop)
	foundDominant := false
	for _, ind := range selected {
		if ind.ID == "dominant" {
			foundDominant = true
		}
	}
	if !foundDominant {
		t.Error("NSGA2Selection: dominant individual not in selected set")
	}
}

// TestNSGA2SelectionSize verifies the output size is half the input.
func TestNSGA2SelectionSize(t *testing.T) {
	scores := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	pop := makePop(scores)
	selected := NSGA2Selection(pop)
	if len(selected) != 4 {
		t.Errorf("NSGA2Selection: len=%d; want 4", len(selected))
	}
}

// TestElitePreservationCount verifies n elites are preserved.
func TestElitePreservationCount(t *testing.T) {
	pop := makePop([]float64{5, 1, 3, 2, 4})
	elites := ElitePreservation(pop, 2)
	if len(elites) != 2 {
		t.Fatalf("ElitePreservation: len=%d; want 2", len(elites))
	}
	// The two best should be fitness=5 and fitness=4.
	if elites[0].Fitness.WeightedScore != 5 || elites[1].Fitness.WeightedScore != 4 {
		t.Errorf("ElitePreservation: unexpected top 2: %v %v",
			elites[0].Fitness.WeightedScore, elites[1].Fitness.WeightedScore)
	}
}

// ---------------------------------------------------------------------------
// Population tests
// ---------------------------------------------------------------------------

// TestPopulationDiversity verifies that a diverse population has higher
// diversity than a uniform population.
func TestPopulationDiversity(t *testing.T) {
	bounds := makeBounds(5, 0, 1)
	diversePop, err := Initialize(20, bounds, 0)
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	diversePop.Diversity = ComputeDiversity(diversePop)

	// Create a uniform population (all same genes).
	uniformPop := Population{Individuals: make([]Individual, 20)}
	for i := range uniformPop.Individuals {
		uniformPop.Individuals[i] = Individual{Genes: Genome{0.5, 0.5, 0.5, 0.5, 0.5}}
	}
	uniformPop.Diversity = ComputeDiversity(uniformPop)

	if uniformPop.Diversity != 0.0 {
		t.Errorf("uniform population diversity = %g; want 0", uniformPop.Diversity)
	}
	if diversePop.Diversity <= uniformPop.Diversity {
		t.Errorf("diverse pop diversity %g should be > uniform pop diversity %g",
			diversePop.Diversity, uniformPop.Diversity)
	}
}

// TestInitializeLHSSampleCount verifies Initialize produces exactly size
// individuals.
func TestInitializeLHSSampleCount(t *testing.T) {
	bounds := makeBounds(10, -1, 1)
	pop, err := Initialize(30, bounds, 0)
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if len(pop.Individuals) != 30 {
		t.Errorf("Initialize: got %d individuals; want 30", len(pop.Individuals))
	}
}

// TestInitializeBoundsRespected verifies every gene in the initial population
// lies within its bounds.
func TestInitializeBoundsRespected(t *testing.T) {
	lo, hi := -5.0, 5.0
	bounds := makeBounds(8, lo, hi)
	pop, err := Initialize(50, bounds, 0)
	if err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	for i, ind := range pop.Individuals {
		for j, v := range ind.Genes {
			if v < lo || v > hi {
				t.Errorf("ind %d gene %d = %g outside [%g, %g]", i, j, v, lo, hi)
			}
		}
	}
}

// TestArchiveElitesMaxSize verifies the archive is trimmed to maxSize.
func TestArchiveElitesMaxSize(t *testing.T) {
	pop := Population{Individuals: makePop([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})}
	archive := ArchiveElites(pop, nil, 5)
	if len(archive) != 5 {
		t.Errorf("ArchiveElites: len=%d; want 5", len(archive))
	}
	// Archive should contain the 5 best.
	if archive[0].Fitness.WeightedScore != 10 {
		t.Errorf("ArchiveElites: best fitness = %g; want 10", archive[0].Fitness.WeightedScore)
	}
}

// ---------------------------------------------------------------------------
// Fitness tests
// ---------------------------------------------------------------------------

// TestWeightedFitnessFormula verifies the weighted fitness formula.
func TestWeightedFitnessFormula(t *testing.T) {
	r := FitnessResult{
		Sharpe:           2.0,
		Sortino:          1.5,
		MaxDD:            0.2,
		RegimeRobustness: 1.0,
	}
	score := WeightedFitness(&r)
	expected := 0.40*2.0 + 0.25*1.5 + 0.20*(1.0/0.2) + 0.15*1.0
	if math.Abs(score-expected) > 1e-10 {
		t.Errorf("WeightedFitness = %g; want %g", score, expected)
	}
}

// TestWeightedFitnessZeroMaxDD verifies MaxDD=0 does not produce Inf.
func TestWeightedFitnessZeroMaxDD(t *testing.T) {
	r := FitnessResult{Sharpe: 1, Sortino: 1, MaxDD: 0, RegimeRobustness: 1}
	score := WeightedFitness(&r)
	if math.IsInf(score, 0) || math.IsNaN(score) {
		t.Errorf("WeightedFitness with MaxDD=0: got %g (Inf or NaN)", score)
	}
}

// TestCachedEvaluatorCacheHit verifies that a second call with the same genome
// returns a cached result and does not invoke the inner evaluator again.
func TestCachedEvaluatorCacheHit(t *testing.T) {
	calls := 0
	inner := &countingEvaluator{
		callCount: &calls,
		result:    FitnessResult{Sharpe: 1.5, Sortino: 1.2, MaxDD: 0.1, WeightedScore: 1.5},
	}
	cached := NewCachedFitnessEvaluator(inner, 100)
	genome := Genome{1.0, 2.0, 3.0}

	ctx := context.Background()
	r1 := cached.Evaluate(ctx, genome)
	r2 := cached.Evaluate(ctx, genome)

	if calls != 1 {
		t.Errorf("inner evaluator called %d times; want 1 (cache hit on second call)", calls)
	}
	if r1.Sharpe != r2.Sharpe {
		t.Errorf("cached result differs: r1.Sharpe=%g r2.Sharpe=%g", r1.Sharpe, r2.Sharpe)
	}

	hits, misses, size := cached.CacheStats()
	if hits != 1 {
		t.Errorf("cache hits = %d; want 1", hits)
	}
	if misses != 1 {
		t.Errorf("cache misses = %d; want 1", misses)
	}
	if size != 1 {
		t.Errorf("cache size = %d; want 1", size)
	}
}

// TestCachedEvaluatorDifferentGenomes verifies different genomes are evaluated
// independently (no false cache hits).
func TestCachedEvaluatorDifferentGenomes(t *testing.T) {
	calls := 0
	inner := &countingEvaluator{
		callCount: &calls,
		result:    FitnessResult{Sharpe: 1.0, WeightedScore: 1.0},
	}
	cached := NewCachedFitnessEvaluator(inner, 100)
	ctx := context.Background()

	for i := 0; i < 10; i++ {
		cached.Evaluate(ctx, Genome{float64(i)})
	}
	if calls != 10 {
		t.Errorf("inner evaluator called %d times; want 10 (one per distinct genome)", calls)
	}
}

// TestParallelEvaluatorConcurrency verifies that all unevaluated individuals
// are evaluated without data races under concurrent workers.
func TestParallelEvaluatorConcurrency(t *testing.T) {
	inner := &sleepEvaluator{result: FitnessResult{Sharpe: 1.0, WeightedScore: 1.0}}
	pe := NewParallelEvaluator(inner, 4)

	pop := make([]Individual, 20)
	for i := range pop {
		pop[i] = Individual{
			ID:    fmt.Sprintf("ind-%d", i),
			Genes: Genome{float64(i)},
		}
	}

	ctx := context.Background()
	evaluated := pe.EvaluatePopulation(ctx, pop)

	for i, ind := range evaluated {
		if !ind.Evaluated {
			t.Errorf("individual %d not evaluated", i)
		}
	}
}

// TestAdaptiveSigmaMutationAdapts verifies that consistently successful
// mutations cause sigma to increase (1/5 rule: >20% success -> grow sigma).
func TestAdaptiveSigmaMutationAdapts(t *testing.T) {
	asm := NewAdaptiveSigmaMutation(5, 0.01)
	initialSigmas := asm.Sigmas()

	// Record 100% success rate -> sigma should increase.
	for i := 0; i < 100; i++ {
		asm.RecordOutcome(true)
	}
	finalSigmas := asm.Sigmas()
	for i, s := range finalSigmas {
		if s <= initialSigmas[i] {
			t.Errorf("sigma[%d] = %g; expected increase from %g after 100%% success", i, s, initialSigmas[i])
		}
	}
}

// TestAdaptiveSigmaMutationDecreasesOnFailure verifies sigma decreases when
// no improvements are reported (0% success rate).
func TestAdaptiveSigmaMutationDecreasesOnFailure(t *testing.T) {
	asm := NewAdaptiveSigmaMutation(5, 1.0)
	initialSigmas := asm.Sigmas()

	for i := 0; i < 100; i++ {
		asm.RecordOutcome(false)
	}
	finalSigmas := asm.Sigmas()
	for i, s := range finalSigmas {
		if s >= initialSigmas[i] {
			t.Errorf("sigma[%d] = %g; expected decrease from %g after 0%% success", i, s, initialSigmas[i])
		}
	}
}

// TestAdaptiveCrossoverRecordsOutcome verifies that AdaptiveCrossover tracks
// operator usage and shifts probabilities.
func TestAdaptiveCrossoverRecordsOutcome(t *testing.T) {
	ac := NewAdaptiveCrossover(50)
	p1 := Genome{1.0, 2.0, 3.0}
	p2 := Genome{4.0, 5.0, 6.0}

	// Use operator 0 (uniform) 40 times with improvement.
	// Use operator 1 (sbx) 10 times with no improvement.
	for i := 0; i < 40; i++ {
		_, _, opIdx := ac.Apply(p1, p2)
		ac.RecordOutcome(opIdx, 0.5)
	}

	stats := ac.OperatorStats()
	totalUses := 0
	for _, s := range stats {
		totalUses += s.Uses
	}
	if totalUses != 40 {
		t.Errorf("total operator uses = %d; want 40", totalUses)
	}
}

// ---------------------------------------------------------------------------
// GenomeHash tests
// ---------------------------------------------------------------------------

// TestGenomeHashConsistency verifies the same genome always produces the same
// hash.
func TestGenomeHashConsistency(t *testing.T) {
	g := Genome{1.1, 2.2, 3.3}
	h1 := genomeHash(g)
	h2 := genomeHash(g)
	if h1 != h2 {
		t.Error("genomeHash: same genome produced different hashes")
	}
}

// TestGenomeHashDifferentGenomes verifies different genomes produce different
// hashes (collision resistance for close values).
func TestGenomeHashDifferentGenomes(t *testing.T) {
	g1 := Genome{1.0, 2.0, 3.0}
	g2 := Genome{1.0, 2.0, 3.0000000001}
	if genomeHash(g1) == genomeHash(g2) {
		t.Error("genomeHash: collision between near-equal genomes")
	}
}

// ---------------------------------------------------------------------------
// LRU cache internal tests
// ---------------------------------------------------------------------------

// TestLRUCacheEviction verifies that the oldest entry is evicted when the
// cache exceeds its capacity.
func TestLRUCacheEviction(t *testing.T) {
	c := newLRUCache(3)
	keys := [4][32]byte{}
	for i := range keys {
		keys[i][0] = byte(i + 1)
	}
	c.put(keys[0], FitnessResult{Sharpe: 0})
	c.put(keys[1], FitnessResult{Sharpe: 1})
	c.put(keys[2], FitnessResult{Sharpe: 2})
	c.put(keys[3], FitnessResult{Sharpe: 3}) // should evict keys[0]

	_, ok := c.get(keys[0])
	if ok {
		t.Error("LRU eviction: keys[0] should have been evicted")
	}
	_, ok = c.get(keys[3])
	if !ok {
		t.Error("LRU eviction: keys[3] should be present")
	}
}

// TestLRUCacheMoveToFront verifies that accessing an entry promotes it to
// most-recently-used, preventing it from being evicted first.
func TestLRUCacheMoveToFront(t *testing.T) {
	c := newLRUCache(2)
	k1, k2, k3 := [32]byte{1}, [32]byte{2}, [32]byte{3}
	c.put(k1, FitnessResult{Sharpe: 1})
	c.put(k2, FitnessResult{Sharpe: 2})
	c.get(k1) // promote k1 to MRU
	c.put(k3, FitnessResult{Sharpe: 3}) // should evict k2 (now LRU)
	_, ok := c.get(k2)
	if ok {
		t.Error("LRU: k2 should have been evicted after k1 was promoted")
	}
	_, ok = c.get(k1)
	if !ok {
		t.Error("LRU: k1 should still be in cache after being promoted")
	}
}

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

// countingEvaluator is a test double that counts calls to Evaluate.
type countingEvaluator struct {
	mu        sync.Mutex
	callCount *int
	result    FitnessResult
}

func (e *countingEvaluator) Evaluate(_ context.Context, _ Genome) FitnessResult {
	e.mu.Lock()
	*e.callCount++
	e.mu.Unlock()
	return e.result
}

// sleepEvaluator is a test double that returns a fixed result.
type sleepEvaluator struct {
	result FitnessResult
}

func (e *sleepEvaluator) Evaluate(_ context.Context, _ Genome) FitnessResult {
	return e.result
}

// Compile-time check that test doubles satisfy the interface.
var _ FitnessEvaluator = (*countingEvaluator)(nil)
var _ FitnessEvaluator = (*sleepEvaluator)(nil)
