package evolution

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Uniform crossover
// ---------------------------------------------------------------------------

// UniformCrossover produces two offspring by independently swapping each gene
// position with probability rate. Both offspring are full-length copies of
// their respective parents with some genes exchanged.
//
// rate=0.5 produces maximum mixing; rate=0.0 returns clones of the parents.
func UniformCrossover(p1, p2 Genome, rate float64) (Genome, Genome) {
	if len(p1) != len(p2) {
		panic("UniformCrossover: parent genomes must have equal length")
	}
	n := len(p1)
	c1 := p1.Clone()
	c2 := p2.Clone()
	for i := 0; i < n; i++ {
		if rand.Float64() < rate {
			c1[i], c2[i] = c2[i], c1[i]
		}
	}
	return c1, c2
}

// ---------------------------------------------------------------------------
// Simulated Binary Crossover (SBX)
// ---------------------------------------------------------------------------

// sbxBeta computes the spread factor beta for SBX given a uniform random u
// and the distribution index eta.
// eta controls the shape of the offspring distribution:
//   - large eta -> offspring close to parents
//   - small eta -> offspring spread farther from parents
func sbxBeta(u, eta float64) float64 {
	if u <= 0.5 {
		return math.Pow(2.0*u, 1.0/(eta+1.0))
	}
	return math.Pow(1.0/(2.0*(1.0-u)), 1.0/(eta+1.0))
}

// SimulatedBinaryCrossover implements SBX for real-valued parameters.
// eta=20 gives offspring distributions concentrated near the parents
// (similar to single-point crossover in binary encodings).
//
// Reference: Deb & Agrawal (1995) "Simulated Binary Crossover for Continuous
// Search Space", Complex Systems 9(2), 115-148.
func SimulatedBinaryCrossover(p1, p2 Genome, eta float64) (Genome, Genome) {
	if len(p1) != len(p2) {
		panic("SimulatedBinaryCrossover: parent genomes must have equal length")
	}
	n := len(p1)
	c1 := p1.Clone()
	c2 := p2.Clone()

	for i := 0; i < n; i++ {
		// Each gene recombined with probability 0.5.
		if rand.Float64() > 0.5 {
			continue
		}
		// Skip if parents are identical at this locus.
		if math.Abs(p1[i]-p2[i]) < 1e-14 {
			continue
		}
		u := rand.Float64()
		beta := sbxBeta(u, eta)
		c1[i] = 0.5*((1.0+beta)*p1[i] + (1.0-beta)*p2[i])
		c2[i] = 0.5*((1.0-beta)*p1[i] + (1.0+beta)*p2[i])
	}
	return c1, c2
}

// ---------------------------------------------------------------------------
// Arithmetic crossover
// ---------------------------------------------------------------------------

// ArithmeticCrossover produces offspring as convex combinations of the parents.
//
//	c1 = alpha*p1 + (1-alpha)*p2
//	c2 = alpha*p2 + (1-alpha)*p1
//
// alpha in [0,1]; alpha=0.5 returns the midpoint in both directions.
func ArithmeticCrossover(p1, p2 Genome, alpha float64) (Genome, Genome) {
	if len(p1) != len(p2) {
		panic("ArithmeticCrossover: parent genomes must have equal length")
	}
	if alpha < 0 || alpha > 1 {
		panic("ArithmeticCrossover: alpha must be in [0,1]")
	}
	n := len(p1)
	c1 := make(Genome, n)
	c2 := make(Genome, n)
	for i := 0; i < n; i++ {
		c1[i] = alpha*p1[i] + (1.0-alpha)*p2[i]
		c2[i] = alpha*p2[i] + (1.0-alpha)*p1[i]
	}
	return c1, c2
}

// ---------------------------------------------------------------------------
// Order crossover (OX) -- for permutation-encoded parameters
// ---------------------------------------------------------------------------

// OrderCrossover implements the OX1 order crossover operator for genomes
// treated as integer permutations (cast from float64). It is suited for
// permutation-encoded sub-sequences such as the blocked-hours ordering in
// LARSA configurations.
//
// The operator:
//  1. Selects a random sub-segment from p1 and copies it into c1.
//  2. Fills remaining positions in order from p2, skipping values already
//     present.
//  3. Symmetrically builds c2 from p2's segment and p1's order.
func OrderCrossover(p1, p2 Genome) (Genome, Genome) {
	if len(p1) != len(p2) {
		panic("OrderCrossover: parent genomes must have equal length")
	}
	n := len(p1)
	if n == 0 {
		return Genome{}, Genome{}
	}

	// Choose two random cut points.
	a := rand.Intn(n)
	b := rand.Intn(n)
	if a > b {
		a, b = b, a
	}

	c1 := oxFill(p1, p2, a, b)
	c2 := oxFill(p2, p1, a, b)
	return c1, c2
}

// oxFill builds one OX offspring: copies donor[a:b+1] into the child, then
// fills remaining slots in circular order from the values in source.
func oxFill(donor, source Genome, a, b int) Genome {
	n := len(donor)
	child := make(Genome, n)
	inSegment := make(map[float64]bool, b-a+1)

	// Copy the segment from donor.
	for i := a; i <= b; i++ {
		child[i] = donor[i]
		inSegment[donor[i]] = true
	}

	// Fill the rest in the order they appear in source (circular).
	pos := (b + 1) % n
	for _, v := range sourceOrder(source, b+1) {
		if !inSegment[v] {
			child[pos] = v
			pos = (pos + 1) % n
			if pos == a {
				pos = (b + 1) % n
			}
		}
	}
	return child
}

// sourceOrder returns the elements of src starting at index start, wrapping
// around, in circular order -- used by OX to maintain relative ordering.
func sourceOrder(src Genome, start int) []float64 {
	n := len(src)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = src[(start+i)%n]
	}
	return out
}

// ---------------------------------------------------------------------------
// Adaptive crossover operator
// ---------------------------------------------------------------------------

// crossoverOp is a function signature shared by all binary crossover
// operators that produce two offspring from two parents and a float64
// parameter.
type crossoverOp func(p1, p2 Genome, param float64) (Genome, Genome)

// operatorRecord tracks historical performance of a single crossover operator.
type operatorRecord struct {
	// wins is the number of times this operator produced the better offspring.
	wins int
	// uses is how many times this operator was used.
	uses int
	// sumImprovement accumulates total fitness improvement from this operator.
	sumImprovement float64
}

// AdaptiveCrossover tracks which crossover operator produced the best
// offspring over the last windowSize pairings and shifts selection
// probability toward the best operator (operator credit assignment).
//
// The credit assignment uses a sliding window of windowSize=100 pairings.
// Each operator's selection probability is proportional to its mean
// fitness improvement over the window. A minimum probability floor of
// minProb prevents any operator being permanently excluded.
type AdaptiveCrossover struct {
	mu         sync.Mutex
	operators  []crossoverOp
	params     []float64
	names      []string
	records    []operatorRecord
	history    []int // ring buffer of which op was used; length = windowSize
	histFit    []float64
	windowSize int
	minProb    float64
	rng        *rand.Rand
}

// NewAdaptiveCrossover constructs an AdaptiveCrossover with the four built-in
// operators pre-registered: Uniform, SBX, Arithmetic, and Order crossover.
// windowSize controls how many recent pairings are tracked (default 100).
func NewAdaptiveCrossover(windowSize int) *AdaptiveCrossover {
	if windowSize <= 0 {
		windowSize = 100
	}
	ac := &AdaptiveCrossover{
		windowSize: windowSize,
		minProb:    0.05,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
		history:    make([]int, 0, windowSize),
		histFit:    make([]float64, 0, windowSize),
	}

	// Register the four built-in operators.
	// Each is wrapped to match the crossoverOp signature.
	ac.Register("uniform", func(p1, p2 Genome, rate float64) (Genome, Genome) {
		return UniformCrossover(p1, p2, rate)
	}, 0.5)

	ac.Register("sbx", func(p1, p2 Genome, eta float64) (Genome, Genome) {
		return SimulatedBinaryCrossover(p1, p2, eta)
	}, 20.0)

	ac.Register("arithmetic", func(p1, p2 Genome, alpha float64) (Genome, Genome) {
		return ArithmeticCrossover(p1, p2, alpha)
	}, 0.5)

	ac.Register("order", func(p1, p2 Genome, _ float64) (Genome, Genome) {
		return OrderCrossover(p1, p2)
	}, 0.0)

	return ac
}

// Register adds a new crossover operator with a given name and default
// parameter value. Operators are assigned initial equal probabilities.
func (ac *AdaptiveCrossover) Register(name string, op crossoverOp, param float64) {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	ac.operators = append(ac.operators, op)
	ac.params = append(ac.params, param)
	ac.names = append(ac.names, name)
	ac.records = append(ac.records, operatorRecord{})
}

// probabilities computes the current selection probability vector.
// Each operator's probability is proportional to its mean improvement
// over the recent window, subject to a minimum floor of ac.minProb.
// Must be called with ac.mu held.
func (ac *AdaptiveCrossover) probabilities() []float64 {
	n := len(ac.operators)
	if n == 0 {
		return nil
	}

	// Count recent uses and improvements from the ring buffer.
	recentUses := make([]int, n)
	recentImp := make([]float64, n)
	for i, opIdx := range ac.history {
		recentUses[opIdx]++
		recentImp[opIdx] += ac.histFit[i]
	}

	scores := make([]float64, n)
	for i := range scores {
		if recentUses[i] > 0 {
			scores[i] = recentImp[i] / float64(recentUses[i])
		} else {
			scores[i] = 0
		}
		if scores[i] < ac.minProb {
			scores[i] = ac.minProb
		}
	}

	// Normalise to a probability distribution.
	sum := 0.0
	for _, s := range scores {
		sum += s
	}
	probs := make([]float64, n)
	for i := range probs {
		probs[i] = scores[i] / sum
	}
	return probs
}

// selectOperator picks an operator index via roulette wheel selection over
// the current probability distribution. Must be called with ac.mu held.
func (ac *AdaptiveCrossover) selectOperator() int {
	probs := ac.probabilities()
	r := ac.rng.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r <= cumulative {
			return i
		}
	}
	return len(probs) - 1
}

// Apply selects an operator adaptively, applies it to (p1, p2), and returns
// both offspring along with the operator name used. The caller should provide
// the improvement signal (e.g. best child fitness - parent fitness) via
// RecordOutcome after evaluating the offspring.
func (ac *AdaptiveCrossover) Apply(p1, p2 Genome) (Genome, Genome, int) {
	ac.mu.Lock()
	opIdx := ac.selectOperator()
	op := ac.operators[opIdx]
	param := ac.params[opIdx]
	ac.mu.Unlock()

	c1, c2 := op(p1, p2, param)
	return c1, c2, opIdx
}

// RecordOutcome updates the ring buffer with the fitness improvement achieved
// by the operator at opIdx. improvement should be max(child fitness) - best
// parent fitness (clamped to >= 0 for clarity, although negatives are valid).
func (ac *AdaptiveCrossover) RecordOutcome(opIdx int, improvement float64) {
	ac.mu.Lock()
	defer ac.mu.Unlock()

	if len(ac.history) >= ac.windowSize {
		// Remove oldest entry.
		ac.history = ac.history[1:]
		ac.histFit = ac.histFit[1:]
	}
	ac.history = append(ac.history, opIdx)
	ac.histFit = append(ac.histFit, improvement)

	// Update lifetime record.
	if opIdx < len(ac.records) {
		ac.records[opIdx].uses++
		ac.records[opIdx].sumImprovement += improvement
		if improvement > 0 {
			ac.records[opIdx].wins++
		}
	}
}

// OperatorStats returns a snapshot of win rates and mean improvements for all
// registered operators.
func (ac *AdaptiveCrossover) OperatorStats() []OperatorStat {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	stats := make([]OperatorStat, len(ac.operators))
	for i, rec := range ac.records {
		var winRate, meanImp float64
		if rec.uses > 0 {
			winRate = float64(rec.wins) / float64(rec.uses)
			meanImp = rec.sumImprovement / float64(rec.uses)
		}
		stats[i] = OperatorStat{
			Name:        ac.names[i],
			Uses:        rec.uses,
			Wins:        rec.wins,
			WinRate:     winRate,
			MeanImprove: meanImp,
		}
	}
	return stats
}

// OperatorStat holds summary statistics for a single crossover operator.
type OperatorStat struct {
	Name        string
	Uses        int
	Wins        int
	WinRate     float64
	MeanImprove float64
}

// CurrentProbabilities returns the current operator selection probabilities
// in the same order as the registered operators.
func (ac *AdaptiveCrossover) CurrentProbabilities() []float64 {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	return ac.probabilities()
}

// OperatorNames returns the names of all registered operators.
func (ac *AdaptiveCrossover) OperatorNames() []string {
	ac.mu.Lock()
	defer ac.mu.Unlock()
	names := make([]string, len(ac.names))
	copy(names, ac.names)
	return names
}
