// Package analysis provides post-hoc and in-flight analysis tools for the
// IAE genome evolution system.
package analysis

import (
	"fmt"
	"math"
	"sync"
)

// ---------------------------------------------------------------------------
// ConvergenceTracker
// ---------------------------------------------------------------------------

// generationRecord stores the best and mean fitness for one generation.
type generationRecord struct {
	Generation  int
	BestFitness float64
	MeanFitness float64
}

// ConvergenceTracker records the best fitness per generation and detects
// when evolution has plateaued.
//
// A plateau is detected when the best fitness has not improved by more than
// minImprovement over the last plateauWindow generations.
type ConvergenceTracker struct {
	mu             sync.RWMutex
	history        []generationRecord
	plateauWindow  int
	minImprovement float64
	// smoothedRate is the exponentially weighted moving average of
	// improvement per generation.
	smoothedRate float64
	ewmaAlpha    float64
}

// NewConvergenceTracker creates a ConvergenceTracker.
// plateauWindow controls how many generations without improvement triggers a
// plateau detection (default 20). minImprovement is the minimum fitness delta
// required to reset the plateau counter (default 1e-4).
func NewConvergenceTracker(plateauWindow int, minImprovement float64) *ConvergenceTracker {
	if plateauWindow <= 0 {
		plateauWindow = 20
	}
	if minImprovement <= 0 {
		minImprovement = 1e-4
	}
	return &ConvergenceTracker{
		plateauWindow:  plateauWindow,
		minImprovement: minImprovement,
		ewmaAlpha:      0.2,
	}
}

// Record adds the best and mean fitness observations for the given generation.
func (c *ConvergenceTracker) Record(generation int, bestFitness, meanFitness float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	rec := generationRecord{
		Generation:  generation,
		BestFitness: bestFitness,
		MeanFitness: meanFitness,
	}
	c.history = append(c.history, rec)

	// Update EWMA of improvement rate.
	n := len(c.history)
	if n >= 2 {
		delta := c.history[n-1].BestFitness - c.history[n-2].BestFitness
		if c.smoothedRate == 0 {
			c.smoothedRate = delta
		} else {
			c.smoothedRate = c.ewmaAlpha*delta + (1-c.ewmaAlpha)*c.smoothedRate
		}
	}
}

// IsPlateaued returns true if the best fitness has not improved by more than
// minImprovement over the last plateauWindow generations.
func (c *ConvergenceTracker) IsPlateaued() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	n := len(c.history)
	if n < c.plateauWindow {
		return false
	}
	window := c.history[n-c.plateauWindow:]
	base := window[0].BestFitness
	for _, rec := range window[1:] {
		if rec.BestFitness-base > c.minImprovement {
			return false
		}
	}
	return true
}

// ImprovementRate returns the EWMA of per-generation fitness improvements.
// Positive values indicate ongoing improvement; values near zero indicate
// convergence.
func (c *ConvergenceTracker) ImprovementRate() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.smoothedRate
}

// BestHistory returns a copy of the best fitness per generation.
func (c *ConvergenceTracker) BestHistory() []float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]float64, len(c.history))
	for i, r := range c.history {
		out[i] = r.BestFitness
	}
	return out
}

// LastN returns up to the last n generationRecords.
func (c *ConvergenceTracker) LastN(n int) []generationRecord {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if n <= 0 || len(c.history) == 0 {
		return nil
	}
	start := len(c.history) - n
	if start < 0 {
		start = 0
	}
	out := make([]generationRecord, len(c.history)-start)
	copy(out, c.history[start:])
	return out
}

// GenerationCount returns how many generations have been recorded.
func (c *ConvergenceTracker) GenerationCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.history)
}

// ---------------------------------------------------------------------------
// FitnessLandscape
// ---------------------------------------------------------------------------

// FitnessLandscape estimates the ruggedness of the fitness landscape by
// performing a random walk and computing the autocorrelation of the fitness
// series. High autocorrelation implies a smooth landscape; low autocorrelation
// implies ruggedness.
type FitnessLandscape struct {
	// WalkLength is the number of steps to take per random walk.
	WalkLength int
	// StepSize is the standard deviation of the Gaussian perturbation per step.
	StepSize float64
	// Evaluator is used to assess fitness at each walk point.
	Evaluator landscapeEvaluator
}

// landscapeEvaluator is a minimal interface used by FitnessLandscape.
type landscapeEvaluator interface {
	EvaluatePoint(genes []float64) float64
}

// EstimateRuggedness performs a random walk starting from seed and returns
// the ruggedness index in [0,1]. 0 = perfectly smooth; 1 = maximally rugged.
//
// The ruggedness index is 1 - |autocorr(fitness_series, lag=1)|. High
// autocorrelation (smooth) yields low ruggedness, and vice versa.
func (fl *FitnessLandscape) EstimateRuggedness(seed []float64) (float64, error) {
	if fl.WalkLength < 3 {
		return 0, fmt.Errorf("FitnessLandscape: WalkLength must be >= 3, got %d", fl.WalkLength)
	}
	if fl.StepSize <= 0 {
		return 0, fmt.Errorf("FitnessLandscape: StepSize must be > 0")
	}
	if fl.Evaluator == nil {
		return 0, fmt.Errorf("FitnessLandscape: Evaluator must not be nil")
	}
	n := len(seed)
	current := make([]float64, n)
	copy(current, seed)

	series := make([]float64, fl.WalkLength)
	for i := 0; i < fl.WalkLength; i++ {
		series[i] = fl.Evaluator.EvaluatePoint(current)
		// Perturb each dimension by Gaussian noise.
		for d := 0; d < n; d++ {
			current[d] += gaussianSample() * fl.StepSize
		}
	}

	ac := autocorrelation(series, 1)
	ruggedness := 1.0 - math.Abs(ac)
	if ruggedness < 0 {
		ruggedness = 0
	}
	if ruggedness > 1 {
		ruggedness = 1
	}
	return ruggedness, nil
}

// autocorrelation computes Pearson autocorrelation at the given lag.
func autocorrelation(series []float64, lag int) float64 {
	n := len(series)
	if n <= lag+1 {
		return 0
	}
	mean := 0.0
	for _, v := range series {
		mean += v
	}
	mean /= float64(n)

	num, den1, den2 := 0.0, 0.0, 0.0
	for i := 0; i < n-lag; i++ {
		a := series[i] - mean
		b := series[i+lag] - mean
		num += a * b
		den1 += a * a
		den2 += b * b
	}
	denom := math.Sqrt(den1 * den2)
	if denom < 1e-14 {
		return 0
	}
	return num / denom
}

// gaussianSample draws from N(0,1) using the Box-Muller transform with the
// package-level LCG source. This is only called from FitnessLandscape random
// walks where full cryptographic quality is not required.
func gaussianSample() float64 {
	u1 := pseudoRandFloat()
	u2 := pseudoRandFloat()
	if u1 < 1e-15 {
		u1 = 1e-15
	}
	return math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
}

// pseudoRandFloat is a simple LCG-based source used only within the landscape
// estimator where full randomness is not critical.
var lcgState uint64 = 0x123456789ABCDEF0

func pseudoRandFloat() float64 {
	lcgState = lcgState*6364136223846793005 + 1442695040888963407
	return float64(lcgState>>11) / float64(1<<53)
}

// ---------------------------------------------------------------------------
// EpochScheduler
// ---------------------------------------------------------------------------

// EvolutionPhase represents the current phase of the evolution run.
type EvolutionPhase int

const (
	// PhaseExploration is the early phase where broad search is preferred.
	PhaseExploration EvolutionPhase = iota
	// PhaseTransition is the middle phase where both are balanced.
	PhaseTransition
	// PhaseExploitation is the late phase where refinement is preferred.
	PhaseExploitation
)

// String returns a human-readable name for the phase.
func (p EvolutionPhase) String() string {
	switch p {
	case PhaseExploration:
		return "exploration"
	case PhaseTransition:
		return "transition"
	case PhaseExploitation:
		return "exploitation"
	}
	return "unknown"
}

// RateConfig holds the operator rate settings for one evolution phase.
type RateConfig struct {
	// MutationRate is the per-gene mutation probability.
	MutationRate float64
	// MutationSigma is the Gaussian step size.
	MutationSigma float64
	// CrossoverRate is the probability that two parents undergo crossover.
	CrossoverRate float64
	// EliteRatio is the fraction of the population to preserve as elites.
	EliteRatio float64
}

// EpochScheduler adjusts mutation and crossover rates based on the current
// convergence phase detected by a ConvergenceTracker.
//
// Phases:
//   - Exploration: high mutation, lower crossover (diversify the population)
//   - Transition:  balanced settings
//   - Exploitation: low mutation, high crossover (refine near best solutions)
type EpochScheduler struct {
	tracker         *ConvergenceTracker
	explorationCfg  RateConfig
	transitionCfg   RateConfig
	exploitationCfg RateConfig
	// plateauRestartThreshold: if plateaued AND smoothed improvement rate is
	// below this, recommend a population restart.
	plateauRestartThreshold float64
}

// NewEpochScheduler constructs an EpochScheduler with sensible defaults.
func NewEpochScheduler(tracker *ConvergenceTracker) *EpochScheduler {
	return &EpochScheduler{
		tracker: tracker,
		explorationCfg: RateConfig{
			MutationRate:  0.3,
			MutationSigma: 0.1,
			CrossoverRate: 0.6,
			EliteRatio:    0.05,
		},
		transitionCfg: RateConfig{
			MutationRate:  0.15,
			MutationSigma: 0.05,
			CrossoverRate: 0.75,
			EliteRatio:    0.10,
		},
		exploitationCfg: RateConfig{
			MutationRate:  0.05,
			MutationSigma: 0.01,
			CrossoverRate: 0.90,
			EliteRatio:    0.20,
		},
		plateauRestartThreshold: 1e-5,
	}
}

// CurrentPhase determines the current evolution phase based on generation
// count and convergence state.
//
//   - Generations 0..explorationGens-1: PhaseExploration
//   - Generations explorationGens..transitionGens-1: PhaseTransition
//   - Generations >= transitionGens: PhaseExploitation (or sooner if converged)
func (es *EpochScheduler) CurrentPhase(explorationGens, transitionGens int) EvolutionPhase {
	gen := es.tracker.GenerationCount()
	if gen < explorationGens {
		return PhaseExploration
	}
	if gen < transitionGens {
		return PhaseTransition
	}
	return PhaseExploitation
}

// Rates returns the RateConfig for the current phase.
func (es *EpochScheduler) Rates(explorationGens, transitionGens int) RateConfig {
	switch es.CurrentPhase(explorationGens, transitionGens) {
	case PhaseExploration:
		return es.explorationCfg
	case PhaseTransition:
		return es.transitionCfg
	default:
		return es.exploitationCfg
	}
}

// ShouldRestart returns true when the population is plateaued and the
// improvement rate has dropped below the restart threshold, indicating
// that the evolution run is stuck in a local optimum.
func (es *EpochScheduler) ShouldRestart() bool {
	if !es.tracker.IsPlateaued() {
		return false
	}
	return math.Abs(es.tracker.ImprovementRate()) < es.plateauRestartThreshold
}

// SetExplorationConfig overrides the default exploration phase rates.
func (es *EpochScheduler) SetExplorationConfig(cfg RateConfig) {
	es.explorationCfg = cfg
}

// SetExploitationConfig overrides the default exploitation phase rates.
func (es *EpochScheduler) SetExploitationConfig(cfg RateConfig) {
	es.exploitationCfg = cfg
}
