// Package parameter_evolution implements self-adaptive evolutionary operators
// for tuning SRFM strategy parameters. It extends the base evolution package
// with per-parameter sigma adaptation and multi-objective ranking.
package parameter_evolution

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// GenomeHistory records the fitness trajectory and parameter history of a
// single genome line for use in adaptive mutation rate control.
type GenomeHistory struct {
	// Fitness is the ordered sequence of fitness values (oldest first).
	Fitness []float64
	// Params is the ordered sequence of parameter snapshots.
	Params []map[string]float64
	// Timestamps records Unix nanoseconds when each entry was recorded.
	Timestamps []int64
}

// Add appends a new observation to the history.
func (h *GenomeHistory) Add(fitness float64, params map[string]float64) {
	cp := make(map[string]float64, len(params))
	for k, v := range params {
		cp[k] = v
	}
	h.Fitness = append(h.Fitness, fitness)
	h.Params = append(h.Params, cp)
	h.Timestamps = append(h.Timestamps, time.Now().UnixNano())
}

// RecentImproveRate returns the fraction of the last n mutations that improved
// fitness. Returns 0 if there are fewer than 2 observations.
func (h *GenomeHistory) RecentImproveRate(n int) float64 {
	total := len(h.Fitness)
	if total < 2 {
		return 0
	}
	start := total - n
	if start < 1 {
		start = 1
	}
	improved := 0
	for i := start; i < total; i++ {
		if h.Fitness[i] > h.Fitness[i-1] {
			improved++
		}
	}
	count := total - start
	if count == 0 {
		return 0
	}
	return float64(improved) / float64(count)
}

// ConstraintSpec describes a feasibility constraint on a set of parameters.
// The constraint is: sum of the named parameters must be <= Limit.
type ConstraintSpec struct {
	// Params names the parameters that participate in this constraint.
	Params []string
	// Limit is the upper bound on their sum (e.g. 1.0 for GARCH alpha+beta<1).
	Limit float64
}

// AdaptiveGenome is a genome that maintains per-parameter mutation sigmas and
// adapts its global mutation rate based on recent fitness history.
type AdaptiveGenome struct {
	mu sync.Mutex

	// Params holds the current parameter values.
	Params map[string]float64
	// Sigmas holds the per-parameter step sizes for Gaussian mutations.
	Sigmas map[string]float64
	// GlobalSigmaScale is a multiplier applied on top of all sigmas.
	GlobalSigmaScale float64
	// History tracks past fitness and parameter snapshots.
	History GenomeHistory
	// Constraints are feasibility constraints enforced after mutation.
	Constraints []ConstraintSpec
	// rng is the genome's private random source.
	rng *rand.Rand
	// tau is the self-adaptation learning rate.
	tau float64
}

// NewAdaptiveGenome creates an AdaptiveGenome with initial parameters.
// sigmaInit is the starting standard deviation for all parameters.
func NewAdaptiveGenome(
	params map[string]float64,
	sigmaInit float64,
	seed int64,
) *AdaptiveGenome {
	if sigmaInit <= 0 {
		sigmaInit = 0.1
	}
	sigmas := make(map[string]float64, len(params))
	for k := range params {
		sigmas[k] = sigmaInit
	}
	n := float64(len(params))
	tau := 1.0 / math.Sqrt(2*math.Sqrt(n))
	if tau == 0 {
		tau = 0.1
	}
	return &AdaptiveGenome{
		Params:           copyParams(params),
		Sigmas:           sigmas,
		GlobalSigmaScale: 1.0,
		rng:              rand.New(rand.NewSource(seed)),
		tau:              tau,
	}
}

// AdaptMutationRate applies the 1/5 success rule to adjust GlobalSigmaScale.
// If fewer than 20% of the last 20 mutations improved fitness, the global
// sigma scale is reduced by 20%. If more than 20% improved, it is increased
// by a factor of 1/(0.80) = 1.25.
func (g *AdaptiveGenome) AdaptMutationRate() {
	g.mu.Lock()
	defer g.mu.Unlock()

	rate := g.History.RecentImproveRate(20)
	if rate < 0.20 {
		g.GlobalSigmaScale *= 0.80
		if g.GlobalSigmaScale < 1e-6 {
			g.GlobalSigmaScale = 1e-6
		}
	} else {
		// More successes than the 1/5 threshold -- cautiously expand.
		g.GlobalSigmaScale *= 1.25
		if g.GlobalSigmaScale > 10.0 {
			g.GlobalSigmaScale = 10.0
		}
	}
}

// SelfAdaptiveGaussian mutates a single parameter using its per-parameter
// sigma. The sigma itself is also mutated via:
//
//	sigma' = sigma * exp(tau * N(0,1))
//
// The updated sigma is stored back into g.Sigmas[param].
// Returns the new value for the parameter.
func (g *AdaptiveGenome) SelfAdaptiveGaussian(param string, value float64) float64 {
	g.mu.Lock()
	defer g.mu.Unlock()

	sigma, ok := g.Sigmas[param]
	if !ok || sigma <= 0 {
		sigma = 0.1
	}
	// Mutate sigma.
	newSigma := sigma * math.Exp(g.tau*g.rng.NormFloat64())
	if newSigma < 1e-8 {
		newSigma = 1e-8
	}
	g.Sigmas[param] = newSigma

	// Mutate value.
	newVal := value + g.GlobalSigmaScale*newSigma*g.rng.NormFloat64()
	return newVal
}

// CorrelatedMutation mutates all parameters simultaneously using a covariance
// matrix to produce correlated perturbations. The Cholesky factor of cov is
// used to transform a standard normal vector into a correlated sample.
//
// cov must be n x n where n == len(params). Parameter order is determined by
// the sorted keys of params.
//
// After mutation the result is projected back to the feasible region defined
// by g.Constraints.
func (g *AdaptiveGenome) CorrelatedMutation(
	params map[string]float64,
	cov [][]float64,
) (map[string]float64, error) {
	keys := sortedKeys(params)
	n := len(keys)
	if len(cov) != n {
		return nil, fmt.Errorf("cov dimension %d does not match params count %d", len(cov), n)
	}
	for i, row := range cov {
		if len(row) != n {
			return nil, fmt.Errorf("cov row %d has length %d, expected %d", i, len(row), n)
		}
	}

	// Cholesky decomposition of cov.
	L, err := cholesky(cov)
	if err != nil {
		// Fall back to independent mutations if cov is not PD.
		result := make(map[string]float64, n)
		for k, v := range params {
			result[k] = g.SelfAdaptiveGaussian(k, v)
		}
		return result, nil
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	// Sample z ~ N(0, I_n).
	z := make([]float64, n)
	for i := range z {
		z[i] = g.rng.NormFloat64()
	}

	// Transform to correlated sample: delta = L * z.
	delta := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			delta[i] += L[i][j] * z[j]
		}
	}

	result := make(map[string]float64, n)
	for i, k := range keys {
		sigma := g.Sigmas[k]
		if sigma <= 0 {
			sigma = 0.1
		}
		result[k] = params[k] + g.GlobalSigmaScale*sigma*delta[i]
	}

	// Repair constraints.
	g.repairConstraints(result)
	return result, nil
}

// RecordFitness appends a fitness observation to the genome's history.
func (g *AdaptiveGenome) RecordFitness(fitness float64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.History.Add(fitness, g.Params)
}

// repairConstraints projects params back to the feasible region by scaling
// down any group of parameters whose sum exceeds the constraint limit.
// Caller must hold g.mu.
func (g *AdaptiveGenome) repairConstraints(params map[string]float64) {
	for _, c := range g.Constraints {
		sum := 0.0
		for _, k := range c.Params {
			if v, ok := params[k]; ok {
				sum += v
			}
		}
		if sum <= c.Limit {
			continue
		}
		// Scale all participants proportionally.
		scale := c.Limit / sum
		for _, k := range c.Params {
			if v, ok := params[k]; ok {
				params[k] = v * scale
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Linear algebra helpers
// ---------------------------------------------------------------------------

// cholesky computes the lower-triangular Cholesky factor L of a symmetric
// positive-definite matrix A, such that A = L * L^T.
// Returns an error if A is not positive definite.
func cholesky(A [][]float64) ([][]float64, error) {
	n := len(A)
	L := make([][]float64, n)
	for i := range L {
		L[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := A[i][j]
			for k := 0; k < j; k++ {
				sum -= L[i][k] * L[j][k]
			}
			if i == j {
				if sum <= 0 {
					return nil, fmt.Errorf("matrix is not positive definite at [%d,%d]", i, j)
				}
				L[i][j] = math.Sqrt(sum)
			} else {
				if L[j][j] == 0 {
					return nil, fmt.Errorf("zero diagonal in Cholesky at [%d,%d]", j, j)
				}
				L[i][j] = sum / L[j][j]
			}
		}
	}
	return L, nil
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

// sortedKeys returns the keys of m in sorted order.
func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// copyParams returns a deep copy of a parameter map.
func copyParams(p map[string]float64) map[string]float64 {
	cp := make(map[string]float64, len(p))
	for k, v := range p {
		cp[k] = v
	}
	return cp
}
