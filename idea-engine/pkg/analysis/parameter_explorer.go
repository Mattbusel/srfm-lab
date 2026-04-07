package analysis

import (
	"crypto/sha256"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// ---------------------------------------------------------------------------
// ParameterExplorer
// ---------------------------------------------------------------------------

// ParameterExplorer systematically explores the parameter space to find
// regions of high fitness and characterise performance boundaries.
//
// ParamBounds maps each parameter name to a [min, max] pair.
// Evaluator takes a named parameter map and returns a fitness scalar.
// Cache stores previously evaluated parameter vectors to avoid redundant
// evaluations; the key is a hex SHA-256 digest of the sorted param values.
type ParameterExplorer struct {
	ParamBounds map[string][2]float64
	Evaluator   func(map[string]float64) float64
	Cache       map[string]float64
	// RNG is the random source. If nil a default source is used.
	RNG *rand.Rand
}

// NewParameterExplorer creates an explorer with the given bounds and evaluator.
func NewParameterExplorer(
	bounds map[string][2]float64,
	evaluator func(map[string]float64) float64,
) *ParameterExplorer {
	return &ParameterExplorer{
		ParamBounds: bounds,
		Evaluator:   evaluator,
		Cache:       make(map[string]float64),
		RNG:         rand.New(rand.NewSource(1337)),
	}
}

// ---------------------------------------------------------------------------
// SearchResult
// ---------------------------------------------------------------------------

// SearchResult holds one evaluated parameter configuration.
type SearchResult struct {
	Params  map[string]float64
	Fitness float64
}

// ---------------------------------------------------------------------------
// GridResult
// ---------------------------------------------------------------------------

// GridResult holds the output of a 2-D grid search over two parameters.
type GridResult struct {
	Param1    string
	Param2    string
	Values1   []float64   // param1 axis values
	Values2   []float64   // param2 axis values
	Fitness   [][]float64 // Fitness[i][j] corresponds to Values1[i], Values2[j]
	BestI     int         // row index of global best
	BestJ     int         // col index of global best
	BestValue float64
}

// GridSearch performs a 2-D grid search over param1 and param2 at the given
// resolution (number of points per axis). All other parameters are held at
// the midpoint of their bounds. Returns a GridResult suitable for
// visualisation or downstream analysis.
func (e *ParameterExplorer) GridSearch(param1, param2 string, resolution int) GridResult {
	if resolution < 2 {
		resolution = 2
	}
	base := e.midpointParams()

	v1 := linspace(e.ParamBounds[param1][0], e.ParamBounds[param1][1], resolution)
	v2 := linspace(e.ParamBounds[param2][0], e.ParamBounds[param2][1], resolution)

	grid := GridResult{
		Param1:  param1,
		Param2:  param2,
		Values1: v1,
		Values2: v2,
		Fitness: make([][]float64, resolution),
	}
	grid.BestValue = math.Inf(-1)

	for i, p1 := range v1 {
		grid.Fitness[i] = make([]float64, resolution)
		for j, p2 := range v2 {
			params := copyParams(base)
			params[param1] = p1
			params[param2] = p2
			f := e.evaluate(params)
			grid.Fitness[i][j] = f
			if f > grid.BestValue {
				grid.BestValue = f
				grid.BestI = i
				grid.BestJ = j
			}
		}
	}
	return grid
}

// RandomSearch draws n uniform random samples from the full parameter space
// and returns the evaluated results sorted by fitness descending.
func (e *ParameterExplorer) RandomSearch(n int) []SearchResult {
	rng := e.rng()
	results := make([]SearchResult, 0, n)
	for i := 0; i < n; i++ {
		params := make(map[string]float64, len(e.ParamBounds))
		for name, bounds := range e.ParamBounds {
			params[name] = bounds[0] + rng.Float64()*(bounds[1]-bounds[0])
		}
		f := e.evaluate(params)
		results = append(results, SearchResult{Params: params, Fitness: f})
	}
	sortResults(results)
	return results
}

// SobolSearch draws n quasi-random samples using a simple bitwise Van der
// Corput sequence (Sobol-inspired) to achieve low-discrepancy coverage.
// For d dimensions, successive dimensions use different bit-reversal bases.
func (e *ParameterExplorer) SobolSearch(n int) []SearchResult {
	names := e.sortedParamNames()
	d := len(names)
	results := make([]SearchResult, 0, n)

	for i := 0; i < n; i++ {
		params := make(map[string]float64, d)
		for dim, name := range names {
			u := vanDerCorput(i+1, primes[dim%len(primes)])
			bounds := e.ParamBounds[name]
			params[name] = bounds[0] + u*(bounds[1]-bounds[0])
		}
		f := e.evaluate(params)
		results = append(results, SearchResult{Params: params, Fitness: f})
	}
	sortResults(results)
	return results
}

// LocalSearch draws n random samples within a hypercube of the given radius
// (as a fraction of each parameter's range) around the center point.
// Results are sorted by fitness descending.
func (e *ParameterExplorer) LocalSearch(center map[string]float64, radius float64, n int) []SearchResult {
	rng := e.rng()
	results := make([]SearchResult, 0, n)
	for i := 0; i < n; i++ {
		params := make(map[string]float64, len(e.ParamBounds))
		for name, bounds := range e.ParamBounds {
			r := (bounds[1] - bounds[0]) * radius
			v := center[name] + (rng.Float64()*2-1)*r
			// clamp to bounds
			if v < bounds[0] {
				v = bounds[0]
			}
			if v > bounds[1] {
				v = bounds[1]
			}
			params[name] = v
		}
		f := e.evaluate(params)
		results = append(results, SearchResult{Params: params, Fitness: f})
	}
	sortResults(results)
	return results
}

// FindFeasibleRegion returns all SearchResults from the given slice whose
// fitness is >= minSharpe. Useful for understanding the extent of the
// acceptable parameter region.
func (e *ParameterExplorer) FindFeasibleRegion(minSharpe float64, results []SearchResult) []SearchResult {
	var out []SearchResult
	for _, r := range results {
		if r.Fitness >= minSharpe {
			out = append(out, r)
		}
	}
	return out
}

// ParameterImportance computes an ANOVA-style importance score for each
// parameter by measuring how much variance in fitness is explained by
// stratifying along that parameter dimension. Higher score = more important.
//
// The score for parameter p is: variance(bin_means) / variance(all_fitnesses).
// Returns a map from parameter name to importance in [0, 1].
func (e *ParameterExplorer) ParameterImportance(results []SearchResult) map[string]float64 {
	if len(results) < 2 {
		return nil
	}

	fits := make([]float64, len(results))
	for i, r := range results {
		fits[i] = r.Fitness
	}
	totalVar := variance(fits)
	if totalVar == 0 {
		// All fitnesses identical -- no parameter matters
		out := make(map[string]float64, len(e.ParamBounds))
		for name := range e.ParamBounds {
			out[name] = 0.0
		}
		return out
	}

	importance := make(map[string]float64, len(e.ParamBounds))
	nBins := 5

	for name, bounds := range e.ParamBounds {
		rangeW := bounds[1] - bounds[0]
		if rangeW == 0 {
			importance[name] = 0.0
			continue
		}
		bins := make([][]float64, nBins)
		for _, r := range results {
			v := r.Params[name]
			b := int((v - bounds[0]) / rangeW * float64(nBins))
			if b >= nBins {
				b = nBins - 1
			}
			bins[b] = append(bins[b], r.Fitness)
		}
		binMeans := make([]float64, 0, nBins)
		for _, bin := range bins {
			if len(bin) > 0 {
				binMeans = append(binMeans, mean(bin))
			}
		}
		if len(binMeans) < 2 {
			importance[name] = 0.0
			continue
		}
		importance[name] = math.Min(1.0, variance(binMeans)/totalVar)
	}
	return importance
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// evaluate calls the Evaluator with caching. The cache key is the SHA-256
// of the canonical string representation of sorted param name=value pairs.
func (e *ParameterExplorer) evaluate(params map[string]float64) float64 {
	key := paramKey(params)
	if v, ok := e.Cache[key]; ok {
		return v
	}
	v := e.Evaluator(params)
	e.Cache[key] = v
	return v
}

// paramKey returns a stable hash key for a parameter map.
func paramKey(params map[string]float64) string {
	names := make([]string, 0, len(params))
	for k := range params {
		names = append(names, k)
	}
	sort.Strings(names)
	s := ""
	for _, n := range names {
		s += fmt.Sprintf("%s=%.10f;", n, params[n])
	}
	h := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%x", h)
}

// midpointParams returns a params map with each parameter set to the midpoint
// of its bounds.
func (e *ParameterExplorer) midpointParams() map[string]float64 {
	out := make(map[string]float64, len(e.ParamBounds))
	for name, bounds := range e.ParamBounds {
		out[name] = (bounds[0] + bounds[1]) / 2.0
	}
	return out
}

// sortedParamNames returns parameter names in sorted order for determinism.
func (e *ParameterExplorer) sortedParamNames() []string {
	names := make([]string, 0, len(e.ParamBounds))
	for k := range e.ParamBounds {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// copyParams returns a shallow copy of a params map.
func copyParams(p map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(p))
	for k, v := range p {
		out[k] = v
	}
	return out
}

// rng returns the explorer's RNG, initialising one if nil.
func (e *ParameterExplorer) rng() *rand.Rand {
	if e.RNG != nil {
		return e.RNG
	}
	e.RNG = rand.New(rand.NewSource(1337))
	return e.RNG
}

// linspace returns n evenly-spaced values from lo to hi inclusive.
func linspace(lo, hi float64, n int) []float64 {
	if n == 1 {
		return []float64{lo}
	}
	out := make([]float64, n)
	step := (hi - lo) / float64(n-1)
	for i := range out {
		out[i] = lo + float64(i)*step
	}
	return out
}

// vanDerCorput returns the Van der Corput sequence value for index n in the
// given base. Produces values in (0, 1).
func vanDerCorput(n, base int) float64 {
	result := 0.0
	denom := 1.0
	for n > 0 {
		denom *= float64(base)
		result += float64(n%base) / denom
		n /= base
	}
	return result
}

// primes is used to select Van der Corput bases per dimension.
var primes = []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

// sortResults sorts results by fitness descending (best first).
func sortResults(results []SearchResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Fitness > results[j].Fitness
	})
}

// variance returns the population variance of xs.
func variance(xs []float64) float64 {
	if len(xs) < 2 {
		return 0.0
	}
	m := mean(xs)
	s := 0.0
	for _, v := range xs {
		d := v - m
		s += d * d
	}
	return s / float64(len(xs))
}
