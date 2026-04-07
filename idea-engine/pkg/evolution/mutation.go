package evolution

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
)

// ---------------------------------------------------------------------------
// Gaussian mutation
// ---------------------------------------------------------------------------

// GaussianMutation adds Gaussian noise N(0, sigma) to each gene independently
// with probability rate. It does not enforce bounds -- callers should clamp
// after mutation if hard bounds are required.
//
// rate=1.0 mutates every gene; rate=1/n mutates one gene on average.
func GaussianMutation(g Genome, sigma, rate float64) Genome {
	out := g.Clone()
	for i := range out {
		if rand.Float64() < rate {
			out[i] += rand.NormFloat64() * sigma
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Polynomial mutation
// ---------------------------------------------------------------------------

// PolynomialMutation applies the polynomial mutation operator for bounded
// real-valued parameters. It preserves the parameter range defined by bounds.
//
// eta controls the perturbation distribution:
//   - large eta -> small perturbations (exploitation)
//   - small eta -> large perturbations (exploration)
//
// bounds[i] = [lo, hi] for gene i. If bounds is nil or shorter than the
// genome, genes beyond bounds are left unchanged.
//
// Reference: Deb & Goyal (1996) "A combined genetic adaptive search (GeneAS)
// for engineering design".
func PolynomialMutation(g Genome, eta float64, bounds [][2]float64) Genome {
	out := g.Clone()
	for i := range out {
		if i >= len(bounds) {
			continue
		}
		lo, hi := bounds[i][0], bounds[i][1]
		if hi <= lo {
			continue
		}
		u := rand.Float64()
		var delta float64
		if u < 0.5 {
			deltaL := (out[i] - lo) / (hi - lo)
			delta = math.Pow(2.0*u+(1.0-2.0*u)*math.Pow(1.0-deltaL, eta+1.0), 1.0/(eta+1.0)) - 1.0
		} else {
			deltaR := (hi - out[i]) / (hi - lo)
			delta = 1.0 - math.Pow(2.0*(1.0-u)+(2.0*u-1.0)*math.Pow(1.0-deltaR, eta+1.0), 1.0/(eta+1.0))
		}
		out[i] = out[i] + delta*(hi-lo)
		// Clamp to bounds.
		if out[i] < lo {
			out[i] = lo
		}
		if out[i] > hi {
			out[i] = hi
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Cauchy mutation
// ---------------------------------------------------------------------------

// cauchySample draws a single sample from the standard Cauchy distribution
// using the inverse CDF method: C(0,1) = tan(pi*(u - 0.5)).
func cauchySample() float64 {
	u := rand.Float64()
	// Avoid the degenerate tails at exactly 0 or 1.
	for u == 0.0 || u == 1.0 {
		u = rand.Float64()
	}
	return math.Tan(math.Pi * (u - 0.5))
}

// CauchyMutation adds Cauchy-distributed noise to every gene in the genome.
// The heavy tail of the Cauchy distribution gives a higher probability of
// large jumps than Gaussian mutation, which helps escape local optima.
//
// scale acts as the half-width parameter of the Cauchy distribution.
// All genes are mutated (no rate parameter -- use the caller to gate per gene
// if needed).
func CauchyMutation(g Genome, scale float64) Genome {
	out := g.Clone()
	for i := range out {
		out[i] += scale * cauchySample()
	}
	return out
}

// ---------------------------------------------------------------------------
// Self-adaptive step size mutation (1/5 rule)
// ---------------------------------------------------------------------------

// AdaptiveSigmaMutation implements self-adaptive Gaussian mutation with one
// sigma per gene. The step sizes are updated according to the 1/5 success
// rule: if more than 1/5 of recent mutations were improvements, increase
// sigma; otherwise decrease.
//
// Reference: Rechenberg (1973) "Evolutionsstrategie: Optimierung technischer
// Systeme nach Prinzipien der biologischen Evolution".
type AdaptiveSigmaMutation struct {
	mu      sync.Mutex
	sigmas  []float64
	// window stores recent success flags (1=improved, 0=no improvement).
	window  []int
	winSize int
	// cFactor is the multiplicative adjustment factor (Rechenberg recommends
	// 0.817 for n-dimensional problems).
	cFactor float64
	// sigmaMin and sigmaMax prevent sigma collapsing to zero or diverging.
	sigmaMin float64
	sigmaMax float64
}

// NewAdaptiveSigmaMutation creates a new AdaptiveSigmaMutation for a genome
// of the given dimension. initSigma is the starting step size for all genes.
func NewAdaptiveSigmaMutation(dim int, initSigma float64) *AdaptiveSigmaMutation {
	sigmas := make([]float64, dim)
	for i := range sigmas {
		sigmas[i] = initSigma
	}
	return &AdaptiveSigmaMutation{
		sigmas:   sigmas,
		winSize:  10 * dim,
		window:   make([]int, 0, 10*dim),
		cFactor:  0.817,
		sigmaMin: 1e-6,
		sigmaMax: 10.0,
	}
}

// Mutate applies per-gene Gaussian mutation using the current sigma vector
// and returns the offspring genome. The caller must call RecordOutcome after
// evaluating the offspring to drive sigma adaptation.
func (a *AdaptiveSigmaMutation) Mutate(g Genome) Genome {
	a.mu.Lock()
	defer a.mu.Unlock()

	out := g.Clone()
	n := len(out)
	if n > len(a.sigmas) {
		// Extend sigmas if the genome grew (should not happen in practice).
		extra := make([]float64, n-len(a.sigmas))
		for i := range extra {
			extra[i] = a.sigmas[len(a.sigmas)-1]
		}
		a.sigmas = append(a.sigmas, extra...)
	}
	for i := range out {
		out[i] += rand.NormFloat64() * a.sigmas[i]
	}
	return out
}

// RecordOutcome informs the adapter whether the last mutation improved
// fitness. improved=true increments the success counter; false decrements it.
// After updating the ring buffer the 1/5 rule is applied to all sigmas.
func (a *AdaptiveSigmaMutation) RecordOutcome(improved bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	val := 0
	if improved {
		val = 1
	}
	if len(a.window) >= a.winSize {
		a.window = a.window[1:]
	}
	a.window = append(a.window, val)

	// Count successes.
	successes := 0
	for _, v := range a.window {
		successes += v
	}
	rate := float64(successes) / float64(len(a.window))

	for i := range a.sigmas {
		if rate > 0.2 {
			a.sigmas[i] /= a.cFactor
		} else {
			a.sigmas[i] *= a.cFactor
		}
		if a.sigmas[i] < a.sigmaMin {
			a.sigmas[i] = a.sigmaMin
		}
		if a.sigmas[i] > a.sigmaMax {
			a.sigmas[i] = a.sigmaMax
		}
	}
}

// Sigmas returns a copy of the current per-gene step sizes.
func (a *AdaptiveSigmaMutation) Sigmas() []float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]float64, len(a.sigmas))
	copy(out, a.sigmas)
	return out
}

// ---------------------------------------------------------------------------
// Parameter-aware mutation
// ---------------------------------------------------------------------------

// ParamSchema is the deserialized form of param_schema.json. Each entry
// defines the bounds and scale for one LARSA parameter.
type ParamSchema struct {
	Parameters []ParamDef `json:"parameters"`
}

// ParamDef holds the definition of a single LARSA parameter slot.
type ParamDef struct {
	// Name is the human-readable parameter name.
	Name string `json:"name"`
	// Min is the hard lower bound.
	Min float64 `json:"min"`
	// Max is the hard upper bound.
	Max float64 `json:"max"`
	// Scale is the suggested mutation scale (sigma = scale * (max-min)).
	Scale float64 `json:"scale"`
	// Type is "continuous", "integer", or "categorical".
	Type string `json:"type"`
}

// ParameterAwareMutation applies different mutation scales and strategies per
// gene based on parameter definitions loaded from a param_schema.json file.
//
// For continuous parameters Gaussian mutation with sigma = scale*(max-min) is
// applied, followed by clamping to [min, max].
// For integer parameters the result is rounded to the nearest integer.
// For categorical parameters a random valid integer in [0, max] is substituted
// with probability rate.
type ParameterAwareMutation struct {
	schema ParamSchema
	// rate is the per-gene mutation probability.
	rate float64
}

// NewParameterAwareMutation loads a ParamSchema from schemaPath and returns a
// ParameterAwareMutation. schemaPath is typically "param_schema.json" relative
// to the working directory of the evolution service.
func NewParameterAwareMutation(schemaPath string, rate float64) (*ParameterAwareMutation, error) {
	data, err := os.ReadFile(schemaPath)
	if err != nil {
		return nil, fmt.Errorf("NewParameterAwareMutation: read %q: %w", schemaPath, err)
	}
	var schema ParamSchema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("NewParameterAwareMutation: parse schema: %w", err)
	}
	if rate <= 0 || rate > 1 {
		return nil, fmt.Errorf("NewParameterAwareMutation: rate must be in (0,1], got %f", rate)
	}
	return &ParameterAwareMutation{schema: schema, rate: rate}, nil
}

// Mutate applies parameter-aware mutation to g and returns the offspring.
// Genes beyond the schema length are mutated with a default Gaussian sigma
// of 0.01*(max-min) if bounds can be inferred, or left unchanged.
func (p *ParameterAwareMutation) Mutate(g Genome) Genome {
	out := g.Clone()
	defs := p.schema.Parameters
	for i := range out {
		if rand.Float64() >= p.rate {
			continue
		}
		if i >= len(defs) {
			// No schema info -- apply a small Gaussian nudge.
			out[i] += rand.NormFloat64() * 0.01
			continue
		}
		def := defs[i]
		switch def.Type {
		case "categorical":
			// Choose uniformly from [0, int(def.Max)].
			max := int(def.Max)
			if max > 0 {
				out[i] = float64(rand.Intn(max + 1))
			}
		case "integer":
			sigma := def.Scale * (def.Max - def.Min)
			if sigma <= 0 {
				sigma = 1.0
			}
			out[i] += rand.NormFloat64() * sigma
			out[i] = math.Round(out[i])
			if out[i] < def.Min {
				out[i] = def.Min
			}
			if out[i] > def.Max {
				out[i] = def.Max
			}
		default: // "continuous"
			sigma := def.Scale * (def.Max - def.Min)
			if sigma <= 0 {
				sigma = 0.01 * (def.Max - def.Min)
			}
			out[i] += rand.NormFloat64() * sigma
			if out[i] < def.Min {
				out[i] = def.Min
			}
			if out[i] > def.Max {
				out[i] = def.Max
			}
		}
	}
	return out
}

// Bounds extracts the [][2]float64 bounds slice from the schema, which can be
// passed to PolynomialMutation.
func (p *ParameterAwareMutation) Bounds() [][2]float64 {
	out := make([][2]float64, len(p.schema.Parameters))
	for i, def := range p.schema.Parameters {
		out[i] = [2]float64{def.Min, def.Max}
	}
	return out
}
