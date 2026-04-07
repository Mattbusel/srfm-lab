package analysis

import (
	"math"
	"math/rand"
)

// ---------------------------------------------------------------------------
// FitnessLandscape
// ---------------------------------------------------------------------------

// FitnessLandscape builds a local model of the fitness landscape around the
// current best genome using perturbation experiments. It does NOT call the
// evaluator in parallel -- callers should set Evaluator to a goroutine-safe
// function if concurrent use is desired.
type FitnessLandscape struct {
	// Center is the genome around which the landscape is explored.
	Center []float64
	// Perturbation is the fraction of the gene range to perturb each gene.
	// A value of 0.05 means each gene is shifted by up to 5% of its range.
	Perturbation float64
	// GeneRanges[i] = [min, max] for gene i. If nil, range defaults to [0,1].
	GeneRanges [][2]float64
	// Evaluator returns the fitness for a given genome.
	Evaluator func([]float64) float64
	// RNG is the random source. If nil, a default source is used.
	RNG *rand.Rand
}

// NewFitnessLandscape creates a FitnessLandscape centred on the given genome.
// perturbation is the fractional perturbation size; evaluator must be set.
func NewFitnessLandscape(center []float64, perturbation float64, evaluator func([]float64) float64) *FitnessLandscape {
	c := make([]float64, len(center))
	copy(c, center)
	return &FitnessLandscape{
		Center:       c,
		Perturbation: perturbation,
		Evaluator:    evaluator,
		RNG:          rand.New(rand.NewSource(42)),
	}
}

// ---------------------------------------------------------------------------
// LandscapeMap
// ---------------------------------------------------------------------------

// LandscapePoint is one sampled point in the fitness landscape.
type LandscapePoint struct {
	Genome  []float64
	Fitness float64
	// Delta is the perturbation vector (genome - center)
	Delta []float64
}

// LandscapeMap stores the results of a perturbation study around a centre point.
type LandscapeMap struct {
	Center      []float64
	Points      []LandscapePoint
	BestFitness float64
	BestGenome  []float64
	// CenterFitness is the fitness of the unperturbed center.
	CenterFitness float64
}

// MapNeighborhood draws nSamples random perturbations around Center, evaluates
// each one, and returns the resulting LandscapeMap.
func (fl *FitnessLandscape) MapNeighborhood(nSamples int) LandscapeMap {
	d := len(fl.Center)
	rng := fl.rng()

	lm := LandscapeMap{
		Center: fl.Center,
	}
	// Evaluate center itself
	lm.CenterFitness = fl.Evaluator(fl.Center)
	lm.BestFitness = lm.CenterFitness
	lm.BestGenome = fl.Center

	for s := 0; s < nSamples; s++ {
		g := make([]float64, d)
		delta := make([]float64, d)
		for k := 0; k < d; k++ {
			rangeK := fl.geneRange(k)
			shift := (rng.Float64()*2 - 1) * fl.Perturbation * rangeK
			delta[k] = shift
			g[k] = fl.Center[k] + shift
		}
		fitness := fl.Evaluator(g)
		pt := LandscapePoint{
			Genome:  g,
			Fitness: fitness,
			Delta:   delta,
		}
		lm.Points = append(lm.Points, pt)
		if fitness > lm.BestFitness {
			lm.BestFitness = fitness
			lm.BestGenome = g
		}
	}
	return lm
}

// ---------------------------------------------------------------------------
// Gradient and curvature analysis
// ---------------------------------------------------------------------------

// EstimateGradient computes the finite-difference gradient of the fitness
// function at Center. Each gene is perturbed by Perturbation * geneRange(k)
// and the central difference formula is used.
func (fl *FitnessLandscape) EstimateGradient() []float64 {
	d := len(fl.Center)
	grad := make([]float64, d)
	for k := 0; k < d; k++ {
		h := fl.Perturbation * fl.geneRange(k)
		if h == 0 {
			h = 1e-6
		}
		fwd := fl.perturbEval(k, h)
		bwd := fl.perturbEval(k, -h)
		grad[k] = (fwd - bwd) / (2 * h)
	}
	return grad
}

// ComputeHessian computes the second-order finite-difference approximation
// of the Hessian matrix at Center. Diagonal entries use the standard
// second-difference formula; off-diagonal entries use a mixed partial
// approximation. Returns a d x d matrix.
func (fl *FitnessLandscape) ComputeHessian() [][]float64 {
	d := len(fl.Center)
	H := make([][]float64, d)
	for i := range H {
		H[i] = make([]float64, d)
	}

	f0 := fl.Evaluator(fl.Center)

	for i := 0; i < d; i++ {
		hi := fl.perturbStep(i)
		// diagonal: (f(x+h) - 2f(x) + f(x-h)) / h^2
		fpi := fl.perturbEval(i, hi)
		fmi := fl.perturbEval(i, -hi)
		H[i][i] = (fpi - 2*f0 + fmi) / (hi * hi)

		for j := i + 1; j < d; j++ {
			hj := fl.perturbStep(j)
			// mixed partial: (f(x+hi,x+hj) - f(x+hi,x-hj) - f(x-hi,x+hj) + f(x-hi,x-hj)) / (4*hi*hj)
			fpp := fl.perturbEval2(i, hi, j, hj)
			fpm := fl.perturbEval2(i, hi, j, -hj)
			fmp := fl.perturbEval2(i, -hi, j, hj)
			fmm := fl.perturbEval2(i, -hi, j, -hj)
			mixed := (fpp - fpm - fmp + fmm) / (4 * hi * hj)
			H[i][j] = mixed
			H[j][i] = mixed
		}
	}
	return H
}

// LocalCurvature returns the average diagonal of the Hessian (trace / d).
// A positive value indicates a bowl-shaped landscape (good for optimisation);
// a negative value indicates a local peak or ridge.
func (fl *FitnessLandscape) LocalCurvature() float64 {
	H := fl.ComputeHessian()
	d := len(H)
	if d == 0 {
		return 0.0
	}
	trace := 0.0
	for i := 0; i < d; i++ {
		trace += H[i][i]
	}
	return trace / float64(d)
}

// RidgednessScore measures the variance in gradient direction across a set of
// perturbation samples. High variance means the landscape has many ridges
// and is difficult to navigate by gradient methods.
func (fl *FitnessLandscape) RidgednessScore() float64 {
	// Sample a number of random nearby points and compute their local gradients,
	// then measure the variance of the angle between each gradient and the
	// gradient at center.
	nSamples := 20
	rng := fl.rng()
	d := len(fl.Center)

	centerGrad := fl.EstimateGradient()
	centerNorm := vecNorm(centerGrad)
	if centerNorm == 0 {
		return 0.0
	}

	cosines := make([]float64, 0, nSamples)
	for s := 0; s < nSamples; s++ {
		// Small perturbation to center
		g := make([]float64, d)
		for k := 0; k < d; k++ {
			h := fl.perturbStep(k)
			g[k] = fl.Center[k] + (rng.Float64()*2-1)*h
		}

		// Local gradient at this perturbed point -- temporarily change center
		savedCenter := fl.Center
		fl.Center = g
		localGrad := fl.EstimateGradient()
		fl.Center = savedCenter

		localNorm := vecNorm(localGrad)
		if localNorm == 0 {
			continue
		}
		dot := 0.0
		for k := 0; k < d; k++ {
			dot += (centerGrad[k] / centerNorm) * (localGrad[k] / localNorm)
		}
		// Clamp to [-1,1] for safety before acos
		dot = math.Max(-1.0, math.Min(1.0, dot))
		cosines = append(cosines, dot)
	}

	if len(cosines) == 0 {
		return 0.0
	}
	m := mean(cosines)
	return stddev(cosines, m)
}

// FunnelnessScore measures the correlation between an individual's distance
// to the current best genome and its fitness (using the LandscapeMap data).
// A strong negative correlation indicates a funnel-shaped landscape, which is
// favourable for evolutionary search.
func (fl *FitnessLandscape) FunnelnessScore() float64 {
	lm := fl.MapNeighborhood(50)
	n := len(lm.Points)
	if n < 2 {
		return 0.0
	}

	dists := make([]float64, n)
	fits := make([]float64, n)
	for i, pt := range lm.Points {
		dists[i] = euclidean(pt.Genome, lm.BestGenome)
		fits[i] = pt.Fitness
	}

	md := mean(dists)
	mf := mean(fits)
	sd := stddev(dists, md)
	sf := stddev(fits, mf)
	return pearson(dists, fits, md, mf, sd, sf)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// geneRange returns the range width of gene k.
// Defaults to 1.0 if GeneRanges is nil or entry is zero-width.
func (fl *FitnessLandscape) geneRange(k int) float64 {
	if fl.GeneRanges == nil || k >= len(fl.GeneRanges) {
		return 1.0
	}
	r := fl.GeneRanges[k][1] - fl.GeneRanges[k][0]
	if r <= 0 {
		return 1.0
	}
	return r
}

// perturbStep returns the step size h for gene k.
func (fl *FitnessLandscape) perturbStep(k int) float64 {
	h := fl.Perturbation * fl.geneRange(k)
	if h == 0 {
		h = 1e-6
	}
	return h
}

// perturbEval evaluates fitness at Center with gene k shifted by delta.
func (fl *FitnessLandscape) perturbEval(k int, delta float64) float64 {
	g := make([]float64, len(fl.Center))
	copy(g, fl.Center)
	g[k] += delta
	return fl.Evaluator(g)
}

// perturbEval2 evaluates fitness at Center with genes i and j shifted.
func (fl *FitnessLandscape) perturbEval2(i int, di float64, j int, dj float64) float64 {
	g := make([]float64, len(fl.Center))
	copy(g, fl.Center)
	g[i] += di
	g[j] += dj
	return fl.Evaluator(g)
}

// rng returns the FitnessLandscape's RNG, initialising one if nil.
func (fl *FitnessLandscape) rng() *rand.Rand {
	if fl.RNG != nil {
		return fl.RNG
	}
	fl.RNG = rand.New(rand.NewSource(42))
	return fl.RNG
}

// vecNorm returns the L2 norm of a vector.
func vecNorm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}
