// Package tests contains integration-level tests for the evolution package
// crossover operators.
package tests

import (
	"math"
	"math/rand"
	"testing"

	"srfm-lab/idea-engine/pkg/evolution"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// makeGenome returns a Genome with n genes linearly spaced in [lo, hi].
func makeGenome(n int, lo, hi float64) evolution.Genome {
	g := make(evolution.Genome, n)
	if n == 1 {
		g[0] = (lo + hi) / 2
		return g
	}
	for i := range g {
		g[i] = lo + float64(i)*(hi-lo)/float64(n-1)
	}
	return g
}

// allWithin returns true when every gene of g is in [lo, hi].
func allWithin(g evolution.Genome, lo, hi float64) bool {
	for _, v := range g {
		if v < lo-1e-9 || v > hi+1e-9 {
			return false
		}
	}
	return true
}

// anyDifferent returns true when at least one gene differs between a and b.
func anyDifferent(a, b evolution.Genome) bool {
	for i := range a {
		if a[i] != b[i] {
			return true
		}
	}
	return false
}

// geneticDistance returns the Euclidean distance between two genomes.
func geneticDistance(a, b evolution.Genome) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// ---------------------------------------------------------------------------
// UniformCrossover tests
// ---------------------------------------------------------------------------

func TestUniformCrossover_LengthPreserved(t *testing.T) {
	cases := []struct {
		n    int
		rate float64
	}{
		{1, 0.5},
		{10, 0.5},
		{50, 0.5},
		{100, 0.0},
		{100, 1.0},
	}
	for _, tc := range cases {
		p1 := makeGenome(tc.n, 0, 1)
		p2 := makeGenome(tc.n, 1, 2)
		c1, c2 := evolution.UniformCrossover(p1, p2, tc.rate)
		if len(c1) != tc.n {
			t.Errorf("n=%d rate=%.1f: c1 length %d, want %d", tc.n, tc.rate, len(c1), tc.n)
		}
		if len(c2) != tc.n {
			t.Errorf("n=%d rate=%.1f: c2 length %d, want %d", tc.n, tc.rate, len(c2), tc.n)
		}
	}
}

func TestUniformCrossover_GenesFromParents(t *testing.T) {
	// Each gene in offspring must come from one of the two parents.
	p1 := makeGenome(20, 0, 1)
	p2 := makeGenome(20, 10, 20)
	c1, c2 := evolution.UniformCrossover(p1, p2, 0.5)
	for i := range c1 {
		if c1[i] != p1[i] && c1[i] != p2[i] {
			t.Errorf("c1[%d]=%.6f not from parent (p1=%.6f, p2=%.6f)", i, c1[i], p1[i], p2[i])
		}
		if c2[i] != p1[i] && c2[i] != p2[i] {
			t.Errorf("c2[%d]=%.6f not from parent (p1=%.6f, p2=%.6f)", i, c2[i], p1[i], p2[i])
		}
	}
}

func TestUniformCrossover_RateZero_ClonesParents(t *testing.T) {
	p1 := makeGenome(30, 0, 1)
	p2 := makeGenome(30, 5, 6)
	c1, c2 := evolution.UniformCrossover(p1, p2, 0.0)
	for i := range p1 {
		if c1[i] != p1[i] {
			t.Errorf("rate=0: c1[%d]=%.6f != p1[%d]=%.6f", i, c1[i], i, p1[i])
		}
		if c2[i] != p2[i] {
			t.Errorf("rate=0: c2[%d]=%.6f != p2[%d]=%.6f", i, c2[i], i, p2[i])
		}
	}
}

func TestUniformCrossover_RateOne_FullSwap(t *testing.T) {
	p1 := makeGenome(30, 0, 1)
	p2 := makeGenome(30, 5, 6)
	c1, c2 := evolution.UniformCrossover(p1, p2, 1.0)
	for i := range p1 {
		if c1[i] != p2[i] {
			t.Errorf("rate=1: c1[%d]=%.6f != p2[%d]=%.6f", i, c1[i], i, p2[i])
		}
		if c2[i] != p1[i] {
			t.Errorf("rate=1: c2[%d]=%.6f != p1[%d]=%.6f", i, c2[i], i, p1[i])
		}
	}
}

func TestUniformCrossover_DiversityMaintained(t *testing.T) {
	// With rate=0.5 and 50 genes, at least one swap should occur in expectation.
	// We run 100 trials -- at least one must produce offspring != parents.
	p1 := makeGenome(50, 0, 1)
	p2 := makeGenome(50, 10, 20)
	anyDiff := false
	for trial := 0; trial < 100; trial++ {
		c1, _ := evolution.UniformCrossover(p1, p2, 0.5)
		if anyDifferent(c1, p1) {
			anyDiff = true
			break
		}
	}
	if !anyDiff {
		t.Error("expected at least one offspring different from parent over 100 trials")
	}
}

// ---------------------------------------------------------------------------
// SBX tests
// ---------------------------------------------------------------------------

func TestSBX_LengthPreserved(t *testing.T) {
	p1 := makeGenome(40, 0, 1)
	p2 := makeGenome(40, 0, 1)
	c1, c2 := evolution.SimulatedBinaryCrossover(p1, p2, 20)
	if len(c1) != 40 || len(c2) != 40 {
		t.Errorf("SBX length mismatch: got %d, %d want 40, 40", len(c1), len(c2))
	}
}

func TestSBX_OffspringBoundedByParents(t *testing.T) {
	// SBX offspring should be close to parents; use a large eta for tight bounds.
	// With eta=100, beta is very close to 1 so offspring nearly equal parents.
	rng := rand.New(rand.NewSource(0))
	for trial := 0; trial < 20; trial++ {
		n := 20
		p1 := make(evolution.Genome, n)
		p2 := make(evolution.Genome, n)
		for i := range p1 {
			p1[i] = rng.Float64() * 10
			p2[i] = rng.Float64() * 10
		}
		c1, c2 := evolution.SimulatedBinaryCrossover(p1, p2, 20)

		// Each offspring gene should lie within 3x the parent spread
		// (SBX can occasionally produce values outside [min,max]).
		for i := range c1 {
			lo := math.Min(p1[i], p2[i])
			hi := math.Max(p1[i], p2[i])
			span := hi - lo + 1e-9
			if c1[i] < lo-3*span || c1[i] > hi+3*span {
				t.Errorf("trial=%d gene=%d: c1=%.4f outside extended bound [%.4f, %.4f]",
					trial, i, c1[i], lo-3*span, hi+3*span)
			}
			if c2[i] < lo-3*span || c2[i] > hi+3*span {
				t.Errorf("trial=%d gene=%d: c2=%.4f outside extended bound [%.4f, %.4f]",
					trial, i, c2[i], lo-3*span, hi+3*span)
			}
		}
	}
}

func TestSBX_HighEtaProducesOffspringCloseToParents(t *testing.T) {
	// With very high eta the spread factor beta -> 1 and offspring -> parents.
	p1 := makeGenome(20, 0, 1)
	p2 := makeGenome(20, 1, 2)
	totalDist := 0.0
	trials := 50
	for i := 0; i < trials; i++ {
		c1, _ := evolution.SimulatedBinaryCrossover(p1, p2, 200)
		totalDist += geneticDistance(c1, p1)
	}
	avgDist := totalDist / float64(trials)
	// Expect very small deviation with eta=200.
	if avgDist > 1.0 {
		t.Errorf("expected average distance < 1.0 with eta=200, got %.4f", avgDist)
	}
}

func TestSBX_DiversityWithLowEta(t *testing.T) {
	// With eta=0.5 offspring should show spread.
	p1 := makeGenome(20, 0, 1)
	p2 := makeGenome(20, 1, 2)
	totalDist := 0.0
	trials := 50
	for i := 0; i < trials; i++ {
		c1, _ := evolution.SimulatedBinaryCrossover(p1, p2, 0.5)
		totalDist += geneticDistance(c1, p1)
	}
	avgDist := totalDist / float64(trials)
	if avgDist < 0.01 {
		t.Errorf("expected non-trivial distance with eta=0.5, got %.6f", avgDist)
	}
}

// ---------------------------------------------------------------------------
// ArithmeticCrossover tests
// ---------------------------------------------------------------------------

func TestArithmeticCrossover_LengthPreserved(t *testing.T) {
	p1 := makeGenome(25, 0, 1)
	p2 := makeGenome(25, 2, 3)
	c1, c2 := evolution.ArithmeticCrossover(p1, p2, 0.5)
	if len(c1) != 25 || len(c2) != 25 {
		t.Errorf("length mismatch: got %d, %d want 25, 25", len(c1), len(c2))
	}
}

func TestArithmeticCrossover_BlendedValues(t *testing.T) {
	// c1[i] = alpha*p1[i] + (1-alpha)*p2[i]
	// c2[i] = alpha*p2[i] + (1-alpha)*p1[i]
	p1 := evolution.Genome{1.0, 2.0, 3.0}
	p2 := evolution.Genome{3.0, 4.0, 5.0}
	alpha := 0.3
	c1, c2 := evolution.ArithmeticCrossover(p1, p2, alpha)
	for i := range p1 {
		wantC1 := alpha*p1[i] + (1-alpha)*p2[i]
		wantC2 := alpha*p2[i] + (1-alpha)*p1[i]
		if math.Abs(c1[i]-wantC1) > 1e-12 {
			t.Errorf("c1[%d]: got %.12f want %.12f", i, c1[i], wantC1)
		}
		if math.Abs(c2[i]-wantC2) > 1e-12 {
			t.Errorf("c2[%d]: got %.12f want %.12f", i, c2[i], wantC2)
		}
	}
}

func TestArithmeticCrossover_AlphaZero(t *testing.T) {
	// alpha=0 -> c1 is p2, c2 is p1.
	p1 := makeGenome(10, 0, 1)
	p2 := makeGenome(10, 5, 6)
	c1, c2 := evolution.ArithmeticCrossover(p1, p2, 0.0)
	for i := range p1 {
		if math.Abs(c1[i]-p2[i]) > 1e-12 {
			t.Errorf("alpha=0 c1[%d]=%.6f want p2[%d]=%.6f", i, c1[i], i, p2[i])
		}
		if math.Abs(c2[i]-p1[i]) > 1e-12 {
			t.Errorf("alpha=0 c2[%d]=%.6f want p1[%d]=%.6f", i, c2[i], i, p1[i])
		}
	}
}

func TestArithmeticCrossover_AlphaOne(t *testing.T) {
	// alpha=1 -> c1 is p1, c2 is p2.
	p1 := makeGenome(10, 0, 1)
	p2 := makeGenome(10, 5, 6)
	c1, c2 := evolution.ArithmeticCrossover(p1, p2, 1.0)
	for i := range p1 {
		if math.Abs(c1[i]-p1[i]) > 1e-12 {
			t.Errorf("alpha=1 c1[%d]=%.6f want p1[%d]=%.6f", i, c1[i], i, p1[i])
		}
		if math.Abs(c2[i]-p2[i]) > 1e-12 {
			t.Errorf("alpha=1 c2[%d]=%.6f want p2[%d]=%.6f", i, c2[i], i, p2[i])
		}
	}
}

func TestArithmeticCrossover_WithinBounds(t *testing.T) {
	// Offspring should always lie within [min(p1,p2), max(p1,p2)] for each gene.
	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 50; trial++ {
		n := 30
		p1 := make(evolution.Genome, n)
		p2 := make(evolution.Genome, n)
		for i := range p1 {
			p1[i] = rng.Float64() * 100
			p2[i] = rng.Float64() * 100
		}
		alpha := rng.Float64()
		c1, c2 := evolution.ArithmeticCrossover(p1, p2, alpha)
		for i := range c1 {
			lo := math.Min(p1[i], p2[i])
			hi := math.Max(p1[i], p2[i])
			if c1[i] < lo-1e-9 || c1[i] > hi+1e-9 {
				t.Errorf("trial=%d gene=%d: c1=%.6f out of [%.6f, %.6f]",
					trial, i, c1[i], lo, hi)
			}
			if c2[i] < lo-1e-9 || c2[i] > hi+1e-9 {
				t.Errorf("trial=%d gene=%d: c2=%.6f out of [%.6f, %.6f]",
					trial, i, c2[i], lo, hi)
			}
		}
	}
}

func TestArithmeticCrossover_OffspringDiffersFromParents(t *testing.T) {
	// With alpha=0.5 and distinct parents, offspring must differ from both parents.
	p1 := makeGenome(10, 0, 1)
	p2 := makeGenome(10, 10, 20)
	c1, c2 := evolution.ArithmeticCrossover(p1, p2, 0.5)
	if !anyDifferent(c1, p1) {
		t.Error("c1 should differ from p1 with alpha=0.5")
	}
	if !anyDifferent(c2, p2) {
		t.Error("c2 should differ from p2 with alpha=0.5")
	}
}
