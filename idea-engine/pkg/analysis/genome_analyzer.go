// Package analysis provides post-hoc and in-flight analysis tools for the
// IAE genome evolution system.
package analysis

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// ---------------------------------------------------------------------------
// GenomeAnalyzer
// ---------------------------------------------------------------------------

// GenomeAnalyzer computes statistics over a population of genomes.
// PopulationSize and GeneNames are informational metadata -- the analyzer
// works on any [][]float64 slice regardless of these fields.
type GenomeAnalyzer struct {
	PopulationSize int
	GeneNames      []string
}

// NewGenomeAnalyzer creates a GenomeAnalyzer with the given gene names.
func NewGenomeAnalyzer(geneNames []string) *GenomeAnalyzer {
	return &GenomeAnalyzer{
		GeneNames: geneNames,
	}
}

// ---------------------------------------------------------------------------
// PopulationStats
// ---------------------------------------------------------------------------

// PopulationStats contains diversity and convergence metrics for one
// snapshot of the population.
type PopulationStats struct {
	Generation     int
	BestFitness    float64
	AvgFitness     float64
	WorstFitness   float64
	Diversity      float64 // mean pairwise Euclidean distance
	Entropy        float64 // Shannon entropy of gene distribution
	ConvergenceIdx float64 // 1.0 = fully converged, 0.0 = fully diverse
}

// ComputeStats aggregates statistics over the given population.
// genomes[i] is the gene vector for individual i; fitnesses[i] is its score.
// Returns zero-value stats if the slices are empty or mismatched.
func (a *GenomeAnalyzer) ComputeStats(genomes [][]float64, fitnesses []float64) PopulationStats {
	n := len(genomes)
	if n == 0 || len(fitnesses) != n {
		return PopulationStats{}
	}

	// -- fitness aggregates
	best := fitnesses[0]
	worst := fitnesses[0]
	sum := 0.0
	for _, f := range fitnesses {
		if f > best {
			best = f
		}
		if f < worst {
			worst = f
		}
		sum += f
	}
	avg := sum / float64(n)

	// -- diversity
	diversity := a.MeanPairwiseDistance(genomes)

	// -- entropy: flatten all gene values across the population
	flat := make([]float64, 0, n*len(genomes[0]))
	for _, g := range genomes {
		flat = append(flat, g...)
	}
	entropy := a.GeneEntropy(flat, 20)

	// -- convergence index: if diversity == 0 we are fully converged
	// We normalise by the theoretical max diversity which we approximate
	// as the diversity of the initial call to this function clamped [0,1].
	// For a simple bounded estimate we use: conv = 1 / (1 + diversity)
	convergenceIdx := 1.0 / (1.0 + diversity)

	return PopulationStats{
		BestFitness:    best,
		AvgFitness:     avg,
		WorstFitness:   worst,
		Diversity:      diversity,
		Entropy:        entropy,
		ConvergenceIdx: convergenceIdx,
	}
}

// MeanPairwiseDistance computes the mean Euclidean distance between all
// distinct pairs of genomes. Complexity is O(n^2 * d) where d is genome
// length; acceptable for typical population sizes of 50-200.
func (a *GenomeAnalyzer) MeanPairwiseDistance(genomes [][]float64) float64 {
	n := len(genomes)
	if n < 2 {
		return 0.0
	}
	total := 0.0
	count := 0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			total += euclidean(genomes[i], genomes[j])
			count++
		}
	}
	if count == 0 {
		return 0.0
	}
	return total / float64(count)
}

// GeneEntropy computes the Shannon entropy of a slice of continuous values
// by first discretising into the given number of histogram bins.
// A larger bin count gives finer resolution but requires more data.
func (a *GenomeAnalyzer) GeneEntropy(values []float64, bins int) float64 {
	if len(values) == 0 || bins <= 0 {
		return 0.0
	}

	minV, maxV := values[0], values[0]
	for _, v := range values {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}

	rangeV := maxV - minV
	if rangeV == 0 {
		return 0.0 // all identical -- zero entropy
	}

	hist := make([]int, bins)
	for _, v := range values {
		b := int((v - minV) / rangeV * float64(bins))
		if b >= bins {
			b = bins - 1
		}
		hist[b]++
	}

	n := float64(len(values))
	entropy := 0.0
	for _, count := range hist {
		if count == 0 {
			continue
		}
		p := float64(count) / n
		entropy -= p * math.Log2(p)
	}
	return entropy
}

// FindNiches groups genomes into clusters where each member is within
// radius Euclidean distance of the cluster seed. Seeds are chosen
// greedily from the remaining unassigned individuals.
// Returns a slice of clusters; each cluster is a slice of genome indices.
func (a *GenomeAnalyzer) FindNiches(genomes [][]float64, radius float64) [][]int {
	n := len(genomes)
	if n == 0 {
		return nil
	}

	assigned := make([]bool, n)
	var clusters [][]int

	for i := 0; i < n; i++ {
		if assigned[i] {
			continue
		}
		cluster := []int{i}
		assigned[i] = true
		for j := i + 1; j < n; j++ {
			if assigned[j] {
				continue
			}
			if euclidean(genomes[i], genomes[j]) <= radius {
				cluster = append(cluster, j)
				assigned[j] = true
			}
		}
		clusters = append(clusters, cluster)
	}
	return clusters
}

// SelectDiverseSubset selects n genomes from the population using a greedy
// maximum-distance strategy. The first genome is chosen randomly (index 0),
// then each subsequent selection maximises the minimum distance to all
// already-selected genomes.
// Returns indices into the original genomes slice.
func (a *GenomeAnalyzer) SelectDiverseSubset(genomes [][]float64, n int) []int {
	total := len(genomes)
	if total == 0 || n <= 0 {
		return nil
	}
	if n >= total {
		idx := make([]int, total)
		for i := range idx {
			idx[i] = i
		}
		return idx
	}

	selected := make([]int, 0, n)
	selected = append(selected, 0) // seed with first genome

	// minDist[i] = min distance from genome i to any selected genome
	minDist := make([]float64, total)
	for i := 1; i < total; i++ {
		minDist[i] = euclidean(genomes[i], genomes[0])
	}

	for len(selected) < n {
		// pick the unselected genome with the largest min-distance
		best := -1
		bestDist := -1.0
		for i := 1; i < total; i++ {
			// skip already selected
			alreadySel := false
			for _, s := range selected {
				if s == i {
					alreadySel = true
					break
				}
			}
			if alreadySel {
				continue
			}
			if minDist[i] > bestDist {
				bestDist = minDist[i]
				best = i
			}
		}
		if best < 0 {
			break
		}
		selected = append(selected, best)
		// update minDist for remaining candidates
		for i := 0; i < total; i++ {
			d := euclidean(genomes[i], genomes[best])
			if d < minDist[i] {
				minDist[i] = d
			}
		}
	}
	return selected
}

// GeneCorrelationMatrix computes the Pearson correlation matrix between
// all gene pairs across the population.
// Returns a d x d matrix where d = len(genomes[0]).
// Returns nil if the population is too small or genomes have zero length.
func (a *GenomeAnalyzer) GeneCorrelationMatrix(genomes [][]float64) [][]float64 {
	n := len(genomes)
	if n < 2 || len(genomes[0]) == 0 {
		return nil
	}
	d := len(genomes[0])

	// Build a d x n matrix of gene values (transpose of genomes)
	genes := make([][]float64, d)
	for k := range genes {
		genes[k] = make([]float64, n)
	}
	for i, g := range genomes {
		for k := 0; k < d && k < len(g); k++ {
			genes[k][i] = g[k]
		}
	}

	// Compute means and std devs
	means := make([]float64, d)
	stds := make([]float64, d)
	for k := 0; k < d; k++ {
		means[k] = mean(genes[k])
		stds[k] = stddev(genes[k], means[k])
	}

	// Build correlation matrix
	corr := make([][]float64, d)
	for k := range corr {
		corr[k] = make([]float64, d)
	}
	for i := 0; i < d; i++ {
		corr[i][i] = 1.0
		for j := i + 1; j < d; j++ {
			c := pearson(genes[i], genes[j], means[i], means[j], stds[i], stds[j])
			corr[i][j] = c
			corr[j][i] = c
		}
	}
	return corr
}

// ---------------------------------------------------------------------------
// StagnationDetector
// ---------------------------------------------------------------------------

// StagnationDetector tracks the best fitness over generations and fires an
// alert if no improvement occurs within a configured window.
type StagnationDetector struct {
	mu           sync.Mutex
	Window       int            // number of generations without improvement to trigger
	Tolerance    float64        // minimum improvement to count as progress
	history      []float64      // best fitness per generation
	AlertHandler func(msg string) // called when stagnation is detected
}

// NewStagnationDetector creates a detector with the given window and
// tolerance. alertHandler is called when stagnation is first detected.
func NewStagnationDetector(window int, tolerance float64, alertHandler func(string)) *StagnationDetector {
	return &StagnationDetector{
		Window:       window,
		Tolerance:    tolerance,
		AlertHandler: alertHandler,
	}
}

// Record adds the best fitness for the current generation. If the best
// fitness has not improved by more than Tolerance in the last Window
// generations, the AlertHandler is invoked.
func (s *StagnationDetector) Record(bestFitness float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.history = append(s.history, bestFitness)
	n := len(s.history)
	if n < s.Window {
		return
	}

	// Check if the last Window entries show any improvement
	window := s.history[n-s.Window:]
	baseline := window[0]
	improved := false
	for _, v := range window[1:] {
		if v-baseline > s.Tolerance {
			improved = true
			break
		}
	}
	if !improved && s.AlertHandler != nil {
		msg := fmt.Sprintf(
			"StagnationDetector: no improvement > %.6f in %d generations (best=%.6f)",
			s.Tolerance, s.Window, bestFitness,
		)
		s.AlertHandler(msg)
	}
}

// IsStagnant returns true if the last Window generations show no improvement
// greater than Tolerance. Returns false if fewer than Window records exist.
func (s *StagnationDetector) IsStagnant() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := len(s.history)
	if n < s.Window {
		return false
	}
	window := s.history[n-s.Window:]
	baseline := window[0]
	for _, v := range window[1:] {
		if v-baseline > s.Tolerance {
			return false
		}
	}
	return true
}

// Reset clears the history, useful when parameters change significantly.
func (s *StagnationDetector) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.history = s.history[:0]
}

// GenerationsRecorded returns the number of generations tracked so far.
func (s *StagnationDetector) GenerationsRecorded() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.history)
}

// LastImprovement returns the generation index and value of the best
// fitness recorded, plus how many generations ago it occurred.
// Returns (-1, 0.0, 0) if no history exists.
func (s *StagnationDetector) LastImprovement() (genIdx int, bestVal float64, generationsAgo int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := len(s.history)
	if n == 0 {
		return -1, 0.0, 0
	}

	best := s.history[0]
	bestIdx := 0
	for i, v := range s.history {
		if v > best {
			best = v
			bestIdx = i
		}
	}
	return bestIdx, best, n - 1 - bestIdx
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// euclidean computes the L2 distance between two float64 slices.
// If the slices differ in length the shorter length is used.
func euclidean(a, b []float64) float64 {
	length := len(a)
	if len(b) < length {
		length = len(b)
	}
	sum := 0.0
	for i := 0; i < length; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// mean returns the arithmetic mean of xs.
func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0.0
	}
	s := 0.0
	for _, v := range xs {
		s += v
	}
	return s / float64(len(xs))
}

// stddev returns the population standard deviation given a precomputed mean.
func stddev(xs []float64, m float64) float64 {
	if len(xs) == 0 {
		return 0.0
	}
	s := 0.0
	for _, v := range xs {
		d := v - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(xs)))
}

// pearson computes the Pearson correlation coefficient given precomputed
// means and standard deviations.
func pearson(xs, ys []float64, mx, my, sx, sy float64) float64 {
	if sx == 0 || sy == 0 {
		return 0.0
	}
	n := len(xs)
	if len(ys) < n {
		n = len(ys)
	}
	cov := 0.0
	for i := 0; i < n; i++ {
		cov += (xs[i] - mx) * (ys[i] - my)
	}
	cov /= float64(n)
	return cov / (sx * sy)
}

// sortedCopy returns a sorted copy of xs (ascending).
func sortedCopy(xs []float64) []float64 {
	c := make([]float64, len(xs))
	copy(c, xs)
	sort.Float64s(c)
	return c
}
