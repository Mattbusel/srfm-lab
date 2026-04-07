// Package signal_discovery implements systematic signal generation, mutation,
// and anti-correlation filtering for the SRFM idea engine.
package signal_discovery

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// Bar represents a single OHLCV bar for one symbol.
type Bar struct {
	Symbol    string
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	// Return is the forward return used for IC computation.
	// Callers must populate this field before passing bars to Validate.
	Return float64
}

// SignalCandidate describes a single candidate signal produced by the scanner.
type SignalCandidate struct {
	// Name is a human-readable identifier, e.g. "momentum_20d_scaled_1.05".
	Name string
	// Formula is a textual description of the signal computation.
	Formula string
	// Params holds the signal's tunable parameters.
	Params map[string]float64
	// TestIC is the Information Coefficient measured on the held-out window.
	TestIC float64
	// TestICIR is the IC Information Ratio on the held-out window.
	TestICIR float64
	// IsNovel is true when the candidate passed anti-correlation filtering.
	IsNovel bool
	// Series caches the signal values aligned to the test bars.
	// Keyed by bar index. Not exported to JSON.
	Series []float64
}

// clone returns a deep copy of a SignalCandidate.
func (sc SignalCandidate) clone() SignalCandidate {
	cp := sc
	cp.Params = make(map[string]float64, len(sc.Params))
	for k, v := range sc.Params {
		cp.Params[k] = v
	}
	if sc.Series != nil {
		cp.Series = make([]float64, len(sc.Series))
		copy(cp.Series, sc.Series)
	}
	return cp
}

// SignalMutation describes a single mutation applied to a signal candidate.
type SignalMutation struct {
	// Type indicates what kind of mutation this is: "scale", "lookback", "combine".
	Type string
	// ParamName is the parameter targeted by the mutation (for "scale" and "lookback").
	ParamName string
	// ScaleFactor is the multiplier applied for "scale" mutations.
	ScaleFactor float64
	// NewLookback replaces the lookback window value for "lookback" mutations.
	NewLookback int
	// CombineWith is the second signal used in a "combine" mutation.
	CombineWith *SignalCandidate
}

// MutateSignal applies a random mutation to base and returns the mutated copy.
// Three mutation types are possible:
//   - scale: multiplies a random parameter by a factor in [0.8, 1.2]
//   - lookback: snaps a "lookback"-prefixed parameter to a preset window
//   - combine: linearly blends base with another known-good signal
//
// The rng argument is used for all random decisions.
func MutateSignal(base SignalCandidate, rng *rand.Rand) SignalCandidate {
	mutant := base.clone()

	// Collect parameter keys so we can pick one deterministically.
	keys := make([]string, 0, len(base.Params))
	for k := range base.Params {
		keys = append(keys, k)
	}
	sort.Strings(keys) // stable order before random pick

	if len(keys) == 0 {
		return mutant
	}

	pick := rng.Intn(3) // 0=scale, 1=lookback, 2=combine
	paramIdx := rng.Intn(len(keys))
	chosenKey := keys[paramIdx]

	switch pick {
	case 0: // scale
		scale := 0.8 + rng.Float64()*0.4 // [0.8, 1.2]
		mutant.Params[chosenKey] = base.Params[chosenKey] * scale
		mutant.Name = fmt.Sprintf("%s_scaled_%.3f", base.Name, scale)
		mutant.Formula = fmt.Sprintf("scale(%s, %.4f)", base.Formula, scale)

	case 1: // lookback swap -- snap to standard windows
		windows := []int{5, 10, 14, 20, 30, 50, 60, 90, 120, 200, 252}
		newWindow := windows[rng.Intn(len(windows))]
		mutant.Params[chosenKey] = float64(newWindow)
		mutant.Name = fmt.Sprintf("%s_lb%d", base.Name, newWindow)
		mutant.Formula = fmt.Sprintf("lookback(%s, %d)", base.Formula, newWindow)

	case 2: // combine -- blend with a simple momentum proxy
		alpha := 0.3 + rng.Float64()*0.4 // [0.3, 0.7]
		mutant.Params["blend_alpha"] = alpha
		mutant.Name = fmt.Sprintf("%s_blend_%.2f", base.Name, alpha)
		mutant.Formula = fmt.Sprintf("blend(%s, momentum, %.4f)", base.Formula, alpha)
	}

	// Mark as not yet tested.
	mutant.TestIC = 0
	mutant.TestICIR = 0
	mutant.IsNovel = false
	mutant.Series = nil
	return mutant
}

// ScannerConfig holds tunable knobs for the SignalScanner.
type ScannerConfig struct {
	// MinICIR is the minimum ICIR a candidate must exceed to be retained.
	MinICIR float64
	// AntiCorrThreshold is the maximum absolute Pearson correlation allowed
	// between a candidate and existing signals. Candidates above this are dropped.
	AntiCorrThreshold float64
	// MutationsPerSeed is how many mutations to generate per seed signal.
	MutationsPerSeed int
	// Workers is the number of goroutines used for parallel candidate testing.
	Workers int
	// TestWindowFraction is the fraction of data reserved for held-out testing.
	TestWindowFraction float64
	// RandSeed seeds the random number generator for reproducibility.
	RandSeed int64
}

// DefaultScannerConfig returns a ScannerConfig with sensible defaults.
func DefaultScannerConfig() ScannerConfig {
	return ScannerConfig{
		MinICIR:            0.3,
		AntiCorrThreshold:  0.7,
		MutationsPerSeed:   20,
		Workers:            4,
		TestWindowFraction: 0.3,
		RandSeed:           42,
	}
}

// SignalScanner systematically generates and tests signal candidates by
// mutating a set of known-good seed signals.
type SignalScanner struct {
	cfg  ScannerConfig
	rng  *rand.Rand
	mu   sync.Mutex
	// seeds are the known-good base signals provided to the scanner.
	seeds []SignalCandidate
	// existing holds signal series already in the portfolio -- used for
	// anti-correlation checks.
	existing []SignalCandidate
}

// NewSignalScanner constructs a SignalScanner with the given config and seeds.
func NewSignalScanner(cfg ScannerConfig, seeds []SignalCandidate) *SignalScanner {
	return &SignalScanner{
		cfg:   cfg,
		rng:   rand.New(rand.NewSource(cfg.RandSeed)),
		seeds: seeds,
	}
}

// SetExisting replaces the set of existing portfolio signals used for
// anti-correlation filtering.
func (s *SignalScanner) SetExisting(existing []SignalCandidate) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.existing = existing
}

// ScanUniverse generates and evaluates signal candidates for all symbols over
// the given date range. It returns candidates that pass the ICIR filter.
//
// The data layer is simulated here -- in production, bars would be fetched
// from the research database and the signal series would be computed by the
// signal library. The function demonstrates the full pipeline structure.
func (s *SignalScanner) ScanUniverse(
	ctx context.Context,
	symbols []string,
	startDate, endDate string,
	barProvider func(symbol, start, end string) ([]Bar, error),
) ([]SignalCandidate, error) {
	if len(symbols) == 0 {
		return nil, fmt.Errorf("symbols list is empty")
	}
	if barProvider == nil {
		return nil, fmt.Errorf("barProvider must not be nil")
	}

	// Build candidate list from seed mutations.
	s.mu.Lock()
	candidates := s.generateCandidates()
	s.mu.Unlock()

	// Collect bars for all symbols.
	allBars := make(map[string][]Bar, len(symbols))
	for _, sym := range symbols {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		bars, err := barProvider(sym, startDate, endDate)
		if err != nil {
			return nil, fmt.Errorf("fetch bars for %s: %w", sym, err)
		}
		allBars[sym] = bars
	}

	// Evaluate each candidate in parallel.
	type evalResult struct {
		idx       int
		candidate SignalCandidate
	}
	jobs := make(chan int, len(candidates))
	results := make(chan evalResult, len(candidates))

	workers := s.cfg.Workers
	if workers <= 0 {
		workers = 1
	}

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				select {
				case <-ctx.Done():
					return
				default:
				}
				cand := candidates[idx]
				// Compute signal series across all symbols and test IC.
				ic, icir, series := s.evaluateCandidate(cand, allBars)
				cand.TestIC = ic
				cand.TestICIR = icir
				cand.Series = series
				results <- evalResult{idx: idx, candidate: cand}
			}
		}()
	}

	for i := range candidates {
		jobs <- i
	}
	close(jobs)

	go func() {
		wg.Wait()
		close(results)
	}()

	var passed []SignalCandidate
	for res := range results {
		if res.candidate.TestICIR >= s.cfg.MinICIR {
			passed = append(passed, res.candidate)
		}
	}

	// Anti-correlation filter against existing portfolio signals.
	s.mu.Lock()
	existing := s.existing
	s.mu.Unlock()
	passed = s.AntiCorrelationFilter(passed, s.cfg.AntiCorrThreshold)
	_ = existing // anti-corr against portfolio is handled inside the filter

	// Sort by ICIR descending.
	sort.Slice(passed, func(i, j int) bool {
		return passed[i].TestICIR > passed[j].TestICIR
	})

	return passed, nil
}

// generateCandidates creates the full set of mutated candidates from seeds.
// Caller must hold s.mu.
func (s *SignalScanner) generateCandidates() []SignalCandidate {
	var out []SignalCandidate
	for _, seed := range s.seeds {
		out = append(out, seed)
		for m := 0; m < s.cfg.MutationsPerSeed; m++ {
			mutant := MutateSignal(seed, s.rng)
			out = append(out, mutant)
		}
	}
	return out
}

// evaluateCandidate computes IC and ICIR for a candidate over the held-out
// window of all symbol bar slices. Returns (ic, icir, aggregated series).
func (s *SignalScanner) evaluateCandidate(
	cand SignalCandidate,
	allBars map[string][]Bar,
) (ic, icir float64, series []float64) {
	var allSignal, allReturn []float64

	for _, bars := range allBars {
		n := len(bars)
		if n < 20 {
			continue
		}
		// Use the last TestWindowFraction of bars as held-out window.
		splitAt := int(float64(n) * (1 - s.cfg.TestWindowFraction))
		testBars := bars[splitAt:]
		if len(testBars) < 5 {
			continue
		}

		sig := computeSignalSeries(cand, testBars)
		for i, v := range sig {
			allSignal = append(allSignal, v)
			allReturn = append(allReturn, testBars[i].Return)
		}
	}

	if len(allSignal) < 10 {
		return 0, 0, nil
	}

	series = allSignal
	ic = pearsonCorrelation(allSignal, allReturn)
	icir = rollingICIR(allSignal, allReturn, 20)
	return ic, icir, series
}

// computeSignalSeries generates a signal value for each bar using the
// candidate's formula type encoded in its Params.
// This is a simplified simulation -- real code would dispatch to the
// signal library.
func computeSignalSeries(cand SignalCandidate, bars []Bar) []float64 {
	n := len(bars)
	out := make([]float64, n)

	lookback := int(getParam(cand.Params, "lookback", 20))
	if lookback < 2 {
		lookback = 2
	}
	scale := getParam(cand.Params, "scale", 1.0)

	// Momentum signal: (close - close[lookback]) / close[lookback]
	for i := 0; i < n; i++ {
		if i < lookback {
			out[i] = 0
			continue
		}
		base := bars[i-lookback].Close
		if base == 0 {
			out[i] = 0
			continue
		}
		out[i] = scale * (bars[i].Close - base) / base
	}

	// Apply blend if specified.
	if alpha, ok := cand.Params["blend_alpha"]; ok {
		// Blend with a 5-day momentum proxy.
		for i := 5; i < n; i++ {
			base := bars[i-5].Close
			if base == 0 {
				continue
			}
			mom5 := (bars[i].Close - base) / base
			out[i] = alpha*out[i] + (1-alpha)*mom5
		}
	}

	return out
}

// AntiCorrelationFilter removes candidates whose absolute Pearson correlation
// with any existing signal series exceeds threshold. It also removes
// candidates that are highly correlated with each other (keeping the one with
// higher ICIR).
func (s *SignalScanner) AntiCorrelationFilter(
	candidates []SignalCandidate,
	threshold float64,
) []SignalCandidate {
	s.mu.Lock()
	existing := s.existing
	s.mu.Unlock()

	// Filter against existing portfolio signals first.
	var filtered []SignalCandidate
	for _, cand := range candidates {
		reject := false
		for _, ex := range existing {
			if len(cand.Series) == 0 || len(ex.Series) == 0 {
				continue
			}
			// Align lengths.
			minLen := len(cand.Series)
			if len(ex.Series) < minLen {
				minLen = len(ex.Series)
			}
			corr := pearsonCorrelation(cand.Series[:minLen], ex.Series[:minLen])
			if math.Abs(corr) > threshold {
				reject = true
				break
			}
		}
		if !reject {
			cand.IsNovel = true
			filtered = append(filtered, cand)
		}
	}

	// De-duplicate within the filtered set -- keep higher ICIR when two
	// candidates are highly correlated with each other.
	// Sort by ICIR descending so we keep the better one first.
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].TestICIR > filtered[j].TestICIR
	})

	var deduped []SignalCandidate
	for i, cand := range filtered {
		dominated := false
		for j := 0; j < i; j++ {
			if len(cand.Series) == 0 || len(deduped[j].Series) == 0 {
				continue
			}
			minLen := len(cand.Series)
			if len(deduped[j].Series) < minLen {
				minLen = len(deduped[j].Series)
			}
			if minLen == 0 {
				continue
			}
			corr := pearsonCorrelation(cand.Series[:minLen], deduped[j].Series[:minLen])
			if math.Abs(corr) > threshold {
				dominated = true
				break
			}
		}
		if !dominated {
			deduped = append(deduped, cand)
		}
	}

	return deduped
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

// pearsonCorrelation computes the Pearson r between x and y.
// Returns 0 if either slice has length < 2 or zero variance.
func pearsonCorrelation(x, y []float64) float64 {
	n := len(x)
	if n < 2 || len(y) < n {
		return 0
	}
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	fn := float64(n)
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	num := fn*sumXY - sumX*sumY
	den := math.Sqrt((fn*sumX2 - sumX*sumX) * (fn*sumY2 - sumY*sumY))
	if den == 0 {
		return 0
	}
	return num / den
}

// rollingICIR computes the IC Information Ratio using a rolling window of
// length window. Returns mean(IC) / stddev(IC) across rolling periods.
func rollingICIR(signal, ret []float64, window int) float64 {
	n := len(signal)
	if n < window*2 {
		// Fall back to single-window computation.
		ic := pearsonCorrelation(signal, ret)
		return ic / 0.05 // approximate ICIR assuming 5% IC stddev
	}

	var ics []float64
	for start := 0; start+window <= n; start += window {
		end := start + window
		ic := pearsonCorrelation(signal[start:end], ret[start:end])
		ics = append(ics, ic)
	}
	if len(ics) < 2 {
		return 0
	}
	mean := mean64(ics)
	std := stddev64(ics, mean)
	if std == 0 {
		return 0
	}
	return mean / std
}

// mean64 returns the arithmetic mean of xs.
func mean64(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range xs {
		sum += v
	}
	return sum / float64(len(xs))
}

// stddev64 returns the sample standard deviation of xs given its mean.
func stddev64(xs []float64, mean float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	sumSq := 0.0
	for _, v := range xs {
		d := v - mean
		sumSq += d * d
	}
	return math.Sqrt(sumSq / float64(len(xs)-1))
}

// getParam retrieves a float64 parameter by name, returning def if absent.
func getParam(params map[string]float64, name string, def float64) float64 {
	if v, ok := params[name]; ok {
		return v
	}
	return def
}
