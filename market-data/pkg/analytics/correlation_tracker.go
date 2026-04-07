// Package analytics provides real-time statistical computations on bar data streams.
package analytics

import (
	"math"
	"sync"
	"time"
)

// CorrelationPair holds the Pearson correlation between two symbols
// over the current rolling window.
type CorrelationPair struct {
	Symbol1     string
	Symbol2     string
	Correlation float64
	WindowBars  int
}

// priceObs is one timestamped price observation for a symbol.
type priceObs struct {
	price float64
	ts    time.Time
}

// ewmState holds EWM accumulators for a single symbol.
// mean and variance are updated incrementally on each new observation.
type ewmState struct {
	mean     float64 // EWM mean of log-returns
	variance float64 // EWM variance of log-returns
	lastLog  float64 // last log-price, for return computation
	count    int     // observations received so far
}

// crossState holds the EWM cross-moment accumulator for a pair (i, j).
// Used to compute rolling covariance.
type crossState struct {
	cov float64 // EWM covariance estimate E[(ri - mean_i)(rj - mean_j)]
}

// CorrelationTracker maintains a rolling correlation matrix for all symbols
// in the universe. Correlations are computed over log-returns using an
// exponential weighted moving average (EWM) with alpha = 2/(window+1).
//
// All public methods are safe for concurrent use.
type CorrelationTracker struct {
	Symbols    []string    // ordered list of tracked symbols
	Window     int         // rolling window in bars (sets EWM alpha)
	Matrix     [][]float64 // current correlation matrix (n x n)
	lastUpdate time.Time

	mu      sync.RWMutex
	idx     map[string]int   // symbol -> row/col index
	alpha   float64          // EWM decay coefficient: 2/(window+1)
	states  []ewmState       // per-symbol EWM accumulators
	crosses [][]crossState   // upper-triangle covariance accumulators
	prices  [][]priceObs     // last Window observations per symbol (ring buffer)
	volWeights []float64     // normalized weights for diversification ratio
}

// NewCorrelationTracker creates a tracker for the given symbols and window.
// Window must be >= 2; if it is less, it is clamped to 2.
func NewCorrelationTracker(symbols []string, window int) *CorrelationTracker {
	if window < 2 {
		window = 2
	}
	n := len(symbols)
	idx := make(map[string]int, n)
	for i, s := range symbols {
		idx[s] = i
	}

	// Allocate matrix (n x n).
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		matrix[i][i] = 1.0 // diagonal is always 1
	}

	// Allocate upper-triangle cross accumulators.
	crosses := make([][]crossState, n)
	for i := range crosses {
		crosses[i] = make([]crossState, n)
	}

	// Allocate ring-buffer price history.
	prices := make([][]priceObs, n)
	for i := range prices {
		prices[i] = make([]priceObs, 0, window)
	}

	weights := make([]float64, n)
	for i := range weights {
		weights[i] = 1.0 / float64(n) // equal weight default
	}

	return &CorrelationTracker{
		Symbols:    append([]string(nil), symbols...),
		Window:     window,
		Matrix:     matrix,
		idx:        idx,
		alpha:      2.0 / float64(window+1),
		states:     make([]ewmState, n),
		crosses:    crosses,
		prices:     prices,
		volWeights: weights,
	}
}

// Update adds a new price observation for symbol at time ts.
// If symbol is not in the tracker's universe, the call is a no-op.
// The function updates EWM means, variances, and cross-moments, then
// refreshes the correlation matrix row/column for symbol.
func (ct *CorrelationTracker) Update(symbol string, price float64, ts time.Time) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	i, ok := ct.idx[symbol]
	if !ok {
		return
	}

	logP := math.Log(price)
	st := &ct.states[i]

	// First observation: initialise and return -- no return yet.
	if st.count == 0 {
		st.lastLog = logP
		st.count = 1
		// Append first observation to price history.
		ct.appendObs(i, price, ts)
		return
	}

	ret := logP - st.lastLog
	st.lastLog = logP
	st.count++

	// EWM mean update.
	prevMean := st.mean
	st.mean = ct.alpha*ret + (1-ct.alpha)*st.mean

	// EWM variance update (incremental formula).
	delta := ret - prevMean
	st.variance = (1 - ct.alpha) * (st.variance + ct.alpha*delta*delta)

	// Update cross-moments with all other symbols that have at least one obs.
	n := len(ct.Symbols)
	for j := 0; j < n; j++ {
		if j == i {
			continue
		}
		other := &ct.states[j]
		if other.count < 2 {
			continue
		}
		// We need the most recent return for j -- stored as last log diff.
		// We approximate using the other symbol's last observed log-return.
		// This is valid for synchronous tick streams; for async streams it is
		// an approximation that still produces consistent signs.
		retJ := ct.lastReturn(j)
		dj := retJ - other.mean
		ct.crosses[i][j].cov = (1-ct.alpha)*ct.crosses[i][j].cov + ct.alpha*delta*dj
		ct.crosses[j][i].cov = ct.crosses[i][j].cov
	}

	// Refresh matrix row i.
	ct.refreshRow(i)

	// Append to price history ring.
	ct.appendObs(i, price, ts)
	ct.lastUpdate = ts
}

// lastReturn returns the approximate last log-return for symbol index j.
// Uses the last two entries in the price history ring buffer.
func (ct *CorrelationTracker) lastReturn(j int) float64 {
	hist := ct.prices[j]
	n := len(hist)
	if n < 2 {
		return 0
	}
	return math.Log(hist[n-1].price) - math.Log(hist[n-2].price)
}

// appendObs appends a price observation to the ring buffer for symbol i.
// When the buffer exceeds Window entries, the oldest is dropped.
func (ct *CorrelationTracker) appendObs(i int, price float64, ts time.Time) {
	ct.prices[i] = append(ct.prices[i], priceObs{price: price, ts: ts})
	if len(ct.prices[i]) > ct.Window {
		ct.prices[i] = ct.prices[i][1:]
	}
}

// refreshRow recomputes the correlation values in row i of the matrix.
func (ct *CorrelationTracker) refreshRow(i int) {
	n := len(ct.Symbols)
	varI := ct.states[i].variance
	if varI <= 0 {
		return
	}
	for j := 0; j < n; j++ {
		if i == j {
			ct.Matrix[i][j] = 1.0
			continue
		}
		varJ := ct.states[j].variance
		if varJ <= 0 {
			ct.Matrix[i][j] = 0
			continue
		}
		cov := ct.crosses[i][j].cov
		corr := cov / math.Sqrt(varI*varJ)
		// Clamp to [-1, 1] due to floating-point drift.
		if corr > 1.0 {
			corr = 1.0
		} else if corr < -1.0 {
			corr = -1.0
		}
		ct.Matrix[i][j] = corr
	}
}

// GetCorrelation returns the Pearson correlation between s1 and s2 over
// the current EWM window. Returns 0 if either symbol is unknown or
// insufficient data has been accumulated.
func (ct *CorrelationTracker) GetCorrelation(s1, s2 string) float64 {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	i, ok1 := ct.idx[s1]
	j, ok2 := ct.idx[s2]
	if !ok1 || !ok2 {
		return 0
	}
	return ct.Matrix[i][j]
}

// GetMatrix returns a deep copy of the current n x n correlation matrix.
func (ct *CorrelationTracker) GetMatrix() [][]float64 {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	n := len(ct.Symbols)
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, n)
		copy(out[i], ct.Matrix[i])
	}
	return out
}

// GetHighlyCorrelated returns all symbol pairs whose absolute correlation
// exceeds threshold (0 < threshold <= 1).
func (ct *CorrelationTracker) GetHighlyCorrelated(threshold float64) []CorrelationPair {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	var pairs []CorrelationPair
	n := len(ct.Symbols)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			c := ct.Matrix[i][j]
			if math.Abs(c) >= threshold {
				pairs = append(pairs, CorrelationPair{
					Symbol1:     ct.Symbols[i],
					Symbol2:     ct.Symbols[j],
					Correlation: c,
					WindowBars:  ct.Window,
				})
			}
		}
	}
	return pairs
}

// SetVolWeights sets portfolio weights for the diversification ratio
// computation. weights must have the same length as Symbols. If the
// lengths differ or weights sum to zero, the call is a no-op.
func (ct *CorrelationTracker) SetVolWeights(weights []float64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if len(weights) != len(ct.Symbols) {
		return
	}
	sum := 0.0
	for _, w := range weights {
		sum += w
	}
	if sum == 0 {
		return
	}
	ct.volWeights = make([]float64, len(weights))
	for i, w := range weights {
		ct.volWeights[i] = w / sum
	}
}

// GetDiversificationRatio returns the ratio of the weighted-average
// individual volatility to the portfolio volatility implied by the
// current correlation matrix.
//
//   DR = (sum_i w_i * sigma_i) / sqrt(w' * Sigma * w)
//
// Returns 1.0 if the portfolio variance is zero or insufficient data.
func (ct *CorrelationTracker) GetDiversificationRatio() float64 {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	n := len(ct.Symbols)
	if n == 0 {
		return 1.0
	}

	// Collect individual vols (EWM std dev of returns).
	vols := make([]float64, n)
	for i, st := range ct.states {
		vols[i] = math.Sqrt(st.variance)
	}

	// Weighted average vol numerator.
	numerator := 0.0
	for i := 0; i < n; i++ {
		numerator += ct.volWeights[i] * vols[i]
	}

	// Portfolio variance: w' Sigma w where Sigma_ij = rho_ij * sigma_i * sigma_j.
	portfolioVar := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			portfolioVar += ct.volWeights[i] * ct.volWeights[j] *
				ct.Matrix[i][j] * vols[i] * vols[j]
		}
	}

	if portfolioVar <= 0 {
		return 1.0
	}
	dr := numerator / math.Sqrt(portfolioVar)
	return dr
}

// LastUpdate returns the timestamp of the most recent observation processed.
func (ct *CorrelationTracker) LastUpdate() time.Time {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	return ct.lastUpdate
}

// Reset clears all accumulators while keeping the symbol list and window.
func (ct *CorrelationTracker) Reset() {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	n := len(ct.Symbols)
	ct.states = make([]ewmState, n)
	for i := range ct.crosses {
		ct.crosses[i] = make([]crossState, n)
	}
	for i := range ct.prices {
		ct.prices[i] = ct.prices[i][:0]
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				ct.Matrix[i][j] = 1.0
			} else {
				ct.Matrix[i][j] = 0
			}
		}
	}
	ct.lastUpdate = time.Time{}
}

// SymbolCount returns the number of tracked symbols.
func (ct *CorrelationTracker) SymbolCount() int {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	return len(ct.Symbols)
}

// HasSymbol reports whether symbol is tracked by this instance.
func (ct *CorrelationTracker) HasSymbol(symbol string) bool {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	_, ok := ct.idx[symbol]
	return ok
}
