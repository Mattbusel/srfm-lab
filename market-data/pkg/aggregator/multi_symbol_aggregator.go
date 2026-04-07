// Package aggregator provides multi-symbol bar aggregation and cross-asset analytics.
package aggregator

import (
	"math"
	"sync"
	"time"
)

// OHLCV holds a single bar's price and volume data.
type OHLCV struct {
	Symbol    string
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Timestamp time.Time
}

// BarSnapshot is a point-in-time snapshot of a single symbol's bar, statistics, and signal.
type BarSnapshot struct {
	Symbol  string
	OHLCV   [5]float64 // [open, high, low, close, volume]
	BHMass  float64    // Black-Hole mass proxy: log(volume) * |close - open|
	Signal  float64    // normalized signal: (close - ema20) / atr
	Regime  string     // "trending_up", "trending_down", "ranging", "unknown"
}

// AggregatedSnapshot is a cross-symbol view produced each bar.
type AggregatedSnapshot struct {
	Symbols           map[string]BarSnapshot
	Timestamp         time.Time
	CorrelationMatrix [][]float64
}

// symbolRing holds a fixed-size ring buffer of close prices for correlation.
type symbolRing struct {
	closes []float64
	head   int
	count  int
	cap    int
}

func newSymbolRing(capacity int) *symbolRing {
	return &symbolRing{
		closes: make([]float64, capacity),
		cap:    capacity,
	}
}

func (r *symbolRing) push(v float64) {
	r.closes[r.head] = v
	r.head = (r.head + 1) % r.cap
	if r.count < r.cap {
		r.count++
	}
}

// values returns the values in chronological order (oldest first).
func (r *symbolRing) values() []float64 {
	if r.count == 0 {
		return nil
	}
	out := make([]float64, r.count)
	start := (r.head - r.count + r.cap) % r.cap
	for i := 0; i < r.count; i++ {
		out[i] = r.closes[(start+i)%r.cap]
	}
	return out
}

// returns returne pct returns from the closes slice.
func returns(closes []float64) []float64 {
	if len(closes) < 2 {
		return nil
	}
	ret := make([]float64, len(closes)-1)
	for i := 1; i < len(closes); i++ {
		if closes[i-1] != 0 {
			ret[i-1] = (closes[i] - closes[i-1]) / closes[i-1]
		}
	}
	return ret
}

// pearson computes Pearson correlation between two equal-length slices.
func pearson(a, b []float64) float64 {
	n := len(a)
	if n != len(b) || n < 2 {
		return 0
	}
	var sumA, sumB, sumAB, sumA2, sumB2 float64
	for i := 0; i < n; i++ {
		sumA += a[i]
		sumB += b[i]
		sumAB += a[i] * b[i]
		sumA2 += a[i] * a[i]
		sumB2 += b[i] * b[i]
	}
	fn := float64(n)
	num := fn*sumAB - sumA*sumB
	den := math.Sqrt((fn*sumA2 - sumA*sumA) * (fn*sumB2 - sumB*sumB))
	if den == 0 {
		return 0
	}
	return num / den
}

// MultiSymbolAggregator collects BarSnapshots for all subscribed symbols,
// computes rolling correlations, and broadcasts AggregatedSnapshots.
type MultiSymbolAggregator struct {
	mu        sync.RWMutex
	symbols   []string
	snapshots map[string]BarSnapshot
	rings     map[string]*symbolRing

	// subscribers receive a copy of each AggregatedSnapshot on each Update call.
	subsMu      sync.Mutex
	subscribers []chan AggregatedSnapshot

	lastSnapshot AggregatedSnapshot
	corrWindow   int
}

// NewMultiSymbolAggregator creates an aggregator for the given symbol list.
// corrWindow controls how many bars of history are used for correlation.
func NewMultiSymbolAggregator(symbols []string) *MultiSymbolAggregator {
	a := &MultiSymbolAggregator{
		symbols:    make([]string, len(symbols)),
		snapshots:  make(map[string]BarSnapshot, len(symbols)),
		rings:      make(map[string]*symbolRing, len(symbols)),
		corrWindow: 60,
	}
	copy(a.symbols, symbols)
	for _, sym := range symbols {
		a.rings[sym] = newSymbolRing(a.corrWindow + 1)
	}
	return a
}

// Subscribe registers a channel that receives AggregatedSnapshots.
// The caller is responsible for draining the channel.
func (a *MultiSymbolAggregator) Subscribe() <-chan AggregatedSnapshot {
	ch := make(chan AggregatedSnapshot, 64)
	a.subsMu.Lock()
	a.subscribers = append(a.subscribers, ch)
	a.subsMu.Unlock()
	return ch
}

// Unsubscribe removes and closes a previously subscribed channel.
func (a *MultiSymbolAggregator) Unsubscribe(ch <-chan AggregatedSnapshot) {
	a.subsMu.Lock()
	defer a.subsMu.Unlock()
	for i, sub := range a.subscribers {
		if sub == ch {
			a.subscribers = append(a.subscribers[:i], a.subscribers[i+1:]...)
			close(sub)
			return
		}
	}
}

// SetCorrelationWindow sets the rolling window length for correlation computation.
func (a *MultiSymbolAggregator) SetCorrelationWindow(window int) {
	if window < 2 {
		window = 2
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.corrWindow = window
	for sym := range a.rings {
		a.rings[sym] = newSymbolRing(window + 1)
	}
}

// Update incorporates a new BarSnapshot for the given symbol.
// After updating, it recomputes the AggregatedSnapshot and broadcasts it.
func (a *MultiSymbolAggregator) Update(symbol string, bar BarSnapshot) {
	a.mu.Lock()
	bar.Symbol = symbol
	a.snapshots[symbol] = bar

	// push close price into ring buffer for correlation
	if r, ok := a.rings[symbol]; ok {
		r.push(bar.OHLCV[3]) // index 3 = close
	} else {
		r := newSymbolRing(a.corrWindow + 1)
		r.push(bar.OHLCV[3])
		a.rings[symbol] = r
	}

	snap := a.buildSnapshotLocked()
	a.lastSnapshot = snap
	a.mu.Unlock()

	a.broadcast(snap)
}

// Snapshot returns the most recent AggregatedSnapshot. Thread-safe.
func (a *MultiSymbolAggregator) Snapshot() AggregatedSnapshot {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.copySnapshot(a.lastSnapshot)
}

// ComputeRollingCorrelation recomputes the N x N Pearson correlation matrix
// using the last `window` bars of close-price returns.
func (a *MultiSymbolAggregator) ComputeRollingCorrelation(window int) [][]float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()

	n := len(a.symbols)
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		matrix[i][i] = 1.0
	}

	// Gather return series for each symbol capped at window.
	retSeries := make([][]float64, n)
	for i, sym := range a.symbols {
		r, ok := a.rings[sym]
		if !ok {
			continue
		}
		closes := r.values()
		if len(closes) > window+1 {
			closes = closes[len(closes)-window-1:]
		}
		retSeries[i] = returns(closes)
	}

	// Compute upper triangle and mirror.
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			// align lengths
			ai, bj := retSeries[i], retSeries[j]
			minLen := len(ai)
			if len(bj) < minLen {
				minLen = len(bj)
			}
			if minLen < 2 {
				continue
			}
			r := pearson(ai[len(ai)-minLen:], bj[len(bj)-minLen:])
			matrix[i][j] = r
			matrix[j][i] = r
		}
	}
	return matrix
}

// buildSnapshotLocked must be called with a.mu write-locked.
func (a *MultiSymbolAggregator) buildSnapshotLocked() AggregatedSnapshot {
	syms := make(map[string]BarSnapshot, len(a.snapshots))
	for k, v := range a.snapshots {
		syms[k] = v
	}
	snap := AggregatedSnapshot{
		Symbols:   syms,
		Timestamp: time.Now().UTC(),
	}

	// Compute correlation matrix using current ring data.
	n := len(a.symbols)
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		matrix[i][i] = 1.0
	}
	retSeries := make([][]float64, n)
	for i, sym := range a.symbols {
		if r, ok := a.rings[sym]; ok {
			retSeries[i] = returns(r.values())
		}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			ai, bj := retSeries[i], retSeries[j]
			minLen := len(ai)
			if len(bj) < minLen {
				minLen = len(bj)
			}
			if minLen < 2 {
				continue
			}
			r := pearson(ai[len(ai)-minLen:], bj[len(bj)-minLen:])
			matrix[i][j] = r
			matrix[j][i] = r
		}
	}
	snap.CorrelationMatrix = matrix
	return snap
}

// copySnapshot does a shallow copy of an AggregatedSnapshot for safe external use.
func (a *MultiSymbolAggregator) copySnapshot(s AggregatedSnapshot) AggregatedSnapshot {
	out := AggregatedSnapshot{
		Timestamp: s.Timestamp,
		Symbols:   make(map[string]BarSnapshot, len(s.Symbols)),
	}
	for k, v := range s.Symbols {
		out.Symbols[k] = v
	}
	if s.CorrelationMatrix != nil {
		n := len(s.CorrelationMatrix)
		out.CorrelationMatrix = make([][]float64, n)
		for i := range s.CorrelationMatrix {
			row := make([]float64, len(s.CorrelationMatrix[i]))
			copy(row, s.CorrelationMatrix[i])
			out.CorrelationMatrix[i] = row
		}
	}
	return out
}

// broadcast sends a copy of snap to every subscriber without blocking.
func (a *MultiSymbolAggregator) broadcast(snap AggregatedSnapshot) {
	a.subsMu.Lock()
	subs := make([]chan AggregatedSnapshot, len(a.subscribers))
	copy(subs, a.subscribers)
	a.subsMu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- snap:
		default:
			// drop if subscriber is not draining fast enough
		}
	}
}

// BHMass computes the Black-Hole mass proxy: log1p(volume) * abs(close - open).
func BHMass(open, close, volume float64) float64 {
	if volume <= 0 {
		return 0
	}
	return math.Log1p(volume) * math.Abs(close-open)
}

// ClassifyRegime returns a regime string based on normalized trend strength.
// strength = (close - ema20) / atr
func ClassifyRegime(trendStrength float64) string {
	switch {
	case trendStrength > 1.5:
		return "trending_up"
	case trendStrength < -1.5:
		return "trending_down"
	case math.Abs(trendStrength) <= 1.5:
		return "ranging"
	default:
		return "unknown"
	}
}
