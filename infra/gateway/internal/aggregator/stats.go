package aggregator

import (
	"math"
	"sync"
	"time"

	"github.com/srfm/gateway/internal/feed"
)

// SymbolStats tracks rolling statistics for a single symbol.
type SymbolStats struct {
	Symbol    string
	UpdatedAt time.Time
	// Rolling window size.
	Window int

	mu       sync.RWMutex
	closes   []float64
	volumes  []float64
	returns  []float64
}

// NewSymbolStats creates a SymbolStats with the given rolling window.
func NewSymbolStats(symbol string, window int) *SymbolStats {
	if window <= 0 {
		window = 20
	}
	return &SymbolStats{
		Symbol: symbol,
		Window: window,
	}
}

// Update incorporates a new bar.
func (ss *SymbolStats) Update(b feed.Bar) {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	prev := 0.0
	if len(ss.closes) > 0 {
		prev = ss.closes[len(ss.closes)-1]
	}

	ss.closes = appendRolling(ss.closes, b.Close, ss.Window)
	ss.volumes = appendRolling(ss.volumes, b.Volume, ss.Window)

	if prev > 0 {
		ret := (b.Close - prev) / prev
		ss.returns = appendRolling(ss.returns, ret, ss.Window)
	}
	ss.UpdatedAt = b.Timestamp
}

// Mean returns the rolling mean of closes.
func (ss *SymbolStats) Mean() float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return mean(ss.closes)
}

// StdDev returns the rolling standard deviation of closes.
func (ss *SymbolStats) StdDev() float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return stdDev(ss.closes)
}

// ReturnMean returns the mean of recent returns.
func (ss *SymbolStats) ReturnMean() float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return mean(ss.returns)
}

// ReturnStdDev returns the stddev of recent returns.
func (ss *SymbolStats) ReturnStdDev() float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return stdDev(ss.returns)
}

// RollingSharpe returns an annualised Sharpe estimate for the window.
// barsPerYear is the number of bars in one year.
func (ss *SymbolStats) RollingSharpe(barsPerYear float64) float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return Sharpe(ss.returns, barsPerYear)
}

// VolRatio returns volume / mean_volume.
func (ss *SymbolStats) VolRatio(lastVol float64) float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	avg := mean(ss.volumes)
	if avg == 0 {
		return 1
	}
	return lastVol / avg
}

// Latest returns the most recent close.
func (ss *SymbolStats) Latest() float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	if len(ss.closes) == 0 {
		return 0
	}
	return ss.closes[len(ss.closes)-1]
}

// ZScore returns how many standard deviations the latest close is from the mean.
func (ss *SymbolStats) ZScore() float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	if len(ss.closes) < 2 {
		return 0
	}
	m := mean(ss.closes)
	s := stdDev(ss.closes)
	if s == 0 {
		return 0
	}
	return (ss.closes[len(ss.closes)-1] - m) / s
}

// Snapshot returns a copy of all computed statistics.
func (ss *SymbolStats) Snapshot() map[string]float64 {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return map[string]float64{
		"close":       last(ss.closes),
		"mean":        mean(ss.closes),
		"std_dev":     stdDev(ss.closes),
		"ret_mean":    mean(ss.returns),
		"ret_std_dev": stdDev(ss.returns),
		"vol_mean":    mean(ss.volumes),
		"z_score":     zScore(ss.closes),
	}
}

// ---- StatsRegistry ----

// StatsRegistry holds per-symbol stats across all tracked symbols.
type StatsRegistry struct {
	mu     sync.RWMutex
	stats  map[string]*SymbolStats
	window int
}

// NewStatsRegistry creates a StatsRegistry.
func NewStatsRegistry(window int) *StatsRegistry {
	return &StatsRegistry{
		stats:  make(map[string]*SymbolStats),
		window: window,
	}
}

// Update incorporates a bar for its symbol.
func (sr *StatsRegistry) Update(b feed.Bar) {
	sr.mu.RLock()
	st, ok := sr.stats[b.Symbol]
	sr.mu.RUnlock()

	if !ok {
		sr.mu.Lock()
		st, ok = sr.stats[b.Symbol]
		if !ok {
			st = NewSymbolStats(b.Symbol, sr.window)
			sr.stats[b.Symbol] = st
		}
		sr.mu.Unlock()
	}
	st.Update(b)
}

// Get returns the SymbolStats for a symbol, or nil.
func (sr *StatsRegistry) Get(symbol string) *SymbolStats {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	return sr.stats[symbol]
}

// Symbols returns all tracked symbols.
func (sr *StatsRegistry) Symbols() []string {
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	out := make([]string, 0, len(sr.stats))
	for sym := range sr.stats {
		out = append(out, sym)
	}
	return out
}

// All returns snapshots for all symbols.
func (sr *StatsRegistry) All() map[string]map[string]float64 {
	sr.mu.RLock()
	syms := make([]string, 0, len(sr.stats))
	for sym := range sr.stats {
		syms = append(syms, sym)
	}
	sr.mu.RUnlock()

	result := make(map[string]map[string]float64, len(syms))
	for _, sym := range syms {
		if st := sr.Get(sym); st != nil {
			result[sym] = st.Snapshot()
		}
	}
	return result
}

// ---- helpers ----

func appendRolling(s []float64, v float64, maxLen int) []float64 {
	s = append(s, v)
	if len(s) > maxLen {
		s = s[len(s)-maxLen:]
	}
	return s
}

func mean(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

func stdDev(s []float64) float64 {
	if len(s) < 2 {
		return 0
	}
	m := mean(s)
	var sq float64
	for _, v := range s {
		d := v - m
		sq += d * d
	}
	variance := sq / float64(len(s)-1)
	return math.Sqrt(variance)
}

func last(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	return s[len(s)-1]
}

func zScore(s []float64) float64 {
	if len(s) < 2 {
		return 0
	}
	m := mean(s)
	sd := stdDev(s)
	if sd == 0 {
		return 0
	}
	return (s[len(s)-1] - m) / sd
}
