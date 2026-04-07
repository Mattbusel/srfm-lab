// Package analytics provides real-time statistical computations on bar data streams.
package analytics

import (
	"math"
	"sync"
	"time"

	"srfm/market-data/aggregator"
)

// Bar is an alias for the canonical bar type used throughout analytics.
type Bar = aggregator.BarEvent

// BarStats holds running statistics computed for a single symbol.
// All fields are updated incrementally using Welford's online algorithm
// and EWMA recurrences -- no full-history recomputation is needed.
type BarStats struct {
	Symbol string `json:"symbol"`

	// Price statistics (Welford online)
	Count    int64   `json:"count"`
	Mean     float64 `json:"mean"`
	Variance float64 `json:"variance"` // population variance (M2 / n)
	StdDev   float64 `json:"std_dev"`

	// Exponential moving averages (close price)
	EMA8  float64 `json:"ema8"`
	EMA21 float64 `json:"ema21"`

	// Average True Range (14-period EWMA of |H-L|)
	ATR float64 `json:"atr"`

	// Volume statistics
	VolEMA    float64 `json:"vol_ema"`    // volume EWMA (14-period)
	VolStd    float64 `json:"vol_std"`    // running volume std (Welford)
	VolZScore float64 `json:"vol_z_score"` // (volume - vol_ema) / vol_std

	// Last bar values for reference
	LastClose  float64   `json:"last_close"`
	LastHigh   float64   `json:"last_high"`
	LastLow    float64   `json:"last_low"`
	LastVolume float64   `json:"last_volume"`
	LastTime   time.Time `json:"last_time"`
}

// symbolState holds mutable accumulator state for one symbol.
// Separated from BarStats so callers always receive a clean value copy.
type symbolState struct {
	mu sync.Mutex

	count int64

	// Welford accumulators for close price
	wMean float64
	wM2   float64

	// EMA multipliers
	ema8Alpha  float64 // 2/(8+1)
	ema21Alpha float64 // 2/(21+1)
	atrAlpha   float64 // 2/(14+1)
	volAlpha   float64 // 2/(14+1) -- same as ATR period

	// EMA state
	ema8  float64
	ema21 float64
	atr   float64

	// Volume Welford + EWMA
	volEMA  float64
	volWM2  float64
	volMean float64 // Welford mean for volume variance

	// Previous close for true range
	prevClose float64
	initialized bool

	// Latest snapshot (updated in place, returned by value)
	last BarStats
}

func newSymbolState(symbol string) *symbolState {
	return &symbolState{
		ema8Alpha:  2.0 / (8.0 + 1.0),
		ema21Alpha: 2.0 / (21.0 + 1.0),
		atrAlpha:   2.0 / (14.0 + 1.0),
		volAlpha:   2.0 / (14.0 + 1.0),
		last: BarStats{
			Symbol: symbol,
		},
	}
}

// update applies a new bar and returns the updated BarStats snapshot.
func (s *symbolState) update(bar Bar) BarStats {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.count++
	n := float64(s.count)
	close_ := bar.Close
	vol := bar.Volume

	// -- Welford online update for close price --
	delta := close_ - s.wMean
	s.wMean += delta / n
	delta2 := close_ - s.wMean
	s.wM2 += delta * delta2

	var variance float64
	if s.count > 1 {
		variance = s.wM2 / n
	}
	stdDev := math.Sqrt(variance)

	// -- Welford update for volume --
	volDelta := vol - s.volMean
	s.volMean += volDelta / n
	volDelta2 := vol - s.volMean
	s.volWM2 += volDelta * volDelta2

	var volVariance float64
	if s.count > 1 {
		volVariance = s.volWM2 / n
	}
	volStd := math.Sqrt(volVariance)

	// -- EMA and ATR update --
	if !s.initialized {
		s.ema8 = close_
		s.ema21 = close_
		s.atr = bar.High - bar.Low
		s.volEMA = vol
		s.prevClose = close_
		s.initialized = true
	} else {
		// EMA recurrence: EMA = price * alpha + prev_EMA * (1-alpha)
		s.ema8 = close_*s.ema8Alpha + s.ema8*(1-s.ema8Alpha)
		s.ema21 = close_*s.ema21Alpha + s.ema21*(1-s.ema21Alpha)

		// True range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
		tr := trueRange(bar.High, bar.Low, s.prevClose)
		s.atr = tr*s.atrAlpha + s.atr*(1-s.atrAlpha)

		// Volume EWMA
		s.volEMA = vol*s.volAlpha + s.volEMA*(1-s.volAlpha)

		s.prevClose = close_
	}

	// Volume z-score: protect against zero std
	var volZ float64
	if volStd > 0 {
		volZ = (vol - s.volEMA) / volStd
	}

	s.last = BarStats{
		Symbol:     s.last.Symbol,
		Count:      s.count,
		Mean:       s.wMean,
		Variance:   variance,
		StdDev:     stdDev,
		EMA8:       s.ema8,
		EMA21:      s.ema21,
		ATR:        s.atr,
		VolEMA:     s.volEMA,
		VolStd:     volStd,
		VolZScore:  volZ,
		LastClose:  close_,
		LastHigh:   bar.High,
		LastLow:    bar.Low,
		LastVolume: vol,
		LastTime:   bar.Timestamp,
	}
	return s.last
}

func (s *symbolState) snapshot() (BarStats, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.count == 0 {
		return BarStats{}, false
	}
	return s.last, true
}

// trueRange computes the true range for a bar given the previous close.
func trueRange(high, low, prevClose float64) float64 {
	hl := high - low
	hpc := math.Abs(high - prevClose)
	lpc := math.Abs(low - prevClose)
	tr := hl
	if hpc > tr {
		tr = hpc
	}
	if lpc > tr {
		tr = lpc
	}
	return tr
}

// BarStatsComputer maintains per-symbol rolling statistics in a sync.Map.
// It is safe for concurrent use by multiple goroutines.
type BarStatsComputer struct {
	states sync.Map // symbol -> *symbolState
}

// NewBarStatsComputer creates a BarStatsComputer ready for use.
func NewBarStatsComputer() *BarStatsComputer {
	return &BarStatsComputer{}
}

// OnBar processes a new bar for the given symbol and returns updated BarStats.
// If this is the first bar for the symbol, a new state entry is created.
func (c *BarStatsComputer) OnBar(symbol string, bar Bar) BarStats {
	val, _ := c.states.LoadOrStore(symbol, newSymbolState(symbol))
	state := val.(*symbolState)
	return state.update(bar)
}

// GetStats performs a thread-safe lookup of current statistics for a symbol.
// Returns (BarStats, true) if data exists, or (BarStats{}, false) if not.
func (c *BarStatsComputer) GetStats(symbol string) (BarStats, bool) {
	val, ok := c.states.Load(symbol)
	if !ok {
		return BarStats{}, false
	}
	state := val.(*symbolState)
	return state.snapshot()
}

// Symbols returns all symbols currently tracked by the computer.
func (c *BarStatsComputer) Symbols() []string {
	var out []string
	c.states.Range(func(k, _ interface{}) bool {
		out = append(out, k.(string))
		return true
	})
	return out
}
