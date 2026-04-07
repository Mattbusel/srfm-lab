package aggregator

import (
	"math"
)

// BarStatistics computes incremental rolling statistics for a single symbol.
// All updates are O(1) using EWMA recurrences and Welford's online algorithm.
type BarStatistics struct {
	symbol string

	// EMA state for three periods
	ema20  float64
	ema50  float64
	ema200 float64

	// ATR state (14-period EWMA of true range)
	atr    float64
	prevClose float64

	// Historical (realized) volatility via Welford on log-returns
	hvCount   int64
	hvMean    float64
	hvM2      float64  // sum of squared deviations
	HVol      float64  // annualized realized vol (stddev of log-returns * sqrt(252))

	// Volume z-score via Welford online algorithm
	volCount int64
	volMean  float64
	volM2    float64
	volStd   float64
	lastVol  float64

	// Smoothing factors
	alpha20  float64
	alpha50  float64
	alpha200 float64
	alphaATR float64

	initialized bool
	barCount    int64
}

// NewBarStatistics creates a BarStatistics tracker for the named symbol.
func NewBarStatistics(symbol string) *BarStatistics {
	return &BarStatistics{
		symbol:   symbol,
		alpha20:  2.0 / (20 + 1),
		alpha50:  2.0 / (50 + 1),
		alpha200: 2.0 / (200 + 1),
		alphaATR: 1.0 / 14.0,
	}
}

// Symbol returns the symbol this tracker is associated with.
func (s *BarStatistics) Symbol() string { return s.symbol }

// Update ingests a new OHLCV bar and updates all statistics in O(1) time.
func (s *BarStatistics) Update(bar OHLCV) {
	close := bar.Close
	high := bar.High
	low := bar.Low
	vol := bar.Volume

	if !s.initialized {
		// Seed all EMAs with the first close.
		s.ema20 = close
		s.ema50 = close
		s.ema200 = close
		s.atr = high - low
		s.prevClose = close
		s.initialized = true
		s.barCount = 1

		// seed volume Welford
		s.volCount = 1
		s.volMean = vol
		s.lastVol = vol
		return
	}

	// EMA updates
	s.ema20 = s.alpha20*close + (1-s.alpha20)*s.ema20
	s.ema50 = s.alpha50*close + (1-s.alpha50)*s.ema50
	s.ema200 = s.alpha200*close + (1-s.alpha200)*s.ema200

	// True range
	prevC := s.prevClose
	tr := trueRange(high, low, prevC)
	s.atr = s.alphaATR*tr + (1-s.alphaATR)*s.atr
	s.prevClose = close
	s.barCount++

	// Realized vol via Welford on log-returns
	if s.hvCount > 0 {
		logRet := math.Log(close / prevC)
		s.hvCount++
		delta := logRet - s.hvMean
		s.hvMean += delta / float64(s.hvCount)
		delta2 := logRet - s.hvMean
		s.hvM2 += delta * delta2
		if s.hvCount > 1 {
			variance := s.hvM2 / float64(s.hvCount-1)
			s.HVol = math.Sqrt(variance) * math.Sqrt(252)
		}
	} else {
		s.hvCount = 1
	}

	// Volume Welford update
	s.volCount++
	volDelta := vol - s.volMean
	s.volMean += volDelta / float64(s.volCount)
	volDelta2 := vol - s.volMean
	s.volM2 += volDelta * volDelta2
	if s.volCount > 1 {
		s.volStd = math.Sqrt(s.volM2 / float64(s.volCount-1))
	}
	s.lastVol = vol
}

// trueRange computes the true range given high, low, and previous close.
func trueRange(high, low, prevClose float64) float64 {
	a := high - low
	b := math.Abs(high - prevClose)
	c := math.Abs(low - prevClose)
	if a < b {
		a = b
	}
	if a < c {
		a = c
	}
	return a
}

// EMA20 returns the current 20-bar EMA of close prices.
func (s *BarStatistics) EMA20() float64 { return s.ema20 }

// EMA50 returns the current 50-bar EMA of close prices.
func (s *BarStatistics) EMA50() float64 { return s.ema50 }

// EMA200 returns the current 200-bar EMA of close prices.
func (s *BarStatistics) EMA200() float64 { return s.ema200 }

// ATR returns the current 14-period Average True Range.
func (s *BarStatistics) ATR() float64 { return s.atr }

// RealizedVol returns the annualized realized (historical) volatility.
func (s *BarStatistics) RealizedVol() float64 { return s.HVol }

// VolumeZScore returns the standardized volume of the last update.
// Returns 0 if there is insufficient history.
func (s *BarStatistics) VolumeZScore() float64 {
	if s.volStd == 0 {
		return 0
	}
	return (s.lastVol - s.volMean) / s.volStd
}

// IsAnomalousVolume returns true when the most recent volume exceeds
// the rolling mean by more than 3 standard deviations.
func (s *BarStatistics) IsAnomalousVolume() bool {
	return s.VolumeZScore() > 3.0
}

// TrendStrength returns (close - EMA200) / ATR, clipped to [-5, 5].
// Positive values indicate price above the long-term moving average.
func (s *BarStatistics) TrendStrength() float64 {
	if s.atr == 0 {
		return 0
	}
	raw := (s.prevClose - s.ema200) / s.atr
	return math.Max(-5.0, math.Min(5.0, raw))
}

// BarCount returns the number of bars processed since creation.
func (s *BarStatistics) BarCount() int64 { return s.barCount }

// Snapshot returns a BarSnapshot populated from current statistics.
// The caller provides the last OHLCV bar so signal / regime can be derived.
func (s *BarStatistics) Snapshot(bar OHLCV) BarSnapshot {
	ohlcv := [5]float64{bar.Open, bar.High, bar.Low, bar.Close, bar.Volume}
	bhMass := BHMass(bar.Open, bar.Close, bar.Volume)
	signal := s.TrendStrength()
	regime := ClassifyRegime(signal)
	return BarSnapshot{
		Symbol:  s.symbol,
		OHLCV:   ohlcv,
		BHMass:  bhMass,
		Signal:  signal,
		Regime:  regime,
	}
}

// StatsSnapshot is a read-only view of all computed statistics.
type StatsSnapshot struct {
	Symbol      string
	BarCount    int64
	EMA20       float64
	EMA50       float64
	EMA200      float64
	ATR         float64
	RealizedVol float64
	VolumeZScore float64
	TrendStrength float64
	IsAnomalousVolume bool
}

// Stats returns a StatsSnapshot of all computed values.
func (s *BarStatistics) Stats() StatsSnapshot {
	return StatsSnapshot{
		Symbol:            s.symbol,
		BarCount:          s.barCount,
		EMA20:             s.ema20,
		EMA50:             s.ema50,
		EMA200:            s.ema200,
		ATR:               s.atr,
		RealizedVol:       s.HVol,
		VolumeZScore:      s.VolumeZScore(),
		TrendStrength:     s.TrendStrength(),
		IsAnomalousVolume: s.IsAnomalousVolume(),
	}
}
