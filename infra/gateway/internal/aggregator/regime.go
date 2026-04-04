package aggregator

import (
	"math"
	"sync"

	"github.com/srfm/gateway/internal/feed"
)

// MarketRegime classifies the current market state.
type MarketRegime int

const (
	RegimeUnknown MarketRegime = iota
	RegimeTrending
	RegimeMeanReverting
	RegimeHighVolatility
	RegimeLowVolatility
	RegimeBreakout
)

func (r MarketRegime) String() string {
	switch r {
	case RegimeTrending:
		return "trending"
	case RegimeMeanReverting:
		return "mean_reverting"
	case RegimeHighVolatility:
		return "high_volatility"
	case RegimeLowVolatility:
		return "low_volatility"
	case RegimeBreakout:
		return "breakout"
	default:
		return "unknown"
	}
}

// RegimeSnapshot holds the detected regime and supporting metrics.
type RegimeSnapshot struct {
	Regime        MarketRegime
	ADX           float64
	Volatility    float64
	HurstExponent float64
	TrendStrength float64
	Label         string
}

// RegimeDetector detects market regime from a slice of bars.
type RegimeDetector struct {
	mu     sync.Mutex
	window int // number of bars to use
}

// NewRegimeDetector creates a RegimeDetector with the given lookback window.
func NewRegimeDetector(window int) *RegimeDetector {
	if window < 20 {
		window = 50
	}
	return &RegimeDetector{window: window}
}

// Detect analyses bars and returns the current regime snapshot.
func (rd *RegimeDetector) Detect(bars []feed.Bar) RegimeSnapshot {
	rd.mu.Lock()
	defer rd.mu.Unlock()

	n := rd.window
	if len(bars) < n {
		n = len(bars)
	}
	if n < 10 {
		return RegimeSnapshot{Regime: RegimeUnknown, Label: "unknown"}
	}
	window := bars[len(bars)-n:]

	adx := computeADX(window, 14)
	vol := annualisedVol(window)
	hurst := hurstExponent(window)
	trend := trendStrength(window)

	// Classification logic.
	regime := classifyRegime(adx, vol, hurst, trend)

	return RegimeSnapshot{
		Regime:        regime,
		ADX:           adx,
		Volatility:    vol,
		HurstExponent: hurst,
		TrendStrength: trend,
		Label:         regime.String(),
	}
}

// classifyRegime determines the regime from computed metrics.
func classifyRegime(adx, vol, hurst, trend float64) MarketRegime {
	// High ADX => trending.
	if adx > 25 && trend > 0.6 {
		return RegimeTrending
	}
	// Very high volatility.
	if vol > 0.40 {
		return RegimeHighVolatility
	}
	// Hurst > 0.6 => persistent (trending), < 0.4 => mean reverting.
	if hurst > 0.60 {
		return RegimeTrending
	}
	if hurst < 0.40 {
		return RegimeMeanReverting
	}
	// Low ADX + recent range expansion => breakout candidate.
	if adx < 15 && vol > 0.20 {
		return RegimeBreakout
	}
	if vol < 0.10 {
		return RegimeLowVolatility
	}
	return RegimeUnknown
}

// annualisedVol computes annualised daily return volatility.
func annualisedVol(bars []feed.Bar) float64 {
	if len(bars) < 2 {
		return 0
	}
	rets := make([]float64, 0, len(bars)-1)
	for i := 1; i < len(bars); i++ {
		if bars[i-1].Close > 0 {
			rets = append(rets, math.Log(bars[i].Close/bars[i-1].Close))
		}
	}
	if len(rets) == 0 {
		return 0
	}
	mean := 0.0
	for _, r := range rets {
		mean += r
	}
	mean /= float64(len(rets))
	variance := 0.0
	for _, r := range rets {
		d := r - mean
		variance += d * d
	}
	variance /= float64(len(rets))
	return math.Sqrt(variance) * math.Sqrt(252)
}

// computeADX is a simplified ADX calculation.
func computeADX(bars []feed.Bar, period int) float64 {
	if len(bars) < period*2 {
		return 0
	}
	type tr struct{ plus, minus, atr float64 }
	trs := make([]tr, len(bars)-1)
	for i := 1; i < len(bars); i++ {
		b, p := bars[i], bars[i-1]
		up := b.High - p.High
		dn := p.Low - b.Low
		dmPlus, dmMinus := 0.0, 0.0
		if up > dn && up > 0 {
			dmPlus = up
		}
		if dn > up && dn > 0 {
			dmMinus = dn
		}
		atr := b.High - b.Low
		if v := math.Abs(b.High - p.Close); v > atr {
			atr = v
		}
		if v := math.Abs(b.Low - p.Close); v > atr {
			atr = v
		}
		trs[i-1] = tr{dmPlus, dmMinus, atr}
	}

	sumATR, sumPlus, sumMinus := 0.0, 0.0, 0.0
	for _, t := range trs[:period] {
		sumATR += t.atr
		sumPlus += t.plus
		sumMinus += t.minus
	}
	adx := 0.0
	if sumATR > 0 {
		diP := sumPlus / sumATR * 100
		diM := sumMinus / sumATR * 100
		if diP+diM > 0 {
			adx = math.Abs(diP-diM) / (diP + diM) * 100
		}
	}
	for _, t := range trs[period:] {
		sumATR = sumATR - sumATR/float64(period) + t.atr
		sumPlus = sumPlus - sumPlus/float64(period) + t.plus
		sumMinus = sumMinus - sumMinus/float64(period) + t.minus
		if sumATR > 0 {
			diP := sumPlus / sumATR * 100
			diM := sumMinus / sumATR * 100
			dx := 0.0
			if diP+diM > 0 {
				dx = math.Abs(diP-diM) / (diP + diM) * 100
			}
			adx = adx - adx/float64(period) + dx/float64(period)
		}
	}
	return adx
}

// hurstExponent estimates the Hurst exponent using the R/S analysis method.
// Values near 0.5 = random walk, >0.5 = persistent, <0.5 = mean reverting.
func hurstExponent(bars []feed.Bar) float64 {
	if len(bars) < 16 {
		return 0.5
	}
	prices := make([]float64, len(bars))
	for i, b := range bars {
		prices[i] = b.Close
	}

	// Use log returns.
	rets := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] > 0 {
			rets[i-1] = math.Log(prices[i] / prices[i-1])
		}
	}

	// Compute R/S for different lags.
	lags := []int{4, 8, 16}
	if len(rets) > 32 {
		lags = append(lags, 32)
	}

	logLags := make([]float64, len(lags))
	logRS := make([]float64, len(lags))

	for li, lag := range lags {
		if lag > len(rets) {
			logLags[li] = math.Log(float64(lag))
			logRS[li] = 0
			continue
		}
		sub := rets[:lag]
		mean := 0.0
		for _, r := range sub {
			mean += r
		}
		mean /= float64(lag)

		// Cumulative deviation.
		cumDev := make([]float64, lag)
		for i, r := range sub {
			if i == 0 {
				cumDev[i] = r - mean
			} else {
				cumDev[i] = cumDev[i-1] + (r - mean)
			}
		}
		maxCD, minCD := cumDev[0], cumDev[0]
		for _, v := range cumDev[1:] {
			if v > maxCD {
				maxCD = v
			}
			if v < minCD {
				minCD = v
			}
		}
		r := maxCD - minCD

		variance := 0.0
		for _, rv := range sub {
			d := rv - mean
			variance += d * d
		}
		s := math.Sqrt(variance / float64(lag))
		if s == 0 {
			logRS[li] = 0
		} else {
			logRS[li] = math.Log(r / s)
		}
		logLags[li] = math.Log(float64(lag))
	}

	// Linear regression of logRS vs logLags → slope = Hurst exponent.
	return linearSlope(logLags, logRS)
}

// linearSlope returns the slope of the OLS linear fit of y on x.
func linearSlope(x, y []float64) float64 {
	n := float64(len(x))
	if n < 2 {
		return 0.5
	}
	sumX, sumY, sumXY, sumX2 := 0.0, 0.0, 0.0, 0.0
	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
	}
	denom := n*sumX2 - sumX*sumX
	if denom == 0 {
		return 0.5
	}
	return (n*sumXY - sumX*sumY) / denom
}

// trendStrength measures how strongly prices are trending using linear regression R².
func trendStrength(bars []feed.Bar) float64 {
	if len(bars) < 5 {
		return 0
	}
	n := float64(len(bars))
	x := make([]float64, len(bars))
	y := make([]float64, len(bars))
	for i, b := range bars {
		x[i] = float64(i)
		y[i] = b.Close
	}
	sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := range x {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	// Pearson r.
	num := n*sumXY - sumX*sumY
	den := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))
	if den == 0 {
		return 0
	}
	r := num / den
	return math.Abs(r) // R (not R²) as trend strength [0,1]
}

// RegimeRegistry tracks regime state per symbol.
type RegimeRegistry struct {
	mu       sync.RWMutex
	detector *RegimeDetector
	regimes  map[string]RegimeSnapshot
}

// NewRegimeRegistry creates a RegimeRegistry.
func NewRegimeRegistry(window int) *RegimeRegistry {
	return &RegimeRegistry{
		detector: NewRegimeDetector(window),
		regimes:  make(map[string]RegimeSnapshot),
	}
}

// Update re-computes the regime for the given symbol using the provided bars.
func (rr *RegimeRegistry) Update(symbol string, bars []feed.Bar) RegimeSnapshot {
	snap := rr.detector.Detect(bars)
	rr.mu.Lock()
	rr.regimes[symbol] = snap
	rr.mu.Unlock()
	return snap
}

// Get returns the last known regime snapshot for a symbol.
func (rr *RegimeRegistry) Get(symbol string) (RegimeSnapshot, bool) {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	s, ok := rr.regimes[symbol]
	return s, ok
}

// All returns a snapshot of all current regimes.
func (rr *RegimeRegistry) All() map[string]RegimeSnapshot {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	out := make(map[string]RegimeSnapshot, len(rr.regimes))
	for k, v := range rr.regimes {
		out[k] = v
	}
	return out
}
