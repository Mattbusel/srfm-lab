// Package indicators -- signal_indicators.go contains SRFM-specific indicators
// that wrap the physics-inspired BH mass model, Hurst exponent estimation, and
// a composite regime classifier.
package indicators

import (
	"math"
)

// -----------------------------------------------------------------------------
// BHMassIndicator -- Black-Hole mass accumulation / decay model
// -----------------------------------------------------------------------------
// The BH mass model treats price momentum as a gravitational accretion process.
// Parameters mirror the Python reference implementation:
//   bh_form    = 1.92  -- multiplier applied when close > open (accreting)
//   bh_decay   = 0.924 -- per-bar multiplicative decay of mass
//   bh_collapse= 0.992 -- collapse factor applied after mass crosses threshold

const (
	defaultBHForm     = 1.92
	defaultBHDecay    = 0.924
	defaultBHCollapse = 0.992
	defaultBHThresh   = 10.0 // mass threshold triggering collapse reset
)

// BHMassIndicator tracks a pseudo-gravitational mass for a price series.
// A higher mass value indicates stronger directional momentum.
type BHMassIndicator struct {
	// Parameters
	bhForm     float64
	bhDecay    float64
	bhCollapse float64
	bhThresh   float64

	// State
	mass     float64
	prevBar  OHLCV
	hasFirst bool
	minkowski float64 // last Minkowski-metric distance contribution
}

// NewBHMassIndicator creates a BHMassIndicator with default SRFM parameters.
func NewBHMassIndicator() *BHMassIndicator {
	return &BHMassIndicator{
		bhForm:     defaultBHForm,
		bhDecay:    defaultBHDecay,
		bhCollapse: defaultBHCollapse,
		bhThresh:   defaultBHThresh,
	}
}

// NewBHMassIndicatorParams creates a BHMassIndicator with custom parameters.
func NewBHMassIndicatorParams(form, decay, collapse, thresh float64) *BHMassIndicator {
	return &BHMassIndicator{
		bhForm:     form,
		bhDecay:    decay,
		bhCollapse: collapse,
		bhThresh:   thresh,
	}
}

// Update feeds a new bar into the BH mass model.
// The Minkowski metric is used to compute the "distance" between successive
// bars in (open, high, low, close) space with p=2 (Euclidean).
func (b *BHMassIndicator) Update(bar OHLCV) {
	if !b.hasFirst {
		b.prevBar = bar
		b.hasFirst = true
		return
	}

	// Minkowski distance (p=2) between consecutive bar vectors.
	dO := bar.Open - b.prevBar.Open
	dH := bar.High - b.prevBar.High
	dL := bar.Low - b.prevBar.Low
	dC := bar.Close - b.prevBar.Close
	b.minkowski = math.Sqrt(dO*dO + dH*dH + dL*dL + dC*dC)

	// Decay existing mass.
	b.mass *= b.bhDecay

	// Accretion: bullish bar adds mass scaled by bh_form.
	if bar.Close > bar.Open {
		b.mass += b.minkowski * b.bhForm
	} else if bar.Close < bar.Open {
		// Bearish bar decays mass further.
		b.mass -= b.minkowski * (2.0 - b.bhDecay)
	}

	// Collapse: if mass exceeds threshold, apply collapse factor.
	if math.Abs(b.mass) > b.bhThresh {
		b.mass *= b.bhCollapse
	}

	b.prevBar = bar
}

// Value returns the current BH mass. Positive values indicate bullish
// momentum; negative values indicate bearish momentum.
func (b *BHMassIndicator) Value() float64 { return b.mass }

// Minkowski returns the last computed bar-to-bar Minkowski distance.
func (b *BHMassIndicator) Minkowski() float64 { return b.minkowski }

// -----------------------------------------------------------------------------
// HurstIndicator -- R/S analysis Hurst exponent estimator
// -----------------------------------------------------------------------------

// HurstIndicator estimates the Hurst exponent using the rescaled-range (R/S)
// method over a rolling window of prices. Returns 0.5 during warm-up.
//
//   H > 0.5  -> trending / persistent
//   H < 0.5  -> mean-reverting
//   H ~ 0.5  -> random walk
type HurstIndicator struct {
	window int
	prices []float64
	h      float64
}

// NewHurstIndicator creates a HurstIndicator with the given window size.
// Minimum window is 8; the default used throughout SRFM is 64.
func NewHurstIndicator(window int) *HurstIndicator {
	if window < 8 {
		window = 8
	}
	return &HurstIndicator{window: window, prices: make([]float64, 0, window)}
}

// Update adds a new price observation and recomputes H if the window is full.
// Returns the current Hurst exponent (0.5 during warm-up).
func (h *HurstIndicator) Update(price float64) float64 {
	h.prices = append(h.prices, price)
	if len(h.prices) > h.window {
		h.prices = h.prices[len(h.prices)-h.window:]
	}
	if len(h.prices) < h.window {
		return 0.5
	}
	h.h = rsHurst(h.prices)
	return h.h
}

// Value returns the current Hurst exponent (0.5 during warm-up).
func (h *HurstIndicator) Value() float64 {
	if len(h.prices) < h.window {
		return 0.5
	}
	return h.h
}

// rsHurst computes H via the classic R/S method using sub-period halving.
func rsHurst(prices []float64) float64 {
	n := len(prices)
	if n < 8 {
		return 0.5
	}

	// Compute log returns.
	logR := make([]float64, n-1)
	for i := 1; i < n; i++ {
		if prices[i-1] <= 0 || prices[i] <= 0 {
			return 0.5
		}
		logR[i-1] = math.Log(prices[i] / prices[i-1])
	}

	// Use multiple sub-period sizes to fit log(R/S) ~ H*log(n).
	var xSum, ySum, xxSum, xySum float64
	var count float64

	sizes := subPeriodSizes(len(logR))
	for _, sz := range sizes {
		if sz < 4 {
			continue
		}
		rs := averageRS(logR, sz)
		if rs <= 0 {
			continue
		}
		x := math.Log(float64(sz))
		y := math.Log(rs)
		xSum += x
		ySum += y
		xxSum += x * x
		xySum += x * y
		count++
	}

	if count < 2 {
		return 0.5
	}
	denom := count*xxSum - xSum*xSum
	if denom == 0 {
		return 0.5
	}
	h := (count*xySum - xSum*ySum) / denom
	if h < 0 {
		h = 0
	}
	if h > 1 {
		h = 1
	}
	return h
}

// subPeriodSizes returns a set of sub-period sizes for R/S analysis.
func subPeriodSizes(n int) []int {
	var sizes []int
	for sz := 8; sz <= n; sz *= 2 {
		sizes = append(sizes, sz)
	}
	return sizes
}

// averageRS computes the mean R/S over non-overlapping sub-periods of length sz.
func averageRS(returns []float64, sz int) float64 {
	n := len(returns)
	if n < sz {
		return 0
	}
	count := n / sz
	total := 0.0
	for i := 0; i < count; i++ {
		sub := returns[i*sz : (i+1)*sz]
		rs := computeRS(sub)
		total += rs
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

// computeRS computes R/S for a single return series.
func computeRS(returns []float64) float64 {
	n := len(returns)
	if n == 0 {
		return 0
	}
	// Mean
	sum := 0.0
	for _, r := range returns {
		sum += r
	}
	mean := sum / float64(n)

	// Cumulative deviation
	cumDev := make([]float64, n)
	cumDev[0] = returns[0] - mean
	for i := 1; i < n; i++ {
		cumDev[i] = cumDev[i-1] + (returns[i] - mean)
	}

	// Range
	minDev, maxDev := cumDev[0], cumDev[0]
	for _, v := range cumDev {
		if v < minDev {
			minDev = v
		}
		if v > maxDev {
			maxDev = v
		}
	}
	R := maxDev - minDev

	// Standard deviation
	variance := 0.0
	for _, r := range returns {
		d := r - mean
		variance += d * d
	}
	std := math.Sqrt(variance / float64(n))
	if std == 0 {
		return 0
	}
	return R / std
}

// -----------------------------------------------------------------------------
// RegimeIndicator -- composite market regime classifier
// -----------------------------------------------------------------------------

// RegimeIndicator combines BH mass, Hurst exponent, and ATR to classify the
// current market regime into one of four states.
type RegimeIndicator struct {
	bh    BHMassIndicator
	hurst HurstIndicator
	atr   ATR
}

// Regime constants returned by Classify.
const (
	RegimeTrendingBull = "TRENDING_BULL"
	RegimeTrendingBear = "TRENDING_BEAR"
	RegimeRanging      = "RANGING"
	RegimeHighVol      = "HIGH_VOL"
)

// NewRegimeIndicator creates a RegimeIndicator with standard parameters.
func NewRegimeIndicator() *RegimeIndicator {
	return &RegimeIndicator{
		bh:    *NewBHMassIndicator(),
		hurst: *NewHurstIndicator(64),
		atr:   *NewATR(14),
	}
}

// Update feeds a new bar into all sub-indicators.
func (r *RegimeIndicator) Update(bar OHLCV) {
	r.bh.Update(bar)
	r.hurst.Update(bar.Close)
	r.atr.Update(bar)
}

// Classify returns the current market regime string.
//
// Logic:
//   HIGH_VOL      -- ATR is primed and ATR/close ratio > 3%
//   TRENDING_BULL -- Hurst > 0.55 and BH mass > 0
//   TRENDING_BEAR -- Hurst > 0.55 and BH mass < 0
//   RANGING       -- everything else (low Hurst, no strong mass)
func (r *RegimeIndicator) Classify() string {
	atrVal := r.atr.Value()
	hurstVal := r.hurst.Value()
	massVal := r.bh.Value()

	// Need ATR reference price from last prevClose -- approximate via
	// checking ATR primed state only; no close stored separately.
	if r.atr.Primed() && atrVal > 0 {
		// HIGH_VOL heuristic: compare ATR to approximate price scale.
		// Use BH prevClose as proxy if available; else skip.
		if r.bh.hasFirst {
			closeProxy := r.bh.prevBar.Close
			if closeProxy > 0 && (atrVal/closeProxy) > 0.03 {
				return RegimeHighVol
			}
		}
	}

	if hurstVal > 0.55 {
		if massVal >= 0 {
			return RegimeTrendingBull
		}
		return RegimeTrendingBear
	}

	return RegimeRanging
}

// BHMass returns the current BH mass value.
func (r *RegimeIndicator) BHMass() float64 { return r.bh.Value() }

// Hurst returns the current Hurst exponent.
func (r *RegimeIndicator) Hurst() float64 { return r.hurst.Value() }

// ATRValue returns the current ATR.
func (r *RegimeIndicator) ATRValue() float64 { return r.atr.Value() }
