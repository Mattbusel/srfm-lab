// Package analytics provides real-time statistical computations on bar data streams.
package analytics

import (
	"math"
	"sort"
)

// Regime represents a classified market state for a symbol.
type Regime int

const (
	// RegimeTrending indicates a strong directional trend (high ADX).
	RegimeTrending Regime = iota
	// RegimeMeanReverting indicates choppy, oscillating price action (low Hurst).
	RegimeMeanReverting
	// RegimeHighVol indicates elevated short-term volatility.
	RegimeHighVol
	// RegimeLowVol indicates suppressed volatility -- potential breakout precursor.
	RegimeLowVol
	// RegimeUnknown is returned when insufficient data is available.
	RegimeUnknown
)

// String returns a human-readable label for the regime.
func (r Regime) String() string {
	switch r {
	case RegimeTrending:
		return "trending"
	case RegimeMeanReverting:
		return "mean_reverting"
	case RegimeHighVol:
		return "high_vol"
	case RegimeLowVol:
		return "low_vol"
	default:
		return "unknown"
	}
}

// RegimeClassification contains the full feature vector produced by
// ClassifyMulti in addition to the discrete Regime label.
type RegimeClassification struct {
	// Regime is the dominant regime label.
	Regime Regime
	// HurstExponent is from R/S analysis (0.5 = random walk, >0.5 = trending,
	// <0.5 = mean-reverting).
	HurstExponent float64
	// VolRatio is short_vol / long_vol; > 1 means recent vol is elevated.
	VolRatio float64
	// ADX is the Average Directional Index (0-100; >25 = trend present).
	ADX float64
	// Confidence in [0,1]: agreement between independent indicators.
	Confidence float64
}

// RegimeClassifier classifies the current market regime based on recent bar
// data for a single symbol.
//
// HurstWindow controls the lookback for R/S analysis (min 20 bars recommended).
// VolWindow controls the long lookback for volatility ratio (short window is
// VolWindow/4).
type RegimeClassifier struct {
	Symbol      string
	HurstWindow int
	VolWindow   int
}

// NewRegimeClassifier constructs a classifier with sensible defaults.
// HurstWindow=50, VolWindow=40 work well for 15-min bars.
func NewRegimeClassifier(symbol string) *RegimeClassifier {
	return &RegimeClassifier{
		Symbol:      symbol,
		HurstWindow: 50,
		VolWindow:   40,
	}
}

// Classify returns a single Regime label from the last min(HurstWindow, len(bars))
// bars. RegimeUnknown is returned when fewer than 5 bars are provided.
func (rc *RegimeClassifier) Classify(bars []Bar) Regime {
	cl := rc.ClassifyMulti(bars)
	return cl.Regime
}

// ClassifyMulti returns the full RegimeClassification including all computed
// features. Uses up to HurstWindow bars from the tail of bars.
func (rc *RegimeClassifier) ClassifyMulti(bars []Bar) RegimeClassification {
	if len(bars) < 5 {
		return RegimeClassification{Regime: RegimeUnknown}
	}

	// Use the most recent HurstWindow bars.
	window := rc.HurstWindow
	if len(bars) < window {
		window = len(bars)
	}
	recent := bars[len(bars)-window:]

	prices := extractClose(recent)
	hurst := ComputeHurst(prices)
	adx := ComputeADX(recent, 14)
	volRatio := ComputeVolRatio(bars, rc.VolWindow/4, rc.VolWindow)

	regime, confidence := determineRegime(hurst, adx, volRatio)

	return RegimeClassification{
		Regime:        regime,
		HurstExponent: hurst,
		VolRatio:      volRatio,
		ADX:           adx,
		Confidence:    confidence,
	}
}

// determineRegime selects the dominant regime from indicator values and
// returns a confidence score based on cross-indicator agreement.
func determineRegime(hurst, adx, volRatio float64) (Regime, float64) {
	// Count votes from each indicator.
	votes := map[Regime]int{}

	// Hurst vote: > 0.55 = trending, < 0.45 = mean-reverting.
	if hurst > 0.55 {
		votes[RegimeTrending]++
	} else if hurst < 0.45 {
		votes[RegimeMeanReverting]++
	}

	// ADX vote: > 25 = trending.
	if adx > 25 {
		votes[RegimeTrending]++
	} else if adx < 15 {
		votes[RegimeMeanReverting]++
	}

	// Vol ratio vote: > 1.3 = high vol, < 0.7 = low vol.
	if volRatio > 1.3 {
		votes[RegimeHighVol]++
	} else if volRatio < 0.7 {
		votes[RegimeLowVol]++
	}

	// Find the winner.
	best := RegimeUnknown
	bestVotes := 0
	for r, v := range votes {
		if v > bestVotes {
			bestVotes = v
			best = r
		}
	}

	// Confidence = fraction of max possible votes that agreed.
	// Max possible = 3 (all indicators vote the same).
	confidence := float64(bestVotes) / 3.0

	return best, confidence
}

// ComputeHurst estimates the Hurst exponent of prices using R/S analysis.
// It requires at least 8 prices; returns 0.5 (random walk) if the slice
// is too short or computation fails.
//
// Algorithm outline:
//  1. Convert prices to log-returns.
//  2. For each sub-interval of size n in {8, 16, 32, ...} <= len/2:
//     a. Split the series into non-overlapping chunks of size n.
//     b. For each chunk compute R/S = range(cumdev) / std(chunk).
//     c. Average R/S across chunks.
//  3. Regress log(R/S) ~ H * log(n) + c via least squares.
func ComputeHurst(prices []float64) float64 {
	if len(prices) < 8 {
		return 0.5
	}

	// Compute log-returns.
	rets := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] <= 0 || prices[i] <= 0 {
			return 0.5
		}
		rets[i-1] = math.Log(prices[i] / prices[i-1])
	}

	type point struct{ logN, logRS float64 }
	var pts []point

	// Collect (n, mean_RS) pairs for powers of 2 from 8 up to len(rets)/2.
	for n := 8; n <= len(rets)/2; n *= 2 {
		chunks := len(rets) / n
		if chunks == 0 {
			break
		}
		totalRS := 0.0
		for c := 0; c < chunks; c++ {
			chunk := rets[c*n : (c+1)*n]
			rs := rsStatistic(chunk)
			if math.IsNaN(rs) || rs <= 0 {
				continue
			}
			totalRS += rs
		}
		meanRS := totalRS / float64(chunks)
		if meanRS > 0 {
			pts = append(pts, point{math.Log(float64(n)), math.Log(meanRS)})
		}
	}

	if len(pts) < 2 {
		return 0.5
	}

	// OLS: H = slope of log(R/S) vs log(n).
	h := olsSlope(pts)
	if math.IsNaN(h) || h < 0 || h > 1 {
		return 0.5
	}
	return h
}

// rsStatistic computes the R/S statistic for a slice of returns.
func rsStatistic(rets []float64) float64 {
	n := len(rets)
	if n < 2 {
		return math.NaN()
	}

	// Mean.
	mean := 0.0
	for _, r := range rets {
		mean += r
	}
	mean /= float64(n)

	// Standard deviation.
	variance := 0.0
	for _, r := range rets {
		d := r - mean
		variance += d * d
	}
	variance /= float64(n)
	std := math.Sqrt(variance)
	if std == 0 {
		return math.NaN()
	}

	// Cumulative deviation series.
	cumDev := make([]float64, n)
	cumDev[0] = rets[0] - mean
	for i := 1; i < n; i++ {
		cumDev[i] = cumDev[i-1] + (rets[i] - mean)
	}

	// Range.
	minCD := cumDev[0]
	maxCD := cumDev[0]
	for _, v := range cumDev[1:] {
		if v < minCD {
			minCD = v
		}
		if v > maxCD {
			maxCD = v
		}
	}
	return (maxCD - minCD) / std
}

// olsSlope computes the OLS slope of the (x, y) point cloud.
type logPoint struct{ logN, logRS float64 }

func olsSlope(pts []struct{ logN, logRS float64 }) float64 {
	n := float64(len(pts))
	sumX, sumY, sumXX, sumXY := 0.0, 0.0, 0.0, 0.0
	for _, p := range pts {
		sumX += p.logN
		sumY += p.logRS
		sumXX += p.logN * p.logN
		sumXY += p.logN * p.logRS
	}
	denom := n*sumXX - sumX*sumX
	if denom == 0 {
		return math.NaN()
	}
	return (n*sumXY - sumX*sumY) / denom
}

// ComputeADX computes the Average Directional Index for bars over period.
// Returns 0 if fewer than period+1 bars are provided.
//
// Standard Wilder ADX using SMMA (Wilder smoothing = EMA with alpha=1/period).
func ComputeADX(bars []Bar, period int) float64 {
	if len(bars) < period+1 {
		return 0
	}

	alpha := 1.0 / float64(period)

	// Compute +DM, -DM, TR for each bar.
	type dmBar struct {
		plusDM  float64
		minusDM float64
		tr      float64
	}
	dms := make([]dmBar, len(bars)-1)
	for i := 1; i < len(bars); i++ {
		curr := bars[i]
		prev := bars[i-1]
		upMove := curr.High - prev.High
		downMove := prev.Low - curr.Low

		plusDM := 0.0
		minusDM := 0.0
		if upMove > downMove && upMove > 0 {
			plusDM = upMove
		}
		if downMove > upMove && downMove > 0 {
			minusDM = downMove
		}

		tr := math.Max(curr.High-curr.Low,
			math.Max(math.Abs(curr.High-prev.Close),
				math.Abs(curr.Low-prev.Close)))
		dms[i-1] = dmBar{plusDM: plusDM, minusDM: minusDM, tr: tr}
	}

	// Seed the first period's smoothed values with simple average.
	seed := dms[:period]
	smPlusDM, smMinusDM, smTR := 0.0, 0.0, 0.0
	for _, d := range seed {
		smPlusDM += d.plusDM
		smMinusDM += d.minusDM
		smTR += d.tr
	}

	dx := func() float64 {
		pdi := 0.0
		mdi := 0.0
		if smTR > 0 {
			pdi = 100 * smPlusDM / smTR
			mdi = 100 * smMinusDM / smTR
		}
		sum := pdi + mdi
		if sum == 0 {
			return 0
		}
		return 100 * math.Abs(pdi-mdi) / sum
	}

	// Seed ADX with first DX.
	adx := dx()

	// Walk remaining bars with Wilder smoothing.
	for _, d := range dms[period:] {
		smPlusDM = (1-alpha)*smPlusDM + alpha*d.plusDM
		smMinusDM = (1-alpha)*smMinusDM + alpha*d.minusDM
		smTR = (1-alpha)*smTR + alpha*d.tr
		adx = (1-alpha)*adx + alpha*dx()
	}

	return adx
}

// ComputeVolRatio computes the ratio of short-window realised volatility to
// long-window realised volatility. A ratio > 1 means recent vol is elevated.
// Returns 1.0 if insufficient bars are provided.
//
// Volatility is defined as the annualised standard deviation of log-returns
// over each window. The annualisation factor cancels in the ratio so it is
// omitted.
func ComputeVolRatio(bars []Bar, short, long int) float64 {
	if short < 1 {
		short = 1
	}
	if long < short {
		long = short
	}
	if len(bars) < long {
		return 1.0
	}

	allPrices := extractClose(bars)
	shortVol := realizedVol(allPrices[len(allPrices)-short-1:])
	longVol := realizedVol(allPrices[len(allPrices)-long-1:])

	if longVol == 0 {
		return 1.0
	}
	return shortVol / longVol
}

// realizedVol returns the sample standard deviation of log-returns.
func realizedVol(prices []float64) float64 {
	if len(prices) < 2 {
		return 0
	}
	rets := make([]float64, len(prices)-1)
	for i := 1; i < len(prices); i++ {
		if prices[i-1] <= 0 || prices[i] <= 0 {
			return 0
		}
		rets[i-1] = math.Log(prices[i] / prices[i-1])
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
	variance /= float64(len(rets) - 1) // Bessel correction
	return math.Sqrt(variance)
}

// extractClose extracts the Close price from a slice of bars.
func extractClose(bars []Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.Close
	}
	return out
}

// TopRegimes summarises the regime distribution over a list of historical
// bars by sliding a window and classifying each step. Returns a sorted
// frequency map.
func (rc *RegimeClassifier) TopRegimes(bars []Bar, stepSize int) map[string]int {
	if stepSize < 1 {
		stepSize = 1
	}
	counts := map[string]int{}
	if len(bars) < rc.HurstWindow {
		return counts
	}
	for start := 0; start+rc.HurstWindow <= len(bars); start += stepSize {
		end := start + rc.HurstWindow
		r := rc.Classify(bars[start:end])
		counts[r.String()]++
	}
	return counts
}

// DominantRegime returns the most frequently observed regime over bars.
func (rc *RegimeClassifier) DominantRegime(bars []Bar) Regime {
	freq := rc.TopRegimes(bars, 1)
	if len(freq) == 0 {
		return RegimeUnknown
	}
	type kv struct {
		k string
		v int
	}
	var pairs []kv
	for k, v := range freq {
		pairs = append(pairs, kv{k, v})
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].v > pairs[j].v })
	switch pairs[0].k {
	case "trending":
		return RegimeTrending
	case "mean_reverting":
		return RegimeMeanReverting
	case "high_vol":
		return RegimeHighVol
	case "low_vol":
		return RegimeLowVol
	default:
		return RegimeUnknown
	}
}
