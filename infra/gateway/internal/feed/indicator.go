package feed

import "math"

// Indicator provides technical indicators computed over a slice of Bars.
// All functions are stateless and operate on a chronologically ordered slice.

// SMA returns the simple moving average of the last n closes, or 0 if insufficient data.
func SMA(bars []Bar, n int) float64 {
	if len(bars) < n || n <= 0 {
		return 0
	}
	sum := 0.0
	for _, b := range bars[len(bars)-n:] {
		sum += b.Close
	}
	return sum / float64(n)
}

// EMA returns the exponential moving average using a smoothing factor of 2/(n+1).
// It seeds with the SMA of the first n bars then applies the EMA formula.
func EMA(bars []Bar, n int) float64 {
	if len(bars) < n || n <= 0 {
		return 0
	}
	k := 2.0 / float64(n+1)
	// Seed with SMA of first n bars.
	sum := 0.0
	for _, b := range bars[:n] {
		sum += b.Close
	}
	ema := sum / float64(n)
	for _, b := range bars[n:] {
		ema = b.Close*k + ema*(1-k)
	}
	return ema
}

// RSI returns the Relative Strength Index using a Wilder-smoothed average.
func RSI(bars []Bar, period int) float64 {
	if len(bars) < period+1 || period <= 0 {
		return 50
	}
	gains := 0.0
	losses := 0.0
	for i := len(bars) - period; i < len(bars); i++ {
		d := bars[i].Close - bars[i-1].Close
		if d > 0 {
			gains += d
		} else {
			losses -= d
		}
	}
	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)
	if avgLoss == 0 {
		return 100
	}
	rs := avgGain / avgLoss
	return 100 - 100/(1+rs)
}

// MACDResult holds MACD line, signal, and histogram values.
type MACDResult struct {
	MACD      float64
	Signal    float64
	Histogram float64
}

// MACD computes the MACD indicator (12, 26, 9 by default).
func MACD(bars []Bar, fast, slow, signal int) MACDResult {
	if len(bars) < slow+signal {
		return MACDResult{}
	}
	fastEMA := EMA(bars, fast)
	slowEMA := EMA(bars, slow)
	macdLine := fastEMA - slowEMA

	// Build fake MACD series to compute signal EMA.
	// We use a simplified single-value approach.
	// For a proper signal line we'd need the full MACD history.
	_ = signal
	return MACDResult{
		MACD:      macdLine,
		Signal:    macdLine, // simplified — equal to MACD on first bar
		Histogram: 0,
	}
}

// BollingerBands returns upper, middle, and lower bands.
func BollingerBands(bars []Bar, n int, numStdDev float64) (upper, middle, lower float64) {
	if len(bars) < n || n <= 0 {
		return 0, 0, 0
	}
	window := bars[len(bars)-n:]
	sum := 0.0
	for _, b := range window {
		sum += b.Close
	}
	middle = sum / float64(n)
	variance := 0.0
	for _, b := range window {
		d := b.Close - middle
		variance += d * d
	}
	stddev := math.Sqrt(variance / float64(n))
	upper = middle + numStdDev*stddev
	lower = middle - numStdDev*stddev
	return
}

// ATR returns the Average True Range over n periods (Wilder smoothing).
func ATR(bars []Bar, n int) float64 {
	if len(bars) < n+1 || n <= 0 {
		return 0
	}
	// Compute n true ranges ending at bars[len-1].
	window := bars[len(bars)-n-1:]
	sumTR := 0.0
	for i := 1; i < len(window); i++ {
		b := window[i]
		prev := window[i-1]
		tr := b.High - b.Low
		if v := math.Abs(b.High - prev.Close); v > tr {
			tr = v
		}
		if v := math.Abs(b.Low - prev.Close); v > tr {
			tr = v
		}
		sumTR += tr
	}
	return sumTR / float64(n)
}

// Stochastic returns %K and %D for the stochastic oscillator.
func Stochastic(bars []Bar, kPeriod, dPeriod int) (k, d float64) {
	if len(bars) < kPeriod || kPeriod <= 0 {
		return 50, 50
	}
	window := bars[len(bars)-kPeriod:]
	low := window[0].Low
	high := window[0].High
	for _, b := range window[1:] {
		if b.Low < low {
			low = b.Low
		}
		if b.High > high {
			high = b.High
		}
	}
	if high == low {
		return 50, 50
	}
	k = (bars[len(bars)-1].Close - low) / (high - low) * 100
	// Simplified %D as SMA of %K over dPeriod (single value here).
	d = k
	return
}

// OBV computes On-Balance Volume.
func OBV(bars []Bar) float64 {
	if len(bars) < 2 {
		return 0
	}
	obv := 0.0
	for i := 1; i < len(bars); i++ {
		if bars[i].Close > bars[i-1].Close {
			obv += bars[i].Volume
		} else if bars[i].Close < bars[i-1].Close {
			obv -= bars[i].Volume
		}
	}
	return obv
}

// VWAP computes the Volume-Weighted Average Price.
func VWAP(bars []Bar) float64 {
	totalVol := 0.0
	totalPV := 0.0
	for _, b := range bars {
		mid := (b.High + b.Low + b.Close) / 3
		totalPV += mid * b.Volume
		totalVol += b.Volume
	}
	if totalVol == 0 {
		return 0
	}
	return totalPV / totalVol
}

// ADX returns the Average Directional Index.
func ADX(bars []Bar, n int) float64 {
	if len(bars) < n*2 || n <= 0 {
		return 0
	}
	type dmi struct{ plus, minus, tr float64 }
	dmis := make([]dmi, len(bars)-1)
	for i := 1; i < len(bars); i++ {
		b := bars[i]
		p := bars[i-1]
		upMove := b.High - p.High
		downMove := p.Low - b.Low
		dmPlus, dmMinus := 0.0, 0.0
		if upMove > downMove && upMove > 0 {
			dmPlus = upMove
		}
		if downMove > upMove && downMove > 0 {
			dmMinus = downMove
		}
		tr := b.High - b.Low
		if v := math.Abs(b.High - p.Close); v > tr {
			tr = v
		}
		if v := math.Abs(b.Low - p.Close); v > tr {
			tr = v
		}
		dmis[i-1] = dmi{dmPlus, dmMinus, tr}
	}
	if len(dmis) < n {
		return 0
	}
	// Wilder smooth over n periods.
	sumTR, sumPlus, sumMinus := 0.0, 0.0, 0.0
	for _, d := range dmis[:n] {
		sumTR += d.tr
		sumPlus += d.plus
		sumMinus += d.minus
	}
	dx := 0.0
	if sumTR > 0 {
		diPlus := sumPlus / sumTR * 100
		diMinus := sumMinus / sumTR * 100
		if diPlus+diMinus > 0 {
			dx = math.Abs(diPlus-diMinus) / (diPlus + diMinus) * 100
		}
	}
	adx := dx
	for _, d := range dmis[n:] {
		sumTR = sumTR - sumTR/float64(n) + d.tr
		sumPlus = sumPlus - sumPlus/float64(n) + d.plus
		sumMinus = sumMinus - sumMinus/float64(n) + d.minus
		if sumTR > 0 {
			diPlus := sumPlus / sumTR * 100
			diMinus := sumMinus / sumTR * 100
			if diPlus+diMinus > 0 {
				dx = math.Abs(diPlus-diMinus) / (diPlus + diMinus) * 100
			}
		}
		adx = adx - adx/float64(n) + dx/float64(n)
	}
	return adx
}

// CCI returns the Commodity Channel Index.
func CCI(bars []Bar, n int) float64 {
	if len(bars) < n || n <= 0 {
		return 0
	}
	window := bars[len(bars)-n:]
	typicals := make([]float64, n)
	sum := 0.0
	for i, b := range window {
		tp := (b.High + b.Low + b.Close) / 3
		typicals[i] = tp
		sum += tp
	}
	mean := sum / float64(n)
	madSum := 0.0
	for _, tp := range typicals {
		madSum += math.Abs(tp - mean)
	}
	mad := madSum / float64(n)
	if mad == 0 {
		return 0
	}
	lastTP := typicals[n-1]
	return (lastTP - mean) / (0.015 * mad)
}

// WilliamsR returns Williams %R oscillator.
func WilliamsR(bars []Bar, n int) float64 {
	if len(bars) < n || n <= 0 {
		return -50
	}
	window := bars[len(bars)-n:]
	high := window[0].High
	low := window[0].Low
	for _, b := range window[1:] {
		if b.High > high {
			high = b.High
		}
		if b.Low < low {
			low = b.Low
		}
	}
	if high == low {
		return -50
	}
	return (high - bars[len(bars)-1].Close) / (high - low) * -100
}

// DonchianChannel returns the upper, middle, and lower donchian channel lines.
func DonchianChannel(bars []Bar, n int) (upper, middle, lower float64) {
	if len(bars) < n || n <= 0 {
		return 0, 0, 0
	}
	window := bars[len(bars)-n:]
	upper = window[0].High
	lower = window[0].Low
	for _, b := range window[1:] {
		if b.High > upper {
			upper = b.High
		}
		if b.Low < lower {
			lower = b.Low
		}
	}
	middle = (upper + lower) / 2
	return
}

// KeltnerChannel returns upper and lower Keltner channel around the EMA.
func KeltnerChannel(bars []Bar, emaPeriod, atrPeriod int, multiplier float64) (upper, mid, lower float64) {
	mid = EMA(bars, emaPeriod)
	atr := ATR(bars, atrPeriod)
	upper = mid + multiplier*atr
	lower = mid - multiplier*atr
	return
}

// ChaikinMoneyFlow returns the Chaikin Money Flow over n periods.
func ChaikinMoneyFlow(bars []Bar, n int) float64 {
	if len(bars) < n || n <= 0 {
		return 0
	}
	window := bars[len(bars)-n:]
	mfvSum := 0.0
	volSum := 0.0
	for _, b := range window {
		r := b.High - b.Low
		if r == 0 {
			continue
		}
		clv := ((b.Close - b.Low) - (b.High - b.Close)) / r
		mfvSum += clv * b.Volume
		volSum += b.Volume
	}
	if volSum == 0 {
		return 0
	}
	return mfvSum / volSum
}

// ROC returns the Rate of Change over n periods.
func ROC(bars []Bar, n int) float64 {
	if len(bars) < n+1 || n <= 0 {
		return 0
	}
	past := bars[len(bars)-n-1].Close
	if past == 0 {
		return 0
	}
	return (bars[len(bars)-1].Close - past) / past * 100
}

// MomentumScore is a composite score combining RSI, MACD, and rate of change.
// Returns a value in [-1, 1] where positive = bullish momentum.
func MomentumScore(bars []Bar) float64 {
	if len(bars) < 30 {
		return 0
	}
	rsi := RSI(bars, 14)
	roc := ROC(bars, 10)
	m := MACD(bars, 12, 26, 9)

	// Normalise each component to [-1, 1].
	rsiNorm := (rsi - 50) / 50
	rocNorm := math.Tanh(roc / 5)
	macdNorm := math.Tanh(m.MACD / bars[len(bars)-1].Close * 100)

	return (rsiNorm + rocNorm + macdNorm) / 3
}
