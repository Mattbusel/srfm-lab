package signals

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

// Bar represents a single OHLCV bar for signal computation.
type Bar struct {
	Timestamp int64
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// Signal is the interface all signals must implement.
type Signal interface {
	Name() string
	Compute(bars []Bar) float64
	Lookback() int
}

// SignalResult holds the output of a signal computation.
type SignalResult struct {
	Name  string
	Value float64
	Valid bool
}

// ---------------------------------------------------------------------------
// Ring buffer utility
// ---------------------------------------------------------------------------

type ringBuf struct {
	data  []float64
	pos   int
	count int
	cap   int
}

func newRingBuf(capacity int) *ringBuf {
	return &ringBuf{data: make([]float64, capacity), cap: capacity}
}

func (r *ringBuf) push(v float64) {
	r.data[r.pos] = v
	r.pos = (r.pos + 1) % r.cap
	if r.count < r.cap {
		r.count++
	}
}

func (r *ringBuf) full() bool { return r.count == r.cap }

func (r *ringBuf) get(ago int) float64 {
	if ago >= r.count {
		return 0
	}
	idx := (r.pos - 1 - ago + r.cap*2) % r.cap
	return r.data[idx]
}

func (r *ringBuf) values() []float64 {
	out := make([]float64, r.count)
	for i := 0; i < r.count; i++ {
		idx := (r.pos - r.count + i + r.cap*2) % r.cap
		out[i] = r.data[idx]
	}
	return out
}

func (r *ringBuf) mean() float64 {
	if r.count == 0 {
		return 0
	}
	s := 0.0
	for i := 0; i < r.count; i++ {
		s += r.data[i]
	}
	return s / float64(r.count)
}

func (r *ringBuf) stddev() float64 {
	if r.count < 2 {
		return 0
	}
	m := r.mean()
	ss := 0.0
	for i := 0; i < r.count; i++ {
		d := r.data[i] - m
		ss += d * d
	}
	return math.Sqrt(ss / float64(r.count-1))
}

func (r *ringBuf) sum() float64 {
	s := 0.0
	for i := 0; i < r.count; i++ {
		s += r.data[i]
	}
	return s
}

func (r *ringBuf) max() float64 {
	m := -math.MaxFloat64
	for i := 0; i < r.count; i++ {
		if r.data[i] > m {
			m = r.data[i]
		}
	}
	return m
}

func (r *ringBuf) min() float64 {
	m := math.MaxFloat64
	for i := 0; i < r.count; i++ {
		if r.data[i] < m {
			m = r.data[i]
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

func closePrices(bars []Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.Close
	}
	return out
}

func returns(bars []Bar) []float64 {
	if len(bars) < 2 {
		return nil
	}
	ret := make([]float64, len(bars)-1)
	for i := 1; i < len(bars); i++ {
		if bars[i-1].Close > 0 {
			ret[i-1] = math.Log(bars[i].Close / bars[i-1].Close)
		}
	}
	return ret
}

func sma(values []float64, period int) float64 {
	if len(values) < period {
		return 0
	}
	s := 0.0
	for i := len(values) - period; i < len(values); i++ {
		s += values[i]
	}
	return s / float64(period)
}

func ema(values []float64, period int) float64 {
	if len(values) < period {
		return 0
	}
	mult := 2.0 / float64(period+1)
	val := sma(values[:period], period)
	for i := period; i < len(values); i++ {
		val = (values[i]-val)*mult + val
	}
	return val
}

func stddev(values []float64) float64 {
	n := len(values)
	if n < 2 {
		return 0
	}
	m := 0.0
	for _, v := range values {
		m += v
	}
	m /= float64(n)
	ss := 0.0
	for _, v := range values {
		d := v - m
		ss += d * d
	}
	return math.Sqrt(ss / float64(n-1))
}

func zScore(value, mean, sd float64) float64 {
	if sd == 0 {
		return 0
	}
	return (value - mean) / sd
}

func rsi(closes []float64, period int) float64 {
	if len(closes) <= period {
		return 50
	}
	avgGain := 0.0
	avgLoss := 0.0
	for i := 1; i <= period; i++ {
		change := closes[i] - closes[i-1]
		if change > 0 {
			avgGain += change
		} else {
			avgLoss -= change
		}
	}
	avgGain /= float64(period)
	avgLoss /= float64(period)
	for i := period + 1; i < len(closes); i++ {
		change := closes[i] - closes[i-1]
		gain := 0.0
		loss := 0.0
		if change > 0 {
			gain = change
		} else {
			loss = -change
		}
		avgGain = (avgGain*float64(period-1) + gain) / float64(period)
		avgLoss = (avgLoss*float64(period-1) + loss) / float64(period)
	}
	if avgLoss == 0 {
		return 100
	}
	rs := avgGain / avgLoss
	return 100 - 100/(1+rs)
}

func atr(bars []Bar, period int) float64 {
	if len(bars) < period+1 {
		return 0
	}
	trSum := 0.0
	for i := len(bars) - period; i < len(bars); i++ {
		tr := bars[i].High - bars[i].Low
		if i > 0 {
			tr = math.Max(tr, math.Max(
				math.Abs(bars[i].High-bars[i-1].Close),
				math.Abs(bars[i].Low-bars[i-1].Close),
			))
		}
		trSum += tr
	}
	return trSum / float64(period)
}

// ---------------------------------------------------------------------------
// Momentum Signals
// ---------------------------------------------------------------------------

// Momentum1M computes 1-month (21-day) price momentum.
type Momentum1M struct{}

func (s *Momentum1M) Name() string        { return "momentum_1m" }
func (s *Momentum1M) Lookback() int        { return 21 }
func (s *Momentum1M) Compute(bars []Bar) float64 {
	if len(bars) < 22 {
		return 0
	}
	old := bars[len(bars)-22].Close
	if old == 0 {
		return 0
	}
	return bars[len(bars)-1].Close/old - 1
}

// Momentum3M computes 3-month (63-day) price momentum.
type Momentum3M struct{}

func (s *Momentum3M) Name() string        { return "momentum_3m" }
func (s *Momentum3M) Lookback() int        { return 63 }
func (s *Momentum3M) Compute(bars []Bar) float64 {
	if len(bars) < 64 {
		return 0
	}
	old := bars[len(bars)-64].Close
	if old == 0 {
		return 0
	}
	return bars[len(bars)-1].Close/old - 1
}

// Momentum6M computes 6-month (126-day) price momentum.
type Momentum6M struct{}

func (s *Momentum6M) Name() string        { return "momentum_6m" }
func (s *Momentum6M) Lookback() int        { return 126 }
func (s *Momentum6M) Compute(bars []Bar) float64 {
	if len(bars) < 127 {
		return 0
	}
	old := bars[len(bars)-127].Close
	if old == 0 {
		return 0
	}
	return bars[len(bars)-1].Close/old - 1
}

// Momentum12M computes 12-month (252-day) price momentum.
type Momentum12M struct{}

func (s *Momentum12M) Name() string        { return "momentum_12m" }
func (s *Momentum12M) Lookback() int        { return 252 }
func (s *Momentum12M) Compute(bars []Bar) float64 {
	if len(bars) < 253 {
		return 0
	}
	old := bars[len(bars)-253].Close
	if old == 0 {
		return 0
	}
	return bars[len(bars)-1].Close/old - 1
}

// Momentum12_1 computes 12-1 month momentum (skip most recent month).
type Momentum12_1 struct{}

func (s *Momentum12_1) Name() string        { return "momentum_12_1" }
func (s *Momentum12_1) Lookback() int        { return 252 }
func (s *Momentum12_1) Compute(bars []Bar) float64 {
	if len(bars) < 253 {
		return 0
	}
	old := bars[len(bars)-253].Close
	recent := bars[len(bars)-22].Close
	if old == 0 {
		return 0
	}
	return recent/old - 1
}

// ---------------------------------------------------------------------------
// Mean Reversion Signals
// ---------------------------------------------------------------------------

// MeanReversion21 computes mean reversion z-score at 21-day window.
type MeanReversion21 struct{}

func (s *MeanReversion21) Name() string    { return "mean_reversion_21d" }
func (s *MeanReversion21) Lookback() int    { return 21 }
func (s *MeanReversion21) Compute(bars []Bar) float64 {
	if len(bars) < 21 {
		return 0
	}
	closes := closePrices(bars[len(bars)-21:])
	m := sma(closes, 21)
	sd := stddev(closes)
	return -zScore(bars[len(bars)-1].Close, m, sd)
}

// MeanReversion63 computes mean reversion z-score at 63-day window.
type MeanReversion63 struct{}

func (s *MeanReversion63) Name() string    { return "mean_reversion_63d" }
func (s *MeanReversion63) Lookback() int    { return 63 }
func (s *MeanReversion63) Compute(bars []Bar) float64 {
	if len(bars) < 63 {
		return 0
	}
	closes := closePrices(bars[len(bars)-63:])
	m := sma(closes, 63)
	sd := stddev(closes)
	return -zScore(bars[len(bars)-1].Close, m, sd)
}

// MeanReversion126 computes mean reversion z-score at 126-day window.
type MeanReversion126 struct{}

func (s *MeanReversion126) Name() string   { return "mean_reversion_126d" }
func (s *MeanReversion126) Lookback() int   { return 126 }
func (s *MeanReversion126) Compute(bars []Bar) float64 {
	if len(bars) < 126 {
		return 0
	}
	closes := closePrices(bars[len(bars)-126:])
	m := sma(closes, 126)
	sd := stddev(closes)
	return -zScore(bars[len(bars)-1].Close, m, sd)
}

// ---------------------------------------------------------------------------
// Volatility Signals
// ---------------------------------------------------------------------------

// RealizedVol computes 21-day realized volatility signal (low vol = positive).
type RealizedVol struct{}

func (s *RealizedVol) Name() string       { return "realized_vol" }
func (s *RealizedVol) Lookback() int       { return 22 }
func (s *RealizedVol) Compute(bars []Bar) float64 {
	ret := returns(bars)
	if len(ret) < 21 {
		return 0
	}
	recent := ret[len(ret)-21:]
	return -stddev(recent) * math.Sqrt(252) // negative = low vol is positive signal
}

// ImpliedVolProxy estimates implied vol from price range.
type ImpliedVolProxy struct{}

func (s *ImpliedVolProxy) Name() string    { return "implied_vol_proxy" }
func (s *ImpliedVolProxy) Lookback() int    { return 21 }
func (s *ImpliedVolProxy) Compute(bars []Bar) float64 {
	if len(bars) < 21 {
		return 0
	}
	// Parkinson estimator
	sum := 0.0
	for i := len(bars) - 21; i < len(bars); i++ {
		if bars[i].Low > 0 {
			hl := math.Log(bars[i].High / bars[i].Low)
			sum += hl * hl
		}
	}
	return -math.Sqrt(sum/(21*4*math.Log(2))) * math.Sqrt(252)
}

// VolOfVol computes volatility of volatility.
type VolOfVol struct{}

func (s *VolOfVol) Name() string           { return "vol_of_vol" }
func (s *VolOfVol) Lookback() int           { return 126 }
func (s *VolOfVol) Compute(bars []Bar) float64 {
	ret := returns(bars)
	if len(ret) < 126 {
		return 0
	}
	// Rolling 21-day vol, then vol of that
	vols := make([]float64, 0)
	for i := 21; i <= len(ret); i++ {
		window := ret[i-21 : i]
		vols = append(vols, stddev(window)*math.Sqrt(252))
	}
	if len(vols) < 21 {
		return 0
	}
	return -stddev(vols[len(vols)-21:]) // negative = stable vol is positive
}

// VolTrend measures whether vol is trending up or down.
type VolTrend struct{}

func (s *VolTrend) Name() string    { return "vol_trend" }
func (s *VolTrend) Lookback() int    { return 63 }
func (s *VolTrend) Compute(bars []Bar) float64 {
	ret := returns(bars)
	if len(ret) < 63 {
		return 0
	}
	shortVol := stddev(ret[len(ret)-10:]) * math.Sqrt(252)
	longVol := stddev(ret[len(ret)-63:]) * math.Sqrt(252)
	if longVol == 0 {
		return 0
	}
	return -(shortVol/longVol - 1) // negative when vol expanding
}

// ---------------------------------------------------------------------------
// Volume Signals
// ---------------------------------------------------------------------------

// RelativeVolume computes volume relative to average.
type RelativeVolume struct{}

func (s *RelativeVolume) Name() string     { return "relative_volume" }
func (s *RelativeVolume) Lookback() int     { return 21 }
func (s *RelativeVolume) Compute(bars []Bar) float64 {
	if len(bars) < 21 {
		return 0
	}
	avgVol := 0.0
	for i := len(bars) - 21; i < len(bars)-1; i++ {
		avgVol += bars[i].Volume
	}
	avgVol /= 20
	if avgVol == 0 {
		return 0
	}
	return bars[len(bars)-1].Volume/avgVol - 1
}

// VolumeTrend measures volume trend direction.
type VolumeTrend struct{}

func (s *VolumeTrend) Name() string        { return "volume_trend" }
func (s *VolumeTrend) Lookback() int        { return 63 }
func (s *VolumeTrend) Compute(bars []Bar) float64 {
	if len(bars) < 63 {
		return 0
	}
	short := 0.0
	for i := len(bars) - 5; i < len(bars); i++ {
		short += bars[i].Volume
	}
	short /= 5
	long := 0.0
	for i := len(bars) - 63; i < len(bars); i++ {
		long += bars[i].Volume
	}
	long /= 63
	if long == 0 {
		return 0
	}
	return short/long - 1
}

// VolumeBreakout detects volume breakouts.
type VolumeBreakout struct{}

func (s *VolumeBreakout) Name() string     { return "volume_breakout" }
func (s *VolumeBreakout) Lookback() int     { return 21 }
func (s *VolumeBreakout) Compute(bars []Bar) float64 {
	if len(bars) < 21 {
		return 0
	}
	vols := make([]float64, 20)
	for i := 0; i < 20; i++ {
		vols[i] = bars[len(bars)-21+i].Volume
	}
	m := sma(vols, 20)
	sd := stddev(vols)
	current := bars[len(bars)-1].Volume
	z := zScore(current, m, sd)
	// Combine with price direction
	priceChange := 0.0
	if bars[len(bars)-2].Close > 0 {
		priceChange = bars[len(bars)-1].Close/bars[len(bars)-2].Close - 1
	}
	if z > 2 && priceChange > 0 {
		return z * priceChange * 100
	} else if z > 2 && priceChange < 0 {
		return z * priceChange * 100
	}
	return 0
}

// OBVSignal computes On-Balance Volume signal.
type OBVSignal struct{}

func (s *OBVSignal) Name() string          { return "obv_signal" }
func (s *OBVSignal) Lookback() int          { return 42 }
func (s *OBVSignal) Compute(bars []Bar) float64 {
	if len(bars) < 42 {
		return 0
	}
	obv := make([]float64, len(bars))
	obv[0] = bars[0].Volume
	for i := 1; i < len(bars); i++ {
		if bars[i].Close > bars[i-1].Close {
			obv[i] = obv[i-1] + bars[i].Volume
		} else if bars[i].Close < bars[i-1].Close {
			obv[i] = obv[i-1] - bars[i].Volume
		} else {
			obv[i] = obv[i-1]
		}
	}
	obvMA := sma(obv, 21)
	current := obv[len(obv)-1]
	if obvMA == 0 {
		return 0
	}
	return (current - obvMA) / math.Abs(obvMA)
}

// ---------------------------------------------------------------------------
// Microstructure Signals
// ---------------------------------------------------------------------------

// SpreadProxy estimates bid-ask spread from OHLC.
type SpreadProxy struct{}

func (s *SpreadProxy) Name() string        { return "spread_proxy" }
func (s *SpreadProxy) Lookback() int        { return 21 }
func (s *SpreadProxy) Compute(bars []Bar) float64 {
	if len(bars) < 21 {
		return 0
	}
	// Corwin-Schultz spread estimator
	sum := 0.0
	count := 0
	for i := len(bars) - 20; i < len(bars); i++ {
		if bars[i].Low > 0 && bars[i-1].Low > 0 {
			beta := math.Log(bars[i].High/bars[i].Low)*math.Log(bars[i].High/bars[i].Low) +
				math.Log(bars[i-1].High/bars[i-1].Low)*math.Log(bars[i-1].High/bars[i-1].Low)
			gamma := math.Log(math.Max(bars[i].High, bars[i-1].High)/math.Min(bars[i].Low, bars[i-1].Low))
			gamma = gamma * gamma
			alpha := (math.Sqrt(2*beta) - math.Sqrt(beta)) / (3 - 2*math.Sqrt(2)) - math.Sqrt(gamma/(3-2*math.Sqrt(2)))
			if alpha > 0 {
				spread := 2 * (math.Exp(alpha) - 1) / (1 + math.Exp(alpha))
				sum += spread
				count++
			}
		}
	}
	if count == 0 {
		return 0
	}
	return -(sum / float64(count)) // negative = narrow spread is positive
}

// DepthProxy estimates market depth from volume and price impact.
type DepthProxy struct{}

func (s *DepthProxy) Name() string         { return "depth_proxy" }
func (s *DepthProxy) Lookback() int         { return 21 }
func (s *DepthProxy) Compute(bars []Bar) float64 {
	if len(bars) < 21 {
		return 0
	}
	// Amihud illiquidity ratio (inverse as signal)
	sum := 0.0
	count := 0
	for i := len(bars) - 21; i < len(bars); i++ {
		if bars[i].Volume > 0 && i > 0 && bars[i-1].Close > 0 {
			absRet := math.Abs(bars[i].Close/bars[i-1].Close - 1)
			sum += absRet / bars[i].Volume
			count++
		}
	}
	if count == 0 {
		return 0
	}
	amihud := sum / float64(count)
	if amihud == 0 {
		return 0
	}
	return -math.Log(amihud) // higher = more liquid = positive
}

// TradeIntensity measures trade frequency proxy.
type TradeIntensity struct{}

func (s *TradeIntensity) Name() string     { return "trade_intensity" }
func (s *TradeIntensity) Lookback() int     { return 42 }
func (s *TradeIntensity) Compute(bars []Bar) float64 {
	if len(bars) < 42 {
		return 0
	}
	// Use volume * abs(return) as proxy for trade intensity
	recent := 0.0
	for i := len(bars) - 5; i < len(bars); i++ {
		if i > 0 && bars[i-1].Close > 0 {
			absRet := math.Abs(bars[i].Close/bars[i-1].Close - 1)
			recent += bars[i].Volume * absRet
		}
	}
	recent /= 5
	hist := 0.0
	for i := len(bars) - 42; i < len(bars)-5; i++ {
		if i > 0 && bars[i-1].Close > 0 {
			absRet := math.Abs(bars[i].Close/bars[i-1].Close - 1)
			hist += bars[i].Volume * absRet
		}
	}
	hist /= 37
	if hist == 0 {
		return 0
	}
	return recent/hist - 1
}

// ---------------------------------------------------------------------------
// Technical Signals
// ---------------------------------------------------------------------------

// RSISignal computes RSI-based signal.
type RSISignal struct {
	period int
}

func NewRSISignal(period int) *RSISignal { return &RSISignal{period: period} }

func (s *RSISignal) Name() string          { return fmt.Sprintf("rsi_%d", s.period) }
func (s *RSISignal) Lookback() int          { return s.period + 1 }
func (s *RSISignal) Compute(bars []Bar) float64 {
	if len(bars) < s.period+1 {
		return 0
	}
	closes := closePrices(bars)
	r := rsi(closes, s.period)
	// Normalize: 50 = neutral, <30 oversold (buy), >70 overbought (sell)
	return (50 - r) / 50
}

// MACDSignal computes MACD-based signal.
type MACDSignal struct{}

func (s *MACDSignal) Name() string         { return "macd_signal" }
func (s *MACDSignal) Lookback() int         { return 35 }
func (s *MACDSignal) Compute(bars []Bar) float64 {
	if len(bars) < 35 {
		return 0
	}
	closes := closePrices(bars)
	fast := ema(closes, 12)
	slow := ema(closes, 26)
	macdLine := fast - slow
	// Build MACD series for signal line
	macdSeries := make([]float64, 0)
	for i := 26; i <= len(closes); i++ {
		f := ema(closes[:i], 12)
		sl := ema(closes[:i], 26)
		macdSeries = append(macdSeries, f-sl)
	}
	if len(macdSeries) < 9 {
		return 0
	}
	signalLine := ema(macdSeries, 9)
	histogram := macdLine - signalLine
	// Normalize by price
	price := bars[len(bars)-1].Close
	if price == 0 {
		return 0
	}
	return histogram / price * 100
}

// BBPctBSignal computes Bollinger Band %B signal.
type BBPctBSignal struct {
	period int
	stdMul float64
}

func NewBBPctBSignal(period int, stdMul float64) *BBPctBSignal {
	return &BBPctBSignal{period: period, stdMul: stdMul}
}

func (s *BBPctBSignal) Name() string       { return fmt.Sprintf("bb_pctb_%d", s.period) }
func (s *BBPctBSignal) Lookback() int       { return s.period }
func (s *BBPctBSignal) Compute(bars []Bar) float64 {
	if len(bars) < s.period {
		return 0
	}
	closes := closePrices(bars[len(bars)-s.period:])
	mid := sma(closes, s.period)
	sd := stddev(closes)
	upper := mid + s.stdMul*sd
	lower := mid - s.stdMul*sd
	if upper == lower {
		return 0
	}
	pctB := (bars[len(bars)-1].Close - lower) / (upper - lower)
	return 0.5 - pctB // centered: >0 near lower band (buy), <0 near upper (sell)
}

// ATRRatioSignal computes ATR ratio (current vs historical).
type ATRRatioSignal struct{}

func (s *ATRRatioSignal) Name() string     { return "atr_ratio" }
func (s *ATRRatioSignal) Lookback() int     { return 63 }
func (s *ATRRatioSignal) Compute(bars []Bar) float64 {
	if len(bars) < 63 {
		return 0
	}
	shortATR := atr(bars[len(bars)-15:], 14)
	longATR := atr(bars[len(bars)-63:], 62)
	if longATR == 0 {
		return 0
	}
	return -(shortATR/longATR - 1) // negative when vol expanding
}

// SMACrossSignal computes SMA crossover signal.
type SMACrossSignal struct {
	fast int
	slow int
}

func NewSMACrossSignal(fast, slow int) *SMACrossSignal {
	return &SMACrossSignal{fast: fast, slow: slow}
}

func (s *SMACrossSignal) Name() string     { return fmt.Sprintf("sma_cross_%d_%d", s.fast, s.slow) }
func (s *SMACrossSignal) Lookback() int     { return s.slow }
func (s *SMACrossSignal) Compute(bars []Bar) float64 {
	if len(bars) < s.slow {
		return 0
	}
	closes := closePrices(bars)
	fastMA := sma(closes, s.fast)
	slowMA := sma(closes, s.slow)
	if slowMA == 0 {
		return 0
	}
	return (fastMA - slowMA) / slowMA
}

// TrendStrength measures trend strength via linear regression.
type TrendStrength struct {
	period int
}

func NewTrendStrength(period int) *TrendStrength { return &TrendStrength{period: period} }

func (s *TrendStrength) Name() string      { return fmt.Sprintf("trend_strength_%d", s.period) }
func (s *TrendStrength) Lookback() int      { return s.period }
func (s *TrendStrength) Compute(bars []Bar) float64 {
	if len(bars) < s.period {
		return 0
	}
	closes := closePrices(bars[len(bars)-s.period:])
	n := float64(len(closes))
	sx, sy, sxy, sx2 := 0.0, 0.0, 0.0, 0.0
	for i, v := range closes {
		x := float64(i)
		sx += x
		sy += v
		sxy += x * v
		sx2 += x * x
	}
	denom := n*sx2 - sx*sx
	if denom == 0 {
		return 0
	}
	slope := (n*sxy - sx*sy) / denom
	avg := sy / n
	if avg == 0 {
		return 0
	}
	return slope * n / avg // normalized slope
}

// PriceAcceleration measures the change in momentum.
type PriceAcceleration struct{}

func (s *PriceAcceleration) Name() string  { return "price_acceleration" }
func (s *PriceAcceleration) Lookback() int  { return 42 }
func (s *PriceAcceleration) Compute(bars []Bar) float64 {
	if len(bars) < 42 {
		return 0
	}
	// Recent momentum vs previous momentum
	recentMom := 0.0
	prevMom := 0.0
	mid := len(bars) - 21
	end := len(bars) - 1
	start := len(bars) - 42
	if bars[mid].Close > 0 {
		recentMom = bars[end].Close/bars[mid].Close - 1
	}
	if bars[start].Close > 0 {
		prevMom = bars[mid].Close/bars[start].Close - 1
	}
	return recentMom - prevMom
}

// HighLowRangeSignal measures current price position within range.
type HighLowRangeSignal struct {
	period int
}

func NewHighLowRangeSignal(period int) *HighLowRangeSignal {
	return &HighLowRangeSignal{period: period}
}

func (s *HighLowRangeSignal) Name() string { return fmt.Sprintf("hl_range_%d", s.period) }
func (s *HighLowRangeSignal) Lookback() int { return s.period }
func (s *HighLowRangeSignal) Compute(bars []Bar) float64 {
	if len(bars) < s.period {
		return 0
	}
	hi := -math.MaxFloat64
	lo := math.MaxFloat64
	for i := len(bars) - s.period; i < len(bars); i++ {
		if bars[i].High > hi {
			hi = bars[i].High
		}
		if bars[i].Low < lo {
			lo = bars[i].Low
		}
	}
	if hi == lo {
		return 0
	}
	pos := (bars[len(bars)-1].Close - lo) / (hi - lo)
	return pos - 0.5 // centered
}

// GapSignal measures overnight gap effect.
type GapSignal struct{}

func (s *GapSignal) Name() string          { return "gap_signal" }
func (s *GapSignal) Lookback() int          { return 2 }
func (s *GapSignal) Compute(bars []Bar) float64 {
	if len(bars) < 2 {
		return 0
	}
	prevClose := bars[len(bars)-2].Close
	if prevClose == 0 {
		return 0
	}
	gap := bars[len(bars)-1].Open/prevClose - 1
	// Mean reversion on gaps
	return -gap
}

// VWAPDeviation measures price deviation from VWAP proxy.
type VWAPDeviation struct{}

func (s *VWAPDeviation) Name() string      { return "vwap_deviation" }
func (s *VWAPDeviation) Lookback() int      { return 1 }
func (s *VWAPDeviation) Compute(bars []Bar) float64 {
	if len(bars) < 1 {
		return 0
	}
	bar := bars[len(bars)-1]
	tp := (bar.High + bar.Low + bar.Close) / 3
	if tp == 0 {
		return 0
	}
	return (tp - bar.Close) / tp // positive when price below VWAP
}

// ---------------------------------------------------------------------------
// Cross-Asset Signals
// ---------------------------------------------------------------------------

// RelativeValueSignal computes relative value between two assets.
type RelativeValueSignal struct {
	name     string
	period   int
	asset1   string
	asset2   string
}

func NewRelativeValueSignal(asset1, asset2 string, period int) *RelativeValueSignal {
	return &RelativeValueSignal{
		name:   fmt.Sprintf("rel_value_%s_%s_%d", asset1, asset2, period),
		period: period,
		asset1: asset1,
		asset2: asset2,
	}
}

func (s *RelativeValueSignal) Name() string    { return s.name }
func (s *RelativeValueSignal) Lookback() int    { return s.period }
func (s *RelativeValueSignal) Compute(bars []Bar) float64 {
	// For single-asset computation, return z-score of price ratio
	// In practice, would need two bar series
	if len(bars) < s.period {
		return 0
	}
	closes := closePrices(bars[len(bars)-s.period:])
	m := sma(closes, s.period)
	sd := stddev(closes)
	return -zScore(bars[len(bars)-1].Close, m, sd)
}

// PairSpreadSignal computes pair spread z-score.
type PairSpreadSignal struct {
	name   string
	period int
}

func NewPairSpreadSignal(name string, period int) *PairSpreadSignal {
	return &PairSpreadSignal{name: fmt.Sprintf("pair_spread_%s_%d", name, period), period: period}
}

func (s *PairSpreadSignal) Name() string    { return s.name }
func (s *PairSpreadSignal) Lookback() int    { return s.period }
func (s *PairSpreadSignal) Compute(bars []Bar) float64 {
	if len(bars) < s.period {
		return 0
	}
	closes := closePrices(bars[len(bars)-s.period:])
	m := sma(closes, s.period)
	sd := stddev(closes)
	return -zScore(bars[len(bars)-1].Close, m, sd)
}

// BetaAdjustedSignal computes beta-adjusted return signal.
type BetaAdjustedSignal struct {
	period int
}

func NewBetaAdjustedSignal(period int) *BetaAdjustedSignal {
	return &BetaAdjustedSignal{period: period}
}

func (s *BetaAdjustedSignal) Name() string  { return fmt.Sprintf("beta_adjusted_%d", s.period) }
func (s *BetaAdjustedSignal) Lookback() int  { return s.period }
func (s *BetaAdjustedSignal) Compute(bars []Bar) float64 {
	// Without market index, use self-beta (always 1) with vol adjustment
	ret := returns(bars)
	if len(ret) < s.period {
		return 0
	}
	recent := ret[len(ret)-s.period:]
	m := 0.0
	for _, r := range recent {
		m += r
	}
	m /= float64(len(recent))
	sd := stddev(recent)
	if sd == 0 {
		return 0
	}
	return m / sd * math.Sqrt(252) // information ratio proxy
}

// ---------------------------------------------------------------------------
// SignalCombiner: IC-weighted, rank, z-score normalization
// ---------------------------------------------------------------------------

// ICWeight holds a signal name and its information coefficient weight.
type ICWeight struct {
	Name   string
	Weight float64
}

// SignalCombiner combines multiple signals into a composite.
type SignalCombiner struct {
	mu       sync.RWMutex
	weights  map[string]float64
	method   string // "ic_weighted", "rank", "equal", "zscore"
}

// NewSignalCombiner creates a new combiner with the given method.
func NewSignalCombiner(method string) *SignalCombiner {
	return &SignalCombiner{
		weights: make(map[string]float64),
		method:  method,
	}
}

// SetWeight sets the weight for a signal.
func (c *SignalCombiner) SetWeight(name string, weight float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.weights[name] = weight
}

// SetWeights sets weights from a slice of ICWeight.
func (c *SignalCombiner) SetWeights(weights []ICWeight) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, w := range weights {
		c.weights[w.Name] = w.Weight
	}
}

// Combine produces a composite signal from individual signal values.
func (c *SignalCombiner) Combine(signals map[string]float64) float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	switch c.method {
	case "ic_weighted":
		return c.icWeighted(signals)
	case "rank":
		return c.rankCombine(signals)
	case "zscore":
		return c.zscoreCombine(signals)
	default:
		return c.equalWeighted(signals)
	}
}

func (c *SignalCombiner) icWeighted(signals map[string]float64) float64 {
	totalWeight := 0.0
	weighted := 0.0
	for name, val := range signals {
		w := 1.0
		if ww, ok := c.weights[name]; ok {
			w = ww
		}
		weighted += val * w
		totalWeight += math.Abs(w)
	}
	if totalWeight == 0 {
		return 0
	}
	return weighted / totalWeight
}

func (c *SignalCombiner) equalWeighted(signals map[string]float64) float64 {
	if len(signals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range signals {
		sum += v
	}
	return sum / float64(len(signals))
}

func (c *SignalCombiner) rankCombine(signals map[string]float64) float64 {
	// Rank each signal value, then average ranks
	if len(signals) == 0 {
		return 0
	}
	type sv struct {
		name string
		val  float64
	}
	items := make([]sv, 0, len(signals))
	for k, v := range signals {
		items = append(items, sv{k, v})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].val < items[j].val
	})
	n := float64(len(items))
	sum := 0.0
	for i, item := range items {
		rank := float64(i) / (n - 1) // 0 to 1
		w := 1.0
		if ww, ok := c.weights[item.name]; ok {
			w = ww
		}
		sum += rank * w
	}
	totalW := 0.0
	for _, item := range items {
		w := 1.0
		if ww, ok := c.weights[item.name]; ok {
			w = ww
		}
		totalW += math.Abs(w)
	}
	if totalW == 0 {
		return 0
	}
	return sum/totalW - 0.5 // centered around 0
}

func (c *SignalCombiner) zscoreCombine(signals map[string]float64) float64 {
	if len(signals) == 0 {
		return 0
	}
	vals := make([]float64, 0, len(signals))
	for _, v := range signals {
		vals = append(vals, v)
	}
	m := 0.0
	for _, v := range vals {
		m += v
	}
	m /= float64(len(vals))
	sd := stddev(vals)
	sum := 0.0
	for name, v := range signals {
		z := zScore(v, m, sd)
		w := 1.0
		if ww, ok := c.weights[name]; ok {
			w = ww
		}
		sum += z * w
	}
	totalW := 0.0
	for name := range signals {
		w := 1.0
		if ww, ok := c.weights[name]; ok {
			w = ww
		}
		totalW += math.Abs(w)
	}
	if totalW == 0 {
		return 0
	}
	return sum / totalW
}

// ---------------------------------------------------------------------------
// SignalEvaluator: rolling IC, IC IR, hit rate, turnover
// ---------------------------------------------------------------------------

// SignalEvaluation holds evaluation metrics for a signal.
type SignalEvaluation struct {
	Name       string
	IC         float64 // Information Coefficient (rank correlation)
	ICIR       float64 // IC Information Ratio (mean IC / std IC)
	HitRate    float64 // fraction of correct sign predictions
	AvgReturn  float64 // average return of top quintile
	Turnover   float64 // average signal turnover
	ICHistory  []float64
}

// SignalEvaluator evaluates signal quality.
type SignalEvaluator struct {
	mu         sync.RWMutex
	history    map[string][]float64
	retHistory []float64
	window     int
}

// NewSignalEvaluator creates a new evaluator.
func NewSignalEvaluator(window int) *SignalEvaluator {
	return &SignalEvaluator{
		history: make(map[string][]float64),
		window:  window,
	}
}

// Record records signal values and subsequent return.
func (e *SignalEvaluator) Record(signals map[string]float64, futureReturn float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for name, val := range signals {
		e.history[name] = append(e.history[name], val)
	}
	e.retHistory = append(e.retHistory, futureReturn)
}

// Evaluate computes evaluation metrics for all signals.
func (e *SignalEvaluator) Evaluate() map[string]SignalEvaluation {
	e.mu.RLock()
	defer e.mu.RUnlock()
	result := make(map[string]SignalEvaluation)
	for name, sigHist := range e.history {
		n := len(sigHist)
		if n < 10 || n != len(e.retHistory) {
			continue
		}
		eval := SignalEvaluation{Name: name}
		// Rolling IC
		step := e.window
		if step > n {
			step = n
		}
		var icVals []float64
		for start := 0; start+step <= n; start += step {
			end := start + step
			ic := rankCorrelation(sigHist[start:end], e.retHistory[start:end])
			icVals = append(icVals, ic)
		}
		eval.ICHistory = icVals
		if len(icVals) > 0 {
			icMean := 0.0
			for _, v := range icVals {
				icMean += v
			}
			icMean /= float64(len(icVals))
			eval.IC = icMean
			icSD := 0.0
			for _, v := range icVals {
				d := v - icMean
				icSD += d * d
			}
			if len(icVals) > 1 {
				icSD = math.Sqrt(icSD / float64(len(icVals)-1))
			}
			if icSD > 0 {
				eval.ICIR = icMean / icSD
			}
		}
		// Hit rate: fraction where sign(signal) == sign(return)
		hits := 0
		for i := 0; i < n; i++ {
			if (sigHist[i] > 0 && e.retHistory[i] > 0) || (sigHist[i] < 0 && e.retHistory[i] < 0) {
				hits++
			}
		}
		eval.HitRate = float64(hits) / float64(n)
		// Turnover
		if n > 1 {
			turnoverSum := 0.0
			for i := 1; i < n; i++ {
				turnoverSum += math.Abs(sigHist[i] - sigHist[i-1])
			}
			eval.Turnover = turnoverSum / float64(n-1)
		}
		result[name] = eval
	}
	return result
}

// rankCorrelation computes Spearman rank correlation.
func rankCorrelation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n < 3 {
		return 0
	}
	rx := ranks(x)
	ry := ranks(y)
	mx, my := 0.0, 0.0
	for i := 0; i < n; i++ {
		mx += rx[i]
		my += ry[i]
	}
	mx /= float64(n)
	my /= float64(n)
	cov, vx, vy := 0.0, 0.0, 0.0
	for i := 0; i < n; i++ {
		dx := rx[i] - mx
		dy := ry[i] - my
		cov += dx * dy
		vx += dx * dx
		vy += dy * dy
	}
	denom := math.Sqrt(vx * vy)
	if denom == 0 {
		return 0
	}
	return cov / denom
}

func ranks(vals []float64) []float64 {
	n := len(vals)
	type iv struct {
		idx int
		val float64
	}
	items := make([]iv, n)
	for i, v := range vals {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].val < items[j].val
	})
	r := make([]float64, n)
	for i, item := range items {
		r[item.idx] = float64(i + 1)
	}
	return r
}

// ---------------------------------------------------------------------------
// SignalDecay: exponential decay estimation
// ---------------------------------------------------------------------------

// DecayResult holds decay analysis for a signal.
type DecayResult struct {
	Name       string
	HalfLife   float64
	DecayRate  float64
	ICAtLag    map[int]float64
	ShouldRetire bool
}

// SignalDecay estimates signal decay and half-life.
type SignalDecay struct {
	retireThreshold float64
}

// NewSignalDecay creates a decay analyzer.
func NewSignalDecay(retireThreshold float64) *SignalDecay {
	return &SignalDecay{retireThreshold: retireThreshold}
}

// Analyze computes decay characteristics for a signal.
func (d *SignalDecay) Analyze(signalValues []float64, futureReturns []float64, maxLag int) DecayResult {
	result := DecayResult{
		ICAtLag: make(map[int]float64),
	}
	n := len(signalValues)
	if n < maxLag+10 || len(futureReturns) < n {
		return result
	}
	// Compute IC at different lags
	for lag := 1; lag <= maxLag; lag++ {
		if n-lag < 10 {
			break
		}
		sig := signalValues[:n-lag]
		ret := futureReturns[lag:]
		minLen := len(sig)
		if len(ret) < minLen {
			minLen = len(ret)
		}
		ic := rankCorrelation(sig[:minLen], ret[:minLen])
		result.ICAtLag[lag] = ic
	}
	// Estimate decay rate via log-linear regression
	var xs, ys []float64
	for lag := 1; lag <= maxLag; lag++ {
		ic, ok := result.ICAtLag[lag]
		if !ok || ic <= 0 {
			continue
		}
		xs = append(xs, float64(lag))
		ys = append(ys, math.Log(ic))
	}
	if len(xs) >= 3 {
		// Linear regression of log(IC) on lag
		nf := float64(len(xs))
		sx, sy, sxy, sx2 := 0.0, 0.0, 0.0, 0.0
		for i := range xs {
			sx += xs[i]
			sy += ys[i]
			sxy += xs[i] * ys[i]
			sx2 += xs[i] * xs[i]
		}
		denom := nf*sx2 - sx*sx
		if denom != 0 {
			slope := (nf*sxy - sx*sy) / denom
			result.DecayRate = -slope
			if slope != 0 {
				result.HalfLife = math.Log(2) / math.Abs(slope)
			}
		}
	}
	// Check if should retire
	if ic1, ok := result.ICAtLag[1]; ok {
		result.ShouldRetire = math.Abs(ic1) < d.retireThreshold
	}
	return result
}

// ---------------------------------------------------------------------------
// SignalLibrary: register, list, evaluate, rank signals
// ---------------------------------------------------------------------------

// SignalLibrary manages a collection of signals.
type SignalLibrary struct {
	mu      sync.RWMutex
	signals map[string]Signal
	order   []string
}

// NewSignalLibrary creates a new signal library.
func NewSignalLibrary() *SignalLibrary {
	return &SignalLibrary{
		signals: make(map[string]Signal),
	}
}

// Register adds a signal to the library.
func (l *SignalLibrary) Register(s Signal) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.signals[s.Name()] = s
	l.order = append(l.order, s.Name())
}

// List returns all registered signal names.
func (l *SignalLibrary) List() []string {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]string, len(l.order))
	copy(out, l.order)
	return out
}

// Get retrieves a signal by name.
func (l *SignalLibrary) Get(name string) (Signal, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	s, ok := l.signals[name]
	return s, ok
}

// ComputeAll computes all signals for given bars.
func (l *SignalLibrary) ComputeAll(bars []Bar) map[string]float64 {
	l.mu.RLock()
	defer l.mu.RUnlock()
	result := make(map[string]float64, len(l.signals))
	for name, sig := range l.signals {
		if len(bars) >= sig.Lookback() {
			result[name] = sig.Compute(bars)
		}
	}
	return result
}

// ComputeSubset computes specific signals.
func (l *SignalLibrary) ComputeSubset(bars []Bar, names []string) map[string]float64 {
	l.mu.RLock()
	defer l.mu.RUnlock()
	result := make(map[string]float64, len(names))
	for _, name := range names {
		if sig, ok := l.signals[name]; ok {
			if len(bars) >= sig.Lookback() {
				result[name] = sig.Compute(bars)
			}
		}
	}
	return result
}

// RankSignals ranks signals by their IC from an evaluator.
func (l *SignalLibrary) RankSignals(evals map[string]SignalEvaluation) []SignalEvaluation {
	ranked := make([]SignalEvaluation, 0, len(evals))
	for _, e := range evals {
		ranked = append(ranked, e)
	}
	sort.Slice(ranked, func(i, j int) bool {
		return math.Abs(ranked[i].IC) > math.Abs(ranked[j].IC)
	})
	return ranked
}

// MaxLookback returns the maximum lookback across all signals.
func (l *SignalLibrary) MaxLookback() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	maxLB := 0
	for _, sig := range l.signals {
		if sig.Lookback() > maxLB {
			maxLB = sig.Lookback()
		}
	}
	return maxLB
}

// Count returns the number of registered signals.
func (l *SignalLibrary) Count() int {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.signals)
}

// RegisterDefaults registers the standard signal set.
func (l *SignalLibrary) RegisterDefaults() {
	// Momentum
	l.Register(&Momentum1M{})
	l.Register(&Momentum3M{})
	l.Register(&Momentum6M{})
	l.Register(&Momentum12M{})
	l.Register(&Momentum12_1{})
	// Mean reversion
	l.Register(&MeanReversion21{})
	l.Register(&MeanReversion63{})
	l.Register(&MeanReversion126{})
	// Volatility
	l.Register(&RealizedVol{})
	l.Register(&ImpliedVolProxy{})
	l.Register(&VolOfVol{})
	l.Register(&VolTrend{})
	// Volume
	l.Register(&RelativeVolume{})
	l.Register(&VolumeTrend{})
	l.Register(&VolumeBreakout{})
	l.Register(&OBVSignal{})
	// Microstructure
	l.Register(&SpreadProxy{})
	l.Register(&DepthProxy{})
	l.Register(&TradeIntensity{})
	// Technical
	l.Register(NewRSISignal(14))
	l.Register(&MACDSignal{})
	l.Register(NewBBPctBSignal(20, 2.0))
	l.Register(&ATRRatioSignal{})
	l.Register(NewSMACrossSignal(10, 50))
	l.Register(NewSMACrossSignal(50, 200))
	l.Register(NewTrendStrength(21))
	l.Register(NewTrendStrength(63))
	l.Register(&PriceAcceleration{})
	l.Register(NewHighLowRangeSignal(21))
	l.Register(NewHighLowRangeSignal(63))
	l.Register(&GapSignal{})
	l.Register(&VWAPDeviation{})
	// Cross-asset
	l.Register(NewBetaAdjustedSignal(63))
}

// ---------------------------------------------------------------------------
// Additional Momentum Signals
// ---------------------------------------------------------------------------

// Momentum1W computes 1-week (5-day) price momentum.
type Momentum1W struct{}

func (s *Momentum1W) Name() string     { return "momentum_1w" }
func (s *Momentum1W) Lookback() int     { return 5 }
func (s *Momentum1W) Compute(bars []Bar) float64 {
	if len(bars) < 6 {
		return 0
	}
	old := bars[len(bars)-6].Close
	if old == 0 {
		return 0
	}
	return bars[len(bars)-1].Close/old - 1
}

// Momentum2W computes 2-week (10-day) price momentum.
type Momentum2W struct{}

func (s *Momentum2W) Name() string     { return "momentum_2w" }
func (s *Momentum2W) Lookback() int     { return 10 }
func (s *Momentum2W) Compute(bars []Bar) float64 {
	if len(bars) < 11 {
		return 0
	}
	old := bars[len(bars)-11].Close
	if old == 0 {
		return 0
	}
	return bars[len(bars)-1].Close/old - 1
}

// ---------------------------------------------------------------------------
// Additional Technical Signals
// ---------------------------------------------------------------------------

// DualMACross computes dual moving average crossover signal.
type DualMACross struct {
	fast int
	slow int
}

func NewDualMACross(fast, slow int) *DualMACross {
	return &DualMACross{fast: fast, slow: slow}
}

func (s *DualMACross) Name() string     { return fmt.Sprintf("dual_ma_%d_%d", s.fast, s.slow) }
func (s *DualMACross) Lookback() int     { return s.slow + 1 }
func (s *DualMACross) Compute(bars []Bar) float64 {
	if len(bars) < s.slow+1 {
		return 0
	}
	closes := closePrices(bars)
	// Current crossover
	fastNow := ema(closes, s.fast)
	slowNow := ema(closes, s.slow)
	// Previous
	fastPrev := ema(closes[:len(closes)-1], s.fast)
	slowPrev := ema(closes[:len(closes)-1], s.slow)
	if slowNow == 0 {
		return 0
	}
	// Crossover signal
	crossUp := fastPrev <= slowPrev && fastNow > slowNow
	crossDown := fastPrev >= slowPrev && fastNow < slowNow
	if crossUp {
		return 1.0
	}
	if crossDown {
		return -1.0
	}
	return (fastNow - slowNow) / slowNow
}

// VolatilityBreakout detects volatility breakouts.
type VolatilityBreakout struct {
	period int
	mult   float64
}

func NewVolatilityBreakout(period int, mult float64) *VolatilityBreakout {
	return &VolatilityBreakout{period: period, mult: mult}
}

func (s *VolatilityBreakout) Name() string  { return fmt.Sprintf("vol_breakout_%d", s.period) }
func (s *VolatilityBreakout) Lookback() int  { return s.period + 1 }
func (s *VolatilityBreakout) Compute(bars []Bar) float64 {
	if len(bars) < s.period+1 {
		return 0
	}
	ret := returns(bars[len(bars)-s.period-1:])
	if len(ret) < s.period {
		return 0
	}
	m := mean(ret)
	sd := stddev(ret)
	current := ret[len(ret)-1]
	if sd == 0 {
		return 0
	}
	z := (current - m) / sd
	if z > s.mult {
		return z - s.mult // positive breakout
	} else if z < -s.mult {
		return z + s.mult // negative breakout
	}
	return 0
}

// MeanReversionBB computes mean reversion based on Bollinger Band position.
type MeanReversionBB struct {
	period int
}

func NewMeanReversionBB(period int) *MeanReversionBB {
	return &MeanReversionBB{period: period}
}

func (s *MeanReversionBB) Name() string    { return fmt.Sprintf("mr_bb_%d", s.period) }
func (s *MeanReversionBB) Lookback() int    { return s.period }
func (s *MeanReversionBB) Compute(bars []Bar) float64 {
	if len(bars) < s.period {
		return 0
	}
	closes := closePrices(bars[len(bars)-s.period:])
	m := sma(closes, s.period)
	sd := stddev(closes)
	if sd == 0 {
		return 0
	}
	z := (bars[len(bars)-1].Close - m) / sd
	return -z // buy when below mean, sell when above
}

// PriceRangePosition measures price position within N-day range.
type PriceRangePosition struct {
	period int
}

func NewPriceRangePosition(period int) *PriceRangePosition {
	return &PriceRangePosition{period: period}
}

func (s *PriceRangePosition) Name() string  { return fmt.Sprintf("range_pos_%d", s.period) }
func (s *PriceRangePosition) Lookback() int  { return s.period }
func (s *PriceRangePosition) Compute(bars []Bar) float64 {
	if len(bars) < s.period {
		return 0
	}
	hi := -math.MaxFloat64
	lo := math.MaxFloat64
	for i := len(bars) - s.period; i < len(bars); i++ {
		if bars[i].High > hi {
			hi = bars[i].High
		}
		if bars[i].Low < lo {
			lo = bars[i].Low
		}
	}
	if hi == lo {
		return 0
	}
	return (bars[len(bars)-1].Close-lo)/(hi-lo)*2 - 1 // -1 to 1
}

// VolumeRSI applies RSI to volume series.
type VolumeRSI struct {
	period int
}

func NewVolumeRSI(period int) *VolumeRSI {
	return &VolumeRSI{period: period}
}

func (s *VolumeRSI) Name() string          { return fmt.Sprintf("vol_rsi_%d", s.period) }
func (s *VolumeRSI) Lookback() int          { return s.period + 1 }
func (s *VolumeRSI) Compute(bars []Bar) float64 {
	if len(bars) < s.period+1 {
		return 0
	}
	vols := make([]float64, len(bars))
	for i, b := range bars {
		vols[i] = b.Volume
	}
	r := rsi(vols, s.period)
	return (50 - r) / 50
}

// EfficiencyRatio measures price efficiency (Kaufman ER).
type EfficiencyRatio struct {
	period int
}

func NewEfficiencyRatio(period int) *EfficiencyRatio {
	return &EfficiencyRatio{period: period}
}

func (s *EfficiencyRatio) Name() string    { return fmt.Sprintf("efficiency_%d", s.period) }
func (s *EfficiencyRatio) Lookback() int    { return s.period }
func (s *EfficiencyRatio) Compute(bars []Bar) float64 {
	if len(bars) < s.period+1 {
		return 0
	}
	direction := math.Abs(bars[len(bars)-1].Close - bars[len(bars)-s.period-1].Close)
	volatility := 0.0
	for i := len(bars) - s.period; i < len(bars); i++ {
		volatility += math.Abs(bars[i].Close - bars[i-1].Close)
	}
	if volatility == 0 {
		return 0
	}
	er := direction / volatility
	// Signed by direction
	if bars[len(bars)-1].Close > bars[len(bars)-s.period-1].Close {
		return er
	}
	return -er
}

// ---------------------------------------------------------------------------
// SignalPortfolio: manage signal allocations
// ---------------------------------------------------------------------------

// SignalAllocation represents a signal's allocation in the portfolio.
type SignalAllocation struct {
	Name   string  `json:"name"`
	Weight float64 `json:"weight"`
	Active bool    `json:"active"`
}

// SignalPortfolio manages a portfolio of signals with allocation weights.
type SignalPortfolio struct {
	mu          sync.RWMutex
	allocations map[string]SignalAllocation
	maxSignals  int
}

// NewSignalPortfolio creates a new signal portfolio.
func NewSignalPortfolio(maxSignals int) *SignalPortfolio {
	return &SignalPortfolio{
		allocations: make(map[string]SignalAllocation),
		maxSignals:  maxSignals,
	}
}

// SetAllocation sets the allocation for a signal.
func (sp *SignalPortfolio) SetAllocation(name string, weight float64, active bool) {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	sp.allocations[name] = SignalAllocation{Name: name, Weight: weight, Active: active}
}

// GetActive returns all active signal allocations.
func (sp *SignalPortfolio) GetActive() []SignalAllocation {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	var active []SignalAllocation
	for _, a := range sp.allocations {
		if a.Active {
			active = append(active, a)
		}
	}
	return active
}

// CompositeSignal computes the weighted composite from individual signals.
func (sp *SignalPortfolio) CompositeSignal(values map[string]float64) float64 {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	totalWeight := 0.0
	weighted := 0.0
	for name, alloc := range sp.allocations {
		if !alloc.Active {
			continue
		}
		val, ok := values[name]
		if !ok {
			continue
		}
		weighted += val * alloc.Weight
		totalWeight += math.Abs(alloc.Weight)
	}
	if totalWeight == 0 {
		return 0
	}
	return weighted / totalWeight
}

// Count returns total signals in the portfolio.
func (sp *SignalPortfolio) Count() int {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	return len(sp.allocations)
}

// ActiveCount returns number of active signals.
func (sp *SignalPortfolio) ActiveCount() int {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	count := 0
	for _, a := range sp.allocations {
		if a.Active {
			count++
		}
	}
	return count
}

// NormalizeWeights normalizes active weights to sum to 1.
func (sp *SignalPortfolio) NormalizeWeights() {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	totalWeight := 0.0
	for _, a := range sp.allocations {
		if a.Active {
			totalWeight += math.Abs(a.Weight)
		}
	}
	if totalWeight == 0 {
		return
	}
	for name, a := range sp.allocations {
		if a.Active {
			a.Weight = a.Weight / totalWeight
			sp.allocations[name] = a
		}
	}
}
