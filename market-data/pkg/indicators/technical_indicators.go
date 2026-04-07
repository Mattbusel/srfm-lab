// Package indicators provides incremental technical indicator implementations
// for SRFM market data. All indicators update one bar at a time to support
// real-time streaming.
package indicators

import (
	"math"
)

// OHLCV is a single price bar with volume.
type OHLCV struct {
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Timestamp int64 // Unix epoch seconds
}

// -----------------------------------------------------------------------------
// EMA -- Exponential Moving Average
// -----------------------------------------------------------------------------

// EMA is an incremental exponential moving average.
type EMA struct {
	span    int
	value   float64
	k       float64 // smoothing factor = 2/(span+1)
	primed  bool
	count   int
	sumSeed float64 // accumulates first span values for SMA seed
}

// NewEMA creates an EMA with the given period.
func NewEMA(span int) *EMA {
	if span < 1 {
		span = 1
	}
	return &EMA{
		span: span,
		k:    2.0 / float64(span+1),
	}
}

// Update feeds a new price into the EMA.
func (e *EMA) Update(price float64) {
	if !e.primed {
		e.sumSeed += price
		e.count++
		if e.count >= e.span {
			e.value = e.sumSeed / float64(e.span)
			e.primed = true
		}
		return
	}
	e.value = price*e.k + e.value*(1-e.k)
}

// Value returns the current EMA value, or 0 during warm-up.
func (e *EMA) Value() float64 {
	if !e.primed {
		return 0
	}
	return e.value
}

// Primed returns true once the EMA has seen enough data.
func (e *EMA) Primed() bool { return e.primed }

// -----------------------------------------------------------------------------
// MACD -- Moving Average Convergence Divergence (12/26/9)
// -----------------------------------------------------------------------------

// MACD computes the classic 12/26/9 MACD line, signal line, and histogram.
type MACD struct {
	fast      EMA
	slow      EMA
	signal    EMA
	line      float64
	histogram float64
}

// NewMACD creates a MACD with default 12/26/9 parameters.
func NewMACD() *MACD {
	return &MACD{
		fast:   *NewEMA(12),
		slow:   *NewEMA(26),
		signal: *NewEMA(9),
	}
}

// Update feeds a new closing price.
func (m *MACD) Update(bar OHLCV) {
	m.fast.Update(bar.Close)
	m.slow.Update(bar.Close)
	if m.fast.Primed() && m.slow.Primed() {
		m.line = m.fast.Value() - m.slow.Value()
		m.signal.Update(m.line)
		if m.signal.Primed() {
			m.histogram = m.line - m.signal.Value()
		}
	}
}

// Value returns the MACD line.
func (m *MACD) Value() float64 { return m.line }

// Signal returns the signal line.
func (m *MACD) Signal() float64 { return m.signal.Value() }

// Histogram returns the MACD histogram (line - signal).
func (m *MACD) Histogram() float64 { return m.histogram }

// -----------------------------------------------------------------------------
// RSI -- Relative Strength Index (14-period Wilder smoothing)
// -----------------------------------------------------------------------------

// RSI computes the 14-period Wilder RSI.
type RSI struct {
	n         int     // period (default 14)
	avgGain   float64
	avgLoss   float64
	prevClose float64
	rsi       float64
	count     int
	sumGain   float64
	sumLoss   float64
	primed    bool
}

// NewRSI creates an RSI with the given period.
func NewRSI(n int) *RSI {
	if n < 1 {
		n = 14
	}
	return &RSI{n: n}
}

// Update feeds a new bar. The RSI is computed from Close prices.
func (r *RSI) Update(bar OHLCV) {
	if r.count == 0 {
		r.prevClose = bar.Close
		r.count++
		return
	}
	change := bar.Close - r.prevClose
	r.prevClose = bar.Close

	gain, loss := 0.0, 0.0
	if change > 0 {
		gain = change
	} else {
		loss = -change
	}

	r.count++
	if !r.primed {
		r.sumGain += gain
		r.sumLoss += loss
		if r.count > r.n {
			r.avgGain = r.sumGain / float64(r.n)
			r.avgLoss = r.sumLoss / float64(r.n)
			r.primed = true
			r.computeRSI()
		}
		return
	}
	// Wilder smoothing
	r.avgGain = (r.avgGain*float64(r.n-1) + gain) / float64(r.n)
	r.avgLoss = (r.avgLoss*float64(r.n-1) + loss) / float64(r.n)
	r.computeRSI()
}

func (r *RSI) computeRSI() {
	if r.avgLoss == 0 {
		r.rsi = 100
		return
	}
	rs := r.avgGain / r.avgLoss
	r.rsi = 100 - 100/(1+rs)
}

// Value returns the current RSI (0 during warm-up).
func (r *RSI) Value() float64 {
	if !r.primed {
		return 0
	}
	return r.rsi
}

// Primed returns true once the RSI has processed enough bars.
func (r *RSI) Primed() bool { return r.primed }

// -----------------------------------------------------------------------------
// ATR -- Average True Range
// -----------------------------------------------------------------------------

// ATR computes Wilder's Average True Range.
type ATR struct {
	period    int
	atr       float64
	prevClose float64
	count     int
	sumTR     float64
	primed    bool
}

// NewATR creates an ATR with the given period.
func NewATR(period int) *ATR {
	if period < 1 {
		period = 14
	}
	return &ATR{period: period}
}

// Update feeds a new bar.
func (a *ATR) Update(bar OHLCV) {
	tr := bar.High - bar.Low
	if a.count > 0 {
		if v := math.Abs(bar.High - a.prevClose); v > tr {
			tr = v
		}
		if v := math.Abs(bar.Low - a.prevClose); v > tr {
			tr = v
		}
	}
	a.prevClose = bar.Close
	a.count++

	if !a.primed {
		a.sumTR += tr
		if a.count >= a.period {
			a.atr = a.sumTR / float64(a.period)
			a.primed = true
		}
		return
	}
	// Wilder smoothing
	a.atr = (a.atr*float64(a.period-1) + tr) / float64(a.period)
}

// Value returns the current ATR (0 during warm-up).
func (a *ATR) Value() float64 {
	if !a.primed {
		return 0
	}
	return a.atr
}

// Primed returns true once ATR has enough data.
func (a *ATR) Primed() bool { return a.primed }

// -----------------------------------------------------------------------------
// BollingerBands -- 20-period SMA +/- 2 standard deviations
// -----------------------------------------------------------------------------

// BollingerBands computes 20-period Bollinger Bands.
type BollingerBands struct {
	period    int
	prices    []float64
	sma       float64
	upper     float64
	lower     float64
	bandwidth float64
}

// NewBollingerBands creates Bollinger Bands with the given period (default 20).
func NewBollingerBands(period int) *BollingerBands {
	if period < 2 {
		period = 20
	}
	return &BollingerBands{period: period, prices: make([]float64, 0, period)}
}

// Update feeds a new bar.
func (b *BollingerBands) Update(bar OHLCV) {
	b.prices = append(b.prices, bar.Close)
	if len(b.prices) > b.period {
		b.prices = b.prices[len(b.prices)-b.period:]
	}
	if len(b.prices) < b.period {
		return
	}
	sum := 0.0
	for _, p := range b.prices {
		sum += p
	}
	b.sma = sum / float64(b.period)

	variance := 0.0
	for _, p := range b.prices {
		d := p - b.sma
		variance += d * d
	}
	std := math.Sqrt(variance / float64(b.period))
	b.upper = b.sma + 2*std
	b.lower = b.sma - 2*std
	if b.sma != 0 {
		b.bandwidth = (b.upper - b.lower) / b.sma
	}
}

// Value returns the SMA (middle band).
func (b *BollingerBands) Value() float64 { return b.sma }

// Upper returns the upper band.
func (b *BollingerBands) Upper() float64 { return b.upper }

// Lower returns the lower band.
func (b *BollingerBands) Lower() float64 { return b.lower }

// Bandwidth returns (upper-lower)/sma.
func (b *BollingerBands) Bandwidth() float64 { return b.bandwidth }

// Primed returns true once enough prices are available.
func (b *BollingerBands) Primed() bool { return len(b.prices) >= b.period }

// -----------------------------------------------------------------------------
// VWAP -- Volume Weighted Average Price (session, resets daily)
// -----------------------------------------------------------------------------

// VWAP computes the intraday VWAP; it resets when the session date changes.
type VWAP struct {
	cumPV    float64 // cumulative price * volume
	cumVol   float64 // cumulative volume
	lastDate int64   // Unix day (ts / 86400)
}

// NewVWAP creates a new VWAP accumulator.
func NewVWAP() *VWAP { return &VWAP{} }

// Update feeds a new bar and resets the session at midnight.
func (v *VWAP) Update(bar OHLCV) {
	day := bar.Timestamp / 86400
	if day != v.lastDate && v.lastDate != 0 {
		// New session -- reset.
		v.cumPV = 0
		v.cumVol = 0
	}
	v.lastDate = day
	typicalPrice := (bar.High + bar.Low + bar.Close) / 3.0
	v.cumPV += typicalPrice * bar.Volume
	v.cumVol += bar.Volume
}

// Value returns the current session VWAP (0 if no volume seen).
func (v *VWAP) Value() float64 {
	if v.cumVol == 0 {
		return 0
	}
	return v.cumPV / v.cumVol
}

// Reset resets the VWAP session manually.
func (v *VWAP) Reset() {
	v.cumPV = 0
	v.cumVol = 0
	v.lastDate = 0
}

// -----------------------------------------------------------------------------
// ADX -- Average Directional Index (14-period)
// -----------------------------------------------------------------------------

// ADX computes the 14-period Average Directional Index.
type ADX struct {
	period    int
	prevHigh  float64
	prevLow   float64
	prevClose float64
	smoothTR  float64
	smoothDMp float64 // +DM
	smoothDMn float64 // -DM
	adx       float64
	dx        float64
	count     int
	primed    bool
	dxBuf     []float64
	dxSum     float64
}

// NewADX creates an ADX with the given period.
func NewADX(period int) *ADX {
	if period < 2 {
		period = 14
	}
	return &ADX{period: period}
}

// Update feeds a new bar.
func (a *ADX) Update(bar OHLCV) {
	if a.count == 0 {
		a.prevHigh = bar.High
		a.prevLow = bar.Low
		a.prevClose = bar.Close
		a.count++
		return
	}

	// True Range
	tr := bar.High - bar.Low
	if v := math.Abs(bar.High - a.prevClose); v > tr {
		tr = v
	}
	if v := math.Abs(bar.Low - a.prevClose); v > tr {
		tr = v
	}

	// Directional Movement
	upMove := bar.High - a.prevHigh
	downMove := a.prevLow - bar.Low

	dmPlus, dmMinus := 0.0, 0.0
	if upMove > downMove && upMove > 0 {
		dmPlus = upMove
	}
	if downMove > upMove && downMove > 0 {
		dmMinus = downMove
	}

	a.prevHigh = bar.High
	a.prevLow = bar.Low
	a.prevClose = bar.Close
	a.count++

	if !a.primed {
		// Seed Wilder smoothing with first-period sum.
		a.smoothTR += tr
		a.smoothDMp += dmPlus
		a.smoothDMn += dmMinus
		if a.count > a.period {
			a.primed = true
			// Initialise ADX buffer.
			a.computeDX()
			a.dxBuf = append(a.dxBuf, a.dx)
			a.dxSum += a.dx
			if len(a.dxBuf) >= a.period {
				a.adx = a.dxSum / float64(a.period)
			}
		}
		return
	}

	// Wilder smoothing
	a.smoothTR = a.smoothTR - a.smoothTR/float64(a.period) + tr
	a.smoothDMp = a.smoothDMp - a.smoothDMp/float64(a.period) + dmPlus
	a.smoothDMn = a.smoothDMn - a.smoothDMn/float64(a.period) + dmMinus

	a.computeDX()

	// ADX = Wilder EMA of DX
	if len(a.dxBuf) < a.period {
		a.dxBuf = append(a.dxBuf, a.dx)
		a.dxSum += a.dx
		if len(a.dxBuf) == a.period {
			a.adx = a.dxSum / float64(a.period)
		}
	} else {
		a.adx = (a.adx*float64(a.period-1) + a.dx) / float64(a.period)
	}
}

func (a *ADX) computeDX() {
	if a.smoothTR == 0 {
		a.dx = 0
		return
	}
	diPlus := 100 * a.smoothDMp / a.smoothTR
	diMinus := 100 * a.smoothDMn / a.smoothTR
	diff := math.Abs(diPlus - diMinus)
	sum := diPlus + diMinus
	if sum == 0 {
		a.dx = 0
		return
	}
	a.dx = 100 * diff / sum
}

// Value returns the current ADX (0 during warm-up).
func (a *ADX) Value() float64 {
	if len(a.dxBuf) < a.period {
		return 0
	}
	return a.adx
}

// Primed returns true once ADX has enough data.
func (a *ADX) Primed() bool { return len(a.dxBuf) >= a.period }

// -----------------------------------------------------------------------------
// IndicatorSet -- named collection of indicators per symbol
// -----------------------------------------------------------------------------

// IndicatorSet manages a named set of indicators for a single symbol.
type IndicatorSet struct {
	Symbol     string
	indicators map[string]Indicator
}

// Indicator is the minimal interface every indicator implements.
type Indicator interface {
	Update(bar OHLCV)
	Value() float64
}

// NewIndicatorSet creates an empty set for symbol.
func NewIndicatorSet(symbol string) *IndicatorSet {
	return &IndicatorSet{Symbol: symbol, indicators: make(map[string]Indicator)}
}

// Register adds an indicator under a name.
func (s *IndicatorSet) Register(name string, ind Indicator) {
	s.indicators[name] = ind
}

// Update feeds all registered indicators with the new bar.
func (s *IndicatorSet) Update(bar OHLCV) {
	for _, ind := range s.indicators {
		ind.Update(bar)
	}
}

// Get returns the named indicator, or nil if not found.
func (s *IndicatorSet) Get(name string) Indicator {
	return s.indicators[name]
}

// Values returns a snapshot map of name -> current value.
func (s *IndicatorSet) Values() map[string]float64 {
	out := make(map[string]float64, len(s.indicators))
	for name, ind := range s.indicators {
		out[name] = ind.Value()
	}
	return out
}

// -----------------------------------------------------------------------------
// IndicatorBundle -- standard bundle of indicators used across SRFM strategies
// -----------------------------------------------------------------------------

// IndicatorBundle holds a complete set of standard indicators for one symbol.
type IndicatorBundle struct {
	EMA20 *EMA
	EMA50 *EMA
	EMA200 *EMA
	MACD   *MACD
	RSI    *RSI
	ATR    *ATR
	BB     *BollingerBands
	VWAP   *VWAP
	ADX    *ADX
}

// NewBundle creates an IndicatorBundle with default periods.
func NewBundle() *IndicatorBundle {
	return &IndicatorBundle{
		EMA20:  NewEMA(20),
		EMA50:  NewEMA(50),
		EMA200: NewEMA(200),
		MACD:   NewMACD(),
		RSI:    NewRSI(14),
		ATR:    NewATR(14),
		BB:     NewBollingerBands(20),
		VWAP:   NewVWAP(),
		ADX:    NewADX(14),
	}
}

// Update feeds all indicators in the bundle.
func (b *IndicatorBundle) Update(bar OHLCV) {
	b.EMA20.Update(bar.Close)
	b.EMA50.Update(bar.Close)
	b.EMA200.Update(bar.Close)
	b.MACD.Update(bar)
	b.RSI.Update(bar)
	b.ATR.Update(bar)
	b.BB.Update(bar)
	b.VWAP.Update(bar)
	b.ADX.Update(bar)
}

// Summary returns a flat map of all indicator values.
func (b *IndicatorBundle) Summary() map[string]float64 {
	return map[string]float64{
		"ema20":      b.EMA20.Value(),
		"ema50":      b.EMA50.Value(),
		"ema200":     b.EMA200.Value(),
		"macd_line":  b.MACD.Value(),
		"macd_sig":   b.MACD.Signal(),
		"macd_hist":  b.MACD.Histogram(),
		"rsi":        b.RSI.Value(),
		"atr":        b.ATR.Value(),
		"bb_upper":   b.BB.Upper(),
		"bb_lower":   b.BB.Lower(),
		"bb_bw":      b.BB.Bandwidth(),
		"vwap":       b.VWAP.Value(),
		"adx":        b.ADX.Value(),
	}
}
