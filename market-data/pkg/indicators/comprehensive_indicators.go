package indicators

import (
	"fmt"
	"math"
	"sync"
)

// Bar represents a single OHLCV bar.
type Bar struct {
	Timestamp int64
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// ---------------------------------------------------------------------------
// Result structs
// ---------------------------------------------------------------------------

// SMAResult holds simple moving average output.
type SMAResult struct {
	Value float64
	Ready bool
}

// EMAResult holds exponential moving average output.
type EMAResult struct {
	Value float64
	Ready bool
}

// DEMAResult holds double exponential moving average output.
type DEMAResult struct {
	Value float64
	Ready bool
}

// TEMAResult holds triple exponential moving average output.
type TEMAResult struct {
	Value float64
	Ready bool
}

// WMAResult holds weighted moving average output.
type WMAResult struct {
	Value float64
	Ready bool
}

// KAMAResult holds Kaufman adaptive moving average output.
type KAMAResult struct {
	Value float64
	Ready bool
}

// HMAResult holds Hull moving average output.
type HMAResult struct {
	Value float64
	Ready bool
}

// VWMAResult holds volume-weighted moving average output.
type VWMAResult struct {
	Value float64
	Ready bool
}

// IchimokuResult holds all five Ichimoku lines.
type IchimokuResult struct {
	Tenkan  float64
	Kijun   float64
	SenkouA float64
	SenkouB float64
	Chikou  float64
	Ready   bool
}

// SupertrendResult holds supertrend output.
type SupertrendResult struct {
	Value     float64
	Direction int // 1 = up, -1 = down
	Ready     bool
}

// ParabolicSARResult holds Parabolic SAR output.
type ParabolicSARResult struct {
	Value float64
	Trend int // 1 = bullish, -1 = bearish
	Ready bool
}

// RSIResult holds RSI output.
type RSIResult struct {
	Value float64
	Ready bool
}

// StochasticResult holds stochastic oscillator output.
type StochasticResult struct {
	K     float64
	D     float64
	Ready bool
}

// MACDResult holds MACD output.
type MACDResult struct {
	Line      float64
	Signal    float64
	Histogram float64
	Ready     bool
}

// CCIResult holds CCI output.
type CCIResult struct {
	Value float64
	Ready bool
}

// WilliamsRResult holds Williams %R output.
type WilliamsRResult struct {
	Value float64
	Ready bool
}

// MFIResult holds Money Flow Index output.
type MFIResult struct {
	Value float64
	Ready bool
}

// ROCResult holds Rate of Change output.
type ROCResult struct {
	Value float64
	Ready bool
}

// MomentumResult holds momentum indicator output.
type MomentumResult struct {
	Value float64
	Ready bool
}

// UltimateOscillatorResult holds Ultimate Oscillator output.
type UltimateOscillatorResult struct {
	Value float64
	Ready bool
}

// StochRSIResult holds Stochastic RSI output.
type StochRSIResult struct {
	K     float64
	D     float64
	Ready bool
}

// ConnorsRSIResult holds Connors RSI output.
type ConnorsRSIResult struct {
	Value float64
	Ready bool
}

// KSTResult holds Know Sure Thing output.
type KSTResult struct {
	Value  float64
	Signal float64
	Ready  bool
}

// BollingerResult holds Bollinger Bands output.
type BollingerResult struct {
	Upper     float64
	Mid       float64
	Lower     float64
	PctB      float64
	Bandwidth float64
	Ready     bool
}

// ATRResult holds Average True Range output.
type ATRResult struct {
	Value float64
	Ready bool
}

// KeltnerResult holds Keltner Channel output.
type KeltnerResult struct {
	Upper  float64
	Mid    float64
	Lower  float64
	Ready  bool
}

// DonchianResult holds Donchian Channel output.
type DonchianResult struct {
	Upper float64
	Mid   float64
	Lower float64
	Ready bool
}

// StdDevResult holds standard deviation output.
type StdDevResult struct {
	Value float64
	Ready bool
}

// OBVResult holds On-Balance Volume output.
type OBVResult struct {
	Value float64
	Ready bool
}

// VWAPResult holds VWAP output.
type VWAPResult struct {
	Value float64
	Ready bool
}

// ADResult holds Accumulation/Distribution output.
type ADResult struct {
	Value float64
	Ready bool
}

// ChaikinMFResult holds Chaikin Money Flow output.
type ChaikinMFResult struct {
	Value float64
	Ready bool
}

// ForceIndexResult holds Force Index output.
type ForceIndexResult struct {
	Value float64
	Ready bool
}

// ElderRayResult holds Elder Ray output.
type ElderRayResult struct {
	BullPower float64
	BearPower float64
	Ready     bool
}

// ---------------------------------------------------------------------------
// Ring buffer utility
// ---------------------------------------------------------------------------

type ringBuffer struct {
	data  []float64
	pos   int
	count int
	cap   int
}

func newRingBuffer(capacity int) *ringBuffer {
	return &ringBuffer{
		data: make([]float64, capacity),
		cap:  capacity,
	}
}

func (r *ringBuffer) push(v float64) {
	r.data[r.pos] = v
	r.pos = (r.pos + 1) % r.cap
	if r.count < r.cap {
		r.count++
	}
}

func (r *ringBuffer) full() bool {
	return r.count == r.cap
}

func (r *ringBuffer) get(ago int) float64 {
	if ago >= r.count {
		return 0
	}
	idx := (r.pos - 1 - ago + r.cap*2) % r.cap
	return r.data[idx]
}

func (r *ringBuffer) sum() float64 {
	s := 0.0
	for i := 0; i < r.count; i++ {
		s += r.data[i]
	}
	return s
}

func (r *ringBuffer) max() float64 {
	if r.count == 0 {
		return 0
	}
	m := -math.MaxFloat64
	start := 0
	if r.count < r.cap {
		start = 0
	}
	for i := 0; i < r.count; i++ {
		idx := (start + i) % r.cap
		if r.data[idx] > m {
			m = r.data[idx]
		}
	}
	return m
}

func (r *ringBuffer) min() float64 {
	if r.count == 0 {
		return 0
	}
	m := math.MaxFloat64
	for i := 0; i < r.count; i++ {
		idx := i % r.cap
		if r.data[idx] < m {
			m = r.data[idx]
		}
	}
	return m
}

func (r *ringBuffer) mean() float64 {
	if r.count == 0 {
		return 0
	}
	return r.sum() / float64(r.count)
}

func (r *ringBuffer) values() []float64 {
	out := make([]float64, r.count)
	for i := 0; i < r.count; i++ {
		idx := (r.pos - r.count + i + r.cap*2) % r.cap
		out[i] = r.data[idx]
	}
	return out
}

func (r *ringBuffer) stddev() float64 {
	if r.count < 2 {
		return 0
	}
	m := r.mean()
	ss := 0.0
	for i := 0; i < r.count; i++ {
		idx := i % r.cap
		d := r.data[idx] - m
		ss += d * d
	}
	return math.Sqrt(ss / float64(r.count))
}

// ---------------------------------------------------------------------------
// SMA - Simple Moving Average
// ---------------------------------------------------------------------------

type SMA struct {
	period int
	buf    *ringBuffer
}

func NewSMA(period int) *SMA {
	return &SMA{period: period, buf: newRingBuffer(period)}
}

func (s *SMA) Update(price float64) SMAResult {
	s.buf.push(price)
	if !s.buf.full() {
		return SMAResult{Ready: false}
	}
	return SMAResult{Value: s.buf.mean(), Ready: true}
}

func (s *SMA) Period() int { return s.period }

// ---------------------------------------------------------------------------
// EMA - Exponential Moving Average
// ---------------------------------------------------------------------------

type EMA struct {
	period int
	mult   float64
	value  float64
	count  int
	sum    float64
}

func NewEMA(period int) *EMA {
	return &EMA{
		period: period,
		mult:   2.0 / float64(period+1),
	}
}

func (e *EMA) Update(price float64) EMAResult {
	e.count++
	if e.count <= e.period {
		e.sum += price
		if e.count == e.period {
			e.value = e.sum / float64(e.period)
			return EMAResult{Value: e.value, Ready: true}
		}
		return EMAResult{Ready: false}
	}
	e.value = (price-e.value)*e.mult + e.value
	return EMAResult{Value: e.value, Ready: true}
}

func (e *EMA) Value() float64 { return e.value }
func (e *EMA) Ready() bool    { return e.count >= e.period }

// ---------------------------------------------------------------------------
// DEMA - Double Exponential Moving Average
// ---------------------------------------------------------------------------

type DEMA struct {
	period int
	ema1   *EMA
	ema2   *EMA
}

func NewDEMA(period int) *DEMA {
	return &DEMA{period: period, ema1: NewEMA(period), ema2: NewEMA(period)}
}

func (d *DEMA) Update(price float64) DEMAResult {
	r1 := d.ema1.Update(price)
	if !r1.Ready {
		return DEMAResult{Ready: false}
	}
	r2 := d.ema2.Update(r1.Value)
	if !r2.Ready {
		return DEMAResult{Ready: false}
	}
	v := 2*r1.Value - r2.Value
	return DEMAResult{Value: v, Ready: true}
}

// ---------------------------------------------------------------------------
// TEMA - Triple Exponential Moving Average
// ---------------------------------------------------------------------------

type TEMA struct {
	period int
	ema1   *EMA
	ema2   *EMA
	ema3   *EMA
}

func NewTEMA(period int) *TEMA {
	return &TEMA{period: period, ema1: NewEMA(period), ema2: NewEMA(period), ema3: NewEMA(period)}
}

func (t *TEMA) Update(price float64) TEMAResult {
	r1 := t.ema1.Update(price)
	if !r1.Ready {
		return TEMAResult{Ready: false}
	}
	r2 := t.ema2.Update(r1.Value)
	if !r2.Ready {
		return TEMAResult{Ready: false}
	}
	r3 := t.ema3.Update(r2.Value)
	if !r3.Ready {
		return TEMAResult{Ready: false}
	}
	v := 3*r1.Value - 3*r2.Value + r3.Value
	return TEMAResult{Value: v, Ready: true}
}

// ---------------------------------------------------------------------------
// WMA - Weighted Moving Average
// ---------------------------------------------------------------------------

type WMA struct {
	period int
	buf    *ringBuffer
	denom  float64
}

func NewWMA(period int) *WMA {
	d := 0.0
	for i := 1; i <= period; i++ {
		d += float64(i)
	}
	return &WMA{period: period, buf: newRingBuffer(period), denom: d}
}

func (w *WMA) Update(price float64) WMAResult {
	w.buf.push(price)
	if !w.buf.full() {
		return WMAResult{Ready: false}
	}
	vals := w.buf.values()
	num := 0.0
	for i, v := range vals {
		num += float64(i+1) * v
	}
	return WMAResult{Value: num / w.denom, Ready: true}
}

// ---------------------------------------------------------------------------
// KAMA - Kaufman Adaptive Moving Average
// ---------------------------------------------------------------------------

type KAMA struct {
	period   int
	fastSC   float64
	slowSC   float64
	buf      *ringBuffer
	value    float64
	ready    bool
	primed   bool
}

func NewKAMA(period, fast, slow int) *KAMA {
	return &KAMA{
		period: period,
		fastSC: 2.0 / float64(fast+1),
		slowSC: 2.0 / float64(slow+1),
		buf:    newRingBuffer(period + 1),
	}
}

func (k *KAMA) Update(price float64) KAMAResult {
	k.buf.push(price)
	if k.buf.count < k.period+1 {
		return KAMAResult{Ready: false}
	}
	if !k.primed {
		k.value = price
		k.primed = true
	}
	direction := math.Abs(price - k.buf.get(k.period))
	volatility := 0.0
	for i := 0; i < k.period; i++ {
		volatility += math.Abs(k.buf.get(i) - k.buf.get(i+1))
	}
	var er float64
	if volatility > 0 {
		er = direction / volatility
	}
	sc := er*(k.fastSC-k.slowSC) + k.slowSC
	sc = sc * sc
	k.value = k.value + sc*(price-k.value)
	k.ready = true
	return KAMAResult{Value: k.value, Ready: true}
}

// ---------------------------------------------------------------------------
// HMA - Hull Moving Average
// ---------------------------------------------------------------------------

type HMA struct {
	period int
	wmaH   *WMA
	wmaF   *WMA
	wmaS   *WMA
	buf    *ringBuffer
	sqrtP  int
}

func NewHMA(period int) *HMA {
	half := period / 2
	sqrtP := int(math.Sqrt(float64(period)))
	return &HMA{
		period: period,
		wmaH:  NewWMA(half),
		wmaF:  NewWMA(period),
		wmaS:  NewWMA(sqrtP),
		sqrtP: sqrtP,
	}
}

func (h *HMA) Update(price float64) HMAResult {
	rH := h.wmaH.Update(price)
	rF := h.wmaF.Update(price)
	if !rH.Ready || !rF.Ready {
		return HMAResult{Ready: false}
	}
	diff := 2*rH.Value - rF.Value
	rS := h.wmaS.Update(diff)
	if !rS.Ready {
		return HMAResult{Ready: false}
	}
	return HMAResult{Value: rS.Value, Ready: true}
}

// ---------------------------------------------------------------------------
// VWMA - Volume-Weighted Moving Average
// ---------------------------------------------------------------------------

type VWMA struct {
	period  int
	pvBuf   *ringBuffer
	volBuf  *ringBuffer
}

func NewVWMA(period int) *VWMA {
	return &VWMA{
		period: period,
		pvBuf:  newRingBuffer(period),
		volBuf: newRingBuffer(period),
	}
}

func (v *VWMA) Update(price, volume float64) VWMAResult {
	v.pvBuf.push(price * volume)
	v.volBuf.push(volume)
	if !v.pvBuf.full() {
		return VWMAResult{Ready: false}
	}
	vs := v.volBuf.sum()
	if vs == 0 {
		return VWMAResult{Value: price, Ready: true}
	}
	return VWMAResult{Value: v.pvBuf.sum() / vs, Ready: true}
}

// ---------------------------------------------------------------------------
// Ichimoku Cloud
// ---------------------------------------------------------------------------

type Ichimoku struct {
	tenkanPeriod  int
	kijunPeriod   int
	senkouBPeriod int
	displacement  int
	highBuf       *ringBuffer
	lowBuf        *ringBuffer
	closeBuf      *ringBuffer
	count         int
}

func NewIchimoku(tenkan, kijun, senkouB, displacement int) *Ichimoku {
	maxP := senkouB
	if kijun > maxP {
		maxP = kijun
	}
	return &Ichimoku{
		tenkanPeriod:  tenkan,
		kijunPeriod:   kijun,
		senkouBPeriod: senkouB,
		displacement:  displacement,
		highBuf:       newRingBuffer(maxP),
		lowBuf:        newRingBuffer(maxP),
		closeBuf:      newRingBuffer(displacement + 1),
	}
}

func NewDefaultIchimoku() *Ichimoku {
	return NewIchimoku(9, 26, 52, 26)
}

func (ich *Ichimoku) highLow(n int) (float64, float64) {
	hi := -math.MaxFloat64
	lo := math.MaxFloat64
	for i := 0; i < n && i < ich.highBuf.count; i++ {
		h := ich.highBuf.get(i)
		l := ich.lowBuf.get(i)
		if h > hi {
			hi = h
		}
		if l < lo {
			lo = l
		}
	}
	return hi, lo
}

func (ich *Ichimoku) Update(bar Bar) IchimokuResult {
	ich.highBuf.push(bar.High)
	ich.lowBuf.push(bar.Low)
	ich.closeBuf.push(bar.Close)
	ich.count++
	if ich.count < ich.senkouBPeriod {
		return IchimokuResult{Ready: false}
	}
	tH, tL := ich.highLow(ich.tenkanPeriod)
	tenkan := (tH + tL) / 2
	kH, kL := ich.highLow(ich.kijunPeriod)
	kijun := (kH + kL) / 2
	senkouA := (tenkan + kijun) / 2
	sH, sL := ich.highLow(ich.senkouBPeriod)
	senkouB := (sH + sL) / 2
	chikou := bar.Close
	if ich.closeBuf.count > ich.displacement {
		chikou = ich.closeBuf.get(ich.displacement)
	}
	return IchimokuResult{
		Tenkan:  tenkan,
		Kijun:   kijun,
		SenkouA: senkouA,
		SenkouB: senkouB,
		Chikou:  chikou,
		Ready:   true,
	}
}

// ---------------------------------------------------------------------------
// Supertrend
// ---------------------------------------------------------------------------

type Supertrend struct {
	period     int
	multiplier float64
	atr        *ATR_
	prevClose  float64
	prevUpper  float64
	prevLower  float64
	direction  int
	count      int
}

func NewSupertrend(period int, multiplier float64) *Supertrend {
	return &Supertrend{
		period:     period,
		multiplier: multiplier,
		atr:        NewATR(period),
		direction:  1,
	}
}

func (s *Supertrend) Update(bar Bar) SupertrendResult {
	atrR := s.atr.Update(bar)
	s.count++
	if !atrR.Ready {
		s.prevClose = bar.Close
		return SupertrendResult{Ready: false}
	}
	hl2 := (bar.High + bar.Low) / 2
	upper := hl2 + s.multiplier*atrR.Value
	lower := hl2 - s.multiplier*atrR.Value
	if s.count > s.period+1 {
		if lower > s.prevLower || s.prevClose < s.prevLower {
			// keep lower
		} else {
			lower = s.prevLower
		}
		if upper < s.prevUpper || s.prevClose > s.prevUpper {
			// keep upper
		} else {
			upper = s.prevUpper
		}
	}
	if s.direction == 1 {
		if bar.Close < lower {
			s.direction = -1
		}
	} else {
		if bar.Close > upper {
			s.direction = 1
		}
	}
	s.prevUpper = upper
	s.prevLower = lower
	s.prevClose = bar.Close
	val := lower
	if s.direction == -1 {
		val = upper
	}
	return SupertrendResult{Value: val, Direction: s.direction, Ready: true}
}

// ---------------------------------------------------------------------------
// Parabolic SAR
// ---------------------------------------------------------------------------

type ParabolicSAR struct {
	afStart float64
	afStep  float64
	afMax   float64
	sar     float64
	ep      float64
	af      float64
	trend   int
	count   int
	prevH   float64
	prevL   float64
}

func NewParabolicSAR(afStart, afStep, afMax float64) *ParabolicSAR {
	return &ParabolicSAR{
		afStart: afStart,
		afStep:  afStep,
		afMax:   afMax,
		af:      afStart,
		trend:   1,
	}
}

func NewDefaultParabolicSAR() *ParabolicSAR {
	return NewParabolicSAR(0.02, 0.02, 0.2)
}

func (p *ParabolicSAR) Update(bar Bar) ParabolicSARResult {
	p.count++
	if p.count == 1 {
		p.sar = bar.Low
		p.ep = bar.High
		p.prevH = bar.High
		p.prevL = bar.Low
		return ParabolicSARResult{Value: p.sar, Trend: p.trend, Ready: false}
	}
	if p.count == 2 {
		if bar.Close >= p.prevH {
			p.trend = 1
			p.sar = p.prevL
			p.ep = bar.High
		} else {
			p.trend = -1
			p.sar = p.prevH
			p.ep = bar.Low
		}
	}
	prevSAR := p.sar
	prevAF := p.af
	prevEP := p.ep
	if p.trend == 1 {
		p.sar = prevSAR + prevAF*(prevEP-prevSAR)
		if p.sar > bar.Low {
			p.trend = -1
			p.sar = prevEP
			p.ep = bar.Low
			p.af = p.afStart
		} else {
			if bar.High > prevEP {
				p.ep = bar.High
				p.af = math.Min(prevAF+p.afStep, p.afMax)
			}
		}
	} else {
		p.sar = prevSAR + prevAF*(prevEP-prevSAR)
		if p.sar < bar.High {
			p.trend = 1
			p.sar = prevEP
			p.ep = bar.High
			p.af = p.afStart
		} else {
			if bar.Low < prevEP {
				p.ep = bar.Low
				p.af = math.Min(prevAF+p.afStep, p.afMax)
			}
		}
	}
	p.prevH = bar.High
	p.prevL = bar.Low
	return ParabolicSARResult{Value: p.sar, Trend: p.trend, Ready: p.count >= 2}
}

// ---------------------------------------------------------------------------
// RSI - Relative Strength Index
// ---------------------------------------------------------------------------

type RSI struct {
	period  int
	avgGain float64
	avgLoss float64
	prev    float64
	count   int
}

func NewRSI(period int) *RSI {
	return &RSI{period: period}
}

func (r *RSI) Update(price float64) RSIResult {
	r.count++
	if r.count == 1 {
		r.prev = price
		return RSIResult{Ready: false}
	}
	change := price - r.prev
	r.prev = price
	gain := 0.0
	loss := 0.0
	if change > 0 {
		gain = change
	} else {
		loss = -change
	}
	if r.count <= r.period+1 {
		r.avgGain += gain
		r.avgLoss += loss
		if r.count == r.period+1 {
			r.avgGain /= float64(r.period)
			r.avgLoss /= float64(r.period)
			if r.avgLoss == 0 {
				return RSIResult{Value: 100, Ready: true}
			}
			rs := r.avgGain / r.avgLoss
			return RSIResult{Value: 100 - 100/(1+rs), Ready: true}
		}
		return RSIResult{Ready: false}
	}
	p := float64(r.period)
	r.avgGain = (r.avgGain*(p-1) + gain) / p
	r.avgLoss = (r.avgLoss*(p-1) + loss) / p
	if r.avgLoss == 0 {
		return RSIResult{Value: 100, Ready: true}
	}
	rs := r.avgGain / r.avgLoss
	return RSIResult{Value: 100 - 100/(1+rs), Ready: true}
}

func (r *RSI) Value() float64 {
	if r.avgLoss == 0 {
		return 100
	}
	rs := r.avgGain / r.avgLoss
	return 100 - 100/(1+rs)
}

func (r *RSI) IsReady() bool { return r.count > r.period }

// ---------------------------------------------------------------------------
// Stochastic Oscillator
// ---------------------------------------------------------------------------

type Stochastic struct {
	kPeriod int
	dPeriod int
	smooth  int
	highBuf *ringBuffer
	lowBuf  *ringBuffer
	kBuf    *ringBuffer
	skBuf   *ringBuffer
}

func NewStochastic(kPeriod, dPeriod, smooth int) *Stochastic {
	return &Stochastic{
		kPeriod: kPeriod,
		dPeriod: dPeriod,
		smooth:  smooth,
		highBuf: newRingBuffer(kPeriod),
		lowBuf:  newRingBuffer(kPeriod),
		kBuf:    newRingBuffer(smooth),
		skBuf:   newRingBuffer(dPeriod),
	}
}

func (s *Stochastic) Update(high, low, close float64) StochasticResult {
	s.highBuf.push(high)
	s.lowBuf.push(low)
	if !s.highBuf.full() {
		return StochasticResult{Ready: false}
	}
	hh := s.highBuf.max()
	ll := s.lowBuf.min()
	var rawK float64
	if hh != ll {
		rawK = 100 * (close - ll) / (hh - ll)
	}
	s.kBuf.push(rawK)
	if !s.kBuf.full() {
		return StochasticResult{Ready: false}
	}
	k := s.kBuf.mean()
	s.skBuf.push(k)
	if !s.skBuf.full() {
		return StochasticResult{Ready: false}
	}
	d := s.skBuf.mean()
	return StochasticResult{K: k, D: d, Ready: true}
}

// ---------------------------------------------------------------------------
// MACD - Moving Average Convergence Divergence
// ---------------------------------------------------------------------------

type MACD struct {
	fastEMA   *EMA
	slowEMA   *EMA
	signalEMA *EMA
}

func NewMACD(fast, slow, signal int) *MACD {
	return &MACD{
		fastEMA:   NewEMA(fast),
		slowEMA:   NewEMA(slow),
		signalEMA: NewEMA(signal),
	}
}

func (m *MACD) Update(price float64) MACDResult {
	rf := m.fastEMA.Update(price)
	rs := m.slowEMA.Update(price)
	if !rf.Ready || !rs.Ready {
		return MACDResult{Ready: false}
	}
	line := rf.Value - rs.Value
	rsig := m.signalEMA.Update(line)
	if !rsig.Ready {
		return MACDResult{Ready: false}
	}
	return MACDResult{
		Line:      line,
		Signal:    rsig.Value,
		Histogram: line - rsig.Value,
		Ready:     true,
	}
}

// ---------------------------------------------------------------------------
// CCI - Commodity Channel Index
// ---------------------------------------------------------------------------

type CCI struct {
	period int
	buf    *ringBuffer
}

func NewCCI(period int) *CCI {
	return &CCI{period: period, buf: newRingBuffer(period)}
}

func (c *CCI) Update(high, low, close float64) CCIResult {
	tp := (high + low + close) / 3
	c.buf.push(tp)
	if !c.buf.full() {
		return CCIResult{Ready: false}
	}
	mean := c.buf.mean()
	md := 0.0
	vals := c.buf.values()
	for _, v := range vals {
		md += math.Abs(v - mean)
	}
	md /= float64(c.period)
	if md == 0 {
		return CCIResult{Value: 0, Ready: true}
	}
	return CCIResult{Value: (tp - mean) / (0.015 * md), Ready: true}
}

// ---------------------------------------------------------------------------
// Williams %R
// ---------------------------------------------------------------------------

type WilliamsR struct {
	period  int
	highBuf *ringBuffer
	lowBuf  *ringBuffer
}

func NewWilliamsR(period int) *WilliamsR {
	return &WilliamsR{
		period:  period,
		highBuf: newRingBuffer(period),
		lowBuf:  newRingBuffer(period),
	}
}

func (w *WilliamsR) Update(high, low, close float64) WilliamsRResult {
	w.highBuf.push(high)
	w.lowBuf.push(low)
	if !w.highBuf.full() {
		return WilliamsRResult{Ready: false}
	}
	hh := w.highBuf.max()
	ll := w.lowBuf.min()
	if hh == ll {
		return WilliamsRResult{Value: -50, Ready: true}
	}
	return WilliamsRResult{Value: -100 * (hh - close) / (hh - ll), Ready: true}
}

// ---------------------------------------------------------------------------
// MFI - Money Flow Index
// ---------------------------------------------------------------------------

type MFI_ struct {
	period    int
	posBuf    *ringBuffer
	negBuf    *ringBuffer
	prevTP    float64
	count     int
}

func NewMFI(period int) *MFI_ {
	return &MFI_{
		period: period,
		posBuf: newRingBuffer(period),
		negBuf: newRingBuffer(period),
	}
}

func (m *MFI_) Update(high, low, close, volume float64) MFIResult {
	tp := (high + low + close) / 3
	mf := tp * volume
	m.count++
	if m.count == 1 {
		m.prevTP = tp
		return MFIResult{Ready: false}
	}
	if tp > m.prevTP {
		m.posBuf.push(mf)
		m.negBuf.push(0)
	} else {
		m.posBuf.push(0)
		m.negBuf.push(mf)
	}
	m.prevTP = tp
	if m.posBuf.count < m.period {
		return MFIResult{Ready: false}
	}
	posSum := m.posBuf.sum()
	negSum := m.negBuf.sum()
	if negSum == 0 {
		return MFIResult{Value: 100, Ready: true}
	}
	ratio := posSum / negSum
	return MFIResult{Value: 100 - 100/(1+ratio), Ready: true}
}

// ---------------------------------------------------------------------------
// ROC - Rate of Change
// ---------------------------------------------------------------------------

type ROC struct {
	period int
	buf    *ringBuffer
}

func NewROC(period int) *ROC {
	return &ROC{period: period, buf: newRingBuffer(period + 1)}
}

func (r *ROC) Update(price float64) ROCResult {
	r.buf.push(price)
	if r.buf.count <= r.period {
		return ROCResult{Ready: false}
	}
	old := r.buf.get(r.period)
	if old == 0 {
		return ROCResult{Value: 0, Ready: true}
	}
	return ROCResult{Value: 100 * (price - old) / old, Ready: true}
}

// ---------------------------------------------------------------------------
// Momentum
// ---------------------------------------------------------------------------

type Momentum_ struct {
	period int
	buf    *ringBuffer
}

func NewMomentum(period int) *Momentum_ {
	return &Momentum_{period: period, buf: newRingBuffer(period + 1)}
}

func (m *Momentum_) Update(price float64) MomentumResult {
	m.buf.push(price)
	if m.buf.count <= m.period {
		return MomentumResult{Ready: false}
	}
	return MomentumResult{Value: price - m.buf.get(m.period), Ready: true}
}

// ---------------------------------------------------------------------------
// Ultimate Oscillator
// ---------------------------------------------------------------------------

type UltimateOscillator struct {
	p1, p2, p3  int
	bp1, bp2, bp3 *ringBuffer
	tr1, tr2, tr3 *ringBuffer
	prevClose   float64
	count       int
}

func NewUltimateOscillator(p1, p2, p3 int) *UltimateOscillator {
	return &UltimateOscillator{
		p1: p1, p2: p2, p3: p3,
		bp1: newRingBuffer(p1), bp2: newRingBuffer(p2), bp3: newRingBuffer(p3),
		tr1: newRingBuffer(p1), tr2: newRingBuffer(p2), tr3: newRingBuffer(p3),
	}
}

func (u *UltimateOscillator) Update(bar Bar) UltimateOscillatorResult {
	u.count++
	if u.count == 1 {
		u.prevClose = bar.Close
		return UltimateOscillatorResult{Ready: false}
	}
	bp := bar.Close - math.Min(bar.Low, u.prevClose)
	tr := math.Max(bar.High, u.prevClose) - math.Min(bar.Low, u.prevClose)
	u.prevClose = bar.Close
	u.bp1.push(bp)
	u.bp2.push(bp)
	u.bp3.push(bp)
	u.tr1.push(tr)
	u.tr2.push(tr)
	u.tr3.push(tr)
	if !u.bp3.full() {
		return UltimateOscillatorResult{Ready: false}
	}
	tr1s := u.tr1.sum()
	tr2s := u.tr2.sum()
	tr3s := u.tr3.sum()
	if tr1s == 0 || tr2s == 0 || tr3s == 0 {
		return UltimateOscillatorResult{Value: 50, Ready: true}
	}
	avg1 := u.bp1.sum() / tr1s
	avg2 := u.bp2.sum() / tr2s
	avg3 := u.bp3.sum() / tr3s
	uo := 100 * (4*avg1 + 2*avg2 + avg3) / 7
	return UltimateOscillatorResult{Value: uo, Ready: true}
}

// ---------------------------------------------------------------------------
// Stochastic RSI
// ---------------------------------------------------------------------------

type StochRSI struct {
	rsi    *RSI
	period int
	rsiBuf *ringBuffer
	kBuf   *ringBuffer
	kSmooth int
	dSmooth int
}

func NewStochRSI(rsiPeriod, stochPeriod, kSmooth, dSmooth int) *StochRSI {
	return &StochRSI{
		rsi:     NewRSI(rsiPeriod),
		period:  stochPeriod,
		rsiBuf:  newRingBuffer(stochPeriod),
		kBuf:    newRingBuffer(dSmooth),
		kSmooth: kSmooth,
		dSmooth: dSmooth,
	}
}

func (s *StochRSI) Update(price float64) StochRSIResult {
	r := s.rsi.Update(price)
	if !r.Ready {
		return StochRSIResult{Ready: false}
	}
	s.rsiBuf.push(r.Value)
	if !s.rsiBuf.full() {
		return StochRSIResult{Ready: false}
	}
	hi := s.rsiBuf.max()
	lo := s.rsiBuf.min()
	var k float64
	if hi != lo {
		k = 100 * (r.Value - lo) / (hi - lo)
	}
	s.kBuf.push(k)
	if !s.kBuf.full() {
		return StochRSIResult{Ready: false}
	}
	d := s.kBuf.mean()
	return StochRSIResult{K: k, D: d, Ready: true}
}

// ---------------------------------------------------------------------------
// Connors RSI
// ---------------------------------------------------------------------------

type ConnorsRSI struct {
	rsiPrice  *RSI
	rsiStreak *RSI
	rocPeriod int
	pctRank   *ringBuffer
	prevPrice float64
	streak    float64
	count     int
}

func NewConnorsRSI(rsiPeriod, streakPeriod, rankPeriod int) *ConnorsRSI {
	return &ConnorsRSI{
		rsiPrice:  NewRSI(rsiPeriod),
		rsiStreak: NewRSI(streakPeriod),
		rocPeriod: rankPeriod,
		pctRank:   newRingBuffer(rankPeriod),
	}
}

func (c *ConnorsRSI) Update(price float64) ConnorsRSIResult {
	c.count++
	if c.count == 1 {
		c.prevPrice = price
		return ConnorsRSIResult{Ready: false}
	}
	if price > c.prevPrice {
		if c.streak > 0 {
			c.streak++
		} else {
			c.streak = 1
		}
	} else if price < c.prevPrice {
		if c.streak < 0 {
			c.streak--
		} else {
			c.streak = -1
		}
	} else {
		c.streak = 0
	}
	change := price - c.prevPrice
	c.prevPrice = price
	rsiP := c.rsiPrice.Update(price)
	rsiS := c.rsiStreak.Update(c.streak)
	c.pctRank.push(change)
	if !rsiP.Ready || !rsiS.Ready || !c.pctRank.full() {
		return ConnorsRSIResult{Ready: false}
	}
	// Percent rank of current ROC
	vals := c.pctRank.values()
	current := vals[len(vals)-1]
	below := 0
	for _, v := range vals[:len(vals)-1] {
		if v < current {
			below++
		}
	}
	pctRank := 100 * float64(below) / float64(len(vals)-1)
	crsi := (rsiP.Value + rsiS.Value + pctRank) / 3
	return ConnorsRSIResult{Value: crsi, Ready: true}
}

// ---------------------------------------------------------------------------
// KST - Know Sure Thing
// ---------------------------------------------------------------------------

type KST struct {
	roc1, roc2, roc3, roc4 *ROC
	sma1, sma2, sma3, sma4 *SMA
	signalSMA              *SMA
}

func NewKST(r1, r2, r3, r4, s1, s2, s3, s4, sig int) *KST {
	return &KST{
		roc1: NewROC(r1), roc2: NewROC(r2), roc3: NewROC(r3), roc4: NewROC(r4),
		sma1: NewSMA(s1), sma2: NewSMA(s2), sma3: NewSMA(s3), sma4: NewSMA(s4),
		signalSMA: NewSMA(sig),
	}
}

func NewDefaultKST() *KST {
	return NewKST(10, 15, 20, 30, 10, 10, 10, 15, 9)
}

func (k *KST) Update(price float64) KSTResult {
	r1 := k.roc1.Update(price)
	r2 := k.roc2.Update(price)
	r3 := k.roc3.Update(price)
	r4 := k.roc4.Update(price)
	if !r1.Ready || !r2.Ready || !r3.Ready || !r4.Ready {
		return KSTResult{Ready: false}
	}
	s1 := k.sma1.Update(r1.Value)
	s2 := k.sma2.Update(r2.Value)
	s3 := k.sma3.Update(r3.Value)
	s4 := k.sma4.Update(r4.Value)
	if !s1.Ready || !s2.Ready || !s3.Ready || !s4.Ready {
		return KSTResult{Ready: false}
	}
	kst := s1.Value*1 + s2.Value*2 + s3.Value*3 + s4.Value*4
	sig := k.signalSMA.Update(kst)
	if !sig.Ready {
		return KSTResult{Ready: false}
	}
	return KSTResult{Value: kst, Signal: sig.Value, Ready: true}
}

// ---------------------------------------------------------------------------
// Bollinger Bands
// ---------------------------------------------------------------------------

type BollingerBands struct {
	period int
	stdMul float64
	buf    *ringBuffer
}

func NewBollingerBands(period int, stdMul float64) *BollingerBands {
	return &BollingerBands{period: period, stdMul: stdMul, buf: newRingBuffer(period)}
}

func (b *BollingerBands) Update(price float64) BollingerResult {
	b.buf.push(price)
	if !b.buf.full() {
		return BollingerResult{Ready: false}
	}
	mid := b.buf.mean()
	sd := b.buf.stddev()
	upper := mid + b.stdMul*sd
	lower := mid - b.stdMul*sd
	var pctB float64
	if upper != lower {
		pctB = (price - lower) / (upper - lower)
	}
	var bw float64
	if mid != 0 {
		bw = (upper - lower) / mid
	}
	return BollingerResult{
		Upper:     upper,
		Mid:       mid,
		Lower:     lower,
		PctB:      pctB,
		Bandwidth: bw,
		Ready:     true,
	}
}

// ---------------------------------------------------------------------------
// ATR - Average True Range
// ---------------------------------------------------------------------------

type ATR_ struct {
	period    int
	prevClose float64
	value     float64
	count     int
	sum       float64
}

func NewATR(period int) *ATR_ {
	return &ATR_{period: period}
}

func (a *ATR_) Update(bar Bar) ATRResult {
	a.count++
	if a.count == 1 {
		a.prevClose = bar.Close
		return ATRResult{Ready: false}
	}
	tr := math.Max(bar.High-bar.Low, math.Max(math.Abs(bar.High-a.prevClose), math.Abs(bar.Low-a.prevClose)))
	a.prevClose = bar.Close
	if a.count <= a.period+1 {
		a.sum += tr
		if a.count == a.period+1 {
			a.value = a.sum / float64(a.period)
			return ATRResult{Value: a.value, Ready: true}
		}
		return ATRResult{Ready: false}
	}
	a.value = (a.value*float64(a.period-1) + tr) / float64(a.period)
	return ATRResult{Value: a.value, Ready: true}
}

func (a *ATR_) Value() float64 { return a.value }
func (a *ATR_) IsReady() bool  { return a.count > a.period }

// ---------------------------------------------------------------------------
// Keltner Channel
// ---------------------------------------------------------------------------

type KeltnerChannel struct {
	ema *EMA
	atr *ATR_
	mul float64
}

func NewKeltnerChannel(period int, atrPeriod int, multiplier float64) *KeltnerChannel {
	return &KeltnerChannel{
		ema: NewEMA(period),
		atr: NewATR(atrPeriod),
		mul: multiplier,
	}
}

func (k *KeltnerChannel) Update(bar Bar) KeltnerResult {
	e := k.ema.Update(bar.Close)
	a := k.atr.Update(bar)
	if !e.Ready || !a.Ready {
		return KeltnerResult{Ready: false}
	}
	return KeltnerResult{
		Upper: e.Value + k.mul*a.Value,
		Mid:   e.Value,
		Lower: e.Value - k.mul*a.Value,
		Ready: true,
	}
}

// ---------------------------------------------------------------------------
// Donchian Channel
// ---------------------------------------------------------------------------

type DonchianChannel struct {
	period  int
	highBuf *ringBuffer
	lowBuf  *ringBuffer
}

func NewDonchianChannel(period int) *DonchianChannel {
	return &DonchianChannel{
		period:  period,
		highBuf: newRingBuffer(period),
		lowBuf:  newRingBuffer(period),
	}
}

func (d *DonchianChannel) Update(bar Bar) DonchianResult {
	d.highBuf.push(bar.High)
	d.lowBuf.push(bar.Low)
	if !d.highBuf.full() {
		return DonchianResult{Ready: false}
	}
	upper := d.highBuf.max()
	lower := d.lowBuf.min()
	return DonchianResult{Upper: upper, Mid: (upper + lower) / 2, Lower: lower, Ready: true}
}

// ---------------------------------------------------------------------------
// Standard Deviation
// ---------------------------------------------------------------------------

type StdDev struct {
	period int
	buf    *ringBuffer
}

func NewStdDev(period int) *StdDev {
	return &StdDev{period: period, buf: newRingBuffer(period)}
}

func (s *StdDev) Update(price float64) StdDevResult {
	s.buf.push(price)
	if !s.buf.full() {
		return StdDevResult{Ready: false}
	}
	return StdDevResult{Value: s.buf.stddev(), Ready: true}
}

// ---------------------------------------------------------------------------
// OBV - On-Balance Volume
// ---------------------------------------------------------------------------

type OBV struct {
	value     float64
	prevClose float64
	count     int
}

func NewOBV() *OBV {
	return &OBV{}
}

func (o *OBV) Update(close, volume float64) OBVResult {
	o.count++
	if o.count == 1 {
		o.prevClose = close
		o.value = volume
		return OBVResult{Value: o.value, Ready: true}
	}
	if close > o.prevClose {
		o.value += volume
	} else if close < o.prevClose {
		o.value -= volume
	}
	o.prevClose = close
	return OBVResult{Value: o.value, Ready: true}
}

// ---------------------------------------------------------------------------
// VWAP - Volume Weighted Average Price
// ---------------------------------------------------------------------------

type VWAP_ struct {
	cumPV  float64
	cumVol float64
	count  int
}

func NewVWAP() *VWAP_ {
	return &VWAP_{}
}

func (v *VWAP_) Update(bar Bar) VWAPResult {
	v.count++
	tp := (bar.High + bar.Low + bar.Close) / 3
	v.cumPV += tp * bar.Volume
	v.cumVol += bar.Volume
	if v.cumVol == 0 {
		return VWAPResult{Value: bar.Close, Ready: true}
	}
	return VWAPResult{Value: v.cumPV / v.cumVol, Ready: true}
}

func (v *VWAP_) Reset() {
	v.cumPV = 0
	v.cumVol = 0
	v.count = 0
}

// ---------------------------------------------------------------------------
// Accumulation/Distribution
// ---------------------------------------------------------------------------

type AD struct {
	value float64
}

func NewAD() *AD {
	return &AD{}
}

func (a *AD) Update(bar Bar) ADResult {
	hl := bar.High - bar.Low
	if hl == 0 {
		return ADResult{Value: a.value, Ready: true}
	}
	clv := ((bar.Close - bar.Low) - (bar.High - bar.Close)) / hl
	a.value += clv * bar.Volume
	return ADResult{Value: a.value, Ready: true}
}

// ---------------------------------------------------------------------------
// Chaikin Money Flow
// ---------------------------------------------------------------------------

type ChaikinMF struct {
	period int
	mfvBuf *ringBuffer
	volBuf *ringBuffer
}

func NewChaikinMF(period int) *ChaikinMF {
	return &ChaikinMF{
		period: period,
		mfvBuf: newRingBuffer(period),
		volBuf: newRingBuffer(period),
	}
}

func (c *ChaikinMF) Update(bar Bar) ChaikinMFResult {
	hl := bar.High - bar.Low
	mfv := 0.0
	if hl != 0 {
		mfm := ((bar.Close - bar.Low) - (bar.High - bar.Close)) / hl
		mfv = mfm * bar.Volume
	}
	c.mfvBuf.push(mfv)
	c.volBuf.push(bar.Volume)
	if !c.mfvBuf.full() {
		return ChaikinMFResult{Ready: false}
	}
	vs := c.volBuf.sum()
	if vs == 0 {
		return ChaikinMFResult{Value: 0, Ready: true}
	}
	return ChaikinMFResult{Value: c.mfvBuf.sum() / vs, Ready: true}
}

// ---------------------------------------------------------------------------
// Force Index
// ---------------------------------------------------------------------------

type ForceIndex struct {
	period    int
	ema       *EMA
	prevClose float64
	count     int
}

func NewForceIndex(period int) *ForceIndex {
	return &ForceIndex{period: period, ema: NewEMA(period)}
}

func (f *ForceIndex) Update(close, volume float64) ForceIndexResult {
	f.count++
	if f.count == 1 {
		f.prevClose = close
		return ForceIndexResult{Ready: false}
	}
	raw := (close - f.prevClose) * volume
	f.prevClose = close
	r := f.ema.Update(raw)
	if !r.Ready {
		return ForceIndexResult{Ready: false}
	}
	return ForceIndexResult{Value: r.Value, Ready: true}
}

// ---------------------------------------------------------------------------
// Elder Ray (Bull/Bear Power)
// ---------------------------------------------------------------------------

type ElderRay struct {
	ema *EMA
}

func NewElderRay(period int) *ElderRay {
	return &ElderRay{ema: NewEMA(period)}
}

func (e *ElderRay) Update(bar Bar) ElderRayResult {
	r := e.ema.Update(bar.Close)
	if !r.Ready {
		return ElderRayResult{Ready: false}
	}
	return ElderRayResult{
		BullPower: bar.High - r.Value,
		BearPower: bar.Low - r.Value,
		Ready:     true,
	}
}

// ---------------------------------------------------------------------------
// IndicatorSignal represents a signal from an indicator.
// ---------------------------------------------------------------------------

// SignalType describes the nature of a trading signal.
type SignalType int

const (
	SignalNone    SignalType = 0
	SignalBuy     SignalType = 1
	SignalSell    SignalType = -1
	SignalNeutral SignalType = 2
)

// IndicatorSignal carries a signal from an indicator.
type IndicatorSignal struct {
	Name       string
	Category   string
	Signal     SignalType
	Value      float64
	Strength   float64
	Ready      bool
}

// ---------------------------------------------------------------------------
// Indicator interface for the engine.
// ---------------------------------------------------------------------------

// Indicator is the common interface for all registered indicators.
type Indicator interface {
	Name() string
	Category() string
	Update(bar Bar) IndicatorSignal
}

// ---------------------------------------------------------------------------
// Wrapped indicators implementing the Indicator interface
// ---------------------------------------------------------------------------

type smaIndicator struct {
	name   string
	period int
	sma    *SMA
	prev   float64
}

func NewSMAIndicator(period int) Indicator {
	return &smaIndicator{
		name:   fmt.Sprintf("SMA_%d", period),
		period: period,
		sma:    NewSMA(period),
	}
}

func (s *smaIndicator) Name() string     { return s.name }
func (s *smaIndicator) Category() string  { return "trend" }
func (s *smaIndicator) Update(bar Bar) IndicatorSignal {
	r := s.sma.Update(bar.Close)
	sig := IndicatorSignal{Name: s.name, Category: "trend", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if bar.Close > r.Value && s.prev <= r.Value && s.prev > 0 {
			sig.Signal = SignalBuy
			sig.Strength = (bar.Close - r.Value) / r.Value
		} else if bar.Close < r.Value && s.prev >= r.Value && s.prev > 0 {
			sig.Signal = SignalSell
			sig.Strength = (r.Value - bar.Close) / r.Value
		}
	}
	if r.Ready {
		s.prev = r.Value
	}
	return sig
}

type emaIndicator struct {
	name string
	ema  *EMA
	prev float64
}

func NewEMAIndicator(period int) Indicator {
	return &emaIndicator{
		name: fmt.Sprintf("EMA_%d", period),
		ema:  NewEMA(period),
	}
}

func (e *emaIndicator) Name() string     { return e.name }
func (e *emaIndicator) Category() string  { return "trend" }
func (e *emaIndicator) Update(bar Bar) IndicatorSignal {
	r := e.ema.Update(bar.Close)
	sig := IndicatorSignal{Name: e.name, Category: "trend", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if bar.Close > r.Value && e.prev <= r.Value && e.prev > 0 {
			sig.Signal = SignalBuy
		} else if bar.Close < r.Value && e.prev >= r.Value && e.prev > 0 {
			sig.Signal = SignalSell
		}
		e.prev = r.Value
	}
	return sig
}

type rsiIndicator struct {
	name string
	rsi  *RSI
}

func NewRSIIndicator(period int) Indicator {
	return &rsiIndicator{
		name: fmt.Sprintf("RSI_%d", period),
		rsi:  NewRSI(period),
	}
}

func (r *rsiIndicator) Name() string     { return r.name }
func (r *rsiIndicator) Category() string  { return "momentum" }
func (r *rsiIndicator) Update(bar Bar) IndicatorSignal {
	res := r.rsi.Update(bar.Close)
	sig := IndicatorSignal{Name: r.name, Category: "momentum", Value: res.Value, Ready: res.Ready}
	if res.Ready {
		if res.Value < 30 {
			sig.Signal = SignalBuy
			sig.Strength = (30 - res.Value) / 30
		} else if res.Value > 70 {
			sig.Signal = SignalSell
			sig.Strength = (res.Value - 70) / 30
		}
	}
	return sig
}

type macdIndicator struct {
	name string
	macd *MACD
	prevHist float64
}

func NewMACDIndicator(fast, slow, signal int) Indicator {
	return &macdIndicator{
		name: fmt.Sprintf("MACD_%d_%d_%d", fast, slow, signal),
		macd: NewMACD(fast, slow, signal),
	}
}

func (m *macdIndicator) Name() string     { return m.name }
func (m *macdIndicator) Category() string  { return "momentum" }
func (m *macdIndicator) Update(bar Bar) IndicatorSignal {
	r := m.macd.Update(bar.Close)
	sig := IndicatorSignal{Name: m.name, Category: "momentum", Value: r.Histogram, Ready: r.Ready}
	if r.Ready {
		if r.Histogram > 0 && m.prevHist <= 0 {
			sig.Signal = SignalBuy
			sig.Strength = math.Abs(r.Histogram)
		} else if r.Histogram < 0 && m.prevHist >= 0 {
			sig.Signal = SignalSell
			sig.Strength = math.Abs(r.Histogram)
		}
		m.prevHist = r.Histogram
	}
	return sig
}

type bollingerIndicator struct {
	name string
	bb   *BollingerBands
}

func NewBollingerIndicator(period int, stdMul float64) Indicator {
	return &bollingerIndicator{
		name: fmt.Sprintf("BB_%d_%.1f", period, stdMul),
		bb:   NewBollingerBands(period, stdMul),
	}
}

func (b *bollingerIndicator) Name() string     { return b.name }
func (b *bollingerIndicator) Category() string  { return "volatility" }
func (b *bollingerIndicator) Update(bar Bar) IndicatorSignal {
	r := b.bb.Update(bar.Close)
	sig := IndicatorSignal{Name: b.name, Category: "volatility", Value: r.PctB, Ready: r.Ready}
	if r.Ready {
		if r.PctB < 0 {
			sig.Signal = SignalBuy
			sig.Strength = math.Abs(r.PctB)
		} else if r.PctB > 1 {
			sig.Signal = SignalSell
			sig.Strength = r.PctB - 1
		}
	}
	return sig
}

type obvIndicator struct {
	name string
	obv  *OBV
	sma  *SMA
}

func NewOBVIndicator(smaPeriod int) Indicator {
	return &obvIndicator{
		name: fmt.Sprintf("OBV_%d", smaPeriod),
		obv:  NewOBV(),
		sma:  NewSMA(smaPeriod),
	}
}

func (o *obvIndicator) Name() string     { return o.name }
func (o *obvIndicator) Category() string  { return "volume" }
func (o *obvIndicator) Update(bar Bar) IndicatorSignal {
	r := o.obv.Update(bar.Close, bar.Volume)
	s := o.sma.Update(r.Value)
	sig := IndicatorSignal{Name: o.name, Category: "volume", Value: r.Value, Ready: s.Ready}
	if s.Ready {
		if r.Value > s.Value {
			sig.Signal = SignalBuy
		} else {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type stochasticIndicator struct {
	name  string
	stoch *Stochastic
}

func NewStochasticIndicator(k, d, smooth int) Indicator {
	return &stochasticIndicator{
		name:  fmt.Sprintf("STOCH_%d_%d_%d", k, d, smooth),
		stoch: NewStochastic(k, d, smooth),
	}
}

func (s *stochasticIndicator) Name() string     { return s.name }
func (s *stochasticIndicator) Category() string  { return "momentum" }
func (s *stochasticIndicator) Update(bar Bar) IndicatorSignal {
	r := s.stoch.Update(bar.High, bar.Low, bar.Close)
	sig := IndicatorSignal{Name: s.name, Category: "momentum", Value: r.K, Ready: r.Ready}
	if r.Ready {
		if r.K < 20 && r.K > r.D {
			sig.Signal = SignalBuy
			sig.Strength = (20 - r.K) / 20
		} else if r.K > 80 && r.K < r.D {
			sig.Signal = SignalSell
			sig.Strength = (r.K - 80) / 20
		}
	}
	return sig
}

type cciIndicator struct {
	name string
	cci  *CCI
}

func NewCCIIndicator(period int) Indicator {
	return &cciIndicator{
		name: fmt.Sprintf("CCI_%d", period),
		cci:  NewCCI(period),
	}
}

func (c *cciIndicator) Name() string     { return c.name }
func (c *cciIndicator) Category() string  { return "momentum" }
func (c *cciIndicator) Update(bar Bar) IndicatorSignal {
	r := c.cci.Update(bar.High, bar.Low, bar.Close)
	sig := IndicatorSignal{Name: c.name, Category: "momentum", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value < -100 {
			sig.Signal = SignalBuy
			sig.Strength = math.Abs(r.Value) / 200
		} else if r.Value > 100 {
			sig.Signal = SignalSell
			sig.Strength = r.Value / 200
		}
	}
	return sig
}

type atrIndicator struct {
	name string
	atr  *ATR_
}

func NewATRIndicator(period int) Indicator {
	return &atrIndicator{
		name: fmt.Sprintf("ATR_%d", period),
		atr:  NewATR(period),
	}
}

func (a *atrIndicator) Name() string     { return a.name }
func (a *atrIndicator) Category() string  { return "volatility" }
func (a *atrIndicator) Update(bar Bar) IndicatorSignal {
	r := a.atr.Update(bar)
	return IndicatorSignal{Name: a.name, Category: "volatility", Value: r.Value, Ready: r.Ready, Signal: SignalNeutral}
}

type supertrendIndicator struct {
	name string
	st   *Supertrend
}

func NewSupertrendIndicator(period int, multiplier float64) Indicator {
	return &supertrendIndicator{
		name: fmt.Sprintf("ST_%d_%.1f", period, multiplier),
		st:   NewSupertrend(period, multiplier),
	}
}

func (s *supertrendIndicator) Name() string     { return s.name }
func (s *supertrendIndicator) Category() string  { return "trend" }
func (s *supertrendIndicator) Update(bar Bar) IndicatorSignal {
	r := s.st.Update(bar)
	sig := IndicatorSignal{Name: s.name, Category: "trend", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Direction == 1 {
			sig.Signal = SignalBuy
		} else {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type williamsRIndicator struct {
	name string
	wr   *WilliamsR
}

func NewWilliamsRIndicator(period int) Indicator {
	return &williamsRIndicator{
		name: fmt.Sprintf("WR_%d", period),
		wr:   NewWilliamsR(period),
	}
}

func (w *williamsRIndicator) Name() string     { return w.name }
func (w *williamsRIndicator) Category() string  { return "momentum" }
func (w *williamsRIndicator) Update(bar Bar) IndicatorSignal {
	r := w.wr.Update(bar.High, bar.Low, bar.Close)
	sig := IndicatorSignal{Name: w.name, Category: "momentum", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value < -80 {
			sig.Signal = SignalBuy
			sig.Strength = (-80 - r.Value) / 20
		} else if r.Value > -20 {
			sig.Signal = SignalSell
			sig.Strength = (r.Value + 20) / 20
		}
	}
	return sig
}

type mfiIndicator struct {
	name string
	mfi  *MFI_
}

func NewMFIIndicator(period int) Indicator {
	return &mfiIndicator{
		name: fmt.Sprintf("MFI_%d", period),
		mfi:  NewMFI(period),
	}
}

func (m *mfiIndicator) Name() string     { return m.name }
func (m *mfiIndicator) Category() string  { return "volume" }
func (m *mfiIndicator) Update(bar Bar) IndicatorSignal {
	r := m.mfi.Update(bar.High, bar.Low, bar.Close, bar.Volume)
	sig := IndicatorSignal{Name: m.name, Category: "volume", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value < 20 {
			sig.Signal = SignalBuy
			sig.Strength = (20 - r.Value) / 20
		} else if r.Value > 80 {
			sig.Signal = SignalSell
			sig.Strength = (r.Value - 80) / 20
		}
	}
	return sig
}

type keltnerIndicator struct {
	name string
	kc   *KeltnerChannel
}

func NewKeltnerIndicator(period, atrPeriod int, multiplier float64) Indicator {
	return &keltnerIndicator{
		name: fmt.Sprintf("KC_%d_%d_%.1f", period, atrPeriod, multiplier),
		kc:   NewKeltnerChannel(period, atrPeriod, multiplier),
	}
}

func (k *keltnerIndicator) Name() string     { return k.name }
func (k *keltnerIndicator) Category() string  { return "volatility" }
func (k *keltnerIndicator) Update(bar Bar) IndicatorSignal {
	r := k.kc.Update(bar)
	sig := IndicatorSignal{Name: k.name, Category: "volatility", Value: r.Mid, Ready: r.Ready}
	if r.Ready {
		if bar.Close < r.Lower {
			sig.Signal = SignalBuy
		} else if bar.Close > r.Upper {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type donchianIndicator struct {
	name string
	dc   *DonchianChannel
}

func NewDonchianIndicator(period int) Indicator {
	return &donchianIndicator{
		name: fmt.Sprintf("DC_%d", period),
		dc:   NewDonchianChannel(period),
	}
}

func (d *donchianIndicator) Name() string     { return d.name }
func (d *donchianIndicator) Category() string  { return "volatility" }
func (d *donchianIndicator) Update(bar Bar) IndicatorSignal {
	r := d.dc.Update(bar)
	sig := IndicatorSignal{Name: d.name, Category: "volatility", Value: r.Mid, Ready: r.Ready}
	if r.Ready {
		if bar.Close >= r.Upper {
			sig.Signal = SignalBuy
			sig.Strength = 1.0
		} else if bar.Close <= r.Lower {
			sig.Signal = SignalSell
			sig.Strength = 1.0
		}
	}
	return sig
}

type elderRayIndicator struct {
	name string
	er   *ElderRay
}

func NewElderRayIndicator(period int) Indicator {
	return &elderRayIndicator{
		name: fmt.Sprintf("ER_%d", period),
		er:   NewElderRay(period),
	}
}

func (e *elderRayIndicator) Name() string     { return e.name }
func (e *elderRayIndicator) Category() string  { return "volume" }
func (e *elderRayIndicator) Update(bar Bar) IndicatorSignal {
	r := e.er.Update(bar)
	sig := IndicatorSignal{Name: e.name, Category: "volume", Value: r.BullPower, Ready: r.Ready}
	if r.Ready {
		if r.BullPower > 0 && r.BearPower > 0 {
			sig.Signal = SignalBuy
		} else if r.BullPower < 0 && r.BearPower < 0 {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type forceIndexIndicator struct {
	name string
	fi   *ForceIndex
}

func NewForceIndexIndicator(period int) Indicator {
	return &forceIndexIndicator{
		name: fmt.Sprintf("FI_%d", period),
		fi:   NewForceIndex(period),
	}
}

func (f *forceIndexIndicator) Name() string     { return f.name }
func (f *forceIndexIndicator) Category() string  { return "volume" }
func (f *forceIndexIndicator) Update(bar Bar) IndicatorSignal {
	r := f.fi.Update(bar.Close, bar.Volume)
	sig := IndicatorSignal{Name: f.name, Category: "volume", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value > 0 {
			sig.Signal = SignalBuy
		} else if r.Value < 0 {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type chaikinMFIndicator struct {
	name string
	cmf  *ChaikinMF
}

func NewChaikinMFIndicator(period int) Indicator {
	return &chaikinMFIndicator{
		name: fmt.Sprintf("CMF_%d", period),
		cmf:  NewChaikinMF(period),
	}
}

func (c *chaikinMFIndicator) Name() string     { return c.name }
func (c *chaikinMFIndicator) Category() string  { return "volume" }
func (c *chaikinMFIndicator) Update(bar Bar) IndicatorSignal {
	r := c.cmf.Update(bar)
	sig := IndicatorSignal{Name: c.name, Category: "volume", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value > 0.05 {
			sig.Signal = SignalBuy
			sig.Strength = r.Value
		} else if r.Value < -0.05 {
			sig.Signal = SignalSell
			sig.Strength = math.Abs(r.Value)
		}
	}
	return sig
}

type rocIndicator struct {
	name string
	roc  *ROC
}

func NewROCIndicator(period int) Indicator {
	return &rocIndicator{
		name: fmt.Sprintf("ROC_%d", period),
		roc:  NewROC(period),
	}
}

func (ro *rocIndicator) Name() string     { return ro.name }
func (ro *rocIndicator) Category() string  { return "momentum" }
func (ro *rocIndicator) Update(bar Bar) IndicatorSignal {
	r := ro.roc.Update(bar.Close)
	sig := IndicatorSignal{Name: ro.name, Category: "momentum", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value > 0 {
			sig.Signal = SignalBuy
			sig.Strength = math.Min(r.Value/10, 1)
		} else if r.Value < 0 {
			sig.Signal = SignalSell
			sig.Strength = math.Min(math.Abs(r.Value)/10, 1)
		}
	}
	return sig
}

type momentumIndicator struct {
	name string
	mom  *Momentum_
}

func NewMomentumIndicator(period int) Indicator {
	return &momentumIndicator{
		name: fmt.Sprintf("MOM_%d", period),
		mom:  NewMomentum(period),
	}
}

func (m *momentumIndicator) Name() string     { return m.name }
func (m *momentumIndicator) Category() string  { return "momentum" }
func (m *momentumIndicator) Update(bar Bar) IndicatorSignal {
	r := m.mom.Update(bar.Close)
	sig := IndicatorSignal{Name: m.name, Category: "momentum", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value > 0 {
			sig.Signal = SignalBuy
		} else if r.Value < 0 {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type vwapIndicator struct {
	name string
	vwap *VWAP_
}

func NewVWAPIndicator() Indicator {
	return &vwapIndicator{
		name: "VWAP",
		vwap: NewVWAP(),
	}
}

func (v *vwapIndicator) Name() string     { return v.name }
func (v *vwapIndicator) Category() string  { return "volume" }
func (v *vwapIndicator) Update(bar Bar) IndicatorSignal {
	r := v.vwap.Update(bar)
	sig := IndicatorSignal{Name: v.name, Category: "volume", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if bar.Close > r.Value {
			sig.Signal = SignalBuy
		} else {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type adIndicator struct {
	name string
	ad   *AD
	sma  *SMA
}

func NewADIndicator(smaPeriod int) Indicator {
	return &adIndicator{
		name: fmt.Sprintf("AD_%d", smaPeriod),
		ad:   NewAD(),
		sma:  NewSMA(smaPeriod),
	}
}

func (a *adIndicator) Name() string     { return a.name }
func (a *adIndicator) Category() string  { return "volume" }
func (a *adIndicator) Update(bar Bar) IndicatorSignal {
	r := a.ad.Update(bar)
	s := a.sma.Update(r.Value)
	sig := IndicatorSignal{Name: a.name, Category: "volume", Value: r.Value, Ready: s.Ready}
	if s.Ready {
		if r.Value > s.Value {
			sig.Signal = SignalBuy
		} else {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type ichimokuIndicator struct {
	name string
	ich  *Ichimoku
}

func NewIchimokuIndicator() Indicator {
	return &ichimokuIndicator{
		name: "ICHIMOKU",
		ich:  NewDefaultIchimoku(),
	}
}

func (i *ichimokuIndicator) Name() string     { return i.name }
func (i *ichimokuIndicator) Category() string  { return "trend" }
func (i *ichimokuIndicator) Update(bar Bar) IndicatorSignal {
	r := i.ich.Update(bar)
	sig := IndicatorSignal{Name: i.name, Category: "trend", Value: r.Tenkan, Ready: r.Ready}
	if r.Ready {
		aboveCloud := bar.Close > r.SenkouA && bar.Close > r.SenkouB
		belowCloud := bar.Close < r.SenkouA && bar.Close < r.SenkouB
		if aboveCloud && r.Tenkan > r.Kijun {
			sig.Signal = SignalBuy
			sig.Strength = 1.0
		} else if belowCloud && r.Tenkan < r.Kijun {
			sig.Signal = SignalSell
			sig.Strength = 1.0
		}
	}
	return sig
}

type psarIndicator struct {
	name string
	psar *ParabolicSAR
}

func NewParabolicSARIndicator() Indicator {
	return &psarIndicator{
		name: "PSAR",
		psar: NewDefaultParabolicSAR(),
	}
}

func (p *psarIndicator) Name() string     { return p.name }
func (p *psarIndicator) Category() string  { return "trend" }
func (p *psarIndicator) Update(bar Bar) IndicatorSignal {
	r := p.psar.Update(bar)
	sig := IndicatorSignal{Name: p.name, Category: "trend", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Trend == 1 {
			sig.Signal = SignalBuy
		} else {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type stochRSIIndicator struct {
	name string
	srsi *StochRSI
}

func NewStochRSIIndicator(rsiPeriod, stochPeriod, kSmooth, dSmooth int) Indicator {
	return &stochRSIIndicator{
		name: fmt.Sprintf("SRSI_%d_%d", rsiPeriod, stochPeriod),
		srsi: NewStochRSI(rsiPeriod, stochPeriod, kSmooth, dSmooth),
	}
}

func (s *stochRSIIndicator) Name() string     { return s.name }
func (s *stochRSIIndicator) Category() string  { return "momentum" }
func (s *stochRSIIndicator) Update(bar Bar) IndicatorSignal {
	r := s.srsi.Update(bar.Close)
	sig := IndicatorSignal{Name: s.name, Category: "momentum", Value: r.K, Ready: r.Ready}
	if r.Ready {
		if r.K < 20 {
			sig.Signal = SignalBuy
		} else if r.K > 80 {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type kstIndicator struct {
	name string
	kst  *KST
}

func NewKSTIndicator() Indicator {
	return &kstIndicator{
		name: "KST",
		kst:  NewDefaultKST(),
	}
}

func (k *kstIndicator) Name() string     { return k.name }
func (k *kstIndicator) Category() string  { return "momentum" }
func (k *kstIndicator) Update(bar Bar) IndicatorSignal {
	r := k.kst.Update(bar.Close)
	sig := IndicatorSignal{Name: k.name, Category: "momentum", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value > r.Signal {
			sig.Signal = SignalBuy
		} else {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type stdDevIndicator struct {
	name string
	sd   *StdDev
}

func NewStdDevIndicator(period int) Indicator {
	return &stdDevIndicator{
		name: fmt.Sprintf("STDDEV_%d", period),
		sd:   NewStdDev(period),
	}
}

func (s *stdDevIndicator) Name() string     { return s.name }
func (s *stdDevIndicator) Category() string  { return "volatility" }
func (s *stdDevIndicator) Update(bar Bar) IndicatorSignal {
	r := s.sd.Update(bar.Close)
	return IndicatorSignal{Name: s.name, Category: "volatility", Value: r.Value, Ready: r.Ready, Signal: SignalNeutral}
}

type uoIndicator struct {
	name string
	uo   *UltimateOscillator
}

func NewUltimateOscillatorIndicator(p1, p2, p3 int) Indicator {
	return &uoIndicator{
		name: fmt.Sprintf("UO_%d_%d_%d", p1, p2, p3),
		uo:   NewUltimateOscillator(p1, p2, p3),
	}
}

func (u *uoIndicator) Name() string     { return u.name }
func (u *uoIndicator) Category() string  { return "momentum" }
func (u *uoIndicator) Update(bar Bar) IndicatorSignal {
	r := u.uo.Update(bar)
	sig := IndicatorSignal{Name: u.name, Category: "momentum", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value < 30 {
			sig.Signal = SignalBuy
		} else if r.Value > 70 {
			sig.Signal = SignalSell
		}
	}
	return sig
}

type connorsRSIIndicator struct {
	name string
	crsi *ConnorsRSI
}

func NewConnorsRSIIndicator(rsiPeriod, streakPeriod, rankPeriod int) Indicator {
	return &connorsRSIIndicator{
		name: fmt.Sprintf("CRSI_%d_%d_%d", rsiPeriod, streakPeriod, rankPeriod),
		crsi: NewConnorsRSI(rsiPeriod, streakPeriod, rankPeriod),
	}
}

func (c *connorsRSIIndicator) Name() string     { return c.name }
func (c *connorsRSIIndicator) Category() string  { return "momentum" }
func (c *connorsRSIIndicator) Update(bar Bar) IndicatorSignal {
	r := c.crsi.Update(bar.Close)
	sig := IndicatorSignal{Name: c.name, Category: "momentum", Value: r.Value, Ready: r.Ready}
	if r.Ready {
		if r.Value < 20 {
			sig.Signal = SignalBuy
		} else if r.Value > 80 {
			sig.Signal = SignalSell
		}
	}
	return sig
}

// ---------------------------------------------------------------------------
// IndicatorEngine: register, update all, collect signals
// ---------------------------------------------------------------------------

// IndicatorEngine manages a collection of indicators and produces signals.
type IndicatorEngine struct {
	mu         sync.RWMutex
	indicators []Indicator
	signals    map[string]IndicatorSignal
	barCount   int
}

// NewIndicatorEngine creates a new engine.
func NewIndicatorEngine() *IndicatorEngine {
	return &IndicatorEngine{
		indicators: make([]Indicator, 0, 32),
		signals:    make(map[string]IndicatorSignal),
	}
}

// Register adds an indicator to the engine.
func (e *IndicatorEngine) Register(ind Indicator) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.indicators = append(e.indicators, ind)
}

// Update feeds a new bar to all registered indicators and collects signals.
func (e *IndicatorEngine) Update(bar Bar) map[string]IndicatorSignal {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.barCount++
	for _, ind := range e.indicators {
		sig := ind.Update(bar)
		e.signals[ind.Name()] = sig
	}
	out := make(map[string]IndicatorSignal, len(e.signals))
	for k, v := range e.signals {
		out[k] = v
	}
	return out
}

// Signals returns the latest signals snapshot.
func (e *IndicatorEngine) Signals() map[string]IndicatorSignal {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make(map[string]IndicatorSignal, len(e.signals))
	for k, v := range e.signals {
		out[k] = v
	}
	return out
}

// BuySignals returns only buy signals that are ready.
func (e *IndicatorEngine) BuySignals() []IndicatorSignal {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var out []IndicatorSignal
	for _, s := range e.signals {
		if s.Ready && s.Signal == SignalBuy {
			out = append(out, s)
		}
	}
	return out
}

// SellSignals returns only sell signals that are ready.
func (e *IndicatorEngine) SellSignals() []IndicatorSignal {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var out []IndicatorSignal
	for _, s := range e.signals {
		if s.Ready && s.Signal == SignalSell {
			out = append(out, s)
		}
	}
	return out
}

// SignalsByCategory returns signals filtered by category.
func (e *IndicatorEngine) SignalsByCategory(cat string) []IndicatorSignal {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var out []IndicatorSignal
	for _, s := range e.signals {
		if s.Category == cat {
			out = append(out, s)
		}
	}
	return out
}

// Consensus returns a weighted consensus: +1 all buy, -1 all sell, 0 mixed.
func (e *IndicatorEngine) Consensus() float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	total := 0.0
	count := 0.0
	for _, s := range e.signals {
		if !s.Ready {
			continue
		}
		switch s.Signal {
		case SignalBuy:
			total += 1
			count++
		case SignalSell:
			total -= 1
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return total / count
}

// BarCount returns the number of bars processed.
func (e *IndicatorEngine) BarCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.barCount
}

// RegisteredCount returns the number of registered indicators.
func (e *IndicatorEngine) RegisteredCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.indicators)
}

// RegisterDefaults registers a standard set of indicators.
func (e *IndicatorEngine) RegisterDefaults() {
	e.Register(NewSMAIndicator(20))
	e.Register(NewSMAIndicator(50))
	e.Register(NewSMAIndicator(200))
	e.Register(NewEMAIndicator(12))
	e.Register(NewEMAIndicator(26))
	e.Register(NewRSIIndicator(14))
	e.Register(NewMACDIndicator(12, 26, 9))
	e.Register(NewBollingerIndicator(20, 2.0))
	e.Register(NewStochasticIndicator(14, 3, 3))
	e.Register(NewCCIIndicator(20))
	e.Register(NewATRIndicator(14))
	e.Register(NewOBVIndicator(20))
	e.Register(NewSupertrendIndicator(10, 3.0))
	e.Register(NewWilliamsRIndicator(14))
	e.Register(NewMFIIndicator(14))
	e.Register(NewKeltnerIndicator(20, 10, 2.0))
	e.Register(NewDonchianIndicator(20))
	e.Register(NewElderRayIndicator(13))
	e.Register(NewForceIndexIndicator(13))
	e.Register(NewChaikinMFIndicator(20))
	e.Register(NewROCIndicator(12))
	e.Register(NewMomentumIndicator(10))
	e.Register(NewVWAPIndicator())
	e.Register(NewADIndicator(20))
	e.Register(NewIchimokuIndicator())
	e.Register(NewParabolicSARIndicator())
	e.Register(NewStochRSIIndicator(14, 14, 3, 3))
	e.Register(NewKSTIndicator())
	e.Register(NewStdDevIndicator(20))
	e.Register(NewUltimateOscillatorIndicator(7, 14, 28))
	e.Register(NewConnorsRSIIndicator(3, 2, 100))
}

// ReadyCount returns how many indicators are currently ready.
func (e *IndicatorEngine) ReadyCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	c := 0
	for _, s := range e.signals {
		if s.Ready {
			c++
		}
	}
	return c
}

// Reset clears the engine state and re-creates all indicators.
func (e *IndicatorEngine) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.signals = make(map[string]IndicatorSignal)
	e.barCount = 0
}

// StrengthSum returns the sum of absolute signal strengths.
func (e *IndicatorEngine) StrengthSum() float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	s := 0.0
	for _, sig := range e.signals {
		if sig.Ready {
			s += math.Abs(sig.Strength)
		}
	}
	return s
}

// CategoryConsensus returns consensus for a specific category.
func (e *IndicatorEngine) CategoryConsensus(cat string) float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	total := 0.0
	count := 0.0
	for _, s := range e.signals {
		if !s.Ready || s.Category != cat {
			continue
		}
		switch s.Signal {
		case SignalBuy:
			total += 1
			count++
		case SignalSell:
			total -= 1
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return total / count
}
