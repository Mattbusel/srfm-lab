package analytics

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

// RegimeType classifies a market regime.
type RegimeType int

const (
	RegimeUnknown       RegimeType = 0
	RegimeLowVol        RegimeType = 1
	RegimeNormalVol      RegimeType = 2
	RegimeHighVol        RegimeType = 3
	RegimeCrisisVol      RegimeType = 4
	RegimeUptrend        RegimeType = 10
	RegimeDowntrend      RegimeType = 11
	RegimeSideways       RegimeType = 12
	RegimeStrongUp       RegimeType = 13
	RegimeStrongDown     RegimeType = 14
	RegimeHighCorr       RegimeType = 20
	RegimeLowCorr        RegimeType = 21
	RegimeHerding        RegimeType = 22
	RegimeMomPositive    RegimeType = 30
	RegimeMomNegative    RegimeType = 31
	RegimeMomNeutral     RegimeType = 32
	RegimeHighLiquidity  RegimeType = 40
	RegimeLowLiquidity   RegimeType = 41
	RegimeNormalLiquidity RegimeType = 42
	RegimeExpansion      RegimeType = 50
	RegimeContraction    RegimeType = 51
	RegimeRiskOn         RegimeType = 52
	RegimeRiskOff        RegimeType = 53
)

// RegimeLabel returns a human-readable label for the regime.
func RegimeLabel(r RegimeType) string {
	switch r {
	case RegimeLowVol:
		return "low_volatility"
	case RegimeNormalVol:
		return "normal_volatility"
	case RegimeHighVol:
		return "high_volatility"
	case RegimeCrisisVol:
		return "crisis_volatility"
	case RegimeUptrend:
		return "uptrend"
	case RegimeDowntrend:
		return "downtrend"
	case RegimeSideways:
		return "sideways"
	case RegimeStrongUp:
		return "strong_uptrend"
	case RegimeStrongDown:
		return "strong_downtrend"
	case RegimeHighCorr:
		return "high_correlation"
	case RegimeLowCorr:
		return "low_correlation"
	case RegimeHerding:
		return "herding"
	case RegimeMomPositive:
		return "positive_momentum"
	case RegimeMomNegative:
		return "negative_momentum"
	case RegimeMomNeutral:
		return "neutral_momentum"
	case RegimeHighLiquidity:
		return "high_liquidity"
	case RegimeLowLiquidity:
		return "low_liquidity"
	case RegimeNormalLiquidity:
		return "normal_liquidity"
	case RegimeExpansion:
		return "expansion"
	case RegimeContraction:
		return "contraction"
	case RegimeRiskOn:
		return "risk_on"
	case RegimeRiskOff:
		return "risk_off"
	default:
		return "unknown"
	}
}

// RegimeResult carries a classification and confidence.
type RegimeResult struct {
	Regime     RegimeType
	Label      string
	Confidence float64
	Value      float64
	Timestamp  int64
}

// RegimeAlert signals a regime change event.
type RegimeAlert struct {
	From       RegimeType
	To         RegimeType
	Confidence float64
	Timestamp  int64
	Detector   string
}

// ---------------------------------------------------------------------------
// Ring buffer for analytics
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

func (r *ringBuf) sum() float64 {
	s := 0.0
	for i := 0; i < r.count; i++ {
		s += r.data[i]
	}
	return s
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

func (r *ringBuf) percentile(pct float64) float64 {
	if r.count == 0 {
		return 0
	}
	vals := r.values()
	sort.Float64s(vals)
	idx := pct / 100.0 * float64(len(vals)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi || hi >= len(vals) {
		return vals[lo]
	}
	frac := idx - float64(lo)
	return vals[lo]*(1-frac) + vals[hi]*frac
}

func (r *ringBuf) max() float64 {
	if r.count == 0 {
		return 0
	}
	m := -math.MaxFloat64
	for i := 0; i < r.count; i++ {
		if r.data[i] > m {
			m = r.data[i]
		}
	}
	return m
}

func (r *ringBuf) min() float64 {
	if r.count == 0 {
		return 0
	}
	m := math.MaxFloat64
	for i := 0; i < r.count; i++ {
		if r.data[i] < m {
			m = r.data[i]
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// VolatilityRegime: EWMA vol with 4 thresholds, historical percentile
// ---------------------------------------------------------------------------

// VolatilityRegime classifies market volatility into regimes.
type VolatilityRegime struct {
	ewmaLambda   float64
	ewmaVar      float64
	histBuf      *ringBuf
	returnBuf    *ringBuf
	prevPrice    float64
	count        int
	lowThresh    float64
	normalThresh float64
	highThresh   float64
	current      RegimeType
	primed       bool
}

// NewVolatilityRegime creates a vol regime detector.
// thresholds are percentiles: e.g. 25, 50, 75 for low/normal/high boundaries.
func NewVolatilityRegime(lambda float64, historyLen int, lowPct, normalPct, highPct float64) *VolatilityRegime {
	return &VolatilityRegime{
		ewmaLambda:   lambda,
		histBuf:      newRingBuf(historyLen),
		returnBuf:    newRingBuf(historyLen),
		lowThresh:    lowPct,
		normalThresh: normalPct,
		highThresh:   highPct,
		current:      RegimeUnknown,
	}
}

// NewDefaultVolatilityRegime creates with standard parameters.
func NewDefaultVolatilityRegime() *VolatilityRegime {
	return NewVolatilityRegime(0.94, 252, 25, 50, 75)
}

// Update processes a new price and returns the vol regime.
func (v *VolatilityRegime) Update(price float64, timestamp int64) RegimeResult {
	v.count++
	if v.count == 1 {
		v.prevPrice = price
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}
	}
	ret := math.Log(price / v.prevPrice)
	v.prevPrice = price
	v.returnBuf.push(ret)
	if !v.primed {
		v.ewmaVar = ret * ret
		v.primed = true
	} else {
		v.ewmaVar = v.ewmaLambda*v.ewmaVar + (1-v.ewmaLambda)*ret*ret
	}
	vol := math.Sqrt(v.ewmaVar * 252)
	v.histBuf.push(vol)
	if !v.histBuf.full() {
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Value: vol, Timestamp: timestamp}
	}
	p25 := v.histBuf.percentile(v.lowThresh)
	p50 := v.histBuf.percentile(v.normalThresh)
	p75 := v.histBuf.percentile(v.highThresh)
	var regime RegimeType
	var conf float64
	if vol <= p25 {
		regime = RegimeLowVol
		conf = 1 - vol/p25
	} else if vol <= p50 {
		regime = RegimeNormalVol
		conf = (vol - p25) / (p50 - p25)
	} else if vol <= p75 {
		regime = RegimeHighVol
		conf = (vol - p50) / (p75 - p50)
	} else {
		regime = RegimeCrisisVol
		conf = math.Min((vol-p75)/p75, 1.0)
	}
	v.current = regime
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: math.Abs(conf),
		Value:      vol,
		Timestamp:  timestamp,
	}
}

// Current returns the last classified regime.
func (v *VolatilityRegime) Current() RegimeType { return v.current }

// AnnualizedVol returns current annualized EWMA volatility.
func (v *VolatilityRegime) AnnualizedVol() float64 {
	return math.Sqrt(v.ewmaVar * 252)
}

// Percentile returns where current vol sits in historical distribution.
func (v *VolatilityRegime) Percentile() float64 {
	if !v.histBuf.full() {
		return 50
	}
	vol := v.AnnualizedVol()
	vals := v.histBuf.values()
	below := 0
	for _, vv := range vals {
		if vv < vol {
			below++
		}
	}
	return 100 * float64(below) / float64(len(vals))
}

// ---------------------------------------------------------------------------
// TrendRegime: dual MA crossover, ADX-like strength, linear regression R2
// ---------------------------------------------------------------------------

// TrendRegime detects trending vs sideways regimes.
type TrendRegime struct {
	fastPeriod int
	slowPeriod int
	adxPeriod  int
	fastBuf    *ringBuf
	slowBuf    *ringBuf
	priceBuf   *ringBuf
	trBuf      *ringBuf
	dmPlusBuf  *ringBuf
	dmMinusBuf *ringBuf
	prevBar    *barData
	count      int
	current    RegimeType
	adxSmooth  float64
	diPSmooth  float64
	diMSmooth  float64
}

type barData struct {
	high  float64
	low   float64
	close float64
}

// NewTrendRegime creates a trend regime detector.
func NewTrendRegime(fastPeriod, slowPeriod, adxPeriod int) *TrendRegime {
	return &TrendRegime{
		fastPeriod: fastPeriod,
		slowPeriod: slowPeriod,
		adxPeriod:  adxPeriod,
		fastBuf:    newRingBuf(fastPeriod),
		slowBuf:    newRingBuf(slowPeriod),
		priceBuf:   newRingBuf(slowPeriod),
		trBuf:      newRingBuf(adxPeriod),
		dmPlusBuf:  newRingBuf(adxPeriod),
		dmMinusBuf: newRingBuf(adxPeriod),
		current:    RegimeUnknown,
	}
}

// NewDefaultTrendRegime creates with standard parameters.
func NewDefaultTrendRegime() *TrendRegime {
	return NewTrendRegime(20, 50, 14)
}

// linearRegR2 computes R-squared of linear regression on values.
func linearRegR2(vals []float64) float64 {
	n := float64(len(vals))
	if n < 3 {
		return 0
	}
	sx, sy, sxy, sx2, sy2 := 0.0, 0.0, 0.0, 0.0, 0.0
	for i, v := range vals {
		x := float64(i)
		sx += x
		sy += v
		sxy += x * v
		sx2 += x * x
		sy2 += v * v
	}
	denom := (n*sx2 - sx*sx) * (n*sy2 - sy*sy)
	if denom <= 0 {
		return 0
	}
	r := (n*sxy - sx*sy) / math.Sqrt(denom)
	return r * r
}

// linearRegSlope computes slope of linear regression.
func linearRegSlope(vals []float64) float64 {
	n := float64(len(vals))
	if n < 2 {
		return 0
	}
	sx, sy, sxy, sx2 := 0.0, 0.0, 0.0, 0.0
	for i, v := range vals {
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
	return (n*sxy - sx*sy) / denom
}

// Update processes a new bar and returns trend regime.
func (t *TrendRegime) Update(high, low, close float64, timestamp int64) RegimeResult {
	t.count++
	t.fastBuf.push(close)
	t.slowBuf.push(close)
	t.priceBuf.push(close)
	// ADX computation
	if t.prevBar != nil {
		tr := math.Max(high-low, math.Max(math.Abs(high-t.prevBar.close), math.Abs(low-t.prevBar.close)))
		dmPlus := 0.0
		dmMinus := 0.0
		upMove := high - t.prevBar.high
		downMove := t.prevBar.low - low
		if upMove > downMove && upMove > 0 {
			dmPlus = upMove
		}
		if downMove > upMove && downMove > 0 {
			dmMinus = downMove
		}
		t.trBuf.push(tr)
		t.dmPlusBuf.push(dmPlus)
		t.dmMinusBuf.push(dmMinus)
	}
	t.prevBar = &barData{high: high, low: low, close: close}
	if !t.slowBuf.full() {
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}
	}
	fastMA := t.fastBuf.mean()
	slowMA := t.slowBuf.mean()
	// ADX
	adx := 0.0
	if t.trBuf.full() {
		trSum := t.trBuf.sum()
		if trSum > 0 {
			diP := 100 * t.dmPlusBuf.sum() / trSum
			diM := 100 * t.dmMinusBuf.sum() / trSum
			diSum := diP + diM
			if diSum > 0 {
				dx := 100 * math.Abs(diP-diM) / diSum
				if t.count > t.adxPeriod+t.slowPeriod {
					t.adxSmooth = (t.adxSmooth*float64(t.adxPeriod-1) + dx) / float64(t.adxPeriod)
				} else {
					t.adxSmooth = dx
				}
				adx = t.adxSmooth
			}
			t.diPSmooth = diP
			t.diMSmooth = diM
		}
	}
	// R2
	r2 := linearRegR2(t.priceBuf.values())
	slope := linearRegSlope(t.priceBuf.values())
	// Classification
	var regime RegimeType
	var conf float64
	if adx > 25 && r2 > 0.6 {
		if slope > 0 {
			if adx > 40 {
				regime = RegimeStrongUp
				conf = math.Min(adx/50, 1)
			} else {
				regime = RegimeUptrend
				conf = adx / 40
			}
		} else {
			if adx > 40 {
				regime = RegimeStrongDown
				conf = math.Min(adx/50, 1)
			} else {
				regime = RegimeDowntrend
				conf = adx / 40
			}
		}
	} else if fastMA > slowMA && slope > 0 {
		regime = RegimeUptrend
		conf = 0.5
	} else if fastMA < slowMA && slope < 0 {
		regime = RegimeDowntrend
		conf = 0.5
	} else {
		regime = RegimeSideways
		conf = 1 - r2
	}
	t.current = regime
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: conf,
		Value:      adx,
		Timestamp:  timestamp,
	}
}

// ADX returns the current ADX value.
func (t *TrendRegime) ADX() float64 { return t.adxSmooth }

// R2 returns the R-squared of the price series.
func (t *TrendRegime) R2() float64 {
	if !t.priceBuf.full() {
		return 0
	}
	return linearRegR2(t.priceBuf.values())
}

// Current returns the current trend regime.
func (t *TrendRegime) Current() RegimeType { return t.current }

// ---------------------------------------------------------------------------
// CorrelationRegime: rolling pairwise correlation, herding detection
// ---------------------------------------------------------------------------

// CorrelationRegime tracks pairwise correlations among assets.
type CorrelationRegime struct {
	period     int
	assets     []string
	returnBufs map[string]*ringBuf
	corrMatrix map[string]float64
	avgCorr    float64
	count      int
	current    RegimeType
	herdThresh float64
	lowThresh  float64
}

// NewCorrelationRegime creates a correlation regime detector.
func NewCorrelationRegime(period int, assets []string, herdingThreshold, lowThreshold float64) *CorrelationRegime {
	bufs := make(map[string]*ringBuf, len(assets))
	for _, a := range assets {
		bufs[a] = newRingBuf(period)
	}
	return &CorrelationRegime{
		period:     period,
		assets:     assets,
		returnBufs: bufs,
		corrMatrix: make(map[string]float64),
		herdThresh: herdingThreshold,
		lowThresh:  lowThreshold,
		current:    RegimeUnknown,
	}
}

func corrKey(a, b string) string {
	if a < b {
		return a + "|" + b
	}
	return b + "|" + a
}

func correlation(xs, ys []float64) float64 {
	n := len(xs)
	if n < 3 || n != len(ys) {
		return 0
	}
	mx, my := 0.0, 0.0
	for i := 0; i < n; i++ {
		mx += xs[i]
		my += ys[i]
	}
	mx /= float64(n)
	my /= float64(n)
	cov, vx, vy := 0.0, 0.0, 0.0
	for i := 0; i < n; i++ {
		dx := xs[i] - mx
		dy := ys[i] - my
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

// UpdateReturns feeds returns for each asset. returns map[asset]logReturn.
func (c *CorrelationRegime) UpdateReturns(returns map[string]float64, timestamp int64) RegimeResult {
	c.count++
	for _, a := range c.assets {
		if r, ok := returns[a]; ok {
			c.returnBufs[a].push(r)
		}
	}
	ready := true
	for _, a := range c.assets {
		if !c.returnBufs[a].full() {
			ready = false
			break
		}
	}
	if !ready {
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}
	}
	// Compute pairwise correlations
	totalCorr := 0.0
	pairs := 0
	for i := 0; i < len(c.assets); i++ {
		for j := i + 1; j < len(c.assets); j++ {
			xs := c.returnBufs[c.assets[i]].values()
			ys := c.returnBufs[c.assets[j]].values()
			corr := correlation(xs, ys)
			key := corrKey(c.assets[i], c.assets[j])
			c.corrMatrix[key] = corr
			totalCorr += corr
			pairs++
		}
	}
	if pairs > 0 {
		c.avgCorr = totalCorr / float64(pairs)
	}
	var regime RegimeType
	var conf float64
	if c.avgCorr >= c.herdThresh {
		regime = RegimeHerding
		conf = math.Min((c.avgCorr-c.herdThresh)/(1-c.herdThresh), 1)
	} else if c.avgCorr >= (c.herdThresh+c.lowThresh)/2 {
		regime = RegimeHighCorr
		conf = (c.avgCorr - c.lowThresh) / (c.herdThresh - c.lowThresh)
	} else {
		regime = RegimeLowCorr
		conf = 1 - c.avgCorr/c.lowThresh
	}
	c.current = regime
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: math.Abs(conf),
		Value:      c.avgCorr,
		Timestamp:  timestamp,
	}
}

// AverageCorrelation returns the current average pairwise correlation.
func (c *CorrelationRegime) AverageCorrelation() float64 { return c.avgCorr }

// PairCorrelation returns correlation between two assets.
func (c *CorrelationRegime) PairCorrelation(a, b string) float64 {
	return c.corrMatrix[corrKey(a, b)]
}

// Current returns the current correlation regime.
func (c *CorrelationRegime) Current() RegimeType { return c.current }

// ---------------------------------------------------------------------------
// MomentumRegime: 12-1 month momentum, cross-sectional rank
// ---------------------------------------------------------------------------

// MomentumRegime classifies momentum conditions.
type MomentumRegime struct {
	longPeriod  int
	skipPeriod  int
	priceBufs   map[string]*ringBuf
	assets      []string
	count       int
	current     RegimeType
}

// NewMomentumRegime creates a momentum regime detector.
func NewMomentumRegime(longPeriod, skipPeriod int, assets []string) *MomentumRegime {
	bufs := make(map[string]*ringBuf, len(assets))
	for _, a := range assets {
		bufs[a] = newRingBuf(longPeriod + 1)
	}
	return &MomentumRegime{
		longPeriod: longPeriod,
		skipPeriod: skipPeriod,
		priceBufs:  bufs,
		assets:     assets,
		current:    RegimeUnknown,
	}
}

// NewDefaultMomentumRegime uses 252d lookback, 21d skip.
func NewDefaultMomentumRegime(assets []string) *MomentumRegime {
	return NewMomentumRegime(252, 21, assets)
}

// MomentumScore holds a single asset's momentum score and rank.
type MomentumScore struct {
	Asset    string
	Momentum float64
	Rank     int
}

// UpdatePrices feeds prices for all assets and returns regime.
func (m *MomentumRegime) UpdatePrices(prices map[string]float64, timestamp int64) (RegimeResult, []MomentumScore) {
	m.count++
	for _, a := range m.assets {
		if p, ok := prices[a]; ok {
			m.priceBufs[a].push(p)
		}
	}
	ready := true
	for _, a := range m.assets {
		if m.priceBufs[a].count < m.longPeriod+1 {
			ready = false
			break
		}
	}
	if !ready {
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}, nil
	}
	scores := make([]MomentumScore, 0, len(m.assets))
	totalMom := 0.0
	for _, a := range m.assets {
		buf := m.priceBufs[a]
		current := buf.get(m.skipPeriod)
		past := buf.get(m.longPeriod)
		mom := 0.0
		if past > 0 {
			mom = current/past - 1
		}
		scores = append(scores, MomentumScore{Asset: a, Momentum: mom})
		totalMom += mom
	}
	// Sort and rank
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Momentum > scores[j].Momentum
	})
	for i := range scores {
		scores[i].Rank = i + 1
	}
	avgMom := totalMom / float64(len(m.assets))
	var regime RegimeType
	var conf float64
	if avgMom > 0.1 {
		regime = RegimeMomPositive
		conf = math.Min(avgMom/0.2, 1)
	} else if avgMom < -0.1 {
		regime = RegimeMomNegative
		conf = math.Min(math.Abs(avgMom)/0.2, 1)
	} else {
		regime = RegimeMomNeutral
		conf = 1 - math.Abs(avgMom)/0.1
	}
	m.current = regime
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: conf,
		Value:      avgMom,
		Timestamp:  timestamp,
	}, scores
}

// Current returns the current momentum regime.
func (m *MomentumRegime) Current() RegimeType { return m.current }

// ---------------------------------------------------------------------------
// LiquidityRegime: spread, depth, volume relative to history
// ---------------------------------------------------------------------------

// LiquidityRegime classifies market liquidity conditions.
type LiquidityRegime struct {
	period     int
	spreadBuf  *ringBuf
	depthBuf   *ringBuf
	volumeBuf  *ringBuf
	count      int
	current    RegimeType
	lowPct     float64
	highPct    float64
}

// NewLiquidityRegime creates a liquidity regime detector.
func NewLiquidityRegime(period int, lowPct, highPct float64) *LiquidityRegime {
	return &LiquidityRegime{
		period:    period,
		spreadBuf: newRingBuf(period),
		depthBuf:  newRingBuf(period),
		volumeBuf: newRingBuf(period),
		lowPct:    lowPct,
		highPct:   highPct,
		current:   RegimeUnknown,
	}
}

// NewDefaultLiquidityRegime creates with standard parameters.
func NewDefaultLiquidityRegime() *LiquidityRegime {
	return NewLiquidityRegime(252, 25, 75)
}

// LiquidityData holds raw liquidity metrics for a single observation.
type LiquidityData struct {
	Spread float64
	Depth  float64
	Volume float64
}

// Update processes new liquidity data.
func (l *LiquidityRegime) Update(data LiquidityData, timestamp int64) RegimeResult {
	l.count++
	l.spreadBuf.push(data.Spread)
	l.depthBuf.push(data.Depth)
	l.volumeBuf.push(data.Volume)
	if !l.spreadBuf.full() {
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}
	}
	// Composite score: low spread + high depth + high volume = high liquidity
	spreadPct := l.spreadBuf.percentile(50)
	depthPct := l.depthBuf.percentile(50)
	volPct := l.volumeBuf.percentile(50)
	// Normalize: spread is inverse (lower = better)
	spreadScore := 0.0
	if spreadPct > 0 {
		spreadScore = 1 - data.Spread/spreadPct
		if spreadScore < -1 {
			spreadScore = -1
		}
	}
	depthScore := 0.0
	if depthPct > 0 {
		depthScore = data.Depth/depthPct - 1
	}
	volScore := 0.0
	if volPct > 0 {
		volScore = data.Volume/volPct - 1
	}
	composite := (spreadScore + depthScore + volScore) / 3
	var regime RegimeType
	var conf float64
	if composite > 0.3 {
		regime = RegimeHighLiquidity
		conf = math.Min(composite, 1)
	} else if composite < -0.3 {
		regime = RegimeLowLiquidity
		conf = math.Min(math.Abs(composite), 1)
	} else {
		regime = RegimeNormalLiquidity
		conf = 1 - math.Abs(composite)/0.3
	}
	l.current = regime
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: conf,
		Value:      composite,
		Timestamp:  timestamp,
	}
}

// Current returns the current liquidity regime.
func (l *LiquidityRegime) Current() RegimeType { return l.current }

// ---------------------------------------------------------------------------
// MacroRegime: yield curve, credit spread, VIX level mapping
// ---------------------------------------------------------------------------

// MacroRegime maps macro indicators to market regimes.
type MacroRegime struct {
	yieldCurveBuf  *ringBuf
	creditSpreadBuf *ringBuf
	vixBuf         *ringBuf
	period         int
	count          int
	current        RegimeType
}

// NewMacroRegime creates a macro regime detector.
func NewMacroRegime(period int) *MacroRegime {
	return &MacroRegime{
		yieldCurveBuf:   newRingBuf(period),
		creditSpreadBuf: newRingBuf(period),
		vixBuf:          newRingBuf(period),
		period:          period,
		current:         RegimeUnknown,
	}
}

// NewDefaultMacroRegime creates with a 252-day window.
func NewDefaultMacroRegime() *MacroRegime {
	return NewMacroRegime(252)
}

// MacroData holds macro indicator values.
type MacroData struct {
	YieldCurveSlope float64 // 10y - 2y spread
	CreditSpread    float64 // HY - IG spread
	VIX             float64
}

// Update processes macro data and returns a macro regime.
func (m *MacroRegime) Update(data MacroData, timestamp int64) RegimeResult {
	m.count++
	m.yieldCurveBuf.push(data.YieldCurveSlope)
	m.creditSpreadBuf.push(data.CreditSpread)
	m.vixBuf.push(data.VIX)
	if !m.yieldCurveBuf.full() {
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}
	}
	// Score components
	ycMean := m.yieldCurveBuf.mean()
	csMean := m.creditSpreadBuf.mean()
	csStd := m.creditSpreadBuf.stddev()
	vixMean := m.vixBuf.mean()
	// Yield curve: positive = expansion, inverted = contraction
	ycScore := 0.0
	if data.YieldCurveSlope > 0 {
		ycScore = 1
	} else {
		ycScore = -1
	}
	_ = ycMean
	// Credit spread: widening = risk off
	csZ := 0.0
	if csStd > 0 {
		csZ = (data.CreditSpread - csMean) / csStd
	}
	csScore := -csZ // higher spread = more risk-off
	// VIX
	vixScore := 0.0
	if data.VIX < 15 {
		vixScore = 1
	} else if data.VIX < 20 {
		vixScore = 0.5
	} else if data.VIX < 30 {
		vixScore = -0.5
	} else {
		vixScore = -1
	}
	_ = vixMean
	composite := (ycScore + csScore + vixScore) / 3
	var regime RegimeType
	var conf float64
	if composite > 0.3 {
		if ycScore > 0 {
			regime = RegimeExpansion
		} else {
			regime = RegimeRiskOn
		}
		conf = math.Min(composite, 1)
	} else if composite < -0.3 {
		if ycScore < 0 {
			regime = RegimeContraction
		} else {
			regime = RegimeRiskOff
		}
		conf = math.Min(math.Abs(composite), 1)
	} else {
		regime = RegimeExpansion
		conf = 0.5
	}
	m.current = regime
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: conf,
		Value:      composite,
		Timestamp:  timestamp,
	}
}

// Current returns the current macro regime.
func (m *MacroRegime) Current() RegimeType { return m.current }

// ---------------------------------------------------------------------------
// RegimeEnsemble: weighted vote, confidence, hysteresis
// ---------------------------------------------------------------------------

// RegimeVote is a single vote from a detector.
type RegimeVote struct {
	Source     string
	Regime     RegimeType
	Confidence float64
	Weight     float64
}

// EnsembleResult holds the ensemble classification.
type EnsembleResult struct {
	Regime      RegimeType
	Label       string
	Confidence  float64
	Votes       []RegimeVote
	Confirmations int
	Timestamp   int64
}

// RegimeEnsemble combines multiple regime detectors with weighted voting.
type RegimeEnsemble struct {
	mu              sync.RWMutex
	votes           map[string]RegimeVote
	weights         map[string]float64
	hysteresisCount int
	confirmNeeded   int
	currentRegime   RegimeType
	pendingRegime   RegimeType
	pendingCount    int
}

// NewRegimeEnsemble creates an ensemble.
func NewRegimeEnsemble(confirmNeeded int) *RegimeEnsemble {
	return &RegimeEnsemble{
		votes:         make(map[string]RegimeVote),
		weights:       make(map[string]float64),
		confirmNeeded: confirmNeeded,
		currentRegime: RegimeUnknown,
	}
}

// SetWeight sets the weight for a detector.
func (e *RegimeEnsemble) SetWeight(source string, weight float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.weights[source] = weight
}

// SubmitVote submits a vote from a detector.
func (e *RegimeEnsemble) SubmitVote(vote RegimeVote) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if w, ok := e.weights[vote.Source]; ok {
		vote.Weight = w
	} else {
		vote.Weight = 1.0
	}
	e.votes[vote.Source] = vote
}

// Evaluate computes the ensemble result with hysteresis.
func (e *RegimeEnsemble) Evaluate(timestamp int64) EnsembleResult {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.votes) == 0 {
		return EnsembleResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}
	}
	// Weighted vote tally
	regimeScores := make(map[RegimeType]float64)
	totalWeight := 0.0
	var allVotes []RegimeVote
	for _, v := range e.votes {
		score := v.Weight * v.Confidence
		regimeScores[v.Regime] += score
		totalWeight += v.Weight
		allVotes = append(allVotes, v)
	}
	// Find winner
	bestRegime := RegimeUnknown
	bestScore := 0.0
	for r, s := range regimeScores {
		if s > bestScore {
			bestScore = s
			bestRegime = r
		}
	}
	confidence := 0.0
	if totalWeight > 0 {
		confidence = bestScore / totalWeight
	}
	// Hysteresis
	if bestRegime != e.currentRegime {
		if bestRegime == e.pendingRegime {
			e.pendingCount++
		} else {
			e.pendingRegime = bestRegime
			e.pendingCount = 1
		}
		if e.pendingCount >= e.confirmNeeded {
			e.currentRegime = bestRegime
			e.pendingCount = 0
		}
	} else {
		e.pendingCount = 0
		e.pendingRegime = RegimeUnknown
	}
	return EnsembleResult{
		Regime:        e.currentRegime,
		Label:         RegimeLabel(e.currentRegime),
		Confidence:    confidence,
		Votes:         allVotes,
		Confirmations: e.pendingCount,
		Timestamp:     timestamp,
	}
}

// Current returns the current ensemble regime.
func (e *RegimeEnsemble) Current() RegimeType {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.currentRegime
}

// ---------------------------------------------------------------------------
// MarkovTransition: transition matrix, expected duration, forecasting
// ---------------------------------------------------------------------------

// MarkovTransition models regime transitions as a Markov chain.
type MarkovTransition struct {
	mu          sync.RWMutex
	states      []RegimeType
	stateIndex  map[RegimeType]int
	transitions [][]int
	counts      []int
	current     RegimeType
	history     []RegimeType
	durations   map[RegimeType][]int
	durCount    int
}

// NewMarkovTransition creates a Markov transition model for given states.
func NewMarkovTransition(states []RegimeType) *MarkovTransition {
	n := len(states)
	idx := make(map[RegimeType]int, n)
	for i, s := range states {
		idx[s] = i
	}
	trans := make([][]int, n)
	for i := range trans {
		trans[i] = make([]int, n)
	}
	return &MarkovTransition{
		states:      states,
		stateIndex:  idx,
		transitions: trans,
		counts:      make([]int, n),
		current:     RegimeUnknown,
		durations:   make(map[RegimeType][]int),
	}
}

// Observe records a regime observation.
func (m *MarkovTransition) Observe(regime RegimeType) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.history = append(m.history, regime)
	if _, ok := m.stateIndex[regime]; !ok {
		return
	}
	idx := m.stateIndex[regime]
	m.counts[idx]++
	if regime != m.current && m.current != RegimeUnknown {
		if prevIdx, ok := m.stateIndex[m.current]; ok {
			m.transitions[prevIdx][idx]++
		}
		if m.durCount > 0 {
			m.durations[m.current] = append(m.durations[m.current], m.durCount)
		}
		m.durCount = 1
	} else {
		m.durCount++
	}
	m.current = regime
}

// TransitionMatrix returns the transition probability matrix.
func (m *MarkovTransition) TransitionMatrix() [][]float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	n := len(m.states)
	mat := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, n)
		rowSum := 0
		for j := 0; j < n; j++ {
			rowSum += m.transitions[i][j]
		}
		if rowSum > 0 {
			for j := 0; j < n; j++ {
				mat[i][j] = float64(m.transitions[i][j]) / float64(rowSum)
			}
		}
	}
	return mat
}

// ExpectedDuration returns the average duration of a regime in observations.
func (m *MarkovTransition) ExpectedDuration(regime RegimeType) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	durs := m.durations[regime]
	if len(durs) == 0 {
		return 0
	}
	s := 0.0
	for _, d := range durs {
		s += float64(d)
	}
	return s / float64(len(durs))
}

// Forecast returns the probability distribution of next regime.
func (m *MarkovTransition) Forecast() map[RegimeType]float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make(map[RegimeType]float64)
	if m.current == RegimeUnknown {
		return result
	}
	idx, ok := m.stateIndex[m.current]
	if !ok {
		return result
	}
	mat := m.TransitionMatrix()
	for j, s := range m.states {
		result[s] = mat[idx][j]
	}
	return result
}

// ForecastN returns the probability distribution N steps ahead.
func (m *MarkovTransition) ForecastN(steps int) map[RegimeType]float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make(map[RegimeType]float64)
	if m.current == RegimeUnknown {
		return result
	}
	idx, ok := m.stateIndex[m.current]
	if !ok {
		return result
	}
	mat := m.TransitionMatrix()
	n := len(m.states)
	// Matrix power by repeated multiplication
	current := make([]float64, n)
	current[idx] = 1.0
	for s := 0; s < steps; s++ {
		next := make([]float64, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				next[j] += current[i] * mat[i][j]
			}
		}
		current = next
	}
	for j, st := range m.states {
		result[st] = current[j]
	}
	return result
}

// History returns the full regime history.
func (m *MarkovTransition) History() []RegimeType {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]RegimeType, len(m.history))
	copy(out, m.history)
	return out
}

// ---------------------------------------------------------------------------
// RegimeConditionalStats: per-regime return, vol, Sharpe, max DD
// ---------------------------------------------------------------------------

// ConditionalStats holds statistics conditional on a regime.
type ConditionalStats struct {
	Regime       RegimeType
	Label        string
	Count        int
	MeanReturn   float64
	Volatility   float64
	Sharpe       float64
	MaxDrawdown  float64
	TotalReturn  float64
	WinRate      float64
}

// RegimeConditionalStats computes statistics per regime.
type RegimeConditionalStats struct {
	mu         sync.RWMutex
	regimeData map[RegimeType][]float64
	riskFree   float64
}

// NewRegimeConditionalStats creates a conditional stats tracker.
func NewRegimeConditionalStats(riskFreeRate float64) *RegimeConditionalStats {
	return &RegimeConditionalStats{
		regimeData: make(map[RegimeType][]float64),
		riskFree:   riskFreeRate,
	}
}

// Record adds a return observation for a regime.
func (r *RegimeConditionalStats) Record(regime RegimeType, ret float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.regimeData[regime] = append(r.regimeData[regime], ret)
}

// Compute returns conditional statistics for all regimes.
func (r *RegimeConditionalStats) Compute() map[RegimeType]ConditionalStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make(map[RegimeType]ConditionalStats)
	for regime, returns := range r.regimeData {
		n := len(returns)
		if n == 0 {
			continue
		}
		// Mean
		sum := 0.0
		for _, v := range returns {
			sum += v
		}
		mean := sum / float64(n)
		// Vol
		ss := 0.0
		for _, v := range returns {
			d := v - mean
			ss += d * d
		}
		vol := 0.0
		if n > 1 {
			vol = math.Sqrt(ss / float64(n-1))
		}
		// Sharpe
		sharpe := 0.0
		if vol > 0 {
			sharpe = (mean*252 - r.riskFree) / (vol * math.Sqrt(252))
		}
		// Max drawdown
		cumRet := 0.0
		peak := 0.0
		maxDD := 0.0
		for _, v := range returns {
			cumRet += v
			if cumRet > peak {
				peak = cumRet
			}
			dd := peak - cumRet
			if dd > maxDD {
				maxDD = dd
			}
		}
		// Win rate
		wins := 0
		for _, v := range returns {
			if v > 0 {
				wins++
			}
		}
		winRate := float64(wins) / float64(n)
		// Total return
		totalRet := math.Exp(sum) - 1
		result[regime] = ConditionalStats{
			Regime:      regime,
			Label:       RegimeLabel(regime),
			Count:       n,
			MeanReturn:  mean,
			Volatility:  vol,
			Sharpe:      sharpe,
			MaxDrawdown: maxDD,
			TotalReturn: totalRet,
			WinRate:     winRate,
		}
	}
	return result
}

// Stats returns conditional statistics for a specific regime.
func (r *RegimeConditionalStats) Stats(regime RegimeType) (ConditionalStats, bool) {
	all := r.Compute()
	s, ok := all[regime]
	return s, ok
}

// ---------------------------------------------------------------------------
// RegimeAlerts: regime change notification with confidence
// ---------------------------------------------------------------------------

// RegimeAlertSystem monitors regime changes and emits alerts.
type RegimeAlertSystem struct {
	mu           sync.RWMutex
	currentMap   map[string]RegimeType
	alerts       []RegimeAlert
	minConfidence float64
	maxAlerts    int
}

// NewRegimeAlertSystem creates an alert system.
func NewRegimeAlertSystem(minConfidence float64, maxAlerts int) *RegimeAlertSystem {
	return &RegimeAlertSystem{
		currentMap:    make(map[string]RegimeType),
		alerts:        make([]RegimeAlert, 0, maxAlerts),
		minConfidence: minConfidence,
		maxAlerts:     maxAlerts,
	}
}

// Check evaluates a regime result and emits an alert if changed.
func (r *RegimeAlertSystem) Check(detector string, result RegimeResult) *RegimeAlert {
	r.mu.Lock()
	defer r.mu.Unlock()
	prev, exists := r.currentMap[detector]
	r.currentMap[detector] = result.Regime
	if !exists || prev == result.Regime {
		return nil
	}
	if result.Confidence < r.minConfidence {
		return nil
	}
	alert := RegimeAlert{
		From:       prev,
		To:         result.Regime,
		Confidence: result.Confidence,
		Timestamp:  result.Timestamp,
		Detector:   detector,
	}
	r.alerts = append(r.alerts, alert)
	if len(r.alerts) > r.maxAlerts {
		r.alerts = r.alerts[len(r.alerts)-r.maxAlerts:]
	}
	return &alert
}

// RecentAlerts returns the last N alerts.
func (r *RegimeAlertSystem) RecentAlerts(n int) []RegimeAlert {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if n > len(r.alerts) {
		n = len(r.alerts)
	}
	out := make([]RegimeAlert, n)
	copy(out, r.alerts[len(r.alerts)-n:])
	return out
}

// AlertCount returns total alerts emitted.
func (r *RegimeAlertSystem) AlertCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.alerts)
}

// CurrentRegimes returns the current regime for each detector.
func (r *RegimeAlertSystem) CurrentRegimes() map[string]RegimeType {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]RegimeType, len(r.currentMap))
	for k, v := range r.currentMap {
		out[k] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Full regime detection pipeline
// ---------------------------------------------------------------------------

// RegimeDetector is a full pipeline combining all detectors.
type RegimeDetector struct {
	mu        sync.RWMutex
	vol       *VolatilityRegime
	trend     *TrendRegime
	corr      *CorrelationRegime
	mom       *MomentumRegime
	liq       *LiquidityRegime
	macro     *MacroRegime
	ensemble  *RegimeEnsemble
	markov    *MarkovTransition
	condStats *RegimeConditionalStats
	alerts    *RegimeAlertSystem
}

// NewRegimeDetector creates a full regime detection pipeline.
func NewRegimeDetector(assets []string) *RegimeDetector {
	states := []RegimeType{
		RegimeLowVol, RegimeNormalVol, RegimeHighVol, RegimeCrisisVol,
		RegimeUptrend, RegimeDowntrend, RegimeSideways,
		RegimeRiskOn, RegimeRiskOff,
	}
	rd := &RegimeDetector{
		vol:       NewDefaultVolatilityRegime(),
		trend:     NewDefaultTrendRegime(),
		mom:       NewDefaultMomentumRegime(assets),
		liq:       NewDefaultLiquidityRegime(),
		macro:     NewDefaultMacroRegime(),
		ensemble:  NewRegimeEnsemble(3),
		markov:    NewMarkovTransition(states),
		condStats: NewRegimeConditionalStats(0.02),
		alerts:    NewRegimeAlertSystem(0.5, 1000),
	}
	if len(assets) > 1 {
		rd.corr = NewCorrelationRegime(60, assets, 0.7, 0.3)
	}
	rd.ensemble.SetWeight("volatility", 1.5)
	rd.ensemble.SetWeight("trend", 2.0)
	rd.ensemble.SetWeight("correlation", 1.0)
	rd.ensemble.SetWeight("momentum", 1.0)
	rd.ensemble.SetWeight("liquidity", 0.5)
	rd.ensemble.SetWeight("macro", 1.0)
	return rd
}

// MarketBar holds all data needed for a full regime update.
type MarketBar struct {
	Timestamp   int64
	Prices      map[string]float64
	Returns     map[string]float64
	PrimaryHigh float64
	PrimaryLow  float64
	PrimaryClose float64
	PrimaryPrice float64
	Liquidity   LiquidityData
	Macro       MacroData
}

// Update processes a full market bar through all detectors.
func (rd *RegimeDetector) Update(bar MarketBar) (EnsembleResult, []RegimeAlert) {
	rd.mu.Lock()
	defer rd.mu.Unlock()
	var newAlerts []RegimeAlert
	// Volatility
	vr := rd.vol.Update(bar.PrimaryPrice, bar.Timestamp)
	rd.ensemble.SubmitVote(RegimeVote{Source: "volatility", Regime: vr.Regime, Confidence: vr.Confidence})
	if a := rd.alerts.Check("volatility", vr); a != nil {
		newAlerts = append(newAlerts, *a)
	}
	// Trend
	tr := rd.trend.Update(bar.PrimaryHigh, bar.PrimaryLow, bar.PrimaryClose, bar.Timestamp)
	rd.ensemble.SubmitVote(RegimeVote{Source: "trend", Regime: tr.Regime, Confidence: tr.Confidence})
	if a := rd.alerts.Check("trend", tr); a != nil {
		newAlerts = append(newAlerts, *a)
	}
	// Correlation
	if rd.corr != nil && len(bar.Returns) > 0 {
		cr := rd.corr.UpdateReturns(bar.Returns, bar.Timestamp)
		rd.ensemble.SubmitVote(RegimeVote{Source: "correlation", Regime: cr.Regime, Confidence: cr.Confidence})
		if a := rd.alerts.Check("correlation", cr); a != nil {
			newAlerts = append(newAlerts, *a)
		}
	}
	// Momentum
	if len(bar.Prices) > 0 {
		mr, _ := rd.mom.UpdatePrices(bar.Prices, bar.Timestamp)
		rd.ensemble.SubmitVote(RegimeVote{Source: "momentum", Regime: mr.Regime, Confidence: mr.Confidence})
		if a := rd.alerts.Check("momentum", mr); a != nil {
			newAlerts = append(newAlerts, *a)
		}
	}
	// Liquidity
	lr := rd.liq.Update(bar.Liquidity, bar.Timestamp)
	rd.ensemble.SubmitVote(RegimeVote{Source: "liquidity", Regime: lr.Regime, Confidence: lr.Confidence})
	if a := rd.alerts.Check("liquidity", lr); a != nil {
		newAlerts = append(newAlerts, *a)
	}
	// Macro
	macr := rd.macro.Update(bar.Macro, bar.Timestamp)
	rd.ensemble.SubmitVote(RegimeVote{Source: "macro", Regime: macr.Regime, Confidence: macr.Confidence})
	if a := rd.alerts.Check("macro", macr); a != nil {
		newAlerts = append(newAlerts, *a)
	}
	// Ensemble
	ensResult := rd.ensemble.Evaluate(bar.Timestamp)
	// Track in Markov
	rd.markov.Observe(ensResult.Regime)
	return ensResult, newAlerts
}

// Forecast returns the Markov probability forecast.
func (rd *RegimeDetector) Forecast(steps int) map[RegimeType]float64 {
	rd.mu.RLock()
	defer rd.mu.RUnlock()
	return rd.markov.ForecastN(steps)
}

// ConditionalStats returns per-regime statistics.
func (rd *RegimeDetector) ConditionalStats() map[RegimeType]ConditionalStats {
	return rd.condStats.Compute()
}

// RecentAlerts returns the latest alerts.
func (rd *RegimeDetector) RecentAlerts(n int) []RegimeAlert {
	return rd.alerts.RecentAlerts(n)
}

// TransitionMatrix returns the Markov transition matrix.
func (rd *RegimeDetector) TransitionMatrix() [][]float64 {
	return rd.markov.TransitionMatrix()
}

// ExpectedDuration returns expected regime duration.
func (rd *RegimeDetector) ExpectedDuration(regime RegimeType) float64 {
	return rd.markov.ExpectedDuration(regime)
}

// String returns a summary string of the current state.
func (rd *RegimeDetector) String() string {
	rd.mu.RLock()
	defer rd.mu.RUnlock()
	return fmt.Sprintf("RegimeDetector[vol=%s trend=%s ensemble=%s]",
		RegimeLabel(rd.vol.Current()),
		RegimeLabel(rd.trend.Current()),
		RegimeLabel(rd.ensemble.Current()),
	)
}

// ---------------------------------------------------------------------------
// RegimeHistory: tracks regime history with timestamps for analysis
// ---------------------------------------------------------------------------

// RegimeObservation is a single timestamped regime observation.
type RegimeObservation struct {
	Timestamp  int64
	Regime     RegimeType
	Confidence float64
	Value      float64
}

// RegimeHistory stores a complete history of regime classifications.
type RegimeHistory struct {
	mu           sync.RWMutex
	observations []RegimeObservation
	maxLen       int
}

// NewRegimeHistory creates a new regime history tracker.
func NewRegimeHistory(maxLen int) *RegimeHistory {
	return &RegimeHistory{
		observations: make([]RegimeObservation, 0, maxLen),
		maxLen:       maxLen,
	}
}

// Record adds an observation.
func (h *RegimeHistory) Record(obs RegimeObservation) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.observations = append(h.observations, obs)
	if len(h.observations) > h.maxLen {
		h.observations = h.observations[len(h.observations)-h.maxLen:]
	}
}

// Last returns the most recent N observations.
func (h *RegimeHistory) Last(n int) []RegimeObservation {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if n > len(h.observations) {
		n = len(h.observations)
	}
	out := make([]RegimeObservation, n)
	copy(out, h.observations[len(h.observations)-n:])
	return out
}

// Count returns total observations.
func (h *RegimeHistory) Count() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.observations)
}

// Duration returns the number of consecutive observations of the current regime.
func (h *RegimeHistory) Duration() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if len(h.observations) == 0 {
		return 0
	}
	current := h.observations[len(h.observations)-1].Regime
	count := 0
	for i := len(h.observations) - 1; i >= 0; i-- {
		if h.observations[i].Regime != current {
			break
		}
		count++
	}
	return count
}

// RegimeDistribution returns the frequency distribution of regimes.
func (h *RegimeHistory) RegimeDistribution() map[RegimeType]float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	dist := make(map[RegimeType]float64)
	if len(h.observations) == 0 {
		return dist
	}
	for _, obs := range h.observations {
		dist[obs.Regime]++
	}
	total := float64(len(h.observations))
	for k := range dist {
		dist[k] /= total
	}
	return dist
}

// TransitionsCount counts transitions between regimes.
func (h *RegimeHistory) TransitionsCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	count := 0
	for i := 1; i < len(h.observations); i++ {
		if h.observations[i].Regime != h.observations[i-1].Regime {
			count++
		}
	}
	return count
}

// AvgConfidence returns the average confidence across all observations.
func (h *RegimeHistory) AvgConfidence() float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if len(h.observations) == 0 {
		return 0
	}
	sum := 0.0
	for _, obs := range h.observations {
		sum += obs.Confidence
	}
	return sum / float64(len(h.observations))
}

// ---------------------------------------------------------------------------
// VolatilityTermStructure: multi-horizon vol analysis
// ---------------------------------------------------------------------------

// VolTermStructure computes volatility at multiple horizons.
type VolTermStructure struct {
	horizons  []int
	returnBuf *ringBuf
	maxHorizon int
}

// NewVolTermStructure creates a vol term structure analyzer.
func NewVolTermStructure(horizons []int) *VolTermStructure {
	maxH := 0
	for _, h := range horizons {
		if h > maxH {
			maxH = h
		}
	}
	return &VolTermStructure{
		horizons:   horizons,
		returnBuf:  newRingBuf(maxH + 1),
		maxHorizon: maxH,
	}
}

// Update adds a return observation and returns vol at each horizon.
func (v *VolTermStructure) Update(ret float64) map[int]float64 {
	v.returnBuf.push(ret)
	result := make(map[int]float64, len(v.horizons))
	if v.returnBuf.count < v.maxHorizon {
		return result
	}
	vals := v.returnBuf.values()
	for _, h := range v.horizons {
		if h > len(vals) {
			continue
		}
		window := vals[len(vals)-h:]
		sd := 0.0
		m := 0.0
		for _, vv := range window {
			m += vv
		}
		m /= float64(len(window))
		for _, vv := range window {
			d := vv - m
			sd += d * d
		}
		sd = math.Sqrt(sd/float64(len(window)-1)) * math.Sqrt(252)
		result[h] = sd
	}
	return result
}

// Slope returns the vol term structure slope (long-short).
func (v *VolTermStructure) Slope(vols map[int]float64) float64 {
	if len(v.horizons) < 2 {
		return 0
	}
	shortest := v.horizons[0]
	longest := v.horizons[len(v.horizons)-1]
	shortVol, ok1 := vols[shortest]
	longVol, ok2 := vols[longest]
	if !ok1 || !ok2 || shortVol == 0 {
		return 0
	}
	return longVol/shortVol - 1
}

// ---------------------------------------------------------------------------
// DispersionRegime: cross-sectional return dispersion
// ---------------------------------------------------------------------------

// DispersionRegime measures cross-sectional return dispersion.
type DispersionRegime struct {
	period      int
	returnBufs  map[string]*ringBuf
	assets      []string
	dispBuf     *ringBuf
	count       int
}

// NewDispersionRegime creates a dispersion regime detector.
func NewDispersionRegime(period int, assets []string) *DispersionRegime {
	bufs := make(map[string]*ringBuf, len(assets))
	for _, a := range assets {
		bufs[a] = newRingBuf(period)
	}
	return &DispersionRegime{
		period:     period,
		returnBufs: bufs,
		assets:     assets,
		dispBuf:    newRingBuf(252),
	}
}

// Update processes new returns and returns dispersion level.
func (d *DispersionRegime) Update(returns map[string]float64, timestamp int64) RegimeResult {
	d.count++
	for _, a := range d.assets {
		if r, ok := returns[a]; ok {
			d.returnBufs[a].push(r)
		}
	}
	// Cross-sectional dispersion: std of returns across assets
	var rets []float64
	for _, a := range d.assets {
		if r, ok := returns[a]; ok {
			rets = append(rets, r)
		}
	}
	if len(rets) < 2 {
		return RegimeResult{Regime: RegimeUnknown, Timestamp: timestamp}
	}
	m := 0.0
	for _, r := range rets {
		m += r
	}
	m /= float64(len(rets))
	ss := 0.0
	for _, r := range rets {
		dd := r - m
		ss += dd * dd
	}
	disp := math.Sqrt(ss / float64(len(rets)-1))
	d.dispBuf.push(disp)
	if !d.dispBuf.full() {
		return RegimeResult{Regime: RegimeUnknown, Value: disp, Timestamp: timestamp}
	}
	pct := d.dispBuf.percentile(50)
	var regime RegimeType
	var conf float64
	if disp > pct*1.5 {
		regime = RegimeLowCorr // high dispersion = low correlation
		conf = math.Min((disp/pct-1)/0.5, 1)
	} else if disp < pct*0.5 {
		regime = RegimeHighCorr // low dispersion = high correlation
		conf = math.Min((1-disp/pct)/0.5, 1)
	} else {
		regime = RegimeUnknown
		conf = 0.5
	}
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: conf,
		Value:      disp,
		Timestamp:  timestamp,
	}
}

// CurrentDispersion returns the latest cross-sectional dispersion.
func (d *DispersionRegime) CurrentDispersion() float64 {
	return d.dispBuf.get(0)
}

// HistoricalPercentile returns where current dispersion sits historically.
func (d *DispersionRegime) HistoricalPercentile() float64 {
	if !d.dispBuf.full() {
		return 50
	}
	current := d.dispBuf.get(0)
	vals := d.dispBuf.values()
	below := 0
	for _, v := range vals {
		if v < current {
			below++
		}
	}
	return 100 * float64(below) / float64(len(vals))
}

// ---------------------------------------------------------------------------
// SentimentRegime: put/call, fear/greed proxy
// ---------------------------------------------------------------------------

// SentimentData holds sentiment indicator values.
type SentimentData struct {
	PutCallRatio float64
	VIXLevel     float64
	AdvDecLine   float64 // advance-decline ratio
	NewHighsLows float64 // new highs minus new lows
}

// SentimentRegime classifies market sentiment.
type SentimentRegime struct {
	period    int
	pcrBuf    *ringBuf
	vixBuf    *ringBuf
	adBuf     *ringBuf
	nhlBuf    *ringBuf
	count     int
	current   RegimeType
}

// NewSentimentRegime creates a sentiment regime detector.
func NewSentimentRegime(period int) *SentimentRegime {
	return &SentimentRegime{
		period:  period,
		pcrBuf:  newRingBuf(period),
		vixBuf:  newRingBuf(period),
		adBuf:   newRingBuf(period),
		nhlBuf:  newRingBuf(period),
		current: RegimeUnknown,
	}
}

// Update processes new sentiment data.
func (s *SentimentRegime) Update(data SentimentData, timestamp int64) RegimeResult {
	s.count++
	s.pcrBuf.push(data.PutCallRatio)
	s.vixBuf.push(data.VIXLevel)
	s.adBuf.push(data.AdvDecLine)
	s.nhlBuf.push(data.NewHighsLows)
	if !s.pcrBuf.full() {
		return RegimeResult{Regime: RegimeUnknown, Label: "unknown", Timestamp: timestamp}
	}
	// Composite fear/greed score
	// Put/Call: high = fear, low = greed
	pcrMean := s.pcrBuf.mean()
	pcrSD := s.pcrBuf.stddev()
	pcrZ := 0.0
	if pcrSD > 0 {
		pcrZ = -(data.PutCallRatio - pcrMean) / pcrSD // inverted: high PCR = fear = negative
	}
	// VIX: high = fear
	vixMean := s.vixBuf.mean()
	vixSD := s.vixBuf.stddev()
	vixZ := 0.0
	if vixSD > 0 {
		vixZ = -(data.VIXLevel - vixMean) / vixSD
	}
	// Advance/Decline: high = greed
	adMean := s.adBuf.mean()
	adSD := s.adBuf.stddev()
	adZ := 0.0
	if adSD > 0 {
		adZ = (data.AdvDecLine - adMean) / adSD
	}
	// New Highs/Lows: high = greed
	nhlMean := s.nhlBuf.mean()
	nhlSD := s.nhlBuf.stddev()
	nhlZ := 0.0
	if nhlSD > 0 {
		nhlZ = (data.NewHighsLows - nhlMean) / nhlSD
	}
	composite := (pcrZ + vixZ + adZ + nhlZ) / 4
	var regime RegimeType
	var conf float64
	if composite > 0.5 {
		regime = RegimeRiskOn
		conf = math.Min(composite, 1)
	} else if composite < -0.5 {
		regime = RegimeRiskOff
		conf = math.Min(math.Abs(composite), 1)
	} else {
		regime = RegimeUnknown
		conf = 1 - math.Abs(composite)/0.5
	}
	s.current = regime
	return RegimeResult{
		Regime:     regime,
		Label:      RegimeLabel(regime),
		Confidence: conf,
		Value:      composite,
		Timestamp:  timestamp,
	}
}

// Current returns the current sentiment regime.
func (s *SentimentRegime) Current() RegimeType { return s.current }
