package analytics

import (
	"math"
	"sort"
	"sync"
)

// ──────────────────────────────────────────────────────────────────────────────
// Core tick/bar types
// ──────────────────────────────────────────────────────────────────────────────

type Tick struct {
	Symbol    string  `json:"symbol"`
	Price     float64 `json:"price"`
	Volume    float64 `json:"volume"`
	Bid       float64 `json:"bid"`
	Ask       float64 `json:"ask"`
	BidSize   float64 `json:"bid_size"`
	AskSize   float64 `json:"ask_size"`
	Side      int     `json:"side"` // 1=buy, -1=sell, 0=unknown
	Timestamp int64   `json:"timestamp_ns"`
}

type OHLCV struct {
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
	Volume    float64 `json:"volume"`
	Timestamp int64   `json:"timestamp_ns"`
}

// ──────────────────────────────────────────────────────────────────────────────
// Real-time volatility estimators
// ──────────────────────────────────────────────────────────────────────────────

type RealTimeVolatility struct {
	mu          sync.RWMutex
	ewmaVar     float64
	ewmaLambda  float64
	prevReturn  float64
	initialized bool

	// Realized variance (sum of squared intraday returns)
	realizedVar float64
	rvCount     int
	prevPrice   float64

	// For Parkinson / Garman-Klass, store recent bars
	bars     []OHLCV
	barCap   int
	barIdx   int
	barFull  bool
}

func NewRealTimeVolatility(ewmaLambda float64, barCapacity int) *RealTimeVolatility {
	return &RealTimeVolatility{
		ewmaLambda: ewmaLambda,
		barCap:     barCapacity,
		bars:       make([]OHLCV, barCapacity),
	}
}

// UpdateTick processes a single price tick for EWMA and realized variance.
func (rv *RealTimeVolatility) UpdateTick(price float64) {
	rv.mu.Lock()
	defer rv.mu.Unlock()

	if rv.prevPrice > 0 {
		ret := math.Log(price / rv.prevPrice)
		if rv.initialized {
			rv.ewmaVar = rv.ewmaLambda*rv.ewmaVar + (1-rv.ewmaLambda)*ret*ret
		} else {
			rv.ewmaVar = ret * ret
			rv.initialized = true
		}
		rv.realizedVar += ret * ret
		rv.rvCount++
		rv.prevReturn = ret
	}
	rv.prevPrice = price
}

// UpdateBar adds an OHLCV bar for range-based estimators.
func (rv *RealTimeVolatility) UpdateBar(bar OHLCV) {
	rv.mu.Lock()
	rv.bars[rv.barIdx] = bar
	rv.barIdx = (rv.barIdx + 1) % rv.barCap
	if rv.barIdx == 0 {
		rv.barFull = true
	}
	rv.mu.Unlock()
}

// EWMA returns annualized EWMA volatility.
func (rv *RealTimeVolatility) EWMA() float64 {
	rv.mu.RLock()
	defer rv.mu.RUnlock()
	return math.Sqrt(rv.ewmaVar * 252)
}

// RealizedVol returns annualized realized volatility from tick returns.
func (rv *RealTimeVolatility) RealizedVol() float64 {
	rv.mu.RLock()
	defer rv.mu.RUnlock()
	if rv.rvCount == 0 {
		return 0
	}
	dailyVar := rv.realizedVar // assume intraday
	return math.Sqrt(dailyVar * 252)
}

// ResetDaily resets intraday realized variance accumulator.
func (rv *RealTimeVolatility) ResetDaily() {
	rv.mu.Lock()
	rv.realizedVar = 0
	rv.rvCount = 0
	rv.mu.Unlock()
}

func (rv *RealTimeVolatility) recentBars() []OHLCV {
	n := rv.barIdx
	if rv.barFull {
		n = rv.barCap
	}
	out := make([]OHLCV, n)
	if rv.barFull {
		copy(out, rv.bars[rv.barIdx:])
		copy(out[rv.barCap-rv.barIdx:], rv.bars[:rv.barIdx])
	} else {
		copy(out, rv.bars[:n])
	}
	return out
}

// Parkinson volatility estimator (range-based).
func (rv *RealTimeVolatility) Parkinson() float64 {
	rv.mu.RLock()
	bars := rv.recentBars()
	rv.mu.RUnlock()

	if len(bars) == 0 {
		return 0
	}
	sum := 0.0
	for _, b := range bars {
		if b.Low > 0 {
			hl := math.Log(b.High / b.Low)
			sum += hl * hl
		}
	}
	n := float64(len(bars))
	return math.Sqrt(sum / (4 * n * math.Log(2)) * 252)
}

// GarmanKlass volatility estimator.
func (rv *RealTimeVolatility) GarmanKlass() float64 {
	rv.mu.RLock()
	bars := rv.recentBars()
	rv.mu.RUnlock()

	if len(bars) == 0 {
		return 0
	}
	sum := 0.0
	for _, b := range bars {
		if b.Low > 0 && b.Open > 0 {
			hl := math.Log(b.High / b.Low)
			co := math.Log(b.Close / b.Open)
			sum += 0.5*hl*hl - (2*math.Log(2)-1)*co*co
		}
	}
	n := float64(len(bars))
	return math.Sqrt(sum / n * 252)
}

// ──────────────────────────────────────────────────────────────────────────────
// Microstructure metrics
// ──────────────────────────────────────────────────────────────────────────────

type MicrostructureMetrics struct {
	mu             sync.RWMutex
	spreadEWMA     float64
	imbalanceEWMA  float64
	lambda         float64
	initialized    bool

	// VPIN approximation
	bucketVolume   float64
	bucketBuyVol   float64
	bucketSellVol  float64
	vpinBuckets    []float64
	vpinBucketCap  int
	vpinIdx        int
	vpinFull       bool
	vpinBucketSize float64

	// Trade flow
	cumDelta       float64
	buyCount       int64
	sellCount      int64
	totalTrades    int64
	largeTrades    []Tick
	largeThreshold float64
}

func NewMicrostructureMetrics(lambda, vpinBucketSize float64, vpinBuckets int, largeThresh float64) *MicrostructureMetrics {
	return &MicrostructureMetrics{
		lambda:         lambda,
		vpinBucketSize: vpinBucketSize,
		vpinBuckets:    make([]float64, vpinBuckets),
		vpinBucketCap:  vpinBuckets,
		largeThreshold: largeThresh,
	}
}

// UpdateQuote processes a quote update for spread and depth metrics.
func (mm *MicrostructureMetrics) UpdateQuote(bid, ask, bidSize, askSize float64) {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	spread := 0.0
	if bid > 0 {
		spread = (ask - bid) / ((ask + bid) / 2)
	}
	imbalance := 0.0
	total := bidSize + askSize
	if total > 0 {
		imbalance = (bidSize - askSize) / total
	}

	if mm.initialized {
		mm.spreadEWMA = mm.lambda*mm.spreadEWMA + (1-mm.lambda)*spread
		mm.imbalanceEWMA = mm.lambda*mm.imbalanceEWMA + (1-mm.lambda)*imbalance
	} else {
		mm.spreadEWMA = spread
		mm.imbalanceEWMA = imbalance
		mm.initialized = true
	}
}

// UpdateTrade processes a trade for VPIN and flow metrics.
func (mm *MicrostructureMetrics) UpdateTrade(tick Tick) {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	vol := tick.Volume
	if tick.Side > 0 {
		mm.bucketBuyVol += vol
		mm.cumDelta += vol
		mm.buyCount++
	} else if tick.Side < 0 {
		mm.bucketSellVol += vol
		mm.cumDelta -= vol
		mm.sellCount++
	} else {
		// Classify by tick rule: split 50/50
		mm.bucketBuyVol += vol / 2
		mm.bucketSellVol += vol / 2
	}
	mm.totalTrades++
	mm.bucketVolume += vol

	// Large trade detection
	if vol >= mm.largeThreshold {
		mm.largeTrades = append(mm.largeTrades, tick)
		if len(mm.largeTrades) > 1000 {
			mm.largeTrades = mm.largeTrades[len(mm.largeTrades)-500:]
		}
	}

	// VPIN bucket completion
	if mm.bucketVolume >= mm.vpinBucketSize {
		totalVol := mm.bucketBuyVol + mm.bucketSellVol
		vpin := 0.0
		if totalVol > 0 {
			vpin = math.Abs(mm.bucketBuyVol-mm.bucketSellVol) / totalVol
		}
		mm.vpinBuckets[mm.vpinIdx] = vpin
		mm.vpinIdx = (mm.vpinIdx + 1) % mm.vpinBucketCap
		if mm.vpinIdx == 0 {
			mm.vpinFull = true
		}
		mm.bucketVolume = 0
		mm.bucketBuyVol = 0
		mm.bucketSellVol = 0
	}
}

// Spread returns current EWMA bid-ask spread (relative).
func (mm *MicrostructureMetrics) Spread() float64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.spreadEWMA
}

// DepthImbalance returns current EWMA depth imbalance [-1, 1].
func (mm *MicrostructureMetrics) DepthImbalance() float64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.imbalanceEWMA
}

// VPIN returns average VPIN across buckets.
func (mm *MicrostructureMetrics) VPIN() float64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	n := mm.vpinIdx
	if mm.vpinFull {
		n = mm.vpinBucketCap
	}
	if n == 0 {
		return 0
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += mm.vpinBuckets[i]
	}
	return sum / float64(n)
}

// CumulativeDelta returns net buy-sell volume.
func (mm *MicrostructureMetrics) CumulativeDelta() float64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.cumDelta
}

// BuySellPressure returns buy_ratio in [0, 1].
func (mm *MicrostructureMetrics) BuySellPressure() float64 {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	total := mm.buyCount + mm.sellCount
	if total == 0 {
		return 0.5
	}
	return float64(mm.buyCount) / float64(total)
}

// TradeFlowToxicity returns fraction of informed flow (VPIN proxy).
func (mm *MicrostructureMetrics) TradeFlowToxicity() float64 {
	return mm.VPIN()
}

// LargeTrades returns recent large trades.
func (mm *MicrostructureMetrics) LargeTrades(n int) []Tick {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	start := len(mm.largeTrades) - n
	if start < 0 {
		start = 0
	}
	out := make([]Tick, len(mm.largeTrades)-start)
	copy(out, mm.largeTrades[start:])
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Regime detector
// ──────────────────────────────────────────────────────────────────────────────

type VolRegime int
type TrendRegime int
type CorrRegime int

const (
	VolLow VolRegime = iota
	VolNormal
	VolHigh
	VolCrisis
)

const (
	TrendUp TrendRegime = iota
	TrendDown
	TrendFlat
)

const (
	CorrLow CorrRegime = iota
	CorrNormal
	CorrHigh
)

func (v VolRegime) String() string {
	return [...]string{"low", "normal", "high", "crisis"}[v]
}

func (t TrendRegime) String() string {
	return [...]string{"up", "down", "flat"}[t]
}

func (c CorrRegime) String() string {
	return [...]string{"low", "normal", "high"}[c]
}

type RegimeState struct {
	Vol   VolRegime   `json:"vol_regime"`
	Trend TrendRegime `json:"trend_regime"`
	Corr  CorrRegime  `json:"corr_regime"`
}

type RegimeDetector struct {
	mu           sync.RWMutex
	volWindow    *ringBuf
	retWindow    *ringBuf
	corrWindow   *ringBuf
	volThresholds [3]float64 // low/normal, normal/high, high/crisis
	trendThresh   float64
	corrThresholds [2]float64 // low/normal, normal/high
	state         RegimeState
}

type ringBuf struct {
	data   []float64
	cursor int
	cap    int
	full   bool
}

func newRingBuf(cap int) *ringBuf {
	return &ringBuf{data: make([]float64, cap), cap: cap}
}

func (rb *ringBuf) add(v float64) {
	rb.data[rb.cursor] = v
	rb.cursor = (rb.cursor + 1) % rb.cap
	if rb.cursor == 0 {
		rb.full = true
	}
}

func (rb *ringBuf) snapshot() []float64 {
	n := rb.cursor
	if rb.full {
		n = rb.cap
	}
	out := make([]float64, n)
	if rb.full {
		copy(out, rb.data[rb.cursor:])
		copy(out[rb.cap-rb.cursor:], rb.data[:rb.cursor])
	} else {
		copy(out, rb.data[:n])
	}
	return out
}

func (rb *ringBuf) last() float64 {
	idx := rb.cursor - 1
	if idx < 0 {
		idx = rb.cap - 1
	}
	return rb.data[idx]
}

func (rb *ringBuf) length() int {
	if rb.full {
		return rb.cap
	}
	return rb.cursor
}

func NewRegimeDetector(window int) *RegimeDetector {
	return &RegimeDetector{
		volWindow:      newRingBuf(window),
		retWindow:      newRingBuf(window),
		corrWindow:     newRingBuf(window),
		volThresholds:  [3]float64{0.10, 0.20, 0.40},
		trendThresh:    0.002,
		corrThresholds: [2]float64{0.30, 0.70},
	}
}

// Update processes a new return and volatility observation.
func (rd *RegimeDetector) Update(ret, vol, avgCorr float64) RegimeState {
	rd.mu.Lock()
	defer rd.mu.Unlock()

	rd.retWindow.add(ret)
	rd.volWindow.add(vol)
	rd.corrWindow.add(avgCorr)

	// Vol regime
	annVol := vol * math.Sqrt(252)
	switch {
	case annVol < rd.volThresholds[0]:
		rd.state.Vol = VolLow
	case annVol < rd.volThresholds[1]:
		rd.state.Vol = VolNormal
	case annVol < rd.volThresholds[2]:
		rd.state.Vol = VolHigh
	default:
		rd.state.Vol = VolCrisis
	}

	// Trend regime: SMA of returns
	rets := rd.retWindow.snapshot()
	if len(rets) > 0 {
		avg := meanF(rets)
		switch {
		case avg > rd.trendThresh:
			rd.state.Trend = TrendUp
		case avg < -rd.trendThresh:
			rd.state.Trend = TrendDown
		default:
			rd.state.Trend = TrendFlat
		}
	}

	// Correlation regime
	switch {
	case avgCorr < rd.corrThresholds[0]:
		rd.state.Corr = CorrLow
	case avgCorr < rd.corrThresholds[1]:
		rd.state.Corr = CorrNormal
	default:
		rd.state.Corr = CorrHigh
	}

	return rd.state
}

func (rd *RegimeDetector) State() RegimeState {
	rd.mu.RLock()
	defer rd.mu.RUnlock()
	return rd.state
}

// ──────────────────────────────────────────────────────────────────────────────
// Lead-lag analyzer
// ──────────────────────────────────────────────────────────────────────────────

type LeadLagResult struct {
	LeaderSymbol  string    `json:"leader"`
	LaggerSymbol  string    `json:"lagger"`
	OptimalLag    int       `json:"optimal_lag"`
	Correlation   float64   `json:"correlation"`
	Correlations  []float64 `json:"lag_correlations"`
}

type LeadLagAnalyzer struct {
	mu       sync.RWMutex
	returns  map[string]*ringBuf
	window   int
	maxLags  int
}

func NewLeadLagAnalyzer(window, maxLags int) *LeadLagAnalyzer {
	return &LeadLagAnalyzer{
		returns: make(map[string]*ringBuf),
		window:  window,
		maxLags: maxLags,
	}
}

func (lla *LeadLagAnalyzer) AddReturn(symbol string, ret float64) {
	lla.mu.Lock()
	if _, ok := lla.returns[symbol]; !ok {
		lla.returns[symbol] = newRingBuf(lla.window)
	}
	lla.returns[symbol].add(ret)
	lla.mu.Unlock()
}

func (lla *LeadLagAnalyzer) Analyze(symbolA, symbolB string) LeadLagResult {
	lla.mu.RLock()
	ra, okA := lla.returns[symbolA]
	rb, okB := lla.returns[symbolB]
	lla.mu.RUnlock()

	result := LeadLagResult{LeaderSymbol: symbolA, LaggerSymbol: symbolB}
	if !okA || !okB {
		return result
	}

	sa := ra.snapshot()
	sb := rb.snapshot()
	n := minInt(len(sa), len(sb))
	if n < lla.maxLags+5 {
		return result
	}
	sa = sa[len(sa)-n:]
	sb = sb[len(sb)-n:]

	bestCorr := 0.0
	bestLag := 0
	corrs := make([]float64, 2*lla.maxLags+1)

	for lag := -lla.maxLags; lag <= lla.maxLags; lag++ {
		c := laggedCorrelation(sa, sb, lag)
		corrs[lag+lla.maxLags] = c
		if math.Abs(c) > math.Abs(bestCorr) {
			bestCorr = c
			bestLag = lag
		}
	}

	result.OptimalLag = bestLag
	result.Correlation = bestCorr
	result.Correlations = corrs
	return result
}

func laggedCorrelation(xs, ys []float64, lag int) float64 {
	n := len(xs)
	var xSlice, ySlice []float64
	if lag >= 0 {
		if lag >= n {
			return 0
		}
		xSlice = xs[:n-lag]
		ySlice = ys[lag:]
	} else {
		aLag := -lag
		if aLag >= n {
			return 0
		}
		xSlice = xs[aLag:]
		ySlice = ys[:n-aLag]
	}
	return pearsonCorr(xSlice, ySlice)
}

// ──────────────────────────────────────────────────────────────────────────────
// Order flow analyzer
// ──────────────────────────────────────────────────────────────────────────────

type OrderFlowAnalyzer struct {
	mu          sync.RWMutex
	cumDelta    float64
	deltaWindow *ringBuf
	volumeWindow *ringBuf
	buyPressure  *ringBuf
	prevPrice   float64
}

func NewOrderFlowAnalyzer(window int) *OrderFlowAnalyzer {
	return &OrderFlowAnalyzer{
		deltaWindow:  newRingBuf(window),
		volumeWindow: newRingBuf(window),
		buyPressure:  newRingBuf(window),
	}
}

func (ofa *OrderFlowAnalyzer) Update(tick Tick) {
	ofa.mu.Lock()
	defer ofa.mu.Unlock()

	delta := tick.Volume
	if tick.Side < 0 {
		delta = -delta
	} else if tick.Side == 0 {
		// Tick rule
		if tick.Price > ofa.prevPrice {
			delta = tick.Volume
		} else if tick.Price < ofa.prevPrice {
			delta = -tick.Volume
		} else {
			delta = 0
		}
	}
	ofa.cumDelta += delta
	ofa.deltaWindow.add(delta)
	ofa.volumeWindow.add(tick.Volume)
	bp := 0.5
	if delta > 0 {
		bp = 1.0
	} else if delta < 0 {
		bp = 0.0
	}
	ofa.buyPressure.add(bp)
	ofa.prevPrice = tick.Price
}

func (ofa *OrderFlowAnalyzer) CumDelta() float64 {
	ofa.mu.RLock()
	defer ofa.mu.RUnlock()
	return ofa.cumDelta
}

func (ofa *OrderFlowAnalyzer) RecentBuyPressure() float64 {
	ofa.mu.RLock()
	defer ofa.mu.RUnlock()
	snap := ofa.buyPressure.snapshot()
	if len(snap) == 0 {
		return 0.5
	}
	return meanF(snap)
}

func (ofa *OrderFlowAnalyzer) DeltaDivergence(priceReturn float64) float64 {
	ofa.mu.RLock()
	defer ofa.mu.RUnlock()
	deltas := ofa.deltaWindow.snapshot()
	if len(deltas) == 0 {
		return 0
	}
	netDelta := sumF(deltas)
	// Divergence: price up but delta negative (or vice versa)
	if priceReturn > 0 && netDelta < 0 {
		return -netDelta // bearish divergence
	}
	if priceReturn < 0 && netDelta > 0 {
		return netDelta // bullish divergence
	}
	return 0
}

// ──────────────────────────────────────────────────────────────────────────────
// Intraday patterns
// ──────────────────────────────────────────────────────────────────────────────

type IntradayPatterns struct {
	mu             sync.RWMutex
	volumeProfile  [48]float64 // 48 half-hour buckets
	spreadProfile  [48]float64
	returnProfile  [48]float64
	counts         [48]int
}

func NewIntradayPatterns() *IntradayPatterns {
	return &IntradayPatterns{}
}

func (ip *IntradayPatterns) Update(halfHourBucket int, volume, spread, ret float64) {
	if halfHourBucket < 0 || halfHourBucket >= 48 {
		return
	}
	ip.mu.Lock()
	n := float64(ip.counts[halfHourBucket])
	// Incremental mean update
	ip.volumeProfile[halfHourBucket] = (ip.volumeProfile[halfHourBucket]*n + volume) / (n + 1)
	ip.spreadProfile[halfHourBucket] = (ip.spreadProfile[halfHourBucket]*n + spread) / (n + 1)
	ip.returnProfile[halfHourBucket] = (ip.returnProfile[halfHourBucket]*n + ret) / (n + 1)
	ip.counts[halfHourBucket]++
	ip.mu.Unlock()
}

func (ip *IntradayPatterns) VolumeProfile() [48]float64 {
	ip.mu.RLock()
	defer ip.mu.RUnlock()
	return ip.volumeProfile
}

func (ip *IntradayPatterns) SpreadProfile() [48]float64 {
	ip.mu.RLock()
	defer ip.mu.RUnlock()
	return ip.spreadProfile
}

func (ip *IntradayPatterns) ReturnProfile() [48]float64 {
	ip.mu.RLock()
	defer ip.mu.RUnlock()
	return ip.returnProfile
}

// ──────────────────────────────────────────────────────────────────────────────
// Anomaly detector
// ──────────────────────────────────────────────────────────────────────────────

type AnomalyType int

const (
	AnomalyNone AnomalyType = iota
	AnomalyZScore
	AnomalyFlashCrash
	AnomalyVolumeSpike
	AnomalySpreadBlow
)

func (a AnomalyType) String() string {
	return [...]string{"none", "zscore", "flash_crash", "volume_spike", "spread_blow"}[a]
}

type Anomaly struct {
	Type      AnomalyType `json:"type"`
	Symbol    string      `json:"symbol"`
	Value     float64     `json:"value"`
	Threshold float64     `json:"threshold"`
	ZScore    float64     `json:"z_score"`
	Timestamp int64       `json:"timestamp_ns"`
}

type AnomalyDetector struct {
	mu               sync.RWMutex
	priceWindows     map[string]*ringBuf
	volWindows       map[string]*ringBuf
	spreadWindows    map[string]*ringBuf
	windowSize       int
	zThreshold       float64
	flashCrashThresh float64 // e.g., -0.05 in one tick
	volSpikeMultiple float64 // e.g., 5x average volume
	anomalies        []Anomaly
	maxAnomalies     int
}

func NewAnomalyDetector(windowSize int, zThresh, flashThresh, volSpike float64) *AnomalyDetector {
	return &AnomalyDetector{
		priceWindows:     make(map[string]*ringBuf),
		volWindows:       make(map[string]*ringBuf),
		spreadWindows:    make(map[string]*ringBuf),
		windowSize:       windowSize,
		zThreshold:       zThresh,
		flashCrashThresh: flashThresh,
		volSpikeMultiple: volSpike,
		maxAnomalies:     10000,
	}
}

func (ad *AnomalyDetector) Check(tick Tick) []Anomaly {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	if ad.priceWindows[tick.Symbol] == nil {
		ad.priceWindows[tick.Symbol] = newRingBuf(ad.windowSize)
		ad.volWindows[tick.Symbol] = newRingBuf(ad.windowSize)
		ad.spreadWindows[tick.Symbol] = newRingBuf(ad.windowSize)
	}

	var anomalies []Anomaly

	// Price z-score
	pw := ad.priceWindows[tick.Symbol]
	psnap := pw.snapshot()
	if len(psnap) >= 20 {
		m := meanF(psnap)
		s := stddevF(psnap)
		if s > 0 {
			z := (tick.Price - m) / s
			if math.Abs(z) > ad.zThreshold {
				a := Anomaly{
					Type: AnomalyZScore, Symbol: tick.Symbol,
					Value: tick.Price, Threshold: ad.zThreshold,
					ZScore: z, Timestamp: tick.Timestamp,
				}
				anomalies = append(anomalies, a)
			}
		}

		// Flash crash: single-tick return
		if pw.length() > 0 {
			prevP := pw.last()
			if prevP > 0 {
				ret := (tick.Price - prevP) / prevP
				if ret < ad.flashCrashThresh {
					anomalies = append(anomalies, Anomaly{
						Type: AnomalyFlashCrash, Symbol: tick.Symbol,
						Value: ret, Threshold: ad.flashCrashThresh,
						Timestamp: tick.Timestamp,
					})
				}
			}
		}
	}
	pw.add(tick.Price)

	// Volume spike
	vw := ad.volWindows[tick.Symbol]
	vsnap := vw.snapshot()
	if len(vsnap) >= 20 && tick.Volume > 0 {
		avgVol := meanF(vsnap)
		if avgVol > 0 && tick.Volume > avgVol*ad.volSpikeMultiple {
			anomalies = append(anomalies, Anomaly{
				Type: AnomalyVolumeSpike, Symbol: tick.Symbol,
				Value: tick.Volume, Threshold: avgVol * ad.volSpikeMultiple,
				Timestamp: tick.Timestamp,
			})
		}
	}
	vw.add(tick.Volume)

	// Spread blowout
	spread := 0.0
	if tick.Bid > 0 {
		spread = (tick.Ask - tick.Bid) / ((tick.Ask + tick.Bid) / 2)
	}
	sw := ad.spreadWindows[tick.Symbol]
	ssnap := sw.snapshot()
	if len(ssnap) >= 20 && spread > 0 {
		avgSpread := meanF(ssnap)
		if avgSpread > 0 && spread > avgSpread*3 {
			anomalies = append(anomalies, Anomaly{
				Type: AnomalySpreadBlow, Symbol: tick.Symbol,
				Value: spread, Threshold: avgSpread * 3,
				Timestamp: tick.Timestamp,
			})
		}
	}
	sw.add(spread)

	ad.anomalies = append(ad.anomalies, anomalies...)
	if len(ad.anomalies) > ad.maxAnomalies {
		ad.anomalies = ad.anomalies[len(ad.anomalies)-ad.maxAnomalies/2:]
	}

	return anomalies
}

func (ad *AnomalyDetector) RecentAnomalies(n int) []Anomaly {
	ad.mu.RLock()
	defer ad.mu.RUnlock()
	start := len(ad.anomalies) - n
	if start < 0 {
		start = 0
	}
	out := make([]Anomaly, len(ad.anomalies)-start)
	copy(out, ad.anomalies[start:])
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Technical signals (streaming/incremental)
// ──────────────────────────────────────────────────────────────────────────────

type TechnicalSignals struct {
	mu     sync.RWMutex
	prices *ringBuf
	highs  *ringBuf
	lows   *ringBuf
	closes *ringBuf
	vols   *ringBuf
	cap    int

	// MACD state
	ema12    float64
	ema26    float64
	signal9  float64
	macdInit bool

	// RSI state
	avgGain  float64
	avgLoss  float64
	rsiInit  bool
	rsiCount int

	// Stochastic state
	stochK float64
	stochD float64

	// OBV
	obv     float64
	prevClose float64

	// MFI accumulators
	mfiPosFlow *ringBuf
	mfiNegFlow *ringBuf

	// ATR
	atr      float64
	atrInit  bool

	// Ichimoku
	ichTenkan  float64
	ichKijun   float64
	ichSenkouA float64
	ichSenkouB float64
}

func NewTechnicalSignals(capacity int) *TechnicalSignals {
	return &TechnicalSignals{
		prices:     newRingBuf(capacity),
		highs:      newRingBuf(capacity),
		lows:       newRingBuf(capacity),
		closes:     newRingBuf(capacity),
		vols:       newRingBuf(capacity),
		cap:        capacity,
		mfiPosFlow: newRingBuf(14),
		mfiNegFlow: newRingBuf(14),
	}
}

// Update processes a new bar and updates all indicators incrementally.
func (ts *TechnicalSignals) Update(bar OHLCV) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	ts.prices.add(bar.Close)
	ts.highs.add(bar.High)
	ts.lows.add(bar.Low)
	ts.closes.add(bar.Close)
	ts.vols.add(bar.Volume)

	ts.updateMACD(bar.Close)
	ts.updateRSI(bar.Close)
	ts.updateOBV(bar.Close, bar.Volume)
	ts.updateATR(bar.High, bar.Low, bar.Close)
	ts.updateStochastic()
	ts.updateMFI(bar)
	ts.updateIchimoku()

	ts.prevClose = bar.Close
}

func (ts *TechnicalSignals) updateMACD(price float64) {
	if !ts.macdInit {
		ts.ema12 = price
		ts.ema26 = price
		ts.signal9 = 0
		ts.macdInit = true
		return
	}
	ts.ema12 = price*2/13 + ts.ema12*(1-2.0/13)
	ts.ema26 = price*2/27 + ts.ema26*(1-2.0/27)
	macd := ts.ema12 - ts.ema26
	ts.signal9 = macd*2/10 + ts.signal9*(1-2.0/10)
}

func (ts *TechnicalSignals) updateRSI(price float64) {
	if ts.prevClose == 0 {
		ts.rsiCount++
		return
	}
	change := price - ts.prevClose
	gain, loss := 0.0, 0.0
	if change > 0 {
		gain = change
	} else {
		loss = -change
	}
	ts.rsiCount++
	if ts.rsiCount <= 14 {
		ts.avgGain += gain / 14
		ts.avgLoss += loss / 14
		if ts.rsiCount == 14 {
			ts.rsiInit = true
		}
	} else if ts.rsiInit {
		ts.avgGain = (ts.avgGain*13 + gain) / 14
		ts.avgLoss = (ts.avgLoss*13 + loss) / 14
	}
}

func (ts *TechnicalSignals) updateOBV(price, volume float64) {
	if ts.prevClose == 0 {
		return
	}
	if price > ts.prevClose {
		ts.obv += volume
	} else if price < ts.prevClose {
		ts.obv -= volume
	}
}

func (ts *TechnicalSignals) updateATR(high, low, close float64) {
	if ts.prevClose == 0 {
		ts.atr = high - low
		ts.atrInit = true
		return
	}
	tr := math.Max(high-low, math.Max(math.Abs(high-ts.prevClose), math.Abs(low-ts.prevClose)))
	if ts.atrInit {
		ts.atr = (ts.atr*13 + tr) / 14
	} else {
		ts.atr = tr
		ts.atrInit = true
	}
}

func (ts *TechnicalSignals) updateStochastic() {
	h := ts.highs.snapshot()
	l := ts.lows.snapshot()
	c := ts.closes.snapshot()
	if len(h) < 14 {
		return
	}
	h14 := h[len(h)-14:]
	l14 := l[len(l)-14:]
	hh := maxSlice(h14)
	ll := minSlice(l14)
	denom := hh - ll
	if denom <= 0 {
		return
	}
	ts.stochK = (c[len(c)-1] - ll) / denom * 100
	// %D = 3-period SMA of %K (simplified: EMA)
	ts.stochD = ts.stochD*2/4 + ts.stochK*(1-2.0/4)
}

func (ts *TechnicalSignals) updateMFI(bar OHLCV) {
	tp := (bar.High + bar.Low + bar.Close) / 3
	rawFlow := tp * bar.Volume
	if ts.prevClose > 0 {
		prevTP := ts.prevClose // simplified
		if tp > prevTP {
			ts.mfiPosFlow.add(rawFlow)
			ts.mfiNegFlow.add(0)
		} else {
			ts.mfiPosFlow.add(0)
			ts.mfiNegFlow.add(rawFlow)
		}
	}
}

func (ts *TechnicalSignals) updateIchimoku() {
	h := ts.highs.snapshot()
	l := ts.lows.snapshot()
	n := len(h)
	if n >= 9 {
		ts.ichTenkan = (maxSlice(h[n-9:]) + minSlice(l[n-9:])) / 2
	}
	if n >= 26 {
		ts.ichKijun = (maxSlice(h[n-26:]) + minSlice(l[n-26:])) / 2
		ts.ichSenkouA = (ts.ichTenkan + ts.ichKijun) / 2
	}
	if n >= 52 {
		ts.ichSenkouB = (maxSlice(h[n-52:]) + minSlice(l[n-52:])) / 2
	}
}

// Getters

func (ts *TechnicalSignals) RSI() float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	if !ts.rsiInit || ts.avgLoss == 0 {
		return 50
	}
	rs := ts.avgGain / ts.avgLoss
	return 100 - 100/(1+rs)
}

func (ts *TechnicalSignals) MACD() (macd, signal, histogram float64) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	macd = ts.ema12 - ts.ema26
	signal = ts.signal9
	histogram = macd - signal
	return
}

func (ts *TechnicalSignals) Bollinger(period int, mult float64) (upper, middle, lower float64) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	snap := ts.closes.snapshot()
	if len(snap) < period {
		return
	}
	window := snap[len(snap)-period:]
	middle = meanF(window)
	sd := stddevF(window)
	upper = middle + mult*sd
	lower = middle - mult*sd
	return
}

func (ts *TechnicalSignals) ATR() float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return ts.atr
}

func (ts *TechnicalSignals) OBV() float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return ts.obv
}

func (ts *TechnicalSignals) MFI() float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	pos := sumF(ts.mfiPosFlow.snapshot())
	neg := sumF(ts.mfiNegFlow.snapshot())
	if neg == 0 {
		return 100
	}
	ratio := pos / neg
	return 100 - 100/(1+ratio)
}

func (ts *TechnicalSignals) Stochastic() (k, d float64) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return ts.stochK, ts.stochD
}

func (ts *TechnicalSignals) CCI(period int) float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	h := ts.highs.snapshot()
	l := ts.lows.snapshot()
	c := ts.closes.snapshot()
	n := minInt(len(h), minInt(len(l), len(c)))
	if n < period {
		return 0
	}
	tps := make([]float64, period)
	for i := 0; i < period; i++ {
		idx := n - period + i
		tps[i] = (h[idx] + l[idx] + c[idx]) / 3
	}
	m := meanF(tps)
	mad := 0.0
	for _, tp := range tps {
		mad += math.Abs(tp - m)
	}
	mad /= float64(period)
	if mad == 0 {
		return 0
	}
	return (tps[period-1] - m) / (0.015 * mad)
}

func (ts *TechnicalSignals) WilliamsR(period int) float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	h := ts.highs.snapshot()
	l := ts.lows.snapshot()
	c := ts.closes.snapshot()
	n := minInt(len(h), minInt(len(l), len(c)))
	if n < period {
		return -50
	}
	hh := maxSlice(h[n-period:])
	ll := minSlice(l[n-period:])
	denom := hh - ll
	if denom <= 0 {
		return -50
	}
	return (hh - c[n-1]) / denom * -100
}

func (ts *TechnicalSignals) Ichimoku() (tenkan, kijun, senkouA, senkouB float64) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return ts.ichTenkan, ts.ichKijun, ts.ichSenkouA, ts.ichSenkouB
}

// SMA returns simple moving average of closes.
func (ts *TechnicalSignals) SMA(period int) float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	snap := ts.closes.snapshot()
	if len(snap) < period {
		return 0
	}
	return meanF(snap[len(snap)-period:])
}

// EMA returns exponential moving average.
func (ts *TechnicalSignals) EMA(period int) float64 {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	snap := ts.closes.snapshot()
	if len(snap) == 0 {
		return 0
	}
	mult := 2.0 / float64(period+1)
	ema := snap[0]
	for i := 1; i < len(snap); i++ {
		ema = snap[i]*mult + ema*(1-mult)
	}
	return ema
}

// AllSignals returns a snapshot of all indicator values.
type SignalSnapshot struct {
	RSI          float64 `json:"rsi"`
	MACD         float64 `json:"macd"`
	MACDSignal   float64 `json:"macd_signal"`
	MACDHist     float64 `json:"macd_histogram"`
	BollingerUp  float64 `json:"bollinger_upper"`
	BollingerMid float64 `json:"bollinger_mid"`
	BollingerLow float64 `json:"bollinger_lower"`
	ATR          float64 `json:"atr"`
	OBV          float64 `json:"obv"`
	MFI          float64 `json:"mfi"`
	StochK       float64 `json:"stoch_k"`
	StochD       float64 `json:"stoch_d"`
	CCI          float64 `json:"cci"`
	WilliamsR    float64 `json:"williams_r"`
	Tenkan       float64 `json:"ichimoku_tenkan"`
	Kijun        float64 `json:"ichimoku_kijun"`
	SenkouA      float64 `json:"ichimoku_senkou_a"`
	SenkouB      float64 `json:"ichimoku_senkou_b"`
}

func (ts *TechnicalSignals) AllSignals() SignalSnapshot {
	macd, sig, hist := ts.MACD()
	bu, bm, bl := ts.Bollinger(20, 2.0)
	sk, sd := ts.Stochastic()
	tk, kj, sa, sb := ts.Ichimoku()
	return SignalSnapshot{
		RSI: ts.RSI(), MACD: macd, MACDSignal: sig, MACDHist: hist,
		BollingerUp: bu, BollingerMid: bm, BollingerLow: bl,
		ATR: ts.ATR(), OBV: ts.OBV(), MFI: ts.MFI(),
		StochK: sk, StochD: sd, CCI: ts.CCI(20), WilliamsR: ts.WilliamsR(14),
		Tenkan: tk, Kijun: kj, SenkouA: sa, SenkouB: sb,
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Math utilities
// ──────────────────────────────────────────────────────────────────────────────

func meanF(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func stddevF(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := meanF(xs)
	s := 0.0
	for _, x := range xs {
		d := x - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(xs)-1))
}

func sumF(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s
}

func pearsonCorr(xs, ys []float64) float64 {
	n := len(xs)
	if n < 2 || n != len(ys) {
		return 0
	}
	mx := meanF(xs)
	my := meanF(ys)
	var num, dx2, dy2 float64
	for i := 0; i < n; i++ {
		dx := xs[i] - mx
		dy := ys[i] - my
		num += dx * dy
		dx2 += dx * dx
		dy2 += dy * dy
	}
	denom := math.Sqrt(dx2 * dy2)
	if denom == 0 {
		return 0
	}
	return num / denom
}

func maxSlice(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	m := xs[0]
	for _, x := range xs[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

func minSlice(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	m := xs[0]
	for _, x := range xs[1:] {
		if x < m {
			m = x
		}
	}
	return m
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ──────────────────────────────────────────────────────────────────────────────
// Composite analytics engine
// ──────────────────────────────────────────────────────────────────────────────

type AnalyticsEngine struct {
	mu          sync.RWMutex
	volatility  map[string]*RealTimeVolatility
	micro       map[string]*MicrostructureMetrics
	regime      *RegimeDetector
	leadLag     *LeadLagAnalyzer
	orderFlow   map[string]*OrderFlowAnalyzer
	intraday    map[string]*IntradayPatterns
	anomaly     *AnomalyDetector
	technicals  map[string]*TechnicalSignals
}

func NewAnalyticsEngine() *AnalyticsEngine {
	return &AnalyticsEngine{
		volatility: make(map[string]*RealTimeVolatility),
		micro:      make(map[string]*MicrostructureMetrics),
		regime:     NewRegimeDetector(63),
		leadLag:    NewLeadLagAnalyzer(252, 10),
		orderFlow:  make(map[string]*OrderFlowAnalyzer),
		intraday:   make(map[string]*IntradayPatterns),
		anomaly:    NewAnomalyDetector(200, 3.0, -0.05, 5.0),
		technicals: make(map[string]*TechnicalSignals),
	}
}

func (ae *AnalyticsEngine) getVol(sym string) *RealTimeVolatility {
	ae.mu.Lock()
	defer ae.mu.Unlock()
	if ae.volatility[sym] == nil {
		ae.volatility[sym] = NewRealTimeVolatility(0.94, 63)
	}
	return ae.volatility[sym]
}

func (ae *AnalyticsEngine) getMicro(sym string) *MicrostructureMetrics {
	ae.mu.Lock()
	defer ae.mu.Unlock()
	if ae.micro[sym] == nil {
		ae.micro[sym] = NewMicrostructureMetrics(0.95, 50000, 50, 10000)
	}
	return ae.micro[sym]
}

func (ae *AnalyticsEngine) getFlow(sym string) *OrderFlowAnalyzer {
	ae.mu.Lock()
	defer ae.mu.Unlock()
	if ae.orderFlow[sym] == nil {
		ae.orderFlow[sym] = NewOrderFlowAnalyzer(500)
	}
	return ae.orderFlow[sym]
}

func (ae *AnalyticsEngine) getIntraday(sym string) *IntradayPatterns {
	ae.mu.Lock()
	defer ae.mu.Unlock()
	if ae.intraday[sym] == nil {
		ae.intraday[sym] = NewIntradayPatterns()
	}
	return ae.intraday[sym]
}

func (ae *AnalyticsEngine) getTech(sym string) *TechnicalSignals {
	ae.mu.Lock()
	defer ae.mu.Unlock()
	if ae.technicals[sym] == nil {
		ae.technicals[sym] = NewTechnicalSignals(500)
	}
	return ae.technicals[sym]
}

// ProcessTick is the main entry point for real-time tick processing.
func (ae *AnalyticsEngine) ProcessTick(tick Tick) []Anomaly {
	ae.getVol(tick.Symbol).UpdateTick(tick.Price)
	ae.getMicro(tick.Symbol).UpdateQuote(tick.Bid, tick.Ask, tick.BidSize, tick.AskSize)
	ae.getMicro(tick.Symbol).UpdateTrade(tick)
	ae.getFlow(tick.Symbol).Update(tick)

	return ae.anomaly.Check(tick)
}

// ProcessBar is the main entry point for bar-level processing.
func (ae *AnalyticsEngine) ProcessBar(sym string, bar OHLCV) {
	ae.getVol(sym).UpdateBar(bar)
	ae.getTech(sym).Update(bar)

	// Compute return for lead-lag and regime
	closes := ae.getTech(sym)
	closes.mu.RLock()
	snap := closes.closes.snapshot()
	closes.mu.RUnlock()
	if len(snap) >= 2 {
		ret := math.Log(snap[len(snap)-1] / snap[len(snap)-2])
		ae.leadLag.AddReturn(sym, ret)
	}
}

// Signals returns all technical signals for a symbol.
func (ae *AnalyticsEngine) Signals(sym string) SignalSnapshot {
	return ae.getTech(sym).AllSignals()
}

// Regime returns current regime state.
func (ae *AnalyticsEngine) Regime() RegimeState {
	return ae.regime.State()
}

// LeadLag analyzes lead-lag between two symbols.
func (ae *AnalyticsEngine) LeadLag(a, b string) LeadLagResult {
	return ae.leadLag.Analyze(a, b)
}

// Anomalies returns recent anomalies.
func (ae *AnalyticsEngine) Anomalies(n int) []Anomaly {
	return ae.anomaly.RecentAnomalies(n)
}

// VolSnapshot returns all vol estimators for a symbol.
type VolSnapshot struct {
	EWMA       float64 `json:"ewma"`
	Realized   float64 `json:"realized"`
	Parkinson  float64 `json:"parkinson"`
	GarmanKlass float64 `json:"garman_klass"`
}

func (ae *AnalyticsEngine) VolSnapshot(sym string) VolSnapshot {
	v := ae.getVol(sym)
	return VolSnapshot{
		EWMA:        v.EWMA(),
		Realized:    v.RealizedVol(),
		Parkinson:   v.Parkinson(),
		GarmanKlass: v.GarmanKlass(),
	}
}

// MicroSnapshot returns microstructure metrics for a symbol.
type MicroSnapshot struct {
	Spread         float64 `json:"spread"`
	DepthImbalance float64 `json:"depth_imbalance"`
	VPIN           float64 `json:"vpin"`
	CumDelta       float64 `json:"cum_delta"`
	BuyPressure    float64 `json:"buy_pressure"`
	Toxicity       float64 `json:"toxicity"`
}

func (ae *AnalyticsEngine) MicroSnapshot(sym string) MicroSnapshot {
	m := ae.getMicro(sym)
	return MicroSnapshot{
		Spread:         m.Spread(),
		DepthImbalance: m.DepthImbalance(),
		VPIN:           m.VPIN(),
		CumDelta:       m.CumulativeDelta(),
		BuyPressure:    m.BuySellPressure(),
		Toxicity:       m.TradeFlowToxicity(),
	}
}

// Symbols returns all tracked symbols sorted.
func (ae *AnalyticsEngine) Symbols() []string {
	ae.mu.RLock()
	defer ae.mu.RUnlock()
	seen := make(map[string]bool)
	for s := range ae.volatility {
		seen[s] = true
	}
	for s := range ae.technicals {
		seen[s] = true
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}
