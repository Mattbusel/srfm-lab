package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// ──────────────────────────────────────────────────────────────────────────────
// Domain types
// ──────────────────────────────────────────────────────────────────────────────

type PriceTick struct {
	Symbol    string  `json:"symbol"`
	Price     float64 `json:"price"`
	Volume    float64 `json:"volume"`
	Bid       float64 `json:"bid"`
	Ask       float64 `json:"ask"`
	Source    string  `json:"source"` // alpaca, binance, etc.
	Timestamp int64   `json:"timestamp_ns"`
}

type QualityScore struct {
	Symbol     string             `json:"symbol"`
	Date       string             `json:"date"`
	Overall    float64            `json:"overall"`
	Components map[string]float64 `json:"components"`
	Issues     []string           `json:"issues"`
}

type QualityAlert struct {
	Symbol    string `json:"symbol"`
	Severity  string `json:"severity"` // info, warning, critical
	Type      string `json:"type"`
	Message   string `json:"message"`
	Timestamp int64  `json:"timestamp_ns"`
	Value     float64 `json:"value,omitempty"`
	Threshold float64 `json:"threshold,omitempty"`
}

type SymbolReport struct {
	Symbol        string          `json:"symbol"`
	ScoreHistory  []QualityScore  `json:"score_history"`
	AlertHistory  []QualityAlert  `json:"alert_history"`
	CurrentScore  float64         `json:"current_score"`
	GapCount      int             `json:"gap_count"`
	OutlierCount  int             `json:"outlier_count"`
	StaleCount    int             `json:"stale_count"`
	SplitEvents   []SplitEvent    `json:"split_events"`
	SourceDiverg  []SourceDivergence `json:"source_divergences"`
}

type SplitEvent struct {
	Symbol    string  `json:"symbol"`
	Date      string  `json:"date"`
	Ratio     float64 `json:"ratio"`
	PricePre  float64 `json:"price_pre"`
	PricePost float64 `json:"price_post"`
}

type SourceDivergence struct {
	Symbol    string  `json:"symbol"`
	SourceA   string  `json:"source_a"`
	SourceB   string  `json:"source_b"`
	PriceA    float64 `json:"price_a"`
	PriceB    float64 `json:"price_b"`
	DivPct    float64 `json:"divergence_pct"`
	Timestamp int64   `json:"timestamp_ns"`
}

// ──────────────────────────────────────────────────────────────────────────────
// Feed validator: gaps, stale, outliers, negatives
// ──────────────────────────────────────────────────────────────────────────────

type FeedValidator struct {
	mu             sync.RWMutex
	lastTick       map[string]PriceTick
	tickCounts     map[string]int
	gapThresholdNs int64   // nanoseconds
	staleThreshNs  int64
	maxPriceJump   float64 // max single-tick return
}

func NewFeedValidator() *FeedValidator {
	return &FeedValidator{
		lastTick:       make(map[string]PriceTick),
		tickCounts:     make(map[string]int),
		gapThresholdNs: int64(5 * time.Minute),
		staleThreshNs:  int64(15 * time.Minute),
		maxPriceJump:   0.15,
	}
}

type FeedCheck struct {
	IsGap       bool
	IsStale     bool
	IsOutlier   bool
	IsNegative  bool
	GapDuration int64
	JumpPct     float64
	Messages    []string
}

func (fv *FeedValidator) Validate(tick PriceTick) FeedCheck {
	fv.mu.Lock()
	defer fv.mu.Unlock()

	var check FeedCheck

	// Negative price
	if tick.Price <= 0 {
		check.IsNegative = true
		check.Messages = append(check.Messages, fmt.Sprintf("negative/zero price: %.4f", tick.Price))
	}

	prev, exists := fv.lastTick[tick.Symbol]
	if exists {
		// Gap detection
		gap := tick.Timestamp - prev.Timestamp
		if gap > fv.gapThresholdNs {
			check.IsGap = true
			check.GapDuration = gap
			check.Messages = append(check.Messages, fmt.Sprintf("data gap: %v", time.Duration(gap)))
		}

		// Price jump / outlier
		if prev.Price > 0 {
			jump := (tick.Price - prev.Price) / prev.Price
			check.JumpPct = jump
			if math.Abs(jump) > fv.maxPriceJump {
				check.IsOutlier = true
				check.Messages = append(check.Messages, fmt.Sprintf("price jump: %.2f%%", jump*100))
			}
		}
	}

	fv.lastTick[tick.Symbol] = tick
	fv.tickCounts[tick.Symbol]++
	return check
}

func (fv *FeedValidator) CheckStale(now int64) map[string]FeedCheck {
	fv.mu.RLock()
	defer fv.mu.RUnlock()
	stale := make(map[string]FeedCheck)
	for sym, tick := range fv.lastTick {
		if now-tick.Timestamp > fv.staleThreshNs {
			stale[sym] = FeedCheck{
				IsStale:  true,
				Messages: []string{fmt.Sprintf("stale for %v", time.Duration(now-tick.Timestamp))},
			}
		}
	}
	return stale
}

// ──────────────────────────────────────────────────────────────────────────────
// Timeseries validator: z-score, IQR, Grubbs
// ──────────────────────────────────────────────────────────────────────────────

type TimeseriesValidator struct {
	mu       sync.RWMutex
	windows  map[string]*rollingWindow
	capacity int
}

type rollingWindow struct {
	data   []float64
	cursor int
	full   bool
	cap    int
}

func newRollingWindow(cap int) *rollingWindow {
	return &rollingWindow{data: make([]float64, cap), cap: cap}
}

func (rw *rollingWindow) add(v float64) {
	rw.data[rw.cursor] = v
	rw.cursor = (rw.cursor + 1) % rw.cap
	if rw.cursor == 0 {
		rw.full = true
	}
}

func (rw *rollingWindow) snapshot() []float64 {
	n := rw.cursor
	if rw.full {
		n = rw.cap
	}
	out := make([]float64, n)
	if rw.full {
		copy(out, rw.data[rw.cursor:])
		copy(out[rw.cap-rw.cursor:], rw.data[:rw.cursor])
	} else {
		copy(out, rw.data[:n])
	}
	return out
}

func NewTimeseriesValidator(windowSize int) *TimeseriesValidator {
	return &TimeseriesValidator{windows: make(map[string]*rollingWindow), capacity: windowSize}
}

type OutlierResult struct {
	IsOutlier   bool    `json:"is_outlier"`
	ZScore      float64 `json:"z_score"`
	IQROutlier  bool    `json:"iqr_outlier"`
	GrubbsStat  float64 `json:"grubbs_stat"`
	GrubbsCrit  float64 `json:"grubbs_critical"`
	GrubbsFlag  bool    `json:"grubbs_flag"`
}

func (tv *TimeseriesValidator) Check(symbol string, value float64) OutlierResult {
	tv.mu.Lock()
	if _, ok := tv.windows[symbol]; !ok {
		tv.windows[symbol] = newRollingWindow(tv.capacity)
	}
	tv.windows[symbol].add(value)
	snap := tv.windows[symbol].snapshot()
	tv.mu.Unlock()

	if len(snap) < 10 {
		return OutlierResult{}
	}

	m := mean(snap)
	s := stddev(snap)
	var result OutlierResult

	// Z-score
	if s > 0 {
		result.ZScore = (value - m) / s
	}
	if math.Abs(result.ZScore) > 3.0 {
		result.IsOutlier = true
	}

	// IQR
	sorted := make([]float64, len(snap))
	copy(sorted, snap)
	sort.Float64s(sorted)
	q1 := percentileSorted(sorted, 0.25)
	q3 := percentileSorted(sorted, 0.75)
	iqr := q3 - q1
	if value < q1-1.5*iqr || value > q3+1.5*iqr {
		result.IQROutlier = true
		result.IsOutlier = true
	}

	// Grubbs test
	n := float64(len(snap))
	if s > 0 {
		result.GrubbsStat = math.Abs(value-m) / s
		// Critical value approximation (t-distribution approx)
		tCrit := 2.0 + 0.3/math.Sqrt(n) // simplified
		result.GrubbsCrit = ((n - 1) / math.Sqrt(n)) * math.Sqrt(tCrit*tCrit/(n-2+tCrit*tCrit))
		if result.GrubbsStat > result.GrubbsCrit {
			result.GrubbsFlag = true
			result.IsOutlier = true
		}
	}

	return result
}

// ──────────────────────────────────────────────────────────────────────────────
// Cross-source validator
// ──────────────────────────────────────────────────────────────────────────────

type CrossSourceValidator struct {
	mu          sync.RWMutex
	prices      map[string]map[string]PriceTick // symbol -> source -> tick
	maxDivPct   float64
	divergences []SourceDivergence
}

func NewCrossSourceValidator(maxDivPct float64) *CrossSourceValidator {
	return &CrossSourceValidator{
		prices:    make(map[string]map[string]PriceTick),
		maxDivPct: maxDivPct,
	}
}

func (csv *CrossSourceValidator) Update(tick PriceTick) []SourceDivergence {
	csv.mu.Lock()
	defer csv.mu.Unlock()

	if csv.prices[tick.Symbol] == nil {
		csv.prices[tick.Symbol] = make(map[string]PriceTick)
	}
	csv.prices[tick.Symbol][tick.Source] = tick

	var divs []SourceDivergence
	sources := csv.prices[tick.Symbol]
	sourceNames := make([]string, 0, len(sources))
	for s := range sources {
		sourceNames = append(sourceNames, s)
	}
	sort.Strings(sourceNames)

	for i := 0; i < len(sourceNames); i++ {
		for j := i + 1; j < len(sourceNames); j++ {
			a := sources[sourceNames[i]]
			b := sources[sourceNames[j]]
			// Only compare if within 30s of each other
			if absInt64(a.Timestamp-b.Timestamp) > int64(30*time.Second) {
				continue
			}
			mid := (a.Price + b.Price) / 2
			if mid <= 0 {
				continue
			}
			divPct := math.Abs(a.Price-b.Price) / mid * 100
			if divPct > csv.maxDivPct {
				d := SourceDivergence{
					Symbol: tick.Symbol, SourceA: sourceNames[i], SourceB: sourceNames[j],
					PriceA: a.Price, PriceB: b.Price, DivPct: divPct,
					Timestamp: tick.Timestamp,
				}
				divs = append(divs, d)
				csv.divergences = append(csv.divergences, d)
			}
		}
	}
	return divs
}

func (csv *CrossSourceValidator) RecentDivergences(n int) []SourceDivergence {
	csv.mu.RLock()
	defer csv.mu.RUnlock()
	start := len(csv.divergences) - n
	if start < 0 {
		start = 0
	}
	out := make([]SourceDivergence, len(csv.divergences)-start)
	copy(out, csv.divergences[start:])
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Split detector
// ──────────────────────────────────────────────────────────────────────────────

type SplitDetector struct {
	mu         sync.RWMutex
	prevClose  map[string]float64
	splits     []SplitEvent
	ratios     []float64 // common split ratios to check
}

func NewSplitDetector() *SplitDetector {
	return &SplitDetector{
		prevClose: make(map[string]float64),
		ratios:    []float64{2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 10.0, 15.0, 20.0, 0.5, 0.333, 0.25, 0.1},
	}
}

func (sd *SplitDetector) Check(symbol string, close float64, date string) *SplitEvent {
	sd.mu.Lock()
	defer sd.mu.Unlock()

	prev, exists := sd.prevClose[symbol]
	sd.prevClose[symbol] = close

	if !exists || prev <= 0 || close <= 0 {
		return nil
	}

	ratio := prev / close
	// Check if ratio is close to a common split ratio
	for _, r := range sd.ratios {
		if math.Abs(ratio-r)/r < 0.05 { // within 5%
			ev := &SplitEvent{
				Symbol:    symbol,
				Date:      date,
				Ratio:     r,
				PricePre:  prev,
				PricePost: close,
			}
			sd.splits = append(sd.splits, *ev)
			return ev
		}
	}
	return nil
}

func (sd *SplitDetector) Events() []SplitEvent {
	sd.mu.RLock()
	defer sd.mu.RUnlock()
	out := make([]SplitEvent, len(sd.splits))
	copy(out, sd.splits)
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Quality scorer
// ──────────────────────────────────────────────────────────────────────────────

type QualityScorer struct {
	mu     sync.RWMutex
	scores map[string][]QualityScore // symbol -> history
}

func NewQualityScorer() *QualityScorer {
	return &QualityScorer{scores: make(map[string][]QualityScore)}
}

func (qs *QualityScorer) Score(symbol, date string, gaps, outliers, stale int, divergences int, hasSplit bool) QualityScore {
	components := make(map[string]float64)
	var issues []string

	// Gap score (0-25 points)
	gapScore := 25.0 - float64(gaps)*5
	if gapScore < 0 {
		gapScore = 0
	}
	components["gaps"] = gapScore
	if gaps > 0 {
		issues = append(issues, fmt.Sprintf("%d data gaps", gaps))
	}

	// Outlier score (0-25 points)
	outlierScore := 25.0 - float64(outliers)*3
	if outlierScore < 0 {
		outlierScore = 0
	}
	components["outliers"] = outlierScore
	if outliers > 0 {
		issues = append(issues, fmt.Sprintf("%d outliers", outliers))
	}

	// Staleness score (0-25 points)
	staleScore := 25.0 - float64(stale)*8
	if staleScore < 0 {
		staleScore = 0
	}
	components["staleness"] = staleScore
	if stale > 0 {
		issues = append(issues, fmt.Sprintf("%d stale periods", stale))
	}

	// Consistency score (0-25 points)
	consScore := 25.0 - float64(divergences)*4
	if consScore < 0 {
		consScore = 0
	}
	components["consistency"] = consScore
	if divergences > 0 {
		issues = append(issues, fmt.Sprintf("%d source divergences", divergences))
	}
	if hasSplit {
		issues = append(issues, "possible unadjusted split detected")
		consScore -= 10
		if consScore < 0 {
			consScore = 0
		}
		components["consistency"] = consScore
	}

	overall := gapScore + outlierScore + staleScore + consScore

	score := QualityScore{
		Symbol:     symbol,
		Date:       date,
		Overall:    overall,
		Components: components,
		Issues:     issues,
	}

	qs.mu.Lock()
	qs.scores[symbol] = append(qs.scores[symbol], score)
	// Keep last 365 days
	if len(qs.scores[symbol]) > 365 {
		qs.scores[symbol] = qs.scores[symbol][len(qs.scores[symbol])-365:]
	}
	qs.mu.Unlock()

	return score
}

func (qs *QualityScorer) GetScores(symbol string) []QualityScore {
	qs.mu.RLock()
	defer qs.mu.RUnlock()
	h := qs.scores[symbol]
	out := make([]QualityScore, len(h))
	copy(out, h)
	return out
}

func (qs *QualityScorer) AllCurrentScores() map[string]float64 {
	qs.mu.RLock()
	defer qs.mu.RUnlock()
	out := make(map[string]float64)
	for sym, history := range qs.scores {
		if len(history) > 0 {
			out[sym] = history[len(history)-1].Overall
		}
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Alert generator
// ──────────────────────────────────────────────────────────────────────────────

type AlertGenerator struct {
	mu     sync.RWMutex
	alerts []QualityAlert
	maxLen int
}

func NewAlertGenerator(maxLen int) *AlertGenerator {
	return &AlertGenerator{maxLen: maxLen}
}

func (ag *AlertGenerator) Fire(alert QualityAlert) {
	ag.mu.Lock()
	ag.alerts = append(ag.alerts, alert)
	if len(ag.alerts) > ag.maxLen {
		ag.alerts = ag.alerts[len(ag.alerts)-ag.maxLen:]
	}
	ag.mu.Unlock()
}

func (ag *AlertGenerator) Recent(n int) []QualityAlert {
	ag.mu.RLock()
	defer ag.mu.RUnlock()
	start := len(ag.alerts) - n
	if start < 0 {
		start = 0
	}
	out := make([]QualityAlert, len(ag.alerts)-start)
	copy(out, ag.alerts[start:])
	return out
}

func (ag *AlertGenerator) BySymbol(symbol string) []QualityAlert {
	ag.mu.RLock()
	defer ag.mu.RUnlock()
	var out []QualityAlert
	for _, a := range ag.alerts {
		if a.Symbol == symbol {
			out = append(out, a)
		}
	}
	return out
}

func (ag *AlertGenerator) BySeverity(sev string) []QualityAlert {
	ag.mu.RLock()
	defer ag.mu.RUnlock()
	var out []QualityAlert
	for _, a := range ag.alerts {
		if a.Severity == sev {
			out = append(out, a)
		}
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Metrics exporter (Prometheus-compatible)
// ──────────────────────────────────────────────────────────────────────────────

type MetricsExporter struct {
	mu          sync.RWMutex
	counters    map[string]int64
	gauges      map[string]float64
}

func NewMetricsExporter() *MetricsExporter {
	return &MetricsExporter{
		counters: make(map[string]int64),
		gauges:   make(map[string]float64),
	}
}

func (me *MetricsExporter) IncCounter(name string) {
	me.mu.Lock()
	me.counters[name]++
	me.mu.Unlock()
}

func (me *MetricsExporter) SetGauge(name string, val float64) {
	me.mu.Lock()
	me.gauges[name] = val
	me.mu.Unlock()
}

func (me *MetricsExporter) Export() string {
	me.mu.RLock()
	defer me.mu.RUnlock()
	var sb strings.Builder
	for k, v := range me.counters {
		fmt.Fprintf(&sb, "# TYPE %s counter\n%s %d\n", k, k, v)
	}
	for k, v := range me.gauges {
		fmt.Fprintf(&sb, "# TYPE %s gauge\n%s %.6f\n", k, k, v)
	}
	return sb.String()
}

// ──────────────────────────────────────────────────────────────────────────────
// Data quality service (orchestrator)
// ──────────────────────────────────────────────────────────────────────────────

type DataQualityService struct {
	feedVal    *FeedValidator
	tsVal      *TimeseriesValidator
	crossVal   *CrossSourceValidator
	splitDet   *SplitDetector
	scorer     *QualityScorer
	alertGen   *AlertGenerator
	metrics    *MetricsExporter

	mu             sync.RWMutex
	symbolCounters map[string]*symbolStats
}

type symbolStats struct {
	gaps     int
	outliers int
	stale    int
	divs     int
	splits   int
	ticks    int
}

func NewDataQualityService() *DataQualityService {
	return &DataQualityService{
		feedVal:        NewFeedValidator(),
		tsVal:          NewTimeseriesValidator(200),
		crossVal:       NewCrossSourceValidator(0.5),
		splitDet:       NewSplitDetector(),
		scorer:         NewQualityScorer(),
		alertGen:       NewAlertGenerator(10000),
		metrics:        NewMetricsExporter(),
		symbolCounters: make(map[string]*symbolStats),
	}
}

func (dqs *DataQualityService) ProcessTick(tick PriceTick) {
	dqs.metrics.IncCounter("dq_ticks_total")

	// Feed validation
	check := dqs.feedVal.Validate(tick)
	if check.IsNegative {
		dqs.alertGen.Fire(QualityAlert{
			Symbol: tick.Symbol, Severity: "critical", Type: "negative_price",
			Message: fmt.Sprintf("Negative price: %.4f", tick.Price), Timestamp: tick.Timestamp,
		})
		dqs.metrics.IncCounter("dq_negative_prices_total")
	}
	if check.IsGap {
		dqs.alertGen.Fire(QualityAlert{
			Symbol: tick.Symbol, Severity: "warning", Type: "data_gap",
			Message: check.Messages[0], Timestamp: tick.Timestamp,
			Value: float64(check.GapDuration), Threshold: float64(dqs.feedVal.gapThresholdNs),
		})
		dqs.metrics.IncCounter("dq_gaps_total")
	}

	// Timeseries validation
	outlier := dqs.tsVal.Check(tick.Symbol, tick.Price)
	if outlier.IsOutlier {
		sev := "warning"
		if math.Abs(outlier.ZScore) > 5 {
			sev = "critical"
		}
		dqs.alertGen.Fire(QualityAlert{
			Symbol: tick.Symbol, Severity: sev, Type: "outlier",
			Message: fmt.Sprintf("Outlier z=%.2f iqr=%v grubbs=%v", outlier.ZScore, outlier.IQROutlier, outlier.GrubbsFlag),
			Timestamp: tick.Timestamp, Value: outlier.ZScore,
		})
		dqs.metrics.IncCounter("dq_outliers_total")
	}

	// Cross-source validation
	divs := dqs.crossVal.Update(tick)
	for _, d := range divs {
		dqs.alertGen.Fire(QualityAlert{
			Symbol: d.Symbol, Severity: "warning", Type: "source_divergence",
			Message: fmt.Sprintf("%s vs %s: %.2f%% divergence", d.SourceA, d.SourceB, d.DivPct),
			Timestamp: d.Timestamp, Value: d.DivPct, Threshold: dqs.crossVal.maxDivPct,
		})
		dqs.metrics.IncCounter("dq_divergences_total")
	}

	// Split detection (daily granularity)
	date := time.Unix(0, tick.Timestamp).Format("2006-01-02")
	if ev := dqs.splitDet.Check(tick.Symbol, tick.Price, date); ev != nil {
		dqs.alertGen.Fire(QualityAlert{
			Symbol: tick.Symbol, Severity: "critical", Type: "split_detected",
			Message: fmt.Sprintf("Possible split %.1f:1, pre=%.2f post=%.2f", ev.Ratio, ev.PricePre, ev.PricePost),
			Timestamp: tick.Timestamp, Value: ev.Ratio,
		})
		dqs.metrics.IncCounter("dq_splits_total")
	}

	// Update per-symbol counters
	dqs.mu.Lock()
	if dqs.symbolCounters[tick.Symbol] == nil {
		dqs.symbolCounters[tick.Symbol] = &symbolStats{}
	}
	st := dqs.symbolCounters[tick.Symbol]
	st.ticks++
	if check.IsGap {
		st.gaps++
	}
	if outlier.IsOutlier || check.IsOutlier {
		st.outliers++
	}
	if check.IsStale {
		st.stale++
	}
	st.divs += len(divs)
	dqs.mu.Unlock()

	// Update quality gauge
	dqs.metrics.SetGauge(fmt.Sprintf("dq_quality_score{symbol=%q}", tick.Symbol), dqs.currentScore(tick.Symbol, date))
}

func (dqs *DataQualityService) currentScore(symbol, date string) float64 {
	dqs.mu.RLock()
	st := dqs.symbolCounters[symbol]
	dqs.mu.RUnlock()
	if st == nil {
		return 100
	}
	score := dqs.scorer.Score(symbol, date, st.gaps, st.outliers, st.stale, st.divs, st.splits > 0)
	return score.Overall
}

func (dqs *DataQualityService) CheckStale() {
	now := time.Now().UnixNano()
	stale := dqs.feedVal.CheckStale(now)
	for sym, check := range stale {
		if check.IsStale {
			dqs.alertGen.Fire(QualityAlert{
				Symbol: sym, Severity: "warning", Type: "stale_data",
				Message: check.Messages[0], Timestamp: now,
			})
			dqs.mu.Lock()
			if dqs.symbolCounters[sym] == nil {
				dqs.symbolCounters[sym] = &symbolStats{}
			}
			dqs.symbolCounters[sym].stale++
			dqs.mu.Unlock()
			dqs.metrics.IncCounter("dq_stale_total")
		}
	}
}

func (dqs *DataQualityService) GetReport(symbol string) SymbolReport {
	scores := dqs.scorer.GetScores(symbol)
	alerts := dqs.alertGen.BySymbol(symbol)
	splits := dqs.splitDet.Events()

	var symSplits []SplitEvent
	for _, s := range splits {
		if s.Symbol == symbol {
			symSplits = append(symSplits, s)
		}
	}

	current := 100.0
	if len(scores) > 0 {
		current = scores[len(scores)-1].Overall
	}

	dqs.mu.RLock()
	st := dqs.symbolCounters[symbol]
	dqs.mu.RUnlock()

	report := SymbolReport{
		Symbol:       symbol,
		ScoreHistory: scores,
		AlertHistory: alerts,
		CurrentScore: current,
		SplitEvents:  symSplits,
		SourceDiverg: dqs.crossVal.RecentDivergences(50),
	}
	if st != nil {
		report.GapCount = st.gaps
		report.OutlierCount = st.outliers
		report.StaleCount = st.stale
	}
	return report
}

// ──────────────────────────────────────────────────────────────────────────────
// Background monitoring loop
// ──────────────────────────────────────────────────────────────────────────────

func (dqs *DataQualityService) MonitorLoop(stop <-chan struct{}) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	scoreTicker := time.NewTicker(5 * time.Minute)
	defer scoreTicker.Stop()

	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			dqs.CheckStale()
		case <-scoreTicker.C:
			// Recompute all scores
			dqs.mu.RLock()
			symbols := make([]string, 0, len(dqs.symbolCounters))
			for s := range dqs.symbolCounters {
				symbols = append(symbols, s)
			}
			dqs.mu.RUnlock()
			date := time.Now().Format("2006-01-02")
			for _, sym := range symbols {
				dqs.currentScore(sym, date)
			}
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// HTTP handlers
// ──────────────────────────────────────────────────────────────────────────────

type Server struct {
	dqs *DataQualityService
	mux *http.ServeMux
}

func NewServer(dqs *DataQualityService) *Server {
	s := &Server{dqs: dqs, mux: http.NewServeMux()}
	s.mux.HandleFunc("/quality/scores", s.handleScores)
	s.mux.HandleFunc("/quality/alerts", s.handleAlerts)
	s.mux.HandleFunc("/quality/report/", s.handleReport)
	s.mux.HandleFunc("/metrics", s.handleMetrics)
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/quality/ingest", s.handleIngest)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func (s *Server) handleScores(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.dqs.scorer.AllCurrentScores())
}

func (s *Server) handleAlerts(w http.ResponseWriter, r *http.Request) {
	n := 100
	sev := r.URL.Query().Get("severity")
	if sev != "" {
		writeJSON(w, http.StatusOK, s.dqs.alertGen.BySeverity(sev))
		return
	}
	writeJSON(w, http.StatusOK, s.dqs.alertGen.Recent(n))
}

func (s *Server) handleReport(w http.ResponseWriter, r *http.Request) {
	symbol := r.URL.Path[len("/quality/report/"):]
	if symbol == "" {
		http.Error(w, "symbol required", http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusOK, s.dqs.GetReport(symbol))
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	w.Write([]byte(s.dqs.metrics.Export()))
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var tick PriceTick
	if err := json.NewDecoder(r.Body).Decode(&tick); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	r.Body.Close()
	s.dqs.ProcessTick(tick)
	w.WriteHeader(http.StatusNoContent)
}

// ──────────────────────────────────────────────────────────────────────────────
// Math helpers
// ──────────────────────────────────────────────────────────────────────────────

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func stddev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := mean(xs)
	s := 0.0
	for _, x := range xs {
		d := x - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(xs)-1))
}

func percentileSorted(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	idx := p * float64(n-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi || hi >= n {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

func absInt64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

func main() {
	addr := ":8091"
	if v := os.Getenv("DQ_ADDR"); v != "" {
		addr = v
	}
	dqs := NewDataQualityService()
	stop := make(chan struct{})
	go dqs.MonitorLoop(stop)

	srv := NewServer(dqs)
	log.Printf("data-quality service listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, srv))
}
