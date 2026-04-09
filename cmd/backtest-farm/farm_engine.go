package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// =============================================================================
// farm_engine.go — execution engine: queue, workers, result store, grid search,
// landscape builder, notification engine.
// =============================================================================

// ---- configuration ---------------------------------------------------------

type FarmConfig struct {
	MaxWorkers       int    `json:"max_workers"`
	JobTimeoutSec    int    `json:"job_timeout_sec"`
	ResultRetentionS int    `json:"result_retention_sec"`
	NotificationURL  string `json:"notification_webhook"`
}

// ---- job status constants --------------------------------------------------

const (
	StatusQueued    = "queued"
	StatusRunning   = "running"
	StatusCompleted = "completed"
	StatusFailed    = "failed"
	StatusCancelled = "cancelled"
)

// ---- core structs ----------------------------------------------------------

type BacktestConfig struct {
	Strategy   string                 `json:"strategy"`
	Symbol     string                 `json:"symbol"`
	Timeframe  string                 `json:"timeframe"`
	Parameters map[string]interface{} `json:"parameters"`
	CostBPS    float64                `json:"cost_bps"`
	DateRange  string                 `json:"date_range"`
	Priority   int                    `json:"priority"`
}

type BacktestResult struct {
	Sharpe       float64 `json:"sharpe"`
	Sortino      float64 `json:"sortino"`
	Calmar       float64 `json:"calmar"`
	TotalReturn  float64 `json:"total_return"`
	MaxDrawdown  float64 `json:"max_drawdown"`
	WinRate      float64 `json:"win_rate"`
	ProfitFactor float64 `json:"profit_factor"`
	NTrades      int     `json:"n_trades"`
	IC           float64 `json:"ic"`
}

type BacktestJob struct {
	ID          string          `json:"id"`
	Config      BacktestConfig  `json:"config"`
	Status      string          `json:"status"`
	Result      *BacktestResult `json:"result,omitempty"`
	Error       string          `json:"error,omitempty"`
	SubmittedAt time.Time       `json:"submitted_at"`
	StartedAt   *time.Time      `json:"started_at,omitempty"`
	CompletedAt *time.Time      `json:"completed_at,omitempty"`
	DurationMs  int64           `json:"duration_ms"`
	WorkerID    int             `json:"worker_id"`
	BatchID     string          `json:"batch_id,omitempty"`
	cancelCh    chan struct{}
}

// ---- ID generation ---------------------------------------------------------

func newJobID() string {
	b := make([]byte, 12)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("job-%d", time.Now().UnixNano())
	}
	return "bt-" + hex.EncodeToString(b)
}

func newBatchID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("batch-%d", time.Now().UnixNano())
	}
	return "batch-" + hex.EncodeToString(b)
}

// ---- priority queue --------------------------------------------------------

type JobQueue struct {
	mu    sync.Mutex
	cond  *sync.Cond
	items []*BacktestJob
	closed bool
}

func NewJobQueue() *JobQueue {
	q := &JobQueue{}
	q.cond = sync.NewCond(&q.mu)
	return q
}

func (q *JobQueue) Push(job *BacktestJob) {
	q.mu.Lock()
	defer q.mu.Unlock()
	// insert maintaining priority order (higher priority first)
	idx := sort.Search(len(q.items), func(i int) bool {
		return q.items[i].Config.Priority < job.Config.Priority
	})
	q.items = append(q.items, nil)
	copy(q.items[idx+1:], q.items[idx:])
	q.items[idx] = job
	q.cond.Signal()
}

func (q *JobQueue) Pop() (*BacktestJob, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for len(q.items) == 0 && !q.closed {
		q.cond.Wait()
	}
	if q.closed && len(q.items) == 0 {
		return nil, false
	}
	job := q.items[0]
	q.items = q.items[1:]
	return job, true
}

func (q *JobQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.items)
}

func (q *JobQueue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	q.cond.Broadcast()
}

func (q *JobQueue) Drain() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	n := len(q.items)
	for _, j := range q.items {
		j.Status = StatusCancelled
		close(j.cancelCh)
	}
	q.items = q.items[:0]
	return n
}

func (q *JobQueue) Remove(id string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i, j := range q.items {
		if j.ID == id {
			j.Status = StatusCancelled
			close(j.cancelCh)
			q.items = append(q.items[:i], q.items[i+1:]...)
			return true
		}
	}
	return false
}

func (q *JobQueue) Snapshot() []*BacktestJob {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make([]*BacktestJob, len(q.items))
	copy(out, q.items)
	return out
}

// ---- result store ----------------------------------------------------------

type ResultStore struct {
	mu      sync.RWMutex
	results map[string]*BacktestJob
	ttl     time.Duration
}

func NewResultStore(ttlSeconds int) *ResultStore {
	return &ResultStore{
		results: make(map[string]*BacktestJob, 1024),
		ttl:     time.Duration(ttlSeconds) * time.Second,
	}
}

func (rs *ResultStore) Put(job *BacktestJob) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.results[job.ID] = job
}

func (rs *ResultStore) Get(id string) (*BacktestJob, bool) {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	j, ok := rs.results[id]
	return j, ok
}

func (rs *ResultStore) All() []*BacktestJob {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	out := make([]*BacktestJob, 0, len(rs.results))
	for _, j := range rs.results {
		out = append(out, j)
	}
	return out
}

func (rs *ResultStore) Completed() []*BacktestJob {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	out := make([]*BacktestJob, 0, len(rs.results))
	for _, j := range rs.results {
		if j.Status == StatusCompleted && j.Result != nil {
			out = append(out, j)
		}
	}
	return out
}

func (rs *ResultStore) ByStatus(status string) []*BacktestJob {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	out := make([]*BacktestJob, 0)
	for _, j := range rs.results {
		if j.Status == status {
			out = append(out, j)
		}
	}
	return out
}

func (rs *ResultStore) Count() map[string]int {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	counts := map[string]int{
		StatusQueued:    0,
		StatusRunning:   0,
		StatusCompleted: 0,
		StatusFailed:    0,
		StatusCancelled: 0,
	}
	for _, j := range rs.results {
		counts[j.Status]++
	}
	return counts
}

func (rs *ResultStore) PurgeCompleted() int {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	n := 0
	for id, j := range rs.results {
		if j.Status == StatusCompleted || j.Status == StatusFailed || j.Status == StatusCancelled {
			delete(rs.results, id)
			n++
		}
	}
	return n
}

func (rs *ResultStore) CleanExpired() int {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	now := time.Now()
	n := 0
	for id, j := range rs.results {
		if j.CompletedAt != nil && now.Sub(*j.CompletedAt) > rs.ttl {
			delete(rs.results, id)
			n++
		}
	}
	return n
}

func (rs *ResultStore) Size() int {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	return len(rs.results)
}

// ---- farm metrics ----------------------------------------------------------

type FarmMetrics struct {
	TotalSubmitted    int64   `json:"total_submitted"`
	TotalCompleted    int64   `json:"total_completed"`
	TotalFailed       int64   `json:"total_failed"`
	AvgDurationMs     float64 `json:"avg_duration_ms"`
	ThroughputPerSec  float64 `json:"throughput_per_sec"`
	totalDurationMs   int64
	firstSubmitTime   time.Time
}

func (fm *FarmMetrics) RecordSubmit() {
	atomic.AddInt64(&fm.TotalSubmitted, 1)
	if fm.firstSubmitTime.IsZero() {
		fm.firstSubmitTime = time.Now()
	}
}

func (fm *FarmMetrics) RecordComplete(durationMs int64) {
	atomic.AddInt64(&fm.TotalCompleted, 1)
	atomic.AddInt64(&fm.totalDurationMs, durationMs)
}

func (fm *FarmMetrics) RecordFail() {
	atomic.AddInt64(&fm.TotalFailed, 1)
}

func (fm *FarmMetrics) Snapshot() FarmMetrics {
	completed := atomic.LoadInt64(&fm.TotalCompleted)
	totalDur := atomic.LoadInt64(&fm.totalDurationMs)
	snap := FarmMetrics{
		TotalSubmitted: atomic.LoadInt64(&fm.TotalSubmitted),
		TotalCompleted: completed,
		TotalFailed:    atomic.LoadInt64(&fm.TotalFailed),
	}
	if completed > 0 {
		snap.AvgDurationMs = float64(totalDur) / float64(completed)
	}
	if !fm.firstSubmitTime.IsZero() {
		elapsed := time.Since(fm.firstSubmitTime).Seconds()
		if elapsed > 0 {
			snap.ThroughputPerSec = float64(completed) / elapsed
		}
	}
	return snap
}

// ---- worker ----------------------------------------------------------------

type Worker struct {
	id      int
	engine  *FarmEngine
	stopCh  chan struct{}
}

func NewWorker(id int, engine *FarmEngine) *Worker {
	return &Worker{
		id:     id,
		engine: engine,
		stopCh: make(chan struct{}),
	}
}

func (w *Worker) Run() {
	log.Printf("worker %d started", w.id)
	for {
		select {
		case <-w.stopCh:
			log.Printf("worker %d stopped", w.id)
			return
		default:
		}

		job, ok := w.engine.queue.Pop()
		if !ok {
			return
		}

		w.execute(job)
	}
}

func (w *Worker) execute(job *BacktestJob) {
	now := time.Now()
	job.StartedAt = &now
	job.Status = StatusRunning
	job.WorkerID = w.id

	w.engine.store.Put(job)

	defer func() {
		if r := recover(); r != nil {
			log.Printf("worker %d: panic running job %s: %v", w.id, job.ID, r)
			job.Status = StatusFailed
			job.Error = fmt.Sprintf("panic: %v", r)
			end := time.Now()
			job.CompletedAt = &end
			job.DurationMs = end.Sub(now).Milliseconds()
			w.engine.metrics.RecordFail()
			w.engine.store.Put(job)
		}
	}()

	result, err := w.runBacktest(job)

	end := time.Now()
	job.CompletedAt = &end
	job.DurationMs = end.Sub(now).Milliseconds()

	if err != nil {
		job.Status = StatusFailed
		job.Error = err.Error()
		w.engine.metrics.RecordFail()
	} else {
		job.Status = StatusCompleted
		job.Result = result
		w.engine.metrics.RecordComplete(job.DurationMs)
		w.engine.notifier.OnJobComplete(job)
	}

	w.engine.store.Put(job)
}

func (w *Worker) runBacktest(job *BacktestJob) (*BacktestResult, error) {
	cfg := job.Config

	// Check for cancellation
	select {
	case <-job.cancelCh:
		return nil, fmt.Errorf("cancelled")
	default:
	}

	// Deterministic seed from config
	seed := hashConfig(cfg)
	rng := newLCG(seed)

	// Generate synthetic price series (252 trading days)
	nDays := 252
	if cfg.DateRange != "" {
		nDays = parseDateRangeDays(cfg.DateRange)
	}
	prices := generatePrices(rng, nDays, cfg.Symbol)

	// Generate signal based on strategy
	signals := generateSignal(rng, prices, cfg.Strategy, cfg.Parameters)

	// Compute returns
	returns := computeReturns(prices)

	// Apply signal to get strategy returns, accounting for costs
	costPerTrade := cfg.CostBPS / 10000.0
	stratReturns, nTrades := applySignal(signals, returns, costPerTrade)

	// Check cancellation mid-computation
	select {
	case <-job.cancelCh:
		return nil, fmt.Errorf("cancelled")
	default:
	}

	// Compute performance metrics
	sharpe := computeSharpe(stratReturns)
	sortino := computeSortino(stratReturns)
	totalReturn := computeTotalReturn(stratReturns)
	maxDD := computeMaxDrawdown(stratReturns)
	calmar := computeCalmar(totalReturn, maxDD)
	winRate := computeWinRate(stratReturns)
	pf := computeProfitFactor(stratReturns)
	ic := computeIC(signals, returns)

	result := &BacktestResult{
		Sharpe:       roundTo(sharpe, 4),
		Sortino:      roundTo(sortino, 4),
		Calmar:       roundTo(calmar, 4),
		TotalReturn:  roundTo(totalReturn, 6),
		MaxDrawdown:  roundTo(maxDD, 6),
		WinRate:      roundTo(winRate, 4),
		ProfitFactor: roundTo(pf, 4),
		NTrades:      nTrades,
		IC:           roundTo(ic, 4),
	}

	return result, nil
}

func (w *Worker) Stop() {
	close(w.stopCh)
}

// ---- synthetic data helpers ------------------------------------------------

type lcg struct {
	state uint64
}

func newLCG(seed uint64) *lcg {
	if seed == 0 {
		seed = 42
	}
	return &lcg{state: seed}
}

func (l *lcg) Next() uint64 {
	l.state = l.state*6364136223846793005 + 1442695040888963407
	return l.state
}

func (l *lcg) Float64() float64 {
	return float64(l.Next()>>11) / float64(1<<53)
}

func (l *lcg) NormFloat64() float64 {
	// Box-Muller
	u1 := l.Float64()
	u2 := l.Float64()
	if u1 < 1e-15 {
		u1 = 1e-15
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func hashConfig(cfg BacktestConfig) uint64 {
	h := uint64(0)
	for _, c := range cfg.Strategy {
		h = h*31 + uint64(c)
	}
	for _, c := range cfg.Symbol {
		h = h*37 + uint64(c)
	}
	for _, c := range cfg.Timeframe {
		h = h*41 + uint64(c)
	}
	// Hash parameters deterministically
	keys := make([]string, 0, len(cfg.Parameters))
	for k := range cfg.Parameters {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		for _, c := range k {
			h = h*43 + uint64(c)
		}
		v := fmt.Sprintf("%v", cfg.Parameters[k])
		for _, c := range v {
			h = h*47 + uint64(c)
		}
	}
	return h
}

func parseDateRangeDays(dr string) int {
	// format: "2020-01-01:2021-01-01" or just a number
	parts := strings.Split(dr, ":")
	if len(parts) == 2 {
		t1, e1 := time.Parse("2006-01-02", parts[0])
		t2, e2 := time.Parse("2006-01-02", parts[1])
		if e1 == nil && e2 == nil {
			days := int(t2.Sub(t1).Hours() / 24)
			if days > 0 {
				return days
			}
		}
	}
	return 252
}

func generatePrices(rng *lcg, n int, symbol string) []float64 {
	prices := make([]float64, n)
	// Start price depends on symbol
	p := 100.0
	switch {
	case strings.Contains(symbol, "BTC"):
		p = 30000.0
	case strings.Contains(symbol, "ETH"):
		p = 2000.0
	case strings.Contains(symbol, "SPY"):
		p = 400.0
	case strings.Contains(symbol, "EUR"):
		p = 1.10
	}

	drift := 0.0002  // slight upward drift
	vol := 0.015     // daily vol
	prices[0] = p
	for i := 1; i < n; i++ {
		ret := drift + vol*rng.NormFloat64()
		prices[i] = prices[i-1] * (1 + ret)
		if prices[i] < 0.01 {
			prices[i] = 0.01
		}
	}
	return prices
}

func generateSignal(rng *lcg, prices []float64, strategy string, params map[string]interface{}) []float64 {
	n := len(prices)
	signals := make([]float64, n)

	switch strings.ToLower(strategy) {
	case "momentum", "mom":
		lookback := getParamInt(params, "lookback", 20)
		for i := lookback; i < n; i++ {
			ret := (prices[i] - prices[i-lookback]) / prices[i-lookback]
			signals[i] = ret
		}
	case "mean_reversion", "mr", "meanrev":
		window := getParamInt(params, "window", 20)
		nStd := getParamFloat(params, "n_std", 2.0)
		for i := window; i < n; i++ {
			sum := 0.0
			for j := i - window; j < i; j++ {
				sum += prices[j]
			}
			mean := sum / float64(window)
			sumSq := 0.0
			for j := i - window; j < i; j++ {
				d := prices[j] - mean
				sumSq += d * d
			}
			std := math.Sqrt(sumSq / float64(window))
			if std < 1e-10 {
				std = 1e-10
			}
			z := (prices[i] - mean) / std
			signals[i] = -z / nStd // short when above band, long when below
		}
	case "breakout", "bo":
		window := getParamInt(params, "window", 50)
		for i := window; i < n; i++ {
			high := prices[i-window]
			low := prices[i-window]
			for j := i - window + 1; j < i; j++ {
				if prices[j] > high {
					high = prices[j]
				}
				if prices[j] < low {
					low = prices[j]
				}
			}
			rng_ := high - low
			if rng_ < 1e-10 {
				rng_ = 1e-10
			}
			pos := (prices[i] - low) / rng_
			signals[i] = pos*2 - 1 // -1 to +1
		}
	case "pairs", "stat_arb":
		halflife := getParamInt(params, "halflife", 10)
		decay := math.Exp(-1.0 / float64(halflife))
		ewma := prices[0]
		for i := 1; i < n; i++ {
			ewma = decay*ewma + (1-decay)*prices[i]
			signals[i] = -(prices[i] - ewma) / ewma * 100
		}
	case "trend_follow", "tf":
		fastPeriod := getParamInt(params, "fast", 10)
		slowPeriod := getParamInt(params, "slow", 50)
		fast := ema(prices, fastPeriod)
		slow := ema(prices, slowPeriod)
		for i := 0; i < n; i++ {
			if slow[i] != 0 {
				signals[i] = (fast[i] - slow[i]) / slow[i] * 100
			}
		}
	case "volatility", "vol_target":
		targetVol := getParamFloat(params, "target_vol", 0.15)
		window := getParamInt(params, "window", 20)
		rets := computeReturns(prices)
		for i := window; i < n; i++ {
			sum := 0.0
			sumSq := 0.0
			for j := i - window; j < i; j++ {
				sum += rets[j]
				sumSq += rets[j] * rets[j]
			}
			mean := sum / float64(window)
			variance := sumSq/float64(window) - mean*mean
			if variance < 1e-15 {
				variance = 1e-15
			}
			realizedVol := math.Sqrt(variance) * math.Sqrt(252)
			scalar := targetVol / realizedVol
			if scalar > 3 {
				scalar = 3
			}
			// Trend signal scaled by vol
			if i > 0 {
				signals[i] = scalar * rets[i-1]
			}
		}
	case "rsi":
		period := getParamInt(params, "period", 14)
		overbought := getParamFloat(params, "overbought", 70)
		oversold := getParamFloat(params, "oversold", 30)
		rsiVals := computeRSI(prices, period)
		for i := 0; i < n; i++ {
			if rsiVals[i] > overbought {
				signals[i] = -(rsiVals[i] - overbought) / (100 - overbought)
			} else if rsiVals[i] < oversold {
				signals[i] = (oversold - rsiVals[i]) / oversold
			}
		}
	case "macd":
		fastP := getParamInt(params, "fast", 12)
		slowP := getParamInt(params, "slow", 26)
		sigP := getParamInt(params, "signal", 9)
		fast := ema(prices, fastP)
		slow := ema(prices, slowP)
		macdLine := make([]float64, n)
		for i := 0; i < n; i++ {
			macdLine[i] = fast[i] - slow[i]
		}
		sigLine := ema(macdLine, sigP)
		for i := 0; i < n; i++ {
			signals[i] = macdLine[i] - sigLine[i]
			if prices[i] != 0 {
				signals[i] /= prices[i] // normalize
			}
		}
	default:
		// Random signal with slight alpha
		alpha := getParamFloat(params, "alpha", 0.02)
		for i := 1; i < n; i++ {
			noise := rng.NormFloat64()
			ret := (prices[i] - prices[i-1]) / prices[i-1]
			signals[i] = alpha*ret + (1-alpha)*noise*0.1
		}
	}

	// Clip signals to [-1, 1]
	for i := range signals {
		if signals[i] > 1 {
			signals[i] = 1
		} else if signals[i] < -1 {
			signals[i] = -1
		}
	}

	return signals
}

func computeRSI(prices []float64, period int) []float64 {
	n := len(prices)
	rsi := make([]float64, n)
	if n < period+1 {
		return rsi
	}
	gains := 0.0
	losses := 0.0
	for i := 1; i <= period; i++ {
		diff := prices[i] - prices[i-1]
		if diff > 0 {
			gains += diff
		} else {
			losses -= diff
		}
	}
	avgGain := gains / float64(period)
	avgLoss := losses / float64(period)
	if avgLoss == 0 {
		rsi[period] = 100
	} else {
		rs := avgGain / avgLoss
		rsi[period] = 100 - 100/(1+rs)
	}
	for i := period + 1; i < n; i++ {
		diff := prices[i] - prices[i-1]
		if diff > 0 {
			avgGain = (avgGain*float64(period-1) + diff) / float64(period)
			avgLoss = (avgLoss * float64(period-1)) / float64(period)
		} else {
			avgGain = (avgGain * float64(period-1)) / float64(period)
			avgLoss = (avgLoss*float64(period-1) - diff) / float64(period)
		}
		if avgLoss == 0 {
			rsi[i] = 100
		} else {
			rs := avgGain / avgLoss
			rsi[i] = 100 - 100/(1+rs)
		}
	}
	return rsi
}

func ema(data []float64, period int) []float64 {
	n := len(data)
	out := make([]float64, n)
	if n == 0 || period <= 0 {
		return out
	}
	alpha := 2.0 / float64(period+1)
	out[0] = data[0]
	for i := 1; i < n; i++ {
		out[i] = alpha*data[i] + (1-alpha)*out[i-1]
	}
	return out
}

func computeReturns(prices []float64) []float64 {
	n := len(prices)
	if n < 2 {
		return nil
	}
	rets := make([]float64, n)
	rets[0] = 0
	for i := 1; i < n; i++ {
		if prices[i-1] != 0 {
			rets[i] = (prices[i] - prices[i-1]) / prices[i-1]
		}
	}
	return rets
}

func applySignal(signals, returns []float64, costPerTrade float64) ([]float64, int) {
	n := len(signals)
	if n == 0 {
		return nil, 0
	}
	stratRets := make([]float64, n)
	nTrades := 0
	prevPos := 0.0
	for i := 1; i < n; i++ {
		pos := signals[i-1] // trade on previous signal
		stratRets[i] = pos * returns[i]
		// Cost for position change
		turnover := math.Abs(pos - prevPos)
		if turnover > 0.01 {
			nTrades++
			stratRets[i] -= turnover * costPerTrade
		}
		prevPos = pos
	}
	return stratRets, nTrades
}

func computeSharpe(returns []float64) float64 {
	if len(returns) < 2 {
		return 0
	}
	mean, std := meanStd(returns)
	if std < 1e-10 {
		return 0
	}
	return (mean / std) * math.Sqrt(252)
}

func computeSortino(returns []float64) float64 {
	if len(returns) < 2 {
		return 0
	}
	mean := 0.0
	for _, r := range returns {
		mean += r
	}
	mean /= float64(len(returns))

	downVar := 0.0
	nDown := 0
	for _, r := range returns {
		if r < 0 {
			downVar += r * r
			nDown++
		}
	}
	if nDown == 0 {
		return 10.0 // cap
	}
	downStd := math.Sqrt(downVar / float64(len(returns)))
	if downStd < 1e-10 {
		return 10.0
	}
	return (mean / downStd) * math.Sqrt(252)
}

func computeTotalReturn(returns []float64) float64 {
	cum := 1.0
	for _, r := range returns {
		cum *= (1 + r)
	}
	return cum - 1
}

func computeMaxDrawdown(returns []float64) float64 {
	peak := 1.0
	equity := 1.0
	maxDD := 0.0
	for _, r := range returns {
		equity *= (1 + r)
		if equity > peak {
			peak = equity
		}
		dd := (peak - equity) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

func computeCalmar(totalReturn, maxDD float64) float64 {
	if maxDD < 1e-10 {
		return 0
	}
	annualized := totalReturn // simplified (assume ~1yr of data)
	return annualized / maxDD
}

func computeWinRate(returns []float64) float64 {
	wins := 0
	total := 0
	for _, r := range returns {
		if r != 0 {
			total++
			if r > 0 {
				wins++
			}
		}
	}
	if total == 0 {
		return 0
	}
	return float64(wins) / float64(total)
}

func computeProfitFactor(returns []float64) float64 {
	grossProfit := 0.0
	grossLoss := 0.0
	for _, r := range returns {
		if r > 0 {
			grossProfit += r
		} else if r < 0 {
			grossLoss -= r
		}
	}
	if grossLoss < 1e-15 {
		return 10.0
	}
	return grossProfit / grossLoss
}

func computeIC(signals, returns []float64) float64 {
	// Information coefficient = rank correlation between signal and next return
	n := len(signals)
	if n < 10 {
		return 0
	}
	// Use Pearson on signal[i] vs return[i+1] as approximation
	pairs := 0
	sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := 0; i < n-1; i++ {
		x := signals[i]
		y := returns[i+1]
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
		sumY2 += y * y
		pairs++
	}
	if pairs == 0 {
		return 0
	}
	nf := float64(pairs)
	num := nf*sumXY - sumX*sumY
	den := math.Sqrt((nf*sumX2 - sumX*sumX) * (nf*sumY2 - sumY*sumY))
	if den < 1e-15 {
		return 0
	}
	return num / den
}

func meanStd(data []float64) (float64, float64) {
	n := float64(len(data))
	if n == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean := sum / n
	sumSq := 0.0
	for _, v := range data {
		d := v - mean
		sumSq += d * d
	}
	return mean, math.Sqrt(sumSq / n)
}

func getParamInt(params map[string]interface{}, key string, def int) int {
	v, ok := params[key]
	if !ok {
		return def
	}
	switch t := v.(type) {
	case float64:
		return int(t)
	case int:
		return t
	case json.Number:
		i, err := t.Int64()
		if err == nil {
			return int(i)
		}
	}
	return def
}

func getParamFloat(params map[string]interface{}, key string, def float64) float64 {
	v, ok := params[key]
	if !ok {
		return def
	}
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case json.Number:
		f, err := t.Float64()
		if err == nil {
			return f
		}
	}
	return def
}

func roundTo(v float64, decimals int) float64 {
	pow := math.Pow(10, float64(decimals))
	return math.Round(v*pow) / pow
}

// ---- grid search generator -------------------------------------------------

type GridSearchGenerator struct{}

func (g *GridSearchGenerator) Generate(paramGrid map[string][]interface{}) []map[string]interface{} {
	keys := make([]string, 0, len(paramGrid))
	for k := range paramGrid {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	values := make([][]interface{}, len(keys))
	for i, k := range keys {
		values[i] = paramGrid[k]
	}

	totalCombos := 1
	for _, v := range values {
		totalCombos *= len(v)
		if totalCombos > 100000 {
			totalCombos = 100000
			break
		}
	}

	combos := make([]map[string]interface{}, 0, totalCombos)
	indices := make([]int, len(keys))

	for {
		if len(combos) >= 100000 {
			break
		}
		combo := make(map[string]interface{}, len(keys))
		for i, k := range keys {
			combo[k] = values[i][indices[i]]
		}
		combos = append(combos, combo)

		// Increment indices
		carry := true
		for i := len(indices) - 1; i >= 0 && carry; i-- {
			indices[i]++
			if indices[i] < len(values[i]) {
				carry = false
			} else {
				indices[i] = 0
			}
		}
		if carry {
			break
		}
	}

	return combos
}

// ---- landscape builder -----------------------------------------------------

type LandscapePoint struct {
	Strategy   string                 `json:"strategy"`
	Symbol     string                 `json:"symbol"`
	Timeframe  string                 `json:"timeframe"`
	Parameters map[string]interface{} `json:"parameters"`
	Sharpe     float64                `json:"sharpe"`
	Sortino    float64                `json:"sortino"`
	Calmar     float64                `json:"calmar"`
	Return     float64                `json:"total_return"`
	MaxDD      float64                `json:"max_drawdown"`
	NTrades    int                    `json:"n_trades"`
}

type AlphaLandscape struct {
	Strategy             string                     `json:"strategy"`
	TotalPoints          int                        `json:"total_points"`
	BestOverall          *LandscapePoint            `json:"best_overall"`
	BestBySymbol         map[string]*LandscapePoint `json:"best_by_symbol"`
	BestByTimeframe      map[string]*LandscapePoint `json:"best_by_timeframe"`
	ParameterSensitivity map[string]float64         `json:"parameter_sensitivity"`
	RegimeBreakdown      map[string]float64         `json:"regime_breakdown"`
	SharpeDistribution   DistributionStats          `json:"sharpe_distribution"`
	Points               []LandscapePoint           `json:"points"`
}

type DistributionStats struct {
	Mean        float64   `json:"mean"`
	Std         float64   `json:"std"`
	Median      float64   `json:"median"`
	Skew        float64   `json:"skew"`
	Kurtosis    float64   `json:"kurtosis"`
	Min         float64   `json:"min"`
	Max         float64   `json:"max"`
	Percentiles []float64 `json:"percentiles_5_25_50_75_95"`
}

type HeatmapCell struct {
	Param1 interface{} `json:"param1"`
	Param2 interface{} `json:"param2"`
	Sharpe float64     `json:"sharpe"`
	Return float64     `json:"total_return"`
	MaxDD  float64     `json:"max_drawdown"`
}

type HeatmapData struct {
	Strategy   string        `json:"strategy"`
	Symbol     string        `json:"symbol"`
	Param1Name string        `json:"param1_name"`
	Param2Name string        `json:"param2_name"`
	Cells      []HeatmapCell `json:"cells"`
	BestCell   *HeatmapCell  `json:"best_cell"`
}

type LandscapeBuilder struct {
	store *ResultStore
}

func NewLandscapeBuilder(store *ResultStore) *LandscapeBuilder {
	return &LandscapeBuilder{store: store}
}

func (lb *LandscapeBuilder) BuildForStrategy(strategy string) *AlphaLandscape {
	completed := lb.store.Completed()
	points := make([]LandscapePoint, 0)

	for _, j := range completed {
		if !strings.EqualFold(j.Config.Strategy, strategy) {
			continue
		}
		pt := LandscapePoint{
			Strategy:   j.Config.Strategy,
			Symbol:     j.Config.Symbol,
			Timeframe:  j.Config.Timeframe,
			Parameters: j.Config.Parameters,
			Sharpe:     j.Result.Sharpe,
			Sortino:    j.Result.Sortino,
			Calmar:     j.Result.Calmar,
			Return:     j.Result.TotalReturn,
			MaxDD:      j.Result.MaxDrawdown,
			NTrades:    j.Result.NTrades,
		}
		points = append(points, pt)
	}

	if len(points) == 0 {
		return &AlphaLandscape{Strategy: strategy, TotalPoints: 0}
	}

	landscape := &AlphaLandscape{
		Strategy:        strategy,
		TotalPoints:     len(points),
		BestBySymbol:    make(map[string]*LandscapePoint),
		BestByTimeframe: make(map[string]*LandscapePoint),
		Points:          points,
	}

	// Find best overall
	var best *LandscapePoint
	for i := range points {
		if best == nil || points[i].Sharpe > best.Sharpe {
			best = &points[i]
		}
	}
	landscape.BestOverall = best

	// Best by symbol
	for i := range points {
		sym := points[i].Symbol
		cur, exists := landscape.BestBySymbol[sym]
		if !exists || points[i].Sharpe > cur.Sharpe {
			p := points[i]
			landscape.BestBySymbol[sym] = &p
		}
	}

	// Best by timeframe
	for i := range points {
		tf := points[i].Timeframe
		cur, exists := landscape.BestByTimeframe[tf]
		if !exists || points[i].Sharpe > cur.Sharpe {
			p := points[i]
			landscape.BestByTimeframe[tf] = &p
		}
	}

	// Sharpe distribution
	sharpes := make([]float64, len(points))
	for i, pt := range points {
		sharpes[i] = pt.Sharpe
	}
	landscape.SharpeDistribution = computeDistribution(sharpes)

	// Parameter sensitivity
	landscape.ParameterSensitivity = lb.computeParamSensitivity(points)

	// Regime breakdown (simplified: split by return sign of underlying)
	landscape.RegimeBreakdown = lb.computeRegimeBreakdown(points)

	return landscape
}

func (lb *LandscapeBuilder) BuildHeatmap(strategy, symbol string) *HeatmapData {
	completed := lb.store.Completed()
	cells := make([]HeatmapCell, 0)

	// Collect all parameter names used
	paramNames := make(map[string]bool)
	for _, j := range completed {
		if !strings.EqualFold(j.Config.Strategy, strategy) || !strings.EqualFold(j.Config.Symbol, symbol) {
			continue
		}
		for k := range j.Config.Parameters {
			paramNames[k] = true
		}
	}

	// Pick first two parameter names alphabetically
	names := make([]string, 0, len(paramNames))
	for k := range paramNames {
		names = append(names, k)
	}
	sort.Strings(names)

	p1Name := ""
	p2Name := ""
	if len(names) >= 1 {
		p1Name = names[0]
	}
	if len(names) >= 2 {
		p2Name = names[1]
	}

	var bestCell *HeatmapCell
	for _, j := range completed {
		if !strings.EqualFold(j.Config.Strategy, strategy) || !strings.EqualFold(j.Config.Symbol, symbol) {
			continue
		}
		cell := HeatmapCell{
			Sharpe: j.Result.Sharpe,
			Return: j.Result.TotalReturn,
			MaxDD:  j.Result.MaxDrawdown,
		}
		if p1Name != "" {
			cell.Param1 = j.Config.Parameters[p1Name]
		}
		if p2Name != "" {
			cell.Param2 = j.Config.Parameters[p2Name]
		}
		cells = append(cells, cell)
		if bestCell == nil || cell.Sharpe > bestCell.Sharpe {
			c := cell
			bestCell = &c
		}
	}

	return &HeatmapData{
		Strategy:   strategy,
		Symbol:     symbol,
		Param1Name: p1Name,
		Param2Name: p2Name,
		Cells:      cells,
		BestCell:   bestCell,
	}
}

func (lb *LandscapeBuilder) computeParamSensitivity(points []LandscapePoint) map[string]float64 {
	sensitivity := make(map[string]float64)
	if len(points) < 2 {
		return sensitivity
	}

	// For each parameter, compute std of Sharpe grouped by parameter value
	paramValues := make(map[string]map[string][]float64) // param -> value_str -> sharpes
	for _, pt := range points {
		for k, v := range pt.Parameters {
			if paramValues[k] == nil {
				paramValues[k] = make(map[string][]float64)
			}
			vs := fmt.Sprintf("%v", v)
			paramValues[k][vs] = append(paramValues[k][vs], pt.Sharpe)
		}
	}

	for param, groups := range paramValues {
		if len(groups) < 2 {
			sensitivity[param] = 0
			continue
		}
		means := make([]float64, 0, len(groups))
		for _, sharpes := range groups {
			sum := 0.0
			for _, s := range sharpes {
				sum += s
			}
			means = append(means, sum/float64(len(sharpes)))
		}
		_, std := meanStd(means)
		sensitivity[param] = roundTo(std, 4)
	}
	return sensitivity
}

func (lb *LandscapeBuilder) computeRegimeBreakdown(points []LandscapePoint) map[string]float64 {
	// Simplified: classify by total return sign
	regimes := map[string][]float64{
		"positive_return": {},
		"negative_return": {},
		"high_vol":        {},
		"low_vol":         {},
	}
	for _, pt := range points {
		if pt.Return >= 0 {
			regimes["positive_return"] = append(regimes["positive_return"], pt.Sharpe)
		} else {
			regimes["negative_return"] = append(regimes["negative_return"], pt.Sharpe)
		}
		if pt.MaxDD > 0.15 {
			regimes["high_vol"] = append(regimes["high_vol"], pt.Sharpe)
		} else {
			regimes["low_vol"] = append(regimes["low_vol"], pt.Sharpe)
		}
	}

	result := make(map[string]float64)
	for regime, sharpes := range regimes {
		if len(sharpes) == 0 {
			continue
		}
		sum := 0.0
		for _, s := range sharpes {
			sum += s
		}
		result[regime] = roundTo(sum/float64(len(sharpes)), 4)
	}
	return result
}

func computeDistribution(data []float64) DistributionStats {
	n := len(data)
	if n == 0 {
		return DistributionStats{}
	}

	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)

	mean, std := meanStd(data)

	// Skew
	skew := 0.0
	if std > 1e-10 {
		for _, v := range data {
			d := (v - mean) / std
			skew += d * d * d
		}
		skew /= float64(n)
	}

	// Kurtosis
	kurt := 0.0
	if std > 1e-10 {
		for _, v := range data {
			d := (v - mean) / std
			kurt += d * d * d * d
		}
		kurt = kurt/float64(n) - 3
	}

	percentile := func(p float64) float64 {
		idx := p / 100.0 * float64(n-1)
		lo := int(math.Floor(idx))
		hi := int(math.Ceil(idx))
		if lo == hi || hi >= n {
			return sorted[lo]
		}
		frac := idx - float64(lo)
		return sorted[lo]*(1-frac) + sorted[hi]*frac
	}

	return DistributionStats{
		Mean:     roundTo(mean, 4),
		Std:      roundTo(std, 4),
		Median:   roundTo(percentile(50), 4),
		Skew:     roundTo(skew, 4),
		Kurtosis: roundTo(kurt, 4),
		Min:      roundTo(sorted[0], 4),
		Max:      roundTo(sorted[n-1], 4),
		Percentiles: []float64{
			roundTo(percentile(5), 4),
			roundTo(percentile(25), 4),
			roundTo(percentile(50), 4),
			roundTo(percentile(75), 4),
			roundTo(percentile(95), 4),
		},
	}
}

// ---- notification engine ---------------------------------------------------

type NotificationEngine struct {
	webhookURL string
	mu         sync.Mutex
	bestSharpe float64
	client     *http.Client
}

func NewNotificationEngine(url string) *NotificationEngine {
	return &NotificationEngine{
		webhookURL: url,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

func (ne *NotificationEngine) OnJobComplete(job *BacktestJob) {
	if ne.webhookURL == "" {
		return
	}
	if job.Result == nil {
		return
	}

	ne.mu.Lock()
	newRecord := job.Result.Sharpe > ne.bestSharpe
	if newRecord {
		ne.bestSharpe = job.Result.Sharpe
	}
	ne.mu.Unlock()

	if newRecord {
		ne.sendWebhook(map[string]interface{}{
			"event":    "new_record",
			"job_id":   job.ID,
			"strategy": job.Config.Strategy,
			"symbol":   job.Config.Symbol,
			"sharpe":   job.Result.Sharpe,
			"return":   job.Result.TotalReturn,
		})
	}
}

func (ne *NotificationEngine) OnBatchComplete(batchID string, count int, bestSharpe float64) {
	if ne.webhookURL == "" {
		return
	}
	ne.sendWebhook(map[string]interface{}{
		"event":       "batch_complete",
		"batch_id":    batchID,
		"jobs":        count,
		"best_sharpe": bestSharpe,
	})
}

func (ne *NotificationEngine) sendWebhook(payload map[string]interface{}) {
	data, err := json.Marshal(payload)
	if err != nil {
		log.Printf("notification marshal error: %v", err)
		return
	}
	go func() {
		resp, err := ne.client.Post(ne.webhookURL, "application/json", bytes.NewReader(data))
		if err != nil {
			log.Printf("webhook error: %v", err)
			return
		}
		resp.Body.Close()
	}()
}

// ---- farm engine (orchestrator) --------------------------------------------

type FarmEngine struct {
	Config    FarmConfig
	queue     *JobQueue
	store     *ResultStore
	metrics   *FarmMetrics
	landscape *LandscapeBuilder
	notifier  *NotificationEngine
	workers   []*Worker
	startedAt time.Time
	stopCh    chan struct{}
	running   sync.Map // id -> *BacktestJob for running jobs
}

func NewFarmEngine(cfg FarmConfig) *FarmEngine {
	store := NewResultStore(cfg.ResultRetentionS)
	return &FarmEngine{
		Config:    cfg,
		queue:     NewJobQueue(),
		store:     store,
		metrics:   &FarmMetrics{},
		landscape: NewLandscapeBuilder(store),
		notifier:  NewNotificationEngine(cfg.NotificationURL),
		stopCh:    make(chan struct{}),
	}
}

func (fe *FarmEngine) Start() {
	fe.startedAt = time.Now()
	fe.workers = make([]*Worker, fe.Config.MaxWorkers)
	for i := 0; i < fe.Config.MaxWorkers; i++ {
		w := NewWorker(i, fe)
		fe.workers[i] = w
		go w.Run()
	}

	// TTL cleanup goroutine
	go fe.cleanupLoop()

	log.Printf("farm engine started with %d workers", fe.Config.MaxWorkers)
}

func (fe *FarmEngine) Stop() {
	close(fe.stopCh)
	fe.queue.Close()
	for _, w := range fe.workers {
		w.Stop()
	}
	log.Println("farm engine stopped")
}

func (fe *FarmEngine) cleanupLoop() {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-fe.stopCh:
			return
		case <-ticker.C:
			n := fe.store.CleanExpired()
			if n > 0 {
				log.Printf("cleaned %d expired results", n)
			}
		}
	}
}

func (fe *FarmEngine) Submit(cfg BacktestConfig) *BacktestJob {
	job := &BacktestJob{
		ID:          newJobID(),
		Config:      cfg,
		Status:      StatusQueued,
		SubmittedAt: time.Now(),
		cancelCh:    make(chan struct{}),
	}
	fe.store.Put(job)
	fe.metrics.RecordSubmit()
	fe.queue.Push(job)
	return job
}

func (fe *FarmEngine) Status() map[string]interface{} {
	counts := fe.store.Count()
	counts[StatusQueued] += fe.queue.Len()
	m := fe.metrics.Snapshot()
	return map[string]interface{}{
		"queued":    counts[StatusQueued],
		"running":   counts[StatusRunning],
		"completed": counts[StatusCompleted],
		"failed":    counts[StatusFailed],
		"cancelled": counts[StatusCancelled],
		"total":     fe.store.Size(),
		"metrics":   m,
		"workers":   fe.Config.MaxWorkers,
		"uptime_s":  int(time.Since(fe.startedAt).Seconds()),
	}
}

func (fe *FarmEngine) ListJobs(status string, limit int) []*BacktestJob {
	var jobs []*BacktestJob
	if status != "" {
		jobs = fe.store.ByStatus(status)
	} else {
		jobs = fe.store.All()
	}
	// Sort by submitted_at descending
	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].SubmittedAt.After(jobs[j].SubmittedAt)
	})
	if len(jobs) > limit {
		jobs = jobs[:limit]
	}
	return jobs
}

func (fe *FarmEngine) GetJob(id string) (*BacktestJob, bool) {
	return fe.store.Get(id)
}

func (fe *FarmEngine) TopResults(top int, sortBy string) []*BacktestJob {
	completed := fe.store.Completed()

	sort.Slice(completed, func(i, j int) bool {
		ri, rj := completed[i].Result, completed[j].Result
		switch sortBy {
		case "sharpe":
			return ri.Sharpe > rj.Sharpe
		case "sortino":
			return ri.Sortino > rj.Sortino
		case "calmar":
			return ri.Calmar > rj.Calmar
		case "return", "total_return":
			return ri.TotalReturn > rj.TotalReturn
		case "win_rate":
			return ri.WinRate > rj.WinRate
		case "profit_factor":
			return ri.ProfitFactor > rj.ProfitFactor
		case "ic":
			return ri.IC > rj.IC
		case "drawdown", "max_drawdown":
			return ri.MaxDrawdown < rj.MaxDrawdown // lower is better
		default:
			return ri.Sharpe > rj.Sharpe
		}
	})

	if len(completed) > top {
		completed = completed[:top]
	}
	return completed
}

func (fe *FarmEngine) LandscapeForStrategy(strategy string) *AlphaLandscape {
	return fe.landscape.BuildForStrategy(strategy)
}

func (fe *FarmEngine) HeatmapData(strategy, symbol string) *HeatmapData {
	return fe.landscape.BuildHeatmap(strategy, symbol)
}

func (fe *FarmEngine) Cancel(id string) bool {
	// Try removing from queue
	if fe.queue.Remove(id) {
		return true
	}
	// Try cancelling a running job
	job, ok := fe.store.Get(id)
	if ok && job.Status == StatusRunning {
		job.Status = StatusCancelled
		select {
		case <-job.cancelCh:
		default:
			close(job.cancelCh)
		}
		return true
	}
	return false
}

func (fe *FarmEngine) CancelAll() int {
	n := fe.queue.Drain()
	// Also cancel running jobs
	all := fe.store.ByStatus(StatusRunning)
	for _, j := range all {
		j.Status = StatusCancelled
		select {
		case <-j.cancelCh:
		default:
			close(j.cancelCh)
		}
		n++
	}
	return n
}

func (fe *FarmEngine) Purge() int {
	return fe.store.PurgeCompleted()
}
