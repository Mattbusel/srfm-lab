package main

import (
	"math"
	"sync"
	"time"
)

// ringBufferSize is the capacity of each in-memory ring buffer.
const ringBufferSize = 1000

// ---------------------------------------------------------------------------
// Metric value types
// ---------------------------------------------------------------------------

// GenomeMetric records the state of a single evolution generation.
type GenomeMetric struct {
	// Generation is the generation number.
	Generation int `json:"generation"`
	// BestFitness is the fitness of the top individual.
	BestFitness float64 `json:"best_fitness"`
	// MeanFitness is the arithmetic mean across the population.
	MeanFitness float64 `json:"mean_fitness"`
	// Diversity is the normalised average pairwise distance in gene space.
	Diversity float64 `json:"diversity"`
	// EvalCount is the cumulative number of fitness evaluations performed.
	EvalCount int64 `json:"eval_count"`
	// Timestamp is when this metric was recorded.
	Timestamp time.Time `json:"timestamp"`
}

// ParamMetric records a single parameter update event.
type ParamMetric struct {
	// ParamName is the name of the LARSA parameter that changed.
	ParamName string `json:"param_name"`
	// Value is the new parameter value.
	Value float64 `json:"value"`
	// PrevValue is the value before this update.
	PrevValue float64 `json:"prev_value"`
	// DeltaPct is the signed percentage change: (Value-PrevValue)/abs(PrevValue)*100.
	// Zero when PrevValue is zero.
	DeltaPct float64 `json:"delta_pct"`
	// Source identifies what triggered the update (e.g. "mutation", "evolution", "manual").
	Source string `json:"source"`
	// Timestamp is when the update occurred.
	Timestamp time.Time `json:"timestamp"`
}

// EvalMetric records the outcome of a single backtest evaluation.
type EvalMetric struct {
	// Sharpe is the Sharpe ratio returned by the backtest.
	Sharpe float64 `json:"sharpe"`
	// MaxDD is the maximum drawdown as a positive fraction.
	MaxDD float64 `json:"max_drawdown"`
	// EvalTimeMs is how long the evaluation took in milliseconds.
	EvalTimeMs int64 `json:"eval_time_ms"`
	// Timestamp is when the evaluation completed.
	Timestamp time.Time `json:"timestamp"`
}

// ---------------------------------------------------------------------------
// Ring buffers
// ---------------------------------------------------------------------------

// genomeRing is a fixed-size ring buffer for GenomeMetrics.
type genomeRing struct {
	buf  [ringBufferSize]GenomeMetric
	head int
	size int
}

func (r *genomeRing) push(m GenomeMetric) {
	r.buf[r.head] = m
	r.head = (r.head + 1) % ringBufferSize
	if r.size < ringBufferSize {
		r.size++
	}
}

// slice returns all elements in insertion order (oldest first).
func (r *genomeRing) slice() []GenomeMetric {
	out := make([]GenomeMetric, r.size)
	start := r.head - r.size
	if start < 0 {
		start += ringBufferSize
	}
	for i := 0; i < r.size; i++ {
		out[i] = r.buf[(start+i)%ringBufferSize]
	}
	return out
}

// last returns the most recent n elements. If n > size, all elements
// are returned.
func (r *genomeRing) last(n int) []GenomeMetric {
	all := r.slice()
	if n >= len(all) {
		return all
	}
	return all[len(all)-n:]
}

// paramRing is a fixed-size ring buffer for ParamMetrics.
type paramRing struct {
	buf  [ringBufferSize]ParamMetric
	head int
	size int
}

func (r *paramRing) push(m ParamMetric) {
	r.buf[r.head] = m
	r.head = (r.head + 1) % ringBufferSize
	if r.size < ringBufferSize {
		r.size++
	}
}

func (r *paramRing) slice() []ParamMetric {
	out := make([]ParamMetric, r.size)
	start := r.head - r.size
	if start < 0 {
		start += ringBufferSize
	}
	for i := 0; i < r.size; i++ {
		out[i] = r.buf[(start+i)%ringBufferSize]
	}
	return out
}

// evalRing is a fixed-size ring buffer for EvalMetrics.
type evalRing struct {
	buf  [ringBufferSize]EvalMetric
	head int
	size int
}

func (r *evalRing) push(m EvalMetric) {
	r.buf[r.head] = m
	r.head = (r.head + 1) % ringBufferSize
	if r.size < ringBufferSize {
		r.size++
	}
}

func (r *evalRing) slice() []EvalMetric {
	out := make([]EvalMetric, r.size)
	start := r.head - r.size
	if start < 0 {
		start += ringBufferSize
	}
	for i := 0; i < r.size; i++ {
		out[i] = r.buf[(start+i)%ringBufferSize]
	}
	return out
}

// ---------------------------------------------------------------------------
// IAEMetrics -- in-memory store
// ---------------------------------------------------------------------------

// IAEMetrics is the central in-memory metrics store for the metrics-server.
// All public methods are safe for concurrent use.
type IAEMetrics struct {
	mu      sync.RWMutex
	genomes genomeRing
	params  paramRing
	evals   evalRing

	// totalEvals is the lifetime count of recorded evaluations.
	totalEvals int64
	// startTime is when the store was created (used for eval rate calculation).
	startTime time.Time
}

// NewIAEMetrics constructs an empty IAEMetrics.
func NewIAEMetrics() *IAEMetrics {
	return &IAEMetrics{
		startTime: time.Now(),
	}
}

// RecordGenome appends a genome generation metric to the ring buffer.
// If gm.DeltaPct is uninitialised, it is computed automatically from
// the previous generation's BestFitness.
func (m *IAEMetrics) RecordGenome(gm GenomeMetric) {
	if gm.Timestamp.IsZero() {
		gm.Timestamp = time.Now()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.genomes.push(gm)
}

// RecordParamUpdate appends a parameter change event to the ring buffer.
// DeltaPct is computed automatically if not already set.
func (m *IAEMetrics) RecordParamUpdate(pm ParamMetric) {
	if pm.Timestamp.IsZero() {
		pm.Timestamp = time.Now()
	}
	// Compute delta percentage if the caller did not set it.
	if pm.DeltaPct == 0 && pm.PrevValue != 0 {
		pm.DeltaPct = (pm.Value - pm.PrevValue) / math.Abs(pm.PrevValue) * 100
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.params.push(pm)
}

// RecordEvaluation appends an evaluation result and increments the
// lifetime counter.
func (m *IAEMetrics) RecordEvaluation(sharpe, maxDD float64, evalTimeMs int64) {
	em := EvalMetric{
		Sharpe:     sharpe,
		MaxDD:      maxDD,
		EvalTimeMs: evalTimeMs,
		Timestamp:  time.Now(),
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.evals.push(em)
	m.totalEvals++
}

// EvaluationsPerHour returns the mean number of evaluations per hour based
// on all evaluations currently in the ring buffer. Returns 0 if fewer than
// two evaluations have been recorded.
func (m *IAEMetrics) EvaluationsPerHour() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	evals := m.evals.slice()
	if len(evals) < 2 {
		return 0
	}
	oldest := evals[0].Timestamp
	newest := evals[len(evals)-1].Timestamp
	elapsed := newest.Sub(oldest).Hours()
	if elapsed <= 0 {
		return 0
	}
	// Number of intervals between n points is n-1.
	return float64(len(evals)-1) / elapsed
}

// FitnessImprovement returns the absolute improvement in BestFitness over the
// last n genome records. If fewer than 2 records exist, returns 0.
// A positive value means fitness improved; negative means it deteriorated.
func (m *IAEMetrics) FitnessImprovement(n int) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	recent := m.genomes.last(n)
	if len(recent) < 2 {
		return 0
	}
	return recent[len(recent)-1].BestFitness - recent[0].BestFitness
}

// LatestGenome returns the most recently recorded GenomeMetric, and false
// if no genomes have been recorded yet.
func (m *IAEMetrics) LatestGenome() (GenomeMetric, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.genomes.size == 0 {
		return GenomeMetric{}, false
	}
	idx := (m.genomes.head - 1 + ringBufferSize) % ringBufferSize
	return m.genomes.buf[idx], true
}

// RecentGenomes returns up to n genome metrics in oldest-first order.
func (m *IAEMetrics) RecentGenomes(n int) []GenomeMetric {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.genomes.last(n)
}

// RecentParams returns up to n param update metrics in oldest-first order.
func (m *IAEMetrics) RecentParams(n int) []ParamMetric {
	m.mu.RLock()
	defer m.mu.RUnlock()
	all := m.params.slice()
	if n >= len(all) {
		return all
	}
	return all[len(all)-n:]
}

// RecentEvals returns all evaluation metrics in the ring buffer in
// oldest-first order (up to ringBufferSize entries).
func (m *IAEMetrics) RecentEvals() []EvalMetric {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.evals.slice()
}

// TotalEvals returns the lifetime evaluation count.
func (m *IAEMetrics) TotalEvals() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.totalEvals
}

// MeanEvalTimeMs returns the arithmetic mean evaluation time across all
// evaluations in the ring buffer. Returns 0 if no evaluations recorded.
func (m *IAEMetrics) MeanEvalTimeMs() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	evals := m.evals.slice()
	if len(evals) == 0 {
		return 0
	}
	var total int64
	for _, e := range evals {
		total += e.EvalTimeMs
	}
	return float64(total) / float64(len(evals))
}

// MeanSharpe returns the arithmetic mean Sharpe ratio across all evaluations
// in the ring buffer.
func (m *IAEMetrics) MeanSharpe() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	evals := m.evals.slice()
	if len(evals) == 0 {
		return 0
	}
	var sum float64
	for _, e := range evals {
		sum += e.Sharpe
	}
	return sum / float64(len(evals))
}
