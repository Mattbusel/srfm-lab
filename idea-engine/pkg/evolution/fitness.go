package evolution

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// FitnessResult -- the output of a single backtest evaluation
// ---------------------------------------------------------------------------

// FitnessResult holds all metrics returned by a backtest evaluation.
type FitnessResult struct {
	// Sharpe is the annualised Sharpe ratio.
	Sharpe float64 `json:"sharpe"`
	// Sortino is the annualised Sortino ratio.
	Sortino float64 `json:"sortino"`
	// MaxDD is the maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
	MaxDD float64 `json:"max_drawdown"`
	// AnnualisedReturn is the geometric mean annual return.
	AnnualisedReturn float64 `json:"annualised_return"`
	// WinRate is the fraction of trades that were profitable.
	WinRate float64 `json:"win_rate"`
	// RegimeRobustness is the harmonic mean of per-regime Sharpe ratios.
	// A higher value indicates the strategy performs consistently across
	// different market regimes (trending, ranging, volatile).
	RegimeRobustness float64 `json:"regime_robustness"`
	// WeightedScore is the composite scalar fitness.
	// Computed by WeightedFitness() after the other fields are populated.
	WeightedScore float64 `json:"weighted_score"`
	// CrowdingDistance is set by the NSGA-II crowding distance assignment.
	// Not populated by the evaluator -- populated by NSGA2Selection.
	CrowdingDistance float64 `json:"crowding_distance,omitempty"`
	// Error carries any evaluation error (empty on success).
	Error string `json:"error,omitempty"`
}

// IsValid returns true when the result contains meaningful (non-error) data.
func (f FitnessResult) IsValid() bool {
	return f.Error == "" && !math.IsNaN(f.Sharpe) && !math.IsInf(f.Sharpe, 0)
}

// ---------------------------------------------------------------------------
// Weighted fitness scalar
// ---------------------------------------------------------------------------

// WeightedFitness computes the composite scalar fitness score from a
// FitnessResult using the fixed weighting scheme:
//
//	score = 0.40*Sharpe + 0.25*Sortino + 0.20*(1/MaxDD) + 0.15*RegimeRobustness
//
// MaxDD of zero is replaced with 1e-6 to avoid division by zero. The result
// is stored in f.WeightedScore and also returned.
func WeightedFitness(f *FitnessResult) float64 {
	maxDD := f.MaxDD
	if maxDD < 1e-6 {
		maxDD = 1e-6
	}
	score := 0.40*f.Sharpe +
		0.25*f.Sortino +
		0.20*(1.0/maxDD) +
		0.15*f.RegimeRobustness
	f.WeightedScore = score
	return score
}

// ---------------------------------------------------------------------------
// FitnessEvaluator interface
// ---------------------------------------------------------------------------

// FitnessEvaluator is the core interface for all fitness evaluation backends.
type FitnessEvaluator interface {
	// Evaluate runs a backtest for the given genome and returns fitness metrics.
	// ctx allows the caller to cancel long-running subprocess evaluations.
	Evaluate(ctx context.Context, genome Genome) FitnessResult
}

// ---------------------------------------------------------------------------
// BacktestFitness -- calls Python backtest subprocess
// ---------------------------------------------------------------------------

// BacktestConfig configures the Python subprocess evaluator.
type BacktestConfig struct {
	// PythonPath is the path to the Python interpreter (default: "python3").
	PythonPath string
	// ScriptPath is the path to the backtest runner script.
	ScriptPath string
	// Timeout is the maximum wall-clock time allowed per evaluation.
	Timeout time.Duration
	// ExtraArgs are appended to the subprocess command after the genome JSON.
	ExtraArgs []string
}

// backtestOutput mirrors the JSON structure written to stdout by the
// Python backtest runner script.
type backtestOutput struct {
	Sharpe           float64            `json:"sharpe"`
	Sortino          float64            `json:"sortino"`
	MaxDrawdown      float64            `json:"max_drawdown"`
	AnnualisedReturn float64            `json:"annualised_return"`
	WinRate          float64            `json:"win_rate"`
	RegimeSharpes    map[string]float64 `json:"regime_sharpes"`
}

// BacktestFitness evaluates a genome by launching a Python backtest subprocess.
// The genome is serialised to JSON and passed as the first argument. The script
// must write a single JSON object to stdout matching backtestOutput.
type BacktestFitness struct {
	cfg BacktestConfig
}

// NewBacktestFitness constructs a BacktestFitness with the given config.
// If PythonPath is empty it defaults to "python3".
func NewBacktestFitness(cfg BacktestConfig) *BacktestFitness {
	if cfg.PythonPath == "" {
		cfg.PythonPath = "python3"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}
	return &BacktestFitness{cfg: cfg}
}

// Evaluate serialises genome to JSON, invokes the Python backtest script, and
// parses the resulting JSON from stdout.
func (b *BacktestFitness) Evaluate(ctx context.Context, genome Genome) FitnessResult {
	genesJSON, err := json.Marshal([]float64(genome))
	if err != nil {
		return FitnessResult{Error: fmt.Sprintf("marshal genome: %v", err)}
	}

	timeout := b.cfg.Timeout
	evalCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args := append([]string{b.cfg.ScriptPath, string(genesJSON)}, b.cfg.ExtraArgs...)
	cmd := exec.CommandContext(evalCtx, b.cfg.PythonPath, args...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		errMsg := fmt.Sprintf("backtest subprocess: %v", err)
		if stderr.Len() > 0 {
			errMsg += ": " + stderr.String()
		}
		return FitnessResult{Error: errMsg}
	}

	var out backtestOutput
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		return FitnessResult{Error: fmt.Sprintf("parse backtest output: %v; raw: %s", err, stdout.String())}
	}

	result := FitnessResult{
		Sharpe:           out.Sharpe,
		Sortino:          out.Sortino,
		MaxDD:            out.MaxDrawdown,
		AnnualisedReturn: out.AnnualisedReturn,
		WinRate:          out.WinRate,
		RegimeRobustness: harmonicMeanRegimeSharpeSharpes(out.RegimeSharpes),
	}
	WeightedFitness(&result)
	return result
}

// harmonicMeanRegimeSharpeSharpes computes the harmonic mean of the per-regime
// Sharpe values. Regimes with Sharpe <= 0 are treated as 0.001 to avoid
// degenerate harmonic mean results while still penalising negative regimes.
func harmonicMeanRegimeSharpeSharpes(regimes map[string]float64) float64 {
	if len(regimes) == 0 {
		return 0
	}
	sumRecip := 0.0
	for _, v := range regimes {
		if v < 0.001 {
			v = 0.001
		}
		sumRecip += 1.0 / v
	}
	return float64(len(regimes)) / sumRecip
}

// ---------------------------------------------------------------------------
// CachedFitnessEvaluator -- LRU cache over genome hash
// ---------------------------------------------------------------------------

// lruEntry is one slot in the LRU cache.
type lruEntry struct {
	key    [32]byte
	result FitnessResult
	prev   *lruEntry
	next   *lruEntry
}

// lruCache is a simple doubly-linked-list LRU cache for FitnessResult values.
// It is not safe for concurrent use -- the caller must hold a lock.
type lruCache struct {
	capacity int
	table    map[[32]byte]*lruEntry
	head     *lruEntry // most recently used
	tail     *lruEntry // least recently used
}

func newLRUCache(capacity int) *lruCache {
	return &lruCache{
		capacity: capacity,
		table:    make(map[[32]byte]*lruEntry, capacity),
	}
}

func (c *lruCache) get(key [32]byte) (FitnessResult, bool) {
	e, ok := c.table[key]
	if !ok {
		return FitnessResult{}, false
	}
	c.moveToFront(e)
	return e.result, true
}

func (c *lruCache) put(key [32]byte, result FitnessResult) {
	if e, ok := c.table[key]; ok {
		e.result = result
		c.moveToFront(e)
		return
	}
	e := &lruEntry{key: key, result: result}
	c.table[key] = e
	c.addFront(e)
	if len(c.table) > c.capacity {
		c.evictTail()
	}
}

func (c *lruCache) addFront(e *lruEntry) {
	e.prev = nil
	e.next = c.head
	if c.head != nil {
		c.head.prev = e
	}
	c.head = e
	if c.tail == nil {
		c.tail = e
	}
}

func (c *lruCache) moveToFront(e *lruEntry) {
	if e == c.head {
		return
	}
	if e.prev != nil {
		e.prev.next = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	}
	if e == c.tail {
		c.tail = e.prev
	}
	e.prev = nil
	e.next = c.head
	if c.head != nil {
		c.head.prev = e
	}
	c.head = e
}

func (c *lruCache) evictTail() {
	if c.tail == nil {
		return
	}
	delete(c.table, c.tail.key)
	if c.tail.prev != nil {
		c.tail.prev.next = nil
	}
	c.tail = c.tail.prev
	if c.tail == nil {
		c.head = nil
	}
}

// genomeHash computes a SHA-256 hash of the genome's float64 values.
// Two genomes with identical gene values will produce the same hash.
func genomeHash(g Genome) [32]byte {
	buf := make([]byte, 8*len(g))
	for i, v := range g {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	return sha256.Sum256(buf)
}

// CachedFitnessEvaluator wraps another FitnessEvaluator and caches results
// keyed by the SHA-256 hash of the genome. Cache capacity defaults to 10000
// entries.
type CachedFitnessEvaluator struct {
	mu    sync.Mutex
	inner FitnessEvaluator
	cache *lruCache
	hits  int64
	misses int64
}

// NewCachedFitnessEvaluator wraps inner with an LRU cache of the given
// capacity (use 0 for the default of 10000).
func NewCachedFitnessEvaluator(inner FitnessEvaluator, capacity int) *CachedFitnessEvaluator {
	if capacity <= 0 {
		capacity = 10000
	}
	return &CachedFitnessEvaluator{
		inner: inner,
		cache: newLRUCache(capacity),
	}
}

// Evaluate returns a cached FitnessResult if one exists for this genome;
// otherwise it delegates to the inner evaluator and caches the result.
func (c *CachedFitnessEvaluator) Evaluate(ctx context.Context, genome Genome) FitnessResult {
	key := genomeHash(genome)

	c.mu.Lock()
	if result, ok := c.cache.get(key); ok {
		c.hits++
		c.mu.Unlock()
		return result
	}
	c.misses++
	c.mu.Unlock()

	result := c.inner.Evaluate(ctx, genome)

	c.mu.Lock()
	c.cache.put(key, result)
	c.mu.Unlock()

	return result
}

// CacheStats returns (hits, misses, size) for monitoring.
func (c *CachedFitnessEvaluator) CacheStats() (hits, misses int64, size int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses, len(c.cache.table)
}

// ---------------------------------------------------------------------------
// ParallelEvaluator -- concurrent worker pool
// ---------------------------------------------------------------------------

// evalJob is one unit of work submitted to the worker pool.
type evalJob struct {
	idx    int
	genome Genome
}

// evalResult carries the result back from a worker.
type evalResult struct {
	idx    int
	result FitnessResult
}

// ParallelEvaluator evaluates a population concurrently using a fixed-size
// worker pool. Results are written back into the Individuals in-place.
type ParallelEvaluator struct {
	inner   FitnessEvaluator
	workers int
}

// NewParallelEvaluator creates a ParallelEvaluator with the given number of
// workers. workers <= 0 defaults to 4.
func NewParallelEvaluator(inner FitnessEvaluator, workers int) *ParallelEvaluator {
	if workers <= 0 {
		workers = 4
	}
	return &ParallelEvaluator{inner: inner, workers: workers}
}

// EvaluatePopulation evaluates all unevaluated individuals in pop concurrently
// and writes their FitnessResult back. Already-evaluated individuals
// (Evaluated==true) are skipped.
func (p *ParallelEvaluator) EvaluatePopulation(ctx context.Context, pop []Individual) []Individual {
	jobs := make(chan evalJob, len(pop))
	results := make(chan evalResult, len(pop))

	// Launch workers.
	var wg sync.WaitGroup
	for w := 0; w < p.workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				r := p.inner.Evaluate(ctx, job.genome)
				results <- evalResult{idx: job.idx, result: r}
			}
		}()
	}

	// Submit jobs for unevaluated individuals.
	submitted := 0
	for i, ind := range pop {
		if !ind.Evaluated {
			jobs <- evalJob{idx: i, genome: ind.Genes}
			submitted++
		}
	}
	close(jobs)

	// Wait for all workers and close results channel.
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results.
	out := make([]Individual, len(pop))
	copy(out, pop)
	for res := range results {
		out[res.idx].Fitness = res.result
		out[res.idx].Evaluated = true
	}
	return out
}

// Workers returns the configured number of workers.
func (p *ParallelEvaluator) Workers() int {
	return p.workers
}
