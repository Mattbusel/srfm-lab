// Package evaluation provides the genome evaluation pipeline for the IAE.
// It runs Python backtests as subprocesses, collects their JSON output, and
// aggregates multi-period results into a single fitness score.
package evaluation

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// EvalConfig -- configuration for a GenomeEvaluator
// ---------------------------------------------------------------------------

// EvalConfig holds all static settings for a GenomeEvaluator instance.
type EvalConfig struct {
	// DataPath is the root directory that the Python backtest reads from.
	DataPath string
	// StartDate is the backtest start in "YYYY-MM-DD" format.
	StartDate string
	// EndDate is the backtest end in "YYYY-MM-DD" format.
	EndDate string
	// PyPath is the Python interpreter to invoke (default: "python").
	PyPath string
	// Timeout is the maximum duration for a single subprocess call.
	// Zero means use the package default (300 seconds).
	Timeout time.Duration
}

// defaults fills in zero-value fields with sensible defaults.
func (c *EvalConfig) defaults() {
	if c.PyPath == "" {
		c.PyPath = "python"
	}
	if c.Timeout == 0 {
		c.Timeout = 300 * time.Second
	}
}

// ---------------------------------------------------------------------------
// EvalResult -- metrics returned by a single backtest run
// ---------------------------------------------------------------------------

// EvalResult holds the performance metrics parsed from the Python backtest
// script's JSON stdout output.
type EvalResult struct {
	// Sharpe is the annualised Sharpe ratio.
	Sharpe float64 `json:"sharpe"`
	// MaxDrawdown is the maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
	MaxDrawdown float64 `json:"max_drawdown"`
	// Calmar is the annualised return divided by the maximum drawdown.
	Calmar float64 `json:"calmar"`
	// WinRate is the fraction of filled orders that were profitable.
	WinRate float64 `json:"win_rate"`
	// NFills is the total number of order fills.
	NFills int `json:"n_fills"`
	// AnnualisedReturn is the geometric mean annual return.
	AnnualisedReturn float64 `json:"annualised_return"`
	// EvalDurationMs is how long the subprocess took in milliseconds.
	EvalDurationMs int64 `json:"eval_duration_ms"`
	// Error is set if the subprocess returned a non-zero exit code.
	// A non-empty Error means the other fields should not be trusted.
	Error string `json:"error,omitempty"`
}

// IsValid returns true when the result does not carry an error and has at
// least one fill -- a prerequisite for any fitness calculation.
func (r EvalResult) IsValid() bool {
	return r.Error == "" && r.NFills > 0
}

// ---------------------------------------------------------------------------
// GenomeEvaluator
// ---------------------------------------------------------------------------

// GenomeEvaluator evaluates a single genome by calling the Python backtest
// script as a subprocess and parsing the JSON result from stdout.
type GenomeEvaluator struct {
	cfg EvalConfig
}

// NewGenomeEvaluator constructs a GenomeEvaluator with the given config.
func NewGenomeEvaluator(cfg EvalConfig) *GenomeEvaluator {
	cfg.defaults()
	return &GenomeEvaluator{cfg: cfg}
}

// Evaluate serialises genome to a temporary JSON file, invokes the Python
// backtest, and returns the parsed EvalResult.
//
// The subprocess is expected to write a single JSON object to stdout matching
// the EvalResult struct. Any non-zero exit code is treated as a failure.
func (e *GenomeEvaluator) Evaluate(genome map[string]float64) (EvalResult, error) {
	start := time.Now()

	// Write params to a temp file so the subprocess can read them safely.
	tmpFile, err := writeTempGenomeJSON(genome)
	if err != nil {
		return EvalResult{Error: err.Error()}, fmt.Errorf("write temp file: %w", err)
	}
	defer os.Remove(tmpFile)

	ctx, cancel := context.WithTimeout(context.Background(), e.cfg.Timeout)
	defer cancel()

	args := e.buildArgs(tmpFile)
	cmd := exec.CommandContext(ctx, e.cfg.PyPath, args...)

	out, err := cmd.Output()
	durationMs := time.Since(start).Milliseconds()

	if err != nil {
		msg := err.Error()
		if ctx.Err() == context.DeadlineExceeded {
			msg = fmt.Sprintf("backtest timed out after %s", e.cfg.Timeout)
		}
		return EvalResult{Error: msg, EvalDurationMs: durationMs}, fmt.Errorf("subprocess: %w", err)
	}

	var result EvalResult
	if err := json.Unmarshal(out, &result); err != nil {
		return EvalResult{Error: fmt.Sprintf("parse output: %v", err), EvalDurationMs: durationMs},
			fmt.Errorf("parse: %w", err)
	}
	result.EvalDurationMs = durationMs
	return result, nil
}

// buildArgs constructs the argument list for the Python subprocess.
func (e *GenomeEvaluator) buildArgs(genomePath string) []string {
	args := []string{"-m", "tools.larsa_v18_backtest", "--genome-file", genomePath}
	if e.cfg.DataPath != "" {
		args = append(args, "--data-path", e.cfg.DataPath)
	}
	if e.cfg.StartDate != "" {
		args = append(args, "--start-date", e.cfg.StartDate)
	}
	if e.cfg.EndDate != "" {
		args = append(args, "--end-date", e.cfg.EndDate)
	}
	return args
}

// writeTempGenomeJSON marshals genome to a temporary JSON file and returns
// the file path. The caller is responsible for removing the file.
func writeTempGenomeJSON(genome map[string]float64) (string, error) {
	data, err := json.Marshal(genome)
	if err != nil {
		return "", fmt.Errorf("marshal genome: %w", err)
	}
	f, err := os.CreateTemp("", "iae-genome-*.json")
	if err != nil {
		return "", fmt.Errorf("create temp: %w", err)
	}
	defer f.Close()
	if _, err := f.Write(data); err != nil {
		_ = os.Remove(f.Name())
		return "", fmt.Errorf("write temp: %w", err)
	}
	return filepath.Clean(f.Name()), nil
}

// ---------------------------------------------------------------------------
// BatchEvaluator
// ---------------------------------------------------------------------------

// BatchEvaluator evaluates multiple genomes in parallel using a fixed-size
// goroutine pool.
type BatchEvaluator struct {
	workers   int
	evaluator *GenomeEvaluator
}

// NewBatchEvaluator constructs a BatchEvaluator.
// workers must be >= 1; values > 32 are capped at 32 to prevent excessive
// resource usage.
func NewBatchEvaluator(workers int, evaluator *GenomeEvaluator) *BatchEvaluator {
	if workers < 1 {
		workers = 1
	}
	if workers > 32 {
		workers = 32
	}
	return &BatchEvaluator{workers: workers, evaluator: evaluator}
}

// EvaluateBatch evaluates all genomes in parallel and returns results in the
// same order as the input slice. If a genome fails evaluation, the
// corresponding EvalResult will have a non-empty Error field.
func (b *BatchEvaluator) EvaluateBatch(genomes []map[string]float64) []EvalResult {
	results := make([]EvalResult, len(genomes))

	type work struct {
		idx    int
		genome map[string]float64
	}

	jobs := make(chan work, len(genomes))
	for i, g := range genomes {
		jobs <- work{idx: i, genome: g}
	}
	close(jobs)

	var wg sync.WaitGroup
	for w := 0; w < b.workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobs {
				res, err := b.evaluator.Evaluate(job.genome)
				if err != nil && res.Error == "" {
					res.Error = err.Error()
				}
				results[job.idx] = res
			}
		}()
	}
	wg.Wait()
	return results
}
