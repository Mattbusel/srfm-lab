package evaluation

import (
	"math"
	"os"
	"testing"
)

// ---------------------------------------------------------------------------
// EvalResult helpers
// ---------------------------------------------------------------------------

func TestEvalResultIsValidTrue(t *testing.T) {
	r := EvalResult{Sharpe: 1.5, NFills: 20}
	if !r.IsValid() {
		t.Error("expected IsValid=true")
	}
}

func TestEvalResultIsValidFalseNoFills(t *testing.T) {
	r := EvalResult{Sharpe: 1.5, NFills: 0}
	if r.IsValid() {
		t.Error("expected IsValid=false for zero fills")
	}
}

func TestEvalResultIsValidFalseError(t *testing.T) {
	r := EvalResult{Sharpe: 1.5, NFills: 50, Error: "timeout"}
	if r.IsValid() {
		t.Error("expected IsValid=false when Error is set")
	}
}

// ---------------------------------------------------------------------------
// AdjustedSharpe
// ---------------------------------------------------------------------------

func TestAdjustedSharpeFullDeflation(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	// 25 trades out of 100 target -> factor = sqrt(0.25) = 0.5
	adj := agg.AdjustedSharpe(2.0, 25)
	want := 2.0 * 0.5
	if math.Abs(adj-want) > 1e-9 {
		t.Errorf("expected %.4f, got %.4f", want, adj)
	}
}

func TestAdjustedSharpeNoDeflation(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	// 200 trades -> factor capped at 1.0
	adj := agg.AdjustedSharpe(1.8, 200)
	if math.Abs(adj-1.8) > 1e-9 {
		t.Errorf("expected 1.8 (no deflation), got %f", adj)
	}
}

func TestAdjustedSharpeZeroTrades(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	adj := agg.AdjustedSharpe(3.0, 0)
	if adj != 0 {
		t.Errorf("expected 0 for zero trades, got %f", adj)
	}
}

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------

func TestAggregateEmpty(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	if s := agg.Aggregate(nil); s != 0 {
		t.Errorf("expected 0 for empty input, got %f", s)
	}
}

func TestAggregateSingleValidResult(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	results := []EvalResult{
		{Sharpe: 2.0, MaxDrawdown: 0.05, NFills: 100},
	}
	score := agg.Aggregate(results)
	if score <= 0 {
		t.Errorf("expected positive score, got %f", score)
	}
}

func TestAggregateDrawdownPenalty(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	noPenalty := []EvalResult{
		{Sharpe: 1.0, MaxDrawdown: 0.10, NFills: 100},
		{Sharpe: 1.0, MaxDrawdown: 0.10, NFills: 100},
		{Sharpe: 1.0, MaxDrawdown: 0.10, NFills: 100},
	}
	withPenalty := []EvalResult{
		{Sharpe: 1.0, MaxDrawdown: 0.25, NFills: 100},
		{Sharpe: 1.0, MaxDrawdown: 0.25, NFills: 100},
		{Sharpe: 1.0, MaxDrawdown: 0.25, NFills: 100},
	}
	scoreNP := agg.Aggregate(noPenalty)
	scoreWP := agg.Aggregate(withPenalty)
	if scoreWP >= scoreNP {
		t.Errorf("high drawdown score (%f) should be < low drawdown score (%f)", scoreWP, scoreNP)
	}
}

func TestAggregateThreePeriodsWeighted(t *testing.T) {
	// With equal Sharpe and 100 trades across all periods, no deflation and
	// no drawdown penalty, the weighted score should equal the raw Sharpe.
	agg := NewFitnessAggregator(AggregatorConfig{})
	results := []EvalResult{
		{Sharpe: 2.0, MaxDrawdown: 0.05, NFills: 100},
		{Sharpe: 2.0, MaxDrawdown: 0.05, NFills: 100},
		{Sharpe: 2.0, MaxDrawdown: 0.05, NFills: 100},
	}
	score := agg.Aggregate(results)
	if math.Abs(score-2.0) > 1e-6 {
		t.Errorf("expected score~2.0, got %f", score)
	}
}

func TestAggregateInvalidResultsSkipped(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	results := []EvalResult{
		{Sharpe: 2.0, MaxDrawdown: 0.05, NFills: 100},
		{Error: "failed"},
		{Error: "failed"},
	}
	score := agg.Aggregate(results)
	if score <= 0 {
		t.Errorf("expected positive score when one valid period, got %f", score)
	}
}

func TestAggregateCustomWeights(t *testing.T) {
	// With weight 1.0 on 3m and 0 on 6m/12m, only the first period matters.
	agg := NewFitnessAggregator(AggregatorConfig{
		Weight3M: 1.0, Weight6M: 0.001, Weight12M: 0.001,
	})
	results := []EvalResult{
		{Sharpe: 3.0, MaxDrawdown: 0.05, NFills: 100},
		{Sharpe: 0.1, MaxDrawdown: 0.05, NFills: 100},
		{Sharpe: 0.1, MaxDrawdown: 0.05, NFills: 100},
	}
	score := agg.Aggregate(results)
	// Score should be close to 3.0 since the first period dominates.
	if score < 2.5 {
		t.Errorf("expected score close to 3.0, got %f", score)
	}
}

// ---------------------------------------------------------------------------
// ParetoScore
// ---------------------------------------------------------------------------

func TestParetoScoreEmpty(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	if s := agg.ParetoScore(nil); s != 0 {
		t.Errorf("expected 0 for empty, got %f", s)
	}
}

func TestParetoScoreSingleResult(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	results := []EvalResult{{Sharpe: 1.5, NFills: 50, Calmar: 0.8, MaxDrawdown: 0.1}}
	s := agg.ParetoScore(results)
	if s != 1.0 {
		t.Errorf("expected 1.0 for single result, got %f", s)
	}
}

func TestParetoScoreNonZeroForValidResults(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	results := []EvalResult{
		{Sharpe: 0.5, Calmar: 0.3, MaxDrawdown: 0.25, NFills: 50},
		{Sharpe: 2.0, Calmar: 1.5, MaxDrawdown: 0.10, NFills: 100},
	}
	score := agg.ParetoScore(results)
	if score <= 0 {
		t.Errorf("expected positive Pareto score, got %f", score)
	}
}

func TestParetoScoreAllInvalid(t *testing.T) {
	agg := NewFitnessAggregator(AggregatorConfig{})
	results := []EvalResult{
		{Error: "fail1"},
		{Error: "fail2"},
	}
	if s := agg.ParetoScore(results); s != 0 {
		t.Errorf("expected 0 for all-invalid results, got %f", s)
	}
}

// ---------------------------------------------------------------------------
// WriteTempGenomeJSON
// ---------------------------------------------------------------------------

func TestWriteTempGenomeJSONRoundTrip(t *testing.T) {
	genome := map[string]float64{"fast": 5, "slow": 20, "threshold": 0.002}
	path, err := writeTempGenomeJSON(genome)
	if err != nil {
		t.Fatalf("writeTempGenomeJSON: %v", err)
	}
	defer os.Remove(path)
	if path == "" {
		t.Error("expected non-empty path")
	}
	// File should be readable.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read temp file: %v", err)
	}
	if len(data) == 0 {
		t.Error("expected non-empty JSON file")
	}
}

// ---------------------------------------------------------------------------
// BatchEvaluator construction
// ---------------------------------------------------------------------------

func TestNewBatchEvaluatorCapsWorkers(t *testing.T) {
	ev := NewGenomeEvaluator(EvalConfig{})
	b := NewBatchEvaluator(100, ev)
	if b.workers != 32 {
		t.Errorf("expected workers capped at 32, got %d", b.workers)
	}
}

func TestNewBatchEvaluatorMinWorkers(t *testing.T) {
	ev := NewGenomeEvaluator(EvalConfig{})
	b := NewBatchEvaluator(0, ev)
	if b.workers != 1 {
		t.Errorf("expected workers=1 minimum, got %d", b.workers)
	}
}

// ---------------------------------------------------------------------------
// SortBySharpeDESC
// ---------------------------------------------------------------------------

func TestSortBySharpeDESC(t *testing.T) {
	results := []EvalResult{
		{Sharpe: 0.5},
		{Sharpe: 2.1},
		{Sharpe: 1.3},
	}
	SortBySharpeDESC(results)
	if results[0].Sharpe != 2.1 {
		t.Errorf("expected 2.1 first, got %f", results[0].Sharpe)
	}
	if results[2].Sharpe != 0.5 {
		t.Errorf("expected 0.5 last, got %f", results[2].Sharpe)
	}
}
