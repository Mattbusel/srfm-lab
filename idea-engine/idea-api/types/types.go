// Package types defines the Go structs that mirror the idea_engine.db schema.
// These are used as return types from the query layer and as JSON response
// bodies from the HTTP handlers.
package types

import (
	"encoding/json"
	"time"
)

// Hypothesis represents a row from the hypotheses table.
type Hypothesis struct {
	// HypothesisID is the unique identifier (UUID).
	HypothesisID string `json:"hypothesis_id" db:"hypothesis_id"`
	// Statement is the natural-language description of the hypothesis.
	Statement string `json:"statement" db:"statement"`
	// Status is one of: pending, queued, running, done, rejected.
	Status string `json:"status" db:"status"`
	// PriorityRank is used by the scheduler to order experiments; lower = higher priority.
	PriorityRank int `json:"priority_rank" db:"priority_rank"`
	// ExpectedAlpha is the model-predicted annualised alpha.
	ExpectedAlpha float64 `json:"expected_alpha" db:"expected_alpha"`
	// ConfidenceScore is the prior confidence in the hypothesis (0–1).
	ConfidenceScore float64 `json:"confidence_score" db:"confidence_score"`
	// SourcePatternIDs is the JSON-encoded list of pattern IDs that spawned this.
	SourcePatternIDs json.RawMessage `json:"source_pattern_ids,omitempty" db:"source_pattern_ids"`
	// CreatedAt is when the hypothesis was generated.
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	// UpdatedAt is the last time the row was modified.
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// Genome represents a row from the genomes table.
type Genome struct {
	// GenomeID is the unique identifier.
	GenomeID string `json:"genome_id" db:"genome_id"`
	// Generation is the evolutionary generation this individual belongs to.
	Generation int `json:"generation" db:"generation"`
	// Chromosome is the JSON-encoded parameter vector.
	Chromosome json.RawMessage `json:"chromosome" db:"chromosome"`
	// FitnessScore is the composite fitness after evaluation; null if not yet evaluated.
	FitnessScore *float64 `json:"fitness_score,omitempty" db:"fitness_score"`
	// SharpeRatio from the evaluation backtest.
	SharpeRatio *float64 `json:"sharpe_ratio,omitempty" db:"sharpe_ratio"`
	// MaxDrawdown from the evaluation backtest.
	MaxDrawdown *float64 `json:"max_drawdown,omitempty" db:"max_drawdown"`
	// IsElite indicates this genome was preserved in the elite archive.
	IsElite bool `json:"is_elite" db:"is_elite"`
	// ParentIDs is the JSON-encoded list of parent genome IDs (empty for generation 0).
	ParentIDs json.RawMessage `json:"parent_ids,omitempty" db:"parent_ids"`
	// CreatedAt is when the genome was initialised or bred.
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	// EvaluatedAt is when the fitness evaluation completed.
	EvaluatedAt *time.Time `json:"evaluated_at,omitempty" db:"evaluated_at"`
}

// FitnessPoint is a time-series data point for the fitness history chart.
type FitnessPoint struct {
	// Generation is the evolutionary generation.
	Generation int `json:"generation" db:"generation"`
	// BestFitness is the highest fitness score seen in this generation.
	BestFitness float64 `json:"best_fitness" db:"best_fitness"`
	// MeanFitness is the population mean fitness for this generation.
	MeanFitness float64 `json:"mean_fitness" db:"mean_fitness"`
	// RecordedAt is when this generation's evaluation completed.
	RecordedAt time.Time `json:"recorded_at" db:"recorded_at"`
}

// ShadowVariant represents a row from the shadow_variants table.
type ShadowVariant struct {
	// VariantID is the unique identifier.
	VariantID string `json:"variant_id" db:"variant_id"`
	// Name is the human-readable label for this variant.
	Name string `json:"name" db:"name"`
	// HypothesisID links this variant to its originating hypothesis.
	HypothesisID string `json:"hypothesis_id" db:"hypothesis_id"`
	// Strategy is the JSON-encoded strategy parameters.
	Strategy json.RawMessage `json:"strategy" db:"strategy"`
	// LatestScore is the most recent shadow score.
	LatestScore *float64 `json:"latest_score,omitempty" db:"latest_score"`
	// CumulativePnL is the total paper-trading P&L accumulated.
	CumulativePnL float64 `json:"cumulative_pnl" db:"cumulative_pnl"`
	// IsPromoted indicates this variant has been promoted to live trading.
	IsPromoted bool `json:"is_promoted" db:"is_promoted"`
	// CreatedAt is when the variant was registered.
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	// UpdatedAt is the last time the row was modified.
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// ShadowScore represents a historical score record for a shadow variant.
type ShadowScore struct {
	// ScoreID is the unique identifier.
	ScoreID string `json:"score_id" db:"score_id"`
	// VariantID links to the variant.
	VariantID string `json:"variant_id" db:"variant_id"`
	// CycleID identifies the shadow cycle that produced this score.
	CycleID string `json:"cycle_id" db:"cycle_id"`
	// Score is the composite score for this cycle.
	Score float64 `json:"score" db:"score"`
	// Pnl is the P&L for this cycle.
	Pnl float64 `json:"pnl" db:"pnl"`
	// SharpeContribution is the Sharpe contribution from this cycle.
	SharpeContribution float64 `json:"sharpe_contribution" db:"sharpe_contribution"`
	// ScoredAt is when the score was computed.
	ScoredAt time.Time `json:"scored_at" db:"scored_at"`
}

// Pattern represents a row from the patterns table.
type Pattern struct {
	// PatternID is the unique identifier.
	PatternID string `json:"pattern_id" db:"pattern_id"`
	// PatternType categorises the pattern (e.g. "momentum", "mean_reversion").
	PatternType string `json:"pattern_type" db:"pattern_type"`
	// Description is a natural-language description of the pattern.
	Description string `json:"description" db:"description"`
	// Confidence is the statistical confidence in the pattern (0–1).
	Confidence float64 `json:"confidence" db:"confidence"`
	// Frequency is how often this pattern has been observed.
	Frequency int `json:"frequency" db:"frequency"`
	// Features is the JSON-encoded feature vector associated with the pattern.
	Features json.RawMessage `json:"features,omitempty" db:"features"`
	// DiscoveredAt is when the pattern was first identified.
	DiscoveredAt time.Time `json:"discovered_at" db:"discovered_at"`
	// RunID links to the mining run that produced this pattern.
	RunID string `json:"run_id" db:"run_id"`
}

// Experiment represents a row from the experiments table.
type Experiment struct {
	// ExperimentID is the unique identifier.
	ExperimentID string `json:"experiment_id" db:"experiment_id"`
	// HypothesisID links to the originating hypothesis.
	HypothesisID string `json:"hypothesis_id" db:"hypothesis_id"`
	// ExperimentType is one of: genome, counterfactual, shadow, causal, academic.
	ExperimentType string `json:"experiment_type" db:"experiment_type"`
	// Status is one of: queued, running, done, failed.
	Status string `json:"status" db:"status"`
	// Priority drives scheduling order; lower number = higher priority.
	Priority int `json:"priority" db:"priority"`
	// Config is the JSON-encoded configuration passed to the experiment runner.
	Config json.RawMessage `json:"config,omitempty" db:"config"`
	// Result is the JSON-encoded result written by the experiment runner.
	Result json.RawMessage `json:"result,omitempty" db:"result"`
	// ErrorMsg contains the error message if the experiment failed.
	ErrorMsg string `json:"error_msg,omitempty" db:"error_msg"`
	// RetryCount is how many times this experiment has been retried.
	RetryCount int `json:"retry_count" db:"retry_count"`
	// CreatedAt is when the experiment was created.
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	// StartedAt is when the experiment began running.
	StartedAt *time.Time `json:"started_at,omitempty" db:"started_at"`
	// CompletedAt is when the experiment reached a terminal state.
	CompletedAt *time.Time `json:"completed_at,omitempty" db:"completed_at"`
}

// Narrative represents a row from the narratives table.
type Narrative struct {
	// NarrativeID is the unique identifier.
	NarrativeID string `json:"narrative_id" db:"narrative_id"`
	// Title is the short title of the narrative report.
	Title string `json:"title" db:"title"`
	// Body is the full markdown/text body of the narrative.
	Body string `json:"body" db:"body"`
	// HypothesisIDs is the JSON-encoded list of hypothesis IDs covered.
	HypothesisIDs json.RawMessage `json:"hypothesis_ids,omitempty" db:"hypothesis_ids"`
	// ExperimentIDs is the JSON-encoded list of experiment IDs referenced.
	ExperimentIDs json.RawMessage `json:"experiment_ids,omitempty" db:"experiment_ids"`
	// GeneratedAt is when the narrative was produced by the LLM.
	GeneratedAt time.Time `json:"generated_at" db:"generated_at"`
}

// ---- Request / response helpers ----

// PaginatedResponse wraps a list result with pagination metadata.
type PaginatedResponse struct {
	Data   interface{} `json:"data"`
	Total  int         `json:"total"`
	Limit  int         `json:"limit"`
	Offset int         `json:"offset"`
}

// ErrorResponse is the canonical JSON error body.
type ErrorResponse struct {
	Error string `json:"error"`
	Code  int    `json:"code,omitempty"`
}
