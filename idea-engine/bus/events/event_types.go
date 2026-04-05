// Package events defines the shared event types used by the Idea Automation Engine
// event bus. Every message travelling the bus is represented as an Event with a
// typed JSON payload.
package events

import (
	"encoding/json"
	"time"
)

// Event is the canonical envelope for every message on the bus.
type Event struct {
	// EventID is a globally unique identifier assigned by the bus on publish.
	EventID string `json:"event_id"`
	// Topic is one of the Topic* constants defined in the bus topics file.
	Topic string `json:"topic"`
	// Payload carries the raw JSON bytes of the topic-specific payload struct.
	Payload json.RawMessage `json:"payload"`
	// ProducedAt is the UTC timestamp when the event was published.
	ProducedAt time.Time `json:"produced_at"`
	// ProducerName is the logical name of the service that produced the event.
	ProducerName string `json:"producer_name"`
}

// ---- Typed payload structs for each topic ----

// PatternsDiscoveredPayload is the payload for TopicPatternsDiscovered.
type PatternsDiscoveredPayload struct {
	// RunID identifies the pattern-mining run that produced these patterns.
	RunID string `json:"run_id"`
	// Patterns is the list of newly discovered patterns.
	Patterns []PatternSummary `json:"patterns"`
	// PatternCount is the total number of patterns found in this run.
	PatternCount int `json:"pattern_count"`
	// DiscoveredAt is when the mining run completed.
	DiscoveredAt time.Time `json:"discovered_at"`
}

// PatternSummary carries the lightweight summary embedded in a bus event.
type PatternSummary struct {
	PatternID   string  `json:"pattern_id"`
	PatternType string  `json:"pattern_type"`
	Confidence  float64 `json:"confidence"`
	Description string  `json:"description"`
}

// HypothesesCreatedPayload is the payload for TopicHypothesesCreated.
type HypothesesCreatedPayload struct {
	// SourceRunID links back to the pattern mining run that spawned these.
	SourceRunID string `json:"source_run_id"`
	// Hypotheses is the list of freshly generated hypotheses.
	Hypotheses []HypothesisSummary `json:"hypotheses"`
	// GeneratedAt is the timestamp of hypothesis generation.
	GeneratedAt time.Time `json:"generated_at"`
}

// HypothesisSummary carries the lightweight summary embedded in a bus event.
type HypothesisSummary struct {
	HypothesisID  string  `json:"hypothesis_id"`
	Statement     string  `json:"statement"`
	PriorityRank  int     `json:"priority_rank"`
	ExpectedAlpha float64 `json:"expected_alpha"`
}

// GenomeEvaluatedPayload is the payload for TopicGenomeEvaluated.
type GenomeEvaluatedPayload struct {
	// GenomeID is the unique identifier of the evaluated genome.
	GenomeID string `json:"genome_id"`
	// Generation is the evolutionary generation this genome belongs to.
	Generation int `json:"generation"`
	// FitnessScore is the composite fitness value after evaluation.
	FitnessScore float64 `json:"fitness_score"`
	// SharpeRatio is the in-sample Sharpe from the backtest.
	SharpeRatio float64 `json:"sharpe_ratio"`
	// MaxDrawdown is the worst peak-to-trough drawdown from the backtest.
	MaxDrawdown float64 `json:"max_drawdown"`
	// EvaluatedAt is when the evaluation completed.
	EvaluatedAt time.Time `json:"evaluated_at"`
}

// ShadowCycleCompletePayload is the payload for TopicShadowCycleComplete.
type ShadowCycleCompletePayload struct {
	// CycleID uniquely identifies this shadow-runner cycle.
	CycleID string `json:"cycle_id"`
	// VariantsRun is the count of variants scored in this cycle.
	VariantsRun int `json:"variants_run"`
	// LeaderVariantID is the ID of the top-ranked variant.
	LeaderVariantID string `json:"leader_variant_id"`
	// LeaderScore is the score of the leading variant.
	LeaderScore float64 `json:"leader_score"`
	// CompletedAt is when the cycle finished.
	CompletedAt time.Time `json:"completed_at"`
}

// CounterfactualDonePayload is the payload for TopicCounterfactualDone.
type CounterfactualDonePayload struct {
	// ExperimentID identifies the counterfactual experiment.
	ExperimentID string `json:"experiment_id"`
	// HypothesisID links back to the hypothesis under test.
	HypothesisID string `json:"hypothesis_id"`
	// CounterfactualType describes the type of counterfactual (e.g. "regime_swap").
	CounterfactualType string `json:"counterfactual_type"`
	// ResultSummary is a human-readable one-liner of the finding.
	ResultSummary string `json:"result_summary"`
	// PValueAccepted indicates whether the hypothesis survived the counterfactual.
	PValueAccepted bool `json:"p_value_accepted"`
	// CompletedAt is when the counterfactual finished.
	CompletedAt time.Time `json:"completed_at"`
}

// AcademicIdeaExtractedPayload is the payload for TopicAcademicIdeaExtracted.
type AcademicIdeaExtractedPayload struct {
	// SourcePaperID is the identifier (e.g. arXiv ID) of the source paper.
	SourcePaperID string `json:"source_paper_id"`
	// Title is the title of the academic paper.
	Title string `json:"title"`
	// ExtractedIdeas is a list of actionable ideas extracted from the paper.
	ExtractedIdeas []AcademicIdeaSummary `json:"extracted_ideas"`
	// ExtractedAt is when extraction completed.
	ExtractedAt time.Time `json:"extracted_at"`
}

// AcademicIdeaSummary is a lightweight idea extracted from an academic paper.
type AcademicIdeaSummary struct {
	IdeaID      string  `json:"idea_id"`
	Description string  `json:"description"`
	Relevance   float64 `json:"relevance"`
	Keywords    []string `json:"keywords"`
}

// SerendipitySurprisePayload is the payload for TopicSerendipitySurprise.
type SerendipitySurprisePayload struct {
	// SurpriseID uniquely identifies this serendipitous discovery.
	SurpriseID string `json:"surprise_id"`
	// Description is a human-readable description of the surprise.
	Description string `json:"description"`
	// SurpriseScore measures how unexpected the finding is (0–1).
	SurpriseScore float64 `json:"surprise_score"`
	// RelatedPatternIDs lists any patterns this surprise is linked to.
	RelatedPatternIDs []string `json:"related_pattern_ids"`
	// DiscoveredAt is when the surprise was identified.
	DiscoveredAt time.Time `json:"discovered_at"`
}

// CausalDagUpdatedPayload is the payload for TopicCausalDagUpdated.
type CausalDagUpdatedPayload struct {
	// DagVersion is the monotonically incrementing version of the causal DAG.
	DagVersion int `json:"dag_version"`
	// NodesAdded is the count of new nodes added to the DAG.
	NodesAdded int `json:"nodes_added"`
	// EdgesAdded is the count of new causal edges added.
	EdgesAdded int `json:"edges_added"`
	// EdgesRemoved is the count of edges removed due to refutation.
	EdgesRemoved int `json:"edges_removed"`
	// UpdatedAt is when the DAG update completed.
	UpdatedAt time.Time `json:"updated_at"`
}

// ExperimentCompletedPayload is the payload for TopicExperimentCompleted.
type ExperimentCompletedPayload struct {
	// ExperimentID identifies the completed experiment.
	ExperimentID string `json:"experiment_id"`
	// ExperimentType is e.g. "genome", "counterfactual", "shadow", "causal".
	ExperimentType string `json:"experiment_type"`
	// HypothesisID links back to the originating hypothesis, if any.
	HypothesisID string `json:"hypothesis_id,omitempty"`
	// Status is "done" or "failed".
	Status string `json:"status"`
	// Result carries the raw result JSON for downstream consumers.
	Result json.RawMessage `json:"result,omitempty"`
	// CompletedAt is when the experiment finished.
	CompletedAt time.Time `json:"completed_at"`
}
