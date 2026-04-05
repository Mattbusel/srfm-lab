// Package bus defines the event bus for the Idea Automation Engine.
// This file contains the canonical topic name constants shared across all
// producers and consumers in the system.
package bus

// Topic name constants.  All inter-service communication travels through
// the bus using these string identifiers.  Keeping them as typed constants
// prevents typos and makes it easy to grep the codebase for all usages of
// a given topic.
const (
	// TopicPatternsDiscovered is published by the pattern-mining module
	// whenever a batch of new market patterns has been identified.
	TopicPatternsDiscovered = "patterns.discovered"

	// TopicHypothesesCreated is published by the hypothesis generator after
	// it has synthesised a new cohort of hypotheses from recent patterns.
	TopicHypothesesCreated = "hypotheses.created"

	// TopicGenomeEvaluated is published by the genome engine after it has
	// run a backtest evaluation on a single genome individual.
	TopicGenomeEvaluated = "genome.evaluated"

	// TopicShadowCycleComplete is published by the shadow runner after every
	// full live-paper-trading scoring cycle across all active variants.
	TopicShadowCycleComplete = "shadow.cycle.complete"

	// TopicCounterfactualDone is published by the counterfactual module when
	// a counterfactual experiment has finished executing.
	TopicCounterfactualDone = "counterfactual.done"

	// TopicAcademicIdeaExtracted is published by the academic miner when it
	// has extracted actionable ideas from a new paper ingestion batch.
	TopicAcademicIdeaExtracted = "academic.idea.extracted"

	// TopicSerendipitySurprise is published by the serendipity engine when it
	// detects an unexpected but potentially valuable anomaly.
	TopicSerendipitySurprise = "serendipity.surprise"

	// TopicCausalDagUpdated is published by the causal inference module after
	// it has revised the causal directed acyclic graph.
	TopicCausalDagUpdated = "causal.dag.updated"

	// TopicExperimentCompleted is published by the scheduler after any
	// experiment (of any type) has reached a terminal state (done or failed).
	TopicExperimentCompleted = "experiment.completed"
)

// AllTopics returns a slice of every registered topic constant.  It is used
// by the persistence layer to validate incoming event topics and by the HTTP
// adapter to reject unknown topics quickly.
func AllTopics() []string {
	return []string{
		TopicPatternsDiscovered,
		TopicHypothesesCreated,
		TopicGenomeEvaluated,
		TopicShadowCycleComplete,
		TopicCounterfactualDone,
		TopicAcademicIdeaExtracted,
		TopicSerendipitySurprise,
		TopicCausalDagUpdated,
		TopicExperimentCompleted,
	}
}

// IsValidTopic reports whether topic is one of the registered constants.
func IsValidTopic(topic string) bool {
	for _, t := range AllTopics() {
		if t == topic {
			return true
		}
	}
	return false
}
