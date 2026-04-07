package tests

import (
	"context"
	"math"
	"testing"

	"srfm-lab/idea-engine/pkg/evolution"
	"srfm-lab/idea-engine/pkg/parameter_evolution"
)

// noopEval is a no-op EvaluatorFunc used in NSGA-II tests that only exercise
// sorting/ranking logic (no actual fitness evaluation needed).
var noopEval = parameter_evolution.EvaluatorFunc(func(_ context.Context, _ *parameter_evolution.Individual) error {
	return nil
})

// ---------------------------------------------------------------------------
// TournamentSelection tests
// ---------------------------------------------------------------------------

func makeIndividual(id string, score float64) evolution.Individual {
	return evolution.Individual{
		ID:        id,
		Evaluated: true,
		Fitness:   evolution.FitnessResult{WeightedScore: score, Sharpe: score},
	}
}

func TestTournamentSelection_ReturnsBestOfK(t *testing.T) {
	// With k == population size, the global best must always be returned.
	pop := []evolution.Individual{
		makeIndividual("a", 1.0),
		makeIndividual("b", 5.0),
		makeIndividual("c", 2.0),
		makeIndividual("d", 3.0),
	}
	for trial := 0; trial < 20; trial++ {
		winner := evolution.TournamentSelection(pop, len(pop))
		if winner.ID != "b" {
			t.Errorf("trial=%d: expected winner 'b' (score=5), got '%s' (score=%.2f)",
				trial, winner.ID, winner.Fitness.WeightedScore)
		}
	}
}

func TestTournamentSelection_KOne_RandomUniform(t *testing.T) {
	// With k=1, the selection is random -- all IDs should appear over enough trials.
	pop := make([]evolution.Individual, 5)
	for i := range pop {
		pop[i] = makeIndividual(string(rune('a'+i)), float64(i+1))
	}
	seen := make(map[string]int, 5)
	for trial := 0; trial < 200; trial++ {
		w := evolution.TournamentSelection(pop, 1)
		seen[w.ID]++
	}
	for _, ind := range pop {
		if seen[ind.ID] == 0 {
			t.Errorf("individual '%s' never selected with k=1 over 200 trials", ind.ID)
		}
	}
}

func TestTournamentSelection_HigherScoreWinsMoreOften(t *testing.T) {
	// With k=2, the highest-fitness individual should win more than 50% of the time.
	pop := []evolution.Individual{
		makeIndividual("weak", 1.0),
		makeIndividual("strong", 10.0),
	}
	strongWins := 0
	trials := 200
	for i := 0; i < trials; i++ {
		w := evolution.TournamentSelection(pop, 2)
		if w.ID == "strong" {
			strongWins++
		}
	}
	winRate := float64(strongWins) / float64(trials)
	if winRate < 0.60 {
		t.Errorf("strong individual win rate %.2f < 0.60 over %d trials", winRate, trials)
	}
}

// ---------------------------------------------------------------------------
// NSGA-II NonDominatedSort tests
// ---------------------------------------------------------------------------

// makeObjectives is a convenience for building an Individual with given
// objective values. IsMaximize=true for all objectives.
func makeNSGAInd(id string, objs ...float64) parameter_evolution.Individual {
	objectives := make([]parameter_evolution.Objective, len(objs))
	for i, v := range objs {
		objectives[i] = parameter_evolution.Objective{
			Name:       string(rune('A' + i)),
			Value:      v,
			IsMaximize: true,
		}
	}
	return parameter_evolution.Individual{
		ID:         id,
		Objectives: objectives,
	}
}

func TestNonDominatedSort_ParetoFrontCorrect(t *testing.T) {
	// Hand-crafted 2-objective case (maximise both).
	//   A=(3,3) -- Pareto optimal
	//   B=(2,3) -- dominated by A
	//   C=(3,2) -- dominated by A
	//   D=(1,1) -- dominated by A, B, C
	pop := []parameter_evolution.Individual{
		makeNSGAInd("A", 3, 3),
		makeNSGAInd("B", 2, 3),
		makeNSGAInd("C", 3, 2),
		makeNSGAInd("D", 1, 1),
	}
	opt := parameter_evolution.NewMultiObjectiveOptimizer(parameter_evolution.DefaultMOConfig(), noopEval)
	// We only call NonDominatedSort, which doesn't need an evaluator.
	fronts := opt.NonDominatedSort(pop)

	if len(fronts) < 1 {
		t.Fatal("expected at least one front")
	}

	// Front 0 should contain exactly "A".
	front0IDs := make(map[string]bool)
	for _, idx := range fronts[0] {
		front0IDs[pop[idx].ID] = true
	}
	if !front0IDs["A"] {
		t.Errorf("expected 'A' in front 0, got IDs: %v", front0IDs)
	}
	if len(fronts[0]) != 1 {
		t.Errorf("front 0 should contain exactly 1 individual, got %d: %v", len(fronts[0]), front0IDs)
	}
}

func TestNonDominatedSort_MultipleParetoOptimal(t *testing.T) {
	// A=(5,1), B=(4,2), C=(3,3), D=(2,4), E=(1,5) -- all Pareto optimal
	// F=(2,2) dominated by B, C, D
	pop := []parameter_evolution.Individual{
		makeNSGAInd("A", 5, 1),
		makeNSGAInd("B", 4, 2),
		makeNSGAInd("C", 3, 3),
		makeNSGAInd("D", 2, 4),
		makeNSGAInd("E", 1, 5),
		makeNSGAInd("F", 2, 2),
	}
	opt := parameter_evolution.NewMultiObjectiveOptimizer(
		parameter_evolution.DefaultMOConfig(),
		nil,
	)
	fronts := opt.NonDominatedSort(pop)

	if len(fronts) < 2 {
		t.Fatalf("expected at least 2 fronts, got %d", len(fronts))
	}
	if len(fronts[0]) != 5 {
		t.Errorf("expected 5 individuals in front 0 (the Pareto front), got %d", len(fronts[0]))
	}
	// F must be in a later front.
	fInFront0 := false
	for _, idx := range fronts[0] {
		if pop[idx].ID == "F" {
			fInFront0 = true
		}
	}
	if fInFront0 {
		t.Error("individual F should NOT be in front 0")
	}
}

func TestNonDominatedSort_AllEqualIsAllPareto(t *testing.T) {
	// All individuals identical -- none dominates any other, all in front 0.
	pop := []parameter_evolution.Individual{
		makeNSGAInd("a", 2, 2),
		makeNSGAInd("b", 2, 2),
		makeNSGAInd("c", 2, 2),
	}
	opt := parameter_evolution.NewMultiObjectiveOptimizer(parameter_evolution.DefaultMOConfig(), noopEval)
	fronts := opt.NonDominatedSort(pop)

	if len(fronts[0]) != 3 {
		t.Errorf("expected all 3 individuals in front 0, got %d", len(fronts[0]))
	}
}

func TestNonDominatedSort_StrictChain(t *testing.T) {
	// A strictly dominates B which strictly dominates C -- three separate fronts.
	pop := []parameter_evolution.Individual{
		makeNSGAInd("A", 3, 3),
		makeNSGAInd("B", 2, 2),
		makeNSGAInd("C", 1, 1),
	}
	opt := parameter_evolution.NewMultiObjectiveOptimizer(parameter_evolution.DefaultMOConfig(), noopEval)
	fronts := opt.NonDominatedSort(pop)

	if len(fronts) < 3 {
		t.Fatalf("expected 3 fronts for strict chain, got %d", len(fronts))
	}
	for rank, front := range fronts {
		if len(front) != 1 {
			t.Errorf("front %d should have exactly 1 individual, got %d", rank, len(front))
		}
	}
}

// ---------------------------------------------------------------------------
// CrowdingDistance tests
// ---------------------------------------------------------------------------

func TestCrowdingDistance_BoundaryInfinite(t *testing.T) {
	// The best and worst on each objective always receive Inf distance.
	front := []parameter_evolution.Individual{
		makeNSGAInd("lo", 1, 5),
		makeNSGAInd("mid", 3, 3),
		makeNSGAInd("hi", 5, 1),
	}
	distances := parameter_evolution.CrowdingDistance(front, 2)
	if !math.IsInf(distances[0], 1) && !math.IsInf(distances[2], 1) {
		t.Error("expected boundary individuals to have Inf crowding distance")
	}
}

func TestCrowdingDistance_MiddleFinite(t *testing.T) {
	front := []parameter_evolution.Individual{
		makeNSGAInd("a", 1, 5),
		makeNSGAInd("b", 3, 3),
		makeNSGAInd("c", 5, 1),
	}
	distances := parameter_evolution.CrowdingDistance(front, 2)
	middleIdx := -1
	for i, d := range distances {
		if !math.IsInf(d, 1) {
			middleIdx = i
		}
	}
	if middleIdx < 0 {
		t.Fatal("expected at least one finite crowding distance for the middle individual")
	}
	if distances[middleIdx] <= 0 {
		t.Errorf("middle individual crowding distance should be > 0, got %.6f", distances[middleIdx])
	}
}

func TestCrowdingDistance_TwoOrFewerInfinity(t *testing.T) {
	// Fronts of size <= 2 should give all Inf.
	for _, size := range []int{1, 2} {
		front := make([]parameter_evolution.Individual, size)
		for i := range front {
			front[i] = makeNSGAInd(string(rune('a'+i)), float64(i), float64(i))
		}
		distances := parameter_evolution.CrowdingDistance(front, 2)
		for i, d := range distances {
			if !math.IsInf(d, 1) {
				t.Errorf("size=%d: index %d expected Inf, got %.6f", size, i, d)
			}
		}
	}
}

func TestCrowdingDistance_DenseClusterLowDistance(t *testing.T) {
	// When interior individuals are closely clustered, distances should be small.
	front := []parameter_evolution.Individual{
		makeNSGAInd("lo", 0, 10),
		makeNSGAInd("m1", 5.0, 5.0),
		makeNSGAInd("m2", 5.1, 4.9),
		makeNSGAInd("hi", 10, 0),
	}
	distances := parameter_evolution.CrowdingDistance(front, 2)
	// m1 and m2 are very close to each other -- their distances should both be small.
	// Find indices of m1 and m2.
	var innerDist []float64
	for i, d := range distances {
		if !math.IsInf(d, 1) {
			_ = front[i]
			innerDist = append(innerDist, d)
		}
	}
	if len(innerDist) == 0 {
		t.Fatal("expected finite crowding distances for inner individuals")
	}
	for _, d := range innerDist {
		if d > 1.5 {
			t.Errorf("closely clustered inner individual has unexpectedly large crowding distance %.4f", d)
		}
	}
}
