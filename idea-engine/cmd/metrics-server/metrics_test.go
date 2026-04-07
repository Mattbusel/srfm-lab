package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// IAEMetrics unit tests
// ---------------------------------------------------------------------------

func TestRecordAndRetrieveGenome(t *testing.T) {
	m := NewIAEMetrics()
	gm := GenomeMetric{
		Generation:  1,
		BestFitness: 1.5,
		MeanFitness: 1.1,
		Diversity:   0.4,
		EvalCount:   10,
		Timestamp:   time.Now(),
	}
	m.RecordGenome(gm)

	latest, ok := m.LatestGenome()
	if !ok {
		t.Fatal("expected a genome to be recorded")
	}
	if latest.BestFitness != 1.5 {
		t.Errorf("expected BestFitness=1.5, got %f", latest.BestFitness)
	}
}

func TestLatestGenomeEmpty(t *testing.T) {
	m := NewIAEMetrics()
	_, ok := m.LatestGenome()
	if ok {
		t.Error("expected false for empty store")
	}
}

func TestFitnessImprovementPositive(t *testing.T) {
	m := NewIAEMetrics()
	for i := 0; i < 5; i++ {
		m.RecordGenome(GenomeMetric{Generation: i, BestFitness: float64(i) * 0.1})
	}
	imp := m.FitnessImprovement(5)
	if imp <= 0 {
		t.Errorf("expected positive improvement, got %f", imp)
	}
}

func TestFitnessImprovementTooFewPoints(t *testing.T) {
	m := NewIAEMetrics()
	m.RecordGenome(GenomeMetric{Generation: 1, BestFitness: 2.0})
	imp := m.FitnessImprovement(5)
	if imp != 0 {
		t.Errorf("expected 0 for single record, got %f", imp)
	}
}

func TestRecordParamUpdateDeltaAutoCalc(t *testing.T) {
	m := NewIAEMetrics()
	m.RecordParamUpdate(ParamMetric{
		ParamName: "fast_period",
		Value:     12.0,
		PrevValue: 10.0,
		Source:    "mutation",
	})
	params := m.RecentParams(1)
	if len(params) != 1 {
		t.Fatalf("expected 1 param, got %d", len(params))
	}
	// DeltaPct should be 20%.
	if params[0].DeltaPct < 19.9 || params[0].DeltaPct > 20.1 {
		t.Errorf("expected DeltaPct~20, got %f", params[0].DeltaPct)
	}
}

func TestRecordParamPrevValueZero(t *testing.T) {
	m := NewIAEMetrics()
	m.RecordParamUpdate(ParamMetric{
		ParamName: "threshold",
		Value:     5.0,
		PrevValue: 0.0,
		Source:    "manual",
	})
	params := m.RecentParams(1)
	// Should not divide by zero; DeltaPct stays at 0.
	if params[0].DeltaPct != 0 {
		t.Errorf("expected DeltaPct=0 when PrevValue=0, got %f", params[0].DeltaPct)
	}
}

func TestEvaluationsPerHourFewPoints(t *testing.T) {
	m := NewIAEMetrics()
	// Only one evaluation -- should return 0.
	m.RecordEvaluation(1.2, 0.05, 1000)
	rate := m.EvaluationsPerHour()
	if rate != 0 {
		t.Errorf("expected 0 with single eval, got %f", rate)
	}
}

func TestTotalEvalsIncrement(t *testing.T) {
	m := NewIAEMetrics()
	for i := 0; i < 7; i++ {
		m.RecordEvaluation(1.0, 0.05, 500)
	}
	if m.TotalEvals() != 7 {
		t.Errorf("expected 7 total evals, got %d", m.TotalEvals())
	}
}

func TestRingBufferOverflow(t *testing.T) {
	m := NewIAEMetrics()
	// Insert more than ringBufferSize entries.
	for i := 0; i < ringBufferSize+50; i++ {
		m.RecordGenome(GenomeMetric{Generation: i, BestFitness: float64(i)})
	}
	genomes := m.RecentGenomes(ringBufferSize)
	if len(genomes) != ringBufferSize {
		t.Errorf("expected %d genomes after overflow, got %d", ringBufferSize, len(genomes))
	}
	// The oldest entry should not be generation 0 after overflow.
	if genomes[0].Generation == 0 {
		t.Error("expected ring buffer to have evicted generation 0")
	}
}

func TestMeanSharpeEmpty(t *testing.T) {
	m := NewIAEMetrics()
	if m.MeanSharpe() != 0 {
		t.Error("expected MeanSharpe=0 for empty store")
	}
}

func TestMeanEvalTimeMsCalculation(t *testing.T) {
	m := NewIAEMetrics()
	m.RecordEvaluation(1.0, 0.1, 100)
	m.RecordEvaluation(1.2, 0.08, 200)
	m.RecordEvaluation(0.9, 0.12, 300)
	mean := m.MeanEvalTimeMs()
	if mean < 199.9 || mean > 200.1 {
		t.Errorf("expected mean eval time 200ms, got %.2f", mean)
	}
}

// ---------------------------------------------------------------------------
// Handler HTTP tests
// ---------------------------------------------------------------------------

func newTestHandler() *MetricsHandler {
	return NewMetricsHandler(NewIAEMetrics())
}

func TestHandleHealthOK(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rr := httptest.NewRecorder()
	h.HandleHealth(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	var resp healthResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("expected status=ok, got %s", resp.Status)
	}
}

func TestHandleHealthMethodNotAllowed(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodPost, "/health", nil)
	rr := httptest.NewRecorder()
	h.HandleHealth(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rr.Code)
	}
}

func TestHandleMetricsPrometheusFormat(t *testing.T) {
	h := newTestHandler()
	h.store.RecordGenome(GenomeMetric{Generation: 3, BestFitness: 1.8, MeanFitness: 1.3, Diversity: 0.5})
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	h.HandleMetrics(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	body := rr.Body.String()
	if len(body) == 0 {
		t.Error("expected non-empty Prometheus output")
	}
	// Check for a known metric name.
	if !bytes.Contains([]byte(body), []byte("iae_total_evaluations")) {
		t.Error("expected iae_total_evaluations in Prometheus output")
	}
}

func TestHandleIngestGenome(t *testing.T) {
	h := newTestHandler()
	gm := GenomeMetric{Generation: 10, BestFitness: 2.1}
	body, _ := json.Marshal(gm)
	req := httptest.NewRequest(http.MethodPost, "/ingest/genome", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	h.HandleIngestGenome(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	latest, ok := h.store.LatestGenome()
	if !ok || latest.Generation != 10 {
		t.Errorf("expected ingested genome generation=10, got %v", latest)
	}
}

func TestHandleEvolutionJSON(t *testing.T) {
	h := newTestHandler()
	h.store.RecordGenome(GenomeMetric{Generation: 5, BestFitness: 1.5})
	req := httptest.NewRequest(http.MethodGet, "/metrics/evolution", nil)
	rr := httptest.NewRecorder()
	h.HandleEvolution(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	var resp evolutionResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode evolution response: %v", err)
	}
	if resp.CurrentGeneration != 5 {
		t.Errorf("expected generation=5, got %d", resp.CurrentGeneration)
	}
}

func TestHandleParamsNParam(t *testing.T) {
	h := newTestHandler()
	for i := 0; i < 10; i++ {
		h.store.RecordParamUpdate(ParamMetric{ParamName: "p", Value: float64(i)})
	}
	req := httptest.NewRequest(http.MethodGet, "/metrics/params?n=5", nil)
	rr := httptest.NewRecorder()
	h.HandleParams(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	var resp paramsResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if resp.Count != 5 {
		t.Errorf("expected count=5, got %d", resp.Count)
	}
}
