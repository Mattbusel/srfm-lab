package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// newTestBacktestHandler creates a BacktestHandler backed by an in-memory
// SQLite database. The worker pool is NOT started (queue is never drained)
// so tests remain deterministic.
func newTestBacktestHandler(t *testing.T) *BacktestHandler {
	t.Helper()
	h, err := NewBacktestHandler(":memory:", "python")
	if err != nil {
		t.Fatalf("NewBacktestHandler: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	return h
}

func TestSubmitJobReturnsAccepted(t *testing.T) {
	h := newTestBacktestHandler(t)
	body, _ := json.Marshal(submitRequest{
		GenomeID: "g1",
		Params:   map[string]float64{"fast": 5, "slow": 20},
	})
	req := httptest.NewRequest(http.MethodPost, "/backtest/submit", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	h.SubmitJob(rr, req)
	if rr.Code != http.StatusAccepted {
		t.Errorf("expected 202, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp map[string]interface{}
	_ = json.NewDecoder(rr.Body).Decode(&resp)
	if resp["job_id"] == "" {
		t.Error("expected job_id in response")
	}
}

func TestSubmitJobMissingGenomeID(t *testing.T) {
	h := newTestBacktestHandler(t)
	body, _ := json.Marshal(submitRequest{Params: map[string]float64{"x": 1}})
	req := httptest.NewRequest(http.MethodPost, "/backtest/submit", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	h.SubmitJob(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestSubmitJobEmptyParams(t *testing.T) {
	h := newTestBacktestHandler(t)
	body, _ := json.Marshal(submitRequest{GenomeID: "g2", Params: map[string]float64{}})
	req := httptest.NewRequest(http.MethodPost, "/backtest/submit", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	h.SubmitJob(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestGetStatusFoundInMemory(t *testing.T) {
	h := newTestBacktestHandler(t)
	// Manually insert a job without going through the queue.
	job := &BacktestJob{
		JobID:       "job-abc",
		GenomeID:    "g-test",
		Status:      "queued",
		SubmittedAt: time.Now(),
		Params:      map[string]float64{"p": 1},
	}
	h.mu.Lock()
	h.jobs[job.JobID] = job
	h.mu.Unlock()

	req := httptest.NewRequest(http.MethodGet, "/backtest/status?job_id=job-abc", nil)
	rr := httptest.NewRecorder()
	h.GetStatus(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	var resp BacktestJob
	_ = json.NewDecoder(rr.Body).Decode(&resp)
	if resp.JobID != "job-abc" {
		t.Errorf("expected job_id=job-abc, got %s", resp.JobID)
	}
}

func TestGetStatusNotFound(t *testing.T) {
	h := newTestBacktestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/backtest/status?job_id=nope", nil)
	rr := httptest.NewRecorder()
	h.GetStatus(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", rr.Code)
	}
}

func TestGetResultsForGenome(t *testing.T) {
	h := newTestBacktestHandler(t)
	now := time.Now()
	result := &BacktestResult{Sharpe: 1.5, MaxDrawdown: 0.08}
	job := &BacktestJob{
		JobID:       "j1",
		GenomeID:    "gX",
		Status:      "completed",
		SubmittedAt: now,
		CompletedAt: &now,
		Result:      result,
		Params:      map[string]float64{"a": 1},
	}
	h.mu.Lock()
	h.jobs[job.JobID] = job
	h.mu.Unlock()

	req := httptest.NewRequest(http.MethodGet, "/backtest/results?genome_id=gX", nil)
	rr := httptest.NewRecorder()
	h.GetResults(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	var resp BacktestJob
	_ = json.NewDecoder(rr.Body).Decode(&resp)
	if resp.Result == nil || resp.Result.Sharpe != 1.5 {
		t.Errorf("expected result.Sharpe=1.5, got %+v", resp.Result)
	}
}

func TestGetResultsMissingGenomeID(t *testing.T) {
	h := newTestBacktestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/backtest/results", nil)
	rr := httptest.NewRecorder()
	h.GetResults(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", rr.Code)
	}
}

func TestGetHistoryEmpty(t *testing.T) {
	h := newTestBacktestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/backtest/history?n=10", nil)
	rr := httptest.NewRecorder()
	h.GetHistory(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestSubmitAndPersistToDB(t *testing.T) {
	h := newTestBacktestHandler(t)
	body, _ := json.Marshal(submitRequest{
		GenomeID: "persist-test",
		Params:   map[string]float64{"k": 99},
	})
	req := httptest.NewRequest(http.MethodPost, "/backtest/submit", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	h.SubmitJob(rr, req)
	if rr.Code != http.StatusAccepted {
		t.Fatalf("submit failed: %d", rr.Code)
	}

	// Verify the job appears in history (DB query).
	histReq := httptest.NewRequest(http.MethodGet, "/backtest/history?n=10", nil)
	histRR := httptest.NewRecorder()
	h.GetHistory(histRR, histReq)
	var resp map[string]interface{}
	_ = json.NewDecoder(histRR.Body).Decode(&resp)
	count := int(resp["count"].(float64))
	if count < 1 {
		t.Errorf("expected at least 1 job in history, got %d", count)
	}
}

func TestMethodNotAllowedOnSubmit(t *testing.T) {
	h := newTestBacktestHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/backtest/submit", nil)
	rr := httptest.NewRecorder()
	h.SubmitJob(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", rr.Code)
	}
}
