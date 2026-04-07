package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func newTestPatternHandler(t *testing.T) *PatternHandler {
	t.Helper()
	h, err := NewPatternHandler(":memory:")
	if err != nil {
		t.Fatalf("NewPatternHandler: %v", err)
	}
	t.Cleanup(func() { h.Close() })
	return h
}

func TestGetConfirmedEmpty(t *testing.T) {
	h := newTestPatternHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/patterns/confirmed", nil)
	rr := httptest.NewRecorder()
	h.GetConfirmed(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	var resp map[string]interface{}
	_ = json.NewDecoder(rr.Body).Decode(&resp)
	if resp["count"].(float64) != 0 {
		t.Error("expected empty result set")
	}
}

func TestCreateAndGetPattern(t *testing.T) {
	h := newTestPatternHandler(t)
	p := Pattern{
		ID:           "p1",
		PatternType:  "momentum",
		Symbol:       "BTC-USD",
		Confidence:   0.85,
		DiscoveredAt: time.Now(),
	}
	if err := h.CreatePattern(p); err != nil {
		t.Fatalf("CreatePattern: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/patterns/p1", nil)
	rr := httptest.NewRecorder()
	h.GetPattern(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	var got Pattern
	_ = json.NewDecoder(rr.Body).Decode(&got)
	if got.Confidence != 0.85 {
		t.Errorf("expected confidence=0.85, got %f", got.Confidence)
	}
}

func TestGetPatternNotFound(t *testing.T) {
	h := newTestPatternHandler(t)
	req := httptest.NewRequest(http.MethodGet, "/patterns/does-not-exist", nil)
	rr := httptest.NewRecorder()
	h.GetPattern(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", rr.Code)
	}
}

func TestConfirmPattern(t *testing.T) {
	h := newTestPatternHandler(t)
	p := Pattern{ID: "p2", PatternType: "mean_reversion", Symbol: "ETH-USD",
		Confidence: 0.7, DiscoveredAt: time.Now()}
	_ = h.CreatePattern(p)

	body, _ := json.Marshal(confirmRequest{ID: "p2", Metadata: map[string]interface{}{"analyst": "auto"}})
	req := httptest.NewRequest(http.MethodPost, "/patterns/confirm", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	h.ConfirmPattern(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	var got Pattern
	_ = json.NewDecoder(rr.Body).Decode(&got)
	if got.ConfirmedAt == nil {
		t.Error("expected ConfirmedAt to be set after confirmation")
	}
}

func TestConfirmPatternNotFound(t *testing.T) {
	h := newTestPatternHandler(t)
	body, _ := json.Marshal(confirmRequest{ID: "ghost"})
	req := httptest.NewRequest(http.MethodPost, "/patterns/confirm", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	h.ConfirmPattern(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", rr.Code)
	}
}

func TestDeletePattern(t *testing.T) {
	h := newTestPatternHandler(t)
	p := Pattern{ID: "del1", PatternType: "regime", Symbol: "SPY",
		Confidence: 0.5, DiscoveredAt: time.Now()}
	_ = h.CreatePattern(p)

	req := httptest.NewRequest(http.MethodDelete, "/patterns/del1", nil)
	rr := httptest.NewRecorder()
	h.DeletePattern(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}

	// Confirm the pattern no longer appears in confirmed list.
	confReq := httptest.NewRequest(http.MethodGet, "/patterns/confirmed", nil)
	confRR := httptest.NewRecorder()
	h.GetConfirmed(confRR, confReq)
	var resp map[string]interface{}
	_ = json.NewDecoder(confRR.Body).Decode(&resp)
	if resp["count"].(float64) != 0 {
		t.Error("invalidated pattern should not appear in confirmed list")
	}
}

func TestDeletePatternNotFound(t *testing.T) {
	h := newTestPatternHandler(t)
	req := httptest.NewRequest(http.MethodDelete, "/patterns/nobody", nil)
	rr := httptest.NewRecorder()
	h.DeletePattern(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", rr.Code)
	}
}

func TestGetStatsMultipleTypes(t *testing.T) {
	h := newTestPatternHandler(t)
	for i, pt := range []string{"momentum", "momentum", "mean_reversion"} {
		_ = h.CreatePattern(Pattern{
			ID: fmt.Sprintf("s%d", i), PatternType: pt, Symbol: "X",
			Confidence: 0.6, DiscoveredAt: time.Now(),
		})
	}
	req := httptest.NewRequest(http.MethodGet, "/patterns/stats", nil)
	rr := httptest.NewRecorder()
	h.GetStats(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
	var stats patternStats
	_ = json.NewDecoder(rr.Body).Decode(&stats)
	if stats.TotalPatterns != 3 {
		t.Errorf("expected 3 total patterns, got %d", stats.TotalPatterns)
	}
	if stats.ByType["momentum"] != 2 {
		t.Errorf("expected 2 momentum, got %d", stats.ByType["momentum"])
	}
}

func TestGetConfirmedSinceFilter(t *testing.T) {
	h := newTestPatternHandler(t)
	old := time.Now().Add(-2 * time.Hour)
	confirmed := old
	p := Pattern{ID: "old1", PatternType: "t", Symbol: "S", Confidence: 0.5, DiscoveredAt: old}
	_ = h.CreatePattern(p)
	_, _ = h.db.Exec(`UPDATE patterns SET confirmed_at = ? WHERE id = 'old1'`, confirmed)

	// Query with a "since" that is after the confirmation -- should return nothing.
	future := time.Now().Add(1 * time.Hour).Format(time.RFC3339)
	req := httptest.NewRequest(http.MethodGet, "/patterns/confirmed?since="+future, nil)
	rr := httptest.NewRecorder()
	h.GetConfirmed(rr, req)
	var resp map[string]interface{}
	_ = json.NewDecoder(rr.Body).Decode(&resp)
	if resp["count"].(float64) != 0 {
		t.Errorf("expected 0 patterns since future timestamp, got %v", resp["count"])
	}
}

func TestExtractPathID(t *testing.T) {
	cases := []struct {
		path   string
		prefix string
		want   string
	}{
		{"/patterns/abc123", "/patterns/", "abc123"},
		{"/patterns/confirmed", "/patterns/", "confirmed"},
		{"/patterns/", "/patterns/", ""},
		{"/other/abc", "/patterns/", ""},
	}
	for _, tc := range cases {
		got := extractPathID(tc.path, tc.prefix)
		if got != tc.want {
			t.Errorf("extractPathID(%q, %q) = %q, want %q", tc.path, tc.prefix, got, tc.want)
		}
	}
}
