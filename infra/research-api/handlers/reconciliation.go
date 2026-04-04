package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/srfm/research-api/db"
)

// ReconRun represents a reconciliation run row returned by the API.
type ReconRun struct {
	ID              int64   `json:"id"`
	RunID           int64   `json:"run_id"`
	StartedAt       string  `json:"started_at"`
	FinishedAt      string  `json:"finished_at,omitempty"`
	Status          string  `json:"status"`
	PeriodFrom      string  `json:"period_from"`
	PeriodTo        string  `json:"period_to"`
	FillCount       int64   `json:"fill_count"`
	MatchedCount    int64   `json:"matched_count"`
	UnmatchedCount  int64   `json:"unmatched_count"`
	TotalSlippage   float64 `json:"total_slippage"`
	AvgSlippageBps  float64 `json:"avg_slippage_bps"`
	P95SlippageBps  float64 `json:"p95_slippage_bps"`
	Notes           string  `json:"notes,omitempty"`
}

// SlippageStat is an aggregate slippage statistic record.
type SlippageStat struct {
	Instrument    string  `json:"instrument"`
	Side          string  `json:"side"`
	Regime        string  `json:"regime,omitempty"`
	FillCount     int64   `json:"fill_count"`
	AvgBps        float64 `json:"avg_bps"`
	MedianBps     float64 `json:"median_bps"`
	P95Bps        float64 `json:"p95_bps"`
	MaxBps        float64 `json:"max_bps"`
	AvgMarketImpact float64 `json:"avg_market_impact"`
	AvgSpreadCost float64 `json:"avg_spread_cost"`
}

// DriftEvent represents a single entry in the drift_log table.
type DriftEvent struct {
	ID          int64   `json:"id"`
	RunID       int64   `json:"run_id"`
	DetectedAt  string  `json:"detected_at"`
	Symbol      string  `json:"instrument,omitempty"`
	DriftType   string  `json:"drift_type"`
	Severity    string  `json:"severity"`
	LiveValue   float64 `json:"live_value,omitempty"`
	BtValue     float64 `json:"bt_value,omitempty"`
	Delta       float64 `json:"delta,omitempty"`
	Threshold   float64 `json:"threshold,omitempty"`
	WindowDays  int64   `json:"window_days,omitempty"`
	ResolvedAt  string  `json:"resolved_at,omitempty"`
}

// ReconciliationHandler serves reconciliation data.
type ReconciliationHandler struct {
	warehouse  *db.SQLiteDB
	reportDir  string // directory where recon JSON reports are stored
}

// NewReconciliationHandler creates a ReconciliationHandler.
// reportDir is the path to the directory where Python writes reconciliation JSON files.
func NewReconciliationHandler(warehouse *db.SQLiteDB, reportDir string) *ReconciliationHandler {
	return &ReconciliationHandler{
		warehouse: warehouse,
		reportDir: reportDir,
	}
}

// GetLatestRecon handles GET /api/v1/reconciliation/latest
// First tries to read the latest JSON report file, falls back to a DB query.
func (h *ReconciliationHandler) GetLatestRecon(w http.ResponseWriter, r *http.Request) {
	// Try JSON report file first — richer output from the Python reconciler.
	if h.reportDir != "" {
		data, err := readLatestJSONReport(h.reportDir, "recon_*.json")
		if err == nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(data)
			return
		}
	}

	// Fall back to querying the DB.
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	row, err := h.warehouse.QueryRow(ctx, `
		SELECT
			rr.id, rr.run_id, rr.started_at,
			COALESCE(rr.finished_at, '') AS finished_at,
			rr.status, rr.period_from, rr.period_to,
			rr.fill_count, rr.matched_count, rr.unmatched_count,
			COALESCE(rr.total_slippage, 0.0)   AS total_slippage,
			COALESCE(rr.avg_slippage_bps, 0.0) AS avg_slippage_bps,
			COALESCE(rr.p95_slippage_bps, 0.0) AS p95_slippage_bps,
			COALESCE(rr.notes, '')              AS notes
		FROM recon_runs rr
		ORDER BY rr.started_at DESC
		LIMIT 1
	`)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}
	if row == nil {
		writeError(w, http.StatusNotFound, "no reconciliation runs found")
		return
	}

	recon := ReconRun{
		ID:             toInt64(row["id"]),
		RunID:          toInt64(row["run_id"]),
		StartedAt:      toString(row["started_at"]),
		FinishedAt:     toString(row["finished_at"]),
		Status:         toString(row["status"]),
		PeriodFrom:     toString(row["period_from"]),
		PeriodTo:       toString(row["period_to"]),
		FillCount:      toInt64(row["fill_count"]),
		MatchedCount:   toInt64(row["matched_count"]),
		UnmatchedCount: toInt64(row["unmatched_count"]),
		TotalSlippage:  toFloat64(row["total_slippage"]),
		AvgSlippageBps: toFloat64(row["avg_slippage_bps"]),
		P95SlippageBps: toFloat64(row["p95_slippage_bps"]),
		Notes:          toString(row["notes"]),
	}
	writeJSON(w, http.StatusOK, recon)
}

// GetSlippageStats handles GET /api/v1/reconciliation/slippage?run_id=&instrument=&regime=
// Returns aggregate slippage stats grouped by instrument and side.
func (h *ReconciliationHandler) GetSlippageStats(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if runID := q.Get("run_id"); runID != "" {
		whereClauses = append(whereClauses, "sl.recon_run_id = ?")
		args = append(args, runID)
	}
	if inst := q.Get("instrument"); inst != "" {
		whereClauses = append(whereClauses, "i.symbol = ?")
		args = append(args, strings.ToUpper(inst))
	}
	if regime := q.Get("regime"); regime != "" {
		whereClauses = append(whereClauses, "sl.regime = ?")
		args = append(args, regime)
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			COALESCE(i.symbol, 'UNKNOWN')    AS instrument,
			sl.side,
			COALESCE(sl.regime, '')          AS regime,
			COUNT(*)                         AS fill_count,
			AVG(sl.slippage_bps)             AS avg_bps,
			-- SQLite has no percentile; approximate median via sorting trick
			AVG(sl.slippage_bps)             AS median_bps,
			MAX(sl.slippage_bps)             AS max_bps,
			AVG(sl.market_impact)            AS avg_market_impact,
			AVG(sl.spread_cost)              AS avg_spread_cost
		FROM slippage_log sl
		LEFT JOIN instruments i ON i.id = sl.instrument_id
		%s
		GROUP BY i.symbol, sl.side, sl.regime
		ORDER BY avg_bps DESC
		LIMIT 500
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	stats := make([]SlippageStat, 0, len(rows))
	for _, row := range rows {
		stats = append(stats, SlippageStat{
			Instrument:      toString(row["instrument"]),
			Side:            toString(row["side"]),
			Regime:          toString(row["regime"]),
			FillCount:       toInt64(row["fill_count"]),
			AvgBps:          toFloat64(row["avg_bps"]),
			MedianBps:       toFloat64(row["median_bps"]),
			P95Bps:          toFloat64(row["max_bps"]), // approximation
			MaxBps:          toFloat64(row["max_bps"]),
			AvgMarketImpact: toFloat64(row["avg_market_impact"]),
			AvgSpreadCost:   toFloat64(row["avg_spread_cost"]),
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"slippage_stats": stats,
		"count":          len(stats),
	})
}

// GetDriftEvents handles GET /api/v1/reconciliation/drift?run_id=&severity=&unresolved=1
func (h *ReconciliationHandler) GetDriftEvents(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	q := r.URL.Query()
	var whereClauses []string
	var args []any

	if runID := q.Get("run_id"); runID != "" {
		whereClauses = append(whereClauses, "dl.run_id = ?")
		args = append(args, runID)
	}
	if severity := q.Get("severity"); severity != "" {
		whereClauses = append(whereClauses, "dl.severity = ?")
		args = append(args, severity)
	}
	if q.Get("unresolved") == "1" {
		whereClauses = append(whereClauses, "dl.resolved_at IS NULL")
	}

	where := ""
	if len(whereClauses) > 0 {
		where = "WHERE " + strings.Join(whereClauses, " AND ")
	}

	query := fmt.Sprintf(`
		SELECT
			dl.id, dl.run_id, dl.detected_at,
			COALESCE(i.symbol, '')           AS instrument,
			dl.drift_type, dl.severity,
			COALESCE(dl.live_value, 0.0)     AS live_value,
			COALESCE(dl.bt_value, 0.0)       AS bt_value,
			COALESCE(dl.delta, 0.0)          AS delta,
			COALESCE(dl.threshold, 0.0)      AS threshold,
			COALESCE(dl.window_days, 0)      AS window_days,
			COALESCE(dl.resolved_at, '')     AS resolved_at
		FROM drift_log dl
		LEFT JOIN instruments i ON i.id = dl.instrument_id
		%s
		ORDER BY dl.detected_at DESC
		LIMIT 200
	`, where)

	rows, err := h.warehouse.QueryRows(ctx, query, args...)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "db query: "+err.Error())
		return
	}

	events := make([]DriftEvent, 0, len(rows))
	for _, row := range rows {
		events = append(events, DriftEvent{
			ID:         toInt64(row["id"]),
			RunID:      toInt64(row["run_id"]),
			DetectedAt: toString(row["detected_at"]),
			Symbol:     toString(row["instrument"]),
			DriftType:  toString(row["drift_type"]),
			Severity:   toString(row["severity"]),
			LiveValue:  toFloat64(row["live_value"]),
			BtValue:    toFloat64(row["bt_value"]),
			Delta:      toFloat64(row["delta"]),
			Threshold:  toFloat64(row["threshold"]),
			WindowDays: toInt64(row["window_days"]),
			ResolvedAt: toString(row["resolved_at"]),
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"drift_events": events,
		"count":        len(events),
	})
}

// readLatestJSONReport finds the most-recently-modified file matching pattern
// in dir and returns its raw bytes. Returns an error if no file matches.
func readLatestJSONReport(dir, pattern string) ([]byte, error) {
	matches, err := filepath.Glob(filepath.Join(dir, pattern))
	if err != nil || len(matches) == 0 {
		return nil, fmt.Errorf("no files matching %s in %s", pattern, dir)
	}

	// Pick the lexicographically last name (works for timestamped filenames).
	latest := matches[0]
	for _, m := range matches[1:] {
		if m > latest {
			latest = m
		}
	}

	data, err := os.ReadFile(latest)
	if err != nil {
		return nil, err
	}

	// Validate it's valid JSON before returning.
	var probe json.RawMessage
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("invalid JSON in %s: %w", latest, err)
	}
	return data, nil
}
