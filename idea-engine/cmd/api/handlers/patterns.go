// Package handlers -- patterns.go implements the /patterns/* REST endpoints.
//
// Routes:
//   GET    /patterns/confirmed  -- confirmed patterns since a timestamp
//   GET    /patterns/stats      -- aggregate pattern statistics
//   GET    /patterns/:id        -- single pattern by ID
//   POST   /patterns/confirm    -- mark a pattern as confirmed
//   DELETE /patterns/:id        -- invalidate a pattern
package handlers

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

// Pattern represents a discovered trading signal pattern.
type Pattern struct {
	// ID is the unique identifier for this pattern.
	ID string `json:"id"`
	// PatternType describes the signal class (e.g. "momentum", "mean_reversion").
	PatternType string `json:"pattern_type"`
	// Symbol is the instrument this pattern was observed on.
	Symbol string `json:"symbol"`
	// Confidence is the model confidence score in [0, 1].
	Confidence float64 `json:"confidence"`
	// DiscoveredAt is when the pattern was first observed.
	DiscoveredAt time.Time `json:"discovered_at"`
	// ConfirmedAt is when a human or automated system validated the pattern.
	// Nil for unconfirmed patterns.
	ConfirmedAt *time.Time `json:"confirmed_at,omitempty"`
	// Metadata holds arbitrary JSON-serialisable annotations.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	// Invalidated is true if the pattern has been deleted/rejected.
	Invalidated bool `json:"invalidated"`
}

// ---------------------------------------------------------------------------
// PatternHandler
// ---------------------------------------------------------------------------

// PatternHandler handles all /patterns/* routes backed by SQLite.
type PatternHandler struct {
	db *sql.DB
}

// NewPatternHandler constructs a PatternHandler and ensures the schema exists.
// dbPath may be ":memory:" for tests.
func NewPatternHandler(dbPath string) (*PatternHandler, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	if err := migratePatternDB(db); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return &PatternHandler{db: db}, nil
}

// Close releases the database connection.
func (h *PatternHandler) Close() { _ = h.db.Close() }

// ---------------------------------------------------------------------------
// GET /patterns/confirmed
// ---------------------------------------------------------------------------

// GetConfirmed returns all confirmed, non-invalidated patterns that were
// confirmed at or after the `since` query parameter (RFC3339 timestamp).
// If `since` is absent, all confirmed patterns are returned.
func (h *PatternHandler) GetConfirmed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	var since time.Time
	if s := r.URL.Query().Get("since"); s != "" {
		var err error
		since, err = time.Parse(time.RFC3339, s)
		if err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid since timestamp: %v", err))
			return
		}
	}

	var rows *sql.Rows
	var err error
	if since.IsZero() {
		rows, err = h.db.Query(
			`SELECT id, pattern_type, symbol, confidence, discovered_at, confirmed_at, metadata_json, invalidated
			 FROM patterns WHERE confirmed_at IS NOT NULL AND invalidated = 0 ORDER BY confirmed_at DESC`)
	} else {
		rows, err = h.db.Query(
			`SELECT id, pattern_type, symbol, confidence, discovered_at, confirmed_at, metadata_json, invalidated
			 FROM patterns WHERE confirmed_at IS NOT NULL AND confirmed_at >= ? AND invalidated = 0
			 ORDER BY confirmed_at DESC`, since)
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("query: %v", err))
		return
	}
	defer rows.Close()

	patterns, err := scanPatterns(rows)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("scan: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"patterns": patterns,
		"count":    len(patterns),
	})
}

// ---------------------------------------------------------------------------
// GET /patterns/:id
// ---------------------------------------------------------------------------

// GetPattern returns a single pattern by ID.
// The ID is extracted from the URL path segment after /patterns/.
func (h *PatternHandler) GetPattern(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	id := extractPathID(r.URL.Path, "/patterns/")
	if id == "" {
		writeError(w, http.StatusBadRequest, "pattern ID required in path")
		return
	}

	p, err := h.loadPattern(id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, fmt.Sprintf("pattern %s not found", id))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("load: %v", err))
		return
	}
	writeJSON(w, http.StatusOK, p)
}

// ---------------------------------------------------------------------------
// POST /patterns/confirm
// ---------------------------------------------------------------------------

// confirmRequest is the body for POST /patterns/confirm.
type confirmRequest struct {
	// ID is the pattern to confirm.
	ID string `json:"id"`
	// Metadata is additional annotations to store alongside the confirmation.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ConfirmPattern marks an existing pattern as confirmed.
// If the pattern is already confirmed, the request is idempotent.
func (h *PatternHandler) ConfirmPattern(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req confirmRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid body: %v", err))
		return
	}
	if req.ID == "" {
		writeError(w, http.StatusBadRequest, "id is required")
		return
	}

	p, err := h.loadPattern(req.ID)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, fmt.Sprintf("pattern %s not found", req.ID))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("load: %v", err))
		return
	}
	if p.Invalidated {
		writeError(w, http.StatusConflict, "cannot confirm an invalidated pattern")
		return
	}

	// Merge incoming metadata with existing metadata.
	if p.Metadata == nil {
		p.Metadata = make(map[string]interface{})
	}
	for k, v := range req.Metadata {
		p.Metadata[k] = v
	}

	now := time.Now()
	p.ConfirmedAt = &now

	metaJSON, _ := json.Marshal(p.Metadata)
	_, err = h.db.Exec(
		`UPDATE patterns SET confirmed_at = ?, metadata_json = ? WHERE id = ?`,
		now, string(metaJSON), req.ID,
	)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("update: %v", err))
		return
	}
	writeJSON(w, http.StatusOK, p)
}

// ---------------------------------------------------------------------------
// DELETE /patterns/:id
// ---------------------------------------------------------------------------

// DeletePattern marks a pattern as invalidated (soft delete).
// The data is retained in the database for audit purposes.
func (h *PatternHandler) DeletePattern(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		writeError(w, http.StatusMethodNotAllowed, "use DELETE")
		return
	}
	id := extractPathID(r.URL.Path, "/patterns/")
	if id == "" {
		writeError(w, http.StatusBadRequest, "pattern ID required in path")
		return
	}

	result, err := h.db.Exec(`UPDATE patterns SET invalidated = 1 WHERE id = ?`, id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("update: %v", err))
		return
	}
	n, _ := result.RowsAffected()
	if n == 0 {
		writeError(w, http.StatusNotFound, fmt.Sprintf("pattern %s not found", id))
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"status": "invalidated",
		"id":     id,
	})
}

// ---------------------------------------------------------------------------
// GET /patterns/stats
// ---------------------------------------------------------------------------

// patternStats is the JSON body for GET /patterns/stats.
type patternStats struct {
	// TotalPatterns is the count of all non-invalidated patterns.
	TotalPatterns int `json:"total_patterns"`
	// ConfirmedPatterns is the count of confirmed non-invalidated patterns.
	ConfirmedPatterns int `json:"confirmed_patterns"`
	// ByType maps PatternType to count.
	ByType map[string]int `json:"by_type"`
	// AvgConfidence is the mean confidence across all non-invalidated patterns.
	AvgConfidence float64 `json:"avg_confidence"`
	// DiscoveryRatePerDay is patterns discovered per day over the last 30 days.
	DiscoveryRatePerDay float64 `json:"discovery_rate_per_day"`
	// OldestDiscovery is the timestamp of the earliest known pattern.
	OldestDiscovery *time.Time `json:"oldest_discovery,omitempty"`
}

// GetStats returns aggregate statistics about all stored patterns.
func (h *PatternHandler) GetStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	var stats patternStats

	// Total and confirmed counts.
	row := h.db.QueryRow(
		`SELECT COUNT(*), SUM(CASE WHEN confirmed_at IS NOT NULL THEN 1 ELSE 0 END), AVG(confidence)
		 FROM patterns WHERE invalidated = 0`)
	var avgConf sql.NullFloat64
	_ = row.Scan(&stats.TotalPatterns, &stats.ConfirmedPatterns, &avgConf)
	if avgConf.Valid {
		stats.AvgConfidence = avgConf.Float64
	}

	// Count by type.
	rows, err := h.db.Query(
		`SELECT pattern_type, COUNT(*) FROM patterns WHERE invalidated = 0 GROUP BY pattern_type`)
	if err == nil {
		defer rows.Close()
		stats.ByType = make(map[string]int)
		for rows.Next() {
			var pt string
			var cnt int
			if err := rows.Scan(&pt, &cnt); err == nil {
				stats.ByType[pt] = cnt
			}
		}
	}

	// 30-day discovery rate.
	since30 := time.Now().AddDate(0, 0, -30)
	var cnt30 int
	_ = h.db.QueryRow(`SELECT COUNT(*) FROM patterns WHERE discovered_at >= ? AND invalidated = 0`, since30).Scan(&cnt30)
	stats.DiscoveryRatePerDay = float64(cnt30) / 30.0

	// Oldest discovery.
	var oldest sql.NullTime
	_ = h.db.QueryRow(`SELECT MIN(discovered_at) FROM patterns WHERE invalidated = 0`).Scan(&oldest)
	if oldest.Valid {
		t := oldest.Time
		stats.OldestDiscovery = &t
	}

	writeJSON(w, http.StatusOK, stats)
}

// ---------------------------------------------------------------------------
// POST /patterns -- create a new pattern (internal helper used by tests)
// ---------------------------------------------------------------------------

// CreatePattern inserts a new Pattern record.
// This is an internal helper method; not directly registered as an HTTP route
// but may be wired up by the main server.
func (h *PatternHandler) CreatePattern(p Pattern) error {
	if p.ID == "" {
		p.ID = generateID()
	}
	if p.DiscoveredAt.IsZero() {
		p.DiscoveredAt = time.Now()
	}
	metaJSON, _ := json.Marshal(p.Metadata)
	_, err := h.db.Exec(
		`INSERT INTO patterns (id, pattern_type, symbol, confidence, discovered_at, confirmed_at, metadata_json, invalidated)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		p.ID, p.PatternType, p.Symbol, p.Confidence,
		p.DiscoveredAt, p.ConfirmedAt, string(metaJSON), p.Invalidated,
	)
	return err
}

// ---------------------------------------------------------------------------
// SQLite helpers
// ---------------------------------------------------------------------------

// migratePatternDB creates the patterns table if it does not exist.
func migratePatternDB(db *sql.DB) error {
	_, err := db.Exec(`CREATE TABLE IF NOT EXISTS patterns (
		id            TEXT PRIMARY KEY,
		pattern_type  TEXT NOT NULL,
		symbol        TEXT NOT NULL,
		confidence    REAL NOT NULL DEFAULT 0,
		discovered_at DATETIME NOT NULL,
		confirmed_at  DATETIME,
		metadata_json TEXT,
		invalidated   INTEGER NOT NULL DEFAULT 0
	)`)
	return err
}

// loadPattern retrieves a single pattern by ID.
func (h *PatternHandler) loadPattern(id string) (Pattern, error) {
	row := h.db.QueryRow(
		`SELECT id, pattern_type, symbol, confidence, discovered_at, confirmed_at, metadata_json, invalidated
		 FROM patterns WHERE id = ?`, id)
	return scanPattern(row)
}

// scanPatterns converts a *sql.Rows result to a slice of Pattern.
func scanPatterns(rows *sql.Rows) ([]Pattern, error) {
	var out []Pattern
	for rows.Next() {
		p, err := scanPattern(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	if out == nil {
		out = []Pattern{}
	}
	return out, rows.Err()
}

// scanPattern reads one row into a Pattern.
func scanPattern(row rowScanner) (Pattern, error) {
	var (
		p           Pattern
		confirmedAt sql.NullTime
		metaJSON    sql.NullString
		invalidated int
	)
	if err := row.Scan(&p.ID, &p.PatternType, &p.Symbol, &p.Confidence,
		&p.DiscoveredAt, &confirmedAt, &metaJSON, &invalidated); err != nil {
		return p, err
	}
	if confirmedAt.Valid {
		t := confirmedAt.Time
		p.ConfirmedAt = &t
	}
	if metaJSON.Valid && metaJSON.String != "" && metaJSON.String != "null" {
		_ = json.Unmarshal([]byte(metaJSON.String), &p.Metadata)
	}
	p.Invalidated = invalidated != 0
	return p, nil
}

// extractPathID pulls the trailing ID segment from a URL path.
// e.g. "/patterns/abc123" with prefix "/patterns/" returns "abc123".
func extractPathID(path, prefix string) string {
	if !strings.HasPrefix(path, prefix) {
		return ""
	}
	id := strings.TrimPrefix(path, prefix)
	id = strings.TrimRight(id, "/")
	// Reject sub-routes like "confirmed" or "stats".
	if id == "" || strings.Contains(id, "/") {
		return ""
	}
	return id
}
