// Package queries provides typed query functions over idea_engine.db.
package queries

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"srfm-lab/idea-engine/idea-api/db"
	"srfm-lab/idea-engine/idea-api/types"
)

// HypothesisStore wraps IdeaDB and exposes hypothesis-specific queries.
type HypothesisStore struct {
	db *db.IdeaDB
}

// NewHypothesisStore constructs a HypothesisStore.
func NewHypothesisStore(d *db.IdeaDB) *HypothesisStore {
	return &HypothesisStore{db: d}
}

// GetHypotheses returns a paginated, optionally filtered list of hypotheses.
// Pass an empty string for status to return all statuses.
func (s *HypothesisStore) GetHypotheses(ctx context.Context, status string, limit, offset int) ([]types.Hypothesis, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}

	var (
		rows []db.Row
		err  error
	)

	if status != "" {
		rows, err = s.db.QueryRows(ctx, `
			SELECT hypothesis_id, statement, status, priority_rank,
			       expected_alpha, confidence_score, source_pattern_ids,
			       created_at, updated_at
			FROM   hypotheses
			WHERE  status = ?
			ORDER  BY priority_rank ASC, created_at DESC
			LIMIT  ? OFFSET ?
		`, status, limit, offset)
	} else {
		rows, err = s.db.QueryRows(ctx, `
			SELECT hypothesis_id, statement, status, priority_rank,
			       expected_alpha, confidence_score, source_pattern_ids,
			       created_at, updated_at
			FROM   hypotheses
			ORDER  BY priority_rank ASC, created_at DESC
			LIMIT  ? OFFSET ?
		`, limit, offset)
	}
	if err != nil {
		return nil, fmt.Errorf("GetHypotheses: %w", err)
	}

	return parseHypotheses(rows)
}

// GetHypothesisByID returns a single hypothesis by its UUID.
// Returns sql.ErrNoRows if not found.
func (s *HypothesisStore) GetHypothesisByID(ctx context.Context, id string) (types.Hypothesis, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT hypothesis_id, statement, status, priority_rank,
		       expected_alpha, confidence_score, source_pattern_ids,
		       created_at, updated_at
		FROM   hypotheses
		WHERE  hypothesis_id = ?
		LIMIT  1
	`, id)
	if err != nil {
		return types.Hypothesis{}, fmt.Errorf("GetHypothesisByID: %w", err)
	}
	if len(rows) == 0 {
		return types.Hypothesis{}, sql.ErrNoRows
	}
	hs, err := parseHypotheses(rows)
	if err != nil {
		return types.Hypothesis{}, err
	}
	return hs[0], nil
}

// UpdateHypothesisStatus sets the status field of the hypothesis identified by id.
// Returns sql.ErrNoRows if no row was updated.
func (s *HypothesisStore) UpdateHypothesisStatus(ctx context.Context, id, status string) error {
	n, err := s.db.Execute(ctx, `
		UPDATE hypotheses
		SET    status = ?, updated_at = ?
		WHERE  hypothesis_id = ?
	`, status, time.Now().UTC().Format(time.RFC3339), id)
	if err != nil {
		return fmt.Errorf("UpdateHypothesisStatus: %w", err)
	}
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// GetTopHypotheses returns the n highest-priority hypotheses regardless of status.
func (s *HypothesisStore) GetTopHypotheses(ctx context.Context, n int) ([]types.Hypothesis, error) {
	if n <= 0 {
		n = 10
	}
	rows, err := s.db.QueryRows(ctx, `
		SELECT hypothesis_id, statement, status, priority_rank,
		       expected_alpha, confidence_score, source_pattern_ids,
		       created_at, updated_at
		FROM   hypotheses
		ORDER  BY priority_rank ASC, expected_alpha DESC
		LIMIT  ?
	`, n)
	if err != nil {
		return nil, fmt.Errorf("GetTopHypotheses: %w", err)
	}
	return parseHypotheses(rows)
}

// CountHypotheses returns the total number of hypotheses, optionally filtered
// by status (pass empty string for all).
func (s *HypothesisStore) CountHypotheses(ctx context.Context, status string) (int, error) {
	var row db.Row
	var err error
	if status != "" {
		row, err = s.db.QueryRow(ctx, `SELECT COUNT(*) AS n FROM hypotheses WHERE status = ?`, status)
	} else {
		row, err = s.db.QueryRow(ctx, `SELECT COUNT(*) AS n FROM hypotheses`)
	}
	if err != nil {
		return 0, fmt.Errorf("CountHypotheses: %w", err)
	}
	if row == nil {
		return 0, nil
	}
	return toInt(row["n"]), nil
}

// parseHypotheses converts raw query rows into typed Hypothesis slices.
func parseHypotheses(rows []db.Row) ([]types.Hypothesis, error) {
	out := make([]types.Hypothesis, 0, len(rows))
	for _, r := range rows {
		h := types.Hypothesis{
			HypothesisID:    toString(r["hypothesis_id"]),
			Statement:       toString(r["statement"]),
			Status:          toString(r["status"]),
			PriorityRank:    toInt(r["priority_rank"]),
			ExpectedAlpha:   toFloat64(r["expected_alpha"]),
			ConfidenceScore: toFloat64(r["confidence_score"]),
		}

		if sp, ok := r["source_pattern_ids"]; ok && sp != nil {
			raw := toString(sp)
			if raw != "" && raw != "null" {
				h.SourcePatternIDs = json.RawMessage(raw)
			}
		}

		var err error
		h.CreatedAt, err = parseTime(toString(r["created_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseHypotheses created_at: %w", err)
		}
		h.UpdatedAt, err = parseTime(toString(r["updated_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseHypotheses updated_at: %w", err)
		}
		out = append(out, h)
	}
	return out, nil
}

// ---- shared helpers ----

func toString(v any) string {
	if v == nil {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	case []byte:
		return string(t)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func toInt(v any) int {
	if v == nil {
		return 0
	}
	switch t := v.(type) {
	case int64:
		return int(t)
	case float64:
		return int(t)
	case int:
		return t
	default:
		return 0
	}
}

func toFloat64(v any) float64 {
	if v == nil {
		return 0
	}
	switch t := v.(type) {
	case float64:
		return t
	case int64:
		return float64(t)
	default:
		return 0
	}
}

func toBool(v any) bool {
	if v == nil {
		return false
	}
	switch t := v.(type) {
	case bool:
		return t
	case int64:
		return t != 0
	default:
		return false
	}
}

func parseTime(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, nil
	}
	// Try RFC3339Nano first, fall back to RFC3339.
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		t, err = time.Parse(time.RFC3339, s)
		if err != nil {
			return time.Time{}, fmt.Errorf("parse time %q: %w", s, err)
		}
	}
	return t.UTC(), nil
}

func parseTimePtr(s string) (*time.Time, error) {
	if s == "" {
		return nil, nil
	}
	t, err := parseTime(s)
	if err != nil {
		return nil, err
	}
	return &t, nil
}

func toFloat64Ptr(v any) *float64 {
	if v == nil {
		return nil
	}
	f := toFloat64(v)
	return &f
}
