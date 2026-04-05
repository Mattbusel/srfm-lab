package queries

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	"srfm-lab/idea-engine/idea-api/db"
	"srfm-lab/idea-engine/idea-api/types"
)

// PatternStore wraps IdeaDB and exposes pattern-specific queries.
type PatternStore struct {
	db *db.IdeaDB
}

// NewPatternStore constructs a PatternStore.
func NewPatternStore(d *db.IdeaDB) *PatternStore {
	return &PatternStore{db: d}
}

// GetPatterns returns a paginated list of patterns, optionally filtered by
// patternType. Pass an empty string for patternType to return all types.
func (s *PatternStore) GetPatterns(ctx context.Context, patternType string, limit, offset int) ([]types.Pattern, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}

	var (
		rows []db.Row
		err  error
	)

	if patternType != "" {
		rows, err = s.db.QueryRows(ctx, `
			SELECT pattern_id, pattern_type, description, confidence,
			       frequency, features, discovered_at, run_id
			FROM   patterns
			WHERE  pattern_type = ?
			ORDER  BY confidence DESC, discovered_at DESC
			LIMIT  ? OFFSET ?
		`, patternType, limit, offset)
	} else {
		rows, err = s.db.QueryRows(ctx, `
			SELECT pattern_id, pattern_type, description, confidence,
			       frequency, features, discovered_at, run_id
			FROM   patterns
			ORDER  BY confidence DESC, discovered_at DESC
			LIMIT  ? OFFSET ?
		`, limit, offset)
	}
	if err != nil {
		return nil, fmt.Errorf("GetPatterns: %w", err)
	}
	return parsePatterns(rows)
}

// GetPatternByID returns a single pattern by its UUID.
// Returns sql.ErrNoRows if not found.
func (s *PatternStore) GetPatternByID(ctx context.Context, id string) (types.Pattern, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT pattern_id, pattern_type, description, confidence,
		       frequency, features, discovered_at, run_id
		FROM   patterns
		WHERE  pattern_id = ?
		LIMIT  1
	`, id)
	if err != nil {
		return types.Pattern{}, fmt.Errorf("GetPatternByID: %w", err)
	}
	if len(rows) == 0 {
		return types.Pattern{}, sql.ErrNoRows
	}
	ps, err := parsePatterns(rows)
	if err != nil {
		return types.Pattern{}, err
	}
	return ps[0], nil
}

// GetPatternsByRunID returns all patterns produced by a specific mining run.
func (s *PatternStore) GetPatternsByRunID(ctx context.Context, runID string) ([]types.Pattern, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT pattern_id, pattern_type, description, confidence,
		       frequency, features, discovered_at, run_id
		FROM   patterns
		WHERE  run_id = ?
		ORDER  BY confidence DESC
	`, runID)
	if err != nil {
		return nil, fmt.Errorf("GetPatternsByRunID: %w", err)
	}
	return parsePatterns(rows)
}

// CountPatterns returns the total number of patterns, optionally filtered by
// patternType.
func (s *PatternStore) CountPatterns(ctx context.Context, patternType string) (int, error) {
	var (
		row db.Row
		err error
	)
	if patternType != "" {
		row, err = s.db.QueryRow(ctx, `SELECT COUNT(*) AS n FROM patterns WHERE pattern_type = ?`, patternType)
	} else {
		row, err = s.db.QueryRow(ctx, `SELECT COUNT(*) AS n FROM patterns`)
	}
	if err != nil {
		return 0, fmt.Errorf("CountPatterns: %w", err)
	}
	if row == nil {
		return 0, nil
	}
	return toInt(row["n"]), nil
}

// parsePatterns converts raw query rows into typed Pattern slices.
func parsePatterns(rows []db.Row) ([]types.Pattern, error) {
	out := make([]types.Pattern, 0, len(rows))
	for _, r := range rows {
		p := types.Pattern{
			PatternID:   toString(r["pattern_id"]),
			PatternType: toString(r["pattern_type"]),
			Description: toString(r["description"]),
			Confidence:  toFloat64(r["confidence"]),
			Frequency:   toInt(r["frequency"]),
			RunID:       toString(r["run_id"]),
		}
		if f := toString(r["features"]); f != "" && f != "null" {
			p.Features = json.RawMessage(f)
		}
		var err error
		p.DiscoveredAt, err = parseTime(toString(r["discovered_at"]))
		if err != nil {
			return nil, fmt.Errorf("parsePatterns discovered_at: %w", err)
		}
		out = append(out, p)
	}
	return out, nil
}
