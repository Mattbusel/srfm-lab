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

// NarrativeStore wraps IdeaDB and exposes narrative-specific queries.
type NarrativeStore struct {
	db *db.IdeaDB
}

// NewNarrativeStore constructs a NarrativeStore.
func NewNarrativeStore(d *db.IdeaDB) *NarrativeStore {
	return &NarrativeStore{db: d}
}

// GetNarratives returns a paginated list of narratives ordered by
// generated_at descending. Pass a zero time for since to return all.
func (s *NarrativeStore) GetNarratives(ctx context.Context, limit int, since time.Time) ([]types.Narrative, error) {
	if limit <= 0 || limit > 500 {
		limit = 50
	}

	var (
		rows []db.Row
		err  error
	)

	if !since.IsZero() {
		rows, err = s.db.QueryRows(ctx, `
			SELECT narrative_id, title, body, hypothesis_ids, experiment_ids, generated_at
			FROM   narratives
			WHERE  generated_at >= ?
			ORDER  BY generated_at DESC
			LIMIT  ?
		`, since.UTC().Format(time.RFC3339), limit)
	} else {
		rows, err = s.db.QueryRows(ctx, `
			SELECT narrative_id, title, body, hypothesis_ids, experiment_ids, generated_at
			FROM   narratives
			ORDER  BY generated_at DESC
			LIMIT  ?
		`, limit)
	}
	if err != nil {
		return nil, fmt.Errorf("GetNarratives: %w", err)
	}
	return parseNarratives(rows)
}

// GetNarrativeByID returns a single narrative by its UUID.
// Returns sql.ErrNoRows if not found.
func (s *NarrativeStore) GetNarrativeByID(ctx context.Context, id string) (types.Narrative, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT narrative_id, title, body, hypothesis_ids, experiment_ids, generated_at
		FROM   narratives
		WHERE  narrative_id = ?
		LIMIT  1
	`, id)
	if err != nil {
		return types.Narrative{}, fmt.Errorf("GetNarrativeByID: %w", err)
	}
	if len(rows) == 0 {
		return types.Narrative{}, sql.ErrNoRows
	}
	ns, err := parseNarratives(rows)
	if err != nil {
		return types.Narrative{}, err
	}
	return ns[0], nil
}

// GetLatestNarrative returns the single most recently generated narrative.
// Returns sql.ErrNoRows if the narratives table is empty.
func (s *NarrativeStore) GetLatestNarrative(ctx context.Context) (types.Narrative, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT narrative_id, title, body, hypothesis_ids, experiment_ids, generated_at
		FROM   narratives
		ORDER  BY generated_at DESC
		LIMIT  1
	`)
	if err != nil {
		return types.Narrative{}, fmt.Errorf("GetLatestNarrative: %w", err)
	}
	if len(rows) == 0 {
		return types.Narrative{}, sql.ErrNoRows
	}
	ns, err := parseNarratives(rows)
	if err != nil {
		return types.Narrative{}, err
	}
	return ns[0], nil
}

// CountNarratives returns the total number of narratives in the table.
func (s *NarrativeStore) CountNarratives(ctx context.Context) (int, error) {
	row, err := s.db.QueryRow(ctx, `SELECT COUNT(*) AS n FROM narratives`)
	if err != nil {
		return 0, fmt.Errorf("CountNarratives: %w", err)
	}
	if row == nil {
		return 0, nil
	}
	return toInt(row["n"]), nil
}

// parseNarratives converts raw rows into typed Narrative slices.
func parseNarratives(rows []db.Row) ([]types.Narrative, error) {
	out := make([]types.Narrative, 0, len(rows))
	for _, r := range rows {
		n := types.Narrative{
			NarrativeID: toString(r["narrative_id"]),
			Title:       toString(r["title"]),
			Body:        toString(r["body"]),
		}
		if hi := toString(r["hypothesis_ids"]); hi != "" && hi != "null" {
			n.HypothesisIDs = json.RawMessage(hi)
		}
		if ei := toString(r["experiment_ids"]); ei != "" && ei != "null" {
			n.ExperimentIDs = json.RawMessage(ei)
		}
		var err error
		n.GeneratedAt, err = parseTime(toString(r["generated_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseNarratives generated_at: %w", err)
		}
		out = append(out, n)
	}
	return out, nil
}
