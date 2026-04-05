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

// ExperimentStore wraps IdeaDB and exposes experiment-specific queries.
type ExperimentStore struct {
	db *db.IdeaDB
}

// NewExperimentStore constructs an ExperimentStore.
func NewExperimentStore(d *db.IdeaDB) *ExperimentStore {
	return &ExperimentStore{db: d}
}

// GetExperiments returns a paginated list of experiments, optionally filtered
// by status. Pass an empty string to return all statuses.
func (s *ExperimentStore) GetExperiments(ctx context.Context, status string, limit, offset int) ([]types.Experiment, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	var (
		rows []db.Row
		err  error
	)
	if status != "" {
		rows, err = s.db.QueryRows(ctx, `
			SELECT experiment_id, hypothesis_id, experiment_type, status,
			       priority, config, result, error_msg, retry_count,
			       created_at, started_at, completed_at
			FROM   experiments
			WHERE  status = ?
			ORDER  BY priority ASC, created_at DESC
			LIMIT  ? OFFSET ?
		`, status, limit, offset)
	} else {
		rows, err = s.db.QueryRows(ctx, `
			SELECT experiment_id, hypothesis_id, experiment_type, status,
			       priority, config, result, error_msg, retry_count,
			       created_at, started_at, completed_at
			FROM   experiments
			ORDER  BY priority ASC, created_at DESC
			LIMIT  ? OFFSET ?
		`, limit, offset)
	}
	if err != nil {
		return nil, fmt.Errorf("GetExperiments: %w", err)
	}
	return parseExperiments(rows)
}

// GetExperimentByID returns a single experiment by its UUID.
// Returns sql.ErrNoRows if not found.
func (s *ExperimentStore) GetExperimentByID(ctx context.Context, id string) (types.Experiment, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT experiment_id, hypothesis_id, experiment_type, status,
		       priority, config, result, error_msg, retry_count,
		       created_at, started_at, completed_at
		FROM   experiments
		WHERE  experiment_id = ?
		LIMIT  1
	`, id)
	if err != nil {
		return types.Experiment{}, fmt.Errorf("GetExperimentByID: %w", err)
	}
	if len(rows) == 0 {
		return types.Experiment{}, sql.ErrNoRows
	}
	es, err := parseExperiments(rows)
	if err != nil {
		return types.Experiment{}, err
	}
	return es[0], nil
}

// CreateExperiment inserts a new experiment row. If exp.ExperimentID is empty,
// the caller must set it before passing.
func (s *ExperimentStore) CreateExperiment(ctx context.Context, exp types.Experiment) error {
	configJSON := "{}"
	if len(exp.Config) > 0 {
		configJSON = string(exp.Config)
	}
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.Execute(ctx, `
		INSERT INTO experiments
		    (experiment_id, hypothesis_id, experiment_type, status, priority,
		     config, retry_count, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`,
		exp.ExperimentID,
		exp.HypothesisID,
		exp.ExperimentType,
		exp.Status,
		exp.Priority,
		configJSON,
		exp.RetryCount,
		now,
	)
	if err != nil {
		return fmt.Errorf("CreateExperiment: %w", err)
	}
	return nil
}

// UpdateExperimentStatus updates the status (and optionally result) of the
// experiment identified by id. Pass a nil result to leave it unchanged.
func (s *ExperimentStore) UpdateExperimentStatus(ctx context.Context, id, status string, result json.RawMessage) error {
	now := time.Now().UTC().Format(time.RFC3339)

	var (
		n   int64
		err error
	)
	if result != nil {
		n, err = s.db.Execute(ctx, `
			UPDATE experiments
			SET    status = ?, result = ?, completed_at = ?, updated_at = ?
			WHERE  experiment_id = ?
		`, status, string(result), now, now, id)
	} else if status == "running" {
		n, err = s.db.Execute(ctx, `
			UPDATE experiments
			SET    status = ?, started_at = ?, updated_at = ?
			WHERE  experiment_id = ?
		`, status, now, now, id)
	} else {
		n, err = s.db.Execute(ctx, `
			UPDATE experiments
			SET    status = ?, updated_at = ?
			WHERE  experiment_id = ?
		`, status, now, id)
	}

	if err != nil {
		return fmt.Errorf("UpdateExperimentStatus: %w", err)
	}
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// MarkExperimentFailed records a failure with an error message and increments
// the retry count.
func (s *ExperimentStore) MarkExperimentFailed(ctx context.Context, id, errMsg string) error {
	now := time.Now().UTC().Format(time.RFC3339)
	n, err := s.db.Execute(ctx, `
		UPDATE experiments
		SET    status = 'failed', error_msg = ?, completed_at = ?,
		       updated_at = ?, retry_count = retry_count + 1
		WHERE  experiment_id = ?
	`, errMsg, now, now, id)
	if err != nil {
		return fmt.Errorf("MarkExperimentFailed: %w", err)
	}
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// CountExperiments returns the total number of experiments, optionally
// filtered by status.
func (s *ExperimentStore) CountExperiments(ctx context.Context, status string) (int, error) {
	var (
		row db.Row
		err error
	)
	if status != "" {
		row, err = s.db.QueryRow(ctx, `SELECT COUNT(*) AS n FROM experiments WHERE status = ?`, status)
	} else {
		row, err = s.db.QueryRow(ctx, `SELECT COUNT(*) AS n FROM experiments`)
	}
	if err != nil {
		return 0, fmt.Errorf("CountExperiments: %w", err)
	}
	if row == nil {
		return 0, nil
	}
	return toInt(row["n"]), nil
}

// parseExperiments converts raw rows into typed Experiment slices.
func parseExperiments(rows []db.Row) ([]types.Experiment, error) {
	out := make([]types.Experiment, 0, len(rows))
	for _, r := range rows {
		e := types.Experiment{
			ExperimentID:   toString(r["experiment_id"]),
			HypothesisID:   toString(r["hypothesis_id"]),
			ExperimentType: toString(r["experiment_type"]),
			Status:         toString(r["status"]),
			Priority:       toInt(r["priority"]),
			ErrorMsg:       toString(r["error_msg"]),
			RetryCount:     toInt(r["retry_count"]),
		}
		if c := toString(r["config"]); c != "" && c != "null" {
			e.Config = json.RawMessage(c)
		}
		if res := toString(r["result"]); res != "" && res != "null" {
			e.Result = json.RawMessage(res)
		}
		var err error
		e.CreatedAt, err = parseTime(toString(r["created_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseExperiments created_at: %w", err)
		}
		e.StartedAt, err = parseTimePtr(toString(r["started_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseExperiments started_at: %w", err)
		}
		e.CompletedAt, err = parseTimePtr(toString(r["completed_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseExperiments completed_at: %w", err)
		}
		out = append(out, e)
	}
	return out, nil
}
