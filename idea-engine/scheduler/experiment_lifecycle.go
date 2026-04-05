package scheduler

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"go.uber.org/zap"
)

// ExperimentState is an enumeration of experiment lifecycle states.
type ExperimentState string

const (
	StateQueued  ExperimentState = "queued"
	StateRunning ExperimentState = "running"
	StateDone    ExperimentState = "done"
	StateFailed  ExperimentState = "failed"
)

// ExperimentLifecycle manages the state machine transitions for experiments
// stored in idea_engine.db. It wraps the raw *sql.DB so that the scheduler
// can update status without depending on the full idea-api query layer.
type ExperimentLifecycle struct {
	db     *sql.DB
	pq     *PriorityQueue
	budget *BudgetManager
	retry  *RetryPolicy
	rq     *RetryQueue
	log    *zap.Logger

	// dispatchers maps ExperimentType to the function that runs the experiment.
	dispatchers map[string]DispatchFunc
}

// DispatchFunc is the signature for a module-specific experiment runner.
// It receives the experiment item and must return the JSON result or an error.
type DispatchFunc func(ctx context.Context, item *ExperimentItem) (json.RawMessage, error)

// NewExperimentLifecycle constructs an ExperimentLifecycle.
func NewExperimentLifecycle(
	db *sql.DB,
	pq *PriorityQueue,
	budget *BudgetManager,
	retry *RetryPolicy,
	rq *RetryQueue,
	log *zap.Logger,
) *ExperimentLifecycle {
	return &ExperimentLifecycle{
		db:          db,
		pq:          pq,
		budget:      budget,
		retry:       retry,
		rq:          rq,
		log:         log,
		dispatchers: make(map[string]DispatchFunc),
	}
}

// RegisterDispatcher associates an ExperimentType with its runner function.
func (el *ExperimentLifecycle) RegisterDispatcher(expType string, fn DispatchFunc) {
	el.dispatchers[expType] = fn
}

// Dispatch selects the next experiment from the priority queue, acquires a
// budget slot, transitions it to running, and invokes the registered
// dispatcher in a new goroutine.
//
// Dispatch should be called by a tight loop in the scheduler main loop.
// Returns false if there is nothing to dispatch or no budget is available.
func (el *ExperimentLifecycle) Dispatch(ctx context.Context) bool {
	item := el.pq.Peek()
	if item == nil {
		return false
	}

	if !el.budget.CanRun(item.ExperimentType) {
		el.log.Debug("no budget slot available",
			zap.String("type", item.ExperimentType),
		)
		return false
	}

	// Pop only after confirming budget is available.
	item = el.pq.Pop()
	if item == nil {
		return false
	}

	if err := el.budget.Acquire(item.ExperimentType); err != nil {
		// Race: re-queue and try again later.
		el.pq.Push(item)
		return false
	}

	dispatchFn, ok := el.dispatchers[item.ExperimentType]
	if !ok {
		el.budget.Release(item.ExperimentType)
		el.log.Error("no dispatcher registered",
			zap.String("type", item.ExperimentType),
			zap.String("id", item.ExperimentID),
		)
		_ = el.persistStatus(item.ExperimentID, StateFailed, nil, "no dispatcher for type: "+item.ExperimentType)
		return false
	}

	_ = el.persistStatus(item.ExperimentID, StateRunning, nil, "")

	go func() {
		defer el.budget.Release(item.ExperimentType)
		el.runExperiment(ctx, item, dispatchFn)
	}()

	return true
}

// runExperiment calls the dispatcher, handles success/failure, and updates the DB.
func (el *ExperimentLifecycle) runExperiment(ctx context.Context, item *ExperimentItem, fn DispatchFunc) {
	el.log.Info("dispatching experiment",
		zap.String("id", item.ExperimentID),
		zap.String("type", item.ExperimentType),
		zap.String("hypothesis_id", item.HypothesisID),
	)

	result, err := fn(ctx, item)
	if err != nil {
		el.log.Error("experiment failed",
			zap.String("id", item.ExperimentID),
			zap.String("type", item.ExperimentType),
			zap.Int("retry_count", item.RetryCount),
			zap.Error(err),
		)
		el.handleFailure(item, err.Error())
		return
	}

	el.log.Info("experiment completed",
		zap.String("id", item.ExperimentID),
		zap.String("type", item.ExperimentType),
	)
	_ = el.persistStatus(item.ExperimentID, StateDone, result, "")
}

// handleFailure decides whether to retry or permanently fail the experiment.
func (el *ExperimentLifecycle) handleFailure(item *ExperimentItem, errMsg string) {
	decision := el.retry.Evaluate(item.ExperimentID, item.RetryCount, errMsg)
	if !decision.ShouldRetry {
		_ = el.persistStatus(item.ExperimentID, StateFailed, nil, errMsg)
		return
	}

	// Mark as queued again in the DB; the RetryQueue will re-push it later.
	_ = el.persistStatus(item.ExperimentID, StateQueued, nil, errMsg)

	item.RetryCount++
	ok := el.rq.Enqueue(RetryItem{Item: item, RetryAt: decision.RetryAt})
	if !ok {
		el.log.Error("retry queue full; experiment permanently failed",
			zap.String("id", item.ExperimentID),
		)
		_ = el.persistStatus(item.ExperimentID, StateFailed, nil, "retry queue full")
	}
}

// HandleCompletion is a callback invoked by external runners (e.g. a long-
// running subprocess) when they write results out-of-band. It transitions the
// experiment from running → done and persists the result JSON.
func (el *ExperimentLifecycle) HandleCompletion(expID string, result json.RawMessage) error {
	return el.persistStatus(expID, StateDone, result, "")
}

// persistStatus writes the new state and optional result to the experiments table.
func (el *ExperimentLifecycle) persistStatus(
	expID string,
	state ExperimentState,
	result json.RawMessage,
	errMsg string,
) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	now := time.Now().UTC().Format(time.RFC3339)

	switch state {
	case StateRunning:
		_, err := el.db.ExecContext(ctx, `
			UPDATE experiments
			SET    status = 'running', started_at = ?, updated_at = ?
			WHERE  experiment_id = ?
		`, now, now, expID)
		return err

	case StateDone:
		resultStr := "{}"
		if len(result) > 0 {
			resultStr = string(result)
		}
		_, err := el.db.ExecContext(ctx, `
			UPDATE experiments
			SET    status = 'done', result = ?, completed_at = ?, updated_at = ?
			WHERE  experiment_id = ?
		`, resultStr, now, now, expID)
		return err

	case StateFailed:
		_, err := el.db.ExecContext(ctx, `
			UPDATE experiments
			SET    status = 'failed', error_msg = ?, completed_at = ?,
			       updated_at = ?, retry_count = retry_count + 1
			WHERE  experiment_id = ?
		`, errMsg, now, now, expID)
		return err

	case StateQueued:
		_, err := el.db.ExecContext(ctx, `
			UPDATE experiments
			SET    status = 'queued', updated_at = ?
			WHERE  experiment_id = ?
		`, now, expID)
		return err

	default:
		return fmt.Errorf("experiment_lifecycle: unknown state %q", state)
	}
}

// EnqueueFromDB loads all queued experiments from the database at startup and
// pushes them into the priority queue so the scheduler can resume after a
// crash or restart.
func (el *ExperimentLifecycle) EnqueueFromDB(ctx context.Context) (int, error) {
	rows, err := el.db.QueryContext(ctx, `
		SELECT experiment_id, hypothesis_id, experiment_type, priority, config, retry_count
		FROM   experiments
		WHERE  status IN ('queued', 'running')
		ORDER  BY priority ASC, created_at ASC
	`)
	if err != nil {
		return 0, fmt.Errorf("EnqueueFromDB: %w", err)
	}
	defer rows.Close()

	n := 0
	for rows.Next() {
		var (
			expID, hypID, expType string
			priority, retryCount  int
			configStr             string
		)
		if err := rows.Scan(&expID, &hypID, &expType, &priority, &configStr, &retryCount); err != nil {
			return n, fmt.Errorf("EnqueueFromDB scan: %w", err)
		}
		el.pq.Push(&ExperimentItem{
			ExperimentID:   expID,
			HypothesisID:   hypID,
			ExperimentType: expType,
			Config:         []byte(configStr),
			Priority:       priority,
			RetryCount:     retryCount,
			EnqueuedAt:     time.Now().UTC(),
		})
		n++
	}
	if err := rows.Err(); err != nil {
		return n, fmt.Errorf("EnqueueFromDB iterate: %w", err)
	}
	return n, nil
}
