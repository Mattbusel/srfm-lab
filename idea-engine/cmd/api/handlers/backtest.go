// Package handlers -- backtest.go implements the /backtest/* REST endpoints.
//
// Routes:
//   GET  /backtest/results     -- results for a specific genome (query: genome_id)
//   GET  /backtest/history     -- last N backtest results (query: n)
//   POST /backtest/submit      -- submit a genome for backtest evaluation
//   GET  /backtest/status/:job_id -- check job status (query: job_id)
package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const (
	// backtestWorkers is the number of parallel backtest goroutines.
	backtestWorkers = 4
	// backtestTimeout is the maximum time a single backtest job may run.
	backtestTimeout = 300 * time.Second
	// backtestHistoryCap is the maximum rows returned by /backtest/history.
	backtestHistoryCap = 200
)

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

// BacktestResult holds the metrics returned by a completed backtest run.
type BacktestResult struct {
	// Sharpe is the annualised Sharpe ratio.
	Sharpe float64 `json:"sharpe"`
	// MaxDrawdown is the maximum drawdown as a positive fraction.
	MaxDrawdown float64 `json:"max_drawdown"`
	// Calmar is the Calmar ratio (annualised return / max drawdown).
	Calmar float64 `json:"calmar"`
	// WinRate is the fraction of profitable trades.
	WinRate float64 `json:"win_rate"`
	// NFills is the total number of executed fills.
	NFills int `json:"n_fills"`
	// AnnualisedReturn is the geometric mean annual return.
	AnnualisedReturn float64 `json:"annualised_return"`
}

// BacktestJob represents a single queued or completed backtest evaluation.
type BacktestJob struct {
	// JobID is the unique identifier for this job, generated at submit time.
	JobID string `json:"job_id"`
	// GenomeID is the identifier of the genome being evaluated.
	GenomeID string `json:"genome_id"`
	// Params is the parameter map to pass to the Python backtest script.
	Params map[string]float64 `json:"params"`
	// Status is one of: "queued", "running", "completed", "failed".
	Status string `json:"status"`
	// SubmittedAt is when the job was created.
	SubmittedAt time.Time `json:"submitted_at"`
	// CompletedAt is populated when the job finishes (success or failure).
	CompletedAt *time.Time `json:"completed_at,omitempty"`
	// Result is populated on successful completion.
	Result *BacktestResult `json:"result,omitempty"`
	// Error holds the error message if Status is "failed".
	Error string `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// BacktestHandler
// ---------------------------------------------------------------------------

// BacktestHandler handles all /backtest/* routes. It manages an in-memory
// job map, a SQLite persistence layer, and a fixed goroutine worker pool.
type BacktestHandler struct {
	db      *sql.DB
	jobs    map[string]*BacktestJob
	mu      sync.RWMutex
	queue   chan *BacktestJob
	wg      sync.WaitGroup
	// pyPath is the Python interpreter to use (default: "python").
	pyPath string
}

// NewBacktestHandler constructs a BacktestHandler and starts the worker pool.
// dbPath is the SQLite file path; use ":memory:" for tests.
// pyPath is the Python executable (e.g. "/usr/bin/python3").
func NewBacktestHandler(dbPath, pyPath string) (*BacktestHandler, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	if err := migrateBacktestDB(db); err != nil {
		return nil, fmt.Errorf("migrate db: %w", err)
	}

	if pyPath == "" {
		pyPath = "python"
	}

	h := &BacktestHandler{
		db:     db,
		jobs:   make(map[string]*BacktestJob),
		queue:  make(chan *BacktestJob, 256),
		pyPath: pyPath,
	}

	// Start fixed-size worker pool.
	for i := 0; i < backtestWorkers; i++ {
		h.wg.Add(1)
		go h.worker()
	}

	// Reload any previously queued/running jobs as failed (process restarted).
	if err := h.reloadJobsFromDB(); err != nil {
		return nil, fmt.Errorf("reload jobs: %w", err)
	}

	return h, nil
}

// Close drains the queue, waits for workers, and closes the database.
func (h *BacktestHandler) Close() {
	close(h.queue)
	h.wg.Wait()
	_ = h.db.Close()
}

// ---------------------------------------------------------------------------
// GET /backtest/results
// ---------------------------------------------------------------------------

// GetResults returns the most recent BacktestJob for a specific genome.
// Query param: genome_id (required).
func (h *BacktestHandler) GetResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	genomeID := r.URL.Query().Get("genome_id")
	if genomeID == "" {
		writeError(w, http.StatusBadRequest, "genome_id is required")
		return
	}

	h.mu.RLock()
	var found *BacktestJob
	for _, j := range h.jobs {
		if j.GenomeID == genomeID {
			if found == nil || j.SubmittedAt.After(found.SubmittedAt) {
				found = j
			}
		}
	}
	h.mu.RUnlock()

	if found == nil {
		writeError(w, http.StatusNotFound, fmt.Sprintf("no job found for genome_id %s", genomeID))
		return
	}
	writeJSON(w, http.StatusOK, found)
}

// ---------------------------------------------------------------------------
// GET /backtest/history
// ---------------------------------------------------------------------------

// GetHistory returns the last N completed backtest jobs, newest first.
// Query param: n (default 50, max 200).
func (h *BacktestHandler) GetHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	n := 50
	if s := r.URL.Query().Get("n"); s != "" {
		fmt.Sscanf(s, "%d", &n)
	}
	if n < 1 {
		n = 1
	}
	if n > backtestHistoryCap {
		n = backtestHistoryCap
	}

	rows, err := h.db.Query(
		`SELECT job_id, genome_id, params_json, status, submitted_at, completed_at, result_json, error_msg
		 FROM backtest_jobs ORDER BY submitted_at DESC LIMIT ?`, n)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("query: %v", err))
		return
	}
	defer rows.Close()

	jobs := make([]*BacktestJob, 0, n)
	for rows.Next() {
		j, err := scanBacktestJob(rows)
		if err != nil {
			continue
		}
		jobs = append(jobs, j)
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"jobs":  jobs,
		"count": len(jobs),
	})
}

// ---------------------------------------------------------------------------
// POST /backtest/submit
// ---------------------------------------------------------------------------

// submitRequest is the JSON body for POST /backtest/submit.
type submitRequest struct {
	// GenomeID identifies the genome being evaluated.
	GenomeID string `json:"genome_id"`
	// Params is the full parameter map to backtest.
	Params map[string]float64 `json:"params"`
}

// SubmitJob enqueues a new backtest job and returns the assigned JobID.
func (h *BacktestHandler) SubmitJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req submitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid body: %v", err))
		return
	}
	if req.GenomeID == "" {
		writeError(w, http.StatusBadRequest, "genome_id is required")
		return
	}
	if len(req.Params) == 0 {
		writeError(w, http.StatusBadRequest, "params must not be empty")
		return
	}

	job := &BacktestJob{
		JobID:       generateID(),
		GenomeID:    req.GenomeID,
		Params:      req.Params,
		Status:      "queued",
		SubmittedAt: time.Now(),
	}

	// Persist immediately so the job survives a restart.
	if err := h.persistJob(job); err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("persist: %v", err))
		return
	}

	h.mu.Lock()
	h.jobs[job.JobID] = job
	h.mu.Unlock()

	// Non-blocking send -- queue is buffered at 256.
	select {
	case h.queue <- job:
	default:
		writeError(w, http.StatusServiceUnavailable, "job queue is full, retry later")
		return
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"job_id":       job.JobID,
		"genome_id":    job.GenomeID,
		"status":       job.Status,
		"submitted_at": job.SubmittedAt,
	})
}

// ---------------------------------------------------------------------------
// GET /backtest/status
// ---------------------------------------------------------------------------

// GetStatus returns the current status and result (if complete) for a job.
// Query param: job_id (required).
func (h *BacktestHandler) GetStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	jobID := r.URL.Query().Get("job_id")
	if jobID == "" {
		// Also support path segment /backtest/status/<id>.
		parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
		if len(parts) >= 3 {
			jobID = parts[2]
		}
	}
	if jobID == "" {
		writeError(w, http.StatusBadRequest, "job_id is required")
		return
	}

	h.mu.RLock()
	job, ok := h.jobs[jobID]
	h.mu.RUnlock()

	if !ok {
		// Try to load from DB in case the server restarted.
		dbJob, err := h.loadJobFromDB(jobID)
		if err != nil {
			writeError(w, http.StatusNotFound, fmt.Sprintf("job not found: %s", jobID))
			return
		}
		job = dbJob
	}

	writeJSON(w, http.StatusOK, job)
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

// worker drains the job queue, running each backtest with a timeout.
func (h *BacktestHandler) worker() {
	defer h.wg.Done()
	for job := range h.queue {
		h.runJob(job)
	}
}

// runJob executes the Python backtest for the given job.
func (h *BacktestHandler) runJob(job *BacktestJob) {
	h.setJobStatus(job, "running", "", nil)

	ctx, cancel := context.WithTimeout(context.Background(), backtestTimeout)
	defer cancel()

	paramsJSON, err := json.Marshal(job.Params)
	if err != nil {
		h.setJobStatus(job, "failed", fmt.Sprintf("marshal params: %v", err), nil)
		return
	}

	cmd := exec.CommandContext(ctx, h.pyPath,
		"-m", "tools.larsa_v18_backtest",
		"--genome-json", string(paramsJSON),
	)

	out, err := cmd.Output()
	now := time.Now()
	if err != nil {
		errMsg := err.Error()
		if ctx.Err() == context.DeadlineExceeded {
			errMsg = "backtest timed out after 300s"
		}
		h.setJobStatus(job, "failed", errMsg, &now)
		return
	}

	var result BacktestResult
	if err := json.Unmarshal(out, &result); err != nil {
		h.setJobStatus(job, "failed", fmt.Sprintf("parse result: %v", err), &now)
		return
	}

	h.mu.Lock()
	job.Status = "completed"
	job.CompletedAt = &now
	job.Result = &result
	h.mu.Unlock()

	_ = h.persistJob(job)
}

// setJobStatus updates a job's status fields atomically.
func (h *BacktestHandler) setJobStatus(job *BacktestJob, status, errMsg string, completedAt *time.Time) {
	h.mu.Lock()
	job.Status = status
	job.Error = errMsg
	job.CompletedAt = completedAt
	h.mu.Unlock()
	_ = h.persistJob(job)
}

// ---------------------------------------------------------------------------
// SQLite persistence
// ---------------------------------------------------------------------------

// migrateBacktestDB creates the backtest_jobs table if it does not exist.
func migrateBacktestDB(db *sql.DB) error {
	_, err := db.Exec(`CREATE TABLE IF NOT EXISTS backtest_jobs (
		job_id       TEXT PRIMARY KEY,
		genome_id    TEXT NOT NULL,
		params_json  TEXT NOT NULL,
		status       TEXT NOT NULL,
		submitted_at DATETIME NOT NULL,
		completed_at DATETIME,
		result_json  TEXT,
		error_msg    TEXT
	)`)
	return err
}

// persistJob upserts a job into the database.
func (h *BacktestHandler) persistJob(job *BacktestJob) error {
	h.mu.RLock()
	paramsJSON, _ := json.Marshal(job.Params)
	var resultJSON []byte
	if job.Result != nil {
		resultJSON, _ = json.Marshal(job.Result)
	}
	h.mu.RUnlock()

	_, err := h.db.Exec(`INSERT OR REPLACE INTO backtest_jobs
		(job_id, genome_id, params_json, status, submitted_at, completed_at, result_json, error_msg)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		job.JobID, job.GenomeID, string(paramsJSON), job.Status,
		job.SubmittedAt, job.CompletedAt, string(resultJSON), job.Error,
	)
	return err
}

// loadJobFromDB loads a single job by ID from the database.
func (h *BacktestHandler) loadJobFromDB(jobID string) (*BacktestJob, error) {
	row := h.db.QueryRow(`SELECT job_id, genome_id, params_json, status, submitted_at, completed_at, result_json, error_msg
		FROM backtest_jobs WHERE job_id = ?`, jobID)
	return scanBacktestJob(row)
}

// reloadJobsFromDB reads unfinished jobs from DB at startup and marks them failed.
func (h *BacktestHandler) reloadJobsFromDB() error {
	rows, err := h.db.Query(`SELECT job_id, genome_id, params_json, status, submitted_at, completed_at, result_json, error_msg
		FROM backtest_jobs WHERE status IN ('queued','running')`)
	if err != nil {
		return err
	}
	defer rows.Close()

	now := time.Now()
	for rows.Next() {
		j, err := scanBacktestJob(rows)
		if err != nil {
			continue
		}
		j.Status = "failed"
		j.Error = "service restarted while job was in-flight"
		j.CompletedAt = &now
		h.mu.Lock()
		h.jobs[j.JobID] = j
		h.mu.Unlock()
		_ = h.persistJob(j)
	}
	return rows.Err()
}

// rowScanner unifies *sql.Row and *sql.Rows for scanBacktestJob.
type rowScanner interface {
	Scan(dest ...interface{}) error
}

// scanBacktestJob reads one row into a BacktestJob.
func scanBacktestJob(row rowScanner) (*BacktestJob, error) {
	var (
		j           BacktestJob
		paramsJSON  string
		resultJSON  sql.NullString
		completedAt sql.NullTime
		errMsg      sql.NullString
	)
	if err := row.Scan(&j.JobID, &j.GenomeID, &paramsJSON, &j.Status,
		&j.SubmittedAt, &completedAt, &resultJSON, &errMsg); err != nil {
		return nil, err
	}
	_ = json.Unmarshal([]byte(paramsJSON), &j.Params)
	if resultJSON.Valid && resultJSON.String != "" && resultJSON.String != "null" {
		var res BacktestResult
		if err := json.Unmarshal([]byte(resultJSON.String), &res); err == nil {
			j.Result = &res
		}
	}
	if completedAt.Valid {
		t := completedAt.Time
		j.CompletedAt = &t
	}
	if errMsg.Valid {
		j.Error = errMsg.String
	}
	return &j, nil
}

// generateID generates a simple time-based unique ID.
// In production you would use uuid.New().String() from github.com/google/uuid.
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}
