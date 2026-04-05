package handlers

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"go.uber.org/zap"

	"srfm-lab/idea-engine/idea-api/db/queries"
	"srfm-lab/idea-engine/idea-api/types"
)

// ExperimentHandler handles HTTP requests for the /api/v1/experiments routes.
type ExperimentHandler struct {
	store        *queries.ExperimentStore
	schedulerURL string
	log          *zap.Logger
	httpClient   *http.Client
}

// NewExperimentHandler constructs an ExperimentHandler.
// schedulerURL is the base URL of the scheduler service (e.g. http://localhost:8769).
func NewExperimentHandler(store *queries.ExperimentStore, schedulerURL string, log *zap.Logger) *ExperimentHandler {
	return &ExperimentHandler{
		store:        store,
		schedulerURL: schedulerURL,
		log:          log,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// GetExperiments handles GET /api/v1/experiments
// Query params: status (optional), limit (default 100), offset (default 0)
func (h *ExperimentHandler) GetExperiments(w http.ResponseWriter, r *http.Request) {
	status := r.URL.Query().Get("status")
	limit := queryInt(r, "limit", 100)
	offset := queryInt(r, "offset", 0)

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	exps, err := h.store.GetExperiments(ctx, status, limit, offset)
	if err != nil {
		h.log.Error("GetExperiments failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch experiments")
		return
	}

	total, err := h.store.CountExperiments(ctx, status)
	if err != nil {
		h.log.Warn("CountExperiments failed", zap.Error(err))
		total = len(exps)
	}

	writeJSON(w, http.StatusOK, types.PaginatedResponse{
		Data:   exps,
		Total:  total,
		Limit:  limit,
		Offset: offset,
	})
}

// GetExperimentByID handles GET /api/v1/experiments/{id}
func (h *ExperimentHandler) GetExperimentByID(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "experiment id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	exp, err := h.store.GetExperimentByID(ctx, id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "experiment not found")
		return
	}
	if err != nil {
		h.log.Error("GetExperimentByID failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch experiment")
		return
	}

	writeJSON(w, http.StatusOK, exp)
}

// createExperimentRequest is the body accepted by CreateExperiment.
type createExperimentRequest struct {
	HypothesisID   string          `json:"hypothesis_id"`
	ExperimentType string          `json:"experiment_type"`
	Priority       int             `json:"priority"`
	Config         json.RawMessage `json:"config"`
}

// CreateExperiment handles POST /api/v1/experiments
func (h *ExperimentHandler) CreateExperiment(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var req createExperimentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}

	if req.HypothesisID == "" {
		writeError(w, http.StatusBadRequest, "hypothesis_id is required")
		return
	}
	validTypes := map[string]bool{
		"genome": true, "counterfactual": true, "shadow": true,
		"causal": true, "academic": true, "serendipity": true,
	}
	if !validTypes[req.ExperimentType] {
		writeError(w, http.StatusBadRequest,
			fmt.Sprintf("experiment_type must be one of: genome, counterfactual, shadow, causal, academic, serendipity; got %q", req.ExperimentType))
		return
	}

	exp := types.Experiment{
		ExperimentID:   generateID(),
		HypothesisID:   req.HypothesisID,
		ExperimentType: req.ExperimentType,
		Status:         "queued",
		Priority:       req.Priority,
		Config:         req.Config,
		RetryCount:     0,
		CreatedAt:      time.Now().UTC(),
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	if err := h.store.CreateExperiment(ctx, exp); err != nil {
		h.log.Error("CreateExperiment failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to create experiment")
		return
	}

	h.log.Info("experiment created",
		zap.String("id", exp.ExperimentID),
		zap.String("type", exp.ExperimentType),
		zap.String("hypothesis_id", exp.HypothesisID),
	)
	writeJSON(w, http.StatusCreated, exp)
}

// RunExperiment handles POST /api/v1/experiments/{id}/run
// Dispatches the experiment to the scheduler service.
func (h *ExperimentHandler) RunExperiment(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "experiment id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	exp, err := h.store.GetExperimentByID(ctx, id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "experiment not found")
		return
	}
	if err != nil {
		h.log.Error("RunExperiment: fetch failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch experiment")
		return
	}
	if exp.Status == "running" {
		writeError(w, http.StatusConflict, "experiment is already running")
		return
	}
	if exp.Status == "done" {
		writeError(w, http.StatusConflict, "experiment has already completed")
		return
	}

	// Dispatch to scheduler via HTTP POST.
	if err := h.dispatchToScheduler(ctx, exp); err != nil {
		h.log.Error("RunExperiment: dispatch failed",
			zap.String("id", id),
			zap.String("scheduler", h.schedulerURL),
			zap.Error(err),
		)
		writeError(w, http.StatusBadGateway,
			"failed to dispatch to scheduler: "+err.Error())
		return
	}

	h.log.Info("experiment dispatched to scheduler",
		zap.String("id", id),
		zap.String("type", exp.ExperimentType),
	)
	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"experiment_id": id,
		"status":        "queued",
		"dispatched_at": time.Now().UTC().Format(time.RFC3339),
	})
}

// dispatchToScheduler POSTs the experiment to the scheduler service.
func (h *ExperimentHandler) dispatchToScheduler(ctx context.Context, exp types.Experiment) error {
	if h.schedulerURL == "" {
		return fmt.Errorf("scheduler URL not configured")
	}

	body, err := json.Marshal(exp)
	if err != nil {
		return fmt.Errorf("marshal experiment: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx,
		http.MethodPost,
		h.schedulerURL+"/api/v1/dispatch",
		bytes.NewReader(body),
	)
	if err != nil {
		return fmt.Errorf("build scheduler request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("scheduler POST: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("scheduler returned %d", resp.StatusCode)
	}
	return nil
}

// generateID returns a new UUID-like identifier using the current nanosecond
// timestamp combined with a pseudo-random suffix.  In production this would
// use github.com/google/uuid; here we keep the dependency tree minimal.
func generateID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())
}
