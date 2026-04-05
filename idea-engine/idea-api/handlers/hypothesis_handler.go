// Package handlers provides chi HTTP handlers for the idea-api service.
package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"go.uber.org/zap"

	"srfm-lab/idea-engine/idea-api/db/queries"
	"srfm-lab/idea-engine/idea-api/types"
)

// HypothesisHandler handles HTTP requests for the /api/v1/hypotheses routes.
type HypothesisHandler struct {
	store *queries.HypothesisStore
	log   *zap.Logger
}

// NewHypothesisHandler constructs a HypothesisHandler.
func NewHypothesisHandler(store *queries.HypothesisStore, log *zap.Logger) *HypothesisHandler {
	return &HypothesisHandler{store: store, log: log}
}

// GetHypotheses handles GET /api/v1/hypotheses
// Query params: status (optional), limit (default 100), offset (default 0)
func (h *HypothesisHandler) GetHypotheses(w http.ResponseWriter, r *http.Request) {
	status := r.URL.Query().Get("status")
	limit := queryInt(r, "limit", 100)
	offset := queryInt(r, "offset", 0)

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	hyps, err := h.store.GetHypotheses(ctx, status, limit, offset)
	if err != nil {
		h.log.Error("GetHypotheses query failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to query hypotheses")
		return
	}

	total, err := h.store.CountHypotheses(ctx, status)
	if err != nil {
		h.log.Warn("CountHypotheses failed", zap.Error(err))
		total = len(hyps)
	}

	writeJSON(w, http.StatusOK, types.PaginatedResponse{
		Data:   hyps,
		Total:  total,
		Limit:  limit,
		Offset: offset,
	})
}

// GetHypothesisByID handles GET /api/v1/hypotheses/{id}
func (h *HypothesisHandler) GetHypothesisByID(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "hypothesis id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	hyp, err := h.store.GetHypothesisByID(ctx, id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "hypothesis not found")
		return
	}
	if err != nil {
		h.log.Error("GetHypothesisByID failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch hypothesis")
		return
	}

	writeJSON(w, http.StatusOK, hyp)
}

// ApproveHypothesis handles POST /api/v1/hypotheses/{id}/approve
// Transitions a hypothesis from pending → queued.
func (h *HypothesisHandler) ApproveHypothesis(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "hypothesis id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	// Verify the hypothesis exists and is in a promotable state.
	hyp, err := h.store.GetHypothesisByID(ctx, id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "hypothesis not found")
		return
	}
	if err != nil {
		h.log.Error("ApproveHypothesis: fetch failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch hypothesis")
		return
	}
	if hyp.Status != "pending" {
		writeError(w, http.StatusConflict,
			"hypothesis must be in 'pending' status to be approved; current status: "+hyp.Status)
		return
	}

	if err := h.store.UpdateHypothesisStatus(ctx, id, "queued"); err != nil {
		h.log.Error("ApproveHypothesis: update failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to approve hypothesis")
		return
	}

	h.log.Info("hypothesis approved", zap.String("id", id))
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"hypothesis_id": id,
		"status":        "queued",
		"approved_at":   time.Now().UTC().Format(time.RFC3339),
	})
}

// RejectHypothesis handles POST /api/v1/hypotheses/{id}/reject
// Transitions a hypothesis to rejected status.
func (h *HypothesisHandler) RejectHypothesis(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "hypothesis id is required")
		return
	}

	// Parse optional rejection reason from body.
	var body struct {
		Reason string `json:"reason"`
	}
	if r.Body != nil {
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
		_ = json.NewDecoder(r.Body).Decode(&body)
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	hyp, err := h.store.GetHypothesisByID(ctx, id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "hypothesis not found")
		return
	}
	if err != nil {
		h.log.Error("RejectHypothesis: fetch failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch hypothesis")
		return
	}
	if hyp.Status == "rejected" {
		writeError(w, http.StatusConflict, "hypothesis is already rejected")
		return
	}
	if hyp.Status == "done" {
		writeError(w, http.StatusConflict, "cannot reject a completed hypothesis")
		return
	}

	if err := h.store.UpdateHypothesisStatus(ctx, id, "rejected"); err != nil {
		h.log.Error("RejectHypothesis: update failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to reject hypothesis")
		return
	}

	h.log.Info("hypothesis rejected", zap.String("id", id), zap.String("reason", body.Reason))
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"hypothesis_id": id,
		"status":        "rejected",
		"reason":        body.Reason,
		"rejected_at":   time.Now().UTC().Format(time.RFC3339),
	})
}

// GetTopHypotheses handles GET /api/v1/hypotheses/top
// Query params: n (default 10)
func (h *HypothesisHandler) GetTopHypotheses(w http.ResponseWriter, r *http.Request) {
	n := queryInt(r, "n", 10)

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	hyps, err := h.store.GetTopHypotheses(ctx, n)
	if err != nil {
		h.log.Error("GetTopHypotheses failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch top hypotheses")
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"hypotheses": hyps,
		"count":      len(hyps),
	})
}

// ---- shared handler helpers ----

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, types.ErrorResponse{Error: msg, Code: status})
}

func queryInt(r *http.Request, key string, defaultVal int) int {
	s := r.URL.Query().Get(key)
	if s == "" {
		return defaultVal
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 0 {
		return defaultVal
	}
	return n
}
