package handlers

import (
	"context"
	"database/sql"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"go.uber.org/zap"

	"srfm-lab/idea-engine/idea-api/db/queries"
	"srfm-lab/idea-engine/idea-api/types"
)

// PatternHandler handles HTTP requests for the /api/v1/patterns routes.
type PatternHandler struct {
	store *queries.PatternStore
	log   *zap.Logger
}

// NewPatternHandler constructs a PatternHandler.
func NewPatternHandler(store *queries.PatternStore, log *zap.Logger) *PatternHandler {
	return &PatternHandler{store: store, log: log}
}

// GetPatterns handles GET /api/v1/patterns
// Query params: type (optional), limit (default 100), offset (default 0)
func (h *PatternHandler) GetPatterns(w http.ResponseWriter, r *http.Request) {
	patternType := r.URL.Query().Get("type")
	limit := queryInt(r, "limit", 100)
	offset := queryInt(r, "offset", 0)

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	patterns, err := h.store.GetPatterns(ctx, patternType, limit, offset)
	if err != nil {
		h.log.Error("GetPatterns failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch patterns")
		return
	}

	total, err := h.store.CountPatterns(ctx, patternType)
	if err != nil {
		h.log.Warn("CountPatterns failed", zap.Error(err))
		total = len(patterns)
	}

	writeJSON(w, http.StatusOK, types.PaginatedResponse{
		Data:   patterns,
		Total:  total,
		Limit:  limit,
		Offset: offset,
	})
}

// GetPatternByID handles GET /api/v1/patterns/{id}
func (h *PatternHandler) GetPatternByID(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "pattern id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	pattern, err := h.store.GetPatternByID(ctx, id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "pattern not found")
		return
	}
	if err != nil {
		h.log.Error("GetPatternByID failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch pattern")
		return
	}

	writeJSON(w, http.StatusOK, pattern)
}
