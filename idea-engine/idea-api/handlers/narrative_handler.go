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

// NarrativeHandler handles HTTP requests for the /api/v1/narratives routes.
type NarrativeHandler struct {
	store *queries.NarrativeStore
	log   *zap.Logger
}

// NewNarrativeHandler constructs a NarrativeHandler.
func NewNarrativeHandler(store *queries.NarrativeStore, log *zap.Logger) *NarrativeHandler {
	return &NarrativeHandler{store: store, log: log}
}

// GetNarratives handles GET /api/v1/narratives
// Query params: limit (default 50), since (RFC3339, optional)
func (h *NarrativeHandler) GetNarratives(w http.ResponseWriter, r *http.Request) {
	limit := queryInt(r, "limit", 50)

	var since time.Time
	if s := r.URL.Query().Get("since"); s != "" {
		t, err := time.Parse(time.RFC3339, s)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid 'since' timestamp; use RFC3339")
			return
		}
		since = t
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	narratives, err := h.store.GetNarratives(ctx, limit, since)
	if err != nil {
		h.log.Error("GetNarratives failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch narratives")
		return
	}

	total, err := h.store.CountNarratives(ctx)
	if err != nil {
		h.log.Warn("CountNarratives failed", zap.Error(err))
		total = len(narratives)
	}

	writeJSON(w, http.StatusOK, types.PaginatedResponse{
		Data:   narratives,
		Total:  total,
		Limit:  limit,
		Offset: 0,
	})
}

// GetNarrativeByID handles GET /api/v1/narratives/{id}
func (h *NarrativeHandler) GetNarrativeByID(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "narrative id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	narrative, err := h.store.GetNarrativeByID(ctx, id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "narrative not found")
		return
	}
	if err != nil {
		h.log.Error("GetNarrativeByID failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch narrative")
		return
	}

	writeJSON(w, http.StatusOK, narrative)
}

// GetLatestNarrative handles GET /api/v1/narratives/latest
func (h *NarrativeHandler) GetLatestNarrative(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	narrative, err := h.store.GetLatestNarrative(ctx)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, "no narratives found")
		return
	}
	if err != nil {
		h.log.Error("GetLatestNarrative failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch latest narrative")
		return
	}

	writeJSON(w, http.StatusOK, narrative)
}
