package handlers

import (
	"context"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"go.uber.org/zap"

	"srfm-lab/idea-engine/idea-api/db/queries"
)

// ShadowHandler handles HTTP requests for the /api/v1/shadow routes.
type ShadowHandler struct {
	store *queries.ShadowStore
	log   *zap.Logger
}

// NewShadowHandler constructs a ShadowHandler.
func NewShadowHandler(store *queries.ShadowStore, log *zap.Logger) *ShadowHandler {
	return &ShadowHandler{store: store, log: log}
}

// GetLeaderboard handles GET /api/v1/shadow/leaderboard
// Query params: limit (default 50)
func (h *ShadowHandler) GetLeaderboard(w http.ResponseWriter, r *http.Request) {
	limit := queryInt(r, "limit", 50)

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	variants, err := h.store.GetLeaderboard(ctx, limit)
	if err != nil {
		h.log.Error("GetLeaderboard failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch leaderboard")
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"variants": variants,
		"count":    len(variants),
		"limit":    limit,
	})
}

// GetVariantHistory handles GET /api/v1/shadow/variants/{id}/history
func (h *ShadowHandler) GetVariantHistory(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "variant id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	// Confirm the variant exists.
	variant, err := h.store.GetVariantByID(ctx, id)
	if err != nil {
		h.log.Warn("GetVariantHistory: variant not found", zap.String("id", id))
		writeError(w, http.StatusNotFound, "shadow variant not found")
		return
	}

	history, err := h.store.GetVariantHistory(ctx, id)
	if err != nil {
		h.log.Error("GetVariantHistory failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch variant history")
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"variant": variant,
		"history": history,
		"count":   len(history),
	})
}

// PromoteVariant handles POST /api/v1/shadow/variants/{id}/promote
// Marks the variant as promoted to live trading.
func (h *ShadowHandler) PromoteVariant(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "variant id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	// Verify variant exists before promoting.
	variant, err := h.store.GetVariantByID(ctx, id)
	if err != nil {
		writeError(w, http.StatusNotFound, "shadow variant not found")
		return
	}
	if variant.IsPromoted {
		writeError(w, http.StatusConflict, "variant is already promoted")
		return
	}

	if err := h.store.PromoteVariant(ctx, id); err != nil {
		h.log.Error("PromoteVariant failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to promote variant")
		return
	}

	h.log.Info("shadow variant promoted", zap.String("id", id), zap.String("name", variant.Name))
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"variant_id":  id,
		"name":        variant.Name,
		"is_promoted": true,
		"promoted_at": time.Now().UTC().Format(time.RFC3339),
	})
}
