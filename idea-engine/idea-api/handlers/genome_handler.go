package handlers

import (
	"context"
	"net/http"
	"strconv"
	"time"

	"go.uber.org/zap"

	"srfm-lab/idea-engine/idea-api/db/queries"
)

// GenomeHandler handles HTTP requests for the /api/v1/genomes routes.
type GenomeHandler struct {
	store *queries.GenomeStore
	log   *zap.Logger
}

// NewGenomeHandler constructs a GenomeHandler.
func NewGenomeHandler(store *queries.GenomeStore, log *zap.Logger) *GenomeHandler {
	return &GenomeHandler{store: store, log: log}
}

// GetPopulation handles GET /api/v1/genomes/population
// Query params: generation (optional, default -1 = latest)
func (h *GenomeHandler) GetPopulation(w http.ResponseWriter, r *http.Request) {
	generation := -1
	if s := r.URL.Query().Get("generation"); s != "" {
		n, err := strconv.Atoi(s)
		if err == nil && n >= 0 {
			generation = n
		}
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	genomes, err := h.store.GetGenomePopulation(ctx, generation)
	if err != nil {
		h.log.Error("GetPopulation failed", zap.Int("generation", generation), zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch genome population")
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"generation": generation,
		"genomes":    genomes,
		"count":      len(genomes),
	})
}

// GetArchive handles GET /api/v1/genomes/archive
// Query params: limit (default 100)
func (h *GenomeHandler) GetArchive(w http.ResponseWriter, r *http.Request) {
	limit := queryInt(r, "limit", 100)

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	genomes, err := h.store.GetArchive(ctx, limit)
	if err != nil {
		h.log.Error("GetArchive failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch genome archive")
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"genomes": genomes,
		"count":   len(genomes),
		"limit":   limit,
	})
}

// GetFitnessHistory handles GET /api/v1/genomes/fitness-history
func (h *GenomeHandler) GetFitnessHistory(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	history, err := h.store.GetFitnessHistory(ctx)
	if err != nil {
		h.log.Error("GetFitnessHistory failed", zap.Error(err))
		writeError(w, http.StatusInternalServerError, "failed to fetch fitness history")
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"history":     history,
		"generations": len(history),
	})
}

// GetGenomeByID handles GET /api/v1/genomes/{id}
func (h *GenomeHandler) GetGenomeByID(w http.ResponseWriter, r *http.Request) {
	// chi URL param extraction — id is set by the router
	id := r.URL.Query().Get("id")
	if id == "" {
		// Try path param via chi context.
		if v := r.Context().Value(struct{ key string }{"id"}); v != nil {
			id, _ = v.(string)
		}
	}
	if id == "" {
		writeError(w, http.StatusBadRequest, "genome id is required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	genome, err := h.store.GetGenomeByID(ctx, id)
	if err != nil {
		h.log.Error("GetGenomeByID failed", zap.String("id", id), zap.Error(err))
		writeError(w, http.StatusNotFound, "genome not found")
		return
	}

	writeJSON(w, http.StatusOK, genome)
}
