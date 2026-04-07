// Package handlers provides HTTP request handlers for the IAE genome
// evolution REST API. These handlers expose the running evolution state
// over the routes documented below.
//
// Routes:
//   GET  /genome/best          -- best individual with fitness
//   GET  /genome/population    -- full population with fitness scores
//   POST /genome/seed          -- inject a genome into the population
//   GET  /genome/stats         -- generation, diversity, convergence stats
//   GET  /genome/lineage       -- lineage graph for current run
//   POST /genome/evolve/start  -- start an evolution run
//   POST /genome/evolve/stop   -- stop the current run
package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"

	"srfm-lab/idea-engine/pkg/analysis"
	"srfm-lab/idea-engine/pkg/evolution"
)

// ---------------------------------------------------------------------------
// EvolutionEngine -- interface the handler talks to
// ---------------------------------------------------------------------------

// EvolutionEngine is the minimal interface the GenomeHandler requires from
// the running evolution service. This allows the handler to be tested with
// a mock without importing the full engine.
type EvolutionEngine interface {
	// BestIndividual returns the best individual seen so far.
	BestIndividual() (evolution.Individual, error)
	// Population returns all individuals in the current generation.
	Population() ([]evolution.Individual, error)
	// SeedIndividual injects a genome into the next generation.
	SeedIndividual(genes evolution.Genome) error
	// Stats returns summary statistics for the current run.
	Stats() (EvolutionStats, error)
	// StartRun begins an evolution run with the given config.
	StartRun(cfg RunConfig) error
	// StopRun halts the current run gracefully.
	StopRun() error
	// IsRunning reports whether a run is currently active.
	IsRunning() bool
}

// EvolutionStats holds summary statistics for a running or completed evolution.
type EvolutionStats struct {
	// Generation is the current generation number.
	Generation int `json:"generation"`
	// Diversity is the normalised average pairwise Euclidean distance.
	Diversity float64 `json:"diversity"`
	// ConvergenceRate is the EWMA of per-generation fitness improvements.
	ConvergenceRate float64 `json:"convergence_rate"`
	// IsPlateaued is true when no significant improvement over recent gens.
	IsPlateaued bool `json:"is_plateaued"`
	// BestFitnessHistory contains the best fitness for the last 50 generations.
	BestFitnessHistory []float64 `json:"best_fitness_history"`
	// PopulationSize is the current number of individuals.
	PopulationSize int `json:"population_size"`
	// Running is true while an evolution run is active.
	Running bool `json:"running"`
}

// RunConfig parametrises a new evolution run.
type RunConfig struct {
	// PopulationSize is the number of individuals per generation.
	PopulationSize int `json:"population_size"`
	// MaxGenerations is the number of generations to evolve (0 = unlimited).
	MaxGenerations int `json:"max_generations"`
	// NumWorkers is the number of parallel fitness evaluator workers.
	NumWorkers int `json:"num_workers"`
	// EliteRatio is the fraction of the population to preserve as elites.
	EliteRatio float64 `json:"elite_ratio"`
	// MutationRate is the initial per-gene mutation probability.
	MutationRate float64 `json:"mutation_rate"`
	// CrossoverRate is the initial crossover probability.
	CrossoverRate float64 `json:"crossover_rate"`
}

// ---------------------------------------------------------------------------
// GenomeHandler
// ---------------------------------------------------------------------------

// GenomeHandler handles all /genome/* REST endpoints.
type GenomeHandler struct {
	engine  EvolutionEngine
	lineage *analysis.GenomeLineage
	mu      sync.RWMutex
}

// NewGenomeHandler constructs a GenomeHandler.
// lineage may be nil if lineage tracking is not configured.
func NewGenomeHandler(engine EvolutionEngine, lineage *analysis.GenomeLineage) *GenomeHandler {
	return &GenomeHandler{
		engine:  engine,
		lineage: lineage,
	}
}

// ---------------------------------------------------------------------------
// GET /genome/best
// ---------------------------------------------------------------------------

// GetBest returns the current best individual with its full fitness breakdown.
//
// Response 200:
//
//	{
//	  "id": "...",
//	  "generation": 42,
//	  "genes": [1.2, 3.4, ...],
//	  "fitness": { "sharpe": ..., "sortino": ..., ... },
//	  "parent_ids": ["...", "..."]
//	}
func (h *GenomeHandler) GetBest(w http.ResponseWriter, r *http.Request) {
	best, err := h.engine.BestIndividual()
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to get best individual: %v", err))
		return
	}
	writeJSON(w, http.StatusOK, individualResponse(best))
}

// ---------------------------------------------------------------------------
// GET /genome/population
// ---------------------------------------------------------------------------

// GetPopulation returns all individuals in the current generation with their
// fitness scores. Query params:
//   - evaluated_only=true  (default false) -- filter to only evaluated individuals
//
// Response 200:
//
//	{
//	  "individuals": [...],
//	  "count": 50,
//	  "generation": 42
//	}
func (h *GenomeHandler) GetPopulation(w http.ResponseWriter, r *http.Request) {
	pop, err := h.engine.Population()
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to get population: %v", err))
		return
	}

	evaluatedOnly := r.URL.Query().Get("evaluated_only") == "true"
	responses := make([]map[string]interface{}, 0, len(pop))
	for _, ind := range pop {
		if evaluatedOnly && !ind.Evaluated {
			continue
		}
		responses = append(responses, individualResponse(ind))
	}

	gen := 0
	if len(pop) > 0 {
		gen = pop[0].Generation
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"individuals": responses,
		"count":       len(responses),
		"generation":  gen,
	})
}

// ---------------------------------------------------------------------------
// POST /genome/seed
// ---------------------------------------------------------------------------

// seedRequest is the body for POST /genome/seed.
type seedRequest struct {
	// Genes is the float64 parameter vector to inject.
	Genes []float64 `json:"genes"`
}

// SeedGenome injects a genome provided by the caller into the next generation.
// This is used by the Python optimiser to warm-start evolution from a known
// good solution.
//
// Request body:
//
//	{ "genes": [1.2, 3.4, ...] }
//
// Response 200:
//
//	{ "message": "genome seeded", "gene_count": 42 }
func (h *GenomeHandler) SeedGenome(w http.ResponseWriter, r *http.Request) {
	var req seedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid request body: %v", err))
		return
	}
	if len(req.Genes) == 0 {
		writeError(w, http.StatusBadRequest, "genes must not be empty")
		return
	}

	genome := evolution.Genome(req.Genes)
	if err := h.engine.SeedIndividual(genome); err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to seed genome: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"message":    "genome seeded",
		"gene_count": len(req.Genes),
	})
}

// ---------------------------------------------------------------------------
// GET /genome/stats
// ---------------------------------------------------------------------------

// GetStats returns summary statistics for the current or most recent run.
//
// Response 200:
//
//	{
//	  "generation": 42,
//	  "diversity": 0.34,
//	  "convergence_rate": 0.002,
//	  "is_plateaued": false,
//	  "best_fitness_history": [1.1, 1.2, ...],
//	  "population_size": 50,
//	  "running": true
//	}
func (h *GenomeHandler) GetStats(w http.ResponseWriter, r *http.Request) {
	stats, err := h.engine.Stats()
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to get stats: %v", err))
		return
	}
	writeJSON(w, http.StatusOK, stats)
}

// ---------------------------------------------------------------------------
// GET /genome/lineage
// ---------------------------------------------------------------------------

// GetLineage returns the lineage graph for the current run in a JSON
// representation suitable for rendering as a DAG in the front-end.
//
// Query params:
//   - breakthroughs_only=true (default false)
//   - generation=N  (filter to nodes from generation N)
//
// Response 200:
//
//	{
//	  "nodes": [{ "id": "...", "generation": N, "fitness_score": 1.2, ... }],
//	  "node_count": 1234,
//	  "breakthroughs": [...]
//	}
func (h *GenomeHandler) GetLineage(w http.ResponseWriter, r *http.Request) {
	if h.lineage == nil {
		writeError(w, http.StatusServiceUnavailable, "lineage tracking not configured")
		return
	}

	q := r.URL.Query()
	breakthroughsOnly := q.Get("breakthroughs_only") == "true"
	genFilter := -1
	if s := q.Get("generation"); s != "" {
		if n, err := strconv.Atoi(s); err == nil {
			genFilter = n
		}
	}

	allNodes := h.lineage.Graph().AllNodes()
	filtered := make([]analysis.LineageNode, 0, len(allNodes))
	for _, n := range allNodes {
		if genFilter >= 0 && n.Generation != genFilter {
			continue
		}
		if breakthroughsOnly && !n.IsBreakthrough {
			continue
		}
		filtered = append(filtered, n)
	}

	// Identify breakthroughs inline.
	breakthroughs := make([]analysis.LineageNode, 0)
	for _, n := range filtered {
		if n.IsBreakthrough {
			breakthroughs = append(breakthroughs, n)
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"nodes":         filtered,
		"node_count":    len(filtered),
		"breakthroughs": breakthroughs,
	})
}

// ---------------------------------------------------------------------------
// POST /genome/evolve/start
// ---------------------------------------------------------------------------

// StartEvolution begins a new evolution run with the provided configuration.
// If a run is already active, HTTP 409 Conflict is returned.
//
// Request body: RunConfig JSON (all fields optional; defaults apply).
//
// Response 200:
//
//	{ "message": "evolution started", "config": { ... } }
func (h *GenomeHandler) StartEvolution(w http.ResponseWriter, r *http.Request) {
	if h.engine.IsRunning() {
		writeError(w, http.StatusConflict, "evolution run already active")
		return
	}

	var cfg RunConfig
	// Apply defaults before decoding.
	cfg.PopulationSize = 50
	cfg.MaxGenerations = 0
	cfg.NumWorkers = 4
	cfg.EliteRatio = 0.10
	cfg.MutationRate = 0.15
	cfg.CrossoverRate = 0.75

	if r.Body != nil && r.ContentLength != 0 {
		if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid run config: %v", err))
			return
		}
	}

	if err := validateRunConfig(cfg); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	if err := h.engine.StartRun(cfg); err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to start evolution: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"message": "evolution started",
		"config":  cfg,
	})
}

// validateRunConfig checks that the RunConfig fields are within acceptable
// ranges.
func validateRunConfig(cfg RunConfig) error {
	if cfg.PopulationSize < 2 {
		return fmt.Errorf("population_size must be >= 2, got %d", cfg.PopulationSize)
	}
	if cfg.NumWorkers < 1 {
		return fmt.Errorf("num_workers must be >= 1, got %d", cfg.NumWorkers)
	}
	if cfg.EliteRatio < 0 || cfg.EliteRatio > 0.5 {
		return fmt.Errorf("elite_ratio must be in [0, 0.5], got %f", cfg.EliteRatio)
	}
	if cfg.MutationRate < 0 || cfg.MutationRate > 1 {
		return fmt.Errorf("mutation_rate must be in [0, 1], got %f", cfg.MutationRate)
	}
	if cfg.CrossoverRate < 0 || cfg.CrossoverRate > 1 {
		return fmt.Errorf("crossover_rate must be in [0, 1], got %f", cfg.CrossoverRate)
	}
	return nil
}

// ---------------------------------------------------------------------------
// POST /genome/evolve/stop
// ---------------------------------------------------------------------------

// StopEvolution gracefully halts the current evolution run. If no run is
// active, HTTP 409 Conflict is returned.
//
// Response 200:
//
//	{ "message": "evolution stopping" }
func (h *GenomeHandler) StopEvolution(w http.ResponseWriter, r *http.Request) {
	if !h.engine.IsRunning() {
		writeError(w, http.StatusConflict, "no evolution run is active")
		return
	}
	if err := h.engine.StopRun(); err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to stop evolution: %v", err))
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"message": "evolution stopping",
	})
}

// ---------------------------------------------------------------------------
// Response helpers
// ---------------------------------------------------------------------------

// individualResponse converts an Individual to a JSON-serialisable map.
func individualResponse(ind evolution.Individual) map[string]interface{} {
	return map[string]interface{}{
		"id":         ind.ID,
		"generation": ind.Generation,
		"genes":      []float64(ind.Genes),
		"fitness":    ind.Fitness,
		"evaluated":  ind.Evaluated,
		"parent_ids": ind.ParentIDs,
	}
}

// writeJSON encodes v as JSON and writes it to w with the given status code.
func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// writeError writes a JSON error response with the given status and message.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]interface{}{
		"error": msg,
		"code":  status,
	})
}

// ---------------------------------------------------------------------------
// NoOpEngine -- safe default for tests and dev mode
// ---------------------------------------------------------------------------

// NoOpEngine is a placeholder EvolutionEngine that returns sensible zero
// values. Useful for tests and for running the API without a live evolution
// backend.
type NoOpEngine struct {
	mu      sync.RWMutex
	running bool
	pop     []evolution.Individual
	seeded  []evolution.Individual
}

// BestIndividual returns the first individual if any exist, otherwise an
// empty Individual.
func (e *NoOpEngine) BestIndividual() (evolution.Individual, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if len(e.pop) == 0 {
		return evolution.Individual{}, nil
	}
	best := e.pop[0]
	for _, ind := range e.pop[1:] {
		if ind.Fitness.WeightedScore > best.Fitness.WeightedScore {
			best = ind
		}
	}
	return best, nil
}

// Population returns a shallow copy of the current population.
func (e *NoOpEngine) Population() ([]evolution.Individual, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]evolution.Individual, len(e.pop))
	copy(out, e.pop)
	return out, nil
}

// SeedIndividual appends a new individual to the population.
func (e *NoOpEngine) SeedIndividual(genes evolution.Genome) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	ind := evolution.Individual{
		Genes:      genes,
		Generation: 0,
	}
	e.pop = append(e.pop, ind)
	e.seeded = append(e.seeded, ind)
	return nil
}

// Stats returns zero-value stats.
func (e *NoOpEngine) Stats() (EvolutionStats, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return EvolutionStats{
		PopulationSize: len(e.pop),
		Running:        e.running,
	}, nil
}

// StartRun marks the engine as running.
func (e *NoOpEngine) StartRun(_ RunConfig) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.running {
		return fmt.Errorf("already running")
	}
	e.running = true
	return nil
}

// StopRun marks the engine as stopped.
func (e *NoOpEngine) StopRun() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.running = false
	return nil
}

// IsRunning returns the running state.
func (e *NoOpEngine) IsRunning() bool {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.running
}

// ---------------------------------------------------------------------------
// RegisterRoutes -- wires all handlers into an http.ServeMux or chi router
// ---------------------------------------------------------------------------

// RegisterRoutes registers all /genome/* routes onto mux. This expects a
// *http.ServeMux; for chi routers use the individual handler methods directly.
func RegisterRoutes(mux *http.ServeMux, h *GenomeHandler) {
	mux.HandleFunc("/genome/best", methodGuard(http.MethodGet, h.GetBest))
	mux.HandleFunc("/genome/population", methodGuard(http.MethodGet, h.GetPopulation))
	mux.HandleFunc("/genome/seed", methodGuard(http.MethodPost, h.SeedGenome))
	mux.HandleFunc("/genome/stats", methodGuard(http.MethodGet, h.GetStats))
	mux.HandleFunc("/genome/lineage", methodGuard(http.MethodGet, h.GetLineage))
	mux.HandleFunc("/genome/evolve/start", methodGuard(http.MethodPost, h.StartEvolution))
	mux.HandleFunc("/genome/evolve/stop", methodGuard(http.MethodPost, h.StopEvolution))
}

// methodGuard wraps a handler to return 405 Method Not Allowed when the
// request method does not match the expected method.
func methodGuard(method string, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			writeError(w, http.StatusMethodNotAllowed,
				fmt.Sprintf("method %s not allowed; expected %s", r.Method, method))
			return
		}
		h(w, r)
	}
}

// WithRequestTimeout wraps a handler to enforce a per-request context timeout.
func WithRequestTimeout(timeout time.Duration, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), timeout)
		defer cancel()
		h(w, r.WithContext(ctx))
	}
}
