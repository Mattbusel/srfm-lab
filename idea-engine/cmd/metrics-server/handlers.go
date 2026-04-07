package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// ---------------------------------------------------------------------------
// MetricsHandler -- HTTP handler struct
// ---------------------------------------------------------------------------

// MetricsHandler holds the shared IAEMetrics store and serves all HTTP routes.
type MetricsHandler struct {
	store *IAEMetrics
	// startTime records when the handler was constructed, used in /health.
	startTime time.Time
}

// NewMetricsHandler constructs a MetricsHandler backed by store.
func NewMetricsHandler(store *IAEMetrics) *MetricsHandler {
	return &MetricsHandler{
		store:     store,
		startTime: time.Now(),
	}
}

// ---------------------------------------------------------------------------
// GET /metrics -- Prometheus text format
// ---------------------------------------------------------------------------

// HandleMetrics returns an OpenMetrics-compatible Prometheus scrape payload.
// All IAE metrics are prefixed with iae_.
func (h *MetricsHandler) HandleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}

	latest, hasGenome := h.store.LatestGenome()
	totalEvals := h.store.TotalEvals()
	evalsPerHour := h.store.EvaluationsPerHour()
	meanEvalMs := h.store.MeanEvalTimeMs()
	meanSharpe := h.store.MeanSharpe()
	fitnessImprov50 := h.store.FitnessImprovement(50)

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)

	uptime := time.Since(h.startTime).Seconds()
	fmt.Fprintf(w, "# HELP iae_uptime_seconds Seconds since the metrics-server started\n")
	fmt.Fprintf(w, "# TYPE iae_uptime_seconds gauge\n")
	fmt.Fprintf(w, "iae_uptime_seconds %.3f\n\n", uptime)

	fmt.Fprintf(w, "# HELP iae_total_evaluations Lifetime count of fitness evaluations\n")
	fmt.Fprintf(w, "# TYPE iae_total_evaluations counter\n")
	fmt.Fprintf(w, "iae_total_evaluations %d\n\n", totalEvals)

	fmt.Fprintf(w, "# HELP iae_evaluations_per_hour Rolling evaluations per hour\n")
	fmt.Fprintf(w, "# TYPE iae_evaluations_per_hour gauge\n")
	fmt.Fprintf(w, "iae_evaluations_per_hour %.4f\n\n", evalsPerHour)

	fmt.Fprintf(w, "# HELP iae_mean_eval_time_ms Mean evaluation time in milliseconds\n")
	fmt.Fprintf(w, "# TYPE iae_mean_eval_time_ms gauge\n")
	fmt.Fprintf(w, "iae_mean_eval_time_ms %.2f\n\n", meanEvalMs)

	fmt.Fprintf(w, "# HELP iae_mean_sharpe Mean Sharpe ratio across recent evaluations\n")
	fmt.Fprintf(w, "# TYPE iae_mean_sharpe gauge\n")
	fmt.Fprintf(w, "iae_mean_sharpe %.6f\n\n", meanSharpe)

	fmt.Fprintf(w, "# HELP iae_fitness_improvement_50gen Fitness improvement over last 50 generations\n")
	fmt.Fprintf(w, "# TYPE iae_fitness_improvement_50gen gauge\n")
	fmt.Fprintf(w, "iae_fitness_improvement_50gen %.6f\n\n", fitnessImprov50)

	if hasGenome {
		fmt.Fprintf(w, "# HELP iae_current_generation Current evolution generation number\n")
		fmt.Fprintf(w, "# TYPE iae_current_generation gauge\n")
		fmt.Fprintf(w, "iae_current_generation %d\n\n", latest.Generation)

		fmt.Fprintf(w, "# HELP iae_best_fitness Best individual fitness in the current generation\n")
		fmt.Fprintf(w, "# TYPE iae_best_fitness gauge\n")
		fmt.Fprintf(w, "iae_best_fitness %.6f\n\n", latest.BestFitness)

		fmt.Fprintf(w, "# HELP iae_mean_fitness Mean fitness across the current population\n")
		fmt.Fprintf(w, "# TYPE iae_mean_fitness gauge\n")
		fmt.Fprintf(w, "iae_mean_fitness %.6f\n\n", latest.MeanFitness)

		fmt.Fprintf(w, "# HELP iae_diversity Normalised population diversity\n")
		fmt.Fprintf(w, "# TYPE iae_diversity gauge\n")
		fmt.Fprintf(w, "iae_diversity %.6f\n\n", latest.Diversity)
	}
}

// ---------------------------------------------------------------------------
// GET /metrics/evolution -- JSON evolution stats
// ---------------------------------------------------------------------------

// evolutionResponse is the JSON body for GET /metrics/evolution.
type evolutionResponse struct {
	CurrentGeneration int            `json:"current_generation"`
	BestFitness       float64        `json:"best_fitness"`
	MeanFitness       float64        `json:"mean_fitness"`
	Diversity         float64        `json:"diversity"`
	CumulativeEvals   int64          `json:"cumulative_evals"`
	FitnessImprov10   float64        `json:"fitness_improvement_10gen"`
	FitnessImprov50   float64        `json:"fitness_improvement_50gen"`
	RecentGenerations []GenomeMetric `json:"recent_generations"`
	Timestamp         time.Time      `json:"timestamp"`
}

// HandleEvolution returns current generation statistics as JSON.
func (h *MetricsHandler) HandleEvolution(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}

	latest, hasGenome := h.store.LatestGenome()
	resp := evolutionResponse{
		CumulativeEvals: h.store.TotalEvals(),
		FitnessImprov10: h.store.FitnessImprovement(10),
		FitnessImprov50: h.store.FitnessImprovement(50),
		RecentGenerations: h.store.RecentGenomes(20),
		Timestamp:       time.Now(),
	}
	if hasGenome {
		resp.CurrentGeneration = latest.Generation
		resp.BestFitness = latest.BestFitness
		resp.MeanFitness = latest.MeanFitness
		resp.Diversity = latest.Diversity
	}
	writeMetricsJSON(w, http.StatusOK, resp)
}

// ---------------------------------------------------------------------------
// GET /metrics/params -- recent parameter updates
// ---------------------------------------------------------------------------

// paramsResponse is the JSON body for GET /metrics/params.
type paramsResponse struct {
	Count   int           `json:"count"`
	Params  []ParamMetric `json:"params"`
	Fetched time.Time     `json:"fetched_at"`
}

// HandleParams returns recent parameter update events.
// Query param: n (default 50, max 1000).
func (h *MetricsHandler) HandleParams(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}

	n := queryInt(r, "n", 50, 1, ringBufferSize)
	params := h.store.RecentParams(n)
	writeMetricsJSON(w, http.StatusOK, paramsResponse{
		Count:   len(params),
		Params:  params,
		Fetched: time.Now(),
	})
}

// ---------------------------------------------------------------------------
// GET /metrics/evaluations -- throughput stats
// ---------------------------------------------------------------------------

// evaluationsResponse is the JSON body for GET /metrics/evaluations.
type evaluationsResponse struct {
	TotalEvals     int64      `json:"total_evals"`
	EvalsPerHour   float64    `json:"evals_per_hour"`
	MeanEvalTimeMs float64    `json:"mean_eval_time_ms"`
	MeanSharpe     float64    `json:"mean_sharpe"`
	RecentEvals    []EvalMetric `json:"recent_evals"`
	Fetched        time.Time  `json:"fetched_at"`
}

// HandleEvaluations returns evaluation throughput statistics.
func (h *MetricsHandler) HandleEvaluations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}

	recent := h.store.RecentEvals()
	// Cap the returned slice at 100 entries to keep payloads manageable.
	if len(recent) > 100 {
		recent = recent[len(recent)-100:]
	}

	writeMetricsJSON(w, http.StatusOK, evaluationsResponse{
		TotalEvals:     h.store.TotalEvals(),
		EvalsPerHour:   h.store.EvaluationsPerHour(),
		MeanEvalTimeMs: h.store.MeanEvalTimeMs(),
		MeanSharpe:     h.store.MeanSharpe(),
		RecentEvals:    recent,
		Fetched:        time.Now(),
	})
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

// healthResponse is the JSON body for GET /health.
type healthResponse struct {
	Status      string    `json:"status"`
	UptimeS     float64   `json:"uptime_seconds"`
	TotalEvals  int64     `json:"total_evals"`
	Timestamp   time.Time `json:"timestamp"`
}

// HandleHealth returns a liveness check payload.
func (h *MetricsHandler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	writeMetricsJSON(w, http.StatusOK, healthResponse{
		Status:     "ok",
		UptimeS:    time.Since(h.startTime).Seconds(),
		TotalEvals: h.store.TotalEvals(),
		Timestamp:  time.Now(),
	})
}

// ---------------------------------------------------------------------------
// Ingest endpoints -- called by IAE services to push metrics
// ---------------------------------------------------------------------------

// HandleIngestGenome accepts a GenomeMetric via POST and records it.
func (h *MetricsHandler) HandleIngestGenome(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	var gm GenomeMetric
	if err := json.NewDecoder(r.Body).Decode(&gm); err != nil {
		writeMetricsError(w, http.StatusBadRequest, fmt.Sprintf("invalid body: %v", err))
		return
	}
	h.store.RecordGenome(gm)
	writeMetricsJSON(w, http.StatusOK, map[string]string{"status": "recorded"})
}

// HandleIngestParam accepts a ParamMetric via POST and records it.
func (h *MetricsHandler) HandleIngestParam(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	var pm ParamMetric
	if err := json.NewDecoder(r.Body).Decode(&pm); err != nil {
		writeMetricsError(w, http.StatusBadRequest, fmt.Sprintf("invalid body: %v", err))
		return
	}
	h.store.RecordParamUpdate(pm)
	writeMetricsJSON(w, http.StatusOK, map[string]string{"status": "recorded"})
}

// ingestEvalRequest is the body for POST /ingest/evaluation.
type ingestEvalRequest struct {
	Sharpe     float64 `json:"sharpe"`
	MaxDD      float64 `json:"max_drawdown"`
	EvalTimeMs int64   `json:"eval_time_ms"`
}

// HandleIngestEvaluation accepts a backtest evaluation result via POST.
func (h *MetricsHandler) HandleIngestEvaluation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	var req ingestEvalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeMetricsError(w, http.StatusBadRequest, fmt.Sprintf("invalid body: %v", err))
		return
	}
	h.store.RecordEvaluation(req.Sharpe, req.MaxDD, req.EvalTimeMs)
	writeMetricsJSON(w, http.StatusOK, map[string]string{"status": "recorded"})
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// writeMetricsJSON writes v as indented JSON with the given status code.
func writeMetricsJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// writeMetricsError writes a JSON error body.
func writeMetricsError(w http.ResponseWriter, status int, msg string) {
	writeMetricsJSON(w, status, map[string]interface{}{
		"error": msg,
		"code":  status,
	})
}

// methodNotAllowed writes a 405 response.
func methodNotAllowed(w http.ResponseWriter, allowed string) {
	w.Header().Set("Allow", allowed)
	writeMetricsError(w, http.StatusMethodNotAllowed,
		fmt.Sprintf("method not allowed; use %s", allowed))
}

// queryInt reads an integer query parameter, clamping to [min, max].
// Returns def if the parameter is absent or unparseable.
func queryInt(r *http.Request, key string, def, min, max int) int {
	s := r.URL.Query().Get(key)
	if s == "" {
		return def
	}
	var n int
	if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
		return def
	}
	if n < min {
		return min
	}
	if n > max {
		return max
	}
	return n
}
