package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// =============================================================================
// main.go — HTTP API for the backtest farm job queue
// Listens on :11439, stdlib only, no external dependencies.
// =============================================================================

const (
	listenAddr     = ":11439"
	defaultWorkers = 8
	defaultTimeout = 300 // seconds
)

// ---- request / response helpers -------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		log.Printf("writeJSON encode error: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func readBody(r *http.Request, dst interface{}) error {
	defer r.Body.Close()
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}

func pathParam(path, prefix string) string {
	s := strings.TrimPrefix(path, prefix)
	s = strings.TrimPrefix(s, "/")
	if idx := strings.Index(s, "/"); idx >= 0 {
		return s[:idx]
	}
	return s
}

func pathParamTwo(path, prefix string) (string, string) {
	s := strings.TrimPrefix(path, prefix)
	s = strings.TrimPrefix(s, "/")
	parts := strings.SplitN(s, "/", 3)
	if len(parts) >= 2 {
		return parts[0], parts[1]
	}
	if len(parts) == 1 {
		return parts[0], ""
	}
	return "", ""
}

// ---- multiplexer ----------------------------------------------------------

type farmMux struct {
	engine   *FarmEngine
	analyzer *ResultAnalyzer
}

func newFarmMux(engine *FarmEngine) *farmMux {
	return &farmMux{
		engine:   engine,
		analyzer: NewResultAnalyzer(engine),
	}
}

func (m *farmMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	// health
	if path == "/health" && r.Method == http.MethodGet {
		m.handleHealth(w, r)
		return
	}

	// submit
	if path == "/farm/submit" && r.Method == http.MethodPost {
		m.handleSubmit(w, r)
		return
	}
	if path == "/farm/submit-grid" && r.Method == http.MethodPost {
		m.handleSubmitGrid(w, r)
		return
	}

	// status / statistics
	if path == "/farm/status" && r.Method == http.MethodGet {
		m.handleStatus(w, r)
		return
	}
	if path == "/farm/statistics" && r.Method == http.MethodGet {
		m.handleStatistics(w, r)
		return
	}

	// jobs listing
	if path == "/farm/jobs" && r.Method == http.MethodGet {
		m.handleJobsList(w, r)
		return
	}

	// specific job
	if strings.HasPrefix(path, "/farm/jobs/") && r.Method == http.MethodGet {
		m.handleJobGet(w, r)
		return
	}

	// results
	if path == "/farm/results" && r.Method == http.MethodGet {
		m.handleResults(w, r)
		return
	}

	// landscape
	if strings.HasPrefix(path, "/farm/landscape/") && r.Method == http.MethodGet {
		m.handleLandscape(w, r)
		return
	}

	// cancel
	if strings.HasPrefix(path, "/farm/cancel-all") && r.Method == http.MethodPost {
		m.handleCancelAll(w, r)
		return
	}
	if strings.HasPrefix(path, "/farm/cancel/") && r.Method == http.MethodPost {
		m.handleCancel(w, r)
		return
	}

	// purge
	if path == "/farm/purge" && r.Method == http.MethodDelete {
		m.handlePurge(w, r)
		return
	}

	writeError(w, http.StatusNotFound, "not found")
}

// ---- handler: health ------------------------------------------------------

func (m *farmMux) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"workers":   m.engine.Config.MaxWorkers,
		"uptime_s":  int(time.Since(m.engine.startedAt).Seconds()),
	})
}

// ---- handler: submit batch ------------------------------------------------

func (m *farmMux) handleSubmit(w http.ResponseWriter, r *http.Request) {
	var configs []BacktestConfig
	if err := readBody(r, &configs); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if len(configs) == 0 {
		writeError(w, http.StatusBadRequest, "empty config array")
		return
	}
	ids := make([]string, 0, len(configs))
	for i := range configs {
		if configs[i].Strategy == "" {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("config[%d]: strategy required", i))
			return
		}
		if configs[i].Symbol == "" {
			configs[i].Symbol = "BTCUSD"
		}
		if configs[i].Timeframe == "" {
			configs[i].Timeframe = "1h"
		}
		if configs[i].CostBPS == 0 {
			configs[i].CostBPS = 5.0
		}
		job := m.engine.Submit(configs[i])
		ids = append(ids, job.ID)
	}
	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"submitted": len(ids),
		"job_ids":   ids,
	})
}

// ---- handler: submit grid -------------------------------------------------

type GridSearchRequest struct {
	Strategy   string                    `json:"strategy"`
	Symbol     string                    `json:"symbol"`
	Timeframe  string                    `json:"timeframe"`
	CostBPS    float64                   `json:"cost_bps"`
	DateRange  string                    `json:"date_range"`
	ParamGrid  map[string][]interface{}  `json:"param_grid"`
	Priority   int                       `json:"priority"`
}

func (m *farmMux) handleSubmitGrid(w http.ResponseWriter, r *http.Request) {
	var req GridSearchRequest
	if err := readBody(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.Strategy == "" {
		writeError(w, http.StatusBadRequest, "strategy required")
		return
	}
	if len(req.ParamGrid) == 0 {
		writeError(w, http.StatusBadRequest, "param_grid required")
		return
	}
	if req.Symbol == "" {
		req.Symbol = "BTCUSD"
	}
	if req.Timeframe == "" {
		req.Timeframe = "1h"
	}
	if req.CostBPS == 0 {
		req.CostBPS = 5.0
	}

	gen := &GridSearchGenerator{}
	combos := gen.Generate(req.ParamGrid)

	ids := make([]string, 0, len(combos))
	for _, params := range combos {
		cfg := BacktestConfig{
			Strategy:   req.Strategy,
			Symbol:     req.Symbol,
			Timeframe:  req.Timeframe,
			Parameters: params,
			CostBPS:    req.CostBPS,
			DateRange:  req.DateRange,
			Priority:   req.Priority,
		}
		job := m.engine.Submit(cfg)
		ids = append(ids, job.ID)
	}
	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"submitted":   len(ids),
		"job_ids":     ids,
		"grid_points": len(combos),
	})
}

// ---- handler: farm status -------------------------------------------------

func (m *farmMux) handleStatus(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, m.engine.Status())
}

// ---- handler: list jobs ---------------------------------------------------

func (m *farmMux) handleJobsList(w http.ResponseWriter, r *http.Request) {
	status := r.URL.Query().Get("status")
	limitStr := r.URL.Query().Get("limit")
	limit := 100
	if limitStr != "" {
		if v, err := strconv.Atoi(limitStr); err == nil && v > 0 {
			limit = v
		}
	}
	jobs := m.engine.ListJobs(status, limit)
	writeJSON(w, http.StatusOK, jobs)
}

// ---- handler: get single job ----------------------------------------------

func (m *farmMux) handleJobGet(w http.ResponseWriter, r *http.Request) {
	id := pathParam(r.URL.Path, "/farm/jobs/")
	if id == "" {
		writeError(w, http.StatusBadRequest, "job id required")
		return
	}
	job, ok := m.engine.GetJob(id)
	if !ok {
		writeError(w, http.StatusNotFound, "job not found")
		return
	}
	writeJSON(w, http.StatusOK, job)
}

// ---- handler: results -----------------------------------------------------

func (m *farmMux) handleResults(w http.ResponseWriter, r *http.Request) {
	topStr := r.URL.Query().Get("top")
	top := 20
	if topStr != "" {
		if v, err := strconv.Atoi(topStr); err == nil && v > 0 {
			top = v
		}
	}
	sortBy := r.URL.Query().Get("sort")
	if sortBy == "" {
		sortBy = "sharpe"
	}
	results := m.engine.TopResults(top, sortBy)
	writeJSON(w, http.StatusOK, results)
}

// ---- handler: landscape ---------------------------------------------------

func (m *farmMux) handleLandscape(w http.ResponseWriter, r *http.Request) {
	strategy, symbol := pathParamTwo(r.URL.Path, "/farm/landscape/")
	if strategy == "" {
		writeError(w, http.StatusBadRequest, "strategy required")
		return
	}
	if symbol == "" {
		// alpha landscape for one strategy across all symbols
		landscape := m.engine.LandscapeForStrategy(strategy)
		writeJSON(w, http.StatusOK, landscape)
		return
	}
	// 2D heatmap: strategy + symbol
	heatmap := m.engine.HeatmapData(strategy, symbol)
	writeJSON(w, http.StatusOK, heatmap)
}

// ---- handler: cancel ------------------------------------------------------

func (m *farmMux) handleCancel(w http.ResponseWriter, r *http.Request) {
	id := pathParam(r.URL.Path, "/farm/cancel/")
	if id == "" {
		writeError(w, http.StatusBadRequest, "job id required")
		return
	}
	if m.engine.Cancel(id) {
		writeJSON(w, http.StatusOK, map[string]string{"cancelled": id})
	} else {
		writeError(w, http.StatusNotFound, "job not found or not cancellable")
	}
}

func (m *farmMux) handleCancelAll(w http.ResponseWriter, _ *http.Request) {
	n := m.engine.CancelAll()
	writeJSON(w, http.StatusOK, map[string]int{"cancelled": n})
}

// ---- handler: purge -------------------------------------------------------

func (m *farmMux) handlePurge(w http.ResponseWriter, _ *http.Request) {
	n := m.engine.Purge()
	writeJSON(w, http.StatusOK, map[string]int{"purged": n})
}

// ---- handler: statistics --------------------------------------------------

func (m *farmMux) handleStatistics(w http.ResponseWriter, _ *http.Request) {
	stats := m.analyzer.AggregateStatistics()
	writeJSON(w, http.StatusOK, stats)
}

// ---- main -----------------------------------------------------------------

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds | log.Lshortfile)
	log.Println("backtest-farm starting")

	cfg := FarmConfig{
		MaxWorkers:        defaultWorkers,
		JobTimeoutSec:     defaultTimeout,
		ResultRetentionS:  86400,
		NotificationURL:   os.Getenv("FARM_WEBHOOK_URL"),
	}
	if wStr := os.Getenv("FARM_WORKERS"); wStr != "" {
		if v, err := strconv.Atoi(wStr); err == nil && v > 0 {
			cfg.MaxWorkers = v
		}
	}

	engine := NewFarmEngine(cfg)
	engine.Start()

	mux := newFarmMux(engine)

	srv := &http.Server{
		Addr:         listenAddr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		log.Printf("listening on %s", listenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen error: %v", err)
		}
	}()

	<-sigCh
	log.Println("shutting down...")
	engine.Stop()
	log.Println("done")
}
