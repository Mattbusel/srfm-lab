package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go.uber.org/zap"
)

// MCResult represents a parsed Monte Carlo result file.
type MCResult struct {
	RunID       string  `json:"run_id"`
	ComputedAt  string  `json:"computed_at"`
	NSims       int64   `json:"n_sims"`
	Horizon     int64   `json:"horizon_days"`
	MedianFinal float64 `json:"median_final"`
	P5Final     float64 `json:"p5_final"`
	P25Final    float64 `json:"p25_final"`
	P75Final    float64 `json:"p75_final"`
	P95Final    float64 `json:"p95_final"`
	ProbProfit  float64 `json:"prob_profit"`
	MaxDD5      float64 `json:"max_dd_p5"`  // 5th percentile max drawdown (worst)
	MaxDD50     float64 `json:"max_dd_p50"`
	SharpeMedian float64 `json:"sharpe_median"`
	FileName    string  `json:"file_name"`
}

// MCRunRequest is the request body for POST /api/v1/mc/run.
type MCRunRequest struct {
	StrategyRunID int64  `json:"strategy_run_id"`
	NSims         int    `json:"n_sims"`
	HorizonDays   int    `json:"horizon_days"`
	Seed          int64  `json:"seed"`
}

// MCHandler serves Monte Carlo simulation results and can trigger new runs.
type MCHandler struct {
	resultDir  string // directory where mc.py writes JSON results
	scriptPath string // path to mc.py
	log        *zap.Logger
}

// NewMCHandler creates an MCHandler.
// resultDir is the directory containing MC result JSON files (e.g. research/mc_results/).
// scriptPath is the path to the Python mc.py script.
func NewMCHandler(resultDir, scriptPath string, log *zap.Logger) *MCHandler {
	return &MCHandler{
		resultDir:  resultDir,
		scriptPath: scriptPath,
		log:        log,
	}
}

// GetLatestMC handles GET /api/v1/mc/latest
// Reads the most recent MC result JSON file from resultDir.
func (h *MCHandler) GetLatestMC(w http.ResponseWriter, r *http.Request) {
	data, fname, err := h.readLatestResult()
	if err != nil {
		writeError(w, http.StatusNotFound, "no MC results found: "+err.Error())
		return
	}

	// Inject the file name into the response.
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		// Fall back to serving the raw JSON if it doesn't match our schema.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(data)
		return
	}
	raw["file_name"] = filepath.Base(fname)

	writeJSON(w, http.StatusOK, raw)
}

// GetMCHistory handles GET /api/v1/mc/history?limit=
// Returns metadata for all MC result files in resultDir, newest first.
func (h *MCHandler) GetMCHistory(w http.ResponseWriter, r *http.Request) {
	limit := 50
	if ls := r.URL.Query().Get("limit"); ls != "" {
		var n int
		if _, err := fmt.Sscanf(ls, "%d", &n); err == nil && n > 0 && n <= 500 {
			limit = n
		}
	}

	entries, err := h.listResultFiles()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "list results: "+err.Error())
		return
	}

	// Sort newest first (files are named with timestamps so lexicographic = chronological).
	sort.Slice(entries, func(i, j int) bool { return entries[i] > entries[j] })
	if len(entries) > limit {
		entries = entries[:limit]
	}

	results := make([]map[string]any, 0, len(entries))
	for _, fname := range entries {
		data, err := os.ReadFile(filepath.Join(h.resultDir, fname))
		if err != nil {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal(data, &obj); err != nil {
			continue
		}
		obj["file_name"] = fname
		results = append(results, obj)
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"results": results,
		"count":   len(results),
	})
}

// RunMC handles POST /api/v1/mc/run
// Spawns the Python mc.py script and streams its stdout output as newline-delimited
// log lines until completion, then returns the result JSON.
func (h *MCHandler) RunMC(w http.ResponseWriter, r *http.Request) {
	var req MCRunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err.Error() != "EOF" {
		writeError(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return
	}

	if h.scriptPath == "" {
		writeError(w, http.StatusServiceUnavailable, "mc script path not configured")
		return
	}
	if _, err := os.Stat(h.scriptPath); os.IsNotExist(err) {
		writeError(w, http.StatusServiceUnavailable, "mc.py not found at "+h.scriptPath)
		return
	}

	// Build args.
	args := []string{h.scriptPath}
	if req.StrategyRunID > 0 {
		args = append(args, fmt.Sprintf("--run-id=%d", req.StrategyRunID))
	}
	if req.NSims > 0 {
		args = append(args, fmt.Sprintf("--n-sims=%d", req.NSims))
	} else {
		args = append(args, "--n-sims=1000")
	}
	if req.HorizonDays > 0 {
		args = append(args, fmt.Sprintf("--horizon=%d", req.HorizonDays))
	}
	if req.Seed != 0 {
		args = append(args, fmt.Sprintf("--seed=%d", req.Seed))
	}
	if h.resultDir != "" {
		args = append(args, fmt.Sprintf("--output-dir=%s", h.resultDir))
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()

	h.log.Info("spawning mc.py", zap.Strings("args", args))

	cmd := exec.CommandContext(ctx, "python3", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "pipe: "+err.Error())
		return
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "pipe: "+err.Error())
		return
	}

	if err := cmd.Start(); err != nil {
		writeError(w, http.StatusInternalServerError, "start mc.py: "+err.Error())
		return
	}

	// Stream stdout as Server-Sent Events style text/plain while the process runs.
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(http.StatusOK)

	flusher, canFlush := w.(http.Flusher)

	scanAndStream := func(scanner *bufio.Scanner, prefix string) {
		for scanner.Scan() {
			line := scanner.Text()
			_, _ = fmt.Fprintf(w, "%s%s\n", prefix, line)
			if canFlush {
				flusher.Flush()
			}
			h.log.Debug("mc.py output", zap.String("line", prefix+line))
		}
	}

	go scanAndStream(bufio.NewScanner(stderr), "[stderr] ")
	scanAndStream(bufio.NewScanner(stdout), "")

	if err := cmd.Wait(); err != nil {
		_, _ = fmt.Fprintf(w, "\nERROR: mc.py exited: %v\n", err)
		if canFlush {
			flusher.Flush()
		}
		return
	}

	// Append the result JSON at the end of the stream.
	data, fname, err := h.readLatestResult()
	if err == nil {
		_, _ = fmt.Fprintf(w, "\nRESULT_FILE: %s\n", filepath.Base(fname))
		_, _ = w.Write(data)
		_, _ = fmt.Fprintln(w)
	}
	if canFlush {
		flusher.Flush()
	}
}

// readLatestResult returns the raw bytes and path of the newest MC result JSON.
func (h *MCHandler) readLatestResult() ([]byte, string, error) {
	files, err := h.listResultFiles()
	if err != nil || len(files) == 0 {
		return nil, "", fmt.Errorf("no MC result files in %s", h.resultDir)
	}
	sort.Slice(files, func(i, j int) bool { return files[i] > files[j] })
	path := filepath.Join(h.resultDir, files[0])
	data, err := os.ReadFile(path)
	return data, path, err
}

// listResultFiles returns base file names matching mc_*.json in resultDir.
func (h *MCHandler) listResultFiles() ([]string, error) {
	matches, err := filepath.Glob(filepath.Join(h.resultDir, "mc_*.json"))
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(matches))
	for _, m := range matches {
		names = append(names, filepath.Base(m))
	}
	// Also accept result_*.json naming.
	alt, _ := filepath.Glob(filepath.Join(h.resultDir, "result_*.json"))
	for _, m := range alt {
		names = append(names, filepath.Base(m))
	}
	// Deduplicate.
	seen := make(map[string]bool, len(names))
	out := names[:0]
	for _, n := range names {
		if !seen[n] && strings.HasSuffix(n, ".json") {
			seen[n] = true
			out = append(out, n)
		}
	}
	return out, nil
}
