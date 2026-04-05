// Command scheduler is the experiment scheduler for the Idea Automation Engine.
// It manages a priority queue of pending experiments, enforces per-module CPU
// budgets, dispatches experiments to Rust or Python runtimes, and handles
// retries with exponential backoff.
//
// It also runs market-close and pattern-count triggers that initiate new
// experiment cycles automatically.
//
// Listens on :8769 by default (configurable via SCHEDULER_ADDR env var).
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chimiddleware "github.com/go-chi/chi/v5/middleware"
	_ "github.com/mattn/go-sqlite3"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"srfm-lab/idea-engine/scheduler"
	"srfm-lab/idea-engine/scheduler/dispatchers"
	"srfm-lab/idea-engine/scheduler/triggers"
)

const defaultListenAddr = ":8769"

type config struct {
	listenAddr     string
	dbPath         string
	busURL         string
	rustBinaryPath string
	pythonBin      string
	totalSlots     int
	logLevel       string
}

func main() {
	cfg := parseFlags()

	log := buildLogger(cfg.logLevel)
	defer log.Sync() //nolint:errcheck

	log.Info("starting idea-engine scheduler", zap.String("listen", cfg.listenAddr))

	if err := run(cfg, log); err != nil {
		log.Fatal("scheduler exited with error", zap.Error(err))
	}
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.listenAddr, "listen", envOr("SCHEDULER_ADDR", defaultListenAddr), "HTTP listen address")
	flag.StringVar(&cfg.dbPath, "db", envOr("IDEA_ENGINE_DB", "./idea_engine.db"), "Path to idea_engine.db")
	flag.StringVar(&cfg.busURL, "bus", envOr("BUS_URL", "http://localhost:8768"), "Bus service base URL")
	flag.StringVar(&cfg.rustBinaryPath, "rust-binary", envOr("GENOME_BINARY", "idea-genome-engine"), "Rust genome binary path")
	flag.StringVar(&cfg.pythonBin, "python", envOr("PYTHON_BIN", "python"), "Python interpreter")
	flag.IntVar(&cfg.totalSlots, "slots", 8, "Total parallel experiment slots")
	flag.StringVar(&cfg.logLevel, "log-level", envOr("LOG_LEVEL", "info"), "Log level")
	flag.Parse()
	return cfg
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func run(cfg config, log *zap.Logger) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// ---- Database ----
	dsn := fmt.Sprintf("file:%s?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=on&cache=shared", cfg.dbPath)
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return fmt.Errorf("open sqlite3: %w", err)
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(4)
	db.SetConnMaxLifetime(time.Hour)

	pingCtx, pingCancel := context.WithTimeout(ctx, 5*time.Second)
	defer pingCancel()
	if err := db.PingContext(pingCtx); err != nil {
		return fmt.Errorf("ping sqlite3: %w", err)
	}
	log.Info("scheduler: database connected", zap.String("path", cfg.dbPath))

	// ---- Scheduler components ----
	pq := scheduler.NewPriorityQueue()
	budget := scheduler.NewBudgetManager(cfg.totalSlots)
	retryPolicy := scheduler.NewRetryPolicy(log)
	retryQueue := scheduler.NewRetryQueue()

	lifecycle := scheduler.NewExperimentLifecycle(db, pq, budget, retryPolicy, retryQueue, log)

	// ---- Dispatchers ----
	rustDisp := dispatchers.NewRustDispatcher(cfg.rustBinaryPath, 2*time.Hour, "", log)
	pyDisp := dispatchers.NewPythonDispatcher(cfg.pythonBin, "idea_engine", 30*time.Minute, log)

	// Register genome experiments to the Rust dispatcher.
	lifecycle.RegisterDispatcher("genome", func(ctx context.Context, item *scheduler.ExperimentItem) (json.RawMessage, error) {
		return rustDisp.Dispatch(ctx, &dispatchers.DispatchItem{
			ExperimentID:   item.ExperimentID,
			HypothesisID:   item.HypothesisID,
			ExperimentType: item.ExperimentType,
			Config:         item.Config,
		})
	})

	// Register Python-backed experiment types.
	for _, expType := range []string{"counterfactual", "shadow", "causal", "academic", "serendipity"} {
		et := expType // capture loop var
		lifecycle.RegisterDispatcher(et, func(ctx context.Context, item *scheduler.ExperimentItem) (json.RawMessage, error) {
			return pyDisp.Dispatch(ctx, &dispatchers.DispatchItem{
				ExperimentID:   item.ExperimentID,
				HypothesisID:   item.HypothesisID,
				ExperimentType: item.ExperimentType,
				Config:         item.Config,
			})
		})
	}

	// ---- Load queued experiments from DB at startup (crash recovery) ----
	loadCtx, loadCancel := context.WithTimeout(ctx, 15*time.Second)
	n, err := lifecycle.EnqueueFromDB(loadCtx)
	loadCancel()
	if err != nil {
		log.Warn("scheduler: could not load queued experiments from DB", zap.Error(err))
	} else {
		log.Info("scheduler: loaded queued experiments from DB", zap.Int("count", n))
	}

	// ---- Retry queue drain goroutine ----
	doneCh := make(chan struct{})
	go retryQueue.Drain(func(item *scheduler.ExperimentItem) {
		pq.Push(item)
		log.Info("scheduler: retry experiment re-queued",
			zap.String("id", item.ExperimentID),
			zap.Int("retry_count", item.RetryCount),
		)
	}, doneCh)

	// ---- Market close trigger ----
	mct := triggers.NewMarketCloseTrigger(func(ctx context.Context, isWeekly bool) {
		log.Info("market_close_trigger: cycle starting", zap.Bool("is_weekly", isWeekly))
		// In a full implementation this would POST to the ingestion service.
		// For now it logs intent; the Python ingestion pipeline runs independently
		// and publishes events to the bus which the scheduler listens to.
		log.Info("market_close_trigger: cycle complete", zap.Bool("is_weekly", isWeekly))
	}, log)
	go mct.Run(ctx)

	// ---- Pattern trigger ----
	pt := triggers.NewPatternTrigger(func(ctx context.Context, patternIDs []string) {
		log.Info("pattern_trigger: hypothesis generation triggered",
			zap.Int("pattern_count", len(patternIDs)),
		)
		// POST to the hypothesis generator Python service.
		// (Implemented by the idea_engine Python stack via the bus.)
	}, cfg.busURL, 5, log)
	go pt.PollBus(ctx, 2*time.Minute)

	// ---- Main dispatch loop ----
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				for lifecycle.Dispatch(ctx) {
					// Keep dispatching until no more work or no budget.
				}
			}
		}
	}()

	// ---- HTTP API ----
	r := chi.NewRouter()
	r.Use(chimiddleware.RequestID)
	r.Use(chimiddleware.RealIP)
	r.Use(chimiddleware.Recoverer)
	r.Use(chimiddleware.Timeout(30 * time.Second))

	r.Route("/api/v1", func(r chi.Router) {
		r.Get("/health", healthHandler(db, budget, pq, log))
		r.Get("/stats", statsHandler(budget, pq, pt))
		r.Post("/dispatch", dispatchHandler(pq, log))
		r.Post("/trigger/market-close", func(w http.ResponseWriter, r *http.Request) {
			mct.FireNow(ctx)
			writeJSON(w, http.StatusAccepted, map[string]string{"status": "fired"})
		})
	})

	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "route not found"})
	})

	srv := &http.Server{
		Addr:         cfg.listenAddr,
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Info("scheduler listening", zap.String("addr", cfg.listenAddr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- fmt.Errorf("ListenAndServe: %w", err)
		}
		close(errCh)
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigCh:
		log.Info("received signal, shutting down", zap.String("signal", sig.String()))
	case err := <-errCh:
		if err != nil {
			close(doneCh)
			return err
		}
		close(doneCh)
		return nil
	case <-ctx.Done():
	}

	close(doneCh)
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", zap.Error(err))
		return err
	}
	log.Info("scheduler stopped cleanly")
	return nil
}

// dispatchHandler accepts an experiment from the idea-api and pushes it onto
// the priority queue.
func dispatchHandler(pq *scheduler.PriorityQueue, log *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

		var exp struct {
			ExperimentID   string          `json:"experiment_id"`
			HypothesisID   string          `json:"hypothesis_id"`
			ExperimentType string          `json:"experiment_type"`
			Priority       int             `json:"priority"`
			Config         json.RawMessage `json:"config"`
		}
		if err := json.NewDecoder(r.Body).Decode(&exp); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
			return
		}
		if exp.ExperimentID == "" || exp.ExperimentType == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{
				"error": "experiment_id and experiment_type are required",
			})
			return
		}

		item := &scheduler.ExperimentItem{
			ExperimentID:   exp.ExperimentID,
			HypothesisID:   exp.HypothesisID,
			ExperimentType: exp.ExperimentType,
			Priority:       exp.Priority,
			Config:         exp.Config,
			EnqueuedAt:     time.Now().UTC(),
		}
		pq.Push(item)

		log.Info("scheduler: experiment enqueued via API",
			zap.String("id", exp.ExperimentID),
			zap.String("type", exp.ExperimentType),
			zap.Int("priority", exp.Priority),
			zap.Int("queue_len", pq.Len()),
		)

		writeJSON(w, http.StatusAccepted, map[string]interface{}{
			"experiment_id": exp.ExperimentID,
			"queue_len":     pq.Len(),
			"enqueued_at":   time.Now().UTC().Format(time.RFC3339),
		})
	}
}

// healthHandler returns JSON health with DB ping, budget stats, and queue depth.
func healthHandler(db *sql.DB, budget *scheduler.BudgetManager, pq *scheduler.PriorityQueue, log *zap.Logger) http.HandlerFunc {
	start := time.Now()
	return func(w http.ResponseWriter, r *http.Request) {
		dbStatus := "ok"
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := db.PingContext(ctx); err != nil {
			dbStatus = "error: " + err.Error()
			log.Warn("health check db ping failed", zap.Error(err))
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"status":        "healthy",
			"service":       "idea-engine-scheduler",
			"version":       "0.1.0",
			"uptime_s":      time.Since(start).Seconds(),
			"timestamp":     time.Now().UTC().Format(time.RFC3339),
			"db":            dbStatus,
			"queue_depth":   pq.Len(),
			"total_running": budget.TotalRunning(),
		})
	}
}

// statsHandler returns detailed scheduling metrics.
func statsHandler(budget *scheduler.BudgetManager, pq *scheduler.PriorityQueue, pt *triggers.PatternTrigger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"queue_depth":      pq.Len(),
			"total_running":    budget.TotalRunning(),
			"budgets":          budget.Stats(),
			"pattern_pending":  pt.PendingCount(),
			"pattern_last_fired": pt.LastFired().UTC().Format(time.RFC3339),
			"timestamp":        time.Now().UTC().Format(time.RFC3339),
		})
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func buildLogger(level string) *zap.Logger {
	lvl := zap.InfoLevel
	switch level {
	case "debug":
		lvl = zap.DebugLevel
	case "warn":
		lvl = zap.WarnLevel
	case "error":
		lvl = zap.ErrorLevel
	}
	encoderCfg := zap.NewProductionEncoderConfig()
	encoderCfg.EncodeTime = zapcore.ISO8601TimeEncoder
	cfg := zap.Config{
		Level:            zap.NewAtomicLevelAt(lvl),
		Development:      false,
		Encoding:         "json",
		EncoderConfig:    encoderCfg,
		OutputPaths:      []string{"stdout"},
		ErrorOutputPaths: []string{"stderr"},
	}
	l, err := cfg.Build()
	if err != nil {
		panic("build logger: " + err.Error())
	}
	return l
}
