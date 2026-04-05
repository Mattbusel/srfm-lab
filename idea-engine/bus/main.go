// Command bus is the in-process event bus service for the Idea Automation
// Engine. It exposes:
//
//   - POST /events        — for Python modules to publish events onto the bus
//   - GET  /topics        — lists all registered topic constants
//   - GET  /replay        — replays persisted events for crash recovery
//   - GET  /api/v1/health — health check with DB status
//   - GET  /api/v1/stats  — subscription counts per topic
//
// Listens on :8768 by default (configurable via BUS_ADDR env var).
package main

import (
	"context"
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
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"srfm-lab/idea-engine/bus"
	"srfm-lab/idea-engine/bus/adapters"
)

const defaultListenAddr = ":8768"

type config struct {
	listenAddr string
	dbPath     string
	logLevel   string
}

func main() {
	cfg := parseFlags()

	log := buildLogger(cfg.logLevel)
	defer log.Sync() //nolint:errcheck

	log.Info("starting idea-engine bus", zap.String("listen", cfg.listenAddr))

	if err := run(cfg, log); err != nil {
		log.Fatal("bus exited with error", zap.Error(err))
	}
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.listenAddr, "listen", envOr("BUS_ADDR", defaultListenAddr),
		"HTTP listen address")
	flag.StringVar(&cfg.dbPath, "db", envOr("IDEA_ENGINE_DB", "./idea_engine.db"),
		"Path to idea_engine.db SQLite database")
	flag.StringVar(&cfg.logLevel, "log-level", envOr("LOG_LEVEL", "info"),
		"Log level: debug|info|warn|error")
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

	// ---- Persistence layer ----
	store, err := bus.NewEventStore(cfg.dbPath, log)
	if err != nil {
		// Non-fatal: bus still operates in-memory without persistence.
		log.Warn("could not open event store; events will not be persisted",
			zap.String("path", cfg.dbPath),
			zap.Error(err),
		)
		store = nil
	}
	if store != nil {
		defer store.Close()
	}

	// ---- Router ----
	router := bus.NewRouter(log, store)

	// ---- Python HTTP adapter ----
	adapter := adapters.NewPythonAdapter(router, log)

	// ---- Chi router ----
	r := chi.NewRouter()

	r.Use(chimiddleware.RequestID)
	r.Use(chimiddleware.RealIP)
	r.Use(loggingMiddleware(log))
	r.Use(corsMiddleware)
	r.Use(chimiddleware.Recoverer)
	r.Use(chimiddleware.Timeout(30 * time.Second))

	// Python inbound adapter
	r.Post("/events", adapter.HandlePublish)
	r.Get("/topics", adapter.HandleTopics)
	r.Get("/replay", func(w http.ResponseWriter, req *http.Request) {
		if store == nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{
				"error": "event store not available",
			})
			return
		}
		adapter.HandleReplay(store, w, req)
	})

	// API endpoints
	r.Route("/api/v1", func(r chi.Router) {
		r.Get("/health", healthHandler(store, log))
		r.Get("/stats", statsHandler(router))
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
		log.Info("bus listening", zap.String("addr", cfg.listenAddr))
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
			return err
		}
		return nil
	case <-ctx.Done():
	}

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", zap.Error(err))
		return err
	}
	log.Info("bus stopped cleanly")
	return nil
}

// healthHandler returns JSON health including DB reachability.
func healthHandler(store *bus.EventStore, log *zap.Logger) http.HandlerFunc {
	start := time.Now()
	return func(w http.ResponseWriter, r *http.Request) {
		dbStatus := "ok"
		if store == nil {
			dbStatus = "unavailable"
		}
		status := http.StatusOK
		if dbStatus != "ok" {
			status = http.StatusServiceUnavailable
		}
		writeJSON(w, status, map[string]interface{}{
			"status":    statusLabel(dbStatus),
			"service":   "idea-engine-bus",
			"version":   "0.1.0",
			"uptime_s":  time.Since(start).Seconds(),
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"db":        dbStatus,
		})
	}
}

// statsHandler returns subscription counts per topic.
func statsHandler(router *bus.Router) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"subscriptions": router.TopicCounts(),
			"timestamp":     time.Now().UTC().Format(time.RFC3339),
		})
	}
}

func statusLabel(db string) string {
	if db == "ok" {
		return "healthy"
	}
	return "degraded"
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// corsMiddleware adds permissive CORS headers for local development.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, X-Request-ID")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// loggingMiddleware produces per-request zap log lines.
func loggingMiddleware(log *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			next.ServeHTTP(w, r)
			log.Info("request",
				zap.String("method", r.Method),
				zap.String("path", r.URL.Path),
				zap.Duration("duration", time.Since(start)),
			)
		})
	}
}

// buildLogger constructs a production zap logger at the requested level.
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
