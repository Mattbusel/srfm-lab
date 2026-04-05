// Command idea-api is the HTTP API server for the Idea Automation Engine.
// It serves hypotheses, genome populations, shadow-runner leaderboards,
// experiments, patterns, and narratives from idea_engine.db.
//
// Listens on :8767 by default (configurable via IDEA_API_PORT env var).
package main

import (
	"context"
	"encoding/json"
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

	"srfm-lab/idea-engine/idea-api/config"
	ideadb "srfm-lab/idea-engine/idea-api/db"
	"srfm-lab/idea-engine/idea-api/db/queries"
	"srfm-lab/idea-engine/idea-api/handlers"
	apimiddleware "srfm-lab/idea-engine/idea-api/middleware"
	"srfm-lab/idea-engine/idea-api/types"
	ideaws "srfm-lab/idea-engine/idea-api/websocket"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	log := buildLogger(cfg.LogLevel)
	defer log.Sync() //nolint:errcheck

	log.Info("starting idea-api",
		zap.String("listen", cfg.ListenAddr),
		zap.String("db", cfg.DBPath),
		zap.String("bus", cfg.BusURL),
	)

	if err := run(cfg, log); err != nil {
		log.Fatal("idea-api exited with error", zap.Error(err))
	}
}

func run(cfg *config.Config, log *zap.Logger) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// ---- Database ----
	var database *ideadb.IdeaDB
	d, err := ideadb.NewIdeaDB(cfg.DBPath)
	if err != nil {
		log.Warn("could not open idea_engine.db; DB-dependent routes will return 503",
			zap.String("path", cfg.DBPath),
			zap.Error(err),
		)
	} else {
		database = d
		defer database.Close()
	}

	// ---- Stores ----
	var (
		hypoStore *queries.HypothesisStore
		genStore  *queries.GenomeStore
		shadStore *queries.ShadowStore
		expStore  *queries.ExperimentStore
		patStore  *queries.PatternStore
		narStore  *queries.NarrativeStore
	)
	if database != nil {
		hypoStore = queries.NewHypothesisStore(database)
		genStore = queries.NewGenomeStore(database)
		shadStore = queries.NewShadowStore(database)
		expStore = queries.NewExperimentStore(database)
		patStore = queries.NewPatternStore(database)
		narStore = queries.NewNarrativeStore(database)
	}

	// ---- Handlers ----
	hypoH := handlers.NewHypothesisHandler(hypoStore, log)
	genH := handlers.NewGenomeHandler(genStore, log)
	shadH := handlers.NewShadowHandler(shadStore, log)
	expH := handlers.NewExperimentHandler(expStore, "http://localhost:8769", log)
	patH := handlers.NewPatternHandler(patStore, log)
	narH := handlers.NewNarrativeHandler(narStore, log)

	// ---- WebSocket hub ----
	wsHub := ideaws.NewHub(log)

	// ---- Rate limiter ----
	rl := apimiddleware.NewRateLimiter(cfg.RateLimitRPS, cfg.RateLimitBurst, log)

	// ---- Chi router ----
	r := chi.NewRouter()

	r.Use(chimiddleware.RequestID)
	r.Use(chimiddleware.RealIP)
	r.Use(loggingMiddleware(log))
	r.Use(apimiddleware.CORS)
	r.Use(rl.Middleware)
	r.Use(chimiddleware.Recoverer)
	r.Use(chimiddleware.Timeout(30 * time.Second))

	// ---- Routes ----
	r.Route("/api/v1", func(r chi.Router) {
		r.Get("/health", healthHandler(database, log))

		// Hypotheses
		r.Get("/hypotheses", guardDB(database, hypoH.GetHypotheses))
		r.Get("/hypotheses/top", guardDB(database, hypoH.GetTopHypotheses))
		r.Get("/hypotheses/{id}", guardDB(database, hypoH.GetHypothesisByID))
		r.Post("/hypotheses/{id}/approve", guardDB(database, hypoH.ApproveHypothesis))
		r.Post("/hypotheses/{id}/reject", guardDB(database, hypoH.RejectHypothesis))

		// Genomes
		r.Get("/genomes/population", guardDB(database, genH.GetPopulation))
		r.Get("/genomes/archive", guardDB(database, genH.GetArchive))
		r.Get("/genomes/fitness-history", guardDB(database, genH.GetFitnessHistory))

		// Shadow runner
		r.Get("/shadow/leaderboard", guardDB(database, shadH.GetLeaderboard))
		r.Get("/shadow/variants/{id}/history", guardDB(database, shadH.GetVariantHistory))
		r.Post("/shadow/variants/{id}/promote", guardDB(database, shadH.PromoteVariant))

		// Experiments
		r.Get("/experiments", guardDB(database, expH.GetExperiments))
		r.Post("/experiments", guardDB(database, expH.CreateExperiment))
		r.Get("/experiments/{id}", guardDB(database, expH.GetExperimentByID))
		r.Post("/experiments/{id}/run", guardDB(database, expH.RunExperiment))

		// Patterns
		r.Get("/patterns", guardDB(database, patH.GetPatterns))
		r.Get("/patterns/{id}", guardDB(database, patH.GetPatternByID))

		// Narratives — /latest must be registered before /{id} to avoid shadowing.
		r.Get("/narratives", guardDB(database, narH.GetNarratives))
		r.Get("/narratives/latest", guardDB(database, narH.GetLatestNarrative))
		r.Get("/narratives/{id}", guardDB(database, narH.GetNarrativeByID))
	})

	// WebSocket endpoint for live shadow score updates.
	r.Get("/ws", wsHub.ServeWS)

	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "route not found"})
	})

	srv := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		log.Info("idea-api listening", zap.String("addr", cfg.ListenAddr))
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
	log.Info("idea-api stopped cleanly")
	return nil
}

// guardDB wraps a handler and returns 503 if the database is unavailable.
func guardDB(database *ideadb.IdeaDB, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if database == nil {
			writeJSON(w, http.StatusServiceUnavailable,
				types.ErrorResponse{Error: "database unavailable", Code: http.StatusServiceUnavailable})
			return
		}
		h(w, r)
	}
}

// healthHandler returns a JSON health response with DB status and uptime.
func healthHandler(database *ideadb.IdeaDB, log *zap.Logger) http.HandlerFunc {
	start := time.Now()
	return func(w http.ResponseWriter, r *http.Request) {
		dbStatus := "ok"
		if database == nil {
			dbStatus = "unavailable"
		} else {
			pingCtx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
			defer cancel()
			if err := database.Ping(pingCtx); err != nil {
				dbStatus = "error: " + err.Error()
				log.Warn("health check db ping failed", zap.Error(err))
			}
		}
		status := http.StatusOK
		if dbStatus != "ok" {
			status = http.StatusServiceUnavailable
		}
		writeJSON(w, status, map[string]interface{}{
			"status":    statusLabel(dbStatus),
			"service":   "idea-api",
			"version":   "0.1.0",
			"uptime_s":  time.Since(start).Seconds(),
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"db":        dbStatus,
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

// loggingMiddleware logs every request using zap.
func loggingMiddleware(log *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			wrapped := &responseWriter{ResponseWriter: w, status: http.StatusOK}
			reqID := r.Header.Get("X-Request-ID")
			if reqID == "" {
				reqID = fmt.Sprintf("%d", start.UnixNano())
			}
			wrapped.Header().Set("X-Request-ID", reqID)
			defer func() {
				elapsed := time.Since(start)
				fields := []zap.Field{
					zap.String("method", r.Method),
					zap.String("path", r.URL.Path),
					zap.Int("status", wrapped.status),
					zap.Duration("duration", elapsed),
					zap.String("request_id", reqID),
				}
				if r.URL.Path == "/api/v1/health" {
					log.Debug("request", fields...)
				} else if wrapped.status >= 500 {
					log.Error("request", fields...)
				} else {
					log.Info("request", fields...)
				}
			}()
			next.ServeHTTP(wrapped, r)
		})
	}
}

// responseWriter captures the status code for logging.
type responseWriter struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (rw *responseWriter) WriteHeader(code int) {
	if rw.wroteHeader {
		return
	}
	rw.status = code
	rw.wroteHeader = true
	rw.ResponseWriter.WriteHeader(code)
}

// buildLogger constructs a production zap logger.
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
