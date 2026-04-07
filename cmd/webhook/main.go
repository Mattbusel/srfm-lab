// cmd/webhook/main.go -- SRFM Lab webhook receiver service.
//
// HTTP server on :8796. Receives webhooks from:
//   - Alpaca (fill events)        POST /webhook/fills
//   - IAE genome engine (params)  POST /webhook/params/update
//   - All services (health)       POST /webhook/health/{service}
//
// Summary endpoint:
//   GET /webhook/health/summary

package main

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	_ "github.com/mattn/go-sqlite3"

	"srfm-webhook/handlers"
	"srfm-webhook/middleware"
)

const (
	listenAddr   = ":8796"
	elixirAddr   = "http://localhost:8781"
	alerterAddr  = "http://localhost:8795"
	dbPath       = "data/webhook.db"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	slog.Info("starting SRFM webhook service", "addr", listenAddr)

	// Open SQLite database.
	if err := os.MkdirAll("data", 0755); err != nil {
		slog.Error("could not create data directory", "err", err)
		os.Exit(1)
	}
	db, err := sql.Open("sqlite3", dbPath+"?_journal=WAL&_busy_timeout=5000")
	if err != nil {
		slog.Error("could not open sqlite", "err", err)
		os.Exit(1)
	}
	defer db.Close()

	if err := migrateDB(db, logger); err != nil {
		slog.Error("db migration failed", "err", err)
		os.Exit(1)
	}

	// Auth middleware.
	alpacaSecret := os.Getenv("ALPACA_WEBHOOK_SECRET")
	if alpacaSecret == "" {
		slog.Warn("ALPACA_WEBHOOK_SECRET not set -- signature verification disabled")
	}
	auth := middleware.NewHMACAuth(alpacaSecret, logger)
	rl := middleware.NewRateLimiter(logger)

	// Handlers.
	fillsH := handlers.NewAlpacaFillsHandler(db, elixirAddr, alerterAddr, logger)
	paramsH := handlers.NewParameterWebhookHandler(db, elixirAddr, logger)
	healthH := handlers.NewHealthWebhookHandler(alerterAddr, logger)

	// Routes.
	mux := http.NewServeMux()

	// Alpaca fills: HMAC auth + rate limit.
	mux.Handle("/webhook/fills",
		rl.Limit("fills", 100, 10)(
			auth.Verify(http.HandlerFunc(fillsH.HandleFill)),
		),
	)

	// Parameter updates: rate limit (no HMAC -- IAE is internal).
	mux.Handle("/webhook/params/update",
		rl.Limit("params", 5, 1)(
			http.HandlerFunc(paramsH.HandleParamUpdate),
		),
	)

	// Health webhooks: per-service routes.
	mux.HandleFunc("/webhook/health/", func(w http.ResponseWriter, r *http.Request) {
		// /webhook/health/summary is a GET.
		tail := strings.TrimPrefix(r.URL.Path, "/webhook/health/")
		if tail == "summary" && r.Method == http.MethodGet {
			healthH.HandleSummary(w, r)
			return
		}
		if r.Method == http.MethodPost {
			healthH.HandleServiceHealth(w, r)
			return
		}
		http.NotFound(w, r)
	})

	// Liveness / info.
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"status":"ok","service":"srfm-webhook","uptime":%q}`,
			time.Since(startTime).Truncate(time.Second).String())
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"service":"srfm-webhook","version":"0.1.0","status":"running"}`)
	})

	httpSrv := &http.Server{
		Addr:         listenAddr,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		slog.Info("shutting down webhook service")
		cancel()
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutCancel()
		_ = httpSrv.Shutdown(shutCtx)
	}()

	_ = ctx // used by handlers via request context

	slog.Info("webhook HTTP server listening", "addr", listenAddr)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		slog.Error("HTTP server error", "err", err)
		os.Exit(1)
	}
}

// migrateDB creates the required tables if they do not exist.
func migrateDB(db *sql.DB, logger *slog.Logger) error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS fills (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			order_id        TEXT NOT NULL,
			symbol          TEXT NOT NULL,
			qty             REAL NOT NULL,
			price           REAL NOT NULL,
			side            TEXT NOT NULL,
			fill_type       TEXT NOT NULL,
			realized_pnl    REAL NOT NULL DEFAULT 0,
			timestamp       TEXT NOT NULL,
			created_at      TEXT NOT NULL DEFAULT (datetime('now')),
			raw_payload     TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol)`,
		`CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id)`,

		`CREATE TABLE IF NOT EXISTS positions (
			symbol          TEXT PRIMARY KEY,
			qty             REAL NOT NULL DEFAULT 0,
			avg_cost        REAL NOT NULL DEFAULT 0,
			side            TEXT NOT NULL DEFAULT 'flat',
			unrealized_pnl  REAL NOT NULL DEFAULT 0,
			realized_pnl    REAL NOT NULL DEFAULT 0,
			updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
		)`,

		`CREATE TABLE IF NOT EXISTS param_proposals (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			genome_hash     TEXT NOT NULL,
			fitness_score   REAL NOT NULL,
			schema_valid    INTEGER NOT NULL DEFAULT 0,
			forwarded       INTEGER NOT NULL DEFAULT 0,
			proposed_at     TEXT NOT NULL DEFAULT (datetime('now')),
			raw_payload     TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_params_proposed_at ON param_proposals(proposed_at)`,
	}

	for _, stmt := range statements {
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("migrate: %w", err)
		}
	}

	logger.Info("database migration complete", "path", dbPath)
	return nil
}

var startTime = time.Now()
