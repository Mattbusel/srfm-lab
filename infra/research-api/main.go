// Command research-api is the HTTP API server for the SRFM research platform.
// It serves backtesting results, reconciliation data, signal analytics, regime
// detections, portfolio constructions, and Monte Carlo results from the
// SQLite warehouse and on-disk JSON artefacts produced by the Python research
// stack.
//
// Listens on :8766 (one above the Spacetime Arena gateway on :8765).
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	chimiddleware "github.com/go-chi/chi/v5/middleware"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/srfm/research-api/db"
	"github.com/srfm/research-api/handlers"
	apimiddleware "github.com/srfm/research-api/middleware"
)

const defaultListenAddr = ":8766"

// config holds runtime configuration.
type config struct {
	listenAddr    string
	warehousePath string
	livePath      string
	mcResultDir   string
	mcScriptPath  string
	reportDir     string
	logLevel      string
}

func main() {
	cfg := parseFlags()

	log := buildLogger(cfg.logLevel)
	defer log.Sync() //nolint:errcheck

	log.Info("starting research-api", zap.String("listen", cfg.listenAddr))

	if err := run(cfg, log); err != nil {
		log.Fatal("research-api exited with error", zap.Error(err))
	}
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.listenAddr, "listen", envOr("RESEARCH_API_ADDR", defaultListenAddr),
		"HTTP listen address")
	flag.StringVar(&cfg.warehousePath, "warehouse", envOr("WAREHOUSE_DB", "./data/warehouse.db"),
		"Path to warehouse SQLite database")
	flag.StringVar(&cfg.livePath, "livedb", envOr("LIVE_TRADES_DB", "./data/live_trades.db"),
		"Path to live trades SQLite database")
	flag.StringVar(&cfg.mcResultDir, "mc-dir", envOr("MC_RESULT_DIR", "./data/mc_results"),
		"Directory containing mc_*.json result files")
	flag.StringVar(&cfg.mcScriptPath, "mc-script", envOr("MC_SCRIPT", ""),
		"Path to mc.py script for triggering new MC runs")
	flag.StringVar(&cfg.reportDir, "report-dir", envOr("REPORT_DIR", "./data/reports"),
		"Directory containing reconciliation JSON report files")
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

	// ---- Open databases ----
	warehouseDB, err := db.NewSQLiteDB(cfg.warehousePath)
	if err != nil {
		// Non-fatal: API will return 503 on DB-dependent routes.
		log.Warn("could not open warehouse db", zap.String("path", cfg.warehousePath), zap.Error(err))
		warehouseDB = nil
	}
	if warehouseDB != nil {
		defer warehouseDB.Close()
	}

	liveDB, err := db.NewSQLiteDB(cfg.livePath)
	if err != nil {
		log.Warn("could not open live trades db", zap.String("path", cfg.livePath), zap.Error(err))
		liveDB = nil
	}
	if liveDB != nil {
		defer liveDB.Close()
	}

	// ---- Build handlers ----
	tradesH := handlers.NewTradesHandler(warehouseDB, liveDB)
	reconH := handlers.NewReconciliationHandler(warehouseDB, cfg.reportDir)
	signalsH := handlers.NewSignalsHandler(warehouseDB)
	regimesH := handlers.NewRegimesHandler(warehouseDB)
	portfolioH := handlers.NewPortfolioHandler(warehouseDB)
	mcH := handlers.NewMCHandler(cfg.mcResultDir, cfg.mcScriptPath, log)
	wfH := handlers.NewWalkForwardHandler(warehouseDB)

	// ---- Live event broadcaster for WebSocket ----
	hub := newWSHub()

	// ---- Router ----
	r := chi.NewRouter()

	// Global middleware stack.
	r.Use(chimiddleware.RequestID)
	r.Use(chimiddleware.RealIP)
	r.Use(apimiddleware.Logging(log))
	r.Use(apimiddleware.CORS)
	r.Use(chimiddleware.Recoverer)
	r.Use(chimiddleware.Timeout(30 * time.Second))

	// ---- API v1 routes ----
	r.Route("/api/v1", func(r chi.Router) {
		// Health
		r.Get("/health", healthHandler(warehouseDB, log))

		// Trades
		r.Get("/trades", tradesH.GetTrades)
		r.Get("/trades/stats", tradesH.GetTradeStats)

		// Reconciliation
		r.Get("/reconciliation/latest", reconH.GetLatestRecon)
		r.Get("/reconciliation/slippage", reconH.GetSlippageStats)
		r.Get("/reconciliation/drift", reconH.GetDriftEvents)

		// Walk-forward
		r.Get("/walkforward/runs", wfH.GetRuns)
		r.Get("/walkforward/runs/{id}", wfH.GetRunDetail)

		// Signals
		r.Get("/signals/ic", signalsH.GetICHistory)
		r.Get("/signals/factor-returns", signalsH.GetFactorReturns)
		r.Get("/signals/alpha-decay", signalsH.GetAlphaDecay)

		// Regimes
		r.Get("/regimes/current", regimesH.GetCurrentRegime)
		r.Get("/regimes/history", regimesH.GetRegimeHistory)
		r.Get("/regimes/transition-matrix", regimesH.GetTransitionMatrix)

		// Portfolio
		r.Get("/portfolio/weights", portfolioH.GetWeights)
		r.Get("/portfolio/risk", portfolioH.GetRiskMetrics)
		r.Get("/portfolio/correlation", portfolioH.GetCorrelation)

		// Monte Carlo
		r.Get("/mc/latest", mcH.GetLatestMC)
		r.Get("/mc/history", mcH.GetMCHistory)
		r.Post("/mc/run", mcH.RunMC)

		// Stress (alias to reconciliation stress results)
		r.Get("/stress/latest", stressLatestHandler(warehouseDB))
	})

	// WebSocket endpoint for live trade events.
	r.Get("/ws/live", hub.serveWS)

	// Not-found handler.
	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "route not found"})
	})

	srv := &http.Server{
		Addr:         cfg.listenAddr,
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second, // long for MC streaming
		IdleTimeout:  120 * time.Second,
	}

	// ---- Graceful shutdown ----
	errCh := make(chan error, 1)
	go func() {
		log.Info("research-api listening", zap.String("addr", cfg.listenAddr))
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
	log.Info("research-api stopped cleanly")
	return nil
}

// healthHandler returns a JSON health response with DB status and uptime.
func healthHandler(warehouse *db.SQLiteDB, log *zap.Logger) http.HandlerFunc {
	start := time.Now()
	return func(w http.ResponseWriter, r *http.Request) {
		dbStatus := "ok"
		if warehouse == nil {
			dbStatus = "unavailable"
		} else {
			ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
			defer cancel()
			if err := warehouse.Ping(ctx); err != nil {
				dbStatus = "error: " + err.Error()
				log.Warn("health check db ping failed", zap.Error(err))
			}
		}

		status := http.StatusOK
		if dbStatus != "ok" {
			status = http.StatusServiceUnavailable
		}

		writeJSON(w, status, map[string]any{
			"status":    statusLabel(dbStatus),
			"service":   "research-api",
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

// stressLatestHandler returns the latest stress test results.
func stressLatestHandler(warehouse *db.SQLiteDB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if warehouse == nil {
			writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "warehouse db unavailable"})
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()

		rows, err := warehouse.QueryRows(ctx, `
			SELECT
				sr.id, sr.run_id, sr.scenario_name, sr.scenario_type,
				sr.computed_at,
				COALESCE(sr.portfolio_pnl, 0.0)    AS portfolio_pnl,
				COALESCE(sr.benchmark_pnl, 0.0)    AS benchmark_pnl,
				COALESCE(sr.max_drawdown, 0.0)      AS max_drawdown,
				COALESCE(sr.var_95, 0.0)            AS var_95,
				COALESCE(sr.cvar_95, 0.0)           AS cvar_95,
				sr.margin_call,
				sr.factor_shocks,
				sr.notes
			FROM stress_results sr
			ORDER BY sr.computed_at DESC
			LIMIT 20
		`)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"stress_results": rows, "count": len(rows)})
	}
}

// writeJSON encodes v as indented JSON and writes it to w.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// ---- WebSocket live-trade hub ----

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		// Allow same origins as the CORS middleware.
		origin := r.Header.Get("Origin")
		return origin == "" ||
			origin == "http://localhost:5173" ||
			origin == "http://localhost:5174" ||
			origin == "http://localhost:5175"
	},
	ReadBufferSize:  1024,
	WriteBufferSize: 4096,
}

// wsClient represents a connected WebSocket client.
type wsClient struct {
	conn *websocket.Conn
	send chan []byte
}

// wsHub manages WebSocket clients and broadcasts messages to all of them.
type wsHub struct {
	mu      sync.RWMutex
	clients map[*wsClient]bool
	broadcast chan []byte
}

func newWSHub() *wsHub {
	h := &wsHub{
		clients:   make(map[*wsClient]bool),
		broadcast: make(chan []byte, 256),
	}
	go h.run()
	return h
}

func (h *wsHub) run() {
	for msg := range h.broadcast {
		h.mu.RLock()
		for c := range h.clients {
			select {
			case c.send <- msg:
			default:
				// slow client: skip this message
			}
		}
		h.mu.RUnlock()
	}
}

func (h *wsHub) register(c *wsClient) {
	h.mu.Lock()
	h.clients[c] = true
	h.mu.Unlock()
}

func (h *wsHub) unregister(c *wsClient) {
	h.mu.Lock()
	delete(h.clients, c)
	h.mu.Unlock()
}

// Broadcast sends a message to all connected WebSocket clients.
func (h *wsHub) Broadcast(msg []byte) {
	select {
	case h.broadcast <- msg:
	default:
		// drop if the broadcast channel is full
	}
}

// serveWS upgrades HTTP to WebSocket and pumps messages to the client.
func (h *wsHub) serveWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}

	c := &wsClient{conn: conn, send: make(chan []byte, 64)}
	h.register(c)

	// Send an initial greeting.
	hello, _ := json.Marshal(map[string]string{
		"type":    "connected",
		"service": "research-api",
		"ts":      time.Now().UTC().Format(time.RFC3339),
	})
	c.send <- hello

	// Writer goroutine.
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer func() {
			ticker.Stop()
			conn.Close()
		}()
		for {
			select {
			case msg, ok := <-c.send:
				conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
				if !ok {
					_ = conn.WriteMessage(websocket.CloseMessage, []byte{})
					return
				}
				if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
					return
				}
			case <-ticker.C:
				conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
				if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					return
				}
			}
		}
	}()

	// Reader loop (drain incoming messages / handle pong).
	conn.SetReadLimit(512)
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})
	for {
		_, _, err := conn.ReadMessage()
		if err != nil {
			break
		}
	}
	h.unregister(c)
	close(c.send)
}

// buildLogger constructs a zap logger at the requested level.
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

	log, err := cfg.Build()
	if err != nil {
		panic("build logger: " + err.Error())
	}
	return log
}
