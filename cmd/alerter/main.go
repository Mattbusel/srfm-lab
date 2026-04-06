// cmd/alerter/main.go — SRFM Lab alert service.
//
// HTTP server on :8795. Reads alert_rules.yaml, polls Prometheus (:9090),
// health (:8799), and heartbeat (:8783). Routes alerts to Slack / log / stdout.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

const (
	listenAddr      = ":8795"
	prometheusAddr  = "http://localhost:9090"
	healthAddr      = "http://localhost:8799"
	heartbeatAddr   = "http://localhost:8783"
	alertRulesPath  = "config/alert_rules.yaml"
	alertsLogPath   = "logs/alerts.log"
	evalInterval    = 10 * time.Second
)

// Server wires together the alert engine, notifiers, and HTTP endpoints.
type Server struct {
	engine    *AlertEngine
	notifiers CompositeNotifier
	metrics   *MetricsClient
	health    *HealthClient
	heartbeat *HeartbeatClient
	logger    *slog.Logger
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	slog.Info("starting SRFM alerter", "addr", listenAddr)

	// Load alert rules.
	rules, err := LoadAlertRules(alertRulesPath)
	if err != nil {
		slog.Warn("could not load alert_rules.yaml, using defaults", "err", err)
		rules = DefaultAlertRules()
	}
	slog.Info("loaded alert rules", "count", len(rules))

	// Build notifiers.
	var notifiers []Notifier
	notifiers = append(notifiers, NewLogNotifier(alertsLogPath, logger))
	notifiers = append(notifiers, NewStdoutNotifier(logger))
	if webhookURL := os.Getenv("SLACK_WEBHOOK_URL"); webhookURL != "" {
		slog.Info("Slack notifier enabled")
		notifiers = append(notifiers, NewSlackNotifier(webhookURL, logger))
	}
	if customURL := os.Getenv("ALERT_WEBHOOK_URL"); customURL != "" {
		notifiers = append(notifiers, NewWebhookNotifier(customURL, logger))
	}
	composite := NewCompositeNotifier(notifiers)

	// Build clients.
	mc := NewMetricsClient(prometheusAddr, 5*time.Second)
	hc := NewHealthClient(healthAddr, 5*time.Second)
	bc := NewHeartbeatClient(heartbeatAddr, 5*time.Second)

	engine := NewAlertEngine(rules, composite, logger)

	srv := &Server{
		engine:    engine,
		notifiers: composite,
		metrics:   mc,
		health:    hc,
		heartbeat: bc,
		logger:    logger,
	}

	// Start alert evaluation loop.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go srv.runEvalLoop(ctx)

	// HTTP API.
	mux := http.NewServeMux()
	mux.HandleFunc("/", srv.handleRoot)
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/alerts/active", srv.handleActiveAlerts)
	mux.HandleFunc("/alerts/history", srv.handleAlertHistory)
	mux.HandleFunc("/alerts/rules", srv.handleRules)
	mux.HandleFunc("/metrics", srv.handleMetrics)

	httpSrv := &http.Server{
		Addr:         listenAddr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	// Graceful shutdown on SIGINT/SIGTERM.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		slog.Info("shutting down alerter")
		cancel()
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutCancel()
		_ = httpSrv.Shutdown(shutCtx)
	}()

	slog.Info("alerter HTTP server listening", "addr", listenAddr)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		slog.Error("HTTP server error", "err", err)
		os.Exit(1)
	}
}

// runEvalLoop collects metrics and evaluates alert rules every evalInterval.
func (s *Server) runEvalLoop(ctx context.Context) {
	ticker := time.NewTicker(evalInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.evalOnce(ctx)
		}
	}
}

func (s *Server) evalOnce(ctx context.Context) {
	snapshot := MetricSnapshot{}

	// Collect from Prometheus.
	if pm, err := s.metrics.Scrape(ctx); err == nil {
		snapshot.Equity = pm.Equity
		snapshot.DrawdownPct = pm.DrawdownPct
		snapshot.OpenPositions = pm.OpenPositions
		snapshot.CircuitBreakerState = pm.CircuitBreakerState
		snapshot.BhActiveCount = pm.BhActiveCount
		snapshot.DayPnlPct = pm.DayPnlPct
	} else {
		s.logger.Warn("prometheus scrape failed", "err", err)
	}

	// Collect health checks.
	if hs, err := s.health.Check(ctx); err == nil {
		snapshot.HealthChecks = hs
	} else {
		s.logger.Warn("health check failed", "err", err)
	}

	// Check heartbeat liveness.
	snapshot.TraderAlive = s.heartbeat.IsAlive(ctx)

	s.engine.Evaluate(ctx, snapshot)
}

// ─── HTTP Handlers ────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"service":  "srfm-alerter",
		"version":  "0.1.0",
		"status":   "running",
		"uptime":   fmt.Sprintf("%s", time.Since(startTime).Truncate(time.Second)),
		"endpoints": []string{
			"/health", "/alerts/active", "/alerts/history", "/alerts/rules", "/metrics",
		},
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleActiveAlerts(w http.ResponseWriter, r *http.Request) {
	alerts := s.engine.ActiveAlerts()
	writeJSON(w, http.StatusOK, map[string]any{
		"count":  len(alerts),
		"alerts": alerts,
	})
}

func (s *Server) handleAlertHistory(w http.ResponseWriter, r *http.Request) {
	history := s.engine.History()
	writeJSON(w, http.StatusOK, map[string]any{
		"count":   len(history),
		"history": history,
	})
}

func (s *Server) handleRules(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"rules": s.engine.Rules(),
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	pm, err := s.metrics.Scrape(ctx)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, pm)
}

var startTime = time.Now()
