// cmd/alerter/main.go -- SRFM Lab alert service.
//
// HTTP server on :8795. Integrates the rule engine (engine/rules.go) and
// alert router (engine/routing.go) alongside the existing AlertEngine.
//
// Endpoints:
//   GET  /                      -- service info
//   GET  /health                -- liveness probe
//   GET  /alerts/active         -- currently firing alerts (both engines)
//   GET  /alerts/history        -- last 1000 alert events
//   GET  /alerts/rules          -- configured rule names
//   GET  /metrics               -- latest prometheus snapshot
//   POST /alerts/silence        -- suppress a rule for N minutes
//   POST /maintenance/set       -- set maintenance window

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

	"srfm-alerter/engine"
)

const (
	listenAddr     = ":8795"
	prometheusAddr = "http://localhost:9090"
	healthAddr     = "http://localhost:8799"
	heartbeatAddr  = "http://localhost:8783"
	alertRulesPath = "config/alert_rules.yaml"
	alertsLogPath  = "logs/alerts.log"
	evalInterval   = 10 * time.Second
)

// Server wires together both alert engines, router, notifiers, and HTTP endpoints.
type Server struct {
	// Legacy YAML-driven engine.
	engine    *AlertEngine
	notifiers CompositeNotifier

	// New typed rule engine.
	ruleEngine *engine.RuleEngine
	router     *engine.AlertRouter

	metrics   *MetricsClient
	health    *HealthClient
	heartbeat *HeartbeatClient
	logger    *slog.Logger

	// Context state for constructing AlertContext.
	lastParamUpdate   time.Time
	btcEthCorrelation float64
	serviceRestarts   map[string]int
	feedLastSeen      map[string]time.Time
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	slog.Info("starting SRFM alerter", "addr", listenAddr)

	// ---- Legacy rule engine ----
	rules, err := LoadAlertRules(alertRulesPath)
	if err != nil {
		slog.Warn("could not load alert_rules.yaml, using defaults", "err", err)
		rules = DefaultAlertRules()
	}
	slog.Info("loaded alert rules", "count", len(rules))

	var notifiers []Notifier
	notifiers = append(notifiers, NewLogNotifier(alertsLogPath, logger))
	notifiers = append(notifiers, NewStdoutNotifier(logger))
	if webhookURL := os.Getenv("SLACK_WEBHOOK_URL"); webhookURL != "" {
		slog.Info("Slack notifier enabled (legacy)")
		notifiers = append(notifiers, NewSlackNotifier(webhookURL, logger))
	}
	if customURL := os.Getenv("ALERT_WEBHOOK_URL"); customURL != "" {
		notifiers = append(notifiers, NewWebhookNotifier(customURL, logger))
	}
	composite := NewCompositeNotifier(notifiers)

	mc := NewMetricsClient(prometheusAddr, 5*time.Second)
	hc := NewHealthClient(healthAddr, 5*time.Second)
	bc := NewHeartbeatClient(heartbeatAddr, 5*time.Second)

	legacyEngine := NewAlertEngine(rules, composite, logger)

	// ---- New typed rule engine + router ----
	routerOpts := []engine.RouterOption{}
	if slackURL := os.Getenv("SLACK_WEBHOOK_URL"); slackURL != "" {
		slackCh := os.Getenv("SLACK_CHANNEL")
		if slackCh == "" {
			slackCh = "#alerts"
		}
		slackN := engine.NewSlackNotifier(slackURL, slackCh, 5, logger)
		routerOpts = append(routerOpts, engine.WithSlack(slackN))
		slog.Info("Slack notifier enabled (rule engine)", "channel", slackCh)
	}
	if pdKey := os.Getenv("PAGERDUTY_ROUTING_KEY"); pdKey != "" {
		pdN := engine.NewPagerDutyNotifier(pdKey, logger)
		routerOpts = append(routerOpts, engine.WithPagerDuty(pdN))
		slog.Info("PagerDuty notifier enabled")
	}
	if sentryDSN := os.Getenv("SENTRY_DSN"); sentryDSN != "" {
		sentryN := engine.NewSentryNotifier(sentryDSN, logger)
		routerOpts = append(routerOpts, engine.WithSentry(sentryN))
		slog.Info("Sentry notifier enabled")
	}

	router := engine.NewAlertRouter(logger, routerOpts...)

	dispatch := func(ctx context.Context, alert *engine.Alert) {
		if err := router.Route(ctx, alert); err != nil {
			logger.Error("route error", "rule", alert.RuleName, "err", err)
		}
	}

	ruleEngine := engine.NewRuleEngine(logger, dispatch)
	slog.Info("rule engine initialized", "rules", ruleEngine.RuleNames())

	srv := &Server{
		engine:           legacyEngine,
		notifiers:        composite,
		ruleEngine:       ruleEngine,
		router:           router,
		metrics:          mc,
		health:           hc,
		heartbeat:        bc,
		logger:           logger,
		serviceRestarts:  make(map[string]int),
		feedLastSeen:     make(map[string]time.Time),
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start legacy eval loop.
	go srv.runEvalLoop(ctx)

	// Start new rule engine loop.
	go ruleEngine.Run(ctx, srv.buildAlertContext)

	// HTTP API.
	mux := http.NewServeMux()
	mux.HandleFunc("/", srv.handleRoot)
	mux.HandleFunc("/health", srv.handleHealth)
	mux.HandleFunc("/alerts/active", srv.handleActiveAlerts)
	mux.HandleFunc("/alerts/history", srv.handleAlertHistory)
	mux.HandleFunc("/alerts/rules", srv.handleRules)
	mux.HandleFunc("/metrics", srv.handleMetrics)
	mux.HandleFunc("/alerts/silence", srv.handleSilence)
	mux.HandleFunc("/maintenance/set", srv.handleMaintenanceSet)

	httpSrv := &http.Server{
		Addr:         listenAddr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

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

// buildAlertContext constructs an AlertContext from the most recent scrape data.
// This is called by the rule engine on each tick.
func (s *Server) buildAlertContext() engine.AlertContext {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ac := engine.AlertContext{
		CollectedAt:       time.Now(),
		LastParamUpdateAt: s.lastParamUpdate,
		BTCETHCorrelation: s.btcEthCorrelation,
	}

	// Prometheus scrape.
	if pm, err := s.metrics.Scrape(ctx); err == nil {
		ac.Equity = pm.Equity
		ac.DrawdownPct = pm.DrawdownPct
		ac.DayPnlPct = pm.DayPnlPct
		ac.BHActiveCount = pm.BhActiveCount

		// Derive peak equity from drawdown and current equity.
		if pm.DrawdownPct < 0 && pm.Equity > 0 {
			ac.PeakEquity = pm.Equity / (1 + pm.DrawdownPct)
		} else {
			ac.PeakEquity = pm.Equity
		}

		// Pull extended metrics from raw map.
		if v, ok := pm.Raw["srfm_var95_day"]; ok {
			ac.VaR95Day = v
		}
		if v, ok := pm.Raw["srfm_realized_loss_pct"]; ok {
			ac.RealizedLossPct = v
		}
		if v, ok := pm.Raw["srfm_signal_bullish"]; ok {
			ac.SignalBullish = int(v)
		}
		if v, ok := pm.Raw["srfm_signal_bearish"]; ok {
			ac.SignalBearish = int(v)
		}
		if v, ok := pm.Raw["srfm_btceth_corr"]; ok {
			ac.BTCETHCorrelation = v
		}
		if v, ok := pm.Raw["srfm_equity_drop_bar1"]; ok {
			ac.EquityDropBars = map[int]float64{1: v}
		}
	}

	// Health checks -> service info.
	if hs, err := s.health.Check(ctx); err == nil {
		for name, healthy := range hs {
			restarts := 0
			// Fall back to zero restarts if unknown.
			ac.Services = append(ac.Services, engine.ServiceRestartInfo{
				Name:    name,
				Healthy: healthy,
				Restarts: restarts,
			})
		}
	}

	// Feed staleness -- use whatever we've been told via webhook updates.
	for sym, lastSeen := range s.feedLastSeen {
		ac.Feeds = append(ac.Feeds, engine.FeedInfo{
			Symbol:   sym,
			LastSeen: lastSeen,
			Healthy:  time.Since(lastSeen) < 30*time.Minute,
		})
	}

	return ac
}

// runEvalLoop drives the legacy AlertEngine.
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

	if hs, err := s.health.Check(ctx); err == nil {
		snapshot.HealthChecks = hs
	} else {
		s.logger.Warn("health check failed", "err", err)
	}

	snapshot.TraderAlive = s.heartbeat.IsAlive(ctx)
	s.engine.Evaluate(ctx, snapshot)
}

// ---------------------------------------------------------------------------
// HTTP Handlers
// ---------------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"service": "srfm-alerter",
		"version": "0.2.0",
		"status":  "running",
		"uptime":  fmt.Sprintf("%s", time.Since(startTime).Truncate(time.Second)),
		"endpoints": []string{
			"/health", "/alerts/active", "/alerts/history",
			"/alerts/rules", "/metrics", "/alerts/silence", "/maintenance/set",
		},
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// handleActiveAlerts merges results from both engines.
func (s *Server) handleActiveAlerts(w http.ResponseWriter, r *http.Request) {
	legacy := s.engine.ActiveAlerts()
	typed := s.ruleEngine.ActiveAlerts()
	writeJSON(w, http.StatusOK, map[string]any{
		"legacy_count": len(legacy),
		"legacy":       legacy,
		"engine_count": len(typed),
		"engine":       typed,
		"total":        len(legacy) + len(typed),
	})
}

func (s *Server) handleAlertHistory(w http.ResponseWriter, r *http.Request) {
	history := s.engine.History()
	typed := s.ruleEngine.History()
	writeJSON(w, http.StatusOK, map[string]any{
		"legacy_count": len(history),
		"legacy":       history,
		"engine_count": len(typed),
		"engine":       typed,
	})
}

func (s *Server) handleRules(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"legacy_rules": s.engine.Rules(),
		"engine_rules": s.ruleEngine.RuleNames(),
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

// silenceRequest is the JSON body for POST /alerts/silence.
type silenceRequest struct {
	RuleName    string `json:"rule_name"`
	DurationMin int    `json:"duration_minutes"`
}

// handleSilence silences a specific rule in the typed rule engine router.
func (s *Server) handleSilence(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "POST only"})
		return
	}
	var req silenceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}
	if req.RuleName == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "rule_name required"})
		return
	}
	dur := time.Duration(req.DurationMin) * time.Minute
	if dur <= 0 {
		dur = 60 * time.Minute // default: 1 hour
	}
	s.router.Silence(req.RuleName, dur)
	writeJSON(w, http.StatusOK, map[string]any{
		"silenced":         req.RuleName,
		"duration_minutes": int(dur.Minutes()),
		"expires_at":       time.Now().Add(dur).Format(time.RFC3339),
	})
}

// maintenanceSetRequest is the JSON body for POST /maintenance/set.
type maintenanceSetRequest struct {
	Windows []struct {
		Label    string `json:"label"`
		StartsAt string `json:"starts_at"` // RFC3339
		EndsAt   string `json:"ends_at"`   // RFC3339
	} `json:"windows"`
}

// handleMaintenanceSet replaces the maintenance window configuration.
func (s *Server) handleMaintenanceSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "POST only"})
		return
	}
	var req maintenanceSetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}

	windows := make([]engine.MaintenanceWindow, 0, len(req.Windows))
	for _, win := range req.Windows {
		starts, err := time.Parse(time.RFC3339, win.StartsAt)
		if err != nil {
			writeJSON(w, http.StatusBadRequest,
				map[string]string{"error": fmt.Sprintf("bad starts_at for %q: %v", win.Label, err)})
			return
		}
		ends, err := time.Parse(time.RFC3339, win.EndsAt)
		if err != nil {
			writeJSON(w, http.StatusBadRequest,
				map[string]string{"error": fmt.Sprintf("bad ends_at for %q: %v", win.Label, err)})
			return
		}
		windows = append(windows, engine.MaintenanceWindow{
			Label:    win.Label,
			StartsAt: starts,
			EndsAt:   ends,
		})
	}

	s.router.SetMaintenance(windows)
	writeJSON(w, http.StatusOK, map[string]any{
		"windows_set": len(windows),
		"active":      s.router.ActiveMaintenance(),
	})
}

var startTime = time.Now()
