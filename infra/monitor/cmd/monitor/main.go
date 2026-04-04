// Command monitor is the main entry point for the SRFM monitoring service.
// It watches the BH engine state file, polls the Alpaca paper account,
// evaluates alert rules, and serves a live monitoring dashboard.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/srfm/monitor/internal/alerting"
	"github.com/srfm/monitor/internal/dashboard"
	"github.com/srfm/monitor/internal/watcher"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

type config struct {
	ListenAddr      string
	AlpacaAPIKey    string
	AlpacaSecret    string
	AlpacaPaper     bool
	BHStateFile     string
	BHPollEvery     time.Duration
	PortfolioPoll   time.Duration
	AlertFile       string
	WebhookURL      string
	LogLevel        string
}

func loadConfig() config {
	cfg := config{
		ListenAddr:    ":8081",
		AlpacaPaper:   true,
		BHStateFile:   watcher.BHStateFile,
		BHPollEvery:   10 * time.Second,
		PortfolioPoll: 30 * time.Second,
		AlertFile:     "./data/alerts.csv",
		LogLevel:      "info",
	}
	if v := os.Getenv("LISTEN_ADDR"); v != "" {
		cfg.ListenAddr = v
	}
	if v := os.Getenv("ALPACA_API_KEY"); v != "" {
		cfg.AlpacaAPIKey = v
	}
	if v := os.Getenv("ALPACA_SECRET"); v != "" {
		cfg.AlpacaSecret = v
	}
	if v := os.Getenv("BH_STATE_FILE"); v != "" {
		cfg.BHStateFile = v
	}
	if v := os.Getenv("ALERT_FILE"); v != "" {
		cfg.AlertFile = v
	}
	if v := os.Getenv("WEBHOOK_URL"); v != "" {
		cfg.WebhookURL = v
	}
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}
	return cfg
}

func main() {
	flag.Parse()
	cfg := loadConfig()
	log := buildLogger(cfg.LogLevel)
	defer log.Sync()

	log.Info("starting monitor", zap.String("listen", cfg.ListenAddr))

	if err := run(cfg, log); err != nil {
		log.Fatal("monitor error", zap.Error(err))
	}
}

func run(cfg config, log *zap.Logger) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// ----- Dashboard state -----
	dashState := dashboard.NewDashboardState(43200)
	dashServer := dashboard.NewServer(dashState, log)

	// ----- Alert manager -----
	am := alerting.NewAlertManager(log)
	for _, rule := range alerting.DefaultRules() {
		am.AddRule(rule)
	}

	// Notifiers.
	logNotifier := alerting.NewLogNotifier(log)
	am.AddNotifier(logNotifier)

	if cfg.WebhookURL != "" {
		am.AddNotifier(alerting.NewWebhookNotifier(cfg.WebhookURL, log))
	}

	if cfg.AlertFile != "" {
		if err := os.MkdirAll("./data", 0o755); err == nil {
			fn, err := alerting.NewFileNotifier(cfg.AlertFile, log)
			if err != nil {
				log.Warn("create file notifier", zap.Error(err))
			} else {
				am.AddNotifier(fn)
			}
		}
	}

	// ----- Portfolio watcher -----
	if cfg.AlpacaAPIKey != "" {
		pw := watcher.NewPortfolioWatcher(watcher.AlpacaConfig{
			APIKey:    cfg.AlpacaAPIKey,
			Secret:    cfg.AlpacaSecret,
			Paper:     cfg.AlpacaPaper,
			PollEvery: cfg.PortfolioPoll,
		}, log)

		pw.AddHandler(func(state alerting.PortfolioState) {
			dashState.UpdatePortfolio(state)
			dashState.AddEquityPoint(dashboard.EquityPoint{
				Timestamp: state.Timestamp,
				Equity:    state.Equity,
				DailyPnL:  state.DailyPnL,
			})

			alerts := am.EvaluatePortfolio(state)
			for _, a := range alerts {
				dashState.AddAlert(a)
				am.SendAlert(a)
			}

			// Push SSE update.
			dashServer.PushSSE(dashState.Snapshot())
		})

		go pw.Run(ctx)
		log.Info("portfolio watcher started")
	} else {
		log.Warn("no Alpaca credentials — portfolio watcher disabled")
	}

	// ----- BH watcher -----
	bhCfg := watcher.DefaultBHConfig()
	bhCfg.StateFilePath = cfg.BHStateFile
	bhCfg.PollEvery = cfg.BHPollEvery

	bhWatcher := watcher.NewBHWatcher(bhCfg, log)
	bhWatcher.AddHandler(func(evt alerting.BHEvent) {
		dashState.UpdateBHMasses(bhWatcher.CurrentMasses())
		alerts := am.EvaluateBHEvent(evt)
		for _, a := range alerts {
			dashState.AddAlert(a)
			am.SendAlert(a)
		}
		dashServer.PushSSE(dashState.Snapshot())
	})
	go bhWatcher.Run(ctx)
	log.Info("bh watcher started", zap.String("state_file", cfg.BHStateFile))

	// Periodic SSE push even without events.
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				dashServer.PushSSE(dashState.Snapshot())
			}
		}
	}()

	// ----- HTTP router -----
	r := chi.NewRouter()
	r.Mount("/", dashServer.Routes())

	srv := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // SSE needs no timeout
		IdleTimeout:  120 * time.Second,
	}

	serverErr := make(chan error, 1)
	go func() {
		log.Info("dashboard listening", zap.String("addr", cfg.ListenAddr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-quit:
		log.Info("signal received", zap.String("signal", sig.String()))
	case err := <-serverErr:
		log.Error("http server error", zap.Error(err))
	}

	cancel()
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Warn("shutdown error", zap.Error(err))
	}
	log.Info("monitor stopped")
	return nil
}

func buildLogger(level string) *zap.Logger {
	lvl := zapcore.InfoLevel
	switch level {
	case "debug":
		lvl = zapcore.DebugLevel
	case "warn":
		lvl = zapcore.WarnLevel
	case "error":
		lvl = zapcore.ErrorLevel
	}
	cfg := zap.NewProductionConfig()
	cfg.Level = zap.NewAtomicLevelAt(lvl)
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	log, err := cfg.Build()
	if err != nil {
		fmt.Fprintf(os.Stderr, "build logger: %v\n", err)
		os.Exit(1)
	}
	return log
}
