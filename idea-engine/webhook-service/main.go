// Package main is the entry point for the IAE Webhook Service.
//
// The webhook service listens on :8770 and receives external signals
// (TradingView alerts, custom webhooks, regime alerts, news events) then
// publishes them to the idea-bus at :8768 so downstream IAE components
// can react in real time.
//
// Environment variables:
//
//	PORT              — listening port (default: 8770)
//	BUS_URL           — idea-bus base URL (default: http://localhost:8768)
//	TV_HMAC_SECRET    — TradingView HMAC-SHA256 shared secret
//	API_KEY           — global API key for simple key-based auth
//	RATE_LIMIT_RPM    — webhooks per minute per IP (default: 60)
//	LOG_LEVEL         — "debug" | "info" | "warn" | "error" (default: "info")
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"srfm-lab/idea-engine/webhook-service/handlers"
	custommw "srfm-lab/idea-engine/webhook-service/middleware"
)

// Config holds all runtime configuration for the service.
type Config struct {
	Port          string
	BusURL        string
	TVHMACSecret  string
	APIKey        string
	RateLimitRPM  int
	LogLevel      string
}

func configFromEnv() Config {
	rpm, err := strconv.Atoi(getenv("RATE_LIMIT_RPM", "60"))
	if err != nil {
		rpm = 60
	}
	return Config{
		Port:         getenv("PORT", "8770"),
		BusURL:       getenv("BUS_URL", "http://localhost:8768"),
		TVHMACSecret: getenv("TV_HMAC_SECRET", ""),
		APIKey:       getenv("API_KEY", ""),
		RateLimitRPM: rpm,
		LogLevel:     getenv("LOG_LEVEL", "info"),
	}
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func buildLogger(level string) *zap.Logger {
	cfg := zap.NewProductionConfig()
	switch level {
	case "debug":
		cfg.Level = zap.NewAtomicLevelAt(zapcore.DebugLevel)
	case "warn":
		cfg.Level = zap.NewAtomicLevelAt(zapcore.WarnLevel)
	case "error":
		cfg.Level = zap.NewAtomicLevelAt(zapcore.ErrorLevel)
	default:
		cfg.Level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	}
	cfg.EncoderConfig.TimeKey = "ts"
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	logger, _ := cfg.Build()
	return logger
}

func main() {
	cfg := configFromEnv()
	logger := buildLogger(cfg.LogLevel)
	defer logger.Sync() //nolint:errcheck

	logger.Info("starting IAE webhook service",
		zap.String("port", cfg.Port),
		zap.String("bus_url", cfg.BusURL),
	)

	// Shared bus client
	busClient := handlers.NewBusClient(cfg.BusURL, logger)

	// Shared metrics registry
	metrics := handlers.NewMetricsRegistry()

	// Handler dependencies
	tvHandler := handlers.NewTradingViewHandler(busClient, logger, metrics, cfg.TVHMACSecret, cfg.RateLimitRPM)
	customHandler := handlers.NewCustomHandler(busClient, logger, metrics)
	healthHandler := handlers.NewHealthHandler(metrics, logger)

	// Auth middleware factory
	authMW := custommw.NewAuthMiddleware(cfg.TVHMACSecret, cfg.APIKey, logger)

	// Router
	r := chi.NewRouter()

	// Global middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(zapLoggerMiddleware(logger))
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(30 * time.Second))

	// Health / metrics — no auth required
	r.Get("/health", healthHandler.Health)
	r.Get("/metrics", healthHandler.Metrics)
	r.Get("/ready", healthHandler.Ready)

	// TradingView webhooks — HMAC auth
	r.Group(func(r chi.Router) {
		r.Use(authMW.HMACOrAPIKey)
		r.Post("/webhooks/tradingview", tvHandler.HandleAlert)
	})

	// Custom webhooks — API key auth
	r.Group(func(r chi.Router) {
		r.Use(authMW.APIKeyOrOpen)
		r.Post("/webhooks/custom", customHandler.HandleCustom)
		r.Post("/webhooks/regime-alert", customHandler.HandleRegimeAlert)
		r.Post("/webhooks/news-event", customHandler.HandleNewsEvent)
	})

	// 404 handler
	r.NotFound(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"not found"}`, http.StatusNotFound)
	})

	srv := &http.Server{
		Addr:         fmt.Sprintf(":%s", cfg.Port),
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		logger.Info("webhook service listening", zap.String("addr", srv.Addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("listen error", zap.Error(err))
		}
	}()

	<-quit
	logger.Info("shutdown signal received, draining connections ...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("server shutdown error", zap.Error(err))
	}
	logger.Info("webhook service stopped")
}

// zapLoggerMiddleware returns a chi-compatible middleware that logs each
// request using the provided zap logger.
func zapLoggerMiddleware(logger *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)
			logger.Info("request",
				zap.String("method", r.Method),
				zap.String("path", r.URL.Path),
				zap.Int("status", ww.Status()),
				zap.Duration("latency", time.Since(start)),
				zap.String("request_id", middleware.GetReqID(r.Context())),
				zap.String("remote_addr", r.RemoteAddr),
			)
		})
	}
}
