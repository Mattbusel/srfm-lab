// Package main is the entry point for the risk-aggregator microservice.
// The service listens on :8792 and provides HTTP endpoints that aggregate
// risk metrics from the Python risk API (:8791), the options analytics
// service, and the live-trader position feed.
//
// Graceful shutdown: SIGINT/SIGTERM triggers a 10-second drain window.
package main

import (
	"context"
	"errors"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/srfm-lab/risk-aggregator/aggregator"
	"github.com/srfm-lab/risk-aggregator/handler"
)

// Config holds runtime configuration sourced from environment variables.
type Config struct {
	ListenAddr     string
	RiskAPIBase    string // Python risk API base URL, default http://localhost:8791
	TraderAPIBase  string // Live-trader API base URL, default http://localhost:8790
	OptionsAPIBase string // Options analytics base URL, default http://localhost:8793
	LogLevel       string
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
	IdleTimeout    time.Duration
	ShutdownGrace  time.Duration
}

func configFromEnv() Config {
	getEnvOrDefault := func(key, def string) string {
		if v := os.Getenv(key); v != "" {
			return v
		}
		return def
	}
	return Config{
		ListenAddr:     getEnvOrDefault("RISK_AGGREGATOR_ADDR", ":8792"),
		RiskAPIBase:    getEnvOrDefault("RISK_API_BASE", "http://localhost:8791"),
		TraderAPIBase:  getEnvOrDefault("TRADER_API_BASE", "http://localhost:8790"),
		OptionsAPIBase: getEnvOrDefault("OPTIONS_API_BASE", "http://localhost:8793"),
		LogLevel:       getEnvOrDefault("LOG_LEVEL", "info"),
		ReadTimeout:    15 * time.Second,
		WriteTimeout:   60 * time.Second, // stress tests can be slow
		IdleTimeout:    120 * time.Second,
		ShutdownGrace:  10 * time.Second,
	}
}

func setupLogger(level string) {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	lvl, err := zerolog.ParseLevel(level)
	if err != nil {
		lvl = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(lvl)
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339})
}

// buildRouter wires all HTTP routes onto a gin engine.
func buildRouter(cfg Config, agg *aggregator.Aggregator) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()

	// Structured request logging middleware.
	r.Use(requestLogger())
	r.Use(gin.Recovery())

	// Health / readiness probes.
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok", "ts": time.Now().UTC()})
	})
	r.GET("/ready", func(c *gin.Context) {
		if err := agg.Ping(c.Request.Context()); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"ready": false, "error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, gin.H{"ready": true})
	})

	// Portfolio risk
	portfolioH := handler.NewPortfolioHandler(agg)
	r.GET("/portfolio/risk", portfolioH.GetPortfolioRisk)

	// Stress tests
	stressH := handler.NewStressHandler(agg)
	r.POST("/stress/run", stressH.RunStress)
	r.GET("/stress/scenarios", stressH.ListScenarios)

	// Limit breaches
	limitsH := handler.NewLimitsHandler(agg)
	r.GET("/limits/breaches", limitsH.GetBreaches)
	r.GET("/limits/all", limitsH.GetAllLimits)

	// Attribution
	attrH := handler.NewAttributionHandler(agg)
	r.GET("/attribution/daily", attrH.GetDailyAttribution)

	// Correlation / PCA
	r.GET("/correlation/matrix", func(c *gin.Context) {
		matrix, err := agg.GetCorrelationMatrix(c.Request.Context())
		if err != nil {
			log.Error().Err(err).Msg("failed to compute correlation matrix")
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, matrix)
	})
	r.GET("/correlation/pca", func(c *gin.Context) {
		pca, err := agg.GetPCA(c.Request.Context())
		if err != nil {
			log.Error().Err(err).Msg("failed to compute PCA")
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusOK, pca)
	})

	return r
}

// requestLogger returns a gin middleware that logs each request with zerolog.
func requestLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		c.Next()
		log.Info().
			Str("method", c.Request.Method).
			Str("path", c.Request.URL.Path).
			Int("status", c.Writer.Status()).
			Dur("latency_ms", time.Since(start)).
			Str("client_ip", c.ClientIP()).
			Msg("request")
	}
}

func main() {
	cfg := configFromEnv()
	setupLogger(cfg.LogLevel)

	log.Info().
		Str("addr", cfg.ListenAddr).
		Str("risk_api", cfg.RiskAPIBase).
		Str("trader_api", cfg.TraderAPIBase).
		Msg("starting risk-aggregator")

	// Build the aggregator which holds HTTP client connections to upstream services.
	agg := aggregator.New(aggregator.Config{
		RiskAPIBase:    cfg.RiskAPIBase,
		TraderAPIBase:  cfg.TraderAPIBase,
		OptionsAPIBase: cfg.OptionsAPIBase,
	})

	router := buildRouter(cfg, agg)

	srv := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      router,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		IdleTimeout:  cfg.IdleTimeout,
	}

	// Start the server in a goroutine so we can listen for OS signals.
	serverErr := make(chan error, 1)
	go func() {
		log.Info().Str("addr", cfg.ListenAddr).Msg("HTTP server listening")
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	// Wait for interrupt or server error.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serverErr:
		log.Fatal().Err(err).Msg("server error")
	case sig := <-quit:
		log.Info().Str("signal", sig.String()).Msg("shutdown signal received")
	}

	// Graceful shutdown -- drain in-flight requests.
	ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Error().Err(err).Msg("server forced shutdown")
		os.Exit(1)
	}

	log.Info().Msg("server exited cleanly")
}
