// Package config provides runtime configuration for the idea-api service.
// Values are read from environment variables with sensible defaults so the
// service works out-of-the-box in local development without any configuration.
package config

import (
	"fmt"
	"os"
	"strconv"
)

// Config holds all runtime configuration for the idea-api.
type Config struct {
	// Port is the TCP port the HTTP server listens on.
	Port int
	// ListenAddr is the full listen address derived from Port.
	ListenAddr string
	// DBPath is the file-system path to idea_engine.db.
	DBPath string
	// BusURL is the base URL of the event bus service (bus/).
	BusURL string
	// LogLevel is one of debug|info|warn|error.
	LogLevel string
	// RateLimitRPS is the per-client request rate limit (requests per second).
	RateLimitRPS float64
	// RateLimitBurst is the per-client burst allowance.
	RateLimitBurst int
}

// Load reads configuration from environment variables, falling back to the
// defaults listed below when a variable is absent or empty.
//
// Environment variables:
//
//	IDEA_API_PORT          default 8767
//	IDEA_ENGINE_DB         default ./idea_engine.db
//	BUS_URL                default http://localhost:8768
//	LOG_LEVEL              default info
//	RATE_LIMIT_RPS         default 60
//	RATE_LIMIT_BURST       default 120
func Load() (*Config, error) {
	port, err := intEnvOr("IDEA_API_PORT", 8767)
	if err != nil {
		return nil, fmt.Errorf("config: IDEA_API_PORT: %w", err)
	}

	rps, err := floatEnvOr("RATE_LIMIT_RPS", 60.0)
	if err != nil {
		return nil, fmt.Errorf("config: RATE_LIMIT_RPS: %w", err)
	}

	burst, err := intEnvOr("RATE_LIMIT_BURST", 120)
	if err != nil {
		return nil, fmt.Errorf("config: RATE_LIMIT_BURST: %w", err)
	}

	cfg := &Config{
		Port:           port,
		ListenAddr:     fmt.Sprintf(":%d", port),
		DBPath:         strEnvOr("IDEA_ENGINE_DB", "./idea_engine.db"),
		BusURL:         strEnvOr("BUS_URL", "http://localhost:8768"),
		LogLevel:       strEnvOr("LOG_LEVEL", "info"),
		RateLimitRPS:   rps,
		RateLimitBurst: burst,
	}
	return cfg, nil
}

// strEnvOr returns the value of the environment variable key, or fallback if
// the variable is not set or is empty.
func strEnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// intEnvOr returns the integer value of key, or fallback if key is unset.
func intEnvOr(key string, fallback int) (int, error) {
	v := os.Getenv(key)
	if v == "" {
		return fallback, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("parse %q as int: %w", v, err)
	}
	return n, nil
}

// floatEnvOr returns the float64 value of key, or fallback if key is unset.
func floatEnvOr(key string, fallback float64) (float64, error) {
	v := os.Getenv(key)
	if v == "" {
		return fallback, nil
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return 0, fmt.Errorf("parse %q as float64: %w", v, err)
	}
	return f, nil
}
