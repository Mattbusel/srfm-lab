package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// fullConfig holds all monitor configuration.
type fullConfig struct {
	// Server settings.
	ListenAddr string `yaml:"listen_addr"`

	// Alpaca paper account.
	AlpacaAPIKey  string `yaml:"alpaca_api_key"`
	AlpacaSecret  string `yaml:"alpaca_secret"`
	AlpacaPaper   bool   `yaml:"alpaca_paper"`
	PortfolioPoll string `yaml:"portfolio_poll"` // duration string

	// BH watcher.
	BHStateFile       string  `yaml:"bh_state_file"`
	BHPollEvery       string  `yaml:"bh_poll_every"`
	BHMassThreshold   float64 `yaml:"bh_mass_threshold"`
	BHCollapseThresh  float64 `yaml:"bh_collapse_threshold"`

	// Gateway watcher.
	GatewayURL  string `yaml:"gateway_url"`
	GatewayPoll string `yaml:"gateway_poll"`

	// Alerting.
	AlertFile   string `yaml:"alert_file"`
	WebhookURL  string `yaml:"webhook_url"`
	EventLogFile string `yaml:"event_log_file"`
	MaxAlertHistoryMB int `yaml:"max_alert_history_mb"`

	// Dashboard.
	MaxEquityPoints int `yaml:"max_equity_points"`

	// Logging.
	LogLevel string `yaml:"log_level"`

	// Alert rules (loaded from YAML).
	AlertRules []alertRuleConfig `yaml:"alert_rules"`
}

type alertRuleConfig struct {
	Name      string  `yaml:"name"`
	Symbol    string  `yaml:"symbol"`
	Metric    string  `yaml:"metric"`
	Operator  string  `yaml:"operator"`
	Threshold float64 `yaml:"threshold"`
	Level     string  `yaml:"level"`
	Cooldown  string  `yaml:"cooldown"`
	Message   string  `yaml:"message"`
}

// defaultFullConfig returns defaults for all settings.
func defaultFullConfig() fullConfig {
	return fullConfig{
		ListenAddr:        ":8081",
		AlpacaPaper:       true,
		PortfolioPoll:     "30s",
		BHPollEvery:       "10s",
		BHMassThreshold:   0.7,
		BHCollapseThresh:  0.3,
		GatewayURL:        "http://localhost:8080",
		GatewayPoll:       "10s",
		AlertFile:         "./data/alerts.csv",
		EventLogFile:      "./data/events.jsonl",
		MaxAlertHistoryMB: 50,
		MaxEquityPoints:   43200,
		LogLevel:          "info",
	}
}

// loadFullConfig reads from a YAML file then overlays environment variables.
func loadFullConfig(path string) (fullConfig, error) {
	cfg := defaultFullConfig()

	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil && !os.IsNotExist(err) {
			return cfg, fmt.Errorf("read config %q: %w", path, err)
		}
		if err == nil {
			if err := yaml.Unmarshal(data, &cfg); err != nil {
				return cfg, fmt.Errorf("parse config: %w", err)
			}
		}
	}

	// Environment variable overlay.
	envStr := func(key string, target *string) {
		if v := os.Getenv(key); v != "" {
			*target = v
		}
	}
	envBool := func(key string, target *bool) {
		if v := os.Getenv(key); v != "" {
			*target = strings.ToLower(v) == "true" || v == "1"
		}
	}
	envFloat := func(key string, target *float64) {
		if v := os.Getenv(key); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				*target = f
			}
		}
	}
	envInt := func(key string, target *int) {
		if v := os.Getenv(key); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				*target = n
			}
		}
	}

	envStr("LISTEN_ADDR", &cfg.ListenAddr)
	envStr("ALPACA_API_KEY", &cfg.AlpacaAPIKey)
	envStr("ALPACA_SECRET", &cfg.AlpacaSecret)
	envBool("ALPACA_PAPER", &cfg.AlpacaPaper)
	envStr("BH_STATE_FILE", &cfg.BHStateFile)
	envStr("GATEWAY_URL", &cfg.GatewayURL)
	envStr("ALERT_FILE", &cfg.AlertFile)
	envStr("WEBHOOK_URL", &cfg.WebhookURL)
	envStr("EVENT_LOG_FILE", &cfg.EventLogFile)
	envStr("LOG_LEVEL", &cfg.LogLevel)
	envFloat("BH_MASS_THRESHOLD", &cfg.BHMassThreshold)
	envFloat("BH_COLLAPSE_THRESHOLD", &cfg.BHCollapseThresh)
	envInt("MAX_EQUITY_POINTS", &cfg.MaxEquityPoints)

	return cfg, nil
}

// parseDuration parses a duration string, returning the fallback on error.
func parseDuration(s string, fallback time.Duration) time.Duration {
	if s == "" {
		return fallback
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return fallback
	}
	return d
}
