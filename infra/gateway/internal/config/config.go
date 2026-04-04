// Package config loads gateway configuration from environment variables and
// an optional YAML file. Environment variables override file values.
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration structure for the gateway.
type Config struct {
	// Alpaca credentials and settings.
	AlpacaAPIKey string `yaml:"alpaca_api_key"`
	AlpacaSecret string `yaml:"alpaca_secret"`
	AlpacaPaper  bool   `yaml:"alpaca_paper"`

	// Binance credentials.
	BinanceAPIKey string `yaml:"binance_api_key"`
	BinanceSecret string `yaml:"binance_secret"`

	// Feed enable flags.
	EnableAlpaca   bool `yaml:"enable_alpaca"`
	EnableBinance  bool `yaml:"enable_binance"`
	EnableSimulator bool `yaml:"enable_simulator"`

	// Simulator settings.
	SimulatorMu        float64 `yaml:"simulator_mu"`
	SimulatorSigma     float64 `yaml:"simulator_sigma"`
	SimulatorTickSize  float64 `yaml:"simulator_tick_size"`
	SimulatorVolMean   float64 `yaml:"simulator_vol_mean"`
	SimulatorSpeedMult float64 `yaml:"simulator_speed_mult"` // 1=realtime, 1000=fast

	// Symbols to subscribe to across all feeds.
	Symbols []string `yaml:"symbols"`

	// CryptoSymbols for Binance / Alpaca-crypto feeds.
	CryptoSymbols []string `yaml:"crypto_symbols"`

	// ListenAddr is the HTTP/WS server listen address.
	ListenAddr string `yaml:"listen_addr"`

	// MaxBarCache is the maximum number of bars stored per symbol per timeframe.
	MaxBarCache int `yaml:"max_bar_cache"`

	// AggregationTimeframes are the target bar sizes above 1m.
	// Accepted values: "5m", "15m", "1h", "4h", "1d".
	AggregationTimeframes []string `yaml:"aggregation_timeframes"`

	// ParquetDir is the root directory for parquet files.
	ParquetDir string `yaml:"parquet_dir"`

	// InfluxDB settings.
	InfluxURL    string `yaml:"influx_url"`
	InfluxToken  string `yaml:"influx_token"`
	InfluxOrg    string `yaml:"influx_org"`
	InfluxBucket string `yaml:"influx_bucket"`

	// WebSocket heartbeat interval.
	WSHeartbeat time.Duration `yaml:"ws_heartbeat"`

	// LogLevel is one of debug, info, warn, error.
	LogLevel string `yaml:"log_level"`
}

// Default returns a Config populated with sensible defaults.
func Default() *Config {
	return &Config{
		ListenAddr:            ":8080",
		MaxBarCache:           10000,
		EnableSimulator:       true,
		SimulatorMu:           0.0001,
		SimulatorSigma:        0.015,
		SimulatorTickSize:     0.01,
		SimulatorVolMean:      100000,
		SimulatorSpeedMult:    1.0,
		Symbols:               []string{"AAPL", "MSFT", "SPY"},
		CryptoSymbols:         []string{"BTCUSDT", "ETHUSDT"},
		AggregationTimeframes: []string{"5m", "15m", "1h", "4h", "1d"},
		ParquetDir:            "./data/parquet",
		InfluxURL:             "http://localhost:8086",
		InfluxOrg:             "srfm",
		InfluxBucket:          "market_data",
		WSHeartbeat:           30 * time.Second,
		LogLevel:              "info",
	}
}

// Load reads config from an optional YAML file then overlays environment variables.
func Load(path string) (*Config, error) {
	cfg := Default()

	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("reading config file %q: %w", path, err)
		}
		if err == nil {
			if err := yaml.Unmarshal(data, cfg); err != nil {
				return nil, fmt.Errorf("parsing config file %q: %w", path, err)
			}
		}
	}

	// Overlay environment variables.
	if v := os.Getenv("ALPACA_API_KEY"); v != "" {
		cfg.AlpacaAPIKey = v
	}
	if v := os.Getenv("ALPACA_SECRET"); v != "" {
		cfg.AlpacaSecret = v
	}
	if v := os.Getenv("ALPACA_PAPER"); v != "" {
		cfg.AlpacaPaper = strings.ToLower(v) == "true" || v == "1"
	}
	if v := os.Getenv("BINANCE_API_KEY"); v != "" {
		cfg.BinanceAPIKey = v
	}
	if v := os.Getenv("BINANCE_SECRET"); v != "" {
		cfg.BinanceSecret = v
	}
	if v := os.Getenv("LISTEN_ADDR"); v != "" {
		cfg.ListenAddr = v
	}
	if v := os.Getenv("MAX_BAR_CACHE"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("invalid MAX_BAR_CACHE %q: %w", v, err)
		}
		cfg.MaxBarCache = n
	}
	if v := os.Getenv("SYMBOLS"); v != "" {
		cfg.Symbols = strings.Split(v, ",")
	}
	if v := os.Getenv("CRYPTO_SYMBOLS"); v != "" {
		cfg.CryptoSymbols = strings.Split(v, ",")
	}
	if v := os.Getenv("ENABLE_ALPACA"); v != "" {
		cfg.EnableAlpaca = strings.ToLower(v) == "true" || v == "1"
	}
	if v := os.Getenv("ENABLE_BINANCE"); v != "" {
		cfg.EnableBinance = strings.ToLower(v) == "true" || v == "1"
	}
	if v := os.Getenv("ENABLE_SIMULATOR"); v != "" {
		cfg.EnableSimulator = strings.ToLower(v) == "true" || v == "1"
	}
	if v := os.Getenv("INFLUX_URL"); v != "" {
		cfg.InfluxURL = v
	}
	if v := os.Getenv("INFLUX_TOKEN"); v != "" {
		cfg.InfluxToken = v
	}
	if v := os.Getenv("INFLUX_ORG"); v != "" {
		cfg.InfluxOrg = v
	}
	if v := os.Getenv("INFLUX_BUCKET"); v != "" {
		cfg.InfluxBucket = v
	}
	if v := os.Getenv("PARQUET_DIR"); v != "" {
		cfg.ParquetDir = v
	}
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}

	return cfg, nil
}

// AllSymbols returns the combined list of equity + crypto symbols.
func (c *Config) AllSymbols() []string {
	seen := make(map[string]struct{})
	out := make([]string, 0, len(c.Symbols)+len(c.CryptoSymbols))
	for _, s := range c.Symbols {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	for _, s := range c.CryptoSymbols {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	return out
}
