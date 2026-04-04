package config

import (
	"fmt"
	"strings"
	"time"
)

// ValidationError holds a list of config validation issues.
type ValidationError struct {
	Errors []string
}

func (e *ValidationError) Error() string {
	return "config validation failed:\n  - " + strings.Join(e.Errors, "\n  - ")
}

// Validate checks the configuration for common mistakes and returns a
// ValidationError if any issues are found.
func (c *Config) Validate() error {
	var errs []string

	// Network.
	if c.ListenAddr == "" {
		errs = append(errs, "listen_addr must not be empty")
	}

	// Cache.
	if c.MaxBarCache <= 0 {
		errs = append(errs, "max_bar_cache must be > 0")
	}
	if c.MaxBarCache > 1_000_000 {
		errs = append(errs, "max_bar_cache > 1M bars may exhaust memory")
	}

	// Symbols.
	if len(c.AllSymbols()) == 0 {
		errs = append(errs, "at least one symbol must be configured")
	}
	for _, sym := range c.AllSymbols() {
		if strings.TrimSpace(sym) == "" {
			errs = append(errs, "symbol list contains empty string")
			break
		}
	}

	// Alpaca.
	if c.EnableAlpaca {
		if c.AlpacaAPIKey == "" {
			errs = append(errs, "alpaca_api_key is required when enable_alpaca=true")
		}
		if c.AlpacaSecret == "" {
			errs = append(errs, "alpaca_secret is required when enable_alpaca=true")
		}
	}

	// Binance.
	if c.EnableBinance {
		if c.BinanceAPIKey == "" {
			errs = append(errs, "binance_api_key is required when enable_binance=true")
		}
		if c.BinanceSecret == "" {
			errs = append(errs, "binance_secret is required when enable_binance=true")
		}
		if len(c.CryptoSymbols) == 0 {
			errs = append(errs, "crypto_symbols must not be empty when enable_binance=true")
		}
	}

	// Simulator.
	if c.EnableSimulator {
		if c.SimulatorMu < -1 || c.SimulatorMu > 1 {
			errs = append(errs, "simulator_mu should be in [-1, 1]")
		}
		if c.SimulatorSigma <= 0 {
			errs = append(errs, "simulator_sigma must be > 0")
		}
		if c.SimulatorSpeedMult <= 0 {
			errs = append(errs, "simulator_speed_mult must be > 0")
		}
		if c.SimulatorSpeedMult > 100000 {
			errs = append(errs, "simulator_speed_mult > 100000 is extremely fast")
		}
		if c.SimulatorVolMean <= 0 {
			errs = append(errs, "simulator_vol_mean must be > 0")
		}
	}

	if !c.EnableAlpaca && !c.EnableBinance && !c.EnableSimulator {
		errs = append(errs, "at least one feed must be enabled (alpaca, binance, or simulator)")
	}

	// Aggregation timeframes.
	valid := map[string]bool{"1m": true, "5m": true, "15m": true, "30m": true, "1h": true, "4h": true, "1d": true}
	for _, tf := range c.AggregationTimeframes {
		if !valid[tf] {
			errs = append(errs, fmt.Sprintf("unknown aggregation timeframe %q", tf))
		}
	}

	// InfluxDB.
	if c.InfluxURL != "" {
		if !strings.HasPrefix(c.InfluxURL, "http://") && !strings.HasPrefix(c.InfluxURL, "https://") {
			errs = append(errs, "influx_url must start with http:// or https://")
		}
	}

	// WebSocket heartbeat.
	if c.WSHeartbeat > 0 && c.WSHeartbeat < 5*time.Second {
		errs = append(errs, "ws_heartbeat < 5s may cause excessive ping traffic")
	}

	if len(errs) > 0 {
		return &ValidationError{Errors: errs}
	}
	return nil
}

// Sanitize applies defaults and normalisation to catch common mistakes.
func (c *Config) Sanitize() {
	if c.ListenAddr == "" {
		c.ListenAddr = ":8080"
	}
	if c.MaxBarCache <= 0 {
		c.MaxBarCache = 10000
	}
	if c.WSHeartbeat == 0 {
		c.WSHeartbeat = 30 * time.Second
	}
	if c.SimulatorSpeedMult == 0 {
		c.SimulatorSpeedMult = 1.0
	}
	if c.SimulatorSigma == 0 {
		c.SimulatorSigma = 0.015
	}
	if c.SimulatorVolMean == 0 {
		c.SimulatorVolMean = 100000
	}
	if c.LogLevel == "" {
		c.LogLevel = "info"
	}
	if len(c.AggregationTimeframes) == 0 {
		c.AggregationTimeframes = []string{"5m", "15m", "1h", "4h", "1d"}
	}
	if c.ParquetDir == "" {
		c.ParquetDir = "./data/parquet"
	}

	// Normalise symbols: trim spaces and upper-case.
	for i, sym := range c.Symbols {
		c.Symbols[i] = strings.TrimSpace(strings.ToUpper(sym))
	}
	for i, sym := range c.CryptoSymbols {
		c.CryptoSymbols[i] = strings.TrimSpace(strings.ToUpper(sym))
	}
}

// Print returns a human-readable representation of the config,
// masking secrets.
func (c *Config) Print() string {
	mask := func(s string) string {
		if s == "" {
			return "(empty)"
		}
		if len(s) <= 8 {
			return "****"
		}
		return s[:4] + "****" + s[len(s)-4:]
	}

	var sb strings.Builder
	sb.WriteString("Gateway Configuration:\n")
	sb.WriteString(fmt.Sprintf("  ListenAddr:        %s\n", c.ListenAddr))
	sb.WriteString(fmt.Sprintf("  LogLevel:          %s\n", c.LogLevel))
	sb.WriteString(fmt.Sprintf("  Symbols:           %v\n", c.Symbols))
	sb.WriteString(fmt.Sprintf("  CryptoSymbols:     %v\n", c.CryptoSymbols))
	sb.WriteString(fmt.Sprintf("  MaxBarCache:       %d\n", c.MaxBarCache))
	sb.WriteString(fmt.Sprintf("  Timeframes:        %v\n", c.AggregationTimeframes))
	sb.WriteString(fmt.Sprintf("  WSHeartbeat:       %s\n", c.WSHeartbeat))
	sb.WriteString(fmt.Sprintf("  ParquetDir:        %s\n", c.ParquetDir))
	sb.WriteString(fmt.Sprintf("  EnableAlpaca:      %v\n", c.EnableAlpaca))
	sb.WriteString(fmt.Sprintf("  AlpacaAPIKey:      %s\n", mask(c.AlpacaAPIKey)))
	sb.WriteString(fmt.Sprintf("  AlpacaSecret:      %s\n", mask(c.AlpacaSecret)))
	sb.WriteString(fmt.Sprintf("  AlpacaPaper:       %v\n", c.AlpacaPaper))
	sb.WriteString(fmt.Sprintf("  EnableBinance:     %v\n", c.EnableBinance))
	sb.WriteString(fmt.Sprintf("  BinanceAPIKey:     %s\n", mask(c.BinanceAPIKey)))
	sb.WriteString(fmt.Sprintf("  EnableSimulator:   %v\n", c.EnableSimulator))
	sb.WriteString(fmt.Sprintf("  SimulatorSpeedMult:%.1f\n", c.SimulatorSpeedMult))
	sb.WriteString(fmt.Sprintf("  SimulatorSigma:    %.4f\n", c.SimulatorSigma))
	sb.WriteString(fmt.Sprintf("  InfluxURL:         %s\n", c.InfluxURL))
	sb.WriteString(fmt.Sprintf("  InfluxOrg:         %s\n", c.InfluxOrg))
	sb.WriteString(fmt.Sprintf("  InfluxBucket:      %s\n", c.InfluxBucket))
	return sb.String()
}
