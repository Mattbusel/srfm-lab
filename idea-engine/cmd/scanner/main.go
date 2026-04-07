// Command scanner runs signal discovery scans over a universe of symbols and
// writes a JSON report of the candidates that pass quality gates.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"srfm-lab/idea-engine/pkg/signal_discovery"
)

// flags holds the parsed command-line options.
type flags struct {
	symbols        string
	start          string
	end            string
	outputJSON     string
	minICIR        float64
	maxCorrelation float64
	workers        int
	seed           int64
}

func main() {
	f := parseFlags()
	if f.symbols == "" {
		log.Fatal("--symbols is required")
	}
	if f.start == "" || f.end == "" {
		log.Fatal("--start and --end are required")
	}

	syms := splitSymbols(f.symbols)
	if len(syms) == 0 {
		log.Fatal("--symbols produced an empty list")
	}

	cfg := signal_discovery.DefaultScannerConfig()
	cfg.MinICIR = f.minICIR
	cfg.AntiCorrThreshold = f.maxCorrelation
	cfg.Workers = f.workers
	cfg.RandSeed = f.seed

	seeds := buildDefaultSeeds()
	scanner := signal_discovery.NewSignalScanner(cfg, seeds)

	// Check the research DB for existing signals and set them as the
	// anti-correlation reference set.
	existing, err := loadExistingSignals()
	if err != nil {
		log.Printf("warning: could not load existing signals: %v", err)
	} else {
		scanner.SetExisting(existing)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle SIGINT / SIGTERM for graceful shutdown.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("received shutdown signal, cancelling scan")
		cancel()
	}()

	log.Printf("scanning %d symbols from %s to %s", len(syms), f.start, f.end)
	startTime := time.Now()

	candidates, err := scanner.ScanUniverse(ctx, syms, f.start, f.end, mockBarProvider)
	if err != nil {
		log.Fatalf("scan failed: %v", err)
	}

	elapsed := time.Since(startTime)
	log.Printf("scan complete: %d candidates passed, elapsed %s", len(candidates), elapsed)

	// Write JSON output.
	outputPath := f.outputJSON
	if outputPath == "" {
		outputPath = fmt.Sprintf("scan_%s.json", time.Now().Format("20060102_150405"))
	}
	if err := GenerateScanReport(candidates, outputPath); err != nil {
		log.Fatalf("generate report: %v", err)
	}
	log.Printf("report written to %s", outputPath)
}

// parseFlags parses the command-line flags and returns a flags struct.
func parseFlags() flags {
	var f flags
	flag.StringVar(&f.symbols, "symbols", "", "Comma-separated list of symbols to scan")
	flag.StringVar(&f.start, "start", "", "Start date (YYYY-MM-DD)")
	flag.StringVar(&f.end, "end", "", "End date (YYYY-MM-DD)")
	flag.StringVar(&f.outputJSON, "output-json", "", "Path for JSON output (default: scan_<timestamp>.json)")
	flag.Float64Var(&f.minICIR, "min-icir", 0.3, "Minimum ICIR for a candidate to pass")
	flag.Float64Var(&f.maxCorrelation, "max-correlation", 0.7, "Maximum absolute correlation to existing signals")
	flag.IntVar(&f.workers, "workers", 4, "Number of parallel evaluation workers")
	flag.Int64Var(&f.seed, "seed", 42, "Random seed for reproducibility")
	flag.Parse()
	return f
}

// splitSymbols splits a comma-separated symbol string into a slice.
func splitSymbols(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// buildDefaultSeeds returns the set of known-good seed signals for the scanner.
// In production these would be loaded from the signal library database.
func buildDefaultSeeds() []signal_discovery.SignalCandidate {
	return []signal_discovery.SignalCandidate{
		{
			Name:    "momentum_20d",
			Formula: "momentum(close, 20)",
			Params:  map[string]float64{"lookback": 20, "scale": 1.0},
		},
		{
			Name:    "momentum_60d",
			Formula: "momentum(close, 60)",
			Params:  map[string]float64{"lookback": 60, "scale": 1.0},
		},
		{
			Name:    "mean_reversion_10d",
			Formula: "mean_reversion(close, 10)",
			Params:  map[string]float64{"lookback": 10, "scale": -1.0},
		},
		{
			Name:    "volatility_adj_momentum_20d",
			Formula: "vol_adj_momentum(close, 20)",
			Params:  map[string]float64{"lookback": 20, "vol_lookback": 20, "scale": 1.0},
		},
	}
}

// loadExistingSignals reads the top-N signals from the research database and
// returns them as SignalCandidate values. In production this queries the signal
// library; here we return an empty slice to avoid a hard dependency on the DB.
func loadExistingSignals() ([]signal_discovery.SignalCandidate, error) {
	// No research DB wired up yet -- return empty to skip the portfolio filter.
	return nil, nil
}

// mockBarProvider generates synthetic OHLCV bars for testing.
// In production this would query the market data store.
func mockBarProvider(symbol, start, end string) ([]signal_discovery.Bar, error) {
	rng := rand.New(rand.NewSource(hashSymbol(symbol)))
	startT, err := time.Parse("2006-01-02", start)
	if err != nil {
		return nil, fmt.Errorf("parse start date: %w", err)
	}
	endT, err := time.Parse("2006-01-02", end)
	if err != nil {
		return nil, fmt.Errorf("parse end date: %w", err)
	}

	var bars []signal_discovery.Bar
	price := 100.0
	for t := startT; !t.After(endT); t = t.AddDate(0, 0, 1) {
		// Skip weekends (simplified).
		if t.Weekday() == time.Saturday || t.Weekday() == time.Sunday {
			continue
		}
		ret := rng.NormFloat64() * 0.01
		open := price
		price *= (1 + ret)
		bars = append(bars, signal_discovery.Bar{
			Symbol:    symbol,
			Timestamp: t,
			Open:      open,
			High:      price * (1 + rng.Float64()*0.005),
			Low:       price * (1 - rng.Float64()*0.005),
			Close:     price,
			Volume:    1e6 * (0.5 + rng.Float64()),
			Return:    ret,
		})
	}
	return bars, nil
}

// hashSymbol returns a deterministic int64 seed from a symbol string.
func hashSymbol(sym string) int64 {
	var h int64 = 5381
	for _, c := range sym {
		h = h*33 + int64(c)
	}
	return h
}

// jsonExport holds the machine-readable output for a scan run.
type jsonExport struct {
	GeneratedAt string                          `json:"generated_at"`
	Candidates  []signal_discovery.SignalCandidate `json:"candidates"`
}

// exportJSON writes candidates to a JSON file at outputPath.
func exportJSON(candidates []signal_discovery.SignalCandidate, outputPath string) error {
	// Strip Series from JSON output (it is large and not useful downstream).
	clean := make([]signal_discovery.SignalCandidate, len(candidates))
	for i, c := range candidates {
		c.Series = nil
		clean[i] = c
	}
	payload := jsonExport{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Candidates:  clean,
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal JSON: %w", err)
	}
	if err := os.WriteFile(outputPath, data, 0644); err != nil {
		return fmt.Errorf("write JSON: %w", err)
	}
	return nil
}
