// Command tsdb is a CLI tool for time series database operations.
// It can write bars from CSV/parquet files to InfluxDB and run analytics queries.
package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/srfm/timeseries/internal/duckdb"
	"github.com/srfm/timeseries/internal/influx"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	influxURL    = flag.String("influx-url", envOrDefault("INFLUX_URL", "http://localhost:8086"), "InfluxDB URL")
	influxToken  = flag.String("influx-token", os.Getenv("INFLUX_TOKEN"), "InfluxDB auth token")
	influxOrg    = flag.String("influx-org", envOrDefault("INFLUX_ORG", "srfm"), "InfluxDB org")
	influxBucket = flag.String("influx-bucket", envOrDefault("INFLUX_BUCKET", "market_data"), "InfluxDB bucket")
	duckProxyURL = flag.String("duckdb-proxy", os.Getenv("DUCKDB_PROXY_URL"), "DuckDB proxy URL")
	logLevel     = flag.String("log-level", "info", "Log level (debug|info|warn|error)")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `Usage: tsdb <command> [flags]

Commands:
  import <csv-glob>           Import CSV bar files to InfluxDB
  query <flux-sql>            Run a Flux query against InfluxDB
  analytics daily <symbol>    Compute daily returns via DuckDB
  analytics sharpe <symbol>   Compute rolling Sharpe ratio
  analytics corr <sym1,sym2>  Compute correlation matrix

Flags:
`)
		flag.PrintDefaults()
	}
	flag.Parse()

	log := buildLogger(*logLevel)
	defer log.Sync()

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	ctx := context.Background()

	influxCfg := influx.Config{
		URL:           *influxURL,
		Token:         *influxToken,
		Org:           *influxOrg,
		Bucket:        *influxBucket,
		BatchSize:     1000,
		FlushInterval: 5 * time.Second,
	}
	influxClient := influx.New(influxCfg, log)
	defer influxClient.Close()

	duckCfg := duckdb.Config{
		ProxyURL: *duckProxyURL,
	}
	analytics, err := duckdb.New(duckCfg, log)
	if err != nil {
		log.Fatal("init duckdb", zap.Error(err))
	}

	switch args[0] {
	case "import":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: tsdb import <csv-glob>")
			os.Exit(1)
		}
		if err := runImport(ctx, args[1], influxClient, log); err != nil {
			log.Fatal("import", zap.Error(err))
		}

	case "query":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: tsdb query <flux-query>")
			os.Exit(1)
		}
		// For query we use the raw flux query via InfluxDB client.
		// Build a quick test query.
		rows, err := analytics.RunQuery(ctx, args[1])
		if err != nil {
			log.Fatal("query", zap.Error(err))
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(rows)

	case "analytics":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: tsdb analytics <subcommand> <symbol>")
			os.Exit(1)
		}
		from := time.Now().AddDate(0, -1, 0)
		to := time.Now()
		switch args[1] {
		case "daily":
			rets, err := analytics.DailyReturns(ctx, args[2], from, to)
			if err != nil {
				log.Fatal("daily returns", zap.Error(err))
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(rets)

		case "sharpe":
			window := 20
			metrics, err := analytics.RollingSharpe(ctx, args[2], window, from, to)
			if err != nil {
				log.Fatal("rolling sharpe", zap.Error(err))
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(metrics)

		case "corr":
			symbols := splitComma(args[2])
			cm, err := analytics.CorrelationMatrix(ctx, symbols, from, to)
			if err != nil {
				log.Fatal("correlation matrix", zap.Error(err))
			}
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			enc.Encode(cm)

		default:
			fmt.Fprintf(os.Stderr, "unknown analytics subcommand: %s\n", args[1])
			os.Exit(1)
		}

	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", args[0])
		flag.Usage()
		os.Exit(1)
	}
}

// runImport reads CSV bar files matching glob and writes them to InfluxDB.
func runImport(ctx context.Context, globPattern string, client *influx.Client, log *zap.Logger) error {
	matches, err := filepath.Glob(globPattern)
	if err != nil {
		return fmt.Errorf("glob %q: %w", globPattern, err)
	}
	if len(matches) == 0 {
		log.Warn("no files matched glob", zap.String("pattern", globPattern))
		return nil
	}

	total := 0
	for _, path := range matches {
		n, err := importCSVFile(path, client, log)
		if err != nil {
			log.Warn("import file", zap.String("path", path), zap.Error(err))
			continue
		}
		total += n
		log.Info("imported file", zap.String("path", path), zap.Int("bars", n))
	}
	log.Info("import complete", zap.Int("total_bars", total), zap.Int("files", len(matches)))
	return nil
}

// importCSVFile reads a single CSV bar file and writes bars to InfluxDB.
func importCSVFile(path string, client *influx.Client, log *zap.Logger) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return 0, fmt.Errorf("read csv: %w", err)
	}

	if len(rows) == 0 {
		return 0, nil
	}

	// Detect column indices from header.
	header := rows[0]
	colIdx := make(map[string]int)
	for i, h := range header {
		colIdx[h] = i
	}
	required := []string{"timestamp", "symbol", "open", "high", "low", "close", "volume"}
	for _, req := range required {
		if _, ok := colIdx[req]; !ok {
			return 0, fmt.Errorf("missing column %q in %s", req, path)
		}
	}

	// Infer timeframe from path (e.g. /data/AAPL/1m/2024-01-15.csv).
	timeframe := inferTimeframe(path)

	count := 0
	for _, row := range rows[1:] {
		if len(row) < len(header) {
			continue
		}
		ts, err := time.Parse(time.RFC3339, row[colIdx["timestamp"]])
		if err != nil {
			continue
		}
		bar := influx.Bar{
			Symbol:    row[colIdx["symbol"]],
			Timestamp: ts,
			Open:      parseF(row[colIdx["open"]]),
			High:      parseF(row[colIdx["high"]]),
			Low:       parseF(row[colIdx["low"]]),
			Close:     parseF(row[colIdx["close"]]),
			Volume:    parseF(row[colIdx["volume"]]),
			Timeframe: timeframe,
		}
		if s, ok := colIdx["source"]; ok {
			bar.Source = row[s]
		}
		client.WriteBar(bar)
		count++
	}
	return count, nil
}

func inferTimeframe(path string) string {
	// Walk path components looking for known timeframe strings.
	parts := filepath.SplitList(path)
	for _, p := range parts {
		switch p {
		case "1m", "5m", "15m", "30m", "1h", "4h", "1d":
			return p
		}
	}
	// Also try splitting by path separator.
	segments := splitPath(path)
	for _, seg := range segments {
		switch seg {
		case "1m", "5m", "15m", "30m", "1h", "4h", "1d":
			return seg
		}
	}
	return "1m"
}

func splitPath(p string) []string {
	var parts []string
	for _, ch := range []byte(p) {
		if ch == '/' || ch == '\\' {
			// we'll split manually
		}
	}
	// Use filepath.Dir approach.
	cur := p
	for cur != "." && cur != "/" && cur != "" {
		base := filepath.Base(cur)
		parts = append(parts, base)
		next := filepath.Dir(cur)
		if next == cur {
			break
		}
		cur = next
	}
	return parts
}

func parseF(s string) float64 {
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func splitComma(s string) []string {
	var out []string
	for _, p := range filepath.SplitList(s) {
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		// Fall back to comma split since filepath.SplitList uses OS list sep.
		for _, p := range splitString(s, ',') {
			if p != "" {
				out = append(out, p)
			}
		}
	}
	return out
}

func splitString(s string, sep rune) []string {
	var parts []string
	start := 0
	for i, c := range s {
		if c == sep {
			parts = append(parts, s[start:i])
			start = i + 1
		}
	}
	parts = append(parts, s[start:])
	return parts
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
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
		fmt.Fprintf(os.Stderr, "logger: %v\n", err)
		os.Exit(1)
	}
	return log
}
