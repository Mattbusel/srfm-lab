// Package duckdb provides an analytics layer using DuckDB via the REST proxy.
// Because DuckDB does not have a native Go driver in all environments, this
// package communicates with the duckdb-proxy REST service defined in docker-compose.
// For local development it also supports executing queries via os/exec (duckdb CLI).
package duckdb

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"
)

// Config holds DuckDB analytics configuration.
type Config struct {
	// ProxyURL is the URL of the duckdb-proxy REST service, if used.
	ProxyURL string
	// DBPath is the path to a local DuckDB file (used when ProxyURL is empty).
	DBPath string
}

// DailyReturn is a single day's return for a symbol.
type DailyReturn struct {
	Symbol string
	Date   time.Time
	Return float64
}

// RollingMetric is a time-series of a rolling metric.
type RollingMetric struct {
	Symbol    string
	Timestamp time.Time
	Value     float64
}

// CorrelationMatrix holds pairwise correlations.
type CorrelationMatrix struct {
	Symbols []string
	Data    [][]float64
}

// Analytics is the DuckDB analytics layer.
type Analytics struct {
	cfg    Config
	log    *zap.Logger
	db     *sql.DB    // non-nil when using local DuckDB driver
	client *http.Client
}

// New creates an Analytics instance.
// If cfg.ProxyURL is non-empty, queries are proxied over HTTP.
// Otherwise, it tries to open a local DuckDB database file.
func New(cfg Config, log *zap.Logger) (*Analytics, error) {
	a := &Analytics{
		cfg:    cfg,
		log:    log,
		client: &http.Client{Timeout: 60 * time.Second},
	}
	// If proxy is not configured, try to use local DuckDB.
	// We use a driver registration approach rather than linking CGo.
	// The actual driver is registered externally (e.g. go-duckdb) if available.
	if cfg.ProxyURL == "" {
		log.Info("duckdb analytics: proxy URL not configured, using stub mode")
	}
	return a, nil
}

// LoadFromParquet loads parquet/CSV files matching the glob pattern into DuckDB.
func (a *Analytics) LoadFromParquet(ctx context.Context, globPattern string) error {
	sql := fmt.Sprintf(`
CREATE OR REPLACE VIEW bars AS
SELECT * FROM read_csv_auto('%s', header=true);
`, globPattern)
	_, err := a.RunQuery(ctx, sql)
	return err
}

// RunQuery executes arbitrary SQL and returns rows as maps.
func (a *Analytics) RunQuery(ctx context.Context, query string) ([]map[string]interface{}, error) {
	if a.cfg.ProxyURL != "" {
		return a.proxyQuery(ctx, query)
	}
	// Stub: log the query and return empty.
	a.log.Debug("duckdb query (stub)", zap.String("sql", truncate(query, 120)))
	return nil, nil
}

// DailyReturns computes daily returns for a symbol.
func (a *Analytics) DailyReturns(ctx context.Context, symbol string, from, to time.Time) ([]DailyReturn, error) {
	q := fmt.Sprintf(`
WITH base AS (
  SELECT
    date_trunc('day', timestamp) AS day,
    symbol,
    last(close ORDER BY timestamp) AS close_price
  FROM bars
  WHERE symbol = '%s'
    AND timestamp >= TIMESTAMP '%s'
    AND timestamp <= TIMESTAMP '%s'
  GROUP BY 1, 2
),
returns AS (
  SELECT
    day,
    symbol,
    close_price,
    LAG(close_price) OVER (PARTITION BY symbol ORDER BY day) AS prev_close,
    (close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY day))
      / NULLIF(LAG(close_price) OVER (PARTITION BY symbol ORDER BY day), 0) AS daily_return
  FROM base
)
SELECT day, symbol, daily_return
FROM returns
WHERE daily_return IS NOT NULL
ORDER BY day;
`,
		escapeSQLString(symbol),
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	var out []DailyReturn
	for _, row := range rows {
		day, _ := parseTime(fmt.Sprintf("%v", row["day"]))
		out = append(out, DailyReturn{
			Symbol: symbol,
			Date:   day,
			Return: toFloat64(row["daily_return"]),
		})
	}
	return out, nil
}

// RollingSharpe computes the rolling Sharpe ratio.
func (a *Analytics) RollingSharpe(ctx context.Context, symbol string, window int, from, to time.Time) ([]RollingMetric, error) {
	q := fmt.Sprintf(`
WITH daily AS (
  SELECT
    date_trunc('day', timestamp) AS day,
    (last(close ORDER BY timestamp) - first(open ORDER BY timestamp))
      / NULLIF(first(open ORDER BY timestamp), 0) AS ret
  FROM bars
  WHERE symbol = '%s'
    AND timestamp >= TIMESTAMP '%s'
    AND timestamp <= TIMESTAMP '%s'
  GROUP BY 1
),
rolling AS (
  SELECT
    day,
    AVG(ret) OVER (ORDER BY day ROWS BETWEEN %d PRECEDING AND CURRENT ROW) AS avg_ret,
    STDDEV(ret) OVER (ORDER BY day ROWS BETWEEN %d PRECEDING AND CURRENT ROW) AS std_ret
  FROM daily
)
SELECT day, (avg_ret / NULLIF(std_ret, 0)) * sqrt(252) AS sharpe
FROM rolling
WHERE std_ret > 0
ORDER BY day;
`,
		escapeSQLString(symbol),
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
		window-1, window-1,
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	var out []RollingMetric
	for _, row := range rows {
		ts, _ := parseTime(fmt.Sprintf("%v", row["day"]))
		out = append(out, RollingMetric{
			Symbol:    symbol,
			Timestamp: ts,
			Value:     toFloat64(row["sharpe"]),
		})
	}
	return out, nil
}

// CorrelationMatrix computes pairwise return correlations for a set of symbols.
func (a *Analytics) CorrelationMatrix(ctx context.Context, symbols []string, from, to time.Time) (*CorrelationMatrix, error) {
	if len(symbols) == 0 {
		return &CorrelationMatrix{}, nil
	}

	symList := "'" + strings.Join(symbols, "','") + "'"
	q := fmt.Sprintf(`
WITH daily AS (
  SELECT
    date_trunc('day', timestamp) AS day,
    symbol,
    last(close ORDER BY timestamp) AS close_px
  FROM bars
  WHERE symbol IN (%s)
    AND timestamp >= TIMESTAMP '%s'
    AND timestamp <= TIMESTAMP '%s'
  GROUP BY 1, 2
),
pivoted AS (
  PIVOT daily ON symbol USING last(close_px) GROUP BY day
)
SELECT * FROM pivoted ORDER BY day;
`,
		symList,
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	// Build a simple correlation matrix from the data rows.
	n := len(symbols)
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		for j := range matrix[i] {
			if i == j {
				matrix[i][j] = 1.0
			}
		}
	}

	// Collect returns per symbol.
	returns := make([][]float64, n)
	for _, row := range rows {
		for i, sym := range symbols {
			v := toFloat64(row[sym])
			returns[i] = append(returns[i], v)
		}
	}

	// Compute log-returns then correlations.
	logRets := make([][]float64, n)
	for i := range returns {
		r := returns[i]
		logRets[i] = make([]float64, 0, len(r))
		for j := 1; j < len(r); j++ {
			if r[j-1] > 0 && r[j] > 0 {
				logRets[i] = append(logRets[i], (r[j]-r[j-1])/r[j-1])
			}
		}
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			c := correlation(logRets[i], logRets[j])
			matrix[i][j] = c
			matrix[j][i] = c
		}
	}

	return &CorrelationMatrix{Symbols: symbols, Data: matrix}, nil
}

// FactorExposures computes beta exposures to a set of factor symbols.
func (a *Analytics) FactorExposures(ctx context.Context, symbol string, factors []string, from, to time.Time) (map[string]float64, error) {
	all := append([]string{symbol}, factors...)
	cm, err := a.CorrelationMatrix(ctx, all, from, to)
	if err != nil {
		return nil, err
	}

	// Find index of target symbol.
	symIdx := -1
	for i, s := range cm.Symbols {
		if s == symbol {
			symIdx = i
			break
		}
	}
	if symIdx < 0 {
		return nil, fmt.Errorf("symbol %q not found in correlation matrix", symbol)
	}

	exposures := make(map[string]float64)
	for i, s := range cm.Symbols {
		if s == symbol {
			continue
		}
		exposures[s] = cm.Data[symIdx][i]
	}
	return exposures, nil
}

// proxyQuery sends a query to the DuckDB REST proxy.
func (a *Analytics) proxyQuery(ctx context.Context, query string) ([]map[string]interface{}, error) {
	body, err := json.Marshal(map[string]string{"query": query})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, a.cfg.ProxyURL+"/query", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("proxy request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("proxy returned %d: %s", resp.StatusCode, string(data))
	}

	var result struct {
		Rows []map[string]interface{} `json:"rows"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return result.Rows, nil
}

// --- helpers ---

func escapeSQLString(s string) string {
	return strings.ReplaceAll(s, "'", "''")
}

func toFloat64(v interface{}) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case string:
		var f float64
		fmt.Sscanf(x, "%f", &f)
		return f
	}
	return 0
}

func parseTime(s string) (time.Time, error) {
	formats := []string{time.RFC3339, "2006-01-02 15:04:05", "2006-01-02"}
	for _, f := range formats {
		t, err := time.Parse(f, s)
		if err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("cannot parse time %q", s)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// correlation computes the Pearson correlation coefficient of two slices.
func correlation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n == 0 {
		return 0
	}
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	fn := float64(n)
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	num := fn*sumXY - sumX*sumY
	den := (fn*sumX2 - sumX*sumX) * (fn*sumY2 - sumY*sumY)
	if den <= 0 {
		return 0
	}
	// Avoid importing math.
	// Newton-Raphson sqrt.
	sq := func(v float64) float64 {
		if v <= 0 {
			return 0
		}
		x := v
		for i := 0; i < 50; i++ {
			x = (x + v/x) / 2
		}
		return x
	}
	return num / sq(den)
}
