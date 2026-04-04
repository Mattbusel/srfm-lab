// Package influx provides an InfluxDB v2 client wrapper for market data.
package influx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

// Config holds InfluxDB connection settings.
type Config struct {
	URL    string
	Token  string
	Org    string
	Bucket string
	// BatchSize is the number of points to accumulate before flushing.
	BatchSize int
	// FlushInterval is the maximum time between flushes.
	FlushInterval time.Duration
}

// DefaultConfig returns reasonable defaults.
func DefaultConfig() Config {
	return Config{
		URL:           "http://localhost:8086",
		Org:           "srfm",
		Bucket:        "market_data",
		BatchSize:     1000,
		FlushInterval: 5 * time.Second,
	}
}

// Bar mirrors feed.Bar without the import cycle.
type Bar struct {
	Symbol    string
	Timestamp time.Time
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Source    string
	Timeframe string
}

// Trade mirrors feed.Trade.
type Trade struct {
	Symbol    string
	Timestamp time.Time
	Price     float64
	Size      float64
	Side      string
	Source    string
}

// EquityPoint is a single portfolio equity snapshot.
type EquityPoint struct {
	Timestamp time.Time
	Account   string
	Equity    float64
	Cash      float64
	DailyPnL  float64
	Positions map[string]float64 // symbol -> market value
}

// Client wraps the InfluxDB v2 HTTP API.
type Client struct {
	cfg    Config
	log    *zap.Logger
	http   *http.Client
	mu     sync.Mutex
	batch  []string // line protocol lines
	stopCh chan struct{}
	wg     sync.WaitGroup
}

// New creates a new InfluxDB client and starts the background flush goroutine.
func New(cfg Config, log *zap.Logger) *Client {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1000
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = 5 * time.Second
	}
	c := &Client{
		cfg:    cfg,
		log:    log,
		http:   &http.Client{Timeout: 30 * time.Second},
		stopCh: make(chan struct{}),
	}
	c.wg.Add(1)
	go c.flushLoop()
	return c
}

// Close flushes remaining points and stops the background goroutine.
func (c *Client) Close() {
	close(c.stopCh)
	c.wg.Wait()
	c.flushNow(context.Background())
}

// WriteBar enqueues a bar as an InfluxDB line protocol point.
func (c *Client) WriteBar(b Bar) {
	tf := b.Timeframe
	if tf == "" {
		tf = "1m"
	}
	line := fmt.Sprintf(
		"bar,symbol=%s,source=%s,timeframe=%s open=%f,high=%f,low=%f,close=%f,volume=%f %d",
		escapeLPTag(b.Symbol),
		escapeLPTag(b.Source),
		escapeLPTag(tf),
		b.Open, b.High, b.Low, b.Close, b.Volume,
		b.Timestamp.UnixNano(),
	)
	c.enqueue(line)
}

// WriteTrade enqueues a trade as a line protocol point.
func (c *Client) WriteTrade(t Trade) {
	side := t.Side
	if side == "" {
		side = "unknown"
	}
	line := fmt.Sprintf(
		"trade,symbol=%s,source=%s,side=%s price=%f,size=%f %d",
		escapeLPTag(t.Symbol),
		escapeLPTag(t.Source),
		escapeLPTag(side),
		t.Price, t.Size,
		t.Timestamp.UnixNano(),
	)
	c.enqueue(line)
}

// WriteEquity enqueues an equity snapshot.
func (c *Client) WriteEquity(pt EquityPoint) {
	sb := strings.Builder{}
	sb.WriteString(fmt.Sprintf(
		"equity,account=%s equity=%f,cash=%f,daily_pnl=%f",
		escapeLPTag(pt.Account),
		pt.Equity, pt.Cash, pt.DailyPnL,
	))
	for sym, mv := range pt.Positions {
		sb.WriteString(fmt.Sprintf(",position_%s=%f", escapeLPTag(sym), mv))
	}
	sb.WriteString(fmt.Sprintf(" %d", pt.Timestamp.UnixNano()))
	c.enqueue(sb.String())
}

// QueryBars runs a Flux query to retrieve bars.
func (c *Client) QueryBars(ctx context.Context, symbol, timeframe string, from, to time.Time) ([]Bar, error) {
	fluxQuery := fmt.Sprintf(`
from(bucket: "%s")
  |> range(start: %s, stop: %s)
  |> filter(fn: (r) => r._measurement == "bar")
  |> filter(fn: (r) => r.symbol == "%s")
  |> filter(fn: (r) => r.timeframe == "%s")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
`,
		c.cfg.Bucket,
		from.UTC().Format(time.RFC3339),
		to.UTC().Format(time.RFC3339),
		symbol, timeframe,
	)

	rows, err := c.query(ctx, fluxQuery)
	if err != nil {
		return nil, err
	}

	var bars []Bar
	for _, row := range rows {
		ts, _ := time.Parse(time.RFC3339, fmt.Sprintf("%v", row["_time"]))
		bars = append(bars, Bar{
			Symbol:    fmt.Sprintf("%v", row["symbol"]),
			Timestamp: ts,
			Open:      toFloat(row["open"]),
			High:      toFloat(row["high"]),
			Low:       toFloat(row["low"]),
			Close:     toFloat(row["close"]),
			Volume:    toFloat(row["volume"]),
			Source:    fmt.Sprintf("%v", row["source"]),
			Timeframe: timeframe,
		})
	}
	return bars, nil
}

// QueryEquity retrieves equity curve points.
func (c *Client) QueryEquity(ctx context.Context, account string, from, to time.Time) ([]EquityPoint, error) {
	fluxQuery := fmt.Sprintf(`
from(bucket: "%s")
  |> range(start: %s, stop: %s)
  |> filter(fn: (r) => r._measurement == "equity")
  |> filter(fn: (r) => r.account == "%s")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
`,
		c.cfg.Bucket,
		from.UTC().Format(time.RFC3339),
		to.UTC().Format(time.RFC3339),
		account,
	)

	rows, err := c.query(ctx, fluxQuery)
	if err != nil {
		return nil, err
	}

	var points []EquityPoint
	for _, row := range rows {
		ts, _ := time.Parse(time.RFC3339, fmt.Sprintf("%v", row["_time"]))
		points = append(points, EquityPoint{
			Timestamp: ts,
			Account:   account,
			Equity:    toFloat(row["equity"]),
			Cash:      toFloat(row["cash"]),
			DailyPnL:  toFloat(row["daily_pnl"]),
		})
	}
	return points, nil
}

// enqueue adds a line protocol string to the write batch.
func (c *Client) enqueue(line string) {
	c.mu.Lock()
	c.batch = append(c.batch, line)
	shouldFlush := len(c.batch) >= c.cfg.BatchSize
	c.mu.Unlock()

	if shouldFlush {
		go c.flushNow(context.Background())
	}
}

// flushLoop periodically flushes the batch.
func (c *Client) flushLoop() {
	defer c.wg.Done()
	ticker := time.NewTicker(c.cfg.FlushInterval)
	defer ticker.Stop()
	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			c.flushNow(context.Background())
		}
	}
}

// flushNow writes the accumulated batch to InfluxDB.
func (c *Client) flushNow(ctx context.Context) {
	c.mu.Lock()
	if len(c.batch) == 0 {
		c.mu.Unlock()
		return
	}
	lines := c.batch
	c.batch = make([]string, 0, c.cfg.BatchSize)
	c.mu.Unlock()

	body := strings.Join(lines, "\n")
	u := fmt.Sprintf("%s/api/v2/write?org=%s&bucket=%s&precision=ns",
		c.cfg.URL,
		url.QueryEscape(c.cfg.Org),
		url.QueryEscape(c.cfg.Bucket))

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewBufferString(body))
	if err != nil {
		c.log.Warn("influx write request", zap.Error(err))
		return
	}
	req.Header.Set("Authorization", "Token "+c.cfg.Token)
	req.Header.Set("Content-Type", "text/plain; charset=utf-8")

	resp, err := c.http.Do(req)
	if err != nil {
		c.log.Warn("influx write", zap.Error(err), zap.Int("points", len(lines)))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		c.log.Warn("influx write non-2xx",
			zap.Int("status", resp.StatusCode),
			zap.String("body", string(body)))
		return
	}

	c.log.Debug("influx batch flushed", zap.Int("points", len(lines)))
}

// query executes a Flux query and returns rows as maps.
func (c *Client) query(ctx context.Context, fluxQuery string) ([]map[string]interface{}, error) {
	u := fmt.Sprintf("%s/api/v2/query?org=%s", c.cfg.URL, url.QueryEscape(c.cfg.Org))

	reqBody, _ := json.Marshal(map[string]string{"query": fluxQuery, "type": "flux"})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("build query request: %w", err)
	}
	req.Header.Set("Authorization", "Token "+c.cfg.Token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/csv")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("query influx: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("query returned %d: %s", resp.StatusCode, string(body))
	}

	return parseFluxCSV(resp.Body)
}

// parseFluxCSV parses the annotated CSV returned by InfluxDB Flux queries.
func parseFluxCSV(r io.Reader) ([]map[string]interface{}, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(data), "\n")
	var headers []string
	var rows []map[string]interface{}

	for _, line := range lines {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if headers == nil {
			headers = fields
			continue
		}
		if len(fields) != len(headers) {
			continue
		}
		row := make(map[string]interface{}, len(headers))
		for i, h := range headers {
			h = strings.TrimSpace(h)
			v := strings.TrimSpace(fields[i])
			// Try to parse as float.
			if fv, err := strconv.ParseFloat(v, 64); err == nil {
				row[h] = fv
			} else {
				row[h] = v
			}
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// escapeLPTag escapes special characters in InfluxDB line protocol tag values.
func escapeLPTag(s string) string {
	s = strings.ReplaceAll(s, " ", "\\ ")
	s = strings.ReplaceAll(s, ",", "\\,")
	s = strings.ReplaceAll(s, "=", "\\=")
	return s
}

func toFloat(v interface{}) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case string:
		f, _ := strconv.ParseFloat(x, 64)
		return f
	}
	return 0
}
