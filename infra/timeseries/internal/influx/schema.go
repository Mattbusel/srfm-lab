// Package influx — schema.go documents the InfluxDB measurement schemas and
// provides additional helper write/query methods.
package influx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// MeasurementInfo describes an InfluxDB measurement.
type MeasurementInfo struct {
	Name        string
	Tags        []string
	Fields      []string
	Description string
}

// Measurements returns the schema documentation for all SRFM measurements.
func Measurements() []MeasurementInfo {
	return []MeasurementInfo{
		{
			Name:        "bar",
			Tags:        []string{"symbol", "source", "timeframe"},
			Fields:      []string{"open", "high", "low", "close", "volume"},
			Description: "OHLCV candlestick bar data",
		},
		{
			Name:        "trade",
			Tags:        []string{"symbol", "source", "side"},
			Fields:      []string{"price", "size"},
			Description: "Individual trade executions",
		},
		{
			Name:        "equity",
			Tags:        []string{"account"},
			Fields:      []string{"equity", "cash", "daily_pnl", "position_*"},
			Description: "Portfolio equity snapshots",
		},
		{
			Name:        "quote",
			Tags:        []string{"symbol", "source"},
			Fields:      []string{"bid_price", "bid_size", "ask_price", "ask_size", "spread_bps"},
			Description: "Best bid/ask quotes",
		},
		{
			Name:        "bh_state",
			Tags:        []string{"symbol", "timeframe"},
			Fields:      []string{"mass", "active"},
			Description: "Black-hole engine state",
		},
	}
}

// WriteQuotePoint writes a quote as a line protocol point.
func (c *Client) WriteQuotePoint(symbol, source string, bidPx, bidSz, askPx, askSz float64, ts time.Time) {
	spread := 0.0
	mid := (bidPx + askPx) / 2
	if mid > 0 {
		spread = (askPx - bidPx) / mid * 10000
	}
	line := fmt.Sprintf(
		"quote,symbol=%s,source=%s bid_price=%f,bid_size=%f,ask_price=%f,ask_size=%f,spread_bps=%f %d",
		escapeLPTag(symbol), escapeLPTag(source),
		bidPx, bidSz, askPx, askSz, spread,
		ts.UnixNano(),
	)
	c.enqueue(line)
}

// WriteBHState writes a BH engine state point.
func (c *Client) WriteBHState(symbol, timeframe string, mass float64, active bool, ts time.Time) {
	activeInt := 0
	if active {
		activeInt = 1
	}
	line := fmt.Sprintf(
		"bh_state,symbol=%s,timeframe=%s mass=%f,active=%di %d",
		escapeLPTag(symbol), escapeLPTag(timeframe),
		mass, activeInt,
		ts.UnixNano(),
	)
	c.enqueue(line)
}

// Ping checks connectivity to InfluxDB.
func (c *Client) Ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.cfg.URL+"/ping", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Token "+c.cfg.Token)
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("ping: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 && resp.StatusCode != 204 {
		return fmt.Errorf("ping returned %d", resp.StatusCode)
	}
	return nil
}

// CreateBucket creates the configured bucket if it doesn't exist.
func (c *Client) CreateBucket(ctx context.Context, retentionDays int) error {
	orgID, err := c.getOrgID(ctx)
	if err != nil {
		return fmt.Errorf("get org id: %w", err)
	}

	retentionSecs := retentionDays * 86400
	body := fmt.Sprintf(
		`{"name":"%s","orgID":"%s","retentionRules":[{"type":"expire","everySeconds":%d}]}`,
		c.cfg.Bucket, orgID, retentionSecs,
	)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.cfg.URL+"/api/v2/buckets", bytes.NewBufferString(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Token "+c.cfg.Token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("create bucket: %w", err)
	}
	defer resp.Body.Close()

	// 201 = created, 422 = already exists (both are fine).
	if resp.StatusCode != 201 && resp.StatusCode != 422 {
		return fmt.Errorf("create bucket returned %d", resp.StatusCode)
	}
	return nil
}

func (c *Client) getOrgID(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		c.cfg.URL+"/api/v2/orgs?org="+c.cfg.Org, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Token "+c.cfg.Token)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Orgs []struct {
			ID string `json:"id"`
		} `json:"orgs"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if len(result.Orgs) == 0 {
		return "", fmt.Errorf("org %q not found", c.cfg.Org)
	}
	return result.Orgs[0].ID, nil
}

// QueryLatestBar retrieves the most recent bar for a symbol+timeframe.
func (c *Client) QueryLatestBar(ctx context.Context, symbol, timeframe string) (*Bar, error) {
	to := time.Now()
	from := to.Add(-24 * time.Hour)
	bars, err := c.QueryBars(ctx, symbol, timeframe, from, to)
	if err != nil {
		return nil, err
	}
	if len(bars) == 0 {
		return nil, nil
	}
	b := bars[len(bars)-1]
	return &b, nil
}

// QueryVolume returns the total volume for a symbol+timeframe in a time range.
func (c *Client) QueryVolume(ctx context.Context, symbol, timeframe string, from, to time.Time) (float64, error) {
	q := fmt.Sprintf(`
from(bucket: "%s")
  |> range(start: %s, stop: %s)
  |> filter(fn: (r) => r._measurement == "bar")
  |> filter(fn: (r) => r.symbol == "%s")
  |> filter(fn: (r) => r.timeframe == "%s")
  |> filter(fn: (r) => r._field == "volume")
  |> sum()
`,
		c.cfg.Bucket,
		from.UTC().Format(time.RFC3339),
		to.UTC().Format(time.RFC3339),
		symbol, timeframe,
	)

	rows, err := c.query(ctx, q)
	if err != nil {
		return 0, err
	}
	if len(rows) == 0 {
		return 0, nil
	}
	return toFloat(rows[0]["_value"]), nil
}

// QueryBHHistory retrieves BH mass history for a symbol+timeframe.
func (c *Client) QueryBHHistory(ctx context.Context, symbol, timeframe string, from, to time.Time) ([]struct {
	Timestamp time.Time
	Mass      float64
	Active    bool
}, error) {
	q := fmt.Sprintf(`
from(bucket: "%s")
  |> range(start: %s, stop: %s)
  |> filter(fn: (r) => r._measurement == "bh_state")
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

	rows, err := c.query(ctx, q)
	if err != nil {
		return nil, err
	}

	type point struct {
		Timestamp time.Time
		Mass      float64
		Active    bool
	}
	var out []point
	for _, row := range rows {
		ts, _ := time.Parse(time.RFC3339, fmt.Sprintf("%v", row["_time"]))
		active := toFloat(row["active"]) > 0.5
		out = append(out, point{
			Timestamp: ts,
			Mass:      toFloat(row["mass"]),
			Active:    active,
		})
	}
	return out, nil
}

// dummyBodyClose is a helper to drain and close response bodies.
func dummyBodyClose(rc io.ReadCloser) {
	io.Copy(io.Discard, rc) //nolint:errcheck
	rc.Close()
}
