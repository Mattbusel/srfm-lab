// metrics_client.go — Clients for scraping Prometheus, health, and heartbeat endpoints.
//
// MetricsClient: parse Prometheus /metrics text exposition format.
// HealthClient: poll /health JSON and extract check statuses.
// HeartbeatClient: verify trader liveness from :8783.

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ─── PrometheusMetrics ────────────────────────────────────────────────────────

// PrometheusMetrics holds the scraped metric values we care about.
type PrometheusMetrics struct {
	Equity              float64 `json:"equity"`
	DrawdownPct         float64 `json:"drawdown_pct"`
	OpenPositions       int     `json:"open_positions"`
	CircuitBreakerState int     `json:"circuit_breaker_state"`
	BhActiveCount       int     `json:"bh_active_count"`
	DayPnlPct           float64 `json:"day_pnl_pct"`
	ScrapedAt           time.Time `json:"scraped_at"`
	// Raw key-value pairs for all other metrics.
	Raw map[string]float64 `json:"raw,omitempty"`
}

// ─── MetricsClient ────────────────────────────────────────────────────────────

// MetricsClient scrapes a Prometheus /metrics text endpoint.
type MetricsClient struct {
	baseURL string
	client  *http.Client
	logger  *slog.Logger
}

func NewMetricsClient(baseURL string, timeout time.Duration) *MetricsClient {
	return &MetricsClient{
		baseURL: baseURL,
		client:  buildHTTPClient(timeout),
		logger:  slog.Default(),
	}
}

// Scrape fetches and parses /metrics from the Prometheus endpoint.
func (m *MetricsClient) Scrape(ctx context.Context) (*PrometheusMetrics, error) {
	url := m.baseURL + "/metrics"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("metrics request: %w", err)
	}
	req.Header.Set("Accept", "text/plain; version=0.0.4")

	resp, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("metrics scrape: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("prometheus returned HTTP %d", resp.StatusCode)
	}

	return parsePrometheusText(resp.Body)
}

// parsePrometheusText parses Prometheus exposition text format.
// Format: metric_name{labels} value [timestamp]
func parsePrometheusText(r io.Reader) (*PrometheusMetrics, error) {
	pm := &PrometheusMetrics{
		Raw:       make(map[string]float64),
		ScrapedAt: time.Now(),
	}

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") { continue }

		// Split off optional timestamp.
		parts := strings.Fields(line)
		if len(parts) < 2 { continue }

		metricFull := parts[0] // may include {labels}
		valueStr := parts[1]

		// Strip label set: find first '{' and '}', extract base name.
		metricName := metricFull
		if idx := strings.IndexByte(metricFull, '{'); idx >= 0 {
			metricName = metricFull[:idx]
		}

		value, err := strconv.ParseFloat(valueStr, 64)
		if err != nil { continue }

		pm.Raw[metricName] = value

		// Map known metric names to structured fields.
		switch metricName {
		case "srfm_equity", "portfolio_equity", "equity":
			pm.Equity = value
		case "srfm_drawdown_pct", "drawdown_pct", "max_drawdown_pct":
			pm.DrawdownPct = value
		case "srfm_open_positions", "open_positions", "positions_count":
			pm.OpenPositions = int(value)
		case "srfm_circuit_breaker", "circuit_breaker_state", "circuit_breaker_active":
			pm.CircuitBreakerState = int(value)
		case "srfm_bh_active", "bh_active_count", "black_hole_count":
			pm.BhActiveCount = int(value)
		case "srfm_day_pnl_pct", "day_pnl_pct", "daily_pnl_pct":
			pm.DayPnlPct = value
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("parsing prometheus text: %w", err)
	}
	return pm, nil
}

// Get retrieves a specific metric by name from the last scrape.
func (m *MetricsClient) Get(ctx context.Context, name string) (float64, bool, error) {
	pm, err := m.Scrape(ctx)
	if err != nil { return 0, false, err }
	v, ok := pm.Raw[name]
	return v, ok, nil
}

// ─── HealthClient ─────────────────────────────────────────────────────────────

// HealthClient polls a /health endpoint and extracts check statuses.
type HealthClient struct {
	baseURL string
	client  *http.Client
	logger  *slog.Logger
}

func NewHealthClient(baseURL string, timeout time.Duration) *HealthClient {
	return &HealthClient{
		baseURL: baseURL,
		client:  buildHTTPClient(timeout),
		logger:  slog.Default(),
	}
}

// healthResponse is the expected JSON structure from /health.
type healthResponse struct {
	Status string                 `json:"status"`
	Checks map[string]checkDetail `json:"checks,omitempty"`
	// Flat map fallback.
	Services map[string]string `json:"services,omitempty"`
}

type checkDetail struct {
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
	Latency string `json:"latency,omitempty"`
}

// Check polls /health and returns a map of check-name → healthy.
func (h *HealthClient) Check(ctx context.Context) (map[string]bool, error) {
	url := h.baseURL + "/health"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil { return nil, err }
	req.Header.Set("Accept", "application/json")

	resp, err := h.client.Do(req)
	if err != nil { return nil, fmt.Errorf("health check: %w", err) }
	defer resp.Body.Close()

	result := map[string]bool{}

	// Top-level status.
	result["http"] = resp.StatusCode < 300

	var hr healthResponse
	if err := json.NewDecoder(resp.Body).Decode(&hr); err != nil {
		// Non-JSON /health still counts as up if HTTP 200.
		result["overall"] = resp.StatusCode == http.StatusOK
		return result, nil
	}

	result["overall"] = strings.ToLower(hr.Status) == "ok" ||
		strings.ToLower(hr.Status) == "healthy" ||
		strings.ToLower(hr.Status) == "up"

	for name, detail := range hr.Checks {
		ok := strings.ToLower(detail.Status) == "ok" ||
			strings.ToLower(detail.Status) == "healthy" ||
			strings.ToLower(detail.Status) == "pass"
		result[name] = ok
	}

	for name, status := range hr.Services {
		ok := strings.ToLower(status) == "ok" || strings.ToLower(status) == "up"
		result[name] = ok
	}

	return result, nil
}

// IsHealthy returns true if the overall health check passes.
func (h *HealthClient) IsHealthy(ctx context.Context) bool {
	checks, err := h.Check(ctx)
	if err != nil { return false }
	return checks["overall"]
}

// ─── HeartbeatClient ─────────────────────────────────────────────────────────

// HeartbeatClient verifies that the trader process is alive via its heartbeat endpoint.
// It checks :8783 for a JSON body containing {"alive": true} or HTTP 200.
type HeartbeatClient struct {
	baseURL    string
	client     *http.Client
	logger     *slog.Logger
	lastAlive  time.Time
	staleAfter time.Duration
}

func NewHeartbeatClient(baseURL string, timeout time.Duration) *HeartbeatClient {
	return &HeartbeatClient{
		baseURL:    baseURL,
		client:     buildHTTPClient(timeout),
		logger:     slog.Default(),
		staleAfter: 30 * time.Second,
	}
}

type heartbeatResponse struct {
	Alive     bool      `json:"alive"`
	Timestamp time.Time `json:"timestamp,omitempty"`
	Uptime    string    `json:"uptime,omitempty"`
	PID       int       `json:"pid,omitempty"`
}

// IsAlive returns true if the heartbeat endpoint responds and signals liveness.
func (hb *HeartbeatClient) IsAlive(ctx context.Context) bool {
	url := hb.baseURL + "/heartbeat"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil { return false }

	resp, err := hb.client.Do(req)
	if err != nil {
		hb.logger.Debug("heartbeat unreachable", "url", url, "err", err)
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false
	}

	var hbr heartbeatResponse
	if err := json.NewDecoder(resp.Body).Decode(&hbr); err != nil {
		// Plain 200 with non-JSON body counts as alive.
		hb.lastAlive = time.Now()
		return true
	}

	if hbr.Alive {
		hb.lastAlive = time.Now()
	}

	// Accept explicit alive field or recent timestamp.
	if !hbr.Timestamp.IsZero() {
		return time.Since(hbr.Timestamp) < hb.staleAfter
	}
	return hbr.Alive
}

// LastAlive returns the last time the heartbeat was confirmed alive.
func (hb *HeartbeatClient) LastAlive() time.Time { return hb.lastAlive }

// StaleDuration returns how long since the last confirmed heartbeat.
func (hb *HeartbeatClient) StaleDuration() time.Duration {
	if hb.lastAlive.IsZero() { return -1 }
	return time.Since(hb.lastAlive)
}

// ─── Shared HTTP Client Builder ───────────────────────────────────────────────

func buildHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			DialContext: (&net.Dialer{
				Timeout:   3 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			MaxIdleConns:        10,
			MaxIdleConnsPerHost: 5,
			IdleConnTimeout:     90 * time.Second,
		},
	}
}
