package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/srfm/monitor/internal/alerting"
	"go.uber.org/zap"
)

// GatewayConfig holds settings for the gateway health watcher.
type GatewayConfig struct {
	GatewayURL  string
	PollEvery   time.Duration
	StaleBarAge time.Duration // warn if latest bar is older than this
}

// DefaultGatewayConfig returns default gateway watcher settings.
func DefaultGatewayConfig() GatewayConfig {
	return GatewayConfig{
		GatewayURL:  "http://localhost:8080",
		PollEvery:   10 * time.Second,
		StaleBarAge: 5 * time.Minute,
	}
}

// GatewayHealth is a snapshot of gateway health metrics.
type GatewayHealth struct {
	Status          string
	UptimeSeconds   float64
	BarCount        int
	SubscriberCount int
	CacheHitRate    float64
	ActiveSymbols   int
	AsOf            time.Time
}

// GatewayHealthHandler is called when a new health snapshot is available.
type GatewayHealthHandler func(health GatewayHealth)

// GatewayWatcher polls the gateway REST API for health and bar data.
type GatewayWatcher struct {
	cfg      GatewayConfig
	log      *zap.Logger
	client   *http.Client
	mu       sync.Mutex
	handlers []GatewayHealthHandler
	lastBars map[string]time.Time // symbol -> last bar time
}

// NewGatewayWatcher creates a GatewayWatcher.
func NewGatewayWatcher(cfg GatewayConfig, log *zap.Logger) *GatewayWatcher {
	return &GatewayWatcher{
		cfg:      cfg,
		log:      log,
		client:   &http.Client{Timeout: 10 * time.Second},
		lastBars: make(map[string]time.Time),
	}
}

// AddHandler registers a health callback.
func (gw *GatewayWatcher) AddHandler(h GatewayHealthHandler) {
	gw.mu.Lock()
	defer gw.mu.Unlock()
	gw.handlers = append(gw.handlers, h)
}

// Run polls the gateway in a loop. Cancel ctx to stop.
func (gw *GatewayWatcher) Run(ctx context.Context) {
	ticker := time.NewTicker(gw.cfg.PollEvery)
	defer ticker.Stop()

	gw.poll(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			gw.poll(ctx)
		}
	}
}

func (gw *GatewayWatcher) poll(ctx context.Context) {
	health, err := gw.fetchHealth(ctx)
	if err != nil {
		gw.log.Warn("gateway watcher: health fetch failed", zap.Error(err))
		return
	}

	gw.mu.Lock()
	hs := gw.handlers
	gw.mu.Unlock()
	for _, h := range hs {
		h(*health)
	}
}

func (gw *GatewayWatcher) fetchHealth(ctx context.Context) (*GatewayHealth, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		gw.cfg.GatewayURL+"/health", nil)
	if err != nil {
		return nil, err
	}
	resp, err := gw.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gateway health request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gateway health returned %d", resp.StatusCode)
	}

	var raw struct {
		Status          string  `json:"status"`
		UptimeSeconds   float64 `json:"uptime_seconds"`
		BarCount        int     `json:"bar_count"`
		SubscriberCount int     `json:"subscriber_count"`
		CacheHitRate    float64 `json:"cache_hit_rate"`
		ActiveSymbols   int     `json:"active_symbols"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("decode health: %w", err)
	}

	return &GatewayHealth{
		Status:          raw.Status,
		UptimeSeconds:   raw.UptimeSeconds,
		BarCount:        raw.BarCount,
		SubscriberCount: raw.SubscriberCount,
		CacheHitRate:    raw.CacheHitRate,
		ActiveSymbols:   raw.ActiveSymbols,
		AsOf:            time.Now(),
	}, nil
}

// FetchSymbols fetches the list of active symbols from the gateway.
func (gw *GatewayWatcher) FetchSymbols(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		gw.cfg.GatewayURL+"/symbols", nil)
	if err != nil {
		return nil, err
	}
	resp, err := gw.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Symbols []string `json:"symbols"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Symbols, nil
}

// FetchLatestBar fetches the most recent bar for a symbol.
func (gw *GatewayWatcher) FetchLatestBar(ctx context.Context, symbol, timeframe string) (*alerting.Bar, error) {
	now := time.Now()
	from := now.Add(-1 * time.Hour)
	url := fmt.Sprintf("%s/bars/%s/%s?from=%d&to=%d&limit=1",
		gw.cfg.GatewayURL, symbol, timeframe,
		from.Unix(), now.Unix())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := gw.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Bars []struct {
			Symbol    string  `json:"Symbol"`
			Timestamp string  `json:"Timestamp"`
			Open      float64 `json:"Open"`
			High      float64 `json:"High"`
			Low       float64 `json:"Low"`
			Close     float64 `json:"Close"`
			Volume    float64 `json:"Volume"`
			Source    string  `json:"Source"`
		} `json:"bars"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	if len(result.Bars) == 0 {
		return nil, nil
	}
	b := result.Bars[len(result.Bars)-1]
	ts, _ := time.Parse(time.RFC3339, b.Timestamp)
	return &alerting.Bar{
		Symbol:    b.Symbol,
		Timestamp: ts,
		Open:      b.Open,
		High:      b.High,
		Low:       b.Low,
		Close:     b.Close,
		Volume:    b.Volume,
		Source:    b.Source,
	}, nil
}
