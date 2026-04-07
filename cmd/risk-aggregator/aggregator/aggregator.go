package aggregator

import (
	"context"
	"fmt"
	"time"

	"github.com/go-resty/resty/v2"
)

// Config holds the base URLs for all upstream services.
type Config struct {
	RiskAPIBase    string
	TraderAPIBase  string
	OptionsAPIBase string
}

// Aggregator is the central struct that owns HTTP client connections and
// orchestrates calls to upstream services.
type Aggregator struct {
	cfg        Config
	httpClient *resty.Client
}

// New creates a new Aggregator with a pre-configured resty HTTP client.
func New(cfg Config) *Aggregator {
	client := resty.New().
		SetTimeout(30 * time.Second).
		SetRetryCount(2).
		SetRetryWaitTime(200 * time.Millisecond).
		SetRetryMaxWaitTime(2 * time.Second).
		SetHeader("Content-Type", "application/json").
		SetHeader("Accept", "application/json")

	return &Aggregator{
		cfg:        cfg,
		httpClient: client,
	}
}

// Ping checks that all upstream services are reachable by calling their
// /health endpoints. Returns an error if any are unreachable.
func (a *Aggregator) Ping(ctx context.Context) error {
	endpoints := []struct {
		name string
		url  string
	}{
		{"risk_api", a.cfg.RiskAPIBase + "/health"},
		{"trader_api", a.cfg.TraderAPIBase + "/health"},
		{"options_api", a.cfg.OptionsAPIBase + "/health"},
	}

	for _, ep := range endpoints {
		resp, err := a.httpClient.R().SetContext(ctx).Get(ep.url)
		if err != nil {
			return fmt.Errorf("%s unreachable: %w", ep.name, err)
		}
		if resp.IsError() {
			return fmt.Errorf("%s returned %d", ep.name, resp.StatusCode())
		}
	}
	return nil
}
