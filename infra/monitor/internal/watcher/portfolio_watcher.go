// Package watcher monitors external state sources and emits structured events.
package watcher

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/srfm/monitor/internal/alerting"
	"go.uber.org/zap"
)

// AlpacaConfig holds credentials for the Alpaca paper API.
type AlpacaConfig struct {
	APIKey    string
	Secret    string
	Paper     bool
	PollEvery time.Duration
}

// alpacaAccount is the wire format for GET /v2/account.
type alpacaAccount struct {
	Equity          string `json:"equity"`
	Cash            string `json:"cash"`
	PortfolioValue  string `json:"portfolio_value"`
	LastEquity      string `json:"last_equity"`
	BuyingPower     string `json:"buying_power"`
	DaytradingBuyingPower string `json:"daytrading_buying_power"`
}

// alpacaPosition is the wire format for GET /v2/positions.
type alpacaPosition struct {
	Symbol        string `json:"symbol"`
	Qty           string `json:"qty"`
	AvgEntryPrice string `json:"avg_entry_price"`
	MarketValue   string `json:"market_value"`
	UnrealizedPL  string `json:"unrealized_pl"`
	Side          string `json:"side"`
}

// PortfolioStateHandler is called each time a new PortfolioState is computed.
type PortfolioStateHandler func(state alerting.PortfolioState)

// PortfolioWatcher polls the Alpaca paper API and emits PortfolioState events.
type PortfolioWatcher struct {
	cfg      AlpacaConfig
	log      *zap.Logger
	handlers []PortfolioStateHandler
	mu       sync.Mutex
	client   *http.Client

	// Rolling state for drawdown computation.
	dailyHigh     float64
	dailyPnLStart float64
	lastEquity    float64
	dayStarted    time.Time
}

// NewPortfolioWatcher creates a PortfolioWatcher.
func NewPortfolioWatcher(cfg AlpacaConfig, log *zap.Logger) *PortfolioWatcher {
	if cfg.PollEvery == 0 {
		cfg.PollEvery = 30 * time.Second
	}
	return &PortfolioWatcher{
		cfg:    cfg,
		log:    log,
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// AddHandler registers a callback for new PortfolioState snapshots.
func (pw *PortfolioWatcher) AddHandler(h PortfolioStateHandler) {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	pw.handlers = append(pw.handlers, h)
}

// Run starts polling. Call in a goroutine; cancel ctx to stop.
func (pw *PortfolioWatcher) Run(ctx context.Context) {
	ticker := time.NewTicker(pw.cfg.PollEvery)
	defer ticker.Stop()

	// Poll immediately on start.
	pw.poll(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pw.poll(ctx)
		}
	}
}

func (pw *PortfolioWatcher) poll(ctx context.Context) {
	account, err := pw.fetchAccount(ctx)
	if err != nil {
		pw.log.Warn("portfolio watcher: fetch account", zap.Error(err))
		return
	}
	positions, err := pw.fetchPositions(ctx)
	if err != nil {
		pw.log.Warn("portfolio watcher: fetch positions", zap.Error(err))
		return
	}

	equity := parseFloat(account.Equity)
	cash := parseFloat(account.Cash)
	lastEquity := parseFloat(account.LastEquity)

	now := time.Now()

	pw.mu.Lock()
	// Reset daily tracking at start of each trading day.
	if pw.dayStarted.IsZero() || now.Day() != pw.dayStarted.Day() {
		pw.dayStarted = now
		pw.dailyHigh = equity
		pw.dailyPnLStart = lastEquity
		pw.lastEquity = equity
	}

	if equity > pw.dailyHigh {
		pw.dailyHigh = equity
	}

	intradayDD := 0.0
	if pw.dailyHigh > 0 {
		intradayDD = (pw.dailyHigh - equity) / pw.dailyHigh * 100
	}

	dailyPnL := equity - pw.dailyPnLStart
	hwm := pw.dailyHigh
	pw.lastEquity = equity
	pw.mu.Unlock()

	posMap := make(map[string]alerting.Position)
	totalExposure := 0.0
	for _, p := range positions {
		mv := parseFloat(p.MarketValue)
		totalExposure += math.Abs(mv)
		posMap[p.Symbol] = alerting.Position{
			Symbol:    p.Symbol,
			Qty:       parseFloat(p.Qty),
			AvgCost:   parseFloat(p.AvgEntryPrice),
			MarketVal: mv,
			UnrealPnL: parseFloat(p.UnrealizedPL),
			Side:      p.Side,
		}
	}

	state := alerting.PortfolioState{
		Timestamp:     now,
		Equity:        equity,
		Cash:          cash,
		DailyPnL:      dailyPnL,
		IntradayDD:    intradayDD,
		HighWaterMark: hwm,
		Positions:     posMap,
		TotalExposure: totalExposure,
	}

	pw.log.Debug("portfolio state",
		zap.Float64("equity", equity),
		zap.Float64("daily_pnl", dailyPnL),
		zap.Float64("intraday_dd_pct", intradayDD),
		zap.Int("positions", len(posMap)))

	pw.mu.Lock()
	hs := pw.handlers
	pw.mu.Unlock()
	for _, h := range hs {
		h(state)
	}
}

func (pw *PortfolioWatcher) baseURL() string {
	if pw.cfg.Paper {
		return "https://paper-api.alpaca.markets"
	}
	return "https://api.alpaca.markets"
}

func (pw *PortfolioWatcher) newRequest(ctx context.Context, path string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, pw.baseURL()+path, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("APCA-API-KEY-ID", pw.cfg.APIKey)
	req.Header.Set("APCA-API-SECRET-KEY", pw.cfg.Secret)
	return req, nil
}

func (pw *PortfolioWatcher) fetchAccount(ctx context.Context) (*alpacaAccount, error) {
	req, err := pw.newRequest(ctx, "/v2/account")
	if err != nil {
		return nil, err
	}
	resp, err := pw.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch account: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("account API returned %d", resp.StatusCode)
	}
	var acc alpacaAccount
	if err := json.NewDecoder(resp.Body).Decode(&acc); err != nil {
		return nil, fmt.Errorf("decode account: %w", err)
	}
	return &acc, nil
}

func (pw *PortfolioWatcher) fetchPositions(ctx context.Context) ([]alpacaPosition, error) {
	req, err := pw.newRequest(ctx, "/v2/positions")
	if err != nil {
		return nil, err
	}
	resp, err := pw.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch positions: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("positions API returned %d", resp.StatusCode)
	}
	var pos []alpacaPosition
	if err := json.NewDecoder(resp.Body).Decode(&pos); err != nil {
		return nil, fmt.Errorf("decode positions: %w", err)
	}
	return pos, nil
}

func parseFloat(s string) float64 {
	var v float64
	fmt.Sscanf(s, "%f", &v)
	return v
}
