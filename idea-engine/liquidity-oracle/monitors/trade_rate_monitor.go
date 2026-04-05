package monitors

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

const (
	binanceRecentTradesURL  = "https://api.binance.com/api/v3/trades?symbol=%s&limit=500"
	tradeRatePollInterval   = 30 * time.Second
	tradeRateWindow         = time.Minute
	tradeRateHistoryBuckets = 24
	tradeRateHistoryPerHour = 120
)

// TradeRateEvent classifies trade rate conditions.
type TradeRateEvent string

const (
	TradeRateEventNormal     TradeRateEvent = "NORMAL"
	TradeRateEventDead       TradeRateEvent = "DEAD"       // very low trade rate
	TradeRateEventMomentum   TradeRateEvent = "MOMENTUM"   // high rate + directional bias
)

// TradeRateSnapshot holds a trade rate measurement.
type TradeRateSnapshot struct {
	Symbol          string
	TradesPerMinute float64
	TypicalTPM      float64 // typical for this hour-of-day
	Ratio           float64
	BuyRatio        float64 // fraction of trades that were buys (taker side)
	Hour            int
	Event           TradeRateEvent
	Timestamp       time.Time
}

// tradeEntry is a Binance recent trade.
type tradeEntry struct {
	ID           int64   `json:"id"`
	Price        string  `json:"price"`
	Qty          string  `json:"qty"`
	Time         int64   `json:"time"` // ms
	IsBuyerMaker bool    `json:"isBuyerMaker"`
}

// TradeRateMonitor tracks trades-per-minute per symbol.
type TradeRateMonitor struct {
	mu      sync.RWMutex
	client  *http.Client
	log     *slog.Logger
	current map[string]TradeRateSnapshot
	history map[string][tradeRateHistoryBuckets][]float64
	symbols []string
}

// NewTradeRateMonitor creates a TradeRateMonitor.
func NewTradeRateMonitor(symbols []string, log *slog.Logger) *TradeRateMonitor {
	return &TradeRateMonitor{
		client:  &http.Client{Timeout: 8 * time.Second},
		log:     log,
		current: make(map[string]TradeRateSnapshot),
		history: make(map[string][tradeRateHistoryBuckets][]float64),
		symbols: symbols,
	}
}

// Run starts the polling loop.
func (t *TradeRateMonitor) Run(ctx context.Context) {
	ticker := time.NewTicker(tradeRatePollInterval)
	defer ticker.Stop()
	t.pollAll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			t.pollAll(ctx)
		}
	}
}

func (t *TradeRateMonitor) pollAll(ctx context.Context) {
	for _, sym := range t.symbols {
		if err := t.pollSymbol(ctx, sym); err != nil {
			t.log.Warn("trade_rate: poll failed", "symbol", sym, "err", err)
		}
	}
}

func (t *TradeRateMonitor) pollSymbol(ctx context.Context, symbol string) error {
	url := fmt.Sprintf(binanceRecentTradesURL, symbol)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("trade_rate: %s returned %d", symbol, resp.StatusCode)
	}

	var trades []tradeEntry
	if err := json.NewDecoder(resp.Body).Decode(&trades); err != nil {
		return fmt.Errorf("trade_rate: decode %s: %w", symbol, err)
	}

	now := time.Now().UTC()
	cutoff := now.Add(-tradeRateWindow).UnixMilli()

	recentCount := 0
	buyCount := 0
	for _, tr := range trades {
		if tr.Time >= cutoff {
			recentCount++
			// isBuyerMaker=true means the buyer is the maker, so the taker is a seller.
			if !tr.IsBuyerMaker {
				buyCount++
			}
		}
	}

	tpm := float64(recentCount) // trades in last minute
	buyRatio := 0.0
	if recentCount > 0 {
		buyRatio = float64(buyCount) / float64(recentCount)
	}

	hour := now.Hour()

	t.mu.Lock()
	hist, _ := t.history[symbol]
	bucket := hist[hour]
	if len(bucket) >= tradeRateHistoryPerHour {
		bucket = bucket[1:]
	}
	bucket = append(bucket, tpm)
	hist[hour] = bucket
	t.history[symbol] = hist
	typical := median(bucket)
	t.mu.Unlock()

	ratio := 0.0
	if typical > 0 {
		ratio = tpm / typical
	}

	event := TradeRateEventNormal
	switch {
	case tpm < 5 || (typical > 0 && ratio < 0.10):
		event = TradeRateEventDead
	case ratio > 3.0 && (buyRatio > 0.7 || buyRatio < 0.3):
		event = TradeRateEventMomentum
	}

	t.mu.Lock()
	t.current[symbol] = TradeRateSnapshot{
		Symbol:          symbol,
		TradesPerMinute: tpm,
		TypicalTPM:      typical,
		Ratio:           ratio,
		BuyRatio:        buyRatio,
		Hour:            hour,
		Event:           event,
		Timestamp:       now,
	}
	t.mu.Unlock()

	return nil
}

// Current returns the latest TradeRateSnapshot for symbol.
func (t *TradeRateMonitor) Current(symbol string) (TradeRateSnapshot, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	snap, ok := t.current[symbol]
	return snap, ok
}

// AllCurrent returns all current snapshots.
func (t *TradeRateMonitor) AllCurrent() map[string]TradeRateSnapshot {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make(map[string]TradeRateSnapshot, len(t.current))
	for k, v := range t.current {
		out[k] = v
	}
	return out
}
