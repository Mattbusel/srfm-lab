package monitors

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"sync"
	"time"
)

const (
	binanceDepthURL      = "https://api.binance.com/api/v3/depth?symbol=%s&limit=20"
	depthPollInterval    = 10 * time.Second
	depthWindowPct       = 0.001 // 0.1% of mid price
	depthHistorySize     = 500
	depthThinPercentile  = 20.0
)

// DepthSnapshot represents a single order book depth measurement.
type DepthSnapshot struct {
	Symbol        string
	BidDepth      float64 // total bid notional within 0.1% of mid
	AskDepth      float64 // total ask notional within 0.1% of mid
	Imbalance     float64 // (bid-ask)/(bid+ask)
	TotalDepth    float64
	Timestamp     time.Time
	Thin          bool // below historical 20th percentile
}

type depthHistory struct {
	totals []float64
}

func (h *depthHistory) push(v float64) {
	if len(h.totals) >= depthHistorySize {
		h.totals = h.totals[1:]
	}
	h.totals = append(h.totals, v)
}

// percentile20 returns the 20th percentile of total depths.
func (h *depthHistory) percentile20() float64 {
	if len(h.totals) == 0 {
		return 0
	}
	sorted := make([]float64, len(h.totals))
	copy(sorted, h.totals)
	sort.Float64s(sorted)
	idx := int(float64(len(sorted)) * 0.20)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// DepthMonitor polls L2 order book depth per symbol.
type DepthMonitor struct {
	mu      sync.RWMutex
	client  *http.Client
	log     *slog.Logger
	current map[string]DepthSnapshot
	history map[string]*depthHistory
	symbols []string
}

// NewDepthMonitor creates a DepthMonitor for the given Binance symbols.
func NewDepthMonitor(symbols []string, log *slog.Logger) *DepthMonitor {
	return &DepthMonitor{
		client:  &http.Client{Timeout: 8 * time.Second},
		log:     log,
		current: make(map[string]DepthSnapshot),
		history: make(map[string]*depthHistory),
		symbols: symbols,
	}
}

// Run starts the polling loop.
func (d *DepthMonitor) Run(ctx context.Context) {
	ticker := time.NewTicker(depthPollInterval)
	defer ticker.Stop()
	d.pollAll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.pollAll(ctx)
		}
	}
}

type depthResponse struct {
	Bids [][]json.RawMessage `json:"bids"` // [[price, qty], ...]
	Asks [][]json.RawMessage `json:"asks"`
}

func (d *DepthMonitor) pollAll(ctx context.Context) {
	for _, sym := range d.symbols {
		if err := d.pollSymbol(ctx, sym); err != nil {
			d.log.Warn("depth: poll failed", "symbol", sym, "err", err)
		}
	}
}

func (d *DepthMonitor) pollSymbol(ctx context.Context, symbol string) error {
	url := fmt.Sprintf(binanceDepthURL, symbol)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("depth: %s returned %d", symbol, resp.StatusCode)
	}

	var book depthResponse
	if err := json.NewDecoder(resp.Body).Decode(&book); err != nil {
		return fmt.Errorf("depth: decode %s: %w", symbol, err)
	}

	// Compute mid from best bid/ask.
	bestBid := parseLevelPrice(book.Bids, 0)
	bestAsk := parseLevelPrice(book.Asks, 0)
	if bestBid <= 0 || bestAsk <= 0 {
		return nil
	}
	mid := (bestBid + bestAsk) / 2.0
	threshold := mid * depthWindowPct

	bidDepth := 0.0
	for _, level := range book.Bids {
		price := parseLevelPriceRaw(level[0])
		qty := parseLevelPriceRaw(level[1])
		if mid-price <= threshold {
			bidDepth += price * qty
		}
	}
	askDepth := 0.0
	for _, level := range book.Asks {
		price := parseLevelPriceRaw(level[0])
		qty := parseLevelPriceRaw(level[1])
		if price-mid <= threshold {
			askDepth += price * qty
		}
	}

	total := bidDepth + askDepth
	imbalance := 0.0
	if total > 0 {
		imbalance = (bidDepth - askDepth) / total
	}

	d.mu.Lock()
	h, ok := d.history[symbol]
	if !ok {
		h = &depthHistory{}
		d.history[symbol] = h
	}
	h.push(total)
	p20 := h.percentile20()
	d.mu.Unlock()

	snap := DepthSnapshot{
		Symbol:     symbol,
		BidDepth:   bidDepth,
		AskDepth:   askDepth,
		Imbalance:  imbalance,
		TotalDepth: total,
		Timestamp:  time.Now().UTC(),
		Thin:       total < p20,
	}

	d.mu.Lock()
	d.current[symbol] = snap
	d.mu.Unlock()

	return nil
}

// Current returns the latest DepthSnapshot for symbol.
func (d *DepthMonitor) Current(symbol string) (DepthSnapshot, bool) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	snap, ok := d.current[symbol]
	return snap, ok
}

// AllCurrent returns all current depth snapshots.
func (d *DepthMonitor) AllCurrent() map[string]DepthSnapshot {
	d.mu.RLock()
	defer d.mu.RUnlock()
	out := make(map[string]DepthSnapshot, len(d.current))
	for k, v := range d.current {
		out[k] = v
	}
	return out
}

// Percentile20 returns the 20th percentile total depth for symbol.
func (d *DepthMonitor) Percentile20(symbol string) float64 {
	d.mu.RLock()
	defer d.mu.RUnlock()
	h, ok := d.history[symbol]
	if !ok {
		return 0
	}
	return h.percentile20()
}

func parseLevelPrice(levels [][]json.RawMessage, idx int) float64 {
	if len(levels) == 0 {
		return 0
	}
	return parseLevelPriceRaw(levels[0][0])
}

func parseLevelPriceRaw(raw json.RawMessage) float64 {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return 0
	}
	return parseF(s)
}
