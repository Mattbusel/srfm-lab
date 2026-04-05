// Package monitors contains per-metric liquidity monitors.
package monitors

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"sync"
	"time"
)

const (
	binanceBookTickerURL    = "https://api.binance.com/api/v3/bookTicker"
	spreadPollInterval      = 5 * time.Second
	spreadAlertMultiplier   = 2.0
	spreadHistoryBuckets    = 24 // hours
	spreadHistoryPerBucket  = 500
)

// SpreadObservation is one bid-ask spread measurement.
type SpreadObservation struct {
	Symbol        string
	Bid           float64
	Ask           float64
	Mid           float64
	EffectiveSpread float64 // (ask - bid) / mid
	Hour          int // 0-23 UTC
	Timestamp     time.Time
}

// SpreadAlert fires when current spread exceeds 2x hourly average.
type SpreadAlert struct {
	Symbol          string    `json:"symbol"`
	CurrentSpread   float64   `json:"current_spread"`
	HourlyAverage   float64   `json:"hourly_average"`
	Ratio           float64   `json:"ratio"`
	Hour            int       `json:"hour"`
	AlertedAt       time.Time `json:"alerted_at"`
}

// hourBucket holds rolling spread observations for one hour-of-day.
type hourBucket struct {
	obs []float64
	pos int
}

func (b *hourBucket) push(v float64) {
	if len(b.obs) < spreadHistoryPerBucket {
		b.obs = append(b.obs, v)
	} else {
		b.obs[b.pos] = v
		b.pos = (b.pos + 1) % spreadHistoryPerBucket
	}
}

func (b *hourBucket) mean() float64 {
	if len(b.obs) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range b.obs {
		sum += v
	}
	return sum / float64(len(b.obs))
}

// SpreadMonitor monitors bid-ask spreads per symbol.
type SpreadMonitor struct {
	mu      sync.RWMutex
	client  *http.Client
	log     *slog.Logger
	current map[string]SpreadObservation
	history map[string][spreadHistoryBuckets]*hourBucket
	alerts  chan SpreadAlert
}

// NewSpreadMonitor creates a SpreadMonitor. alerts is a channel to receive alerts on.
func NewSpreadMonitor(log *slog.Logger, alerts chan SpreadAlert) *SpreadMonitor {
	return &SpreadMonitor{
		client:  &http.Client{Timeout: 8 * time.Second},
		log:     log,
		current: make(map[string]SpreadObservation),
		history: make(map[string][spreadHistoryBuckets]*hourBucket),
		alerts:  alerts,
	}
}

// Run starts the polling loop.
func (s *SpreadMonitor) Run(ctx context.Context) {
	ticker := time.NewTicker(spreadPollInterval)
	defer ticker.Stop()
	s.poll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.poll(ctx)
		}
	}
}

type bookTickerEntry struct {
	Symbol   string `json:"symbol"`
	BidPrice string `json:"bidPrice"`
	AskPrice string `json:"askPrice"`
}

func (s *SpreadMonitor) poll(ctx context.Context) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, binanceBookTickerURL, nil)
	if err != nil {
		s.log.Error("spread: build request", "err", err)
		return
	}
	resp, err := s.client.Do(req)
	if err != nil {
		s.log.Error("spread: request failed", "err", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		s.log.Error("spread: non-200", "status", resp.StatusCode)
		return
	}

	var tickers []bookTickerEntry
	if err := json.NewDecoder(resp.Body).Decode(&tickers); err != nil {
		s.log.Error("spread: decode", "err", err)
		return
	}

	now := time.Now().UTC()
	hour := now.Hour()

	for _, t := range tickers {
		if !isTrackedSymbol(t.Symbol) {
			continue
		}
		bid := parseF(t.BidPrice)
		ask := parseF(t.AskPrice)
		if bid <= 0 || ask <= 0 {
			continue
		}
		mid := (bid + ask) / 2.0
		spread := (ask - bid) / mid

		obs := SpreadObservation{
			Symbol:          t.Symbol,
			Bid:             bid,
			Ask:             ask,
			Mid:             mid,
			EffectiveSpread: spread,
			Hour:            hour,
			Timestamp:       now,
		}

		s.mu.Lock()
		s.current[t.Symbol] = obs

		hist, ok := s.history[t.Symbol]
		if !ok {
			// Zero value initializes all pointers to nil.
		}
		if hist[hour] == nil {
			hist[hour] = &hourBucket{}
		}
		hist[hour].push(spread)
		s.history[t.Symbol] = hist
		avg := hist[hour].mean()
		s.mu.Unlock()

		if avg > 0 && spread > spreadAlertMultiplier*avg {
			alert := SpreadAlert{
				Symbol:        t.Symbol,
				CurrentSpread: spread,
				HourlyAverage: avg,
				Ratio:         spread / avg,
				Hour:          hour,
				AlertedAt:     now,
			}
			select {
			case s.alerts <- alert:
			default:
			}
		}
	}
}

// Current returns the latest SpreadObservation for symbol.
func (s *SpreadMonitor) Current(symbol string) (SpreadObservation, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	obs, ok := s.current[symbol]
	return obs, ok
}

// AllCurrent returns all latest observations.
func (s *SpreadMonitor) AllCurrent() map[string]SpreadObservation {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[string]SpreadObservation, len(s.current))
	for k, v := range s.current {
		out[k] = v
	}
	return out
}

// HourlyAverage returns the average spread for symbol at the given hour.
func (s *SpreadMonitor) HourlyAverage(symbol string, hour int) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	hist, ok := s.history[symbol]
	if !ok || hist[hour] == nil {
		return 0
	}
	return hist[hour].mean()
}

// trackedSymbols is the set of Binance symbols to track.
var trackedSymbols = map[string]struct{}{
	"BTCUSDT": {}, "ETHUSDT": {}, "SOLUSDT": {}, "BNBUSDT": {}, "XRPUSDT": {},
	"ADAUSDT": {}, "DOGEUSDT": {}, "AVAXUSDT": {}, "LINKUSDT": {}, "DOTUSDT": {},
}

func isTrackedSymbol(s string) bool {
	_, ok := trackedSymbols[s]
	return ok
}

// parseF converts a decimal string to float64.
func parseF(s string) float64 {
	if s == "" {
		return 0
	}
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}

// worstSpread returns the worst (highest) spread seen across all hours for a symbol.
func (s *SpreadMonitor) WorstSpread(symbol string) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	hist, ok := s.history[symbol]
	if !ok {
		return 0
	}
	worst := 0.0
	for _, b := range hist {
		if b == nil {
			continue
		}
		for _, v := range b.obs {
			worst = math.Max(worst, v)
		}
	}
	return worst
}
