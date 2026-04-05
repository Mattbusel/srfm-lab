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
	binanceKlinesURL     = "https://api.binance.com/api/v3/klines?symbol=%s&interval=1h&limit=2"
	volumePollInterval   = 60 * time.Second
	volumeHistoryBuckets = 24 // hours
	volumeHistoryPerHour = 90 // ~90 days
	volumeThinThreshold  = 0.30 // < 30% of typical = thin
	volumeSpikeThreshold = 3.0  // > 3x typical = spike
)

// VolumeEvent classifies the current volume condition.
type VolumeEvent string

const (
	VolumeEventNormal VolumeEvent = "NORMAL"
	VolumeEventThin   VolumeEvent = "THIN"
	VolumeEventSpike  VolumeEvent = "SPIKE"
)

// VolumeSnapshot holds a rolling-1h volume observation for a symbol.
type VolumeSnapshot struct {
	Symbol        string
	CurrentVolume float64
	TypicalVolume float64 // median for this hour-of-day
	Ratio         float64 // current / typical
	Hour          int
	Event         VolumeEvent
	Timestamp     time.Time
}

// VolumeMonitor tracks rolling 1h volume per symbol.
type VolumeMonitor struct {
	mu      sync.RWMutex
	client  *http.Client
	log     *slog.Logger
	current map[string]VolumeSnapshot
	// history[symbol][hour] = ring of volumes
	history map[string][volumeHistoryBuckets][]float64
	symbols []string
}

// NewVolumeMonitor creates a VolumeMonitor for the given symbols.
func NewVolumeMonitor(symbols []string, log *slog.Logger) *VolumeMonitor {
	return &VolumeMonitor{
		client:  &http.Client{Timeout: 8 * time.Second},
		log:     log,
		current: make(map[string]VolumeSnapshot),
		history: make(map[string][volumeHistoryBuckets][]float64),
		symbols: symbols,
	}
}

// Run starts the polling loop.
func (v *VolumeMonitor) Run(ctx context.Context) {
	ticker := time.NewTicker(volumePollInterval)
	defer ticker.Stop()
	v.pollAll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			v.pollAll(ctx)
		}
	}
}

// binance kline: [open_time, open, high, low, close, volume, ...]
type binanceKline [12]json.RawMessage

func (v *VolumeMonitor) pollAll(ctx context.Context) {
	for _, sym := range v.symbols {
		if err := v.pollSymbol(ctx, sym); err != nil {
			v.log.Warn("volume: poll failed", "symbol", sym, "err", err)
		}
	}
}

func (v *VolumeMonitor) pollSymbol(ctx context.Context, symbol string) error {
	url := fmt.Sprintf(binanceKlinesURL, symbol)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := v.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("volume: %s returned %d", symbol, resp.StatusCode)
	}

	var klines []binanceKline
	if err := json.NewDecoder(resp.Body).Decode(&klines); err != nil {
		return fmt.Errorf("volume: decode %s: %w", symbol, err)
	}
	if len(klines) == 0 {
		return nil
	}

	// Use the most recent completed candle (index 0 if limit=2).
	var volStr string
	if err := json.Unmarshal(klines[0][5], &volStr); err != nil {
		return fmt.Errorf("volume: parse vol %s: %w", symbol, err)
	}
	currentVol := parseF(volStr)

	// Determine which hour the candle belongs to.
	var openTimeMs int64
	if err := json.Unmarshal(klines[0][0], &openTimeMs); err != nil {
		return fmt.Errorf("volume: parse open_time %s: %w", symbol, err)
	}
	candleHour := time.UnixMilli(openTimeMs).UTC().Hour()

	v.mu.Lock()
	hist, ok := v.history[symbol]
	if !ok {
		// Zero value is fine.
	}
	bucket := hist[candleHour]
	if len(bucket) >= volumeHistoryPerHour {
		bucket = bucket[1:]
	}
	bucket = append(bucket, currentVol)
	hist[candleHour] = bucket
	v.history[symbol] = hist
	typical := median(bucket)
	v.mu.Unlock()

	ratio := 0.0
	if typical > 0 {
		ratio = currentVol / typical
	}

	event := VolumeEventNormal
	switch {
	case ratio > 0 && ratio < volumeThinThreshold:
		event = VolumeEventThin
	case ratio >= volumeSpikeThreshold:
		event = VolumeEventSpike
	}

	v.mu.Lock()
	v.current[symbol] = VolumeSnapshot{
		Symbol:        symbol,
		CurrentVolume: currentVol,
		TypicalVolume: typical,
		Ratio:         ratio,
		Hour:          candleHour,
		Event:         event,
		Timestamp:     time.Now().UTC(),
	}
	v.mu.Unlock()

	return nil
}

// Current returns the latest VolumeSnapshot for symbol.
func (v *VolumeMonitor) Current(symbol string) (VolumeSnapshot, bool) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	snap, ok := v.current[symbol]
	return snap, ok
}

// AllCurrent returns all current volume snapshots.
func (v *VolumeMonitor) AllCurrent() map[string]VolumeSnapshot {
	v.mu.RLock()
	defer v.mu.RUnlock()
	out := make(map[string]VolumeSnapshot, len(v.current))
	for k, vv := range v.current {
		out[k] = vv
	}
	return out
}

// median computes the median of a float64 slice (sorted copy).
func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	// Simple insertion sort for small slices.
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return sorted[n/2]
}
