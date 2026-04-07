// Package server provides HTTP handlers for the v2 market data API.
package server

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"srfm/market-data/aggregator"
	"srfm/market-data/pkg/analytics"
	"srfm/market-data/pkg/feed"
	pkstorage "srfm/market-data/pkg/storage"
)

// -- Dependency interfaces --

// FeedHealthReader exposes feed health status.
type FeedHealthReader interface {
	Health() feed.BinanceFeedV2Health
}

// HandlersV2 holds dependencies for the v2 HTTP handlers.
type HandlersV2 struct {
	cache       *pkstorage.TimeSeriesCache
	spreads     *analytics.SpreadMonitor
	stats       *analytics.BarStatsComputer
	volClock    *analytics.VolumeClock
	feedHealths []FeedHealthReader
}

// NewHandlersV2 creates HandlersV2.
func NewHandlersV2(
	cache *pkstorage.TimeSeriesCache,
	spreads *analytics.SpreadMonitor,
	stats *analytics.BarStatsComputer,
	volClock *analytics.VolumeClock,
	feedHealths []FeedHealthReader,
) *HandlersV2 {
	return &HandlersV2{
		cache:       cache,
		spreads:     spreads,
		stats:       stats,
		volClock:    volClock,
		feedHealths: feedHealths,
	}
}

// RegisterRoutes attaches all v2 routes to the given ServeMux.
func (h *HandlersV2) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v2/bars/", h.GetBars)
	mux.HandleFunc("/v2/spread/", h.GetSpread)
	mux.HandleFunc("/v2/stats/", h.GetStats)
	mux.HandleFunc("/v2/volume-bars/", h.GetVolumeBars)
	mux.HandleFunc("/v2/feed/health", h.GetFeedHealth)
}

// -- GET /v2/bars/{symbol}/{timeframe}?n=100 --

// GetBars returns the last n bars from the TimeSeriesCache.
// Path: /v2/bars/{symbol}/{timeframe}
// Query: n (default 100, max 500)
func (h *HandlersV2) GetBars(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		v2WriteError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Strip prefix "/v2/bars/" and split into symbol/timeframe
	tail := strings.TrimPrefix(r.URL.Path, "/v2/bars/")
	parts := strings.SplitN(tail, "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		v2WriteError(w, "path must be /v2/bars/{symbol}/{timeframe}", http.StatusBadRequest)
		return
	}

	symbol := strings.ToUpper(parts[0])
	timeframe := parts[1]

	n := v2ParseIntQuery(r, "n", 100)
	if n > 500 {
		n = 500
	}

	bars := h.cache.GetLast(symbol, timeframe, n)
	if bars == nil {
		bars = []aggregator.BarEvent{}
	}

	v2WriteJSON(w, map[string]interface{}{
		"symbol":    symbol,
		"timeframe": timeframe,
		"n":         n,
		"count":     len(bars),
		"bars":      bars,
	})
}

// -- GET /v2/spread/{symbol} --

// GetSpread returns the current spread in basis points and a liquidity score.
// Path: /v2/spread/{symbol}
func (h *HandlersV2) GetSpread(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		v2WriteError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	symbol := strings.ToUpper(strings.TrimPrefix(r.URL.Path, "/v2/spread/"))
	if symbol == "" {
		v2WriteError(w, "symbol required", http.StatusBadRequest)
		return
	}

	spreadBps, ok := h.spreads.GetSpread(symbol)
	if !ok {
		v2WriteError(w, "no spread data for symbol", http.StatusNotFound)
		return
	}

	snap, _ := h.spreads.GetLatestSnapshot(symbol)
	liquidityScore := h.spreads.GetLiquidityScore(symbol)
	isWide := h.spreads.IsWide(symbol)

	pct25 := h.spreads.GetSpreadPercentile(symbol, 25)
	pct50 := h.spreads.GetSpreadPercentile(symbol, 50)
	pct75 := h.spreads.GetSpreadPercentile(symbol, 75)
	pct95 := h.spreads.GetSpreadPercentile(symbol, 95)

	v2WriteJSON(w, map[string]interface{}{
		"symbol":          symbol,
		"spread_bps":      spreadBps,
		"is_wide":         isWide,
		"liquidity_score": liquidityScore,
		"bid":             snap.Bid,
		"ask":             snap.Ask,
		"depth_bid":       snap.DepthBid,
		"depth_ask":       snap.DepthAsk,
		"timestamp":       snap.Timestamp.UTC().Format(time.RFC3339Nano),
		"percentiles": map[string]float64{
			"p25": pct25,
			"p50": pct50,
			"p75": pct75,
			"p95": pct95,
		},
	})
}

// -- GET /v2/stats/{symbol} --

// GetStats returns the real-time BarStats for a symbol.
// Path: /v2/stats/{symbol}
func (h *HandlersV2) GetStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		v2WriteError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	symbol := strings.ToUpper(strings.TrimPrefix(r.URL.Path, "/v2/stats/"))
	if symbol == "" {
		v2WriteError(w, "symbol required", http.StatusBadRequest)
		return
	}

	stats, ok := h.stats.GetStats(symbol)
	if !ok {
		v2WriteError(w, "no stats for symbol", http.StatusNotFound)
		return
	}

	v2WriteJSON(w, map[string]interface{}{
		"symbol":      symbol,
		"count":       stats.Count,
		"mean":        stats.Mean,
		"variance":    stats.Variance,
		"std_dev":     stats.StdDev,
		"ema8":        stats.EMA8,
		"ema21":       stats.EMA21,
		"atr":         stats.ATR,
		"vol_ema":     stats.VolEMA,
		"vol_std":     stats.VolStd,
		"vol_z_score": stats.VolZScore,
		"last_close":  stats.LastClose,
		"last_high":   stats.LastHigh,
		"last_low":    stats.LastLow,
		"last_volume": stats.LastVolume,
		"last_time":   stats.LastTime.UTC().Format(time.RFC3339Nano),
	})
}

// -- GET /v2/volume-bars/{symbol}?n=50 --

// GetVolumeBars returns the last n completed volume-clock bars for a symbol.
// Path: /v2/volume-bars/{symbol}
// Query: n (default 50, max 500)
func (h *HandlersV2) GetVolumeBars(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		v2WriteError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	symbol := strings.ToUpper(strings.TrimPrefix(r.URL.Path, "/v2/volume-bars/"))
	if symbol == "" {
		v2WriteError(w, "symbol required", http.StatusBadRequest)
		return
	}

	n := v2ParseIntQuery(r, "n", 50)
	if n > 500 {
		n = 500
	}

	bars := h.volClock.GetBars(symbol, n)
	if bars == nil {
		bars = []analytics.VolumeBar{}
	}

	targetVol := h.volClock.GetTargetVolume(symbol)

	v2WriteJSON(w, map[string]interface{}{
		"symbol":        symbol,
		"n":             n,
		"count":         len(bars),
		"target_volume": targetVol,
		"bars":          bars,
	})
}

// -- GET /v2/feed/health --

// GetFeedHealth returns health status for all registered feeds with metrics.
// Path: /v2/feed/health
func (h *HandlersV2) GetFeedHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		v2WriteError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	type feedEntry struct {
		Name           string  `json:"name"`
		IsConnected    bool    `json:"is_connected"`
		IsHealthy      bool    `json:"is_healthy"`
		BarsReceived   int64   `json:"bars_received"`
		ErrorsCount    int64   `json:"errors_count"`
		LatencyMs      int64   `json:"latency_ms"`
		ReconnectCount int64   `json:"reconnect_count"`
		UptimePct      float64 `json:"uptime_pct"`
		LastBarTime    string  `json:"last_bar_time"`
		UptimeSec      float64 `json:"uptime_sec"`
	}

	entries := make([]feedEntry, 0, len(h.feedHealths))
	allHealthy := true

	for _, fh := range h.feedHealths {
		snap := fh.Health()

		lbt := ""
		if !snap.LastBarTime.IsZero() {
			lbt = snap.LastBarTime.UTC().Format(time.RFC3339)
		}

		if !snap.IsHealthy || !snap.IsConnected {
			allHealthy = false
		}

		entries = append(entries, feedEntry{
			Name:           snap.Name,
			IsConnected:    snap.IsConnected,
			IsHealthy:      snap.IsHealthy,
			BarsReceived:   snap.BarsReceived,
			ErrorsCount:    snap.ErrorsCount,
			LatencyMs:      snap.LatencyMs,
			ReconnectCount: snap.ReconnectCount,
			UptimePct:      snap.UptimePct,
			LastBarTime:    lbt,
			UptimeSec:      snap.Uptime.Seconds(),
		})
	}

	status := "ok"
	if !allHealthy {
		status = "degraded"
	}

	v2WriteJSON(w, map[string]interface{}{
		"status":      status,
		"all_healthy": allHealthy,
		"feeds":       entries,
		"timestamp":   time.Now().UTC().Format(time.RFC3339),
	})
}

// -- helpers --

func v2WriteJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("[handlers-v2] encode error: %v", err)
	}
}

func v2WriteError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]interface{}{ //nolint:errcheck
		"error": msg,
		"code":  code,
	})
}

func v2ParseIntQuery(r *http.Request, key string, def int) int {
	s := r.URL.Query().Get(key)
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return def
	}
	return n
}
