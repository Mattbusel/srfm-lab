package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/srfm/gateway/internal/cache"
	"github.com/srfm/gateway/internal/feed"
	"go.uber.org/zap"
)

// SSEHandler provides Server-Sent Events for bar data.
// Clients connect to /stream/{symbol}/{timeframe} and receive bar events
// as they arrive without needing a WebSocket.
type SSEHandler struct {
	cache   *cache.BarCache
	barCh   <-chan barMsg
	clients map[string]map[chan barMsg]struct{} // key: symbol+tf
	log     *zap.Logger
}

type barMsg struct {
	Timeframe string
	Bar       feed.Bar
}

// StreamBar is the JSON envelope for SSE bar messages.
type StreamBar struct {
	Type      string    `json:"type"`
	Symbol    string    `json:"symbol"`
	Timeframe string    `json:"timeframe"`
	Timestamp time.Time `json:"timestamp"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	Source    string    `json:"source"`
	IsPartial bool      `json:"is_partial,omitempty"`
}

// CandleHandler handles requests for chart-ready OHLCV data in various formats.
type CandleHandler struct {
	cache *cache.BarCache
	log   *zap.Logger
}

// NewCandleHandler creates a CandleHandler.
func NewCandleHandler(c *cache.BarCache, log *zap.Logger) *CandleHandler {
	return &CandleHandler{cache: c, log: log}
}

// Routes returns a chi router with candle endpoints.
func (ch *CandleHandler) Routes() http.Handler {
	r := chi.NewRouter()
	r.Get("/{symbol}/{timeframe}", ch.getCandles)
	r.Get("/{symbol}/{timeframe}/latest", ch.getLatestCandle)
	r.Get("/{symbol}/{timeframe}/ohlc", ch.getOHLC)
	return r
}

// getCandles handles GET /candles/{symbol}/{timeframe}?from=&to=&limit=
// Returns bars in a TradingView-compatible format.
func (ch *CandleHandler) getCandles(w http.ResponseWriter, r *http.Request) {
	symbol := chi.URLParam(r, "symbol")
	timeframe := chi.URLParam(r, "timeframe")
	q := r.URL.Query()

	var from, to time.Time
	if fs := q.Get("from"); fs != "" {
		t, err := parseTime(fs)
		if err == nil {
			from = t
		}
	}
	if from.IsZero() {
		from = time.Now().Add(-7 * 24 * time.Hour)
	}
	if ts := q.Get("to"); ts != "" {
		t, err := parseTime(ts)
		if err == nil {
			to = t
		}
	}
	if to.IsZero() {
		to = time.Now()
	}

	limit := 0
	if ls := q.Get("limit"); ls != "" {
		if n, err := strconv.Atoi(ls); err == nil {
			limit = n
		}
	}

	bars := ch.cache.GetBars(symbol, timeframe, from, to)
	if bars == nil {
		bars = []feed.Bar{}
	}
	if limit > 0 && len(bars) > limit {
		bars = bars[len(bars)-limit:]
	}

	// TradingView-compatible format.
	type tvCandle struct {
		Time   int64   `json:"time"`
		Open   float64 `json:"open"`
		High   float64 `json:"high"`
		Low    float64 `json:"low"`
		Close  float64 `json:"close"`
		Volume float64 `json:"volume"`
	}
	candles := make([]tvCandle, len(bars))
	for i, b := range bars {
		candles[i] = tvCandle{
			Time:   b.Timestamp.Unix(),
			Open:   b.Open,
			High:   b.High,
			Low:    b.Low,
			Close:  b.Close,
			Volume: b.Volume,
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"symbol":    symbol,
		"timeframe": timeframe,
		"candles":   candles,
		"count":     len(candles),
	})
}

// getLatestCandle handles GET /candles/{symbol}/{timeframe}/latest
func (ch *CandleHandler) getLatestCandle(w http.ResponseWriter, r *http.Request) {
	symbol := chi.URLParam(r, "symbol")
	timeframe := chi.URLParam(r, "timeframe")

	b := ch.cache.GetLatest(symbol, timeframe)
	if b == nil {
		writeError(w, http.StatusNotFound, "no data")
		return
	}
	writeJSON(w, http.StatusOK, b)
}

// getOHLC handles GET /candles/{symbol}/{timeframe}/ohlc
// Returns just the current OHLC as a scalar summary.
func (ch *CandleHandler) getOHLC(w http.ResponseWriter, r *http.Request) {
	symbol := chi.URLParam(r, "symbol")
	timeframe := chi.URLParam(r, "timeframe")

	b := ch.cache.GetLatest(symbol, timeframe)
	if b == nil {
		writeError(w, http.StatusNotFound, "no data")
		return
	}

	from := b.Timestamp.Truncate(24 * time.Hour)
	to := from.Add(24 * time.Hour)
	daily := ch.cache.GetBars(symbol, "1d", from, to)

	prevClose := 0.0
	if len(daily) > 1 {
		prevClose = daily[len(daily)-2].Close
	}

	changePct := 0.0
	if prevClose > 0 {
		changePct = (b.Close - prevClose) / prevClose * 100
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"symbol":     symbol,
		"timeframe":  timeframe,
		"timestamp":  b.Timestamp,
		"open":       b.Open,
		"high":       b.High,
		"low":        b.Low,
		"close":      b.Close,
		"volume":     b.Volume,
		"prev_close": prevClose,
		"change_pct": changePct,
		"source":     b.Source,
	})
}

// SSE streaming for live bar data.

// ServeBarStream streams live bars as SSE for a specific symbol+timeframe.
// The client receives data: <json> events as bars arrive.
func ServeBarStream(
	w http.ResponseWriter,
	r *http.Request,
	symbol, timeframe string,
	subscription chan feed.Bar,
	log *zap.Logger,
) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "SSE not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("X-Accel-Buffering", "no")

	log.Info("SSE stream connected",
		zap.String("symbol", symbol),
		zap.String("timeframe", timeframe),
		zap.String("remote", r.RemoteAddr))

	heartbeat := time.NewTicker(30 * time.Second)
	defer heartbeat.Stop()

	for {
		select {
		case <-r.Context().Done():
			return
		case bar, ok := <-subscription:
			if !ok {
				return
			}
			sb := StreamBar{
				Type:      "bar",
				Symbol:    bar.Symbol,
				Timeframe: timeframe,
				Timestamp: bar.Timestamp,
				Open:      bar.Open,
				High:      bar.High,
				Low:       bar.Low,
				Close:     bar.Close,
				Volume:    bar.Volume,
				Source:    bar.Source,
				IsPartial: bar.IsPartial,
			}
			data, err := json.Marshal(sb)
			if err != nil {
				log.Warn("SSE marshal bar", zap.Error(err))
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()

		case <-heartbeat.C:
			fmt.Fprintf(w, ": heartbeat %d\n\n", time.Now().Unix())
			flusher.Flush()
		}
	}
}
