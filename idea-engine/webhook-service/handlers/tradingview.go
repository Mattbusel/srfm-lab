// Package handlers — tradingview.go
//
// TradingViewHandler receives TradingView webhook alerts, validates their
// HMAC-SHA256 signature, converts the alert to an IAE event, and publishes
// it to the idea-bus under the topic "signal.external".
//
// Rate limiting: 60 webhooks per minute per IP (sliding window).
// Ticker format: BTCUSD → BTC/USD (6-char pairs split at position 3;
// longer symbols use the exchange-provided format).
package handlers

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"go.uber.org/zap"
)

// ---------------------------------------------------------------------------
// TradingView alert payload
// ---------------------------------------------------------------------------

// TVAlert is the JSON payload sent by TradingView alert webhooks.
type TVAlert struct {
	// Ticker is the instrument symbol in TradingView format (e.g. "BTCUSD").
	Ticker string `json:"ticker"`
	// Close is the close price at the time the alert fired.
	Close float64 `json:"close"`
	// Action is the trade direction: "buy" | "sell" | "close_long" | "close_short".
	Action string `json:"action"`
	// Strategy is the name of the TradingView strategy that fired the alert.
	Strategy string `json:"strategy"`
	// Exchange is the exchange name (optional, e.g. "BINANCE").
	Exchange string `json:"exchange,omitempty"`
	// Interval is the chart interval that produced the alert (e.g. "1D", "4H").
	Interval string `json:"interval,omitempty"`
	// Time is the bar timestamp in ISO8601/Unix format (TV sends as string).
	Time string `json:"time,omitempty"`
	// Volume is the bar volume (optional).
	Volume float64 `json:"volume,omitempty"`
	// Comment is any free-text comment added to the alert.
	Comment string `json:"comment,omitempty"`
}

// Validate returns an error if the alert is missing required fields or
// contains an invalid action.
func (a *TVAlert) Validate() error {
	if strings.TrimSpace(a.Ticker) == "" {
		return fmt.Errorf("ticker is required")
	}
	if a.Close <= 0 {
		return fmt.Errorf("close price must be positive, got %v", a.Close)
	}
	switch strings.ToLower(a.Action) {
	case "buy", "sell", "close_long", "close_short", "long", "short", "exit":
		// valid
	default:
		return fmt.Errorf("unknown action %q", a.Action)
	}
	return nil
}

// NormalisedTicker converts a TradingView compound symbol to slash format.
//
// Rules:
//   - 6-char all-caps symbols with no slash: BTCUSD → BTC/USD
//   - Already contains slash: returned as-is
//   - Exchange-prefixed (BINANCE:BTCUSDT): strip exchange prefix, then apply above
//   - All other cases: return as-is with a best-effort split
func NormalisedTicker(raw string) string {
	// Strip exchange prefix
	if idx := strings.Index(raw, ":"); idx != -1 {
		raw = raw[idx+1:]
	}
	if strings.Contains(raw, "/") {
		return strings.ToUpper(raw)
	}
	raw = strings.ToUpper(raw)
	// Stablecoins and fiat quote currencies
	quoteCurrencies := []string{"USDT", "USDC", "BUSD", "USD", "EUR", "GBP", "BTC", "ETH"}
	for _, q := range quoteCurrencies {
		if strings.HasSuffix(raw, q) && len(raw) > len(q) {
			base := raw[:len(raw)-len(q)]
			if utf8.RuneCountInString(base) >= 2 {
				return base + "/" + q
			}
		}
	}
	// Fallback: split at midpoint for 6-char symbols
	n := len(raw)
	if n == 6 {
		return raw[:3] + "/" + raw[3:]
	}
	return raw
}

// ---------------------------------------------------------------------------
// Rate limiter (sliding window, per IP)
// ---------------------------------------------------------------------------

type ipWindow struct {
	mu        sync.Mutex
	timestamps []time.Time
}

func (w *ipWindow) allow(limit int) bool {
	w.mu.Lock()
	defer w.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-time.Minute)

	// Evict old entries
	filtered := w.timestamps[:0]
	for _, t := range w.timestamps {
		if t.After(cutoff) {
			filtered = append(filtered, t)
		}
	}
	w.timestamps = filtered

	if len(w.timestamps) >= limit {
		return false
	}
	w.timestamps = append(w.timestamps, now)
	return true
}

type rateLimiter struct {
	mu      sync.Mutex
	windows map[string]*ipWindow
	limit   int
}

func newRateLimiter(rpmLimit int) *rateLimiter {
	rl := &rateLimiter{
		windows: make(map[string]*ipWindow),
		limit:   rpmLimit,
	}
	// Periodically evict stale IP entries (every 5 minutes)
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		for range ticker.C {
			rl.evict()
		}
	}()
	return rl
}

func (rl *rateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	w, ok := rl.windows[ip]
	if !ok {
		w = &ipWindow{}
		rl.windows[ip] = w
	}
	rl.mu.Unlock()
	return w.allow(rl.limit)
}

func (rl *rateLimiter) evict() {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	cutoff := time.Now().Add(-time.Minute)
	for ip, w := range rl.windows {
		w.mu.Lock()
		allOld := true
		for _, t := range w.timestamps {
			if t.After(cutoff) {
				allOld = false
				break
			}
		}
		w.mu.Unlock()
		if allOld {
			delete(rl.windows, ip)
		}
	}
}

// ---------------------------------------------------------------------------
// TradingViewHandler
// ---------------------------------------------------------------------------

// TradingViewHandler processes POST /webhooks/tradingview.
type TradingViewHandler struct {
	bus        *BusClient
	logger     *zap.Logger
	metrics    *MetricsRegistry
	hmacSecret []byte
	rl         *rateLimiter
}

// NewTradingViewHandler creates a new handler.
func NewTradingViewHandler(
	bus *BusClient,
	logger *zap.Logger,
	metrics *MetricsRegistry,
	hmacSecret string,
	rateLimitRPM int,
) *TradingViewHandler {
	return &TradingViewHandler{
		bus:        bus,
		logger:     logger,
		metrics:    metrics,
		hmacSecret: []byte(hmacSecret),
		rl:         newRateLimiter(rateLimitRPM),
	}
}

// HandleAlert is the HTTP handler for POST /webhooks/tradingview.
func (h *TradingViewHandler) HandleAlert(w http.ResponseWriter, r *http.Request) {
	h.metrics.WebhooksReceived.Add(1)

	// Rate limit by client IP
	ip := extractIP(r)
	if !h.rl.allow(ip) {
		h.logger.Warn("rate limit exceeded", zap.String("ip", ip))
		respondError(w, http.StatusTooManyRequests, "rate limit exceeded: 60 requests/minute")
		h.metrics.WebhookErrors.Add(1)
		return
	}

	// Read and size-limit body (max 64 KB)
	body, err := io.ReadAll(io.LimitReader(r.Body, 64*1024))
	if err != nil {
		h.logger.Error("read body", zap.Error(err))
		respondError(w, http.StatusBadRequest, "could not read request body")
		h.metrics.WebhookErrors.Add(1)
		return
	}

	// Verify HMAC signature if a secret is configured
	if len(h.hmacSecret) > 0 {
		sig := r.Header.Get("X-Webhook-Signature")
		if !verifyHMAC(h.hmacSecret, body, sig) {
			h.logger.Warn("invalid HMAC signature", zap.String("ip", ip))
			respondError(w, http.StatusUnauthorized, "invalid webhook signature")
			h.metrics.WebhookErrors.Add(1)
			return
		}
	}

	// Parse alert
	var alert TVAlert
	if err := json.Unmarshal(body, &alert); err != nil {
		h.logger.Warn("invalid JSON", zap.Error(err), zap.String("ip", ip))
		respondError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	if err := alert.Validate(); err != nil {
		h.logger.Warn("validation failed", zap.Error(err))
		respondError(w, http.StatusUnprocessableEntity, err.Error())
		h.metrics.WebhookErrors.Add(1)
		return
	}

	// Normalise ticker format
	normTicker := NormalisedTicker(alert.Ticker)

	h.logger.Info("tradingview alert received",
		zap.String("ticker", normTicker),
		zap.String("action", alert.Action),
		zap.String("strategy", alert.Strategy),
		zap.Float64("close", alert.Close),
		zap.String("ip", ip),
	)

	// Build and publish bus event
	extra := map[string]interface{}{
		"close":      alert.Close,
		"strategy":   alert.Strategy,
		"exchange":   alert.Exchange,
		"interval":   alert.Interval,
		"alert_time": alert.Time,
		"volume":     alert.Volume,
		"comment":    alert.Comment,
		"raw_ticker": alert.Ticker,
		"source_ip":  ip,
	}

	if err := h.bus.PublishSignal(r.Context(), "tradingview", normTicker, strings.ToLower(alert.Action), extra); err != nil {
		h.logger.Error("publish to bus failed", zap.Error(err))
		respondError(w, http.StatusServiceUnavailable, "could not publish to event bus")
		h.metrics.WebhookErrors.Add(1)
		return
	}

	h.metrics.WebhooksProcessed.Add(1)
	respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":         "accepted",
		"ticker":         normTicker,
		"action":         strings.ToLower(alert.Action),
		"bus_topic":      "signal.external",
	})
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// verifyHMAC validates that sig == HMAC-SHA256(secret, body) in hex.
func verifyHMAC(secret, body []byte, sig string) bool {
	mac := hmac.New(sha256.New, secret)
	mac.Write(body)
	expected := hex.EncodeToString(mac.Sum(nil))
	return hmac.Equal([]byte(expected), []byte(sig))
}

// extractIP returns the best-effort real client IP from the request.
func extractIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.SplitN(xff, ",", 2)
		return strings.TrimSpace(parts[0])
	}
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return strings.TrimSpace(xri)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// respondJSON serialises v as JSON and writes it with the given status code.
func respondJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// respondError writes a JSON error response.
func respondError(w http.ResponseWriter, status int, msg string) {
	respondJSON(w, status, map[string]string{"error": msg})
}
