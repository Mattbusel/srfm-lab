package api

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"srfm/market-data/aggregator"
	"srfm/market-data/feeds"
	"srfm/market-data/monitoring"
	"srfm/market-data/streaming"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// BarQuerier supports historical bar queries.
type BarQuerier interface {
	QueryBars(symbol, timeframe string, start, end time.Time, limit int) ([]aggregator.BarEvent, error)
	LatestBar(symbol, timeframe string) (*aggregator.BarEvent, error)
}

// BarCacher supports cached reads.
type BarCacher interface {
	Get(symbol, timeframe string, n int) []aggregator.BarEvent
	Latest(symbol, timeframe string) *aggregator.BarEvent
	HitRate() float64
}

// FeedManagerI exposes feed health.
type FeedManagerI interface {
	GetHealth() (feeds.FeedHealth, feeds.FeedHealth)
	PrimaryFeed() string
}

// Handlers holds all HTTP handler dependencies.
type Handlers struct {
	store    BarQuerier
	cache    BarCacher
	subMgr   *streaming.SubscriptionManager
	metrics  *monitoring.Metrics
	hub      *streaming.WebSocketHub
	symbols  []string
	replayer *streaming.Replayer
}

// NewHandlers creates handlers.
func NewHandlers(
	store BarQuerier,
	cache BarCacher,
	subMgr *streaming.SubscriptionManager,
	metrics *monitoring.Metrics,
	hub *streaming.WebSocketHub,
	symbols []string,
) *Handlers {
	return &Handlers{
		store:   store,
		cache:   cache,
		subMgr:  subMgr,
		metrics: metrics,
		hub:     hub,
		symbols: symbols,
	}
}

// SetReplayer injects the replayer (called from main after both are created).
func (h *Handlers) SetReplayer(r *streaming.Replayer) {
	h.replayer = r
}

// --- /bars/{symbol}/{timeframe} ---

func (h *Handlers) GetBars(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse path: /bars/{symbol}/{timeframe}
	parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/bars/"), "/")
	if len(parts) != 2 {
		writeError(w, "path must be /bars/{symbol}/{timeframe}", http.StatusBadRequest)
		return
	}
	symbol := strings.ToUpper(parts[0])
	timeframe := parts[1]

	q := r.URL.Query()
	limit := parseIntDefault(q.Get("limit"), 500)
	if limit > 5000 {
		limit = 5000
	}

	now := time.Now().UTC()
	start := parseTimeDefault(q.Get("start"), now.Add(-7*24*time.Hour))
	end := parseTimeDefault(q.Get("end"), now)

	// Try cache first
	cached := h.cache.Get(symbol, timeframe, limit)
	if len(cached) > 0 {
		h.metrics.CacheHit()
		writeJSONCompressed(w, r, map[string]interface{}{
			"symbol":    symbol,
			"timeframe": timeframe,
			"bars":      cached,
			"source":    "cache",
		})
		return
	}

	h.metrics.CacheMiss()
	bars, err := h.store.QueryBars(symbol, timeframe, start, end, limit)
	if err != nil {
		log.Printf("[handler] QueryBars error: %v", err)
		writeError(w, "internal error", http.StatusInternalServerError)
		return
	}

	writeJSONCompressed(w, r, map[string]interface{}{
		"symbol":    symbol,
		"timeframe": timeframe,
		"bars":      bars,
		"source":    "store",
	})
}

// --- /snapshot/{symbol} ---

func (h *Handlers) GetSnapshot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	symbol := strings.ToUpper(strings.TrimPrefix(r.URL.Path, "/snapshot/"))
	if symbol == "" {
		writeError(w, "symbol required", http.StatusBadRequest)
		return
	}

	snapshot := make(map[string]*aggregator.BarEvent)
	for _, tf := range aggregator.Timeframes {
		latest := h.cache.Latest(symbol, tf)
		if latest == nil {
			bar, err := h.store.LatestBar(symbol, tf)
			if err == nil && bar != nil {
				latest = bar
			}
		}
		snapshot[tf] = latest
	}

	writeJSON(w, map[string]interface{}{
		"symbol":   symbol,
		"snapshot": snapshot,
	})
}

// --- /status ---

func (h *Handlers) GetStatus(mgr FeedManagerI) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		alpacaH, binanceH := mgr.GetHealth()

		type feedStatus struct {
			Name        string  `json:"name"`
			Connected   bool    `json:"connected"`
			LastBarTime string  `json:"last_bar_time"`
			BarsRcvd    int64   `json:"bars_received"`
			Errors      int64   `json:"errors"`
			LatencyMs   int64   `json:"latency_ms"`
			UptimeSec   float64 `json:"uptime_sec"`
		}

		fmtFeed := func(fh feeds.FeedHealth) feedStatus {
			lbt := ""
			if !fh.LastBarTime.IsZero() {
				lbt = fh.LastBarTime.UTC().Format(time.RFC3339)
			}
			return feedStatus{
				Name:        fh.Name,
				Connected:   fh.IsConnected,
				LastBarTime: lbt,
				BarsRcvd:    fh.BarsReceived,
				Errors:      fh.ErrorsCount,
				LatencyMs:   fh.LatencyMs,
				UptimeSec:   fh.Uptime.Seconds(),
			}
		}

		writeJSON(w, map[string]interface{}{
			"primary_feed":   mgr.PrimaryFeed(),
			"feeds":          []feedStatus{fmtFeed(alpacaH), fmtFeed(binanceH)},
			"ws_clients":     h.hub.ClientCount(),
			"cache_hit_rate": h.cache.HitRate(),
			"metrics":        h.metrics.Snapshot(),
			"timestamp":      time.Now().UTC().Format(time.RFC3339),
		})
	}
}

// --- /health ---

func (h *Handlers) GetHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, map[string]string{
		"status": "ok",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// --- /stream (WebSocket) ---

var wsUpgrader = websocket.Upgrader{
	CheckOrigin:     func(r *http.Request) bool { return true },
	ReadBufferSize:  1024,
	WriteBufferSize: 8192,
}

func (h *Handlers) HandleWebSocket(hub *streaming.WebSocketHub, subMgr *streaming.SubscriptionManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if hub.ClientCount() >= 100 {
			writeError(w, "max clients reached", http.StatusServiceUnavailable)
			return
		}

		conn, err := wsUpgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("[ws] upgrade error: %v", err)
			return
		}

		clientID := uuid.New().String()
		client := hub.Register(conn, clientID)

		// Read loop: process subscription messages
		go func() {
			defer func() {
				hub.Unregister(client)
				subMgr.RemoveClient(clientID)
				conn.Close()
			}()

			conn.SetReadLimit(512 * 1024)
			conn.SetReadDeadline(time.Now().Add(60 * time.Second))
			conn.SetPongHandler(func(string) error {
				conn.SetReadDeadline(time.Now().Add(60 * time.Second))
				return nil
			})

			for {
				_, msg, err := conn.ReadMessage()
				if err != nil {
					if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
						log.Printf("[ws] client %s read error: %v", clientID, err)
					}
					return
				}
				conn.SetReadDeadline(time.Now().Add(60 * time.Second))
				subMgr.HandleMessage(client, msg)
			}
		}()
	}
}

// --- /replay/start ---

func (h *Handlers) StartReplay(store interface {
	QueryBars(symbol, timeframe string, start, end time.Time, limit int) ([]aggregator.BarEvent, error)
}) http.HandlerFunc {
	replayer := streaming.NewReplayer(store, h.hub, h.subMgr)
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req streaming.ReplayRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, fmt.Sprintf("invalid body: %v", err), http.StatusBadRequest)
			return
		}

		sessionID, err := replayer.StartSession(req)
		if err != nil {
			writeError(w, err.Error(), http.StatusBadRequest)
			return
		}

		writeJSON(w, map[string]string{"session_id": sessionID})
	}
}

// --- helpers ---

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func writeJSONCompressed(w http.ResponseWriter, r *http.Request, v interface{}) {
	data, err := json.Marshal(v)
	if err != nil {
		writeError(w, "marshal error", http.StatusInternalServerError)
		return
	}

	if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") && len(data) > 1024 {
		w.Header().Set("Content-Encoding", "gzip")
		w.Header().Set("Content-Type", "application/json")
		gz := gzip.NewWriter(w)
		defer gz.Close()
		gz.Write(data) //nolint:errcheck
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(data) //nolint:errcheck
}

func writeError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]interface{}{ //nolint:errcheck
		"error": msg,
		"code":  code,
	})
}

func parseIntDefault(s string, def int) int {
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil || n <= 0 {
		return def
	}
	return n
}

func parseTimeDefault(s string, def time.Time) time.Time {
	if s == "" {
		return def
	}
	formats := []string{time.RFC3339, "2006-01-02", "2006-01-02T15:04:05"}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t.UTC()
		}
	}
	return def
}
