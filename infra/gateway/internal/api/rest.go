// Package api implements the gateway's REST and WebSocket API.
package api

import (
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/srfm/gateway/internal/cache"
	"github.com/srfm/gateway/internal/feed"
	"github.com/srfm/gateway/internal/hub"
	"go.uber.org/zap"
)

// QuoteStore is satisfied by any type that can return the latest quote for a symbol.
type QuoteStore interface {
	LatestQuote(symbol string) *feed.Quote
}

// SymbolRegistry tracks active symbols and their bar counts.
type SymbolRegistry interface {
	ActiveSymbols() []string
	TotalBars() int
}

// RESTHandler contains dependencies for REST endpoints.
type RESTHandler struct {
	cache    *cache.BarCache
	hub      *hub.Hub
	quotes   QuoteStore
	registry SymbolRegistry
	uptime   time.Time
	log      *zap.Logger
}

// NewRESTHandler creates a RESTHandler.
func NewRESTHandler(
	c *cache.BarCache,
	h *hub.Hub,
	qs QuoteStore,
	reg SymbolRegistry,
	log *zap.Logger,
) *RESTHandler {
	return &RESTHandler{
		cache:    c,
		hub:      h,
		quotes:   qs,
		registry: reg,
		uptime:   time.Now(),
		log:      log,
	}
}

// Routes returns a chi router with all REST endpoints mounted.
func (rh *RESTHandler) Routes() http.Handler {
	r := chi.NewRouter()
	r.Get("/bars/{symbol}/{timeframe}", rh.getBars)
	r.Get("/quote/{symbol}", rh.getQuote)
	r.Get("/symbols", rh.getSymbols)
	r.Get("/health", rh.getHealth)
	r.Get("/subscribers", rh.getSubscribers)
	return r
}

// getBars handles GET /bars/{symbol}/{timeframe}?from=&to=&limit=
func (rh *RESTHandler) getBars(w http.ResponseWriter, r *http.Request) {
	symbol := chi.URLParam(r, "symbol")
	timeframe := chi.URLParam(r, "timeframe")

	q := r.URL.Query()
	var from, to time.Time

	if fs := q.Get("from"); fs != "" {
		t, err := parseTime(fs)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid from: "+err.Error())
			return
		}
		from = t
	} else {
		from = time.Now().Add(-7 * 24 * time.Hour)
	}

	if ts := q.Get("to"); ts != "" {
		t, err := parseTime(ts)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid to: "+err.Error())
			return
		}
		to = t
	} else {
		to = time.Now()
	}

	if !from.IsZero() && !to.IsZero() && from.After(to) {
		writeError(w, http.StatusBadRequest, "from must be before to")
		return
	}

	limit := 0
	if ls := q.Get("limit"); ls != "" {
		n, err := strconv.Atoi(ls)
		if err != nil || n <= 0 {
			writeError(w, http.StatusBadRequest, "invalid limit")
			return
		}
		limit = n
	}

	bars := rh.cache.GetBars(symbol, timeframe, from, to)
	if bars == nil {
		bars = []feed.Bar{}
	}

	// Sort by timestamp ascending.
	sort.Slice(bars, func(i, j int) bool {
		return bars[i].Timestamp.Before(bars[j].Timestamp)
	})

	if limit > 0 && len(bars) > limit {
		bars = bars[len(bars)-limit:]
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"symbol":    symbol,
		"timeframe": timeframe,
		"from":      from,
		"to":        to,
		"count":     len(bars),
		"bars":      bars,
	})
}

// getQuote handles GET /quote/{symbol}
func (rh *RESTHandler) getQuote(w http.ResponseWriter, r *http.Request) {
	symbol := chi.URLParam(r, "symbol")
	q := rh.quotes.LatestQuote(symbol)
	if q == nil {
		// Fall back to latest bar close.
		latest := rh.cache.GetLatest(symbol, "1m")
		if latest == nil {
			writeError(w, http.StatusNotFound, "no data for symbol "+symbol)
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"symbol":    symbol,
			"timestamp": latest.Timestamp,
			"price":     latest.Close,
			"source":    latest.Source,
			"type":      "bar_close",
		})
		return
	}
	writeJSON(w, http.StatusOK, q)
}

// getSymbols handles GET /symbols
func (rh *RESTHandler) getSymbols(w http.ResponseWriter, r *http.Request) {
	symbols := rh.registry.ActiveSymbols()
	sort.Strings(symbols)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"symbols": symbols,
		"count":   len(symbols),
	})
}

// getHealth handles GET /health
func (rh *RESTHandler) getHealth(w http.ResponseWriter, r *http.Request) {
	uptime := time.Since(rh.uptime)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":           "ok",
		"uptime_seconds":   uptime.Seconds(),
		"uptime":           uptime.String(),
		"bar_count":        rh.registry.TotalBars(),
		"subscriber_count": rh.hub.SubscriberCount(),
		"cache_hit_rate":   rh.cache.HitRate(),
		"active_symbols":   len(rh.registry.ActiveSymbols()),
	})
}

// getSubscribers handles GET /subscribers — returns WS subscriber details.
func (rh *RESTHandler) getSubscribers(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"subscribers": rh.hub.SubscriberInfo(),
		"count":       rh.hub.SubscriberCount(),
	})
}

// --- helpers ---

func parseTime(s string) (time.Time, error) {
	// Try RFC3339 first, then Unix timestamp.
	t, err := time.Parse(time.RFC3339, s)
	if err == nil {
		return t, nil
	}
	n, err2 := strconv.ParseInt(s, 10, 64)
	if err2 == nil {
		return time.Unix(n, 0).UTC(), nil
	}
	return time.Time{}, err
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
