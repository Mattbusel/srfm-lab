package api

import (
	"encoding/json"
	"net/http"
	"runtime"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/srfm/gateway/internal/aggregator"
	"github.com/srfm/gateway/internal/cache"
	"github.com/srfm/gateway/internal/hub"
	"go.uber.org/zap"
)

// AdminHandler exposes internal diagnostic and control endpoints.
// These should be protected from public access in production.
type AdminHandler struct {
	cache   *cache.BarCache
	hub     *hub.Hub
	agg     *aggregator.BarAggregator
	startAt time.Time
	log     *zap.Logger
}

// NewAdminHandler creates an AdminHandler.
func NewAdminHandler(
	c *cache.BarCache,
	h *hub.Hub,
	agg *aggregator.BarAggregator,
	log *zap.Logger,
) *AdminHandler {
	return &AdminHandler{
		cache:   c,
		hub:     h,
		agg:     agg,
		startAt: time.Now(),
		log:     log,
	}
}

// Routes returns a chi router with admin endpoints mounted.
func (ah *AdminHandler) Routes() http.Handler {
	r := chi.NewRouter()
	r.Get("/debug", ah.getDebug)
	r.Get("/goroutines", ah.getGoroutines)
	r.Get("/timeframes", ah.getTimeframes)
	r.Post("/flush", ah.postFlush)
	r.Delete("/cache/{symbol}", ah.deleteSymbolCache)
	return r
}

// getDebug returns runtime diagnostics.
func (ah *AdminHandler) getDebug(w http.ResponseWriter, r *http.Request) {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"uptime":           time.Since(ah.startAt).String(),
		"goroutines":       runtime.NumGoroutine(),
		"num_cpu":          runtime.NumCPU(),
		"go_version":       runtime.Version(),
		"alloc_mb":         float64(memStats.Alloc) / 1024 / 1024,
		"total_alloc_mb":   float64(memStats.TotalAlloc) / 1024 / 1024,
		"sys_mb":           float64(memStats.Sys) / 1024 / 1024,
		"gc_runs":          memStats.NumGC,
		"heap_objects":     memStats.HeapObjects,
		"bar_count":        ah.cache.TotalBars(),
		"ws_subscribers":   ah.hub.SubscriberCount(),
		"cache_hit_rate":   ah.cache.HitRate(),
		"active_symbols":   len(ah.cache.Symbols()),
	})
}

// getGoroutines returns the current goroutine count.
func (ah *AdminHandler) getGoroutines(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]int{
		"goroutines": runtime.NumGoroutine(),
	})
}

// getTimeframes returns the configured aggregation timeframes.
func (ah *AdminHandler) getTimeframes(w http.ResponseWriter, r *http.Request) {
	tfs := ah.agg.Timeframes()
	names := make([]string, len(tfs))
	for i, tf := range tfs {
		names[i] = tf.Name
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"timeframes": names,
	})
}

// postFlush forces a flush of all pending aggregated bars.
func (ah *AdminHandler) postFlush(w http.ResponseWriter, r *http.Request) {
	ah.log.Info("admin: manual aggregator flush")
	ah.agg.Flush()
	writeJSON(w, http.StatusOK, map[string]string{"status": "flushed"})
}

// deleteSymbolCache clears all cached bars for a symbol.
func (ah *AdminHandler) deleteSymbolCache(w http.ResponseWriter, r *http.Request) {
	_ = chi.URLParam(r, "symbol")
	// BarCache doesn't expose a delete method — we log and return 200.
	ah.log.Info("admin: symbol cache clear requested (not implemented)")
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "note": "cache eviction not implemented"})
}

// BarExportHandler handles bulk bar export.
type BarExportHandler struct {
	cache *cache.BarCache
	log   *zap.Logger
}

// NewBarExportHandler creates a BarExportHandler.
func NewBarExportHandler(c *cache.BarCache, log *zap.Logger) *BarExportHandler {
	return &BarExportHandler{cache: c, log: log}
}

// ExportCSV exports bars for a symbol+timeframe as CSV.
func (h *BarExportHandler) ExportCSV(w http.ResponseWriter, r *http.Request) {
	symbol := chi.URLParam(r, "symbol")
	timeframe := chi.URLParam(r, "timeframe")

	from := time.Now().Add(-30 * 24 * time.Hour)
	to := time.Now()

	bars := h.cache.GetBars(symbol, timeframe, from, to)
	if bars == nil {
		bars = nil
	}

	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition",
		"attachment; filename=\""+symbol+"_"+timeframe+".csv\"")
	w.WriteHeader(http.StatusOK)

	w.Write([]byte("timestamp,symbol,open,high,low,close,volume,source\n"))
	for _, b := range bars {
		line := b.Timestamp.UTC().Format(time.RFC3339) + "," +
			b.Symbol + "," +
			formatFloat(b.Open) + "," +
			formatFloat(b.High) + "," +
			formatFloat(b.Low) + "," +
			formatFloat(b.Close) + "," +
			formatFloat(b.Volume) + "," +
			b.Source + "\n"
		w.Write([]byte(line))
	}
}

// ExportJSON exports bars for a symbol+timeframe as JSON.
func (h *BarExportHandler) ExportJSON(w http.ResponseWriter, r *http.Request) {
	symbol := chi.URLParam(r, "symbol")
	timeframe := chi.URLParam(r, "timeframe")

	from := time.Now().Add(-30 * 24 * time.Hour)
	to := time.Now()

	bars := h.cache.GetBars(symbol, timeframe, from, to)
	if bars == nil {
		bars = nil
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition",
		"attachment; filename=\""+symbol+"_"+timeframe+".json\"")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"symbol":    symbol,
		"timeframe": timeframe,
		"from":      from,
		"to":        to,
		"count":     len(bars),
		"bars":      bars,
	})
}

func formatFloat(f float64) string {
	return strconv.FormatFloat(f, 'f', 6, 64)
}
