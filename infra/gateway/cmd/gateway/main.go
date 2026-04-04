// Command gateway is the main entry point for the SRFM market data gateway.
// It wires together all feeds, aggregators, caches, and API servers.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/srfm/gateway/internal/aggregator"
	"github.com/srfm/gateway/internal/api"
	"github.com/srfm/gateway/internal/cache"
	"github.com/srfm/gateway/internal/config"
	"github.com/srfm/gateway/internal/feed"
	"github.com/srfm/gateway/internal/hub"
	gwmetrics "github.com/srfm/gateway/internal/metrics"
	"github.com/srfm/gateway/internal/storage"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func main() {
	cfgPath := flag.String("config", "", "Path to YAML config file")
	flag.Parse()

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load config: %v\n", err)
		os.Exit(1)
	}

	log := buildLogger(cfg.LogLevel)
	defer log.Sync()

	log.Info("starting gateway", zap.String("listen", cfg.ListenAddr))

	if err := run(cfg, log); err != nil {
		log.Fatal("gateway exited with error", zap.Error(err))
	}
}

func run(cfg *config.Config, log *zap.Logger) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// ----- Prometheus metrics -----
	m := gwmetrics.New(nil)

	// ----- Bar cache -----
	barCache := cache.NewBarCache(cfg.MaxBarCache, log)
	cacheFile := "./data/cache.json"
	if err := barCache.LoadFromFile(cacheFile); err != nil {
		log.Warn("could not restore cache from disk", zap.Error(err))
	}

	// ----- Parquet writer -----
	parquetWriter := storage.NewParquetWriter(cfg.ParquetDir, log)

	// ----- Bar aggregator -----
	agg, err := aggregator.NewBarAggregator(cfg.AggregationTimeframes, log)
	if err != nil {
		return fmt.Errorf("create aggregator: %w", err)
	}

	// ----- WS hub -----
	wsHub := hub.New(cfg.WSHeartbeat, log)

	// ----- Quote store -----
	qs := newInMemoryQuoteStore()

	// ----- Symbol registry -----
	reg := newSymbolRegistry(barCache)

	// ----- REST & WS handlers -----
	restHandler := api.NewRESTHandler(barCache, wsHub, qs, reg, log)
	wsHandler := api.NewWSHandler(wsHub, log)

	// ----- Aggregator output handlers -----
	agg.AddHandler(func(tf aggregator.Timeframe, b feed.Bar) {
		barCache.Push(b.Symbol, tf.Name, b)
		wsHub.BroadcastBar(tf.Name, b)
		m.AggregatedBarsTotal.WithLabelValues(tf.Name).Inc()
		if err := parquetWriter.WriteBar(b.Symbol, tf.Name, b); err != nil {
			log.Warn("parquet write error", zap.Error(err),
				zap.String("symbol", b.Symbol),
				zap.String("timeframe", tf.Name))
			m.ParquetWriteErrors.WithLabelValues(b.Symbol).Inc()
		}
	})

	// ----- Event channel -----
	eventCh := make(chan feed.Event, 65536)

	// ----- Feeds -----
	var feeds []interface {
		Start(context.Context)
		Stop()
	}

	if cfg.EnableAlpaca && cfg.AlpacaAPIKey != "" {
		alpacaFeed := feed.NewAlpacaFeed(feed.AlpacaConfig{
			APIKey:        cfg.AlpacaAPIKey,
			Secret:        cfg.AlpacaSecret,
			Paper:         cfg.AlpacaPaper,
			StockSymbols:  cfg.Symbols,
			CryptoSymbols: cfg.CryptoSymbols,
		}, eventCh, log)
		feeds = append(feeds, alpacaFeed)
	}

	if cfg.EnableBinance && cfg.BinanceAPIKey != "" {
		binanceFeed := feed.NewBinanceFeed(feed.BinanceConfig{
			APIKey:  cfg.BinanceAPIKey,
			Secret:  cfg.BinanceSecret,
			Symbols: cfg.CryptoSymbols,
		}, eventCh, log)
		feeds = append(feeds, binanceFeed)
	}

	if cfg.EnableSimulator {
		simCfg := feed.SimulatorConfig{
			Symbols:      cfg.AllSymbols(),
			Mu:           cfg.SimulatorMu,
			Sigma:        cfg.SimulatorSigma,
			TickSize:     cfg.SimulatorTickSize,
			VolumeMean:   cfg.SimulatorVolMean,
			SpeedMult:    cfg.SimulatorSpeedMult,
			BarInterval:  time.Minute,
			InitialPrice: 100.0,
		}
		simFeed := feed.NewSimulatorFeed(simCfg, eventCh, log)
		feeds = append(feeds, simFeed)
	}

	// Start all feeds.
	for _, f := range feeds {
		f.Start(ctx)
	}
	log.Info("feeds started", zap.Int("count", len(feeds)))

	// ----- Event processor goroutine -----
	var processorWG sync.WaitGroup
	processorWG.Add(1)
	go func() {
		defer processorWG.Done()
		processEvents(ctx, eventCh, agg, barCache, wsHub, qs, m, log)
	}()

	// ----- Metrics refresh goroutine -----
	processorWG.Add(1)
	go func() {
		defer processorWG.Done()
		refreshMetrics(ctx, m, wsHub, barCache, eventCh, log)
	}()

	// ----- End-of-day flush goroutine -----
	processorWG.Add(1)
	go func() {
		defer processorWG.Done()
		runEODFlush(ctx, agg, barCache, parquetWriter, cacheFile, log)
	}()

	// ----- HTTP router -----
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)
	r.Use(middleware.RequestID)
	r.Use(loggingMiddleware(log))

	r.Mount("/", restHandler.Routes())
	r.Handle("/ws", wsHandler)
	r.Handle("/metrics", promhttp.Handler())

	srv := &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      r,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Start HTTP server.
	serverErr := make(chan error, 1)
	go func() {
		log.Info("http server listening", zap.String("addr", cfg.ListenAddr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	// Wait for shutdown signal.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-quit:
		log.Info("shutdown signal received", zap.String("signal", sig.String()))
	case err := <-serverErr:
		log.Error("http server error", zap.Error(err))
	}

	// Graceful shutdown.
	log.Info("shutting down gracefully...")
	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Warn("http server shutdown error", zap.Error(err))
	}

	// Stop feeds.
	for _, f := range feeds {
		f.Stop()
	}

	// Flush aggregator.
	agg.Flush()
	parquetWriter.FlushAll()

	// Save cache.
	if err := barCache.SaveToFile(cacheFile); err != nil {
		log.Warn("could not save cache to disk", zap.Error(err))
	}

	processorWG.Wait()
	log.Info("gateway stopped")
	return nil
}

// processEvents is the core event dispatch loop.
func processEvents(
	ctx context.Context,
	eventCh <-chan feed.Event,
	agg *aggregator.BarAggregator,
	barCache *cache.BarCache,
	wsHub *hub.Hub,
	qs *inMemoryQuoteStore,
	m *gwmetrics.Metrics,
	log *zap.Logger,
) {
	for {
		select {
		case <-ctx.Done():
			// Drain remaining events.
			for {
				select {
				case evt := <-eventCh:
					dispatchEvent(evt, agg, barCache, wsHub, qs, m)
				default:
					return
				}
			}
		case evt := <-eventCh:
			dispatchEvent(evt, agg, barCache, wsHub, qs, m)
		}
	}
}

func dispatchEvent(
	evt feed.Event,
	agg *aggregator.BarAggregator,
	barCache *cache.BarCache,
	wsHub *hub.Hub,
	qs *inMemoryQuoteStore,
	m *gwmetrics.Metrics,
) {
	switch evt.Kind {
	case feed.EventBar:
		b := evt.Bar
		if b == nil {
			return
		}
		// Record 1m bars in cache.
		if !b.IsPartial {
			barCache.Push(b.Symbol, "1m", *b)
			m.BarsReceivedTotal.WithLabelValues(b.Symbol, b.Source).Inc()
			latency := time.Since(b.Timestamp).Milliseconds()
			m.BarLatencyMs.WithLabelValues(b.Source).Observe(float64(latency))
		}
		// Feed into aggregator.
		agg.Push(*b)
		// Always broadcast 1m bars on WS (including partial).
		wsHub.BroadcastBar("1m", *b)

	case feed.EventTrade:
		if evt.Trade != nil {
			wsHub.BroadcastTrade(*evt.Trade)
		}

	case feed.EventQuote:
		if evt.Quote != nil {
			qs.Update(*evt.Quote)
			wsHub.BroadcastQuote(*evt.Quote)
		}
	}
}

// refreshMetrics periodically updates gauge metrics.
func refreshMetrics(
	ctx context.Context,
	m *gwmetrics.Metrics,
	wsHub *hub.Hub,
	barCache *cache.BarCache,
	eventCh <-chan feed.Event,
	log *zap.Logger,
) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	var lastBarCount int64
	var lastTime time.Time = time.Now()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.WSSubscribers.Set(float64(wsHub.SubscriberCount()))
			m.CacheHitRate.Set(barCache.HitRate())
			m.EventChannelDepth.Set(float64(len(eventCh)))
			m.ActiveSymbols.Set(float64(len(barCache.Symbols())))

			currentCount := int64(barCache.TotalBars())
			elapsed := time.Since(lastTime).Seconds()
			if elapsed > 0 {
				bps := float64(currentCount-lastBarCount) / elapsed
				m.BarsPerSecond.Set(bps)
			}
			lastBarCount = currentCount
			lastTime = time.Now()
		}
	}
}

// runEODFlush handles end-of-day flushing at 4pm ET (21:00 UTC weekdays).
func runEODFlush(
	ctx context.Context,
	agg *aggregator.BarAggregator,
	barCache *cache.BarCache,
	pw *storage.ParquetWriter,
	cacheFile string,
	log *zap.Logger,
) {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-ticker.C:
			utc := t.UTC()
			// 21:00 UTC = 4pm ET (ignoring DST for simplicity).
			if utc.Hour() == 21 && utc.Minute() == 0 {
				log.Info("end-of-day flush")
				agg.Flush()
				pw.FlushDate(utc.Format("2006-01-02"))
				if err := barCache.SaveToFile(cacheFile); err != nil {
					log.Warn("eod cache save", zap.Error(err))
				}
			}
		}
	}
}

// ----- Symbol registry -----

type symbolRegistry struct {
	cache *cache.BarCache
}

func newSymbolRegistry(c *cache.BarCache) *symbolRegistry {
	return &symbolRegistry{cache: c}
}

func (r *symbolRegistry) ActiveSymbols() []string {
	return r.cache.Symbols()
}

func (r *symbolRegistry) TotalBars() int {
	return r.cache.TotalBars()
}

// ----- In-memory quote store -----

type inMemoryQuoteStore struct {
	mu     sync.RWMutex
	quotes map[string]feed.Quote
}

func newInMemoryQuoteStore() *inMemoryQuoteStore {
	return &inMemoryQuoteStore{quotes: make(map[string]feed.Quote)}
}

func (s *inMemoryQuoteStore) Update(q feed.Quote) {
	s.mu.Lock()
	s.quotes[q.Symbol] = q
	s.mu.Unlock()
}

func (s *inMemoryQuoteStore) LatestQuote(symbol string) *feed.Quote {
	s.mu.RLock()
	q, ok := s.quotes[symbol]
	s.mu.RUnlock()
	if !ok {
		return nil
	}
	return &q
}

// ----- Logger -----

func buildLogger(level string) *zap.Logger {
	lvl := zapcore.InfoLevel
	switch level {
	case "debug":
		lvl = zapcore.DebugLevel
	case "warn":
		lvl = zapcore.WarnLevel
	case "error":
		lvl = zapcore.ErrorLevel
	}

	cfg := zap.NewProductionConfig()
	cfg.Level = zap.NewAtomicLevelAt(lvl)
	cfg.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	log, err := cfg.Build()
	if err != nil {
		panic(err)
	}
	return log
}

// ----- Middleware -----

// barCounter is used for logging middleware.
var requestCounter atomic.Int64

func loggingMiddleware(log *zap.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)
			log.Debug("http request",
				zap.String("method", r.Method),
				zap.String("path", r.URL.Path),
				zap.Int("status", ww.Status()),
				zap.Duration("duration", time.Since(start)),
				zap.String("remote", r.RemoteAddr))
		})
	}
}
