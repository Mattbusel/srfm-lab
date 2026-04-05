package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"srfm/market-data/aggregator"
	"srfm/market-data/api"
	"srfm/market-data/feeds"
	"srfm/market-data/monitoring"
	"srfm/market-data/storage"
	"srfm/market-data/streaming"

	_ "github.com/mattn/go-sqlite3"
)

const (
	listenAddr = ":8780"
	dbPath     = "./market_data.db"
)

// Symbols tracked by the service
var trackedSymbols = []string{
	"BTC", "ETH", "SOL", "BNB", "XRP",
	"ADA", "AVAX", "DOGE", "MATIC", "DOT",
	"LINK", "UNI", "ATOM", "LTC", "BCH",
	"ALGO", "XLM", "VET", "FIL", "AAVE",
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Printf("SRFM Market Data Service starting on %s", listenAddr)

	// Open SQLite database
	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		log.Fatalf("failed to open database: %v", err)
	}
	defer db.Close()

	db.SetMaxOpenConns(1) // SQLite single-writer
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)

	// Initialize storage
	barStore, err := storage.NewBarStore(db)
	if err != nil {
		log.Fatalf("failed to initialize bar store: %v", err)
	}

	barCache := storage.NewBarCache(500)

	// Initialize metrics
	metrics := monitoring.NewMetrics()

	// Initialize alerter
	alerter := monitoring.NewAlerter()

	// Initialize aggregator
	barAgg := aggregator.NewBarAggregator(barStore, barCache, metrics)

	// Initialize WebSocket hub
	hub := streaming.NewWebSocketHub(metrics)
	go hub.Run()

	subManager := streaming.NewSubscriptionManager(hub)

	// Wire aggregator -> hub broadcast
	barAgg.SetBroadcastFunc(func(evt aggregator.BarEvent) {
		subManager.Broadcast(evt)
	})

	// Initialize feeds
	alpacaFeed := feeds.NewAlpacaFeed(trackedSymbols, barAgg.EventChan(), metrics)
	binanceFeed := feeds.NewBinanceFeed(trackedSymbols, barAgg.EventChan(), metrics)
	feedMgr := feeds.NewFeedManager(alpacaFeed, binanceFeed, alerter, metrics)

	// Build HTTP mux
	mux := http.NewServeMux()

	// Apply middleware stack
	handler := api.NewHandlers(barStore, barCache, subManager, metrics, hub, trackedSymbols)

	mw := api.NewMiddleware(metrics)

	mux.HandleFunc("/bars/", handler.GetBars)
	mux.HandleFunc("/snapshot/", handler.GetSnapshot)
	mux.HandleFunc("/status", handler.GetStatus(feedMgr))
	mux.HandleFunc("/health", handler.GetHealth)
	mux.HandleFunc("/stream", handler.HandleWebSocket(hub, subManager))
	mux.HandleFunc("/replay/start", handler.StartReplay(barStore))
	mux.HandleFunc("/metrics", metrics.Handler())

	srv := &http.Server{
		Addr:         listenAddr,
		Handler:      mw.Apply(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Startup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Load historical data
	alpacaKey := os.Getenv("ALPACA_API_KEY")
	alpacaSecret := os.Getenv("ALPACA_API_SECRET")
	if alpacaKey != "" && alpacaSecret != "" {
		loader := storage.NewHistoricalLoader(barStore, barCache, trackedSymbols, alpacaKey, alpacaSecret)
		go func() {
			log.Println("Starting historical data load...")
			if err := loader.Load(ctx); err != nil {
				log.Printf("Historical load error: %v", err)
			} else {
				log.Println("Historical data load complete")
			}
		}()
	} else {
		log.Println("ALPACA_API_KEY/ALPACA_API_SECRET not set, skipping historical load")
	}

	// Start bar aggregator
	go barAgg.Run(ctx)

	// Start feed manager (connects feeds, manages failover)
	go feedMgr.Run(ctx)

	// Start alerter
	go alerter.Run(ctx, feedMgr)

	// Start HTTP server
	go func() {
		log.Printf("HTTP server listening on %s", listenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
	sig := <-sigCh
	fmt.Printf("\nReceived signal %v, shutting down...\n", sig)

	// Graceful shutdown
	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Printf("HTTP shutdown error: %v", err)
	}

	hub.Stop()
	barAgg.Flush()

	log.Println("Market data service stopped")
}
