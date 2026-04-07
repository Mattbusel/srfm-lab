// Package main runs the IAE metrics collection service on :8786.
// The service collects and exposes evolution metrics, fitness history,
// and parameter update events for the Idea Adaptation Engine.
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

const (
	// defaultAddr is the listen address for the metrics server.
	defaultAddr = ":8786"
	// shutdownTimeout is the maximum time to wait for in-flight requests.
	shutdownTimeout = 15 * time.Second
	// readTimeout is the maximum duration for reading the full request.
	readTimeout = 10 * time.Second
	// writeTimeout is the maximum duration for writing a response.
	writeTimeout = 30 * time.Second
	// idleTimeout is the maximum duration to keep an idle connection open.
	idleTimeout = 60 * time.Second
)

func main() {
	addr := defaultAddr
	if v := os.Getenv("METRICS_ADDR"); v != "" {
		addr = v
	}

	store := NewIAEMetrics()
	mux := buildMux(store)

	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  readTimeout,
		WriteTimeout: writeTimeout,
		IdleTimeout:  idleTimeout,
	}

	// Run the server in a goroutine so we can listen for shutdown signals.
	serverErr := make(chan error, 1)
	go func() {
		log.Printf("[metrics-server] listening on %s", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- fmt.Errorf("ListenAndServe: %w", err)
		}
		close(serverErr)
	}()

	// Block until a signal or a server error arrives.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serverErr:
		if err != nil {
			log.Fatalf("[metrics-server] fatal: %v", err)
		}
	case sig := <-quit:
		log.Printf("[metrics-server] received signal %s, shutting down", sig)
	}

	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("[metrics-server] graceful shutdown failed: %v", err)
		// Force close if shutdown timed out.
		_ = srv.Close()
		return
	}

	log.Printf("[metrics-server] shutdown complete")
}

// buildMux constructs the HTTP router and registers all routes.
func buildMux(store *IAEMetrics) *http.ServeMux {
	h := NewMetricsHandler(store)
	mux := http.NewServeMux()

	// Prometheus-style scrape endpoint.
	mux.HandleFunc("/metrics", h.HandleMetrics)

	// Structured JSON endpoints.
	mux.HandleFunc("/metrics/evolution", h.HandleEvolution)
	mux.HandleFunc("/metrics/params", h.HandleParams)
	mux.HandleFunc("/metrics/evaluations", h.HandleEvaluations)

	// Health and liveness.
	mux.HandleFunc("/health", h.HandleHealth)

	// Ingest endpoints -- called by other IAE services.
	mux.HandleFunc("/ingest/genome", h.HandleIngestGenome)
	mux.HandleFunc("/ingest/param", h.HandleIngestParam)
	mux.HandleFunc("/ingest/evaluation", h.HandleIngestEvaluation)

	return mux
}
