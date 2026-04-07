package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

// envOrDefault returns the value of an environment variable or a fallback string.
func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	if v := os.Getenv(key); v != "" {
		b, err := strconv.ParseBool(v)
		if err == nil {
			return b
		}
	}
	return fallback
}

func main() {
	cfg := FIXConfig{
		SenderCompID:      envOrDefault("FIX_SENDER_COMP_ID", "SRFM"),
		TargetCompID:      envOrDefault("FIX_TARGET_COMP_ID", "BROKER"),
		Host:              envOrDefault("FIX_HOST", "127.0.0.1"),
		Port:              envInt("FIX_PORT", 9876),
		HeartbeatInterval: envInt("FIX_HEARTBEAT_INTERVAL", 30),
		ResetOnLogon:      envBool("FIX_RESET_ON_LOGON", true),
		BeginString:       envOrDefault("FIX_BEGIN_STRING", "FIX.4.2"),
	}

	apiAddr := envOrDefault("FIX_API_ADDR", ":8080")
	metricsAddr := envOrDefault("FIX_METRICS_ADDR", ":9090")

	log.Printf("fix-gateway starting: sender=%s target=%s %s:%d",
		cfg.SenderCompID, cfg.TargetCompID, cfg.Host, cfg.Port)

	// Build session, router, and API server.
	session := NewFIXSession(cfg)
	router := NewFIXOrderRouter(session)
	apiSrv := NewFIXAPIServer(apiAddr, session, router)

	// Root context with cancel for graceful shutdown.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start router timeout scanner.
	router.Start()

	// Start FIX session with auto-reconnect in background.
	go session.ConnectWithRetry()

	// Start REST API server.
	apiErrCh := make(chan error, 1)
	go func() {
		if err := apiSrv.ListenAndServe(); err != nil {
			apiErrCh <- err
		}
	}()

	// Start minimal metrics/health server.
	metricsErrCh := make(chan error, 1)
	go func() {
		mux := http.NewServeMux()
		mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			// Expose basic counters in Prometheus text format.
			_, _ = w.Write([]byte(formatMetrics(session, router)))
		})
		mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
			if session.IsActive() {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("ok"))
			} else {
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte("session not active"))
			}
		})
		srv := &http.Server{Addr: metricsAddr, Handler: mux}
		log.Printf("fix-gateway: metrics server on %s", metricsAddr)
		if err := srv.ListenAndServe(); err != nil {
			metricsErrCh <- err
		}
	}()

	// Wait for termination signal or error.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigCh:
		log.Printf("fix-gateway: received signal %s, shutting down", sig)
	case err := <-apiErrCh:
		log.Printf("fix-gateway: API server error: %v", err)
	case err := <-metricsErrCh:
		log.Printf("fix-gateway: metrics server error: %v", err)
	}

	cancel()
	router.Stop()

	// Allow a brief window for in-flight operations to complete.
	shutdownTimer := time.NewTimer(5 * time.Second)
	defer shutdownTimer.Stop()

	// Send Logout and close.
	if err := session.Close(); err != nil {
		log.Printf("fix-gateway: session close error: %v", err)
	}

	<-shutdownTimer.C
	log.Println("fix-gateway: shutdown complete")
	_ = ctx // context used by background goroutines
}

// formatMetrics returns a minimal Prometheus-compatible metrics text block.
func formatMetrics(session *FIXSession, router *FIXOrderRouter) string {
	state := session.State()
	isActive := 0
	if state == StateActive {
		isActive = 1
	}
	return "# HELP fix_session_active 1 if session is in ACTIVE state\n" +
		"# TYPE fix_session_active gauge\n" +
		"fix_session_active " + strconv.Itoa(isActive) + "\n" +
		"# HELP fix_seq_num_out Next outgoing sequence number\n" +
		"# TYPE fix_seq_num_out counter\n" +
		"fix_seq_num_out " + strconv.FormatInt(session.SeqNumOut(), 10) + "\n" +
		"# HELP fix_orders_submitted_total Total orders submitted\n" +
		"# TYPE fix_orders_submitted_total counter\n" +
		"fix_orders_submitted_total " + strconv.FormatInt(router.SubmitCount(), 10) + "\n" +
		"# HELP fix_orders_pending Current pending order count\n" +
		"# TYPE fix_orders_pending gauge\n" +
		"fix_orders_pending " + strconv.Itoa(len(router.PendingOrders())) + "\n"
}
