package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// ---------------------------------------------------------------------------
// HTTP response helpers
// ---------------------------------------------------------------------------

type apiResponse struct {
	OK      bool        `json:"ok"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
	Elapsed string      `json:"elapsed,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func respondOK(w http.ResponseWriter, data interface{}) {
	writeJSON(w, http.StatusOK, apiResponse{OK: true, Data: data})
}

func respondError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, apiResponse{OK: false, Error: msg})
}

func decodeBody(r *http.Request, dst interface{}) error {
	defer r.Body.Close()
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	return dec.Decode(dst)
}

func requestID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// ---------------------------------------------------------------------------
// WebSocket-lite: SSE-based real-time stream (stdlib only, no gorilla)
// ---------------------------------------------------------------------------

type streamHub struct {
	mu      sync.Mutex
	clients map[chan []byte]struct{}
}

func newStreamHub() *streamHub {
	return &streamHub{clients: make(map[chan []byte]struct{})}
}

func (h *streamHub) subscribe() chan []byte {
	ch := make(chan []byte, 256)
	h.mu.Lock()
	h.clients[ch] = struct{}{}
	h.mu.Unlock()
	return ch
}

func (h *streamHub) unsubscribe(ch chan []byte) {
	h.mu.Lock()
	delete(h.clients, ch)
	h.mu.Unlock()
	close(ch)
}

func (h *streamHub) broadcast(data []byte) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.clients {
		select {
		case ch <- data:
		default:
			// slow consumer – drop
		}
	}
}

func (h *streamHub) numClients() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.clients)
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

type server struct {
	orchestrator *Orchestrator
	hub          *streamHub
	startTime    time.Time
}

func newServer() *server {
	hub := newStreamHub()
	orch := NewOrchestrator(hub)
	return &server{
		orchestrator: orch,
		hub:          hub,
		startTime:    time.Now(),
	}
}

// POST /exchange/start
func (s *server) handleStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		respondError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var cfg ExchangeConfig
	if err := decodeBody(r, &cfg); err != nil {
		respondError(w, http.StatusBadRequest, "bad config: "+err.Error())
		return
	}
	if err := cfg.Validate(); err != nil {
		respondError(w, http.StatusBadRequest, "invalid config: "+err.Error())
		return
	}
	if err := s.orchestrator.Start(cfg); err != nil {
		respondError(w, http.StatusConflict, err.Error())
		return
	}
	respondOK(w, map[string]string{"status": "started", "session_id": s.orchestrator.SessionID()})
}

// POST /exchange/stop
func (s *server) handleStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		respondError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	if err := s.orchestrator.Stop(); err != nil {
		respondError(w, http.StatusConflict, err.Error())
		return
	}
	respondOK(w, map[string]string{"status": "stopped"})
}

// POST /exchange/inject-event
func (s *server) handleInjectEvent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		respondError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}
	var evt EventInjection
	if err := decodeBody(r, &evt); err != nil {
		respondError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := s.orchestrator.InjectEvent(evt); err != nil {
		respondError(w, http.StatusBadRequest, err.Error())
		return
	}
	respondOK(w, map[string]string{"status": "injected", "event": evt.Type})
}

// GET /exchange/state
func (s *server) handleState(w http.ResponseWriter, r *http.Request) {
	st, err := s.orchestrator.GetState()
	if err != nil {
		respondError(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	respondOK(w, st)
}

// GET /exchange/metrics
func (s *server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	m := s.orchestrator.GetMetrics()
	respondOK(w, m)
}

// GET /exchange/orderbook/{symbol}
func (s *server) handleOrderbook(w http.ResponseWriter, r *http.Request) {
	symbol := strings.TrimPrefix(r.URL.Path, "/exchange/orderbook/")
	if symbol == "" {
		respondError(w, http.StatusBadRequest, "symbol required")
		return
	}
	ob, err := s.orchestrator.GetOrderBook(symbol)
	if err != nil {
		respondError(w, http.StatusNotFound, err.Error())
		return
	}
	respondOK(w, ob)
}

// GET /exchange/trades/{symbol}?n=100
func (s *server) handleTrades(w http.ResponseWriter, r *http.Request) {
	symbol := strings.TrimPrefix(r.URL.Path, "/exchange/trades/")
	if symbol == "" {
		respondError(w, http.StatusBadRequest, "symbol required")
		return
	}
	n := 100
	if ns := r.URL.Query().Get("n"); ns != "" {
		if parsed, err := strconv.Atoi(ns); err == nil && parsed > 0 {
			n = parsed
		}
	}
	trades, err := s.orchestrator.GetRecentTrades(symbol, n)
	if err != nil {
		respondError(w, http.StatusNotFound, err.Error())
		return
	}
	respondOK(w, trades)
}

// GET /exchange/agents
func (s *server) handleAgents(w http.ResponseWriter, r *http.Request) {
	summary := s.orchestrator.GetAgentSummary()
	respondOK(w, summary)
}

// GET /exchange/agents/{id}
func (s *server) handleAgentByID(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/exchange/agents/")
	if id == "" {
		respondError(w, http.StatusBadRequest, "agent id required")
		return
	}
	info, err := s.orchestrator.GetAgentInfo(id)
	if err != nil {
		respondError(w, http.StatusNotFound, err.Error())
		return
	}
	respondOK(w, info)
}

// GET /exchange/stream (SSE)
func (s *server) handleStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		respondError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	ch := s.hub.subscribe()
	defer s.hub.unsubscribe(ch)

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case data, ok := <-ch:
			if !ok {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

// GET /health
func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	respondOK(w, map[string]interface{}{
		"status":     "healthy",
		"uptime_sec": time.Since(s.startTime).Seconds(),
		"running":    s.orchestrator.IsRunning(),
		"streams":    s.hub.numClients(),
	})
}

// routeExchangeAgents dispatches /exchange/agents vs /exchange/agents/{id}
func (s *server) routeExchangeAgents(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/exchange/agents")
	path = strings.TrimPrefix(path, "/")
	if path == "" {
		s.handleAgents(w, r)
	} else {
		s.handleAgentByID(w, r)
	}
}

func (s *server) buildMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/exchange/start", s.handleStart)
	mux.HandleFunc("/exchange/stop", s.handleStop)
	mux.HandleFunc("/exchange/inject-event", s.handleInjectEvent)
	mux.HandleFunc("/exchange/state", s.handleState)
	mux.HandleFunc("/exchange/metrics", s.handleMetrics)
	mux.HandleFunc("/exchange/orderbook/", s.handleOrderbook)
	mux.HandleFunc("/exchange/trades/", s.handleTrades)
	mux.HandleFunc("/exchange/agents/", s.routeExchangeAgents)
	mux.HandleFunc("/exchange/agents", s.handleAgents)
	mux.HandleFunc("/exchange/stream", s.handleStream)
	mux.HandleFunc("/health", s.handleHealth)
	return mux
}

func main() {
	addr := ":11438"
	if env := os.Getenv("EXCHANGE_ADDR"); env != "" {
		addr = env
	}

	srv := newServer()
	mux := srv.buildMux()

	httpSrv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		log.Printf("[exchange] listening on %s", addr)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[exchange] listen error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("[exchange] shutting down …")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = srv.orchestrator.Stop()
	if err := httpSrv.Shutdown(ctx); err != nil {
		log.Fatalf("[exchange] shutdown error: %v", err)
	}
	log.Println("[exchange] stopped")
}
