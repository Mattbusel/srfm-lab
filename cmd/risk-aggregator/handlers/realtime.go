// Package handlers provides HTTP and WebSocket handlers for the risk-aggregator service.
// This file implements the real-time WebSocket risk streaming endpoint.
package handlers

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/rs/zerolog/log"
)

// PositionRisk holds per-symbol risk decomposition for the snapshot.
type PositionRisk struct {
	Symbol       string  `json:"symbol"`
	Value        float64 `json:"value_usd"`
	VaR          float64 `json:"var_usd"`
	ComponentVaR float64 `json:"component_var_usd"`
	Beta         float64 `json:"beta"`
}

// RiskSnapshot is the payload broadcast to all WebSocket subscribers every 5 seconds.
type RiskSnapshot struct {
	Timestamp             int64                 `json:"timestamp_ms"`
	PortfolioVaR99        float64               `json:"portfolio_var_99"`
	PortfolioVaR95        float64               `json:"portfolio_var_95"`
	MaxDrawdown           float64               `json:"max_drawdown_pct"`
	GrossExposure         float64               `json:"gross_exposure_usd"`
	Leverage              float64               `json:"leverage"`
	TopRiskyPositions     []PositionRisk        `json:"top_risky_positions"`
	CircuitBreakerStatus  map[string]string     `json:"circuit_breaker_status"`
}

// RiskProvider is a dependency interface so the handler can be tested
// without a real aggregator.
type RiskProvider interface {
	LatestSnapshot() (RiskSnapshot, error)
}

// client represents a single connected WebSocket subscriber.
type client struct {
	conn   *websocket.Conn
	send   chan []byte
	closed bool
}

// RealtimeRiskHandler manages the WebSocket hub and streams risk snapshots
// to all connected clients at a fixed interval.
type RealtimeRiskHandler struct {
	provider RiskProvider
	upgrader websocket.Upgrader

	mu      sync.Mutex
	clients map[*client]struct{}

	broadcast  chan []byte
	register   chan *client
	unregister chan *client

	done chan struct{}
}

// NewRealtimeRiskHandler constructs the handler and starts the hub goroutine.
func NewRealtimeRiskHandler(provider RiskProvider) *RealtimeRiskHandler {
	h := &RealtimeRiskHandler{
		provider: provider,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  1024,
			WriteBufferSize: 4096,
			CheckOrigin: func(r *http.Request) bool {
				// Allow all origins -- tighten per deployment policy.
				return true
			},
		},
		clients:    make(map[*client]struct{}),
		broadcast:  make(chan []byte, 64),
		register:   make(chan *client, 16),
		unregister: make(chan *client, 16),
		done:       make(chan struct{}),
	}
	go h.runHub()
	go h.runSnapshotter()
	return h
}

// ServeWS is the HTTP handler for GET /ws/risk.
// It upgrades the connection and registers the client with the hub.
func (h *RealtimeRiskHandler) ServeWS(w http.ResponseWriter, r *http.Request) {
	conn, err := h.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Error().Err(err).Msg("ws upgrade failed")
		return
	}

	c := &client{
		conn: conn,
		send: make(chan []byte, 32),
	}
	h.register <- c

	go h.writePump(c)
	go h.readPump(c)
}

// Close shuts down the hub and stops background goroutines.
func (h *RealtimeRiskHandler) Close() {
	close(h.done)
}

// ConnectedClients returns the number of currently connected WebSocket clients.
func (h *RealtimeRiskHandler) ConnectedClients() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.clients)
}

// runHub processes register, unregister and broadcast events serially to avoid
// data races on the clients map.
func (h *RealtimeRiskHandler) runHub() {
	for {
		select {
		case <-h.done:
			// Drain and close all clients.
			h.mu.Lock()
			for c := range h.clients {
				h.closeClient(c)
			}
			h.mu.Unlock()
			return

		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = struct{}{}
			h.mu.Unlock()
			log.Info().
				Int("total_clients", len(h.clients)).
				Msg("ws client registered")

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				h.closeClient(c)
			}
			h.mu.Unlock()

		case msg := <-h.broadcast:
			h.mu.Lock()
			for c := range h.clients {
				select {
				case c.send <- msg:
				default:
					// Slow client -- drop and disconnect.
					log.Warn().Msg("ws client too slow, disconnecting")
					h.closeClient(c)
				}
			}
			h.mu.Unlock()
		}
	}
}

// closeClient closes the send channel and removes the client from the map.
// Caller must hold h.mu.
func (h *RealtimeRiskHandler) closeClient(c *client) {
	if !c.closed {
		c.closed = true
		close(c.send)
		delete(h.clients, c)
	}
}

// runSnapshotter polls the RiskProvider every 5 seconds and broadcasts results.
func (h *RealtimeRiskHandler) runSnapshotter() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-h.done:
			return
		case <-ticker.C:
			snap, err := h.provider.LatestSnapshot()
			if err != nil {
				log.Error().Err(err).Msg("failed to fetch risk snapshot")
				continue
			}
			snap.Timestamp = time.Now().UnixMilli()

			payload, err := json.Marshal(snap)
			if err != nil {
				log.Error().Err(err).Msg("failed to marshal risk snapshot")
				continue
			}

			select {
			case h.broadcast <- payload:
			default:
				log.Warn().Msg("broadcast channel full, dropping snapshot")
			}
		}
	}
}

// writePump drains the client's send channel and forwards messages over the
// WebSocket connection. It also handles periodic ping frames.
func (h *RealtimeRiskHandler) writePump(c *client) {
	pingTicker := time.NewTicker(30 * time.Second)
	defer func() {
		pingTicker.Stop()
		c.conn.Close()
	}()

	const writeTimeout = 10 * time.Second

	for {
		select {
		case msg, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeTimeout)) //nolint:errcheck
			if !ok {
				// Channel closed -- send close frame.
				c.conn.WriteMessage(websocket.CloseMessage, //nolint:errcheck
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				log.Debug().Err(err).Msg("ws write error")
				h.unregister <- c
				return
			}

		case <-pingTicker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeTimeout)) //nolint:errcheck
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				log.Debug().Err(err).Msg("ws ping error")
				h.unregister <- c
				return
			}
		}
	}
}

// readPump consumes incoming frames (pong, close) to keep the connection alive.
// It drives the pong handler and triggers unregister on any read error.
func (h *RealtimeRiskHandler) readPump(c *client) {
	defer func() {
		h.unregister <- c
		c.conn.Close()
	}()

	const (
		pongWait   = 60 * time.Second
		maxMsgSize = 512
	)

	c.conn.SetReadLimit(maxMsgSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait)) //nolint:errcheck
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait)) //nolint:errcheck
		return nil
	})

	for {
		// We do not expect data frames from clients -- just drain.
		_, _, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err,
				websocket.CloseGoingAway,
				websocket.CloseNormalClosure,
				websocket.CloseNoStatusReceived) {
				log.Debug().Err(err).Msg("ws unexpected close")
			}
			return
		}
	}
}
