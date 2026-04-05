// Package websocket provides the WebSocket hub for the idea-api service.
// It broadcasts live shadow runner score updates to all connected clients.
package websocket

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		origin := r.Header.Get("Origin")
		return origin == "" ||
			origin == "http://localhost:5173" ||
			origin == "http://localhost:5174" ||
			origin == "http://localhost:5175" ||
			origin == "http://localhost:3000"
	},
	ReadBufferSize:  1024,
	WriteBufferSize: 4096,
}

// client represents a single connected WebSocket client.
type client struct {
	conn *websocket.Conn
	send chan []byte
	// topic filter: if non-empty, only messages matching this topic are forwarded.
	topicFilter string
}

// Hub manages all connected WebSocket clients and fan-out message broadcasting.
// It is safe for concurrent use.
type Hub struct {
	mu        sync.RWMutex
	clients   map[*client]struct{}
	broadcast chan []byte
	register  chan *client
	unregister chan *client
	log       *zap.Logger
}

// NewHub creates a Hub and starts the internal dispatch goroutine.
func NewHub(log *zap.Logger) *Hub {
	h := &Hub{
		clients:    make(map[*client]struct{}),
		broadcast:  make(chan []byte, 512),
		register:   make(chan *client, 64),
		unregister: make(chan *client, 64),
		log:        log,
	}
	go h.run()
	return h
}

// run is the central event loop. It processes register, unregister, and
// broadcast commands serially so that the client map is always consistent.
func (h *Hub) run() {
	for {
		select {
		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = struct{}{}
			h.mu.Unlock()
			h.log.Debug("ws client registered", zap.Int("total", h.clientCount()))

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				close(c.send)
			}
			h.mu.Unlock()
			h.log.Debug("ws client unregistered", zap.Int("total", h.clientCount()))

		case msg := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- msg:
				default:
					// Slow client — skip this message to avoid blocking the hub.
					h.log.Warn("ws: slow client, dropping message")
				}
			}
			h.mu.RUnlock()
		}
	}
}

// Broadcast sends msg to all connected clients.
func (h *Hub) Broadcast(msg []byte) {
	select {
	case h.broadcast <- msg:
	default:
		h.log.Warn("ws: broadcast channel full, dropping message")
	}
}

// BroadcastJSON marshals v to JSON and broadcasts it to all clients.
func (h *Hub) BroadcastJSON(v interface{}) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	h.Broadcast(b)
	return nil
}

// BroadcastShadowScore broadcasts a shadow score update message.
func (h *Hub) BroadcastShadowScore(variantID, name string, score, pnl float64, cycleID string) {
	msg := map[string]interface{}{
		"type":       "shadow_score",
		"variant_id": variantID,
		"name":       name,
		"score":      score,
		"pnl":        pnl,
		"cycle_id":   cycleID,
		"ts":         time.Now().UTC().Format(time.RFC3339),
	}
	_ = h.BroadcastJSON(msg)
}

// ServeWS upgrades an HTTP connection to WebSocket and handles the client
// lifecycle. Register it on GET /ws.
func (h *Hub) ServeWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.log.Warn("ws: upgrade failed", zap.Error(err))
		return
	}

	topicFilter := r.URL.Query().Get("topic")

	c := &client{
		conn:        conn,
		send:        make(chan []byte, 128),
		topicFilter: topicFilter,
	}
	h.register <- c

	// Send an initial connection acknowledgement.
	hello, _ := json.Marshal(map[string]interface{}{
		"type":    "connected",
		"service": "idea-api",
		"ts":      time.Now().UTC().Format(time.RFC3339),
	})
	c.send <- hello

	// Writer goroutine.
	go h.writePump(c)

	// Reader loop (drains incoming messages and handles pong frames).
	h.readPump(c)
}

// writePump serialises outbound messages to the WebSocket connection.
func (h *Hub) writePump(c *client) {
	ticker := time.NewTicker(30 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case msg, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				_ = c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				h.log.Debug("ws: write error", zap.Error(err))
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// readPump drains incoming messages and keeps the read deadline fresh on pong.
func (h *Hub) readPump(c *client) {
	defer func() {
		h.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(512)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, _, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err,
				websocket.CloseGoingAway,
				websocket.CloseAbnormalClosure) {
				h.log.Debug("ws: unexpected close", zap.Error(err))
			}
			break
		}
	}
}

// clientCount returns the number of currently connected clients.
func (h *Hub) clientCount() int {
	h.mu.RLock()
	n := len(h.clients)
	h.mu.RUnlock()
	return n
}

// Stats returns runtime metrics for the health endpoint.
func (h *Hub) Stats() map[string]interface{} {
	return map[string]interface{}{
		"connected_clients": h.clientCount(),
	}
}
