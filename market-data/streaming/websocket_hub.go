package streaming

import (
	"log"
	"net/http"
	"sync"
	"time"

	"srfm/market-data/monitoring"

	"github.com/gorilla/websocket"
)

const (
	maxClients    = 100
	pingInterval  = 30 * time.Second
	writeTimeout  = 10 * time.Second
	writeWait     = 10 * time.Second
	pongWait      = 60 * time.Second
	maxMessageSize = 512 * 1024 // 512 KB
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
	ReadBufferSize:  1024,
	WriteBufferSize: 8192,
}

// Client represents a connected WebSocket consumer.
type Client struct {
	id      string
	conn    *websocket.Conn
	send    chan []byte
	hub     *WebSocketHub
	filters map[SubscriptionKey]struct{} // nil = receive all
}

// WebSocketHub manages all connected clients and broadcasts messages.
type WebSocketHub struct {
	mu         sync.RWMutex
	clients    map[*Client]struct{}
	register   chan *Client
	unregister chan *Client
	broadcast  chan []byte
	metrics    *monitoring.Metrics
	stopped    chan struct{}
}

// NewWebSocketHub creates a new hub.
func NewWebSocketHub(metrics *monitoring.Metrics) *WebSocketHub {
	return &WebSocketHub{
		clients:    make(map[*Client]struct{}),
		register:   make(chan *Client, 16),
		unregister: make(chan *Client, 16),
		broadcast:  make(chan []byte, 4096),
		metrics:    metrics,
		stopped:    make(chan struct{}),
	}
}

// Run is the hub's main event loop. Must be called in a goroutine.
func (h *WebSocketHub) Run() {
	pingTicker := time.NewTicker(pingInterval)
	defer pingTicker.Stop()

	for {
		select {
		case <-h.stopped:
			h.mu.Lock()
			for c := range h.clients {
				c.conn.Close()
				close(c.send)
			}
			h.clients = make(map[*Client]struct{})
			h.mu.Unlock()
			return

		case c := <-h.register:
			h.mu.Lock()
			h.clients[c] = struct{}{}
			h.metrics.WSClientsActive(len(h.clients))
			h.mu.Unlock()
			log.Printf("[hub] client %s connected (%d total)", c.id, h.ClientCount())

		case c := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[c]; ok {
				delete(h.clients, c)
				close(c.send)
				h.metrics.WSClientsActive(len(h.clients))
			}
			h.mu.Unlock()
			log.Printf("[hub] client %s disconnected (%d total)", c.id, h.ClientCount())

		case msg := <-h.broadcast:
			h.mu.RLock()
			for c := range h.clients {
				select {
				case c.send <- msg:
				default:
					// Slow client; drop message
					log.Printf("[hub] slow client %s, dropping message", c.id)
				}
			}
			h.mu.RUnlock()

		case <-pingTicker.C:
			h.mu.RLock()
			for c := range h.clients {
				c.conn.SetWriteDeadline(time.Now().Add(writeTimeout))
				if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					// Will be cleaned up by reader
				}
			}
			h.mu.RUnlock()
		}
	}
}

// Stop shuts down the hub.
func (h *WebSocketHub) Stop() {
	close(h.stopped)
}

// Broadcast sends a raw message to all connected clients.
func (h *WebSocketHub) Broadcast(msg []byte) {
	select {
	case h.broadcast <- msg:
	default:
		log.Println("[hub] broadcast channel full, dropping message")
	}
}

// BroadcastFiltered sends msg only to clients whose filters match key.
func (h *WebSocketHub) BroadcastFiltered(key SubscriptionKey, msg []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for c := range h.clients {
		if c.filters == nil {
			// subscribed to everything
			select {
			case c.send <- msg:
			default:
			}
			continue
		}
		if _, ok := c.filters[key]; ok {
			select {
			case c.send <- msg:
			default:
			}
		}
	}
}

// Register adds a new client and starts its read/write pumps.
func (h *WebSocketHub) Register(conn *websocket.Conn, id string) *Client {
	c := &Client{
		id:   id,
		conn: conn,
		send: make(chan []byte, 256),
		hub:  h,
	}
	h.register <- c
	go c.writePump()
	return c
}

// Unregister removes a client.
func (h *WebSocketHub) Unregister(c *Client) {
	h.unregister <- c
}

// ClientCount returns the number of connected clients.
func (h *WebSocketHub) ClientCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.clients)
}

// SetFilters updates the subscription filters for a client.
func (h *WebSocketHub) SetFilters(c *Client, filters map[SubscriptionKey]struct{}) {
	h.mu.Lock()
	c.filters = filters
	h.mu.Unlock()
}

// writePump pumps messages from the send channel to the WebSocket connection.
func (c *Client) writePump() {
	defer func() {
		c.conn.Close()
	}()

	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})

	for msg := range c.send {
		c.conn.SetWriteDeadline(time.Now().Add(writeWait))
		if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
			return
		}
	}
}
