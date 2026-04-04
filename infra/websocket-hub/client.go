// client.go — Client struct: read/write pumps, ping/pong keepalive, backpressure.
package wshub

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// Client configuration
// ─────────────────────────────────────────────────────────────────────────────

// ClientConfig holds per-client tuning parameters.
type ClientConfig struct {
	// WriteTimeout is the deadline for a single write operation.
	WriteTimeout time.Duration
	// PongTimeout is how long to wait for a pong response.
	PongTimeout time.Duration
	// PingInterval is how often to send pings.
	PingInterval time.Duration
	// MaxMessageSize is the maximum size of an incoming message (bytes).
	MaxMessageSize int64
	// SendQueueSize is the size of the outbound message queue.
	SendQueueSize int
	// MaxRooms is the maximum number of rooms a single client can join.
	MaxRooms int
}

// DefaultClientConfig returns sensible defaults.
func DefaultClientConfig() ClientConfig {
	return ClientConfig{
		WriteTimeout:   10 * time.Second,
		PongTimeout:    60 * time.Second,
		PingInterval:   30 * time.Second,
		MaxMessageSize: 4096,
		SendQueueSize:  256,
		MaxRooms:       50,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Client
// ─────────────────────────────────────────────────────────────────────────────

// ClientState represents the lifecycle state of a Client.
type ClientState int32

const (
	ClientStateConnecting ClientState = iota
	ClientStateConnected
	ClientStateClosing
	ClientStateClosed
)

// Client represents a connected WebSocket peer.
type Client struct {
	// ID is a unique client identifier (UUID).
	ID string

	// AccountID is the authenticated account ID (empty if anonymous).
	AccountID string

	// UserID is the authenticated user/subject (from JWT or API key).
	UserID string

	// Roles holds the client's authorisation roles.
	Roles []string

	// conn is the underlying WebSocket connection.
	conn *websocket.Conn

	// sendCh is the outbound message queue.
	sendCh chan []byte

	// rooms is the set of room names this client has joined.
	roomsMu sync.RWMutex
	rooms   map[string]bool

	// hub is a back-reference to the hub (for unregistration).
	hub *Hub

	// codec serialises/deserialises messages.
	codec *Codec

	// cfg holds per-client parameters.
	cfg ClientConfig

	// log is a client-scoped logger.
	log *zap.Logger

	// state tracks the client lifecycle.
	state atomic.Int32

	// Stats.
	messagesSent     atomic.Int64
	messagesReceived atomic.Int64
	bytesSent        atomic.Int64
	bytesReceived    atomic.Int64
	connectedAt      time.Time
	lastActivity     atomic.Value // time.Time

	// ctx / cancel for graceful shutdown.
	ctx    context.Context
	cancel context.CancelFunc

	// done is closed when both pumps have exited.
	done chan struct{}
}

// newClient constructs a Client.
func newClient(hub *Hub, conn *websocket.Conn, cfg ClientConfig, log *zap.Logger) *Client {
	ctx, cancel := context.WithCancel(context.Background())
	c := &Client{
		ID:          uuid.New().String(),
		conn:        conn,
		sendCh:      make(chan []byte, cfg.SendQueueSize),
		rooms:       make(map[string]bool),
		hub:         hub,
		codec:       hub.codec,
		cfg:         cfg,
		log:         log.With(zap.String("client_id", uuid.New().String())),
		connectedAt: time.Now().UTC(),
		ctx:         ctx,
		cancel:      cancel,
		done:        make(chan struct{}),
	}
	c.state.Store(int32(ClientStateConnecting))
	c.lastActivity.Store(time.Now())
	return c
}

// ─────────────────────────────────────────────────────────────────────────────
// Pumps
// ─────────────────────────────────────────────────────────────────────────────

// run starts the read and write pumps. Blocks until the client disconnects.
func (c *Client) run() {
	defer close(c.done)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		c.writePump()
	}()
	go func() {
		defer wg.Done()
		c.readPump()
	}()
	wg.Wait()
}

// readPump reads incoming messages from the WebSocket connection.
func (c *Client) readPump() {
	defer c.hub.unregister(c)
	defer c.cancel()

	c.conn.SetReadLimit(c.cfg.MaxMessageSize)
	_ = c.conn.SetReadDeadline(time.Now().Add(c.cfg.PongTimeout))
	c.conn.SetPongHandler(func(appData string) error {
		_ = c.conn.SetReadDeadline(time.Now().Add(c.cfg.PongTimeout))
		c.lastActivity.Store(time.Now())
		return nil
	})
	c.conn.SetPingHandler(func(appData string) error {
		_ = c.conn.SetReadDeadline(time.Now().Add(c.cfg.PongTimeout))
		c.lastActivity.Store(time.Now())
		// Echo pong.
		_ = c.conn.SetWriteDeadline(time.Now().Add(c.cfg.WriteTimeout))
		_ = c.conn.WriteMessage(websocket.PongMessage, []byte(appData))
		return nil
	})

	c.state.Store(int32(ClientStateConnected))

	for {
		msgType, data, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				c.log.Debug("unexpected close", zap.Error(err))
			}
			return
		}
		if msgType != websocket.TextMessage && msgType != websocket.BinaryMessage {
			continue
		}

		c.messagesReceived.Add(1)
		c.bytesReceived.Add(int64(len(data)))
		c.lastActivity.Store(time.Now())

		msg, err := c.codec.Decode(data)
		if err != nil {
			c.sendError("", ErrCodeBadRequest, "invalid message format", err.Error())
			continue
		}

		c.hub.handleClientMessage(c, msg)
	}
}

// writePump drains the sendCh and writes to the WebSocket connection.
// Also sends periodic pings to detect dead connections.
func (c *Client) writePump() {
	ticker := time.NewTicker(c.cfg.PingInterval)
	defer ticker.Stop()
	defer func() {
		_ = c.conn.Close()
	}()

	for {
		select {
		case <-c.ctx.Done():
			// Send close frame.
			_ = c.conn.SetWriteDeadline(time.Now().Add(c.cfg.WriteTimeout))
			_ = c.conn.WriteMessage(websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
			return

		case data, ok := <-c.sendCh:
			_ = c.conn.SetWriteDeadline(time.Now().Add(c.cfg.WriteTimeout))
			if !ok {
				_ = c.conn.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
			if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
				c.log.Debug("write error", zap.Error(err))
				return
			}
			c.messagesSent.Add(1)
			c.bytesSent.Add(int64(len(data)))

		case <-ticker.C:
			_ = c.conn.SetWriteDeadline(time.Now().Add(c.cfg.WriteTimeout))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Send helpers
// ─────────────────────────────────────────────────────────────────────────────

// Send queues a pre-encoded message for delivery. Returns false if dropped.
func (c *Client) Send(data []byte) bool {
	if ClientState(c.state.Load()) >= ClientStateClosing {
		return false
	}
	select {
	case c.sendCh <- data:
		return true
	default:
		// Queue full — back pressure.
		c.log.Warn("send queue full, dropping message", zap.String("client", c.ID))
		hubDroppedTotal.WithLabelValues("send_queue_full").Inc()
		return false
	}
}

// SendMessage encodes and queues a Message.
func (c *Client) SendMessage(msg *Message) bool {
	data, err := c.codec.Encode(msg)
	if err != nil {
		c.log.Error("encode message failed", zap.Error(err))
		return false
	}
	return c.Send(data)
}

// sendError encodes and sends an error message.
func (c *Client) sendError(requestID string, code int, errMsg, detail string) bool {
	return c.SendMessage(NewErrorMessage(requestID, code, errMsg, detail))
}

// ─────────────────────────────────────────────────────────────────────────────
// Room management (client-side)
// ─────────────────────────────────────────────────────────────────────────────

// JoinRoom adds a room to the client's membership.
func (c *Client) JoinRoom(room string) bool {
	c.roomsMu.Lock()
	defer c.roomsMu.Unlock()
	if len(c.rooms) >= c.cfg.MaxRooms {
		return false
	}
	c.rooms[room] = true
	return true
}

// LeaveRoom removes a room from the client's membership.
func (c *Client) LeaveRoom(room string) {
	c.roomsMu.Lock()
	defer c.roomsMu.Unlock()
	delete(c.rooms, room)
}

// InRoom reports whether the client is in the given room.
func (c *Client) InRoom(room string) bool {
	c.roomsMu.RLock()
	defer c.roomsMu.RUnlock()
	return c.rooms[room]
}

// Rooms returns a copy of the room membership set.
func (c *Client) Rooms() []string {
	c.roomsMu.RLock()
	defer c.roomsMu.RUnlock()
	rooms := make([]string, 0, len(c.rooms))
	for r := range c.rooms {
		rooms = append(rooms, r)
	}
	return rooms
}

// RoomCount returns the number of rooms the client is in.
func (c *Client) RoomCount() int {
	c.roomsMu.RLock()
	defer c.roomsMu.RUnlock()
	return len(c.rooms)
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

// ClientStats holds per-client metrics snapshot.
type ClientStats struct {
	ID               string
	AccountID        string
	UserID           string
	Rooms            []string
	MessagesSent     int64
	MessagesReceived int64
	BytesSent        int64
	BytesReceived    int64
	ConnectedAt      time.Time
	LastActivity     time.Time
	QueueDepth       int
	State            ClientState
}

// Stats returns a snapshot of current client metrics.
func (c *Client) Stats() ClientStats {
	lastAct, _ := c.lastActivity.Load().(time.Time)
	return ClientStats{
		ID:               c.ID,
		AccountID:        c.AccountID,
		UserID:           c.UserID,
		Rooms:            c.Rooms(),
		MessagesSent:     c.messagesSent.Load(),
		MessagesReceived: c.messagesReceived.Load(),
		BytesSent:        c.bytesSent.Load(),
		BytesReceived:    c.bytesReceived.Load(),
		ConnectedAt:      c.connectedAt,
		LastActivity:     lastAct,
		QueueDepth:       len(c.sendCh),
		State:            ClientState(c.state.Load()),
	}
}

// Close initiates graceful shutdown of the client connection.
func (c *Client) Close() {
	if c.state.CompareAndSwap(int32(ClientStateConnected), int32(ClientStateClosing)) {
		c.cancel()
	}
}

// Wait blocks until the client's pumps have exited.
func (c *Client) Wait() {
	<-c.done
}
