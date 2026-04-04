// hub.go — Hub: central coordinator for client registry, broadcasts, and rooms.
package wshub

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// Hub configuration
// ─────────────────────────────────────────────────────────────────────────────

// HubConfig holds global hub parameters.
type HubConfig struct {
	// ServerID uniquely identifies this hub instance (for multi-server setups).
	ServerID string

	// MaxClients is the global connection limit. 0 = unlimited.
	MaxClients int

	// BroadcastWorkers is the number of goroutines draining the broadcast queue.
	BroadcastWorkers int

	// BroadcastQueueSize is the depth of the broadcast work queue.
	BroadcastQueueSize int

	// CleanupInterval is how often to remove empty rooms.
	CleanupInterval time.Duration

	// StatsInterval is how often to update Prometheus gauges.
	StatsInterval time.Duration

	// ClientCfg holds per-client defaults.
	ClientCfg ClientConfig
}

// DefaultHubConfig returns production-ready defaults.
func DefaultHubConfig() HubConfig {
	return HubConfig{
		ServerID:           "hub-1",
		MaxClients:         10_000,
		BroadcastWorkers:   16,
		BroadcastQueueSize: 8192,
		CleanupInterval:    5 * time.Minute,
		StatsInterval:      10 * time.Second,
		ClientCfg:          DefaultClientConfig(),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Broadcast job
// ─────────────────────────────────────────────────────────────────────────────

type broadcastJob struct {
	room string
	data []byte
}

// ─────────────────────────────────────────────────────────────────────────────
// Hub
// ─────────────────────────────────────────────────────────────────────────────

// Hub manages all WebSocket clients, rooms, and broadcasts.
type Hub struct {
	cfg     HubConfig
	log     *zap.Logger
	codec   *Codec
	rooms   *RoomRegistry
	auth    *Authenticator
	rl      *HubRateLimiter

	// Client registry.
	clientsMu sync.RWMutex
	clients   map[string]*Client // client ID → Client

	// Broadcast pipeline.
	broadcastCh chan broadcastJob

	// Sequence counter for outgoing messages.
	seqCounter atomic.Int64

	// Lifecycle.
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// Metrics.
	clientCount atomic.Int64
}

// NewHub creates a Hub and starts background workers.
func NewHub(cfg HubConfig, auth *Authenticator, rl *HubRateLimiter, log *zap.Logger) *Hub {
	ctx, cancel := context.WithCancel(context.Background())
	h := &Hub{
		cfg:         cfg,
		log:         log,
		codec:       NewCodec(),
		rooms:       NewRoomRegistry(),
		auth:        auth,
		rl:          rl,
		clients:     make(map[string]*Client),
		broadcastCh: make(chan broadcastJob, cfg.BroadcastQueueSize),
		ctx:         ctx,
		cancel:      cancel,
	}

	for i := 0; i < cfg.BroadcastWorkers; i++ {
		h.wg.Add(1)
		go h.broadcastWorker()
	}

	h.wg.Add(1)
	go h.cleanupLoop()

	h.wg.Add(1)
	go h.statsLoop()

	return h
}

// ─────────────────────────────────────────────────────────────────────────────
// Client lifecycle
// ─────────────────────────────────────────────────────────────────────────────

// RegisterClient adds a newly upgraded client to the hub.
// Sends the "connected" welcome message.
func (h *Hub) RegisterClient(c *Client) error {
	if h.cfg.MaxClients > 0 && int(h.clientCount.Load()) >= h.cfg.MaxClients {
		return fmt.Errorf("max clients reached (%d)", h.cfg.MaxClients)
	}

	h.clientsMu.Lock()
	h.clients[c.ID] = c
	h.clientCount.Add(1)
	h.clientsMu.Unlock()

	hubConnectedTotal.Inc()
	hubActiveConnections.Set(float64(h.clientCount.Load()))

	h.log.Info("client registered", zap.String("client", c.ID), zap.String("user", c.UserID))

	// Send welcome message.
	welcomeMsg, _ := NewMessage(MsgTypeConnected, "", "", &ConnectedPayload{
		ClientID: c.ID,
		ServerID: h.cfg.ServerID,
		Features: []string{"subscriptions", "streaming", "snapshots", "heartbeat"},
	})
	c.SendMessage(welcomeMsg)

	return nil
}

// unregister removes a client from all rooms and the hub registry.
// Called automatically from readPump.
func (h *Hub) unregister(c *Client) {
	h.clientsMu.Lock()
	if _, ok := h.clients[c.ID]; !ok {
		h.clientsMu.Unlock()
		return
	}
	delete(h.clients, c.ID)
	h.clientCount.Add(-1)
	h.clientsMu.Unlock()

	// Remove from all rooms.
	for _, roomName := range c.Rooms() {
		room := h.rooms.Get(roomName)
		if room != nil {
			room.Remove(c)
		}
	}

	hubActiveConnections.Set(float64(h.clientCount.Load()))
	h.log.Info("client unregistered", zap.String("client", c.ID))
}

// ─────────────────────────────────────────────────────────────────────────────
// Message dispatch
// ─────────────────────────────────────────────────────────────────────────────

// handleClientMessage processes an incoming message from a client.
func (h *Hub) handleClientMessage(c *Client, msg *Message) {
	switch msg.Type {
	case MsgTypeSubscribe:
		h.handleSubscribe(c, msg)
	case MsgTypeUnsubscribe:
		h.handleUnsubscribe(c, msg)
	case MsgTypePing:
		h.handlePing(c, msg)
	case MsgTypeAuth:
		h.handleAuth(c, msg)
	default:
		c.sendError(msg.ID, ErrCodeBadRequest, "unknown message type", string(msg.Type))
	}
}

func (h *Hub) handleSubscribe(c *Client, msg *Message) {
	sub, err := DecodePayload[SubscribePayload](msg)
	if err != nil {
		c.sendError(msg.ID, ErrCodeBadRequest, "invalid subscribe payload", err.Error())
		return
	}

	var joined, denied []string
	for _, roomName := range sub.Rooms {
		room := h.rooms.GetOrCreate(roomName)

		if ok, reason := room.CanJoin(c); !ok {
			denied = append(denied, roomName+" ("+reason+")")
			continue
		}
		if !c.JoinRoom(roomName) {
			denied = append(denied, roomName+" (max rooms reached)")
			continue
		}
		room.Add(c)
		joined = append(joined, roomName)

		hubRoomSubscriptions.WithLabelValues(roomName).Inc()

		// Send snapshot for the room.
		h.sendSnapshot(c, room)
	}

	ackData, _ := json.Marshal(map[string]interface{}{
		"joined": joined,
		"denied": denied,
	})
	ackMsg := &Message{
		Type:      MsgTypeAck,
		ID:        msg.ID,
		Timestamp: time.Now().UTC(),
		Payload:   ackData,
	}
	c.SendMessage(ackMsg)

	h.log.Debug("client subscribed",
		zap.String("client", c.ID),
		zap.Strings("joined", joined))
}

func (h *Hub) handleUnsubscribe(c *Client, msg *Message) {
	unsub, err := DecodePayload[UnsubscribePayload](msg)
	if err != nil {
		c.sendError(msg.ID, ErrCodeBadRequest, "invalid unsubscribe payload", err.Error())
		return
	}

	for _, roomName := range unsub.Rooms {
		c.LeaveRoom(roomName)
		room := h.rooms.Get(roomName)
		if room != nil {
			room.Remove(c)
			hubRoomSubscriptions.WithLabelValues(roomName).Dec()
		}
	}

	ackMsg, _ := NewMessage(MsgTypeAck, "", msg.ID, &AckPayload{
		RequestID: msg.ID,
		Success:   true,
		Message:   fmt.Sprintf("unsubscribed from %d rooms", len(unsub.Rooms)),
	})
	c.SendMessage(ackMsg)
}

func (h *Hub) handlePing(c *Client, msg *Message) {
	pongMsg, _ := NewMessage(MsgTypePong, "", msg.ID, &HeartbeatPayload{
		ServerTime: time.Now().UTC(),
	})
	c.SendMessage(pongMsg)
}

func (h *Hub) handleAuth(c *Client, msg *Message) {
	if h.auth == nil {
		c.sendError(msg.ID, ErrCodeInternal, "auth not configured", "")
		return
	}

	authPayload, err := DecodePayload[AuthPayload](msg)
	if err != nil {
		c.sendError(msg.ID, ErrCodeBadRequest, "invalid auth payload", err.Error())
		return
	}

	claims, err := h.auth.Authenticate(authPayload.Token, authPayload.APIKey)
	if err != nil {
		c.sendError(msg.ID, ErrCodeUnauthorized, "authentication failed", err.Error())
		return
	}

	c.UserID = claims.Subject
	c.AccountID = claims.AccountID
	c.Roles = claims.Roles

	ackMsg, _ := NewMessage(MsgTypeAck, "", msg.ID, &AckPayload{
		RequestID: msg.ID,
		Success:   true,
		Message:   "authenticated as " + claims.Subject,
	})
	c.SendMessage(ackMsg)
}

// sendSnapshot sends a data snapshot to a newly subscribed client.
func (h *Hub) sendSnapshot(c *Client, room *Room) {
	snapData, _ := json.Marshal(map[string]interface{}{
		"room":       room.Name,
		"type":       room.Type,
		"symbol":     room.Symbol,
		"subscribed": true,
	})
	snapMsg := &Message{
		Type:      MsgTypeSnapshot,
		Room:      room.Name,
		Timestamp: time.Now().UTC(),
		Payload:   snapData,
	}
	c.SendMessage(snapMsg)
}

// ─────────────────────────────────────────────────────────────────────────────
// Broadcast
// ─────────────────────────────────────────────────────────────────────────────

// Broadcast enqueues a message for delivery to all clients in room.
func (h *Hub) Broadcast(room string, data []byte) {
	select {
	case h.broadcastCh <- broadcastJob{room: room, data: data}:
		hubBroadcastTotal.Inc()
	default:
		h.log.Warn("broadcast queue full, dropping message", zap.String("room", room))
		hubDroppedTotal.WithLabelValues("broadcast_queue_full").Inc()
	}
}

// BroadcastMessage encodes and broadcasts a Message to a room.
func (h *Hub) BroadcastMessage(room string, msg *Message) error {
	msg.Sequence = h.seqCounter.Add(1)
	if msg.Timestamp.IsZero() {
		msg.Timestamp = time.Now().UTC()
	}
	data, err := h.codec.Encode(msg)
	if err != nil {
		return err
	}
	h.Broadcast(room, data)
	return nil
}

// BroadcastUpdate broadcasts a typed update to a room.
func (h *Hub) BroadcastUpdate(room, dataType string, payload interface{}) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}
	update := &UpdatePayload{
		Room:      room,
		DataType:  dataType,
		Data:      raw,
		Sequence:  h.seqCounter.Add(1),
		Timestamp: time.Now().UTC(),
	}
	msg := &Message{
		Type:      MsgTypeUpdate,
		Room:      room,
		Sequence:  update.Sequence,
		Timestamp: update.Timestamp,
	}
	updateRaw, _ := json.Marshal(update)
	msg.Payload = updateRaw
	data, err := h.codec.Encode(msg)
	if err != nil {
		return err
	}
	h.Broadcast(room, data)
	return nil
}

// SendToClient delivers a message directly to a specific client by ID.
func (h *Hub) SendToClient(clientID string, msg *Message) bool {
	h.clientsMu.RLock()
	c, ok := h.clients[clientID]
	h.clientsMu.RUnlock()
	if !ok {
		return false
	}
	return c.SendMessage(msg)
}

// SendToAccount delivers a message to all connections from a given account.
func (h *Hub) SendToAccount(accountID string, msg *Message) int {
	data, err := h.codec.Encode(msg)
	if err != nil {
		return 0
	}
	h.clientsMu.RLock()
	var targets []*Client
	for _, c := range h.clients {
		if c.AccountID == accountID {
			targets = append(targets, c)
		}
	}
	h.clientsMu.RUnlock()

	sent := 0
	for _, c := range targets {
		if c.Send(data) {
			sent++
		}
	}
	return sent
}

// broadcastWorker drains the broadcastCh and delivers to room members.
func (h *Hub) broadcastWorker() {
	defer h.wg.Done()
	for {
		select {
		case <-h.ctx.Done():
			return
		case job := <-h.broadcastCh:
			room := h.rooms.Get(job.room)
			if room == nil {
				continue
			}
			n := room.Broadcast(job.data)
			hubMessagesSentTotal.Add(float64(n))
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Global broadcast helpers
// ─────────────────────────────────────────────────────────────────────────────

// BroadcastAlert sends an alert to all connected clients.
func (h *Hub) BroadcastAlert(alert *AlertPayload) {
	msg, err := NewMessage(MsgTypeAlert, "", "", alert)
	if err != nil {
		return
	}
	data, err := h.codec.Encode(msg)
	if err != nil {
		return
	}

	h.clientsMu.RLock()
	clients := make([]*Client, 0, len(h.clients))
	for _, c := range h.clients {
		clients = append(clients, c)
	}
	h.clientsMu.RUnlock()

	for _, c := range clients {
		c.Send(data)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Management / admin
// ─────────────────────────────────────────────────────────────────────────────

// ClientCount returns the number of currently connected clients.
func (h *Hub) ClientCount() int {
	return int(h.clientCount.Load())
}

// RoomCount returns the number of active rooms.
func (h *Hub) RoomCount() int {
	return h.rooms.Count()
}

// ListRooms returns metadata for all active rooms.
func (h *Hub) ListRooms() []RoomMeta {
	return h.rooms.MetaList()
}

// DisconnectClient forcefully closes a client connection.
func (h *Hub) DisconnectClient(clientID, reason string) bool {
	h.clientsMu.RLock()
	c, ok := h.clients[clientID]
	h.clientsMu.RUnlock()
	if !ok {
		return false
	}
	errMsg, _ := NewMessage(MsgTypeDisconnected, "", "", map[string]string{"reason": reason})
	c.SendMessage(errMsg)
	c.Close()
	return true
}

// HubStats returns a snapshot of hub-level statistics.
type HubStats struct {
	ServerID       string
	ClientCount    int
	RoomCount      int
	TotalSubs      int
	BroadcastQueueDepth int
}

// Stats returns current hub metrics.
func (h *Hub) Stats() HubStats {
	return HubStats{
		ServerID:            h.cfg.ServerID,
		ClientCount:         int(h.clientCount.Load()),
		RoomCount:           h.rooms.Count(),
		TotalSubs:           h.rooms.TotalSubscribers(),
		BroadcastQueueDepth: len(h.broadcastCh),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Background loops
// ─────────────────────────────────────────────────────────────────────────────

func (h *Hub) cleanupLoop() {
	defer h.wg.Done()
	ticker := time.NewTicker(h.cfg.CleanupInterval)
	defer ticker.Stop()
	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			removed := h.rooms.RemoveEmpty()
			if removed > 0 {
				h.log.Debug("removed empty rooms", zap.Int("count", removed))
			}
		}
	}
}

func (h *Hub) statsLoop() {
	defer h.wg.Done()
	ticker := time.NewTicker(h.cfg.StatsInterval)
	defer ticker.Stop()
	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			stats := h.Stats()
			hubActiveConnections.Set(float64(stats.ClientCount))
			hubActiveRooms.Set(float64(stats.RoomCount))
			hubBroadcastQueueDepth.Set(float64(stats.BroadcastQueueDepth))
		}
	}
}

// Shutdown gracefully closes the hub.
func (h *Hub) Shutdown(ctx context.Context) {
	h.cancel()

	// Close all client connections.
	h.clientsMu.RLock()
	clients := make([]*Client, 0, len(h.clients))
	for _, c := range h.clients {
		clients = append(clients, c)
	}
	h.clientsMu.RUnlock()

	for _, c := range clients {
		c.Close()
	}

	// Wait for workers.
	done := make(chan struct{})
	go func() {
		h.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-ctx.Done():
		h.log.Warn("hub shutdown timed out")
	}
}
