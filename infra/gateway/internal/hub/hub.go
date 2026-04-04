// Package hub manages WebSocket subscriber connections and bar fan-out.
package hub

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"github.com/srfm/gateway/internal/feed"
	"go.uber.org/zap"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  4096,
	WriteBufferSize: 4096,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

// subscribeMsg is the JSON control message sent by clients.
type subscribeMsg struct {
	Action     string   `json:"action"`     // "subscribe" | "unsubscribe"
	Symbols    []string `json:"symbols"`    // symbol filter, empty = all
	Timeframes []string `json:"timeframes"` // timeframe filter, empty = all
	EventTypes []string `json:"event_types"` // "bar","trade","quote", empty = bar only
}

// outMsg is the JSON message sent to clients.
type outMsg struct {
	Type      string    `json:"type"`
	Symbol    string    `json:"symbol"`
	Timeframe string    `json:"timeframe,omitempty"`
	Timestamp time.Time `json:"timestamp"`
	Open      float64   `json:"open,omitempty"`
	High      float64   `json:"high,omitempty"`
	Low       float64   `json:"low,omitempty"`
	Close     float64   `json:"close,omitempty"`
	Volume    float64   `json:"volume,omitempty"`
	Price     float64   `json:"price,omitempty"`
	Size      float64   `json:"size,omitempty"`
	Side      string    `json:"side,omitempty"`
	BidPrice  float64   `json:"bid_price,omitempty"`
	AskPrice  float64   `json:"ask_price,omitempty"`
	Source    string    `json:"source,omitempty"`
	IsPartial bool      `json:"is_partial,omitempty"`
}

// Subscription represents a single WebSocket client's subscription state.
type Subscription struct {
	id         uint64
	conn       *websocket.Conn
	symbols    map[string]struct{}
	timeframes map[string]struct{}
	eventTypes map[string]struct{}
	send       chan []byte
	done       chan struct{}
}

func newSubscription(id uint64, conn *websocket.Conn) *Subscription {
	return &Subscription{
		id:         id,
		conn:       conn,
		symbols:    make(map[string]struct{}),
		timeframes: make(map[string]struct{}),
		eventTypes: map[string]struct{}{"bar": {}},
		send:       make(chan []byte, 512),
		done:       make(chan struct{}),
	}
}

func (s *Subscription) matches(symbol, timeframe, eventType string) bool {
	// Symbol filter.
	if len(s.symbols) > 0 {
		if _, ok := s.symbols[symbol]; !ok {
			return false
		}
	}
	// Timeframe filter.
	if len(s.timeframes) > 0 && timeframe != "" {
		if _, ok := s.timeframes[timeframe]; !ok {
			return false
		}
	}
	// Event type filter.
	if len(s.eventTypes) > 0 {
		if _, ok := s.eventTypes[eventType]; !ok {
			return false
		}
	}
	return true
}

// Hub manages all WebSocket subscribers.
type Hub struct {
	log       *zap.Logger
	heartbeat time.Duration

	mu   sync.RWMutex
	subs map[uint64]*Subscription

	nextID atomic.Uint64

	// Subscriber count exposed for metrics.
	Count atomic.Int64
}

// New creates a new Hub.
func New(heartbeat time.Duration, log *zap.Logger) *Hub {
	return &Hub{
		log:       log,
		heartbeat: heartbeat,
		subs:      make(map[uint64]*Subscription),
	}
}

// HandleConn upgrades an HTTP connection to WebSocket and registers the subscriber.
func (h *Hub) HandleConn(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.log.Warn("ws upgrade failed", zap.Error(err))
		return
	}

	id := h.nextID.Add(1)
	sub := newSubscription(id, conn)

	h.mu.Lock()
	h.subs[id] = sub
	h.mu.Unlock()
	h.Count.Add(1)

	h.log.Info("ws client connected", zap.Uint64("id", id), zap.String("remote", r.RemoteAddr))

	// Writer goroutine.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		h.writePump(sub)
	}()

	// Read pump (blocking).
	h.readPump(sub)

	// Clean up.
	close(sub.done)
	conn.Close()
	h.mu.Lock()
	delete(h.subs, id)
	h.mu.Unlock()
	h.Count.Add(-1)
	wg.Wait()
	h.log.Info("ws client disconnected", zap.Uint64("id", id))
}

// readPump handles incoming control messages from the client.
func (h *Hub) readPump(sub *Subscription) {
	conn := sub.conn
	conn.SetReadLimit(4096)
	conn.SetReadDeadline(time.Now().Add(h.heartbeat * 2))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(h.heartbeat * 2))
		return nil
	})

	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				h.log.Warn("ws read error", zap.Uint64("id", sub.id), zap.Error(err))
			}
			return
		}

		var msg subscribeMsg
		if err := json.Unmarshal(raw, &msg); err != nil {
			h.log.Warn("ws invalid message", zap.Uint64("id", sub.id), zap.Error(err))
			continue
		}
		h.applySubscribeMsg(sub, msg)
	}
}

// writePump sends queued messages and heartbeats to the client.
func (h *Hub) writePump(sub *Subscription) {
	ticker := time.NewTicker(h.heartbeat)
	defer ticker.Stop()

	for {
		select {
		case <-sub.done:
			return
		case msg, ok := <-sub.send:
			if !ok {
				return
			}
			sub.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := sub.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
				h.log.Warn("ws write error", zap.Uint64("id", sub.id), zap.Error(err))
				return
			}
		case <-ticker.C:
			sub.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := sub.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// applySubscribeMsg updates subscription state based on a client control message.
func (h *Hub) applySubscribeMsg(sub *Subscription, msg subscribeMsg) {
	h.mu.Lock()
	defer h.mu.Unlock()

	switch msg.Action {
	case "subscribe":
		for _, s := range msg.Symbols {
			sub.symbols[s] = struct{}{}
		}
		for _, tf := range msg.Timeframes {
			sub.timeframes[tf] = struct{}{}
		}
		if len(msg.EventTypes) > 0 {
			for _, et := range msg.EventTypes {
				sub.eventTypes[et] = struct{}{}
			}
		}
		h.log.Debug("ws subscribe",
			zap.Uint64("id", sub.id),
			zap.Strings("symbols", msg.Symbols),
			zap.Strings("timeframes", msg.Timeframes))

	case "unsubscribe":
		for _, s := range msg.Symbols {
			delete(sub.symbols, s)
		}
		for _, tf := range msg.Timeframes {
			delete(sub.timeframes, tf)
		}
		for _, et := range msg.EventTypes {
			delete(sub.eventTypes, et)
		}

	default:
		h.log.Warn("ws unknown action", zap.String("action", msg.Action), zap.Uint64("id", sub.id))
	}
}

// BroadcastBar fans out a completed bar to all matching subscribers.
func (h *Hub) BroadcastBar(timeframe string, b feed.Bar) {
	msg := outMsg{
		Type:      "bar",
		Symbol:    b.Symbol,
		Timeframe: timeframe,
		Timestamp: b.Timestamp,
		Open:      b.Open,
		High:      b.High,
		Low:       b.Low,
		Close:     b.Close,
		Volume:    b.Volume,
		Source:    b.Source,
		IsPartial: b.IsPartial,
	}
	raw, err := json.Marshal(msg)
	if err != nil {
		h.log.Error("marshal bar message", zap.Error(err))
		return
	}
	h.broadcast(b.Symbol, timeframe, "bar", raw)
}

// BroadcastTrade fans out a trade event to all matching subscribers.
func (h *Hub) BroadcastTrade(t feed.Trade) {
	msg := outMsg{
		Type:      "trade",
		Symbol:    t.Symbol,
		Timestamp: t.Timestamp,
		Price:     t.Price,
		Size:      t.Size,
		Side:      t.Side,
		Source:    t.Source,
	}
	raw, err := json.Marshal(msg)
	if err != nil {
		h.log.Error("marshal trade message", zap.Error(err))
		return
	}
	h.broadcast(t.Symbol, "", "trade", raw)
}

// BroadcastQuote fans out a quote event to all matching subscribers.
func (h *Hub) BroadcastQuote(q feed.Quote) {
	msg := outMsg{
		Type:      "quote",
		Symbol:    q.Symbol,
		Timestamp: q.Timestamp,
		BidPrice:  q.BidPrice,
		AskPrice:  q.AskPrice,
		Source:    q.Source,
	}
	raw, err := json.Marshal(msg)
	if err != nil {
		h.log.Error("marshal quote message", zap.Error(err))
		return
	}
	h.broadcast(q.Symbol, "", "quote", raw)
}

func (h *Hub) broadcast(symbol, timeframe, eventType string, raw []byte) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for _, sub := range h.subs {
		if sub.matches(symbol, timeframe, eventType) {
			select {
			case sub.send <- raw:
			default:
				h.log.Warn("ws subscriber send buffer full, dropping",
					zap.Uint64("id", sub.id),
					zap.String("symbol", symbol))
			}
		}
	}
}

// SubscriberCount returns the number of active WebSocket connections.
func (h *Hub) SubscriberCount() int {
	return int(h.Count.Load())
}

// SubscriberInfo returns a summary of all active subscriptions for diagnostics.
func (h *Hub) SubscriberInfo() []map[string]interface{} {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make([]map[string]interface{}, 0, len(h.subs))
	for _, sub := range h.subs {
		info := map[string]interface{}{
			"id":          fmt.Sprintf("%d", sub.id),
			"symbols":     keys(sub.symbols),
			"timeframes":  keys(sub.timeframes),
			"event_types": keys(sub.eventTypes),
			"queue_depth": len(sub.send),
		}
		out = append(out, info)
	}
	return out
}

func keys(m map[string]struct{}) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
