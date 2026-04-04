// bridge.go — Integration: bridges Redis Pub/Sub event-bus events → WebSocket broadcasts.
// Subscribes to the event bus and fans out typed updates to the correct hub rooms.
package wshub

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// BridgeConfig
// ─────────────────────────────────────────────────────────────────────────────

// BridgeConfig configures the event-bus → WebSocket bridge.
type BridgeConfig struct {
	// RedisAddr is the Redis server address.
	RedisAddr string
	// RedisPassword is the Redis auth password.
	RedisPassword string
	// RedisDB is the database index.
	RedisDB int

	// TopicPatterns is the list of Redis Pub/Sub channel patterns to subscribe to.
	// May use Redis pattern syntax (e.g., "market.bars.*", "bh.signal.*").
	TopicPatterns []string

	// Workers is the number of goroutines processing inbound events.
	Workers int

	// QueueSize is the internal event processing queue depth.
	QueueSize int

	// ThrottlePerRoom is how many messages per second to allow per room.
	ThrottlePerRoom float64
}

// DefaultBridgeConfig returns sensible defaults.
func DefaultBridgeConfig(redisAddr string) BridgeConfig {
	return BridgeConfig{
		RedisAddr: redisAddr,
		TopicPatterns: []string{
			"market.bars.*",
			"market.quotes.*",
			"market.book.*",
			"bh.signal.*",
			"bh.state.*",
			"trade.executed",
			"risk.breach",
			"risk.event.*",
			"portfolio.update.*",
			"system.alert",
		},
		Workers:         4,
		QueueSize:       4096,
		ThrottlePerRoom: 100,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// RedisBridge
// ─────────────────────────────────────────────────────────────────────────────

// RedisBridge subscribes to Redis Pub/Sub channels and routes events to the
// WebSocket hub, broadcasting them to the appropriate rooms.
type RedisBridge struct {
	cfg      BridgeConfig
	hub      *Hub
	rl       *HubRateLimiter
	log      *zap.Logger
	rdb      *redis.Client
	ps       *redis.PubSub

	// Processing pipeline.
	processCh chan *inboundEvent

	// Lifecycle.
	ctx    context.Context
	cancel context.CancelFunc
}

type inboundEvent struct {
	channel string
	payload []byte
}

// NewRedisBridge creates a RedisBridge and connects to Redis.
func NewRedisBridge(cfg BridgeConfig, hub *Hub, rl *HubRateLimiter, log *zap.Logger) (*RedisBridge, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:     cfg.RedisAddr,
		Password: cfg.RedisPassword,
		DB:       cfg.RedisDB,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}

	bridgeCtx, bridgeCancel := context.WithCancel(context.Background())
	b := &RedisBridge{
		cfg:       cfg,
		hub:       hub,
		rl:        rl,
		log:       log,
		rdb:       rdb,
		processCh: make(chan *inboundEvent, cfg.QueueSize),
		ctx:       bridgeCtx,
		cancel:    bridgeCancel,
	}
	return b, nil
}

// Start subscribes to Redis Pub/Sub and starts background workers.
func (b *RedisBridge) Start() error {
	b.ps = b.rdb.PSubscribe(b.ctx, b.cfg.TopicPatterns...)

	// Start processing workers.
	for i := 0; i < b.cfg.Workers; i++ {
		go b.processWorker()
	}

	// Start Redis receiver.
	go b.receiveLoop()

	b.log.Info("Redis bridge started",
		zap.Strings("patterns", b.cfg.TopicPatterns),
		zap.Int("workers", b.cfg.Workers))
	return nil
}

// Stop gracefully shuts down the bridge.
func (b *RedisBridge) Stop() {
	b.cancel()
	if b.ps != nil {
		_ = b.ps.Close()
	}
	_ = b.rdb.Close()
}

// receiveLoop reads messages from Redis Pub/Sub and enqueues them.
func (b *RedisBridge) receiveLoop() {
	ch := b.ps.Channel()
	for {
		select {
		case <-b.ctx.Done():
			return
		case msg, ok := <-ch:
			if !ok {
				b.log.Warn("Redis PubSub channel closed")
				return
			}
			evt := &inboundEvent{
				channel: msg.Channel,
				payload: []byte(msg.Payload),
			}
			select {
			case b.processCh <- evt:
			default:
				b.log.Warn("bridge process queue full, dropping event",
					zap.String("channel", msg.Channel))
				hubDroppedTotal.WithLabelValues("bridge_queue_full").Inc()
			}
		}
	}
}

// processWorker drains the process queue and routes events to hub rooms.
func (b *RedisBridge) processWorker() {
	for {
		select {
		case <-b.ctx.Done():
			return
		case evt := <-b.processCh:
			b.routeEvent(evt)
		}
	}
}

// routeEvent inspects the channel name and routes the event to the correct room.
func (b *RedisBridge) routeEvent(evt *inboundEvent) {
	ch := evt.channel

	// Decode the event envelope.
	var envelope struct {
		Type    string          `json:"type"`
		Topic   string          `json:"topic"`
		Payload json.RawMessage `json:"payload"`
		Source  string          `json:"source"`
	}
	if err := json.Unmarshal(evt.payload, &envelope); err != nil {
		b.log.Warn("bridge decode failed", zap.String("channel", ch), zap.Error(err))
		return
	}

	switch {
	case strings.HasPrefix(ch, "market.bars."):
		symbol := strings.TrimPrefix(ch, "market.bars.")
		b.broadcastToRoom(BarRoom(symbol, extractTimeframe(envelope.Payload)), "bar", envelope.Payload)

	case strings.HasPrefix(ch, "market.quotes."):
		symbol := strings.TrimPrefix(ch, "market.quotes.")
		b.broadcastToRoom(QuoteRoom(symbol), "quote", envelope.Payload)

	case strings.HasPrefix(ch, "market.book."):
		symbol := strings.TrimPrefix(ch, "market.book.")
		b.broadcastToRoom(OrderBookRoom(symbol), "orderbook", envelope.Payload)

	case strings.HasPrefix(ch, "bh.signal."):
		symbol := strings.TrimPrefix(ch, "bh.signal.")
		b.broadcastSignal(symbol, envelope.Payload)

	case strings.HasPrefix(ch, "bh.state."):
		symbol := strings.TrimPrefix(ch, "bh.state.")
		b.broadcastToRoom(SignalRoom(symbol, "bh_state"), "bh_state", envelope.Payload)

	case ch == "trade.executed":
		b.broadcastTrade(envelope.Payload)

	case ch == "risk.breach":
		b.broadcastRiskBreach(envelope.Payload)

	case strings.HasPrefix(ch, "risk.event."):
		accountID := strings.TrimPrefix(ch, "risk.event.")
		b.broadcastToRoom(RiskRoom(accountID), "risk_event", envelope.Payload)

	case strings.HasPrefix(ch, "portfolio.update."):
		accountID := strings.TrimPrefix(ch, "portfolio.update.")
		b.broadcastToRoom(PortfolioRoom(accountID), "portfolio_update", envelope.Payload)

	case ch == "system.alert":
		b.broadcastSystemAlert(envelope.Payload)
	}
}

// broadcastToRoom encodes a typed update and sends it to a room.
func (b *RedisBridge) broadcastToRoom(room, dataType string, payload json.RawMessage) {
	if b.rl != nil && !b.rl.AllowBroadcast(room) {
		hubRateLimitedTotal.WithLabelValues("bridge_room").Inc()
		return
	}

	if err := b.hub.BroadcastUpdate(room, dataType, payload); err != nil {
		b.log.Warn("broadcast failed", zap.String("room", room), zap.Error(err))
	}
}

// broadcastSignal routes a signal event to all strategy rooms for this symbol.
func (b *RedisBridge) broadcastSignal(symbol string, payload json.RawMessage) {
	// Decode strategy ID from payload.
	var sig struct {
		StrategyID string `json:"strategy_id"`
	}
	_ = json.Unmarshal(payload, &sig)

	// Broadcast to the specific strategy room.
	if sig.StrategyID != "" {
		b.broadcastToRoom(SignalRoom(symbol, sig.StrategyID), "signal", payload)
	}
	// Also broadcast to the wildcard "signals:<symbol>:*" room consumers.
	b.broadcastToRoom(SignalRoom(symbol, "*"), "signal", payload)
}

// broadcastTrade routes a trade event to the account's portfolio room.
func (b *RedisBridge) broadcastTrade(payload json.RawMessage) {
	var trade struct {
		AccountID string `json:"account_id"`
		Symbol    string `json:"symbol"`
	}
	_ = json.Unmarshal(payload, &trade)

	if trade.AccountID != "" {
		b.broadcastToRoom(PortfolioRoom(trade.AccountID), "trade_executed", payload)
	}
}

// broadcastRiskBreach routes a risk breach event.
func (b *RedisBridge) broadcastRiskBreach(payload json.RawMessage) {
	var breach struct {
		AccountID string `json:"account_id"`
		Severity  string `json:"severity"`
	}
	_ = json.Unmarshal(payload, &breach)

	// Per-account room.
	if breach.AccountID != "" {
		b.broadcastToRoom(RiskRoom(breach.AccountID), "risk_breach", payload)
	}

	// Critical/halt alerts go to all clients.
	if breach.Severity == "critical" || breach.Severity == "halt" {
		b.hub.BroadcastAlert(&AlertPayload{
			Level: breach.Severity,
			Title: "Risk Breach",
			Body:  fmt.Sprintf("Risk breach for account %s", breach.AccountID),
		})
	}
}

// broadcastSystemAlert fans out system alerts to all connected clients.
func (b *RedisBridge) broadcastSystemAlert(payload json.RawMessage) {
	var alert AlertPayload
	if err := json.Unmarshal(payload, &alert); err != nil {
		return
	}
	b.hub.BroadcastAlert(&alert)
}

// extractTimeframe reads the "timeframe" field from a bar payload.
func extractTimeframe(payload json.RawMessage) string {
	var bar struct {
		Timeframe string `json:"timeframe"`
	}
	_ = json.Unmarshal(payload, &bar)
	if bar.Timeframe == "" {
		return "1d"
	}
	return bar.Timeframe
}

// ─────────────────────────────────────────────────────────────────────────────
// DirectBridge — in-process bridge (no Redis; for single-binary deployments)
// ─────────────────────────────────────────────────────────────────────────────

// DirectBridgeHandler is the type of function that can be registered as an
// in-process event handler bridging to the hub.
type DirectBridgeHandler func(channel string, payload []byte)

// DirectBridge allows an in-process event bus to push events directly to the hub
// without going through Redis Pub/Sub.
type DirectBridge struct {
	hub *Hub
	rl  *HubRateLimiter
	log *zap.Logger
}

// NewDirectBridge creates a DirectBridge.
func NewDirectBridge(hub *Hub, rl *HubRateLimiter, log *zap.Logger) *DirectBridge {
	return &DirectBridge{hub: hub, rl: rl, log: log}
}

// HandleBar broadcasts a bar event directly (call from BarPublisher).
func (b *DirectBridge) HandleBar(symbol, timeframe string, payload interface{}) {
	room := BarRoom(symbol, timeframe)
	if err := b.hub.BroadcastUpdate(room, "bar", payload); err != nil {
		b.log.Warn("direct bridge bar broadcast failed", zap.Error(err))
	}
}

// HandleQuote broadcasts a quote event.
func (b *DirectBridge) HandleQuote(symbol string, payload interface{}) {
	room := QuoteRoom(symbol)
	if err := b.hub.BroadcastUpdate(room, "quote", payload); err != nil {
		b.log.Warn("direct bridge quote broadcast failed", zap.Error(err))
	}
}

// HandleSignal broadcasts a signal event.
func (b *DirectBridge) HandleSignal(symbol, strategyID string, payload interface{}) {
	room := SignalRoom(symbol, strategyID)
	if err := b.hub.BroadcastUpdate(room, "signal", payload); err != nil {
		b.log.Warn("direct bridge signal broadcast failed", zap.Error(err))
	}
}

// HandleRiskBreach broadcasts a risk breach event to the account's risk room.
func (b *DirectBridge) HandleRiskBreach(accountID string, payload interface{}, severity string) {
	room := RiskRoom(accountID)
	if err := b.hub.BroadcastUpdate(room, "risk_breach", payload); err != nil {
		b.log.Warn("direct bridge risk broadcast failed", zap.Error(err))
	}
	if severity == "critical" || severity == "halt" {
		raw, _ := json.Marshal(payload)
		var alert AlertPayload
		_ = json.Unmarshal(raw, &alert)
		alert.Level = severity
		b.hub.BroadcastAlert(&alert)
	}
}

// HandleTrade broadcasts a trade event to the account's portfolio room.
func (b *DirectBridge) HandleTrade(accountID string, payload interface{}) {
	room := PortfolioRoom(accountID)
	if err := b.hub.BroadcastUpdate(room, "trade_executed", payload); err != nil {
		b.log.Warn("direct bridge trade broadcast failed", zap.Error(err))
	}
}

// HandleOrderBook broadcasts an order book update.
func (b *DirectBridge) HandleOrderBook(symbol string, payload interface{}) {
	room := OrderBookRoom(symbol)
	if err := b.hub.BroadcastUpdate(room, "orderbook", payload); err != nil {
		b.log.Warn("direct bridge orderbook broadcast failed", zap.Error(err))
	}
}
