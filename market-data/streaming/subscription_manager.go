package streaming

import (
	"encoding/json"
	"log"
	"sync"

	"srfm/market-data/aggregator"
)

// SubscriptionKey identifies a (symbol, timeframe) subscription.
type SubscriptionKey struct {
	Symbol    string
	Timeframe string
}

// subscribeRequest is the JSON message clients send to subscribe.
type subscribeRequest struct {
	Action     string   `json:"action"`
	Symbols    []string `json:"symbols"`
	Timeframes []string `json:"timeframes"`
}

// SubscriptionManager tracks which clients want which (symbol, timeframe) pairs.
type SubscriptionManager struct {
	hub *WebSocketHub
	mu  sync.RWMutex
	// clientSubs maps client id -> set of SubscriptionKeys
	clientSubs map[string]map[SubscriptionKey]struct{}
}

// NewSubscriptionManager creates a SubscriptionManager.
func NewSubscriptionManager(hub *WebSocketHub) *SubscriptionManager {
	return &SubscriptionManager{
		hub:        hub,
		clientSubs: make(map[string]map[SubscriptionKey]struct{}),
	}
}

// HandleMessage processes a raw WebSocket message from a client.
func (m *SubscriptionManager) HandleMessage(c *Client, data []byte) {
	var req subscribeRequest
	if err := json.Unmarshal(data, &req); err != nil {
		log.Printf("[sub] invalid message from %s: %v", c.id, err)
		return
	}

	switch req.Action {
	case "subscribe":
		m.subscribe(c, req.Symbols, req.Timeframes)
	case "unsubscribe":
		m.unsubscribe(c, req.Symbols, req.Timeframes)
	case "subscribe_all":
		m.subscribeAll(c)
	default:
		log.Printf("[sub] unknown action %q from %s", req.Action, c.id)
	}
}

func (m *SubscriptionManager) subscribe(c *Client, symbols, timeframes []string) {
	keys := make(map[SubscriptionKey]struct{})
	for _, sym := range symbols {
		for _, tf := range timeframes {
			keys[SubscriptionKey{sym, tf}] = struct{}{}
		}
	}

	m.mu.Lock()
	existing := m.clientSubs[c.id]
	if existing == nil {
		existing = make(map[SubscriptionKey]struct{})
	}
	for k := range keys {
		existing[k] = struct{}{}
	}
	m.clientSubs[c.id] = existing
	m.mu.Unlock()

	m.hub.SetFilters(c, existing)
	log.Printf("[sub] client %s subscribed to %d pairs", c.id, len(existing))
}

func (m *SubscriptionManager) unsubscribe(c *Client, symbols, timeframes []string) {
	m.mu.Lock()
	existing := m.clientSubs[c.id]
	if existing != nil {
		for _, sym := range symbols {
			for _, tf := range timeframes {
				delete(existing, SubscriptionKey{sym, tf})
			}
		}
		m.clientSubs[c.id] = existing
	}
	m.mu.Unlock()

	m.hub.SetFilters(c, existing)
}

func (m *SubscriptionManager) subscribeAll(c *Client) {
	m.mu.Lock()
	m.clientSubs[c.id] = nil // nil = receive all
	m.mu.Unlock()
	m.hub.SetFilters(c, nil)
	log.Printf("[sub] client %s subscribed to all", c.id)
}

// RemoveClient cleans up subscription state when a client disconnects.
func (m *SubscriptionManager) RemoveClient(clientID string) {
	m.mu.Lock()
	delete(m.clientSubs, clientID)
	m.mu.Unlock()
}

// Broadcast sends a BarEvent to subscribed clients.
func (m *SubscriptionManager) Broadcast(evt aggregator.BarEvent) {
	data, err := json.Marshal(evt)
	if err != nil {
		log.Printf("[sub] marshal error: %v", err)
		return
	}
	key := SubscriptionKey{Symbol: evt.Symbol, Timeframe: evt.Timeframe}
	m.hub.BroadcastFiltered(key, data)
}
