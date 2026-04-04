// snapshot_provider.go — Snapshot providers that deliver initial data to newly
// subscribed clients, ensuring they receive current state before streaming deltas.
package wshub

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// SnapshotProvider interface
// ─────────────────────────────────────────────────────────────────────────────

// SnapshotProvider supplies an initial data snapshot for a given room.
// Implementations fetch from the appropriate backing store (gRPC service,
// Redis cache, in-memory state, etc.).
type SnapshotProvider interface {
	// GetSnapshot returns a JSON payload representing the current state of room.
	// Returns nil, nil if no snapshot is available.
	GetSnapshot(ctx context.Context, room string) (json.RawMessage, error)
}

// ─────────────────────────────────────────────────────────────────────────────
// CachingSnapshotProvider
// ─────────────────────────────────────────────────────────────────────────────

// CachedSnapshot wraps a snapshot with a TTL.
type CachedSnapshot struct {
	Data      json.RawMessage
	CachedAt  time.Time
	TTL       time.Duration
}

// IsExpired returns true if the snapshot is past its TTL.
func (cs *CachedSnapshot) IsExpired() bool {
	return time.Since(cs.CachedAt) > cs.TTL
}

// CachingSnapshotProvider wraps another provider with an in-memory cache.
type CachingSnapshotProvider struct {
	inner    SnapshotProvider
	mu       sync.RWMutex
	cache    map[string]*CachedSnapshot
	defaultTTL time.Duration
	log      *zap.Logger
}

// NewCachingSnapshotProvider creates a caching snapshot provider.
func NewCachingSnapshotProvider(inner SnapshotProvider, defaultTTL time.Duration, log *zap.Logger) *CachingSnapshotProvider {
	return &CachingSnapshotProvider{
		inner:      inner,
		cache:      make(map[string]*CachedSnapshot),
		defaultTTL: defaultTTL,
		log:        log,
	}
}

// GetSnapshot returns the snapshot, using cache when available.
func (p *CachingSnapshotProvider) GetSnapshot(ctx context.Context, room string) (json.RawMessage, error) {
	p.mu.RLock()
	cached, ok := p.cache[room]
	p.mu.RUnlock()

	if ok && !cached.IsExpired() {
		return cached.Data, nil
	}

	// Fetch fresh.
	data, err := p.inner.GetSnapshot(ctx, room)
	if err != nil {
		if ok {
			// Return stale data on error.
			p.log.Warn("snapshot fetch failed, using stale cache",
				zap.String("room", room), zap.Error(err))
			return cached.Data, nil
		}
		return nil, err
	}

	p.mu.Lock()
	p.cache[room] = &CachedSnapshot{Data: data, CachedAt: time.Now(), TTL: p.defaultTTL}
	p.mu.Unlock()

	return data, nil
}

// Invalidate removes a room's cached snapshot.
func (p *CachingSnapshotProvider) Invalidate(room string) {
	p.mu.Lock()
	delete(p.cache, room)
	p.mu.Unlock()
}

// ─────────────────────────────────────────────────────────────────────────────
// CompositeSnapshotProvider
// ─────────────────────────────────────────────────────────────────────────────

// CompositeSnapshotProvider tries multiple providers in order, returning the
// first non-nil snapshot.
type CompositeSnapshotProvider struct {
	providers []SnapshotProvider
	log       *zap.Logger
}

// NewCompositeSnapshotProvider creates a CompositeSnapshotProvider.
func NewCompositeSnapshotProvider(providers []SnapshotProvider, log *zap.Logger) *CompositeSnapshotProvider {
	return &CompositeSnapshotProvider{providers: providers, log: log}
}

// GetSnapshot tries each provider in order.
func (p *CompositeSnapshotProvider) GetSnapshot(ctx context.Context, room string) (json.RawMessage, error) {
	for _, provider := range p.providers {
		data, err := provider.GetSnapshot(ctx, room)
		if err != nil {
			p.log.Warn("snapshot provider error", zap.Error(err))
			continue
		}
		if data != nil {
			return data, nil
		}
	}
	return nil, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemorySnapshotStore — stores the latest snapshot per room
// ─────────────────────────────────────────────────────────────────────────────

// InMemorySnapshotStore caches the most recent snapshot for each room.
// The hub's bridge layer calls UpdateSnapshot whenever a full snapshot is broadcast.
type InMemorySnapshotStore struct {
	mu        sync.RWMutex
	snapshots map[string]json.RawMessage
	updatedAt map[string]time.Time
}

// NewInMemorySnapshotStore creates an InMemorySnapshotStore.
func NewInMemorySnapshotStore() *InMemorySnapshotStore {
	return &InMemorySnapshotStore{
		snapshots: make(map[string]json.RawMessage),
		updatedAt: make(map[string]time.Time),
	}
}

// UpdateSnapshot sets the current snapshot for a room.
func (s *InMemorySnapshotStore) UpdateSnapshot(room string, data json.RawMessage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := make(json.RawMessage, len(data))
	copy(cp, data)
	s.snapshots[room] = cp
	s.updatedAt[room] = time.Now().UTC()
}

// GetSnapshot implements SnapshotProvider.
func (s *InMemorySnapshotStore) GetSnapshot(ctx context.Context, room string) (json.RawMessage, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.snapshots[room]
	if !ok {
		return nil, nil
	}
	cp := make(json.RawMessage, len(data))
	copy(cp, data)
	return cp, nil
}

// LastUpdated returns when a room's snapshot was last updated.
func (s *InMemorySnapshotStore) LastUpdated(room string) (time.Time, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	t, ok := s.updatedAt[room]
	return t, ok
}

// RoomCount returns the number of rooms with snapshots.
func (s *InMemorySnapshotStore) RoomCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.snapshots)
}

// ─────────────────────────────────────────────────────────────────────────────
// SnapshotDispatcher — sends snapshots to newly subscribed clients
// ─────────────────────────────────────────────────────────────────────────────

// SnapshotDispatcher dispatches snapshots to clients on room join.
type SnapshotDispatcher struct {
	provider SnapshotProvider
	hub      *Hub
	codec    *Codec
	log      *zap.Logger
}

// NewSnapshotDispatcher creates a SnapshotDispatcher.
func NewSnapshotDispatcher(provider SnapshotProvider, hub *Hub, log *zap.Logger) *SnapshotDispatcher {
	return &SnapshotDispatcher{
		provider: provider,
		hub:      hub,
		codec:    NewCodec(),
		log:      log,
	}
}

// SendSnapshot fetches and sends a snapshot for a room to a specific client.
func (sd *SnapshotDispatcher) SendSnapshot(ctx context.Context, c *Client, room string) error {
	data, err := sd.provider.GetSnapshot(ctx, room)
	if err != nil {
		sd.log.Warn("snapshot fetch failed",
			zap.String("room", room),
			zap.String("client", c.ID),
			zap.Error(err))
		return err
	}
	if data == nil {
		// No snapshot available — send an empty ack.
		msg := &Message{
			Type:      MsgTypeSnapshot,
			Room:      room,
			Timestamp: time.Now().UTC(),
			Payload:   json.RawMessage(`{"status":"no_snapshot_available"}`),
		}
		c.SendMessage(msg)
		return nil
	}

	snapPayload := &SnapshotPayload{
		Room:      room,
		DataType:  inferDataType(room),
		Data:      data,
		Timestamp: time.Now().UTC(),
	}
	raw, err := json.Marshal(snapPayload)
	if err != nil {
		return err
	}
	msg := &Message{
		Type:      MsgTypeSnapshot,
		Room:      room,
		Timestamp: time.Now().UTC(),
		Payload:   raw,
	}
	c.SendMessage(msg)
	return nil
}

// SendSnapshotsForRooms sends snapshots for all of a client's rooms.
func (sd *SnapshotDispatcher) SendSnapshotsForRooms(ctx context.Context, c *Client) {
	for _, room := range c.Rooms() {
		if err := sd.SendSnapshot(ctx, c, room); err != nil {
			sd.log.Warn("snapshot dispatch failed",
				zap.String("room", room),
				zap.Error(err))
		}
	}
}

// inferDataType returns the data type string for a room.
func inferDataType(room string) string {
	rt := inferRoomType(room)
	switch rt {
	case RoomTypeBars:
		return "bar"
	case RoomTypeQuotes:
		return "quote"
	case RoomTypeSignals:
		return "signal"
	case RoomTypeRisk:
		return "risk"
	case RoomTypePortfolio:
		return "portfolio"
	case RoomTypeOrderBook:
		return "orderbook"
	default:
		return "unknown"
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// SnapshotEnricher — enriches hub broadcasts with snapshot metadata
// ─────────────────────────────────────────────────────────────────────────────

// SnapshotEnricher wraps the hub's BroadcastUpdate to update the snapshot store
// whenever a full snapshot is broadcast.
type SnapshotEnricher struct {
	hub      *Hub
	store    *InMemorySnapshotStore
	log      *zap.Logger
}

// NewSnapshotEnricher creates a SnapshotEnricher.
func NewSnapshotEnricher(hub *Hub, store *InMemorySnapshotStore, log *zap.Logger) *SnapshotEnricher {
	return &SnapshotEnricher{hub: hub, store: store, log: log}
}

// BroadcastSnapshot broadcasts a full snapshot update and updates the snapshot store.
func (se *SnapshotEnricher) BroadcastSnapshot(room, dataType string, payload interface{}) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal snapshot: %w", err)
	}
	se.store.UpdateSnapshot(room, raw)
	return se.hub.BroadcastUpdate(room, dataType, payload)
}

// BroadcastDelta broadcasts a delta update (does not update snapshot store).
func (se *SnapshotEnricher) BroadcastDelta(room, dataType string, payload interface{}) error {
	return se.hub.BroadcastUpdate(room, dataType, payload)
}
