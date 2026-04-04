// rooms.go — Room management: join/leave, metadata, subscriber counts.
package wshub

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Room
// ─────────────────────────────────────────────────────────────────────────────

// RoomType categorises what kind of data a room broadcasts.
type RoomType string

const (
	RoomTypeBars      RoomType = "bars"
	RoomTypeQuotes    RoomType = "quotes"
	RoomTypeSignals   RoomType = "signals"
	RoomTypeRisk      RoomType = "risk"
	RoomTypePortfolio RoomType = "portfolio"
	RoomTypeOrderBook RoomType = "orderbook"
	RoomTypeSystem    RoomType = "system"
)

// Room tracks the membership and metadata for a single broadcast channel.
type Room struct {
	mu sync.RWMutex

	// Name is the unique room identifier.
	Name string

	// Type identifies the data category.
	Type RoomType

	// Symbol is set for per-symbol rooms (bars, quotes, book, signals).
	Symbol string

	// StrategyID is set for strategy signal rooms.
	StrategyID string

	// AccountID is set for per-account rooms (risk, portfolio).
	AccountID string

	// RequireAuth: if true, only authenticated clients may join.
	RequireAuth bool

	// AllowedRoles: if non-empty, clients must have at least one of these roles.
	AllowedRoles []string

	// clients is the set of members.
	clients map[string]*Client // client ID → Client

	// Metadata.
	CreatedAt   time.Time
	LastBroadcast time.Time
	BroadcastCount int64
}

// newRoom creates a new Room.
func newRoom(name string, roomType RoomType) *Room {
	return &Room{
		Name:      name,
		Type:      roomType,
		clients:   make(map[string]*Client),
		CreatedAt: time.Now().UTC(),
	}
}

// Add adds a client to the room. Returns false if already present.
func (r *Room) Add(c *Client) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.clients[c.ID]; ok {
		return false
	}
	r.clients[c.ID] = c
	return true
}

// Remove removes a client from the room.
func (r *Room) Remove(c *Client) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.clients, c.ID)
}

// Count returns the number of clients in the room.
func (r *Room) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.clients)
}

// IsEmpty returns true if no clients are in the room.
func (r *Room) IsEmpty() bool {
	return r.Count() == 0
}

// Broadcast sends data to all clients in the room.
// Returns the number of clients successfully enqueued.
func (r *Room) Broadcast(data []byte) int {
	r.mu.RLock()
	clients := make([]*Client, 0, len(r.clients))
	for _, c := range r.clients {
		clients = append(clients, c)
	}
	r.mu.RUnlock()

	sent := 0
	for _, c := range clients {
		if c.Send(data) {
			sent++
		}
	}

	r.mu.Lock()
	r.LastBroadcast = time.Now().UTC()
	r.BroadcastCount++
	r.mu.Unlock()

	return sent
}

// Clients returns a copy of the client list.
func (r *Room) Clients() []*Client {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*Client, 0, len(r.clients))
	for _, c := range r.clients {
		out = append(out, c)
	}
	return out
}

// CanJoin checks whether a client is allowed to join this room.
func (r *Room) CanJoin(c *Client) (bool, string) {
	if r.RequireAuth && c.UserID == "" {
		return false, "room requires authentication"
	}
	if len(r.AllowedRoles) > 0 {
		for _, required := range r.AllowedRoles {
			for _, has := range c.Roles {
				if has == required || has == "admin" {
					return true, ""
				}
			}
		}
		return false, fmt.Sprintf("room requires one of roles: %v", r.AllowedRoles)
	}
	return true, ""
}

// RoomMeta returns a read-only snapshot of room metadata.
type RoomMeta struct {
	Name           string
	Type           RoomType
	Symbol         string
	StrategyID     string
	AccountID      string
	SubscriberCount int
	BroadcastCount int64
	CreatedAt      time.Time
	LastBroadcast  time.Time
}

func (r *Room) Meta() RoomMeta {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return RoomMeta{
		Name:            r.Name,
		Type:            r.Type,
		Symbol:          r.Symbol,
		StrategyID:      r.StrategyID,
		AccountID:       r.AccountID,
		SubscriberCount: len(r.clients),
		BroadcastCount:  r.BroadcastCount,
		CreatedAt:       r.CreatedAt,
		LastBroadcast:   r.LastBroadcast,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// RoomRegistry
// ─────────────────────────────────────────────────────────────────────────────

// RoomRegistry manages all active rooms.
type RoomRegistry struct {
	mu    sync.RWMutex
	rooms map[string]*Room
}

// NewRoomRegistry creates a new RoomRegistry.
func NewRoomRegistry() *RoomRegistry {
	return &RoomRegistry{rooms: make(map[string]*Room)}
}

// GetOrCreate returns an existing room or creates it if it doesn't exist.
func (rr *RoomRegistry) GetOrCreate(name string) *Room {
	rr.mu.RLock()
	if room, ok := rr.rooms[name]; ok {
		rr.mu.RUnlock()
		return room
	}
	rr.mu.RUnlock()

	rr.mu.Lock()
	defer rr.mu.Unlock()
	// Double-check after lock upgrade.
	if room, ok := rr.rooms[name]; ok {
		return room
	}
	room := newRoom(name, inferRoomType(name))
	room.Symbol = inferSymbol(name)
	room.StrategyID = inferStrategyID(name)
	room.AccountID = inferAccountID(name)
	rr.rooms[name] = room
	return room
}

// Get returns an existing room, or nil if it doesn't exist.
func (rr *RoomRegistry) Get(name string) *Room {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	return rr.rooms[name]
}

// Remove deletes a room from the registry.
func (rr *RoomRegistry) Remove(name string) {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	delete(rr.rooms, name)
}

// RemoveEmpty removes all rooms with no subscribers.
func (rr *RoomRegistry) RemoveEmpty() int {
	rr.mu.Lock()
	defer rr.mu.Unlock()
	removed := 0
	for name, room := range rr.rooms {
		if room.IsEmpty() {
			delete(rr.rooms, name)
			removed++
		}
	}
	return removed
}

// All returns a list of all room names.
func (rr *RoomRegistry) All() []string {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	names := make([]string, 0, len(rr.rooms))
	for name := range rr.rooms {
		names = append(names, name)
	}
	return names
}

// Count returns the number of active rooms.
func (rr *RoomRegistry) Count() int {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	return len(rr.rooms)
}

// MetaList returns metadata for all rooms.
func (rr *RoomRegistry) MetaList() []RoomMeta {
	rr.mu.RLock()
	rooms := make([]*Room, 0, len(rr.rooms))
	for _, r := range rr.rooms {
		rooms = append(rooms, r)
	}
	rr.mu.RUnlock()

	out := make([]RoomMeta, len(rooms))
	for i, r := range rooms {
		out[i] = r.Meta()
	}
	return out
}

// TotalSubscribers returns the sum of subscribers across all rooms.
// Note: a client in N rooms is counted N times.
func (rr *RoomRegistry) TotalSubscribers() int {
	rr.mu.RLock()
	defer rr.mu.RUnlock()
	total := 0
	for _, r := range rr.rooms {
		total += r.Count()
	}
	return total
}

// ─────────────────────────────────────────────────────────────────────────────
// Room name inference helpers
// ─────────────────────────────────────────────────────────────────────────────

func inferRoomType(name string) RoomType {
	switch {
	case strings.HasPrefix(name, "bars:"):
		return RoomTypeBars
	case strings.HasPrefix(name, "quotes:"):
		return RoomTypeQuotes
	case strings.HasPrefix(name, "signals:"):
		return RoomTypeSignals
	case strings.HasPrefix(name, "risk:"):
		return RoomTypeRisk
	case strings.HasPrefix(name, "portfolio:"):
		return RoomTypePortfolio
	case strings.HasPrefix(name, "book:"):
		return RoomTypeOrderBook
	default:
		return RoomTypeSystem
	}
}

func inferSymbol(name string) string {
	// Patterns: "bars:AAPL:1d", "quotes:AAPL", "book:AAPL", "signals:AAPL:bh_v2"
	for _, prefix := range []string{"bars:", "quotes:", "book:", "signals:"} {
		if strings.HasPrefix(name, prefix) {
			rest := name[len(prefix):]
			// Symbol is first segment before ':'
			if idx := strings.Index(rest, ":"); idx >= 0 {
				return rest[:idx]
			}
			return rest
		}
	}
	return ""
}

func inferStrategyID(name string) string {
	// Pattern: "signals:AAPL:bh_v2"
	if strings.HasPrefix(name, "signals:") {
		parts := strings.SplitN(name[len("signals:"):], ":", 2)
		if len(parts) == 2 {
			return parts[1]
		}
	}
	return ""
}

func inferAccountID(name string) string {
	for _, prefix := range []string{"risk:", "portfolio:"} {
		if strings.HasPrefix(name, prefix) {
			return name[len(prefix):]
		}
	}
	return ""
}
