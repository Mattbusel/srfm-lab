// session_manager.go — Session management: tracks client sessions, handles
// reconnection state, persists room subscriptions across reconnects.
package wshub

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// ─────────────────────────────────────────────────────────────────────────────
// SessionConfig
// ─────────────────────────────────────────────────────────────────────────────

// SessionConfig configures session management.
type SessionConfig struct {
	// SessionTTL is how long a session remains valid after disconnect.
	// On reconnect within this window, the client's subscriptions are restored.
	SessionTTL time.Duration

	// MaxSessionsPerUser caps how many concurrent sessions a user may have.
	MaxSessionsPerUser int

	// PersistSessions stores sessions in Redis (true) or in-memory only (false).
	PersistSessions bool

	// RedisKeyPrefix is the Redis key prefix for session storage.
	RedisKeyPrefix string
}

// DefaultSessionConfig returns sensible defaults.
func DefaultSessionConfig() SessionConfig {
	return SessionConfig{
		SessionTTL:         5 * time.Minute,
		MaxSessionsPerUser: 5,
		PersistSessions:    true,
		RedisKeyPrefix:     "wshub:session:",
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Session
// ─────────────────────────────────────────────────────────────────────────────

// Session represents a persistent client session that can survive disconnects.
type Session struct {
	// SessionID is a unique identifier for this session.
	SessionID string `json:"session_id"`

	// UserID is the authenticated user.
	UserID string `json:"user_id"`

	// AccountID is the associated account.
	AccountID string `json:"account_id"`

	// Roles holds the user's authorisation roles.
	Roles []string `json:"roles"`

	// Rooms is the list of rooms the session is subscribed to.
	Rooms []string `json:"rooms"`

	// ClientID is the current WebSocket client ID (changes on reconnect).
	ClientID string `json:"client_id,omitempty"`

	// CreatedAt is when the session was first established.
	CreatedAt time.Time `json:"created_at"`

	// LastSeen is the last time the session was active.
	LastSeen time.Time `json:"last_seen"`

	// Metadata holds arbitrary session data.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// SessionStore manages session persistence.
type SessionStore interface {
	Save(ctx context.Context, session *Session, ttl time.Duration) error
	Load(ctx context.Context, sessionID string) (*Session, error)
	Delete(ctx context.Context, sessionID string) error
	ListByUser(ctx context.Context, userID string) ([]*Session, error)
	UpdateLastSeen(ctx context.Context, sessionID string) error
}

// ─────────────────────────────────────────────────────────────────────────────
// RedisSessionStore
// ─────────────────────────────────────────────────────────────────────────────

// RedisSessionStore persists sessions in Redis.
type RedisSessionStore struct {
	rdb    *redis.Client
	prefix string
	log    *zap.Logger
}

// NewRedisSessionStore creates a RedisSessionStore.
func NewRedisSessionStore(rdb *redis.Client, prefix string, log *zap.Logger) *RedisSessionStore {
	return &RedisSessionStore{rdb: rdb, prefix: prefix, log: log}
}

func (s *RedisSessionStore) key(sessionID string) string {
	return s.prefix + sessionID
}

func (s *RedisSessionStore) userIndexKey(userID string) string {
	return s.prefix + "user:" + userID
}

// Save stores a session in Redis with the given TTL.
func (s *RedisSessionStore) Save(ctx context.Context, session *Session, ttl time.Duration) error {
	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("marshal session: %w", err)
	}
	pipe := s.rdb.Pipeline()
	pipe.Set(ctx, s.key(session.SessionID), data, ttl)
	// Add to user's session index.
	pipe.SAdd(ctx, s.userIndexKey(session.UserID), session.SessionID)
	pipe.Expire(ctx, s.userIndexKey(session.UserID), ttl+time.Minute)
	_, err = pipe.Exec(ctx)
	return err
}

// Load retrieves a session by ID.
func (s *RedisSessionStore) Load(ctx context.Context, sessionID string) (*Session, error) {
	data, err := s.rdb.Get(ctx, s.key(sessionID)).Bytes()
	if err != nil {
		if err == redis.Nil {
			return nil, nil // session expired or not found
		}
		return nil, fmt.Errorf("redis GET: %w", err)
	}
	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("unmarshal session: %w", err)
	}
	return &session, nil
}

// Delete removes a session.
func (s *RedisSessionStore) Delete(ctx context.Context, sessionID string) error {
	return s.rdb.Del(ctx, s.key(sessionID)).Err()
}

// ListByUser returns all active sessions for a user.
func (s *RedisSessionStore) ListByUser(ctx context.Context, userID string) ([]*Session, error) {
	ids, err := s.rdb.SMembers(ctx, s.userIndexKey(userID)).Result()
	if err != nil {
		return nil, fmt.Errorf("SMembers: %w", err)
	}
	var sessions []*Session
	for _, id := range ids {
		sess, err := s.Load(ctx, id)
		if err != nil {
			s.log.Warn("failed to load session", zap.String("id", id), zap.Error(err))
			continue
		}
		if sess != nil {
			sessions = append(sessions, sess)
		} else {
			// Expired — remove from index.
			_ = s.rdb.SRem(ctx, s.userIndexKey(userID), id).Err()
		}
	}
	return sessions, nil
}

// UpdateLastSeen updates the lastSeen timestamp without changing other fields.
func (s *RedisSessionStore) UpdateLastSeen(ctx context.Context, sessionID string) error {
	sess, err := s.Load(ctx, sessionID)
	if err != nil || sess == nil {
		return err
	}
	sess.LastSeen = time.Now().UTC()
	data, _ := json.Marshal(sess)
	return s.rdb.Set(ctx, s.key(sessionID), data, redis.KeepTTL).Err()
}

// ─────────────────────────────────────────────────────────────────────────────
// InMemorySessionStore
// ─────────────────────────────────────────────────────────────────────────────

// InMemorySessionStore is a non-persistent session store for testing/single-node.
type InMemorySessionStore struct {
	mu       sync.RWMutex
	sessions map[string]*inMemoryEntry
	log      *zap.Logger
}

type inMemoryEntry struct {
	session   *Session
	expiresAt time.Time
}

// NewInMemorySessionStore creates an InMemorySessionStore.
func NewInMemorySessionStore(log *zap.Logger) *InMemorySessionStore {
	store := &InMemorySessionStore{
		sessions: make(map[string]*inMemoryEntry),
		log:      log,
	}
	go store.evictionLoop()
	return store
}

func (s *InMemorySessionStore) Save(ctx context.Context, session *Session, ttl time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	cp := *session
	s.sessions[session.SessionID] = &inMemoryEntry{
		session:   &cp,
		expiresAt: time.Now().Add(ttl),
	}
	return nil
}

func (s *InMemorySessionStore) Load(ctx context.Context, sessionID string) (*Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	entry, ok := s.sessions[sessionID]
	if !ok || time.Now().After(entry.expiresAt) {
		return nil, nil
	}
	cp := *entry.session
	return &cp, nil
}

func (s *InMemorySessionStore) Delete(ctx context.Context, sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, sessionID)
	return nil
}

func (s *InMemorySessionStore) ListByUser(ctx context.Context, userID string) ([]*Session, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []*Session
	now := time.Now()
	for _, entry := range s.sessions {
		if entry.session.UserID == userID && now.Before(entry.expiresAt) {
			cp := *entry.session
			out = append(out, &cp)
		}
	}
	return out, nil
}

func (s *InMemorySessionStore) UpdateLastSeen(ctx context.Context, sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.sessions[sessionID]
	if !ok {
		return nil
	}
	entry.session.LastSeen = time.Now().UTC()
	return nil
}

func (s *InMemorySessionStore) evictionLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		s.mu.Lock()
		now := time.Now()
		for id, entry := range s.sessions {
			if now.After(entry.expiresAt) {
				delete(s.sessions, id)
			}
		}
		s.mu.Unlock()
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionManager
// ─────────────────────────────────────────────────────────────────────────────

// SessionManager handles session lifecycle and reconnect state restoration.
type SessionManager struct {
	cfg   SessionConfig
	store SessionStore
	hub   *Hub
	log   *zap.Logger
}

// NewSessionManager creates a SessionManager.
func NewSessionManager(cfg SessionConfig, store SessionStore, hub *Hub, log *zap.Logger) *SessionManager {
	return &SessionManager{cfg: cfg, store: store, hub: hub, log: log}
}

// CreateSession creates a new session for an authenticated client.
func (sm *SessionManager) CreateSession(ctx context.Context, c *Client) (*Session, error) {
	session := &Session{
		SessionID: uuid.New().String(),
		UserID:    c.UserID,
		AccountID: c.AccountID,
		Roles:     c.Roles,
		ClientID:  c.ID,
		Rooms:     c.Rooms(),
		CreatedAt: time.Now().UTC(),
		LastSeen:  time.Now().UTC(),
	}

	// Enforce max sessions per user.
	if sm.cfg.MaxSessionsPerUser > 0 {
		existing, err := sm.store.ListByUser(ctx, c.UserID)
		if err == nil && len(existing) >= sm.cfg.MaxSessionsPerUser {
			// Evict the oldest session.
			oldest := existing[0]
			for _, s := range existing[1:] {
				if s.LastSeen.Before(oldest.LastSeen) {
					oldest = s
				}
			}
			_ = sm.store.Delete(ctx, oldest.SessionID)
		}
	}

	if err := sm.store.Save(ctx, session, sm.cfg.SessionTTL); err != nil {
		return nil, fmt.Errorf("save session: %w", err)
	}
	return session, nil
}

// ResumeSession re-attaches a client to an existing session, restoring subscriptions.
func (sm *SessionManager) ResumeSession(ctx context.Context, c *Client, sessionID string) (*Session, error) {
	session, err := sm.store.Load(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("load session: %w", err)
	}
	if session == nil {
		return nil, fmt.Errorf("session %s expired or not found", sessionID)
	}

	// Update client with session credentials.
	c.UserID = session.UserID
	c.AccountID = session.AccountID
	c.Roles = session.Roles

	// Restore room subscriptions.
	for _, roomName := range session.Rooms {
		room := sm.hub.rooms.GetOrCreate(roomName)
		if c.JoinRoom(roomName) {
			room.Add(c)
		}
	}

	// Update session.
	session.ClientID = c.ID
	session.LastSeen = time.Now().UTC()
	if err := sm.store.Save(ctx, session, sm.cfg.SessionTTL); err != nil {
		sm.log.Warn("update session failed", zap.Error(err))
	}

	sm.log.Info("session resumed",
		zap.String("session", sessionID),
		zap.String("user", session.UserID),
		zap.Int("rooms", len(session.Rooms)))
	return session, nil
}

// UpdateSession persists the current room subscriptions for a session.
func (sm *SessionManager) UpdateSession(ctx context.Context, c *Client, sessionID string) error {
	session, err := sm.store.Load(ctx, sessionID)
	if err != nil || session == nil {
		return err
	}
	session.Rooms = c.Rooms()
	session.LastSeen = time.Now().UTC()
	return sm.store.Save(ctx, session, sm.cfg.SessionTTL)
}

// ExpireSession removes a session on clean client disconnect.
func (sm *SessionManager) ExpireSession(ctx context.Context, sessionID string) error {
	return sm.store.Delete(ctx, sessionID)
}

// ListUserSessions returns all active sessions for a user.
func (sm *SessionManager) ListUserSessions(ctx context.Context, userID string) ([]*Session, error) {
	return sm.store.ListByUser(ctx, userID)
}
