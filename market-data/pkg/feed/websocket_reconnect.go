// Package feed provides enhanced WebSocket feed infrastructure with automatic
// reconnection, health monitoring, and per-feed metrics.
package feed

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// ReconnectConfig controls the exponential-backoff reconnect policy.
type ReconnectConfig struct {
	// InitialDelay is the wait before the first reconnect attempt.
	InitialDelay time.Duration
	// MaxDelay caps the exponential growth of backoff.
	MaxDelay time.Duration
	// BackoffMultiplier is the factor applied after each failure (e.g. 2.0).
	BackoffMultiplier float64
	// MaxRetries is the maximum number of attempts; 0 = unlimited.
	MaxRetries int
	// JitterPct adds randomness: actual delay = base * (1 +/- JitterPct).
	// Must be in [0, 1). 0.20 means +/- 20%.
	JitterPct float64
}

// DefaultReconnectConfig returns a sensible default reconnect policy.
func DefaultReconnectConfig() ReconnectConfig {
	return ReconnectConfig{
		InitialDelay:      1 * time.Second,
		MaxDelay:          60 * time.Second,
		BackoffMultiplier: 2.0,
		MaxRetries:        0,
		JitterPct:         0.20,
	}
}

// applyJitter returns delay +/- jitterPct of delay.
func applyJitter(delay time.Duration, jitterPct float64) time.Duration {
	if jitterPct <= 0 {
		return delay
	}
	jitter := float64(delay) * jitterPct * (2*rand.Float64() - 1)
	result := float64(delay) + jitter
	if result < 0 {
		result = 0
	}
	return time.Duration(result)
}

// nextDelay computes the next backoff duration given the current one.
func nextDelay(current time.Duration, cfg ReconnectConfig) time.Duration {
	next := time.Duration(float64(current) * cfg.BackoffMultiplier)
	if next > cfg.MaxDelay {
		next = cfg.MaxDelay
	}
	return applyJitter(next, cfg.JitterPct)
}

// reconnectMetrics holds atomic counters for connection health.
type reconnectMetrics struct {
	reconnectCount      atomic.Int64
	totalUptime         atomic.Int64 // nanoseconds of connected time
	totalLifetime       atomic.Int64 // total nanoseconds since first connect
	lastDisconnectNanos atomic.Int64 // unix nano of last disconnect
}

// lastDisconnectReason is set on disconnect; read under the ws mutex.
type ManagedWebSocket struct {
	url    string
	cfg    ReconnectConfig
	dialer *websocket.Dialer

	// OnReconnect is called after each successful reconnection.
	// It should re-subscribe to all channels. It receives the new connection.
	OnReconnect func(conn *websocket.Conn) error

	mu              sync.Mutex
	conn            *websocket.Conn
	connectedAt     time.Time
	lastDisconnect  time.Time
	disconnectReason string

	msgCh  chan []byte
	errCh  chan error

	metrics reconnectMetrics

	startedAt time.Time
}

// NewManagedWebSocket creates a ManagedWebSocket.
// dialer may be nil; websocket.DefaultDialer will be used in that case.
func NewManagedWebSocket(url string, cfg ReconnectConfig, dialer *websocket.Dialer) *ManagedWebSocket {
	if dialer == nil {
		dialer = websocket.DefaultDialer
	}
	return &ManagedWebSocket{
		url:    url,
		cfg:    cfg,
		dialer: dialer,
		msgCh:  make(chan []byte, 256),
		errCh:  make(chan error, 1),
	}
}

// Connect starts the reconnect loop. It blocks in the background;
// call ReadMessage to receive messages. Cancel ctx to shut down.
func (m *ManagedWebSocket) Connect(ctx context.Context) error {
	m.startedAt = time.Now()
	m.metrics.totalLifetime.Store(0)

	// Attempt first connection synchronously so the caller knows immediately
	// if the URL is unreachable at startup.
	if err := m.dial(ctx); err != nil {
		return err
	}

	go m.reconnectLoop(ctx)
	return nil
}

// dial opens a single WebSocket connection and launches its read goroutine.
func (m *ManagedWebSocket) dial(ctx context.Context) error {
	conn, _, err := m.dialer.DialContext(ctx, m.url, nil)
	if err != nil {
		return fmt.Errorf("dial %s: %w", m.url, err)
	}

	conn.SetPingHandler(func(data string) error {
		return conn.WriteMessage(websocket.PongMessage, []byte(data))
	})

	m.mu.Lock()
	m.conn = conn
	m.connectedAt = time.Now()
	m.mu.Unlock()

	if m.OnReconnect != nil {
		if err := m.OnReconnect(conn); err != nil {
			conn.Close()
			return fmt.Errorf("OnReconnect: %w", err)
		}
	}

	go m.readPump(ctx, conn)
	return nil
}

func (m *ManagedWebSocket) readPump(ctx context.Context, conn *websocket.Conn) {
	defer func() {
		connectedDuration := time.Since(m.connectedAt)
		m.metrics.totalUptime.Add(int64(connectedDuration))

		m.mu.Lock()
		m.lastDisconnect = time.Now()
		m.metrics.lastDisconnectNanos.Store(m.lastDisconnect.UnixNano())
		m.conn = nil
		m.mu.Unlock()

		conn.Close()
	}()

	for {
		conn.SetReadDeadline(time.Now().Add(90 * time.Second))
		_, data, err := conn.ReadMessage()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
			}

			m.mu.Lock()
			m.disconnectReason = err.Error()
			m.mu.Unlock()

			// Signal the reconnect loop
			select {
			case m.errCh <- err:
			default:
			}
			return
		}

		select {
		case m.msgCh <- data:
		case <-ctx.Done():
			return
		}
	}
}

func (m *ManagedWebSocket) reconnectLoop(ctx context.Context) {
	delay := m.cfg.InitialDelay
	attempt := 0

	for {
		select {
		case <-ctx.Done():
			m.mu.Lock()
			if m.conn != nil {
				m.conn.Close()
			}
			m.mu.Unlock()
			return
		case <-m.errCh:
			// Connection died; attempt reconnect with backoff
		}

		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			attempt++
			if m.cfg.MaxRetries > 0 && attempt > m.cfg.MaxRetries {
				log.Printf("[ws] %s: exceeded max retries (%d), giving up", m.url, m.cfg.MaxRetries)
				select {
				case m.errCh <- fmt.Errorf("max retries exceeded"):
				default:
				}
				return
			}

			jittered := applyJitter(delay, m.cfg.JitterPct)
			log.Printf("[ws] %s: reconnect attempt %d in %v", m.url, attempt, jittered)

			select {
			case <-ctx.Done():
				return
			case <-time.After(jittered):
			}

			if err := m.dial(ctx); err != nil {
				log.Printf("[ws] %s: reconnect error: %v", m.url, err)
				delay = nextDelay(delay, m.cfg)
				continue
			}

			// Reconnect succeeded
			m.metrics.reconnectCount.Add(1)
			log.Printf("[ws] %s: reconnected (attempt %d)", m.url, attempt)
			attempt = 0
			delay = m.cfg.InitialDelay
			break
		}
	}
}

// ReadMessage blocks until a message is available, or ctx is cancelled.
// Returns (nil, ctx.Err()) on cancellation.
func (m *ManagedWebSocket) ReadMessage() ([]byte, error) {
	select {
	case msg := <-m.msgCh:
		return msg, nil
	case err := <-m.errCh:
		return nil, err
	}
}

// ReadMessageContext blocks until a message is available or ctx is cancelled.
func (m *ManagedWebSocket) ReadMessageContext(ctx context.Context) ([]byte, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case msg := <-m.msgCh:
		return msg, nil
	case err := <-m.errCh:
		return nil, err
	}
}

// WriteMessage sends a message on the current connection.
// Returns an error if not currently connected.
func (m *ManagedWebSocket) WriteMessage(msgType int, data []byte) error {
	m.mu.Lock()
	conn := m.conn
	m.mu.Unlock()
	if conn == nil {
		return fmt.Errorf("not connected")
	}
	return conn.WriteMessage(msgType, data)
}

// ReconnectCount returns the number of reconnections since Connect was called.
func (m *ManagedWebSocket) ReconnectCount() int64 {
	return m.metrics.reconnectCount.Load()
}

// LastDisconnectReason returns the error string from the most recent disconnect.
func (m *ManagedWebSocket) LastDisconnectReason() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.disconnectReason
}

// UptimePct returns the fraction of total wall-clock time the connection has
// been in the CONNECTED state. Returns 0 before the first connection.
func (m *ManagedWebSocket) UptimePct() float64 {
	lifetime := time.Since(m.startedAt)
	if lifetime <= 0 {
		return 0
	}
	uptime := time.Duration(m.metrics.totalUptime.Load())

	// Add current session if connected
	m.mu.Lock()
	conn := m.conn
	connAt := m.connectedAt
	m.mu.Unlock()

	if conn != nil {
		uptime += time.Since(connAt)
	}

	pct := float64(uptime) / float64(lifetime)
	return math.Min(pct, 1.0)
}

// IsConnected returns true if the WebSocket is currently open.
func (m *ManagedWebSocket) IsConnected() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.conn != nil
}

// -- FeedHealthMonitor --

// FeedHealthMonitor watches message rates per feed and flags stale feeds.
// A feed is stale if its message rate drops below a configurable threshold.
type FeedHealthMonitor struct {
	mu       sync.RWMutex
	feeds    map[string]*feedRecord
	stopCh   chan struct{}
	doneCh   chan struct{}
}

type feedRecord struct {
	mu               sync.Mutex
	name             string
	expectedInterval time.Duration // expected time between messages
	lastMessageTime  time.Time
	messageCount     int64
	stale            bool
}

// NewFeedHealthMonitor creates a FeedHealthMonitor.
// checkInterval controls how often the monitor checks feed staleness.
func NewFeedHealthMonitor(checkInterval time.Duration) *FeedHealthMonitor {
	m := &FeedHealthMonitor{
		feeds:  make(map[string]*feedRecord),
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}
	go m.run(checkInterval)
	return m
}

// Register adds a feed to monitoring.
// expectedInterval is the expected maximum silence period before the feed
// is flagged as stale (e.g. 2 * 15min for 15m bars = 30m).
func (m *FeedHealthMonitor) Register(name string, expectedInterval time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.feeds[name] = &feedRecord{
		name:             name,
		expectedInterval: expectedInterval,
		lastMessageTime:  time.Now(),
	}
}

// RecordMessage records that a message was received from the named feed.
func (m *FeedHealthMonitor) RecordMessage(name string) {
	m.mu.RLock()
	rec, ok := m.feeds[name]
	m.mu.RUnlock()
	if !ok {
		return
	}
	rec.mu.Lock()
	rec.lastMessageTime = time.Now()
	rec.messageCount++
	rec.stale = false
	rec.mu.Unlock()
}

// IsHealthy returns true if the feed has received a message within
// 2x its expected interval.
func (m *FeedHealthMonitor) IsHealthy(name string) bool {
	m.mu.RLock()
	rec, ok := m.feeds[name]
	m.mu.RUnlock()
	if !ok {
		return false
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	return !rec.stale
}

// MessageCount returns the total messages recorded for a feed.
func (m *FeedHealthMonitor) MessageCount(name string) int64 {
	m.mu.RLock()
	rec, ok := m.feeds[name]
	m.mu.RUnlock()
	if !ok {
		return 0
	}
	rec.mu.Lock()
	defer rec.mu.Unlock()
	return rec.messageCount
}

// AllHealthy returns true if every registered feed is healthy.
func (m *FeedHealthMonitor) AllHealthy() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, rec := range m.feeds {
		rec.mu.Lock()
		stale := rec.stale
		rec.mu.Unlock()
		if stale {
			return false
		}
	}
	return true
}

func (m *FeedHealthMonitor) run(checkInterval time.Duration) {
	defer close(m.doneCh)
	ticker := time.NewTicker(checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopCh:
			return
		case <-ticker.C:
			m.checkAll()
		}
	}
}

func (m *FeedHealthMonitor) checkAll() {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, rec := range m.feeds {
		rec.mu.Lock()
		threshold := 2 * rec.expectedInterval
		if time.Since(rec.lastMessageTime) > threshold {
			if !rec.stale {
				log.Printf("[health] feed %q stale: no message for %v (threshold %v)",
					rec.name, time.Since(rec.lastMessageTime).Round(time.Second), threshold)
			}
			rec.stale = true
		}
		rec.mu.Unlock()
	}
}

// Stop shuts down the background health checker.
func (m *FeedHealthMonitor) Stop() {
	close(m.stopCh)
	<-m.doneCh
}
