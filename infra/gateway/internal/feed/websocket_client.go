package feed

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// WSClientConfig configures a generic WebSocket client.
type WSClientConfig struct {
	URL              string
	Headers          map[string]string
	HandshakeTimeout time.Duration
	ReadTimeout      time.Duration
	PingInterval     time.Duration
	MaxMessageBytes  int64
	TLSSkipVerify    bool
	ProxyURL         string
}

// DefaultWSClientConfig returns sensible defaults for a WebSocket client.
func DefaultWSClientConfig(wsURL string) WSClientConfig {
	return WSClientConfig{
		URL:              wsURL,
		HandshakeTimeout: 10 * time.Second,
		ReadTimeout:      90 * time.Second,
		PingInterval:     30 * time.Second,
		MaxMessageBytes:  1 << 20, // 1 MB
	}
}

// WSMessage is a received WebSocket message.
type WSMessage struct {
	Kind int    // websocket.TextMessage etc.
	Data []byte
	Err  error
}

// WSClient is a generic, reconnecting WebSocket client.
type WSClient struct {
	cfg       WSClientConfig
	log       *zap.Logger
	dialer    *websocket.Dialer
	mu        sync.Mutex
	conn      *websocket.Conn
	connected atomic.Bool
	stats     wsClientStats
}

type wsClientStats struct {
	messagesReceived atomic.Int64
	messagesSent     atomic.Int64
	bytesReceived    atomic.Int64
	bytesSent        atomic.Int64
	errors           atomic.Int64
	reconnects       atomic.Int64
	connectedAt      atomic.Value // time.Time
}

// NewWSClient creates a WSClient.
func NewWSClient(cfg WSClientConfig, log *zap.Logger) *WSClient {
	tlsCfg := &tls.Config{}
	if cfg.TLSSkipVerify {
		tlsCfg.InsecureSkipVerify = true
	}

	dialer := &websocket.Dialer{
		HandshakeTimeout: cfg.HandshakeTimeout,
		TLSClientConfig:  tlsCfg,
	}

	if cfg.ProxyURL != "" {
		proxyURL, err := url.Parse(cfg.ProxyURL)
		if err == nil {
			dialer.Proxy = http.ProxyURL(proxyURL)
		}
	}

	return &WSClient{
		cfg:    cfg,
		log:    log,
		dialer: dialer,
	}
}

// Connect establishes a WebSocket connection.
func (c *WSClient) Connect(ctx context.Context) error {
	headers := http.Header{}
	for k, v := range c.cfg.Headers {
		headers.Set(k, v)
	}

	conn, resp, err := c.dialer.DialContext(ctx, c.cfg.URL, headers)
	if err != nil {
		if resp != nil {
			return fmt.Errorf("dial %s (HTTP %d): %w", c.cfg.URL, resp.StatusCode, err)
		}
		return fmt.Errorf("dial %s: %w", c.cfg.URL, err)
	}

	if c.cfg.MaxMessageBytes > 0 {
		conn.SetReadLimit(c.cfg.MaxMessageBytes)
	}
	conn.SetReadDeadline(time.Now().Add(c.cfg.ReadTimeout))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(c.cfg.ReadTimeout))
		return nil
	})

	c.mu.Lock()
	if c.conn != nil {
		c.conn.Close()
	}
	c.conn = conn
	c.connected.Store(true)
	c.mu.Unlock()

	c.stats.connectedAt.Store(time.Now())
	return nil
}

// Close closes the WebSocket connection.
func (c *WSClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		c.conn.WriteMessage(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		c.conn.Close()
		c.conn = nil
	}
	c.connected.Store(false)
}

// SendJSON marshals v to JSON and sends it as a text message.
func (c *WSClient) SendJSON(v interface{}) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	return c.Send(data)
}

// Send sends raw bytes as a text message.
func (c *WSClient) Send(data []byte) error {
	c.mu.Lock()
	conn := c.conn
	c.mu.Unlock()
	if conn == nil {
		return fmt.Errorf("not connected")
	}
	conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	err := conn.WriteMessage(websocket.TextMessage, data)
	if err == nil {
		c.stats.messagesSent.Add(1)
		c.stats.bytesSent.Add(int64(len(data)))
	} else {
		c.stats.errors.Add(1)
	}
	return err
}

// ReadMessage reads the next message from the WebSocket.
func (c *WSClient) ReadMessage() (int, []byte, error) {
	c.mu.Lock()
	conn := c.conn
	c.mu.Unlock()
	if conn == nil {
		return 0, nil, fmt.Errorf("not connected")
	}
	kind, data, err := conn.ReadMessage()
	if err != nil {
		c.stats.errors.Add(1)
		c.connected.Store(false)
		return 0, nil, err
	}
	c.stats.messagesReceived.Add(1)
	c.stats.bytesReceived.Add(int64(len(data)))
	return kind, data, nil
}

// StartPing launches a background goroutine that sends periodic pings.
// Cancel the context or close the connection to stop it.
func (c *WSClient) StartPing(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(c.cfg.PingInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				c.mu.Lock()
				conn := c.conn
				c.mu.Unlock()
				if conn == nil {
					return
				}
				conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
				if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					return
				}
			}
		}
	}()
}

// IsConnected returns true if the client currently has an open connection.
func (c *WSClient) IsConnected() bool {
	return c.connected.Load()
}

// Stats returns a snapshot of client statistics.
func (c *WSClient) Stats() map[string]int64 {
	return map[string]int64{
		"messages_received": c.stats.messagesReceived.Load(),
		"messages_sent":     c.stats.messagesSent.Load(),
		"bytes_received":    c.stats.bytesReceived.Load(),
		"bytes_sent":        c.stats.bytesSent.Load(),
		"errors":            c.stats.errors.Load(),
		"reconnects":        c.stats.reconnects.Load(),
	}
}

// RunLoop runs a typical connect-read-reconnect loop.
// onConnect is called after each successful connection (for auth/subscribe).
// onMessage is called for each received message.
// The loop runs until ctx is cancelled.
func (c *WSClient) RunLoop(
	ctx context.Context,
	onConnect func(client *WSClient) error,
	onMessage func(data []byte) error,
	reconnectCfg ReconnectConfig,
) {
	reconnector := NewReconnector(reconnectCfg, c.cfg.URL, c.log)
	reconnector.Run(ctx, func(ctx context.Context) error {
		if err := c.Connect(ctx); err != nil {
			c.stats.reconnects.Add(1)
			return err
		}
		c.StartPing(ctx)
		if onConnect != nil {
			if err := onConnect(c); err != nil {
				c.Close()
				return fmt.Errorf("onConnect: %w", err)
			}
		}
		c.log.Info("ws client connected", zap.String("url", c.cfg.URL))

		for {
			select {
			case <-ctx.Done():
				c.Close()
				return nil
			default:
			}
			_, data, err := c.ReadMessage()
			if err != nil {
				c.Close()
				return fmt.Errorf("read: %w", err)
			}
			if onMessage != nil {
				if err := onMessage(data); err != nil {
					c.log.Warn("onMessage error", zap.Error(err))
				}
			}
		}
	}, func() {
		c.stats.reconnects.Add(1)
	})
}
