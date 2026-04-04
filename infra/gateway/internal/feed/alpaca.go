// Package feed contains market data feed implementations.
package feed

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

const (
	alpacaStockWSURL  = "wss://stream.data.alpaca.markets/v2/iex"
	alpacaCryptoWSURL = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
)

// AlpacaConfig holds credentials and symbol lists for the Alpaca feed.
type AlpacaConfig struct {
	APIKey        string
	Secret        string
	Paper         bool
	StockSymbols  []string
	CryptoSymbols []string
}

// alpacaAuthMsg is sent to authenticate on the WebSocket.
type alpacaAuthMsg struct {
	Action string `json:"action"`
	Key    string `json:"key"`
	Secret string `json:"secret"`
}

// alpacaSubMsg subscribes to data channels.
type alpacaSubMsg struct {
	Action string   `json:"action"`
	Bars   []string `json:"bars,omitempty"`
	Trades []string `json:"trades,omitempty"`
	Quotes []string `json:"quotes,omitempty"`
}

// alpacaMessage is the envelope used by Alpaca's streaming protocol.
type alpacaMessage struct {
	T  string          `json:"T"`
	S  string          `json:"S"`
	Msg string         `json:"msg,omitempty"`
	Raw json.RawMessage `json:"-"`
}

// alpacaBar maps to the Alpaca bar wire format.
type alpacaBar struct {
	T  string  `json:"T"`
	S  string  `json:"S"`
	O  float64 `json:"o"`
	H  float64 `json:"h"`
	L  float64 `json:"l"`
	C  float64 `json:"c"`
	V  float64 `json:"v"`
	Ts string  `json:"t"`
}

// alpacaTrade maps to the Alpaca trade wire format.
type alpacaTrade struct {
	T  string  `json:"T"`
	S  string  `json:"S"`
	P  float64 `json:"p"`
	Sz float64 `json:"s"`
	Ts string  `json:"t"`
	C  []string `json:"c"`
}

// alpacaQuote maps to the Alpaca quote wire format.
type alpacaQuote struct {
	T  string  `json:"T"`
	S  string  `json:"S"`
	Bp float64 `json:"bp"`
	Bs float64 `json:"bs"`
	Ap float64 `json:"ap"`
	As float64 `json:"as"`
	Ts string  `json:"t"`
}

// AlpacaFeed manages WebSocket connections to Alpaca market data.
type AlpacaFeed struct {
	cfg    AlpacaConfig
	log    *zap.Logger
	out    chan<- Event
	mu     sync.Mutex
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewAlpacaFeed creates a new AlpacaFeed that emits events onto out.
func NewAlpacaFeed(cfg AlpacaConfig, out chan<- Event, log *zap.Logger) *AlpacaFeed {
	return &AlpacaFeed{cfg: cfg, out: out, log: log}
}

// Start begins streaming in the background. Cancel ctx to stop.
func (f *AlpacaFeed) Start(ctx context.Context) {
	ctx, f.cancel = context.WithCancel(ctx)
	if len(f.cfg.StockSymbols) > 0 {
		f.wg.Add(1)
		go f.runStream(ctx, alpacaStockWSURL, f.cfg.StockSymbols, "alpaca-stocks")
	}
	if len(f.cfg.CryptoSymbols) > 0 {
		f.wg.Add(1)
		go f.runStream(ctx, alpacaCryptoWSURL, f.cfg.CryptoSymbols, "alpaca-crypto")
	}
}

// Stop gracefully shuts down the feed.
func (f *AlpacaFeed) Stop() {
	if f.cancel != nil {
		f.cancel()
	}
	f.wg.Wait()
}

// runStream is the core loop for a single WebSocket endpoint.
func (f *AlpacaFeed) runStream(ctx context.Context, wsURL string, symbols []string, source string) {
	defer f.wg.Done()
	backoff := time.Second
	maxBackoff := 2 * time.Minute

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		err := f.connect(ctx, wsURL, symbols, source)
		if err != nil {
			f.log.Warn("alpaca stream disconnected",
				zap.String("source", source),
				zap.Error(err),
				zap.Duration("backoff", backoff))
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		backoff = time.Duration(math.Min(float64(backoff*2), float64(maxBackoff)))
	}
}

// connect establishes a single WebSocket session.
func (f *AlpacaFeed) connect(ctx context.Context, wsURL string, symbols []string, source string) error {
	u, err := url.Parse(wsURL)
	if err != nil {
		return fmt.Errorf("parse url: %w", err)
	}

	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, _, err := dialer.DialContext(ctx, u.String(), nil)
	if err != nil {
		return fmt.Errorf("dial %s: %w", wsURL, err)
	}
	defer conn.Close()

	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	// Wait for "connected" message.
	if err := f.waitForConnected(conn); err != nil {
		return fmt.Errorf("waiting for connected: %w", err)
	}

	// Authenticate.
	authMsg := alpacaAuthMsg{Action: "auth", Key: f.cfg.APIKey, Secret: f.cfg.Secret}
	if err := conn.WriteJSON(authMsg); err != nil {
		return fmt.Errorf("sending auth: %w", err)
	}
	if err := f.waitForAuthorized(conn); err != nil {
		return fmt.Errorf("authorization: %w", err)
	}

	// Subscribe.
	subMsg := alpacaSubMsg{Action: "subscribe", Bars: symbols}
	if err := conn.WriteJSON(subMsg); err != nil {
		return fmt.Errorf("sending subscribe: %w", err)
	}

	f.log.Info("alpaca feed connected",
		zap.String("source", source),
		zap.Strings("symbols", symbols))

	// Start a ping goroutine.
	pingStop := make(chan struct{})
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-pingStop:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
				if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					return
				}
			}
		}
	}()
	defer close(pingStop)

	// Read loop.
	for {
		select {
		case <-ctx.Done():
			conn.WriteMessage(websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
			return nil
		default:
		}

		conn.SetReadDeadline(time.Now().Add(90 * time.Second))
		_, raw, err := conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("read: %w", err)
		}

		if err := f.handleMessages(raw, source); err != nil {
			f.log.Warn("alpaca handle message error", zap.Error(err))
		}
	}
}

// waitForConnected reads until it sees a "connected" status message.
func (f *AlpacaFeed) waitForConnected(conn *websocket.Conn) error {
	conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	_, raw, err := conn.ReadMessage()
	if err != nil {
		return err
	}
	var msgs []map[string]interface{}
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return fmt.Errorf("parse connected: %w", err)
	}
	for _, m := range msgs {
		if t, _ := m["T"].(string); t == "success" {
			if msg, _ := m["msg"].(string); msg == "connected" {
				return nil
			}
		}
	}
	return fmt.Errorf("unexpected initial message: %s", string(raw))
}

// waitForAuthorized reads until it sees an "authorized" success message.
func (f *AlpacaFeed) waitForAuthorized(conn *websocket.Conn) error {
	conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	_, raw, err := conn.ReadMessage()
	if err != nil {
		return err
	}
	var msgs []map[string]interface{}
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return fmt.Errorf("parse auth response: %w", err)
	}
	for _, m := range msgs {
		if t, _ := m["T"].(string); t == "success" {
			if msg, _ := m["msg"].(string); msg == "authenticated" {
				return nil
			}
		}
		if t, _ := m["T"].(string); t == "error" {
			code, _ := m["code"].(float64)
			msg, _ := m["msg"].(string)
			return fmt.Errorf("auth error %d: %s", int(code), msg)
		}
	}
	return fmt.Errorf("unexpected auth response: %s", string(raw))
}

// handleMessages parses a raw WebSocket message and emits events.
func (f *AlpacaFeed) handleMessages(raw []byte, source string) error {
	// Alpaca always sends arrays.
	var msgs []json.RawMessage
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return fmt.Errorf("unmarshal array: %w", err)
	}
	for _, m := range msgs {
		var hdr alpacaMessage
		if err := json.Unmarshal(m, &hdr); err != nil {
			continue
		}
		switch hdr.T {
		case "b": // bar
			var b alpacaBar
			if err := json.Unmarshal(m, &b); err != nil {
				f.log.Warn("parse alpaca bar", zap.Error(err))
				continue
			}
			ts, _ := time.Parse(time.RFC3339Nano, b.Ts)
			f.emit(Event{
				Kind: EventBar,
				Bar: &Bar{
					Symbol:    b.S,
					Timestamp: ts,
					Open:      b.O,
					High:      b.H,
					Low:       b.L,
					Close:     b.C,
					Volume:    b.V,
					Source:    source,
				},
			})
		case "t": // trade
			var t alpacaTrade
			if err := json.Unmarshal(m, &t); err != nil {
				f.log.Warn("parse alpaca trade", zap.Error(err))
				continue
			}
			ts, _ := time.Parse(time.RFC3339Nano, t.Ts)
			f.emit(Event{
				Kind: EventTrade,
				Trade: &Trade{
					Symbol:    t.S,
					Timestamp: ts,
					Price:     t.P,
					Size:      t.Sz,
					Source:    source,
				},
			})
		case "q": // quote
			var q alpacaQuote
			if err := json.Unmarshal(m, &q); err != nil {
				f.log.Warn("parse alpaca quote", zap.Error(err))
				continue
			}
			ts, _ := time.Parse(time.RFC3339Nano, q.Ts)
			f.emit(Event{
				Kind: EventQuote,
				Quote: &Quote{
					Symbol:    q.S,
					Timestamp: ts,
					BidPrice:  q.Bp,
					BidSize:   q.Bs,
					AskPrice:  q.Ap,
					AskSize:   q.As,
					Source:    source,
				},
			})
		}
	}
	return nil
}

// emit sends an event on the output channel in a non-blocking fashion.
func (f *AlpacaFeed) emit(e Event) {
	select {
	case f.out <- e:
	default:
		f.log.Warn("alpaca event channel full, dropping event")
	}
}
