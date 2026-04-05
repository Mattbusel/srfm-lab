package feeds

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"srfm/market-data/aggregator"
	"srfm/market-data/monitoring"

	"github.com/gorilla/websocket"
)

const (
	alpacaWSURL    = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
	alpacaMaxRetry = 60 * time.Second
	alpacaBaseRetry = time.Second
)

// AlpacaFeed connects to Alpaca crypto WebSocket feed.
type AlpacaFeed struct {
	symbols   []string
	outCh     chan<- aggregator.RawTick
	metrics   *monitoring.Metrics
	apiKey    string
	apiSecret string

	mu            sync.RWMutex
	connected     bool
	lastMsgTime   time.Time
	connectedAt   time.Time
	msgsReceived  atomic.Int64
	errCount      atomic.Int64
	latencyMs     atomic.Int64

	stopCh chan struct{}
	doneCh chan struct{}
}

// alpaca wire message types
type alpacaMsg struct {
	T   string          `json:"T"`
	Raw json.RawMessage `json:"-"`
}

type alpacaAuthMsg struct {
	Action string `json:"action"`
	Key    string `json:"key"`
	Secret string `json:"secret"`
}

type alpacaSubscribeMsg struct {
	Action string   `json:"action"`
	Bars   []string `json:"bars"`
	Trades []string `json:"trades"`
}

type alpacaBarMsg struct {
	T  string  `json:"T"`
	S  string  `json:"S"`
	O  float64 `json:"o"`
	H  float64 `json:"h"`
	L  float64 `json:"l"`
	C  float64 `json:"c"`
	V  float64 `json:"v"`
	Ts string  `json:"t"`
}

type alpacaTradeMsg struct {
	T  string  `json:"T"`
	S  string  `json:"S"`
	P  float64 `json:"p"`
	S2 float64 `json:"s"`
	Ts string  `json:"t"`
}

// NewAlpacaFeed creates an Alpaca feed. apiKey/apiSecret are read from environment
// at connect time; pass empty strings here to use env vars.
func NewAlpacaFeed(symbols []string, outCh chan<- aggregator.RawTick, metrics *monitoring.Metrics) *AlpacaFeed {
	return &AlpacaFeed{
		symbols: symbols,
		outCh:   outCh,
		metrics: metrics,
		stopCh:  make(chan struct{}),
		doneCh:  make(chan struct{}),
	}
}

// SetCredentials sets API credentials (called by feed manager from env vars).
func (f *AlpacaFeed) SetCredentials(key, secret string) {
	f.mu.Lock()
	f.apiKey = key
	f.apiSecret = secret
	f.mu.Unlock()
}

// Start begins the feed. Blocks until Stop() is called.
func (f *AlpacaFeed) Start() {
	defer close(f.doneCh)
	backoff := alpacaBaseRetry
	for {
		select {
		case <-f.stopCh:
			return
		default:
		}

		err := f.connect()
		if err != nil {
			f.errCount.Add(1)
			f.metrics.FeedError("alpaca")
			log.Printf("[alpaca] connection error: %v, retrying in %v", err, backoff)
		}

		f.mu.Lock()
		f.connected = false
		f.mu.Unlock()

		select {
		case <-f.stopCh:
			return
		case <-time.After(backoff):
			backoff = time.Duration(math.Min(float64(backoff*2), float64(alpacaMaxRetry)))
		}
	}
}

// Stop signals the feed to stop.
func (f *AlpacaFeed) Stop() {
	close(f.stopCh)
	<-f.doneCh
}

// Health returns current feed health snapshot.
func (f *AlpacaFeed) Health() FeedHealth {
	f.mu.RLock()
	defer f.mu.RUnlock()
	var uptime time.Duration
	if f.connected {
		uptime = time.Since(f.connectedAt)
	}
	return FeedHealth{
		Name:          "alpaca",
		IsConnected:   f.connected,
		LastBarTime:   f.lastMsgTime,
		BarsReceived:  f.msgsReceived.Load(),
		ErrorsCount:   f.errCount.Load(),
		LatencyMs:     f.latencyMs.Load(),
		Uptime:        uptime,
	}
}

func (f *AlpacaFeed) connect() error {
	dialer := websocket.DefaultDialer
	conn, _, err := dialer.Dial(alpacaWSURL, nil)
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	conn.SetReadDeadline(time.Now().Add(30 * time.Second))

	// Read initial connected message
	if err := f.readExpected(conn, "connected"); err != nil {
		return fmt.Errorf("expected connected: %w", err)
	}

	// Authenticate
	f.mu.RLock()
	key, secret := f.apiKey, f.apiSecret
	f.mu.RUnlock()

	authMsg := alpacaAuthMsg{Action: "auth", Key: key, Secret: secret}
	if err := conn.WriteJSON(authMsg); err != nil {
		return fmt.Errorf("auth write: %w", err)
	}

	if err := f.readExpected(conn, "authenticated"); err != nil {
		return fmt.Errorf("auth failed: %w", err)
	}

	// Subscribe to bars for all symbols
	barSyms := make([]string, len(f.symbols))
	for i, s := range f.symbols {
		barSyms[i] = s + "/USD"
	}
	subMsg := alpacaSubscribeMsg{
		Action: "subscribe",
		Bars:   barSyms,
		Trades: barSyms,
	}
	if err := conn.WriteJSON(subMsg); err != nil {
		return fmt.Errorf("subscribe write: %w", err)
	}
	if err := f.readExpected(conn, "subscription"); err != nil {
		return fmt.Errorf("subscription failed: %w", err)
	}

	f.mu.Lock()
	f.connected = true
	f.connectedAt = time.Now()
	f.mu.Unlock()
	log.Printf("[alpaca] connected and subscribed to %d symbols", len(f.symbols))

	// Reset backoff on successful connect
	conn.SetReadDeadline(time.Time{}) // clear deadline for streaming

	return f.readLoop(conn)
}

func (f *AlpacaFeed) readExpected(conn *websocket.Conn, msgType string) error {
	_, data, err := conn.ReadMessage()
	if err != nil {
		return err
	}
	var msgs []json.RawMessage
	if err := json.Unmarshal(data, &msgs); err != nil {
		return fmt.Errorf("parse: %w", err)
	}
	for _, m := range msgs {
		var base struct {
			T string `json:"T"`
		}
		json.Unmarshal(m, &base)
		if base.T == msgType {
			return nil
		}
		if base.T == "error" {
			var errMsg struct {
				Code int    `json:"code"`
				Msg  string `json:"msg"`
			}
			json.Unmarshal(m, &errMsg)
			return fmt.Errorf("server error %d: %s", errMsg.Code, errMsg.Msg)
		}
	}
	return fmt.Errorf("expected %q, got: %s", msgType, string(data))
}

func (f *AlpacaFeed) readLoop(conn *websocket.Conn) error {
	conn.SetPingHandler(func(data string) error {
		return conn.WriteMessage(websocket.PongMessage, []byte(data))
	})

	stopCh := f.stopCh
	for {
		select {
		case <-stopCh:
			return nil
		default:
		}

		conn.SetReadDeadline(time.Now().Add(90 * time.Second))
		_, data, err := conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("read: %w", err)
		}

		recvTime := time.Now()

		var msgs []json.RawMessage
		if err := json.Unmarshal(data, &msgs); err != nil {
			log.Printf("[alpaca] parse error: %v", err)
			continue
		}

		for _, raw := range msgs {
			var base struct {
				T string `json:"T"`
			}
			json.Unmarshal(raw, &base)

			switch base.T {
			case "b": // bar
				var bar alpacaBarMsg
				if err := json.Unmarshal(raw, &bar); err != nil {
					continue
				}
				ts, err := time.Parse(time.RFC3339Nano, bar.Ts)
				if err != nil {
					continue
				}
				sym := stripUSDSuffix(bar.S)
				tick := aggregator.RawTick{
					Symbol:    sym,
					Open:      bar.O,
					High:      bar.H,
					Low:       bar.L,
					Close:     bar.C,
					Volume:    bar.V,
					Timestamp: ts.UTC(),
					Source:    "alpaca",
					IsBar:     true,
				}
				f.emit(tick, recvTime, ts)

			case "t": // trade
				var trade alpacaTradeMsg
				if err := json.Unmarshal(raw, &trade); err != nil {
					continue
				}
				ts, err := time.Parse(time.RFC3339Nano, trade.Ts)
				if err != nil {
					continue
				}
				sym := stripUSDSuffix(trade.S)
				tick := aggregator.RawTick{
					Symbol:    sym,
					Close:     trade.P,
					Volume:    trade.S2,
					Timestamp: ts.UTC(),
					Source:    "alpaca",
					IsBar:     false,
				}
				f.emit(tick, recvTime, ts)

			case "error":
				f.errCount.Add(1)
				log.Printf("[alpaca] server error: %s", string(raw))
			}
		}
	}
}

func (f *AlpacaFeed) emit(tick aggregator.RawTick, recvTime time.Time, msgTime time.Time) {
	latency := recvTime.Sub(msgTime).Milliseconds()
	if latency < 0 {
		latency = 0
	}
	f.latencyMs.Store(latency)
	f.msgsReceived.Add(1)
	f.metrics.BarReceived("alpaca")

	f.mu.Lock()
	f.lastMsgTime = recvTime
	f.mu.Unlock()

	select {
	case f.outCh <- tick:
	default:
		log.Printf("[alpaca] output channel full, dropping tick for %s", tick.Symbol)
	}
}

func stripUSDSuffix(s string) string {
	if len(s) > 4 && s[len(s)-4:] == "/USD" {
		return s[:len(s)-4]
	}
	return s
}
