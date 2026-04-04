package feed

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

const (
	binanceWSBaseURL = "wss://stream.binance.com:9443/ws"
)

// BinanceConfig holds settings for the Binance feed.
type BinanceConfig struct {
	APIKey  string
	Secret  string
	Symbols []string // e.g. ["BTCUSDT", "ETHUSDT"]
}

// binanceKlineEvent is the wire format for kline/candlestick stream events.
type binanceKlineEvent struct {
	E  string        `json:"e"` // event type "kline"
	Es string        `json:"E"` // event time (ms epoch)
	S  string        `json:"s"` // symbol
	K  binanceKline  `json:"k"`
}

type binanceKline struct {
	T  int64  `json:"t"` // open time ms
	TC int64  `json:"T"` // close time ms
	S  string `json:"s"` // symbol
	I  string `json:"i"` // interval
	O  string `json:"o"`
	H  string `json:"h"`
	L  string `json:"l"`
	C  string `json:"c"`
	V  string `json:"v"`
	X  bool   `json:"x"` // is bar closed?
	N  int    `json:"n"` // number of trades
}

// binanceSubMsg is used to subscribe/unsubscribe from streams.
type binanceSubMsg struct {
	Method string   `json:"method"`
	Params []string `json:"params"`
	ID     int      `json:"id"`
}

// BinanceFeed streams kline data from Binance.
type BinanceFeed struct {
	cfg    BinanceConfig
	log    *zap.Logger
	out    chan<- Event
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// partialBars holds the most recent partial (in-progress) bar per symbol.
	mu          sync.Mutex
	partialBars map[string]*Bar
}

// NewBinanceFeed creates a new BinanceFeed.
func NewBinanceFeed(cfg BinanceConfig, out chan<- Event, log *zap.Logger) *BinanceFeed {
	return &BinanceFeed{
		cfg:         cfg,
		out:         out,
		log:         log,
		partialBars: make(map[string]*Bar),
	}
}

// Start begins the stream in the background.
func (f *BinanceFeed) Start(ctx context.Context) {
	ctx, f.cancel = context.WithCancel(ctx)
	f.wg.Add(1)
	go f.run(ctx)
}

// Stop gracefully stops the feed.
func (f *BinanceFeed) Stop() {
	if f.cancel != nil {
		f.cancel()
	}
	f.wg.Wait()
}

// GetPartialBar returns the current partial bar for the given symbol, or nil.
func (f *BinanceFeed) GetPartialBar(symbol string) *Bar {
	f.mu.Lock()
	defer f.mu.Unlock()
	if b, ok := f.partialBars[symbol]; ok {
		cp := *b
		return &cp
	}
	return nil
}

func (f *BinanceFeed) run(ctx context.Context) {
	defer f.wg.Done()
	backoff := time.Second
	maxBackoff := 2 * time.Minute

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		err := f.connect(ctx)
		if err != nil {
			f.log.Warn("binance stream error",
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

// streamNames builds Binance combined stream names for all configured symbols.
func (f *BinanceFeed) streamNames() []string {
	streams := make([]string, 0, len(f.cfg.Symbols))
	for _, sym := range f.cfg.Symbols {
		lower := strings.ToLower(sym)
		streams = append(streams, fmt.Sprintf("%s@kline_1m", lower))
	}
	return streams
}

func (f *BinanceFeed) connect(ctx context.Context) error {
	streams := f.streamNames()
	if len(streams) == 0 {
		return nil
	}

	// Use the combined stream endpoint when subscribing to multiple streams.
	combinedPath := strings.Join(streams, "/")
	rawURL := fmt.Sprintf("%s/%s", binanceWSBaseURL, combinedPath)
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("parse binance url: %w", err)
	}

	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, _, err := dialer.DialContext(ctx, u.String(), nil)
	if err != nil {
		return fmt.Errorf("dial binance: %w", err)
	}
	defer conn.Close()

	conn.SetReadDeadline(time.Now().Add(90 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(90 * time.Second))
		return nil
	})

	f.log.Info("binance feed connected", zap.Strings("streams", streams))

	// Ping goroutine — Binance requires pong within 10 min, we ping every 3 min.
	pingStop := make(chan struct{})
	go func() {
		ticker := time.NewTicker(3 * time.Minute)
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

		if err := f.handleMessage(raw); err != nil {
			f.log.Warn("binance handle message", zap.Error(err))
		}
	}
}

func (f *BinanceFeed) handleMessage(raw []byte) error {
	// Combined stream wraps in {"stream":"...","data":{...}}.
	var wrapper struct {
		Stream string          `json:"stream"`
		Data   json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(raw, &wrapper); err != nil {
		return fmt.Errorf("unmarshal wrapper: %w", err)
	}

	payload := wrapper.Data
	if payload == nil {
		// Single stream — data is the raw message.
		payload = raw
	}

	var evt binanceKlineEvent
	if err := json.Unmarshal(payload, &evt); err != nil {
		return fmt.Errorf("unmarshal kline: %w", err)
	}
	if evt.E != "kline" {
		return nil
	}

	bar, err := klineToBar(evt)
	if err != nil {
		return err
	}

	if bar.IsPartial {
		// Update the partial bar cache but don't fan-out.
		f.mu.Lock()
		f.partialBars[bar.Symbol] = bar
		f.mu.Unlock()

		// Still emit partial bars so consumers can show live prices.
		f.emit(Event{Kind: EventBar, Bar: bar})
	} else {
		// Completed bar: remove from partial cache and emit.
		f.mu.Lock()
		delete(f.partialBars, bar.Symbol)
		f.mu.Unlock()
		f.emit(Event{Kind: EventBar, Bar: bar})
	}
	return nil
}

// klineToBar converts a Binance kline event to a Bar.
func klineToBar(evt binanceKlineEvent) (*Bar, error) {
	parseFloat := func(s string) (float64, error) {
		var v float64
		_, err := fmt.Sscanf(s, "%f", &v)
		return v, err
	}

	o, err := parseFloat(evt.K.O)
	if err != nil {
		return nil, fmt.Errorf("parse open: %w", err)
	}
	h, err := parseFloat(evt.K.H)
	if err != nil {
		return nil, fmt.Errorf("parse high: %w", err)
	}
	l, err := parseFloat(evt.K.L)
	if err != nil {
		return nil, fmt.Errorf("parse low: %w", err)
	}
	c, err := parseFloat(evt.K.C)
	if err != nil {
		return nil, fmt.Errorf("parse close: %w", err)
	}
	v, err := parseFloat(evt.K.V)
	if err != nil {
		return nil, fmt.Errorf("parse volume: %w", err)
	}

	ts := time.Unix(0, evt.K.T*int64(time.Millisecond)).UTC()

	return &Bar{
		Symbol:    evt.S,
		Timestamp: ts,
		Open:      o,
		High:      h,
		Low:       l,
		Close:     c,
		Volume:    v,
		Source:    "binance",
		IsPartial: !evt.K.X,
	}, nil
}

func (f *BinanceFeed) emit(e Event) {
	select {
	case f.out <- e:
	default:
		f.log.Warn("binance event channel full, dropping event",
			zap.String("symbol", func() string {
				if e.Bar != nil {
					return e.Bar.Symbol
				}
				return ""
			}()))
	}
}
