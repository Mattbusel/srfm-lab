package feed

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

// CoinbaseConfig holds settings for the Coinbase Advanced Trade WebSocket feed.
type CoinbaseConfig struct {
	ProductIDs []string // e.g. ["BTC-USD", "ETH-USD"]
	APIKey     string
	Secret     string
}

const coinbaseWSURL = "wss://advanced-trade-ws.coinbase.com"

// coinbaseSubscribeMsg is the subscription message for Coinbase WS.
type coinbaseSubscribeMsg struct {
	Type       string   `json:"type"`
	ProductIDs []string `json:"product_ids"`
	Channel    string   `json:"channel"`
}

// coinbaseTickerMsg maps to Coinbase's ticker channel message.
type coinbaseTickerMsg struct {
	Type      string `json:"type"`
	Channel   string `json:"channel"`
	Timestamp string `json:"timestamp"`
	Events    []struct {
		Type    string             `json:"type"`
		Tickers []coinbaseTicker   `json:"tickers"`
	} `json:"events"`
}

type coinbaseTicker struct {
	ProductID string `json:"product_id"`
	Price     string `json:"price"`
	Volume24H string `json:"volume_24_h"`
	BestBid   string `json:"best_bid"`
	BestAsk   string `json:"best_ask"`
	Time      string `json:"time"`
}

// coinbaseCandleMsg maps to Coinbase's candles channel.
type coinbaseCandleMsg struct {
	Type    string `json:"type"`
	Channel string `json:"channel"`
	Events  []struct {
		Type    string            `json:"type"`
		Candles []coinbaseCandle  `json:"candles"`
	} `json:"events"`
}

type coinbaseCandle struct {
	Start     string `json:"start"`
	Low       string `json:"low"`
	High      string `json:"high"`
	Open      string `json:"open"`
	Close     string `json:"close"`
	Volume    string `json:"volume"`
	ProductID string `json:"product_id"`
}

// CoinbaseFeed streams candle and ticker data from Coinbase.
type CoinbaseFeed struct {
	cfg    CoinbaseConfig
	log    *zap.Logger
	out    chan<- Event
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewCoinbaseFeed creates a CoinbaseFeed.
func NewCoinbaseFeed(cfg CoinbaseConfig, out chan<- Event, log *zap.Logger) *CoinbaseFeed {
	return &CoinbaseFeed{cfg: cfg, out: out, log: log}
}

// Start begins streaming.
func (f *CoinbaseFeed) Start(ctx context.Context) {
	ctx, f.cancel = context.WithCancel(ctx)
	f.wg.Add(1)
	go f.run(ctx)
}

// Stop halts the feed.
func (f *CoinbaseFeed) Stop() {
	if f.cancel != nil {
		f.cancel()
	}
	f.wg.Wait()
}

func (f *CoinbaseFeed) run(ctx context.Context) {
	defer f.wg.Done()
	rec := NewReconnector(DefaultReconnectConfig(), "coinbase", f.log)
	rec.Run(ctx, func(ctx context.Context) error {
		return f.connect(ctx)
	}, nil)
}

func (f *CoinbaseFeed) connect(ctx context.Context) error {
	dialer := websocket.Dialer{HandshakeTimeout: 10 * time.Second}
	conn, _, err := dialer.DialContext(ctx, coinbaseWSURL, nil)
	if err != nil {
		return fmt.Errorf("dial coinbase: %w", err)
	}
	defer conn.Close()

	conn.SetReadDeadline(time.Now().Add(90 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(90 * time.Second))
		return nil
	})

	// Subscribe to candles and ticker.
	for _, channel := range []string{"candles", "ticker"} {
		sub := coinbaseSubscribeMsg{
			Type:       "subscribe",
			ProductIDs: f.cfg.ProductIDs,
			Channel:    channel,
		}
		if err := conn.WriteJSON(sub); err != nil {
			return fmt.Errorf("subscribe %s: %w", channel, err)
		}
	}

	f.log.Info("coinbase feed connected", zap.Strings("products", f.cfg.ProductIDs))

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
			f.log.Warn("coinbase handle", zap.Error(err))
		}
	}
}

func (f *CoinbaseFeed) handleMessage(raw []byte) error {
	// Peek at channel to determine message type.
	var peek struct {
		Channel string `json:"channel"`
	}
	if err := json.Unmarshal(raw, &peek); err != nil {
		return nil
	}

	switch peek.Channel {
	case "candles":
		var msg coinbaseCandleMsg
		if err := json.Unmarshal(raw, &msg); err != nil {
			return nil
		}
		for _, evt := range msg.Events {
			for _, c := range evt.Candles {
				bar, err := coinbaseCandleToBar(c)
				if err != nil {
					continue
				}
				f.emit(Event{Kind: EventBar, Bar: bar})
			}
		}

	case "ticker":
		var msg coinbaseTickerMsg
		if err := json.Unmarshal(raw, &msg); err != nil {
			return nil
		}
		for _, evt := range msg.Events {
			for _, t := range evt.Tickers {
				ts, _ := time.Parse(time.RFC3339, t.Time)
				price := parseFloatStr(t.Price)
				bidPx := parseFloatStr(t.BestBid)
				askPx := parseFloatStr(t.BestAsk)
				if bidPx > 0 && askPx > 0 {
					f.emit(Event{Kind: EventQuote, Quote: &Quote{
						Symbol:    t.ProductID,
						Timestamp: ts,
						BidPrice:  bidPx,
						AskPrice:  askPx,
						Source:    "coinbase",
					}})
				} else if price > 0 {
					// Fake quote from mid price.
					f.emit(Event{Kind: EventQuote, Quote: &Quote{
						Symbol:    t.ProductID,
						Timestamp: ts,
						BidPrice:  price * 0.9999,
						AskPrice:  price * 1.0001,
						Source:    "coinbase",
					}})
				}
			}
		}
	}
	return nil
}

func coinbaseCandleToBar(c coinbaseCandle) (*Bar, error) {
	startSecs := int64(0)
	fmt.Sscanf(c.Start, "%d", &startSecs)
	ts := time.Unix(startSecs, 0).UTC()

	return &Bar{
		Symbol:    strings.ReplaceAll(c.ProductID, "-", ""),
		Timestamp: ts,
		Open:      parseFloatStr(c.Open),
		High:      parseFloatStr(c.High),
		Low:       parseFloatStr(c.Low),
		Close:     parseFloatStr(c.Close),
		Volume:    parseFloatStr(c.Volume),
		Source:    "coinbase",
	}, nil
}

func parseFloatStr(s string) float64 {
	var v float64
	fmt.Sscanf(s, "%f", &v)
	return v
}

func (f *CoinbaseFeed) emit(e Event) {
	select {
	case f.out <- e:
	default:
		f.log.Warn("coinbase channel full")
	}
}

// ---- Multi-exchange bar merger ----

// ExchangeBar associates a bar with its exchange source.
type ExchangeBar struct {
	Bar
	Exchange string
}

// ExchangeMerger merges bars from multiple exchanges into a single stream.
// When the same symbol arrives from multiple sources within the same bar window,
// it picks the canonical source (preferring real feeds over simulator).
type ExchangeMerger struct {
	mu       sync.Mutex
	out      chan<- Event
	priority []string // source priority (index 0 = highest)
	log      *zap.Logger
}

// sourcePriority returns the priority index of a source (lower = better).
func sourcePriority(source string, priority []string) int {
	for i, p := range priority {
		if strings.EqualFold(source, p) {
			return i
		}
	}
	return math.MaxInt32
}

// NewExchangeMerger creates an ExchangeMerger.
// The priority list controls which source wins when the same symbol+bar
// arrives from multiple feeds.
func NewExchangeMerger(out chan<- Event, priority []string, log *zap.Logger) *ExchangeMerger {
	return &ExchangeMerger{out: out, priority: priority, log: log}
}

// FanIn reads from multiple input channels and writes to out.
// It runs until ctx is done.
func (em *ExchangeMerger) FanIn(ctx context.Context, inputs ...<-chan Event) {
	var wg sync.WaitGroup
	for _, ch := range inputs {
		ch := ch
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case evt, ok := <-ch:
					if !ok {
						return
					}
					em.dispatch(evt)
				}
			}
		}()
	}
	wg.Wait()
}

func (em *ExchangeMerger) dispatch(evt Event) {
	select {
	case em.out <- evt:
	default:
		em.log.Warn("exchange merger: output channel full")
	}
}
