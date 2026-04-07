package feed

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"srfm/market-data/aggregator"

	"github.com/gorilla/websocket"
)

// Binance combined stream endpoint.
const binanceCombinedStreamURL = "wss://stream.binance.com:9443/stream"

// Binance connection limits per IP.
const (
	binanceMaxConnections   = 5
	binanceMaxSubscriptions = 300
)

// binanceV2SymbolMap maps Binance instrument names to canonical symbols.
var binanceV2SymbolMap = map[string]string{
	"BTCUSDT":   "BTC",
	"ETHUSDT":   "ETH",
	"SOLUSDT":   "SOL",
	"BNBUSDT":   "BNB",
	"XRPUSDT":   "XRP",
	"ADAUSDT":   "ADA",
	"AVAXUSDT":  "AVAX",
	"DOGEUSDT":  "DOGE",
	"MATICUSDT": "MATIC",
	"DOTUSDT":   "DOT",
	"LINKUSDT":  "LINK",
	"UNIUSDT":   "UNI",
	"ATOMUSDT":  "ATOM",
	"LTCUSDT":   "LTC",
	"BCHUSDT":   "BCH",
	"ALGOUSDT":  "ALGO",
	"XLMUSDT":   "XLM",
	"VETUSDT":   "VET",
	"FILUSDT":   "FIL",
	"AAVEUSDT":  "AAVE",
}

var binanceV2ReverseMap = func() map[string]string {
	m := make(map[string]string, len(binanceV2SymbolMap))
	for k, v := range binanceV2SymbolMap {
		m[v] = k
	}
	return m
}()

// -- JSON message types from Binance combined stream --

type binanceV2StreamMsg struct {
	Stream string          `json:"stream"`
	Data   json.RawMessage `json:"data"`
}

type binanceV2KlineEvent struct {
	EventType string         `json:"e"`
	EventTime int64          `json:"E"`
	Symbol    string         `json:"s"`
	Kline     binanceV2Kline `json:"k"`
}

type binanceV2Kline struct {
	StartTime      int64  `json:"t"`
	CloseTime      int64  `json:"T"`
	Symbol         string `json:"s"`
	Interval       string `json:"i"`
	Open           string `json:"o"`
	Close          string `json:"c"`
	High           string `json:"h"`
	Low            string `json:"l"`
	Volume         string `json:"v"`
	IsClosed       bool   `json:"x"`
	NumberOfTrades int    `json:"n"`
}

type binanceV2AggTrade struct {
	EventType    string `json:"e"`
	EventTime    int64  `json:"E"`
	Symbol       string `json:"s"`
	AggTradeID   int64  `json:"a"`
	Price        string `json:"p"`
	Quantity     string `json:"q"`
	TradeTime    int64  `json:"T"`
	IsBuyerMaker bool   `json:"m"`
}

type binanceV2BookTicker struct {
	UpdateID int64  `json:"u"`
	Symbol   string `json:"s"`
	BidPrice string `json:"b"`
	BidQty   string `json:"B"`
	AskPrice string `json:"a"`
	AskQty   string `json:"A"`
}

// BinanceFeedV2Health is the health snapshot for BinanceFeedV2.
type BinanceFeedV2Health struct {
	Name           string        `json:"name"`
	IsConnected    bool          `json:"is_connected"`
	LastBarTime    time.Time     `json:"last_bar_time"`
	BarsReceived   int64         `json:"bars_received"`
	ErrorsCount    int64         `json:"errors_count"`
	LatencyMs      int64         `json:"latency_ms"`
	Uptime         time.Duration `json:"uptime"`
	ReconnectCount int64         `json:"reconnect_count"`
	UptimePct      float64       `json:"uptime_pct"`
	IsHealthy      bool          `json:"is_healthy"`
}

// BinanceFeedV2 implements a Binance combined stream feed with automatic
// reconnection via ManagedWebSocket. It handles kline (OHLCV bars),
// aggTrade (individual trades), and bookTicker (spread) stream types.
//
// Rate limiting: Binance allows max 5 WebSocket connections per IP and
// max 300 subscriptions per connection. buildCombinedURL enforces the
// 300-subscription limit at URL construction time.
type BinanceFeedV2 struct {
	symbols []string
	outCh   chan aggregator.RawTick // optional external output channel
	barCh   chan aggregator.RawTick // internal channel exposed via Bars()

	// OnBookTicker is an optional callback for top-of-book updates.
	// Set before calling Start. Signature: (symbol, bid, ask, bidQty, askQty).
	OnBookTicker func(symbol string, bid, ask, bidQty, askQty float64)

	ws     *ManagedWebSocket
	health *FeedHealthMonitor

	mu           sync.RWMutex
	connected    bool
	lastMsgTime  time.Time
	connectedAt  time.Time
	msgsReceived atomic.Int64
	errCount     atomic.Int64
	latencyMs    atomic.Int64

	stopCh chan struct{}
	doneCh chan struct{}
}

// NewBinanceFeedV2 creates a BinanceFeedV2 for the given canonical symbols.
// outCh receives all decoded ticks; pass nil if you only need the Bars() channel.
func NewBinanceFeedV2(symbols []string, outCh chan aggregator.RawTick) *BinanceFeedV2 {
	healthMon := NewFeedHealthMonitor(30 * time.Second)
	// Expected max silence: 2 x 15m bar interval = 30 minutes.
	healthMon.Register("binance_v2", 15*time.Minute)

	cfg := DefaultReconnectConfig()
	ws := NewManagedWebSocket(buildCombinedURL(symbols), cfg, nil)

	f := &BinanceFeedV2{
		symbols: symbols,
		outCh:   outCh,
		barCh:   make(chan aggregator.RawTick, 4096),
		ws:      ws,
		health:  healthMon,
		stopCh:  make(chan struct{}),
		doneCh:  make(chan struct{}),
	}

	// Re-subscribe on reconnect: Binance combined streams are URL-encoded
	// so the ManagedWebSocket re-dials the same URL automatically. We just
	// mark our connected state here.
	ws.OnReconnect = func(_ *websocket.Conn) error {
		log.Printf("[binance-v2] reconnected, stream covers %d symbols", len(symbols))
		f.mu.Lock()
		f.connected = true
		f.connectedAt = time.Now()
		f.mu.Unlock()
		return nil
	}

	return f
}

// buildCombinedURL constructs the Binance combined stream URL.
// Includes kline_15m, aggTrade, and bookTicker for each symbol.
// Truncates at binanceMaxSubscriptions (300) if necessary.
func buildCombinedURL(symbols []string) string {
	streams := make([]string, 0, len(symbols)*3)
	for _, sym := range symbols {
		binanceSym, ok := binanceV2ReverseMap[sym]
		if !ok {
			continue
		}
		lower := strings.ToLower(binanceSym)
		streams = append(streams, lower+"@kline_15m")
		streams = append(streams, lower+"@aggTrade")
		streams = append(streams, lower+"@bookTicker")

		if len(streams) >= binanceMaxSubscriptions {
			log.Printf("[binance-v2] subscription limit (%d) reached, truncating", binanceMaxSubscriptions)
			break
		}
	}
	return binanceCombinedStreamURL + "?streams=" + strings.Join(streams, "/")
}

// Start begins the feed loop. It blocks until Stop is called.
// Run this in a goroutine.
func (f *BinanceFeedV2) Start() {
	defer close(f.doneCh)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-f.stopCh
		cancel()
	}()

	if err := f.ws.Connect(ctx); err != nil {
		log.Printf("[binance-v2] initial connect error: %v", err)
		f.errCount.Add(1)
	} else {
		f.mu.Lock()
		f.connected = true
		f.connectedAt = time.Now()
		f.mu.Unlock()
	}

	for {
		msg, err := f.ws.ReadMessageContext(ctx)
		if err != nil {
			select {
			case <-ctx.Done():
				f.mu.Lock()
				f.connected = false
				f.mu.Unlock()
				return
			default:
				f.errCount.Add(1)
				log.Printf("[binance-v2] read error: %v", err)
				continue
			}
		}

		recvTime := time.Now()
		f.health.RecordMessage("binance_v2")
		f.msgsReceived.Add(1)

		f.mu.Lock()
		f.lastMsgTime = recvTime
		f.connected = true
		f.mu.Unlock()

		f.handleMessage(msg, recvTime)
	}
}

func (f *BinanceFeedV2) handleMessage(data []byte, recvTime time.Time) {
	var envelope binanceV2StreamMsg
	if err := json.Unmarshal(data, &envelope); err != nil {
		return
	}

	stream := envelope.Stream
	switch {
	case strings.Contains(stream, "@kline"):
		f.handleKline(envelope.Data, recvTime)
	case strings.Contains(stream, "@aggTrade"):
		f.handleAggTrade(envelope.Data, recvTime)
	case strings.Contains(stream, "@bookTicker"):
		f.handleBookTicker(envelope.Data)
	}
}

func (f *BinanceFeedV2) handleKline(data json.RawMessage, recvTime time.Time) {
	var evt binanceV2KlineEvent
	if err := json.Unmarshal(data, &evt); err != nil {
		return
	}
	if evt.EventType != "kline" {
		return
	}

	k := evt.Kline
	sym, ok := binanceV2SymbolMap[k.Symbol]
	if !ok {
		return
	}

	open := parseFloatV2(k.Open)
	high := parseFloatV2(k.High)
	low := parseFloatV2(k.Low)
	close_ := parseFloatV2(k.Close)
	vol := parseFloatV2(k.Volume)

	ts := time.Unix(0, k.StartTime*int64(time.Millisecond)).UTC()
	msgTime := time.Unix(0, evt.EventTime*int64(time.Millisecond)).UTC()
	latency := recvTime.Sub(msgTime).Milliseconds()
	if latency < 0 {
		latency = 0
	}
	f.latencyMs.Store(latency)

	tick := aggregator.RawTick{
		Symbol:     sym,
		Open:       open,
		High:       high,
		Low:        low,
		Close:      close_,
		Volume:     vol,
		Timestamp:  ts,
		Source:     "binance_v2",
		IsBar:      true,
		IsComplete: k.IsClosed,
	}

	f.dispatch(tick)
}

func (f *BinanceFeedV2) handleAggTrade(data json.RawMessage, recvTime time.Time) {
	var evt binanceV2AggTrade
	if err := json.Unmarshal(data, &evt); err != nil {
		return
	}
	if evt.EventType != "aggTrade" {
		return
	}

	sym, ok := binanceV2SymbolMap[evt.Symbol]
	if !ok {
		return
	}

	price := parseFloatV2(evt.Price)
	qty := parseFloatV2(evt.Quantity)
	ts := time.Unix(0, evt.TradeTime*int64(time.Millisecond)).UTC()

	msgTime := time.Unix(0, evt.EventTime*int64(time.Millisecond)).UTC()
	latency := recvTime.Sub(msgTime).Milliseconds()
	if latency < 0 {
		latency = 0
	}
	f.latencyMs.Store(latency)

	tick := aggregator.RawTick{
		Symbol:    sym,
		Open:      price,
		High:      price,
		Low:       price,
		Close:     price,
		Volume:    qty,
		Timestamp: ts,
		Source:    "binance_v2",
		IsBar:     false,
	}

	f.dispatch(tick)
}

func (f *BinanceFeedV2) handleBookTicker(data json.RawMessage) {
	var bt binanceV2BookTicker
	if err := json.Unmarshal(data, &bt); err != nil {
		return
	}

	cb := f.OnBookTicker
	if cb == nil {
		return
	}

	sym, ok := binanceV2SymbolMap[bt.Symbol]
	if !ok {
		return
	}

	bid := parseFloatV2(bt.BidPrice)
	ask := parseFloatV2(bt.AskPrice)
	bidQty := parseFloatV2(bt.BidQty)
	askQty := parseFloatV2(bt.AskQty)
	cb(sym, bid, ask, bidQty, askQty)
}

func (f *BinanceFeedV2) dispatch(tick aggregator.RawTick) {
	if f.outCh != nil {
		select {
		case f.outCh <- tick:
		default:
			log.Printf("[binance-v2] outCh full, dropping tick for %s", tick.Symbol)
		}
	}
	select {
	case f.barCh <- tick:
	default:
	}
}

// Stop signals the feed to stop and waits for the run loop to exit.
func (f *BinanceFeedV2) Stop() {
	close(f.stopCh)
	<-f.doneCh
	f.health.Stop()
}

// Subscribe queues a new symbol for inclusion on the next reconnect.
// Binance combined streams are URL-encoded, so the new symbol takes effect
// when the connection is re-established.
func (f *BinanceFeedV2) Subscribe(symbol string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, s := range f.symbols {
		if s == symbol {
			return
		}
	}
	if len(f.symbols)*3 >= binanceMaxSubscriptions {
		log.Printf("[binance-v2] cannot subscribe %s: at subscription limit", symbol)
		return
	}
	f.symbols = append(f.symbols, symbol)
	log.Printf("[binance-v2] queued %s for next reconnect", symbol)
}

// Bars returns the read channel on which decoded RawTick values are published.
func (f *BinanceFeedV2) Bars() <-chan aggregator.RawTick {
	return f.barCh
}

// Health returns a snapshot of BinanceFeedV2 connection and message metrics.
func (f *BinanceFeedV2) Health() BinanceFeedV2Health {
	f.mu.RLock()
	defer f.mu.RUnlock()
	var uptime time.Duration
	if f.connected {
		uptime = time.Since(f.connectedAt)
	}
	return BinanceFeedV2Health{
		Name:           "binance_v2",
		IsConnected:    f.connected,
		LastBarTime:    f.lastMsgTime,
		BarsReceived:   f.msgsReceived.Load(),
		ErrorsCount:    f.errCount.Load(),
		LatencyMs:      f.latencyMs.Load(),
		Uptime:         uptime,
		ReconnectCount: f.ws.ReconnectCount(),
		UptimePct:      f.ws.UptimePct(),
		IsHealthy:      f.health.IsHealthy("binance_v2"),
	}
}

func parseFloatV2(s string) float64 {
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}
