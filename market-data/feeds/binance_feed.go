package feeds

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"srfm/market-data/aggregator"
	"srfm/market-data/monitoring"

	"github.com/gorilla/websocket"
)

const (
	binanceWSURL    = "wss://stream.binance.com:9443/stream"
	binanceMaxRetry = 60 * time.Second
	binanceBaseRetry = time.Second
)

// binanceSymbolMap maps Binance pair names to canonical symbols.
var binanceSymbolMap = map[string]string{
	"BTCUSDT":  "BTC",
	"ETHUSDT":  "ETH",
	"SOLUSDT":  "SOL",
	"BNBUSDT":  "BNB",
	"XRPUSDT":  "XRP",
	"ADAUSDT":  "ADA",
	"AVAXUSDT": "AVAX",
	"DOGEUSDT": "DOGE",
	"MATICUSDT": "MATIC",
	"DOTUSDT":  "DOT",
	"LINKUSDT": "LINK",
	"UNIUSDT":  "UNI",
	"ATOMUSDT": "ATOM",
	"LTCUSDT":  "LTC",
	"BCHUSDT":  "BCH",
	"ALGOUSDT": "ALGO",
	"XLMUSDT":  "XLM",
	"VETUSDT":  "VET",
	"FILUSDT":  "FIL",
	"AAVEUSDT": "AAVE",
}

// reverseSymbolMap maps canonical -> Binance
var reverseSymbolMap = func() map[string]string {
	m := make(map[string]string, len(binanceSymbolMap))
	for k, v := range binanceSymbolMap {
		m[v] = k
	}
	return m
}()

// BinanceFeed connects to Binance WebSocket kline feed.
type BinanceFeed struct {
	symbols  []string
	outCh    chan<- aggregator.RawTick
	metrics  *monitoring.Metrics

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

// Binance WebSocket stream message
type binanceStreamMsg struct {
	Stream string          `json:"stream"`
	Data   json.RawMessage `json:"data"`
}

type binanceKlineEvent struct {
	EventType string       `json:"e"`
	EventTime int64        `json:"E"`
	Symbol    string       `json:"s"`
	Kline     binanceKline `json:"k"`
}

type binanceKline struct {
	StartTime            int64  `json:"t"`
	CloseTime            int64  `json:"T"`
	Symbol               string `json:"s"`
	Interval             string `json:"i"`
	Open                 string `json:"o"`
	Close                string `json:"c"`
	High                 string `json:"h"`
	Low                  string `json:"l"`
	Volume               string `json:"v"`
	IsClosed             bool   `json:"x"`
	NumberOfTrades       int    `json:"n"`
}

// NewBinanceFeed creates a Binance kline feed.
func NewBinanceFeed(symbols []string, outCh chan<- aggregator.RawTick, metrics *monitoring.Metrics) *BinanceFeed {
	return &BinanceFeed{
		symbols: symbols,
		outCh:   outCh,
		metrics: metrics,
		stopCh:  make(chan struct{}),
		doneCh:  make(chan struct{}),
	}
}

// Start begins the feed. Blocks until Stop() is called.
func (f *BinanceFeed) Start() {
	defer close(f.doneCh)
	backoff := binanceBaseRetry
	for {
		select {
		case <-f.stopCh:
			return
		default:
		}

		err := f.connect()
		if err != nil {
			f.errCount.Add(1)
			f.metrics.FeedError("binance")
			log.Printf("[binance] connection error: %v, retrying in %v", err, backoff)
		}

		f.mu.Lock()
		f.connected = false
		f.mu.Unlock()

		select {
		case <-f.stopCh:
			return
		case <-time.After(backoff):
			backoff = time.Duration(math.Min(float64(backoff*2), float64(binanceMaxRetry)))
		}
	}
}

// Stop signals the feed to stop.
func (f *BinanceFeed) Stop() {
	close(f.stopCh)
	<-f.doneCh
}

// Health returns current feed health snapshot.
func (f *BinanceFeed) Health() FeedHealth {
	f.mu.RLock()
	defer f.mu.RUnlock()
	var uptime time.Duration
	if f.connected {
		uptime = time.Since(f.connectedAt)
	}
	return FeedHealth{
		Name:         "binance",
		IsConnected:  f.connected,
		LastBarTime:  f.lastMsgTime,
		BarsReceived: f.msgsReceived.Load(),
		ErrorsCount:  f.errCount.Load(),
		LatencyMs:    f.latencyMs.Load(),
		Uptime:       uptime,
	}
}

func (f *BinanceFeed) buildStreamURL() string {
	streams := make([]string, 0, len(f.symbols))
	for _, sym := range f.symbols {
		binanceSym, ok := reverseSymbolMap[sym]
		if !ok {
			continue
		}
		streams = append(streams, strings.ToLower(binanceSym)+"@kline_1m")
	}
	return binanceWSURL + "?streams=" + strings.Join(streams, "/")
}

func (f *BinanceFeed) connect() error {
	url := f.buildStreamURL()
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	f.mu.Lock()
	f.connected = true
	f.connectedAt = time.Now()
	f.mu.Unlock()
	log.Printf("[binance] connected, subscribed to %d kline streams", len(f.symbols))

	conn.SetPingHandler(func(data string) error {
		return conn.WriteMessage(websocket.PongMessage, []byte(data))
	})

	return f.readLoop(conn)
}

func (f *BinanceFeed) readLoop(conn *websocket.Conn) error {
	for {
		select {
		case <-f.stopCh:
			return nil
		default:
		}

		conn.SetReadDeadline(time.Now().Add(90 * time.Second))
		_, data, err := conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("read: %w", err)
		}

		recvTime := time.Now()

		var msg binanceStreamMsg
		if err := json.Unmarshal(data, &msg); err != nil {
			log.Printf("[binance] parse error: %v", err)
			continue
		}

		var evt binanceKlineEvent
		if err := json.Unmarshal(msg.Data, &evt); err != nil {
			continue
		}

		if evt.EventType != "kline" {
			continue
		}

		k := evt.Kline
		sym, ok := binanceSymbolMap[k.Symbol]
		if !ok {
			continue
		}

		open := parseFloat(k.Open)
		high := parseFloat(k.High)
		low := parseFloat(k.Low)
		close_ := parseFloat(k.Close)
		vol := parseFloat(k.Volume)

		ts := time.Unix(0, k.StartTime*int64(time.Millisecond)).UTC()
		msgTime := time.Unix(0, evt.EventTime*int64(time.Millisecond)).UTC()
		latency := recvTime.Sub(msgTime).Milliseconds()
		if latency < 0 {
			latency = 0
		}
		f.latencyMs.Store(latency)
		f.msgsReceived.Add(1)
		f.metrics.BarReceived("binance")

		f.mu.Lock()
		f.lastMsgTime = recvTime
		f.mu.Unlock()

		tick := aggregator.RawTick{
			Symbol:      sym,
			Open:        open,
			High:        high,
			Low:         low,
			Close:       close_,
			Volume:      vol,
			Timestamp:   ts,
			Source:      "binance",
			IsBar:       true,
			IsComplete:  k.IsClosed,
		}

		select {
		case f.outCh <- tick:
		default:
			log.Printf("[binance] output channel full, dropping tick for %s", sym)
		}
	}
}

func parseFloat(s string) float64 {
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}
