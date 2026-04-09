package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Order Book
// ---------------------------------------------------------------------------

// PriceLevel is one level in the order book.
type PriceLevel struct {
	Price    float64 `json:"price"`
	Qty      float64 `json:"qty"`
	NumOrders int    `json:"num_orders"`
}

// OrderBookEntry is a resting order on the book.
type OrderBookEntry struct {
	OrderID   string
	AgentID   string
	Price     float64
	Qty       float64
	Side      Side
	Timestamp time.Time
}

// OrderBook is a price-time priority limit order book.
type OrderBook struct {
	mu        sync.Mutex
	symbol    string
	tickSize  float64
	bids      []OrderBookEntry // sorted descending by price
	asks      []OrderBookEntry // sorted ascending by price
	lastPrice float64
	fillSeq   int64
}

// NewOrderBook creates a new order book.
func NewOrderBook(symbol string, tickSize float64) *OrderBook {
	return &OrderBook{
		symbol:   symbol,
		tickSize: tickSize,
	}
}

// ProcessOrder matches an incoming order against the book.
func (ob *OrderBook) ProcessOrder(o Order) []Fill {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	var fills []Fill

	if o.Type == OrderTypeMarket || (o.Type == OrderTypeLimit && o.Price > 0) {
		fills = ob.match(o)
	}

	// if limit order with remaining qty, rest on book
	if o.Type == OrderTypeLimit && o.Qty > 0 && o.Price > 0 {
		entry := OrderBookEntry{
			OrderID:   o.ID,
			AgentID:   o.AgentID,
			Price:     o.Price,
			Qty:       o.Qty,
			Side:      o.Side,
			Timestamp: o.Timestamp,
		}
		if o.Side == SideBuy {
			ob.insertBid(entry)
		} else {
			ob.insertAsk(entry)
		}
	}

	return fills
}

func (ob *OrderBook) match(o Order) []Fill {
	var fills []Fill

	if o.Side == SideBuy {
		for len(ob.asks) > 0 && o.Qty > 0 {
			best := &ob.asks[0]
			if o.Type == OrderTypeLimit && o.Price < best.Price {
				break
			}
			fillQty := math.Min(o.Qty, best.Qty)
			ob.fillSeq++
			f := Fill{
				ID:        fmt.Sprintf("f-%s-%d", ob.symbol, ob.fillSeq),
				Symbol:    ob.symbol,
				Price:     best.Price,
				Qty:       fillQty,
				BuyerID:   o.AgentID,
				SellerID:  best.AgentID,
				Timestamp: time.Now(),
				Aggressor: SideBuy,
			}
			fills = append(fills, f)
			ob.lastPrice = best.Price
			o.Qty -= fillQty
			best.Qty -= fillQty
			if best.Qty <= 1e-12 {
				ob.asks = ob.asks[1:]
			}
		}
	} else {
		for len(ob.bids) > 0 && o.Qty > 0 {
			best := &ob.bids[0]
			if o.Type == OrderTypeLimit && o.Price > best.Price {
				break
			}
			fillQty := math.Min(o.Qty, best.Qty)
			ob.fillSeq++
			f := Fill{
				ID:        fmt.Sprintf("f-%s-%d", ob.symbol, ob.fillSeq),
				Symbol:    ob.symbol,
				Price:     best.Price,
				Qty:       fillQty,
				BuyerID:   best.AgentID,
				SellerID:  o.AgentID,
				Timestamp: time.Now(),
				Aggressor: SideSell,
			}
			fills = append(fills, f)
			ob.lastPrice = best.Price
			o.Qty -= fillQty
			best.Qty -= fillQty
			if best.Qty <= 1e-12 {
				ob.bids = ob.bids[1:]
			}
		}
	}

	return fills
}

func (ob *OrderBook) insertBid(e OrderBookEntry) {
	i := sort.Search(len(ob.bids), func(i int) bool {
		return ob.bids[i].Price < e.Price
	})
	ob.bids = append(ob.bids, OrderBookEntry{})
	copy(ob.bids[i+1:], ob.bids[i:])
	ob.bids[i] = e
}

func (ob *OrderBook) insertAsk(e OrderBookEntry) {
	i := sort.Search(len(ob.asks), func(i int) bool {
		return ob.asks[i].Price > e.Price
	})
	ob.asks = append(ob.asks, OrderBookEntry{})
	copy(ob.asks[i+1:], ob.asks[i:])
	ob.asks[i] = e
}

// BestBidAsk returns [bestBid, bestAsk]. Zero means no orders.
func (ob *OrderBook) BestBidAsk() [2]float64 {
	ob.mu.Lock()
	defer ob.mu.Unlock()
	var result [2]float64
	if len(ob.bids) > 0 {
		result[0] = ob.bids[0].Price
	}
	if len(ob.asks) > 0 {
		result[1] = ob.asks[0].Price
	}
	return result
}

// LastPrice returns the last traded price.
func (ob *OrderBook) LastPrice() float64 {
	ob.mu.Lock()
	defer ob.mu.Unlock()
	return ob.lastPrice
}

// Depth returns [numBidLevels, numAskLevels].
func (ob *OrderBook) Depth() [2]int {
	ob.mu.Lock()
	defer ob.mu.Unlock()
	return [2]int{len(ob.bids), len(ob.asks)}
}

// Snapshot returns a MarketDataSnapshot.
func (ob *OrderBook) Snapshot() MarketDataSnapshot {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	snap := MarketDataSnapshot{
		Symbol:    ob.symbol,
		LastPrice: ob.lastPrice,
		Timestamp: time.Now(),
	}

	if len(ob.bids) > 0 {
		snap.BestBid = ob.bids[0].Price
		for _, b := range ob.bids {
			snap.BidDepth += b.Qty
		}
	}
	if len(ob.asks) > 0 {
		snap.BestAsk = ob.asks[0].Price
		for _, a := range ob.asks {
			snap.AskDepth += a.Qty
		}
	}
	if snap.BestBid > 0 && snap.BestAsk > 0 {
		snap.Spread = snap.BestAsk - snap.BestBid
		snap.MidPrice = (snap.BestBid + snap.BestAsk) / 2.0
	}

	snap.NumBidLevels = len(ob.bids)
	snap.NumAskLevels = len(ob.asks)

	return snap
}

// OrderBookView is a serialisable depth-of-book view.
type OrderBookView struct {
	Symbol    string       `json:"symbol"`
	Bids      []PriceLevel `json:"bids"`
	Asks      []PriceLevel `json:"asks"`
	BestBid   float64      `json:"best_bid"`
	BestAsk   float64      `json:"best_ask"`
	Spread    float64      `json:"spread"`
	LastPrice float64      `json:"last_price"`
	Timestamp string       `json:"timestamp"`
}

// View returns an aggregated view of the top N levels.
func (ob *OrderBook) View(n int) OrderBookView {
	ob.mu.Lock()
	defer ob.mu.Unlock()

	v := OrderBookView{
		Symbol:    ob.symbol,
		LastPrice: ob.lastPrice,
		Timestamp: time.Now().Format(time.RFC3339Nano),
	}

	// aggregate bids by price
	bidMap := make(map[float64]*PriceLevel)
	var bidPrices []float64
	for _, b := range ob.bids {
		pl, ok := bidMap[b.Price]
		if !ok {
			pl = &PriceLevel{Price: b.Price}
			bidMap[b.Price] = pl
			bidPrices = append(bidPrices, b.Price)
		}
		pl.Qty += b.Qty
		pl.NumOrders++
	}
	sort.Float64s(bidPrices)
	// reverse for descending
	for i, j := 0, len(bidPrices)-1; i < j; i, j = i+1, j-1 {
		bidPrices[i], bidPrices[j] = bidPrices[j], bidPrices[i]
	}
	for i, p := range bidPrices {
		if i >= n {
			break
		}
		v.Bids = append(v.Bids, *bidMap[p])
	}

	// aggregate asks by price
	askMap := make(map[float64]*PriceLevel)
	var askPrices []float64
	for _, a := range ob.asks {
		pl, ok := askMap[a.Price]
		if !ok {
			pl = &PriceLevel{Price: a.Price}
			askMap[a.Price] = pl
			askPrices = append(askPrices, a.Price)
		}
		pl.Qty += a.Qty
		pl.NumOrders++
	}
	sort.Float64s(askPrices)
	for i, p := range askPrices {
		if i >= n {
			break
		}
		v.Asks = append(v.Asks, *askMap[p])
	}

	if len(v.Bids) > 0 {
		v.BestBid = v.Bids[0].Price
	}
	if len(v.Asks) > 0 {
		v.BestAsk = v.Asks[0].Price
	}
	if v.BestBid > 0 && v.BestAsk > 0 {
		v.Spread = v.BestAsk - v.BestBid
	}

	return v
}

// DrainSide removes a fraction of orders from one side.
func (ob *OrderBook) DrainSide(bids bool, fraction float64) {
	ob.mu.Lock()
	defer ob.mu.Unlock()
	if fraction <= 0 {
		return
	}
	if fraction > 1 {
		fraction = 1
	}
	if bids {
		keep := int(float64(len(ob.bids)) * (1 - fraction))
		if keep < 0 {
			keep = 0
		}
		ob.bids = ob.bids[:keep]
	} else {
		keep := int(float64(len(ob.asks)) * (1 - fraction))
		if keep < 0 {
			keep = 0
		}
		ob.asks = ob.asks[:keep]
	}
}

// ---------------------------------------------------------------------------
// Market data snapshot
// ---------------------------------------------------------------------------

// MarketDataSnapshot captures one point-in-time view of a symbol.
type MarketDataSnapshot struct {
	Symbol       string    `json:"symbol"`
	Timestamp    time.Time `json:"timestamp"`
	LastPrice    float64   `json:"last_price"`
	BestBid      float64   `json:"best_bid"`
	BestAsk      float64   `json:"best_ask"`
	Spread       float64   `json:"spread"`
	MidPrice     float64   `json:"mid_price"`
	BidDepth     float64   `json:"bid_depth"`
	AskDepth     float64   `json:"ask_depth"`
	NumBidLevels int       `json:"num_bid_levels"`
	NumAskLevels int       `json:"num_ask_levels"`
	Bar          OHLCVBar  `json:"bar"`
}

// OHLCVBar is an OHLCV candle.
type OHLCVBar struct {
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	Trades    int64     `json:"trades"`
	VWAP      float64   `json:"vwap"`
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
}

// ---------------------------------------------------------------------------
// Trade log (circular buffer)
// ---------------------------------------------------------------------------

// TradeLog is a fixed-capacity circular buffer of trades.
type TradeLog struct {
	mu   sync.Mutex
	buf  []Fill
	cap  int
	head int
	size int
}

// NewTradeLog creates a trade log with the given capacity.
func NewTradeLog(capacity int) *TradeLog {
	return &TradeLog{
		buf: make([]Fill, capacity),
		cap: capacity,
	}
}

// Add inserts a trade.
func (tl *TradeLog) Add(f Fill) {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	tl.buf[tl.head] = f
	tl.head = (tl.head + 1) % tl.cap
	if tl.size < tl.cap {
		tl.size++
	}
}

// Last returns the last n trades in chronological order.
func (tl *TradeLog) Last(n int) []Fill {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	if n > tl.size {
		n = tl.size
	}
	if n <= 0 {
		return nil
	}
	out := make([]Fill, n)
	start := (tl.head - n + tl.cap) % tl.cap
	for i := 0; i < n; i++ {
		out[i] = tl.buf[(start+i)%tl.cap]
	}
	return out
}

// Len returns the number of stored trades.
func (tl *TradeLog) Len() int {
	tl.mu.Lock()
	defer tl.mu.Unlock()
	return tl.size
}

// ---------------------------------------------------------------------------
// Bar aggregator
// ---------------------------------------------------------------------------

// BarAggregator builds OHLCV bars from tick data.
type BarAggregator struct {
	mu       sync.Mutex
	symbol   string
	freq     time.Duration
	current  OHLCVBar
	history  []OHLCVBar
	maxHist  int
	started  bool
	vwapSum  float64
	vwapQty  float64
}

// NewBarAggregator creates a new aggregator.
func NewBarAggregator(symbol string, freq time.Duration) *BarAggregator {
	return &BarAggregator{
		symbol:  symbol,
		freq:    freq,
		maxHist: 10000,
	}
}

// AddTick feeds a new tick into the aggregator.
func (ba *BarAggregator) AddTick(price, qty float64, ts time.Time) {
	ba.mu.Lock()
	defer ba.mu.Unlock()

	if !ba.started {
		ba.current = OHLCVBar{
			Open:      price,
			High:      price,
			Low:       price,
			Close:     price,
			Volume:    qty,
			Trades:    1,
			StartTime: ts.Truncate(ba.freq),
			EndTime:   ts.Truncate(ba.freq).Add(ba.freq),
		}
		ba.vwapSum = price * qty
		ba.vwapQty = qty
		ba.started = true
		return
	}

	// check if we rolled into a new bar
	barStart := ts.Truncate(ba.freq)
	if barStart.After(ba.current.StartTime) || barStart.Equal(ba.current.EndTime) {
		// finalize current bar
		if ba.vwapQty > 0 {
			ba.current.VWAP = ba.vwapSum / ba.vwapQty
		}
		ba.history = append(ba.history, ba.current)
		if len(ba.history) > ba.maxHist {
			ba.history = ba.history[1:]
		}
		// start new bar
		ba.current = OHLCVBar{
			Open:      price,
			High:      price,
			Low:       price,
			Close:     price,
			Volume:    qty,
			Trades:    1,
			StartTime: barStart,
			EndTime:   barStart.Add(ba.freq),
		}
		ba.vwapSum = price * qty
		ba.vwapQty = qty
		return
	}

	// update current bar
	if price > ba.current.High {
		ba.current.High = price
	}
	if price < ba.current.Low {
		ba.current.Low = price
	}
	ba.current.Close = price
	ba.current.Volume += qty
	ba.current.Trades++
	ba.vwapSum += price * qty
	ba.vwapQty += qty
}

// CurrentBar returns the in-progress bar.
func (ba *BarAggregator) CurrentBar() OHLCVBar {
	ba.mu.Lock()
	defer ba.mu.Unlock()
	bar := ba.current
	if ba.vwapQty > 0 {
		bar.VWAP = ba.vwapSum / ba.vwapQty
	}
	return bar
}

// History returns completed bars.
func (ba *BarAggregator) History(n int) []OHLCVBar {
	ba.mu.Lock()
	defer ba.mu.Unlock()
	if n > len(ba.history) {
		n = len(ba.history)
	}
	if n <= 0 {
		return nil
	}
	out := make([]OHLCVBar, n)
	copy(out, ba.history[len(ba.history)-n:])
	return out
}

// ---------------------------------------------------------------------------
// Data feed (pub/sub)
// ---------------------------------------------------------------------------

// DataFeed distributes market data to subscribers via channels.
type DataFeed struct {
	mu          sync.Mutex
	subscribers map[chan MarketData]struct{}
	bufSize     int
}

// NewDataFeed creates a data feed with the given subscriber buffer size.
func NewDataFeed(bufSize int) *DataFeed {
	return &DataFeed{
		subscribers: make(map[chan MarketData]struct{}),
		bufSize:     bufSize,
	}
}

// Subscribe returns a channel that receives market data updates.
func (df *DataFeed) Subscribe() chan MarketData {
	ch := make(chan MarketData, df.bufSize)
	df.mu.Lock()
	df.subscribers[ch] = struct{}{}
	df.mu.Unlock()
	return ch
}

// Unsubscribe removes a subscriber.
func (df *DataFeed) Unsubscribe(ch chan MarketData) {
	df.mu.Lock()
	delete(df.subscribers, ch)
	df.mu.Unlock()
	close(ch)
}

// Publish sends market data to all subscribers.
func (df *DataFeed) Publish(md MarketData) {
	df.mu.Lock()
	defer df.mu.Unlock()
	for ch := range df.subscribers {
		select {
		case ch <- md:
		default:
			// slow subscriber, drop
		}
	}
}

// NumSubscribers returns the count.
func (df *DataFeed) NumSubscribers() int {
	df.mu.Lock()
	defer df.mu.Unlock()
	return len(df.subscribers)
}

// ---------------------------------------------------------------------------
// Historical recorder
// ---------------------------------------------------------------------------

// HistoricalRecord is one line in the JSONL file.
type HistoricalRecord struct {
	Bar       int64              `json:"bar"`
	Symbol    string             `json:"symbol"`
	Timestamp string             `json:"timestamp"`
	Snapshot  MarketDataSnapshot `json:"snapshot"`
}

// HistoricalRecorder saves simulation data to a JSONL file.
type HistoricalRecorder struct {
	mu     sync.Mutex
	file   *os.File
	writer *bufio.Writer
	count  int64
}

// NewHistoricalRecorder opens a file for recording.
func NewHistoricalRecorder(path string) (*HistoricalRecorder, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create recorder file: %w", err)
	}
	return &HistoricalRecorder{
		file:   f,
		writer: bufio.NewWriterSize(f, 64*1024),
	}, nil
}

// WriteSnapshot records one snapshot.
func (hr *HistoricalRecorder) WriteSnapshot(bar int64, symbol string, snap MarketDataSnapshot) error {
	hr.mu.Lock()
	defer hr.mu.Unlock()

	rec := HistoricalRecord{
		Bar:       bar,
		Symbol:    symbol,
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Snapshot:  snap,
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	_, err = hr.writer.Write(data)
	if err != nil {
		return err
	}
	_, err = hr.writer.WriteString("\n")
	hr.count++

	// flush periodically
	if hr.count%1000 == 0 {
		_ = hr.writer.Flush()
	}
	return err
}

// Close flushes and closes the file.
func (hr *HistoricalRecorder) Close() error {
	hr.mu.Lock()
	defer hr.mu.Unlock()
	if err := hr.writer.Flush(); err != nil {
		return err
	}
	return hr.file.Close()
}

// Count returns the number of records written.
func (hr *HistoricalRecorder) Count() int64 {
	hr.mu.Lock()
	defer hr.mu.Unlock()
	return hr.count
}

// ---------------------------------------------------------------------------
// Replay engine
// ---------------------------------------------------------------------------

// ReplayEngine reads a JSONL recording and replays it.
type ReplayEngine struct {
	mu       sync.Mutex
	path     string
	records  []HistoricalRecord
	loaded   bool
	position int
	speed    float64 // playback speed multiplier
	playing  bool
	stopCh   chan struct{}
	feed     *DataFeed
}

// NewReplayEngine creates a replay engine for the given file.
func NewReplayEngine(path string, feed *DataFeed) *ReplayEngine {
	return &ReplayEngine{
		path:  path,
		speed: 1.0,
		feed:  feed,
	}
}

// Load reads all records from disk into memory.
func (re *ReplayEngine) Load() error {
	re.mu.Lock()
	defer re.mu.Unlock()

	f, err := os.Open(re.path)
	if err != nil {
		return fmt.Errorf("open replay file: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 256*1024), 256*1024)

	re.records = nil
	for scanner.Scan() {
		var rec HistoricalRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			continue // skip malformed lines
		}
		re.records = append(re.records, rec)
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan replay file: %w", err)
	}

	re.loaded = true
	re.position = 0
	return nil
}

// NumRecords returns how many records are loaded.
func (re *ReplayEngine) NumRecords() int {
	re.mu.Lock()
	defer re.mu.Unlock()
	return len(re.records)
}

// SetSpeed sets the playback speed multiplier.
func (re *ReplayEngine) SetSpeed(s float64) {
	re.mu.Lock()
	defer re.mu.Unlock()
	if s <= 0 {
		s = 1.0
	}
	re.speed = s
}

// Play starts replaying records, publishing to the feed.
func (re *ReplayEngine) Play() error {
	re.mu.Lock()
	if !re.loaded || len(re.records) == 0 {
		re.mu.Unlock()
		return fmt.Errorf("no records loaded")
	}
	if re.playing {
		re.mu.Unlock()
		return fmt.Errorf("already playing")
	}
	re.playing = true
	re.stopCh = make(chan struct{})
	speed := re.speed
	records := re.records
	pos := re.position
	re.mu.Unlock()

	go func() {
		defer func() {
			re.mu.Lock()
			re.playing = false
			re.mu.Unlock()
		}()

		for i := pos; i < len(records); i++ {
			select {
			case <-re.stopCh:
				re.mu.Lock()
				re.position = i
				re.mu.Unlock()
				return
			default:
			}

			rec := records[i]

			// compute delay from record timestamps
			if i > 0 {
				prevTS, _ := time.Parse(time.RFC3339Nano, records[i-1].Timestamp)
				curTS, _ := time.Parse(time.RFC3339Nano, rec.Timestamp)
				if !prevTS.IsZero() && !curTS.IsZero() {
					delay := curTS.Sub(prevTS)
					if speed > 0 {
						delay = time.Duration(float64(delay) / speed)
					}
					if delay > 0 && delay < 10*time.Second {
						time.Sleep(delay)
					}
				}
			}

			// publish as MarketData
			md := MarketData{
				Bar:       rec.Bar,
				Timestamp: time.Now(),
				Snapshots: map[string]MarketDataSnapshot{
					rec.Symbol: rec.Snapshot,
				},
			}
			re.feed.Publish(md)

			re.mu.Lock()
			re.position = i + 1
			re.mu.Unlock()
		}
	}()

	return nil
}

// Stop halts playback.
func (re *ReplayEngine) Stop() {
	re.mu.Lock()
	defer re.mu.Unlock()
	if re.playing && re.stopCh != nil {
		close(re.stopCh)
	}
}

// Reset rewinds to the beginning.
func (re *ReplayEngine) Reset() {
	re.mu.Lock()
	defer re.mu.Unlock()
	re.position = 0
}

// Progress returns (current, total).
func (re *ReplayEngine) Progress() (int, int) {
	re.mu.Lock()
	defer re.mu.Unlock()
	return re.position, len(re.records)
}

// IsPlaying returns true if currently replaying.
func (re *ReplayEngine) IsPlaying() bool {
	re.mu.Lock()
	defer re.mu.Unlock()
	return re.playing
}

// SeekTo jumps to a specific record index.
func (re *ReplayEngine) SeekTo(pos int) error {
	re.mu.Lock()
	defer re.mu.Unlock()
	if pos < 0 || pos >= len(re.records) {
		return fmt.Errorf("position %d out of range [0, %d)", pos, len(re.records))
	}
	re.position = pos
	return nil
}

// GetRecord returns a specific record by index.
func (re *ReplayEngine) GetRecord(idx int) (HistoricalRecord, error) {
	re.mu.Lock()
	defer re.mu.Unlock()
	if idx < 0 || idx >= len(re.records) {
		return HistoricalRecord{}, fmt.Errorf("index %d out of range", idx)
	}
	return re.records[idx], nil
}

// SliceRecords returns records in range [from, to).
func (re *ReplayEngine) SliceRecords(from, to int) []HistoricalRecord {
	re.mu.Lock()
	defer re.mu.Unlock()
	if from < 0 {
		from = 0
	}
	if to > len(re.records) {
		to = len(re.records)
	}
	if from >= to {
		return nil
	}
	out := make([]HistoricalRecord, to-from)
	copy(out, re.records[from:to])
	return out
}

// FilterBySymbol returns records matching the symbol.
func (re *ReplayEngine) FilterBySymbol(symbol string) []HistoricalRecord {
	re.mu.Lock()
	defer re.mu.Unlock()
	var out []HistoricalRecord
	for _, r := range re.records {
		if r.Symbol == symbol {
			out = append(out, r)
		}
	}
	return out
}

// Stats returns summary statistics of the loaded recording.
func (re *ReplayEngine) Stats() map[string]interface{} {
	re.mu.Lock()
	defer re.mu.Unlock()
	if len(re.records) == 0 {
		return map[string]interface{}{"records": 0}
	}
	symbols := make(map[string]int)
	var minBar, maxBar int64
	minBar = math.MaxInt64
	for _, r := range re.records {
		symbols[r.Symbol]++
		if r.Bar < minBar {
			minBar = r.Bar
		}
		if r.Bar > maxBar {
			maxBar = r.Bar
		}
	}
	return map[string]interface{}{
		"records":    len(re.records),
		"symbols":    symbols,
		"bar_range":  [2]int64{minBar, maxBar},
		"first_time": re.records[0].Timestamp,
		"last_time":  re.records[len(re.records)-1].Timestamp,
	}
}
