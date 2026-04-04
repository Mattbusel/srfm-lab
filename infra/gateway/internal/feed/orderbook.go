package feed

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// PriceLevel represents a single bid or ask level in an order book.
type PriceLevel struct {
	Price float64
	Size  float64
}

// OrderBook is an in-memory bid/ask order book for a single symbol.
// It maintains bids sorted descending and asks sorted ascending.
type OrderBook struct {
	mu     sync.RWMutex
	Symbol string
	Bids   []PriceLevel // descending by price
	Asks   []PriceLevel // ascending by price
	SeqNum int64
	AsOf   time.Time
}

// NewOrderBook creates an empty OrderBook.
func NewOrderBook(symbol string) *OrderBook {
	return &OrderBook{Symbol: symbol}
}

// UpdateBid sets the size at a bid price level (removes if size == 0).
func (ob *OrderBook) UpdateBid(price, size float64) {
	ob.mu.Lock()
	defer ob.mu.Unlock()
	ob.Bids = updateLevel(ob.Bids, price, size, false)
	ob.AsOf = time.Now()
}

// UpdateAsk sets the size at an ask price level (removes if size == 0).
func (ob *OrderBook) UpdateAsk(price, size float64) {
	ob.mu.Lock()
	defer ob.mu.Unlock()
	ob.Asks = updateLevel(ob.Asks, price, size, true)
	ob.AsOf = time.Now()
}

// BestBid returns the highest bid price and size, or (0,0) if empty.
func (ob *OrderBook) BestBid() (price, size float64) {
	ob.mu.RLock()
	defer ob.mu.RUnlock()
	if len(ob.Bids) == 0 {
		return 0, 0
	}
	return ob.Bids[0].Price, ob.Bids[0].Size
}

// BestAsk returns the lowest ask price and size, or (0,0) if empty.
func (ob *OrderBook) BestAsk() (price, size float64) {
	ob.mu.RLock()
	defer ob.mu.RUnlock()
	if len(ob.Asks) == 0 {
		return 0, 0
	}
	return ob.Asks[0].Price, ob.Asks[0].Size
}

// Spread returns the bid-ask spread.
func (ob *OrderBook) Spread() float64 {
	bp, _ := ob.BestBid()
	ap, _ := ob.BestAsk()
	if bp <= 0 || ap <= 0 {
		return 0
	}
	return ap - bp
}

// SpreadBps returns the spread in basis points.
func (ob *OrderBook) SpreadBps() float64 {
	bp, _ := ob.BestBid()
	ap, _ := ob.BestAsk()
	if bp <= 0 || ap <= 0 {
		return 0
	}
	mid := (bp + ap) / 2
	if mid == 0 {
		return 0
	}
	return (ap - bp) / mid * 10000
}

// MidPrice returns (bid+ask)/2.
func (ob *OrderBook) MidPrice() float64 {
	bp, _ := ob.BestBid()
	ap, _ := ob.BestAsk()
	if bp <= 0 || ap <= 0 {
		return 0
	}
	return (bp + ap) / 2
}

// VWAP computes the volume-weighted average price for the top n levels.
func (ob *OrderBook) VWAP(n int) (bidVWAP, askVWAP float64) {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	bidVWAP = vwap(ob.Bids, n)
	askVWAP = vwap(ob.Asks, n)
	return
}

// ToQuote converts the top-of-book to a Quote.
func (ob *OrderBook) ToQuote() Quote {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	var bp, bs, ap, as float64
	if len(ob.Bids) > 0 {
		bp = ob.Bids[0].Price
		bs = ob.Bids[0].Size
	}
	if len(ob.Asks) > 0 {
		ap = ob.Asks[0].Price
		as = ob.Asks[0].Size
	}
	return Quote{
		Symbol:    ob.Symbol,
		Timestamp: ob.AsOf,
		BidPrice:  bp,
		BidSize:   bs,
		AskPrice:  ap,
		AskSize:   as,
		Source:    "orderbook",
	}
}

// Depth returns up to n levels on each side.
func (ob *OrderBook) Depth(n int) (bids, asks []PriceLevel) {
	ob.mu.RLock()
	defer ob.mu.RUnlock()
	if n > len(ob.Bids) {
		n = len(ob.Bids)
	}
	bids = make([]PriceLevel, n)
	copy(bids, ob.Bids[:n])

	askN := n
	if askN > len(ob.Asks) {
		askN = len(ob.Asks)
	}
	asks = make([]PriceLevel, askN)
	copy(asks, ob.Asks[:askN])
	return
}

// Clear resets the order book.
func (ob *OrderBook) Clear() {
	ob.mu.Lock()
	ob.Bids = nil
	ob.Asks = nil
	ob.mu.Unlock()
}

// ImbalanceRatio returns (bid_vol - ask_vol) / (bid_vol + ask_vol) for top n levels.
// Ranges from -1 (pure ask pressure) to +1 (pure bid pressure).
func (ob *OrderBook) ImbalanceRatio(n int) float64 {
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	bidVol := totalVol(ob.Bids, n)
	askVol := totalVol(ob.Asks, n)
	total := bidVol + askVol
	if total == 0 {
		return 0
	}
	return (bidVol - askVol) / total
}

// String returns a human-readable snapshot of the book.
func (ob *OrderBook) String() string {
	ob.mu.RLock()
	defer ob.mu.RUnlock()
	n := 5
	bids := ob.Bids
	asks := ob.Asks
	if len(bids) > n {
		bids = bids[:n]
	}
	if len(asks) > n {
		asks = asks[:n]
	}
	s := fmt.Sprintf("OrderBook[%s] spread=%.4f\n", ob.Symbol, ob.Spread())
	s += "  ASKS:\n"
	for i := len(asks) - 1; i >= 0; i-- {
		s += fmt.Sprintf("    %.4f x %.2f\n", asks[i].Price, asks[i].Size)
	}
	s += "  BIDS:\n"
	for _, l := range bids {
		s += fmt.Sprintf("    %.4f x %.2f\n", l.Price, l.Size)
	}
	return s
}

// ---- helpers ----

// updateLevel inserts or removes a price level in a sorted slice.
// ascending=true sorts ascending (for asks), false=descending (for bids).
func updateLevel(levels []PriceLevel, price, size float64, ascending bool) []PriceLevel {
	// Find existing level by price.
	for i, l := range levels {
		if math.Abs(l.Price-price) < 1e-12 {
			if size == 0 {
				return append(levels[:i], levels[i+1:]...)
			}
			levels[i].Size = size
			return levels
		}
	}
	if size == 0 {
		return levels
	}
	levels = append(levels, PriceLevel{Price: price, Size: size})
	if ascending {
		sort.Slice(levels, func(i, j int) bool { return levels[i].Price < levels[j].Price })
	} else {
		sort.Slice(levels, func(i, j int) bool { return levels[i].Price > levels[j].Price })
	}
	return levels
}

func vwap(levels []PriceLevel, n int) float64 {
	if n > len(levels) {
		n = len(levels)
	}
	var pv, v float64
	for i := 0; i < n; i++ {
		pv += levels[i].Price * levels[i].Size
		v += levels[i].Size
	}
	if v == 0 {
		return 0
	}
	return pv / v
}

func totalVol(levels []PriceLevel, n int) float64 {
	if n > len(levels) {
		n = len(levels)
	}
	var vol float64
	for i := 0; i < n; i++ {
		vol += levels[i].Size
	}
	return vol
}

// OrderBookRegistry manages OrderBooks for multiple symbols.
type OrderBookRegistry struct {
	mu    sync.RWMutex
	books map[string]*OrderBook
}

// NewOrderBookRegistry creates an OrderBookRegistry.
func NewOrderBookRegistry() *OrderBookRegistry {
	return &OrderBookRegistry{books: make(map[string]*OrderBook)}
}

// GetOrCreate returns the OrderBook for symbol, creating it if needed.
func (r *OrderBookRegistry) GetOrCreate(symbol string) *OrderBook {
	r.mu.RLock()
	ob, ok := r.books[symbol]
	r.mu.RUnlock()
	if ok {
		return ob
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	ob, ok = r.books[symbol]
	if !ok {
		ob = NewOrderBook(symbol)
		r.books[symbol] = ob
	}
	return ob
}

// Get returns the OrderBook for symbol, or nil.
func (r *OrderBookRegistry) Get(symbol string) *OrderBook {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.books[symbol]
}

// AllQuotes returns top-of-book quotes for all symbols.
func (r *OrderBookRegistry) AllQuotes() []Quote {
	r.mu.RLock()
	syms := make([]string, 0, len(r.books))
	for sym := range r.books {
		syms = append(syms, sym)
	}
	r.mu.RUnlock()

	quotes := make([]Quote, 0, len(syms))
	for _, sym := range syms {
		ob := r.Get(sym)
		if ob != nil {
			q := ob.ToQuote()
			if q.BidPrice > 0 || q.AskPrice > 0 {
				quotes = append(quotes, q)
			}
		}
	}
	return quotes
}
