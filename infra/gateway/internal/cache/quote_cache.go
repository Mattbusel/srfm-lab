package cache

import (
	"sync"
	"time"

	"github.com/srfm/gateway/internal/feed"
)

// QuoteCache stores the latest quote and best-execution data per symbol.
// It is goroutine-safe.
type QuoteCache struct {
	mu     sync.RWMutex
	quotes map[string]TimestampedQuote
	stats  map[string]*quoteStats
}

// TimestampedQuote wraps a Quote with a received-at timestamp.
type TimestampedQuote struct {
	feed.Quote
	ReceivedAt time.Time
}

// quoteStats tracks per-symbol quote statistics.
type quoteStats struct {
	count       int64
	lastBidAsk  float64 // spread in bps
	sumSpread   float64
	minSpread   float64
	maxSpread   float64
}

// NewQuoteCache creates a QuoteCache.
func NewQuoteCache() *QuoteCache {
	return &QuoteCache{
		quotes: make(map[string]TimestampedQuote),
		stats:  make(map[string]*quoteStats),
	}
}

// Update stores the latest quote for a symbol.
func (qc *QuoteCache) Update(q feed.Quote) {
	now := time.Now()
	tq := TimestampedQuote{Quote: q, ReceivedAt: now}

	spread := 0.0
	if q.BidPrice > 0 {
		mid := (q.BidPrice + q.AskPrice) / 2
		if mid > 0 {
			spread = (q.AskPrice - q.BidPrice) / mid * 10000 // bps
		}
	}

	qc.mu.Lock()
	qc.quotes[q.Symbol] = tq

	st, ok := qc.stats[q.Symbol]
	if !ok {
		st = &quoteStats{minSpread: 1e18}
		qc.stats[q.Symbol] = st
	}
	st.count++
	st.lastBidAsk = spread
	st.sumSpread += spread
	if spread < st.minSpread {
		st.minSpread = spread
	}
	if spread > st.maxSpread {
		st.maxSpread = spread
	}
	qc.mu.Unlock()
}

// Latest returns the most recent quote for a symbol, or nil.
func (qc *QuoteCache) Latest(symbol string) *TimestampedQuote {
	qc.mu.RLock()
	q, ok := qc.quotes[symbol]
	qc.mu.RUnlock()
	if !ok {
		return nil
	}
	return &q
}

// Symbols returns all symbols with a cached quote.
func (qc *QuoteCache) Symbols() []string {
	qc.mu.RLock()
	defer qc.mu.RUnlock()
	out := make([]string, 0, len(qc.quotes))
	for sym := range qc.quotes {
		out = append(out, sym)
	}
	return out
}

// SpreadStats returns bid-ask spread statistics for a symbol.
func (qc *QuoteCache) SpreadStats(symbol string) (min, max, avg, last float64) {
	qc.mu.RLock()
	st, ok := qc.stats[symbol]
	qc.mu.RUnlock()
	if !ok {
		return 0, 0, 0, 0
	}
	avg = 0
	if st.count > 0 {
		avg = st.sumSpread / float64(st.count)
	}
	return st.minSpread, st.maxSpread, avg, st.lastBidAsk
}

// MidPrice returns the midpoint price for a symbol (average of bid/ask).
// Falls back to the last quote price if bid or ask is unavailable.
func (qc *QuoteCache) MidPrice(symbol string) (float64, bool) {
	qc.mu.RLock()
	q, ok := qc.quotes[symbol]
	qc.mu.RUnlock()
	if !ok {
		return 0, false
	}
	if q.BidPrice > 0 && q.AskPrice > 0 {
		return (q.BidPrice + q.AskPrice) / 2, true
	}
	return q.AskPrice, true
}

// Stale returns true if the cached quote is older than the given duration.
func (qc *QuoteCache) Stale(symbol string, maxAge time.Duration) bool {
	qc.mu.RLock()
	q, ok := qc.quotes[symbol]
	qc.mu.RUnlock()
	if !ok {
		return true
	}
	return time.Since(q.ReceivedAt) > maxAge
}
