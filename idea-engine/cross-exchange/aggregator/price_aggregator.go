// Package aggregator maintains the canonical price state across all exchanges.
package aggregator

import (
	"sync"
	"time"
)

// ExchangePrice holds a single price observation from one exchange.
type ExchangePrice struct {
	Exchange  string
	Symbol    string
	Bid       float64
	Ask       float64
	Mid       float64
	Volume    float64 // 24h volume; 0 if unavailable
	Timestamp time.Time
}

// ConsensusPrices holds aggregated data for one symbol.
type ConsensusPrices struct {
	Symbol       string
	ConsensusMid float64
	Prices       []ExchangePrice
	ComputedAt   time.Time
}

// PriceAggregator stores the latest price from every exchange and computes
// consensus values. It is safe for concurrent use.
type PriceAggregator struct {
	mu     sync.RWMutex
	latest map[string]map[string]ExchangePrice // symbol -> exchange -> price
	staleTTL time.Duration
}

// New creates a PriceAggregator. staleTTL is how old a price can be before it
// is excluded from consensus (typically 60 seconds).
func New(staleTTL time.Duration) *PriceAggregator {
	return &PriceAggregator{
		latest:   make(map[string]map[string]ExchangePrice),
		staleTTL: staleTTL,
	}
}

// Update stores the latest price for the given exchange and symbol.
func (a *PriceAggregator) Update(p ExchangePrice) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.latest[p.Symbol] == nil {
		a.latest[p.Symbol] = make(map[string]ExchangePrice)
	}
	a.latest[p.Symbol][p.Exchange] = p
}

// Consensus returns the aggregated view for every known symbol.
func (a *PriceAggregator) Consensus() []ConsensusPrices {
	a.mu.RLock()
	defer a.mu.RUnlock()
	now := time.Now()
	out := make([]ConsensusPrices, 0, len(a.latest))
	for sym, byExchange := range a.latest {
		cp := a.computeConsensus(sym, byExchange, now)
		out = append(out, cp)
	}
	return out
}

// ConsensusForSymbol returns the aggregated view for a single symbol.
func (a *PriceAggregator) ConsensusForSymbol(symbol string) (ConsensusPrices, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	byExchange, ok := a.latest[symbol]
	if !ok {
		return ConsensusPrices{}, false
	}
	return a.computeConsensus(symbol, byExchange, time.Now()), true
}

// AllPrices returns every stored ExchangePrice for every symbol.
func (a *PriceAggregator) AllPrices() map[string][]ExchangePrice {
	a.mu.RLock()
	defer a.mu.RUnlock()
	out := make(map[string][]ExchangePrice, len(a.latest))
	for sym, byExchange := range a.latest {
		prices := make([]ExchangePrice, 0, len(byExchange))
		for _, p := range byExchange {
			prices = append(prices, p)
		}
		out[sym] = prices
	}
	return out
}

// computeConsensus calculates the consensus mid price using volume-weighted
// mean when volumes are available, falling back to simple mean. Stale prices
// (older than staleTTL) are excluded.
func (a *PriceAggregator) computeConsensus(symbol string, byExchange map[string]ExchangePrice, now time.Time) ConsensusPrices {
	fresh := make([]ExchangePrice, 0, len(byExchange))
	for _, p := range byExchange {
		if now.Sub(p.Timestamp) <= a.staleTTL {
			fresh = append(fresh, p)
		}
	}

	var consensusMid float64
	if len(fresh) > 0 {
		totalVolume := 0.0
		for _, p := range fresh {
			totalVolume += p.Volume
		}
		if totalVolume > 0 {
			sum := 0.0
			for _, p := range fresh {
				sum += p.Mid * p.Volume
			}
			consensusMid = sum / totalVolume
		} else {
			sum := 0.0
			for _, p := range fresh {
				sum += p.Mid
			}
			consensusMid = sum / float64(len(fresh))
		}
	}

	allPrices := make([]ExchangePrice, 0, len(byExchange))
	for _, p := range byExchange {
		allPrices = append(allPrices, p)
	}

	return ConsensusPrices{
		Symbol:       symbol,
		ConsensusMid: consensusMid,
		Prices:       allPrices,
		ComputedAt:   now,
	}
}
