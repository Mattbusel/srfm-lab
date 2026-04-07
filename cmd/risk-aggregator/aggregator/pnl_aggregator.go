// Package aggregator contains the core computation engines for the risk-aggregator service.
// This file implements real-time P&L computation with FIFO cost basis tracking.
package aggregator

import (
	"sync"
	"time"
)

// fill represents a single purchase lot stored in the FIFO queue.
type fill struct {
	qty       float64 // always positive
	costBasis float64 // price paid per unit
}

// symbolState holds all accounting state for one symbol.
type symbolState struct {
	fifo         []fill  // FIFO queue of open longs
	realizedPnL  float64 // cumulative realized P&L
	latestPrice  float64 // last known mark price
	currentQty   float64 // net position (can be negative for shorts)
}

// PnLAttribution breaks down total P&L by symbol and sector.
type PnLAttribution struct {
	BySymbol map[string]float64 `json:"by_symbol"`
	BySector map[string]float64 `json:"by_sector"`
	Total    float64            `json:"total"`
}

// PnLAggregator computes real-time unrealized and realized P&L with FIFO
// cost basis tracking.  All methods are safe for concurrent use.
type PnLAggregator struct {
	mu       sync.RWMutex
	symbols  map[string]*symbolState
	sectorOf map[string]string // symbol -> sector, populated via SetSectorMap

	// ytdBase is the sum of realized P&L carried from before today (UTC).
	// It is set when the daily reset happens so YTDPnL can still be returned.
	ytdBase float64

	// dailyStartUnrealized is the unrealized P&L snapshot taken at each
	// midnight reset so that DailyPnL accounts for overnight positions.
	dailyStartUnrealized float64

	lastResetDate time.Time // UTC calendar date of last midnight reset
}

// NewPnLAggregator constructs a zeroed aggregator.
func NewPnLAggregator() *PnLAggregator {
	now := time.Now().UTC()
	return &PnLAggregator{
		symbols:       make(map[string]*symbolState),
		sectorOf:      make(map[string]string),
		lastResetDate: time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC),
	}
}

// SetSectorMap registers a symbol-to-sector mapping used by Attribution.
// Can be called at any time; it replaces the previous map atomically.
func (a *PnLAggregator) SetSectorMap(m map[string]string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sectorOf = make(map[string]string, len(m))
	for k, v := range m {
		a.sectorOf[k] = v
	}
}

// Update records a trade.  qty > 0 is a buy, qty < 0 is a sell.
// price is the execution price per unit.
func (a *PnLAggregator) Update(symbol string, qty float64, price float64) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.maybeResetDaily()

	st := a.getOrCreate(symbol)
	st.latestPrice = price

	if qty > 0 {
		// Buy -- add to FIFO queue.
		st.fifo = append(st.fifo, fill{qty: qty, costBasis: price})
		st.currentQty += qty
		return
	}

	// Sell -- match against FIFO queue and book realized P&L.
	remaining := -qty // positive quantity to consume
	for remaining > 0 && len(st.fifo) > 0 {
		lot := &st.fifo[0]
		if lot.qty <= remaining {
			// Consume the entire lot.
			st.realizedPnL += lot.qty * (price - lot.costBasis)
			remaining -= lot.qty
			st.fifo = st.fifo[1:]
		} else {
			// Partially consume the lot.
			st.realizedPnL += remaining * (price - lot.costBasis)
			lot.qty -= remaining
			remaining = 0
		}
	}
	st.currentQty += qty // qty is negative here
}

// MarkPrice updates the latest mark price for a symbol without recording a trade.
// This allows unrealized P&L to reflect current market prices between trades.
func (a *PnLAggregator) MarkPrice(symbol string, price float64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	st := a.getOrCreate(symbol)
	st.latestPrice = price
}

// UnrealizedPnL returns the current unrealized P&L for each symbol.
func (a *PnLAggregator) UnrealizedPnL() map[string]float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()

	out := make(map[string]float64, len(a.symbols))
	for sym, st := range a.symbols {
		out[sym] = a.unrealizedForState(st)
	}
	return out
}

// RealizedPnL returns the cumulative realized P&L for each symbol.
func (a *PnLAggregator) RealizedPnL() map[string]float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()

	out := make(map[string]float64, len(a.symbols))
	for sym, st := range a.symbols {
		out[sym] = st.realizedPnL
	}
	return out
}

// DailyPnL returns total P&L (realized + unrealized) since midnight UTC.
// It is reset to zero at each midnight; the reset happens lazily on the next
// call that observes a new UTC calendar date.
func (a *PnLAggregator) DailyPnL() float64 {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.maybeResetDaily()

	totalRealized := 0.0
	totalUnrealized := 0.0
	for _, st := range a.symbols {
		totalRealized += st.realizedPnL
		totalUnrealized += a.unrealizedForState(st)
	}

	// DailyPnL = (current total) - (total at reset time)
	// At reset time ytdBase was set to previous total realized, and
	// dailyStartUnrealized was snapshotted.
	dailyRealized := totalRealized - a.ytdBase
	dailyUnrealized := totalUnrealized - a.dailyStartUnrealized
	return dailyRealized + dailyUnrealized
}

// YTDPnL returns total P&L since the beginning of the year.
// It accumulates realized P&L across daily resets.
func (a *PnLAggregator) YTDPnL() float64 {
	a.mu.RLock()
	defer a.mu.RUnlock()

	total := 0.0
	for _, st := range a.symbols {
		total += st.realizedPnL + a.unrealizedForState(st)
	}
	return total
}

// Attribution computes P&L decomposition by symbol and sector.
func (a *PnLAggregator) Attribution() PnLAttribution {
	a.mu.RLock()
	defer a.mu.RUnlock()

	attr := PnLAttribution{
		BySymbol: make(map[string]float64),
		BySector: make(map[string]float64),
	}

	for sym, st := range a.symbols {
		symPnL := st.realizedPnL + a.unrealizedForState(st)
		attr.BySymbol[sym] = symPnL
		attr.Total += symPnL

		sector := a.sectorOf[sym]
		if sector == "" {
			sector = "unknown"
		}
		attr.BySector[sector] += symPnL
	}

	return attr
}

// -- internal helpers --

// getOrCreate returns the symbolState for sym, creating it if necessary.
// Caller must hold a.mu at write level.
func (a *PnLAggregator) getOrCreate(symbol string) *symbolState {
	if st, ok := a.symbols[symbol]; ok {
		return st
	}
	st := &symbolState{}
	a.symbols[symbol] = st
	return st
}

// unrealizedForState computes unrealized P&L from the FIFO queue.
// The mark price is latestPrice; if no price is set, unrealized is zero.
// Caller must hold at least a read lock.
func (a *PnLAggregator) unrealizedForState(st *symbolState) float64 {
	if st.latestPrice == 0 {
		return 0
	}
	total := 0.0
	for _, lot := range st.fifo {
		total += lot.qty * (st.latestPrice - lot.costBasis)
	}
	return total
}

// maybeResetDaily checks if the UTC calendar date has advanced and, if so,
// snapshots the current totals so DailyPnL restarts from zero.
// Caller must hold a.mu at write level.
func (a *PnLAggregator) maybeResetDaily() {
	now := time.Now().UTC()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	if !today.After(a.lastResetDate) {
		return
	}

	// Snapshot totals to serve as the new baseline.
	totalRealized := 0.0
	totalUnrealized := 0.0
	for _, st := range a.symbols {
		totalRealized += st.realizedPnL
		totalUnrealized += a.unrealizedForState(st)
	}
	a.ytdBase = totalRealized
	a.dailyStartUnrealized = totalUnrealized
	a.lastResetDate = today
}
