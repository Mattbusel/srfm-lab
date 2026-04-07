package analytics

import (
	"math"
	"sort"
	"sync"
	"time"
)

// SpreadSnapshot captures a single point-in-time view of the bid-ask spread
// and order-book depth for one symbol.
type SpreadSnapshot struct {
	Symbol    string    `json:"symbol"`
	Bid       float64   `json:"bid"`
	Ask       float64   `json:"ask"`
	SpreadBps float64   `json:"spread_bps"` // (ask-bid)/mid * 10000
	DepthBid  float64   `json:"depth_bid"`  // total bid-side quantity at best level
	DepthAsk  float64   `json:"depth_ask"`  // total ask-side quantity at best level
	Timestamp time.Time `json:"timestamp"`
}

// spreadHistory maintains a fixed-size circular buffer of spread readings.
const spreadHistorySize = 100

type spreadHistory struct {
	mu   sync.RWMutex
	buf  [spreadHistorySize]SpreadSnapshot
	head int  // index of next write slot
	size int  // number of valid entries (0..spreadHistorySize)

	// EWMA of spread for the 24h-mean estimate.
	// alpha = 2/(N+1) with N=100 gives approximately 100-tick horizon.
	spreadEWMA  float64
	spreadAlpha float64
	initialized bool
}

func newSpreadHistory() *spreadHistory {
	return &spreadHistory{
		spreadAlpha: 2.0 / (100.0 + 1.0),
	}
}

// push records a new snapshot into the circular buffer.
func (h *spreadHistory) push(snap SpreadSnapshot) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.buf[h.head] = snap
	h.head = (h.head + 1) % spreadHistorySize
	if h.size < spreadHistorySize {
		h.size++
	}

	if !h.initialized {
		h.spreadEWMA = snap.SpreadBps
		h.initialized = true
	} else {
		h.spreadEWMA = snap.SpreadBps*h.spreadAlpha + h.spreadEWMA*(1-h.spreadAlpha)
	}
}

// latest returns the most recently pushed snapshot.
// Returns (SpreadSnapshot{}, false) if buffer is empty.
func (h *spreadHistory) latest() (SpreadSnapshot, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.size == 0 {
		return SpreadSnapshot{}, false
	}
	idx := (h.head - 1 + spreadHistorySize) % spreadHistorySize
	return h.buf[idx], true
}

// currentSpreadBps returns the most recent spread in bps.
func (h *spreadHistory) currentSpreadBps() (float64, bool) {
	snap, ok := h.latest()
	if !ok {
		return 0, false
	}
	return snap.SpreadBps, true
}

// meanSpreadBps returns the EWMA-based spread estimate (approximates 24h mean).
func (h *spreadHistory) meanSpreadBps() float64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.spreadEWMA
}

// copyEntries returns a slice of all current history entries in insertion order,
// oldest first. Caller should not hold the mutex.
func (h *spreadHistory) copyEntries() []SpreadSnapshot {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.size == 0 {
		return nil
	}

	out := make([]SpreadSnapshot, h.size)
	// oldest entry is at head when buffer is full, or 0 when not yet full.
	var startIdx int
	if h.size < spreadHistorySize {
		startIdx = 0
	} else {
		startIdx = h.head // head points at the oldest entry when full
	}

	for i := 0; i < h.size; i++ {
		out[i] = h.buf[(startIdx+i)%spreadHistorySize]
	}
	return out
}

// percentile computes the p-th percentile (0-100) of historical spread_bps values.
// Uses linear interpolation. Returns 0 if no data.
func (h *spreadHistory) percentile(p float64) float64 {
	entries := h.copyEntries()
	if len(entries) == 0 {
		return 0
	}

	spreads := make([]float64, len(entries))
	for i, e := range entries {
		spreads[i] = e.SpreadBps
	}
	sort.Float64s(spreads)

	n := float64(len(spreads))
	if p <= 0 {
		return spreads[0]
	}
	if p >= 100 {
		return spreads[len(spreads)-1]
	}

	// Linear interpolation between indices.
	rank := (p / 100.0) * (n - 1)
	lo := int(math.Floor(rank))
	hi := lo + 1
	if hi >= len(spreads) {
		return spreads[lo]
	}
	frac := rank - float64(lo)
	return spreads[lo]*(1-frac) + spreads[hi]*frac
}

// liquidityScore computes a 0-1 score. Higher is better (tighter + deeper).
// score = depthComponent * spreadComponent
// spreadComponent: 1 if spread == 0, 0 if spread >= 50 bps (cap at 50)
// depthComponent: based on combined bid+ask depth normalised to a 100-unit baseline
func (h *spreadHistory) liquidityScore() float64 {
	snap, ok := h.latest()
	if !ok {
		return 0
	}

	// Spread component: linearly decays from 1 (0 bps) to 0 (50 bps)
	const maxSpread = 50.0
	spreadComp := 1.0 - math.Min(snap.SpreadBps/maxSpread, 1.0)

	// Depth component: tanh-normalised so 100 units -> 0.76, 200 -> 0.96
	totalDepth := snap.DepthBid + snap.DepthAsk
	depthComp := math.Tanh(totalDepth / 100.0)

	return spreadComp * depthComp
}

// SpreadMonitor tracks bid-ask spreads and liquidity conditions across symbols.
// It is safe for concurrent use.
type SpreadMonitor struct {
	histories sync.Map // symbol -> *spreadHistory
}

// NewSpreadMonitor creates a ready-to-use SpreadMonitor.
func NewSpreadMonitor() *SpreadMonitor {
	return &SpreadMonitor{}
}

// OnBookUpdate records a new L2 book top-of-book update for a symbol.
// bid/ask are the best bid and ask prices; depthBid/depthAsk are the
// total quantities at those price levels.
func (m *SpreadMonitor) OnBookUpdate(symbol string, bid, ask, depthBid, depthAsk float64) {
	if bid <= 0 || ask <= 0 || ask < bid {
		return
	}

	mid := (bid + ask) / 2.0
	var spreadBps float64
	if mid > 0 {
		spreadBps = (ask-bid) / mid * 10_000
	}

	snap := SpreadSnapshot{
		Symbol:    symbol,
		Bid:       bid,
		Ask:       ask,
		SpreadBps: spreadBps,
		DepthBid:  depthBid,
		DepthAsk:  depthAsk,
		Timestamp: time.Now().UTC(),
	}

	val, _ := m.histories.LoadOrStore(symbol, newSpreadHistory())
	hist := val.(*spreadHistory)
	hist.push(snap)
}

// OnSnapshot is a convenience wrapper that accepts a pre-built SpreadSnapshot.
func (m *SpreadMonitor) OnSnapshot(snap SpreadSnapshot) {
	val, _ := m.histories.LoadOrStore(snap.Symbol, newSpreadHistory())
	hist := val.(*spreadHistory)
	hist.push(snap)
}

// GetSpread returns the current spread in basis points for a symbol.
// Returns (0, false) if no data is available.
func (m *SpreadMonitor) GetSpread(symbol string) (float64, bool) {
	val, ok := m.histories.Load(symbol)
	if !ok {
		return 0, false
	}
	hist := val.(*spreadHistory)
	return hist.currentSpreadBps()
}

// GetSpreadPercentile returns the p-th percentile (0-100) of the rolling
// spread history for a symbol. Returns 0 if no data.
func (m *SpreadMonitor) GetSpreadPercentile(symbol string, pct float64) float64 {
	val, ok := m.histories.Load(symbol)
	if !ok {
		return 0
	}
	hist := val.(*spreadHistory)
	return hist.percentile(pct)
}

// IsWide returns true if the current spread is greater than 2x the
// rolling EWMA-based mean spread for that symbol.
func (m *SpreadMonitor) IsWide(symbol string) bool {
	val, ok := m.histories.Load(symbol)
	if !ok {
		return false
	}
	hist := val.(*spreadHistory)
	current, ok := hist.currentSpreadBps()
	if !ok {
		return false
	}
	mean := hist.meanSpreadBps()
	if mean <= 0 {
		return false
	}
	return current > 2*mean
}

// GetLiquidityScore returns a [0,1] score combining spread tightness and book depth.
// 1.0 = tight spread with deep book; 0.0 = no data or very wide / shallow.
func (m *SpreadMonitor) GetLiquidityScore(symbol string) float64 {
	val, ok := m.histories.Load(symbol)
	if !ok {
		return 0
	}
	hist := val.(*spreadHistory)
	return hist.liquidityScore()
}

// GetLatestSnapshot returns the most recent SpreadSnapshot for a symbol.
// Returns (SpreadSnapshot{}, false) if no data exists.
func (m *SpreadMonitor) GetLatestSnapshot(symbol string) (SpreadSnapshot, bool) {
	val, ok := m.histories.Load(symbol)
	if !ok {
		return SpreadSnapshot{}, false
	}
	hist := val.(*spreadHistory)
	return hist.latest()
}

// Symbols returns the list of all symbols currently tracked.
func (m *SpreadMonitor) Symbols() []string {
	var out []string
	m.histories.Range(func(k, _ interface{}) bool {
		out = append(out, k.(string))
		return true
	})
	return out
}
