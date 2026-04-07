// Package storage provides storage and caching utilities for market data.
package storage

import (
	"sync"
	"time"
	"unsafe"

	"srfm/market-data/aggregator"
)

// barRingBuffer is a fixed-capacity circular buffer for aggregator.BarEvent values.
// It is not safe for concurrent use; callers must hold an external lock.
type barRingBuffer struct {
	buf      []aggregator.BarEvent
	capacity int
	head     int // index of next write slot
	size     int // number of valid entries
}

func newBarRingBuffer(capacity int) *barRingBuffer {
	return &barRingBuffer{
		buf:      make([]aggregator.BarEvent, capacity),
		capacity: capacity,
	}
}

// push inserts a bar into the ring buffer, overwriting the oldest entry when full.
func (r *barRingBuffer) push(bar aggregator.BarEvent) {
	r.buf[r.head] = bar
	r.head = (r.head + 1) % r.capacity
	if r.size < r.capacity {
		r.size++
	}
}

// lastN returns the most recent n bars, oldest first.
// If n <= 0 or n >= size, all available entries are returned.
func (r *barRingBuffer) lastN(n int) []aggregator.BarEvent {
	if r.size == 0 {
		return nil
	}
	if n <= 0 || n > r.size {
		n = r.size
	}
	out := make([]aggregator.BarEvent, n)
	// The oldest of the n bars we want is at position (head - n) in the ring.
	startOffset := r.size - n
	var absStart int
	if r.size < r.capacity {
		// Buffer not yet wrapped; entries begin at 0.
		absStart = startOffset
	} else {
		// Buffer is full; head points at the oldest entry.
		absStart = (r.head + startOffset) % r.capacity
	}
	for i := 0; i < n; i++ {
		out[i] = r.buf[(absStart+i)%r.capacity]
	}
	return out
}

// since returns all bars whose Timestamp is >= the given time, oldest first.
func (r *barRingBuffer) since(t time.Time) []aggregator.BarEvent {
	if r.size == 0 {
		return nil
	}

	// Walk from oldest to newest
	var absStart int
	if r.size < r.capacity {
		absStart = 0
	} else {
		absStart = r.head
	}

	var out []aggregator.BarEvent
	for i := 0; i < r.size; i++ {
		bar := r.buf[(absStart+i)%r.capacity]
		if !bar.Timestamp.Before(t) {
			out = append(out, bar)
		}
	}
	return out
}

// estimateBytes returns the approximate number of bytes this buffer occupies.
func (r *barRingBuffer) estimateBytes() int64 {
	// Each BarEvent is two strings + 5 float64s + time.Time + bool.
	// Measure the struct size via unsafe.Sizeof and add string overhead.
	barSize := int64(unsafe.Sizeof(aggregator.BarEvent{}))
	// Add average string length estimate: symbol ~4, timeframe ~3
	barSize += 7
	return barSize * int64(r.capacity)
}

// tsKey uniquely identifies a (symbol, timeframe) series.
type tsKey struct {
	symbol    string
	timeframe string
}

// seriesEntry holds the ring buffer and its RWMutex for one (symbol, timeframe).
type seriesEntry struct {
	mu  sync.RWMutex
	buf *barRingBuffer
}

// TimeSeriesCache is a thread-safe, per-symbol per-timeframe circular buffer
// storing the last 500 bars for each series. Older bars are automatically
// evicted when the capacity is reached.
type TimeSeriesCache struct {
	mu       sync.RWMutex
	series   map[tsKey]*seriesEntry
	capacity int // bars per series
}

// defaultCapacity is the default number of bars kept per (symbol, timeframe).
const defaultCapacity = 500

// NewTimeSeriesCache creates a TimeSeriesCache with the default capacity (500 bars).
func NewTimeSeriesCache() *TimeSeriesCache {
	return NewTimeSeriesCacheWithCapacity(defaultCapacity)
}

// NewTimeSeriesCacheWithCapacity creates a TimeSeriesCache with a custom per-series capacity.
func NewTimeSeriesCacheWithCapacity(capacity int) *TimeSeriesCache {
	if capacity <= 0 {
		capacity = defaultCapacity
	}
	return &TimeSeriesCache{
		series:   make(map[tsKey]*seriesEntry),
		capacity: capacity,
	}
}

// getOrCreate returns the seriesEntry for (symbol, timeframe), creating it if needed.
func (c *TimeSeriesCache) getOrCreate(symbol, timeframe string) *seriesEntry {
	k := tsKey{symbol: symbol, timeframe: timeframe}

	c.mu.RLock()
	e, ok := c.series[k]
	c.mu.RUnlock()
	if ok {
		return e
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	// Double-check after acquiring write lock.
	if e, ok = c.series[k]; ok {
		return e
	}
	e = &seriesEntry{buf: newBarRingBuffer(c.capacity)}
	c.series[k] = e
	return e
}

// Store writes a bar into the cache for the given (symbol, timeframe).
// If the buffer is at capacity, the oldest bar is overwritten.
func (c *TimeSeriesCache) Store(symbol, timeframe string, bar aggregator.BarEvent) {
	e := c.getOrCreate(symbol, timeframe)
	e.mu.Lock()
	e.buf.push(bar)
	e.mu.Unlock()
}

// StoreEvent is a convenience wrapper that reads symbol and timeframe from the BarEvent.
func (c *TimeSeriesCache) StoreEvent(bar aggregator.BarEvent) {
	c.Store(bar.Symbol, bar.Timeframe, bar)
}

// GetLast returns the last n bars for (symbol, timeframe), oldest first.
// Returns nil if no data exists for that key.
func (c *TimeSeriesCache) GetLast(symbol, timeframe string, n int) []aggregator.BarEvent {
	k := tsKey{symbol: symbol, timeframe: timeframe}
	c.mu.RLock()
	e, ok := c.series[k]
	c.mu.RUnlock()
	if !ok {
		return nil
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.buf.lastN(n)
}

// GetSince returns all bars for (symbol, timeframe) whose timestamp is >= since.
// Returns nil if no data exists.
func (c *TimeSeriesCache) GetSince(symbol, timeframe string, since time.Time) []aggregator.BarEvent {
	k := tsKey{symbol: symbol, timeframe: timeframe}
	c.mu.RLock()
	e, ok := c.series[k]
	c.mu.RUnlock()
	if !ok {
		return nil
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.buf.since(since)
}

// Latest returns the single most recent bar for (symbol, timeframe).
// Returns nil if no data exists.
func (c *TimeSeriesCache) Latest(symbol, timeframe string) *aggregator.BarEvent {
	bars := c.GetLast(symbol, timeframe, 1)
	if len(bars) == 0 {
		return nil
	}
	b := bars[len(bars)-1]
	return &b
}

// EstimateSize returns the approximate total memory usage in bytes across all series.
// This is a fast estimate based on buffer capacities, not an exact measurement.
func (c *TimeSeriesCache) EstimateSize() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var total int64
	for _, e := range c.series {
		e.mu.RLock()
		total += e.buf.estimateBytes()
		e.mu.RUnlock()
	}
	return total
}

// SeriesCount returns the number of distinct (symbol, timeframe) series stored.
func (c *TimeSeriesCache) SeriesCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.series)
}

// Symbols returns all symbols that have at least one series stored.
func (c *TimeSeriesCache) Symbols() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	seen := make(map[string]struct{})
	for k := range c.series {
		seen[k.symbol] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for sym := range seen {
		out = append(out, sym)
	}
	return out
}

// Timeframes returns all timeframes stored for a given symbol.
func (c *TimeSeriesCache) Timeframes(symbol string) []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	var out []string
	for k := range c.series {
		if k.symbol == symbol {
			out = append(out, k.timeframe)
		}
	}
	return out
}

// Clear removes all stored data. Primarily for testing.
func (c *TimeSeriesCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.series = make(map[tsKey]*seriesEntry)
}
