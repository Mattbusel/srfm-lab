package storage

import (
	"sync"

	"srfm/market-data/aggregator"
)

// cacheKey identifies a (symbol, timeframe) series.
type cacheKey struct {
	symbol    string
	timeframe string
}

// seriesCache holds up to capacity bars for one (symbol, timeframe) pair.
type seriesCache struct {
	bars     []aggregator.BarEvent
	capacity int
	hits     int64
	misses   int64
}

func newSeriesCache(capacity int) *seriesCache {
	return &seriesCache{
		bars:     make([]aggregator.BarEvent, 0, capacity),
		capacity: capacity,
	}
}

func (sc *seriesCache) put(evt aggregator.BarEvent) {
	// If bar with same timestamp exists, update it
	for i := len(sc.bars) - 1; i >= 0; i-- {
		if sc.bars[i].Timestamp.Equal(evt.Timestamp) {
			sc.bars[i] = evt
			return
		}
		if sc.bars[i].Timestamp.Before(evt.Timestamp) {
			break
		}
	}

	// Append and keep sorted (bars arrive roughly in order)
	sc.bars = append(sc.bars, evt)

	// Keep capacity: evict oldest
	if len(sc.bars) > sc.capacity {
		sc.bars = sc.bars[len(sc.bars)-sc.capacity:]
	}
}

func (sc *seriesCache) get(n int) []aggregator.BarEvent {
	if n <= 0 || n >= len(sc.bars) {
		out := make([]aggregator.BarEvent, len(sc.bars))
		copy(out, sc.bars)
		return out
	}
	out := make([]aggregator.BarEvent, n)
	copy(out, sc.bars[len(sc.bars)-n:])
	return out
}

func (sc *seriesCache) latest() *aggregator.BarEvent {
	if len(sc.bars) == 0 {
		return nil
	}
	b := sc.bars[len(sc.bars)-1]
	return &b
}

// BarCache is an in-memory LRU-style cache of recent bars per (symbol, timeframe).
type BarCache struct {
	mu       sync.RWMutex
	series   map[cacheKey]*seriesCache
	capacity int
	hits     int64
	misses   int64
}

// NewBarCache creates a BarCache with given per-series capacity.
func NewBarCache(capacity int) *BarCache {
	return &BarCache{
		series:   make(map[cacheKey]*seriesCache),
		capacity: capacity,
	}
}

// Put inserts or updates a bar in the cache.
func (c *BarCache) Put(evt aggregator.BarEvent) {
	key := cacheKey{symbol: evt.Symbol, timeframe: evt.Timeframe}
	c.mu.Lock()
	sc, ok := c.series[key]
	if !ok {
		sc = newSeriesCache(c.capacity)
		c.series[key] = sc
	}
	sc.put(evt)
	c.mu.Unlock()
}

// Get returns the last n bars for a (symbol, timeframe) pair.
// Returns nil if no data cached.
func (c *BarCache) Get(symbol, timeframe string, n int) []aggregator.BarEvent {
	key := cacheKey{symbol: symbol, timeframe: timeframe}
	c.mu.RLock()
	sc, ok := c.series[key]
	c.mu.RUnlock()
	if !ok {
		c.mu.Lock()
		c.misses++
		c.mu.Unlock()
		return nil
	}
	c.mu.Lock()
	c.hits++
	c.mu.Unlock()
	return sc.get(n)
}

// Latest returns the most recent bar for a (symbol, timeframe) pair.
func (c *BarCache) Latest(symbol, timeframe string) *aggregator.BarEvent {
	key := cacheKey{symbol: symbol, timeframe: timeframe}
	c.mu.RLock()
	sc, ok := c.series[key]
	c.mu.RUnlock()
	if !ok {
		return nil
	}
	return sc.latest()
}

// WarmFrom populates the cache from a slice of bars.
func (c *BarCache) WarmFrom(bars []aggregator.BarEvent) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, b := range bars {
		key := cacheKey{symbol: b.Symbol, timeframe: b.Timeframe}
		sc, ok := c.series[key]
		if !ok {
			sc = newSeriesCache(c.capacity)
			c.series[key] = sc
		}
		sc.put(b)
	}
}

// HitRate returns the cache hit rate as a float in [0, 1].
func (c *BarCache) HitRate() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	total := c.hits + c.misses
	if total == 0 {
		return 0
	}
	return float64(c.hits) / float64(total)
}
