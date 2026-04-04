// Package cache provides an in-memory LRU bar cache with optional JSON persistence.
package cache

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/srfm/gateway/internal/feed"
	"go.uber.org/zap"
)

// CacheKey uniquely identifies a bar series.
type CacheKey struct {
	Symbol    string
	Timeframe string
}

// barList is a bounded list of bars sorted by ascending timestamp.
type barList struct {
	bars    []feed.Bar
	maxSize int
}

func newBarList(maxSize int) *barList {
	return &barList{
		bars:    make([]feed.Bar, 0, min(maxSize, 256)),
		maxSize: maxSize,
	}
}

func (bl *barList) push(b feed.Bar) {
	// Append and evict oldest if over capacity.
	bl.bars = append(bl.bars, b)
	if len(bl.bars) > bl.maxSize {
		// Evict oldest 10% to amortise the copy cost.
		evict := bl.maxSize / 10
		if evict < 1 {
			evict = 1
		}
		bl.bars = bl.bars[evict:]
	}
}

func (bl *barList) latest() *feed.Bar {
	if len(bl.bars) == 0 {
		return nil
	}
	b := bl.bars[len(bl.bars)-1]
	return &b
}

func (bl *barList) getRange(from, to time.Time) []feed.Bar {
	var out []feed.Bar
	for _, b := range bl.bars {
		if !b.Timestamp.Before(from) && !b.Timestamp.After(to) {
			out = append(out, b)
		}
	}
	return out
}

// BarCache is an in-memory cache of bars keyed by symbol + timeframe.
// It is goroutine-safe.
type BarCache struct {
	mu      sync.RWMutex
	data    map[CacheKey]*barList
	maxBars int
	log     *zap.Logger

	// Stats.
	hits   int64
	misses int64
}

// NewBarCache creates a new BarCache.
// maxBarsPerSeries is the maximum number of bars stored per symbol+timeframe.
func NewBarCache(maxBarsPerSeries int, log *zap.Logger) *BarCache {
	return &BarCache{
		data:    make(map[CacheKey]*barList),
		maxBars: maxBarsPerSeries,
		log:     log,
	}
}

// Push adds or updates a bar in the cache.
func (c *BarCache) Push(symbol, timeframe string, b feed.Bar) {
	key := CacheKey{symbol, timeframe}
	c.mu.Lock()
	defer c.mu.Unlock()
	bl, ok := c.data[key]
	if !ok {
		bl = newBarList(c.maxBars)
		c.data[key] = bl
	}
	bl.push(b)
}

// GetLatest returns the most recent bar for symbol+timeframe, or nil.
func (c *BarCache) GetLatest(symbol, timeframe string) *feed.Bar {
	key := CacheKey{symbol, timeframe}
	c.mu.RLock()
	bl, ok := c.data[key]
	c.mu.RUnlock()
	if !ok {
		c.miss()
		return nil
	}
	b := bl.latest()
	if b == nil {
		c.miss()
		return nil
	}
	c.hit()
	return b
}

// GetBars returns bars in [from, to] for symbol+timeframe.
func (c *BarCache) GetBars(symbol, timeframe string, from, to time.Time) []feed.Bar {
	key := CacheKey{symbol, timeframe}
	c.mu.RLock()
	bl, ok := c.data[key]
	c.mu.RUnlock()
	if !ok {
		c.miss()
		return nil
	}
	bars := bl.getRange(from, to)
	if len(bars) == 0 {
		c.miss()
	} else {
		c.hit()
	}
	return bars
}

// Symbols returns the list of all symbols known to the cache.
func (c *BarCache) Symbols() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	seen := make(map[string]struct{})
	for k := range c.data {
		seen[k.Symbol] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for s := range seen {
		out = append(out, s)
	}
	return out
}

// TotalBars returns the total number of bars stored across all series.
func (c *BarCache) TotalBars() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	total := 0
	for _, bl := range c.data {
		total += len(bl.bars)
	}
	return total
}

// HitRate returns the cache hit rate since start.
func (c *BarCache) HitRate() float64 {
	c.mu.RLock()
	h := c.hits
	m := c.misses
	c.mu.RUnlock()
	total := h + m
	if total == 0 {
		return 0
	}
	return float64(h) / float64(total)
}

func (c *BarCache) hit() {
	c.mu.Lock()
	c.hits++
	c.mu.Unlock()
}

func (c *BarCache) miss() {
	c.mu.Lock()
	c.misses++
	c.mu.Unlock()
}

// --- Persistence ---

// persistenceRecord is the on-disk format for a single series.
type persistenceRecord struct {
	Symbol    string     `json:"symbol"`
	Timeframe string     `json:"timeframe"`
	Bars      []feed.Bar `json:"bars"`
}

// SaveToFile serialises the full cache contents to a JSON file.
func (c *BarCache) SaveToFile(path string) error {
	c.mu.RLock()
	records := make([]persistenceRecord, 0, len(c.data))
	for k, bl := range c.data {
		bars := make([]feed.Bar, len(bl.bars))
		copy(bars, bl.bars)
		records = append(records, persistenceRecord{
			Symbol:    k.Symbol,
			Timeframe: k.Timeframe,
			Bars:      bars,
		})
	}
	c.mu.RUnlock()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create tmp: %w", err)
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(records); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("encode: %w", err)
	}
	f.Close()
	return os.Rename(tmp, path)
}

// LoadFromFile restores a previously saved cache from a JSON file.
func (c *BarCache) LoadFromFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // nothing to restore
		}
		return fmt.Errorf("open %q: %w", path, err)
	}
	defer f.Close()

	var records []persistenceRecord
	if err := json.NewDecoder(f).Decode(&records); err != nil {
		return fmt.Errorf("decode: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	for _, rec := range records {
		key := CacheKey{rec.Symbol, rec.Timeframe}
		bl := newBarList(c.maxBars)
		for _, b := range rec.Bars {
			bl.push(b)
		}
		c.data[key] = bl
	}
	c.log.Info("cache restored from file",
		zap.String("path", path),
		zap.Int("series", len(records)))
	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
