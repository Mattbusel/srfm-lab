// Package cache provides a two-level bar cache: L1 (in-memory ring buffers)
// and L2 (SQLite persistence). L1 holds up to 500 bars per symbol per
// timeframe. When the ring wraps, evicted bars are flushed to L2. L2 retains
// 252 trading days of daily bars, 63 days of hourly bars, and 21 days of
// 15-minute bars; all other timeframes follow the 21-day retention rule.
package cache

import (
	"database/sql"
	"fmt"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// Bar is a single OHLCV bar stored in the cache.
type Bar struct {
	Symbol    string
	Timeframe string
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
	Timestamp time.Time
}

// CacheStats holds hit-rate and usage counters.
type CacheStats struct {
	TotalQueries int64
	L1Hits       int64
	L2Hits       int64
	Misses       int64
	L1HitRate    float64
	L2HitRate    float64
}

// ringBuffer is a fixed-size circular buffer for Bars.
// Not goroutine-safe; callers must hold the parent lock.
type ringBuffer struct {
	buf      []Bar
	capacity int
	head     int // next write position
	size     int // number of valid entries
}

func newRingBuffer(cap int) *ringBuffer {
	return &ringBuffer{buf: make([]Bar, cap), capacity: cap}
}

// push inserts a bar. Returns the evicted bar and true when the ring is full
// before the push (the oldest bar is evicted to make room).
func (r *ringBuffer) push(b Bar) (evicted Bar, wasEvicted bool) {
	if r.size == r.capacity {
		evicted = r.buf[r.head]
		wasEvicted = true
	}
	r.buf[r.head] = b
	r.head = (r.head + 1) % r.capacity
	if r.size < r.capacity {
		r.size++
	}
	return
}

// lastN returns up to n bars, oldest first.
func (r *ringBuffer) lastN(n int) []Bar {
	if r.size == 0 {
		return nil
	}
	if n <= 0 || n > r.size {
		n = r.size
	}
	out := make([]Bar, n)
	startOff := r.size - n
	var absStart int
	if r.size < r.capacity {
		absStart = startOff
	} else {
		absStart = (r.head + startOff) % r.capacity
	}
	for i := 0; i < n; i++ {
		out[i] = r.buf[(absStart+i)%r.capacity]
	}
	return out
}

// clear resets the ring buffer to empty.
func (r *ringBuffer) clear() {
	r.head = 0
	r.size = 0
}

// l1Key is the composite key for the in-memory map.
type l1Key struct {
	symbol    string
	timeframe string
}

// TieredCache is a two-level bar cache.
//   - L1: per-symbol/per-timeframe ring buffers (500 bars each), protected by a
//     RWMutex.
//   - L2: SQLite tables per timeframe, queried on L1 miss and written when bars
//     are evicted from L1.
type TieredCache struct {
	mu sync.RWMutex

	// L1 ring buffers keyed by (symbol, timeframe).
	rings map[l1Key]*ringBuffer

	// L2 SQLite database.
	db *sql.DB

	// counters (accessed under mu or atomically via stats())
	totalQueries int64
	l1Hits       int64
	l2Hits       int64
	misses       int64

	// retention limits per timeframe (days).
	retentionDays map[string]int
}

// retentionDefaults returns the default retention policy (days) per timeframe.
func retentionDefaults() map[string]int {
	return map[string]int{
		"1d":  252,
		"1h":  63,
		"15m": 21,
		"5m":  21,
		"1m":  21,
		"4h":  63,
	}
}

// NewTieredCache opens (or creates) a SQLite database at dbPath and
// initialises the tiered cache. Call Close() when done.
func NewTieredCache(dbPath string) (*TieredCache, error) {
	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		return nil, fmt.Errorf("cache: open sqlite: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite is single-writer

	c := &TieredCache{
		rings:         make(map[l1Key]*ringBuffer),
		db:            db,
		retentionDays: retentionDefaults(),
	}
	if err := c.initSchema(); err != nil {
		db.Close()
		return nil, err
	}
	return c, nil
}

// Close releases the SQLite connection.
func (c *TieredCache) Close() error {
	return c.db.Close()
}

// tableName returns the SQLite table name for a timeframe.
func tableName(timeframe string) string {
	switch timeframe {
	case "1m":
		return "bars_1m"
	case "5m":
		return "bars_5m"
	case "15m":
		return "bars_15m"
	case "1h":
		return "bars_1h"
	case "4h":
		return "bars_4h"
	case "1d":
		return "bars_1d"
	default:
		return "bars_custom"
	}
}

// initSchema ensures all L2 tables exist.
func (c *TieredCache) initSchema() error {
	timeframes := []string{"1m", "5m", "15m", "1h", "4h", "1d", "custom"}
	for _, tf := range timeframes {
		tbl := "bars_" + tf
		if tf == "custom" {
			tbl = "bars_custom"
		}
		ddl := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
			symbol    TEXT    NOT NULL,
			timeframe TEXT    NOT NULL,
			ts        INTEGER NOT NULL,
			open      REAL    NOT NULL,
			high      REAL    NOT NULL,
			low       REAL    NOT NULL,
			close     REAL    NOT NULL,
			volume    REAL    NOT NULL,
			PRIMARY KEY (symbol, ts)
		)`, tbl)
		if _, err := c.db.Exec(ddl); err != nil {
			return fmt.Errorf("cache: create table %s: %w", tbl, err)
		}
		idx := fmt.Sprintf(`CREATE INDEX IF NOT EXISTS idx_%s_sym_ts ON %s (symbol, ts DESC)`, tbl, tbl)
		if _, err := c.db.Exec(idx); err != nil {
			return fmt.Errorf("cache: create index on %s: %w", tbl, err)
		}
	}
	return nil
}

// ring returns the L1 ring buffer for the given key, creating it if necessary.
// Callers must hold mu (write lock when creating).
func (c *TieredCache) ring(key l1Key) *ringBuffer {
	r, ok := c.rings[key]
	if !ok {
		r = newRingBuffer(500)
		c.rings[key] = r
	}
	return r
}

// Get returns up to limit bars for the given symbol and timeframe, oldest
// first. It checks L1 first; on a miss it falls back to L2.
func (c *TieredCache) Get(symbol string, timeframe string, limit int) ([]Bar, error) {
	key := l1Key{symbol, timeframe}

	c.mu.Lock()
	c.totalQueries++

	r := c.ring(key)
	bars := r.lastN(limit)
	if len(bars) >= limit {
		c.l1Hits++
		c.mu.Unlock()
		return bars, nil
	}
	c.mu.Unlock()

	// L2 fallback.
	l2Bars, err := c.queryL2(symbol, timeframe, limit)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	if len(l2Bars) > 0 {
		c.l2Hits++
	} else {
		c.misses++
	}
	c.mu.Unlock()

	return l2Bars, nil
}

// Put stores bars into L1. When L1 rings wrap, evicted bars are written to L2.
func (c *TieredCache) Put(symbol string, timeframe string, bars []Bar) error {
	key := l1Key{symbol, timeframe}

	var evicted []Bar

	c.mu.Lock()
	r := c.ring(key)
	for _, b := range bars {
		if ev, was := r.push(b); was {
			evicted = append(evicted, ev)
		}
	}
	c.mu.Unlock()

	if len(evicted) > 0 {
		if err := c.writeL2(evicted); err != nil {
			return err
		}
	}
	return nil
}

// Invalidate removes all L1 bars for symbol across every timeframe.
// L2 data is NOT purged (historical bars remain queryable).
func (c *TieredCache) Invalidate(symbol string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for key, r := range c.rings {
		if key.symbol == symbol {
			r.clear()
		}
	}
}

// Stats returns a snapshot of cache performance counters.
func (c *TieredCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	s := CacheStats{
		TotalQueries: c.totalQueries,
		L1Hits:       c.l1Hits,
		L2Hits:       c.l2Hits,
		Misses:       c.misses,
	}
	if s.TotalQueries > 0 {
		s.L1HitRate = float64(s.L1Hits) / float64(s.TotalQueries)
		s.L2HitRate = float64(s.L2Hits) / float64(s.TotalQueries)
	}
	return s
}

// writeL2 inserts bars into the appropriate SQLite table.
func (c *TieredCache) writeL2(bars []Bar) error {
	if len(bars) == 0 {
		return nil
	}
	// Group bars by timeframe to use per-table batch inserts.
	byTF := make(map[string][]Bar)
	for _, b := range bars {
		byTF[b.Timeframe] = append(byTF[b.Timeframe], b)
	}
	for tf, group := range byTF {
		tbl := tableName(tf)
		tx, err := c.db.Begin()
		if err != nil {
			return fmt.Errorf("cache: begin tx: %w", err)
		}
		stmt, err := tx.Prepare(fmt.Sprintf(
			`INSERT OR REPLACE INTO %s (symbol,timeframe,ts,open,high,low,close,volume)
			 VALUES (?,?,?,?,?,?,?,?)`, tbl))
		if err != nil {
			tx.Rollback()
			return fmt.Errorf("cache: prepare insert: %w", err)
		}
		for _, b := range group {
			if _, err := stmt.Exec(b.Symbol, b.Timeframe, b.Timestamp.UnixNano(),
				b.Open, b.High, b.Low, b.Close, b.Volume); err != nil {
				stmt.Close()
				tx.Rollback()
				return fmt.Errorf("cache: insert bar: %w", err)
			}
		}
		stmt.Close()
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("cache: commit: %w", err)
		}
	}
	return nil
}

// queryL2 retrieves up to limit bars from SQLite, oldest first.
func (c *TieredCache) queryL2(symbol string, timeframe string, limit int) ([]Bar, error) {
	tbl := tableName(timeframe)
	retDays, ok := c.retentionDays[timeframe]
	if !ok {
		retDays = 21
	}
	cutoff := time.Now().AddDate(0, 0, -retDays).UnixNano()

	query := fmt.Sprintf(`
		SELECT symbol, timeframe, ts, open, high, low, close, volume
		FROM %s
		WHERE symbol = ? AND ts >= ?
		ORDER BY ts DESC
		LIMIT ?`, tbl)

	rows, err := c.db.Query(query, symbol, cutoff, limit)
	if err != nil {
		return nil, fmt.Errorf("cache: query L2: %w", err)
	}
	defer rows.Close()

	var bars []Bar
	for rows.Next() {
		var b Bar
		var tsNano int64
		if err := rows.Scan(&b.Symbol, &b.Timeframe, &tsNano,
			&b.Open, &b.High, &b.Low, &b.Close, &b.Volume); err != nil {
			return nil, fmt.Errorf("cache: scan bar: %w", err)
		}
		b.Timestamp = time.Unix(0, tsNano)
		bars = append(bars, b)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("cache: rows error: %w", err)
	}

	// Reverse so result is oldest-first.
	for i, j := 0, len(bars)-1; i < j; i, j = i+1, j-1 {
		bars[i], bars[j] = bars[j], bars[i]
	}
	return bars, nil
}

// PruneL2 deletes bars older than the retention window from every table.
// This can be called periodically (e.g., daily) to keep the database compact.
func (c *TieredCache) PruneL2() error {
	for tf, days := range c.retentionDays {
		tbl := tableName(tf)
		cutoff := time.Now().AddDate(0, 0, -days).UnixNano()
		if _, err := c.db.Exec(
			fmt.Sprintf(`DELETE FROM %s WHERE ts < ?`, tbl), cutoff); err != nil {
			return fmt.Errorf("cache: prune %s: %w", tbl, err)
		}
	}
	return nil
}

// SetRetentionDays overrides the retention policy for a timeframe.
func (c *TieredCache) SetRetentionDays(timeframe string, days int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.retentionDays[timeframe] = days
}

// L1Size returns the number of bars currently in L1 for a symbol/timeframe pair.
func (c *TieredCache) L1Size(symbol, timeframe string) int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	key := l1Key{symbol, timeframe}
	r, ok := c.rings[key]
	if !ok {
		return 0
	}
	return r.size
}

// WarmL1 loads up to limit bars from L2 into L1 for the given symbol and
// timeframe. Called by the BarCacheWarmer on startup.
func (c *TieredCache) WarmL1(symbol, timeframe string, limit int) error {
	bars, err := c.queryL2(symbol, timeframe, limit)
	if err != nil {
		return err
	}
	if len(bars) == 0 {
		return nil
	}
	key := l1Key{symbol, timeframe}
	c.mu.Lock()
	r := c.ring(key)
	for _, b := range bars {
		r.push(b) // warming does not evict to L2 -- bars came from L2
	}
	c.mu.Unlock()
	return nil
}
