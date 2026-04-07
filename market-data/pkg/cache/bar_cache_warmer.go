// Package cache -- BarCacheWarmer pre-warms L1 cache on startup, either from
// L2 SQLite or directly from CSV files.
package cache

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// BarCacheWarmer pre-populates L1 from L2 (or CSV) so that early queries avoid
// cold-cache misses.
type BarCacheWarmer struct {
	cache *TieredCache

	// Progress tracking: completed work items vs total.
	total     int64
	completed int64
}

// NewBarCacheWarmer creates a warmer backed by the given TieredCache.
func NewBarCacheWarmer(cache *TieredCache) *BarCacheWarmer {
	return &BarCacheWarmer{cache: cache}
}

// Warm sequentially loads the last 500 bars from L2 into L1 for every
// symbol x timeframe combination. It returns the first error encountered.
func (w *BarCacheWarmer) Warm(symbols []string, timeframes []string) error {
	items := int64(len(symbols) * len(timeframes))
	atomic.StoreInt64(&w.total, items)
	atomic.StoreInt64(&w.completed, 0)

	for _, sym := range symbols {
		for _, tf := range timeframes {
			if err := w.cache.WarmL1(sym, tf, 500); err != nil {
				return fmt.Errorf("warmer: warm %s/%s: %w", sym, tf, err)
			}
			atomic.AddInt64(&w.completed, 1)
		}
	}
	return nil
}

// warmOne is the per-item work function used by WarmConcurrent.
type warmItem struct {
	symbol    string
	timeframe string
}

// WarmConcurrent loads L2->L1 using a pool of workers goroutines. workers
// should be >= 1. Returns the first error from any worker; remaining items
// are drained before returning.
func (w *BarCacheWarmer) WarmConcurrent(symbols []string, timeframes []string, workers int) error {
	if workers < 1 {
		workers = 1
	}

	// Build work queue.
	total := len(symbols) * len(timeframes)
	atomic.StoreInt64(&w.total, int64(total))
	atomic.StoreInt64(&w.completed, 0)

	jobs := make(chan warmItem, total)
	for _, sym := range symbols {
		for _, tf := range timeframes {
			jobs <- warmItem{sym, tf}
		}
	}
	close(jobs)

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		firstErr error
	)

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range jobs {
				err := w.cache.WarmL1(item.symbol, item.timeframe, 500)
				atomic.AddInt64(&w.completed, 1)
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = fmt.Errorf("warmer: warm %s/%s: %w", item.symbol, item.timeframe, err)
					}
					mu.Unlock()
				}
			}
		}()
	}

	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	return firstErr
}

// Progress returns a value in [0.0, 1.0] representing warming completion.
// Returns 0 before Warm/WarmConcurrent is called and 1.0 when done.
func (w *BarCacheWarmer) Progress() float64 {
	total := atomic.LoadInt64(&w.total)
	if total == 0 {
		return 0.0
	}
	done := atomic.LoadInt64(&w.completed)
	if done >= total {
		return 1.0
	}
	return float64(done) / float64(total)
}

// WarmFromFile reads a CSV file and loads bars for the given symbol and
// timeframe directly into both L1 and L2 of the cache.
//
// Expected CSV columns (header row required):
//
//	timestamp,open,high,low,close,volume
//
// timestamp must be a Unix epoch in seconds (integer).
func (w *BarCacheWarmer) WarmFromFile(csvPath string, symbol string, timeframe string) error {
	f, err := os.Open(csvPath)
	if err != nil {
		return fmt.Errorf("warmer: open csv %s: %w", csvPath, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.TrimLeadingSpace = true

	// Read and validate header.
	header, err := r.Read()
	if err != nil {
		return fmt.Errorf("warmer: read header: %w", err)
	}
	colIdx, err := csvColumnIndex(header)
	if err != nil {
		return fmt.Errorf("warmer: csv header: %w", err)
	}

	var bars []Bar
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("warmer: read csv row: %w", err)
		}
		b, err := parseCsvBar(rec, colIdx, symbol, timeframe)
		if err != nil {
			return fmt.Errorf("warmer: parse row: %w", err)
		}
		bars = append(bars, b)
	}

	if len(bars) == 0 {
		return nil
	}

	// Write all bars to L2 first for persistence.
	if err := w.cache.writeL2(bars); err != nil {
		return fmt.Errorf("warmer: write L2: %w", err)
	}

	// Load the most recent 500 into L1.
	start := 0
	if len(bars) > 500 {
		start = len(bars) - 500
	}
	if err := w.cache.Put(symbol, timeframe, bars[start:]); err != nil {
		return fmt.Errorf("warmer: put L1: %w", err)
	}

	return nil
}

// csvColumnIndex returns a map of required column name -> index.
func csvColumnIndex(header []string) (map[string]int, error) {
	required := []string{"timestamp", "open", "high", "low", "close", "volume"}
	idx := make(map[string]int, len(required))
	for i, h := range header {
		idx[h] = i
	}
	for _, col := range required {
		if _, ok := idx[col]; !ok {
			return nil, fmt.Errorf("missing column %q", col)
		}
	}
	return idx, nil
}

// parseCsvBar converts a CSV record into a Bar using the column index.
func parseCsvBar(rec []string, idx map[string]int, symbol, timeframe string) (Bar, error) {
	parse := func(col string) (float64, error) {
		return strconv.ParseFloat(rec[idx[col]], 64)
	}
	tsRaw, err := strconv.ParseInt(rec[idx["timestamp"]], 10, 64)
	if err != nil {
		return Bar{}, fmt.Errorf("timestamp: %w", err)
	}
	open, err := parse("open")
	if err != nil {
		return Bar{}, fmt.Errorf("open: %w", err)
	}
	high, err := parse("high")
	if err != nil {
		return Bar{}, fmt.Errorf("high: %w", err)
	}
	low, err := parse("low")
	if err != nil {
		return Bar{}, fmt.Errorf("low: %w", err)
	}
	close_, err := parse("close")
	if err != nil {
		return Bar{}, fmt.Errorf("close: %w", err)
	}
	vol, err := parse("volume")
	if err != nil {
		return Bar{}, fmt.Errorf("volume: %w", err)
	}
	return Bar{
		Symbol:    symbol,
		Timeframe: timeframe,
		Open:      open,
		High:      high,
		Low:       low,
		Close:     close_,
		Volume:    vol,
		Timestamp: time.Unix(tsRaw, 0).UTC(),
	}, nil
}
