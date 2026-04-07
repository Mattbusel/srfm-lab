package cache

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func newTestCache(t *testing.T) *TieredCache {
	t.Helper()
	dir := t.TempDir()
	c, err := NewTieredCache(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("NewTieredCache: %v", err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func makeBar(sym, tf string, price float64, ts time.Time) Bar {
	return Bar{
		Symbol: sym, Timeframe: tf,
		Open: price, High: price + 1, Low: price - 1, Close: price,
		Volume: 1000, Timestamp: ts,
	}
}

// TestPutGet_L1Hit verifies that bars put into the cache are returned from L1.
func TestPutGet_L1Hit(t *testing.T) {
	c := newTestCache(t)
	now := time.Now().UTC().Truncate(time.Minute)

	bars := []Bar{
		makeBar("AAPL", "1h", 150, now.Add(-2*time.Hour)),
		makeBar("AAPL", "1h", 151, now.Add(-1*time.Hour)),
		makeBar("AAPL", "1h", 152, now),
	}
	if err := c.Put("AAPL", "1h", bars); err != nil {
		t.Fatalf("Put: %v", err)
	}

	got, err := c.Get("AAPL", "1h", 3)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 bars, got %d", len(got))
	}
	if got[2].Close != 152 {
		t.Errorf("expected last close=152, got %v", got[2].Close)
	}
}

// TestGet_L2Fallback verifies that bars evicted from L1 are retrieved from L2.
func TestGet_L2Fallback(t *testing.T) {
	c := newTestCache(t)
	now := time.Now().UTC().Truncate(time.Minute)

	// Fill L1 ring (capacity 500) with 501 bars -- first bar gets evicted to L2.
	var bars []Bar
	for i := 0; i < 501; i++ {
		bars = append(bars, makeBar("MSFT", "1h", float64(100+i), now.Add(time.Duration(i)*time.Hour)))
	}
	if err := c.Put("MSFT", "1h", bars); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// L1 now holds bars[1..500]. Request 501 bars -- should fall back to L2.
	got, err := c.Get("MSFT", "1h", 501)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("expected bars from L2 fallback, got none")
	}
}

// TestInvalidate clears L1 for a symbol.
func TestInvalidate(t *testing.T) {
	c := newTestCache(t)
	now := time.Now().UTC()

	if err := c.Put("GOOGL", "1d", []Bar{makeBar("GOOGL", "1d", 2800, now)}); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if c.L1Size("GOOGL", "1d") != 1 {
		t.Fatal("expected L1 size 1 before invalidate")
	}
	c.Invalidate("GOOGL")
	if c.L1Size("GOOGL", "1d") != 0 {
		t.Fatal("expected L1 size 0 after invalidate")
	}
}

// TestStats_L1HitRate verifies hit-rate tracking.
func TestStats_L1HitRate(t *testing.T) {
	c := newTestCache(t)
	now := time.Now().UTC()

	bars := make([]Bar, 10)
	for i := range bars {
		bars[i] = makeBar("SPY", "15m", float64(400+i), now.Add(time.Duration(i)*15*time.Minute))
	}
	if err := c.Put("SPY", "15m", bars); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Two L1 hits.
	c.Get("SPY", "15m", 5)
	c.Get("SPY", "15m", 5)

	stats := c.Stats()
	if stats.TotalQueries < 2 {
		t.Errorf("expected >= 2 queries, got %d", stats.TotalQueries)
	}
	if stats.L1HitRate == 0 {
		t.Error("expected non-zero L1 hit rate")
	}
}

// TestPutGet_MultipleSymbols ensures isolation between symbols.
func TestPutGet_MultipleSymbols(t *testing.T) {
	c := newTestCache(t)
	now := time.Now().UTC()

	c.Put("AAPL", "1h", []Bar{makeBar("AAPL", "1h", 150, now)})
	c.Put("TSLA", "1h", []Bar{makeBar("TSLA", "1h", 800, now)})

	gotA, _ := c.Get("AAPL", "1h", 1)
	gotT, _ := c.Get("TSLA", "1h", 1)

	if len(gotA) == 0 || gotA[0].Close != 150 {
		t.Errorf("AAPL: unexpected %v", gotA)
	}
	if len(gotT) == 0 || gotT[0].Close != 800 {
		t.Errorf("TSLA: unexpected %v", gotT)
	}
}

// TestWarmFromFile validates CSV loading.
func TestWarmFromFile(t *testing.T) {
	c := newTestCache(t)
	w := NewBarCacheWarmer(c)

	// Write a temp CSV.
	dir := t.TempDir()
	csvPath := filepath.Join(dir, "bars.csv")
	f, err := os.Create(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintln(f, "timestamp,open,high,low,close,volume")
	base := time.Now().UTC().Truncate(24*time.Hour).Unix()
	for i := 0; i < 5; i++ {
		fmt.Fprintf(f, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
			base+int64(i*3600), 100+float64(i), 101+float64(i), 99+float64(i), 100+float64(i), 1000)
	}
	f.Close()

	if err := w.WarmFromFile(csvPath, "TEST", "1h"); err != nil {
		t.Fatalf("WarmFromFile: %v", err)
	}
	if c.L1Size("TEST", "1h") == 0 {
		t.Error("expected bars in L1 after WarmFromFile")
	}
}

// TestWarmerProgress checks progress tracking.
func TestWarmerProgress(t *testing.T) {
	c := newTestCache(t)
	w := NewBarCacheWarmer(c)

	if w.Progress() != 0.0 {
		t.Error("expected 0 progress before Warm")
	}
	w.Warm([]string{"AAPL"}, []string{"1h", "1d"})
	if w.Progress() != 1.0 {
		t.Errorf("expected 1.0 progress after Warm, got %v", w.Progress())
	}
}

// TestWarmConcurrent verifies concurrent warming completes without error.
func TestWarmConcurrent(t *testing.T) {
	c := newTestCache(t)
	now := time.Now().UTC()

	syms := []string{"AAPL", "MSFT", "GOOGL", "AMZN"}
	tfs := []string{"1h", "1d"}
	for _, sym := range syms {
		for _, tf := range tfs {
			bars := []Bar{makeBar(sym, tf, 100, now)}
			c.writeL2(bars)
		}
	}

	w := NewBarCacheWarmer(c)
	if err := w.WarmConcurrent(syms, tfs, 4); err != nil {
		t.Fatalf("WarmConcurrent: %v", err)
	}
	if w.Progress() != 1.0 {
		t.Errorf("expected 1.0, got %v", w.Progress())
	}
}

// TestPruneL2 verifies old bars are deleted by PruneL2.
func TestPruneL2(t *testing.T) {
	c := newTestCache(t)
	// Override retention to 1 day so our old bar falls outside window.
	c.SetRetentionDays("1h", 1)

	oldBar := makeBar("NVDA", "1h", 500, time.Now().AddDate(0, 0, -5))
	c.writeL2([]Bar{oldBar})

	if err := c.PruneL2(); err != nil {
		t.Fatalf("PruneL2: %v", err)
	}

	bars, err := c.queryL2("NVDA", "1h", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(bars) != 0 {
		t.Errorf("expected 0 bars after prune, got %d", len(bars))
	}
}
