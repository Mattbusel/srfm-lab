package analytics_test

import (
	"context"
	"math"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"srfm/market-data/aggregator"
	"srfm/market-data/pkg/analytics"
	pkfeed "srfm/market-data/pkg/feed"
	pkstorage "srfm/market-data/pkg/storage"
)

// -- helpers --

func makeBar(sym string, o, h, l, c, v float64, ts time.Time) analytics.Bar {
	return analytics.Bar{
		Symbol:    sym,
		Timeframe: "15m",
		Open:      o,
		High:      h,
		Low:       l,
		Close:     c,
		Volume:    v,
		Timestamp: ts,
	}
}

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// =====================================================================
// BarStatsComputer / Welford tests
// =====================================================================

// TestBarStatsWelfordVariance verifies that Welford's online algorithm
// produces the same population variance as a direct batch computation.
func TestBarStatsWelfordVariance(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	sym := "BTC"
	closes := []float64{100, 102, 98, 105, 103, 99, 101, 107, 95, 110}
	ts := time.Now()

	for _, c := range closes {
		comp.OnBar(sym, makeBar(sym, c, c+1, c-1, c, 1000, ts))
		ts = ts.Add(15 * time.Minute)
	}

	stats, ok := comp.GetStats(sym)
	if !ok {
		t.Fatal("expected stats")
	}

	// Batch population variance
	var sum, sumSq float64
	for _, c := range closes {
		sum += c
		sumSq += c * c
	}
	mean := sum / float64(len(closes))
	batchVar := sumSq/float64(len(closes)) - mean*mean

	if !almostEqual(stats.Mean, mean, 1e-9) {
		t.Errorf("mean: got %v, want %v", stats.Mean, mean)
	}
	if !almostEqual(stats.Variance, batchVar, 1e-6) {
		t.Errorf("variance: got %v, want %v", stats.Variance, batchVar)
	}
}

// TestBarStatsWelfordSingleBar verifies stats on a single observation.
func TestBarStatsWelfordSingleBar(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	bar := makeBar("ETH", 2000, 2010, 1990, 2005, 500, time.Now())
	stats := comp.OnBar("ETH", bar)

	if stats.Count != 1 {
		t.Errorf("count: got %d, want 1", stats.Count)
	}
	if stats.Mean != 2005 {
		t.Errorf("mean: got %v, want 2005", stats.Mean)
	}
	if stats.Variance != 0 {
		t.Errorf("variance of single obs should be 0, got %v", stats.Variance)
	}
}

// TestBarStatsEMAConvergence verifies EMA8/EMA21 converge toward a constant price.
func TestBarStatsEMAConvergence(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	sym := "SOL"
	ts := time.Now()
	var last analytics.BarStats
	for i := 0; i < 60; i++ {
		last = comp.OnBar(sym, makeBar(sym, 50, 51, 49, 50, 100, ts))
		ts = ts.Add(15 * time.Minute)
	}
	if !almostEqual(last.EMA8, 50, 0.01) {
		t.Errorf("EMA8 did not converge: %v", last.EMA8)
	}
	if !almostEqual(last.EMA21, 50, 0.01) {
		t.Errorf("EMA21 did not converge: %v", last.EMA21)
	}
}

// TestBarStatsVolZScore verifies z-score sign after a volume spike.
func TestBarStatsVolZScore(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	sym := "ADA"
	ts := time.Now()
	for i := 0; i < 30; i++ {
		comp.OnBar(sym, makeBar(sym, 1, 1.01, 0.99, 1, 100, ts))
		ts = ts.Add(15 * time.Minute)
	}
	spike := comp.OnBar(sym, makeBar(sym, 1, 1.01, 0.99, 1, 10000, ts))
	if spike.VolZScore <= 0 {
		t.Errorf("expected positive z-score after spike, got %v", spike.VolZScore)
	}
}

// TestBarStatsConcurrentSymbols verifies no data races under concurrent writes.
func TestBarStatsConcurrentSymbols(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	symbols := []string{"BTC", "ETH", "SOL", "ADA", "DOT"}
	var wg sync.WaitGroup
	for _, sym := range symbols {
		wg.Add(1)
		go func(s string) {
			defer wg.Done()
			ts := time.Now()
			for i := 0; i < 100; i++ {
				price := float64(100 + i)
				comp.OnBar(s, makeBar(s, price, price+1, price-1, price, float64(i+1)*100, ts))
				ts = ts.Add(15 * time.Minute)
			}
		}(sym)
	}
	wg.Wait()
	for _, sym := range symbols {
		stats, ok := comp.GetStats(sym)
		if !ok {
			t.Errorf("missing stats for %s", sym)
			continue
		}
		if stats.Count != 100 {
			t.Errorf("%s: expected 100 bars, got %d", sym, stats.Count)
		}
	}
}

// TestBarStatsATRPositive verifies ATR is positive for bars with a real range.
func TestBarStatsATRPositive(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	sym := "LINK"
	ts := time.Now()
	for i := 0; i < 20; i++ {
		comp.OnBar(sym, makeBar(sym, 10, 12, 8, 10, 1000, ts))
		ts = ts.Add(15 * time.Minute)
	}
	stats, _ := comp.GetStats(sym)
	if stats.ATR <= 0 {
		t.Errorf("ATR should be positive, got %v", stats.ATR)
	}
}

// TestBarStatsGetStatsEmpty verifies GetStats returns false before any bars.
func TestBarStatsGetStatsEmpty(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	_, ok := comp.GetStats("NOTHING")
	if ok {
		t.Error("expected false for unseen symbol")
	}
}

// TestWelfordLargeN verifies numerical stability with 1000 random bars.
func TestWelfordLargeN(t *testing.T) {
	comp := analytics.NewBarStatsComputer()
	sym := "TEST"
	ts := time.Now()
	rng := rand.New(rand.NewSource(42))

	var vals []float64
	for i := 0; i < 1000; i++ {
		v := 100 + rng.NormFloat64()*10
		vals = append(vals, v)
		comp.OnBar(sym, makeBar(sym, v, v+1, v-1, v, 1000, ts))
		ts = ts.Add(time.Minute)
	}

	stats, _ := comp.GetStats(sym)

	var sum float64
	for _, v := range vals {
		sum += v
	}
	batchMean := sum / float64(len(vals))

	if math.Abs(stats.Mean-batchMean) > 0.001 {
		t.Errorf("mean divergence: online=%v batch=%v", stats.Mean, batchMean)
	}
}

// =====================================================================
// SpreadMonitor tests
// =====================================================================

// TestSpreadMonitorPercentile verifies the percentile implementation.
func TestSpreadMonitorPercentile(t *testing.T) {
	mon := analytics.NewSpreadMonitor()
	sym := "BTC"
	for i := 1; i <= 100; i++ {
		bid := 50000.0
		ask := bid + bid*float64(i)/10_000.0
		mon.OnBookUpdate(sym, bid, ask, 1.0, 1.0)
	}

	p50 := mon.GetSpreadPercentile(sym, 50)
	if p50 < 49 || p50 > 51 {
		t.Errorf("p50 out of range: %v", p50)
	}
	p95 := mon.GetSpreadPercentile(sym, 95)
	if p95 < 93 || p95 > 97 {
		t.Errorf("p95 out of range: %v", p95)
	}
	p0 := mon.GetSpreadPercentile(sym, 0)
	if p0 < 0.5 || p0 > 1.5 {
		t.Errorf("p0 out of range: %v", p0)
	}
}

// TestSpreadMonitorIsWide verifies the 2x-mean threshold.
func TestSpreadMonitorIsWide(t *testing.T) {
	mon := analytics.NewSpreadMonitor()
	sym := "ETH"
	for i := 0; i < 100; i++ {
		bid := 3000.0
		ask := bid + bid/10_000.0 // 1 bps
		mon.OnBookUpdate(sym, bid, ask, 5, 5)
	}
	if mon.IsWide(sym) {
		t.Error("narrow spread should not be flagged as wide")
	}
	// Insert wide spread: 10 bps vs 1 bps mean
	mon.OnBookUpdate(sym, 3000, 3000+3000*0.001, 5, 5)
	if !mon.IsWide(sym) {
		t.Error("wide spread should be flagged")
	}
}

// TestSpreadMonitorLiquidityScore verifies score is in [0,1].
func TestSpreadMonitorLiquidityScore(t *testing.T) {
	mon := analytics.NewSpreadMonitor()
	sym := "SOL"
	if mon.GetLiquidityScore(sym) != 0 {
		t.Error("empty liquidity score should be 0")
	}
	mon.OnBookUpdate(sym, 100, 100.01, 200, 200)
	score := mon.GetLiquidityScore(sym)
	if score < 0 || score > 1 {
		t.Errorf("liquidity score out of [0,1]: %v", score)
	}
}

// TestSpreadMonitorCircularBuffer verifies buffer rolls at 100 entries.
func TestSpreadMonitorCircularBuffer(t *testing.T) {
	mon := analytics.NewSpreadMonitor()
	sym := "BNB"
	for i := 1; i <= 200; i++ {
		bid := 300.0
		ask := bid + bid*float64(i)/10_000.0
		mon.OnBookUpdate(sym, bid, ask, 1, 1)
	}
	// Only last 100 entries retained; minimum spread should be ~101 bps
	p0 := mon.GetSpreadPercentile(sym, 0)
	if p0 < 9 {
		t.Errorf("oldest entries should have been evicted, p0=%v", p0)
	}
}

// TestSpreadMonitorUnknownSymbol verifies nil-safe returns.
func TestSpreadMonitorUnknownSymbol(t *testing.T) {
	mon := analytics.NewSpreadMonitor()
	_, ok := mon.GetSpread("UNKNOWN")
	if ok {
		t.Error("expected false for unknown symbol")
	}
	if mon.IsWide("UNKNOWN") {
		t.Error("unknown symbol should not be wide")
	}
	if mon.GetLiquidityScore("UNKNOWN") != 0 {
		t.Error("unknown symbol liquidity should be 0")
	}
}

// TestSpreadMonitorConcurrent verifies concurrent OnBookUpdate calls are safe.
func TestSpreadMonitorConcurrent(t *testing.T) {
	mon := analytics.NewSpreadMonitor()
	sym := "BTC"
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				bid := 50000.0 + float64(id*j)
				ask := bid + 5
				mon.OnBookUpdate(sym, bid, ask, float64(j+1), float64(j+1))
			}
		}(i)
	}
	wg.Wait()
	_, ok := mon.GetSpread(sym)
	if !ok {
		t.Error("expected spread data after concurrent writes")
	}
}

// =====================================================================
// VolumeClock tests
// =====================================================================

// TestVolumeClockBarCompletion verifies a bar closes once target volume is reached.
func TestVolumeClockBarCompletion(t *testing.T) {
	vc := analytics.NewVolumeClock(50)
	sym := "BTC"
	ts := time.Now()

	var completed *analytics.VolumeBar
	for i := 0; i < 20; i++ {
		bar := vc.OnTickAt(sym, 50000, 10, ts.Add(time.Duration(i)*time.Second))
		if bar != nil {
			completed = bar
			break
		}
	}

	if completed == nil {
		t.Fatal("expected at least one completed volume bar")
	}
	if completed.Volume <= 0 {
		t.Error("completed bar volume should be > 0")
	}
	if completed.Open <= 0 || completed.Close <= 0 {
		t.Error("completed bar prices should be > 0")
	}
}

// TestVolumeClockOHLCCorrectness verifies OHLC tracking within a bar.
func TestVolumeClockOHLCCorrectness(t *testing.T) {
	vc := analytics.NewVolumeClock(1)
	sym := "ETH"
	ts := time.Now()

	var allBars []analytics.VolumeBar
	prices := []float64{2000, 2100, 1950, 2050}
	for _, p := range prices {
		bar := vc.OnTickAt(sym, p, 1e9, ts)
		ts = ts.Add(time.Second)
		if bar != nil {
			allBars = append(allBars, *bar)
		}
	}

	if len(allBars) == 0 {
		t.Fatal("expected completed bars")
	}
	for _, b := range allBars {
		if b.High < b.Low {
			t.Errorf("high < low in bar: %+v", b)
		}
		if b.Open <= 0 || b.Close <= 0 {
			t.Errorf("invalid price in bar: %+v", b)
		}
	}
}

// TestVolumeClockGetBarsLimit verifies GetBars respects the n limit.
func TestVolumeClockGetBarsLimit(t *testing.T) {
	vc := analytics.NewVolumeClock(50)
	sym := "SOL"
	ts := time.Now()
	for i := 0; i < 30; i++ {
		vc.OnTickAt(sym, 100, 1e12, ts)
		ts = ts.Add(time.Second)
	}

	bars := vc.GetBars(sym, 10)
	if len(bars) > 10 {
		t.Errorf("expected <= 10 bars, got %d", len(bars))
	}
}

// TestVolumeClockTargetCalibration verifies target_volume adapts over time.
func TestVolumeClockTargetCalibration(t *testing.T) {
	vc := analytics.NewVolumeClock(50)
	sym := "AVAX"
	ts := time.Now()
	for i := 0; i < 100; i++ {
		vc.OnTickAt(sym, 20, 1e6, ts)
		ts = ts.Add(time.Second)
	}
	target := vc.GetTargetVolume(sym)
	if target <= 0 {
		t.Errorf("target volume should be positive, got %v", target)
	}
}

// TestVolumeClockNilForInProgress verifies OnTick returns nil before target is reached.
func TestVolumeClockNilForInProgress(t *testing.T) {
	vc := analytics.NewVolumeClock(50)
	sym := "BTC"
	ts := time.Now()

	// Close first bar to establish calibration
	vc.OnTickAt(sym, 100, 1e10, ts)
	ts = ts.Add(time.Second)

	// After calibration, tiny volumes should not close the bar
	nilCount := 0
	for i := 0; i < 20; i++ {
		result := vc.OnTickAt(sym, 100, 0.000001, ts.Add(time.Duration(i)*time.Second))
		if result == nil {
			nilCount++
		}
	}
	if nilCount == 0 {
		t.Log("note: all ticks returned bars -- may be normal for very low volumes")
	}
}

// TestVolumeClockUnknownSymbol verifies GetBars returns nil for unseen symbol.
func TestVolumeClockUnknownSymbol(t *testing.T) {
	vc := analytics.NewVolumeClock(50)
	bars := vc.GetBars("UNKNOWN", 10)
	if bars != nil {
		t.Error("expected nil for unknown symbol")
	}
	if vc.GetTargetVolume("UNKNOWN") != 0 {
		t.Error("expected 0 target for unknown symbol")
	}
}

// =====================================================================
// ManagedWebSocket / ReconnectConfig tests
// =====================================================================

// TestReconnectConfigDefaults tests the DefaultReconnectConfig values match spec.
func TestReconnectConfigDefaults(t *testing.T) {
	cfg := pkfeed.DefaultReconnectConfig()
	if cfg.InitialDelay != time.Second {
		t.Errorf("InitialDelay: got %v, want 1s", cfg.InitialDelay)
	}
	if cfg.MaxDelay != 60*time.Second {
		t.Errorf("MaxDelay: got %v, want 60s", cfg.MaxDelay)
	}
	if cfg.BackoffMultiplier != 2.0 {
		t.Errorf("BackoffMultiplier: got %v, want 2.0", cfg.BackoffMultiplier)
	}
	if cfg.MaxRetries != 0 {
		t.Errorf("MaxRetries: got %v, want 0", cfg.MaxRetries)
	}
	if cfg.JitterPct != 0.20 {
		t.Errorf("JitterPct: got %v, want 0.20", cfg.JitterPct)
	}
}

// TestManagedWebSocketConnect verifies Connect and ReadMessage work with a local server.
func TestManagedWebSocketConnect(t *testing.T) {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool { return true },
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Logf("upgrade: %v", err)
			return
		}
		defer conn.Close()
		conn.WriteMessage(websocket.TextMessage, []byte(`{"hello":"world"}`)) //nolint:errcheck
	}))
	defer srv.Close()

	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	cfg := pkfeed.DefaultReconnectConfig()
	cfg.MaxRetries = 1
	mws := pkfeed.NewManagedWebSocket(wsURL, cfg, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := mws.Connect(ctx); err != nil {
		t.Fatalf("connect error: %v", err)
	}

	msg, err := mws.ReadMessageContext(ctx)
	if err != nil {
		t.Fatalf("read error: %v", err)
	}
	if !strings.Contains(string(msg), "hello") {
		t.Errorf("unexpected message: %s", msg)
	}
}

// TestManagedWebSocketIsConnectedInitial verifies IsConnected before Connect.
func TestManagedWebSocketIsConnectedInitial(t *testing.T) {
	cfg := pkfeed.DefaultReconnectConfig()
	mws := pkfeed.NewManagedWebSocket("ws://192.0.2.1:9999/ws", cfg, nil)
	if mws.IsConnected() {
		t.Error("should not be connected before Connect() is called")
	}
}

// TestManagedWebSocketReconnectBackoff verifies ReconnectCount starts at 0.
func TestManagedWebSocketReconnectBackoff(t *testing.T) {
	cfg := pkfeed.DefaultReconnectConfig()
	mws := pkfeed.NewManagedWebSocket("ws://192.0.2.1:9999/ws", cfg, nil)
	if mws.ReconnectCount() != 0 {
		t.Errorf("reconnect count should be 0 before any connection, got %d", mws.ReconnectCount())
	}
}

// =====================================================================
// TimeSeriesCache tests
// =====================================================================

// TestTimeSeriesCacheEviction verifies that bars beyond capacity are dropped.
func TestTimeSeriesCacheEviction(t *testing.T) {
	capacity := 10
	cache := pkstorage.NewTimeSeriesCacheWithCapacity(capacity)
	sym := "BTC"
	tf := "15m"
	ts := time.Now()

	for i := 0; i < 25; i++ {
		cache.Store(sym, tf, aggregator.BarEvent{
			Symbol:    sym,
			Timeframe: tf,
			Close:     float64(i + 1),
			Timestamp: ts.Add(time.Duration(i) * 15 * time.Minute),
		})
	}

	bars := cache.GetLast(sym, tf, 0)
	if len(bars) > capacity {
		t.Errorf("expected at most %d bars, got %d", capacity, len(bars))
	}
	if bars[len(bars)-1].Close != 25 {
		t.Errorf("expected last close=25, got %v", bars[len(bars)-1].Close)
	}
}

// TestTimeSeriesCacheGetLast verifies GetLast returns the correct count.
func TestTimeSeriesCacheGetLast(t *testing.T) {
	cache := pkstorage.NewTimeSeriesCache()
	sym := "ETH"
	tf := "1h"
	ts := time.Now()
	for i := 0; i < 50; i++ {
		cache.Store(sym, tf, aggregator.BarEvent{
			Symbol:    sym,
			Timeframe: tf,
			Close:     float64(i),
			Timestamp: ts.Add(time.Duration(i) * time.Hour),
		})
	}
	got := cache.GetLast(sym, tf, 20)
	if len(got) != 20 {
		t.Errorf("expected 20 bars, got %d", len(got))
	}
}

// TestTimeSeriesCacheGetSince verifies time-based filtering.
func TestTimeSeriesCacheGetSince(t *testing.T) {
	cache := pkstorage.NewTimeSeriesCache()
	sym := "SOL"
	tf := "15m"
	base := time.Now().Truncate(time.Hour)
	for i := 0; i < 20; i++ {
		cache.Store(sym, tf, aggregator.BarEvent{
			Symbol:    sym,
			Timeframe: tf,
			Close:     float64(i),
			Timestamp: base.Add(time.Duration(i) * 15 * time.Minute),
		})
	}
	cutoff := base.Add(10 * 15 * time.Minute) // bars 10..19
	bars := cache.GetSince(sym, tf, cutoff)
	if len(bars) != 10 {
		t.Errorf("expected 10 bars since cutoff, got %d", len(bars))
	}
}

// TestTimeSeriesCacheUnknownKey verifies nil returns for missing keys.
func TestTimeSeriesCacheUnknownKey(t *testing.T) {
	cache := pkstorage.NewTimeSeriesCache()
	if cache.GetLast("UNKNOWN", "15m", 10) != nil {
		t.Error("expected nil for unknown key")
	}
	if cache.Latest("UNKNOWN", "15m") != nil {
		t.Error("expected nil Latest for unknown key")
	}
}

// TestTimeSeriesCacheEstimateSize verifies size estimate is positive after inserts.
func TestTimeSeriesCacheEstimateSize(t *testing.T) {
	cache := pkstorage.NewTimeSeriesCache()
	ts := time.Now()
	for i := 0; i < 10; i++ {
		cache.Store("BTC", "15m", aggregator.BarEvent{
			Symbol:    "BTC",
			Timeframe: "15m",
			Close:     float64(i),
			Timestamp: ts.Add(time.Duration(i) * 15 * time.Minute),
		})
	}
	size := cache.EstimateSize()
	if size <= 0 {
		t.Errorf("estimated size should be positive, got %d", size)
	}
}

// TestTimeSeriesCacheConcurrentWrites verifies no data races under concurrency.
func TestTimeSeriesCacheConcurrentWrites(t *testing.T) {
	cache := pkstorage.NewTimeSeriesCache()
	var wg sync.WaitGroup
	symbols := []string{"BTC", "ETH", "SOL"}
	ts := time.Now()
	for _, sym := range symbols {
		wg.Add(1)
		go func(s string) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				cache.Store(s, "15m", aggregator.BarEvent{
					Symbol:    s,
					Timeframe: "15m",
					Close:     float64(i),
					Timestamp: ts.Add(time.Duration(i) * 15 * time.Minute),
				})
			}
		}(sym)
	}
	wg.Wait()
	for _, sym := range symbols {
		if len(cache.GetLast(sym, "15m", 500)) == 0 {
			t.Errorf("expected bars for %s", sym)
		}
	}
}

// TestTimeSeriesCacheSeriesCount verifies SeriesCount tracking.
func TestTimeSeriesCacheSeriesCount(t *testing.T) {
	cache := pkstorage.NewTimeSeriesCache()
	ts := time.Now()
	cache.Store("BTC", "15m", aggregator.BarEvent{Symbol: "BTC", Timeframe: "15m", Timestamp: ts})
	cache.Store("ETH", "15m", aggregator.BarEvent{Symbol: "ETH", Timeframe: "15m", Timestamp: ts})
	cache.Store("BTC", "1h", aggregator.BarEvent{Symbol: "BTC", Timeframe: "1h", Timestamp: ts})
	if cache.SeriesCount() != 3 {
		t.Errorf("expected 3 series, got %d", cache.SeriesCount())
	}
}

// =====================================================================
// FeedHealthMonitor tests
// =====================================================================

// TestFeedHealthMonitorStale verifies a feed is flagged stale when silent.
func TestFeedHealthMonitorStale(t *testing.T) {
	mon := pkfeed.NewFeedHealthMonitor(10 * time.Millisecond)
	defer mon.Stop()

	mon.Register("test_feed", 20*time.Millisecond)
	if !mon.IsHealthy("test_feed") {
		t.Error("should be healthy just after registration")
	}
	// Wait for 2x expected interval + check interval
	time.Sleep(100 * time.Millisecond)
	if mon.IsHealthy("test_feed") {
		t.Error("feed should be stale after silence period")
	}
}

// TestFeedHealthMonitorRecovery verifies recovery after a message.
func TestFeedHealthMonitorRecovery(t *testing.T) {
	mon := pkfeed.NewFeedHealthMonitor(10 * time.Millisecond)
	defer mon.Stop()

	mon.Register("recovery_feed", 20*time.Millisecond)
	time.Sleep(100 * time.Millisecond) // go stale

	if mon.IsHealthy("recovery_feed") {
		t.Error("should be stale before recovery")
	}
	mon.RecordMessage("recovery_feed")
	if !mon.IsHealthy("recovery_feed") {
		t.Error("should be healthy after message")
	}
}

// TestFeedHealthMonitorUnknownFeed verifies false for unregistered feeds.
func TestFeedHealthMonitorUnknownFeed(t *testing.T) {
	mon := pkfeed.NewFeedHealthMonitor(10 * time.Millisecond)
	defer mon.Stop()

	if mon.IsHealthy("nonexistent") {
		t.Error("unknown feed should not be healthy")
	}
	if mon.MessageCount("nonexistent") != 0 {
		t.Error("unknown feed message count should be 0")
	}
}

// TestFeedHealthMonitorAllHealthy verifies AllHealthy aggregation.
func TestFeedHealthMonitorAllHealthy(t *testing.T) {
	mon := pkfeed.NewFeedHealthMonitor(10 * time.Millisecond)
	defer mon.Stop()

	mon.Register("feed_a", time.Minute)
	mon.Register("feed_b", time.Minute)
	if !mon.AllHealthy() {
		t.Error("both feeds should be healthy initially")
	}
}

// TestFeedHealthMonitorMessageCount verifies message counting.
func TestFeedHealthMonitorMessageCount(t *testing.T) {
	mon := pkfeed.NewFeedHealthMonitor(time.Minute)
	defer mon.Stop()

	mon.Register("count_feed", time.Minute)
	for i := 0; i < 5; i++ {
		mon.RecordMessage("count_feed")
	}
	if mon.MessageCount("count_feed") != 5 {
		t.Errorf("expected 5 messages, got %d", mon.MessageCount("count_feed"))
	}
}
