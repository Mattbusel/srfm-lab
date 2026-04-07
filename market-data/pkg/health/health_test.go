package health

import (
	"testing"
	"time"
)

// TestReportBar_NotStale verifies a recently reported bar is not marked stale.
func TestReportBar_NotStale(t *testing.T) {
	m := NewFeedHealthMonitor()
	now := time.Now()
	m.now = func() time.Time { return now }

	m.ReportBar("AAPL", "1h", now.Add(-30*time.Minute))

	statuses := m.GetStatus("AAPL")
	st, ok := statuses["1h"]
	if !ok {
		t.Fatal("no status for 1h")
	}
	if st.IsStale {
		t.Error("should not be stale -- last bar 30m ago, threshold 65m")
	}
}

// TestReportBar_Stale verifies a feed with a very old bar is marked stale.
func TestReportBar_Stale(t *testing.T) {
	m := NewFeedHealthMonitor()
	past := time.Now().Add(-3 * time.Hour)
	m.now = func() time.Time { return time.Now() }

	m.ReportBar("MSFT", "1h", past)

	statuses := m.GetStatus("MSFT")
	st := statuses["1h"]
	if !st.IsStale {
		t.Error("should be stale -- last bar 3h ago, threshold 65m")
	}
}

// TestGapDetection verifies GapCount increments on missing bars.
func TestGapDetection(t *testing.T) {
	m := NewFeedHealthMonitor()
	base := time.Now().Truncate(time.Hour)

	// First bar -- no gap possible.
	m.ReportBar("SPY", "1h", base)
	// Next bar 3 hours later (expected 1h, got 3h -- gap).
	m.ReportBar("SPY", "1h", base.Add(3*time.Hour))

	statuses := m.GetStatus("SPY")
	st := statuses["1h"]
	if st.GapCount != 1 {
		t.Errorf("expected GapCount=1, got %d", st.GapCount)
	}
}

// TestNoGap verifies back-to-back bars don't trigger a gap.
func TestNoGap(t *testing.T) {
	m := NewFeedHealthMonitor()
	base := time.Now().Truncate(time.Minute)

	m.ReportBar("QQQ", "1m", base)
	m.ReportBar("QQQ", "1m", base.Add(time.Minute))

	statuses := m.GetStatus("QQQ")
	if statuses["1m"].GapCount != 0 {
		t.Errorf("expected GapCount=0, got %d", statuses["1m"].GapCount)
	}
}

// TestStaleFeeds returns only stale feeds.
func TestStaleFeeds(t *testing.T) {
	m := NewFeedHealthMonitor()
	now := time.Now()
	m.now = func() time.Time { return now }

	// Fresh feed.
	m.ReportBar("AAPL", "15m", now.Add(-5*time.Minute))
	// Stale feed (25m old, threshold 20m).
	m.ReportBar("TSLA", "15m", now.Add(-25*time.Minute))

	stale := m.StaleFeeds(0)
	if len(stale) == 0 {
		t.Fatal("expected at least one stale feed")
	}
	found := false
	for _, s := range stale {
		if s.Symbol == "TSLA" {
			found = true
		}
	}
	if !found {
		t.Error("TSLA should be in stale feeds")
	}
}

// TestReportError increments ErrorCount.
func TestReportError(t *testing.T) {
	m := NewFeedHealthMonitor()
	m.ReportBar("NVDA", "1d", time.Now())
	m.ReportError("NVDA", "1d")
	m.ReportError("NVDA", "1d")

	statuses := m.GetStatus("NVDA")
	if statuses["1d"].ErrorCount != 2 {
		t.Errorf("expected ErrorCount=2, got %d", statuses["1d"].ErrorCount)
	}
}

// TestLatencyMonitor_Avg verifies average latency computation.
func TestLatencyMonitor_Avg(t *testing.T) {
	lm := NewFeedLatencyMonitor()
	base := time.Now()

	lm.Record("BTC", base, base.Add(10*time.Millisecond))
	lm.Record("BTC", base, base.Add(20*time.Millisecond))
	lm.Record("BTC", base, base.Add(30*time.Millisecond))

	avg := lm.AvgLatency("BTC")
	wantMs := 20 * time.Millisecond
	if avg != wantMs {
		t.Errorf("expected avg 20ms, got %v", avg)
	}
}

// TestLatencyMonitor_P99 verifies P99 returns a value >= avg.
func TestLatencyMonitor_P99(t *testing.T) {
	lm := NewFeedLatencyMonitor()
	base := time.Now()

	for i := 1; i <= 100; i++ {
		lm.Record("ETH", base, base.Add(time.Duration(i)*time.Millisecond))
	}

	p99 := lm.P99Latency("ETH")
	avg := lm.AvgLatency("ETH")
	if p99 < avg {
		t.Errorf("P99 (%v) should be >= avg (%v)", p99, avg)
	}
	// P99 of 1..100ms should be around 99ms.
	if p99 < 95*time.Millisecond {
		t.Errorf("expected P99 near 99ms, got %v", p99)
	}
}

// TestLatencyMonitor_ClockSkew verifies negative latency is clamped to zero.
func TestLatencyMonitor_ClockSkew(t *testing.T) {
	lm := NewFeedLatencyMonitor()
	now := time.Now()
	// received before bar (clock skew) -- should clamp to 0.
	lm.Record("SOL", now.Add(10*time.Millisecond), now)

	if lm.AvgLatency("SOL") != 0 {
		t.Errorf("expected 0 for clock-skewed record, got %v", lm.AvgLatency("SOL"))
	}
}

// TestLatencyMonitor_NoSamples returns zero for unknown symbol.
func TestLatencyMonitor_NoSamples(t *testing.T) {
	lm := NewFeedLatencyMonitor()
	if lm.AvgLatency("UNKNOWN") != 0 {
		t.Error("expected 0 for unknown symbol avg")
	}
	if lm.P99Latency("UNKNOWN") != 0 {
		t.Error("expected 0 for unknown symbol P99")
	}
}

// TestAllStatuses returns all tracked feeds.
func TestAllStatuses(t *testing.T) {
	m := NewFeedHealthMonitor()
	m.ReportBar("AAPL", "1h", time.Now())
	m.ReportBar("MSFT", "1d", time.Now())

	all := m.AllStatuses()
	if len(all) != 2 {
		t.Errorf("expected 2 statuses, got %d", len(all))
	}
}
