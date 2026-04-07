// Package health provides feed health monitoring and latency tracking for SRFM
// market data feeds.
package health

import (
	"sync"
	"time"
)

// staleThresholds maps timeframe strings to the duration after which a feed is
// considered stale (no bar received within threshold of the expected bar time).
var staleThresholds = map[string]time.Duration{
	"1m":  2 * time.Minute,
	"5m":  7 * time.Minute,
	"15m": 20 * time.Minute,
	"1h":  65 * time.Minute,
	"4h":  4*time.Hour + 5*time.Minute,
	"1d":  25 * time.Hour,
}

// defaultStaleThreshold is used for timeframes not in staleThresholds.
const defaultStaleThreshold = 30 * time.Minute

// FeedStatus holds health metrics for one (symbol, timeframe) feed.
type FeedStatus struct {
	Symbol     string
	Timeframe  string
	LastBar    time.Time
	IsStale    bool
	GapCount   int // number of detected gaps (missed expected bars)
	ErrorCount int // number of errors reported against this feed
}

// feedKey identifies a (symbol, timeframe) pair.
type feedKey struct {
	symbol    string
	timeframe string
}

// FeedHealthMonitor tracks bar arrival times and detects stale feeds.
type FeedHealthMonitor struct {
	mu       sync.RWMutex
	statuses map[feedKey]*FeedStatus
	errors   map[feedKey]int

	// now is a replaceable clock for testing.
	now func() time.Time
}

// NewFeedHealthMonitor creates a new FeedHealthMonitor.
func NewFeedHealthMonitor() *FeedHealthMonitor {
	return &FeedHealthMonitor{
		statuses: make(map[feedKey]*FeedStatus),
		errors:   make(map[feedKey]int),
		now:      time.Now,
	}
}

// ReportBar records that a bar arrived for the given symbol and timeframe at
// the given timestamp. It updates gap detection by comparing ts to the
// expected next bar time.
func (m *FeedHealthMonitor) ReportBar(symbol string, timeframe string, ts time.Time) {
	key := feedKey{symbol, timeframe}
	m.mu.Lock()
	defer m.mu.Unlock()

	st, exists := m.statuses[key]
	if !exists {
		st = &FeedStatus{Symbol: symbol, Timeframe: timeframe}
		m.statuses[key] = st
	}

	if exists && !st.LastBar.IsZero() {
		// Gap detection: compare ts to expected next bar time.
		expectedNext := st.LastBar.Add(timeframeDuration(timeframe))
		// Allow 10% tolerance on the bar period.
		tolerance := timeframeDuration(timeframe) / 10
		if ts.After(expectedNext.Add(tolerance)) {
			st.GapCount++
		}
	}

	st.LastBar = ts
	st.IsStale = false
}

// ReportError increments the error counter for a feed.
func (m *FeedHealthMonitor) ReportError(symbol string, timeframe string) {
	key := feedKey{symbol, timeframe}
	m.mu.Lock()
	defer m.mu.Unlock()

	st, exists := m.statuses[key]
	if !exists {
		st = &FeedStatus{Symbol: symbol, Timeframe: timeframe}
		m.statuses[key] = st
	}
	st.ErrorCount++
}

// GetStatus returns a map of timeframe -> FeedStatus for the given symbol.
// Stale status is recomputed at query time.
func (m *FeedHealthMonitor) GetStatus(symbol string) map[string]FeedStatus {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := m.now()
	out := make(map[string]FeedStatus)
	for key, st := range m.statuses {
		if key.symbol != symbol {
			continue
		}
		threshold := staleThreshold(key.timeframe)
		st.IsStale = !st.LastBar.IsZero() && now.Sub(st.LastBar) > threshold
		out[key.timeframe] = *st
	}
	return out
}

// StaleFeeds returns all FeedStatus entries whose last bar is older than
// threshold (or whose built-in timeframe threshold has elapsed).
// Pass 0 to use per-timeframe defaults.
func (m *FeedHealthMonitor) StaleFeeds(threshold time.Duration) []FeedStatus {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := m.now()
	var out []FeedStatus
	for key, st := range m.statuses {
		if st.LastBar.IsZero() {
			continue
		}
		thr := threshold
		if thr == 0 {
			thr = staleThreshold(key.timeframe)
		}
		if now.Sub(st.LastBar) > thr {
			st.IsStale = true
			out = append(out, *st)
		}
	}
	return out
}

// AllStatuses returns a snapshot of all tracked feeds.
func (m *FeedHealthMonitor) AllStatuses() []FeedStatus {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := m.now()
	var out []FeedStatus
	for key, st := range m.statuses {
		thr := staleThreshold(key.timeframe)
		st.IsStale = !st.LastBar.IsZero() && now.Sub(st.LastBar) > thr
		out = append(out, *st)
	}
	return out
}

// Reset clears all tracked state (used in testing).
func (m *FeedHealthMonitor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.statuses = make(map[feedKey]*FeedStatus)
}

// staleThreshold returns the appropriate stale duration for a timeframe.
func staleThreshold(timeframe string) time.Duration {
	if d, ok := staleThresholds[timeframe]; ok {
		return d
	}
	return defaultStaleThreshold
}

// timeframeDuration returns the expected bar period for gap detection.
func timeframeDuration(tf string) time.Duration {
	switch tf {
	case "1m":
		return time.Minute
	case "5m":
		return 5 * time.Minute
	case "15m":
		return 15 * time.Minute
	case "1h":
		return time.Hour
	case "4h":
		return 4 * time.Hour
	case "1d":
		return 24 * time.Hour
	default:
		return time.Hour
	}
}
