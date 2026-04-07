// Package health -- latency_monitor.go measures bar delivery latency per
// symbol and provides average and P99 latency queries.
package health

import (
	"sort"
	"sync"
	"time"
)

const (
	// maxSamples is the maximum number of latency samples retained per symbol.
	// P99 accuracy requires at least 100 samples; 1000 is a comfortable buffer.
	maxSamples = 1000
)

// latencySamples holds a rolling window of latency measurements for one symbol.
type latencySamples struct {
	values []time.Duration // ring-buffer style; newest appended to end
}

func (s *latencySamples) add(d time.Duration) {
	s.values = append(s.values, d)
	if len(s.values) > maxSamples {
		s.values = s.values[len(s.values)-maxSamples:]
	}
}

func (s *latencySamples) avg() time.Duration {
	if len(s.values) == 0 {
		return 0
	}
	var sum time.Duration
	for _, v := range s.values {
		sum += v
	}
	return sum / time.Duration(len(s.values))
}

func (s *latencySamples) p99() time.Duration {
	if len(s.values) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(s.values))
	copy(sorted, s.values)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(float64(len(sorted)) * 0.99)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// FeedLatencyMonitor measures the latency between bar production (bar_ts) and
// receipt at the consumer (received_ts) per symbol.
type FeedLatencyMonitor struct {
	mu      sync.RWMutex
	samples map[string]*latencySamples
}

// NewFeedLatencyMonitor creates an empty FeedLatencyMonitor.
func NewFeedLatencyMonitor() *FeedLatencyMonitor {
	return &FeedLatencyMonitor{
		samples: make(map[string]*latencySamples),
	}
}

// Record stores one latency observation for symbol.
// barTs is the bar's own timestamp (when it was produced at the exchange);
// receivedTs is when the local system received it.
func (m *FeedLatencyMonitor) Record(symbol string, barTs time.Time, receivedTs time.Time) {
	latency := receivedTs.Sub(barTs)
	if latency < 0 {
		latency = 0 // clock skew guard
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	s, ok := m.samples[symbol]
	if !ok {
		s = &latencySamples{}
		m.samples[symbol] = s
	}
	s.add(latency)
}

// AvgLatency returns the mean delivery latency for symbol.
// Returns 0 if no samples have been recorded.
func (m *FeedLatencyMonitor) AvgLatency(symbol string) time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s, ok := m.samples[symbol]
	if !ok {
		return 0
	}
	return s.avg()
}

// P99Latency returns the 99th-percentile delivery latency for symbol.
// Returns 0 if no samples have been recorded.
func (m *FeedLatencyMonitor) P99Latency(symbol string) time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s, ok := m.samples[symbol]
	if !ok {
		return 0
	}
	return s.p99()
}

// SampleCount returns the number of latency samples recorded for symbol.
func (m *FeedLatencyMonitor) SampleCount(symbol string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s, ok := m.samples[symbol]
	if !ok {
		return 0
	}
	return len(s.values)
}

// Symbols returns all symbols for which samples have been recorded.
func (m *FeedLatencyMonitor) Symbols() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]string, 0, len(m.samples))
	for sym := range m.samples {
		out = append(out, sym)
	}
	return out
}
