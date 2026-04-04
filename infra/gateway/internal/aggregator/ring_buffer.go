// Package aggregator provides bar aggregation and ring buffer utilities.
package aggregator

import (
	"sync"
	"time"

	"github.com/srfm/gateway/internal/feed"
)

// RingBuffer is a fixed-capacity circular buffer for bars, safe for concurrent use.
type RingBuffer struct {
	mu   sync.RWMutex
	data []feed.Bar
	cap  int
	head int // index of the oldest element
	size int // number of valid elements
}

// NewRingBuffer allocates a RingBuffer with the given capacity.
func NewRingBuffer(capacity int) *RingBuffer {
	if capacity <= 0 {
		capacity = 1000
	}
	return &RingBuffer{
		data: make([]feed.Bar, capacity),
		cap:  capacity,
	}
}

// Push appends a bar. When full, the oldest bar is overwritten.
func (rb *RingBuffer) Push(b feed.Bar) {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	if rb.size < rb.cap {
		// Buffer not yet full: write at head+size.
		idx := (rb.head + rb.size) % rb.cap
		rb.data[idx] = b
		rb.size++
	} else {
		// Full: overwrite oldest (head) and advance head.
		rb.data[rb.head] = b
		rb.head = (rb.head + 1) % rb.cap
	}
}

// Len returns the current number of bars stored.
func (rb *RingBuffer) Len() int {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	return rb.size
}

// GetLast returns the last n bars in chronological order.
// If fewer than n bars are stored, all stored bars are returned.
func (rb *RingBuffer) GetLast(n int) []feed.Bar {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	if n <= 0 || rb.size == 0 {
		return nil
	}
	if n > rb.size {
		n = rb.size
	}

	out := make([]feed.Bar, n)
	// The last n elements start at index (head + size - n) % cap.
	startIdx := (rb.head + rb.size - n + rb.cap*2) % rb.cap
	for i := 0; i < n; i++ {
		out[i] = rb.data[(startIdx+i)%rb.cap]
	}
	return out
}

// GetAll returns all stored bars in chronological order.
func (rb *RingBuffer) GetAll() []feed.Bar {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	if rb.size == 0 {
		return nil
	}
	out := make([]feed.Bar, rb.size)
	for i := 0; i < rb.size; i++ {
		out[i] = rb.data[(rb.head+i)%rb.cap]
	}
	return out
}

// GetRange returns bars with Timestamp in [from, to] (inclusive), in order.
func (rb *RingBuffer) GetRange(from, to time.Time) []feed.Bar {
	rb.mu.RLock()
	defer rb.mu.RUnlock()

	var out []feed.Bar
	for i := 0; i < rb.size; i++ {
		b := rb.data[(rb.head+i)%rb.cap]
		if !b.Timestamp.Before(from) && !b.Timestamp.After(to) {
			out = append(out, b)
		}
	}
	return out
}

// Latest returns the most recently pushed bar, or nil if empty.
func (rb *RingBuffer) Latest() *feed.Bar {
	rb.mu.RLock()
	defer rb.mu.RUnlock()
	if rb.size == 0 {
		return nil
	}
	b := rb.data[(rb.head+rb.size-1)%rb.cap]
	return &b
}

// SymbolRingBuffers maintains a RingBuffer per symbol per timeframe key.
type SymbolRingBuffers struct {
	mu       sync.RWMutex
	bufs     map[string]*RingBuffer // key = symbol+"|"+timeframe
	capacity int
}

// NewSymbolRingBuffers creates a new SymbolRingBuffers with the given per-buffer capacity.
func NewSymbolRingBuffers(capacity int) *SymbolRingBuffers {
	return &SymbolRingBuffers{
		bufs:     make(map[string]*RingBuffer),
		capacity: capacity,
	}
}

func bufKey(symbol, timeframe string) string {
	return symbol + "|" + timeframe
}

// Push stores a bar under symbol + timeframe.
func (s *SymbolRingBuffers) Push(symbol, timeframe string, b feed.Bar) {
	key := bufKey(symbol, timeframe)
	s.mu.RLock()
	buf, ok := s.bufs[key]
	s.mu.RUnlock()

	if !ok {
		s.mu.Lock()
		buf, ok = s.bufs[key]
		if !ok {
			buf = NewRingBuffer(s.capacity)
			s.bufs[key] = buf
		}
		s.mu.Unlock()
	}
	buf.Push(b)
}

// GetLast returns the last n bars for symbol + timeframe.
func (s *SymbolRingBuffers) GetLast(symbol, timeframe string, n int) []feed.Bar {
	key := bufKey(symbol, timeframe)
	s.mu.RLock()
	buf, ok := s.bufs[key]
	s.mu.RUnlock()
	if !ok {
		return nil
	}
	return buf.GetLast(n)
}

// GetRange returns bars in [from, to] for symbol + timeframe.
func (s *SymbolRingBuffers) GetRange(symbol, timeframe string, from, to time.Time) []feed.Bar {
	key := bufKey(symbol, timeframe)
	s.mu.RLock()
	buf, ok := s.bufs[key]
	s.mu.RUnlock()
	if !ok {
		return nil
	}
	return buf.GetRange(from, to)
}

// Latest returns the most recent bar for symbol + timeframe.
func (s *SymbolRingBuffers) Latest(symbol, timeframe string) *feed.Bar {
	key := bufKey(symbol, timeframe)
	s.mu.RLock()
	buf, ok := s.bufs[key]
	s.mu.RUnlock()
	if !ok {
		return nil
	}
	return buf.Latest()
}

// Symbols returns all distinct symbols known to this store.
func (s *SymbolRingBuffers) Symbols() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	seen := make(map[string]struct{})
	for k := range s.bufs {
		parts := splitKey(k)
		seen[parts[0]] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for sym := range seen {
		out = append(out, sym)
	}
	return out
}

func splitKey(key string) [2]string {
	for i, c := range key {
		if c == '|' {
			return [2]string{key[:i], key[i+1:]}
		}
	}
	return [2]string{key, ""}
}
