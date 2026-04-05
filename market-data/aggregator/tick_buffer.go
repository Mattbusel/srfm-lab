package aggregator

import (
	"sync"
)

// TickBuffer is a fixed-size ring buffer for raw ticks per symbol.
// Lock-free reads (via snapshot), mutex writes.
type TickBuffer struct {
	mu       sync.Mutex
	data     []RawTick
	size     int
	head     int // points to next write position
	count    int
}

// NewTickBuffer creates a ring buffer with the given capacity.
func NewTickBuffer(size int) *TickBuffer {
	return &TickBuffer{
		data: make([]RawTick, size),
		size: size,
	}
}

// Push adds a tick to the buffer. Overwrites oldest when full.
func (b *TickBuffer) Push(tick RawTick) {
	b.mu.Lock()
	b.data[b.head] = tick
	b.head = (b.head + 1) % b.size
	if b.count < b.size {
		b.count++
	}
	b.mu.Unlock()
}

// Snapshot returns a copy of all ticks for the given symbol (most recent last).
// Lock-free from caller perspective: we briefly hold the lock only to copy.
func (b *TickBuffer) Snapshot(symbol string) []RawTick {
	b.mu.Lock()
	count := b.count
	head := b.head
	dataCopy := make([]RawTick, b.size)
	copy(dataCopy, b.data)
	b.mu.Unlock()

	result := make([]RawTick, 0, 64)
	// Walk from oldest to newest
	start := (head - count + b.size) % b.size
	for i := 0; i < count; i++ {
		idx := (start + i) % b.size
		if dataCopy[idx].Symbol == symbol {
			result = append(result, dataCopy[idx])
		}
	}
	return result
}

// SnapshotAll returns all ticks regardless of symbol (most recent last).
func (b *TickBuffer) SnapshotAll() []RawTick {
	b.mu.Lock()
	count := b.count
	head := b.head
	dataCopy := make([]RawTick, b.size)
	copy(dataCopy, b.data)
	b.mu.Unlock()

	result := make([]RawTick, count)
	start := (head - count + b.size) % b.size
	for i := 0; i < count; i++ {
		result[i] = dataCopy[(start+i)%b.size]
	}
	return result
}

// LastPrice returns the most recent close price for a symbol, or 0 if none.
func (b *TickBuffer) LastPrice(symbol string) float64 {
	b.mu.Lock()
	count := b.count
	head := b.head
	dataCopy := make([]RawTick, b.size)
	copy(dataCopy, b.data)
	b.mu.Unlock()

	// Walk backwards from newest
	for i := 1; i <= count; i++ {
		idx := (head - i + b.size) % b.size
		if dataCopy[idx].Symbol == symbol {
			return dataCopy[idx].Close
		}
	}
	return 0
}

// VWAP computes the volume-weighted average price for a symbol over all buffered ticks.
func (b *TickBuffer) VWAP(symbol string) float64 {
	ticks := b.Snapshot(symbol)
	var sumPV, sumV float64
	for _, t := range ticks {
		sumPV += t.Close * t.Volume
		sumV += t.Volume
	}
	if sumV == 0 {
		return 0
	}
	return sumPV / sumV
}

// Count returns the number of stored ticks.
func (b *TickBuffer) Count() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.count
}
