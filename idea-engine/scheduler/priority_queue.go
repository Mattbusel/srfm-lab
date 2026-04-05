// Package scheduler implements the experiment scheduler for the Idea Automation
// Engine. It receives experiment requests from the idea-api and dispatches them
// to the appropriate runtime (Rust binary or Python module) subject to
// per-module CPU budget constraints.
package scheduler

import (
	"container/heap"
	"sync"
	"time"
)

// ExperimentItem is an element in the priority queue.
type ExperimentItem struct {
	// ExperimentID is the unique identifier of the experiment.
	ExperimentID string
	// HypothesisID links back to the originating hypothesis.
	HypothesisID string
	// ExperimentType identifies the runtime: genome, counterfactual, shadow, etc.
	ExperimentType string
	// Config is the raw JSON configuration for the experiment runner.
	Config []byte
	// Priority is the computed scheduling priority.  Lower values run first.
	// Priority = hypothesis.priority_rank * urgency_multiplier
	Priority int
	// EnqueuedAt is when the item was pushed onto the queue.
	EnqueuedAt time.Time
	// RetryCount tracks how many times this experiment has been retried.
	RetryCount int
	// index is maintained by the heap.Interface implementation.
	index int
}

// pqHeap is the underlying min-heap for ExperimentItems.
// Lower Priority values are dequeued first (min-heap).
type pqHeap []*ExperimentItem

func (h pqHeap) Len() int { return len(h) }

func (h pqHeap) Less(i, j int) bool {
	// Primary sort: lower Priority first.
	if h[i].Priority != h[j].Priority {
		return h[i].Priority < h[j].Priority
	}
	// Secondary sort: earlier enqueue time first (FIFO within same priority).
	return h[i].EnqueuedAt.Before(h[j].EnqueuedAt)
}

func (h pqHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *pqHeap) Push(x interface{}) {
	n := len(*h)
	item := x.(*ExperimentItem)
	item.index = n
	*h = append(*h, item)
}

func (h *pqHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // prevent memory leak
	item.index = -1 // mark as removed
	*h = old[:n-1]
	return item
}

// PriorityQueue is a thread-safe heap-based experiment priority queue.
// It is the single point of truth for which experiment runs next.
type PriorityQueue struct {
	mu    sync.Mutex
	items pqHeap
}

// NewPriorityQueue constructs an empty PriorityQueue.
func NewPriorityQueue() *PriorityQueue {
	pq := &PriorityQueue{}
	heap.Init(&pq.items)
	return pq
}

// Push adds an ExperimentItem to the queue.
// If EnqueuedAt is zero it is set to time.Now().
func (q *PriorityQueue) Push(item *ExperimentItem) {
	if item.EnqueuedAt.IsZero() {
		item.EnqueuedAt = time.Now().UTC()
	}
	q.mu.Lock()
	heap.Push(&q.items, item)
	q.mu.Unlock()
}

// Pop removes and returns the highest-priority (lowest Priority value) item.
// Returns nil if the queue is empty.
func (q *PriorityQueue) Pop() *ExperimentItem {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.items.Len() == 0 {
		return nil
	}
	return heap.Pop(&q.items).(*ExperimentItem)
}

// Peek returns the highest-priority item without removing it.
// Returns nil if the queue is empty.
func (q *PriorityQueue) Peek() *ExperimentItem {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.items.Len() == 0 {
		return nil
	}
	return q.items[0]
}

// Len returns the number of items currently in the queue.
func (q *PriorityQueue) Len() int {
	q.mu.Lock()
	n := q.items.Len()
	q.mu.Unlock()
	return n
}

// Remove removes the item with the given ExperimentID from the queue.
// It is a no-op if the item is not present.
func (q *PriorityQueue) Remove(experimentID string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for i, item := range q.items {
		if item.ExperimentID == experimentID {
			heap.Remove(&q.items, i)
			return true
		}
	}
	return false
}

// UpdatePriority changes the priority of an existing item and re-heapifies.
// Returns false if the item was not found.
func (q *PriorityQueue) UpdatePriority(experimentID string, newPriority int) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	for _, item := range q.items {
		if item.ExperimentID == experimentID {
			item.Priority = newPriority
			heap.Fix(&q.items, item.index)
			return true
		}
	}
	return false
}

// Snapshot returns a copy of all items in the queue without removing them.
// The order is unspecified (heap order, not sorted order).
func (q *PriorityQueue) Snapshot() []*ExperimentItem {
	q.mu.Lock()
	defer q.mu.Unlock()
	out := make([]*ExperimentItem, len(q.items))
	copy(out, q.items)
	return out
}

// ComputePriority calculates the scheduling priority for an experiment.
// priorityRank comes from the hypothesis; urgencyMultiplier boosts experiments
// that have already been waiting for a long time.
func ComputePriority(priorityRank, urgencyMultiplier int) int {
	if urgencyMultiplier <= 0 {
		urgencyMultiplier = 1
	}
	return priorityRank * urgencyMultiplier
}
