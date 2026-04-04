package cache

import (
	"container/list"
	"sync"
)

// lruNode is an entry in the LRU cache.
type lruNode struct {
	key   string
	value interface{}
}

// LRUCache is a thread-safe LRU cache with a fixed capacity.
type LRUCache struct {
	mu       sync.Mutex
	cap      int
	list     *list.List
	items    map[string]*list.Element
	onEvict  func(key string, value interface{})
}

// NewLRUCache creates an LRUCache with the given capacity.
// onEvict is called (with the lock released) whenever an entry is evicted.
func NewLRUCache(capacity int, onEvict func(key string, value interface{})) *LRUCache {
	if capacity <= 0 {
		capacity = 128
	}
	return &LRUCache{
		cap:     capacity,
		list:    list.New(),
		items:   make(map[string]*list.Element),
		onEvict: onEvict,
	}
}

// Set inserts or updates a key-value pair.
func (c *LRUCache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.items[key]; ok {
		c.list.MoveToFront(elem)
		elem.Value.(*lruNode).value = value
		return
	}

	elem := c.list.PushFront(&lruNode{key: key, value: value})
	c.items[key] = elem

	if c.list.Len() > c.cap {
		c.evictOldest()
	}
}

// Get retrieves a value by key, returning (value, true) or (nil, false).
func (c *LRUCache) Get(key string) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.items[key]
	if !ok {
		return nil, false
	}
	c.list.MoveToFront(elem)
	return elem.Value.(*lruNode).value, true
}

// Delete removes a key from the cache.
func (c *LRUCache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if elem, ok := c.items[key]; ok {
		c.removeElement(elem)
	}
}

// Keys returns all cached keys in most-recently-used order.
func (c *LRUCache) Keys() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	keys := make([]string, 0, c.list.Len())
	for elem := c.list.Front(); elem != nil; elem = elem.Next() {
		keys = append(keys, elem.Value.(*lruNode).key)
	}
	return keys
}

// Len returns the number of items in the cache.
func (c *LRUCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.list.Len()
}

// Clear removes all items.
func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.list.Init()
	c.items = make(map[string]*list.Element)
}

// evictOldest removes the least-recently-used element.
// Caller must hold the lock.
func (c *LRUCache) evictOldest() {
	elem := c.list.Back()
	if elem != nil {
		node := c.removeElement(elem)
		if c.onEvict != nil {
			go c.onEvict(node.key, node.value)
		}
	}
}

// removeElement removes an element from the list and map.
// Returns the node. Caller must hold the lock.
func (c *LRUCache) removeElement(elem *list.Element) *lruNode {
	c.list.Remove(elem)
	node := elem.Value.(*lruNode)
	delete(c.items, node.key)
	return node
}

// ForEach iterates over all cache entries in MRU order.
// The iteration function receives (key, value); return false to stop.
func (c *LRUCache) ForEach(fn func(key string, value interface{}) bool) {
	c.mu.Lock()
	snapshot := make([]*lruNode, 0, c.list.Len())
	for elem := c.list.Front(); elem != nil; elem = elem.Next() {
		snapshot = append(snapshot, elem.Value.(*lruNode))
	}
	c.mu.Unlock()

	for _, node := range snapshot {
		if !fn(node.key, node.value) {
			break
		}
	}
}

// Contains returns true if the key is in the cache (without updating recency).
func (c *LRUCache) Contains(key string) bool {
	c.mu.Lock()
	_, ok := c.items[key]
	c.mu.Unlock()
	return ok
}

// Peek returns the value for a key without updating recency.
func (c *LRUCache) Peek(key string) (interface{}, bool) {
	c.mu.Lock()
	elem, ok := c.items[key]
	c.mu.Unlock()
	if !ok {
		return nil, false
	}
	return elem.Value.(*lruNode).value, true
}

// Resize changes the cache capacity, evicting as needed.
func (c *LRUCache) Resize(newCap int) {
	if newCap <= 0 {
		return
	}
	c.mu.Lock()
	c.cap = newCap
	for c.list.Len() > c.cap {
		c.evictOldest()
	}
	c.mu.Unlock()
}
