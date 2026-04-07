package fitness_cache

import (
	"time"
)

// CacheStats holds snapshot metrics for a DistributedFitnessCache.
type CacheStats struct {
	// HitRate is the fraction of lookups that returned a fresh hit.
	HitRate float64
	// Hits is the cumulative number of cache hits.
	Hits uint64
	// Misses is the cumulative number of cache misses.
	Misses uint64
	// AvgLookupMs is the rolling average lookup latency in milliseconds.
	AvgLookupMs float64
	// DbSize is the approximate number of rows in the SQLite table.
	DbSize int64
	// LRUSize is the current number of entries in the in-memory LRU.
	LRUSize int
}

// Stats returns a point-in-time snapshot of cache performance metrics.
func (c *DistributedFitnessCache) Stats() CacheStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	total := c.statsHits + c.statsMisses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(c.statsHits) / float64(total)
	}

	avgMs := 0.0
	if len(c.lookupDurations) > 0 {
		var sum time.Duration
		for _, d := range c.lookupDurations {
			sum += d
		}
		avgMs = float64(sum.Nanoseconds()) / float64(len(c.lookupDurations)) / 1e6
	}

	var dbRows int64
	row := c.db.QueryRow(`SELECT COUNT(*) FROM fitness_cache`)
	_ = row.Scan(&dbRows)

	lruSize := len(c.lru.items)

	return CacheStats{
		HitRate:     hitRate,
		Hits:        c.statsHits,
		Misses:      c.statsMisses,
		AvgLookupMs: avgMs,
		DbSize:      dbRows,
		LRUSize:     lruSize,
	}
}

// Cleanup deletes cache entries older than olderThan and returns the number
// of rows deleted.
func (c *DistributedFitnessCache) Cleanup(olderThan time.Duration) int {
	cutoff := time.Now().Add(-olderThan).UTC().Format(time.RFC3339Nano)

	c.mu.Lock()
	defer c.mu.Unlock()

	res, err := c.db.Exec(`DELETE FROM fitness_cache WHERE eval_time < ?`, cutoff)
	if err != nil {
		return 0
	}
	n, _ := res.RowsAffected()

	// Evict any stale entries from the LRU. We do a full scan of the LRU
	// because there is no TTL index on the in-memory layer.
	evictKeys := make([]string, 0)
	for key, node := range c.lru.items {
		if time.Since(node.val.EvalTime) > olderThan {
			evictKeys = append(evictKeys, key)
		}
	}
	for _, k := range evictKeys {
		if node, ok := c.lru.items[k]; ok {
			c.lru.remove(node)
			delete(c.lru.items, k)
		}
	}

	return int(n)
}

// ResetStats zeroes out the hit/miss counters and lookup duration history.
func (c *DistributedFitnessCache) ResetStats() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.statsHits = 0
	c.statsMisses = 0
	c.lookupDurations = c.lookupDurations[:0]
}
