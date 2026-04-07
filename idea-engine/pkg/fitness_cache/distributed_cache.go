// Package fitness_cache implements a two-layer fitness cache backed by an
// in-memory LRU store and a SQLite persistence layer. Entries are keyed by
// the SHA-256 hash of the serialised parameter map.
package fitness_cache

import (
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// CacheEntry holds the cached evaluation result for a single genome.
type CacheEntry struct {
	// GenomeHash is the hex-encoded SHA-256 hash of the genome parameters.
	GenomeHash string
	// Fitness is the composite weighted fitness scalar.
	Fitness float64
	// Sharpe is the annualised Sharpe ratio.
	Sharpe float64
	// MaxDD is the maximum drawdown as a positive fraction.
	MaxDD float64
	// EvalTime is when the entry was created.
	EvalTime time.Time
	// DataFingerprint is the SHA-256 of the training data date range used
	// when this entry was computed. Entries from different data windows are
	// treated as stale.
	DataFingerprint string
}

// IsStale returns true when the entry is older than maxAge or was computed
// with a different data fingerprint than currentFingerprint.
func (e CacheEntry) IsStale(maxAge time.Duration, currentFingerprint string) bool {
	if time.Since(e.EvalTime) > maxAge {
		return true
	}
	if currentFingerprint != "" && e.DataFingerprint != currentFingerprint {
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// LRU layer
// ---------------------------------------------------------------------------

type lruNode struct {
	key  string
	val  CacheEntry
	prev *lruNode
	next *lruNode
}

type lruLayer struct {
	cap   int
	items map[string]*lruNode
	head  *lruNode // most recently used sentinel
	tail  *lruNode // least recently used sentinel
}

func newLRULayer(capacity int) *lruLayer {
	head := &lruNode{}
	tail := &lruNode{}
	head.next = tail
	tail.prev = head
	return &lruLayer{
		cap:   capacity,
		items: make(map[string]*lruNode, capacity),
		head:  head,
		tail:  tail,
	}
}

func (l *lruLayer) get(key string) (CacheEntry, bool) {
	n, ok := l.items[key]
	if !ok {
		return CacheEntry{}, false
	}
	l.remove(n)
	l.insertFront(n)
	return n.val, true
}

func (l *lruLayer) put(key string, val CacheEntry) {
	if n, ok := l.items[key]; ok {
		n.val = val
		l.remove(n)
		l.insertFront(n)
		return
	}
	n := &lruNode{key: key, val: val}
	l.items[key] = n
	l.insertFront(n)
	if len(l.items) > l.cap {
		lru := l.tail.prev
		l.remove(lru)
		delete(l.items, lru.key)
	}
}

func (l *lruLayer) remove(n *lruNode) {
	n.prev.next = n.next
	n.next.prev = n.prev
}

func (l *lruLayer) insertFront(n *lruNode) {
	n.prev = l.head
	n.next = l.head.next
	l.head.next.prev = n
	l.head.next = n
}

// ---------------------------------------------------------------------------
// DistributedFitnessCache
// ---------------------------------------------------------------------------

// DistributedFitnessCache is a two-level cache (in-memory LRU + SQLite) for
// genome fitness results.
type DistributedFitnessCache struct {
	mu sync.Mutex
	db *sql.DB
	// lru is the fast in-memory layer (up to lruCap entries).
	lru *lruLayer
	// dataFingerprint is the expected fingerprint for non-stale entries.
	dataFingerprint string
	// maxAge is the maximum age of a valid cache entry (default 7 days).
	maxAge time.Duration
	// statsHits and statsMisses are counters for Stats().
	statsHits   uint64
	statsMisses uint64
	// lookupDurations accumulates lookup times for average latency tracking.
	lookupDurations []time.Duration
}

// NewDistributedFitnessCache opens (or creates) the SQLite database at dbPath
// and returns a ready-to-use cache. lruCap sets the in-memory LRU capacity
// (0 defaults to 1000). dataFingerprint identifies the current training window.
func NewDistributedFitnessCache(
	dbPath string,
	lruCap int,
	dataFingerprint string,
) (*DistributedFitnessCache, error) {
	if lruCap <= 0 {
		lruCap = 1000
	}
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	if err := migrateSchema(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrate schema: %w", err)
	}
	c := &DistributedFitnessCache{
		db:              db,
		lru:             newLRULayer(lruCap),
		dataFingerprint: dataFingerprint,
		maxAge:          7 * 24 * time.Hour,
	}
	return c, nil
}

// Close releases the SQLite connection.
func (c *DistributedFitnessCache) Close() error {
	return c.db.Close()
}

// SetDataFingerprint updates the expected data fingerprint. Any entry with a
// different fingerprint will be treated as stale on the next lookup.
func (c *DistributedFitnessCache) SetDataFingerprint(fp string) {
	c.mu.Lock()
	c.dataFingerprint = fp
	c.mu.Unlock()
}

// Get looks up the cache entry for the given genome parameter map.
// Returns the entry and true if a fresh entry is found, or the zero value
// and false otherwise.
func (c *DistributedFitnessCache) Get(genome map[string]float64) (CacheEntry, bool) {
	start := time.Now()
	key := genomeHash(genome)

	c.mu.Lock()
	defer func() {
		elapsed := time.Since(start)
		c.lookupDurations = append(c.lookupDurations, elapsed)
		c.mu.Unlock()
	}()

	// Check LRU first.
	if entry, ok := c.lru.get(key); ok {
		if !entry.IsStale(c.maxAge, c.dataFingerprint) {
			c.statsHits++
			return entry, true
		}
		// Stale entry in LRU -- treat as miss.
	}

	// Query SQLite.
	entry, found := c.queryDB(key)
	if !found {
		c.statsMisses++
		return CacheEntry{}, false
	}
	if entry.IsStale(c.maxAge, c.dataFingerprint) {
		c.statsMisses++
		return CacheEntry{}, false
	}

	// Warm the LRU.
	c.lru.put(key, entry)
	c.statsHits++
	return entry, true
}

// Set stores a cache entry for the given genome parameter map.
func (c *DistributedFitnessCache) Set(genome map[string]float64, entry CacheEntry) {
	key := genomeHash(genome)
	entry.GenomeHash = key

	c.mu.Lock()
	defer c.mu.Unlock()

	c.lru.put(key, entry)
	c.upsertDB(entry)
}

// queryDB retrieves a CacheEntry from SQLite by genome hash. Caller must hold mu.
func (c *DistributedFitnessCache) queryDB(key string) (CacheEntry, bool) {
	row := c.db.QueryRow(`
		SELECT genome_hash, fitness, sharpe, max_dd, eval_time, data_fingerprint
		FROM fitness_cache
		WHERE genome_hash = ?
	`, key)

	var e CacheEntry
	var evalTimeStr string
	err := row.Scan(&e.GenomeHash, &e.Fitness, &e.Sharpe, &e.MaxDD, &evalTimeStr, &e.DataFingerprint)
	if err == sql.ErrNoRows {
		return CacheEntry{}, false
	}
	if err != nil {
		return CacheEntry{}, false
	}
	t, err := time.Parse(time.RFC3339Nano, evalTimeStr)
	if err == nil {
		e.EvalTime = t
	}
	return e, true
}

// upsertDB inserts or replaces a cache entry in SQLite. Caller must hold mu.
func (c *DistributedFitnessCache) upsertDB(entry CacheEntry) {
	_, _ = c.db.Exec(`
		INSERT OR REPLACE INTO fitness_cache
			(genome_hash, fitness, sharpe, max_dd, eval_time, data_fingerprint)
		VALUES (?, ?, ?, ?, ?, ?)
	`,
		entry.GenomeHash,
		entry.Fitness,
		entry.Sharpe,
		entry.MaxDD,
		entry.EvalTime.UTC().Format(time.RFC3339Nano),
		entry.DataFingerprint,
	)
}

// migrateSchema creates the fitness_cache table if it does not exist.
func migrateSchema(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS fitness_cache (
			genome_hash      TEXT PRIMARY KEY,
			fitness          REAL NOT NULL,
			sharpe           REAL NOT NULL,
			max_dd           REAL NOT NULL,
			eval_time        TEXT NOT NULL,
			data_fingerprint TEXT NOT NULL
		)
	`)
	if err != nil {
		return err
	}
	_, err = db.Exec(`
		CREATE INDEX IF NOT EXISTS idx_fitness_cache_eval_time
		ON fitness_cache (eval_time)
	`)
	return err
}

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

// genomeHash returns the hex-encoded SHA-256 of the parameter map.
// Parameters are serialised in deterministic sorted-key order.
func genomeHash(genome map[string]float64) string {
	keys := make([]string, 0, len(genome))
	for k := range genome {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	h := sha256.New()
	buf := make([]byte, 8)
	for _, k := range keys {
		h.Write([]byte(k))
		binary.LittleEndian.PutUint64(buf, math.Float64bits(genome[k]))
		h.Write(buf)
	}
	sum := h.Sum(nil)
	return fmt.Sprintf("%x", sum)
}

// DataFingerprint hashes a training window descriptor (e.g. "2020-01-01/2023-12-31")
// to a short hex string suitable for use as a CacheEntry.DataFingerprint.
func DataFingerprint(startDate, endDate string) string {
	payload, _ := json.Marshal([]string{startDate, endDate})
	sum := sha256.Sum256(payload)
	return fmt.Sprintf("%x", sum[:8])
}
