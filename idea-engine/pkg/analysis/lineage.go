package analysis

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ---------------------------------------------------------------------------
// LineageNode -- one node in the genome DAG
// ---------------------------------------------------------------------------

// LineageNode represents one individual in the lineage DAG.
type LineageNode struct {
	// ID is the unique genome identifier.
	ID string
	// Generation is which generation this individual belongs to.
	Generation int
	// ParentIDs are the IDs of the parent genomes (one for mutation,
	// two for crossover, zero for generation 0).
	ParentIDs []string
	// FitnessScore is the weighted composite fitness score.
	FitnessScore float64
	// Sharpe is the Sharpe ratio from the backtest evaluation.
	Sharpe float64
	// IsBreakthrough is set by FindBreakthrough.
	IsBreakthrough bool
	// CreatedAt is the wall-clock time the genome was created.
	CreatedAt time.Time
}

// ---------------------------------------------------------------------------
// LineageGraph -- directed acyclic graph of genome ancestry
// ---------------------------------------------------------------------------

// LineageGraph stores the full ancestry DAG for one evolution run.
// Nodes are keyed by genome ID; edges point from parent to child.
type LineageGraph struct {
	mu       sync.RWMutex
	nodes    map[string]*LineageNode
	children map[string][]string // parentID -> []childID
}

// NewLineageGraph creates an empty LineageGraph.
func NewLineageGraph() *LineageGraph {
	return &LineageGraph{
		nodes:    make(map[string]*LineageNode),
		children: make(map[string][]string),
	}
}

// AddNode inserts a genome into the graph. If a node with the same ID already
// exists it is overwritten.
func (g *LineageGraph) AddNode(node LineageNode) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.nodes[node.ID] = &node
	for _, pid := range node.ParentIDs {
		g.children[pid] = appendUnique(g.children[pid], node.ID)
	}
}

// Node retrieves a node by ID, returning (node, true) or (zero, false).
func (g *LineageGraph) Node(id string) (LineageNode, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	n, ok := g.nodes[id]
	if !ok {
		return LineageNode{}, false
	}
	return *n, true
}

// NodeCount returns the number of nodes in the graph.
func (g *LineageGraph) NodeCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.nodes)
}

// AllNodes returns a snapshot of all nodes in the graph.
func (g *LineageGraph) AllNodes() []LineageNode {
	g.mu.RLock()
	defer g.mu.RUnlock()
	out := make([]LineageNode, 0, len(g.nodes))
	for _, n := range g.nodes {
		out = append(out, *n)
	}
	return out
}

// Children returns the IDs of all direct children of the given node.
func (g *LineageGraph) Children(id string) []string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	c := g.children[id]
	out := make([]string, len(c))
	copy(out, c)
	return out
}

// appendUnique appends s to slice only if it is not already present.
func appendUnique(slice []string, s string) []string {
	for _, v := range slice {
		if v == s {
			return slice
		}
	}
	return append(slice, s)
}

// ---------------------------------------------------------------------------
// GenomeLineage -- tracks parent-child relationships for one run
// ---------------------------------------------------------------------------

// GenomeLineage manages the full lifecycle of lineage tracking: it receives
// new genomes, maintains the in-memory DAG, and persists records to SQLite
// for post-run analysis.
type GenomeLineage struct {
	graph *LineageGraph
	db    *sql.DB
	mu    sync.Mutex
}

// NewGenomeLineage opens (or creates) the SQLite database at dbPath and
// returns a GenomeLineage ready for use. Call Close() when done.
func NewGenomeLineage(dbPath string) (*GenomeLineage, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("NewGenomeLineage: open db: %w", err)
	}
	if err := initLineageDB(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("NewGenomeLineage: init schema: %w", err)
	}
	gl := &GenomeLineage{
		graph: NewLineageGraph(),
		db:    db,
	}
	if err := gl.loadFromDB(); err != nil {
		// Non-fatal: proceed with empty in-memory graph.
		// In production this would be logged.
		_ = err
	}
	return gl, nil
}

// initLineageDB creates the lineage table if it does not already exist.
func initLineageDB(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS genome_lineage (
			id            TEXT PRIMARY KEY,
			generation    INTEGER NOT NULL,
			parent_ids    TEXT NOT NULL DEFAULT '[]',
			fitness_score REAL NOT NULL DEFAULT 0,
			sharpe        REAL NOT NULL DEFAULT 0,
			is_breakthrough INTEGER NOT NULL DEFAULT 0,
			created_at    TEXT NOT NULL
		)
	`)
	return err
}

// Record adds a genome to the in-memory DAG and persists it to SQLite.
func (gl *GenomeLineage) Record(node LineageNode) error {
	gl.mu.Lock()
	defer gl.mu.Unlock()

	gl.graph.AddNode(node)

	parentJSON, err := json.Marshal(node.ParentIDs)
	if err != nil {
		return fmt.Errorf("GenomeLineage.Record: marshal parent_ids: %w", err)
	}
	_, err = gl.db.Exec(`
		INSERT INTO genome_lineage (id, generation, parent_ids, fitness_score, sharpe, is_breakthrough, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			fitness_score   = excluded.fitness_score,
			sharpe          = excluded.sharpe,
			is_breakthrough = excluded.is_breakthrough
	`,
		node.ID,
		node.Generation,
		string(parentJSON),
		node.FitnessScore,
		node.Sharpe,
		boolToInt(node.IsBreakthrough),
		node.CreatedAt.UTC().Format(time.RFC3339Nano),
	)
	if err != nil {
		return fmt.Errorf("GenomeLineage.Record: upsert: %w", err)
	}
	return nil
}

// Graph returns the underlying LineageGraph.
func (gl *GenomeLineage) Graph() *LineageGraph {
	return gl.graph
}

// Close closes the SQLite connection.
func (gl *GenomeLineage) Close() error {
	return gl.db.Close()
}

// loadFromDB populates the in-memory graph from all rows in the database.
func (gl *GenomeLineage) loadFromDB() error {
	rows, err := gl.db.Query(`
		SELECT id, generation, parent_ids, fitness_score, sharpe, is_breakthrough, created_at
		FROM genome_lineage
	`)
	if err != nil {
		return fmt.Errorf("loadFromDB: query: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var node LineageNode
		var parentJSON string
		var isBreakthrough int
		var createdAtStr string

		if err := rows.Scan(
			&node.ID,
			&node.Generation,
			&parentJSON,
			&node.FitnessScore,
			&node.Sharpe,
			&isBreakthrough,
			&createdAtStr,
		); err != nil {
			return fmt.Errorf("loadFromDB: scan: %w", err)
		}
		if err := json.Unmarshal([]byte(parentJSON), &node.ParentIDs); err != nil {
			node.ParentIDs = nil
		}
		node.IsBreakthrough = isBreakthrough != 0
		t, err := time.Parse(time.RFC3339Nano, createdAtStr)
		if err == nil {
			node.CreatedAt = t
		}
		gl.graph.AddNode(node)
	}
	return rows.Err()
}

// boolToInt converts a bool to 0 or 1 for SQLite storage.
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// ---------------------------------------------------------------------------
// FindBreakthrough -- identify genomes that caused the largest fitness jumps
// ---------------------------------------------------------------------------

// BreakthroughEvent records one genome that caused a significant fitness jump.
type BreakthroughEvent struct {
	// Node is the genome that achieved the breakthrough.
	Node LineageNode
	// ParentBestFitness is the best fitness among the node's parents.
	ParentBestFitness float64
	// FitnessJump is Node.FitnessScore - ParentBestFitness.
	FitnessJump float64
}

// FindBreakthrough scans the lineage graph and identifies genomes that
// produced the largest absolute fitness jumps relative to their parents.
// Only jumps above minJump are reported. Results are sorted by FitnessJump
// descending. topN controls how many to return (0 = all).
func FindBreakthrough(graph *LineageGraph, minJump float64, topN int) []BreakthroughEvent {
	nodes := graph.AllNodes()
	events := make([]BreakthroughEvent, 0)

	for _, node := range nodes {
		if len(node.ParentIDs) == 0 {
			continue // generation 0 -- no parents to compare against
		}
		// Find the best fitness among this node's parents.
		parentBest := math.Inf(-1)
		for _, pid := range node.ParentIDs {
			p, ok := graph.Node(pid)
			if !ok {
				continue
			}
			if p.FitnessScore > parentBest {
				parentBest = p.FitnessScore
			}
		}
		if math.IsInf(parentBest, -1) {
			continue // no parent data available
		}
		jump := node.FitnessScore - parentBest
		if jump >= minJump {
			events = append(events, BreakthroughEvent{
				Node:              node,
				ParentBestFitness: parentBest,
				FitnessJump:       jump,
			})
		}
	}

	// Sort by jump descending.
	sort.Slice(events, func(i, j int) bool {
		return events[i].FitnessJump > events[j].FitnessJump
	})

	// Mark those nodes as breakthroughs in the graph.
	graph.mu.Lock()
	for _, ev := range events {
		if n, ok := graph.nodes[ev.Node.ID]; ok {
			n.IsBreakthrough = true
		}
	}
	graph.mu.Unlock()

	if topN > 0 && len(events) > topN {
		events = events[:topN]
	}
	return events
}

// MarkBreakthroughs calls FindBreakthrough and persists the updated
// is_breakthrough flags back to the SQLite database.
func (gl *GenomeLineage) MarkBreakthroughs(minJump float64, topN int) ([]BreakthroughEvent, error) {
	events := FindBreakthrough(gl.graph, minJump, topN)

	// Persist the breakthrough flags.
	tx, err := gl.db.Begin()
	if err != nil {
		return events, fmt.Errorf("MarkBreakthroughs: begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	stmt, err := tx.Prepare(`UPDATE genome_lineage SET is_breakthrough = 1 WHERE id = ?`)
	if err != nil {
		return events, fmt.Errorf("MarkBreakthroughs: prepare: %w", err)
	}
	defer stmt.Close()

	for _, ev := range events {
		if _, err := stmt.Exec(ev.Node.ID); err != nil {
			return events, fmt.Errorf("MarkBreakthroughs: update %s: %w", ev.Node.ID, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return events, fmt.Errorf("MarkBreakthroughs: commit: %w", err)
	}
	return events, nil
}

// QueryBreakthroughs returns all nodes marked as breakthroughs from the
// SQLite database, ordered by fitness_score descending.
func (gl *GenomeLineage) QueryBreakthroughs() ([]LineageNode, error) {
	rows, err := gl.db.Query(`
		SELECT id, generation, parent_ids, fitness_score, sharpe, is_breakthrough, created_at
		FROM genome_lineage
		WHERE is_breakthrough = 1
		ORDER BY fitness_score DESC
	`)
	if err != nil {
		return nil, fmt.Errorf("QueryBreakthroughs: %w", err)
	}
	defer rows.Close()

	var result []LineageNode
	for rows.Next() {
		var node LineageNode
		var parentJSON string
		var isBreakthrough int
		var createdAtStr string
		if err := rows.Scan(
			&node.ID,
			&node.Generation,
			&parentJSON,
			&node.FitnessScore,
			&node.Sharpe,
			&isBreakthrough,
			&createdAtStr,
		); err != nil {
			return nil, fmt.Errorf("QueryBreakthroughs: scan: %w", err)
		}
		_ = json.Unmarshal([]byte(parentJSON), &node.ParentIDs)
		node.IsBreakthrough = isBreakthrough != 0
		if t, err := time.Parse(time.RFC3339Nano, createdAtStr); err == nil {
			node.CreatedAt = t
		}
		result = append(result, node)
	}
	return result, rows.Err()
}

// GenerationSummary returns per-generation statistics from the SQLite
// database: count of genomes, best fitness, and count of breakthroughs.
type GenerationSummary struct {
	Generation     int
	Count          int
	BestFitness    float64
	Breakthroughs  int
}

// GenerationSummaries returns a slice of GenerationSummary ordered by
// generation ascending.
func (gl *GenomeLineage) GenerationSummaries() ([]GenerationSummary, error) {
	rows, err := gl.db.Query(`
		SELECT generation,
		       COUNT(*)                          AS cnt,
		       MAX(fitness_score)                AS best_fitness,
		       SUM(is_breakthrough)              AS breakthroughs
		FROM genome_lineage
		GROUP BY generation
		ORDER BY generation ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("GenerationSummaries: %w", err)
	}
	defer rows.Close()

	var result []GenerationSummary
	for rows.Next() {
		var s GenerationSummary
		if err := rows.Scan(&s.Generation, &s.Count, &s.BestFitness, &s.Breakthroughs); err != nil {
			return nil, fmt.Errorf("GenerationSummaries: scan: %w", err)
		}
		result = append(result, s)
	}
	return result, rows.Err()
}
