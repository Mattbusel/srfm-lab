// Package persistence handles SQLite-backed storage for the IAE genome
// evolution system. It stores genome records, lineage trees, and fitness
// histories and provides utilities for pruning old data.
package persistence

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ---------------------------------------------------------------------------
// GenomeRecord
// ---------------------------------------------------------------------------

// GenomeRecord is one genome's stored state. It captures the full gene
// vector, the fitness score, the IDs of its parents, and the genetic
// operator that created it.
type GenomeRecord struct {
	ID         string
	Generation int
	Genes      []float64
	Fitness    float64
	ParentIDs  []string
	Operator   string // "crossover", "mutation", or "elite"
	CreatedAt  time.Time
}

// ---------------------------------------------------------------------------
// GenomeStore
// ---------------------------------------------------------------------------

// GenomeStore handles all SQLite persistence for the IAE.
// All public methods are safe for use from multiple goroutines provided the
// underlying *sql.DB allows concurrent access (use WAL mode in practice).
type GenomeStore struct {
	DB *sql.DB
}

// NewGenomeStore creates a GenomeStore and runs the schema migration.
func NewGenomeStore(db *sql.DB) (*GenomeStore, error) {
	gs := &GenomeStore{DB: db}
	if err := gs.migrate(); err != nil {
		return nil, fmt.Errorf("GenomeStore migrate: %w", err)
	}
	return gs, nil
}

// migrate creates the genomes table if it does not already exist.
//
// Schema:
//
//	id          TEXT PRIMARY KEY
//	generation  INTEGER  -- which generation this individual belongs to
//	genes       BLOB     -- JSON-encoded []float64
//	fitness     REAL     -- composite fitness score
//	parent_ids  TEXT     -- JSON-encoded []string
//	operator    TEXT     -- "crossover", "mutation", "elite"
//	created_at  INTEGER  -- Unix timestamp (seconds)
func (gs *GenomeStore) migrate() error {
	_, err := gs.DB.Exec(`
		CREATE TABLE IF NOT EXISTS genomes (
			id          TEXT    PRIMARY KEY,
			generation  INTEGER NOT NULL,
			genes       BLOB    NOT NULL,
			fitness     REAL    NOT NULL,
			parent_ids  TEXT    NOT NULL DEFAULT '[]',
			operator    TEXT    NOT NULL DEFAULT '',
			created_at  INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_genomes_generation ON genomes(generation);
		CREATE INDEX IF NOT EXISTS idx_genomes_fitness     ON genomes(fitness DESC);
	`)
	return err
}

// SaveGenome inserts or replaces a GenomeRecord in the database.
func (gs *GenomeStore) SaveGenome(g GenomeRecord) error {
	genesJSON, err := json.Marshal(g.Genes)
	if err != nil {
		return fmt.Errorf("SaveGenome marshal genes: %w", err)
	}
	parentJSON, err := json.Marshal(g.ParentIDs)
	if err != nil {
		return fmt.Errorf("SaveGenome marshal parents: %w", err)
	}

	_, err = gs.DB.Exec(`
		INSERT OR REPLACE INTO genomes (id, generation, genes, fitness, parent_ids, operator, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`,
		g.ID,
		g.Generation,
		genesJSON,
		g.Fitness,
		string(parentJSON),
		g.Operator,
		g.CreatedAt.Unix(),
	)
	if err != nil {
		return fmt.Errorf("SaveGenome insert: %w", err)
	}
	return nil
}

// LoadGeneration returns all GenomeRecords belonging to the given generation,
// sorted by fitness descending. Returns an empty slice if none exist.
func (gs *GenomeStore) LoadGeneration(gen int) []GenomeRecord {
	rows, err := gs.DB.Query(`
		SELECT id, generation, genes, fitness, parent_ids, operator, created_at
		FROM genomes
		WHERE generation = ?
		ORDER BY fitness DESC
	`, gen)
	if err != nil {
		return nil
	}
	defer rows.Close()
	return gs.scanRows(rows)
}

// GetLineage traverses the parent_ids graph up to depth levels starting from
// the genome with the given id. Returns a flat slice of GenomeRecords in
// breadth-first order (the genome itself is first). If depth is 0 only the
// root genome is returned.
func (gs *GenomeStore) GetLineage(id string, depth int) []GenomeRecord {
	seen := make(map[string]bool)
	var result []GenomeRecord
	queue := []string{id}

	for level := 0; level <= depth && len(queue) > 0; level++ {
		var nextQueue []string
		for _, qid := range queue {
			if seen[qid] {
				continue
			}
			seen[qid] = true
			rec := gs.loadByID(qid)
			if rec == nil {
				continue
			}
			result = append(result, *rec)
			for _, pid := range rec.ParentIDs {
				if !seen[pid] {
					nextQueue = append(nextQueue, pid)
				}
			}
		}
		queue = nextQueue
	}
	return result
}

// GetFitnessHistory returns the fitness scores of the genome with the given
// id across all generations in which it appears, ordered by generation asc.
// In practice a genome exists in only one generation, so this is most useful
// for genomes that are carried over as elites across multiple generations
// with the same id.
func (gs *GenomeStore) GetFitnessHistory(id string) []float64 {
	rows, err := gs.DB.Query(`
		SELECT fitness FROM genomes WHERE id = ? ORDER BY generation ASC
	`, id)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var history []float64
	for rows.Next() {
		var f float64
		if err := rows.Scan(&f); err != nil {
			continue
		}
		history = append(history, f)
	}
	return history
}

// PruneOldGenerations deletes all genomes whose generation number is older
// than the keepLast most recent generations. It identifies the current
// maximum generation and removes anything below (max - keepLast).
// Returns the number of rows deleted.
func (gs *GenomeStore) PruneOldGenerations(keepLast int) (int64, error) {
	if keepLast <= 0 {
		return 0, fmt.Errorf("PruneOldGenerations: keepLast must be > 0")
	}
	var maxGen sql.NullInt64
	if err := gs.DB.QueryRow(`SELECT MAX(generation) FROM genomes`).Scan(&maxGen); err != nil {
		return 0, fmt.Errorf("PruneOldGenerations query max: %w", err)
	}
	if !maxGen.Valid {
		return 0, nil // table is empty
	}
	cutoff := int(maxGen.Int64) - keepLast
	if cutoff <= 0 {
		return 0, nil // nothing old enough to prune
	}
	result, err := gs.DB.Exec(`DELETE FROM genomes WHERE generation < ?`, cutoff)
	if err != nil {
		return 0, fmt.Errorf("PruneOldGenerations delete: %w", err)
	}
	return result.RowsAffected()
}

// ComputeGeneticDiversity returns the mean pairwise Euclidean distance between
// all genomes in the given generation. Returns 0.0 if fewer than 2 exist.
func (gs *GenomeStore) ComputeGeneticDiversity(gen int) float64 {
	records := gs.LoadGeneration(gen)
	n := len(records)
	if n < 2 {
		return 0.0
	}
	genomes := make([][]float64, n)
	for i, r := range records {
		genomes[i] = r.Genes
	}
	total := 0.0
	count := 0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			total += l2Distance(genomes[i], genomes[j])
			count++
		}
	}
	if count == 0 {
		return 0.0
	}
	return total / float64(count)
}

// ListGenerations returns a sorted list of all distinct generation numbers
// present in the database.
func (gs *GenomeStore) ListGenerations() []int {
	rows, err := gs.DB.Query(`SELECT DISTINCT generation FROM genomes ORDER BY generation ASC`)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var gens []int
	for rows.Next() {
		var g int
		if err := rows.Scan(&g); err != nil {
			continue
		}
		gens = append(gens, g)
	}
	return gens
}

// GetTopN returns the top n genomes by fitness across all generations.
func (gs *GenomeStore) GetTopN(n int) []GenomeRecord {
	rows, err := gs.DB.Query(`
		SELECT id, generation, genes, fitness, parent_ids, operator, created_at
		FROM genomes
		ORDER BY fitness DESC
		LIMIT ?
	`, n)
	if err != nil {
		return nil
	}
	defer rows.Close()
	return gs.scanRows(rows)
}

// GetByID returns the GenomeRecord with the given id, or nil if not found.
func (gs *GenomeStore) GetByID(id string) *GenomeRecord {
	return gs.loadByID(id)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// loadByID fetches a single record by primary key.
func (gs *GenomeStore) loadByID(id string) *GenomeRecord {
	rows, err := gs.DB.Query(`
		SELECT id, generation, genes, fitness, parent_ids, operator, created_at
		FROM genomes WHERE id = ? LIMIT 1
	`, id)
	if err != nil {
		return nil
	}
	defer rows.Close()
	recs := gs.scanRows(rows)
	if len(recs) == 0 {
		return nil
	}
	return &recs[0]
}

// scanRows reads GenomeRecords from an open *sql.Rows cursor.
func (gs *GenomeStore) scanRows(rows *sql.Rows) []GenomeRecord {
	var out []GenomeRecord
	for rows.Next() {
		var r GenomeRecord
		var genesJSON []byte
		var parentJSON string
		var createdAt int64
		if err := rows.Scan(
			&r.ID, &r.Generation, &genesJSON, &r.Fitness,
			&parentJSON, &r.Operator, &createdAt,
		); err != nil {
			continue
		}
		if err := json.Unmarshal(genesJSON, &r.Genes); err != nil {
			continue
		}
		// parent_ids may be stored as a space-separated list or JSON array
		if strings.HasPrefix(strings.TrimSpace(parentJSON), "[") {
			_ = json.Unmarshal([]byte(parentJSON), &r.ParentIDs)
		} else if parentJSON != "" {
			r.ParentIDs = strings.Fields(parentJSON)
		}
		r.CreatedAt = time.Unix(createdAt, 0)
		out = append(out, r)
	}
	return out
}

// l2Distance computes the L2 distance between two gene vectors.
func l2Distance(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}
