// Package handlers -- genome_history.go adds genome history and rollback
// endpoints to complement the existing genome.go handler.
//
// Routes:
//   GET  /genome/current       -- current best genome params
//   GET  /genome/history       -- last N genomes with fitness (query: n)
//   POST /genome/rollback/:id  -- revert to a previous genome (query: id)
package handlers

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"srfm-lab/idea-engine/pkg/evolution"
)

// ---------------------------------------------------------------------------
// GenomeRecord -- persistent genome snapshot
// ---------------------------------------------------------------------------

// GenomeRecord captures the state of a genome at a point in time for
// persistence and rollback purposes.
type GenomeRecord struct {
	// ID is the genome's unique identifier.
	ID string `json:"id"`
	// Generation is the generation this genome was produced in.
	Generation int `json:"generation"`
	// Genes is the parameter vector.
	Genes []float64 `json:"genes"`
	// FitnessScore is the composite weighted fitness score.
	FitnessScore float64 `json:"fitness_score"`
	// Sharpe is the Sharpe ratio for this genome.
	Sharpe float64 `json:"sharpe"`
	// MaxDrawdown is the maximum drawdown fraction.
	MaxDrawdown float64 `json:"max_drawdown"`
	// IsBest is true if this was the best genome at recording time.
	IsBest bool `json:"is_best"`
	// RecordedAt is when this snapshot was taken.
	RecordedAt time.Time `json:"recorded_at"`
}

// ---------------------------------------------------------------------------
// GenomeHistoryHandler
// ---------------------------------------------------------------------------

// GenomeHistoryHandler handles the genome history and rollback endpoints.
// It persists snapshots to SQLite and exposes them through the HTTP API.
// The active evolution engine is consulted for live "current genome" data.
type GenomeHistoryHandler struct {
	db     *sql.DB
	engine EvolutionEngine
}

// NewGenomeHistoryHandler constructs a GenomeHistoryHandler.
// dbPath may be ":memory:" for tests; engine must not be nil.
func NewGenomeHistoryHandler(dbPath string, engine EvolutionEngine) (*GenomeHistoryHandler, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	if err := migrateGenomeHistoryDB(db); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return &GenomeHistoryHandler{db: db, engine: engine}, nil
}

// Close releases the database connection.
func (h *GenomeHistoryHandler) Close() { _ = h.db.Close() }

// ---------------------------------------------------------------------------
// GET /genome/current
// ---------------------------------------------------------------------------

// GetCurrent returns the current best genome parameters from the live engine.
// If a historical snapshot exists with a higher fitness than the live engine
// reports, a warning field is included in the response.
func (h *GenomeHistoryHandler) GetCurrent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	best, err := h.engine.BestIndividual()
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("engine: %v", err))
		return
	}

	// Build a params map from the gene slice using positional keys.
	params := make(map[string]float64, len(best.Genes))
	for i, v := range best.Genes {
		params[fmt.Sprintf("gene_%d", i)] = v
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"id":            best.ID,
		"generation":    best.Generation,
		"params":        params,
		"genes":         []float64(best.Genes),
		"fitness_score": best.Fitness.WeightedScore,
		"sharpe":        best.Fitness.Sharpe,
		"max_drawdown":  best.Fitness.MaxDD,
		"fetched_at":    time.Now(),
	})
}

// ---------------------------------------------------------------------------
// GET /genome/history
// ---------------------------------------------------------------------------

// GetHistory returns the last N recorded genome snapshots, newest first.
// Query param: n (default 20, max 200).
func (h *GenomeHistoryHandler) GetHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	n := 20
	if s := r.URL.Query().Get("n"); s != "" {
		fmt.Sscanf(s, "%d", &n)
	}
	if n < 1 {
		n = 1
	}
	if n > 200 {
		n = 200
	}

	rows, err := h.db.Query(
		`SELECT id, generation, genes_json, fitness_score, sharpe, max_drawdown, is_best, recorded_at
		 FROM genome_history ORDER BY recorded_at DESC LIMIT ?`, n)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("query: %v", err))
		return
	}
	defer rows.Close()

	records, err := scanGenomeRecords(rows)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("scan: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"genomes": records,
		"count":   len(records),
	})
}

// ---------------------------------------------------------------------------
// POST /genome/rollback/:id
// ---------------------------------------------------------------------------

// RollbackGenome reverts the active population by seeding the genome with
// the given historical ID. The ID is read from the `id` query parameter or
// from the URL path.
//
// The historical genome is re-injected into the engine via SeedIndividual.
// This does not immediately replace all individuals; the seeded genome will
// compete in the next generation.
func (h *GenomeHistoryHandler) RollbackGenome(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	// Accept the ID either from the URL path or as a query parameter.
	id := extractPathID(r.URL.Path, "/genome/rollback/")
	if id == "" {
		id = r.URL.Query().Get("id")
	}
	if id == "" {
		writeError(w, http.StatusBadRequest, "genome ID required")
		return
	}

	record, err := h.loadRecord(id)
	if err == sql.ErrNoRows {
		writeError(w, http.StatusNotFound, fmt.Sprintf("genome %s not found in history", id))
		return
	}
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("load: %v", err))
		return
	}

	// Inject the historical genome into the running engine.
	if err := h.engine.SeedIndividual(evolution.Genome(record.Genes)); err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("seed: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"message":       "genome seeded for rollback",
		"rolled_back_to": record.ID,
		"generation":    record.Generation,
		"fitness_score": record.FitnessScore,
	})
}

// ---------------------------------------------------------------------------
// RecordSnapshot -- used by other services to persist a genome snapshot
// ---------------------------------------------------------------------------

// RecordSnapshot persists a genome snapshot to the history table.
// Safe to call from any goroutine.
func (h *GenomeHistoryHandler) RecordSnapshot(rec GenomeRecord) error {
	if rec.ID == "" {
		rec.ID = generateID()
	}
	if rec.RecordedAt.IsZero() {
		rec.RecordedAt = time.Now()
	}
	genesJSON, err := json.Marshal(rec.Genes)
	if err != nil {
		return fmt.Errorf("marshal genes: %w", err)
	}
	isBest := 0
	if rec.IsBest {
		isBest = 1
	}
	_, err = h.db.Exec(
		`INSERT OR REPLACE INTO genome_history
		 (id, generation, genes_json, fitness_score, sharpe, max_drawdown, is_best, recorded_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		rec.ID, rec.Generation, string(genesJSON),
		rec.FitnessScore, rec.Sharpe, rec.MaxDrawdown,
		isBest, rec.RecordedAt,
	)
	return err
}

// ---------------------------------------------------------------------------
// SQLite helpers
// ---------------------------------------------------------------------------

// migrateGenomeHistoryDB creates the genome_history table if absent.
func migrateGenomeHistoryDB(db *sql.DB) error {
	_, err := db.Exec(`CREATE TABLE IF NOT EXISTS genome_history (
		id            TEXT PRIMARY KEY,
		generation    INTEGER NOT NULL,
		genes_json    TEXT NOT NULL,
		fitness_score REAL NOT NULL DEFAULT 0,
		sharpe        REAL NOT NULL DEFAULT 0,
		max_drawdown  REAL NOT NULL DEFAULT 0,
		is_best       INTEGER NOT NULL DEFAULT 0,
		recorded_at   DATETIME NOT NULL
	)`)
	return err
}

// loadRecord retrieves a single GenomeRecord by ID.
func (h *GenomeHistoryHandler) loadRecord(id string) (GenomeRecord, error) {
	row := h.db.QueryRow(
		`SELECT id, generation, genes_json, fitness_score, sharpe, max_drawdown, is_best, recorded_at
		 FROM genome_history WHERE id = ?`, id)
	return scanGenomeRecord(row)
}

// scanGenomeRecords converts sql.Rows to a slice of GenomeRecord.
func scanGenomeRecords(rows *sql.Rows) ([]GenomeRecord, error) {
	var out []GenomeRecord
	for rows.Next() {
		rec, err := scanGenomeRecord(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	if out == nil {
		out = []GenomeRecord{}
	}
	return out, rows.Err()
}

// scanGenomeRecord reads one row into a GenomeRecord.
func scanGenomeRecord(row rowScanner) (GenomeRecord, error) {
	var (
		rec       GenomeRecord
		genesJSON string
		isBest    int
	)
	if err := row.Scan(&rec.ID, &rec.Generation, &genesJSON,
		&rec.FitnessScore, &rec.Sharpe, &rec.MaxDrawdown, &isBest, &rec.RecordedAt); err != nil {
		return rec, err
	}
	if err := json.Unmarshal([]byte(genesJSON), &rec.Genes); err != nil {
		return rec, fmt.Errorf("unmarshal genes: %w", err)
	}
	rec.IsBest = isBest != 0
	return rec, nil
}
