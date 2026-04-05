package queries

import (
	"context"
	"encoding/json"
	"fmt"

	"srfm-lab/idea-engine/idea-api/db"
	"srfm-lab/idea-engine/idea-api/types"
)

// GenomeStore wraps IdeaDB and exposes genome-specific queries.
type GenomeStore struct {
	db *db.IdeaDB
}

// NewGenomeStore constructs a GenomeStore.
func NewGenomeStore(d *db.IdeaDB) *GenomeStore {
	return &GenomeStore{db: d}
}

// GetGenomePopulation returns all genomes belonging to the given generation.
// Pass -1 to get the current (latest) generation.
func (s *GenomeStore) GetGenomePopulation(ctx context.Context, generation int) ([]types.Genome, error) {
	var rows []db.Row
	var err error

	if generation < 0 {
		// Fetch the latest generation number first.
		genRow, qErr := s.db.QueryRow(ctx, `SELECT MAX(generation) AS g FROM genomes`)
		if qErr != nil {
			return nil, fmt.Errorf("GetGenomePopulation max gen: %w", qErr)
		}
		if genRow == nil {
			return nil, nil
		}
		generation = toInt(genRow["g"])
	}

	rows, err = s.db.QueryRows(ctx, `
		SELECT genome_id, generation, chromosome, fitness_score, sharpe_ratio,
		       max_drawdown, is_elite, parent_ids, created_at, evaluated_at
		FROM   genomes
		WHERE  generation = ?
		ORDER  BY fitness_score DESC NULLS LAST
	`, generation)
	if err != nil {
		return nil, fmt.Errorf("GetGenomePopulation: %w", err)
	}
	return parseGenomes(rows)
}

// GetArchive returns the elite genome archive — genomes with is_elite = 1
// ordered by fitness, limited to limit entries.
func (s *GenomeStore) GetArchive(ctx context.Context, limit int) ([]types.Genome, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	rows, err := s.db.QueryRows(ctx, `
		SELECT genome_id, generation, chromosome, fitness_score, sharpe_ratio,
		       max_drawdown, is_elite, parent_ids, created_at, evaluated_at
		FROM   genomes
		WHERE  is_elite = 1
		ORDER  BY fitness_score DESC NULLS LAST
		LIMIT  ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("GetArchive: %w", err)
	}
	return parseGenomes(rows)
}

// GetFitnessHistory returns the per-generation fitness statistics ordered
// chronologically. It derives statistics from the genomes table directly.
func (s *GenomeStore) GetFitnessHistory(ctx context.Context) ([]types.FitnessPoint, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT generation,
		       MAX(fitness_score)  AS best_fitness,
		       AVG(fitness_score)  AS mean_fitness,
		       MAX(evaluated_at)   AS recorded_at
		FROM   genomes
		WHERE  fitness_score IS NOT NULL
		GROUP  BY generation
		ORDER  BY generation ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("GetFitnessHistory: %w", err)
	}

	out := make([]types.FitnessPoint, 0, len(rows))
	for _, r := range rows {
		fp := types.FitnessPoint{
			Generation:  toInt(r["generation"]),
			BestFitness: toFloat64(r["best_fitness"]),
			MeanFitness: toFloat64(r["mean_fitness"]),
		}
		t, err := parseTime(toString(r["recorded_at"]))
		if err != nil {
			return nil, fmt.Errorf("GetFitnessHistory recorded_at: %w", err)
		}
		fp.RecordedAt = t
		out = append(out, fp)
	}
	return out, nil
}

// GetGenomeByID returns a single genome by its UUID.
func (s *GenomeStore) GetGenomeByID(ctx context.Context, id string) (types.Genome, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT genome_id, generation, chromosome, fitness_score, sharpe_ratio,
		       max_drawdown, is_elite, parent_ids, created_at, evaluated_at
		FROM   genomes
		WHERE  genome_id = ?
		LIMIT  1
	`, id)
	if err != nil {
		return types.Genome{}, fmt.Errorf("GetGenomeByID: %w", err)
	}
	if len(rows) == 0 {
		return types.Genome{}, fmt.Errorf("genome %q not found", id)
	}
	gs, err := parseGenomes(rows)
	if err != nil {
		return types.Genome{}, err
	}
	return gs[0], nil
}

// parseGenomes converts raw query rows into typed Genome slices.
func parseGenomes(rows []db.Row) ([]types.Genome, error) {
	out := make([]types.Genome, 0, len(rows))
	for _, r := range rows {
		g := types.Genome{
			GenomeID:    toString(r["genome_id"]),
			Generation:  toInt(r["generation"]),
			FitnessScore: toFloat64Ptr(r["fitness_score"]),
			SharpeRatio:  toFloat64Ptr(r["sharpe_ratio"]),
			MaxDrawdown:  toFloat64Ptr(r["max_drawdown"]),
			IsElite:     toBool(r["is_elite"]),
		}

		if ch := toString(r["chromosome"]); ch != "" && ch != "null" {
			g.Chromosome = json.RawMessage(ch)
		}
		if pi := toString(r["parent_ids"]); pi != "" && pi != "null" {
			g.ParentIDs = json.RawMessage(pi)
		}

		var err error
		g.CreatedAt, err = parseTime(toString(r["created_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseGenomes created_at: %w", err)
		}
		g.EvaluatedAt, err = parseTimePtr(toString(r["evaluated_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseGenomes evaluated_at: %w", err)
		}
		out = append(out, g)
	}
	return out, nil
}
