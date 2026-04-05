package queries

import (
	"context"
	"encoding/json"
	"fmt"

	"srfm-lab/idea-engine/idea-api/db"
	"srfm-lab/idea-engine/idea-api/types"
)

// ShadowStore wraps IdeaDB and exposes shadow-runner-specific queries.
type ShadowStore struct {
	db *db.IdeaDB
}

// NewShadowStore constructs a ShadowStore.
func NewShadowStore(d *db.IdeaDB) *ShadowStore {
	return &ShadowStore{db: d}
}

// GetLeaderboard returns the top limit shadow variants ranked by latest_score
// descending.
func (s *ShadowStore) GetLeaderboard(ctx context.Context, limit int) ([]types.ShadowVariant, error) {
	if limit <= 0 || limit > 200 {
		limit = 50
	}
	rows, err := s.db.QueryRows(ctx, `
		SELECT variant_id, name, hypothesis_id, strategy, latest_score,
		       cumulative_pnl, is_promoted, created_at, updated_at
		FROM   shadow_variants
		ORDER  BY latest_score DESC NULLS LAST, cumulative_pnl DESC
		LIMIT  ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("GetLeaderboard: %w", err)
	}
	return parseShadowVariants(rows)
}

// GetVariantHistory returns the scoring history for the shadow variant
// identified by variantID, ordered chronologically.
func (s *ShadowStore) GetVariantHistory(ctx context.Context, variantID string) ([]types.ShadowScore, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT score_id, variant_id, cycle_id, score, pnl,
		       sharpe_contribution, scored_at
		FROM   shadow_scores
		WHERE  variant_id = ?
		ORDER  BY scored_at ASC
	`, variantID)
	if err != nil {
		return nil, fmt.Errorf("GetVariantHistory: %w", err)
	}
	return parseShadowScores(rows)
}

// GetVariantByID returns a single shadow variant.
func (s *ShadowStore) GetVariantByID(ctx context.Context, id string) (types.ShadowVariant, error) {
	rows, err := s.db.QueryRows(ctx, `
		SELECT variant_id, name, hypothesis_id, strategy, latest_score,
		       cumulative_pnl, is_promoted, created_at, updated_at
		FROM   shadow_variants
		WHERE  variant_id = ?
		LIMIT  1
	`, id)
	if err != nil {
		return types.ShadowVariant{}, fmt.Errorf("GetVariantByID: %w", err)
	}
	if len(rows) == 0 {
		return types.ShadowVariant{}, fmt.Errorf("shadow variant %q not found", id)
	}
	vs, err := parseShadowVariants(rows)
	if err != nil {
		return types.ShadowVariant{}, err
	}
	return vs[0], nil
}

// PromoteVariant marks the shadow variant identified by id as promoted.
func (s *ShadowStore) PromoteVariant(ctx context.Context, id string) error {
	n, err := s.db.Execute(ctx, `
		UPDATE shadow_variants
		SET    is_promoted = 1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
		WHERE  variant_id = ?
	`, id)
	if err != nil {
		return fmt.Errorf("PromoteVariant: %w", err)
	}
	if n == 0 {
		return fmt.Errorf("shadow variant %q not found", id)
	}
	return nil
}

// parseShadowVariants converts raw rows to typed ShadowVariant slices.
func parseShadowVariants(rows []db.Row) ([]types.ShadowVariant, error) {
	out := make([]types.ShadowVariant, 0, len(rows))
	for _, r := range rows {
		v := types.ShadowVariant{
			VariantID:     toString(r["variant_id"]),
			Name:          toString(r["name"]),
			HypothesisID:  toString(r["hypothesis_id"]),
			LatestScore:   toFloat64Ptr(r["latest_score"]),
			CumulativePnL: toFloat64(r["cumulative_pnl"]),
			IsPromoted:    toBool(r["is_promoted"]),
		}
		if st := toString(r["strategy"]); st != "" && st != "null" {
			v.Strategy = json.RawMessage(st)
		}
		var err error
		v.CreatedAt, err = parseTime(toString(r["created_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseShadowVariants created_at: %w", err)
		}
		v.UpdatedAt, err = parseTime(toString(r["updated_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseShadowVariants updated_at: %w", err)
		}
		out = append(out, v)
	}
	return out, nil
}

// parseShadowScores converts raw rows to typed ShadowScore slices.
func parseShadowScores(rows []db.Row) ([]types.ShadowScore, error) {
	out := make([]types.ShadowScore, 0, len(rows))
	for _, r := range rows {
		ss := types.ShadowScore{
			ScoreID:            toString(r["score_id"]),
			VariantID:          toString(r["variant_id"]),
			CycleID:            toString(r["cycle_id"]),
			Score:              toFloat64(r["score"]),
			Pnl:                toFloat64(r["pnl"]),
			SharpeContribution: toFloat64(r["sharpe_contribution"]),
		}
		var err error
		ss.ScoredAt, err = parseTime(toString(r["scored_at"]))
		if err != nil {
			return nil, fmt.Errorf("parseShadowScores scored_at: %w", err)
		}
		out = append(out, ss)
	}
	return out, nil
}
