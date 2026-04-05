// Package store provides SQLite-backed time-series storage for the liquidity oracle.
package store

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"srfm-lab/idea-engine/liquidity-oracle/scorer"
)

const ringBufferSize = 100_000 // max rows per symbol before oldest are pruned

// TimeseriesStore is a SQLite ring buffer for liquidity scores.
type TimeseriesStore struct {
	db *sql.DB
}

// Open opens or creates the SQLite database at path.
func Open(path string) (*TimeseriesStore, error) {
	db, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("store: open %s: %w", path, err)
	}
	db.SetMaxOpenConns(1)
	s := &TimeseriesStore{db: db}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

// Close closes the underlying database.
func (s *TimeseriesStore) Close() error {
	return s.db.Close()
}

func (s *TimeseriesStore) migrate() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS liquidity_scores (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			symbol          TEXT    NOT NULL,
			composite       REAL    NOT NULL,
			spread_score    REAL    NOT NULL,
			depth_score     REAL    NOT NULL,
			volume_score    REAL    NOT NULL,
			recommendation  TEXT    NOT NULL,
			spread_pct      REAL    NOT NULL,
			bid_depth       REAL    NOT NULL,
			ask_depth       REAL    NOT NULL,
			volume_ratio    REAL    NOT NULL,
			ts              INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_liq_symbol_ts ON liquidity_scores(symbol, ts DESC);

		CREATE TABLE IF NOT EXISTS spread_observations (
			id       INTEGER PRIMARY KEY AUTOINCREMENT,
			symbol   TEXT    NOT NULL,
			spread   REAL    NOT NULL,
			hour     INTEGER NOT NULL,
			ts       INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_spread_symbol_ts ON spread_observations(symbol, ts DESC);
	`)
	return err
}

// InsertScore persists a liquidity score.
func (s *TimeseriesStore) InsertScore(sc scorer.LiquidityScore) error {
	_, err := s.db.Exec(
		`INSERT INTO liquidity_scores(symbol,composite,spread_score,depth_score,volume_score,recommendation,spread_pct,bid_depth,ask_depth,volume_ratio,ts)
		 VALUES(?,?,?,?,?,?,?,?,?,?,?)`,
		sc.Symbol, sc.Composite, sc.SpreadScore, sc.DepthScore, sc.VolumeScore,
		string(sc.Recommendation), sc.SpreadPct, sc.BidDepth, sc.AskDepth, sc.VolumeRatio,
		sc.ScoredAt.UnixMilli(),
	)
	if err != nil {
		return err
	}
	// Enforce ring buffer: prune oldest rows beyond limit.
	_, err = s.db.Exec(
		`DELETE FROM liquidity_scores WHERE symbol=? AND id NOT IN (
			SELECT id FROM liquidity_scores WHERE symbol=? ORDER BY ts DESC LIMIT ?
		)`,
		sc.Symbol, sc.Symbol, ringBufferSize,
	)
	return err
}

// InsertSpread persists one spread observation.
func (s *TimeseriesStore) InsertSpread(symbol string, spread float64, hour int, ts time.Time) error {
	_, err := s.db.Exec(
		`INSERT INTO spread_observations(symbol,spread,hour,ts) VALUES(?,?,?,?)`,
		symbol, spread, hour, ts.UnixMilli(),
	)
	return err
}

// RecentScores returns the last n liquidity scores for symbol.
func (s *TimeseriesStore) RecentScores(symbol string, n int) ([]scorer.LiquidityScore, error) {
	rows, err := s.db.Query(
		`SELECT symbol,composite,spread_score,depth_score,volume_score,recommendation,spread_pct,bid_depth,ask_depth,volume_ratio,ts
		 FROM liquidity_scores WHERE symbol=? ORDER BY ts DESC LIMIT ?`,
		symbol, n,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []scorer.LiquidityScore
	for rows.Next() {
		var sc scorer.LiquidityScore
		var rec string
		var tsMs int64
		if err := rows.Scan(&sc.Symbol, &sc.Composite, &sc.SpreadScore, &sc.DepthScore, &sc.VolumeScore,
			&rec, &sc.SpreadPct, &sc.BidDepth, &sc.AskDepth, &sc.VolumeRatio, &tsMs); err != nil {
			return nil, err
		}
		sc.Recommendation = scorer.Recommendation(rec)
		sc.ScoredAt = time.UnixMilli(tsMs).UTC()
		out = append(out, sc)
	}
	return out, rows.Err()
}

// HourlySpreadAvg returns the average spread for symbol at the given hour from stored data.
func (s *TimeseriesStore) HourlySpreadAvg(symbol string, hour int) (float64, error) {
	var avg sql.NullFloat64
	err := s.db.QueryRow(
		`SELECT AVG(spread) FROM spread_observations WHERE symbol=? AND hour=?`,
		symbol, hour,
	).Scan(&avg)
	if err != nil {
		return 0, err
	}
	if !avg.Valid {
		return 0, nil
	}
	return avg.Float64, nil
}
