package storage

import (
	"database/sql"
	"fmt"
	"time"

	"srfm/market-data/aggregator"
)

const createTableSQL = `
CREATE TABLE IF NOT EXISTS bars (
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_bars_lookup ON bars (symbol, timeframe, timestamp DESC);
`

// BarStore persists bars to SQLite.
type BarStore struct {
	db *sql.DB
}

// NewBarStore initializes the bar store and creates schema.
func NewBarStore(db *sql.DB) (*BarStore, error) {
	if _, err := db.Exec(createTableSQL); err != nil {
		return nil, fmt.Errorf("create schema: %w", err)
	}
	return &BarStore{db: db}, nil
}

// InsertBar upserts a single bar.
func (s *BarStore) InsertBar(evt aggregator.BarEvent) error {
	_, err := s.db.Exec(`
		INSERT OR REPLACE INTO bars (symbol, timeframe, timestamp, open, high, low, close, volume)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		evt.Symbol, evt.Timeframe, evt.Timestamp.UTC().Unix(),
		evt.Open, evt.High, evt.Low, evt.Close, evt.Volume,
	)
	return err
}

// InsertBatch inserts multiple bars in a single transaction.
func (s *BarStore) InsertBatch(evts []aggregator.BarEvent) error {
	if len(evts) == 0 {
		return nil
	}
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	stmt, err := tx.Prepare(`
		INSERT OR REPLACE INTO bars (symbol, timeframe, timestamp, open, high, low, close, volume)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return fmt.Errorf("prepare: %w", err)
	}
	defer stmt.Close()

	for _, evt := range evts {
		if _, err := stmt.Exec(
			evt.Symbol, evt.Timeframe, evt.Timestamp.UTC().Unix(),
			evt.Open, evt.High, evt.Low, evt.Close, evt.Volume,
		); err != nil {
			tx.Rollback()
			return fmt.Errorf("insert %s/%s: %w", evt.Symbol, evt.Timeframe, err)
		}
	}
	return tx.Commit()
}

// QueryBars returns bars for a symbol/timeframe between start and end, up to limit rows.
func (s *BarStore) QueryBars(symbol, timeframe string, start, end time.Time, limit int) ([]aggregator.BarEvent, error) {
	if limit <= 0 || limit > 5000 {
		limit = 500
	}
	rows, err := s.db.Query(`
		SELECT timestamp, open, high, low, close, volume
		FROM bars
		WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ?
		ORDER BY timestamp ASC
		LIMIT ?`,
		symbol, timeframe, start.UTC().Unix(), end.UTC().Unix(), limit,
	)
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	var bars []aggregator.BarEvent
	for rows.Next() {
		var ts int64
		var b aggregator.BarEvent
		b.Symbol = symbol
		b.Timeframe = timeframe
		if err := rows.Scan(&ts, &b.Open, &b.High, &b.Low, &b.Close, &b.Volume); err != nil {
			return nil, err
		}
		b.Timestamp = time.Unix(ts, 0).UTC()
		b.IsComplete = true
		bars = append(bars, b)
	}
	return bars, rows.Err()
}

// LatestBar returns the most recent bar for a symbol/timeframe.
func (s *BarStore) LatestBar(symbol, timeframe string) (*aggregator.BarEvent, error) {
	row := s.db.QueryRow(`
		SELECT timestamp, open, high, low, close, volume
		FROM bars
		WHERE symbol = ? AND timeframe = ?
		ORDER BY timestamp DESC
		LIMIT 1`,
		symbol, timeframe,
	)
	var ts int64
	var b aggregator.BarEvent
	b.Symbol = symbol
	b.Timeframe = timeframe
	if err := row.Scan(&ts, &b.Open, &b.High, &b.Low, &b.Close, &b.Volume); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	b.Timestamp = time.Unix(ts, 0).UTC()
	b.IsComplete = true
	return &b, nil
}

// SymbolTimeframeExists returns true if any data exists for a symbol/timeframe.
func (s *BarStore) SymbolTimeframeExists(symbol, timeframe string) (bool, error) {
	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM bars WHERE symbol=? AND timeframe=? LIMIT 1`, symbol, timeframe).Scan(&count)
	return count > 0, err
}

// OldestTimestamp returns the earliest bar timestamp for a symbol/timeframe.
func (s *BarStore) OldestTimestamp(symbol, timeframe string) (time.Time, error) {
	var ts int64
	err := s.db.QueryRow(`SELECT MIN(timestamp) FROM bars WHERE symbol=? AND timeframe=?`, symbol, timeframe).Scan(&ts)
	if err != nil || ts == 0 {
		return time.Time{}, err
	}
	return time.Unix(ts, 0).UTC(), nil
}
