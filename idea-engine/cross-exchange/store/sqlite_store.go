// Package store provides SQLite-backed persistence for price and signal history.
package store

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"srfm-lab/idea-engine/cross-exchange/aggregator"
	"srfm-lab/idea-engine/cross-exchange/divergence"
)

// Store wraps a SQLite database for the cross-exchange service.
type Store struct {
	db *sql.DB
}

// Open opens (or creates) the SQLite database at path and runs migrations.
func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("store: open %s: %w", path, err)
	}
	db.SetMaxOpenConns(1) // SQLite only supports one writer
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

// Close closes the underlying database.
func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) migrate() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS exchange_prices (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			exchange   TEXT    NOT NULL,
			symbol     TEXT    NOT NULL,
			bid        REAL    NOT NULL,
			ask        REAL    NOT NULL,
			mid        REAL    NOT NULL,
			volume     REAL    NOT NULL DEFAULT 0,
			ts         INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_prices_symbol_ts ON exchange_prices(symbol, ts DESC);

		CREATE TABLE IF NOT EXISTS divergence_signals (
			id               INTEGER PRIMARY KEY AUTOINCREMENT,
			symbol           TEXT    NOT NULL,
			max_spread       REAL    NOT NULL,
			leading_exchange TEXT    NOT NULL,
			lagging_exchange TEXT    NOT NULL,
			spread_pct       REAL    NOT NULL,
			zscore           REAL    NOT NULL,
			consensus_price  REAL    NOT NULL,
			ts               INTEGER NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_div_symbol_ts ON divergence_signals(symbol, ts DESC);
	`)
	if err != nil {
		return fmt.Errorf("store: migrate: %w", err)
	}
	return nil
}

// InsertPrice persists one ExchangePrice observation.
func (s *Store) InsertPrice(p aggregator.ExchangePrice) error {
	_, err := s.db.Exec(
		`INSERT INTO exchange_prices(exchange,symbol,bid,ask,mid,volume,ts) VALUES(?,?,?,?,?,?,?)`,
		p.Exchange, p.Symbol, p.Bid, p.Ask, p.Mid, p.Volume, p.Timestamp.UnixMilli(),
	)
	return err
}

// InsertDivergenceSignal persists a divergence signal.
func (s *Store) InsertDivergenceSignal(sig divergence.DivergenceSignal) error {
	_, err := s.db.Exec(
		`INSERT INTO divergence_signals(symbol,max_spread,leading_exchange,lagging_exchange,spread_pct,zscore,consensus_price,ts)
		 VALUES(?,?,?,?,?,?,?,?)`,
		sig.Symbol, sig.MaxSpread, sig.LeadingExchange, sig.LaggingExchange,
		sig.SpreadPct, sig.Zscore, sig.ConsensusPrice, sig.DetectedAt.UnixMilli(),
	)
	return err
}

// RecentPrices returns the last n price observations for the given symbol.
func (s *Store) RecentPrices(symbol string, n int) ([]aggregator.ExchangePrice, error) {
	rows, err := s.db.Query(
		`SELECT exchange,symbol,bid,ask,mid,volume,ts FROM exchange_prices
		 WHERE symbol=? ORDER BY ts DESC LIMIT ?`,
		symbol, n,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []aggregator.ExchangePrice
	for rows.Next() {
		var p aggregator.ExchangePrice
		var tsMs int64
		if err := rows.Scan(&p.Exchange, &p.Symbol, &p.Bid, &p.Ask, &p.Mid, &p.Volume, &tsMs); err != nil {
			return nil, err
		}
		p.Timestamp = time.UnixMilli(tsMs).UTC()
		out = append(out, p)
	}
	return out, rows.Err()
}

// RecentDivergenceSignals returns the last n divergence signals for symbol.
func (s *Store) RecentDivergenceSignals(symbol string, n int) ([]divergence.DivergenceSignal, error) {
	rows, err := s.db.Query(
		`SELECT symbol,max_spread,leading_exchange,lagging_exchange,spread_pct,zscore,consensus_price,ts
		 FROM divergence_signals WHERE symbol=? ORDER BY ts DESC LIMIT ?`,
		symbol, n,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []divergence.DivergenceSignal
	for rows.Next() {
		var sig divergence.DivergenceSignal
		var tsMs int64
		if err := rows.Scan(&sig.Symbol, &sig.MaxSpread, &sig.LeadingExchange, &sig.LaggingExchange,
			&sig.SpreadPct, &sig.Zscore, &sig.ConsensusPrice, &tsMs); err != nil {
			return nil, err
		}
		sig.DetectedAt = time.UnixMilli(tsMs).UTC()
		out = append(out, sig)
	}
	return out, rows.Err()
}
