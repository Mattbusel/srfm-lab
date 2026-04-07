-- Migration 017: Options positions table
-- Tracks open and closed options positions with Greeks

-- UP

CREATE TABLE IF NOT EXISTS options_positions (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol         TEXT    NOT NULL,
  expiry         TEXT    NOT NULL,  -- YYYY-MM-DD
  strike         REAL    NOT NULL,
  right          TEXT    NOT NULL   CHECK(right IN ('CALL','PUT')),
  qty            INTEGER NOT NULL,
  entry_price    REAL    NOT NULL,
  entry_time     TEXT    NOT NULL,  -- ISO-8601
  exit_price     REAL,
  exit_time      TEXT,
  current_price  REAL,
  -- Greeks snapshot at last mark
  delta          REAL,
  gamma          REAL,
  vega           REAL,
  theta          REAL,
  rho            REAL,
  -- Implied vol at entry
  iv_entry       REAL,
  iv_current     REAL,
  -- Underlying price at entry
  underlying_entry REAL,
  -- P&L
  realized_pnl   REAL,
  unrealized_pnl REAL,
  -- Strategy linkage
  strategy_version TEXT,
  regime_at_entry  TEXT,
  signal_source    TEXT,
  -- Status
  status         TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','CLOSED','EXPIRED','ASSIGNED')),
  notes          TEXT,
  created_at     TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_options_positions_symbol  ON options_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_options_positions_expiry  ON options_positions(expiry);
CREATE INDEX IF NOT EXISTS idx_options_positions_status  ON options_positions(status);
CREATE INDEX IF NOT EXISTS idx_options_positions_created ON options_positions(created_at);

-- DOWN

DROP INDEX IF EXISTS idx_options_positions_created;
DROP INDEX IF EXISTS idx_options_positions_status;
DROP INDEX IF EXISTS idx_options_positions_expiry;
DROP INDEX IF EXISTS idx_options_positions_symbol;
DROP TABLE IF EXISTS options_positions;
