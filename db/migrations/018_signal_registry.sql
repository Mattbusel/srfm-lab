-- Migration 018: Signal registry for deployed IAE signals
-- Stores metadata and live state for every deployed signal module

-- UP

CREATE TABLE IF NOT EXISTS signal_registry (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_name     TEXT    NOT NULL,
  signal_class    TEXT    NOT NULL,   -- fully qualified Python class name
  version         TEXT    NOT NULL,   -- semver or commit hash
  description     TEXT,
  -- Deployment metadata
  deployed_at     TEXT    NOT NULL DEFAULT (datetime('now')),
  deployed_by     TEXT,
  is_active       INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
  deactivated_at  TEXT,
  deactivation_reason TEXT,
  -- Signal configuration stored as JSON
  config_json     TEXT    NOT NULL DEFAULT '{}',
  -- Symbols and timeframes this signal handles
  symbols         TEXT    NOT NULL DEFAULT '[]',   -- JSON array
  timeframes      TEXT    NOT NULL DEFAULT '[]',   -- JSON array
  -- Performance summary (updated periodically)
  total_signals_fired   INTEGER NOT NULL DEFAULT 0,
  true_positive_count   INTEGER NOT NULL DEFAULT 0,
  false_positive_count  INTEGER NOT NULL DEFAULT 0,
  win_rate_pct          REAL,
  avg_return_pct        REAL,
  sharpe_contribution   REAL,
  last_fired_at         TEXT,
  -- Source tracking
  iae_genome_id         TEXT,   -- genome that produced this signal
  parent_signal_name    TEXT,
  -- Schema and output contract
  output_schema_json    TEXT,   -- JSON describing signal output fields
  required_features     TEXT,   -- JSON array of feature names needed
  -- Audit
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(signal_name, version)
);

CREATE TABLE IF NOT EXISTS signal_history (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_name     TEXT    NOT NULL,
  symbol          TEXT    NOT NULL,
  timeframe       TEXT    NOT NULL,
  bar_time        TEXT    NOT NULL,
  -- Raw signal value and direction
  signal_value    REAL    NOT NULL,
  direction       TEXT    CHECK(direction IN ('LONG','SHORT','FLAT','NEUTRAL')),
  confidence      REAL,
  -- Component values stored as JSON
  components_json TEXT,
  -- Market context at signal time
  price           REAL,
  volume          REAL,
  volatility      REAL,
  regime          TEXT,
  -- Feature values snapshot
  features_json   TEXT,
  -- Whether this signal led to a trade
  trade_id        INTEGER,
  acted_on        INTEGER NOT NULL DEFAULT 0 CHECK(acted_on IN (0,1)),
  -- Ex-post outcome (filled in later)
  fwd_return_1bar  REAL,
  fwd_return_5bar  REAL,
  fwd_return_20bar REAL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(signal_name) REFERENCES signal_registry(signal_name)
);

CREATE INDEX IF NOT EXISTS idx_signal_registry_name      ON signal_registry(signal_name);
CREATE INDEX IF NOT EXISTS idx_signal_registry_active    ON signal_registry(is_active);
CREATE INDEX IF NOT EXISTS idx_signal_history_name_sym   ON signal_history(signal_name, symbol);
CREATE INDEX IF NOT EXISTS idx_signal_history_bar_time   ON signal_history(bar_time);
CREATE INDEX IF NOT EXISTS idx_signal_history_symbol     ON signal_history(symbol);

-- DOWN

DROP INDEX IF EXISTS idx_signal_history_symbol;
DROP INDEX IF EXISTS idx_signal_history_bar_time;
DROP INDEX IF EXISTS idx_signal_history_name_sym;
DROP INDEX IF EXISTS idx_signal_registry_active;
DROP INDEX IF EXISTS idx_signal_registry_name;
DROP TABLE IF EXISTS signal_history;
DROP TABLE IF EXISTS signal_registry;
