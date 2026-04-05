-- regime-oracle/schema_extension.sql
-- Regime Oracle schema extension for idea_engine.db
-- Applied automatically by RegimeOracle on first use.

CREATE TABLE IF NOT EXISTS regime_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    symbol          TEXT    NOT NULL DEFAULT 'BTC',
    regime          TEXT    NOT NULL,
    bull_prob       REAL,
    bear_prob       REAL,
    neutral_prob    REAL,
    crisis_prob     REAL,
    recovery_prob   REAL,
    topping_prob    REAL,
    features_json   TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE(ts, symbol)
);

CREATE TABLE IF NOT EXISTS genome_regime_tags (
    genome_id       INTEGER NOT NULL,
    regime          TEXT    NOT NULL,
    regime_sharpe   REAL,
    regime_trades   INTEGER,
    is_best         INTEGER NOT NULL DEFAULT 0,
    is_worst        INTEGER NOT NULL DEFAULT 0,
    tagged_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (genome_id, regime)
);

-- Index for fast time-series lookup of regime history
CREATE INDEX IF NOT EXISTS idx_regime_history_ts_sym
    ON regime_history (symbol, ts DESC);

-- Index for routing table queries
CREATE INDEX IF NOT EXISTS idx_genome_regime_tags_regime
    ON genome_regime_tags (regime, regime_sharpe DESC);
