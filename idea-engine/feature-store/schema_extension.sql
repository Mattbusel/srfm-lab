-- idea-engine/feature-store/schema_extension.sql
-- ===================================================
-- Schema extension for the SRFM Feature Store.
-- Apply to idea_engine.db (WAL mode recommended):
--     sqlite3 idea_engine.db < schema_extension.sql
--
-- Tables
-- ------
--   feature_cache    : per-symbol per-signal cached values
--   feature_metadata : per-signal metadata and IC statistics
--   ic_history       : rolling IC history for adaptive weighting / dashboard

PRAGMA journal_mode=WAL;

-- ---------------------------------------------------------------------------
-- feature_cache
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    signal_name     TEXT    NOT NULL,
    ts              TEXT    NOT NULL,   -- ISO-8601 UTC timestamp
    value           REAL,              -- NULL-able (NaN stored as NULL)
    computed_at     TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(symbol, signal_name, ts)
);

-- Primary lookup index: symbol + signal + time range queries
CREATE INDEX IF NOT EXISTS idx_feature_cache_lookup
    ON feature_cache(symbol, signal_name, ts);

-- Allow fast "give me all signals for symbol in range" queries
CREATE INDEX IF NOT EXISTS idx_feature_cache_symbol_ts
    ON feature_cache(symbol, ts);

-- Allow fast "give me all symbols for signal in range" queries
CREATE INDEX IF NOT EXISTS idx_feature_cache_signal_ts
    ON feature_cache(signal_name, ts);

-- ---------------------------------------------------------------------------
-- feature_metadata
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS feature_metadata (
    signal_name     TEXT    PRIMARY KEY,
    category        TEXT    NOT NULL,
    lookback        INTEGER NOT NULL,
    signal_type     TEXT    NOT NULL DEFAULT 'continuous',
    last_computed   TEXT,              -- ISO-8601 UTC of most recent compute
    n_symbols       INTEGER DEFAULT 0,
    mean_ic         REAL,              -- mean IC across symbols / periods
    icir            REAL,              -- IC information ratio (mean/std)
    updated_at      TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ---------------------------------------------------------------------------
-- ic_history
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ic_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name     TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    window_end_ts   TEXT    NOT NULL,   -- timestamp of last bar in IC window
    forward_bars    INTEGER NOT NULL,   -- forward-return horizon used
    ic_value        REAL    NOT NULL,
    method          TEXT    NOT NULL DEFAULT 'spearman',  -- spearman | pearson
    n_obs           INTEGER,            -- number of observations used
    computed_at     TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(signal_name, symbol, window_end_ts, forward_bars, method)
);

CREATE INDEX IF NOT EXISTS idx_ic_history_signal
    ON ic_history(signal_name, window_end_ts);

CREATE INDEX IF NOT EXISTS idx_ic_history_symbol
    ON ic_history(symbol, window_end_ts);

-- ---------------------------------------------------------------------------
-- View: latest feature staleness status
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_feature_staleness AS
SELECT
    fc.symbol,
    fc.signal_name,
    fm.category,
    MAX(fc.ts)                                    AS last_ts,
    MAX(fc.computed_at)                           AS last_computed,
    CAST(
        (julianday('now') - julianday(MAX(fc.computed_at))) * 24 AS REAL
    )                                             AS age_hours
FROM feature_cache fc
LEFT JOIN feature_metadata fm USING (signal_name)
GROUP BY fc.symbol, fc.signal_name;

-- ---------------------------------------------------------------------------
-- View: top signals by IC information ratio
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_top_signals_by_icir AS
SELECT
    signal_name,
    category,
    mean_ic,
    icir,
    n_symbols,
    last_computed
FROM feature_metadata
WHERE icir IS NOT NULL
ORDER BY icir DESC;
