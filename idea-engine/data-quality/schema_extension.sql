-- data-quality schema extension for idea_engine.db
-- Apply once via:  sqlite3 idea_engine.db < schema_extension.sql
-- Or applied automatically by DataQualityChecker._ensure_schema()

CREATE TABLE IF NOT EXISTS data_quality_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    data_source     TEXT    NOT NULL,
    quality_score   REAL    NOT NULL,
    issues_json     TEXT    NOT NULL,
    n_gaps          INTEGER DEFAULT 0,
    n_outliers      INTEGER DEFAULT 0,
    n_stale_bars    INTEGER DEFAULT 0,
    checked_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_dqr_symbol   ON data_quality_reports (symbol);
CREATE INDEX IF NOT EXISTS idx_dqr_score    ON data_quality_reports (quality_score);
CREATE INDEX IF NOT EXISTS idx_dqr_checked  ON data_quality_reports (checked_at DESC);

CREATE TABLE IF NOT EXISTS feed_health (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    ts              TEXT    NOT NULL,
    is_healthy      INTEGER NOT NULL,
    latency_ms      REAL,
    price           REAL,
    issue_type      TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_fh_symbol    ON feed_health (symbol);
CREATE INDEX IF NOT EXISTS idx_fh_ts        ON feed_health (ts DESC);
CREATE INDEX IF NOT EXISTS idx_fh_healthy   ON feed_health (symbol, is_healthy);
