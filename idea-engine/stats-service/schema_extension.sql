-- Stats Service schema extension for idea_engine.db
-- Run once at startup via: sqlite3 idea_engine.db < schema_extension.sql

CREATE TABLE IF NOT EXISTS stats_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    report_type     TEXT    NOT NULL,
    content_json    TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_stats_reports_run_id
    ON stats_reports (run_id);

CREATE TABLE IF NOT EXISTS optimization_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    method          TEXT    NOT NULL,  -- 'bayesian', 'grid', 'pareto'
    param_bounds    TEXT    NOT NULL,  -- JSON
    best_params     TEXT,              -- JSON
    best_score      REAL,
    n_iterations    INTEGER,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
