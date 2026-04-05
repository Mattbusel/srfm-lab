-- Counterfactual Oracle schema extension for idea_engine.db
-- Run once via: sqlite3 idea_engine.db < schema_extension.sql
-- Or applied automatically by CounterfactualOracle._ensure_schema()

CREATE TABLE IF NOT EXISTS counterfactual_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    baseline_run_id TEXT    NOT NULL,
    param_delta     TEXT    NOT NULL,  -- JSON dict of changed params
    params_full     TEXT    NOT NULL,  -- JSON dict of complete genome
    sharpe          REAL,
    max_dd          REAL,
    calmar          REAL,
    total_return    REAL,
    win_rate        REAL,
    num_trades      INTEGER,
    improvement     REAL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_cf_run_id   ON counterfactual_results (baseline_run_id);
CREATE INDEX IF NOT EXISTS idx_cf_improve  ON counterfactual_results (baseline_run_id, improvement DESC);

CREATE TABLE IF NOT EXISTS sensitivity_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    param_name      TEXT    NOT NULL,
    sobol_index     REAL,
    tornado_range   REAL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_sens_run    ON sensitivity_reports (run_id);
CREATE INDEX IF NOT EXISTS idx_sens_param  ON sensitivity_reports (param_name);
