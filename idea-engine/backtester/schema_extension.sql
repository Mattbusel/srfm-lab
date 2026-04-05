-- Backtester Abstraction Layer schema extension for idea_engine.db
-- Apply once via:  sqlite3 idea_engine.db < schema_extension.sql
-- Or applied automatically by BacktestRunner._ensure_schema()

CREATE TABLE IF NOT EXISTS sim_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    params_hash      TEXT    NOT NULL,
    params_json      TEXT    NOT NULL,
    label            TEXT,
    hypothesis_id    INTEGER,
    experiment_id    INTEGER,
    sharpe           REAL,
    calmar           REAL,
    max_dd           REAL,
    total_return     REAL,
    win_rate         REAL,
    num_trades       INTEGER,
    duration_seconds REAL,
    status           TEXT    NOT NULL DEFAULT 'completed',
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_simruns_hash       ON sim_runs (params_hash);
CREATE INDEX IF NOT EXISTS idx_simruns_sharpe     ON sim_runs (sharpe DESC);
CREATE INDEX IF NOT EXISTS idx_simruns_hypothesis ON sim_runs (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_simruns_experiment ON sim_runs (experiment_id);
CREATE INDEX IF NOT EXISTS idx_simruns_status     ON sim_runs (status);

CREATE TABLE IF NOT EXISTS backtest_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    priority     INTEGER NOT NULL DEFAULT 50,
    params_json  TEXT    NOT NULL,
    label        TEXT,
    hypothesis_id INTEGER,
    status       TEXT    NOT NULL DEFAULT 'queued',
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    started_at   TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_btqueue_priority ON backtest_queue (priority ASC, id ASC);
CREATE INDEX IF NOT EXISTS idx_btqueue_status   ON backtest_queue (status);
CREATE INDEX IF NOT EXISTS idx_btqueue_hyp      ON backtest_queue (hypothesis_id);
