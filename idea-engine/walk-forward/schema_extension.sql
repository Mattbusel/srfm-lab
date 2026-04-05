-- walk-forward/schema_extension.sql
-- Walk-Forward Analysis schema extension for idea_engine.db
-- Applied automatically by WalkForwardEngine on first use.

CREATE TABLE IF NOT EXISTS wfa_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id   INTEGER NOT NULL,
    fold_number     INTEGER NOT NULL,
    is_start        TEXT    NOT NULL,
    is_end          TEXT    NOT NULL,
    oos_start       TEXT    NOT NULL,
    oos_end         TEXT    NOT NULL,
    is_sharpe       REAL,
    oos_sharpe      REAL,
    is_maxdd        REAL,
    oos_maxdd       REAL,
    is_trades       INTEGER,
    oos_trades      INTEGER,
    efficiency_ratio REAL,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS wfa_verdicts (
    hypothesis_id   INTEGER PRIMARY KEY,
    verdict         TEXT    NOT NULL,
    efficiency_ratio REAL,
    degradation     REAL,
    stability       REAL,
    n_folds         INTEGER,
    verdict_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- Index for fast lookup of fold results by hypothesis
CREATE INDEX IF NOT EXISTS idx_wfa_results_hyp
    ON wfa_results (hypothesis_id, fold_number);
