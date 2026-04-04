-- =============================================================================
-- Migration 012: Add walk-forward analysis tables
-- Applied: 2026-04-04
-- =============================================================================
-- UP

-- wf_runs: Metadata for each walk-forward (WF) or CPCV analysis run.
CREATE TABLE IF NOT EXISTS wf_runs (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     TIMESTAMPTZ,
    status          VARCHAR(20)     NOT NULL DEFAULT 'running'
                                    CHECK (status IN ('running', 'complete', 'failed', 'cancelled')),
    wf_type         VARCHAR(20)     NOT NULL DEFAULT 'anchored'
                                    CHECK (wf_type IN ('anchored', 'rolling', 'cpcv', 'purged_cv')),
    instrument_id   INTEGER         REFERENCES instruments (id),
    universe        VARCHAR(100),           -- e.g. 'sp500' or comma-separated syms
    in_sample_bars  INTEGER         NOT NULL,
    oos_bars        INTEGER         NOT NULL,
    n_folds         INTEGER         NOT NULL,
    n_paths         INTEGER,                -- for CPCV
    metric          VARCHAR(30)     NOT NULL DEFAULT 'sharpe'
                                    CHECK (metric IN ('sharpe', 'cagr', 'sortino', 'calmar', 'pbo')),
    total_is_sharpe DECIMAL(10,4),
    total_oos_sharpe DECIMAL(10,4),
    efficiency_ratio DECIMAL(10,6),        -- OOS / IS performance
    pbo_score       DECIMAL(10,6),         -- Probability of Backtest Overfitting
    best_params     JSONB           NOT NULL DEFAULT '{}',
    notes           TEXT,
    config          JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_wf_runs_run
    ON wf_runs (run_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_wf_runs_status
    ON wf_runs (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_wf_runs_type
    ON wf_runs (wf_type, started_at DESC);

-- wf_folds: Per-fold results for each walk-forward run.
CREATE TABLE IF NOT EXISTS wf_folds (
    id              SERIAL          PRIMARY KEY,
    wf_run_id       INTEGER         NOT NULL REFERENCES wf_runs (id) ON DELETE CASCADE,
    fold_index      INTEGER         NOT NULL,
    is_start        TIMESTAMPTZ     NOT NULL,
    is_end          TIMESTAMPTZ     NOT NULL,
    oos_start       TIMESTAMPTZ     NOT NULL,
    oos_end         TIMESTAMPTZ     NOT NULL,
    is_sharpe       DECIMAL(10,4),
    oos_sharpe      DECIMAL(10,4),
    is_cagr         DECIMAL(10,6),
    oos_cagr        DECIMAL(10,6),
    is_max_dd       DECIMAL(10,6),
    oos_max_dd      DECIMAL(10,6),
    is_trades       INTEGER,
    oos_trades      INTEGER,
    best_params     JSONB           NOT NULL DEFAULT '{}',
    opt_surface     JSONB           NOT NULL DEFAULT '{}', -- sampled param grid
    UNIQUE (wf_run_id, fold_index)
);

CREATE INDEX IF NOT EXISTS idx_wf_folds_run
    ON wf_folds (wf_run_id, fold_index);

-- cpcv_paths: CPCV (Combinatorially Purged Cross-Validation) path results.
-- Each row is one synthetic backtest path drawn from OOS fold combinations.
CREATE TABLE IF NOT EXISTS cpcv_paths (
    id              SERIAL          PRIMARY KEY,
    wf_run_id       INTEGER         NOT NULL REFERENCES wf_runs (id) ON DELETE CASCADE,
    path_index      INTEGER         NOT NULL,
    fold_combo      INTEGER[]       NOT NULL,  -- fold indices used for this path
    sharpe          DECIMAL(10,4),
    cagr            DECIMAL(10,6),
    max_drawdown    DECIMAL(10,6),
    sortino         DECIMAL(10,4),
    calmar          DECIMAL(10,4),
    win_rate        DECIMAL(10,6),
    trade_count     INTEGER,
    UNIQUE (wf_run_id, path_index)
);

CREATE INDEX IF NOT EXISTS idx_cpcv_paths_run
    ON cpcv_paths (wf_run_id, path_index);

-- param_trials: Full record of every parameter combination tested during optimisation.
CREATE TABLE IF NOT EXISTS param_trials (
    id              SERIAL          PRIMARY KEY,
    wf_run_id       INTEGER         NOT NULL REFERENCES wf_runs (id) ON DELETE CASCADE,
    fold_index      INTEGER         NOT NULL,
    trial_index     INTEGER         NOT NULL,
    params          JSONB           NOT NULL DEFAULT '{}',
    is_sharpe       DECIMAL(10,4),
    is_cagr         DECIMAL(10,6),
    is_max_dd       DECIMAL(10,6),
    is_trades       INTEGER,
    oos_sharpe      DECIMAL(10,4),  -- populated after OOS evaluation if selected
    sampler         VARCHAR(30)     DEFAULT 'grid'
                                    CHECK (sampler IN ('grid', 'random', 'tpe', 'cma-es', 'sobol')),
    evaluated_at    TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_param_trials_run_fold
    ON param_trials (wf_run_id, fold_index, is_sharpe DESC);

INSERT INTO _schema_migrations (version, name) VALUES (12, 'add_walkforward')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS param_trials;
-- DROP TABLE IF EXISTS cpcv_paths;
-- DROP TABLE IF EXISTS wf_folds;
-- DROP TABLE IF EXISTS wf_runs;
-- DELETE FROM _schema_migrations WHERE version = 12;
