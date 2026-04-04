-- =============================================================================
-- Migration 014: Add regime lab tables
-- Applied: 2026-04-04
-- =============================================================================
-- UP

-- regime_detections: Detected regime periods per instrument per detection method.
-- Extends the existing regime_periods table with multi-method support and
-- additional calibration metadata.
CREATE TABLE IF NOT EXISTS regime_detections (
    id              SERIAL          PRIMARY KEY,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    method          VARCHAR(40)     NOT NULL CHECK (method IN (
                        'hmm', 'bh_state', 'vol_regime', 'trend_filter',
                        'macro_overlay', 'breakout_cluster', 'garch', 'custom'
                    )),
    started_at      TIMESTAMPTZ     NOT NULL,
    ended_at        TIMESTAMPTZ,
    regime          VARCHAR(30)     NOT NULL,   -- e.g. 'bull_low_vol', 'bear_high_vol', 'range'
    regime_index    INTEGER,                    -- numeric label for ML models
    confidence      DECIMAL(10,6),             -- posterior probability (for probabilistic methods)
    n_states        INTEGER,                    -- number of states in the model
    timeframe       VARCHAR(10)     NOT NULL DEFAULT '1d',
    model_version   VARCHAR(40),               -- e.g. git hash of model
    volatility      DECIMAL(14,8),             -- realised vol during period
    trend_strength  DECIMAL(10,6),             -- |slope| / vol
    duration_bars   INTEGER,
    is_current      BOOLEAN         NOT NULL DEFAULT FALSE,
    params          JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_regime_detections_instrument
    ON regime_detections (instrument_id, method, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_regime_detections_current
    ON regime_detections (instrument_id, method, is_current) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_regime_detections_method
    ON regime_detections (method, started_at DESC);

-- stress_results: Stress test scenario results.
-- Each row stores the outcome of applying a historical or hypothetical scenario
-- to a portfolio or strategy run.
CREATE TABLE IF NOT EXISTS stress_results (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    scenario_name   VARCHAR(80)     NOT NULL,   -- e.g. 'covid_crash', 'gfc_2008', 'taper_tantrum'
    scenario_type   VARCHAR(20)     NOT NULL DEFAULT 'historical'
                                    CHECK (scenario_type IN (
                                        'historical', 'hypothetical', 'factor_shock', 'montecarlo'
                                    )),
    computed_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    shock_start     TIMESTAMPTZ,
    shock_end       TIMESTAMPTZ,
    portfolio_pnl   DECIMAL(14,6),             -- portfolio return during scenario
    benchmark_pnl   DECIMAL(14,6),
    max_drawdown    DECIMAL(10,6),
    recovery_bars   INTEGER,
    var_95          DECIMAL(14,6),
    cvar_95         DECIMAL(14,6),
    margin_call     BOOLEAN         NOT NULL DEFAULT FALSE,
    factor_shocks   JSONB           NOT NULL DEFAULT '{}',  -- {factor: shock_pct}
    position_pnls   JSONB           NOT NULL DEFAULT '{}',  -- {symbol: pnl}
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_stress_results_run
    ON stress_results (run_id, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_stress_results_scenario
    ON stress_results (scenario_name, computed_at DESC);

-- regime_params: Calibrated regime model parameters per instrument per method.
-- Stores fitted HMM emission params, vol thresholds, etc.
CREATE TABLE IF NOT EXISTS regime_params (
    id              SERIAL          PRIMARY KEY,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    method          VARCHAR(40)     NOT NULL,
    calibrated_at   TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_from      TIMESTAMPTZ     NOT NULL,
    valid_to        TIMESTAMPTZ,
    n_states        INTEGER         NOT NULL,
    params          JSONB           NOT NULL DEFAULT '{}',  -- method-specific params blob
    train_log_prob  DECIMAL(14,8),   -- training log-likelihood for HMM
    aic             DECIMAL(14,4),
    bic             DECIMAL(14,4),
    model_version   VARCHAR(40),
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_regime_params_instrument
    ON regime_params (instrument_id, method, calibrated_at DESC);
CREATE INDEX IF NOT EXISTS idx_regime_params_active
    ON regime_params (instrument_id, method, is_active) WHERE is_active = TRUE;

-- transition_matrices: Stored Markov transition matrices for each regime model.
-- Provides probability of moving from one regime to another given current regime.
CREATE TABLE IF NOT EXISTS transition_matrices (
    id              SERIAL          PRIMARY KEY,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    method          VARCHAR(40)     NOT NULL,
    from_regime     VARCHAR(30)     NOT NULL,
    to_regime       VARCHAR(30)     NOT NULL,
    probability     DECIMAL(10,8)   NOT NULL CHECK (probability BETWEEN 0 AND 1),
    expected_duration_bars DECIMAL(10,2),  -- 1/(1-p_stay) for self-transitions
    sample_count    INTEGER         NOT NULL DEFAULT 0,
    computed_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    regime_params_id INTEGER        REFERENCES regime_params (id) ON DELETE SET NULL,
    UNIQUE (instrument_id, method, from_regime, to_regime)
);

CREATE INDEX IF NOT EXISTS idx_transition_matrices_instrument
    ON transition_matrices (instrument_id, method, computed_at DESC);

INSERT INTO _schema_migrations (version, name) VALUES (14, 'add_regime_lab')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS transition_matrices;
-- DROP TABLE IF EXISTS regime_params;
-- DROP TABLE IF EXISTS stress_results;
-- DROP TABLE IF EXISTS regime_detections;
-- DELETE FROM _schema_migrations WHERE version = 14;
