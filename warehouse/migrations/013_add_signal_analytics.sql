-- =============================================================================
-- Migration 013: Add signal analytics tables
-- Applied: 2026-04-04
-- =============================================================================
-- UP

-- ic_series: Rolling Information Coefficient (IC) per signal per instrument.
-- IC = Spearman correlation between signal rank and forward return rank
-- over a rolling window of observations.
CREATE TABLE IF NOT EXISTS ic_series (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id),
    signal_name     VARCHAR(60)     NOT NULL,
    timeframe       VARCHAR(10)     NOT NULL,
    computed_at     TIMESTAMPTZ     NOT NULL,
    window_bars     INTEGER         NOT NULL,  -- lookback used for rolling IC
    ic              DECIMAL(10,6)   NOT NULL,  -- Spearman IC
    ic_ir           DECIMAL(10,6),             -- IC / std(IC) — information ratio of IC
    ic_t_stat       DECIMAL(10,4),
    hit_rate        DECIMAL(10,6),             -- fraction of obs where sign(signal) = sign(fwd_ret)
    obs_count       INTEGER         NOT NULL,
    mean_abs_signal DECIMAL(18,8),
    fwd_horizon     INTEGER         NOT NULL DEFAULT 1  -- forward bars used for IC computation
);

CREATE INDEX IF NOT EXISTS idx_ic_series_run_signal
    ON ic_series (run_id, signal_name, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_ic_series_instrument
    ON ic_series (instrument_id, signal_name, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_ic_series_time
    ON ic_series (computed_at DESC);

-- factor_returns: Time-series of factor (signal) returns.
-- Captures long-minus-short quintile return for a signal at each rebalance period.
CREATE TABLE IF NOT EXISTS factor_returns (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    signal_name     VARCHAR(60)     NOT NULL,
    timeframe       VARCHAR(10)     NOT NULL,
    period_start    TIMESTAMPTZ     NOT NULL,
    period_end      TIMESTAMPTZ     NOT NULL,
    universe        VARCHAR(100),
    q1_return       DECIMAL(14,8),  -- bottom quintile avg return
    q2_return       DECIMAL(14,8),
    q3_return       DECIMAL(14,8),
    q4_return       DECIMAL(14,8),
    q5_return       DECIMAL(14,8),  -- top quintile avg return
    ls_return       DECIMAL(14,8),  -- q5 - q1 (long-short spread)
    ls_sharpe       DECIMAL(10,4),
    turnover        DECIMAL(10,6),
    long_count      INTEGER,
    short_count     INTEGER
);

CREATE INDEX IF NOT EXISTS idx_factor_returns_signal
    ON factor_returns (signal_name, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_factor_returns_run
    ON factor_returns (run_id, period_start DESC);

-- alpha_decay: Signal decay measurements.
-- Measures how quickly a signal's predictive power diminishes with holding horizon.
CREATE TABLE IF NOT EXISTS alpha_decay (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    instrument_id   INTEGER         REFERENCES instruments (id),
    signal_name     VARCHAR(60)     NOT NULL,
    computed_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    horizon_bars    INTEGER         NOT NULL,  -- forward bars (1, 2, 5, 10, 21, ...)
    ic              DECIMAL(10,6),
    ic_se           DECIMAL(10,6),             -- standard error of IC
    t_stat          DECIMAL(10,4),
    is_significant  BOOLEAN         NOT NULL DEFAULT FALSE,
    half_life_bars  DECIMAL(10,2),             -- populated on final row of series
    decay_rate      DECIMAL(10,6),             -- exponential decay coefficient lambda
    universe        VARCHAR(100),
    obs_count       INTEGER
);

CREATE INDEX IF NOT EXISTS idx_alpha_decay_signal
    ON alpha_decay (signal_name, computed_at DESC, horizon_bars);
CREATE INDEX IF NOT EXISTS idx_alpha_decay_run
    ON alpha_decay (run_id, signal_name, horizon_bars);

-- quantile_returns: Quintile (or decile) analysis results per signal per period.
CREATE TABLE IF NOT EXISTS quantile_returns (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    signal_name     VARCHAR(60)     NOT NULL,
    period_start    TIMESTAMPTZ     NOT NULL,
    period_end      TIMESTAMPTZ     NOT NULL,
    n_quantiles     INTEGER         NOT NULL DEFAULT 5,
    quantile        INTEGER         NOT NULL,  -- 1 = bottom, n_quantiles = top
    avg_return      DECIMAL(14,8),
    median_return   DECIMAL(14,8),
    std_return      DECIMAL(14,8),
    sharpe          DECIMAL(10,4),
    count           INTEGER         NOT NULL,
    win_rate        DECIMAL(10,6),
    universe        VARCHAR(100),
    fwd_horizon     INTEGER         NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_quantile_returns_signal
    ON quantile_returns (signal_name, period_start DESC, quantile);
CREATE INDEX IF NOT EXISTS idx_quantile_returns_run
    ON quantile_returns (run_id, signal_name, period_start DESC);

INSERT INTO _schema_migrations (version, name) VALUES (13, 'add_signal_analytics')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS quantile_returns;
-- DROP TABLE IF EXISTS alpha_decay;
-- DROP TABLE IF EXISTS factor_returns;
-- DROP TABLE IF EXISTS ic_series;
-- DELETE FROM _schema_migrations WHERE version = 13;
