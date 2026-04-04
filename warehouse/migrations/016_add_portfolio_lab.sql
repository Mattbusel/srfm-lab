-- =============================================================================
-- Migration 016: Add portfolio lab tables
-- Applied: 2026-04-04
-- =============================================================================
-- UP

-- portfolio_constructions: Stored portfolio weight solutions from the optimiser.
-- Each row represents one solved portfolio (e.g. MVO, HRP, risk-parity, etc.)
-- at a given point in time.
CREATE TABLE IF NOT EXISTS portfolio_constructions (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         REFERENCES strategy_runs (id) ON DELETE SET NULL,
    constructed_at  TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    method          VARCHAR(40)     NOT NULL CHECK (method IN (
                        'mvo', 'min_variance', 'max_sharpe', 'risk_parity',
                        'hrp', 'herc', 'equal_weight', 'black_litterman',
                        'robust_mvo', 'cvar_opt', 'custom'
                    )),
    universe        VARCHAR(100),
    n_assets        INTEGER         NOT NULL,
    weights         JSONB           NOT NULL DEFAULT '{}',  -- {symbol: weight}
    expected_return DECIMAL(14,8),
    expected_vol    DECIMAL(14,8),
    expected_sharpe DECIMAL(10,4),
    expected_var95  DECIMAL(14,8),
    expected_cvar95 DECIMAL(14,8),
    turnover_vs_prev DECIMAL(10,6),
    max_weight      DECIMAL(10,6),
    min_weight      DECIMAL(10,6),
    concentration   DECIMAL(10,6),  -- HHI of weights
    lookback_days   INTEGER,
    rebalance_freq  VARCHAR(10),
    constraints     JSONB           NOT NULL DEFAULT '{}',  -- position limits, sector caps, etc.
    solver_status   VARCHAR(20)     NOT NULL DEFAULT 'optimal'
                                    CHECK (solver_status IN ('optimal', 'feasible', 'infeasible', 'timeout')),
    solve_time_ms   INTEGER,
    is_live         BOOLEAN         NOT NULL DEFAULT FALSE,  -- whether this construction is in prod
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_portfolio_constructions_run
    ON portfolio_constructions (run_id, constructed_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_constructions_method
    ON portfolio_constructions (method, constructed_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_constructions_live
    ON portfolio_constructions (is_live, constructed_at DESC) WHERE is_live = TRUE;

-- rebalance_events: Rebalance history — orders actually placed to move from
-- one portfolio_construction to the next.
CREATE TABLE IF NOT EXISTS rebalance_events (
    id              SERIAL          PRIMARY KEY,
    from_construction_id INTEGER    REFERENCES portfolio_constructions (id) ON DELETE SET NULL,
    to_construction_id   INTEGER    NOT NULL REFERENCES portfolio_constructions (id) ON DELETE CASCADE,
    rebalanced_at   TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status          VARCHAR(20)     NOT NULL DEFAULT 'pending'
                                    CHECK (status IN (
                                        'pending', 'executing', 'complete', 'partial', 'failed'
                                    )),
    gross_turnover  DECIMAL(10,6),   -- sum of |weight changes|
    net_turnover    DECIMAL(10,6),   -- net weight change
    total_cost_bps  DECIMAL(10,4),   -- estimated transaction cost in bps
    actual_cost_bps DECIMAL(10,4),   -- realised transaction cost (after fills)
    orders_placed   INTEGER         NOT NULL DEFAULT 0,
    orders_filled   INTEGER         NOT NULL DEFAULT 0,
    cash_delta      DECIMAL(14,4),   -- net cash movement
    slippage_bps    DECIMAL(10,4),
    trigger         VARCHAR(30)     NOT NULL DEFAULT 'scheduled'
                                    CHECK (trigger IN (
                                        'scheduled', 'drift', 'risk_breach',
                                        'signal', 'manual'
                                    )),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_rebalance_events_construction
    ON rebalance_events (to_construction_id, rebalanced_at DESC);
CREATE INDEX IF NOT EXISTS idx_rebalance_events_status
    ON rebalance_events (status, rebalanced_at DESC);

-- risk_metrics: Computed VaR/CVaR/Sharpe and other risk metrics per portfolio snapshot.
CREATE TABLE IF NOT EXISTS risk_metrics (
    id                      SERIAL          PRIMARY KEY,
    construction_id         INTEGER         NOT NULL REFERENCES portfolio_constructions (id) ON DELETE CASCADE,
    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    period_days             INTEGER         NOT NULL DEFAULT 252,
    sharpe                  DECIMAL(10,4),
    sortino                 DECIMAL(10,4),
    calmar                  DECIMAL(10,4),
    cagr                    DECIMAL(14,8),
    total_return            DECIMAL(14,8),
    volatility_ann          DECIMAL(14,8),
    max_drawdown            DECIMAL(10,6),
    max_drawdown_duration   INTEGER,        -- bars in max drawdown
    var_95_1d               DECIMAL(14,8),  -- 1-day 95% VaR
    var_99_1d               DECIMAL(14,8),
    cvar_95_1d              DECIMAL(14,8),
    cvar_99_1d              DECIMAL(14,8),
    var_method              VARCHAR(20)     NOT NULL DEFAULT 'historical'
                                            CHECK (var_method IN (
                                                'historical', 'parametric', 'monte_carlo', 'cornish_fisher'
                                            )),
    beta_to_spy             DECIMAL(10,6),
    alpha_ann               DECIMAL(14,8),
    tracking_error          DECIMAL(14,8),
    info_ratio              DECIMAL(10,4),
    tail_ratio              DECIMAL(10,4),
    skewness                DECIMAL(10,6),
    kurtosis                DECIMAL(10,6),
    factor_exposures        JSONB           NOT NULL DEFAULT '{}',  -- {factor: beta}
    stress_var              JSONB           NOT NULL DEFAULT '{}'   -- scenario → estimated loss
);

CREATE INDEX IF NOT EXISTS idx_risk_metrics_construction
    ON risk_metrics (construction_id, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_time
    ON risk_metrics (computed_at DESC);

-- correlation_snapshots: Stored correlation matrices for a portfolio universe.
CREATE TABLE IF NOT EXISTS correlation_snapshots (
    id              SERIAL          PRIMARY KEY,
    construction_id INTEGER         REFERENCES portfolio_constructions (id) ON DELETE SET NULL,
    computed_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    universe        VARCHAR(100),
    n_assets        INTEGER         NOT NULL,
    lookback_days   INTEGER         NOT NULL DEFAULT 60,
    method          VARCHAR(20)     NOT NULL DEFAULT 'pearson'
                                    CHECK (method IN ('pearson', 'spearman', 'ledoit_wolf', 'oas', 'glasso')),
    symbols         TEXT[]          NOT NULL,   -- ordered list; matrix rows/cols match this order
    matrix          JSONB           NOT NULL,   -- 2D array [[...], [...]]
    avg_correlation DECIMAL(10,6),
    max_correlation DECIMAL(10,6),
    min_correlation DECIMAL(10,6),
    det             DECIMAL(18,10),             -- determinant of corr matrix
    cond_number     DECIMAL(18,4),              -- condition number
    is_positive_definite BOOLEAN    NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_correlation_snapshots_construction
    ON correlation_snapshots (construction_id, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_correlation_snapshots_time
    ON correlation_snapshots (computed_at DESC);

INSERT INTO _schema_migrations (version, name) VALUES (16, 'add_portfolio_lab')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS correlation_snapshots;
-- DROP TABLE IF EXISTS risk_metrics;
-- DROP TABLE IF EXISTS rebalance_events;
-- DROP TABLE IF EXISTS portfolio_constructions;
-- DELETE FROM _schema_migrations WHERE version = 16;
