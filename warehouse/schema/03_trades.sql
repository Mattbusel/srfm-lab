-- =============================================================================
-- 03_trades.sql
-- SRFM Lab — Trade, Order, and Equity Tracking Schema
-- PostgreSQL 15+
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Strategy runs
-- ---------------------------------------------------------------------------
-- Each backtest or live session gets one row.
-- Parameters stores the full config JSON so results are reproducible.
CREATE TABLE IF NOT EXISTS strategy_runs (
    id                  SERIAL          PRIMARY KEY,
    run_name            VARCHAR(200)    NOT NULL,
    strategy_name       VARCHAR(100)    NOT NULL DEFAULT 'larsa',
    strategy_version    VARCHAR(20),
    run_type            VARCHAR(10)     NOT NULL
                        CHECK (run_type IN ('backtest', 'paper', 'live')),
    start_date          DATE            NOT NULL,
    end_date            DATE,
    initial_equity      DECIMAL(15,2)   NOT NULL,
    final_equity        DECIMAL(15,2),
    -- Performance summary (denormalized for fast queries)
    total_return_pct    DECIMAL(10,6),
    cagr                DECIMAL(10,6),
    sharpe              DECIMAL(10,6),
    sortino             DECIMAL(10,6),
    max_drawdown_pct    DECIMAL(10,6),
    calmar              DECIMAL(10,6),
    win_rate            DECIMAL(10,6),
    profit_factor       DECIMAL(10,6),
    n_trades            INTEGER,
    -- Full parameter set used for this run
    parameters          JSONB           NOT NULL DEFAULT '{}',
    -- Tags for filtering (e.g. ['baseline', 'sweep-bh_form', 'with-harvest'])
    tags                TEXT[]          NOT NULL DEFAULT '{}',
    notes               TEXT,
    git_commit          VARCHAR(40),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_strategy_runs_type
    ON strategy_runs (run_type, start_date DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_runs_version
    ON strategy_runs (strategy_version, run_type);
CREATE INDEX IF NOT EXISTS idx_strategy_runs_tags
    ON strategy_runs USING GIN (tags);

COMMENT ON TABLE strategy_runs IS
    'One row per backtest or live session.  parameters captures the full '
    'configuration snapshot, enabling exact reproducibility.  Summary metrics '
    'are denormalized here for dashboard queries without re-computing.';

-- ---------------------------------------------------------------------------
-- Individual trades
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id                      BIGSERIAL       PRIMARY KEY,
    run_id                  INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    instrument_id           INTEGER         NOT NULL REFERENCES instruments (id),
    -- Timing
    entry_time              TIMESTAMPTZ     NOT NULL,
    exit_time               TIMESTAMPTZ,
    -- Prices
    entry_price             DECIMAL(20,8)   NOT NULL,
    exit_price              DECIMAL(20,8),
    -- Sizing
    qty                     DECIMAL(20,8)   NOT NULL,
    notional_entry          DECIMAL(20,8)   GENERATED ALWAYS AS (entry_price * qty) STORED,
    side                    VARCHAR(5)      NOT NULL CHECK (side IN ('long', 'short')),
    -- P&L
    pnl_dollar              DECIMAL(15,4),
    pnl_pct                 DECIMAL(10,6),
    -- Execution costs
    commission              DECIMAL(10,4)   NOT NULL DEFAULT 0,
    slippage                DECIMAL(10,6)   NOT NULL DEFAULT 0,
    pnl_net                 DECIMAL(15,4)   GENERATED ALWAYS AS (
                                pnl_dollar - commission
                            ) STORED,
    -- Trade metadata
    hold_bars               INTEGER,
    hold_hours              DECIMAL(10,2),
    -- Extremes during hold
    mfe_pct                 DECIMAL(10,6),  -- maximum favorable excursion
    mae_pct                 DECIMAL(10,6),  -- maximum adverse excursion
    -- BH engine state at entry (for regression analysis)
    tf_score                SMALLINT        CHECK (tf_score BETWEEN 0 AND 7),
    regime_at_entry         VARCHAR(20),
    bh_mass_1d_at_entry     DECIMAL(10,6),
    bh_mass_1h_at_entry     DECIMAL(10,6),
    bh_mass_15m_at_entry    DECIMAL(10,6),
    bh_dir_at_entry         SMALLINT,
    -- Position sizing state at entry
    pos_floor_at_entry      DECIMAL(10,6),
    pos_frac_at_entry       DECIMAL(10,6),
    portfolio_equity_at_entry DECIMAL(15,2),
    -- Exit reason
    exit_reason             VARCHAR(30)     CHECK (exit_reason IN (
                                'target', 'stop', 'bh_collapse', 'regime_change',
                                'timeout', 'manual', 'end_of_backtest'
                            )),
    -- Computed boolean (stored for index performance)
    is_winner               BOOLEAN         GENERATED ALWAYS AS (pnl_dollar > 0) STORED
);

CREATE INDEX IF NOT EXISTS idx_trades_run_id
    ON trades (run_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_instrument
    ON trades (instrument_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_tf_score
    ON trades (tf_score, run_id);
CREATE INDEX IF NOT EXISTS idx_trades_regime
    ON trades (regime_at_entry, run_id);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time
    ON trades (exit_time DESC) WHERE exit_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_winner
    ON trades (is_winner, run_id);

COMMENT ON TABLE trades IS
    'Every trade from every strategy run.  Entry/exit BH state is captured '
    'for post-hoc regression analysis.  is_winner is a generated column '
    'for fast aggregation.';
COMMENT ON COLUMN trades.tf_score IS
    '0-7 multi-timeframe BH confluence score at time of entry.  '
    'Higher = more timeframes active with directional agreement.';
COMMENT ON COLUMN trades.mfe_pct IS
    'Maximum Favorable Excursion: peak unrealized gain during hold period.  '
    'Used to calibrate profit targets.';
COMMENT ON COLUMN trades.mae_pct IS
    'Maximum Adverse Excursion: peak unrealized loss during hold period.  '
    'Used to calibrate stops.';

-- ---------------------------------------------------------------------------
-- Orders (fills that make up a trade)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id              BIGSERIAL       PRIMARY KEY,
    trade_id        BIGINT          REFERENCES trades (id) ON DELETE CASCADE,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id),
    order_time      TIMESTAMPTZ     NOT NULL,
    fill_time       TIMESTAMPTZ,
    order_type      VARCHAR(10)     NOT NULL CHECK (order_type IN (
                        'market', 'limit', 'stop', 'stop_limit'
                    )),
    side            VARCHAR(5)      NOT NULL CHECK (side IN ('buy', 'sell')),
    qty_ordered     DECIMAL(20,8)   NOT NULL,
    qty_filled      DECIMAL(20,8),
    limit_price     DECIMAL(20,8),
    stop_price      DECIMAL(20,8),
    fill_price      DECIMAL(20,8),
    commission      DECIMAL(10,4),
    status          VARCHAR(12)     NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'partial', 'filled', 'cancelled', 'rejected'
                    )),
    broker_order_id VARCHAR(50),
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_orders_trade
    ON orders (trade_id);
CREATE INDEX IF NOT EXISTS idx_orders_run
    ON orders (run_id, order_time DESC);

-- ---------------------------------------------------------------------------
-- Equity curve snapshots (daily)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS equity_snapshots (
    run_id              INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    snapshot_date       DATE            NOT NULL,
    equity              DECIMAL(15,2)   NOT NULL,
    cash                DECIMAL(15,2),
    positions_value     DECIMAL(15,2),
    day_pnl             DECIMAL(15,2),
    day_return          DECIMAL(10,6),
    drawdown_pct        DECIMAL(10,6),
    -- Running hwm (high water mark)
    hwm                 DECIMAL(15,2),
    -- Number of open positions at close of day
    open_positions      INTEGER,
    PRIMARY KEY (run_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_date
    ON equity_snapshots (snapshot_date DESC);

COMMENT ON TABLE equity_snapshots IS
    'Daily equity curve.  One row per (run, date).  drawdown_pct is from '
    'the running high-water mark.';

-- ---------------------------------------------------------------------------
-- Position snapshots (at each rebalance)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS position_snapshots (
    id              BIGSERIAL       PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id),
    frac            DECIMAL(10,6),      -- fraction of portfolio
    dollar_value    DECIMAL(15,2),
    entry_price     DECIMAL(20,8),
    current_price   DECIMAL(20,8),
    unrealized_pnl  DECIMAL(15,2),
    tf_score_now    SMALLINT
);

CREATE INDEX IF NOT EXISTS idx_position_snapshots_run_ts
    ON position_snapshots (run_id, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Risk events (stops hit, drawdown alerts, etc.)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS risk_events (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    event_time      TIMESTAMPTZ     NOT NULL,
    event_type      VARCHAR(30)     NOT NULL CHECK (event_type IN (
                        'max_drawdown_breach',
                        'daily_loss_limit',
                        'position_limit',
                        'margin_call',
                        'stop_hit',
                        'regime_emergency_exit',
                        'bh_mass_spike'
                    )),
    instrument_id   INTEGER         REFERENCES instruments (id),
    details         JSONB           NOT NULL DEFAULT '{}',
    equity_at_event DECIMAL(15,2),
    drawdown_at_event DECIMAL(10,6)
);

CREATE INDEX IF NOT EXISTS idx_risk_events_run
    ON risk_events (run_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_type
    ON risk_events (event_type, event_time DESC);

-- ---------------------------------------------------------------------------
-- MC simulation runs (linked to strategy runs)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mc_simulations (
    id              SERIAL          PRIMARY KEY,
    source_run_id   INTEGER         REFERENCES strategy_runs (id),
    n_sims          INTEGER         NOT NULL,
    n_trades        INTEGER         NOT NULL,
    initial_equity  DECIMAL(15,2)   NOT NULL,
    -- Result percentiles
    p05_final       DECIMAL(15,2),
    p25_final       DECIMAL(15,2),
    p50_final       DECIMAL(15,2),
    p75_final       DECIMAL(15,2),
    p95_final       DECIMAL(15,2),
    -- Drawdown percentiles
    p05_max_dd      DECIMAL(10,6),
    p50_max_dd      DECIMAL(10,6),
    p95_max_dd      DECIMAL(10,6),
    -- Blowup probability (equity < 50% of initial)
    blowup_prob     DECIMAL(10,6),
    -- Configuration
    use_regime_aware BOOLEAN        NOT NULL DEFAULT TRUE,
    ar1_rho         DECIMAL(10,6),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- View: Open trades (no exit_time)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW open_trades AS
SELECT
    t.id,
    sr.run_name,
    sr.run_type,
    i.symbol,
    t.side,
    t.entry_time,
    t.entry_price,
    t.qty,
    t.tf_score,
    t.regime_at_entry,
    t.bh_mass_1d_at_entry,
    t.pos_frac_at_entry,
    NOW() - t.entry_time   AS hold_duration
FROM trades t
JOIN strategy_runs sr ON sr.id = t.run_id
JOIN instruments    i  ON i.id  = t.instrument_id
WHERE t.exit_time IS NULL
ORDER BY t.entry_time DESC;

-- ---------------------------------------------------------------------------
-- View: Trade summary per run
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW run_trade_summary AS
SELECT
    t.run_id,
    sr.run_name,
    sr.run_type,
    sr.strategy_version,
    COUNT(*)                                                    AS n_trades,
    SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)               AS n_winners,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2)
                                                                AS win_rate_pct,
    ROUND(AVG(t.pnl_pct)::NUMERIC, 4)                          AS avg_return,
    ROUND(STDDEV(t.pnl_pct)::NUMERIC, 4)                       AS return_std,
    ROUND(SUM(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar ELSE 0 END)), 0)::NUMERIC, 3)
                                                                AS profit_factor,
    ROUND(AVG(t.hold_bars)::NUMERIC, 1)                        AS avg_hold_bars,
    ROUND(AVG(t.mfe_pct)::NUMERIC, 4)                          AS avg_mfe,
    ROUND(AVG(t.mae_pct)::NUMERIC, 4)                          AS avg_mae,
    MIN(t.entry_time)                                           AS first_trade,
    MAX(t.exit_time)                                            AS last_trade
FROM trades t
JOIN strategy_runs sr ON sr.id = t.run_id
WHERE t.exit_time IS NOT NULL
GROUP BY t.run_id, sr.run_name, sr.run_type, sr.strategy_version
ORDER BY t.run_id DESC;

-- ---------------------------------------------------------------------------
-- View: Slippage analysis
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW slippage_analysis AS
SELECT
    i.symbol,
    i.asset_class,
    t.side,
    COUNT(*)                            AS n_trades,
    AVG(t.slippage)                     AS avg_slippage,
    STDDEV(t.slippage)                  AS slippage_std,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t.slippage) AS median_slippage,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY t.slippage) AS p95_slippage,
    EXTRACT(HOUR FROM t.entry_time)     AS entry_hour
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.slippage IS NOT NULL
GROUP BY i.symbol, i.asset_class, t.side, EXTRACT(HOUR FROM t.entry_time)
ORDER BY i.symbol, t.side, entry_hour;

COMMENT ON VIEW slippage_analysis IS
    'Average and tail slippage by instrument, side, and hour of entry.  '
    'Use to calibrate slippage assumptions in backtests.';

-- ---------------------------------------------------------------------------
-- Function: compute MAE/MFE for a trade from bar data
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION compute_mfe_mae(
    p_trade_id  BIGINT
)
RETURNS TABLE (mfe_pct DECIMAL, mae_pct DECIMAL)
LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_trade     trades%ROWTYPE;
    v_entry_p   DECIMAL;
    v_side      VARCHAR(5);
BEGIN
    SELECT * INTO v_trade FROM trades WHERE id = p_trade_id;
    v_entry_p := v_trade.entry_price;
    v_side    := v_trade.side;

    RETURN QUERY
    SELECT
        CASE v_side
            WHEN 'long'  THEN (MAX(b.high)  - v_entry_p) / v_entry_p
            WHEN 'short' THEN (v_entry_p - MIN(b.low))   / v_entry_p
        END AS mfe_pct,
        CASE v_side
            WHEN 'long'  THEN (MIN(b.low)   - v_entry_p) / v_entry_p
            WHEN 'short' THEN (v_entry_p - MAX(b.high))  / v_entry_p
        END AS mae_pct
    FROM bars_1h b
    WHERE b.instrument_id = v_trade.instrument_id
      AND b.timestamp >= v_trade.entry_time
      AND b.timestamp <= COALESCE(v_trade.exit_time, CURRENT_TIMESTAMP);
END;
$$;
