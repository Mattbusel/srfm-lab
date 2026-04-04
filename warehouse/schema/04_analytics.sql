-- =============================================================================
-- 04_analytics.sql
-- SRFM Lab — Pre-computed Analytics Tables and Views
-- PostgreSQL 15+
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Materialized view: daily rolling performance metrics per run
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS run_daily_metrics AS
SELECT
    es.run_id,
    es.snapshot_date,
    es.equity,
    es.cash,
    es.day_pnl,
    es.day_return,
    es.drawdown_pct,
    es.hwm,
    es.open_positions,
    -- 20-day rolling Sharpe (annualized)
    AVG(es.day_return)
        OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
              ROWS 19 PRECEDING)
      / NULLIF(STDDEV(es.day_return)
        OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
              ROWS 19 PRECEDING), 0)
      * SQRT(252)                         AS rolling_sharpe_20d,
    -- 60-day rolling Sharpe
    AVG(es.day_return)
        OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
              ROWS 59 PRECEDING)
      / NULLIF(STDDEV(es.day_return)
        OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
              ROWS 59 PRECEDING), 0)
      * SQRT(252)                         AS rolling_sharpe_60d,
    -- Running CAGR from inception
    POWER(
        es.equity / NULLIF(FIRST_VALUE(es.equity)
            OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0),
        365.0 / NULLIF(
            es.snapshot_date
            - FIRST_VALUE(es.snapshot_date)
              OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            + 1, 0)
    ) - 1                                 AS running_cagr,
    -- 20-day rolling volatility (annualized)
    STDDEV(es.day_return)
        OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
              ROWS 19 PRECEDING)
      * SQRT(252)                         AS rolling_vol_20d,
    -- Peak-to-trough drawdown over rolling 60 days
    1 - MIN(es.equity)
          OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
                ROWS 59 PRECEDING)
        / NULLIF(MAX(es.equity)
          OVER (PARTITION BY es.run_id ORDER BY es.snapshot_date
                ROWS 59 PRECEDING), 0)   AS rolling_max_dd_60d
FROM equity_snapshots es;

CREATE UNIQUE INDEX IF NOT EXISTS idx_run_daily_metrics_pk
    ON run_daily_metrics (run_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_run_daily_metrics_sharpe
    ON run_daily_metrics (run_id, rolling_sharpe_20d DESC);

COMMENT ON MATERIALIZED VIEW run_daily_metrics IS
    'Pre-computed rolling performance metrics.  Refresh with: '
    'REFRESH MATERIALIZED VIEW CONCURRENTLY run_daily_metrics;';

-- ---------------------------------------------------------------------------
-- View: trade stats by tf_score
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW trade_stats_by_tf AS
SELECT
    t.run_id,
    sr.run_name,
    t.tf_score,
    COUNT(*)                                                AS n_trades,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                         AS win_rate_pct,
    ROUND(AVG(t.pnl_pct)::NUMERIC, 4)                      AS avg_return,
    ROUND(STDDEV(t.pnl_pct)::NUMERIC, 4)                   AS return_std,
    ROUND(
        SUM(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar ELSE 0 END)), 0)
    ::NUMERIC, 3)                                           AS profit_factor,
    ROUND(AVG(t.hold_bars)::NUMERIC, 1)                    AS avg_hold_bars,
    ROUND(AVG(t.mfe_pct)::NUMERIC, 4)                      AS avg_mfe,
    ROUND(AVG(t.mae_pct)::NUMERIC, 4)                      AS avg_mae,
    -- Edge ratio: avg win / avg loss (absolute)
    ROUND(
        AVG(CASE WHEN t.pnl_pct > 0 THEN t.pnl_pct END)
        / NULLIF(ABS(AVG(CASE WHEN t.pnl_pct < 0 THEN t.pnl_pct END)), 0)
    ::NUMERIC, 3)                                           AS edge_ratio,
    -- Expectancy: expected $ per trade
    ROUND(
        (SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*),0))
        * AVG(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar END)
        - (1 - SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*),0))
        * ABS(AVG(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar END))
    ::NUMERIC, 2)                                           AS expectancy_dollar
FROM trades t
JOIN strategy_runs sr ON sr.id = t.run_id
WHERE t.exit_time IS NOT NULL
GROUP BY t.run_id, sr.run_name, t.tf_score
ORDER BY t.run_id, t.tf_score;

COMMENT ON VIEW trade_stats_by_tf IS
    'Core analytics table: how does performance vary with tf_score?  '
    'This directly validates the BH multi-timeframe confluence thesis.';

-- ---------------------------------------------------------------------------
-- View: regime performance
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW regime_performance AS
SELECT
    t.run_id,
    sr.run_name,
    t.regime_at_entry,
    i.asset_class,
    COUNT(*)                                                AS n_trades,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                         AS win_rate_pct,
    ROUND(AVG(t.pnl_pct)::NUMERIC, 4)                      AS avg_return,
    ROUND(SUM(t.pnl_dollar)::NUMERIC, 2)                   AS total_pnl,
    ROUND(STDDEV(t.pnl_pct)::NUMERIC, 4)                   AS return_std,
    ROUND(
        SUM(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar ELSE 0 END)), 0)
    ::NUMERIC, 3)                                           AS profit_factor,
    ROUND(AVG(t.hold_bars)::NUMERIC, 1)                    AS avg_hold_bars
FROM trades t
JOIN strategy_runs sr ON sr.id = t.run_id
JOIN instruments   i  ON i.id  = t.instrument_id
WHERE t.exit_time IS NOT NULL
GROUP BY t.run_id, sr.run_name, t.regime_at_entry, i.asset_class
ORDER BY t.run_id, t.regime_at_entry, i.asset_class;

-- ---------------------------------------------------------------------------
-- View: BH formation success rate analysis
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW bh_formation_analysis AS
SELECT
    i.symbol,
    i.asset_class,
    f.timeframe,
    f.direction,
    f.regime_at_formation,
    COUNT(*)                                                AS n_formations,
    SUM(CASE WHEN f.was_profitable THEN 1 ELSE 0 END)       AS n_profitable,
    ROUND(100.0 * SUM(CASE WHEN f.was_profitable THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                         AS profitable_pct,
    ROUND(AVG(f.duration_bars)::NUMERIC, 1)                AS avg_duration_bars,
    ROUND(AVG(f.peak_mass)::NUMERIC, 4)                    AS avg_peak_mass,
    ROUND(AVG(f.price_move_pct)::NUMERIC, 4)               AS avg_price_move_pct
FROM bh_formations f
JOIN instruments i ON i.id = f.instrument_id
GROUP BY i.symbol, i.asset_class, f.timeframe, f.direction, f.regime_at_formation
ORDER BY i.symbol, f.timeframe, f.direction;

-- ---------------------------------------------------------------------------
-- View: pos_floor impact analysis
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW pos_floor_analysis AS
SELECT
    t.run_id,
    -- Bucket pos_floor into quintiles
    WIDTH_BUCKET(t.pos_floor_at_entry, 0, 1, 5)    AS floor_quintile,
    ROUND(MIN(t.pos_floor_at_entry)::NUMERIC, 3)    AS floor_min,
    ROUND(MAX(t.pos_floor_at_entry)::NUMERIC, 3)    AS floor_max,
    COUNT(*)                                        AS n_trades,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                 AS win_rate_pct,
    ROUND(AVG(t.pnl_pct)::NUMERIC, 4)              AS avg_return,
    ROUND(AVG(t.hold_bars)::NUMERIC, 1)             AS avg_hold_bars
FROM trades t
WHERE t.pos_floor_at_entry IS NOT NULL
  AND t.exit_time IS NOT NULL
GROUP BY t.run_id, WIDTH_BUCKET(t.pos_floor_at_entry, 0, 1, 5)
ORDER BY t.run_id, floor_quintile;

COMMENT ON VIEW pos_floor_analysis IS
    'Does pos_floor magnitude predict trade duration or outcome?  '
    'Trades taken above a higher floor should have stronger BH signals.';

-- ---------------------------------------------------------------------------
-- View: monthly seasonality of returns
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW monthly_seasonality AS
SELECT
    t.run_id,
    i.symbol,
    i.asset_class,
    EXTRACT(YEAR  FROM t.entry_time)::INT   AS yr,
    EXTRACT(MONTH FROM t.entry_time)::INT   AS mo,
    TO_CHAR(t.entry_time, 'Mon')            AS month_name,
    COUNT(*)                                AS n_trades,
    ROUND(SUM(t.pnl_dollar)::NUMERIC, 2)   AS total_pnl,
    ROUND(AVG(t.pnl_pct)::NUMERIC, 4)      AS avg_return,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)         AS win_rate_pct
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.exit_time IS NOT NULL
GROUP BY t.run_id, i.symbol, i.asset_class,
         EXTRACT(YEAR FROM t.entry_time),
         EXTRACT(MONTH FROM t.entry_time),
         TO_CHAR(t.entry_time, 'Mon')
ORDER BY t.run_id, i.symbol, yr, mo;

-- ---------------------------------------------------------------------------
-- View: cross-instrument BH formation clustering
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW bh_formation_clustering AS
WITH formation_windows AS (
    SELECT
        f.id,
        f.instrument_id,
        f.timeframe,
        f.formed_at,
        f.direction,
        -- Count how many other instruments form a BH within 3 bars (3 days for 1d)
        COUNT(f2.id) AS concurrent_formations
    FROM bh_formations f
    LEFT JOIN bh_formations f2
           ON f2.instrument_id <> f.instrument_id
          AND f2.timeframe    = f.timeframe
          AND f2.formed_at BETWEEN f.formed_at - INTERVAL '3 days'
                               AND f.formed_at + INTERVAL '3 days'
    GROUP BY f.id, f.instrument_id, f.timeframe, f.formed_at, f.direction
)
SELECT
    i.symbol,
    fw.timeframe,
    fw.direction,
    fw.formed_at,
    fw.concurrent_formations,
    f.was_profitable,
    f.price_move_pct
FROM formation_windows fw
JOIN bh_formations f ON f.id = fw.id
JOIN instruments   i ON i.id  = fw.instrument_id
ORDER BY fw.concurrent_formations DESC, fw.formed_at DESC;

COMMENT ON VIEW bh_formation_clustering IS
    'Do simultaneous BH formations across multiple instruments predict '
    'stronger moves?  High concurrent_formations = market-wide momentum event.';

-- ---------------------------------------------------------------------------
-- View: Kelly fraction implied by trade history
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW kelly_fractions AS
SELECT
    t.run_id,
    i.symbol,
    i.asset_class,
    t.tf_score,
    COUNT(*)                AS n_trades,
    -- p = win probability
    AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END)   AS win_prob,
    -- b = average win / average loss (edge ratio)
    ABS(
        AVG(CASE WHEN t.pnl_pct > 0 THEN t.pnl_pct END)
        / NULLIF(AVG(CASE WHEN t.pnl_pct < 0 THEN t.pnl_pct END), 0)
    )                       AS edge_ratio,
    -- Full Kelly: f* = p - (1-p)/b
    AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END)
    - (1 - AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END))
      / NULLIF(ABS(
            AVG(CASE WHEN t.pnl_pct > 0 THEN t.pnl_pct END)
            / NULLIF(AVG(CASE WHEN t.pnl_pct < 0 THEN t.pnl_pct END), 0)
        ), 0)               AS full_kelly,
    -- Half Kelly (recommended)
    0.5 * (
        AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END)
        - (1 - AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END))
          / NULLIF(ABS(
                AVG(CASE WHEN t.pnl_pct > 0 THEN t.pnl_pct END)
                / NULLIF(AVG(CASE WHEN t.pnl_pct < 0 THEN t.pnl_pct END), 0)
            ), 0)
    )                       AS half_kelly
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.exit_time IS NOT NULL
GROUP BY t.run_id, i.symbol, i.asset_class, t.tf_score
HAVING COUNT(*) >= 10
ORDER BY t.run_id, i.symbol, t.tf_score;

COMMENT ON VIEW kelly_fractions IS
    'Empirical Kelly fractions by instrument and tf_score.  '
    'Use to validate position sizing: actual pos_frac should be <= half_kelly.';

-- ---------------------------------------------------------------------------
-- View: BH mass at entry vs trade outcome (correlation table)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW mass_entry_correlation AS
SELECT
    t.run_id,
    i.symbol,
    COUNT(*)                                            AS n_trades,
    CORR(t.bh_mass_1d_at_entry, t.pnl_pct)             AS corr_mass_1d_pnl,
    CORR(t.bh_mass_1h_at_entry, t.pnl_pct)             AS corr_mass_1h_pnl,
    CORR(t.bh_mass_15m_at_entry, t.pnl_pct)            AS corr_mass_15m_pnl,
    CORR(t.tf_score::FLOAT, t.pnl_pct)                 AS corr_tf_score_pnl,
    CORR(t.bh_mass_1d_at_entry, t.hold_bars::FLOAT)    AS corr_mass_1d_hold,
    CORR(t.bh_mass_1d_at_entry, t.mfe_pct)             AS corr_mass_1d_mfe,
    CORR(t.bh_mass_1d_at_entry, t.mae_pct)             AS corr_mass_1d_mae
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.exit_time IS NOT NULL
GROUP BY t.run_id, i.symbol
HAVING COUNT(*) >= 10
ORDER BY t.run_id, i.symbol;

COMMENT ON VIEW mass_entry_correlation IS
    'Pearson correlation between BH mass at entry and trade outcomes.  '
    'Strong positive corr_mass_1d_pnl validates the BH signal.';

-- ---------------------------------------------------------------------------
-- View: drawdown timing relative to regime changes
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW drawdown_vs_regime AS
WITH regime_transitions AS (
    SELECT
        rp.instrument_id,
        rp.regime,
        rp.started_at,
        rp.ended_at,
        LAG(rp.regime) OVER (PARTITION BY rp.instrument_id ORDER BY rp.started_at)
            AS prev_regime
    FROM regime_periods rp
),
dd_events AS (
    SELECT
        es.run_id,
        es.snapshot_date,
        es.drawdown_pct,
        es.day_pnl
    FROM equity_snapshots es
    WHERE es.drawdown_pct < -0.05  -- only meaningful drawdowns
)
SELECT
    rt.instrument_id,
    rt.regime,
    rt.prev_regime,
    de.run_id,
    de.snapshot_date,
    de.drawdown_pct,
    de.snapshot_date - rt.started_at::DATE  AS days_into_regime
FROM dd_events de
CROSS JOIN regime_transitions rt
WHERE de.snapshot_date BETWEEN rt.started_at::DATE AND COALESCE(rt.ended_at::DATE, CURRENT_DATE)
ORDER BY de.drawdown_pct ASC;

-- ---------------------------------------------------------------------------
-- Materialized view: instrument correlation matrix (based on daily returns)
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS instrument_correlation_matrix AS
WITH returns AS (
    SELECT
        instrument_id,
        DATE(timestamp) AS dt,
        log_return
    FROM bars_1d
    WHERE timestamp >= CURRENT_DATE - INTERVAL '252 days'
      AND log_return IS NOT NULL
),
pairs AS (
    SELECT
        a.instrument_id    AS inst_a,
        b.instrument_id    AS inst_b,
        CORR(a.log_return, b.log_return)  AS correlation,
        COUNT(*)                           AS n_obs
    FROM returns a
    JOIN returns b ON b.dt = a.dt AND b.instrument_id > a.instrument_id
    GROUP BY a.instrument_id, b.instrument_id
    HAVING COUNT(*) >= 50
)
SELECT
    ia.symbol   AS symbol_a,
    ib.symbol   AS symbol_b,
    p.correlation,
    p.n_obs
FROM pairs p
JOIN instruments ia ON ia.id = p.inst_a
JOIN instruments ib ON ib.id = p.inst_b;

CREATE UNIQUE INDEX IF NOT EXISTS idx_instrument_correlation_pk
    ON instrument_correlation_matrix (symbol_a, symbol_b);

COMMENT ON MATERIALIZED VIEW instrument_correlation_matrix IS
    'Pairwise return correlation over the last 252 trading days.  '
    'Refresh daily.  Used for portfolio risk estimation.';

-- ---------------------------------------------------------------------------
-- View: optimal tf_score threshold by asset class
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW optimal_tf_threshold AS
WITH cumulative AS (
    SELECT
        t.run_id,
        i.asset_class,
        ts.tf_score,
        SUM(t.pnl_dollar)   OVER (PARTITION BY t.run_id, i.asset_class
                                  ORDER BY ts.tf_score)    AS cum_pnl_above,
        COUNT(t.id)         OVER (PARTITION BY t.run_id, i.asset_class
                                  ORDER BY ts.tf_score)    AS n_trades_above,
        COUNT(t.id)         OVER (PARTITION BY t.run_id, i.asset_class) AS total_trades
    FROM (SELECT DISTINCT tf_score FROM trades) ts
    JOIN trades t ON t.tf_score >= ts.tf_score
    JOIN instruments i ON i.id = t.instrument_id
    WHERE t.exit_time IS NOT NULL
),
ranked AS (
    SELECT *,
           cum_pnl_above / NULLIF(n_trades_above, 0) AS avg_pnl_per_trade,
           ROW_NUMBER() OVER (PARTITION BY run_id, asset_class
                              ORDER BY cum_pnl_above / NULLIF(n_trades_above, 0) DESC) AS rn
    FROM cumulative
    WHERE n_trades_above >= 10
)
SELECT
    run_id,
    asset_class,
    tf_score        AS optimal_min_tf_score,
    avg_pnl_per_trade,
    n_trades_above,
    total_trades,
    ROUND(100.0 * n_trades_above / total_trades, 1) AS pct_trades_included
FROM ranked
WHERE rn = 1
ORDER BY run_id, asset_class;

COMMENT ON VIEW optimal_tf_threshold IS
    'For each run and asset class, what minimum tf_score maximizes '
    'average P&L per trade?  Use to tune entry filters.';

-- ---------------------------------------------------------------------------
-- View: strategy comparison dashboard
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW strategy_comparison AS
SELECT
    sr.id               AS run_id,
    sr.run_name,
    sr.strategy_version,
    sr.run_type,
    sr.start_date,
    sr.end_date,
    sr.initial_equity,
    sr.final_equity,
    ROUND(sr.total_return_pct::NUMERIC * 100, 2)    AS total_return_pct,
    ROUND(sr.cagr::NUMERIC * 100, 2)                AS cagr_pct,
    ROUND(sr.sharpe::NUMERIC, 2)                    AS sharpe,
    ROUND(sr.sortino::NUMERIC, 2)                   AS sortino,
    ROUND(sr.max_drawdown_pct::NUMERIC * 100, 2)    AS max_dd_pct,
    ROUND(sr.calmar::NUMERIC, 2)                    AS calmar,
    ROUND(sr.win_rate::NUMERIC * 100, 2)            AS win_rate_pct,
    ROUND(sr.profit_factor::NUMERIC, 2)             AS profit_factor,
    sr.n_trades,
    -- Days run
    (sr.end_date - sr.start_date)                   AS days_run,
    -- Parameters summary
    sr.parameters ->> 'bh_form'                     AS bh_form,
    sr.parameters ->> 'min_tf_score'                AS min_tf_score,
    sr.parameters ->> 'pos_floor'                   AS pos_floor,
    sr.tags
FROM strategy_runs sr
ORDER BY sr.id DESC;
