-- =============================================================================
-- analytics.sql
-- SRFM Lab — DuckDB Analytics Layer
-- DuckDB 0.10+ (NOT PostgreSQL — different SQL dialect)
-- =============================================================================
-- Usage:
--   duckdb srfm_analytics.db < analytics.sql
--   OR:
--   python warehouse/duckdb/setup.py
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Settings
-- ---------------------------------------------------------------------------
SET threads = 8;
SET memory_limit = '8GB';
SET enable_progress_bar = true;

-- ---------------------------------------------------------------------------
-- Parquet ingestion: load bar data from data/ directory
-- ---------------------------------------------------------------------------
-- DuckDB can read partitioned parquet files with glob patterns.
-- Expected layout: data/bars/{timeframe}/{symbol}/YYYY-MM.parquet

CREATE OR REPLACE TABLE bars_1d AS
SELECT
    symbol,
    timestamp::TIMESTAMPTZ  AS timestamp,
    open::DOUBLE            AS open,
    high::DOUBLE            AS high,
    low::DOUBLE             AS low,
    close::DOUBLE           AS close,
    volume::DOUBLE          AS volume,
    COALESCE(vwap::DOUBLE, (open+high+low+close)/4.0) AS vwap,
    LN(close / LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp))
                            AS log_return,
    high - low              AS hl_range
FROM read_parquet('data/bars/1d/**/*.parquet', union_by_name=true)
WHERE close > 0;

CREATE OR REPLACE TABLE bars_1h AS
SELECT
    symbol,
    timestamp::TIMESTAMPTZ  AS timestamp,
    open::DOUBLE, high::DOUBLE, low::DOUBLE, close::DOUBLE, volume::DOUBLE,
    COALESCE(vwap::DOUBLE, (open+high+low+close)/4.0) AS vwap,
    LN(close / LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp))
                            AS log_return,
    high - low              AS hl_range
FROM read_parquet('data/bars/1h/**/*.parquet', union_by_name=true)
WHERE close > 0;

CREATE OR REPLACE TABLE bars_15m AS
SELECT
    symbol,
    timestamp::TIMESTAMPTZ  AS timestamp,
    open::DOUBLE, high::DOUBLE, low::DOUBLE, close::DOUBLE, volume::DOUBLE,
    LN(close / LAG(close) OVER (PARTITION BY symbol ORDER BY timestamp))
                            AS log_return,
    high - low              AS hl_range
FROM read_parquet('data/bars/15m/**/*.parquet', union_by_name=true)
WHERE close > 0;

-- Trades from PostgreSQL export (parquet dump)
CREATE OR REPLACE TABLE trades AS
SELECT * FROM read_parquet('data/trades/**/*.parquet', union_by_name=true);

CREATE OR REPLACE TABLE equity_snapshots AS
SELECT * FROM read_parquet('data/equity/**/*.parquet', union_by_name=true);

CREATE OR REPLACE TABLE bh_state_1d AS
SELECT * FROM read_parquet('data/bh_state/1d/**/*.parquet', union_by_name=true);

CREATE OR REPLACE TABLE instruments AS
SELECT * FROM read_parquet('data/instruments.parquet', union_by_name=true);

-- ---------------------------------------------------------------------------
-- Rolling Sharpe ratio (vectorized, DuckDB window functions)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW rolling_sharpe AS
SELECT
    symbol,
    timestamp,
    close,
    log_return,
    -- 20-day rolling Sharpe
    AVG(log_return) OVER w20 / NULLIF(STDDEV(log_return) OVER w20, 0) * SQRT(252)
        AS sharpe_20d,
    -- 60-day rolling Sharpe
    AVG(log_return) OVER w60 / NULLIF(STDDEV(log_return) OVER w60, 0) * SQRT(252)
        AS sharpe_60d,
    -- 252-day rolling Sharpe
    AVG(log_return) OVER w252 / NULLIF(STDDEV(log_return) OVER w252, 0) * SQRT(252)
        AS sharpe_252d,
    -- Rolling 20-day volatility (annualized)
    STDDEV(log_return) OVER w20 * SQRT(252)     AS vol_20d,
    -- Rolling drawdown from 252-day high
    1 - close / MAX(close) OVER w252             AS rolling_dd_252d
FROM bars_1d
WINDOW
    w20  AS (PARTITION BY symbol ORDER BY timestamp ROWS 19  PRECEDING),
    w60  AS (PARTITION BY symbol ORDER BY timestamp ROWS 59  PRECEDING),
    w252 AS (PARTITION BY symbol ORDER BY timestamp ROWS 251 PRECEDING);

-- ---------------------------------------------------------------------------
-- Rolling drawdown — maximum drawdown over sliding windows
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW rolling_drawdown AS
WITH cummax AS (
    SELECT
        symbol,
        timestamp,
        close,
        MAX(close) OVER (PARTITION BY symbol ORDER BY timestamp
                         ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS rolling_hwm,
        MAX(close) OVER (PARTITION BY symbol ORDER BY timestamp
                         ROWS UNBOUNDED PRECEDING)                    AS running_hwm
    FROM bars_1d
)
SELECT
    symbol,
    timestamp,
    close,
    rolling_hwm,
    running_hwm,
    (close - rolling_hwm) / rolling_hwm AS rolling_dd_252d,
    (close - running_hwm) / running_hwm AS running_dd_all
FROM cummax;

-- ---------------------------------------------------------------------------
-- ASOF JOIN: match each trade to the bar that was active at entry
-- ---------------------------------------------------------------------------
-- DuckDB ASOF JOIN is highly optimized for time-series matching
CREATE OR REPLACE VIEW trade_bar_context AS
SELECT
    t.id            AS trade_id,
    t.run_id,
    t.symbol,
    t.side,
    t.entry_time,
    t.exit_time,
    t.entry_price,
    t.exit_price,
    t.pnl_dollar,
    t.pnl_pct,
    t.tf_score,
    t.regime_at_entry,
    -- Bar context at entry (via ASOF JOIN)
    b.open          AS bar_open,
    b.high          AS bar_high,
    b.low           AS bar_low,
    b.close         AS bar_close,
    b.volume        AS bar_volume,
    b.log_return    AS bar_log_return,
    b.hl_range      AS bar_hl_range,
    -- Rolling metrics at entry time
    rs.sharpe_20d   AS sharpe_at_entry,
    rs.vol_20d      AS vol_at_entry,
    rs.rolling_dd_252d AS dd_at_entry
FROM trades t
ASOF JOIN bars_1d b
    ON b.symbol = t.symbol
   AND b.timestamp <= t.entry_time
ASOF JOIN rolling_sharpe rs
    ON rs.symbol = t.symbol
   AND rs.timestamp <= t.entry_time;

-- ---------------------------------------------------------------------------
-- PIVOT: regime × instrument performance matrix
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW regime_instrument_matrix AS
PIVOT (
    SELECT
        t.symbol,
        t.regime_at_entry AS regime,
        t.pnl_pct * 100   AS return_pct
    FROM trades t
    WHERE t.exit_time IS NOT NULL
      AND t.regime_at_entry IS NOT NULL
)
ON regime
USING AVG(return_pct)
GROUP BY symbol;

-- Alternative PIVOT: win rate by regime × instrument
CREATE OR REPLACE VIEW regime_instrument_winrate AS
PIVOT (
    SELECT
        t.symbol,
        t.regime_at_entry AS regime,
        (t.pnl_dollar > 0)::INT AS won
    FROM trades t
    WHERE t.exit_time IS NOT NULL
      AND t.regime_at_entry IS NOT NULL
)
ON regime
USING AVG(won) * 100
GROUP BY symbol;

-- ---------------------------------------------------------------------------
-- Walk-forward analysis (rolling 252-day windows, step 63 days)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW walk_forward_periods AS
WITH RECURSIVE windows AS (
    SELECT
        MIN(snapshot_date) AS wf_start,
        MIN(snapshot_date) + 251 AS wf_end,
        MIN(snapshot_date) + 252 AS next_start
    FROM equity_snapshots
    UNION ALL
    SELECT
        next_start,
        next_start + 251,
        next_start + 63
    FROM windows
    WHERE next_start <= (SELECT MAX(snapshot_date) FROM equity_snapshots)
),
period_metrics AS (
    SELECT
        w.wf_start,
        w.wf_end,
        COUNT(*)                            AS n_days,
        LAST(equity ORDER BY snapshot_date)
            / FIRST(equity ORDER BY snapshot_date) - 1
                                            AS period_return,
        STDDEV(day_return) * SQRT(252)      AS period_vol,
        MIN(drawdown_pct)                   AS worst_drawdown
    FROM windows w
    JOIN equity_snapshots es
           ON es.snapshot_date BETWEEN w.wf_start AND w.wf_end
    GROUP BY w.wf_start, w.wf_end
)
SELECT
    *,
    period_return / NULLIF(period_vol, 0)   AS period_sharpe,
    CASE WHEN worst_drawdown < -0.20 THEN 'stressed'
         WHEN period_return < 0      THEN 'losing'
         ELSE 'normal' END                  AS period_label
FROM period_metrics
ORDER BY wf_start;

-- ---------------------------------------------------------------------------
-- Complex CTE: BH mass accumulation simulation from bar data
-- ---------------------------------------------------------------------------
-- This recreates the BH engine in pure SQL for validation / offline analysis.
-- Parameters: CF values embedded here for demonstration.
CREATE OR REPLACE VIEW bh_mass_simulation_1d AS
WITH RECURSIVE bh_sim AS (
    -- Seed: first bar for each symbol
    SELECT
        b.symbol,
        b.timestamp,
        b.close,
        b.log_return,
        0.0::DOUBLE             AS mass,
        FALSE                   AS active,
        0::INTEGER              AS bh_dir,
        1                       AS row_num
    FROM bars_1d b
    WHERE b.log_return IS NOT NULL
      AND b.timestamp = (SELECT MIN(timestamp) FROM bars_1d WHERE symbol = b.symbol)

    UNION ALL

    SELECT
        b.symbol,
        b.timestamp,
        b.close,
        b.log_return,
        -- Mass update: decay * old_mass + (gamma - 1) where gamma = 1/sqrt(1-beta^2)
        -- CF_1D lookup (embedded constants for key symbols)
        LEAST(
            prev.mass * 0.95 + (
                1.0 / SQRT(GREATEST(1.0 - POWER(
                    LEAST(ABS(b.log_return) / CASE b.symbol
                        WHEN 'ES'  THEN 0.005  WHEN 'NQ'  THEN 0.006
                        WHEN 'CL'  THEN 0.015  WHEN 'GC'  THEN 0.008
                        WHEN 'BTC' THEN 0.05   WHEN 'ETH' THEN 0.07
                        ELSE 0.01 END, 1.0 - 1e-9), 2.0))
                - 1.0
            ),
            10.0  -- cap mass to avoid runaway
        )                       AS mass,
        -- Active when mass >= 1.5 (default bh_form)
        (prev.mass * 0.95 + (
            1.0 / SQRT(GREATEST(1.0 - POWER(
                LEAST(ABS(b.log_return) / CASE b.symbol
                    WHEN 'ES'  THEN 0.005 WHEN 'NQ' THEN 0.006
                    WHEN 'CL'  THEN 0.015 WHEN 'GC' THEN 0.008
                    WHEN 'BTC' THEN 0.05  WHEN 'ETH' THEN 0.07
                    ELSE 0.01 END, 1.0 - 1e-9), 2.0))
            - 1.0
        )) >= 1.5               AS active,
        -- Direction: sign of log_return when active
        CASE WHEN b.log_return > 0 THEN 1 ELSE -1 END AS bh_dir,
        prev.row_num + 1        AS row_num
    FROM bars_1d b
    JOIN bh_sim prev
           ON prev.symbol = b.symbol
          AND prev.row_num + 1 = (
              SELECT COUNT(*) FROM bars_1d b2
              WHERE b2.symbol = b.symbol AND b2.timestamp <= b.timestamp
          )
)
SELECT * FROM bh_sim;

-- ---------------------------------------------------------------------------
-- Vectorized correlation matrix (DuckDB columnar, very fast)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW return_correlation_matrix AS
SELECT
    a.symbol    AS sym_a,
    b.symbol    AS sym_b,
    CORR(a.log_return, b.log_return)    AS correlation,
    COUNT(*)                            AS n_obs,
    -- Beta of sym_b vs sym_a (useful for hedging)
    COVAR_POP(a.log_return, b.log_return)
        / NULLIF(VAR_POP(a.log_return), 0) AS beta_b_vs_a
FROM bars_1d a
JOIN bars_1d b
       ON b.timestamp = a.timestamp
      AND b.symbol > a.symbol
WHERE a.timestamp >= CURRENT_DATE - 365
  AND a.log_return IS NOT NULL
  AND b.log_return IS NOT NULL
GROUP BY a.symbol, b.symbol
HAVING COUNT(*) >= 50
ORDER BY ABS(CORR(a.log_return, b.log_return)) DESC;

-- ---------------------------------------------------------------------------
-- Turnover analysis (how often does the strategy rebalance?)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW turnover_analysis AS
WITH position_changes AS (
    SELECT
        run_id,
        symbol,
        timestamp,
        frac,
        LAG(frac) OVER (PARTITION BY run_id, symbol ORDER BY timestamp) AS prev_frac,
        ABS(frac - LAG(frac) OVER (PARTITION BY run_id, symbol ORDER BY timestamp))
            AS frac_change
    FROM (
        SELECT run_id, symbol, timestamp, frac FROM trades
        WHERE exit_time IS NULL  -- currently held positions approximation
    ) t
)
SELECT
    run_id,
    DATE_TRUNC('month', timestamp)  AS month,
    SUM(ABS(frac_change))           AS monthly_turnover,
    AVG(ABS(frac_change))           AS avg_position_change
FROM position_changes
WHERE frac_change IS NOT NULL
GROUP BY run_id, DATE_TRUNC('month', timestamp)
ORDER BY run_id, month;

-- ---------------------------------------------------------------------------
-- Instrument momentum ranking (used for relative strength)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW momentum_ranking AS
WITH mom AS (
    SELECT
        symbol,
        LAST(timestamp ORDER BY timestamp)  AS latest_ts,
        LAST(close ORDER BY timestamp)      AS latest_close,
        -- 1-month momentum
        LAST(close ORDER BY timestamp)
            / FIRST(close ORDER BY timestamp
                    FILTER (WHERE timestamp >= CURRENT_DATE - 21)) - 1
                AS mom_1m,
        -- 3-month momentum
        LAST(close ORDER BY timestamp)
            / FIRST(close ORDER BY timestamp
                    FILTER (WHERE timestamp >= CURRENT_DATE - 63)) - 1
                AS mom_3m,
        -- 12-month momentum (skip last month: [252:21] Fama-French style)
        FIRST(close ORDER BY timestamp DESC
              FILTER (WHERE timestamp <= CURRENT_DATE - 21))
            / FIRST(close ORDER BY timestamp
                    FILTER (WHERE timestamp >= CURRENT_DATE - 252)) - 1
                AS mom_12m_skip1
    FROM bars_1d
    WHERE timestamp >= CURRENT_DATE - 270
    GROUP BY symbol
)
SELECT
    symbol,
    ROUND(mom_1m * 100, 2)          AS mom_1m_pct,
    ROUND(mom_3m * 100, 2)          AS mom_3m_pct,
    ROUND(mom_12m_skip1 * 100, 2)   AS mom_12m_pct,
    RANK() OVER (ORDER BY mom_3m DESC)  AS rank_3m,
    RANK() OVER (ORDER BY mom_12m_skip1 DESC) AS rank_12m
FROM mom
ORDER BY rank_3m;

-- ---------------------------------------------------------------------------
-- Regime-aware MC input data (export for Python MC engine)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW mc_input_by_regime AS
SELECT
    symbol,
    regime_at_entry     AS regime,
    pnl_pct,
    is_winner,
    hold_bars,
    tf_score
FROM trades
WHERE exit_time IS NOT NULL
  AND regime_at_entry IS NOT NULL
ORDER BY symbol, regime_at_entry, entry_time;

-- ---------------------------------------------------------------------------
-- Drawdown duration analysis (time to recovery)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW drawdown_durations AS
WITH dd_periods AS (
    SELECT
        run_id,
        snapshot_date,
        equity,
        drawdown_pct,
        -- Label each drawdown trough group
        SUM(CASE WHEN drawdown_pct >= -0.01 THEN 1 ELSE 0 END)
            OVER (PARTITION BY run_id ORDER BY snapshot_date) AS dd_group
    FROM equity_snapshots
),
dd_summary AS (
    SELECT
        run_id,
        dd_group,
        MIN(snapshot_date)      AS dd_start,
        MAX(snapshot_date)      AS dd_end,
        MIN(drawdown_pct)       AS trough_dd,
        MAX(snapshot_date) - MIN(snapshot_date) AS duration_days
    FROM dd_periods
    WHERE drawdown_pct < -0.01
    GROUP BY run_id, dd_group
)
SELECT
    run_id,
    dd_start,
    dd_end,
    ROUND(trough_dd * 100, 2)   AS trough_dd_pct,
    duration_days,
    -- Was it fully recovered by end of period?
    (SELECT equity FROM equity_snapshots
     WHERE run_id = ds.run_id AND snapshot_date = ds.dd_end)
        >= (SELECT equity FROM equity_snapshots
            WHERE run_id = ds.run_id AND snapshot_date = ds.dd_start)
        AS was_recovered
FROM dd_summary ds
WHERE trough_dd < -0.05  -- meaningful drawdowns only
ORDER BY run_id, dd_start;

-- ---------------------------------------------------------------------------
-- Optimal instrument portfolio weights (inverse volatility, DuckDB)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW inverse_vol_weights AS
WITH vols AS (
    SELECT
        symbol,
        STDDEV(log_return) * SQRT(252) AS ann_vol
    FROM bars_1d
    WHERE timestamp >= CURRENT_DATE - 252
      AND log_return IS NOT NULL
    GROUP BY symbol
    HAVING COUNT(*) >= 100
),
inv_vol AS (
    SELECT symbol, 1.0 / NULLIF(ann_vol, 0) AS inv_vol
    FROM vols
)
SELECT
    symbol,
    ROUND(ann_vol * 100, 2)             AS ann_vol_pct,
    ROUND(inv_vol / SUM(inv_vol) OVER () * 100, 2)  AS weight_pct
FROM vols
JOIN inv_vol USING (symbol)
ORDER BY weight_pct DESC;

-- ---------------------------------------------------------------------------
-- Equity curve statistics (full set for MC comparison)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW equity_curve_stats AS
WITH daily_returns AS (
    SELECT
        run_id,
        snapshot_date,
        day_return,
        drawdown_pct
    FROM equity_snapshots
    WHERE day_return IS NOT NULL
),
stats AS (
    SELECT
        run_id,
        COUNT(*)                    AS n_days,
        AVG(day_return) * 252       AS ann_return,
        STDDEV(day_return) * SQRT(252) AS ann_vol,
        AVG(day_return) / NULLIF(STDDEV(day_return), 0) * SQRT(252) AS sharpe,
        -- Sortino: downside deviation only
        AVG(day_return) / NULLIF(
            SQRT(AVG(POWER(LEAST(day_return, 0), 2))) * SQRT(252), 0
        ) AS sortino,
        MIN(drawdown_pct)           AS max_drawdown,
        -- Calmar
        AVG(day_return) * 252 / NULLIF(ABS(MIN(drawdown_pct)), 0) AS calmar,
        -- Skewness of daily returns
        (AVG(POWER(day_return - AVG(day_return), 3))
          / NULLIF(POWER(STDDEV(day_return), 3), 0))  AS return_skew,
        -- Excess kurtosis
        (AVG(POWER(day_return - AVG(day_return), 4))
          / NULLIF(POWER(STDDEV(day_return), 4), 0) - 3) AS return_excess_kurtosis,
        -- % positive days
        SUM(CASE WHEN day_return > 0 THEN 1 ELSE 0 END)::DOUBLE
            / COUNT(*) * 100        AS pct_positive_days
    FROM daily_returns
    GROUP BY run_id
)
SELECT
    run_id,
    n_days,
    ROUND(ann_return * 100, 2)      AS ann_return_pct,
    ROUND(ann_vol * 100, 2)         AS ann_vol_pct,
    ROUND(sharpe, 3)                AS sharpe,
    ROUND(sortino, 3)               AS sortino,
    ROUND(max_drawdown * 100, 2)    AS max_dd_pct,
    ROUND(calmar, 3)                AS calmar,
    ROUND(return_skew, 3)          AS skewness,
    ROUND(return_excess_kurtosis, 3) AS excess_kurtosis,
    ROUND(pct_positive_days, 1)     AS pct_positive_days
FROM stats
ORDER BY sharpe DESC;

-- ---------------------------------------------------------------------------
-- Export: CSV summaries for report generation
-- ---------------------------------------------------------------------------
COPY (SELECT * FROM equity_curve_stats) TO 'reports/equity_curve_stats.csv' (HEADER, DELIMITER ',');
COPY (SELECT * FROM regime_instrument_matrix) TO 'reports/regime_matrix.csv' (HEADER, DELIMITER ',');
COPY (SELECT * FROM return_correlation_matrix WHERE sym_a < sym_b) TO 'reports/correlation.csv' (HEADER, DELIMITER ',');
COPY (SELECT * FROM momentum_ranking) TO 'reports/momentum.csv' (HEADER, DELIMITER ',');
COPY (SELECT * FROM walk_forward_periods) TO 'reports/walk_forward.csv' (HEADER, DELIMITER ',');

-- ---------------------------------------------------------------------------
-- Parquet export for downstream tools
-- ---------------------------------------------------------------------------
COPY (SELECT * FROM bars_1d WHERE symbol IN ('ES', 'NQ', 'CL', 'GC', 'BTC'))
    TO 'data/exports/key_instruments_1d.parquet' (FORMAT PARQUET, COMPRESSION 'zstd');

COPY (SELECT * FROM trade_bar_context)
    TO 'data/exports/trade_bar_context.parquet' (FORMAT PARQUET, COMPRESSION 'zstd');

PRAGMA database_list;
SUMMARIZE bars_1d;
