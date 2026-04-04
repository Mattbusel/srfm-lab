-- =============================================================================
-- 05_queries.sql
-- SRFM Lab — Named Analytical Queries Library
-- PostgreSQL 15+ compatible
-- Run individual queries with:  \i warehouse/schema/05_queries.sql
-- Or copy-paste the named query you need.
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- Q01: Best performing strategies last 90 days
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q01_best_strategies_90d'
SELECT
    sr.id,
    sr.run_name,
    sr.strategy_version,
    sr.run_type,
    sr.start_date,
    sr.end_date,
    ROUND(sr.total_return_pct * 100, 2)    AS total_return_pct,
    ROUND(sr.cagr * 100, 2)                AS cagr_pct,
    ROUND(sr.sharpe, 2)                    AS sharpe,
    ROUND(sr.max_drawdown_pct * 100, 2)    AS max_dd_pct,
    ROUND(sr.calmar, 2)                    AS calmar,
    sr.n_trades,
    sr.tags
FROM strategy_runs sr
WHERE sr.end_date >= CURRENT_DATE - 90
  AND sr.run_type IN ('backtest', 'live', 'paper')
ORDER BY sr.sharpe DESC NULLS LAST
LIMIT 20;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q02: Win rate by regime for all runs (last N completed backtest runs)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q02_win_rate_by_regime'
WITH recent_runs AS (
    SELECT id FROM strategy_runs
    WHERE run_type = 'backtest'
    ORDER BY created_at DESC
    LIMIT 10
)
SELECT
    t.regime_at_entry,
    i.asset_class,
    COUNT(*)                                                    AS n_trades,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                             AS win_rate_pct,
    ROUND(AVG(t.pnl_pct) * 100, 3)                            AS avg_return_pct,
    ROUND(SUM(t.pnl_dollar), 2)                                AS total_pnl,
    ROUND(
        SUM(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar ELSE 0 END)), 0)
    , 3)                                                        AS profit_factor
FROM trades t
JOIN recent_runs r ON r.id = t.run_id
JOIN instruments  i ON i.id  = t.instrument_id
WHERE t.exit_time IS NOT NULL
  AND t.regime_at_entry IS NOT NULL
GROUP BY t.regime_at_entry, i.asset_class
ORDER BY t.regime_at_entry, i.asset_class;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q03: Correlation of BH mass at entry with trade outcome
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q03_mass_entry_outcome_correlation'
SELECT
    i.symbol,
    i.asset_class,
    COUNT(*)                                            AS n_trades,
    ROUND(CORR(t.bh_mass_1d_at_entry,  t.pnl_pct)::NUMERIC, 4) AS r_mass_1d_pnl,
    ROUND(CORR(t.bh_mass_1h_at_entry,  t.pnl_pct)::NUMERIC, 4) AS r_mass_1h_pnl,
    ROUND(CORR(t.bh_mass_15m_at_entry, t.pnl_pct)::NUMERIC, 4) AS r_mass_15m_pnl,
    ROUND(CORR(t.tf_score::FLOAT,      t.pnl_pct)::NUMERIC, 4) AS r_tf_score_pnl,
    ROUND(CORR(t.bh_mass_1d_at_entry,  t.hold_bars::FLOAT)::NUMERIC, 4) AS r_mass_1d_hold,
    ROUND(CORR(t.bh_mass_1d_at_entry,  t.mfe_pct)::NUMERIC, 4) AS r_mass_1d_mfe,
    ROUND(CORR(t.bh_mass_1d_at_entry,  t.mae_pct)::NUMERIC, 4) AS r_mass_1d_mae
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.exit_time IS NOT NULL
  AND t.bh_mass_1d_at_entry IS NOT NULL
GROUP BY i.symbol, i.asset_class
HAVING COUNT(*) >= 20
ORDER BY r_mass_1d_pnl DESC NULLS LAST;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q04: Optimal tf_score threshold by asset class
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q04_optimal_tf_threshold'
WITH stats AS (
    SELECT
        i.asset_class,
        t.tf_score,
        COUNT(*)                                                AS n_trades,
        ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*), 0), 2)                         AS win_rate_pct,
        ROUND(AVG(t.pnl_pct) * 100, 3)                        AS avg_return_pct,
        ROUND(AVG(t.pnl_pct) / NULLIF(STDDEV(t.pnl_pct), 0), 3) AS sharpe_proxy,
        ROUND(
            SUM(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar ELSE 0 END)
            / NULLIF(ABS(SUM(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar ELSE 0 END)), 0)
        , 3)                                                    AS profit_factor
    FROM trades t
    JOIN instruments i ON i.id = t.instrument_id
    WHERE t.exit_time IS NOT NULL
    GROUP BY i.asset_class, t.tf_score
    HAVING COUNT(*) >= 5
)
SELECT *,
       SUM(n_trades) OVER (PARTITION BY asset_class ORDER BY tf_score DESC)
           AS n_trades_at_min_score
FROM stats
ORDER BY asset_class, tf_score;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q05: Monthly seasonality of returns (all runs combined)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q05_monthly_seasonality'
SELECT
    EXTRACT(MONTH FROM t.entry_time)::INT   AS month_num,
    TO_CHAR(t.entry_time, 'Mon')            AS month_name,
    COUNT(*)                                AS n_trades,
    ROUND(SUM(t.pnl_dollar), 2)            AS total_pnl,
    ROUND(AVG(t.pnl_pct) * 100, 3)        AS avg_return_pct,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)         AS win_rate_pct,
    ROUND(STDDEV(t.pnl_pct) * 100, 3)     AS return_std_pct,
    ROUND(
        AVG(t.pnl_pct) / NULLIF(STDDEV(t.pnl_pct), 0) * SQRT(21)
    , 3)                                    AS monthly_sharpe
FROM trades t
WHERE t.exit_time IS NOT NULL
GROUP BY
    EXTRACT(MONTH FROM t.entry_time),
    TO_CHAR(t.entry_time, 'Mon')
ORDER BY month_num;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q06: Drawdown timing relative to regime changes
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q06_drawdown_vs_regime_timing'
WITH drawdown_events AS (
    SELECT
        run_id,
        snapshot_date,
        drawdown_pct,
        -- Is this the worst drawdown in a 10-day window?
        ROW_NUMBER() OVER (
            PARTITION BY run_id
            ORDER BY drawdown_pct ASC
        ) AS dd_rank
    FROM equity_snapshots
    WHERE drawdown_pct < -0.05
),
regime_changes AS (
    SELECT
        rp.instrument_id,
        rp.started_at::DATE     AS change_date,
        rp.regime               AS new_regime,
        LAG(rp.regime) OVER (PARTITION BY rp.instrument_id
                             ORDER BY rp.started_at) AS old_regime
    FROM regime_periods rp
    WHERE rp.regime <> LAG(rp.regime) OVER (PARTITION BY rp.instrument_id
                                            ORDER BY rp.started_at)
       OR rp.id = 1
)
SELECT
    de.run_id,
    de.snapshot_date,
    ROUND(de.drawdown_pct * 100, 2)         AS drawdown_pct,
    rc.instrument_id,
    i.symbol,
    rc.change_date,
    rc.old_regime,
    rc.new_regime,
    de.snapshot_date - rc.change_date       AS days_after_regime_change
FROM drawdown_events de
CROSS JOIN regime_changes rc
JOIN instruments i ON i.id = rc.instrument_id
WHERE de.dd_rank <= 20
  AND de.snapshot_date BETWEEN rc.change_date AND rc.change_date + 30
ORDER BY de.drawdown_pct ASC, i.symbol;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q07: Impact of pos_floor on trade duration
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q07_pos_floor_vs_duration'
SELECT
    CASE
        WHEN t.pos_floor_at_entry < 0.1  THEN '0-10%'
        WHEN t.pos_floor_at_entry < 0.2  THEN '10-20%'
        WHEN t.pos_floor_at_entry < 0.3  THEN '20-30%'
        WHEN t.pos_floor_at_entry < 0.5  THEN '30-50%'
        ELSE '50%+'
    END                                         AS floor_bucket,
    COUNT(*)                                    AS n_trades,
    ROUND(AVG(t.hold_bars), 1)                 AS avg_hold_bars,
    ROUND(STDDEV(t.hold_bars), 1)              AS std_hold_bars,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.hold_bars), 1)
                                                AS median_hold_bars,
    ROUND(AVG(t.pnl_pct) * 100, 3)            AS avg_return_pct,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)             AS win_rate_pct,
    ROUND(AVG(t.mfe_pct) * 100, 3)            AS avg_mfe_pct,
    ROUND(AVG(t.mae_pct) * 100, 3)            AS avg_mae_pct
FROM trades t
WHERE t.exit_time IS NOT NULL
  AND t.pos_floor_at_entry IS NOT NULL
GROUP BY floor_bucket
ORDER BY MIN(t.pos_floor_at_entry);

-- ─────────────────────────────────────────────────────────────────────────────
-- Q08: Cross-instrument BH formation clustering analysis
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q08_bh_formation_clustering'
WITH formation_pairs AS (
    SELECT
        f1.id                           AS formation_id,
        f1.instrument_id                AS inst_a,
        f2.instrument_id                AS inst_b,
        f1.formed_at,
        f1.timeframe,
        f1.direction                    AS dir_a,
        f2.direction                    AS dir_b,
        CASE WHEN f1.direction = f2.direction THEN 'same' ELSE 'opposite' END
                                        AS direction_rel,
        ABS(EXTRACT(EPOCH FROM (f2.formed_at - f1.formed_at)) / 3600)
                                        AS hours_apart
    FROM bh_formations f1
    JOIN bh_formations f2
           ON f2.instrument_id > f1.instrument_id
          AND f2.timeframe = f1.timeframe
          AND f2.formed_at BETWEEN f1.formed_at - INTERVAL '24 hours'
                               AND f1.formed_at + INTERVAL '24 hours'
),
outcomes AS (
    SELECT
        fp.*,
        ia.symbol   AS symbol_a,
        ib.symbol   AS symbol_b,
        ia.asset_class AS class_a,
        ib.asset_class AS class_b,
        f1.was_profitable AS profitable_a,
        f2.was_profitable AS profitable_b,
        CASE WHEN f1.was_profitable AND f2.was_profitable THEN 'both_win'
             WHEN f1.was_profitable OR  f2.was_profitable THEN 'one_win'
             ELSE 'both_lose' END AS outcome
    FROM formation_pairs fp
    JOIN instruments ia ON ia.id = fp.inst_a
    JOIN instruments ib ON ib.id = fp.inst_b
    JOIN bh_formations f1 ON f1.id = fp.formation_id
    JOIN bh_formations f2 ON f2.instrument_id = fp.inst_b
                         AND f2.timeframe = fp.timeframe
                         AND f2.formed_at BETWEEN fp.formed_at - INTERVAL '24 hours'
                                              AND fp.formed_at + INTERVAL '24 hours'
)
SELECT
    symbol_a,
    symbol_b,
    class_a,
    class_b,
    timeframe,
    direction_rel,
    COUNT(*)                                                AS n_co_formations,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'both_win' THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 1)                         AS both_win_pct,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'one_win'  THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 1)                         AS one_win_pct,
    ROUND(AVG(hours_apart), 1)                             AS avg_hours_apart
FROM outcomes
GROUP BY symbol_a, symbol_b, class_a, class_b, timeframe, direction_rel
HAVING COUNT(*) >= 3
ORDER BY n_co_formations DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q09: Slippage analysis by time of day
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q09_slippage_by_hour'
SELECT
    i.symbol,
    i.asset_class,
    EXTRACT(HOUR FROM t.entry_time)::INT    AS entry_hour,
    t.side,
    COUNT(*)                                AS n_trades,
    ROUND(AVG(t.slippage) * 10000, 2)      AS avg_slippage_bps,
    ROUND(STDDEV(t.slippage) * 10000, 2)   AS std_slippage_bps,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.slippage) * 10000, 2)
                                            AS median_slippage_bps,
    ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY t.slippage) * 10000, 2)
                                            AS p90_slippage_bps,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY t.slippage) * 10000, 2)
                                            AS p95_slippage_bps
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.slippage IS NOT NULL
  AND t.exit_time IS NOT NULL
GROUP BY i.symbol, i.asset_class, EXTRACT(HOUR FROM t.entry_time), t.side
HAVING COUNT(*) >= 5
ORDER BY i.symbol, t.side, entry_hour;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q10: Kelly fraction implied by trade history, by tf_score
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q10_kelly_by_tf_score'
SELECT
    i.asset_class,
    t.tf_score,
    COUNT(*)                                                        AS n_trades,
    ROUND(100.0 * AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END), 2)
                                                                    AS win_rate_pct,
    ROUND(
        ABS(AVG(CASE WHEN t.pnl_pct > 0 THEN t.pnl_pct END)
            / NULLIF(AVG(CASE WHEN t.pnl_pct < 0 THEN t.pnl_pct END), 0))
    ::NUMERIC, 3)                                                   AS edge_ratio,
    ROUND(
        (AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END)
         - (1 - AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END))
           / NULLIF(ABS(AVG(CASE WHEN t.pnl_pct > 0 THEN t.pnl_pct END)
                        / NULLIF(AVG(CASE WHEN t.pnl_pct < 0 THEN t.pnl_pct END), 0)), 0))
        * 100
    ::NUMERIC, 2)                                                   AS full_kelly_pct,
    ROUND(
        0.5 * (AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END)
             - (1 - AVG(CASE WHEN t.is_winner THEN 1.0 ELSE 0.0 END))
               / NULLIF(ABS(AVG(CASE WHEN t.pnl_pct > 0 THEN t.pnl_pct END)
                            / NULLIF(AVG(CASE WHEN t.pnl_pct < 0 THEN t.pnl_pct END), 0)), 0))
        * 100
    ::NUMERIC, 2)                                                   AS half_kelly_pct
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.exit_time IS NOT NULL
GROUP BY i.asset_class, t.tf_score
HAVING COUNT(*) >= 10
ORDER BY i.asset_class, t.tf_score;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q11: MFE/MAE profile by tf_score (for exit calibration)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q11_mfe_mae_by_tf'
SELECT
    t.tf_score,
    COUNT(*)                                                    AS n_trades,
    ROUND(AVG(t.mfe_pct) * 100, 3)                            AS avg_mfe_pct,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY t.mfe_pct) * 100, 3)
                                                                AS p25_mfe_pct,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t.mfe_pct) * 100, 3)
                                                                AS median_mfe_pct,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY t.mfe_pct) * 100, 3)
                                                                AS p75_mfe_pct,
    ROUND(AVG(t.mae_pct) * 100, 3)                            AS avg_mae_pct,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY t.mae_pct) * 100, 3)
                                                                AS p25_mae_pct,
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t.mae_pct) * 100, 3)
                                                                AS median_mae_pct,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY t.mae_pct) * 100, 3)
                                                                AS p75_mae_pct,
    -- Optimal R multiple implied by median MFE/MAE
    ROUND(
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t.mfe_pct)
        / NULLIF(ABS(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY t.mae_pct)), 0)
    ::NUMERIC, 2)                                               AS implied_r_multiple
FROM trades t
WHERE t.exit_time IS NOT NULL
  AND t.mfe_pct IS NOT NULL
  AND t.mae_pct IS NOT NULL
GROUP BY t.tf_score
ORDER BY t.tf_score;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q12: Consecutive loss streak analysis (input for MC model calibration)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q12_loss_streak_analysis'
WITH streaks AS (
    SELECT
        run_id,
        id,
        entry_time,
        is_winner,
        -- streak group: new group starts when is_winner changes
        SUM(CASE WHEN is_winner THEN 1 ELSE 0 END)
            OVER (PARTITION BY run_id ORDER BY entry_time) AS win_cumsum
    FROM trades
    WHERE exit_time IS NOT NULL
),
loss_streaks AS (
    SELECT
        run_id,
        win_cumsum AS streak_key,
        COUNT(*) AS streak_length
    FROM streaks
    WHERE NOT is_winner
    GROUP BY run_id, win_cumsum
)
SELECT
    run_id,
    MAX(streak_length)                                          AS max_loss_streak,
    ROUND(AVG(streak_length), 2)                               AS avg_loss_streak,
    ROUND(STDDEV(streak_length), 2)                            AS std_loss_streak,
    ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY streak_length), 1)
                                                                AS p90_loss_streak,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY streak_length), 1)
                                                                AS p95_loss_streak,
    COUNT(*)                                                    AS n_loss_streaks,
    -- Geometric mean for AR(1) calibration
    EXP(AVG(LN(GREATEST(streak_length, 1))))                   AS geo_mean_streak
FROM loss_streaks
WHERE streak_length > 0
GROUP BY run_id
ORDER BY run_id;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q13: Regime duration statistics (for regime-aware MC)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q13_regime_duration_stats'
SELECT
    i.symbol,
    i.asset_class,
    rp.regime,
    COUNT(*)                                                    AS n_periods,
    ROUND(AVG(rp.duration_bars), 1)                            AS avg_duration_bars,
    ROUND(STDDEV(rp.duration_bars), 1)                         AS std_duration_bars,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rp.duration_bars), 1)
                                                                AS median_duration,
    MIN(rp.duration_bars)                                       AS min_duration,
    MAX(rp.duration_bars)                                       AS max_duration,
    ROUND(AVG(rp.return_during) * 100, 2)                     AS avg_regime_return_pct,
    ROUND(STDDEV(rp.return_during) * 100, 2)                  AS std_regime_return_pct
FROM regime_periods rp
JOIN instruments i ON i.id = rp.instrument_id
WHERE rp.duration_bars IS NOT NULL
GROUP BY i.symbol, i.asset_class, rp.regime
ORDER BY i.symbol, rp.regime;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q14: BH formation success rate by regime
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q14_bh_success_by_regime'
SELECT
    i.symbol,
    i.asset_class,
    f.timeframe,
    f.regime_at_formation,
    f.direction,
    COUNT(*)                                                    AS n_formations,
    SUM(CASE WHEN f.was_profitable THEN 1 ELSE 0 END)          AS n_profitable,
    ROUND(100.0 * SUM(CASE WHEN f.was_profitable THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 1)                             AS success_rate_pct,
    ROUND(AVG(f.peak_mass), 4)                                 AS avg_peak_mass,
    ROUND(AVG(f.duration_bars), 1)                             AS avg_duration_bars,
    ROUND(AVG(f.price_move_pct) * 100, 2)                     AS avg_price_move_pct
FROM bh_formations f
JOIN instruments i ON i.id = f.instrument_id
WHERE f.was_profitable IS NOT NULL
GROUP BY i.symbol, i.asset_class, f.timeframe, f.regime_at_formation, f.direction
HAVING COUNT(*) >= 3
ORDER BY i.symbol, f.timeframe, f.direction;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q15: Walk-forward performance (split into 4 quarters, check consistency)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q15_walk_forward_consistency'
WITH runs AS (
    SELECT id, start_date, end_date, initial_equity
    FROM strategy_runs
    WHERE run_type = 'backtest'
    ORDER BY created_at DESC
    LIMIT 5
),
quarters AS (
    SELECT
        t.run_id,
        r.initial_equity,
        NTILE(4) OVER (PARTITION BY t.run_id ORDER BY t.entry_time) AS quarter,
        t.pnl_dollar,
        t.is_winner,
        t.pnl_pct,
        t.tf_score
    FROM trades t
    JOIN runs r ON r.id = t.run_id
    WHERE t.exit_time IS NOT NULL
)
SELECT
    run_id,
    quarter,
    COUNT(*)                                                    AS n_trades,
    ROUND(SUM(pnl_dollar), 2)                                  AS total_pnl,
    ROUND(100.0 * SUM(CASE WHEN is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                             AS win_rate_pct,
    ROUND(AVG(pnl_pct) * 100, 3)                              AS avg_return_pct,
    ROUND(AVG(pnl_pct) / NULLIF(STDDEV(pnl_pct), 0), 3)      AS sharpe_proxy
FROM quarters
GROUP BY run_id, quarter
ORDER BY run_id, quarter;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q16: Top 10 worst drawdown events with context
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q16_worst_drawdown_events'
WITH dd_ranked AS (
    SELECT
        es.run_id,
        sr.run_name,
        es.snapshot_date,
        es.equity,
        es.drawdown_pct,
        es.day_pnl,
        ROW_NUMBER() OVER (ORDER BY es.drawdown_pct ASC) AS rn
    FROM equity_snapshots es
    JOIN strategy_runs sr ON sr.id = es.run_id
    WHERE es.drawdown_pct IS NOT NULL
)
SELECT
    dr.run_name,
    dr.snapshot_date,
    ROUND(dr.drawdown_pct * 100, 2)     AS drawdown_pct,
    ROUND(dr.equity, 2)                 AS equity,
    ROUND(dr.day_pnl, 2)               AS day_pnl,
    -- How many trades were open on that date?
    (SELECT COUNT(*) FROM trades t
     WHERE t.run_id = dr.run_id
       AND t.entry_time::DATE <= dr.snapshot_date
       AND (t.exit_time IS NULL OR t.exit_time::DATE >= dr.snapshot_date)
    )                                   AS open_trades_count
FROM dd_ranked dr
WHERE dr.rn <= 10
ORDER BY dr.drawdown_pct ASC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q17: Equity index vs crypto BH signal comparison
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q17_equity_vs_crypto_bh'
SELECT
    i.asset_class,
    t.tf_score,
    COUNT(*)                                                    AS n_trades,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                             AS win_rate_pct,
    ROUND(AVG(t.pnl_pct) * 100, 3)                            AS avg_return_pct,
    ROUND(AVG(t.hold_bars), 1)                                 AS avg_hold_bars,
    ROUND(AVG(t.bh_mass_1d_at_entry), 4)                      AS avg_mass_at_entry,
    ROUND(
        SUM(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar ELSE 0 END)), 0)
    ::NUMERIC, 3)                                               AS profit_factor
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.exit_time IS NOT NULL
  AND i.asset_class IN ('equity_index', 'crypto')
GROUP BY i.asset_class, t.tf_score
ORDER BY i.asset_class, t.tf_score;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q18: Instrument pair correlation (BH activation co-occurrence)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q18_bh_activation_correlation'
-- Note: BH activation correlation is often lower than price correlation
-- because BH events are regime-triggered, not price-level-triggered
WITH daily_activations AS (
    SELECT
        s.instrument_id,
        DATE(s.timestamp)   AS dt,
        MAX(s.active::INT)  AS was_active
    FROM bh_state_1d s
    GROUP BY s.instrument_id, DATE(s.timestamp)
),
pairs AS (
    SELECT
        a.instrument_id    AS inst_a,
        b.instrument_id    AS inst_b,
        CORR(a.was_active::FLOAT, b.was_active::FLOAT) AS bh_activation_corr,
        COUNT(*)           AS n_obs
    FROM daily_activations a
    JOIN daily_activations b
           ON b.dt = a.dt
          AND b.instrument_id > a.instrument_id
    GROUP BY a.instrument_id, b.instrument_id
    HAVING COUNT(*) >= 50
)
SELECT
    ia.symbol                           AS symbol_a,
    ib.symbol                           AS symbol_b,
    ia.asset_class                      AS class_a,
    ib.asset_class                      AS class_b,
    ROUND(p.bh_activation_corr::NUMERIC, 4) AS bh_activation_corr,
    p.n_obs
FROM pairs p
JOIN instruments ia ON ia.id = p.inst_a
JOIN instruments ib ON ib.id = p.inst_b
ORDER BY ABS(p.bh_activation_corr) DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q19: Sensitivity analysis — how does Sharpe change with bh_form threshold?
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q19_bh_form_sensitivity'
SELECT
    sr.id,
    sr.run_name,
    (sr.parameters ->> 'bh_form')::FLOAT   AS bh_form,
    sr.sharpe,
    sr.cagr * 100                           AS cagr_pct,
    sr.max_drawdown_pct * 100               AS max_dd_pct,
    sr.win_rate * 100                       AS win_rate_pct,
    sr.n_trades
FROM strategy_runs sr
WHERE sr.parameters ? 'bh_form'
ORDER BY (sr.parameters ->> 'bh_form')::FLOAT ASC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q20: Live vs backtest performance comparison (paper trading validation)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q20_live_vs_backtest'
WITH live AS (
    SELECT
        'live/paper'        AS run_type,
        COUNT(*)            AS n_trades,
        AVG(pnl_pct)        AS avg_return,
        STDDEV(pnl_pct)     AS return_std,
        SUM(CASE WHEN is_winner THEN 1 ELSE 0 END)::FLOAT
                            / NULLIF(COUNT(*), 0) AS win_rate,
        AVG(slippage)       AS avg_slippage,
        AVG(commission)     AS avg_commission
    FROM trades t
    JOIN strategy_runs sr ON sr.id = t.run_id
    WHERE sr.run_type IN ('live', 'paper')
      AND t.exit_time IS NOT NULL
),
bt AS (
    SELECT
        'backtest'          AS run_type,
        COUNT(*)            AS n_trades,
        AVG(pnl_pct)        AS avg_return,
        STDDEV(pnl_pct)     AS return_std,
        SUM(CASE WHEN is_winner THEN 1 ELSE 0 END)::FLOAT
                            / NULLIF(COUNT(*), 0) AS win_rate,
        AVG(slippage)       AS avg_slippage,
        AVG(commission)     AS avg_commission
    FROM trades t
    JOIN strategy_runs sr ON sr.id = t.run_id
    WHERE sr.run_type = 'backtest'
      AND t.exit_time IS NOT NULL
)
SELECT * FROM live
UNION ALL
SELECT * FROM bt;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q21: Position concentration analysis (how many instruments per rebalance)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q21_position_concentration'
SELECT
    ps.run_id,
    DATE(ps.timestamp)                  AS rebalance_date,
    COUNT(DISTINCT ps.instrument_id)    AS n_positions,
    ROUND(SUM(ABS(ps.frac)), 3)        AS total_exposure_frac,
    ROUND(MAX(ABS(ps.frac)), 3)        AS max_single_position,
    ROUND(MIN(ABS(ps.frac)), 3)        AS min_position,
    ROUND(SUM(ps.dollar_value), 2)     AS total_positions_value
FROM position_snapshots ps
GROUP BY ps.run_id, DATE(ps.timestamp)
ORDER BY ps.run_id, rebalance_date DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q22: BH formation rate by asset class and timeframe
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q22_formation_rate'
WITH bar_counts AS (
    SELECT instrument_id, COUNT(*) AS n_bars FROM bars_1d GROUP BY instrument_id
)
SELECT
    i.asset_class,
    f.timeframe,
    COUNT(DISTINCT i.id)            AS n_instruments,
    COUNT(f.id)                     AS n_formations,
    SUM(bc.n_bars)                  AS total_bars,
    ROUND(COUNT(f.id)::FLOAT
          / NULLIF(SUM(bc.n_bars), 0) * 100, 3) AS formation_rate_pct_per_bar,
    ROUND(AVG(f.duration_bars), 1) AS avg_formation_duration_bars
FROM bh_formations f
JOIN instruments i ON i.id = f.instrument_id
LEFT JOIN bar_counts bc ON bc.instrument_id = f.instrument_id
GROUP BY i.asset_class, f.timeframe
ORDER BY i.asset_class, f.timeframe;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q23: Regime transition impact on open trades
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q23_regime_transition_impact'
-- Trades that were open when a regime change occurred
SELECT
    t.run_id,
    i.symbol,
    t.side,
    t.entry_time,
    t.exit_time,
    t.regime_at_entry,
    t.pnl_pct * 100         AS pnl_pct,
    t.exit_reason,
    t.hold_bars,
    -- Was regime change the exit trigger?
    (t.exit_reason = 'regime_change') AS exited_on_regime_change
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.exit_time IS NOT NULL
  AND t.exit_reason IN ('regime_change', 'bh_collapse')
ORDER BY t.entry_time DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q24: P&L attribution by exit reason
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q24_pnl_by_exit_reason'
SELECT
    t.exit_reason,
    COUNT(*)                                                    AS n_trades,
    ROUND(SUM(t.pnl_dollar), 2)                               AS total_pnl,
    ROUND(AVG(t.pnl_pct) * 100, 3)                           AS avg_return_pct,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                             AS win_rate_pct,
    ROUND(AVG(t.hold_bars), 1)                                AS avg_hold_bars
FROM trades t
WHERE t.exit_time IS NOT NULL
  AND t.exit_reason IS NOT NULL
GROUP BY t.exit_reason
ORDER BY total_pnl DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q25: Risk event frequency and equity impact
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q25_risk_event_summary'
SELECT
    re.event_type,
    COUNT(*)                                                    AS n_events,
    ROUND(AVG(re.drawdown_at_event) * 100, 2)                 AS avg_dd_at_event_pct,
    ROUND(MIN(re.equity_at_event), 2)                          AS min_equity_seen,
    MIN(re.event_time)                                          AS first_event,
    MAX(re.event_time)                                          AS last_event
FROM risk_events re
GROUP BY re.event_type
ORDER BY n_events DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q26: Harvest mode (SIDEWAYS regime) Z-score mean reversion performance
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q26_harvest_mode_performance'
SELECT
    i.symbol,
    i.asset_class,
    COUNT(*)                                                    AS n_harvest_trades,
    ROUND(100.0 * SUM(CASE WHEN t.is_winner THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 2)                             AS win_rate_pct,
    ROUND(AVG(t.pnl_pct) * 100, 3)                            AS avg_return_pct,
    ROUND(AVG(t.hold_bars), 1)                                 AS avg_hold_bars,
    ROUND(SUM(t.pnl_dollar), 2)                               AS total_pnl,
    ROUND(
        SUM(CASE WHEN t.pnl_dollar > 0 THEN t.pnl_dollar ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_dollar < 0 THEN t.pnl_dollar ELSE 0 END)), 0)
    ::NUMERIC, 3)                                               AS profit_factor
FROM trades t
JOIN instruments i ON i.id = t.instrument_id
WHERE t.regime_at_entry = 'SIDEWAYS'
  AND t.exit_time IS NOT NULL
GROUP BY i.symbol, i.asset_class
ORDER BY total_pnl DESC;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q27: Intraday BH formation timing (which hour of day formations peak)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q27_formation_time_of_day'
SELECT
    f.timeframe,
    i.asset_class,
    EXTRACT(HOUR FROM f.formed_at)::INT     AS formation_hour,
    COUNT(*)                                AS n_formations,
    ROUND(100.0 * SUM(CASE WHEN f.was_profitable THEN 1 ELSE 0 END)
          / NULLIF(SUM(CASE WHEN f.was_profitable IS NOT NULL THEN 1 ELSE 0 END), 0), 1)
                                            AS success_rate_pct
FROM bh_formations f
JOIN instruments i ON i.id = f.instrument_id
WHERE f.timeframe IN ('15m', '1h')
GROUP BY f.timeframe, i.asset_class, EXTRACT(HOUR FROM f.formed_at)
ORDER BY f.timeframe, i.asset_class, formation_hour;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q28: Parameter sweep summary (bh_form vs performance heatmap data)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q28_param_sweep_heatmap'
SELECT
    (sr.parameters ->> 'bh_form')::FLOAT           AS bh_form,
    (sr.parameters ->> 'min_tf_score')::INT         AS min_tf_score,
    COUNT(*) OVER (PARTITION BY
        (sr.parameters ->> 'bh_form')::FLOAT,
        (sr.parameters ->> 'min_tf_score')::INT)    AS n_seeds,
    ROUND(AVG(sr.sharpe) OVER (PARTITION BY
        (sr.parameters ->> 'bh_form')::FLOAT,
        (sr.parameters ->> 'min_tf_score')::INT)::NUMERIC, 3) AS avg_sharpe,
    ROUND(AVG(sr.cagr) OVER (PARTITION BY
        (sr.parameters ->> 'bh_form')::FLOAT,
        (sr.parameters ->> 'min_tf_score')::INT)::NUMERIC, 4) AS avg_cagr,
    ROUND(AVG(sr.max_drawdown_pct) OVER (PARTITION BY
        (sr.parameters ->> 'bh_form')::FLOAT,
        (sr.parameters ->> 'min_tf_score')::INT)::NUMERIC, 4) AS avg_max_dd
FROM strategy_runs sr
WHERE sr.parameters ? 'bh_form'
  AND sr.parameters ? 'min_tf_score'
ORDER BY bh_form, min_tf_score;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q29: Volatility regime vs BH formation rate
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q29_vol_vs_formation_rate'
WITH vol_buckets AS (
    SELECT
        b.instrument_id,
        b.timestamp,
        -- 20-day realized vol (we compute inline, no stored column here)
        STDDEV(b.log_return) OVER (
            PARTITION BY b.instrument_id
            ORDER BY b.timestamp
            ROWS 19 PRECEDING
        ) * SQRT(252)                   AS realized_vol_ann
    FROM bars_1d b
),
with_formations AS (
    SELECT
        vb.*,
        CASE
            WHEN vb.realized_vol_ann < 0.15  THEN 'low (<15%)'
            WHEN vb.realized_vol_ann < 0.30  THEN 'medium (15-30%)'
            WHEN vb.realized_vol_ann < 0.50  THEN 'high (30-50%)'
            ELSE 'extreme (>50%)'
        END AS vol_bucket,
        (EXISTS (
            SELECT 1 FROM bh_formations f
            WHERE f.instrument_id = vb.instrument_id
              AND f.timeframe = '1d'
              AND f.formed_at::DATE = vb.timestamp::DATE
        ))::INT  AS had_formation
    FROM vol_buckets vb
    WHERE vb.realized_vol_ann IS NOT NULL
)
SELECT
    i.symbol,
    i.asset_class,
    wf.vol_bucket,
    COUNT(*)                                    AS n_bars,
    SUM(wf.had_formation)                       AS n_formations,
    ROUND(100.0 * SUM(wf.had_formation) / NULLIF(COUNT(*), 0), 3)
                                                AS formation_rate_pct
FROM with_formations wf
JOIN instruments i ON i.id = wf.instrument_id
GROUP BY i.symbol, i.asset_class, wf.vol_bucket
ORDER BY i.symbol,
    CASE wf.vol_bucket
        WHEN 'low (<15%)'       THEN 1
        WHEN 'medium (15-30%)' THEN 2
        WHEN 'high (30-50%)'   THEN 3
        ELSE 4
    END;

-- ─────────────────────────────────────────────────────────────────────────────
-- Q30: Comprehensive daily dashboard query (used by monitoring script)
-- ─────────────────────────────────────────────────────────────────────────────
-- \set label 'Q30_daily_dashboard'
WITH live_run AS (
    SELECT id FROM strategy_runs
    WHERE run_type IN ('live', 'paper')
    ORDER BY created_at DESC LIMIT 1
),
today_trades AS (
    SELECT
        i.symbol,
        t.side,
        t.entry_time,
        t.exit_time,
        t.pnl_dollar,
        t.pnl_pct,
        t.tf_score,
        t.exit_reason
    FROM trades t
    JOIN instruments i ON i.id = t.instrument_id
    WHERE t.run_id = (SELECT id FROM live_run)
      AND t.entry_time::DATE = CURRENT_DATE
),
current_equity AS (
    SELECT equity, drawdown_pct
    FROM equity_snapshots
    WHERE run_id = (SELECT id FROM live_run)
    ORDER BY snapshot_date DESC LIMIT 1
)
SELECT
    'TODAY' AS section,
    (SELECT COUNT(*) FROM today_trades WHERE exit_time IS NOT NULL) AS closed_trades,
    (SELECT COUNT(*) FROM today_trades WHERE exit_time IS NULL)     AS open_trades,
    (SELECT SUM(pnl_dollar) FROM today_trades WHERE exit_time IS NOT NULL) AS day_pnl,
    (SELECT equity FROM current_equity)                             AS current_equity,
    (SELECT drawdown_pct * 100 FROM current_equity)                AS current_dd_pct,
    (SELECT COUNT(*) FROM bh_current_state WHERE tf_score >= 3)    AS active_signals_3plus,
    (SELECT COUNT(*) FROM bh_current_state WHERE active_1d = TRUE) AS active_1d_bh_count;
