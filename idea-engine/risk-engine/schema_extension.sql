-- ============================================================
-- risk-engine/schema_extension.sql
-- Schema additions for the Risk Engine subsystem.
-- Apply on top of the base idea_engine.db schema.
-- SQLite 3.x
-- ============================================================

-- ------------------------------------------------------------
-- var_estimates
-- Stores Value at Risk estimates produced by VaRCalculator.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_estimates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT,
    method          TEXT    NOT NULL,   -- 'historical','parametric','cornish_fisher','cvar','monte_carlo','evt_gpd'
    confidence      REAL    NOT NULL,
    var_value       REAL    NOT NULL,
    cvar_value      REAL,               -- NULL for methods that don't compute CVaR alongside
    window_days     INTEGER,            -- lookback window used (if applicable)
    computed_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_var_run_id   ON var_estimates(run_id);
CREATE INDEX IF NOT EXISTS idx_var_method   ON var_estimates(method);
CREATE INDEX IF NOT EXISTS idx_var_computed ON var_estimates(computed_at);

-- ------------------------------------------------------------
-- drawdown_events
-- Discrete drawdown episodes identified by DrawdownController.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drawdown_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts        TEXT    NOT NULL,   -- ISO-8601 timestamp of the peak
    trough_ts       TEXT,               -- ISO-8601 timestamp of the trough
    end_ts          TEXT,               -- ISO-8601 timestamp of recovery (NULL = not recovered)
    peak_equity     REAL    NOT NULL,
    trough_equity   REAL,
    drawdown_pct    REAL,               -- (peak - trough) / peak
    duration_bars   INTEGER,            -- bars from peak to recovery (or end of series)
    regime          TEXT,               -- regime label at peak, if available
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_dd_start_ts    ON drawdown_events(start_ts);
CREATE INDEX IF NOT EXISTS idx_dd_drawdown    ON drawdown_events(drawdown_pct);
CREATE INDEX IF NOT EXISTS idx_dd_created_at  ON drawdown_events(created_at);
