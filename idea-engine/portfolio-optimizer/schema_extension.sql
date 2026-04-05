-- ============================================================
-- portfolio-optimizer/schema_extension.sql
-- Schema additions for the Portfolio Optimizer subsystem.
-- Apply on top of the base idea_engine.db schema.
-- SQLite 3.x
-- ============================================================

-- ------------------------------------------------------------
-- portfolio_allocations
-- Stores the result of each portfolio optimisation run.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS portfolio_allocations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    method          TEXT    NOT NULL,   -- 'mean_variance','hrp','risk_parity','black_litterman', etc.
    genome_ids      TEXT    NOT NULL,   -- JSON list of genome / asset identifiers
    weights         TEXT    NOT NULL,   -- JSON dict { genome_id: weight }
    expected_sharpe REAL,               -- estimated Sharpe ratio for this allocation
    expected_dd     REAL,               -- estimated maximum drawdown
    computed_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_palloc_method     ON portfolio_allocations(method);
CREATE INDEX IF NOT EXISTS idx_palloc_computed   ON portfolio_allocations(computed_at);
CREATE INDEX IF NOT EXISTS idx_palloc_sharpe     ON portfolio_allocations(expected_sharpe);

-- ------------------------------------------------------------
-- rebalance_events
-- Audit log of every portfolio rebalance that was executed.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS rebalance_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,               -- ISO-8601 timestamp of the rebalance
    old_weights     TEXT    NOT NULL,               -- JSON dict of pre-rebalance weights
    new_weights     TEXT    NOT NULL,               -- JSON dict of post-rebalance weights
    trigger         TEXT    NOT NULL,               -- 'drift' | 'scheduled' | 'vol_spike' | 'manual'
    estimated_cost  REAL,                           -- fraction of AUM spent on transaction costs
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_reb_ts       ON rebalance_events(ts);
CREATE INDEX IF NOT EXISTS idx_reb_trigger  ON rebalance_events(trigger);
CREATE INDEX IF NOT EXISTS idx_reb_created  ON rebalance_events(created_at);
