-- =============================================================================
-- Migration 001: Initial schema creation
-- Applied: 2024-01-01
-- =============================================================================
-- UP
\ir ../schema/01_market_data.sql
\ir ../schema/02_bh_state.sql
\ir ../schema/03_trades.sql
\ir ../schema/04_analytics.sql

CREATE TABLE IF NOT EXISTS _schema_migrations (
    version     INTEGER     PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    applied_at  TIMESTAMPTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO _schema_migrations (version, name) VALUES (1, 'initial_schema');

-- DOWN (rollback): DROP all tables in reverse dependency order
-- CAUTION: This destroys all data.
-- DROP MATERIALIZED VIEW IF EXISTS run_daily_metrics CASCADE;
-- DROP MATERIALIZED VIEW IF EXISTS instrument_correlation_matrix CASCADE;
-- DROP TABLE IF EXISTS mc_simulations, risk_events, position_snapshots CASCADE;
-- DROP TABLE IF EXISTS equity_snapshots, orders, trades, strategy_runs CASCADE;
-- DROP TABLE IF EXISTS bh_current_state, bh_confluence_events CASCADE;
-- DROP TABLE IF EXISTS regime_transition_matrix, regime_periods CASCADE;
-- DROP TABLE IF EXISTS bh_formations, bh_state_15m, bh_state_1h, bh_state_1d CASCADE;
-- DROP TABLE IF EXISTS atm_iv, iv_surface, open_interest, funding_rates CASCADE;
-- DROP TABLE IF EXISTS market_trades, quotes CASCADE;
-- DROP TABLE IF EXISTS bars_1m, bars_5m, bars_15m, bars_1h, bars_1d CASCADE;
-- DROP TABLE IF EXISTS trading_calendar, instruments CASCADE;
-- DELETE FROM _schema_migrations WHERE version = 1;
