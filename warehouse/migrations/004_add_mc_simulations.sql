-- =============================================================================
-- Migration 004: Add mc_simulations table, add AR(1) rho column
-- Applied: 2024-04-01
-- =============================================================================
-- UP
CREATE TABLE IF NOT EXISTS mc_simulations (
    id              SERIAL          PRIMARY KEY,
    source_run_id   INTEGER         REFERENCES strategy_runs (id),
    n_sims          INTEGER         NOT NULL,
    n_trades        INTEGER         NOT NULL,
    initial_equity  DECIMAL(15,2)   NOT NULL,
    p05_final       DECIMAL(15,2),
    p25_final       DECIMAL(15,2),
    p50_final       DECIMAL(15,2),
    p75_final       DECIMAL(15,2),
    p95_final       DECIMAL(15,2),
    p05_max_dd      DECIMAL(10,6),
    p50_max_dd      DECIMAL(10,6),
    p95_max_dd      DECIMAL(10,6),
    blowup_prob     DECIMAL(10,6),
    use_regime_aware BOOLEAN        NOT NULL DEFAULT TRUE,
    ar1_rho         DECIMAL(10,6),
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mc_simulations_run
    ON mc_simulations (source_run_id, created_at DESC);

INSERT INTO _schema_migrations (version, name) VALUES (4, 'add_mc_simulations')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS mc_simulations;
-- DELETE FROM _schema_migrations WHERE version = 4;
