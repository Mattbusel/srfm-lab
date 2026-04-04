-- =============================================================================
-- Migration 002: Add timeframe column to regime_periods, add indexes
-- Applied: 2024-02-01
-- =============================================================================
-- UP
ALTER TABLE regime_periods
    ADD COLUMN IF NOT EXISTS timeframe VARCHAR(5) NOT NULL DEFAULT '1d';

CREATE INDEX IF NOT EXISTS idx_regime_periods_tf
    ON regime_periods (timeframe, started_at DESC);

-- Backfill existing rows
UPDATE regime_periods SET timeframe = '1d' WHERE timeframe IS NULL;

INSERT INTO _schema_migrations (version, name) VALUES (2, 'add_regime_periods_timeframe')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- ALTER TABLE regime_periods DROP COLUMN IF EXISTS timeframe;
-- DROP INDEX IF EXISTS idx_regime_periods_tf;
-- DELETE FROM _schema_migrations WHERE version = 2;
