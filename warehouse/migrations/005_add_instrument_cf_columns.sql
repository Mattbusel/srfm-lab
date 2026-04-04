-- =============================================================================
-- Migration 005: Add bh_collapse and bh_decay to instruments
-- Applied: 2024-05-01
-- =============================================================================
-- UP
ALTER TABLE instruments
    ADD COLUMN IF NOT EXISTS bh_collapse DECIMAL(10,4) NOT NULL DEFAULT 1.0;

ALTER TABLE instruments
    ADD COLUMN IF NOT EXISTS bh_decay DECIMAL(10,8) NOT NULL DEFAULT 0.95;

ALTER TABLE instruments
    ADD COLUMN IF NOT EXISTS corr_group VARCHAR(30);

ALTER TABLE instruments
    ADD COLUMN IF NOT EXISTS alpaca_ticker VARCHAR(30);

-- Noisy instruments get faster decay
UPDATE instruments SET bh_decay = 0.92 WHERE symbol IN ('VX', 'VIX', 'VIXY', 'NG', 'CL');

INSERT INTO _schema_migrations (version, name) VALUES (5, 'add_instrument_cf_columns')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- ALTER TABLE instruments
--     DROP COLUMN IF EXISTS bh_collapse,
--     DROP COLUMN IF EXISTS bh_decay,
--     DROP COLUMN IF EXISTS corr_group,
--     DROP COLUMN IF EXISTS alpaca_ticker;
-- DELETE FROM _schema_migrations WHERE version = 5;
