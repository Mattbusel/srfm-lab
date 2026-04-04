-- =============================================================================
-- Migration 003: Add exit_reason to trades, add hold_hours column
-- Applied: 2024-03-01
-- =============================================================================
-- UP
ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS exit_reason VARCHAR(30)
        CHECK (exit_reason IN (
            'target', 'stop', 'bh_collapse', 'regime_change',
            'timeout', 'manual', 'end_of_backtest'
        ));

ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS hold_hours DECIMAL(10,2);

-- Backfill hold_hours from hold_bars (assuming 6.5 hours per trading day)
UPDATE trades
SET hold_hours = hold_bars * 24.0
WHERE hold_hours IS NULL AND hold_bars IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_trades_exit_reason
    ON trades (exit_reason, run_id);

INSERT INTO _schema_migrations (version, name) VALUES (3, 'add_trade_exit_reason')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- ALTER TABLE trades DROP COLUMN IF EXISTS exit_reason;
-- ALTER TABLE trades DROP COLUMN IF EXISTS hold_hours;
-- DROP INDEX IF EXISTS idx_trades_exit_reason;
-- DELETE FROM _schema_migrations WHERE version = 3;
