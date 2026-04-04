-- =============================================================================
-- Migration 007: Add tags array and git_commit to strategy_runs
-- Applied: 2024-07-01
-- =============================================================================
-- UP
ALTER TABLE strategy_runs
    ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}';

ALTER TABLE strategy_runs
    ADD COLUMN IF NOT EXISTS git_commit VARCHAR(40);

ALTER TABLE strategy_runs
    ADD COLUMN IF NOT EXISTS notes TEXT;

CREATE INDEX IF NOT EXISTS idx_strategy_runs_tags_gin
    ON strategy_runs USING GIN (tags);

INSERT INTO _schema_migrations (version, name) VALUES (7, 'add_strategy_run_tags')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- ALTER TABLE strategy_runs DROP COLUMN IF EXISTS tags;
-- ALTER TABLE strategy_runs DROP COLUMN IF EXISTS git_commit;
-- ALTER TABLE strategy_runs DROP COLUMN IF EXISTS notes;
-- DROP INDEX IF EXISTS idx_strategy_runs_tags_gin;
-- DELETE FROM _schema_migrations WHERE version = 7;
