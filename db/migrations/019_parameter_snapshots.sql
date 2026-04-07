-- Migration 019: Parameter snapshots
-- Full system parameter state at every IAE update cycle

-- UP

CREATE TABLE IF NOT EXISTS parameter_snapshots (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_time   TEXT    NOT NULL DEFAULT (datetime('now')),
  -- Source of this parameter update
  source          TEXT    NOT NULL CHECK(source IN (
                    'IAE_CYCLE','MANUAL','REGIME_CHANGE','STARTUP','ROLLBACK','SCHEDULE'
                  )),
  -- Full parameter blob as JSON
  params_json     TEXT    NOT NULL,
  -- Delta from previous snapshot (JSON patch RFC 6902)
  delta_json      TEXT,
  -- Human-readable summary of what changed
  change_summary  TEXT,
  -- Performance context that triggered the update
  trigger_sharpe    REAL,
  trigger_drawdown  REAL,
  trigger_regime    TEXT,
  -- IAE genome that produced these params
  genome_id         TEXT,
  genome_generation INTEGER,
  genome_fitness    REAL,
  -- Validation result
  validation_passed INTEGER NOT NULL DEFAULT 1 CHECK(validation_passed IN (0,1)),
  validation_errors TEXT,
  -- Operator who applied (for manual changes)
  applied_by      TEXT,
  notes           TEXT,
  -- Rollback pointer
  rolled_back_from INTEGER,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_param_snapshots_time   ON parameter_snapshots(snapshot_time);
CREATE INDEX IF NOT EXISTS idx_param_snapshots_source ON parameter_snapshots(source);
CREATE INDEX IF NOT EXISTS idx_param_snapshots_genome ON parameter_snapshots(genome_id);

-- DOWN

DROP INDEX IF EXISTS idx_param_snapshots_genome;
DROP INDEX IF EXISTS idx_param_snapshots_source;
DROP INDEX IF EXISTS idx_param_snapshots_time;
DROP TABLE IF EXISTS parameter_snapshots;
