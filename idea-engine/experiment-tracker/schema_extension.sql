-- experiment-tracker/schema_extension.sql
-- Extends idea_engine.db with tables required by the Experiment Tracker subsystem.
-- All statements use CREATE TABLE IF NOT EXISTS so this script is safe to re-run.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─── Experiments ───────────────────────────────────────────────────────────
-- One row per experiment (hypothesis test, genome run, counterfactual, etc.).
CREATE TABLE IF NOT EXISTS experiments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL,
    hypothesis_id    INTEGER,                             -- FK to hypotheses.id (soft ref)
    genome_id        INTEGER,                             -- FK to genome runs (soft ref)
    status           TEXT    NOT NULL DEFAULT 'running',  -- running | completed | failed | cancelled
    started_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    ended_at         TEXT,
    duration_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis_id
    ON experiments (hypothesis_id);

CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments (status);

CREATE INDEX IF NOT EXISTS idx_experiments_started_at
    ON experiments (started_at DESC);

-- ─── Experiment parameters ─────────────────────────────────────────────────
-- Immutable key-value parameters logged at experiment start.
-- (experiment_id, key) is the PK so each param can only be set once.
CREATE TABLE IF NOT EXISTS experiment_params (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    key           TEXT    NOT NULL,
    value         TEXT    NOT NULL,
    PRIMARY KEY (experiment_id, key)
);

CREATE INDEX IF NOT EXISTS idx_experiment_params_key
    ON experiment_params (key, value);

-- ─── Experiment metrics ────────────────────────────────────────────────────
-- Multiple metric values per key are allowed (forming curves when step is set).
CREATE TABLE IF NOT EXISTS experiment_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    key           TEXT    NOT NULL,
    value         REAL    NOT NULL,
    step          INTEGER,                                -- fold index or training step
    logged_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_experiment_metrics_exp_key
    ON experiment_metrics (experiment_id, key);

CREATE INDEX IF NOT EXISTS idx_experiment_metrics_key_value
    ON experiment_metrics (key, value);

-- ─── Experiment artifacts ──────────────────────────────────────────────────
-- Arbitrary JSON/text blobs associated with an experiment (e.g. serialised
-- DataFrames, plots encoded as base64, config snapshots).
CREATE TABLE IF NOT EXISTS experiment_artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name          TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_experiment_artifacts_exp_name
    ON experiment_artifacts (experiment_id, name);

-- ─── Experiment lineage ────────────────────────────────────────────────────
-- Parent–child relationships between experiments.
-- relationship types: hypothesis_test | parameter_variation | regime_split |
--                     wfa_fold | reproduction
CREATE TABLE IF NOT EXISTS experiment_lineage (
    child_id     INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    parent_id    INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    relationship TEXT    NOT NULL,
    PRIMARY KEY (child_id, parent_id)
);

CREATE INDEX IF NOT EXISTS idx_experiment_lineage_parent
    ON experiment_lineage (parent_id);

CREATE INDEX IF NOT EXISTS idx_experiment_lineage_child
    ON experiment_lineage (child_id);
