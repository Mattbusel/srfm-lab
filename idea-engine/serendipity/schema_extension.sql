-- schema_extension.sql
-- Serendipity Generator — database schema for the Idea Automation Engine
-- Run against idea_engine.db to set up required tables.

CREATE TABLE IF NOT EXISTS serendipity_ideas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    technique       TEXT    NOT NULL,  -- random_combination / domain_borrow /
                                       --   inversion / extremization / analogy
    domain          TEXT,              -- thermodynamics / ecology / etc.
    idea_text       TEXT    NOT NULL,
    rationale       TEXT,
    complexity      TEXT,              -- low / medium / high
    experiment_json TEXT,              -- JSON: full experiment spec
    score           REAL    DEFAULT 0.0,
    status          TEXT    NOT NULL DEFAULT 'new',
                                       -- new / queued / testing / rejected / adopted
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_serendipity_technique
    ON serendipity_ideas (technique);

CREATE INDEX IF NOT EXISTS idx_serendipity_score
    ON serendipity_ideas (score DESC);

CREATE INDEX IF NOT EXISTS idx_serendipity_status
    ON serendipity_ideas (status);

CREATE INDEX IF NOT EXISTS idx_serendipity_created
    ON serendipity_ideas (created_at DESC);
