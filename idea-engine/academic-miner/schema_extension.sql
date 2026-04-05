-- schema_extension.sql
-- Academic Miner — database schema for the Idea Automation Engine
-- Run against idea_engine.db to set up required tables.

CREATE TABLE IF NOT EXISTS academic_papers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT    NOT NULL,  -- 'arxiv', 'ssrn', 'local'
    paper_id        TEXT    UNIQUE,
    title           TEXT    NOT NULL,
    authors         TEXT,              -- JSON array
    abstract        TEXT,
    relevance_score REAL,
    categories      TEXT,              -- JSON array
    url             TEXT,
    mined_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_academic_papers_source
    ON academic_papers (source);

CREATE INDEX IF NOT EXISTS idx_academic_papers_score
    ON academic_papers (relevance_score DESC);

CREATE INDEX IF NOT EXISTS idx_academic_papers_mined
    ON academic_papers (mined_at DESC);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hypothesis_candidates (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper_id  INTEGER REFERENCES academic_papers(id),
    hypothesis_text  TEXT    NOT NULL,
    mapped_component TEXT,   -- entry_signal / exit_rule / position_sizing /
                              --   risk_management / regime_filter
    param_suggestions TEXT,  -- JSON: suggested parameter changes
    confidence       REAL,
    status           TEXT    NOT NULL DEFAULT 'pending',
                              --   pending / tested / rejected / adopted
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_hyp_cand_status
    ON hypothesis_candidates (status);

CREATE INDEX IF NOT EXISTS idx_hyp_cand_confidence
    ON hypothesis_candidates (confidence DESC);

CREATE INDEX IF NOT EXISTS idx_hyp_cand_paper
    ON hypothesis_candidates (source_paper_id);
