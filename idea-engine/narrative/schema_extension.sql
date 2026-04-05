-- schema_extension.sql
-- Narrative Intelligence — database schema for the Idea Automation Engine
-- Run against idea_engine.db to set up required tables.

CREATE TABLE IF NOT EXISTS narrative_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type     TEXT    NOT NULL,  -- weekly / idea_card / genome_bio / regime
    subject_id      TEXT,              -- hypothesis_id / genome_id / regime name
    content         TEXT    NOT NULL,  -- full Markdown content
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_narrative_reports_type
    ON narrative_reports (report_type);

CREATE INDEX IF NOT EXISTS idx_narrative_reports_created
    ON narrative_reports (created_at DESC);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS narrative_alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type   TEXT    NOT NULL,  -- PROMOTE_SHADOW / PARAMETER_UPDATE /
                                    --   RESEARCH_ALERT / EVOLUTION_BREAKTHROUGH
    severity     TEXT    NOT NULL DEFAULT 'info',  -- info / medium / high
    message      TEXT    NOT NULL,
    data_json    TEXT,              -- JSON: supporting event data
    acknowledged INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_narrative_alerts_type
    ON narrative_alerts (alert_type);

CREATE INDEX IF NOT EXISTS idx_narrative_alerts_acked
    ON narrative_alerts (acknowledged);

CREATE INDEX IF NOT EXISTS idx_narrative_alerts_created
    ON narrative_alerts (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_narrative_alerts_severity
    ON narrative_alerts (severity);
