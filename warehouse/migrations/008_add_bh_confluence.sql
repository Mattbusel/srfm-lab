-- =============================================================================
-- Migration 008: Add bh_confluence_events table and refresh function
-- Applied: 2024-08-01
-- =============================================================================
-- UP
CREATE TABLE IF NOT EXISTS bh_confluence_events (
    id              SERIAL      PRIMARY KEY,
    instrument_id   INTEGER     NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    event_time      TIMESTAMPTZ NOT NULL,
    tf_score        SMALLINT    NOT NULL CHECK (tf_score BETWEEN 0 AND 7),
    active_15m      BOOLEAN     NOT NULL DEFAULT FALSE,
    active_1h       BOOLEAN     NOT NULL DEFAULT FALSE,
    active_1d       BOOLEAN     NOT NULL DEFAULT FALSE,
    direction_15m   SMALLINT,
    direction_1h    SMALLINT,
    direction_1d    SMALLINT,
    mass_15m        DECIMAL(10,6),
    mass_1h         DECIMAL(10,6),
    mass_1d         DECIMAL(10,6),
    consensus_dir   SMALLINT    GENERATED ALWAYS AS (
                        SIGN(
                            COALESCE(direction_15m, 0)
                          + COALESCE(direction_1h,  0)
                          + COALESCE(direction_1d,  0)
                        )::SMALLINT
                    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_bh_confluence_inst_ts
    ON bh_confluence_events (instrument_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_bh_confluence_tf_score
    ON bh_confluence_events (tf_score DESC, event_time DESC);

INSERT INTO _schema_migrations (version, name) VALUES (8, 'add_bh_confluence')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS bh_confluence_events;
-- DELETE FROM _schema_migrations WHERE version = 8;
