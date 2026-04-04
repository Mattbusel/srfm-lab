-- =============================================================================
-- Migration 009: Add IV surface and ATM IV tables for options research
-- Applied: 2024-09-01
-- =============================================================================
-- UP
CREATE TABLE IF NOT EXISTS iv_surface (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    expiry_date     DATE            NOT NULL,
    strike          DECIMAL(20,8)   NOT NULL,
    option_type     VARCHAR(4)      NOT NULL CHECK (option_type IN ('call', 'put')),
    iv              DECIMAL(10,6)   NOT NULL,
    delta           DECIMAL(10,6),
    gamma           DECIMAL(10,6),
    theta           DECIMAL(10,6),
    vega            DECIMAL(10,6),
    PRIMARY KEY (instrument_id, timestamp, expiry_date, strike, option_type)
);

CREATE INDEX IF NOT EXISTS idx_iv_surface_inst_ts
    ON iv_surface (instrument_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS atm_iv (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    iv_7d           DECIMAL(10,6),
    iv_30d          DECIMAL(10,6),
    iv_60d          DECIMAL(10,6),
    iv_90d          DECIMAL(10,6),
    iv_180d         DECIMAL(10,6),
    iv_rank         DECIMAL(10,6)   CHECK (iv_rank BETWEEN 0 AND 1),
    iv_percentile   DECIMAL(10,6)   CHECK (iv_percentile BETWEEN 0 AND 1),
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_atm_iv_inst_ts
    ON atm_iv (instrument_id, timestamp DESC);

INSERT INTO _schema_migrations (version, name) VALUES (9, 'add_iv_surface')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS atm_iv;
-- DROP TABLE IF EXISTS iv_surface;
-- DELETE FROM _schema_migrations WHERE version = 9;
