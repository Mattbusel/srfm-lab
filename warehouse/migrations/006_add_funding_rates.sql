-- =============================================================================
-- Migration 006: Add funding_rates and open_interest tables for crypto
-- Applied: 2024-06-01
-- =============================================================================
-- UP
CREATE TABLE IF NOT EXISTS funding_rates (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    rate            DECIMAL(20,10)  NOT NULL,
    predicted_rate  DECIMAL(20,10),
    source          VARCHAR(30),
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_funding_rates_inst_ts
    ON funding_rates (instrument_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS open_interest (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    oi              DECIMAL(20,4)   NOT NULL,
    oi_change_pct   DECIMAL(10,6),
    source          VARCHAR(30),
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_open_interest_inst_ts
    ON open_interest (instrument_id, timestamp DESC);

INSERT INTO _schema_migrations (version, name) VALUES (6, 'add_funding_rates')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS open_interest;
-- DROP TABLE IF EXISTS funding_rates;
-- DELETE FROM _schema_migrations WHERE version = 6;
