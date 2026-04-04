-- =============================================================================
-- Migration 010: Add risk_events table and regime_transition_matrix
-- Applied: 2024-10-01
-- =============================================================================
-- UP
CREATE TABLE IF NOT EXISTS risk_events (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    event_time      TIMESTAMPTZ     NOT NULL,
    event_type      VARCHAR(30)     NOT NULL CHECK (event_type IN (
                        'max_drawdown_breach', 'daily_loss_limit', 'position_limit',
                        'margin_call', 'stop_hit', 'regime_emergency_exit', 'bh_mass_spike'
                    )),
    instrument_id   INTEGER         REFERENCES instruments (id),
    details         JSONB           NOT NULL DEFAULT '{}',
    equity_at_event DECIMAL(15,2),
    drawdown_at_event DECIMAL(10,6)
);

CREATE INDEX IF NOT EXISTS idx_risk_events_run
    ON risk_events (run_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_type
    ON risk_events (event_type, event_time DESC);

CREATE TABLE IF NOT EXISTS regime_transition_matrix (
    instrument_id   INTEGER     NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    from_regime     VARCHAR(20) NOT NULL,
    to_regime       VARCHAR(20) NOT NULL,
    count           INTEGER     NOT NULL DEFAULT 0,
    probability     DECIMAL(10,6),
    avg_duration    DECIMAL(10,2),
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument_id, from_regime, to_regime)
);

-- Function to recompute transition matrix for an instrument
CREATE OR REPLACE FUNCTION refresh_regime_transition_matrix(p_instrument_id INTEGER)
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM regime_transition_matrix WHERE instrument_id = p_instrument_id;

    INSERT INTO regime_transition_matrix (instrument_id, from_regime, to_regime, count, probability)
    WITH ordered AS (
        SELECT
            regime,
            LAG(regime) OVER (ORDER BY started_at) AS prev_regime,
            duration_bars
        FROM regime_periods
        WHERE instrument_id = p_instrument_id
          AND ended_at IS NOT NULL
    ),
    counts AS (
        SELECT prev_regime AS from_regime, regime AS to_regime, COUNT(*) AS cnt
        FROM ordered
        WHERE prev_regime IS NOT NULL
        GROUP BY prev_regime, regime
    ),
    totals AS (
        SELECT from_regime, SUM(cnt) AS total
        FROM counts
        GROUP BY from_regime
    )
    SELECT
        p_instrument_id,
        c.from_regime,
        c.to_regime,
        c.cnt,
        c.cnt::DECIMAL / NULLIF(t.total, 0)
    FROM counts c
    JOIN totals t ON t.from_regime = c.from_regime;
END;
$$;

INSERT INTO _schema_migrations (version, name) VALUES (10, 'add_risk_events')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP FUNCTION IF EXISTS refresh_regime_transition_matrix;
-- DROP TABLE IF EXISTS regime_transition_matrix;
-- DROP TABLE IF EXISTS risk_events;
-- DELETE FROM _schema_migrations WHERE version = 10;
