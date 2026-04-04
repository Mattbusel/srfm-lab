-- =============================================================================
-- 02_bh_state.sql
-- SRFM Lab — Black Hole Physics State Tracking Schema
-- PostgreSQL 15+
-- =============================================================================
-- This schema captures the full time-series state of the BH physics engine
-- at each bar.  The BH engine computes:
--   beta  = |Δp / p| / CF          (relativistic speed of the bar move)
--   gamma = 1 / sqrt(1 - beta²)    (Lorentz factor)
--   mass  += (gamma - 1)           (accumulated gravitational mass)
--   mass  *= decay                 (exponential decay between bars)
-- A BH forms when mass >= bh_form.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- BH state timeseries — daily
-- ---------------------------------------------------------------------------
-- One row per (instrument, bar).  Written by the BH engine after each bar.
CREATE TABLE IF NOT EXISTS bh_state_1d (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    -- Core BH physics state
    mass            DECIMAL(10,6)   NOT NULL DEFAULT 0,
    beta            DECIMAL(10,8),          -- relativistic beta of the bar
    gamma           DECIMAL(10,8),          -- Lorentz factor
    active          BOOLEAN         NOT NULL DEFAULT FALSE,
    bh_dir          SMALLINT        CHECK (bh_dir IN (-1, 0, 1)),
    ctl             INTEGER         NOT NULL DEFAULT 0,
    -- Regime classification at this bar
    regime          VARCHAR(20)     CHECK (regime IN (
                        'BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY',
                        'TRANSITION', 'UNKNOWN'
                    )),
    -- Derived convenience columns
    bars_since_form INTEGER,        -- bars elapsed since last BH formation
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bh_state_1d_inst
    ON bh_state_1d (instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bh_state_1d_active
    ON bh_state_1d (active, timestamp DESC) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_bh_state_1d_regime
    ON bh_state_1d (regime, timestamp DESC);

COMMENT ON TABLE bh_state_1d IS
    'Per-bar BH physics state at daily resolution.  mass is the accumulated '
    'gravitational mass in CF-normalized units.  active=TRUE when mass >= bh_form.';

-- ---------------------------------------------------------------------------
-- BH state timeseries — 1 hour
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bh_state_1h (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    mass            DECIMAL(10,6)   NOT NULL DEFAULT 0,
    beta            DECIMAL(10,8),
    gamma           DECIMAL(10,8),
    active          BOOLEAN         NOT NULL DEFAULT FALSE,
    bh_dir          SMALLINT        CHECK (bh_dir IN (-1, 0, 1)),
    ctl             INTEGER         NOT NULL DEFAULT 0,
    regime          VARCHAR(20)     CHECK (regime IN (
                        'BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY',
                        'TRANSITION', 'UNKNOWN'
                    )),
    bars_since_form INTEGER,
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bh_state_1h_inst
    ON bh_state_1h (instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bh_state_1h_active
    ON bh_state_1h (active, timestamp DESC) WHERE active = TRUE;

-- ---------------------------------------------------------------------------
-- BH state timeseries — 15 minutes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bh_state_15m (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    mass            DECIMAL(10,6)   NOT NULL DEFAULT 0,
    beta            DECIMAL(10,8),
    gamma           DECIMAL(10,8),
    active          BOOLEAN         NOT NULL DEFAULT FALSE,
    bh_dir          SMALLINT        CHECK (bh_dir IN (-1, 0, 1)),
    ctl             INTEGER         NOT NULL DEFAULT 0,
    regime          VARCHAR(20)     CHECK (regime IN (
                        'BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY',
                        'TRANSITION', 'UNKNOWN'
                    )),
    bars_since_form INTEGER,
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bh_state_15m_inst
    ON bh_state_15m (instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bh_state_15m_active
    ON bh_state_15m (active, timestamp DESC) WHERE active = TRUE;

-- ---------------------------------------------------------------------------
-- BH formation events
-- ---------------------------------------------------------------------------
-- One row per BH lifecycle (formation → collapse).
-- Linked to the timeseries tables via formed_at / collapsed_at timestamps.
CREATE TABLE IF NOT EXISTS bh_formations (
    id                      SERIAL      PRIMARY KEY,
    instrument_id           INTEGER     NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timeframe               VARCHAR(5)  NOT NULL CHECK (timeframe IN ('15m', '1h', '1d')),
    formed_at               TIMESTAMPTZ NOT NULL,
    collapsed_at            TIMESTAMPTZ,
    peak_mass               DECIMAL(10,6),
    peak_mass_at            TIMESTAMPTZ,
    duration_bars           INTEGER,
    direction               SMALLINT    NOT NULL CHECK (direction IN (-1, 1)),
    regime_at_formation     VARCHAR(20),
    -- Entry / exit price context
    price_at_formation      DECIMAL(20,8),
    price_at_collapse       DECIMAL(20,8),
    price_move_pct          DECIMAL(10,6)
                            GENERATED ALWAYS AS (
                                (price_at_collapse - price_at_formation)
                                / NULLIF(price_at_formation, 0)
                            ) STORED,
    -- Whether the BH led to a profitable trade (filled in after trade closes)
    was_profitable          BOOLEAN,
    associated_trade_id     BIGINT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bh_formations_inst
    ON bh_formations (instrument_id, formed_at DESC);
CREATE INDEX IF NOT EXISTS idx_bh_formations_tf
    ON bh_formations (timeframe, formed_at DESC);
CREATE INDEX IF NOT EXISTS idx_bh_formations_dir
    ON bh_formations (direction, formed_at DESC);

COMMENT ON TABLE bh_formations IS
    'One row per complete BH lifecycle.  formed_at is when mass first crossed '
    'bh_form; collapsed_at is when mass fell below bh_collapse.  '
    'duration_bars measures longevity of the formation.';

-- ---------------------------------------------------------------------------
-- Multi-timeframe BH confluence events
-- ---------------------------------------------------------------------------
-- When BHs form simultaneously across multiple timeframes on the same
-- instrument, this is a high-conviction signal (tf_score ≥ 3).
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
    -- Net directional consensus (-1, 0, +1)
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

-- ---------------------------------------------------------------------------
-- Regime periods
-- ---------------------------------------------------------------------------
-- Contiguous run of the same regime label for one instrument.
-- start_bar → end_bar define the inclusive range.
CREATE TABLE IF NOT EXISTS regime_periods (
    id                  SERIAL      PRIMARY KEY,
    instrument_id       INTEGER     NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timeframe           VARCHAR(5)  NOT NULL DEFAULT '1d',
    regime              VARCHAR(20) NOT NULL CHECK (regime IN (
                            'BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY',
                            'TRANSITION', 'UNKNOWN'
                        )),
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    duration_bars       INTEGER,
    -- Price performance over the regime period
    price_at_start      DECIMAL(20,8),
    price_at_end        DECIMAL(20,8),
    return_during       DECIMAL(10,6)
                        GENERATED ALWAYS AS (
                            (price_at_end - price_at_start)
                            / NULLIF(price_at_start, 0)
                        ) STORED,
    -- Volatility summary
    realized_vol        DECIMAL(10,6),
    max_drawdown        DECIMAL(10,6),
    -- Count of BH formations during this regime
    bh_formations_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_regime_periods_inst
    ON regime_periods (instrument_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_regime_periods_regime
    ON regime_periods (regime, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_regime_periods_active
    ON regime_periods (ended_at) WHERE ended_at IS NULL;

COMMENT ON TABLE regime_periods IS
    'Contiguous regime runs per instrument.  Ended_at=NULL means the regime '
    'is current.  return_during is computed from price_at_start/end.';

-- ---------------------------------------------------------------------------
-- Regime transition matrix (updated periodically)
-- ---------------------------------------------------------------------------
-- P[from_regime][to_regime] = empirical transition probability
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

-- ---------------------------------------------------------------------------
-- BH mass snapshots — condensed for fast lookups
-- ---------------------------------------------------------------------------
-- Updated at end of each bar by the live engine.
-- Allows O(1) lookup: "what is the current BH state for all instruments?"
CREATE TABLE IF NOT EXISTS bh_current_state (
    instrument_id   INTEGER     NOT NULL PRIMARY KEY REFERENCES instruments (id) ON DELETE CASCADE,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Daily BH state
    mass_1d         DECIMAL(10,6),
    active_1d       BOOLEAN     NOT NULL DEFAULT FALSE,
    dir_1d          SMALLINT,
    regime_1d       VARCHAR(20),
    -- Hourly BH state
    mass_1h         DECIMAL(10,6),
    active_1h       BOOLEAN     NOT NULL DEFAULT FALSE,
    dir_1h          SMALLINT,
    -- 15-minute BH state
    mass_15m        DECIMAL(10,6),
    active_15m      BOOLEAN     NOT NULL DEFAULT FALSE,
    dir_15m         SMALLINT,
    -- Computed tf_score (bit mask: bit0=15m, bit1=1h, bit2=1d; then direction bits)
    tf_score        SMALLINT    NOT NULL DEFAULT 0
);

COMMENT ON TABLE bh_current_state IS
    'Live snapshot: current BH state across all three timeframes for each '
    'instrument.  Updated at the close of each bar.  tf_score is a 3-bit '
    'activation mask combined with direction: the primary entry filter.';

-- ---------------------------------------------------------------------------
-- Function: compute tf_score from three BH states
-- ---------------------------------------------------------------------------
-- tf_score encoding:
--   bits 0-2 = activation flags (15m=bit0, 1h=bit1, 1d=bit2)
--   direction bonus: +1 if all active timeframes agree on direction
--   range 0-7, where 7 = all three active + directional agreement
CREATE OR REPLACE FUNCTION compute_tf_score(
    p_active_15m BOOLEAN,
    p_active_1h  BOOLEAN,
    p_active_1d  BOOLEAN,
    p_dir_15m    SMALLINT,
    p_dir_1h     SMALLINT,
    p_dir_1d     SMALLINT
)
RETURNS SMALLINT LANGUAGE plpgsql IMMUTABLE AS $$
DECLARE
    v_score  SMALLINT := 0;
    v_active INT      := 0;
    v_dir    INT      := 0;
BEGIN
    -- Activation bits
    IF p_active_15m THEN v_score := v_score + 1; v_active := v_active + 1; v_dir := v_dir + COALESCE(p_dir_15m, 0); END IF;
    IF p_active_1h  THEN v_score := v_score + 2; v_active := v_active + 1; v_dir := v_dir + COALESCE(p_dir_1h,  0); END IF;
    IF p_active_1d  THEN v_score := v_score + 4; v_active := v_active + 1; v_dir := v_dir + COALESCE(p_dir_1d,  0); END IF;

    -- Directional bonus: all active TFs agree → max score
    IF v_active > 0 AND ABS(v_dir) = v_active THEN
        v_score := LEAST(v_score + 1, 7);
    END IF;

    RETURN v_score;
END;
$$;

COMMENT ON FUNCTION compute_tf_score IS
    'Encode three BH activation states + directions into a 0-7 score.  '
    'Used as the primary entry filter: only take trades with tf_score >= threshold.';

-- ---------------------------------------------------------------------------
-- Function: update bh_current_state from latest bar state rows
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION refresh_bh_current_state(p_instrument_id INTEGER)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    v_state_1d  bh_state_1d%ROWTYPE;
    v_state_1h  bh_state_1h%ROWTYPE;
    v_state_15m bh_state_15m%ROWTYPE;
    v_tf_score  SMALLINT;
BEGIN
    SELECT * INTO v_state_1d
    FROM bh_state_1d
    WHERE instrument_id = p_instrument_id
    ORDER BY timestamp DESC LIMIT 1;

    SELECT * INTO v_state_1h
    FROM bh_state_1h
    WHERE instrument_id = p_instrument_id
    ORDER BY timestamp DESC LIMIT 1;

    SELECT * INTO v_state_15m
    FROM bh_state_15m
    WHERE instrument_id = p_instrument_id
    ORDER BY timestamp DESC LIMIT 1;

    v_tf_score := compute_tf_score(
        COALESCE(v_state_15m.active, FALSE),
        COALESCE(v_state_1h.active, FALSE),
        COALESCE(v_state_1d.active, FALSE),
        v_state_15m.bh_dir,
        v_state_1h.bh_dir,
        v_state_1d.bh_dir
    );

    INSERT INTO bh_current_state (
        instrument_id, updated_at,
        mass_1d, active_1d, dir_1d, regime_1d,
        mass_1h, active_1h, dir_1h,
        mass_15m, active_15m, dir_15m,
        tf_score
    )
    VALUES (
        p_instrument_id, CURRENT_TIMESTAMP,
        v_state_1d.mass,  v_state_1d.active,  v_state_1d.bh_dir, v_state_1d.regime,
        v_state_1h.mass,  v_state_1h.active,  v_state_1h.bh_dir,
        v_state_15m.mass, v_state_15m.active, v_state_15m.bh_dir,
        v_tf_score
    )
    ON CONFLICT (instrument_id) DO UPDATE SET
        updated_at   = EXCLUDED.updated_at,
        mass_1d      = EXCLUDED.mass_1d,
        active_1d    = EXCLUDED.active_1d,
        dir_1d       = EXCLUDED.dir_1d,
        regime_1d    = EXCLUDED.regime_1d,
        mass_1h      = EXCLUDED.mass_1h,
        active_1h    = EXCLUDED.active_1h,
        dir_1h       = EXCLUDED.dir_1h,
        mass_15m     = EXCLUDED.mass_15m,
        active_15m   = EXCLUDED.active_15m,
        dir_15m      = EXCLUDED.dir_15m,
        tf_score     = EXCLUDED.tf_score;
END;
$$;

-- ---------------------------------------------------------------------------
-- View: Current BH status dashboard
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW bh_dashboard AS
SELECT
    i.symbol,
    i.asset_class,
    i.corr_group,
    cs.tf_score,
    cs.active_1d,
    cs.active_1h,
    cs.active_15m,
    CASE cs.dir_1d WHEN  1 THEN 'LONG' WHEN -1 THEN 'SHORT' ELSE 'FLAT' END AS dir_1d,
    CASE cs.dir_1h WHEN  1 THEN 'LONG' WHEN -1 THEN 'SHORT' ELSE 'FLAT' END AS dir_1h,
    CASE cs.dir_15m WHEN 1 THEN 'LONG' WHEN -1 THEN 'SHORT' ELSE 'FLAT' END AS dir_15m,
    ROUND(cs.mass_1d::NUMERIC,  4)  AS mass_1d,
    ROUND(cs.mass_1h::NUMERIC,  4)  AS mass_1h,
    ROUND(cs.mass_15m::NUMERIC, 4)  AS mass_15m,
    cs.regime_1d                    AS regime,
    cs.updated_at
FROM bh_current_state cs
JOIN instruments i ON i.id = cs.instrument_id
WHERE i.is_active = TRUE
ORDER BY cs.tf_score DESC, i.symbol;

COMMENT ON VIEW bh_dashboard IS
    'Live BH status across all active instruments, ordered by tf_score.  '
    'Use this to monitor which instruments are near formation events.';

-- ---------------------------------------------------------------------------
-- View: Recent BH formations (last 30 days, all timeframes)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW recent_bh_formations AS
SELECT
    i.symbol,
    f.timeframe,
    f.formed_at,
    f.collapsed_at,
    f.duration_bars,
    f.direction,
    ROUND(f.peak_mass::NUMERIC, 4)      AS peak_mass,
    f.regime_at_formation               AS regime,
    ROUND(f.price_move_pct::NUMERIC, 4) AS price_move_pct,
    f.was_profitable
FROM bh_formations f
JOIN instruments i ON i.id = f.instrument_id
WHERE f.formed_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
ORDER BY f.formed_at DESC;
