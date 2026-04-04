-- =============================================================================
-- 01_market_data.sql
-- SRFM Lab — Market Data Schema
-- PostgreSQL 15+
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ---------------------------------------------------------------------------
-- Instruments master table
-- ---------------------------------------------------------------------------
-- Each row represents one tradeable instrument.  The CF (curvature factor)
-- columns hold the per-timeframe "light speed" calibration used by the BH
-- physics engine.  bh_form is the mass threshold at which a Black Hole is
-- considered to have formed on the daily timeframe.
CREATE TABLE IF NOT EXISTS instruments (
    id              SERIAL          PRIMARY KEY,
    symbol          VARCHAR(20)     NOT NULL UNIQUE,
    name            VARCHAR(200),
    asset_class     VARCHAR(20)     NOT NULL
                    CHECK (asset_class IN (
                        'equity_index', 'commodity', 'bond',
                        'forex', 'crypto', 'volatility'
                    )),
    base_currency   VARCHAR(10)     NOT NULL DEFAULT 'USD',
    quote_currency  VARCHAR(10)     NOT NULL DEFAULT 'USD',
    tick_size       DECIMAL(20,10),
    lot_size        DECIMAL(20,10),
    margin_rate     DECIMAL(10,6),
    -- BH curvature factors per timeframe
    cf_15m          DECIMAL(10,8)   NOT NULL,
    cf_1h           DECIMAL(10,8)   NOT NULL,
    cf_1d           DECIMAL(10,8)   NOT NULL,
    -- BH formation threshold (mass units at which BH is declared)
    bh_form         DECIMAL(10,4)   NOT NULL DEFAULT 1.5,
    -- BH collapse threshold (mass fraction at which BH dissolves)
    bh_collapse     DECIMAL(10,4)   NOT NULL DEFAULT 1.0,
    -- Per-bar mass decay multiplier (0 < decay < 1)
    bh_decay        DECIMAL(10,8)   NOT NULL DEFAULT 0.95,
    -- Correlation group for portfolio risk aggregation
    corr_group      VARCHAR(30),
    -- Alpaca ticker (may differ from canonical symbol)
    alpaca_ticker   VARCHAR(30),
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_instruments_asset_class
    ON instruments (asset_class);
CREATE INDEX IF NOT EXISTS idx_instruments_active
    ON instruments (is_active) WHERE is_active = TRUE;

COMMENT ON TABLE instruments IS
    'Master catalog of all tradeable instruments.  CF columns calibrate the '
    'BH physics engine per timeframe.  bh_form is the mass at which a Black '
    'Hole formation event is recorded.';
COMMENT ON COLUMN instruments.cf_15m IS
    'Curvature factor for 15-minute bars: the "light speed" of the instrument '
    'at this resolution.  Beta = |Δp/p| / CF.  Calibrated to ~1 std dev of '
    'intraday returns at 15m resolution.';
COMMENT ON COLUMN instruments.cf_1h IS 'Curvature factor for 1-hour bars.';
COMMENT ON COLUMN instruments.cf_1d IS 'Curvature factor for daily bars.';
COMMENT ON COLUMN instruments.bh_form IS
    'Accumulated mass (in CF units) at which the BH physics engine declares '
    'a Black Hole formation.  Higher values = stricter signal filter.';

-- ---------------------------------------------------------------------------
-- OHLCV bars — daily
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars_1d (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            DECIMAL(20,8)   NOT NULL,
    high            DECIMAL(20,8)   NOT NULL,
    low             DECIMAL(20,8)   NOT NULL,
    close           DECIMAL(20,8)   NOT NULL,
    volume          DECIMAL(20,4)   NOT NULL DEFAULT 0,
    vwap            DECIMAL(20,8),
    trade_count     INTEGER,
    -- Derived columns updated by trigger
    log_return      DECIMAL(20,12)  GENERATED ALWAYS AS (
                        LN(close / NULLIF(open, 0))
                    ) STORED,
    hl_range        DECIMAL(20,8)   GENERATED ALWAYS AS (high - low) STORED,
    PRIMARY KEY (instrument_id, timestamp)
) PARTITION BY LIST (instrument_id);

CREATE INDEX IF NOT EXISTS idx_bars_1d_ts
    ON bars_1d (timestamp DESC);

-- ---------------------------------------------------------------------------
-- OHLCV bars — 1 hour
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars_1h (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            DECIMAL(20,8)   NOT NULL,
    high            DECIMAL(20,8)   NOT NULL,
    low             DECIMAL(20,8)   NOT NULL,
    close           DECIMAL(20,8)   NOT NULL,
    volume          DECIMAL(20,4)   NOT NULL DEFAULT 0,
    vwap            DECIMAL(20,8),
    trade_count     INTEGER,
    log_return      DECIMAL(20,12)  GENERATED ALWAYS AS (
                        LN(close / NULLIF(open, 0))
                    ) STORED,
    hl_range        DECIMAL(20,8)   GENERATED ALWAYS AS (high - low) STORED,
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bars_1h_ts
    ON bars_1h (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bars_1h_inst_ts
    ON bars_1h (instrument_id, timestamp DESC);

-- ---------------------------------------------------------------------------
-- OHLCV bars — 15 minutes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars_15m (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            DECIMAL(20,8)   NOT NULL,
    high            DECIMAL(20,8)   NOT NULL,
    low             DECIMAL(20,8)   NOT NULL,
    close           DECIMAL(20,8)   NOT NULL,
    volume          DECIMAL(20,4)   NOT NULL DEFAULT 0,
    vwap            DECIMAL(20,8),
    trade_count     INTEGER,
    log_return      DECIMAL(20,12)  GENERATED ALWAYS AS (
                        LN(close / NULLIF(open, 0))
                    ) STORED,
    hl_range        DECIMAL(20,8)   GENERATED ALWAYS AS (high - low) STORED,
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bars_15m_inst_ts
    ON bars_15m (instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bars_15m_ts
    ON bars_15m (timestamp DESC);

-- ---------------------------------------------------------------------------
-- OHLCV bars — 5 minutes
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars_5m (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            DECIMAL(20,8)   NOT NULL,
    high            DECIMAL(20,8)   NOT NULL,
    low             DECIMAL(20,8)   NOT NULL,
    close           DECIMAL(20,8)   NOT NULL,
    volume          DECIMAL(20,4)   NOT NULL DEFAULT 0,
    vwap            DECIMAL(20,8),
    trade_count     INTEGER,
    log_return      DECIMAL(20,12)  GENERATED ALWAYS AS (
                        LN(close / NULLIF(open, 0))
                    ) STORED,
    hl_range        DECIMAL(20,8)   GENERATED ALWAYS AS (high - low) STORED,
    PRIMARY KEY (instrument_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_bars_5m_inst_ts
    ON bars_5m (instrument_id, timestamp DESC);

-- ---------------------------------------------------------------------------
-- OHLCV bars — 1 minute
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bars_1m (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    open            DECIMAL(20,8)   NOT NULL,
    high            DECIMAL(20,8)   NOT NULL,
    low             DECIMAL(20,8)   NOT NULL,
    close           DECIMAL(20,8)   NOT NULL,
    volume          DECIMAL(20,4)   NOT NULL DEFAULT 0,
    vwap            DECIMAL(20,8),
    trade_count     INTEGER,
    log_return      DECIMAL(20,12)  GENERATED ALWAYS AS (
                        LN(close / NULLIF(open, 0))
                    ) STORED,
    hl_range        DECIMAL(20,8)   GENERATED ALWAYS AS (high - low) STORED,
    PRIMARY KEY (instrument_id, timestamp)
) PARTITION BY RANGE (timestamp);

CREATE INDEX IF NOT EXISTS idx_bars_1m_inst_ts
    ON bars_1m (instrument_id, timestamp DESC);
-- 1-minute data is high volume; partition monthly
-- CREATE TABLE bars_1m_2024_01 PARTITION OF bars_1m
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ---------------------------------------------------------------------------
-- Best bid/ask quotes (level 1)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS quotes (
    id              BIGSERIAL       PRIMARY KEY,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    bid_price       DECIMAL(20,8),
    bid_size        DECIMAL(20,4),
    ask_price       DECIMAL(20,8),
    ask_size        DECIMAL(20,4),
    mid_price       DECIMAL(20,8)   GENERATED ALWAYS AS (
                        (bid_price + ask_price) / 2.0
                    ) STORED,
    spread          DECIMAL(20,8)   GENERATED ALWAYS AS (
                        ask_price - bid_price
                    ) STORED,
    source          VARCHAR(30),
    received_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_quotes_inst_ts
    ON quotes (instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_quotes_ts
    ON quotes (timestamp DESC);

COMMENT ON TABLE quotes IS
    'Level-1 tick data: best bid and ask at each timestamp.  '
    'Mid and spread are computed columns.';

-- ---------------------------------------------------------------------------
-- Market trades (individual fills on exchange)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_trades (
    id              BIGSERIAL       PRIMARY KEY,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    price           DECIMAL(20,8)   NOT NULL,
    size            DECIMAL(20,4)   NOT NULL,
    side            VARCHAR(4)      CHECK (side IN ('buy', 'sell')),
    trade_id        VARCHAR(50),
    source          VARCHAR(30),
    received_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_trades_inst_ts
    ON market_trades (instrument_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_trades_ts
    ON market_trades (timestamp DESC);

COMMENT ON TABLE market_trades IS
    'Individual exchange trade prints (tick data).  Used for volume profile '
    'and VWAP recalculation.  Extremely high volume for active instruments.';

-- ---------------------------------------------------------------------------
-- Funding rates (for crypto perpetuals)
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- Open interest snapshots (futures / perpetuals)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS open_interest (
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id) ON DELETE CASCADE,
    timestamp       TIMESTAMPTZ     NOT NULL,
    oi              DECIMAL(20,4)   NOT NULL,
    oi_change_pct   DECIMAL(10,6),
    source          VARCHAR(30),
    PRIMARY KEY (instrument_id, timestamp)
);

-- ---------------------------------------------------------------------------
-- Implied volatility surface snapshots
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- ATM implied volatility (derived / convenience)
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- Calendar / trading day metadata
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trading_calendar (
    calendar_date   DATE            PRIMARY KEY,
    is_trading_day  BOOLEAN         NOT NULL DEFAULT TRUE,
    market_open     TIME,
    market_close    TIME,
    notes           TEXT
);

-- Pre-populate weekend non-trading days via a function (called in seed data)
COMMENT ON TABLE trading_calendar IS
    'Which dates are trading days. Used to compute hold_bars correctly across '
    'weekends and holidays.';

-- ---------------------------------------------------------------------------
-- Helper view: latest close prices across all instruments
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW latest_prices AS
SELECT DISTINCT ON (instrument_id)
    i.symbol,
    i.asset_class,
    b.timestamp,
    b.close,
    b.volume,
    b.log_return
FROM bars_1d b
JOIN instruments i ON i.id = b.instrument_id
WHERE i.is_active = TRUE
ORDER BY instrument_id, timestamp DESC;

COMMENT ON VIEW latest_prices IS
    'Most recent daily close for each active instrument.';

-- ---------------------------------------------------------------------------
-- Helper view: intraday session summary (derived from 15m bars)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW intraday_session AS
SELECT
    instrument_id,
    DATE(timestamp)                         AS session_date,
    FIRST_VALUE(open) OVER w                AS session_open,
    MAX(high)         OVER w                AS session_high,
    MIN(low)          OVER w                AS session_low,
    LAST_VALUE(close) OVER w                AS session_close,
    SUM(volume)       OVER w                AS session_volume,
    COUNT(*)          OVER w                AS bars_count
FROM bars_15m
WINDOW w AS (
    PARTITION BY instrument_id, DATE(timestamp)
    ORDER BY timestamp
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
);

-- ---------------------------------------------------------------------------
-- Trigger: keep updated_at current on instruments
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_instruments_updated_at
    BEFORE UPDATE ON instruments
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- Function: upsert bar data (used by data loader)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION upsert_bar_1d(
    p_instrument_id INTEGER,
    p_timestamp     TIMESTAMPTZ,
    p_open          DECIMAL,
    p_high          DECIMAL,
    p_low           DECIMAL,
    p_close         DECIMAL,
    p_volume        DECIMAL,
    p_vwap          DECIMAL  DEFAULT NULL,
    p_trade_count   INTEGER  DEFAULT NULL
)
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO bars_1d (instrument_id, timestamp, open, high, low, close, volume, vwap, trade_count)
    VALUES (p_instrument_id, p_timestamp, p_open, p_high, p_low, p_close, p_volume, p_vwap, p_trade_count)
    ON CONFLICT (instrument_id, timestamp)
    DO UPDATE SET
        open        = EXCLUDED.open,
        high        = EXCLUDED.high,
        low         = EXCLUDED.low,
        close       = EXCLUDED.close,
        volume      = EXCLUDED.volume,
        vwap        = COALESCE(EXCLUDED.vwap, bars_1d.vwap),
        trade_count = COALESCE(EXCLUDED.trade_count, bars_1d.trade_count);
END;
$$;

-- Same function template for bars_1h and bars_15m
CREATE OR REPLACE FUNCTION upsert_bar_1h(
    p_instrument_id INTEGER,
    p_timestamp     TIMESTAMPTZ,
    p_open          DECIMAL,
    p_high          DECIMAL,
    p_low           DECIMAL,
    p_close         DECIMAL,
    p_volume        DECIMAL,
    p_vwap          DECIMAL  DEFAULT NULL,
    p_trade_count   INTEGER  DEFAULT NULL
)
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO bars_1h (instrument_id, timestamp, open, high, low, close, volume, vwap, trade_count)
    VALUES (p_instrument_id, p_timestamp, p_open, p_high, p_low, p_close, p_volume, p_vwap, p_trade_count)
    ON CONFLICT (instrument_id, timestamp)
    DO UPDATE SET
        open        = EXCLUDED.open,
        high        = EXCLUDED.high,
        low         = EXCLUDED.low,
        close       = EXCLUDED.close,
        volume      = EXCLUDED.volume,
        vwap        = COALESCE(EXCLUDED.vwap, bars_1h.vwap),
        trade_count = COALESCE(EXCLUDED.trade_count, bars_1h.trade_count);
END;
$$;

CREATE OR REPLACE FUNCTION upsert_bar_15m(
    p_instrument_id INTEGER,
    p_timestamp     TIMESTAMPTZ,
    p_open          DECIMAL,
    p_high          DECIMAL,
    p_low           DECIMAL,
    p_close         DECIMAL,
    p_volume        DECIMAL,
    p_vwap          DECIMAL  DEFAULT NULL,
    p_trade_count   INTEGER  DEFAULT NULL
)
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO bars_15m (instrument_id, timestamp, open, high, low, close, volume, vwap, trade_count)
    VALUES (p_instrument_id, p_timestamp, p_open, p_high, p_low, p_close, p_volume, p_vwap, p_trade_count)
    ON CONFLICT (instrument_id, timestamp)
    DO UPDATE SET
        open        = EXCLUDED.open,
        high        = EXCLUDED.high,
        low         = EXCLUDED.low,
        close       = EXCLUDED.close,
        volume      = EXCLUDED.volume,
        vwap        = COALESCE(EXCLUDED.vwap, bars_15m.vwap),
        trade_count = COALESCE(EXCLUDED.trade_count, bars_15m.trade_count);
END;
$$;

-- ---------------------------------------------------------------------------
-- Function: compute realized volatility over N bars (annualized)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION realized_vol(
    p_instrument_id INTEGER,
    p_end_ts        TIMESTAMPTZ,
    p_lookback      INTEGER     DEFAULT 20,
    p_timeframe     VARCHAR(5)  DEFAULT '1d',
    p_ann_factor    INTEGER     DEFAULT 252
)
RETURNS DECIMAL LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_stddev DECIMAL;
BEGIN
    IF p_timeframe = '1d' THEN
        SELECT STDDEV(log_return) * SQRT(p_ann_factor)
        INTO v_stddev
        FROM (
            SELECT log_return
            FROM bars_1d
            WHERE instrument_id = p_instrument_id
              AND timestamp <= p_end_ts
            ORDER BY timestamp DESC
            LIMIT p_lookback
        ) sub;
    ELSIF p_timeframe = '1h' THEN
        SELECT STDDEV(log_return) * SQRT(p_ann_factor * 6.5)
        INTO v_stddev
        FROM (
            SELECT log_return
            FROM bars_1h
            WHERE instrument_id = p_instrument_id
              AND timestamp <= p_end_ts
            ORDER BY timestamp DESC
            LIMIT p_lookback
        ) sub;
    ELSIF p_timeframe = '15m' THEN
        SELECT STDDEV(log_return) * SQRT(p_ann_factor * 26)
        INTO v_stddev
        FROM (
            SELECT log_return
            FROM bars_15m
            WHERE instrument_id = p_instrument_id
              AND timestamp <= p_end_ts
            ORDER BY timestamp DESC
            LIMIT p_lookback
        ) sub;
    END IF;
    RETURN v_stddev;
END;
$$;

COMMENT ON FUNCTION realized_vol IS
    'Compute annualized realized volatility from log returns over p_lookback bars.';

-- ---------------------------------------------------------------------------
-- Function: compute rolling z-score of closes
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION close_zscore(
    p_instrument_id INTEGER,
    p_end_ts        TIMESTAMPTZ,
    p_lookback      INTEGER DEFAULT 20
)
RETURNS DECIMAL LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_current  DECIMAL;
    v_mean     DECIMAL;
    v_stddev   DECIMAL;
BEGIN
    SELECT close
    INTO v_current
    FROM bars_1d
    WHERE instrument_id = p_instrument_id
      AND timestamp <= p_end_ts
    ORDER BY timestamp DESC
    LIMIT 1;

    SELECT AVG(close), STDDEV(close)
    INTO v_mean, v_stddev
    FROM (
        SELECT close
        FROM bars_1d
        WHERE instrument_id = p_instrument_id
          AND timestamp <= p_end_ts
        ORDER BY timestamp DESC
        LIMIT p_lookback
    ) sub;

    IF v_stddev IS NULL OR v_stddev = 0 THEN
        RETURN 0;
    END IF;

    RETURN (v_current - v_mean) / v_stddev;
END;
$$;
