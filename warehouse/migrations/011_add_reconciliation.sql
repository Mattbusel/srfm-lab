-- =============================================================================
-- Migration 011: Add reconciliation tables
-- Applied: 2026-04-04
-- =============================================================================
-- UP

-- live_fills: Actual fills received from Alpaca (or other live broker).
-- Captures execution price, quantity, and derived slippage vs the signal price.
CREATE TABLE IF NOT EXISTS live_fills (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id),
    fill_time       TIMESTAMPTZ     NOT NULL,
    order_id        VARCHAR(64)     NOT NULL,
    client_order_id VARCHAR(64),
    side            VARCHAR(4)      NOT NULL CHECK (side IN ('buy', 'sell')),
    qty             DECIMAL(18,8)   NOT NULL,
    fill_price      DECIMAL(18,8)   NOT NULL,
    signal_price    DECIMAL(18,8),          -- mid-price at signal generation time
    commission      DECIMAL(12,4)   NOT NULL DEFAULT 0,
    source          VARCHAR(20)     NOT NULL DEFAULT 'alpaca'
                                    CHECK (source IN ('alpaca', 'paper', 'ib', 'sim')),
    raw_payload     JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_live_fills_run
    ON live_fills (run_id, fill_time DESC);
CREATE INDEX IF NOT EXISTS idx_live_fills_instrument
    ON live_fills (instrument_id, fill_time DESC);
CREATE INDEX IF NOT EXISTS idx_live_fills_order
    ON live_fills (order_id);

-- bt_fills: Corresponding backtest fills for the same signal.
-- Used to compare live execution quality against simulated expectations.
CREATE TABLE IF NOT EXISTS bt_fills (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id),
    signal_time     TIMESTAMPTZ     NOT NULL,
    side            VARCHAR(4)      NOT NULL CHECK (side IN ('buy', 'sell')),
    qty             DECIMAL(18,8)   NOT NULL,
    bt_price        DECIMAL(18,8)   NOT NULL,  -- simulated fill price
    bt_commission   DECIMAL(12,4)   NOT NULL DEFAULT 0,
    fill_model      VARCHAR(30)     NOT NULL DEFAULT 'market_on_close'
                                    CHECK (fill_model IN (
                                        'market_on_open', 'market_on_close',
                                        'vwap', 'twap', 'next_open', 'limit'
                                    )),
    live_fill_id    INTEGER         REFERENCES live_fills (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_bt_fills_run
    ON bt_fills (run_id, signal_time DESC);
CREATE INDEX IF NOT EXISTS idx_bt_fills_instrument
    ON bt_fills (instrument_id, signal_time DESC);
CREATE INDEX IF NOT EXISTS idx_bt_fills_live
    ON bt_fills (live_fill_id);

-- recon_runs: Metadata record for each reconciliation analysis pass.
CREATE TABLE IF NOT EXISTS recon_runs (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     TIMESTAMPTZ,
    status          VARCHAR(20)     NOT NULL DEFAULT 'running'
                                    CHECK (status IN ('running', 'complete', 'failed')),
    period_from     TIMESTAMPTZ     NOT NULL,
    period_to       TIMESTAMPTZ     NOT NULL,
    fill_count      INTEGER         NOT NULL DEFAULT 0,
    matched_count   INTEGER         NOT NULL DEFAULT 0,
    unmatched_count INTEGER         NOT NULL DEFAULT 0,
    total_slippage  DECIMAL(18,8),           -- sum of slippage_bps across all fills
    avg_slippage_bps DECIMAL(10,4),
    p95_slippage_bps DECIMAL(10,4),
    notes           TEXT,
    result_json     JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_recon_runs_run
    ON recon_runs (run_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_recon_runs_status
    ON recon_runs (status, started_at DESC);

-- slippage_log: Per-fill slippage breakdown for detailed analysis.
CREATE TABLE IF NOT EXISTS slippage_log (
    id              SERIAL          PRIMARY KEY,
    recon_run_id    INTEGER         NOT NULL REFERENCES recon_runs (id) ON DELETE CASCADE,
    live_fill_id    INTEGER         NOT NULL REFERENCES live_fills (id) ON DELETE CASCADE,
    bt_fill_id      INTEGER         REFERENCES bt_fills (id) ON DELETE SET NULL,
    instrument_id   INTEGER         NOT NULL REFERENCES instruments (id),
    fill_time       TIMESTAMPTZ     NOT NULL,
    side            VARCHAR(4)      NOT NULL CHECK (side IN ('buy', 'sell')),
    qty             DECIMAL(18,8)   NOT NULL,
    signal_price    DECIMAL(18,8),
    bt_price        DECIMAL(18,8),
    live_price      DECIMAL(18,8)   NOT NULL,
    raw_slippage    DECIMAL(18,8),           -- live_price - bt_price (signed)
    slippage_bps    DECIMAL(10,4),           -- raw_slippage / bt_price * 10000
    market_impact   DECIMAL(18,8),
    spread_cost     DECIMAL(18,8),
    timing_cost     DECIMAL(18,8),
    regime          VARCHAR(20),
    adv_fraction    DECIMAL(10,8)            -- qty / ADV at fill time
);

CREATE INDEX IF NOT EXISTS idx_slippage_log_recon
    ON slippage_log (recon_run_id, fill_time DESC);
CREATE INDEX IF NOT EXISTS idx_slippage_log_instrument
    ON slippage_log (instrument_id, fill_time DESC);
CREATE INDEX IF NOT EXISTS idx_slippage_log_regime
    ON slippage_log (regime, fill_time DESC);

-- drift_log: Signal drift detection events.
-- Fires when live PnL diverges from backtest PnL beyond a threshold.
CREATE TABLE IF NOT EXISTS drift_log (
    id              SERIAL          PRIMARY KEY,
    run_id          INTEGER         NOT NULL REFERENCES strategy_runs (id) ON DELETE CASCADE,
    detected_at     TIMESTAMPTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    instrument_id   INTEGER         REFERENCES instruments (id),
    drift_type      VARCHAR(30)     NOT NULL CHECK (drift_type IN (
                        'pnl_divergence', 'signal_flip', 'fill_ratio',
                        'turnover_spike', 'regime_mismatch', 'cost_overrun'
                    )),
    severity        VARCHAR(10)     NOT NULL DEFAULT 'warning'
                                    CHECK (severity IN ('info', 'warning', 'critical')),
    live_value      DECIMAL(18,8),
    bt_value        DECIMAL(18,8),
    delta           DECIMAL(18,8),
    threshold       DECIMAL(18,8),
    window_days     INTEGER,
    resolved_at     TIMESTAMPTZ,
    details         JSONB           NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_drift_log_run
    ON drift_log (run_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_log_type
    ON drift_log (drift_type, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_log_severity
    ON drift_log (severity, detected_at DESC);

INSERT INTO _schema_migrations (version, name) VALUES (11, 'add_reconciliation')
ON CONFLICT (version) DO NOTHING;

-- DOWN
-- DROP TABLE IF EXISTS drift_log;
-- DROP TABLE IF EXISTS slippage_log;
-- DROP TABLE IF EXISTS recon_runs;
-- DROP TABLE IF EXISTS bt_fills;
-- DROP TABLE IF EXISTS live_fills;
-- DELETE FROM _schema_migrations WHERE version = 11;
