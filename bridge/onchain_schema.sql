-- onchain_schema.sql
-- Schema for all on-chain analytics tables used by OnChainBridge.
-- Apply via: sqlite3 data/onchain_signals.db < bridge/onchain_schema.sql

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Main composite signals table
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS onchain_signals (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol                  TEXT    NOT NULL,                  -- BTC, ETH
    composite_signal        REAL    NOT NULL,                  -- [-1, 1]
    mvrv_signal             REAL,
    sopr_signal             REAL,
    funding_signal          REAL,
    oi_signal               REAL,
    exchange_reserve_signal REAL,
    fear_greed_signal       REAL,
    confidence              REAL,                              -- 0..1
    timestamp               REAL    NOT NULL,                  -- Unix epoch (float)
    created_at              REAL    NOT NULL DEFAULT (unixepoch('now'))
);

CREATE INDEX IF NOT EXISTS idx_onchain_symbol_ts
    ON onchain_signals (symbol, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Raw MVRV readings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mvrv_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    mvrv_ratio      REAL    NOT NULL,   -- market_cap / realized_cap_proxy
    zscore          REAL    NOT NULL,
    signal          REAL    NOT NULL,
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'coingecko'
);

CREATE INDEX IF NOT EXISTS idx_mvrv_symbol_ts
    ON mvrv_readings (symbol, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Raw SOPR readings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sopr_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    sopr_value      REAL    NOT NULL,   -- proxy: price / MA30
    zscore          REAL    NOT NULL,
    signal          REAL    NOT NULL,
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'coingecko'
);

CREATE INDEX IF NOT EXISTS idx_sopr_symbol_ts
    ON sopr_readings (symbol, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Funding rate readings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS funding_rate_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,   -- BTC, ETH
    avg_rate        REAL    NOT NULL,   -- composite average across exchanges
    zscore          REAL    NOT NULL,
    signal          REAL    NOT NULL,
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'binance/bybit'
);

CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts
    ON funding_rate_readings (symbol, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Open interest readings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS oi_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    oi_change_pct   REAL    NOT NULL,   -- recent vs prior window
    zscore          REAL    NOT NULL,
    signal          REAL    NOT NULL,
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'binance'
);

CREATE INDEX IF NOT EXISTS idx_oi_symbol_ts
    ON oi_readings (symbol, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Exchange reserve (volume proxy) readings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exchange_reserve_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    volume_change   REAL    NOT NULL,
    zscore          REAL    NOT NULL,
    signal          REAL    NOT NULL,
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'coingecko'
);

CREATE INDEX IF NOT EXISTS idx_exres_symbol_ts
    ON exchange_reserve_readings (symbol, timestamp DESC);

-- ---------------------------------------------------------------------------
-- Fear and Greed Index readings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fear_greed_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_value       REAL    NOT NULL,   -- 0..100
    zscore          REAL    NOT NULL,
    signal          REAL    NOT NULL,   -- [-1, 1]
    classification  TEXT,               -- 'Extreme Fear', 'Fear', 'Neutral', etc.
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'alternative.me'
);

CREATE INDEX IF NOT EXISTS idx_fg_ts
    ON fear_greed_readings (timestamp DESC);

-- ---------------------------------------------------------------------------
-- Whale signal readings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS whale_signal_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    vol_zscore      REAL    NOT NULL,   -- volume z-score above 24h MA
    signal          REAL    NOT NULL,
    timestamp       REAL    NOT NULL,
    source          TEXT    NOT NULL DEFAULT 'coingecko'
);

CREATE INDEX IF NOT EXISTS idx_whale_symbol_ts
    ON whale_signal_readings (symbol, timestamp DESC);
