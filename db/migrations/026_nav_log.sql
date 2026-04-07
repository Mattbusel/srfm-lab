-- Migration 026: QuatNav state history per bar
-- Records full NAV and portfolio state at every processed bar

-- UP

CREATE TABLE IF NOT EXISTS nav_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  bar_time        TEXT    NOT NULL,
  symbol          TEXT    NOT NULL,
  timeframe       TEXT    NOT NULL DEFAULT '1m',
  -- NAV and capital
  nav             REAL    NOT NULL,
  cash            REAL    NOT NULL,
  gross_market_value REAL NOT NULL,
  net_market_value   REAL NOT NULL,
  -- Intraday P&L breakdown
  realized_pnl_today    REAL NOT NULL DEFAULT 0.0,
  unrealized_pnl        REAL NOT NULL DEFAULT 0.0,
  total_pnl_today       REAL NOT NULL DEFAULT 0.0,
  -- Position snapshot
  num_long_positions    INTEGER NOT NULL DEFAULT 0,
  num_short_positions   INTEGER NOT NULL DEFAULT 0,
  num_options_positions INTEGER NOT NULL DEFAULT 0,
  -- Market data at bar close
  close_price     REAL,
  volume          REAL,
  bid             REAL,
  ask             REAL,
  -- Strategy signals active
  active_signals_json TEXT,  -- JSON array of signal names currently active
  -- QuatNav internal state (JSON for extensibility)
  nav_state_json  TEXT,
  -- BH mass at this bar
  bh_mass         REAL,
  bh_curvature    REAL,
  -- Regime at this bar
  regime          TEXT,
  -- Performance since SOD
  pnl_pct_today   REAL,
  max_drawdown_today REAL,
  -- Fees and commissions incurred today so far
  fees_today      REAL NOT NULL DEFAULT 0.0,
  -- Bar processing metadata
  bar_latency_ms  INTEGER,  -- time to process this bar
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_nav_log_bar_time  ON nav_log(bar_time);
CREATE INDEX IF NOT EXISTS idx_nav_log_symbol    ON nav_log(symbol);
CREATE INDEX IF NOT EXISTS idx_nav_log_sym_tf    ON nav_log(symbol, timeframe);
CREATE UNIQUE INDEX IF NOT EXISTS idx_nav_log_sym_tf_bar ON nav_log(symbol, timeframe, bar_time);

-- DOWN

DROP INDEX IF EXISTS idx_nav_log_sym_tf_bar;
DROP INDEX IF EXISTS idx_nav_log_sym_tf;
DROP INDEX IF EXISTS idx_nav_log_symbol;
DROP INDEX IF EXISTS idx_nav_log_bar_time;
DROP TABLE IF EXISTS nav_log;
