-- Migration 021: Execution quality metrics per trade
-- Tracks slippage, market impact, fill quality vs benchmark

-- UP

CREATE TABLE IF NOT EXISTS execution_quality (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_id        INTEGER NOT NULL,
  symbol          TEXT    NOT NULL,
  order_id        TEXT,
  -- Order details
  order_side      TEXT    NOT NULL CHECK(order_side IN ('BUY','SELL','BUY_TO_OPEN','SELL_TO_CLOSE','SELL_TO_OPEN','BUY_TO_CLOSE')),
  order_type      TEXT    NOT NULL CHECK(order_type IN ('MARKET','LIMIT','STOP','STOP_LIMIT','MOO','MOC','VWAP','TWAP')),
  order_qty       INTEGER NOT NULL,
  limit_price     REAL,
  -- Fill details
  fill_price      REAL    NOT NULL,
  fill_qty        INTEGER NOT NULL,
  fill_time       TEXT    NOT NULL,
  -- Benchmark prices for slippage calc
  arrival_price   REAL,   -- mid at order submission
  decision_price  REAL,   -- price at signal generation
  vwap_price      REAL,   -- VWAP over fill window
  twap_price      REAL,   -- TWAP over fill window
  close_price     REAL,   -- close of the bar
  -- Slippage metrics (bps)
  arrival_slippage_bps  REAL,
  decision_slippage_bps REAL,
  vwap_slippage_bps     REAL,
  -- Market impact estimate
  market_impact_bps     REAL,
  -- Time metrics
  order_submit_time TEXT,
  time_to_fill_ms   INTEGER,
  -- Partial fill tracking
  is_partial_fill   INTEGER NOT NULL DEFAULT 0 CHECK(is_partial_fill IN (0,1)),
  num_partial_fills INTEGER NOT NULL DEFAULT 1,
  -- Broker/venue
  broker          TEXT,
  venue           TEXT,
  commission      REAL,
  -- Fee breakdown
  exchange_fee    REAL,
  sec_fee         REAL,
  taf_fee         REAL,
  total_fees      REAL,
  -- Notes
  notes           TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_exec_quality_trade_id ON execution_quality(trade_id);
CREATE INDEX IF NOT EXISTS idx_exec_quality_symbol   ON execution_quality(symbol);
CREATE INDEX IF NOT EXISTS idx_exec_quality_fill_time ON execution_quality(fill_time);

-- DOWN

DROP INDEX IF EXISTS idx_exec_quality_fill_time;
DROP INDEX IF EXISTS idx_exec_quality_symbol;
DROP INDEX IF EXISTS idx_exec_quality_trade_id;
DROP TABLE IF EXISTS execution_quality;
