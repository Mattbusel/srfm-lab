-- Migration 020: Risk metrics snapshots
-- VaR, CVaR, Greeks aggregates -- written every 15 minutes during market hours

-- UP

CREATE TABLE IF NOT EXISTS risk_metrics (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_time   TEXT    NOT NULL,
  -- Portfolio-level VaR/CVaR
  var_95_1d       REAL,   -- 1-day 95% VaR (dollar)
  var_99_1d       REAL,   -- 1-day 99% VaR
  var_95_5d       REAL,   -- 5-day 95% VaR
  cvar_95_1d      REAL,   -- Conditional VaR (Expected Shortfall)
  cvar_99_1d      REAL,
  -- Method used: hist, parametric, montecarlo
  var_method      TEXT    NOT NULL DEFAULT 'hist',
  -- Aggregate portfolio Greeks (options book)
  portfolio_delta   REAL,
  portfolio_gamma   REAL,
  portfolio_vega    REAL,
  portfolio_theta   REAL,
  portfolio_rho     REAL,
  -- Dollar Greeks
  dollar_delta    REAL,
  dollar_gamma    REAL,
  dollar_vega     REAL,
  -- Concentration metrics
  max_single_position_pct REAL,
  top3_concentration_pct  REAL,
  sector_hhi              REAL,   -- Herfindahl index by sector
  -- Leverage and exposure
  gross_exposure  REAL,
  net_exposure    REAL,
  leverage_ratio  REAL,
  -- Drawdown state
  current_drawdown_pct  REAL,
  peak_nav              REAL,
  current_nav           REAL,
  -- Correlation risk
  avg_pairwise_corr     REAL,
  max_pairwise_corr     REAL,
  -- Liquidity
  estimated_liquidation_days REAL,
  -- Stress test results (JSON: scenario -> pnl)
  stress_results_json   TEXT,
  -- Number of open positions at snapshot time
  open_positions_count  INTEGER,
  open_options_count    INTEGER,
  -- Regime at snapshot
  regime              TEXT,
  volatility_regime   TEXT,
  created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_risk_metrics_time    ON risk_metrics(snapshot_time);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_regime  ON risk_metrics(regime);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_created ON risk_metrics(created_at);

-- DOWN

DROP INDEX IF EXISTS idx_risk_metrics_created;
DROP INDEX IF EXISTS idx_risk_metrics_regime;
DROP INDEX IF EXISTS idx_risk_metrics_time;
DROP TABLE IF EXISTS risk_metrics;
