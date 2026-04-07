-- Migration 022: Regime state history
-- Records every regime transition with full classification detail

-- UP

CREATE TABLE IF NOT EXISTS regime_log (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  transition_time       TEXT    NOT NULL,
  -- Regime classification
  regime                TEXT    NOT NULL,   -- e.g. TRENDING_BULL, MEAN_REVERTING, HIGH_VOL, CRISIS
  previous_regime       TEXT,
  regime_confidence     REAL,              -- 0..1 confidence score
  -- Sub-classifications
  trend_regime          TEXT,   -- BULL, BEAR, FLAT
  volatility_regime     TEXT,   -- LOW, NORMAL, HIGH, EXTREME
  liquidity_regime      TEXT,   -- AMPLE, NORMAL, THIN, CRISIS
  correlation_regime    TEXT,   -- LOW_CORR, NORMAL, HIGH_CORR, RISK_OFF
  -- Classifier outputs (JSON: classifier_name -> {regime, confidence})
  classifier_votes_json TEXT,
  -- Feature values that drove the classification
  vix_level             REAL,
  realized_vol_20d      REAL,
  adv_decline_ratio     REAL,
  credit_spread_bps     REAL,
  yield_curve_slope     REAL,
  market_breadth        REAL,
  bh_mass_spx           REAL,   -- BH mass for SPX
  -- Duration of previous regime
  prev_regime_duration_bars INTEGER,
  -- Transition trigger
  trigger_type          TEXT,   -- THRESHOLD, CLASSIFIER, MANUAL, SCHEDULED
  trigger_detail        TEXT,
  -- Parameter set applied for this regime
  param_snapshot_id     INTEGER,
  -- Impact on active positions
  positions_adjusted    INTEGER NOT NULL DEFAULT 0,
  hedges_added          INTEGER NOT NULL DEFAULT 0,
  -- Operator override
  manual_override       INTEGER NOT NULL DEFAULT 0 CHECK(manual_override IN (0,1)),
  overridden_by         TEXT,
  notes                 TEXT,
  created_at            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_regime_log_time    ON regime_log(transition_time);
CREATE INDEX IF NOT EXISTS idx_regime_log_regime  ON regime_log(regime);
CREATE INDEX IF NOT EXISTS idx_regime_log_created ON regime_log(created_at);

-- DOWN

DROP INDEX IF EXISTS idx_regime_log_created;
DROP INDEX IF EXISTS idx_regime_log_regime;
DROP INDEX IF EXISTS idx_regime_log_time;
DROP TABLE IF EXISTS regime_log;
