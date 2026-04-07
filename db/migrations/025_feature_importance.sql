-- Migration 025: ML feature importance snapshots
-- Periodic snapshots of feature importance from all ML models

-- UP

CREATE TABLE IF NOT EXISTS feature_importance (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_time   TEXT    NOT NULL DEFAULT (datetime('now')),
  -- Model identification
  model_name      TEXT    NOT NULL,
  model_version   TEXT    NOT NULL,
  model_type      TEXT    NOT NULL,  -- RF, GBM, LSTM, LINEAR, XGB, etc.
  -- Scope
  symbol          TEXT,   -- NULL means portfolio-level model
  timeframe       TEXT,
  target_variable TEXT    NOT NULL,  -- e.g. fwd_return_5bar, direction
  -- Training context
  training_start  TEXT,
  training_end    TEXT,
  num_training_samples INTEGER,
  -- Feature importance data (JSON array of {feature, importance, rank})
  importances_json TEXT   NOT NULL,
  -- Top features (denormalized for fast queries)
  feature_rank1   TEXT,
  feature_rank2   TEXT,
  feature_rank3   TEXT,
  feature_rank4   TEXT,
  feature_rank5   TEXT,
  importance_rank1 REAL,
  importance_rank2 REAL,
  importance_rank3 REAL,
  importance_rank4 REAL,
  importance_rank5 REAL,
  -- Model performance metrics
  model_accuracy  REAL,
  model_auc       REAL,
  model_r2        REAL,
  model_mse       REAL,
  -- Feature stability vs previous snapshot
  stability_score REAL,   -- 0..1 cosine similarity with prev snapshot
  prev_snapshot_id INTEGER,
  -- Computation metadata
  importance_method TEXT  NOT NULL DEFAULT 'permutation',  -- permutation, shap, gain, split
  num_features    INTEGER,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_feat_imp_time    ON feature_importance(snapshot_time);
CREATE INDEX IF NOT EXISTS idx_feat_imp_model   ON feature_importance(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_feat_imp_symbol  ON feature_importance(symbol);

-- DOWN

DROP INDEX IF EXISTS idx_feat_imp_symbol;
DROP INDEX IF EXISTS idx_feat_imp_model;
DROP INDEX IF EXISTS idx_feat_imp_time;
DROP TABLE IF EXISTS feature_importance;
