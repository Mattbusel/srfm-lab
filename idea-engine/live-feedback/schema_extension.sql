-- live-feedback/schema_extension.sql
-- Extends idea_engine.db with tables required by the Live Feedback Loop subsystem.
-- All statements use CREATE TABLE IF NOT EXISTS so this script is safe to re-run.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ─── Live hypothesis scores ────────────────────────────────────────────────
-- One row per scoring event.  Many rows may exist for the same hypothesis_id
-- (scored after each poll cycle that attributes new trades).
CREATE TABLE IF NOT EXISTS live_hypothesis_scores (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id     INTEGER NOT NULL,
    window_days       INTEGER NOT NULL DEFAULT 30,
    live_sharpe       REAL,
    live_win_rate     REAL,
    live_avg_pnl      REAL,
    live_trade_count  INTEGER,
    backtest_sharpe   REAL,
    degradation_ratio REAL,
    is_degraded       INTEGER NOT NULL DEFAULT 0,   -- BOOLEAN (0/1)
    scored_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_lhs_hypothesis_id
    ON live_hypothesis_scores (hypothesis_id);

CREATE INDEX IF NOT EXISTS idx_lhs_scored_at
    ON live_hypothesis_scores (scored_at DESC);

-- ─── Trade attributions ────────────────────────────────────────────────────
-- One row per trade (trade_id is PK).  Records which hypothesis / signal /
-- regime the trade is attributed to and how confident that attribution is.
CREATE TABLE IF NOT EXISTS trade_attributions (
    trade_id               TEXT    NOT NULL,
    hypothesis_id          INTEGER,
    signal_name            TEXT,
    regime                 TEXT,
    attribution_confidence REAL,
    created_at             TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    PRIMARY KEY (trade_id)
);

CREATE INDEX IF NOT EXISTS idx_ta_hypothesis_id
    ON trade_attributions (hypothesis_id);

-- ─── Performance snapshots ─────────────────────────────────────────────────
-- Hourly rolling snapshots of aggregate portfolio performance.
-- Used by the dashboard and the Bayesian scorer.
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL,
    equity         REAL    NOT NULL,
    running_sharpe REAL,
    running_dd     REAL,
    win_rate       REAL,
    calmar         REAL,
    expectancy     REAL,
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_ps_ts
    ON performance_snapshots (ts DESC);
