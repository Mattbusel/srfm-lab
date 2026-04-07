"""
db/schema.py -- SRFM SQLite schema management.

Handles migrations, table creation, validation, and schema status reporting.
All 26 table definitions are stored as Python constants for use by other modules.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table name constants -- use these everywhere to prevent typo bugs
# ---------------------------------------------------------------------------

class Tables:
    SCHEMA_VERSION         = "schema_version"
    TRADES                 = "trades"
    POSITIONS              = "positions"
    ORDERS                 = "orders"
    FILLS                  = "fills"
    DAILY_PERFORMANCE      = "daily_performance"
    MONTHLY_PERFORMANCE    = "monthly_performance"
    EQUITY_CURVE           = "equity_curve"
    SYMBOLS                = "symbols"
    BAR_DATA               = "bar_data"
    TICK_DATA              = "tick_data"
    CORPORATE_ACTIONS      = "corporate_actions"
    DIVIDENDS              = "dividends"
    BENCHMARKS             = "benchmarks"
    STRATEGY_CONFIG        = "strategy_config"
    BACKTEST_RUNS          = "backtest_runs"
    OPTIONS_POSITIONS      = "options_positions"
    SIGNAL_REGISTRY        = "signal_registry"
    SIGNAL_HISTORY         = "signal_history"
    PARAMETER_SNAPSHOTS    = "parameter_snapshots"
    RISK_METRICS           = "risk_metrics"
    EXECUTION_QUALITY      = "execution_quality"
    REGIME_LOG             = "regime_log"
    ALERTS_LOG             = "alerts_log"
    GENOME_HISTORY         = "genome_history"
    FEATURE_IMPORTANCE     = "feature_importance"
    NAV_LOG                = "nav_log"


# ---------------------------------------------------------------------------
# SQL for all 26 tables (used both in create_tables() and validate_schema())
# ---------------------------------------------------------------------------

_DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER NOT NULL,
  applied_at TEXT    NOT NULL DEFAULT (datetime('now')),
  migration  TEXT
)
"""

_DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol           TEXT    NOT NULL,
  side             TEXT    NOT NULL CHECK(side IN ('BUY','SELL','SHORT','COVER')),
  qty              INTEGER NOT NULL,
  entry_price      REAL    NOT NULL,
  entry_time       TEXT    NOT NULL,
  exit_price       REAL,
  exit_time        TEXT,
  pnl              REAL,
  pnl_pct          REAL,
  commission       REAL    NOT NULL DEFAULT 0.0,
  slippage         REAL    NOT NULL DEFAULT 0.0,
  strategy_version TEXT,
  signal_name      TEXT,
  regime           TEXT,
  notes            TEXT,
  created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol         TEXT    NOT NULL,
  qty            INTEGER NOT NULL,
  avg_entry      REAL    NOT NULL,
  current_price  REAL,
  unrealized_pnl REAL,
  side           TEXT    NOT NULL CHECK(side IN ('LONG','SHORT')),
  opened_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  strategy       TEXT,
  updated_at     TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  broker_order_id TEXT,
  symbol         TEXT    NOT NULL,
  order_type     TEXT    NOT NULL,
  side           TEXT    NOT NULL,
  qty            INTEGER NOT NULL,
  limit_price    REAL,
  stop_price     REAL,
  status         TEXT    NOT NULL DEFAULT 'PENDING',
  submitted_at   TEXT    NOT NULL DEFAULT (datetime('now')),
  filled_at      TEXT,
  fill_price     REAL,
  fill_qty       INTEGER,
  strategy       TEXT,
  notes          TEXT,
  created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_FILLS = """
CREATE TABLE IF NOT EXISTS fills (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id    INTEGER NOT NULL,
  symbol      TEXT    NOT NULL,
  qty         INTEGER NOT NULL,
  price       REAL    NOT NULL,
  fill_time   TEXT    NOT NULL,
  commission  REAL    NOT NULL DEFAULT 0.0,
  venue       TEXT,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(order_id) REFERENCES orders(id)
)
"""

_DDL_DAILY_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS daily_performance (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  date            TEXT    NOT NULL UNIQUE,
  starting_nav    REAL    NOT NULL,
  ending_nav      REAL    NOT NULL,
  daily_return    REAL    NOT NULL,
  daily_pnl       REAL    NOT NULL,
  realized_pnl    REAL    NOT NULL DEFAULT 0.0,
  unrealized_pnl  REAL    NOT NULL DEFAULT 0.0,
  commissions     REAL    NOT NULL DEFAULT 0.0,
  num_trades      INTEGER NOT NULL DEFAULT 0,
  num_winners     INTEGER NOT NULL DEFAULT 0,
  num_losers      INTEGER NOT NULL DEFAULT 0,
  gross_profit    REAL    NOT NULL DEFAULT 0.0,
  gross_loss      REAL    NOT NULL DEFAULT 0.0,
  max_drawdown    REAL    NOT NULL DEFAULT 0.0,
  regime          TEXT,
  benchmark_return REAL,
  alpha           REAL,
  created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_MONTHLY_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS monthly_performance (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  month           TEXT    NOT NULL UNIQUE,   -- YYYY-MM
  starting_nav    REAL    NOT NULL,
  ending_nav      REAL    NOT NULL,
  monthly_return  REAL    NOT NULL,
  monthly_pnl     REAL    NOT NULL,
  sharpe_ratio    REAL,
  max_drawdown    REAL,
  num_trades      INTEGER NOT NULL DEFAULT 0,
  win_rate        REAL,
  profit_factor   REAL,
  benchmark_return REAL,
  alpha           REAL,
  created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_EQUITY_CURVE = """
CREATE TABLE IF NOT EXISTS equity_curve (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  bar_time    TEXT    NOT NULL UNIQUE,
  nav         REAL    NOT NULL,
  cash        REAL    NOT NULL DEFAULT 0.0,
  market_val  REAL    NOT NULL DEFAULT 0.0,
  high_water  REAL    NOT NULL DEFAULT 0.0,
  drawdown    REAL    NOT NULL DEFAULT 0.0,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_SYMBOLS = """
CREATE TABLE IF NOT EXISTS symbols (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol      TEXT    NOT NULL UNIQUE,
  name        TEXT,
  asset_class TEXT    NOT NULL DEFAULT 'EQUITY',
  exchange    TEXT,
  sector      TEXT,
  industry    TEXT,
  currency    TEXT    NOT NULL DEFAULT 'USD',
  is_active   INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
  first_seen  TEXT,
  last_seen   TEXT,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_BAR_DATA = """
CREATE TABLE IF NOT EXISTS bar_data (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol      TEXT    NOT NULL,
  timeframe   TEXT    NOT NULL,
  bar_time    TEXT    NOT NULL,
  open        REAL    NOT NULL,
  high        REAL    NOT NULL,
  low         REAL    NOT NULL,
  close       REAL    NOT NULL,
  volume      REAL    NOT NULL DEFAULT 0.0,
  vwap        REAL,
  num_trades  INTEGER,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE(symbol, timeframe, bar_time)
)
"""

_DDL_TICK_DATA = """
CREATE TABLE IF NOT EXISTS tick_data (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol      TEXT    NOT NULL,
  tick_time   TEXT    NOT NULL,
  price       REAL    NOT NULL,
  size        INTEGER NOT NULL DEFAULT 0,
  bid         REAL,
  ask         REAL,
  bid_size    INTEGER,
  ask_size    INTEGER,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_CORPORATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS corporate_actions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol      TEXT    NOT NULL,
  action_type TEXT    NOT NULL,  -- SPLIT, MERGER, SPINOFF, DELISTED
  action_date TEXT    NOT NULL,
  ratio       REAL,
  notes       TEXT,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_DIVIDENDS = """
CREATE TABLE IF NOT EXISTS dividends (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol        TEXT    NOT NULL,
  ex_date       TEXT    NOT NULL,
  record_date   TEXT,
  pay_date      TEXT,
  amount        REAL    NOT NULL,
  div_type      TEXT    NOT NULL DEFAULT 'REGULAR',
  created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE(symbol, ex_date)
)
"""

_DDL_BENCHMARKS = """
CREATE TABLE IF NOT EXISTS benchmarks (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol      TEXT    NOT NULL,
  date        TEXT    NOT NULL,
  close       REAL    NOT NULL,
  daily_return REAL,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE(symbol, date)
)
"""

_DDL_STRATEGY_CONFIG = """
CREATE TABLE IF NOT EXISTS strategy_config (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  key         TEXT    NOT NULL UNIQUE,
  value       TEXT    NOT NULL,
  value_type  TEXT    NOT NULL DEFAULT 'str',
  description TEXT,
  updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_by  TEXT,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_BACKTEST_RUNS = """
CREATE TABLE IF NOT EXISTS backtest_runs (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id           TEXT    NOT NULL UNIQUE,
  started_at       TEXT    NOT NULL,
  finished_at      TEXT,
  status           TEXT    NOT NULL DEFAULT 'RUNNING',
  period_start     TEXT    NOT NULL,
  period_end       TEXT    NOT NULL,
  params_json      TEXT    NOT NULL,
  results_json     TEXT,
  sharpe           REAL,
  cagr             REAL,
  max_drawdown     REAL,
  num_trades       INTEGER,
  win_rate         REAL,
  profit_factor    REAL,
  notes            TEXT,
  created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
)
"""

# Tables from migrations 017-026 (abbreviated here; full DDL lives in migration files)

_DDL_OPTIONS_POSITIONS = """
CREATE TABLE IF NOT EXISTS options_positions (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  symbol         TEXT    NOT NULL,
  expiry         TEXT    NOT NULL,
  strike         REAL    NOT NULL,
  right          TEXT    NOT NULL   CHECK(right IN ('CALL','PUT')),
  qty            INTEGER NOT NULL,
  entry_price    REAL    NOT NULL,
  entry_time     TEXT    NOT NULL,
  exit_price     REAL,
  exit_time      TEXT,
  current_price  REAL,
  delta          REAL,
  gamma          REAL,
  vega           REAL,
  theta          REAL,
  rho            REAL,
  iv_entry       REAL,
  iv_current     REAL,
  underlying_entry REAL,
  realized_pnl   REAL,
  unrealized_pnl REAL,
  strategy_version TEXT,
  regime_at_entry  TEXT,
  signal_source    TEXT,
  status         TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN','CLOSED','EXPIRED','ASSIGNED')),
  notes          TEXT,
  created_at     TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_SIGNAL_REGISTRY = """
CREATE TABLE IF NOT EXISTS signal_registry (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_name     TEXT    NOT NULL,
  signal_class    TEXT    NOT NULL,
  version         TEXT    NOT NULL,
  description     TEXT,
  deployed_at     TEXT    NOT NULL DEFAULT (datetime('now')),
  deployed_by     TEXT,
  is_active       INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
  deactivated_at  TEXT,
  deactivation_reason TEXT,
  config_json     TEXT    NOT NULL DEFAULT '{}',
  symbols         TEXT    NOT NULL DEFAULT '[]',
  timeframes      TEXT    NOT NULL DEFAULT '[]',
  total_signals_fired   INTEGER NOT NULL DEFAULT 0,
  true_positive_count   INTEGER NOT NULL DEFAULT 0,
  false_positive_count  INTEGER NOT NULL DEFAULT 0,
  win_rate_pct          REAL,
  avg_return_pct        REAL,
  sharpe_contribution   REAL,
  last_fired_at         TEXT,
  iae_genome_id         TEXT,
  parent_signal_name    TEXT,
  output_schema_json    TEXT,
  required_features     TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(signal_name, version)
)
"""

_DDL_SIGNAL_HISTORY = """
CREATE TABLE IF NOT EXISTS signal_history (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_name     TEXT    NOT NULL,
  symbol          TEXT    NOT NULL,
  timeframe       TEXT    NOT NULL,
  bar_time        TEXT    NOT NULL,
  signal_value    REAL    NOT NULL,
  direction       TEXT    CHECK(direction IN ('LONG','SHORT','FLAT','NEUTRAL')),
  confidence      REAL,
  components_json TEXT,
  price           REAL,
  volume          REAL,
  volatility      REAL,
  regime          TEXT,
  features_json   TEXT,
  trade_id        INTEGER,
  acted_on        INTEGER NOT NULL DEFAULT 0 CHECK(acted_on IN (0,1)),
  fwd_return_1bar  REAL,
  fwd_return_5bar  REAL,
  fwd_return_20bar REAL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(signal_name) REFERENCES signal_registry(signal_name)
)
"""

_DDL_PARAMETER_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS parameter_snapshots (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_time   TEXT    NOT NULL DEFAULT (datetime('now')),
  source          TEXT    NOT NULL CHECK(source IN (
                    'IAE_CYCLE','MANUAL','REGIME_CHANGE','STARTUP','ROLLBACK','SCHEDULE'
                  )),
  params_json     TEXT    NOT NULL,
  delta_json      TEXT,
  change_summary  TEXT,
  trigger_sharpe    REAL,
  trigger_drawdown  REAL,
  trigger_regime    TEXT,
  genome_id         TEXT,
  genome_generation INTEGER,
  genome_fitness    REAL,
  validation_passed INTEGER NOT NULL DEFAULT 1 CHECK(validation_passed IN (0,1)),
  validation_errors TEXT,
  applied_by      TEXT,
  notes           TEXT,
  rolled_back_from INTEGER,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_RISK_METRICS = """
CREATE TABLE IF NOT EXISTS risk_metrics (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_time   TEXT    NOT NULL,
  var_95_1d       REAL,
  var_99_1d       REAL,
  var_95_5d       REAL,
  cvar_95_1d      REAL,
  cvar_99_1d      REAL,
  var_method      TEXT    NOT NULL DEFAULT 'hist',
  portfolio_delta   REAL,
  portfolio_gamma   REAL,
  portfolio_vega    REAL,
  portfolio_theta   REAL,
  portfolio_rho     REAL,
  dollar_delta    REAL,
  dollar_gamma    REAL,
  dollar_vega     REAL,
  max_single_position_pct REAL,
  top3_concentration_pct  REAL,
  sector_hhi              REAL,
  gross_exposure  REAL,
  net_exposure    REAL,
  leverage_ratio  REAL,
  current_drawdown_pct  REAL,
  peak_nav              REAL,
  current_nav           REAL,
  avg_pairwise_corr     REAL,
  max_pairwise_corr     REAL,
  estimated_liquidation_days REAL,
  stress_results_json   TEXT,
  open_positions_count  INTEGER,
  open_options_count    INTEGER,
  regime              TEXT,
  volatility_regime   TEXT,
  created_at          TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_EXECUTION_QUALITY = """
CREATE TABLE IF NOT EXISTS execution_quality (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_id        INTEGER NOT NULL,
  symbol          TEXT    NOT NULL,
  order_id        TEXT,
  order_side      TEXT    NOT NULL,
  order_type      TEXT    NOT NULL,
  order_qty       INTEGER NOT NULL,
  limit_price     REAL,
  fill_price      REAL    NOT NULL,
  fill_qty        INTEGER NOT NULL,
  fill_time       TEXT    NOT NULL,
  arrival_price   REAL,
  decision_price  REAL,
  vwap_price      REAL,
  twap_price      REAL,
  close_price     REAL,
  arrival_slippage_bps  REAL,
  decision_slippage_bps REAL,
  vwap_slippage_bps     REAL,
  market_impact_bps     REAL,
  order_submit_time TEXT,
  time_to_fill_ms   INTEGER,
  is_partial_fill   INTEGER NOT NULL DEFAULT 0 CHECK(is_partial_fill IN (0,1)),
  num_partial_fills INTEGER NOT NULL DEFAULT 1,
  broker          TEXT,
  venue           TEXT,
  commission      REAL,
  exchange_fee    REAL,
  sec_fee         REAL,
  taf_fee         REAL,
  total_fees      REAL,
  notes           TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_REGIME_LOG = """
CREATE TABLE IF NOT EXISTS regime_log (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  transition_time       TEXT    NOT NULL,
  regime                TEXT    NOT NULL,
  previous_regime       TEXT,
  regime_confidence     REAL,
  trend_regime          TEXT,
  volatility_regime     TEXT,
  liquidity_regime      TEXT,
  correlation_regime    TEXT,
  classifier_votes_json TEXT,
  vix_level             REAL,
  realized_vol_20d      REAL,
  adv_decline_ratio     REAL,
  credit_spread_bps     REAL,
  yield_curve_slope     REAL,
  market_breadth        REAL,
  bh_mass_spx           REAL,
  prev_regime_duration_bars INTEGER,
  trigger_type          TEXT,
  trigger_detail        TEXT,
  param_snapshot_id     INTEGER,
  positions_adjusted    INTEGER NOT NULL DEFAULT 0,
  hedges_added          INTEGER NOT NULL DEFAULT 0,
  manual_override       INTEGER NOT NULL DEFAULT 0 CHECK(manual_override IN (0,1)),
  overridden_by         TEXT,
  notes                 TEXT,
  created_at            TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_ALERTS_LOG = """
CREATE TABLE IF NOT EXISTS alerts_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  alert_time      TEXT    NOT NULL DEFAULT (datetime('now')),
  alert_type      TEXT    NOT NULL,
  severity        TEXT    NOT NULL CHECK(severity IN ('DEBUG','INFO','WARNING','ERROR','CRITICAL')),
  alert_code      TEXT    NOT NULL,
  alert_name      TEXT    NOT NULL,
  message         TEXT    NOT NULL,
  symbol          TEXT,
  strategy        TEXT,
  component       TEXT,
  threshold_name  TEXT,
  threshold_value REAL,
  actual_value    REAL,
  breach_pct      REAL,
  trade_id        INTEGER,
  position_id     INTEGER,
  signal_name     TEXT,
  context_json    TEXT,
  notified_email  INTEGER NOT NULL DEFAULT 0 CHECK(notified_email IN (0,1)),
  notified_slack  INTEGER NOT NULL DEFAULT 0 CHECK(notified_slack IN (0,1)),
  notified_sms    INTEGER NOT NULL DEFAULT 0 CHECK(notified_sms IN (0,1)),
  notification_time TEXT,
  resolved        INTEGER NOT NULL DEFAULT 0 CHECK(resolved IN (0,1)),
  resolved_time   TEXT,
  resolved_by     TEXT,
  resolution_note TEXT,
  is_suppressed   INTEGER NOT NULL DEFAULT 0 CHECK(is_suppressed IN (0,1)),
  suppression_expires TEXT,
  duplicate_of    INTEGER,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_GENOME_HISTORY = """
CREATE TABLE IF NOT EXISTS genome_history (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  genome_id         TEXT    NOT NULL UNIQUE,
  generation        INTEGER NOT NULL,
  parent_genome_id  TEXT,
  parent2_genome_id TEXT,
  creation_method   TEXT    NOT NULL CHECK(creation_method IN (
                      'RANDOM','MUTATION','CROSSOVER','ELITE','MANUAL','INJECTION'
                    )),
  mutation_type     TEXT,
  genome_json       TEXT    NOT NULL,
  params_json       TEXT    NOT NULL,
  fitness           REAL,
  fitness_components_json TEXT,
  is_sharpe         REAL,
  is_cagr           REAL,
  is_max_drawdown   REAL,
  is_win_rate       REAL,
  is_num_trades     INTEGER,
  is_period_start   TEXT,
  is_period_end     TEXT,
  oos_sharpe        REAL,
  oos_cagr          REAL,
  oos_max_drawdown  REAL,
  oos_win_rate      REAL,
  oos_num_trades    INTEGER,
  oos_period_start  TEXT,
  oos_period_end    TEXT,
  overfit_penalty   REAL,
  adjusted_fitness  REAL,
  is_elite          INTEGER NOT NULL DEFAULT 0 CHECK(is_elite IN (0,1)),
  was_deployed      INTEGER NOT NULL DEFAULT 0 CHECK(was_deployed IN (0,1)),
  deployed_at       TEXT,
  eval_duration_sec REAL,
  eval_error        TEXT,
  eval_node         TEXT,
  cycle_id          TEXT,
  population_size   INTEGER,
  constraints_passed INTEGER NOT NULL DEFAULT 1 CHECK(constraints_passed IN (0,1)),
  constraint_violations TEXT,
  created_at        TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_FEATURE_IMPORTANCE = """
CREATE TABLE IF NOT EXISTS feature_importance (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_time   TEXT    NOT NULL DEFAULT (datetime('now')),
  model_name      TEXT    NOT NULL,
  model_version   TEXT    NOT NULL,
  model_type      TEXT    NOT NULL,
  symbol          TEXT,
  timeframe       TEXT,
  target_variable TEXT    NOT NULL,
  training_start  TEXT,
  training_end    TEXT,
  num_training_samples INTEGER,
  importances_json TEXT   NOT NULL,
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
  model_accuracy  REAL,
  model_auc       REAL,
  model_r2        REAL,
  model_mse       REAL,
  stability_score REAL,
  prev_snapshot_id INTEGER,
  importance_method TEXT  NOT NULL DEFAULT 'permutation',
  num_features    INTEGER,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_DDL_NAV_LOG = """
CREATE TABLE IF NOT EXISTS nav_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  bar_time        TEXT    NOT NULL,
  symbol          TEXT    NOT NULL,
  timeframe       TEXT    NOT NULL DEFAULT '1m',
  nav             REAL    NOT NULL,
  cash            REAL    NOT NULL,
  gross_market_value REAL NOT NULL,
  net_market_value   REAL NOT NULL,
  realized_pnl_today    REAL NOT NULL DEFAULT 0.0,
  unrealized_pnl        REAL NOT NULL DEFAULT 0.0,
  total_pnl_today       REAL NOT NULL DEFAULT 0.0,
  num_long_positions    INTEGER NOT NULL DEFAULT 0,
  num_short_positions   INTEGER NOT NULL DEFAULT 0,
  num_options_positions INTEGER NOT NULL DEFAULT 0,
  close_price     REAL,
  volume          REAL,
  bid             REAL,
  ask             REAL,
  active_signals_json TEXT,
  nav_state_json  TEXT,
  bh_mass         REAL,
  bh_curvature    REAL,
  regime          TEXT,
  pnl_pct_today   REAL,
  max_drawdown_today REAL,
  fees_today      REAL NOT NULL DEFAULT 0.0,
  bar_latency_ms  INTEGER,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

# Ordered list of (table_name, ddl) tuples -- used by create_tables()
ALL_TABLES: list[tuple[str, str]] = [
    (Tables.SCHEMA_VERSION,      _DDL_SCHEMA_VERSION),
    (Tables.TRADES,              _DDL_TRADES),
    (Tables.POSITIONS,           _DDL_POSITIONS),
    (Tables.ORDERS,              _DDL_ORDERS),
    (Tables.FILLS,               _DDL_FILLS),
    (Tables.DAILY_PERFORMANCE,   _DDL_DAILY_PERFORMANCE),
    (Tables.MONTHLY_PERFORMANCE, _DDL_MONTHLY_PERFORMANCE),
    (Tables.EQUITY_CURVE,        _DDL_EQUITY_CURVE),
    (Tables.SYMBOLS,             _DDL_SYMBOLS),
    (Tables.BAR_DATA,            _DDL_BAR_DATA),
    (Tables.TICK_DATA,           _DDL_TICK_DATA),
    (Tables.CORPORATE_ACTIONS,   _DDL_CORPORATE_ACTIONS),
    (Tables.DIVIDENDS,           _DDL_DIVIDENDS),
    (Tables.BENCHMARKS,          _DDL_BENCHMARKS),
    (Tables.STRATEGY_CONFIG,     _DDL_STRATEGY_CONFIG),
    (Tables.BACKTEST_RUNS,       _DDL_BACKTEST_RUNS),
    (Tables.OPTIONS_POSITIONS,   _DDL_OPTIONS_POSITIONS),
    (Tables.SIGNAL_REGISTRY,     _DDL_SIGNAL_REGISTRY),
    (Tables.SIGNAL_HISTORY,      _DDL_SIGNAL_HISTORY),
    (Tables.PARAMETER_SNAPSHOTS, _DDL_PARAMETER_SNAPSHOTS),
    (Tables.RISK_METRICS,        _DDL_RISK_METRICS),
    (Tables.EXECUTION_QUALITY,   _DDL_EXECUTION_QUALITY),
    (Tables.REGIME_LOG,          _DDL_REGIME_LOG),
    (Tables.ALERTS_LOG,          _DDL_ALERTS_LOG),
    (Tables.GENOME_HISTORY,      _DDL_GENOME_HISTORY),
    (Tables.FEATURE_IMPORTANCE,  _DDL_FEATURE_IMPORTANCE),
    (Tables.NAV_LOG,             _DDL_NAV_LOG),
]

# Expected indexes -- (index_name, table_name)
EXPECTED_INDEXES: list[tuple[str, str]] = [
    ("idx_trades_symbol",          Tables.TRADES),
    ("idx_trades_entry_time",      Tables.TRADES),
    ("idx_trades_strategy",        Tables.TRADES),
    ("idx_positions_symbol",       Tables.POSITIONS),
    ("idx_orders_symbol",          Tables.ORDERS),
    ("idx_fills_order_id",         Tables.FILLS),
    ("idx_bar_data_sym_tf_time",   Tables.BAR_DATA),
    ("idx_tick_data_sym_time",     Tables.TICK_DATA),
    ("idx_equity_curve_time",      Tables.EQUITY_CURVE),
    ("idx_options_positions_symbol",  Tables.OPTIONS_POSITIONS),
    ("idx_options_positions_expiry",  Tables.OPTIONS_POSITIONS),
    ("idx_options_positions_status",  Tables.OPTIONS_POSITIONS),
    ("idx_signal_registry_name",   Tables.SIGNAL_REGISTRY),
    ("idx_signal_history_name_sym",Tables.SIGNAL_HISTORY),
    ("idx_signal_history_bar_time",Tables.SIGNAL_HISTORY),
    ("idx_param_snapshots_time",   Tables.PARAMETER_SNAPSHOTS),
    ("idx_risk_metrics_time",      Tables.RISK_METRICS),
    ("idx_exec_quality_trade_id",  Tables.EXECUTION_QUALITY),
    ("idx_regime_log_time",        Tables.REGIME_LOG),
    ("idx_alerts_log_time",        Tables.ALERTS_LOG),
    ("idx_alerts_log_severity",    Tables.ALERTS_LOG),
    ("idx_genome_history_id",      Tables.GENOME_HISTORY),
    ("idx_genome_history_fitness", Tables.GENOME_HISTORY),
    ("idx_feat_imp_time",          Tables.FEATURE_IMPORTANCE),
    ("idx_nav_log_bar_time",       Tables.NAV_LOG),
    ("idx_nav_log_sym_tf_bar",     Tables.NAV_LOG),
]


# ---------------------------------------------------------------------------
# DBSchema
# ---------------------------------------------------------------------------

class DBSchema:
    """
    Schema management: migrations, table creation, and validation.

    Usage
    -----
    DBSchema.migrate("/path/to/srfm.db")
    DBSchema.migrate("/path/to/srfm.db", target_version=20)
    """

    MIGRATIONS_DIR = Path(__file__).parent / "migrations"

    # The baseline schema below tables 001-016 are assumed already applied
    # when version == 16.  Migrations 017-026 are SQL files.
    BASELINE_VERSION = 16

    @classmethod
    def migrate(
        cls,
        db_path: str | Path,
        target_version: int | None = None,
    ) -> None:
        """
        Apply all pending migration files up to target_version (inclusive).
        If target_version is None apply all available migrations.
        Each migration file is wrapped in a transaction; failure rolls back
        that migration and stops the chain.
        """
        db_path = Path(db_path)
        conn = cls._open_conn(db_path)
        try:
            cls._ensure_schema_version_table(conn)
            current = cls.get_current_version(conn)
            migrations = cls._discover_migrations()

            for version, path in migrations:
                if version <= current:
                    continue
                if target_version is not None and version > target_version:
                    break

                log.info("Applying migration %03d from %s", version, path.name)
                sql = cls._extract_up(path.read_text(encoding="utf-8"))
                try:
                    with conn:
                        conn.executescript(sql)
                        conn.execute(
                            "INSERT INTO schema_version(version, migration) VALUES (?,?)",
                            (version, path.name),
                        )
                    log.info("Migration %03d applied OK", version)
                except Exception as exc:
                    log.error("Migration %03d failed: %s", version, exc)
                    raise
        finally:
            conn.close()

    @classmethod
    def get_current_version(cls, conn: sqlite3.Connection) -> int:
        """Return the highest migration version recorded in schema_version."""
        cls._ensure_schema_version_table(conn)
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        if row and row[0] is not None:
            return int(row[0])
        return 0

    @classmethod
    def create_tables(cls, conn: sqlite3.Connection) -> None:
        """
        Idempotent CREATE TABLE IF NOT EXISTS for all 27 tables.
        Also creates all standard indexes.
        Safe to call on an already-initialized database.
        """
        for _name, ddl in ALL_TABLES:
            conn.executescript(ddl)

        _INDEXES = """
        CREATE INDEX IF NOT EXISTS idx_trades_symbol       ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_entry_time   ON trades(entry_time);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy     ON trades(strategy_version);
        CREATE INDEX IF NOT EXISTS idx_positions_symbol    ON positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_orders_symbol       ON orders(symbol);
        CREATE INDEX IF NOT EXISTS idx_fills_order_id      ON fills(order_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_bar_data_sym_tf_time
            ON bar_data(symbol, timeframe, bar_time);
        CREATE INDEX IF NOT EXISTS idx_tick_data_sym_time  ON tick_data(symbol, tick_time);
        CREATE INDEX IF NOT EXISTS idx_equity_curve_time   ON equity_curve(bar_time);
        CREATE INDEX IF NOT EXISTS idx_options_positions_symbol
            ON options_positions(symbol);
        CREATE INDEX IF NOT EXISTS idx_options_positions_expiry
            ON options_positions(expiry);
        CREATE INDEX IF NOT EXISTS idx_options_positions_status
            ON options_positions(status);
        CREATE INDEX IF NOT EXISTS idx_signal_registry_name
            ON signal_registry(signal_name);
        CREATE INDEX IF NOT EXISTS idx_signal_history_name_sym
            ON signal_history(signal_name, symbol);
        CREATE INDEX IF NOT EXISTS idx_signal_history_bar_time
            ON signal_history(bar_time);
        CREATE INDEX IF NOT EXISTS idx_param_snapshots_time
            ON parameter_snapshots(snapshot_time);
        CREATE INDEX IF NOT EXISTS idx_risk_metrics_time   ON risk_metrics(snapshot_time);
        CREATE INDEX IF NOT EXISTS idx_exec_quality_trade_id
            ON execution_quality(trade_id);
        CREATE INDEX IF NOT EXISTS idx_regime_log_time     ON regime_log(transition_time);
        CREATE INDEX IF NOT EXISTS idx_alerts_log_time     ON alerts_log(alert_time);
        CREATE INDEX IF NOT EXISTS idx_alerts_log_severity ON alerts_log(severity);
        CREATE INDEX IF NOT EXISTS idx_genome_history_id   ON genome_history(genome_id);
        CREATE INDEX IF NOT EXISTS idx_genome_history_fitness
            ON genome_history(fitness);
        CREATE INDEX IF NOT EXISTS idx_feat_imp_time       ON feature_importance(snapshot_time);
        CREATE INDEX IF NOT EXISTS idx_nav_log_bar_time    ON nav_log(bar_time);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_nav_log_sym_tf_bar
            ON nav_log(symbol, timeframe, bar_time);
        """
        conn.executescript(_INDEXES)
        conn.commit()

    @classmethod
    def validate_schema(cls, conn: sqlite3.Connection) -> list[str]:
        """
        Check that all expected tables and indexes exist.
        Returns a list of problem strings; empty list means schema is valid.
        """
        problems: list[str] = []

        # Check tables
        existing_tables: set[str] = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for name, _ in ALL_TABLES:
            if name not in existing_tables:
                problems.append(f"Missing table: {name}")

        # Check indexes
        existing_indexes: set[str] = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        for idx_name, _ in EXPECTED_INDEXES:
            if idx_name not in existing_indexes:
                problems.append(f"Missing index: {idx_name}")

        return problems

    @classmethod
    def get_migration_status(cls, conn: sqlite3.Connection) -> dict[str, Any]:
        """
        Return a dict describing current migration state:
          - current_version: int
          - applied: list of (version, migration, applied_at)
          - pending: list of (version, filename) not yet applied
        """
        cls._ensure_schema_version_table(conn)
        current = cls.get_current_version(conn)

        applied = [
            {"version": row[0], "migration": row[1], "applied_at": row[2]}
            for row in conn.execute(
                "SELECT version, migration, applied_at FROM schema_version ORDER BY version"
            )
        ]

        migrations = cls._discover_migrations()
        pending = [
            {"version": v, "file": p.name}
            for v, p in migrations
            if v > current
        ]

        return {
            "current_version": current,
            "applied": applied,
            "pending": pending,
            "is_up_to_date": len(pending) == 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_conn(db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-32000")  # 32 MB cache
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
              version    INTEGER NOT NULL,
              applied_at TEXT    NOT NULL DEFAULT (datetime('now')),
              migration  TEXT
            )
            """
        )
        conn.commit()

    @classmethod
    def _discover_migrations(cls) -> list[tuple[int, Path]]:
        """
        Return sorted list of (version_number, path) for all .sql files
        in the migrations directory whose names start with a 3-digit number.
        """
        results: list[tuple[int, Path]] = []
        if not cls.MIGRATIONS_DIR.exists():
            return results
        pattern = re.compile(r"^(\d{3})_.*\.sql$")
        for p in sorted(cls.MIGRATIONS_DIR.glob("*.sql")):
            m = pattern.match(p.name)
            if m:
                results.append((int(m.group(1)), p))
        return sorted(results, key=lambda x: x[0])

    @staticmethod
    def _extract_up(sql_text: str) -> str:
        """
        Extract the UP section from a migration file.
        Everything after '-- UP' and before '-- DOWN' (case-insensitive).
        If no markers are found return the full text.
        """
        up_match = re.search(r"--\s*UP\b", sql_text, re.IGNORECASE)
        down_match = re.search(r"--\s*DOWN\b", sql_text, re.IGNORECASE)
        if up_match and down_match:
            return sql_text[up_match.end():down_match.start()]
        if up_match:
            return sql_text[up_match.end():]
        return sql_text
