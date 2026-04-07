-- Migration 023: System alerts audit trail
-- All system alerts, warnings, and notifications are written here

-- UP

CREATE TABLE IF NOT EXISTS alerts_log (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  alert_time      TEXT    NOT NULL DEFAULT (datetime('now')),
  -- Classification
  alert_type      TEXT    NOT NULL,  -- RISK, EXECUTION, DATA, SYSTEM, REGIME, POSITION
  severity        TEXT    NOT NULL CHECK(severity IN ('DEBUG','INFO','WARNING','ERROR','CRITICAL')),
  -- Alert identity
  alert_code      TEXT    NOT NULL,  -- machine-readable code e.g. RISK_VAR_BREACH
  alert_name      TEXT    NOT NULL,  -- human label
  message         TEXT    NOT NULL,
  -- Context
  symbol          TEXT,
  strategy        TEXT,
  component       TEXT,  -- which system component raised this
  -- Threshold breach details
  threshold_name  TEXT,
  threshold_value REAL,
  actual_value    REAL,
  breach_pct      REAL,  -- how far over/under threshold
  -- Related entity IDs
  trade_id        INTEGER,
  position_id     INTEGER,
  signal_name     TEXT,
  -- Context JSON for extra data
  context_json    TEXT,
  -- Notification state
  notified_email  INTEGER NOT NULL DEFAULT 0 CHECK(notified_email IN (0,1)),
  notified_slack  INTEGER NOT NULL DEFAULT 0 CHECK(notified_slack IN (0,1)),
  notified_sms    INTEGER NOT NULL DEFAULT 0 CHECK(notified_sms IN (0,1)),
  notification_time TEXT,
  -- Resolution
  resolved        INTEGER NOT NULL DEFAULT 0 CHECK(resolved IN (0,1)),
  resolved_time   TEXT,
  resolved_by     TEXT,
  resolution_note TEXT,
  -- Suppression (for repeated alerts)
  is_suppressed   INTEGER NOT NULL DEFAULT 0 CHECK(is_suppressed IN (0,1)),
  suppression_expires TEXT,
  duplicate_of    INTEGER,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_alerts_log_time     ON alerts_log(alert_time);
CREATE INDEX IF NOT EXISTS idx_alerts_log_severity ON alerts_log(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_log_type     ON alerts_log(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_log_code     ON alerts_log(alert_code);
CREATE INDEX IF NOT EXISTS idx_alerts_log_resolved ON alerts_log(resolved);

-- DOWN

DROP INDEX IF EXISTS idx_alerts_log_resolved;
DROP INDEX IF EXISTS idx_alerts_log_code;
DROP INDEX IF EXISTS idx_alerts_log_type;
DROP INDEX IF EXISTS idx_alerts_log_severity;
DROP INDEX IF EXISTS idx_alerts_log_time;
DROP TABLE IF EXISTS alerts_log;
