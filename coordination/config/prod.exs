import Config

# ---------------------------------------------------------------------------
# Production settings
# ---------------------------------------------------------------------------

config :srfm_coordination,
  # Standard 30s polling, higher timeouts to tolerate slow services
  health_poll_interval_ms: 30_000,
  health_check_timeout_ms: 8_000,
  health_degraded_threshold: 3,
  health_down_threshold: 5,

  metrics_scrape_interval_ms: 15_000,
  metrics_flush_interval_ms: 300_000,

  # More conservative circuit breaker in production
  circuit_failure_threshold: 5,
  circuit_window_seconds: 60,
  circuit_cooldown_ms: 60_000,

  # Full 30s ACK window, strict 20% rollback threshold
  param_ack_timeout_ms: 30_000,
  param_rollback_threshold: 0.20,

  # Persist alerts to the data directory
  alert_log_path: "/var/log/srfm/alerts.log"

config :logger,
  level: :info

config :logger, :console,
  format: "$dateT$time [$level] $metadata$message\n",
  metadata: [:module, :request_id]
