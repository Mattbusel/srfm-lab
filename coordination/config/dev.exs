import Config

# ---------------------------------------------------------------------------
# Development overrides
# ---------------------------------------------------------------------------

config :srfm_coordination,
  # Faster polling in dev so you see results quickly
  health_poll_interval_ms: 10_000,
  health_check_timeout_ms: 3_000,
  metrics_scrape_interval_ms: 5_000,
  metrics_flush_interval_ms: 60_000,

  # Lower circuit breaker thresholds for easier testing
  circuit_failure_threshold: 3,
  circuit_cooldown_ms: 15_000,

  # Shorter ACK window in dev
  param_ack_timeout_ms: 10_000

config :logger, :console,
  level: :debug,
  format: "[$level] $time [$metadata] $message\n",
  metadata: [:module, :pid, :function]
