import Config

config :srfm_coordination,
  health_poll_interval_ms: 60_000,
  health_check_timeout_ms: 1_000,
  metrics_scrape_interval_ms: 60_000,
  metrics_flush_interval_ms: 600_000,
  circuit_failure_threshold: 3,
  circuit_cooldown_ms: 5_000,
  param_ack_timeout_ms: 5_000,
  riskguard_url: "http://localhost:19999/riskguard/validate"

config :logger, level: :warning
