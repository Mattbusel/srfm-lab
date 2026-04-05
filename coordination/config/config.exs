import Config

# ---------------------------------------------------------------------------
# SRFM Coordination Service — Base Configuration
# ---------------------------------------------------------------------------

config :srfm_coordination,
  # HTTP port for the coordination API
  http_port: 8781,

  # RiskGuard validation endpoint
  riskguard_url: "http://localhost:8790/riskguard/validate",

  # Alert log file path (absolute or relative to CWD)
  alert_log_path: "alerts.log",

  # Health monitor settings
  health_poll_interval_ms: 30_000,
  health_check_timeout_ms: 5_000,
  health_degraded_threshold: 3,
  health_down_threshold: 5,

  # Circuit breaker defaults
  circuit_failure_threshold: 5,
  circuit_window_seconds: 60,
  circuit_cooldown_ms: 60_000,

  # Metrics scrape interval
  metrics_scrape_interval_ms: 15_000,
  metrics_flush_interval_ms: 300_000,

  # Parameter coordinator
  param_ack_timeout_ms: 30_000,
  param_rollback_threshold: 0.20,

  # Known IAE services (populated at runtime via ServiceRegistry.register_service/2)
  services: [
    %{name: :iae_core,      port: 8900, cmd: ["python", "src/iae_core.py"]},
    %{name: :iae_risk,      port: 8901, cmd: ["python", "src/risk_guard.py"]},
    %{name: :iae_feed,      port: 8902, cmd: ["python", "src/market_feed.py"]},
    %{name: :iae_executor,  port: 8903, cmd: ["python", "src/trade_executor.py"]},
    %{name: :iae_analytics, port: 8904, cmd: ["python", "src/analytics.py"]}
  ]

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

config :logger, :console,
  format: "[$level] $time $metadata$message\n",
  metadata: [:module, :pid],
  level: :info

# Import environment-specific config (must be last)
import_config "#{config_env()}.exs"
