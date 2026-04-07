defmodule SrfmCoordination.Application do
  @moduledoc """
  OTP Application entry point for the SRFM Coordination Service.

  Supervision tree (all :one_for_one):
    1.  ServiceRegistry      -- ETS-backed Registry for service PIDs/metadata
    2.  ServiceSupervisor    -- DynamicSupervisor for external process launchers
    3.  HealthMonitor        -- GenServer polling all services every 30s
    4.  CircuitBreakerSup    -- Supervisor owning one CircuitBreaker per API
    5.  EventBus             -- GenServer pub/sub with ETS history
    6.  MetricsCollector     -- GenServer aggregating Prometheus metrics
    7.  ParameterCoordinator -- GenServer coordinating parameter updates
    8.  AlertManager         -- GenServer centralizing alert routing (legacy)
    9.  MetricsBridge        -- Prometheus scraper/aggregator (15s cadence)
    10. PerformanceTracker   -- Equity curve, Sharpe, rollback trigger
    11. ParameterHistory     -- Persistent param change log (ETS + SQLite)
    12. GenomeReceiver       -- Polls Go IAE /genome/best every 5 minutes
    13. DrainController      -- Zero-downtime drain coordinator
    14. Alerting             -- Routes EventBus events to Slack/PagerDuty
    15. HTTPServer           -- Plug.Cowboy on port 8781

  Max restarts: 10 in 60 seconds before the application itself shuts down.
  """

  use Application
  require Logger

  @http_port 8781

  @impl true
  def start(_type, _args) do
    Logger.info("[Application] Starting SRFM Coordination Service v#{app_version()}")

    children = [
      # 1. Named Registry — key: service name (atom), value: metadata map
      {Registry, keys: :unique, name: SrfmCoordination.ServiceRegistry},

      # 2. DynamicSupervisor for managed external processes
      {SrfmCoordination.ServiceSupervisor, []},

      # 3. Health monitor — polls every 30s
      {SrfmCoordination.HealthMonitor, []},

      # 4. Circuit breaker supervisor — starts one breaker per named API
      {SrfmCoordination.CircuitBreakerSupervisor, []},

      # 5. In-process pub/sub bus
      {SrfmCoordination.EventBus, []},

      # 6. Metrics aggregation
      {SrfmCoordination.MetricsCollector, []},

      # 7. Parameter coordination across IAE services
      {SrfmCoordination.ParameterCoordinator, []},

      # 8. Alert manager (legacy)
      {SrfmCoordination.AlertManager, []},

      # 9. Prometheus metrics scraper and aggregator
      {SrfmCoordination.MetricsBridge, []},

      # 10. Live trader performance tracking and rollback trigger
      {SrfmCoordination.PerformanceTracker, []},

      # 11. Persistent parameter change history with analytics
      {SrfmCoordination.ParameterHistory, []},

      # 12. Genome evolution receiver -- polls Go IAE every 5 minutes
      {SrfmCoordination.GenomeReceiver, []},

      # 13. Graceful drain controller for zero-downtime restarts
      {SrfmCoordination.DrainController, []},

      # 14. Alerting -- routes EventBus events to Slack/PagerDuty
      {SrfmCoordination.Alerting, []},

      # 15. HTTP API server
      build_cowboy_spec()
    ]

    opts = [
      strategy: :one_for_one,
      name: SrfmCoordination.Supervisor,
      max_restarts: 10,
      max_seconds: 60
    ]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("[Application] Supervision tree started. HTTP on port #{@http_port}")
        {:ok, pid}

      {:error, reason} = err ->
        Logger.error("[Application] Failed to start supervision tree: #{inspect(reason)}")
        err
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("[Application] Coordination service stopping — goodbye.")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp build_cowboy_spec do
    Plug.Cowboy.child_spec(
      scheme: :http,
      plug: SrfmCoordination.HTTP.Router,
      options: [port: @http_port, transport_options: [num_acceptors: 10]]
    )
  end

  defp app_version do
    case :application.get_key(:srfm_coordination, :vsn) do
      {:ok, vsn} -> List.to_string(vsn)
      _ -> "unknown"
    end
  end
end
