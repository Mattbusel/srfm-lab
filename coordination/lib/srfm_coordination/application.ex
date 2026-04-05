defmodule SrfmCoordination.Application do
  @moduledoc """
  OTP Application entry point for the SRFM Coordination Service.

  Supervision tree (all :one_for_one):
    1. ServiceRegistry      - ETS-backed Registry for service PIDs/metadata
    2. ServiceSupervisor    - DynamicSupervisor for external process launchers
    3. HealthMonitor        - GenServer polling all services every 30s
    4. CircuitBreakerSup    - Supervisor owning one CircuitBreaker per API
    5. EventBus             - GenServer pub/sub with ETS history
    6. MetricsCollector     - GenServer aggregating Prometheus metrics
    7. ParameterCoordinator - GenServer coordinating parameter updates
    8. AlertManager         - GenServer centralizing alert routing
    9. HTTPServer           - Plug.Cowboy on port 8781

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

      # 8. Alert manager
      {SrfmCoordination.AlertManager, []},

      # 9. HTTP API server
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
