defmodule SrfmCoordination.HealthMonitor do
  @moduledoc """
  GenServer that polls all registered services every 30 seconds.

  For each service:
    - HTTP GET /health with a 5-second timeout
    - Records: response_time_ms, last_healthy_at, consecutive_failures
    - 3 consecutive failures  -> mark DEGRADED, emit :service_degraded event
    - 5 consecutive failures  -> mark DOWN, trigger restart via ServiceSupervisor

  Stores per-service health history (last 1000 checks) in ETS table
  `:srfm_health_history`.

  Computes an overall system health score = % of services currently healthy.
  """

  use GenServer
  require Logger

  @poll_interval_ms 30_000
  @http_timeout_ms 5_000
  @history_limit 1_000
  @degraded_threshold 3
  @down_threshold 5
  @table :srfm_health_history

  defstruct poll_ref: nil, start_time: nil

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Force an immediate health check cycle (useful for testing)."
  def check_now do
    GenServer.cast(__MODULE__, :check_now)
  end

  @doc "Return the current system health summary."
  @spec system_health() :: map()
  def system_health do
    GenServer.call(__MODULE__, :system_health)
  end

  @doc "Return the health history for a specific service (newest first)."
  @spec history(atom()) :: [map()]
  def history(service_name) do
    case :ets.lookup(@table, service_name) do
      [{^service_name, entries}] -> entries
      [] -> []
    end
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])
    Logger.info("[HealthMonitor] Initialized, polling every #{@poll_interval_ms}ms")
    ref = schedule_poll()
    {:ok, %__MODULE__{poll_ref: ref, start_time: System.monotonic_time(:second)}}
  end

  @impl true
  def handle_cast(:check_now, state) do
    do_poll_all()
    {:noreply, state}
  end

  @impl true
  def handle_call(:system_health, _from, state) do
    services = SrfmCoordination.ServiceRegistry.list_all()
    total = length(services)

    counts =
      Enum.reduce(services, %{healthy: 0, degraded: 0, down: 0, unknown: 0}, fn svc, acc ->
        Map.update(acc, svc.health_status, 1, &(&1 + 1))
      end)

    score =
      if total == 0, do: 100.0, else: Float.round(counts.healthy / total * 100, 1)

    overall =
      cond do
        total == 0 -> :no_services
        score >= 80.0 -> :healthy
        score >= 50.0 -> :degraded
        true -> :critical
      end

    uptime = System.monotonic_time(:second) - state.start_time

    result = %{
      overall: overall,
      score: score,
      counts: counts,
      total_services: total,
      uptime_seconds: uptime,
      checked_at: DateTime.utc_now()
    }

    {:reply, result, state}
  end

  @impl true
  def handle_info(:poll, state) do
    do_poll_all()
    ref = schedule_poll()
    {:noreply, %{state | poll_ref: ref}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[HealthMonitor] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private — polling logic
  # ---------------------------------------------------------------------------

  defp do_poll_all do
    services = SrfmCoordination.ServiceRegistry.list_all()

    if services == [] do
      Logger.debug("[HealthMonitor] No services registered, skipping poll")
    else
      Logger.debug("[HealthMonitor] Polling #{length(services)} services")

      Task.async_stream(
        services,
        fn svc -> check_service(svc) end,
        max_concurrency: 10,
        timeout: @http_timeout_ms + 2_000,
        on_timeout: :kill_task
      )
      |> Stream.run()
    end
  end

  defp check_service(svc) do
    url = "http://localhost:#{svc.port}/health"
    start = System.monotonic_time(:millisecond)

    result =
      try do
        case HTTPoison.get(url, [], recv_timeout: @http_timeout_ms, timeout: @http_timeout_ms) do
          {:ok, %{status_code: code}} when code in 200..299 ->
            elapsed = System.monotonic_time(:millisecond) - start
            {:ok, elapsed}

          {:ok, %{status_code: code}} ->
            {:error, {:bad_status, code}}

          {:error, reason} ->
            {:error, reason}
        end
      catch
        kind, reason ->
          {:error, {kind, reason}}
      end

    process_check_result(svc, result)
  end

  defp process_check_result(svc, {:ok, response_ms}) do
    record = build_record(svc.name, :ok, response_ms, nil)
    append_history(svc.name, record)
    SrfmCoordination.ServiceRegistry.update_health(svc.name, :healthy, %{
      last_healthy_at: DateTime.utc_now(),
      consecutive_failures: 0,
      last_response_ms: response_ms
    })
  end

  defp process_check_result(svc, {:error, reason}) do
    record = build_record(svc.name, :error, nil, reason)
    append_history(svc.name, record)

    # Compute new consecutive failure count
    failures =
      case :ets.lookup(@table, svc.name) do
        [{_, [latest | _]}] ->
          prev_failures = Map.get(latest, :consecutive_failures, 0)
          if latest.status == :error, do: prev_failures + 1, else: 1

        _ ->
          1
      end

    new_status =
      cond do
        failures >= @down_threshold -> :down
        failures >= @degraded_threshold -> :degraded
        true -> svc.health_status
      end

    SrfmCoordination.ServiceRegistry.update_health(svc.name, new_status, %{
      consecutive_failures: failures
    })

    if new_status == :degraded and svc.health_status != :degraded do
      Logger.warning("[HealthMonitor] :#{svc.name} DEGRADED after #{failures} failures")
      emit(:service_degraded, %{name: svc.name, failures: failures})
    end

    if new_status == :down and svc.health_status != :down do
      Logger.error("[HealthMonitor] :#{svc.name} DOWN after #{failures} failures — triggering restart")
      emit(:service_down, %{name: svc.name, failures: failures})
      trigger_restart(svc.name)
    end
  end

  defp build_record(name, status, response_ms, error) do
    %{
      name: name,
      status: status,
      response_ms: response_ms,
      error: error,
      consecutive_failures: 0,
      checked_at: DateTime.utc_now()
    }
  end

  defp append_history(name, record) do
    existing =
      case :ets.lookup(@table, name) do
        [{^name, list}] -> list
        [] -> []
      end

    trimmed = Enum.take([record | existing], @history_limit)
    :ets.insert(@table, {name, trimmed})
  end

  defp trigger_restart(name) do
    Task.start(fn ->
      Process.sleep(1_000)
      SrfmCoordination.ServiceSupervisor.restart_service(name)
    end)
  end

  defp emit(type, payload) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _ -> SrfmCoordination.EventBus.publish(:service_health, %{type: type, payload: payload})
    end
  end

  defp schedule_poll do
    Process.send_after(self(), :poll, @poll_interval_ms)
  end
end
