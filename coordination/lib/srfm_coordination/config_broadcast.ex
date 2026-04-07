defmodule SrfmCoordination.ConfigBroadcast do
  @moduledoc """
  Broadcasts parameter updates to all registered service endpoints.
  Tracks acknowledgments and retries failed deliveries.

  When a parameter version is published via broadcast_params/2, this module:
    1. Fans out HTTP POST /config/update to every registered healthy service
    2. Records delivery status in the :delivery_tracker ETS table
    3. Waits up to 60 seconds for each service to acknowledge
    4. Retries unacknowledged deliveries with exponential backoff (max 3 retries)
    5. Alerts via AlertManager if a service fails all retries

  Services confirm receipt by calling acknowledge/2 from their HTTP handler,
  which is wired through the coordination HTTP router.
  """

  use GenServer
  require Logger

  @ack_timeout_ms      60_000
  @max_retries         3
  @retry_base_delay_ms 2_000
  @http_timeout_ms     8_000
  @tracker_table       :delivery_tracker

  -- BroadcastResult summarizes the outcome of one fan-out operation.
  defmodule BroadcastResult do
    @moduledoc "Result of a broadcast_params/2 call."
    defstruct [:version, :total, :delivered, :failed, :pending]

    @type t :: %__MODULE__{
      version:   String.t(),
      total:     non_neg_integer(),
      delivered: non_neg_integer(),
      failed:    non_neg_integer(),
      pending:   non_neg_integer()
    }
  end

  -- DeliveryRecord tracks the state of one {version, service} delivery.
  defmodule DeliveryRecord do
    @moduledoc "Per-service delivery state for one config version."
    defstruct [
      :version,
      :service,
      :status,          -- :pending | :delivered | :failed
      :attempts,
      :last_attempt_ts,
      :acked_at
    ]

    @type t :: %__MODULE__{
      version:          String.t(),
      service:          atom(),
      status:           :pending | :delivered | :failed,
      attempts:         non_neg_integer(),
      last_attempt_ts:  integer() | nil,
      acked_at:         integer() | nil
    }
  end

  defstruct [
    active_versions: %{},   -- version => %{params, services_pending}
    retry_timers:    %{}    -- {version, service} => timer_ref
  ]

  -- ---------------------------------------------------------------------------
  -- Public API
  -- ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Broadcast `params` as a new config version to all registered services.
  Returns a BroadcastResult summarizing initial delivery.
  """
  @spec broadcast_params(map(), String.t()) :: {:ok, BroadcastResult.t()} | {:error, term()}
  def broadcast_params(params, version) when is_map(params) and is_binary(version) do
    GenServer.call(__MODULE__, {:broadcast, params, version}, 30_000)
  end

  @doc """
  Record acknowledgment from `service_name` for config `version`.
  Called by the HTTP router when a service POSTs to /config/ack.
  """
  @spec acknowledge(atom(), String.t()) :: :ok
  def acknowledge(service_name, version) when is_atom(service_name) and is_binary(version) do
    GenServer.cast(__MODULE__, {:ack, service_name, version})
  end

  @doc """
  Return a map of delivery records for the given `version`.
  Keys are service names, values are DeliveryRecord structs.
  """
  @spec get_delivery_status(String.t()) :: %{atom() => DeliveryRecord.t()}
  def get_delivery_status(version) when is_binary(version) do
    GenServer.call(__MODULE__, {:delivery_status, version})
  end

  @doc "Return a summary map for all tracked versions."
  @spec list_versions() :: [map()]
  def list_versions do
    GenServer.call(__MODULE__, :list_versions)
  end

  -- ---------------------------------------------------------------------------
  -- GenServer callbacks
  -- ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@tracker_table, [:named_table, :set, :public, read_concurrency: true])
    Logger.info("[ConfigBroadcast] Initialized -- delivery tracker ready")
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call({:broadcast, params, version}, _from, state) do
    services =
      SrfmCoordination.ServiceRegistry.list_all()
      |> Enum.filter(fn svc -> svc.health_status in [:healthy, :degraded] end)

    total = length(services)
    Logger.info("[ConfigBroadcast] Broadcasting version #{version} to #{total} services")

    -- Seed delivery records for all services
    Enum.each(services, fn svc ->
      record = %DeliveryRecord{
        version:         version,
        service:         svc.name,
        status:          :pending,
        attempts:        0,
        last_attempt_ts: nil,
        acked_at:        nil
      }
      :ets.insert(@tracker_table, {{version, svc.name}, record})
    end)

    -- Fan-out synchronously in tasks, collect initial results
    results =
      Task.async_stream(
        services,
        fn svc -> {svc.name, deliver_to_service(svc, version, params)} end,
        max_concurrency: 20,
        timeout: @http_timeout_ms + 2_000,
        on_timeout: :kill_task
      )
      |> Enum.map(fn
        {:ok, {name, result}} -> {name, result}
        {:exit, _reason}      -> {:unknown, :timeout}
      end)

    -- Update ETS with initial attempt outcomes
    now = System.system_time(:millisecond)

    {delivered, failed, pending} =
      Enum.reduce(results, {0, 0, 0}, fn {svc_name, result}, {d, f, p} ->
        case result do
          :ok ->
            -- Mark as delivered; will be confirmed via ack
            update_record(version, svc_name, :delivered, 1, now)
            {d + 1, f, p}

          {:error, _reason} ->
            -- Schedule retry
            schedule_retry(version, svc_name, params, 1)
            update_record(version, svc_name, :pending, 1, now)
            {d, f, p + 1}
        end
      end)

    -- Schedule ack timeout check
    Process.send_after(self(), {:check_ack_timeout, version}, @ack_timeout_ms)

    -- Track in state for ack correlation
    new_active = Map.put(state.active_versions, version, %{
      params:   params,
      total:    total,
      started_at: now
    })

    result = %BroadcastResult{
      version:   version,
      total:     total,
      delivered: delivered,
      failed:    failed,
      pending:   pending
    }

    {:reply, {:ok, result}, %{state | active_versions: new_active}}
  end

  @impl true
  def handle_call({:delivery_status, version}, _from, state) do
    records =
      :ets.match_object(@tracker_table, {{version, :_}, :_})
      |> Map.new(fn {{_v, svc}, record} -> {svc, record} end)

    {:reply, records, state}
  end

  @impl true
  def handle_call(:list_versions, _from, state) do
    summaries =
      Enum.map(state.active_versions, fn {version, meta} ->
        records =
          :ets.match_object(@tracker_table, {{version, :_}, :_})
          |> Enum.map(fn {_key, rec} -> rec end)

        delivered = Enum.count(records, &(&1.status == :delivered))
        failed    = Enum.count(records, &(&1.status == :failed))
        pending   = Enum.count(records, &(&1.status == :pending))

        %{
          version:    version,
          total:      meta.total,
          delivered:  delivered,
          failed:     failed,
          pending:    pending,
          started_at: meta.started_at
        }
      end)

    {:reply, summaries, state}
  end

  @impl true
  def handle_cast({:ack, service_name, version}, state) do
    now = System.system_time(:millisecond)

    case :ets.lookup(@tracker_table, {version, service_name}) do
      [{{^version, ^service_name}, record}] ->
        updated = %{record | status: :delivered, acked_at: now}
        :ets.insert(@tracker_table, {{version, service_name}, updated})
        Logger.debug("[ConfigBroadcast] #{service_name} acked version #{version}")

      [] ->
        Logger.debug("[ConfigBroadcast] Unexpected ack from #{service_name} for #{version}")
    end

    {:noreply, state}
  end

  @impl true
  def handle_info({:retry_delivery, version, service_name, params, attempt}, state) do
    Logger.info(
      "[ConfigBroadcast] Retrying delivery to #{service_name} for #{version} " <>
      "(attempt #{attempt}/#{@max_retries})"
    )

    case :ets.lookup(@tracker_table, {version, service_name}) do
      [{{^version, ^service_name}, record}] when record.status == :delivered ->
        -- Already acked, nothing to do
        {:noreply, state}

      _ ->
        services = SrfmCoordination.ServiceRegistry.list_all()
        svc = Enum.find(services, fn s -> s.name == service_name end)

        if svc do
          now = System.system_time(:millisecond)

          case deliver_to_service(svc, version, params) do
            :ok ->
              update_record(version, service_name, :delivered, attempt, now)
              Logger.info("[ConfigBroadcast] Retry succeeded for #{service_name} v#{version}")

            {:error, reason} when attempt < @max_retries ->
              update_record(version, service_name, :pending, attempt, now)
              schedule_retry(version, service_name, params, attempt + 1)
              Logger.warning(
                "[ConfigBroadcast] Retry #{attempt} failed for #{service_name}: #{inspect(reason)}"
              )

            {:error, reason} ->
              update_record(version, service_name, :failed, attempt, now)
              Logger.error(
                "[ConfigBroadcast] All retries exhausted for #{service_name} v#{version}: " <>
                inspect(reason)
              )
              send_delivery_failure_alert(service_name, version)
          end
        end

        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:check_ack_timeout, version}, state) do
    pending =
      :ets.match_object(@tracker_table, {{version, :_}, :_})
      |> Enum.filter(fn {_key, record} -> record.status == :pending end)

    if pending != [] do
      names = Enum.map(pending, fn {{_v, svc}, _} -> svc end)
      Logger.warning(
        "[ConfigBroadcast] Ack timeout for version #{version} -- " <>
        "pending services: #{inspect(names)}"
      )

      Enum.each(pending, fn {{^version, svc_name}, record} ->
        now = System.system_time(:millisecond)
        updated = %{record | status: :failed, last_attempt_ts: now}
        :ets.insert(@tracker_table, {{version, svc_name}, updated})
        send_delivery_failure_alert(svc_name, version)
      end)
    end

    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[ConfigBroadcast] Terminating: #{inspect(reason)}")
    :ok
  end

  -- ---------------------------------------------------------------------------
  -- Private -- HTTP delivery
  -- ---------------------------------------------------------------------------

  defp deliver_to_service(svc, version, params) do
    url     = "http://localhost:#{svc.port}/config/update"
    payload = Jason.encode!(%{version: version, parameters: params})

    try do
      case HTTPoison.post(
        url,
        payload,
        [{"content-type", "application/json"}],
        recv_timeout: @http_timeout_ms,
        timeout: @http_timeout_ms
      ) do
        {:ok, %{status_code: code}} when code in 200..299 ->
          :ok

        {:ok, %{status_code: code}} ->
          {:error, {:bad_status, code}}

        {:error, reason} ->
          {:error, reason}
      end
    catch
      kind, reason -> {:error, {kind, reason}}
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- delivery tracking
  -- ---------------------------------------------------------------------------

  defp update_record(version, service_name, status, attempts, ts) do
    record =
      case :ets.lookup(@tracker_table, {version, service_name}) do
        [{_key, existing}] -> existing
        [] ->
          %DeliveryRecord{
            version: version,
            service: service_name,
            status:  status,
            attempts: 0,
            last_attempt_ts: nil,
            acked_at: nil
          }
      end

    updated = %{record |
      status:          status,
      attempts:        attempts,
      last_attempt_ts: ts
    }

    :ets.insert(@tracker_table, {{version, service_name}, updated})
  end

  defp schedule_retry(version, service_name, params, attempt) do
    delay = @retry_base_delay_ms * :math.pow(2, attempt - 1) |> round()
    Logger.debug(
      "[ConfigBroadcast] Scheduling retry #{attempt} for #{service_name} in #{delay}ms"
    )
    Process.send_after(
      self(),
      {:retry_delivery, version, service_name, params, attempt},
      delay
    )
  end

  defp send_delivery_failure_alert(service_name, version) do
    case Process.whereis(SrfmCoordination.AlertManager) do
      nil -> :ok
      _ ->
        SrfmCoordination.AlertManager.alert(
          :config_delivery_failed,
          :warning,
          "Config version #{version} failed to deliver to #{service_name}",
          %{service: service_name, version: version}
        )
    end
  end
end
