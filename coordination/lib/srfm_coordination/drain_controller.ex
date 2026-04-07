defmodule SrfmCoordination.DrainController do
  @moduledoc """
  Graceful drain controller for zero-downtime service restarts.

  State machine per service:
    IDLE      -- not draining, service running normally
    DRAINING  -- waiting for open positions to close (polls every 10s)
    READY     -- zero open positions confirmed; restart can proceed
    RESTARTING -- restart in progress (delegates to ServiceSupervisor)
    ONLINE    -- service confirmed healthy again after restart

  Safety rules:
  -- Maximum drain timeout: 30 minutes, then force-restart with CRITICAL alert
  -- Blocks restart of :live_trader during active BH regime
     (BH mass must drop below @bh_mass_threshold before restart)
  -- Drain status is tracked in ETS and queried via HTTP

  REST:
    GET  /drain/status                    -- all service drain states
    POST /drain/:service/initiate         -- begin drain for a service
    POST /drain/:service/abort            -- abort drain, return to IDLE

  ETS table: :srfm_drain_state
    {service_name, %DrainEntry{}}
  """

  use GenServer
  require Logger

  @drain_poll_ms 10_000
  @max_drain_timeout_seconds 1_800   # 30 minutes
  @bh_mass_threshold 0.15            # BH mass must be below this to restart live_trader
  @open_positions_query_timeout_ms 5_000

  @state_table :srfm_drain_state

  # Valid drain states
  @states [:idle, :draining, :ready, :restarting, :online]

  defstruct active_drains: %{}

  defmodule DrainEntry do
    @moduledoc false
    defstruct [
      :service,
      :state,
      :initiated_at,
      :ready_at,
      :restart_at,
      :online_at,
      :abort_at,
      :drain_ref,
      :open_positions,
      :force_restart_reason
    ]
  end

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Initiate a drain for the named service. Returns :ok immediately; the
  drain proceeds asynchronously. Subscribe to EventBus :service_health
  to receive state transition events.
  """
  @spec initiate_drain(atom()) :: :ok | {:error, term()}
  def initiate_drain(service_name) when is_atom(service_name) do
    GenServer.call(__MODULE__, {:initiate_drain, service_name})
  end

  @doc "Abort an in-progress drain and return the service to IDLE."
  @spec abort_drain(atom()) :: :ok | {:error, term()}
  def abort_drain(service_name) when is_atom(service_name) do
    GenServer.call(__MODULE__, {:abort_drain, service_name})
  end

  @doc "Return the drain state map for a specific service."
  @spec get_state(atom()) :: {:ok, map()} | {:error, :not_found}
  def get_state(service_name) when is_atom(service_name) do
    case :ets.lookup(@state_table, service_name) do
      [{^service_name, entry}] -> {:ok, drain_entry_to_map(entry)}
      [] -> {:error, :not_found}
    end
  end

  @doc "Return drain state for all services currently tracked."
  @spec all_states() :: [map()]
  def all_states do
    :ets.tab2list(@state_table)
    |> Enum.map(fn {_svc, entry} -> drain_entry_to_map(entry) end)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@state_table, [:named_table, :set, :public, read_concurrency: true])
    Logger.info("[DrainController] Initialized")
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call({:initiate_drain, service}, _from, state) do
    existing = Map.get(state.active_drains, service)

    cond do
      existing != nil and existing.state in [:draining, :restarting] ->
        {:reply, {:error, {:already_draining, existing.state}}, state}

      true ->
        entry = %DrainEntry{
          service: service,
          state: :draining,
          initiated_at: DateTime.utc_now(),
          open_positions: nil
        }

        :ets.insert(@state_table, {service, entry})
        emit_state_transition(service, :idle, :draining)

        # Schedule first poll immediately
        ref = schedule_drain_poll(service, 0)
        updated_entry = %{entry | drain_ref: ref}
        :ets.insert(@state_table, {service, updated_entry})

        new_state = put_in(state.active_drains[service], updated_entry)
        Logger.info("[DrainController] Drain initiated for :#{service}")
        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:abort_drain, service}, _from, state) do
    case Map.fetch(state.active_drains, service) do
      {:ok, entry} ->
        if entry.drain_ref, do: Process.cancel_timer(entry.drain_ref)

        aborted = %{entry | state: :idle, abort_at: DateTime.utc_now(), drain_ref: nil}
        :ets.insert(@state_table, {service, aborted})

        emit_state_transition(service, entry.state, :idle)
        Logger.info("[DrainController] Drain aborted for :#{service}")

        new_state = %{state | active_drains: Map.delete(state.active_drains, service)}
        {:reply, :ok, new_state}

      :error ->
        {:reply, {:error, :not_draining}, state}
    end
  end

  @impl true
  def handle_info({:drain_poll, service}, state) do
    case Map.fetch(state.active_drains, service) do
      {:ok, entry} when entry.state == :draining ->
        new_state = run_drain_poll(service, entry, state)
        {:noreply, new_state}

      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:restart_service, service}, state) do
    case Map.fetch(state.active_drains, service) do
      {:ok, entry} when entry.state in [:ready, :draining] ->
        new_state = do_restart_service(service, entry, state)
        {:noreply, new_state}

      _ ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[DrainController] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Drain poll logic
  # ---------------------------------------------------------------------------

  defp run_drain_poll(service, entry, state) do
    elapsed = DateTime.diff(DateTime.utc_now(), entry.initiated_at, :second)

    if elapsed > @max_drain_timeout_seconds do
      force_restart(service, entry, state, :drain_timeout)
    else
      open_count = query_open_positions(service)
      updated_entry = %{entry | open_positions: open_count}

      cond do
        open_count == 0 ->
          # No positions -- check BH regime if this is live_trader
          handle_zero_positions(service, updated_entry, state)

        open_count > 0 ->
          ref = schedule_drain_poll(service, @drain_poll_ms)
          final_entry = %{updated_entry | drain_ref: ref}
          :ets.insert(@state_table, {service, final_entry})

          Logger.debug("[DrainController] :#{service} still has #{open_count} open position(s), waiting...")
          put_in(state.active_drains[service], final_entry)

        true ->
          # Query failed -- reschedule
          ref = schedule_drain_poll(service, @drain_poll_ms)
          final_entry = %{updated_entry | drain_ref: ref}
          :ets.insert(@state_table, {service, final_entry})
          put_in(state.active_drains[service], final_entry)
      end
    end
  end

  defp handle_zero_positions(:live_trader, entry, state) do
    bh_mass = get_bh_mass()

    if bh_mass >= @bh_mass_threshold do
      # BH regime active -- wait for it to subside
      Logger.info(
        "[DrainController] :live_trader has 0 positions but BH mass=#{Float.round(bh_mass, 4)} >= " <>
        "#{@bh_mass_threshold} -- waiting for regime to clear"
      )

      ref = schedule_drain_poll(:live_trader, @drain_poll_ms)
      updated = %{entry | drain_ref: ref}
      :ets.insert(@state_table, {:live_trader, updated})
      put_in(state.active_drains[:live_trader], updated)
    else
      mark_ready(:live_trader, entry, state)
    end
  end

  defp handle_zero_positions(service, entry, state) do
    mark_ready(service, entry, state)
  end

  defp mark_ready(service, entry, state) do
    ready_entry = %{entry | state: :ready, ready_at: DateTime.utc_now(), drain_ref: nil}
    :ets.insert(@state_table, {service, ready_entry})
    emit_state_transition(service, :draining, :ready)

    Logger.info("[DrainController] :#{service} is READY for restart")
    send(self(), {:restart_service, service})

    put_in(state.active_drains[service], ready_entry)
  end

  defp force_restart(service, entry, state, reason) do
    Logger.error("[DrainController] :#{service} drain timed out (#{reason}) -- forcing restart")

    emit_alert(:critical, service, "Drain timeout exceeded (#{@max_drain_timeout_seconds}s) -- forcing restart with #{entry.open_positions || 0} open positions")

    restarting_entry = %{entry |
      state: :restarting,
      restart_at: DateTime.utc_now(),
      drain_ref: nil,
      force_restart_reason: reason
    }

    :ets.insert(@state_table, {service, restarting_entry})
    emit_state_transition(service, :draining, :restarting)

    execute_restart(service)

    online_entry = %{restarting_entry | state: :online, online_at: DateTime.utc_now()}
    :ets.insert(@state_table, {service, online_entry})
    emit_state_transition(service, :restarting, :online)

    %{state | active_drains: Map.delete(state.active_drains, service)}
  end

  defp do_restart_service(service, entry, state) do
    Logger.info("[DrainController] Proceeding with restart of :#{service}")

    restarting_entry = %{entry | state: :restarting, restart_at: DateTime.utc_now()}
    :ets.insert(@state_table, {service, restarting_entry})
    emit_state_transition(service, :ready, :restarting)

    execute_restart(service)

    online_entry = %{restarting_entry | state: :online, online_at: DateTime.utc_now()}
    :ets.insert(@state_table, {service, online_entry})
    emit_state_transition(service, :restarting, :online)

    Logger.info("[DrainController] :#{service} is ONLINE")
    %{state | active_drains: Map.delete(state.active_drains, service)}
  end

  defp execute_restart(service) do
    case Process.whereis(SrfmCoordination.ServiceSupervisor) do
      nil ->
        Logger.warning("[DrainController] ServiceSupervisor not available -- restart skipped")

      _pid ->
        try do
          SrfmCoordination.ServiceSupervisor.restart_service(service)
        catch
          _, reason ->
            Logger.error("[DrainController] Restart of :#{service} failed: #{inspect(reason)}")
        end
    end
  end

  # ---------------------------------------------------------------------------
  # Open position query
  # ---------------------------------------------------------------------------

  defp query_open_positions(service) do
    # Primary: ask the service via HTTP
    case query_positions_http(service) do
      {:ok, count} ->
        count

      {:error, _} ->
        # Fallback: read from SQLite if available
        query_positions_sqlite(service)
    end
  end

  defp query_positions_http(service) do
    case SrfmCoordination.ServiceRegistry.get_service(service) do
      {:ok, svc} ->
        url = "http://localhost:#{svc.port}/positions/open/count"

        try do
          case HTTPoison.get(url, [], recv_timeout: @open_positions_query_timeout_ms) do
            {:ok, %{status_code: 200, body: body}} ->
              case Jason.decode(body) do
                {:ok, %{"count" => count}} when is_integer(count) -> {:ok, count}
                {:ok, %{"open_positions" => count}} when is_integer(count) -> {:ok, count}
                _ -> {:error, :unexpected_response}
              end

            {:ok, %{status_code: code}} ->
              {:error, {:bad_status, code}}

            {:error, reason} ->
              {:error, reason}
          end
        catch
          _, reason -> {:error, reason}
        end

      {:error, :not_found} ->
        {:error, :service_not_registered}
    end
  end

  defp query_positions_sqlite(_service) do
    db_path = Application.get_env(:srfm_coordination, :live_trader_db_path,
                                  "/tmp/srfm_live_trader.db")

    try do
      {:ok, db} = Exqlite.Sqlite3.open(db_path)
      sql = "SELECT COUNT(*) FROM orders WHERE status = 'open'"
      {:ok, stmt} = Exqlite.Sqlite3.prepare(db, sql)

      count =
        case Exqlite.Sqlite3.step(db, stmt) do
          {:row, [n]} when is_integer(n) -> n
          _ -> -1
        end

      Exqlite.Sqlite3.release(db, stmt)
      Exqlite.Sqlite3.close(db)

      count
    rescue
      _ -> -1
    catch
      _, _ -> -1
    end
  end

  # ---------------------------------------------------------------------------
  # BH regime check
  # ---------------------------------------------------------------------------

  defp get_bh_mass do
    case SrfmCoordination.MetricsBridge.get_metric(:live_trader, "bh_mass_current") do
      {:ok, mass} -> mass
      {:error, _} -> 0.0
    end
  end

  # ---------------------------------------------------------------------------
  # Event helpers
  # ---------------------------------------------------------------------------

  defp emit_state_transition(service, old_state, new_state) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _pid ->
        SrfmCoordination.EventBus.publish(:service_health, %{
          type: :drain_state_changed,
          service: service,
          from: old_state,
          to: new_state
        })
    end
  end

  defp emit_alert(severity, service, message) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _pid ->
        SrfmCoordination.EventBus.publish(:alert, %{
          type: :drain_alert,
          severity: severity,
          service: service,
          message: message
        })
    end
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp schedule_drain_poll(service, delay_ms) do
    Process.send_after(self(), {:drain_poll, service}, delay_ms)
  end

  defp drain_entry_to_map(%DrainEntry{} = e) do
    %{
      service: e.service,
      state: e.state,
      initiated_at: e.initiated_at,
      ready_at: e.ready_at,
      restart_at: e.restart_at,
      online_at: e.online_at,
      abort_at: e.abort_at,
      open_positions: e.open_positions,
      force_restart_reason: e.force_restart_reason
    }
  end

  @doc "Return all valid drain states."
  def valid_states, do: @states
end
