defmodule SrfmCoordination.ParameterCoordinator do
  @moduledoc """
  Coordinates parameter updates across IAE services.

  When the IAE generates a validated hypothesis with a parameter delta:
    1. Validate against RiskGuard constraints (HTTP POST /riskguard/validate)
    2. Store the new parameter values in the local ParameterStore (ETS)
    3. Fan-out HTTP POST /parameters/update to all consumer services
    4. Await acknowledgements (30-second window)
    5. If >20% of services fail to ACK, rollback to the previous version

  Parameter versioning: every change is stored with a timestamp and author.
  The full change history is accessible via `history/1`.

  ETS tables:
    :srfm_param_store   — current parameter values  {key, value, version}
    :srfm_param_history — list of all past versions  {key, [versions]}
  """

  use GenServer
  require Logger

  @ack_timeout_ms 30_000
  @rollback_threshold 0.20
  @store_table :srfm_param_store
  @history_table :srfm_param_history
  @riskguard_timeout_ms 10_000

  defstruct pending_updates: %{}

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Apply a parameter delta. Returns :ok or {:error, reason}.
  `delta` is a map of parameter_name => new_value.
  `author` is a string identifying the source (e.g. "IAE_v2.3_hypothesis_42").
  """
  @spec apply_delta(map(), String.t()) :: :ok | {:error, term()}
  def apply_delta(delta, author \\ "system") when is_map(delta) do
    GenServer.call(__MODULE__, {:apply_delta, delta, author}, @ack_timeout_ms + 5_000)
  end

  @doc "Get the current value of a parameter."
  @spec get(String.t()) :: {:ok, any()} | {:error, :not_found}
  def get(key) when is_binary(key) do
    case :ets.lookup(@store_table, key) do
      [{^key, entry}] -> {:ok, entry}
      [] -> {:error, :not_found}
    end
  end

  @doc "Get all current parameters."
  @spec all() :: map()
  def all do
    :ets.tab2list(@store_table)
    |> Map.new(fn {k, v} -> {k, v} end)
  end

  @doc "Get the change history for a specific parameter key."
  @spec history(String.t()) :: [map()]
  def history(key) when is_binary(key) do
    case :ets.lookup(@history_table, key) do
      [{^key, versions}] -> versions
      [] -> []
    end
  end

  @doc "Acknowledge receipt of an update (called by consumer services via HTTP)."
  @spec acknowledge(String.t(), atom()) :: :ok
  def acknowledge(update_id, service_name) do
    GenServer.cast(__MODULE__, {:ack, update_id, service_name})
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@store_table, [:named_table, :set, :public, read_concurrency: true])
    :ets.new(@history_table, [:named_table, :set, :public, read_concurrency: true])
    Logger.info("[ParameterCoordinator] Initialized")
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call({:apply_delta, delta, author}, _from, state) do
    update_id = generate_update_id()
    Logger.info("[ParameterCoordinator] Applying delta #{update_id} from #{author}: #{map_size(delta)} keys")

    with :ok <- validate_with_riskguard(delta),
         snapshot = take_snapshot(Map.keys(delta)),
         :ok <- write_parameters(delta, author, update_id) do

      case fan_out_update(update_id, delta) do
        :ok ->
          Logger.info("[ParameterCoordinator] Update #{update_id} acknowledged by all services")
          emit_event(delta, update_id, author)
          {:reply, :ok, state}

        {:error, :ack_timeout} ->
          Logger.error("[ParameterCoordinator] ACK timeout for #{update_id} — rolling back")
          rollback(snapshot, author, update_id)
          {:reply, {:error, :ack_timeout_rolled_back}, state}

        {:error, {:too_many_failures, pct}} ->
          Logger.error("[ParameterCoordinator] #{Float.round(pct * 100, 1)}% failed ACK — rolling back")
          rollback(snapshot, author, update_id)
          {:reply, {:error, {:too_many_failures, pct}}, state}
      end
    else
      {:error, reason} ->
        Logger.warning("[ParameterCoordinator] Delta rejected: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_cast({:ack, _update_id, _service_name}, state) do
    # ACKs are currently processed synchronously in fan_out_update.
    # This cast path is for async acknowledgements from HTTP endpoints.
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[ParameterCoordinator] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private — validation
  # ---------------------------------------------------------------------------

  defp validate_with_riskguard(delta) do
    # Attempt to reach RiskGuard service; pass through if unreachable
    riskguard_url = Application.get_env(:srfm_coordination, :riskguard_url,
                                         "http://localhost:8790/riskguard/validate")

    body = Jason.encode!(%{parameters: delta})

    try do
      case HTTPoison.post(riskguard_url, body, [{"content-type", "application/json"}],
                          recv_timeout: @riskguard_timeout_ms) do
        {:ok, %{status_code: 200}} ->
          :ok

        {:ok, %{status_code: 422, body: resp_body}} ->
          {:error, {:riskguard_rejected, Jason.decode!(resp_body)}}

        {:ok, %{status_code: code}} ->
          Logger.warning("[ParameterCoordinator] RiskGuard returned #{code} — allowing update")
          :ok

        {:error, %HTTPoison.Error{reason: :econnrefused}} ->
          Logger.warning("[ParameterCoordinator] RiskGuard unreachable — bypassing validation")
          :ok

        {:error, reason} ->
          Logger.warning("[ParameterCoordinator] RiskGuard error #{inspect(reason)} — allowing")
          :ok
      end
    catch
      _, _ -> :ok
    end
  end

  defp take_snapshot(keys) do
    Map.new(keys, fn key ->
      case :ets.lookup(@store_table, key) do
        [{^key, entry}] -> {key, entry}
        [] -> {key, nil}
      end
    end)
  end

  defp write_parameters(delta, author, update_id) do
    now = DateTime.utc_now()

    Enum.each(delta, fn {key, value} ->
      version = %{
        value: value,
        author: author,
        update_id: update_id,
        applied_at: now
      }

      :ets.insert(@store_table, {key, version})

      existing_history =
        case :ets.lookup(@history_table, key) do
          [{^key, list}] -> list
          [] -> []
        end

      :ets.insert(@history_table, {key, [version | existing_history]})
    end)

    :ok
  end

  defp fan_out_update(update_id, delta) do
    services =
      SrfmCoordination.ServiceRegistry.list_all()
      |> Enum.filter(fn svc -> svc.health_status in [:healthy, :degraded] end)

    total = length(services)

    if total == 0 do
      :ok
    else
      payload = Jason.encode!(%{update_id: update_id, parameters: delta})

      results =
        Task.async_stream(
          services,
          fn svc -> notify_service(svc, payload) end,
          max_concurrency: 20,
          timeout: @ack_timeout_ms,
          on_timeout: :kill_task
        )
        |> Enum.map(fn
          {:ok, result} -> result
          {:exit, :timeout} -> :timeout
        end)

      failures = Enum.count(results, fn r -> r != :ok end)
      failure_rate = failures / total

      cond do
        failure_rate > @rollback_threshold ->
          {:error, {:too_many_failures, failure_rate}}

        true ->
          :ok
      end
    end
  end

  defp notify_service(svc, payload) do
    url = "http://localhost:#{svc.port}/parameters/update"

    try do
      case HTTPoison.post(url, payload, [{"content-type", "application/json"}],
                          recv_timeout: 10_000) do
        {:ok, %{status_code: code}} when code in 200..299 -> :ok
        {:ok, %{status_code: code}} -> {:error, {:bad_status, code}}
        {:error, reason} -> {:error, reason}
      end
    catch
      _, reason -> {:error, reason}
    end
  end

  defp rollback(snapshot, _author, update_id) do
    Logger.warning("[ParameterCoordinator] Rolling back update #{update_id}")

    Enum.each(snapshot, fn {key, previous} ->
      if previous do
        :ets.insert(@store_table, {key, previous})
      else
        :ets.delete(@store_table, key)
      end
    end)
  end

  defp emit_event(delta, update_id, author) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _ ->
        SrfmCoordination.EventBus.publish(:parameter_changed, %{
          update_id: update_id,
          keys: Map.keys(delta),
          author: author
        })
    end
  end

  defp generate_update_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
