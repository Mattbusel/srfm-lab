defmodule SrfmCoordination.ServiceRegistry do
  @moduledoc """
  Registry for all IAE microservices.

  Each service entry stores:
    - name            :: atom    — unique service identifier
    - pid             :: pid | nil — current managing process (if supervised)
    - port            :: integer — HTTP port the service listens on
    - health_status   :: :healthy | :degraded | :down | :unknown
    - last_heartbeat  :: DateTime.t | nil
    - restart_count   :: non_neg_integer
    - registered_at   :: DateTime.t

  Uses the OTP Registry (started in Application) for PID lookups, and a
  separate ETS table `:srfm_service_meta` for the rich metadata.

  All mutations emit events to EventBus so subscribers can react immediately.
  """

  use GenServer
  require Logger

  @table :srfm_service_meta
  @registry SrfmCoordination.ServiceRegistry

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Register a new service or overwrite an existing entry."
  @spec register_service(atom(), map()) :: :ok | {:error, term()}
  def register_service(name, attrs) when is_atom(name) do
    GenServer.call(__MODULE__, {:register, name, attrs})
  end

  @doc "Remove a service from the registry."
  @spec deregister_service(atom()) :: :ok
  def deregister_service(name) when is_atom(name) do
    GenServer.call(__MODULE__, {:deregister, name})
  end

  @doc "Fetch a service by name. Returns {:ok, map} or {:error, :not_found}."
  @spec get_service(atom()) :: {:ok, map()} | {:error, :not_found}
  def get_service(name) when is_atom(name) do
    case :ets.lookup(@table, name) do
      [{^name, meta}] -> {:ok, meta}
      [] -> {:error, :not_found}
    end
  end

  @doc "Return all registered services as a list of maps."
  @spec list_all() :: [map()]
  def list_all do
    :ets.tab2list(@table) |> Enum.map(fn {_k, v} -> v end)
  end

  @doc "Find a service by its HTTP port."
  @spec find_by_port(integer()) :: {:ok, map()} | {:error, :not_found}
  def find_by_port(port) when is_integer(port) do
    result =
      :ets.tab2list(@table)
      |> Enum.find(fn {_k, v} -> v.port == port end)

    case result do
      {_k, meta} -> {:ok, meta}
      nil -> {:error, :not_found}
    end
  end

  @doc "Update the health status of a service."
  @spec update_health(atom(), atom(), map()) :: :ok | {:error, :not_found}
  def update_health(name, status, extra_attrs \\ %{}) do
    GenServer.call(__MODULE__, {:update_health, name, status, extra_attrs})
  end

  @doc "Increment the restart counter for a service."
  @spec increment_restarts(atom()) :: :ok
  def increment_restarts(name) do
    GenServer.cast(__MODULE__, {:inc_restarts, name})
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    table = :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])
    Logger.info("[ServiceRegistry] ETS table #{@table} created (#{inspect(table)})")
    {:ok, %{table: table}}
  end

  @impl true
  def handle_call({:register, name, attrs}, _from, state) do
    now = DateTime.utc_now()

    entry =
      Map.merge(
        %{
          name: name,
          pid: nil,
          port: 0,
          health_status: :unknown,
          last_heartbeat: nil,
          restart_count: 0,
          registered_at: now
        },
        attrs
      )
      |> Map.put(:name, name)

    :ets.insert(@table, {name, entry})
    Logger.info("[ServiceRegistry] Registered service :#{name} on port #{entry.port}")
    emit_event(:service_registered, entry)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:deregister, name}, _from, state) do
    case :ets.lookup(@table, name) do
      [{^name, meta}] ->
        :ets.delete(@table, name)
        Logger.info("[ServiceRegistry] Deregistered service :#{name}")
        emit_event(:service_deregistered, meta)

      [] ->
        Logger.warning("[ServiceRegistry] Deregister called for unknown service :#{name}")
    end

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:update_health, name, status, extra}, _from, state) do
    case :ets.lookup(@table, name) do
      [{^name, meta}] ->
        updated =
          Map.merge(meta, extra)
          |> Map.put(:health_status, status)
          |> Map.put(:last_heartbeat, DateTime.utc_now())

        :ets.insert(@table, {name, updated})

        if meta.health_status != status do
          Logger.info("[ServiceRegistry] :#{name} health: #{meta.health_status} -> #{status}")
          emit_event(:service_health_changed, %{name: name, old: meta.health_status, new: status})
        end

        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_cast({:inc_restarts, name}, state) do
    case :ets.lookup(@table, name) do
      [{^name, meta}] ->
        updated = Map.update(meta, :restart_count, 1, &(&1 + 1))
        :ets.insert(@table, {name, updated})

      [] ->
        :ok
    end

    {:noreply, state}
  end

  @impl true
  def terminate(reason, _state) do
    Logger.info("[ServiceRegistry] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp emit_event(type, payload) do
    # Guard: EventBus may not be up yet during initial boot
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _pid -> SrfmCoordination.EventBus.publish(:service_health, %{type: type, payload: payload})
    end
  end
end
