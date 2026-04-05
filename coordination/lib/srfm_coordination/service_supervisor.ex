defmodule SrfmCoordination.ServiceSupervisor do
  @moduledoc """
  DynamicSupervisor that manages external process launchers (ServiceWorker).

  Each IAE service (Python/Go/Rust) is wrapped in a ServiceWorker GenServer
  that owns an Erlang Port to the OS process. If the worker crashes, the
  DynamicSupervisor restarts it according to the configured strategy.

  Public API:
    start_service/3  - launch an external service under supervision
    stop_service/1   - gracefully terminate a supervised service
    restart_service/1 - stop + start
    list_services/0  - list all active child specs
  """

  use DynamicSupervisor
  require Logger

  @name __MODULE__

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: @name)
  end

  @doc """
  Start a new supervised external service.
  `cmd` is a list of strings, e.g. ["python", "iae_core.py", "--port", "8900"].
  `opts` may include: restart: :permanent | :transient | :temporary
  """
  @spec start_service(atom(), [String.t()], integer(), keyword()) ::
          {:ok, pid()} | {:error, term()}
  def start_service(name, cmd, port, opts \\ []) do
    Logger.info("[ServiceSupervisor] Starting :#{name} cmd=#{inspect(cmd)} port=#{port}")

    child_spec = {
      SrfmCoordination.ServiceWorker,
      [name: name, cmd: cmd, port: port, opts: opts]
    }

    case DynamicSupervisor.start_child(@name, child_spec) do
      {:ok, pid} = ok ->
        SrfmCoordination.ServiceRegistry.register_service(name, %{pid: pid, port: port})
        ok

      {:error, reason} = err ->
        Logger.error("[ServiceSupervisor] Failed to start :#{name}: #{inspect(reason)}")
        err
    end
  end

  @doc "Stop a managed service gracefully."
  @spec stop_service(atom()) :: :ok | {:error, :not_found}
  def stop_service(name) do
    case find_worker_pid(name) do
      {:ok, pid} ->
        Logger.info("[ServiceSupervisor] Stopping :#{name}")
        DynamicSupervisor.terminate_child(@name, pid)
        SrfmCoordination.ServiceRegistry.deregister_service(name)
        :ok

      :error ->
        {:error, :not_found}
    end
  end

  @doc "Restart a managed service."
  @spec restart_service(atom()) :: {:ok, pid()} | {:error, term()}
  def restart_service(name) do
    Logger.info("[ServiceSupervisor] Restarting :#{name}")

    case find_worker_pid(name) do
      {:ok, old_pid} ->
        # Retrieve spec before killing the worker
        spec = GenServer.call(old_pid, :get_spec)
        DynamicSupervisor.terminate_child(@name, old_pid)
        SrfmCoordination.ServiceRegistry.increment_restarts(name)
        Process.sleep(500)
        start_service(spec.name, spec.cmd, spec.port, spec.opts)

      :error ->
        {:error, :not_found}
    end
  end

  @doc "List all supervised service workers."
  @spec list_services() :: [map()]
  def list_services do
    DynamicSupervisor.which_children(@name)
    |> Enum.map(fn {_id, pid, _type, _modules} ->
      case pid do
        :restarting -> %{pid: nil, status: :restarting}
        pid -> GenServer.call(pid, :get_spec)
      end
    end)
  end

  # ---------------------------------------------------------------------------
  # DynamicSupervisor callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    Logger.info("[ServiceSupervisor] DynamicSupervisor initialized")
    DynamicSupervisor.init(strategy: :one_for_one, max_restarts: 5, max_seconds: 30)
  end

  # ---------------------------------------------------------------------------
  # Private
  # ---------------------------------------------------------------------------

  defp find_worker_pid(name) do
    DynamicSupervisor.which_children(@name)
    |> Enum.find_value(:error, fn
      {_id, :restarting, _type, _mods} ->
        false

      {_id, pid, _type, _mods} ->
        try do
          spec = GenServer.call(pid, :get_spec, 2_000)
          if spec.name == name, do: {:ok, pid}, else: false
        catch
          _, _ -> false
        end
    end)
  end
end

# ---------------------------------------------------------------------------
# ServiceWorker — wraps a single external OS process via Erlang Port
# ---------------------------------------------------------------------------

defmodule SrfmCoordination.ServiceWorker do
  @moduledoc """
  GenServer that owns an Erlang Port for one external IAE service process.
  Handles stdout/stderr logging, detects crashes, and allows graceful stops.
  """

  use GenServer
  require Logger

  defstruct [:name, :cmd, :port_ref, :os_port, :opts, :started_at]

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(args) do
    name = Keyword.fetch!(args, :name)
    GenServer.start_link(__MODULE__, args, name: via(name))
  end

  def child_spec(args) do
    name = Keyword.fetch!(args, :name)

    %{
      id: {__MODULE__, name},
      start: {__MODULE__, :start_link, [args]},
      restart: Keyword.get(args[:opts] || [], :restart, :permanent),
      shutdown: 10_000,
      type: :worker
    }
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(args) do
    name = Keyword.fetch!(args, :name)
    cmd = Keyword.fetch!(args, :cmd)
    port_num = Keyword.fetch!(args, :port)
    opts = Keyword.get(args, :opts, [])

    Logger.info("[ServiceWorker:#{name}] Opening port: #{Enum.join(cmd, " ")}")

    [executable | arguments] = cmd

    port_ref =
      Port.open(
        {:spawn_executable, System.find_executable(executable) || executable},
        [
          :binary,
          :exit_status,
          :stderr_to_stdout,
          {:args, arguments},
          {:env, [{'PORT', String.to_charlist("#{port_num}")}]}
        ]
      )

    state = %__MODULE__{
      name: name,
      cmd: cmd,
      port_ref: port_ref,
      os_port: port_num,
      opts: opts,
      started_at: System.monotonic_time(:second)
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:get_spec, _from, state) do
    spec = %{
      name: state.name,
      cmd: state.cmd,
      port: state.os_port,
      opts: state.opts,
      started_at: state.started_at
    }

    {:reply, spec, state}
  end

  @impl true
  def handle_info({port, {:data, data}}, %{port_ref: port} = state) do
    Logger.debug("[ServiceWorker:#{state.name}] stdout: #{String.trim(data)}")
    {:noreply, state}
  end

  @impl true
  def handle_info({port, {:exit_status, status}}, %{port_ref: port} = state) do
    Logger.warning("[ServiceWorker:#{state.name}] Process exited with status #{status}")
    SrfmCoordination.ServiceRegistry.update_health(state.name, :down, %{exit_status: status})
    SrfmCoordination.EventBus.publish(:service_health, %{type: :process_exited, name: state.name, status: status})
    {:stop, {:process_exited, status}, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(_reason, state) do
    if Port.info(state.port_ref) != nil do
      Port.close(state.port_ref)
    end

    :ok
  end

  defp via(name), do: {:via, Registry, {SrfmCoordination.ServiceRegistry, name}}
end
