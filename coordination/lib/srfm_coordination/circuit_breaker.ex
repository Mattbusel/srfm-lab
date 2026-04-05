defmodule SrfmCoordination.CircuitBreakerSupervisor do
  @moduledoc "Supervisor that starts one CircuitBreaker per named external API."

  use Supervisor
  require Logger

  @circuits [:alpaca, :binance, :coinmetrics, :fear_greed, :alternative_me]

  def start_link(opts \\ []) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    children =
      Enum.map(@circuits, fn name ->
        Supervisor.child_spec(
          {SrfmCoordination.CircuitBreaker, name: name},
          id: {SrfmCoordination.CircuitBreaker, name}
        )
      end)

    Logger.info("[CircuitBreakerSupervisor] Starting #{length(children)} circuit breakers")
    Supervisor.init(children, strategy: :one_for_one)
  end

  def circuits, do: @circuits
end

defmodule SrfmCoordination.CircuitBreaker do
  @moduledoc """
  Circuit breaker GenServer for a single named external API.

  State machine:
    CLOSED    — normal operation; failures tracked in a sliding window
    OPEN      — all calls rejected immediately; 60s cooldown timer running
    HALF_OPEN — one probe call allowed; success -> CLOSED, failure -> OPEN

  Transition rules:
    CLOSED -> OPEN      : 5 failures within 60-second window
    OPEN   -> HALF_OPEN : after 60-second cooldown
    HALF_OPEN -> CLOSED : probe call succeeds
    HALF_OPEN -> OPEN   : probe call fails (reset cooldown)

  Usage:
    SrfmCoordination.CircuitBreaker.call(:alpaca, fn ->
      HTTPoison.get("https://api.alpaca.markets/v2/account")
    end)
  """

  use GenServer
  require Logger

  @failure_threshold 5
  @window_seconds 60
  @cooldown_ms 60_000

  # Possible states: :closed | :open | :half_open
  defstruct name: nil,
            state: :closed,
            failures: [],
            last_failure: nil,
            opened_at: nil,
            probe_allowed: false,
            total_calls: 0,
            total_failures: 0,
            total_rejections: 0

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, name, name: via(name))
  end

  @doc """
  Wrap `fun` with circuit-breaking logic.
  Returns {:ok, result} | {:error, :circuit_open} | {:error, reason}.
  """
  @spec call(atom(), (-> any())) :: {:ok, any()} | {:error, term()}
  def call(circuit_name, fun) when is_atom(circuit_name) and is_function(fun, 0) do
    GenServer.call(via(circuit_name), {:call, fun}, 30_000)
  end

  @doc "Return the current state snapshot for a circuit."
  @spec status(atom()) :: map()
  def status(circuit_name) do
    GenServer.call(via(circuit_name), :status)
  end

  @doc "Manually reset a circuit to CLOSED state."
  @spec reset(atom()) :: :ok
  def reset(circuit_name) do
    GenServer.cast(via(circuit_name), :reset)
  end

  @doc "Return status of all known circuits."
  @spec all_statuses() :: [map()]
  def all_statuses do
    SrfmCoordination.CircuitBreakerSupervisor.circuits()
    |> Enum.map(fn name ->
      try do
        status(name)
      catch
        _, _ -> %{name: name, state: :unknown}
      end
    end)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(name) do
    Logger.info("[CircuitBreaker] #{name} initialized (CLOSED)")
    {:ok, %__MODULE__{name: name}}
  end

  @impl true
  def handle_call({:call, fun}, _from, %{state: :open} = s) do
    now = System.monotonic_time(:millisecond)
    elapsed = now - (s.opened_at || now)

    if elapsed >= @cooldown_ms do
      Logger.info("[CircuitBreaker:#{s.name}] Cooldown elapsed, moving to HALF_OPEN")
      new_s = %{s | state: :half_open, probe_allowed: true}
      execute_and_record(fun, new_s)
    else
      remaining = div(@cooldown_ms - elapsed, 1_000)
      Logger.debug("[CircuitBreaker:#{s.name}] OPEN — rejecting call (#{remaining}s remaining)")
      new_s = %{s | total_rejections: s.total_rejections + 1}
      {:reply, {:error, :circuit_open}, new_s}
    end
  end

  @impl true
  def handle_call({:call, fun}, _from, %{state: :half_open} = s) do
    execute_and_record(fun, s)
  end

  @impl true
  def handle_call({:call, fun}, _from, %{state: :closed} = s) do
    execute_and_record(fun, s)
  end

  @impl true
  def handle_call(:status, _from, s) do
    now = System.monotonic_time(:millisecond)
    in_state_ms = if s.opened_at, do: now - s.opened_at, else: 0

    info = %{
      name: s.name,
      state: s.state,
      failure_count_in_window: count_recent_failures(s.failures),
      last_failure: s.last_failure,
      time_in_state_ms: in_state_ms,
      total_calls: s.total_calls,
      total_failures: s.total_failures,
      total_rejections: s.total_rejections
    }

    {:reply, info, s}
  end

  @impl true
  def handle_cast(:reset, s) do
    Logger.info("[CircuitBreaker:#{s.name}] Manual reset to CLOSED")
    {:noreply, %{s | state: :closed, failures: [], opened_at: nil, probe_allowed: false}}
  end

  @impl true
  def handle_info(:cooldown_expired, %{state: :open} = s) do
    Logger.info("[CircuitBreaker:#{s.name}] Cooldown expired, moving to HALF_OPEN")
    {:noreply, %{s | state: :half_open, probe_allowed: true}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, s) do
    Logger.info("[CircuitBreaker:#{s.name}] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private
  # ---------------------------------------------------------------------------

  defp execute_and_record(fun, s) do
    result =
      try do
        {:ok, fun.()}
      catch
        kind, reason -> {:error, {kind, reason}}
      end

    case result do
      {:ok, value} ->
        new_s = on_success(s)
        {:reply, {:ok, value}, new_s}

      {:error, reason} ->
        new_s = on_failure(s, reason)
        {:reply, {:error, reason}, new_s}
    end
  end

  defp on_success(%{state: :half_open} = s) do
    Logger.info("[CircuitBreaker:#{s.name}] Probe succeeded — HALF_OPEN -> CLOSED")
    emit_event(s.name, :circuit_closed)
    %{s | state: :closed, failures: [], opened_at: nil, probe_allowed: false,
          total_calls: s.total_calls + 1}
  end

  defp on_success(s) do
    %{s | total_calls: s.total_calls + 1}
  end

  defp on_failure(%{state: :half_open} = s, reason) do
    Logger.warning("[CircuitBreaker:#{s.name}] Probe failed — HALF_OPEN -> OPEN")
    now = System.monotonic_time(:millisecond)
    Process.send_after(self(), :cooldown_expired, @cooldown_ms)
    emit_event(s.name, :circuit_opened)

    %{s |
      state: :open,
      opened_at: now,
      last_failure: {DateTime.utc_now(), reason},
      total_calls: s.total_calls + 1,
      total_failures: s.total_failures + 1
    }
  end

  defp on_failure(%{state: :closed} = s, reason) do
    now_ms = System.monotonic_time(:millisecond)
    failures = [now_ms | s.failures]
    recent = count_recent_failures(failures)

    Logger.debug("[CircuitBreaker:#{s.name}] Failure recorded (#{recent}/#{@failure_threshold} in window)")

    new_s = %{s |
      failures: failures,
      last_failure: {DateTime.utc_now(), reason},
      total_calls: s.total_calls + 1,
      total_failures: s.total_failures + 1
    }

    if recent >= @failure_threshold do
      Logger.error("[CircuitBreaker:#{s.name}] Threshold reached — CLOSED -> OPEN")
      Process.send_after(self(), :cooldown_expired, @cooldown_ms)
      emit_event(s.name, :circuit_opened)
      %{new_s | state: :open, opened_at: now_ms}
    else
      new_s
    end
  end

  defp on_failure(s, reason) do
    %{s | last_failure: {DateTime.utc_now(), reason},
          total_calls: s.total_calls + 1,
          total_failures: s.total_failures + 1}
  end

  defp count_recent_failures(failures) do
    cutoff = System.monotonic_time(:millisecond) - @window_seconds * 1_000
    Enum.count(failures, fn ts -> ts >= cutoff end)
  end

  defp emit_event(name, type) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _ -> SrfmCoordination.EventBus.publish(:alert, %{type: type, circuit: name})
    end
  end

  defp via(name), do: {:via, Registry, {SrfmCoordination.ServiceRegistry, {:circuit, name}}}
end
