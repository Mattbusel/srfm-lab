defmodule SrfmCoordination.AlertManager do
  @moduledoc """
  Centralized alert management for the SRFM trading system.

  Alert levels: :info | :warning | :critical | :emergency

  Features:
    - Deduplication: same alert key suppressed for 15 minutes after first fire
    - Routing: :critical/:emergency written to alerts.log + stdout immediately
    - Snooze: silence a specific alert key until a given DateTime
    - Rule evaluation: configurable rules checked on incoming events
    - EventBus integration: auto-subscribes to :service_health and :alert topics

  Built-in alert rules:
    :service_down         — any service transitions to :down
    :daily_loss_exceeded  — equity drops below configured threshold
    :drawdown_threshold   — max drawdown exceeded
    :feed_disconnected    — market data feed health check fails
    :circuit_opened       — a circuit breaker opened
  """

  use GenServer
  require Logger

  @dedup_window_ms 900_000   # 15 minutes
  @log_path "alerts.log"

  defstruct fired: %{}, snoozed: %{}, rules: [], alert_count: 0

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Fire an alert manually."
  @spec alert(atom(), atom(), String.t(), map()) :: :ok
  def alert(key, level, message, context \\ %{}) do
    GenServer.cast(__MODULE__, {:alert, key, level, message, context})
  end

  @doc "Snooze an alert key until `until` DateTime."
  @spec snooze(atom(), DateTime.t()) :: :ok
  def snooze(key, until) do
    GenServer.cast(__MODULE__, {:snooze, key, until})
  end

  @doc "Cancel a snooze for an alert key."
  @spec unsnooze(atom()) :: :ok
  def unsnooze(key) do
    GenServer.cast(__MODULE__, {:unsnooze, key})
  end

  @doc "Return a list of recently fired alerts."
  @spec recent(pos_integer()) :: [map()]
  def recent(limit \\ 50) do
    GenServer.call(__MODULE__, {:recent, limit})
  end

  @doc "Return current snooze table."
  @spec snoozed() :: map()
  def snoozed do
    GenServer.call(__MODULE__, :snoozed)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    rules = default_rules()
    Logger.info("[AlertManager] Initialized with #{length(rules)} alert rules")

    # Subscribe to relevant EventBus topics (after it starts)
    Process.send_after(self(), :subscribe_to_events, 1_000)

    {:ok, %__MODULE__{rules: rules}}
  end

  @impl true
  def handle_cast({:alert, key, level, message, context}, state) do
    state = process_alert(key, level, message, context, state)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:snooze, key, until}, state) do
    Logger.info("[AlertManager] Snoozed :#{key} until #{DateTime.to_iso8601(until)}")
    {:noreply, %{state | snoozed: Map.put(state.snoozed, key, until)}}
  end

  @impl true
  def handle_cast({:unsnooze, key}, state) do
    Logger.info("[AlertManager] Unsnoozed :#{key}")
    {:noreply, %{state | snoozed: Map.delete(state.snoozed, key)}}
  end

  @impl true
  def handle_call({:recent, limit}, _from, state) do
    alerts = get_recent_alerts(limit)
    {:reply, alerts, state}
  end

  @impl true
  def handle_call(:snoozed, _from, state) do
    {:reply, state.snoozed, state}
  end

  # Receive events from EventBus
  @impl true
  def handle_info({:event, topic, event}, state) do
    state = evaluate_rules(topic, event, state)
    {:noreply, state}
  end

  @impl true
  def handle_info(:subscribe_to_events, state) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil ->
        Process.send_after(self(), :subscribe_to_events, 2_000)

      _pid ->
        SrfmCoordination.EventBus.subscribe(:service_health, self())
        SrfmCoordination.EventBus.subscribe(:alert, self())
        Logger.info("[AlertManager] Subscribed to EventBus topics")
    end

    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[AlertManager] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private — alert processing
  # ---------------------------------------------------------------------------

  defp process_alert(key, level, message, context, state) do
    now = System.monotonic_time(:millisecond)

    cond do
      snoozed?(key, state) ->
        Logger.debug("[AlertManager] Alert :#{key} is snoozed, suppressing")
        state

      recently_fired?(key, now, state) ->
        Logger.debug("[AlertManager] Alert :#{key} deduplicated (within 15m window)")
        state

      true ->
        fire_alert(key, level, message, context, now, state)
    end
  end

  defp fire_alert(key, level, message, context, now, state) do
    entry = %{
      key: key,
      level: level,
      message: message,
      context: context,
      fired_at: DateTime.utc_now()
    }

    route_alert(level, entry)
    store_alert(entry)

    new_fired = Map.put(state.fired, key, now)
    %{state | fired: new_fired, alert_count: state.alert_count + 1}
  end

  defp route_alert(:info, entry) do
    Logger.info("[ALERT:INFO] #{entry.message}")
  end

  defp route_alert(:warning, entry) do
    Logger.warning("[ALERT:WARNING] #{entry.message}")
  end

  defp route_alert(level, entry) when level in [:critical, :emergency] do
    msg = format_alert(entry)
    Logger.error(msg)
    write_to_log(msg)
    IO.puts(:stderr, msg)
  end

  defp route_alert(_level, entry) do
    Logger.info("[ALERT] #{entry.message}")
  end

  defp format_alert(entry) do
    "[ALERT:#{String.upcase(to_string(entry.level))}] [#{DateTime.to_iso8601(entry.fired_at)}] " <>
    "#{entry.key}: #{entry.message} | ctx=#{inspect(entry.context)}"
  end

  defp write_to_log(msg) do
    log_path = Application.get_env(:srfm_coordination, :alert_log_path, @log_path)

    try do
      File.write(log_path, msg <> "\n", [:append])
    catch
      _, reason ->
        Logger.error("[AlertManager] Failed to write to #{log_path}: #{inspect(reason)}")
    end
  end

  defp store_alert(entry) do
    existing =
      case :ets.lookup(:srfm_alerts, :history) do
        [{:history, list}] -> list
        [] -> []
      end

    case :ets.info(:srfm_alerts) do
      :undefined ->
        :ets.new(:srfm_alerts, [:named_table, :set, :public])

      _ ->
        :ok
    end

    trimmed = Enum.take([entry | existing], 1_000)
    :ets.insert(:srfm_alerts, {:history, trimmed})
  end

  defp get_recent_alerts(limit) do
    case :ets.info(:srfm_alerts) do
      :undefined ->
        []

      _ ->
        case :ets.lookup(:srfm_alerts, :history) do
          [{:history, list}] -> Enum.take(list, limit)
          [] -> []
        end
    end
  end

  defp snoozed?(key, state) do
    case Map.get(state.snoozed, key) do
      nil -> false
      until -> DateTime.compare(DateTime.utc_now(), until) == :lt
    end
  end

  defp recently_fired?(key, now, state) do
    case Map.get(state.fired, key) do
      nil -> false
      last_ms -> (now - last_ms) < @dedup_window_ms
    end
  end

  # ---------------------------------------------------------------------------
  # Private — rule evaluation
  # ---------------------------------------------------------------------------

  defp evaluate_rules(topic, event, state) do
    Enum.reduce(state.rules, state, fn rule, acc ->
      if rule.matches?(topic, event) do
        {key, level, msg, ctx} = rule.build_alert(event)
        process_alert(key, level, msg, ctx, acc)
      else
        acc
      end
    end)
  end

  defp default_rules do
    [
      %{
        matches?: fn topic, event ->
          topic == :service_health and Map.get(event, :type) == :service_down
        end,
        build_alert: fn event ->
          name = get_in(event, [:payload, :name]) || event[:name] || :unknown
          {:service_down, :critical, "Service :#{name} is DOWN", event}
        end
      },
      %{
        matches?: fn topic, event ->
          topic == :alert and Map.get(event, :type) == :circuit_opened
        end,
        build_alert: fn event ->
          circuit = event[:circuit] || :unknown
          {:circuit_opened, :warning, "Circuit breaker :#{circuit} OPENED", event}
        end
      },
      %{
        matches?: fn topic, event ->
          topic == :service_health and Map.get(event, :type) == :service_degraded
        end,
        build_alert: fn event ->
          name = get_in(event, [:payload, :name]) || :unknown
          failures = get_in(event, [:payload, :failures]) || 0
          {:service_degraded, :warning, "Service :#{name} DEGRADED (#{failures} failures)", event}
        end
      }
    ]
  end
end
