defmodule SrfmCoordination.AlertManager do
  @moduledoc """
  Deduplicates, throttles, and routes alerts to configured channels.
  Prevents alert storms during system degradation events.

  Severity levels: :info | :warning | :critical

  Deduplication:
    The same {severity, title} combination is suppressed for 5 minutes after
    first firing. The suppressed count is incremented and stored so callers
    can see how many duplicates were dropped.

  Rate limiting:
    No more than 10 alerts per severity per 5-minute window. When the limit
    is hit the call returns {:suppressed, :rate_limited} and a single
    "rate limit reached" alert fires (itself subject to dedup).

  Routing:
    :critical -> PagerDuty + Slack (via Alerting module)
    :warning  -> Slack only
    :info     -> EventBus only (no external notifications)

  Lifecycle:
    Alert records live in ETS table :srfm_alert_manager for fast dedup.
    Resolved alerts are marked with resolved_at and sent a resolution note.
  """

  use GenServer
  require Logger

  @dedup_window_ms   300_000   -- 5 minute dedup window
  @max_alerts_per_window 10
  @rate_window_ms    300_000   -- same 5 minutes for rate limiting
  @history_limit     500
  @table             :srfm_alert_manager

  -- AlertRecord stores the full lifecycle of one alert instance.
  defmodule AlertRecord do
    @moduledoc "One alert record in the manager."
    defstruct [
      :id,
      :severity,
      :title,
      :body,
      :metadata,
      :ts,
      :count,        -- number of times this {severity, title} fired in window
      :suppressed,   -- true if this specific record was the suppressed copy
      :resolved_at
    ]

    @type t :: %__MODULE__{
      id:          String.t(),
      severity:    :info | :warning | :critical,
      title:       String.t(),
      body:        String.t(),
      metadata:    map(),
      ts:          integer(),
      count:       non_neg_integer(),
      suppressed:  boolean(),
      resolved_at: integer() | nil
    }
  end

  defstruct [
    fired:         %{},   -- {severity, title} => {first_ts_ms, count}
    rate_counters: %{},   -- severity => [{ts_ms}]
    alert_count:   0
  ]

  -- ---------------------------------------------------------------------------
  -- Public API
  -- ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Send an alert with `severity`, `title`, and `body`.
  Returns :ok if the alert was routed, or {:suppressed, reason} if it was
  dropped by deduplication or rate limiting.
  """
  @spec send_alert(atom(), String.t(), String.t(), map()) ::
    :ok | {:suppressed, :dedup} | {:suppressed, :rate_limited}
  def send_alert(severity, title, body, metadata \\ %{})
      when severity in [:info, :warning, :critical]
      and is_binary(title)
      and is_binary(body) do
    GenServer.call(__MODULE__, {:send_alert, severity, title, body, metadata})
  end

  @doc """
  Acknowledge resolution of alert `alert_id`.
  Marks the record resolved and sends a resolution notification if applicable.
  """
  @spec resolve_alert(String.t()) :: :ok | {:error, :not_found}
  def resolve_alert(alert_id) when is_binary(alert_id) do
    GenServer.call(__MODULE__, {:resolve, alert_id})
  end

  @doc """
  Return all alerts fired within the last hour that have not been resolved.
  Sorted newest-first.
  """
  @spec get_active_alerts() :: [AlertRecord.t()]
  def get_active_alerts do
    GenServer.call(__MODULE__, :get_active_alerts)
  end

  @doc "Return recent alert history (default 50 records, newest first)."
  @spec recent(pos_integer()) :: [AlertRecord.t()]
  def recent(limit \\ 50) when is_integer(limit) and limit > 0 do
    GenServer.call(__MODULE__, {:recent, limit})
  end

  @doc "Manually clear the dedup window for a {severity, title} key."
  @spec clear_dedup(atom(), String.t()) :: :ok
  def clear_dedup(severity, title) do
    GenServer.cast(__MODULE__, {:clear_dedup, severity, title})
  end

  -- Legacy API shim -- preserve compatibility with existing calls.
  @doc false
  @spec alert(atom(), atom(), String.t(), map()) :: :ok
  def alert(key, level, message, context \\ %{}) do
    case send_alert(level, to_string(key), message, context) do
      :ok -> :ok
      {:suppressed, _} -> :ok
    end
  end

  -- ---------------------------------------------------------------------------
  -- GenServer callbacks
  -- ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])
    :ets.insert(@table, {:history, []})
    Logger.info("[AlertManager] Initialized -- dedup=#{@dedup_window_ms}ms rate=#{@max_alerts_per_window}/5m")
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call({:send_alert, severity, title, body, metadata}, _from, state) do
    now = System.system_time(:millisecond)

    cond do
      dedup_suppressed?(severity, title, now, state) ->
        -- Increment counter but do not route
        new_state = increment_dedup_count(severity, title, state)
        Logger.debug("[AlertManager] Dedup suppressed #{severity}:#{title}")
        {:reply, {:suppressed, :dedup}, new_state}

      rate_limited?(severity, now, state) ->
        Logger.warning("[AlertManager] Rate limit hit for #{severity} alerts")
        {:reply, {:suppressed, :rate_limited}, state}

      true ->
        {record, new_state} = build_and_fire(severity, title, body, metadata, now, state)
        {:reply, :ok, new_state}
        |> tap(fn _ -> route_alert(record) end)
    end
  end

  @impl true
  def handle_call({:resolve, alert_id}, _from, state) do
    case lookup_alert(alert_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      record ->
        now     = System.system_time(:millisecond)
        updated = %{record | resolved_at: now}
        store_alert(updated)
        send_resolution_notification(updated)
        Logger.info("[AlertManager] Alert #{alert_id} resolved")
        {:reply, :ok, state}
    end
  end

  @impl true
  def handle_call(:get_active_alerts, _from, state) do
    cutoff = System.system_time(:millisecond) - 3_600_000  -- last hour

    alerts =
      get_history()
      |> Enum.filter(fn r -> r.ts >= cutoff and is_nil(r.resolved_at) end)

    {:reply, alerts, state}
  end

  @impl true
  def handle_call({:recent, limit}, _from, state) do
    alerts = get_history() |> Enum.take(limit)
    {:reply, alerts, state}
  end

  @impl true
  def handle_cast({:clear_dedup, severity, title}, state) do
    new_fired = Map.delete(state.fired, {severity, title})
    {:noreply, %{state | fired: new_fired}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[AlertManager] Terminating: #{inspect(reason)}")
    :ok
  end

  -- ---------------------------------------------------------------------------
  -- Private -- dedup and rate limiting
  -- ---------------------------------------------------------------------------

  defp dedup_suppressed?(severity, title, now, state) do
    case Map.get(state.fired, {severity, title}) do
      nil            -> false
      {first_ts, _c} -> now - first_ts < @dedup_window_ms
    end
  end

  defp rate_limited?(severity, now, state) do
    cutoff = now - @rate_window_ms
    recent_for_severity = Map.get(state.rate_counters, severity, [])
    active = Enum.filter(recent_for_severity, fn ts -> ts >= cutoff end)
    length(active) >= @max_alerts_per_window
  end

  defp increment_dedup_count(severity, title, state) do
    key = {severity, title}
    {ts, count} = Map.get(state.fired, key, {System.system_time(:millisecond), 0})
    %{state | fired: Map.put(state.fired, key, {ts, count + 1})}
  end

  -- ---------------------------------------------------------------------------
  -- Private -- building and routing alerts
  -- ---------------------------------------------------------------------------

  defp build_and_fire(severity, title, body, metadata, now, state) do
    id = generate_id()

    record = %AlertRecord{
      id:         id,
      severity:   severity,
      title:      title,
      body:       body,
      metadata:   metadata,
      ts:         now,
      count:      1,
      suppressed: false,
      resolved_at: nil
    }

    store_alert(record)

    -- Update dedup window
    new_fired = Map.put(state.fired, {severity, title}, {now, 1})

    -- Update rate counter
    existing_counter = Map.get(state.rate_counters, severity, [])
    cutoff           = now - @rate_window_ms
    trimmed_counter  = Enum.filter(existing_counter, fn ts -> ts >= cutoff end)
    new_counters     = Map.put(state.rate_counters, severity, [now | trimmed_counter])

    new_state = %{state |
      fired:         new_fired,
      rate_counters: new_counters,
      alert_count:   state.alert_count + 1
    }

    {record, new_state}
  end

  defp route_alert(%AlertRecord{severity: :info} = record) do
    Logger.info("[ALERT:INFO] #{record.title} -- #{record.body}")
    publish_to_event_bus(record)
  end

  defp route_alert(%AlertRecord{severity: :warning} = record) do
    Logger.warning("[ALERT:WARNING] #{record.title} -- #{record.body}")
    publish_to_event_bus(record)
    notify_slack(record)
  end

  defp route_alert(%AlertRecord{severity: :critical} = record) do
    Logger.error("[ALERT:CRITICAL] #{record.title} -- #{record.body}")
    publish_to_event_bus(record)
    notify_slack(record)
    notify_pagerduty(record)
  end

  defp send_resolution_notification(%AlertRecord{severity: :info}), do: :ok

  defp send_resolution_notification(record) do
    Logger.info("[AlertManager] Resolved: #{record.title} (id=#{record.id})")
    publish_to_event_bus(%{record | title: "[RESOLVED] #{record.title}"})
    notify_slack(%{record | body: "[RESOLVED] #{record.body}"})
  end

  -- ---------------------------------------------------------------------------
  -- Private -- notification sinks
  -- ---------------------------------------------------------------------------

  defp publish_to_event_bus(record) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _ ->
        SrfmCoordination.EventBus.publish(:alert, %{
          type:     :alert_fired,
          id:       record.id,
          severity: record.severity,
          title:    record.title,
          body:     record.body,
          metadata: record.metadata,
          ts:       record.ts
        })
    end
  end

  defp notify_slack(record) do
    case Process.whereis(SrfmCoordination.Alerting) do
      nil -> :ok
      _ ->
        try do
          SrfmCoordination.Alerting.send_slack(
            "[#{String.upcase(to_string(record.severity))}] #{record.title}",
            record.body
          )
        catch
          _, reason ->
            Logger.debug("[AlertManager] Slack notify failed: #{inspect(reason)}")
        end
    end
  end

  defp notify_pagerduty(record) do
    case Process.whereis(SrfmCoordination.Alerting) do
      nil -> :ok
      _ ->
        try do
          SrfmCoordination.Alerting.send_pagerduty(
            record.title,
            record.body,
            record.metadata
          )
        catch
          _, reason ->
            Logger.debug("[AlertManager] PagerDuty notify failed: #{inspect(reason)}")
        end
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- ETS storage
  -- ---------------------------------------------------------------------------

  defp store_alert(record) do
    existing = get_history()
    -- Replace existing record if same ID (for updates like resolve), or prepend
    updated =
      case Enum.find_index(existing, fn r -> r.id == record.id end) do
        nil   -> [record | existing]
        index -> List.replace_at(existing, index, record)
      end

    trimmed = Enum.take(updated, @history_limit)
    :ets.insert(@table, {:history, trimmed})
  end

  defp lookup_alert(id) do
    get_history() |> Enum.find(fn r -> r.id == id end)
  end

  defp get_history do
    case :ets.lookup(@table, :history) do
      [{:history, list}] -> list
      [] -> []
    end
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
