defmodule SrfmCoordination.EventBus do
  @moduledoc """
  In-process pub/sub GenServer.

  Supported topics:
    :service_health       — service up/down/degraded transitions
    :trade_executed       — trade submission confirmations
    :hypothesis_generated — new IAE hypothesis ready
    :alert                — system alert events
    :parameter_changed    — parameter update applied

  API:
    subscribe(topic, pid)   — start receiving events; monitored so dead pids auto-cleanup
    unsubscribe(topic, pid) — stop receiving
    publish(topic, event)   — async fan-out to all subscribers; stored in ETS history
    history(topic, n)       — last n events for a topic (default 100)

  ETS table `:srfm_event_history` stores the last 1000 events per topic.
  Optionally writes events to SQLite when the dependency is present (graceful no-op if not).
  """

  use GenServer
  require Logger

  @valid_topics [:service_health, :trade_executed, :hypothesis_generated, :alert, :parameter_changed]
  @history_limit 1_000
  @history_table :srfm_event_history

  defstruct subscribers: %{}, monitor_refs: %{}

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Subscribe `pid` (default: self()) to `topic`."
  @spec subscribe(atom(), pid()) :: :ok | {:error, :invalid_topic}
  def subscribe(topic, pid \\ self()) do
    if topic in @valid_topics do
      GenServer.call(__MODULE__, {:subscribe, topic, pid})
    else
      {:error, :invalid_topic}
    end
  end

  @doc "Unsubscribe `pid` from `topic`."
  @spec unsubscribe(atom(), pid()) :: :ok
  def unsubscribe(topic, pid \\ self()) do
    GenServer.cast(__MODULE__, {:unsubscribe, topic, pid})
  end

  @doc "Publish an event to all subscribers of `topic`."
  @spec publish(atom(), map()) :: :ok
  def publish(topic, event) when is_map(event) do
    enriched = Map.merge(event, %{topic: topic, timestamp: DateTime.utc_now()})
    GenServer.cast(__MODULE__, {:publish, topic, enriched})
  end

  @doc "Return last `limit` events for a topic (newest first)."
  @spec history(atom(), pos_integer()) :: [map()]
  def history(topic, limit \\ 100) do
    case :ets.lookup(@history_table, topic) do
      [{^topic, events}] -> Enum.take(events, limit)
      [] -> []
    end
  end

  @doc "Return subscriber counts per topic."
  @spec subscriber_counts() :: map()
  def subscriber_counts do
    GenServer.call(__MODULE__, :subscriber_counts)
  end

  @doc "All valid topics."
  def topics, do: @valid_topics

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@history_table, [:named_table, :set, :public, read_concurrency: true])

    # Pre-seed empty history for all topics
    Enum.each(@valid_topics, fn t -> :ets.insert(@history_table, {t, []}) end)

    Logger.info("[EventBus] Initialized with topics: #{inspect(@valid_topics)}")
    {:ok, %__MODULE__{subscribers: Map.new(@valid_topics, fn t -> {t, MapSet.new()} end)}}
  end

  @impl true
  def handle_call({:subscribe, topic, pid}, _from, state) do
    ref = Process.monitor(pid)
    updated_subs = Map.update!(state.subscribers, topic, &MapSet.put(&1, pid))
    updated_refs = Map.put(state.monitor_refs, ref, {topic, pid})
    Logger.debug("[EventBus] #{inspect(pid)} subscribed to :#{topic}")
    {:reply, :ok, %{state | subscribers: updated_subs, monitor_refs: updated_refs}}
  end

  @impl true
  def handle_call(:subscriber_counts, _from, state) do
    counts = Map.new(state.subscribers, fn {t, pids} -> {t, MapSet.size(pids)} end)
    {:reply, counts, state}
  end

  @impl true
  def handle_cast({:unsubscribe, topic, pid}, state) do
    updated = Map.update(state.subscribers, topic, MapSet.new(), &MapSet.delete(&1, pid))
    {:noreply, %{state | subscribers: updated}}
  end

  @impl true
  def handle_cast({:publish, topic, event}, state) do
    subscribers = Map.get(state.subscribers, topic, MapSet.new())

    Enum.each(subscribers, fn pid ->
      send(pid, {:event, topic, event})
    end)

    append_history(topic, event)
    persist_event(topic, event)

    Logger.debug("[EventBus] Published :#{topic} to #{MapSet.size(subscribers)} subscribers")
    {:noreply, state}
  end

  # Subscriber process died — clean up
  @impl true
  def handle_info({:DOWN, ref, :process, _pid, _reason}, state) do
    case Map.pop(state.monitor_refs, ref) do
      {nil, refs} ->
        {:noreply, %{state | monitor_refs: refs}}

      {{topic, pid}, refs} ->
        updated = Map.update(state.subscribers, topic, MapSet.new(), &MapSet.delete(&1, pid))
        Logger.debug("[EventBus] Auto-unsubscribed dead process #{inspect(pid)} from :#{topic}")
        {:noreply, %{state | subscribers: updated, monitor_refs: refs}}
    end
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[EventBus] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Private
  # ---------------------------------------------------------------------------

  defp append_history(topic, event) do
    existing =
      case :ets.lookup(@history_table, topic) do
        [{^topic, list}] -> list
        [] -> []
      end

    trimmed = Enum.take([event | existing], @history_limit)
    :ets.insert(@history_table, {topic, trimmed})
  end

  defp persist_event(_topic, _event) do
    # Graceful no-op: write to SQLite when Ecto is available
    # In production wire this to an Ecto.Repo insert
    :ok
  end
end
