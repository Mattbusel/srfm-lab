defmodule SrfmCoordination.WebSocketHandler do
  @moduledoc """
  WebSocket handler for real-time event streaming to connected clients.

  Clients connect to /ws/events and subscribe to topics using JSON messages.

  Message protocol:
    Client -> Server:
      {"action": "subscribe",   "topics": ["params_updated", "circuit_open"]}
      {"action": "unsubscribe", "topics": ["params_updated"]}
      {"action": "ping"}

    Server -> Client:
      {"topic": "params_updated", "payload": {...}, "ts": 1234567890}
      {"action": "pong"}
      {"action": "error", "message": "..."}
      {"action": "subscribed", "topics": [...]}

  The handler subscribes to EventBus topics and fans out matching events to all
  connected WebSocket sessions that have subscribed to the corresponding topic.

  Topic mapping (EventBus atom -> WebSocket string):
    :parameter_changed -> "params_updated"
    :service_health    -> "service_health"
    :alert             -> "alert"
    :trade_executed    -> "trade_executed"
    :hypothesis_generated -> "hypothesis_generated"
    :circuit_open      -> "circuit_open"
    :circuit_close     -> "circuit_close"
    :rollback          -> "rollback"

  Ping/pong keepalive fires every 30 seconds.
  Dead connections are cleaned up via Registry monitor.
  """

  @behaviour WebSock

  require Logger

  alias SrfmCoordination.EventBus

  @ping_interval_ms 30_000

  # All valid subscribable topics (WebSocket string names)
  @valid_topics ~w[
    params_updated
    service_health
    alert
    trade_executed
    hypothesis_generated
    circuit_open
    circuit_close
    rollback
  ]

  # Map from EventBus atom topics to WS string topics
  @topic_map %{
    parameter_changed: "params_updated",
    service_health: "service_health",
    alert: "alert",
    trade_executed: "trade_executed",
    hypothesis_generated: "hypothesis_generated"
  }

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  @type state :: %{
          subscriptions: MapSet.t(),
          remote_ip: String.t() | nil,
          connected_at: DateTime.t(),
          message_count: non_neg_integer()
        }

  # ---------------------------------------------------------------------------
  # WebSock callbacks
  # ---------------------------------------------------------------------------

  @impl WebSock
  def init(opts) do
    remote_ip = Keyword.get(opts, :remote_ip, "unknown")

    # Subscribe to all EventBus topics so we can filter per-connection
    Enum.each(Map.keys(@topic_map), fn eb_topic ->
      EventBus.subscribe(eb_topic, self())
    end)

    # Schedule ping
    Process.send_after(self(), :send_ping, @ping_interval_ms)

    state = %{
      subscriptions: MapSet.new(),
      remote_ip: remote_ip,
      connected_at: DateTime.utc_now(),
      message_count: 0
    }

    Logger.info("[WebSocketHandler] Client connected from #{remote_ip}")

    {:ok, state}
  end

  @impl WebSock
  def handle_in({data, opcode: :text}, state) do
    state = %{state | message_count: state.message_count + 1}

    case Jason.decode(data) do
      {:ok, msg} ->
        handle_message(msg, state)

      {:error, _} ->
        reply = error_frame("invalid JSON")
        {:reply, :ok, {:text, reply}, state}
    end
  end

  @impl WebSock
  def handle_in({_data, opcode: :binary}, state) do
    reply = error_frame("binary frames not supported")
    {:reply, :ok, {:text, reply}, state}
  end

  @impl WebSock
  def handle_in({_data, opcode: :ping}, state) do
    {:reply, :ok, {:pong, ""}, state}
  end

  @impl WebSock
  def handle_in({_data, opcode: :pong}, state) do
    {:ok, state}
  end

  @impl WebSock
  def handle_info(:send_ping, state) do
    Process.send_after(self(), :send_ping, @ping_interval_ms)
    frame = Jason.encode!(%{action: "ping", ts: unix_ts()})
    {:push, {:text, frame}, state}
  end

  @impl WebSock
  def handle_info({:event, eb_topic, event}, state) do
    case Map.get(@topic_map, eb_topic) do
      nil ->
        {:ok, state}

      ws_topic ->
        if MapSet.member?(state.subscriptions, ws_topic) do
          frame = build_event_frame(ws_topic, event)
          {:push, {:text, frame}, state}
        else
          {:ok, state}
        end
    end
  end

  @impl WebSock
  def handle_info(_msg, state), do: {:ok, state}

  @impl WebSock
  def terminate(reason, state) do
    Logger.info(
      "[WebSocketHandler] Client #{state.remote_ip} disconnected: #{inspect(reason)}, " <>
        "msgs=#{state.message_count}"
    )

    :ok
  end

  # ---------------------------------------------------------------------------
  # Message dispatching
  # ---------------------------------------------------------------------------

  @doc """
  Handle a decoded JSON message from the client.
  Returns {:ok, state} | {:reply, :ok, frame, state}.
  """
  @spec handle_message(map(), state()) ::
          {:ok, state()}
          | {:reply, :ok, {:text, binary()}, state()}
          | {:push, {:text, binary()}, state()}
  def handle_message(%{"action" => "subscribe", "topics" => topics}, state)
      when is_list(topics) do
    valid = Enum.filter(topics, &(&1 in @valid_topics))
    invalid = topics -- valid

    if invalid != [] do
      Logger.debug("[WebSocketHandler] Client subscribed to unknown topics: #{inspect(invalid)}")
    end

    updated = Enum.reduce(valid, state.subscriptions, &MapSet.put(&2, &1))
    state = %{state | subscriptions: updated}

    reply =
      Jason.encode!(%{
        action: "subscribed",
        topics: MapSet.to_list(updated),
        ts: unix_ts()
      })

    {:reply, :ok, {:text, reply}, state}
  end

  def handle_message(%{"action" => "unsubscribe", "topics" => topics}, state)
      when is_list(topics) do
    updated = Enum.reduce(topics, state.subscriptions, &MapSet.delete(&2, &1))
    state = %{state | subscriptions: updated}

    reply =
      Jason.encode!(%{
        action: "unsubscribed",
        topics: topics,
        remaining: MapSet.to_list(updated),
        ts: unix_ts()
      })

    {:reply, :ok, {:text, reply}, state}
  end

  def handle_message(%{"action" => "ping"}, state) do
    reply = Jason.encode!(%{action: "pong", ts: unix_ts()})
    {:reply, :ok, {:text, reply}, state}
  end

  def handle_message(%{"action" => "list_topics"}, state) do
    reply =
      Jason.encode!(%{
        action: "topics",
        available: @valid_topics,
        subscribed: MapSet.to_list(state.subscriptions),
        ts: unix_ts()
      })

    {:reply, :ok, {:text, reply}, state}
  end

  def handle_message(%{"action" => "status"}, state) do
    reply =
      Jason.encode!(%{
        action: "status",
        subscriptions: MapSet.to_list(state.subscriptions),
        message_count: state.message_count,
        connected_since: DateTime.to_iso8601(state.connected_at),
        ts: unix_ts()
      })

    {:reply, :ok, {:text, reply}, state}
  end

  def handle_message(%{"action" => action}, state) do
    reply = error_frame("unknown action: #{action}")
    {:reply, :ok, {:text, reply}, state}
  end

  def handle_message(_msg, state) do
    reply = error_frame("missing or invalid action field")
    {:reply, :ok, {:text, reply}, state}
  end

  @doc """
  Called by an EventBus subscriber process to forward an event to all
  WebSocket connections that have subscribed to the topic.

  In practice, each WebSocket process handles its own EventBus subscription
  and filters by its own subscription set. This function is available for
  explicit fan-out if a central dispatcher is preferred.
  """
  @spec handle_event(map()) :: :ok
  def handle_event(%{topic: topic} = event) do
    ws_topic =
      topic
      |> Atom.to_string()
      |> then(&Map.get(@topic_map, topic, &1))

    Registry.dispatch(SrfmCoordination.WebSocketRegistry, ws_topic, fn entries ->
      frame = build_event_frame(ws_topic, event)

      Enum.each(entries, fn {pid, _} ->
        send(pid, {:push_frame, {:text, frame}})
      end)
    end)

    :ok
  end

  # ---------------------------------------------------------------------------
  # Upgrade helper -- call from Plug router
  # ---------------------------------------------------------------------------

  @doc """
  Upgrades an incoming Plug.Conn to a WebSocket connection.
  Call this from your Plug router for the /ws/events path.

  Example usage in router:
    get "/ws/events" do
      SrfmCoordination.WebSocketHandler.upgrade(conn)
    end
  """
  @spec upgrade(Plug.Conn.t()) :: Plug.Conn.t()
  def upgrade(conn) do
    remote_ip =
      conn.remote_ip
      |> Tuple.to_list()
      |> Enum.join(".")

    WebSockAdapter.upgrade(conn, __MODULE__, [remote_ip: remote_ip], timeout: 60_000)
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp build_event_frame(topic, event) do
    payload = sanitize_event(event)

    Jason.encode!(%{
      topic: topic,
      payload: payload,
      ts: unix_ts()
    })
  end

  defp error_frame(message) do
    Jason.encode!(%{action: "error", message: message, ts: unix_ts()})
  end

  defp unix_ts do
    DateTime.utc_now() |> DateTime.to_unix(:millisecond)
  end

  # Remove non-serializable fields before sending over the wire
  defp sanitize_event(event) when is_map(event) do
    event
    |> Map.drop([:__struct__])
    |> Enum.reduce(%{}, fn
      {k, %DateTime{} = v}, acc -> Map.put(acc, to_string(k), DateTime.to_iso8601(v))
      {k, v}, acc when is_atom(k) -> Map.put(acc, to_string(k), v)
      {k, v}, acc -> Map.put(acc, k, v)
    end)
  end

  defp sanitize_event(event), do: event
end
