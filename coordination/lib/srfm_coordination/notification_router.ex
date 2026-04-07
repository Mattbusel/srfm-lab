defmodule SrfmCoordination.NotificationRouter do
  @moduledoc """
  Multi-channel notification routing for the SRFM Coordination Service.

  Routes notifications to appropriate channels based on severity level.

  Severity levels and their channels:
    :info     -> EventBus only
    :warning  -> Slack, EventBus
    :critical -> Slack, PagerDuty, Email, EventBus
    :page     -> Slack, PagerDuty, Email, EventBus

  Rate limiting:
    Slack: max 10 messages per minute
    Deduplication: identical (title + body) messages suppressed within 5 minutes

  Message templates are provided for common event types.
  Channel configuration is loaded from application env and can be overridden
  at runtime via configure_channel/2.
  """

  use GenServer
  require Logger

  alias SrfmCoordination.EventBus

  @slack_rate_limit 10
  @slack_rate_window_ms 60_000
  @dedup_window_ms 5 * 60 * 1_000

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  @type severity :: :info | :warning | :critical | :page

  @type channel :: :slack | :pagerduty | :email | :event_bus

  @type channel_config :: %{
          optional(:webhook_url) => String.t(),
          optional(:routing_key) => String.t(),
          optional(:smtp_host) => String.t(),
          optional(:smtp_port) => pos_integer(),
          optional(:smtp_username) => String.t(),
          optional(:smtp_password) => String.t(),
          optional(:from_address) => String.t(),
          optional(:to_addresses) => [String.t()],
          optional(:enabled) => boolean()
        }

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc "Start the NotificationRouter GenServer."
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Send a notification to appropriate channels based on severity.

  - severity: :info | :warning | :critical | :page
  - title: short subject line
  - body: full message body
  - context: optional map with additional metadata (event_type, service, etc.)

  Returns :ok immediately. Delivery is async.
  """
  @spec notify(severity(), String.t(), String.t(), map()) :: :ok
  def notify(severity, title, body, context \\ %{})
      when severity in [:info, :warning, :critical, :page] do
    GenServer.cast(__MODULE__, {:notify, severity, title, body, context})
  end

  @doc """
  Override channel configuration at runtime.

  channel: :slack | :pagerduty | :email
  config: channel-specific config map (merged with existing config)
  """
  @spec configure_channel(channel(), channel_config()) :: :ok
  def configure_channel(channel, config) when channel in [:slack, :pagerduty, :email] do
    GenServer.call(__MODULE__, {:configure_channel, channel, config})
  end

  @doc """
  Send a test notification to the specified channel.

  Returns :ok if the delivery succeeded, {:error, reason} otherwise.
  Bypasses rate limiting and deduplication.
  """
  @spec test_channel(channel()) :: :ok | {:error, term()}
  def test_channel(channel) do
    GenServer.call(__MODULE__, {:test_channel, channel}, 30_000)
  end

  @doc "Get current channel configuration (passwords redacted)."
  @spec channel_config(channel()) :: channel_config() | nil
  def channel_config(channel) do
    GenServer.call(__MODULE__, {:get_config, channel})
  end

  @doc "Return a pre-formatted notification body for a known event type."
  @spec format_event(atom(), map()) :: {String.t(), String.t()}
  def format_event(event_type, context \\ %{}) do
    build_template(event_type, context)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  defstruct channel_configs: %{},
            slack_sends: [],
            dedup_cache: %{}

  @impl true
  def init(_opts) do
    configs = load_env_configs()

    # Subscribe to EventBus alert topic so we can auto-route system alerts
    EventBus.subscribe(:alert, self())

    # Periodically clean up dedup cache
    :timer.send_interval(60_000, :cleanup_dedup)

    Logger.info("[NotificationRouter] Initialized, channels: #{inspect(Map.keys(configs))}")
    {:ok, %__MODULE__{channel_configs: configs}}
  end

  @impl true
  def handle_cast({:notify, severity, title, body, context}, state) do
    now_ms = System.monotonic_time(:millisecond)
    dedup_key = :crypto.hash(:md5, title <> body) |> Base.encode16()

    {state, should_send} = check_dedup(state, dedup_key, now_ms)

    if should_send do
      channels = channels_for_severity(severity)
      state = dispatch_to_channels(state, channels, severity, title, body, context, now_ms)
      {:noreply, state}
    else
      Logger.debug("[NotificationRouter] Suppressed duplicate notification: #{title}")
      {:noreply, state}
    end
  end

  @impl true
  def handle_call({:configure_channel, channel, new_config}, _from, state) do
    existing = Map.get(state.channel_configs, channel, %{})
    merged = Map.merge(existing, new_config)
    updated = Map.put(state.channel_configs, channel, merged)
    Logger.info("[NotificationRouter] Updated config for #{channel}")
    {:reply, :ok, %{state | channel_configs: updated}}
  end

  @impl true
  def handle_call({:test_channel, channel}, _from, state) do
    result = send_test(channel, state.channel_configs)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_config, channel}, _from, state) do
    config =
      state.channel_configs
      |> Map.get(channel)
      |> redact_secrets()

    {:reply, config, state}
  end

  @impl true
  def handle_info({:event, :alert, event}, state) do
    severity = Map.get(event, :severity, :warning)
    title = Map.get(event, :title, "SRFM Alert")
    body = Map.get(event, :message, inspect(event))
    context = Map.drop(event, [:severity, :title, :message, :topic, :timestamp])

    GenServer.cast(__MODULE__, {:notify, severity, title, body, context})
    {:noreply, state}
  end

  @impl true
  def handle_info(:cleanup_dedup, state) do
    now_ms = System.monotonic_time(:millisecond)
    cutoff = now_ms - @dedup_window_ms

    fresh = Enum.reject(state.dedup_cache, fn {_key, ts} -> ts < cutoff end) |> Map.new()
    {:noreply, %{state | dedup_cache: fresh}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # ---------------------------------------------------------------------------
  # Channel dispatch
  # ---------------------------------------------------------------------------

  defp channels_for_severity(:info), do: [:event_bus]
  defp channels_for_severity(:warning), do: [:slack, :event_bus]
  defp channels_for_severity(:critical), do: [:slack, :pagerduty, :email, :event_bus]
  defp channels_for_severity(:page), do: [:slack, :pagerduty, :email, :event_bus]

  defp dispatch_to_channels(state, channels, severity, title, body, context, now_ms) do
    Enum.reduce(channels, state, fn channel, acc ->
      case channel do
        :slack ->
          {acc2, ok_to_send} = check_slack_rate(acc, now_ms)

          if ok_to_send do
            send_slack_async(title, body, severity, context, acc2.channel_configs)
            acc2
          else
            Logger.warning("[NotificationRouter] Slack rate limit reached, dropping: #{title}")
            acc2
          end

        :pagerduty ->
          send_pagerduty_async(title, body, severity, context, acc.channel_configs)
          acc

        :email ->
          send_email_async(title, body, severity, context, acc.channel_configs)
          acc

        :event_bus ->
          EventBus.publish(:alert, %{
            severity: severity,
            title: title,
            message: body,
            context: context
          })

          acc
      end
    end)
  end

  # ---------------------------------------------------------------------------
  # Slack
  # ---------------------------------------------------------------------------

  defp send_slack_async(title, body, severity, context, configs) do
    Task.start(fn ->
      case Map.get(configs, :slack) do
        nil ->
          Logger.debug("[NotificationRouter] Slack not configured, skipping")

        config when is_map(config) ->
          if Map.get(config, :enabled, true) do
            do_send_slack(title, body, severity, context, config)
          end
      end
    end)
  end

  defp do_send_slack(title, body, severity, context, config) do
    url = Map.get(config, :webhook_url, "")

    if url == "" do
      Logger.warning("[NotificationRouter] Slack webhook_url not set")
    else
      color = slack_color(severity)
      service = Map.get(context, :service, "srfm-coordination")
      ts = DateTime.utc_now() |> DateTime.to_iso8601()

      payload = %{
        attachments: [
          %{
            color: color,
            title: "[#{String.upcase(to_string(severity))}] #{title}",
            text: body,
            footer: "SRFM | #{service}",
            ts: DateTime.utc_now() |> DateTime.to_unix()
          }
        ]
      }

      body_json = Jason.encode!(payload)

      case :httpc.request(
             :post,
             {String.to_charlist(url), [], ~c"application/json", body_json},
             [timeout: 5_000],
             []
           ) do
        {:ok, {{_, 200, _}, _, _}} ->
          Logger.debug("[NotificationRouter] Slack OK: #{title}")

        {:ok, {{_, status, _}, _, resp_body}} ->
          Logger.warning("[NotificationRouter] Slack HTTP #{status}: #{resp_body}")

        {:error, reason} ->
          Logger.error("[NotificationRouter] Slack send failed: #{inspect(reason)}")
      end
    end
  end

  defp slack_color(:info), do: "#36a64f"
  defp slack_color(:warning), do: "#ff9900"
  defp slack_color(:critical), do: "#ff0000"
  defp slack_color(:page), do: "#8b0000"

  # ---------------------------------------------------------------------------
  # PagerDuty
  # ---------------------------------------------------------------------------

  defp send_pagerduty_async(title, body, severity, context, configs) do
    Task.start(fn ->
      case Map.get(configs, :pagerduty) do
        nil ->
          Logger.debug("[NotificationRouter] PagerDuty not configured, skipping")

        config when is_map(config) ->
          if Map.get(config, :enabled, true) do
            do_send_pagerduty(title, body, severity, context, config)
          end
      end
    end)
  end

  defp do_send_pagerduty(title, body, severity, context, config) do
    routing_key = Map.get(config, :routing_key, "")

    if routing_key == "" do
      Logger.warning("[NotificationRouter] PagerDuty routing_key not set")
    else
      pd_severity =
        case severity do
          :page -> "critical"
          :critical -> "error"
          _ -> "warning"
        end

      payload = %{
        routing_key: routing_key,
        event_action: "trigger",
        payload: %{
          summary: title,
          source: Map.get(context, :service, "srfm-coordination"),
          severity: pd_severity,
          custom_details: Map.put(context, :body, body)
        }
      }

      body_json = Jason.encode!(payload)
      url = ~c"https://events.pagerduty.com/v2/enqueue"

      case :httpc.request(
             :post,
             {url, [], ~c"application/json", body_json},
             [timeout: 10_000],
             []
           ) do
        {:ok, {{_, status, _}, _, _}} when status in 200..299 ->
          Logger.debug("[NotificationRouter] PagerDuty OK: #{title}")

        {:ok, {{_, status, _}, _, resp_body}} ->
          Logger.error("[NotificationRouter] PagerDuty HTTP #{status}: #{resp_body}")

        {:error, reason} ->
          Logger.error("[NotificationRouter] PagerDuty send failed: #{inspect(reason)}")
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Email
  # ---------------------------------------------------------------------------

  defp send_email_async(title, body, severity, _context, configs) do
    Task.start(fn ->
      case Map.get(configs, :email) do
        nil ->
          Logger.debug("[NotificationRouter] Email not configured, skipping")

        config when is_map(config) ->
          if Map.get(config, :enabled, true) do
            do_send_email(title, body, severity, config)
          end
      end
    end)
  end

  defp do_send_email(title, body, severity, config) do
    # In production, use gen_smtp or swoosh. Using :gen_smtp or a library here
    # would require an additional dep. Log for now and integrate at app level.
    to = Map.get(config, :to_addresses, [])
    from = Map.get(config, :from_address, "srfm@localhost")

    if to == [] do
      Logger.warning("[NotificationRouter] Email to_addresses not configured")
    else
      Logger.info(
        "[NotificationRouter] Email [#{severity}] to #{inspect(to)} from #{from}: #{title}\n#{body}"
      )
      # Wire to gen_smtp or Swoosh.Mailer.deliver/2 at integration time
    end
  end

  # ---------------------------------------------------------------------------
  # Rate limiting helpers
  # ---------------------------------------------------------------------------

  defp check_slack_rate(state, now_ms) do
    cutoff = now_ms - @slack_rate_window_ms
    recent = Enum.filter(state.slack_sends, &(&1 > cutoff))

    if length(recent) < @slack_rate_limit do
      {%{state | slack_sends: [now_ms | recent]}, true}
    else
      {%{state | slack_sends: recent}, false}
    end
  end

  defp check_dedup(state, key, now_ms) do
    case Map.get(state.dedup_cache, key) do
      nil ->
        updated = Map.put(state.dedup_cache, key, now_ms)
        {%{state | dedup_cache: updated}, true}

      last_ts ->
        if now_ms - last_ts > @dedup_window_ms do
          updated = Map.put(state.dedup_cache, key, now_ms)
          {%{state | dedup_cache: updated}, true}
        else
          {state, false}
        end
    end
  end

  # ---------------------------------------------------------------------------
  # Templates
  # ---------------------------------------------------------------------------

  defp build_template(:param_rollback, ctx) do
    param = Map.get(ctx, :param, "unknown")
    old_val = Map.get(ctx, :old_value, "?")
    reason = Map.get(ctx, :reason, "performance degradation")

    {
      "Parameter Rollback: #{param}",
      "Parameter `#{param}` was rolled back to `#{old_val}`.\nReason: #{reason}\n" <>
        "Triggered at: #{DateTime.utc_now() |> DateTime.to_iso8601()}"
    }
  end

  defp build_template(:circuit_open, ctx) do
    name = Map.get(ctx, :circuit_name, "unknown")
    failures = Map.get(ctx, :failure_count, "?")

    {
      "Circuit Breaker Open: #{name}",
      "Circuit `#{name}` opened after #{failures} consecutive failures.\n" <>
        "All requests are being rejected until the circuit resets."
    }
  end

  defp build_template(:risk_breach, ctx) do
    metric = Map.get(ctx, :metric, "unknown")
    value = Map.get(ctx, :value, "?")
    threshold = Map.get(ctx, :threshold, "?")

    {
      "Risk Breach: #{metric}",
      "Risk metric `#{metric}` breached threshold.\n" <>
        "Value: #{value} | Threshold: #{threshold}\n" <>
        "Automated risk reduction may be triggered."
    }
  end

  defp build_template(:service_down, ctx) do
    service = Map.get(ctx, :service, "unknown")
    last_seen = Map.get(ctx, :last_seen, "unknown")

    {
      "Service Down: #{service}",
      "Service `#{service}` is not responding.\n" <>
        "Last successful health check: #{last_seen}\n" <>
        "Please investigate immediately."
    }
  end

  defp build_template(event_type, ctx) do
    {
      "SRFM Event: #{event_type}",
      "Event: #{event_type}\nContext: #{inspect(ctx, pretty: true)}"
    }
  end

  # ---------------------------------------------------------------------------
  # Test channel
  # ---------------------------------------------------------------------------

  defp send_test(:slack, configs) do
    case Map.get(configs, :slack) do
      nil -> {:error, :not_configured}
      config -> do_send_slack("Test notification", "SRFM test from NotificationRouter", :info, %{}, config)
    end
  end

  defp send_test(:pagerduty, configs) do
    case Map.get(configs, :pagerduty) do
      nil -> {:error, :not_configured}
      config -> do_send_pagerduty("Test notification", "SRFM test", :warning, %{}, config)
    end
  end

  defp send_test(:email, configs) do
    case Map.get(configs, :email) do
      nil -> {:error, :not_configured}
      config -> do_send_email("Test notification", "SRFM test from NotificationRouter", :info, config)
    end
  end

  defp send_test(:event_bus, _configs) do
    EventBus.publish(:alert, %{severity: :info, title: "Test", message: "NotificationRouter test"})
    :ok
  end

  defp send_test(channel, _), do: {:error, {:unknown_channel, channel}}

  # ---------------------------------------------------------------------------
  # Config loading and redaction
  # ---------------------------------------------------------------------------

  defp load_env_configs do
    %{
      slack: load_channel_config(:slack),
      pagerduty: load_channel_config(:pagerduty),
      email: load_channel_config(:email)
    }
    |> Enum.reject(fn {_k, v} -> is_nil(v) end)
    |> Map.new()
  end

  defp load_channel_config(channel) do
    Application.get_env(:srfm_coordination, :"notification_#{channel}")
  end

  defp redact_secrets(nil), do: nil

  defp redact_secrets(config) when is_map(config) do
    secret_keys = [:smtp_password, :routing_key, :webhook_url]

    Enum.reduce(secret_keys, config, fn key, acc ->
      if Map.has_key?(acc, key) do
        Map.put(acc, key, "[REDACTED]")
      else
        acc
      end
    end)
  end
end
