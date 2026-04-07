defmodule SrfmCoordination.Alerting do
  @moduledoc """
  Central alerting GenServer. Subscribes to EventBus topics and routes
  alerts to Slack and PagerDuty based on severity.

  Subscribed EventBus topics:
    :alert         -- catches service_unhealthy, performance_degraded,
                      rollback_triggered, genome_rejected, genome_applied,
                      circuit_open, metric_anomaly, drain_alert
    :service_health -- catches service health transitions

  Severity mapping:
    INFO     -- genome_applied, drain_state_changed
    WARNING  -- service_unhealthy, circuit_open, metric_anomaly, genome_rejected
    CRITICAL -- rollback_triggered, performance_degraded, drain_alert (force)

  Deduplication: 30-minute suppress window keyed on {event_type, service}.
  PagerDuty: fired only for CRITICAL alerts.

  Audit trail: ETS ring buffer of last 500 alerts (:srfm_alert_audit).

  Environment variables consumed:
    SLACK_WEBHOOK_URL   -- Slack incoming webhook URL
    PAGERDUTY_API_KEY   -- PagerDuty Events API v2 routing key
    PAGERDUTY_SERVICE_ID -- source label
  """

  use GenServer
  require Logger

  @suppress_window_seconds 1_800   # 30 minutes
  @audit_ring_size 500
  @http_timeout_ms 8_000

  @audit_table :srfm_alert_audit
  @suppress_table :srfm_alert_suppress

  # Alert severity atoms
  @severity_levels [:info, :warning, :critical]

  defstruct []

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Manually send a Slack alert. severity is :info | :warning | :critical."
  @spec send_slack(String.t(), atom()) :: :ok | {:error, term()}
  def send_slack(message, severity \\ :info) when severity in @severity_levels do
    GenServer.call(__MODULE__, {:send_slack, message, severity})
  end

  @doc "Manually send a PagerDuty alert. Only fires for :critical."
  @spec send_pagerduty(String.t()) :: :ok | {:error, term()}
  def send_pagerduty(message) do
    GenServer.call(__MODULE__, {:send_pagerduty, message})
  end

  @doc "Return the last `n` entries from the audit ring buffer."
  @spec audit_log(pos_integer()) :: [map()]
  def audit_log(n \\ 50) when is_integer(n) and n > 0 do
    case :ets.lookup(@audit_table, :ring) do
      [{:ring, entries}] -> Enum.take(entries, n)
      [] -> []
    end
  end

  @doc "Return current suppression table contents."
  @spec suppressed_keys() :: [map()]
  def suppressed_keys do
    :ets.tab2list(@suppress_table)
    |> Enum.map(fn {{type, service}, suppress_until} ->
      %{type: type, service: service, suppress_until: suppress_until}
    end)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@audit_table, [:named_table, :set, :public, read_concurrency: true])
    :ets.new(@suppress_table, [:named_table, :set, :public, read_concurrency: true])
    :ets.insert(@audit_table, {:ring, []})

    # Subscribe to all relevant EventBus topics
    SrfmCoordination.EventBus.subscribe(:alert, self())
    SrfmCoordination.EventBus.subscribe(:service_health, self())

    Logger.info("[Alerting] Initialized -- subscribed to :alert and :service_health")
    {:ok, %__MODULE__{}}
  end

  # Handle events delivered by EventBus
  @impl true
  def handle_info({:event, topic, event}, state) do
    handle_event(topic, event)
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def handle_call({:send_slack, message, severity}, _from, state) do
    result = do_send_slack(message, severity)
    append_audit(%{channel: :slack, message: message, severity: severity, result: result})
    {:reply, result, state}
  end

  @impl true
  def handle_call({:send_pagerduty, message}, _from, state) do
    result = do_send_pagerduty(message)
    append_audit(%{channel: :pagerduty, message: message, severity: :critical, result: result})
    {:reply, result, state}
  end

  @impl true
  def terminate(reason, _state) do
    Logger.info("[Alerting] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Event routing
  # ---------------------------------------------------------------------------

  defp handle_event(:alert, %{type: type} = event) do
    {severity, message} = classify_alert(type, event)
    service = Map.get(event, :service, :system)

    if should_send?(type, service) do
      Logger.info("[Alerting] Dispatching #{severity} alert: #{type} / #{service}")
      mark_suppressed(type, service)
      dispatch(severity, format_message(type, event, message))
    else
      Logger.debug("[Alerting] Suppressed #{type}/#{service} (within 30min window)")
    end
  end

  defp handle_event(:service_health, %{type: :service_health_changed, payload: payload} = _event) do
    %{name: service, new: new_status} = payload

    case new_status do
      :down ->
        if should_send?(:service_unhealthy, service) do
          mark_suppressed(:service_unhealthy, service)
          dispatch(:warning, "[WARNING] Service :#{service} is DOWN")
        end

      :degraded ->
        if should_send?(:service_degraded, service) do
          mark_suppressed(:service_degraded, service)
          dispatch(:warning, "[WARNING] Service :#{service} is DEGRADED")
        end

      _ ->
        :ok
    end
  end

  defp handle_event(_topic, _event), do: :ok

  # ---------------------------------------------------------------------------
  # Alert classification
  # ---------------------------------------------------------------------------

  defp classify_alert(:performance_degraded, event) do
    msg =
      "[CRITICAL] Performance degraded -- " <>
      "Sharpe_4h=#{event[:sharpe_4h]} drawdown=#{event[:drawdown_pct]}%"

    {:critical, msg}
  end

  defp classify_alert(:rollback_triggered, event) do
    {:critical, "[CRITICAL] Parameter rollback triggered -- reason: #{inspect(event[:reason])}"}
  end

  defp classify_alert(:drain_alert, %{severity: :critical} = event) do
    {:critical, "[CRITICAL] Drain alert for :#{event[:service]} -- #{event[:message]}"}
  end

  defp classify_alert(:circuit_open, event) do
    {:warning, "[WARNING] Circuit :#{event[:circuit]} opened"}
  end

  defp classify_alert(:service_unhealthy, event) do
    {:warning, "[WARNING] Service :#{event[:service]} is unhealthy"}
  end

  defp classify_alert(:metric_anomaly, event) do
    {:warning,
     "[WARNING] Metric anomaly -- :#{event[:service]}/#{event[:metric]} " <>
     "z=#{Float.round(event[:z_score] || 0.0, 2)}"}
  end

  defp classify_alert(:genome_rejected, event) do
    {:warning, "[WARNING] Genome rejected -- reason: #{event[:reason]}"}
  end

  defp classify_alert(:genome_applied, event) do
    {:info,
     "[INFO] Genome applied -- fitness=#{event[:fitness]} steps=#{event[:step_count]}"}
  end

  defp classify_alert(:drain_state_changed, event) do
    {:info, "[INFO] Drain :#{event[:service]} #{event[:from]} -> #{event[:to]}"}
  end

  defp classify_alert(type, event) do
    {:info, "[INFO] #{type} -- #{inspect(event)}"}
  end

  # ---------------------------------------------------------------------------
  # Deduplication
  # ---------------------------------------------------------------------------

  defp should_send?(type, service) do
    key = {type, service}
    now = now_unix()

    case :ets.lookup(@suppress_table, key) do
      [{^key, suppress_until}] -> now >= suppress_until
      [] -> true
    end
  end

  defp mark_suppressed(type, service) do
    key = {type, service}
    suppress_until = now_unix() + @suppress_window_seconds
    :ets.insert(@suppress_table, {key, suppress_until})
  end

  # ---------------------------------------------------------------------------
  # Dispatch
  # ---------------------------------------------------------------------------

  defp dispatch(:critical, message) do
    do_send_slack(message, :critical)
    do_send_pagerduty(message)
    append_audit(%{channel: :slack_and_pagerduty, message: message, severity: :critical})
  end

  defp dispatch(severity, message) do
    do_send_slack(message, severity)
    append_audit(%{channel: :slack, message: message, severity: severity})
  end

  defp format_message(_type, _event, pre_formatted), do: pre_formatted

  # ---------------------------------------------------------------------------
  # Slack
  # ---------------------------------------------------------------------------

  defp do_send_slack(message, severity) do
    webhook_url = System.get_env("SLACK_WEBHOOK_URL")

    if is_nil(webhook_url) or webhook_url == "" do
      Logger.debug("[Alerting] SLACK_WEBHOOK_URL not set -- Slack alert suppressed: #{message}")
      :ok
    else
      emoji = severity_emoji(severity)
      payload = Jason.encode!(%{text: "#{emoji} #{message}"})

      try do
        case HTTPoison.post(webhook_url, payload, [{"content-type", "application/json"}],
                           recv_timeout: @http_timeout_ms) do
          {:ok, %{status_code: code}} when code in 200..299 ->
            Logger.debug("[Alerting] Slack alert sent (#{severity})")
            :ok

          {:ok, %{status_code: code, body: body}} ->
            Logger.warning("[Alerting] Slack returned #{code}: #{body}")
            {:error, {:slack_error, code}}

          {:error, %HTTPoison.Error{reason: reason}} ->
            Logger.warning("[Alerting] Slack HTTP error: #{inspect(reason)}")
            {:error, reason}
        end
      catch
        kind, reason ->
          Logger.warning("[Alerting] Slack exception: #{inspect({kind, reason})}")
          {:error, {kind, reason}}
      end
    end
  end

  defp severity_emoji(:critical), do: ":rotating_light:"
  defp severity_emoji(:warning), do: ":warning:"
  defp severity_emoji(:info), do: ":information_source:"
  defp severity_emoji(_), do: ":bell:"

  # ---------------------------------------------------------------------------
  # PagerDuty
  # ---------------------------------------------------------------------------

  defp do_send_pagerduty(message) do
    routing_key = System.get_env("PAGERDUTY_API_KEY")
    service_id = System.get_env("PAGERDUTY_SERVICE_ID", "srfm-coordination")

    if is_nil(routing_key) or routing_key == "" do
      Logger.debug("[Alerting] PAGERDUTY_API_KEY not set -- PagerDuty alert suppressed")
      :ok
    else
      payload =
        Jason.encode!(%{
          routing_key: routing_key,
          event_action: "trigger",
          payload: %{
            summary: message,
            source: service_id,
            severity: "critical",
            timestamp: DateTime.to_iso8601(DateTime.utc_now())
          }
        })

      url = "https://events.pagerduty.com/v2/enqueue"

      try do
        case HTTPoison.post(url, payload, [{"content-type", "application/json"}],
                           recv_timeout: @http_timeout_ms) do
          {:ok, %{status_code: code}} when code in 200..202 ->
            Logger.info("[Alerting] PagerDuty alert triggered")
            :ok

          {:ok, %{status_code: code, body: body}} ->
            Logger.warning("[Alerting] PagerDuty returned #{code}: #{body}")
            {:error, {:pagerduty_error, code}}

          {:error, %HTTPoison.Error{reason: reason}} ->
            Logger.warning("[Alerting] PagerDuty HTTP error: #{inspect(reason)}")
            {:error, reason}
        end
      catch
        kind, reason ->
          Logger.warning("[Alerting] PagerDuty exception: #{inspect({kind, reason})}")
          {:error, {kind, reason}}
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Audit ring buffer
  # ---------------------------------------------------------------------------

  defp append_audit(entry) do
    full_entry = Map.merge(entry, %{at: DateTime.utc_now()})

    existing =
      case :ets.lookup(@audit_table, :ring) do
        [{:ring, list}] -> list
        [] -> []
      end

    trimmed = Enum.take([full_entry | existing], @audit_ring_size)
    :ets.insert(@audit_table, {:ring, trimmed})
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp now_unix, do: System.os_time(:second)
end
