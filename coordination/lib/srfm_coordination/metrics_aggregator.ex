defmodule SrfmCoordination.MetricsAggregator do
  @moduledoc """
  Aggregates metrics from all registered services and exposes a
  Prometheus-compatible /metrics endpoint.

  Every 30 seconds, MetricsAggregator polls each registered service at
  GET /metrics/json. Responses must be JSON objects mapping metric name
  strings to numeric values:
    {"sharpe_4h": 1.23, "drawdown_pct": 0.04, "position_count": 5}

  Collected data is stored in ETS table :metrics_store with a timestamp.
  The HTTP router serves /metrics which calls to_prometheus_text/0.

  Key metrics tracked per service:
    sharpe_4h           -- rolling 4-hour Sharpe ratio
    drawdown_pct        -- current drawdown as a decimal fraction
    position_count      -- number of open positions
    open_orders         -- number of outstanding orders
    circuit_breaker_state -- 0=closed, 1=half_open, 2=open (numeric encoding)
    genome_generation   -- current IAE generation number
  """

  use GenServer
  require Logger

  @poll_interval_ms 30_000
  @http_timeout_ms  5_000
  @table            :metrics_store
  @staleness_ms     120_000   -- metrics older than 2 minutes are considered stale

  -- MetricPoint represents one scraped data point.
  defmodule MetricPoint do
    @moduledoc "One scraped metric observation."
    defstruct [:service, :name, :value, :ts]

    @type t :: %__MODULE__{
      service: atom(),
      name:    String.t(),
      value:   float(),
      ts:      integer()
    }
  end

  -- Prometheus help strings for well-known metric names.
  @metric_help %{
    "sharpe_4h"             => {"Rolling 4h Sharpe ratio", "gauge"},
    "sharpe_rolling_4h"     => {"Rolling 4h Sharpe ratio", "gauge"},
    "drawdown_pct"          => {"Current drawdown as decimal fraction", "gauge"},
    "position_count"        => {"Number of open positions", "gauge"},
    "open_orders"           => {"Number of outstanding orders", "gauge"},
    "circuit_breaker_state" => {"Circuit breaker state: 0=closed 1=half_open 2=open", "gauge"},
    "genome_generation"     => {"Current IAE genome generation number", "counter"},
    "pnl_today"             => {"Running PnL for current session", "gauge"},
    "trades_today"          => {"Number of trades in current session", "counter"},
    "latency_ms"            => {"Recent execution latency in milliseconds", "gauge"}
  }

  defstruct [
    poll_ref:   nil,
    poll_count: 0,
    last_poll:  nil
  ]

  -- ---------------------------------------------------------------------------
  -- Public API
  -- ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Return the latest value of `name` for `service`.
  Returns nil if the metric is not found or is stale.
  """
  @spec get_metric(atom(), String.t()) :: float() | nil
  def get_metric(service, name) when is_atom(service) and is_binary(name) do
    cutoff = System.system_time(:millisecond) - @staleness_ms

    case :ets.lookup(@table, {service, name}) do
      [{{^service, ^name}, value, ts}] when ts >= cutoff -> value
      _ -> nil
    end
  end

  @doc "Return all current MetricPoints (non-stale only)."
  @spec get_all_metrics() :: [MetricPoint.t()]
  def get_all_metrics do
    cutoff = System.system_time(:millisecond) - @staleness_ms

    :ets.tab2list(@table)
    |> Enum.filter(fn {{_svc, _name}, _val, ts} -> ts >= cutoff end)
    |> Enum.map(fn {{service, name}, value, ts} ->
      %MetricPoint{service: service, name: name, value: value, ts: ts}
    end)
  end

  @doc """
  Serialize all current metrics into Prometheus text exposition format.
  Returns a binary string ready to serve as the /metrics response body.
  """
  @spec to_prometheus_text() :: String.t()
  def to_prometheus_text do
    GenServer.call(__MODULE__, :to_prometheus_text, 10_000)
  end

  @doc "Force an immediate poll cycle (useful for testing)."
  @spec poll_now() :: :ok
  def poll_now do
    GenServer.cast(__MODULE__, :poll_now)
  end

  @doc "Return aggregator status."
  @spec status() :: map()
  def status do
    GenServer.call(__MODULE__, :status)
  end

  -- ---------------------------------------------------------------------------
  -- GenServer callbacks
  -- ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])
    Logger.info("[MetricsAggregator] Initialized -- polling every #{@poll_interval_ms}ms")
    ref = schedule_poll()
    {:ok, %__MODULE__{poll_ref: ref}}
  end

  @impl true
  def handle_cast(:poll_now, state) do
    do_poll_all()
    {:noreply, %{state | last_poll: System.system_time(:millisecond)}}
  end

  @impl true
  def handle_call(:to_prometheus_text, _from, state) do
    text = build_prometheus_text()
    {:reply, text, state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    all = get_all_metrics()

    info = %{
      poll_count:    state.poll_count,
      last_poll:     state.last_poll,
      metric_count:  length(all),
      service_count: all |> Enum.map(& &1.service) |> Enum.uniq() |> length()
    }

    {:reply, info, state}
  end

  @impl true
  def handle_info(:poll, state) do
    do_poll_all()
    ref = schedule_poll()
    {:noreply, %{state | poll_ref: ref, poll_count: state.poll_count + 1, last_poll: System.system_time(:millisecond)}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[MetricsAggregator] Terminating: #{inspect(reason)}")
    :ok
  end

  -- ---------------------------------------------------------------------------
  -- Private -- polling
  -- ---------------------------------------------------------------------------

  defp do_poll_all do
    services = SrfmCoordination.ServiceRegistry.list_all()

    if services == [] do
      Logger.debug("[MetricsAggregator] No services registered, skipping poll")
    else
      Logger.debug("[MetricsAggregator] Polling metrics from #{length(services)} services")

      Task.async_stream(
        services,
        fn svc -> scrape_service(svc) end,
        max_concurrency: 10,
        timeout: @http_timeout_ms + 2_000,
        on_timeout: :kill_task
      )
      |> Stream.run()
    end

    -- Also ingest coordination-internal metrics
    ingest_internal_metrics()
  end

  defp scrape_service(svc) do
    url = "http://localhost:#{svc.port}/metrics/json"

    try do
      case HTTPoison.get(url, [], recv_timeout: @http_timeout_ms, timeout: @http_timeout_ms) do
        {:ok, %{status_code: code, body: body}} when code in 200..299 ->
          case Jason.decode(body) do
            {:ok, metrics_map} when is_map(metrics_map) ->
              now = System.system_time(:millisecond)
              ingest_service_metrics(svc.name, metrics_map, now)

            {:error, err} ->
              Logger.debug("[MetricsAggregator] JSON decode failed for #{svc.name}: #{inspect(err)}")
          end

        {:ok, %{status_code: code}} ->
          Logger.debug("[MetricsAggregator] #{svc.name} /metrics/json returned #{code}")

        {:error, %{reason: :econnrefused}} ->
          Logger.debug("[MetricsAggregator] #{svc.name} not reachable")

        {:error, reason} ->
          Logger.debug("[MetricsAggregator] #{svc.name} error: #{inspect(reason)}")
      end
    catch
      kind, reason ->
        Logger.debug("[MetricsAggregator] #{svc.name} exception (#{kind}): #{inspect(reason)}")
    end
  end

  defp ingest_service_metrics(service_name, metrics_map, ts) do
    Enum.each(metrics_map, fn {name, value} ->
      numeric =
        cond do
          is_float(value)   -> value
          is_integer(value) -> value / 1.0
          is_boolean(value) -> if value, do: 1.0, else: 0.0
          true              -> nil
        end

      if numeric != nil do
        :ets.insert(@table, {{service_name, to_string(name)}, numeric, ts})
      end
    end)
  end

  defp ingest_internal_metrics do
    now = System.system_time(:millisecond)

    -- Circuit breaker states
    try do
      SrfmCoordination.CircuitBreaker.all_statuses()
      |> Enum.each(fn %{name: name, state: cb_state} ->
        numeric =
          case cb_state do
            :closed    -> 0.0
            :half_open -> 1.0
            :open      -> 2.0
            _ -> 0.0
          end
        :ets.insert(@table, {{:coordination, "circuit_breaker.#{name}"}, numeric, now})
      end)
    catch
      _, _ -> :ok
    end

    -- Genome bridge status
    try do
      case Process.whereis(SrfmCoordination.GenomeBridge) do
        nil -> :ok
        _ ->
          status = SrfmCoordination.GenomeBridge.status()

          if status.last_generation != nil do
            :ets.insert(@table, {{:coordination, "genome_generation"}, status.last_generation / 1.0, now})
          end

          if status.last_best_fitness != nil do
            :ets.insert(@table, {{:coordination, "genome_best_fitness"}, status.last_best_fitness, now})
          end
      end
    catch
      _, _ -> :ok
    end

    -- Session state
    try do
      case Process.whereis(SrfmCoordination.SessionManager) do
        nil -> :ok
        _ ->
          session = SrfmCoordination.SessionManager.get_session()
          status_numeric = case session.status do
            :pre_market  -> 0.0
            :market_open -> 1.0
            :market_close -> 2.0
            :after_hours -> 3.0
          end
          :ets.insert(@table, {{:coordination, "session_status"}, status_numeric, now})
          :ets.insert(@table, {{:coordination, "session_trades_today"}, session.trades_today / 1.0, now})
          :ets.insert(@table, {{:coordination, "session_pnl_today"}, session.pnl_today, now})
      end
    catch
      _, _ -> :ok
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- Prometheus text format serializer
  -- ---------------------------------------------------------------------------

  defp build_prometheus_text do
    points =
      get_all_metrics()
      |> Enum.sort_by(fn p -> {p.name, p.service} end)

    -- Group by metric name and serialize
    points
    |> Enum.group_by(fn p ->
      -- Normalize metric name: use name prefixed with srfm_
      "srfm_#{String.replace(p.name, ".", "_")}"
    end)
    |> Enum.map_join("\n", fn {prom_name, group_points} ->
      {help, type} = Map.get(@metric_help, base_metric_name(prom_name), {"SRFM metric", "gauge"})

      lines = [
        "# HELP #{prom_name} #{help}",
        "# TYPE #{prom_name} #{type}"
      ]

      data_lines =
        Enum.map(group_points, fn p ->
          label   = to_string(p.service)
          value   = format_float(p.value)
          "#{prom_name}{service=\"#{label}\"} #{value}"
        end)

      Enum.join(lines ++ data_lines, "\n")
    end)
    |> then(fn body -> body <> "\n" end)
  end

  -- Strip the srfm_ prefix and replace _ back to look up help strings.
  defp base_metric_name("srfm_" <> rest), do: String.replace(rest, "_", "_", global: false)
  defp base_metric_name(name), do: name

  defp format_float(v) when is_float(v) do
    if v == trunc(v) do
      "#{trunc(v)}.0"
    else
      Float.to_string(v)
    end
  end

  defp format_float(v), do: to_string(v)

  defp schedule_poll do
    interval = Application.get_env(:srfm_coordination, :metrics_poll_interval_ms, @poll_interval_ms)
    Process.send_after(self(), :poll, interval)
  end
end
