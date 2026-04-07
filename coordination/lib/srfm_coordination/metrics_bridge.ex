defmodule SrfmCoordination.MetricsBridge do
  @moduledoc """
  GenServer that scrapes Prometheus metrics from all registered services every
  15 seconds, aggregates them, and exposes trend data for anomaly detection.

  Features:
  -- Scrapes /metrics endpoint from every healthy/degraded registered service
  -- Parses Prometheus text exposition format (gauge, counter, histogram_sum/_count)
  -- Maintains a 1-hour rolling history in ETS for trend analysis
  -- Publishes metric_anomaly events when a metric deviates > 3 sigma from mean
  -- Provides get_metric/2, get_trend/3 for in-process consumers
  -- HTTP handler at GET /metrics/aggregate returns consolidated JSON

  ETS tables:
    :srfm_metrics_current  -- {service_name, %{metric_name => float}}
    :srfm_metrics_history  -- {{service_name, metric_name}, [{ts, value}]}
  """

  use GenServer
  require Logger

  @scrape_interval_ms 15_000
  @history_window_seconds 3_600
  @anomaly_sigma_threshold 3.0
  @http_timeout_ms 5_000

  @current_table :srfm_metrics_current
  @history_table :srfm_metrics_history

  defstruct scrape_ref: nil, last_scrape_at: nil, scrape_errors: %{}

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Get the most recent scraped value for a metric on a given service."
  @spec get_metric(atom(), String.t()) :: {:ok, float()} | {:error, :not_found}
  def get_metric(service, metric_name) when is_atom(service) and is_binary(metric_name) do
    case :ets.lookup(@current_table, service) do
      [{^service, metrics}] ->
        case Map.fetch(metrics, metric_name) do
          {:ok, val} -> {:ok, val}
          :error -> {:error, :not_found}
        end

      [] ->
        {:error, :not_found}
    end
  end

  @doc "Get the rolling history for a metric over the last `minutes` minutes."
  @spec get_trend(atom(), String.t(), pos_integer()) :: {:ok, [float()]} | {:error, term()}
  def get_trend(service, metric_name, minutes)
      when is_atom(service) and is_binary(metric_name) and is_integer(minutes) and minutes > 0 do
    key = {service, metric_name}
    cutoff = now_unix() - minutes * 60

    case :ets.lookup(@history_table, key) do
      [{^key, history}] ->
        values =
          history
          |> Enum.filter(fn {ts, _v} -> ts >= cutoff end)
          |> Enum.map(fn {_ts, v} -> v end)

        {:ok, values}

      [] ->
        {:error, :no_history}
    end
  end

  @doc "Return aggregated metrics snapshot for all services."
  @spec aggregate_snapshot() :: map()
  def aggregate_snapshot do
    :ets.tab2list(@current_table)
    |> Map.new(fn {svc, metrics} -> {svc, metrics} end)
  end

  @doc "Force an immediate scrape cycle."
  def scrape_now do
    GenServer.cast(__MODULE__, :scrape_now)
  end

  @doc "Return current scrape state summary."
  @spec status() :: map()
  def status do
    GenServer.call(__MODULE__, :status)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@current_table, [:named_table, :set, :public, read_concurrency: true])
    :ets.new(@history_table, [:named_table, :set, :public, read_concurrency: true])
    Logger.info("[MetricsBridge] Initialized -- scraping every #{@scrape_interval_ms}ms")
    ref = schedule_scrape()
    {:ok, %__MODULE__{scrape_ref: ref, last_scrape_at: nil, scrape_errors: %{}}}
  end

  @impl true
  def handle_info(:scrape, state) do
    ref = schedule_scrape()
    new_state = do_scrape_all(%{state | scrape_ref: ref, last_scrape_at: DateTime.utc_now()})
    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:scrape_now, state) do
    new_state = do_scrape_all(state)
    {:noreply, %{new_state | last_scrape_at: DateTime.utc_now()}}
  end

  @impl true
  def handle_call(:status, _from, state) do
    services_scraped =
      :ets.tab2list(@current_table)
      |> Enum.map(fn {svc, metrics} -> %{service: svc, metric_count: map_size(metrics)} end)

    reply = %{
      last_scrape_at: state.last_scrape_at,
      scrape_errors: state.scrape_errors,
      services: services_scraped
    }

    {:reply, reply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[MetricsBridge] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Scrape logic
  # ---------------------------------------------------------------------------

  defp do_scrape_all(state) do
    services =
      SrfmCoordination.ServiceRegistry.list_all()
      |> Enum.filter(fn svc -> svc.health_status in [:healthy, :degraded] end)

    results =
      Task.async_stream(
        services,
        fn svc -> {svc.name, scrape_service(svc)} end,
        max_concurrency: 20,
        timeout: @http_timeout_ms + 1_000,
        on_timeout: :kill_task
      )
      |> Enum.map(fn
        {:ok, result} -> result
        {:exit, :timeout} -> {:timeout, :timeout}
      end)

    new_errors =
      Enum.reduce(results, state.scrape_errors, fn
        {name, {:ok, metrics}}, errors ->
          store_metrics(name, metrics)
          check_anomalies(name, metrics)
          Map.delete(errors, name)

        {name, {:error, reason}}, errors ->
          Logger.warning("[MetricsBridge] Failed to scrape :#{name}: #{inspect(reason)}")
          Map.put(errors, name, {reason, DateTime.utc_now()})

        {:timeout, _}, errors ->
          errors
      end)

    %{state | scrape_errors: new_errors}
  end

  defp scrape_service(svc) do
    url = "http://localhost:#{svc.port}/metrics"

    try do
      case HTTPoison.get(url, [], recv_timeout: @http_timeout_ms) do
        {:ok, %{status_code: 200, body: body}} ->
          {:ok, parse_prometheus_text(body)}

        {:ok, %{status_code: code}} ->
          {:error, {:bad_status, code}}

        {:error, %HTTPoison.Error{reason: reason}} ->
          {:error, reason}
      end
    catch
      kind, reason -> {:error, {kind, reason}}
    end
  end

  # ---------------------------------------------------------------------------
  # Prometheus text format parser
  # ---------------------------------------------------------------------------

  @doc false
  def parse_prometheus_text(body) when is_binary(body) do
    body
    |> String.split("\n")
    |> Enum.reject(fn line ->
      trimmed = String.trim(line)
      trimmed == "" or String.starts_with?(trimmed, "#")
    end)
    |> Enum.reduce(%{}, fn line, acc ->
      case parse_metric_line(line) do
        {:ok, name, value} -> Map.put(acc, name, value)
        :error -> acc
      end
    end)
  end

  # Parses lines of the form:
  --   metric_name{label="val",...} 1.23
  --   metric_name 1.23
  --   metric_name{...} 1.23 1234567890
  defp parse_metric_line(line) do
    # Strip optional timestamp at end and trim
    stripped = line |> String.trim() |> strip_timestamp()

    # Pattern: name_possibly_with_braces value
    case Regex.run(~r/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|[+-]?Inf|NaN)$/, stripped) do
      [_, name, _labels, value_str] ->
        parse_float_value(name, value_str)

      [_, name, value_str] ->
        parse_float_value(name, value_str)

      nil ->
        # Try without braces pattern
        case Regex.run(~r/^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|[+-]?Inf|NaN)$/, stripped) do
          [_, name, value_str] -> parse_float_value(name, value_str)
          _ -> :error
        end
    end
  end

  defp strip_timestamp(line) do
    # If the line ends with a Unix ms timestamp (13-digit number), strip it
    case Regex.run(~r/^(.+)\s+\d{13}$/, line) do
      [_, without_ts] -> String.trim(without_ts)
      _ -> line
    end
  end

  defp parse_float_value(name, value_str) do
    case value_str do
      "+Inf" -> {:ok, name, :infinity}
      "-Inf" -> {:ok, name, :neg_infinity}
      "NaN" -> {:ok, name, :nan}
      _ ->
        case Float.parse(value_str) do
          {f, ""} -> {:ok, name, f}
          {f, _} -> {:ok, name, f}
          :error ->
            case Integer.parse(value_str) do
              {i, _} -> {:ok, name, i * 1.0}
              :error -> :error
            end
        end
    end
  end

  # ---------------------------------------------------------------------------
  # ETS storage and history management
  # ---------------------------------------------------------------------------

  defp store_metrics(service, metrics) when is_atom(service) and is_map(metrics) do
    # Filter out non-numeric special atoms before storing
    numeric_metrics =
      Map.reject(metrics, fn {_k, v} -> v in [:infinity, :neg_infinity, :nan] end)

    :ets.insert(@current_table, {service, numeric_metrics})

    ts = now_unix()

    Enum.each(numeric_metrics, fn {metric_name, value} when is_float(value) ->
      key = {service, metric_name}
      cutoff = ts - @history_window_seconds

      existing =
        case :ets.lookup(@history_table, key) do
          [{^key, hist}] -> hist
          [] -> []
        end

      # Prepend new point and evict entries older than 1 hour
      pruned =
        [{ts, value} | existing]
        |> Enum.filter(fn {t, _v} -> t >= cutoff end)

      :ets.insert(@history_table, {key, pruned})
    end)
  end

  # ---------------------------------------------------------------------------
  # Anomaly detection -- 3-sigma rule over 1-hour rolling history
  # ---------------------------------------------------------------------------

  defp check_anomalies(service, metrics) do
    Enum.each(metrics, fn {metric_name, current_value}
        when is_float(current_value) ->
      key = {service, metric_name}

      history =
        case :ets.lookup(@history_table, key) do
          [{^key, hist}] -> Enum.map(hist, fn {_ts, v} -> v end)
          [] -> []
        end

      if length(history) >= 10 do
        mean = Enum.sum(history) / length(history)
        variance = Enum.reduce(history, 0.0, fn v, acc -> acc + (v - mean) * (v - mean) end) / length(history)
        sigma = :math.sqrt(variance)

        if sigma > 0.0 and abs(current_value - mean) > @anomaly_sigma_threshold * sigma do
          z_score = (current_value - mean) / sigma
          Logger.warning("[MetricsBridge] Anomaly detected: :#{service}/#{metric_name} z=#{Float.round(z_score, 2)}")
          emit_anomaly_event(service, metric_name, current_value, mean, sigma, z_score)
        end
      end
    end)
  end

  defp emit_anomaly_event(service, metric_name, value, mean, sigma, z_score) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil ->
        :ok

      _pid ->
        SrfmCoordination.EventBus.publish(:alert, %{
          type: :metric_anomaly,
          service: service,
          metric: metric_name,
          value: value,
          mean: mean,
          sigma: sigma,
          z_score: z_score
        })
    end
  end

  # ---------------------------------------------------------------------------
  # HTTP response helper
  # ---------------------------------------------------------------------------

  @doc "Build aggregated JSON body for GET /metrics/aggregate."
  @spec build_aggregate_json() :: String.t()
  def build_aggregate_json do
    snapshot =
      :ets.tab2list(@current_table)
      |> Map.new(fn {svc, metrics} ->
        {Atom.to_string(svc), metrics}
      end)

    Jason.encode!(%{
      scraped_at: DateTime.utc_now(),
      services: snapshot
    })
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp schedule_scrape do
    Process.send_after(self(), :scrape, @scrape_interval_ms)
  end

  defp now_unix, do: System.os_time(:second)
end
