defmodule SrfmCoordination.Telemetry do
  @moduledoc """
  Telemetry integration for the SRFM Coordination Service.

  Attaches handlers for all key system events using :telemetry.attach/4 and
  aggregates them into counts, rates, and latency histograms (p50/p95/p99).

  Events tracked:
    [:srfm, :param_update, :start/:stop]     -- parameter update latency
    [:srfm, :health_check, :result]          -- health check outcomes
    [:srfm, :circuit_breaker, :state_change] -- circuit breaker transitions
    [:srfm, :order, :submitted/:filled/:rejected] -- order lifecycle
    [:srfm, :rollback, :triggered]           -- parameter rollback events

  Exposes metrics in Telemetry.Metrics-compatible format via metrics/0.
  Stores aggregates in an ETS table for O(1) reads by the HTTP metrics endpoint.
  """

  use GenServer
  require Logger

  @table :srfm_telemetry_aggregates
  @handler_prefix "srfm_coordination"

  # Latency histogram buckets in microseconds
  @latency_buckets [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

  # How many raw latency samples to keep for percentile calculation (ring buffer per event)
  @sample_window 1_000

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  @type metric_name :: atom()
  @type handler_id :: String.t()

  @type latency_summary :: %{
          p50: non_neg_integer(),
          p95: non_neg_integer(),
          p99: non_neg_integer(),
          min: non_neg_integer(),
          max: non_neg_integer(),
          count: non_neg_integer(),
          sum_us: non_neg_integer()
        }

  @type aggregate :: %{
          count: non_neg_integer(),
          last_seen: DateTime.t() | nil,
          rate_1m: float(),
          latency: latency_summary() | nil,
          tags: map()
        }

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc "Start the Telemetry GenServer and attach all handlers."
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Returns a list of Telemetry.Metrics metric definitions for all tracked events.
  Compatible with telemetry_metrics and telemetry_poller.
  """
  @spec metrics() :: [map()]
  def metrics do
    [
      # Parameter update latency
      %{
        name: [:srfm, :param_update, :stop],
        type: :distribution,
        measurement: :duration,
        unit: {:native, :microsecond},
        description: "Parameter update end-to-end latency",
        reporter_options: [buckets: @latency_buckets]
      },
      %{
        name: [:srfm, :param_update, :start],
        type: :counter,
        description: "Parameter update attempts started"
      },
      # Health check outcomes
      %{
        name: [:srfm, :health_check, :result],
        type: :counter,
        tags: [:service, :status],
        description: "Health check results by service and status"
      },
      # Circuit breaker transitions
      %{
        name: [:srfm, :circuit_breaker, :state_change],
        type: :counter,
        tags: [:name, :from, :to],
        description: "Circuit breaker state transitions"
      },
      # Order lifecycle
      %{
        name: [:srfm, :order, :submitted],
        type: :counter,
        tags: [:broker, :symbol],
        description: "Orders submitted to broker"
      },
      %{
        name: [:srfm, :order, :filled],
        type: :counter,
        tags: [:broker, :symbol],
        description: "Orders filled by broker"
      },
      %{
        name: [:srfm, :order, :rejected],
        type: :counter,
        tags: [:broker, :symbol, :reason],
        description: "Orders rejected by broker"
      },
      # Rollback events
      %{
        name: [:srfm, :rollback, :triggered],
        type: :counter,
        tags: [:reason],
        description: "Parameter rollback events triggered"
      }
    ]
  end

  @doc """
  Attach all telemetry handlers. Called at startup from init/1.
  Safe to call multiple times -- detaches existing handlers first.
  """
  @spec attach_handlers() :: :ok
  def attach_handlers do
    GenServer.call(__MODULE__, :attach_handlers)
  end

  @doc "Detach all handlers registered by this module."
  @spec detach_all() :: :ok
  def detach_all do
    handler_ids()
    |> Enum.each(fn id ->
      case :telemetry.detach(id) do
        :ok -> :ok
        {:error, :not_found} -> :ok
      end
    end)

    Logger.info("[Telemetry] All handlers detached")
    :ok
  end

  @doc "Get current aggregate for a specific event key."
  @spec get_aggregate(term()) :: aggregate() | nil
  def get_aggregate(key) do
    case :ets.lookup(@table, {:aggregate, key}) do
      [{_, agg}] -> agg
      [] -> nil
    end
  end

  @doc "Get all aggregates as a map."
  @spec all_aggregates() :: map()
  def all_aggregates do
    :ets.match_object(@table, {{:aggregate, :_}, :_})
    |> Map.new(fn {{:aggregate, key}, val} -> {key, val} end)
  end

  @doc "Get the latency summary for a named event."
  @spec latency_summary(atom()) :: latency_summary() | nil
  def latency_summary(event_name) do
    case :ets.lookup(@table, {:samples, event_name}) do
      [{_, samples}] when samples != [] -> compute_percentiles(samples)
      _ -> nil
    end
  end

  @doc "Reset all aggregates. Useful for testing."
  @spec reset_all() :: :ok
  def reset_all do
    GenServer.call(__MODULE__, :reset_all)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :set, :public, write_concurrency: true, read_concurrency: true])

    # Seed aggregate entries for all known events
    seed_aggregates()

    # Attach handlers
    do_attach_handlers()

    # Schedule rate calculation every 60 seconds
    :timer.send_interval(60_000, :recalculate_rates)

    Logger.info("[Telemetry] Initialized, handlers attached")
    {:ok, %{attached: handler_ids()}}
  end

  @impl true
  def handle_call(:attach_handlers, _from, state) do
    # Detach first to avoid duplicate handler errors
    Enum.each(handler_ids(), fn id ->
      :telemetry.detach(id)
    end)

    do_attach_handlers()
    {:reply, :ok, %{state | attached: handler_ids()}}
  end

  @impl true
  def handle_call(:reset_all, _from, state) do
    :ets.delete_all_objects(@table)
    seed_aggregates()
    Logger.info("[Telemetry] Aggregates reset")
    {:reply, :ok, state}
  end

  @impl true
  def handle_info(:recalculate_rates, state) do
    recalculate_rates()
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(_reason, _state) do
    detach_all()
    :ok
  end

  # ---------------------------------------------------------------------------
  # Handler implementations (called directly by :telemetry dispatch)
  # ---------------------------------------------------------------------------

  @doc false
  def handle_param_update_start(_event, _measurements, metadata, _config) do
    key = {:param_update, :start}
    increment_count(key, metadata)
  end

  @doc false
  def handle_param_update_stop(_event, measurements, metadata, _config) do
    key = {:param_update, :stop}
    increment_count(key, metadata)
    duration_us = System.convert_time_unit(measurements[:duration] || 0, :native, :microsecond)
    record_latency(:param_update, duration_us)
  end

  @doc false
  def handle_health_check_result(_event, _measurements, metadata, _config) do
    service = Map.get(metadata, :service, :unknown)
    status = Map.get(metadata, :status, :unknown)
    key = {:health_check, service, status}
    increment_count(key, metadata)
  end

  @doc false
  def handle_circuit_breaker_change(_event, _measurements, metadata, _config) do
    name = Map.get(metadata, :name, :unknown)
    from = Map.get(metadata, :from, :unknown)
    to = Map.get(metadata, :to, :unknown)
    key = {:circuit_breaker, name, from, to}
    increment_count(key, metadata)
    Logger.warning("[Telemetry] Circuit breaker #{name}: #{from} -> #{to}")
  end

  @doc false
  def handle_order_submitted(_event, _measurements, metadata, _config) do
    broker = Map.get(metadata, :broker, :unknown)
    symbol = Map.get(metadata, :symbol, :unknown)
    key = {:order, :submitted, broker, symbol}
    increment_count(key, metadata)
  end

  @doc false
  def handle_order_filled(_event, _measurements, metadata, _config) do
    broker = Map.get(metadata, :broker, :unknown)
    symbol = Map.get(metadata, :symbol, :unknown)
    key = {:order, :filled, broker, symbol}
    increment_count(key, metadata)
  end

  @doc false
  def handle_order_rejected(_event, _measurements, metadata, _config) do
    broker = Map.get(metadata, :broker, :unknown)
    symbol = Map.get(metadata, :symbol, :unknown)
    reason = Map.get(metadata, :reason, :unknown)
    key = {:order, :rejected, broker, symbol, reason}
    increment_count(key, metadata)
    Logger.warning("[Telemetry] Order rejected: #{symbol} @ #{broker} reason=#{reason}")
  end

  @doc false
  def handle_rollback_triggered(_event, _measurements, metadata, _config) do
    reason = Map.get(metadata, :reason, :unknown)
    key = {:rollback, :triggered, reason}
    increment_count(key, metadata)
    Logger.error("[Telemetry] Rollback triggered: reason=#{reason}")
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp handler_ids do
    [
      "#{@handler_prefix}.param_update.start",
      "#{@handler_prefix}.param_update.stop",
      "#{@handler_prefix}.health_check.result",
      "#{@handler_prefix}.circuit_breaker.state_change",
      "#{@handler_prefix}.order.submitted",
      "#{@handler_prefix}.order.filled",
      "#{@handler_prefix}.order.rejected",
      "#{@handler_prefix}.rollback.triggered"
    ]
  end

  defp do_attach_handlers do
    attachments = [
      {"#{@handler_prefix}.param_update.start",
       [:srfm, :param_update, :start],
       &__MODULE__.handle_param_update_start/4, %{}},

      {"#{@handler_prefix}.param_update.stop",
       [:srfm, :param_update, :stop],
       &__MODULE__.handle_param_update_stop/4, %{}},

      {"#{@handler_prefix}.health_check.result",
       [:srfm, :health_check, :result],
       &__MODULE__.handle_health_check_result/4, %{}},

      {"#{@handler_prefix}.circuit_breaker.state_change",
       [:srfm, :circuit_breaker, :state_change],
       &__MODULE__.handle_circuit_breaker_change/4, %{}},

      {"#{@handler_prefix}.order.submitted",
       [:srfm, :order, :submitted],
       &__MODULE__.handle_order_submitted/4, %{}},

      {"#{@handler_prefix}.order.filled",
       [:srfm, :order, :filled],
       &__MODULE__.handle_order_filled/4, %{}},

      {"#{@handler_prefix}.order.rejected",
       [:srfm, :order, :rejected],
       &__MODULE__.handle_order_rejected/4, %{}},

      {"#{@handler_prefix}.rollback.triggered",
       [:srfm, :rollback, :triggered],
       &__MODULE__.handle_rollback_triggered/4, %{}}
    ]

    Enum.each(attachments, fn {id, event, handler, config} ->
      case :telemetry.attach(id, event, handler, config) do
        :ok ->
          Logger.debug("[Telemetry] Attached handler #{id}")

        {:error, :already_exists} ->
          Logger.debug("[Telemetry] Handler #{id} already attached, skipping")
      end
    end)
  end

  defp increment_count(key, metadata) do
    now = DateTime.utc_now()
    tags = Map.take(metadata, [:service, :broker, :symbol, :reason, :name, :from, :to, :status])

    :ets.update_counter(@table, {:aggregate, key}, [{2, 1}], {{:aggregate, key}, 0})

    # Update last_seen and tags in a separate write (count is the hot path)
    case :ets.lookup(@table, {:meta, key}) do
      [] ->
        :ets.insert(@table, {{:meta, key}, %{last_seen: now, tags: tags}})

      _ ->
        :ets.update_element(@table, {:meta, key}, {2, %{last_seen: now, tags: tags}})
    end
  end

  defp record_latency(event_name, duration_us) do
    samples_key = {:samples, event_name}

    existing =
      case :ets.lookup(@table, samples_key) do
        [{_, s}] -> s
        [] -> []
      end

    updated = [duration_us | Enum.take(existing, @sample_window - 1)]
    :ets.insert(@table, {samples_key, updated})
  end

  defp compute_percentiles([]), do: nil

  defp compute_percentiles(samples) do
    sorted = Enum.sort(samples)
    count = length(sorted)

    %{
      p50: percentile(sorted, count, 0.50),
      p95: percentile(sorted, count, 0.95),
      p99: percentile(sorted, count, 0.99),
      min: List.first(sorted),
      max: List.last(sorted),
      count: count,
      sum_us: Enum.sum(samples)
    }
  end

  defp percentile(sorted, count, pct) do
    idx = max(0, round(pct * count) - 1)
    Enum.at(sorted, min(idx, count - 1))
  end

  defp recalculate_rates do
    # Rate is computed as count / 60s window
    # In a full implementation, keep a sliding counter ring per minute
    # Here we log a snapshot for monitoring purposes
    total_updates =
      case :ets.lookup(@table, {:aggregate, {:param_update, :stop}}) do
        [{_, n}] -> n
        [] -> 0
      end

    Logger.debug("[Telemetry] Rate snapshot: param_updates=#{total_updates}")
  end

  defp seed_aggregates do
    seed_keys = [
      {:param_update, :start},
      {:param_update, :stop},
      {:health_check, :aggregate},
      {:circuit_breaker, :aggregate},
      {:order, :submitted},
      {:order, :filled},
      {:order, :rejected},
      {:rollback, :triggered}
    ]

    Enum.each(seed_keys, fn key ->
      :ets.insert_new(@table, {{:aggregate, key}, 0})
    end)
  end
end
