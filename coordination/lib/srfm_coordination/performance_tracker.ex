defmodule SrfmCoordination.PerformanceTracker do
  @moduledoc """
  Tracks live trader performance metrics and drives automatic rollback decisions.

  Responsibilities:
  -- Records trades and equity curve snapshots into ETS
  -- Computes rolling Sharpe ratio over configurable windows
  -- Tracks peak equity and current drawdown
  -- Triggers rollback via ParameterCoordinator when:
       4-hour rolling Sharpe < -0.5 AND current drawdown > 3%
  -- Publishes performance_degraded events with context
  -- Exposes REST endpoints:
       GET /performance/report
       GET /performance/equity_curve?hours=N

  ETS tables:
    :srfm_equity_curve  -- [{timestamp_unix, equity_float}]  (24h rolling)
    :srfm_trade_log     -- [{trade_id, trade_map}]
    :srfm_perf_state    -- misc state (peak_equity, rollback_sent_at, etc.)

  Annualisation factor: sqrt(4 * 6.5 * 252) for 15-minute bars.
  """

  use GenServer
  require Logger

  @equity_interval_ms 900_000   # 15 minutes
  @equity_history_hours 24
  @equity_history_seconds @equity_history_hours * 3_600

  # Rollback thresholds
  @sharpe_4h_threshold -0.5
  @drawdown_threshold 0.03

  # Annualisation: sqrt(4 bars/hr * 6.5 hr/day * 252 days/yr)
  @annualise_factor :math.sqrt(4 * 6.5 * 252)

  @equity_table :srfm_equity_curve
  @trade_table :srfm_trade_log
  @state_table :srfm_perf_state

  defstruct [
    :equity_ref,
    rollback_last_at: nil,
    degraded_alert_last_at: nil
  ]

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Record a completed trade. `param_snapshot` is the ParameterCoordinator.all()
  snapshot at the time of the fill, used for P&L attribution.
  """
  @spec record_trade(String.t(), float(), DateTime.t(), DateTime.t(), map()) :: :ok
  def record_trade(symbol, pnl, entry_time, exit_time, param_snapshot) do
    GenServer.cast(__MODULE__, {:record_trade, symbol, pnl, entry_time, exit_time, param_snapshot})
  end

  @doc "Record a point on the equity curve. Called every 15 minutes by the live trader."
  @spec record_equity_point(float(), DateTime.t()) :: :ok
  def record_equity_point(equity, timestamp) do
    GenServer.cast(__MODULE__, {:record_equity_point, equity, timestamp})
  end

  @doc """
  Compute rolling Sharpe ratio using 15-minute returns over `window_hours`.
  Returns the annualised Sharpe as a float. Returns 0.0 if insufficient data.
  """
  @spec compute_sharpe(pos_integer()) :: float()
  def compute_sharpe(window_hours) when is_integer(window_hours) and window_hours > 0 do
    GenServer.call(__MODULE__, {:compute_sharpe, window_hours})
  end

  @doc "Return the current drawdown from peak equity as a fraction (e.g. 0.05 = 5%)."
  @spec current_drawdown() :: float()
  def current_drawdown do
    GenServer.call(__MODULE__, :current_drawdown)
  end

  @doc """
  Return a full performance report map.
  Keys: equity, sharpe_4h, sharpe_24h, max_dd, win_rate, avg_hold_minutes, trade_count.
  """
  @spec get_performance_report() :: map()
  def get_performance_report do
    GenServer.call(__MODULE__, :get_performance_report)
  end

  @doc "Return equity curve points for the last `hours` hours, sorted oldest first."
  @spec get_equity_curve(pos_integer()) :: [{integer(), float()}]
  def get_equity_curve(hours \\ 24) when is_integer(hours) and hours > 0 do
    cutoff = now_unix() - hours * 3_600
    all_points = ets_get(@state_table, :equity_points, [])

    all_points
    |> Enum.filter(fn {ts, _eq} -> ts >= cutoff end)
    |> Enum.sort_by(fn {ts, _eq} -> ts end)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@equity_table, [:named_table, :ordered_set, :public, read_concurrency: true])
    :ets.new(@trade_table, [:named_table, :set, :public, read_concurrency: true])
    :ets.new(@state_table, [:named_table, :set, :public, read_concurrency: true])

    :ets.insert(@state_table, {:equity_points, []})
    :ets.insert(@state_table, {:peak_equity, 0.0})
    :ets.insert(@state_table, {:current_equity, 0.0})

    ref = schedule_equity_tick()
    Logger.info("[PerformanceTracker] Initialized")
    {:ok, %__MODULE__{equity_ref: ref}}
  end

  @impl true
  def handle_cast({:record_trade, symbol, pnl, entry_time, exit_time, param_snapshot}, state) do
    trade_id = generate_id()
    hold_minutes = hold_time_minutes(entry_time, exit_time)

    trade = %{
      id: trade_id,
      symbol: symbol,
      pnl: pnl,
      entry_time: entry_time,
      exit_time: exit_time,
      hold_minutes: hold_minutes,
      param_snapshot: param_snapshot,
      recorded_at: DateTime.utc_now()
    }

    :ets.insert(@trade_table, {trade_id, trade})
    Logger.debug("[PerformanceTracker] Trade recorded: #{symbol} pnl=#{Float.round(pnl, 4)}")
    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_equity_point, equity, timestamp}, state) do
    ts = DateTime.to_unix(timestamp)
    cutoff = ts - @equity_history_seconds

    existing = ets_get(@state_table, :equity_points, [])

    pruned =
      [{ts, equity} | existing]
      |> Enum.filter(fn {t, _} -> t >= cutoff end)
      |> Enum.sort_by(fn {t, _} -> t end)

    :ets.insert(@state_table, {:equity_points, pruned})
    :ets.insert(@state_table, {:current_equity, equity})

    # Update peak equity
    peak = ets_get(@state_table, :peak_equity, 0.0)

    if equity > peak do
      :ets.insert(@state_table, {:peak_equity, equity})
    end

    Logger.debug("[PerformanceTracker] Equity point: #{Float.round(equity, 2)} at #{ts}")
    {:noreply, state}
  end

  @impl true
  def handle_call({:compute_sharpe, window_hours}, _from, state) do
    sharpe = do_compute_sharpe(window_hours)
    {:reply, sharpe, state}
  end

  @impl true
  def handle_call(:current_drawdown, _from, state) do
    dd = do_compute_drawdown()
    {:reply, dd, state}
  end

  @impl true
  def handle_call(:get_performance_report, _from, state) do
    report = build_report()
    {:reply, report, state}
  end

  @impl true
  def handle_info(:equity_tick, state) do
    ref = schedule_equity_tick()
    new_state = check_rollback_condition(%{state | equity_ref: ref})
    {:noreply, new_state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[PerformanceTracker] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Sharpe computation
  # ---------------------------------------------------------------------------

  defp do_compute_sharpe(window_hours) do
    cutoff = now_unix() - window_hours * 3_600
    points = ets_get(@state_table, :equity_points, [])

    window_points =
      points
      |> Enum.filter(fn {ts, _} -> ts >= cutoff end)
      |> Enum.sort_by(fn {ts, _} -> ts end)

    returns = compute_returns(window_points)

    case length(returns) do
      n when n < 2 ->
        0.0

      _ ->
        mean_return = Enum.sum(returns) / length(returns)
        variance = Enum.reduce(returns, 0.0, fn r, acc -> acc + (r - mean_return) * (r - mean_return) end) / length(returns)
        std_dev = :math.sqrt(variance)

        if std_dev == 0.0 do
          0.0
        else
          (mean_return / std_dev) * @annualise_factor
        end
    end
  end

  # Compute fractional returns between consecutive equity points
  defp compute_returns([]), do: []
  defp compute_returns([_]), do: []

  defp compute_returns(points) do
    points
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.flat_map(fn
      [{_t1, eq1}, {_t2, eq2}] when eq1 > 0.0 ->
        [(eq2 - eq1) / eq1]

      _ ->
        []
    end)
  end

  # ---------------------------------------------------------------------------
  # Drawdown computation
  # ---------------------------------------------------------------------------

  defp do_compute_drawdown do
    peak = ets_get(@state_table, :peak_equity, 0.0)
    current = ets_get(@state_table, :current_equity, 0.0)

    cond do
      peak <= 0.0 -> 0.0
      current >= peak -> 0.0
      true -> (peak - current) / peak
    end
  end

  # ---------------------------------------------------------------------------
  # Rollback trigger
  # ---------------------------------------------------------------------------

  defp check_rollback_condition(state) do
    sharpe_4h = do_compute_sharpe(4)
    drawdown = do_compute_drawdown()

    trigger = sharpe_4h < @sharpe_4h_threshold and drawdown > @drawdown_threshold

    if trigger do
      # Rate-limit: don't fire more than once per hour
      now = System.monotonic_time(:second)
      last = state.rollback_last_at || 0
      elapsed = now - last

      if elapsed > 3_600 do
        Logger.error(
          "[PerformanceTracker] Rollback condition met: " <>
          "Sharpe_4h=#{Float.round(sharpe_4h, 3)} drawdown=#{Float.round(drawdown * 100, 2)}%"
        )

        notify_rollback(sharpe_4h, drawdown)
        %{state | rollback_last_at: now, degraded_alert_last_at: now}
      else
        state
      end
    else
      state
    end
  end

  defp notify_rollback(sharpe_4h, drawdown) do
    payload = %{
      type: :performance_degraded,
      sharpe_4h: sharpe_4h,
      drawdown_pct: Float.round(drawdown * 100, 3),
      sharpe_threshold: @sharpe_4h_threshold,
      drawdown_threshold_pct: @drawdown_threshold * 100,
      triggered_at: DateTime.utc_now()
    }

    # Emit event for Alerting module to pick up
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _pid -> SrfmCoordination.EventBus.publish(:alert, payload)
    end

    # Notify ParameterCoordinator directly
    case Process.whereis(SrfmCoordination.ParameterCoordinator) do
      nil ->
        Logger.warning("[PerformanceTracker] ParameterCoordinator not running -- rollback skipped")

      _pid ->
        Logger.info("[PerformanceTracker] Notifying ParameterCoordinator to rollback")
        # The coordinator can decide rollback strategy from this cast
        send(SrfmCoordination.ParameterCoordinator, {:performance_rollback_request, payload})
    end
  end

  # ---------------------------------------------------------------------------
  # Report builder
  # ---------------------------------------------------------------------------

  defp build_report do
    trades = :ets.tab2list(@trade_table) |> Enum.map(fn {_id, t} -> t end)
    sharpe_4h = do_compute_sharpe(4)
    sharpe_24h = do_compute_sharpe(24)
    drawdown = do_compute_drawdown()
    current_equity = ets_get(@state_table, :current_equity, 0.0)
    peak_equity = ets_get(@state_table, :peak_equity, 0.0)

    {win_rate, avg_hold} = compute_trade_stats(trades)
    max_dd = compute_max_drawdown()

    %{
      equity: current_equity,
      peak_equity: peak_equity,
      sharpe_4h: sharpe_4h,
      sharpe_24h: sharpe_24h,
      current_drawdown_pct: Float.round(drawdown * 100, 3),
      max_dd_pct: Float.round(max_dd * 100, 3),
      win_rate: win_rate,
      avg_hold_minutes: avg_hold,
      trade_count: length(trades),
      report_generated_at: DateTime.utc_now()
    }
  end

  defp compute_trade_stats([]), do: {0.0, 0.0}

  defp compute_trade_stats(trades) do
    winners = Enum.count(trades, fn t -> t.pnl > 0 end)
    win_rate = if length(trades) > 0, do: winners / length(trades), else: 0.0

    holds = Enum.map(trades, fn t -> t.hold_minutes || 0.0 end)
    avg_hold = if length(holds) > 0, do: Enum.sum(holds) / length(holds), else: 0.0

    {Float.round(win_rate, 4), Float.round(avg_hold, 2)}
  end

  defp compute_max_drawdown do
    points = ets_get(@state_table, :equity_points, [])
    sorted = Enum.sort_by(points, fn {ts, _} -> ts end)

    {_peak, max_dd} =
      Enum.reduce(sorted, {0.0, 0.0}, fn {_ts, equity}, {peak, max_dd} ->
        new_peak = max(peak, equity)
        dd = if new_peak > 0.0, do: (new_peak - equity) / new_peak, else: 0.0
        {new_peak, max(max_dd, dd)}
      end)

    max_dd
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp hold_time_minutes(%DateTime{} = entry, %DateTime{} = exit_time) do
    diff_seconds = DateTime.diff(exit_time, entry, :second)
    diff_seconds / 60.0
  end

  defp hold_time_minutes(_entry, _exit_time), do: 0.0

  defp schedule_equity_tick do
    Process.send_after(self(), :equity_tick, @equity_interval_ms)
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end

  defp now_unix, do: System.os_time(:second)

  defp ets_get(table, key, default) do
    case :ets.lookup(table, key) do
      [{^key, value}] -> value
      [] -> default
    end
  end
end
