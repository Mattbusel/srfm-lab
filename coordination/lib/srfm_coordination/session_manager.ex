defmodule SrfmCoordination.SessionManager do
  @moduledoc """
  Manages trading session open/close lifecycle.
  Coordinates startup checks, parameter loading, and end-of-day procedures.

  The session state machine transitions through:
    :pre_market  -- before @market_open_hour:@market_open_minute
    :market_open -- during regular trading hours
    :market_close -- at or after @market_close_hour:@market_close_minute
    :after_hours -- post-close, IAE eligible

  Checks are run every 30 seconds via a timer loop. Transitions trigger
  open_session/0 and close_session/0 automatically.

  Half-day schedule: session closes at 13:00 ET on dates in @half_days.
  """

  use GenServer
  require Logger

  @market_open_hour    9
  @market_open_minute  30
  @market_close_hour   16
  @market_close_minute 0
  @half_day_close_hour 13
  @check_interval_ms   30_000

  -- Half-day dates as {month, day} tuples -- update annually.
  @half_days [
    {1, 2},    -- Day after New Year's (observed)
    {7, 3},    -- July 3rd
    {11, 29},  -- Day after Thanksgiving (Black Friday)
    {12, 24}   -- Christmas Eve
  ]

  defmodule SessionState do
    @moduledoc "Current session lifecycle state."
    defstruct [
      :status,
      :opened_at,
      :closed_at,
      :trades_today,
      :pnl_today,
      :last_param_update
    ]

    @type t :: %__MODULE__{
      status:            :pre_market | :market_open | :market_close | :after_hours,
      opened_at:         DateTime.t() | nil,
      closed_at:         DateTime.t() | nil,
      trades_today:      non_neg_integer(),
      pnl_today:         float(),
      last_param_update: DateTime.t() | nil
    }
  end

  defstruct [
    session:    nil,
    check_ref:  nil,
    iae_triggered_today: false
  ]

  -- ---------------------------------------------------------------------------
  -- Public API
  -- ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Return the current session state snapshot."
  @spec get_session() :: SessionState.t()
  def get_session do
    GenServer.call(__MODULE__, :get_session)
  end

  @doc """
  Manually trigger session open. Normally called automatically by the timer.
  Runs preflight checks before marking the session open.
  """
  @spec open_session() :: :ok | {:error, term()}
  def open_session do
    GenServer.call(__MODULE__, :open_session, 30_000)
  end

  @doc """
  Manually trigger session close. Normally called automatically by the timer.
  Saves end-of-day state and optionally triggers an IAE evolution cycle.
  """
  @spec close_session() :: :ok
  def close_session do
    GenServer.call(__MODULE__, :close_session, 30_000)
  end

  @doc "Record a trade result for today's session."
  @spec record_trade(float()) :: :ok
  def record_trade(pnl) when is_number(pnl) do
    GenServer.cast(__MODULE__, {:record_trade, pnl / 1.0})
  end

  @doc "Update last_param_update timestamp."
  @spec mark_param_update() :: :ok
  def mark_param_update do
    GenServer.cast(__MODULE__, :mark_param_update)
  end

  -- ---------------------------------------------------------------------------
  -- GenServer callbacks
  -- ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    initial_session = %SessionState{
      status:            :pre_market,
      opened_at:         nil,
      closed_at:         nil,
      trades_today:      0,
      pnl_today:         0.0,
      last_param_update: nil
    }

    ref = schedule_check()
    Logger.info("[SessionManager] Initialized -- checking session state every #{@check_interval_ms}ms")

    {:ok, %__MODULE__{session: initial_session, check_ref: ref}}
  end

  @impl true
  def handle_call(:get_session, _from, state) do
    {:reply, state.session, state}
  end

  @impl true
  def handle_call(:open_session, _from, state) do
    case do_open_session(state) do
      {:ok, new_state} ->
        {:reply, :ok, new_state}

      {:error, reason} = err ->
        Logger.error("[SessionManager] Failed to open session: #{inspect(reason)}")
        {:reply, err, state}
    end
  end

  @impl true
  def handle_call(:close_session, _from, state) do
    new_state = do_close_session(state)
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_cast({:record_trade, pnl}, state) do
    session = %{state.session |
      trades_today: state.session.trades_today + 1,
      pnl_today:    state.session.pnl_today + pnl
    }
    {:noreply, %{state | session: session}}
  end

  @impl true
  def handle_cast(:mark_param_update, state) do
    session = %{state.session | last_param_update: DateTime.utc_now()}
    {:noreply, %{state | session: session}}
  end

  @impl true
  def handle_info(:check_session, state) do
    new_state = evaluate_session_transition(state)
    ref = schedule_check()
    {:noreply, %{new_state | check_ref: ref}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, state) do
    Logger.info("[SessionManager] Terminating: #{inspect(reason)}")

    -- Attempt graceful close if session is open
    if state.session.status == :market_open do
      Logger.warning("[SessionManager] Closing open session on terminate")
      do_close_session(state)
    end

    :ok
  end

  -- ---------------------------------------------------------------------------
  -- Private -- state machine evaluation
  -- ---------------------------------------------------------------------------

  defp evaluate_session_transition(state) do
    now    = DateTime.utc_now()
    status = state.session.status

    cond do
      status == :pre_market and should_open?(now) ->
        Logger.info("[SessionManager] Market open time reached -- opening session")
        case do_open_session(state) do
          {:ok, new_state} -> new_state
          {:error, reason} ->
            Logger.error("[SessionManager] Auto-open failed: #{inspect(reason)}")
            state
        end

      status == :market_open and should_close?(now) ->
        Logger.info("[SessionManager] Market close time reached -- closing session")
        do_close_session(state)

      status in [:market_close, :after_hours] and is_new_trading_day?(state, now) ->
        -- Reset for the next trading day
        Logger.info("[SessionManager] New trading day -- resetting session state")
        reset_for_new_day(state, now)

      true ->
        state
    end
  end

  defp should_open?(now) do
    time = DateTime.to_time(now)

    open_time =
      case Time.new(@market_open_hour, @market_open_minute, 0) do
        {:ok, t} -> t
        _ -> ~T[09:30:00]
      end

    Time.compare(time, open_time) in [:gt, :eq]
  end

  defp should_close?(now) do
    time  = DateTime.to_time(now)
    date  = DateTime.to_date(now)
    month = date.month
    day   = date.day

    close_hour =
      if {month, day} in @half_days do
        @half_day_close_hour
      else
        @market_close_hour
      end

    close_time =
      case Time.new(close_hour, @market_close_minute, 0) do
        {:ok, t} -> t
        _ -> ~T[16:00:00]
      end

    Time.compare(time, close_time) in [:gt, :eq]
  end

  defp is_new_trading_day?(state, now) do
    case state.session.closed_at do
      nil -> false
      closed_at ->
        Date.compare(DateTime.to_date(closed_at), DateTime.to_date(now)) == :lt
    end
  end

  defp reset_for_new_day(state, _now) do
    session = %SessionState{
      status:            :pre_market,
      opened_at:         nil,
      closed_at:         nil,
      trades_today:      0,
      pnl_today:         0.0,
      last_param_update: state.session.last_param_update
    }

    %{state | session: session, iae_triggered_today: false}
  end

  -- ---------------------------------------------------------------------------
  -- Private -- session open procedure
  -- ---------------------------------------------------------------------------

  defp do_open_session(state) do
    Logger.info("[SessionManager] Running pre-open checks")

    with :ok <- load_current_params(),
         :ok <- verify_services_healthy(),
         :ok <- reset_circuit_breakers() do

      now     = DateTime.utc_now()
      session = %{state.session | status: :market_open, opened_at: now}

      publish(:session_opened, %{opened_at: now})

      Logger.info("[SessionManager] Session opened at #{DateTime.to_iso8601(now)}")
      {:ok, %{state | session: session}}
    end
  end

  defp load_current_params do
    case Process.whereis(SrfmCoordination.ParameterCoordinator) do
      nil ->
        Logger.warning("[SessionManager] ParameterCoordinator not running -- skipping param load")
        :ok

      _ ->
        params = SrfmCoordination.ParameterCoordinator.all()
        Logger.info("[SessionManager] Loaded #{map_size(params)} parameters for session open")
        :ok
    end
  end

  defp verify_services_healthy do
    case Process.whereis(SrfmCoordination.HealthMonitor) do
      nil ->
        Logger.warning("[SessionManager] HealthMonitor not running -- skipping health check")
        :ok

      _ ->
        health = SrfmCoordination.HealthMonitor.system_health()

        case health.overall do
          :healthy ->
            Logger.info("[SessionManager] All services healthy (score=#{health.score}%)")
            :ok

          :degraded ->
            Logger.warning("[SessionManager] System degraded (score=#{health.score}%) -- proceeding anyway")
            :ok

          :critical ->
            Logger.error("[SessionManager] System critical (score=#{health.score}%) -- aborting open")
            {:error, {:system_critical, health.score}}

          :no_services ->
            Logger.warning("[SessionManager] No services registered -- proceeding")
            :ok
        end
    end
  end

  defp reset_circuit_breakers do
    case Process.whereis(SrfmCoordination.CircuitBreakerSupervisor) do
      nil ->
        :ok

      _ ->
        circuits = SrfmCoordination.CircuitBreakerSupervisor.circuits()
        Enum.each(circuits, fn name ->
          try do
            SrfmCoordination.CircuitBreaker.reset(name)
          catch
            _, _ -> :ok
          end
        end)
        Logger.info("[SessionManager] Reset #{length(circuits)} circuit breakers")
        :ok
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- session close procedure
  -- ---------------------------------------------------------------------------

  defp do_close_session(state) do
    now = DateTime.utc_now()
    Logger.info(
      "[SessionManager] Closing session -- trades=#{state.session.trades_today} " <>
      "pnl=#{Float.round(state.session.pnl_today, 2)}"
    )

    -- Step 1: save end-of-day P&L to PerformanceLedger
    save_eod_pnl(state.session, now)

    -- Step 2: trigger IAE cycle if eligible and not already done today
    new_state =
      if should_trigger_iae?(state) do
        trigger_iae_cycle()
        %{state | iae_triggered_today: true}
      else
        state
      end

    -- Step 3: update session state
    session = %{new_state.session | status: :market_close, closed_at: now}
    publish(:session_closed, %{
      closed_at:    now,
      trades_today: session.trades_today,
      pnl_today:    session.pnl_today
    })

    Logger.info("[SessionManager] Session closed at #{DateTime.to_iso8601(now)}")
    %{new_state | session: session}
  end

  defp save_eod_pnl(session, now) do
    case Process.whereis(SrfmCoordination.PerformanceLedger) do
      nil ->
        Logger.debug("[SessionManager] PerformanceLedger not running -- skipping EOD save")

      _ ->
        ts = DateTime.to_unix(now, :millisecond)

        entry = %SrfmCoordination.PerformanceLedger.LedgerEntry{
          ts:       ts,
          sharpe:   0.0,   -- will be computed separately by PerformanceTracker
          drawdown: 0.0,
          pnl:      session.pnl_today,
          n_trades: session.trades_today
        }

        SrfmCoordination.PerformanceLedger.record_entry(entry)
        Logger.info("[SessionManager] EOD PnL saved: #{Float.round(session.pnl_today, 2)}")
    end
  end

  defp should_trigger_iae?(state) do
    not state.iae_triggered_today and
      state.session.trades_today > 0
  end

  defp trigger_iae_cycle do
    case Process.whereis(SrfmCoordination.GenomeBridge) do
      nil ->
        Logger.debug("[SessionManager] GenomeBridge not running -- skipping IAE trigger")

      _ ->
        Logger.info("[SessionManager] Triggering post-session IAE evolution cycle")
        -- Use a fire-and-forget task so we don't block session close
        Task.start(fn ->
          Process.sleep(5_000)  -- brief delay before triggering
          SrfmCoordination.GenomeBridge.get_population_stats()
        end)
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- EventBus publishing
  -- ---------------------------------------------------------------------------

  defp publish(type, payload) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _ ->
        SrfmCoordination.EventBus.publish(:service_health, Map.merge(payload, %{type: type}))
    end
  end

  defp schedule_check do
    Process.send_after(self(), :check_session, @check_interval_ms)
  end
end
