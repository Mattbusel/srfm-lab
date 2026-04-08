defmodule SrfmCoordination.StrategyManager do
  @moduledoc """
  Live strategy management GenServer.
  Manages active strategies with performance tracking, A/B testing,
  parameter hot-reload, risk limits, and audit logging.
  """
  use GenServer
  require Logger

  # ---------------------------------------------------------------------------
  # Types & Structs
  # ---------------------------------------------------------------------------
  defmodule StrategyState do
    @enforce_keys [:id, :name, :status]
    defstruct [
      :id,
      :name,
      :status,
      :config,
      :performance,
      :risk_limits,
      :dependencies,
      :shadow_of,
      :created_at,
      :updated_at,
      :version
    ]

    @type status :: :active | :paused | :deprecated | :shadow | :unwinding
    @type t :: %__MODULE__{
            id: binary(),
            name: binary(),
            status: status(),
            config: map(),
            performance: Performance.t() | nil,
            risk_limits: RiskLimits.t() | nil,
            dependencies: [binary()],
            shadow_of: binary() | nil,
            created_at: integer(),
            updated_at: integer(),
            version: non_neg_integer()
          }
  end

  defmodule Performance do
    defstruct [
      :total_pnl,
      :rolling_sharpe,
      :max_drawdown,
      :current_drawdown,
      :win_rate,
      :trade_count,
      :avg_holding_period,
      :returns_history,
      :pnl_history,
      :last_trade_at,
      :peak_equity
    ]

    @type t :: %__MODULE__{
            total_pnl: float(),
            rolling_sharpe: float(),
            max_drawdown: float(),
            current_drawdown: float(),
            win_rate: float(),
            trade_count: non_neg_integer(),
            avg_holding_period: float(),
            returns_history: :queue.queue(),
            pnl_history: [float()],
            last_trade_at: integer() | nil,
            peak_equity: float()
          }

    def new do
      %__MODULE__{
        total_pnl: 0.0,
        rolling_sharpe: 0.0,
        max_drawdown: 0.0,
        current_drawdown: 0.0,
        win_rate: 0.0,
        trade_count: 0,
        avg_holding_period: 0.0,
        returns_history: :queue.new(),
        pnl_history: [],
        last_trade_at: nil,
        peak_equity: 0.0
      }
    end
  end

  defmodule RiskLimits do
    defstruct [
      :max_position_usd,
      :max_drawdown_pct,
      :max_correlation_to_book,
      :max_daily_loss,
      :max_open_trades,
      :max_notional,
      :position_limit_per_asset
    ]

    @type t :: %__MODULE__{
            max_position_usd: float(),
            max_drawdown_pct: float(),
            max_correlation_to_book: float(),
            max_daily_loss: float(),
            max_open_trades: non_neg_integer(),
            max_notional: float(),
            position_limit_per_asset: float()
          }
  end

  defmodule AuditEntry do
    defstruct [:timestamp, :strategy_id, :action, :details, :actor]

    @type t :: %__MODULE__{
            timestamp: integer(),
            strategy_id: binary(),
            action: atom(),
            details: map(),
            actor: binary()
          }
  end

  defmodule State do
    defstruct [
      strategies: %{},
      audit_log: [],
      ab_tests: %{},
      rotation_config: %{},
      book_positions: %{}
    ]
  end

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  def register_strategy(server \\ __MODULE__, name, config, opts \\ []) do
    GenServer.call(server, {:register, name, config, opts})
  end

  def enable_strategy(server \\ __MODULE__, id, actor \\ "system") do
    GenServer.call(server, {:enable, id, actor})
  end

  def disable_strategy(server \\ __MODULE__, id, actor \\ "system", opts \\ []) do
    GenServer.call(server, {:disable, id, actor, opts})
  end

  def pause_strategy(server \\ __MODULE__, id, actor \\ "system") do
    GenServer.call(server, {:pause, id, actor})
  end

  def deprecate_strategy(server \\ __MODULE__, id, actor \\ "system") do
    GenServer.call(server, {:deprecate, id, actor})
  end

  def update_config(server \\ __MODULE__, id, new_config, actor \\ "system") do
    GenServer.call(server, {:update_config, id, new_config, actor})
  end

  def record_trade(server \\ __MODULE__, strategy_id, trade_result) do
    GenServer.cast(server, {:record_trade, strategy_id, trade_result})
  end

  def get_strategy(server \\ __MODULE__, id) do
    GenServer.call(server, {:get_strategy, id})
  end

  def list_strategies(server \\ __MODULE__, filter \\ :all) do
    GenServer.call(server, {:list_strategies, filter})
  end

  def get_performance(server \\ __MODULE__, id) do
    GenServer.call(server, {:get_performance, id})
  end

  def start_ab_test(server \\ __MODULE__, live_id, shadow_config, opts \\ []) do
    GenServer.call(server, {:start_ab_test, live_id, shadow_config, opts})
  end

  def get_ab_test_results(server \\ __MODULE__, test_id) do
    GenServer.call(server, {:ab_results, test_id})
  end

  def promote_shadow(server \\ __MODULE__, shadow_id, actor \\ "system") do
    GenServer.call(server, {:promote_shadow, shadow_id, actor})
  end

  def rotate_strategies(server \\ __MODULE__, opts \\ []) do
    GenServer.call(server, {:rotate, opts})
  end

  def check_risk_limits(server \\ __MODULE__, id) do
    GenServer.call(server, {:check_risk, id})
  end

  def get_audit_log(server \\ __MODULE__, opts \\ []) do
    GenServer.call(server, {:audit_log, opts})
  end

  def update_book_positions(server \\ __MODULE__, positions) do
    GenServer.cast(server, {:update_book, positions})
  end

  # ---------------------------------------------------------------------------
  # Server Callbacks
  # ---------------------------------------------------------------------------
  @impl true
  def init(_opts) do
    schedule_performance_check()
    schedule_risk_check()
    {:ok, %State{}}
  end

  @impl true
  def handle_call({:register, name, config, opts}, _from, state) do
    id = generate_id()
    now = System.system_time(:millisecond)

    risk_limits = Keyword.get(opts, :risk_limits, default_risk_limits())
    dependencies = Keyword.get(opts, :dependencies, [])

    strategy = %StrategyState{
      id: id,
      name: name,
      status: :paused,
      config: config,
      performance: Performance.new(),
      risk_limits: build_risk_limits(risk_limits),
      dependencies: dependencies,
      shadow_of: nil,
      created_at: now,
      updated_at: now,
      version: 1
    }

    entry = audit(:registered, id, %{name: name, config: config}, "system")
    new_state = %{state |
      strategies: Map.put(state.strategies, id, strategy),
      audit_log: [entry | state.audit_log]
    }
    {:reply, {:ok, id}, new_state}
  end

  def handle_call({:enable, id, actor}, _from, state) do
    case Map.get(state.strategies, id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      %{status: :deprecated} ->
        {:reply, {:error, :deprecated}, state}

      strategy ->
        case check_dependencies(strategy, state) do
          :ok ->
            updated = %{strategy | status: :active, updated_at: now()}
            entry = audit(:enabled, id, %{}, actor)
            new_state = %{state |
              strategies: Map.put(state.strategies, id, updated),
              audit_log: [entry | state.audit_log]
            }
            {:reply, :ok, new_state}

          {:error, missing} ->
            {:reply, {:error, {:missing_dependencies, missing}}, state}
        end
    end
  end

  def handle_call({:disable, id, actor, opts}, _from, state) do
    case Map.get(state.strategies, id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      strategy ->
        graceful = Keyword.get(opts, :graceful, true)
        new_status = if graceful, do: :unwinding, else: :paused

        updated = %{strategy | status: new_status, updated_at: now()}
        entry = audit(:disabled, id, %{graceful: graceful}, actor)

        if graceful do
          Process.send_after(self(), {:unwind_complete, id}, 30_000)
        end

        new_state = %{state |
          strategies: Map.put(state.strategies, id, updated),
          audit_log: [entry | state.audit_log]
        }
        {:reply, :ok, new_state}
    end
  end

  def handle_call({:pause, id, actor}, _from, state) do
    with_strategy(state, id, fn strategy ->
      updated = %{strategy | status: :paused, updated_at: now()}
      entry = audit(:paused, id, %{}, actor)
      new_state = %{state |
        strategies: Map.put(state.strategies, id, updated),
        audit_log: [entry | state.audit_log]
      }
      {:reply, :ok, new_state}
    end)
  end

  def handle_call({:deprecate, id, actor}, _from, state) do
    with_strategy(state, id, fn strategy ->
      updated = %{strategy | status: :deprecated, updated_at: now()}
      entry = audit(:deprecated, id, %{}, actor)
      new_state = %{state |
        strategies: Map.put(state.strategies, id, updated),
        audit_log: [entry | state.audit_log]
      }
      {:reply, :ok, new_state}
    end)
  end

  def handle_call({:update_config, id, new_config, actor}, _from, state) do
    with_strategy(state, id, fn strategy ->
      merged = Map.merge(strategy.config, new_config)
      updated = %{strategy |
        config: merged,
        version: strategy.version + 1,
        updated_at: now()
      }
      entry = audit(:config_updated, id, %{
        old_config: strategy.config, new_config: merged,
        version: updated.version
      }, actor)
      new_state = %{state |
        strategies: Map.put(state.strategies, id, updated),
        audit_log: [entry | state.audit_log]
      }
      {:reply, {:ok, updated.version}, new_state}
    end)
  end

  def handle_call({:get_strategy, id}, _from, state) do
    {:reply, Map.get(state.strategies, id), state}
  end

  def handle_call({:list_strategies, filter}, _from, state) do
    result = state.strategies
      |> Map.values()
      |> filter_strategies(filter)
      |> Enum.sort_by(& &1.name)
    {:reply, result, state}
  end

  def handle_call({:get_performance, id}, _from, state) do
    case Map.get(state.strategies, id) do
      nil -> {:reply, {:error, :not_found}, state}
      s -> {:reply, {:ok, s.performance}, state}
    end
  end

  def handle_call({:start_ab_test, live_id, shadow_config, opts}, _from, state) do
    case Map.get(state.strategies, live_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      live_strategy ->
        shadow_id = generate_id()
        test_id = "ab_" <> shadow_id
        duration = Keyword.get(opts, :duration_ms, 86_400_000)

        shadow = %StrategyState{
          id: shadow_id,
          name: live_strategy.name <> "_shadow",
          status: :shadow,
          config: shadow_config,
          performance: Performance.new(),
          risk_limits: live_strategy.risk_limits,
          dependencies: live_strategy.dependencies,
          shadow_of: live_id,
          created_at: now(),
          updated_at: now(),
          version: 1
        }

        ab_test = %{
          id: test_id,
          live_id: live_id,
          shadow_id: shadow_id,
          started_at: now(),
          duration_ms: duration,
          status: :running
        }

        entry = audit(:ab_test_started, live_id, %{
          test_id: test_id, shadow_id: shadow_id
        }, "system")

        Process.send_after(self(), {:ab_test_complete, test_id}, duration)

        new_state = %{state |
          strategies: Map.put(state.strategies, shadow_id, shadow),
          ab_tests: Map.put(state.ab_tests, test_id, ab_test),
          audit_log: [entry | state.audit_log]
        }
        {:reply, {:ok, test_id, shadow_id}, new_state}
    end
  end

  def handle_call({:ab_results, test_id}, _from, state) do
    case Map.get(state.ab_tests, test_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      test ->
        live_perf = get_in(state.strategies, [test.live_id, :performance])
        shadow_perf = get_in(state.strategies, [test.shadow_id, :performance])

        results = compare_performance(live_perf, shadow_perf)
        {:reply, {:ok, results}, state}
    end
  end

  def handle_call({:promote_shadow, shadow_id, actor}, _from, state) do
    case Map.get(state.strategies, shadow_id) do
      %{status: :shadow, shadow_of: live_id} = shadow ->
        live = Map.get(state.strategies, live_id)
        updated_live = %{live | status: :deprecated, updated_at: now()}
        updated_shadow = %{shadow |
          status: :active,
          shadow_of: nil,
          updated_at: now()
        }

        entry = audit(:shadow_promoted, shadow_id, %{
          replaced: live_id
        }, actor)

        new_strategies = state.strategies
          |> Map.put(live_id, updated_live)
          |> Map.put(shadow_id, updated_shadow)

        new_state = %{state |
          strategies: new_strategies,
          audit_log: [entry | state.audit_log]
        }
        {:reply, :ok, new_state}

      _ ->
        {:reply, {:error, :not_shadow}, state}
    end
  end

  def handle_call({:rotate, opts}, _from, state) do
    window = Keyword.get(opts, :window_days, 30)
    min_sharpe = Keyword.get(opts, :min_sharpe, 0.0)
    max_drawdown = Keyword.get(opts, :max_drawdown, 0.20)

    active = state.strategies
      |> Map.values()
      |> Enum.filter(& &1.status == :active)

    {demote, keep} = Enum.split_with(active, fn s ->
      perf = s.performance
      perf.rolling_sharpe < min_sharpe or perf.max_drawdown > max_drawdown
    end)

    paused = state.strategies
      |> Map.values()
      |> Enum.filter(& &1.status == :paused)
      |> Enum.sort_by(& &1.performance.rolling_sharpe, :desc)

    promote = Enum.take(paused, length(demote))

    new_strategies = Enum.reduce(demote, state.strategies, fn s, acc ->
      Map.put(acc, s.id, %{s | status: :paused, updated_at: now()})
    end)
    new_strategies = Enum.reduce(promote, new_strategies, fn s, acc ->
      Map.put(acc, s.id, %{s | status: :active, updated_at: now()})
    end)

    entries = Enum.map(demote, fn s ->
      audit(:demoted, s.id, %{sharpe: s.performance.rolling_sharpe}, "rotation")
    end) ++ Enum.map(promote, fn s ->
      audit(:promoted, s.id, %{}, "rotation")
    end)

    new_state = %{state |
      strategies: new_strategies,
      audit_log: entries ++ state.audit_log
    }
    {:reply, {:ok, %{demoted: length(demote), promoted: length(promote)}}, new_state}
  end

  def handle_call({:check_risk, id}, _from, state) do
    case Map.get(state.strategies, id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      strategy ->
        violations = check_risk_violations(strategy, state)
        {:reply, {:ok, violations}, state}
    end
  end

  def handle_call({:audit_log, opts}, _from, state) do
    limit = Keyword.get(opts, :limit, 100)
    strategy_id = Keyword.get(opts, :strategy_id, nil)

    log = if strategy_id do
      Enum.filter(state.audit_log, & &1.strategy_id == strategy_id)
    else
      state.audit_log
    end
    {:reply, Enum.take(log, limit), state}
  end

  @impl true
  def handle_cast({:record_trade, strategy_id, trade}, state) do
    case Map.get(state.strategies, strategy_id) do
      nil ->
        {:noreply, state}

      strategy ->
        updated_perf = update_performance(strategy.performance, trade)
        updated = %{strategy | performance: updated_perf, updated_at: now()}

        violations = check_risk_violations(updated, state)
        updated = if length(violations) > 0 do
          Logger.warning("Risk violations for #{strategy_id}: #{inspect(violations)}")
          if Enum.any?(violations, &(&1.severity == :critical)) do
            %{updated | status: :paused}
          else
            updated
          end
        else
          updated
        end

        new_state = %{state |
          strategies: Map.put(state.strategies, strategy_id, updated)
        }
        {:noreply, new_state}
    end
  end

  def handle_cast({:update_book, positions}, state) do
    {:noreply, %{state | book_positions: positions}}
  end

  @impl true
  def handle_info({:unwind_complete, id}, state) do
    case Map.get(state.strategies, id) do
      %{status: :unwinding} = strategy ->
        updated = %{strategy | status: :paused, updated_at: now()}
        entry = audit(:unwind_complete, id, %{}, "system")
        new_state = %{state |
          strategies: Map.put(state.strategies, id, updated),
          audit_log: [entry | state.audit_log]
        }
        {:noreply, new_state}

      _ ->
        {:noreply, state}
    end
  end

  def handle_info({:ab_test_complete, test_id}, state) do
    case Map.get(state.ab_tests, test_id) do
      nil ->
        {:noreply, state}

      test ->
        updated_test = %{test | status: :completed}
        entry = audit(:ab_test_complete, test.live_id, %{test_id: test_id}, "system")
        new_state = %{state |
          ab_tests: Map.put(state.ab_tests, test_id, updated_test),
          audit_log: [entry | state.audit_log]
        }
        {:noreply, new_state}
    end
  end

  def handle_info(:performance_check, state) do
    active = state.strategies
      |> Map.values()
      |> Enum.filter(& &1.status == :active)

    Enum.each(active, fn s ->
      Logger.info("Strategy #{s.name}: Sharpe=#{Float.round(s.performance.rolling_sharpe, 3)} DD=#{Float.round(s.performance.max_drawdown, 4)}")
    end)

    schedule_performance_check()
    {:noreply, state}
  end

  def handle_info(:risk_check, state) do
    active = state.strategies
      |> Map.values()
      |> Enum.filter(& &1.status in [:active, :shadow])

    new_state = Enum.reduce(active, state, fn strategy, acc ->
      violations = check_risk_violations(strategy, acc)
      if Enum.any?(violations, &(&1.severity == :critical)) do
        Logger.error("Critical risk violation for #{strategy.name}, pausing")
        updated = %{strategy | status: :paused, updated_at: now()}
        entry = audit(:risk_paused, strategy.id, %{violations: violations}, "risk_system")
        %{acc |
          strategies: Map.put(acc.strategies, strategy.id, updated),
          audit_log: [entry | acc.audit_log]
        }
      else
        acc
      end
    end)

    schedule_risk_check()
    {:noreply, new_state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # ---------------------------------------------------------------------------
  # Private Helpers
  # ---------------------------------------------------------------------------
  defp generate_id do
    :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
  end

  defp now, do: System.system_time(:millisecond)

  defp audit(action, strategy_id, details, actor) do
    %AuditEntry{
      timestamp: now(),
      strategy_id: strategy_id,
      action: action,
      details: details,
      actor: actor
    }
  end

  defp with_strategy(state, id, fun) do
    case Map.get(state.strategies, id) do
      nil -> {:reply, {:error, :not_found}, state}
      strategy -> fun.(strategy)
    end
  end

  defp filter_strategies(strategies, :all), do: strategies
  defp filter_strategies(strategies, :active),
    do: Enum.filter(strategies, & &1.status == :active)
  defp filter_strategies(strategies, :paused),
    do: Enum.filter(strategies, & &1.status == :paused)
  defp filter_strategies(strategies, status) when is_atom(status),
    do: Enum.filter(strategies, & &1.status == status)

  defp check_dependencies(strategy, state) do
    missing = Enum.filter(strategy.dependencies, fn dep ->
      case Map.get(state.strategies, dep) do
        %{status: :active} -> false
        _ -> true
      end
    end)
    if Enum.empty?(missing), do: :ok, else: {:error, missing}
  end

  defp update_performance(perf, trade) do
    pnl = Map.get(trade, :pnl, 0.0)
    holding = Map.get(trade, :holding_period, 0.0)
    won = if pnl > 0, do: 1, else: 0

    new_count = perf.trade_count + 1
    new_total = perf.total_pnl + pnl
    new_win_rate = (perf.win_rate * perf.trade_count + won) / new_count
    new_avg_hold = (perf.avg_holding_period * perf.trade_count + holding) / new_count

    {new_queue, returns_list} = update_rolling_window(perf.returns_history, pnl, 252)
    new_sharpe = compute_rolling_sharpe(returns_list)
    new_peak = max(perf.peak_equity, new_total)
    new_dd = if new_peak > 0, do: (new_peak - new_total) / new_peak, else: 0.0
    new_max_dd = max(perf.max_drawdown, new_dd)

    %{perf |
      total_pnl: new_total,
      trade_count: new_count,
      win_rate: new_win_rate,
      avg_holding_period: new_avg_hold,
      rolling_sharpe: new_sharpe,
      peak_equity: new_peak,
      current_drawdown: new_dd,
      max_drawdown: new_max_dd,
      returns_history: new_queue,
      pnl_history: [pnl | perf.pnl_history],
      last_trade_at: now()
    }
  end

  defp update_rolling_window(queue, value, max_size) do
    queue = :queue.in(value, queue)
    {queue, list} = if :queue.len(queue) > max_size do
      {{:value, _}, q} = :queue.out(queue)
      {q, :queue.to_list(q)}
    else
      {queue, :queue.to_list(queue)}
    end
    {queue, list}
  end

  defp compute_rolling_sharpe(returns) when length(returns) < 10, do: 0.0
  defp compute_rolling_sharpe(returns) do
    mean_r = Enum.sum(returns) / length(returns)
    var_r = Enum.reduce(returns, 0.0, fn r, acc -> acc + (r - mean_r) * (r - mean_r) end) / (length(returns) - 1)
    std_r = :math.sqrt(max(var_r, 1.0e-15))
    mean_r / std_r * :math.sqrt(252)
  end

  defp check_risk_violations(strategy, state) do
    limits = strategy.risk_limits
    perf = strategy.performance
    violations = []

    violations = if limits && perf.max_drawdown > (limits.max_drawdown_pct || 1.0) do
      [%{type: :max_drawdown, severity: :critical,
         value: perf.max_drawdown, limit: limits.max_drawdown_pct} | violations]
    else
      violations
    end

    violations = if limits && limits.max_daily_loss && daily_loss(perf) > limits.max_daily_loss do
      [%{type: :daily_loss, severity: :critical,
         value: daily_loss(perf), limit: limits.max_daily_loss} | violations]
    else
      violations
    end

    violations = if limits && limits.max_correlation_to_book do
      corr = compute_book_correlation(strategy, state.book_positions)
      if corr > limits.max_correlation_to_book do
        [%{type: :book_correlation, severity: :warning,
           value: corr, limit: limits.max_correlation_to_book} | violations]
      else
        violations
      end
    else
      violations
    end

    violations
  end

  defp daily_loss(perf) do
    today_pnl = perf.pnl_history
      |> Enum.take(50)
      |> Enum.sum()
    abs(min(today_pnl, 0.0))
  end

  defp compute_book_correlation(_strategy, book) when map_size(book) == 0, do: 0.0
  defp compute_book_correlation(_strategy, _book), do: 0.0

  defp compare_performance(live_perf, shadow_perf) do
    %{
      live_sharpe: safe_field(live_perf, :rolling_sharpe),
      shadow_sharpe: safe_field(shadow_perf, :rolling_sharpe),
      live_drawdown: safe_field(live_perf, :max_drawdown),
      shadow_drawdown: safe_field(shadow_perf, :max_drawdown),
      live_win_rate: safe_field(live_perf, :win_rate),
      shadow_win_rate: safe_field(shadow_perf, :win_rate),
      live_pnl: safe_field(live_perf, :total_pnl),
      shadow_pnl: safe_field(shadow_perf, :total_pnl),
      recommendation: recommend(live_perf, shadow_perf)
    }
  end

  defp safe_field(nil, _), do: 0.0
  defp safe_field(perf, field), do: Map.get(perf, field, 0.0)

  defp recommend(nil, _), do: :insufficient_data
  defp recommend(_, nil), do: :insufficient_data
  defp recommend(live, shadow) do
    cond do
      shadow.rolling_sharpe > live.rolling_sharpe * 1.2 and
        shadow.max_drawdown < live.max_drawdown -> :promote_shadow
      shadow.rolling_sharpe < live.rolling_sharpe * 0.8 -> :keep_live
      true -> :inconclusive
    end
  end

  defp build_risk_limits(map) when is_map(map) do
    %RiskLimits{
      max_position_usd: Map.get(map, :max_position_usd, 1_000_000.0),
      max_drawdown_pct: Map.get(map, :max_drawdown_pct, 0.10),
      max_correlation_to_book: Map.get(map, :max_correlation_to_book, 0.7),
      max_daily_loss: Map.get(map, :max_daily_loss, 50_000.0),
      max_open_trades: Map.get(map, :max_open_trades, 100),
      max_notional: Map.get(map, :max_notional, 10_000_000.0),
      position_limit_per_asset: Map.get(map, :position_limit_per_asset, 500_000.0)
    }
  end
  defp build_risk_limits(_), do: build_risk_limits(%{})

  defp default_risk_limits do
    %{max_position_usd: 1_000_000.0, max_drawdown_pct: 0.10,
      max_daily_loss: 50_000.0, max_open_trades: 100}
  end

  defp get_in(map, [key | rest]) do
    case Map.get(map, key) do
      nil -> nil
      val when rest == [] -> val
      val when is_map(val) -> get_in(val, rest)
      val when is_struct(val) -> get_in(Map.from_struct(val), rest)
      _ -> nil
    end
  end

  defp schedule_performance_check do
    Process.send_after(self(), :performance_check, 60_000)
  end

  defp schedule_risk_check do
    Process.send_after(self(), :risk_check, 10_000)
  end
end
