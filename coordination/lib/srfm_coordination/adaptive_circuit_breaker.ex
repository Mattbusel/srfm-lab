defmodule SrfmCoordination.AdaptiveCircuitBreaker do
  @moduledoc """
  Adaptive circuit breaker with learning: closed/open/half-open states,
  adaptive thresholds, exponential backoff, partial degradation,
  health scoring, predictive opening, cascading, metrics, alerts.
  """
  use GenServer
  require Logger

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------
  @type breaker_state :: :closed | :open | :half_open
  @type health_score :: 0..100

  defmodule Config do
    defstruct [
      name: "default",
      failure_threshold: 5,
      success_threshold: 3,
      window_size_ms: 60_000,
      base_timeout_ms: 5_000,
      max_timeout_ms: 120_000,
      backoff_multiplier: 2.0,
      half_open_max_calls: 3,
      adaptive: true,
      adaptive_window: 300_000,
      degradation_enabled: true,
      degraded_rate_pct: 50,
      predictive_enabled: true,
      predictive_lookback: 10,
      predictive_slope_threshold: 0.15,
      cascading_enabled: false,
      upstream_breakers: [],
      alert_callback: nil,
      metrics_callback: nil
    ]
  end

  defmodule Metrics do
    defstruct [
      total_calls: 0,
      total_successes: 0,
      total_failures: 0,
      total_timeouts: 0,
      total_rejections: 0,
      state_transitions: [],
      current_failure_rate: 0.0,
      current_success_rate: 1.0,
      current_latency_avg: 0.0,
      current_latency_p99: 0.0,
      latency_window: [],
      health_score: 100,
      last_failure_at: nil,
      last_success_at: nil,
      recovery_times: [],
      opened_count: 0,
      degraded_since: nil
    ]
  end

  defmodule Window do
    defstruct events: [], size_ms: 60_000

    def new(size_ms), do: %__MODULE__{size_ms: size_ms}

    def add(%__MODULE__{} = w, event) do
      now = System.system_time(:millisecond)
      cutoff = now - w.size_ms
      events = [{event, now} | w.events]
        |> Enum.filter(fn {_, ts} -> ts > cutoff end)
      %{w | events: events}
    end

    def count(%__MODULE__{} = w, type) do
      now = System.system_time(:millisecond)
      cutoff = now - w.size_ms
      w.events
      |> Enum.filter(fn {ev, ts} -> ev == type and ts > cutoff end)
      |> length()
    end

    def total(%__MODULE__{} = w) do
      now = System.system_time(:millisecond)
      cutoff = now - w.size_ms
      Enum.count(w.events, fn {_, ts} -> ts > cutoff end)
    end

    def failure_rate(%__MODULE__{} = w) do
      t = total(w)
      if t == 0, do: 0.0, else: count(w, :failure) / t
    end

    def recent_rates(%__MODULE__{} = w, n_buckets) do
      now = System.system_time(:millisecond)
      bucket_size = div(w.size_ms, n_buckets)
      Enum.map(0..(n_buckets - 1), fn i ->
        lo = now - (i + 1) * bucket_size
        hi = now - i * bucket_size
        bucket_events = Enum.filter(w.events, fn {_, ts} -> ts > lo and ts <= hi end)
        total = length(bucket_events)
        fails = Enum.count(bucket_events, fn {ev, _} -> ev == :failure end)
        if total == 0, do: 0.0, else: fails / total
      end)
      |> Enum.reverse()
    end
  end

  defmodule State do
    defstruct [
      config: %Config{},
      breaker_state: :closed,
      window: nil,
      metrics: %Metrics{},
      current_timeout: 5_000,
      open_since: nil,
      half_open_successes: 0,
      half_open_calls: 0,
      baseline_failure_rate: 0.05,
      adaptive_threshold: 5,
      degraded: false,
      call_counter: 0
    ]
  end

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------
  def start_link(opts \\ []) do
    config = struct(Config, Keyword.get(opts, :config, []))
    name = Keyword.get(opts, :name, String.to_atom("cb_#{config.name}"))
    GenServer.start_link(__MODULE__, config, name: name)
  end

  def call(server, fun, timeout \\ 5_000) do
    GenServer.call(server, {:call, fun, timeout}, timeout + 1_000)
  end

  def state(server), do: GenServer.call(server, :state)
  def metrics(server), do: GenServer.call(server, :metrics)
  def health(server), do: GenServer.call(server, :health)
  def reset(server), do: GenServer.call(server, :reset)
  def force_open(server), do: GenServer.call(server, :force_open)
  def force_close(server), do: GenServer.call(server, :force_close)

  # ---------------------------------------------------------------------------
  # Server
  # ---------------------------------------------------------------------------
  @impl true
  def init(config) do
    state = %State{
      config: config,
      window: Window.new(config.window_size_ms),
      current_timeout: config.base_timeout_ms,
      adaptive_threshold: config.failure_threshold
    }
    schedule_health_check(1_000)
    schedule_adaptive_update(config.adaptive_window)
    {:ok, state}
  end

  @impl true
  def handle_call({:call, fun, timeout}, _from, state) do
    case check_state(state) do
      {:ok, updated_state} ->
        execute_call(fun, timeout, updated_state)

      {:rejected, reason, updated_state} ->
        metrics = %{updated_state.metrics |
          total_rejections: updated_state.metrics.total_rejections + 1
        }
        {:reply, {:error, {:circuit_open, reason}}, %{updated_state | metrics: metrics}}

      {:degraded, updated_state} ->
        if should_allow_degraded(updated_state) do
          execute_call(fun, timeout, updated_state)
        else
          metrics = %{updated_state.metrics |
            total_rejections: updated_state.metrics.total_rejections + 1
          }
          {:reply, {:error, :degraded_rejected}, %{updated_state | metrics: metrics}}
        end
    end
  end

  def handle_call(:state, _from, state) do
    {:reply, state.breaker_state, state}
  end

  def handle_call(:metrics, _from, state) do
    {:reply, state.metrics, state}
  end

  def handle_call(:health, _from, state) do
    {:reply, state.metrics.health_score, state}
  end

  def handle_call(:reset, _from, state) do
    new_state = %{state |
      breaker_state: :closed,
      window: Window.new(state.config.window_size_ms),
      metrics: %Metrics{},
      current_timeout: state.config.base_timeout_ms,
      open_since: nil,
      half_open_successes: 0,
      half_open_calls: 0,
      degraded: false
    }
    {:reply, :ok, new_state}
  end

  def handle_call(:force_open, _from, state) do
    {:reply, :ok, transition_to(:open, state, "forced")}
  end

  def handle_call(:force_close, _from, state) do
    {:reply, :ok, transition_to(:closed, state, "forced")}
  end

  @impl true
  def handle_info(:health_check, state) do
    health = compute_health_score(state)
    metrics = %{state.metrics | health_score: health}

    state = %{state | metrics: metrics}

    state = if state.config.predictive_enabled do
      maybe_predictive_open(state)
    else
      state
    end

    state = if state.config.cascading_enabled do
      check_cascading(state)
    else
      state
    end

    schedule_health_check(1_000)
    {:noreply, state}
  end

  def handle_info(:adaptive_update, state) do
    state = if state.config.adaptive do
      update_adaptive_threshold(state)
    else
      state
    end
    schedule_adaptive_update(state.config.adaptive_window)
    {:noreply, state}
  end

  def handle_info(:try_half_open, state) do
    if state.breaker_state == :open do
      {:noreply, transition_to(:half_open, state, "timeout_elapsed")}
    else
      {:noreply, state}
    end
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # ---------------------------------------------------------------------------
  # Private: State checking
  # ---------------------------------------------------------------------------
  defp check_state(%{breaker_state: :closed} = state) do
    {:ok, state}
  end

  defp check_state(%{breaker_state: :open, degraded: true} = state) do
    {:degraded, state}
  end

  defp check_state(%{breaker_state: :open} = state) do
    {:rejected, :open, state}
  end

  defp check_state(%{breaker_state: :half_open} = state) do
    if state.half_open_calls < state.config.half_open_max_calls do
      {:ok, %{state | half_open_calls: state.half_open_calls + 1}}
    else
      {:rejected, :half_open_full, state}
    end
  end

  # ---------------------------------------------------------------------------
  # Private: Execution
  # ---------------------------------------------------------------------------
  defp execute_call(fun, timeout, state) do
    start_time = System.monotonic_time(:millisecond)

    task = Task.async(fn ->
      try do
        {:ok, fun.()}
      rescue
        e -> {:error, e}
      catch
        kind, reason -> {:error, {kind, reason}}
      end
    end)

    result = case Task.yield(task, timeout) || Task.shutdown(task, :brutal_kill) do
      {:ok, {:ok, value}} ->
        latency = System.monotonic_time(:millisecond) - start_time
        {:success, value, latency}
      {:ok, {:error, error}} ->
        latency = System.monotonic_time(:millisecond) - start_time
        {:failure, error, latency}
      nil ->
        {:timeout, nil, timeout}
    end

    case result do
      {:success, value, latency} ->
        new_state = record_success(state, latency)
        {:reply, {:ok, value}, new_state}

      {:failure, error, latency} ->
        new_state = record_failure(state, latency)
        {:reply, {:error, error}, new_state}

      {:timeout, _, latency} ->
        new_state = record_timeout(state, latency)
        {:reply, {:error, :timeout}, new_state}
    end
  end

  defp record_success(state, latency) do
    window = Window.add(state.window, :success)
    metrics = %{state.metrics |
      total_calls: state.metrics.total_calls + 1,
      total_successes: state.metrics.total_successes + 1,
      current_success_rate: 1.0 - Window.failure_rate(window),
      current_failure_rate: Window.failure_rate(window),
      last_success_at: System.system_time(:millisecond),
      latency_window: Enum.take([latency | state.metrics.latency_window], 100),
      current_latency_avg: update_latency_avg(state.metrics.latency_window, latency)
    }

    state = %{state | window: window, metrics: metrics}

    case state.breaker_state do
      :half_open ->
        new_successes = state.half_open_successes + 1
        if new_successes >= state.config.success_threshold do
          transition_to(:closed, %{state | half_open_successes: 0, half_open_calls: 0}, "recovery")
        else
          %{state | half_open_successes: new_successes}
        end

      _ -> state
    end
  end

  defp record_failure(state, latency) do
    window = Window.add(state.window, :failure)
    now = System.system_time(:millisecond)
    metrics = %{state.metrics |
      total_calls: state.metrics.total_calls + 1,
      total_failures: state.metrics.total_failures + 1,
      current_failure_rate: Window.failure_rate(window),
      current_success_rate: 1.0 - Window.failure_rate(window),
      last_failure_at: now,
      latency_window: Enum.take([latency | state.metrics.latency_window], 100),
      current_latency_avg: update_latency_avg(state.metrics.latency_window, latency)
    }

    state = %{state | window: window, metrics: metrics}

    case state.breaker_state do
      :closed ->
        failures = Window.count(window, :failure)
        threshold = if state.config.adaptive, do: state.adaptive_threshold, else: state.config.failure_threshold
        if failures >= threshold do
          maybe_degrade_or_open(state)
        else
          state
        end

      :half_open ->
        transition_to(:open, state, "half_open_failure")

      :open -> state
    end
  end

  defp record_timeout(state, latency) do
    window = Window.add(state.window, :failure)
    metrics = %{state.metrics |
      total_calls: state.metrics.total_calls + 1,
      total_timeouts: state.metrics.total_timeouts + 1,
      total_failures: state.metrics.total_failures + 1,
      current_failure_rate: Window.failure_rate(window),
      last_failure_at: System.system_time(:millisecond),
      latency_window: Enum.take([latency | state.metrics.latency_window], 100)
    }

    state = %{state | window: window, metrics: metrics}

    case state.breaker_state do
      :closed ->
        failures = Window.count(window, :failure)
        threshold = if state.config.adaptive, do: state.adaptive_threshold, else: state.config.failure_threshold
        if failures >= threshold, do: maybe_degrade_or_open(state), else: state

      :half_open ->
        transition_to(:open, state, "half_open_timeout")

      _ -> state
    end
  end

  # ---------------------------------------------------------------------------
  # Private: State transitions
  # ---------------------------------------------------------------------------
  defp transition_to(new_state, state, reason) do
    old_state = state.breaker_state
    if old_state == new_state, do: state, else: do_transition(new_state, old_state, state, reason)
  end

  defp do_transition(:open, old, state, reason) do
    now = System.system_time(:millisecond)
    timeout = state.current_timeout

    Process.send_after(self(), :try_half_open, timeout)

    metrics = %{state.metrics |
      opened_count: state.metrics.opened_count + 1,
      state_transitions: [{old, :open, now, reason} | state.metrics.state_transitions]
    }

    notify_transition(state, old, :open, reason)

    new_timeout = min(
      round(timeout * state.config.backoff_multiplier),
      state.config.max_timeout_ms
    )

    Logger.warning("Circuit breaker #{state.config.name}: #{old} -> open (#{reason}), next retry in #{timeout}ms")

    %{state |
      breaker_state: :open,
      open_since: now,
      metrics: metrics,
      current_timeout: new_timeout,
      half_open_successes: 0,
      half_open_calls: 0
    }
  end

  defp do_transition(:half_open, old, state, reason) do
    now = System.system_time(:millisecond)
    metrics = %{state.metrics |
      state_transitions: [{old, :half_open, now, reason} | state.metrics.state_transitions]
    }
    notify_transition(state, old, :half_open, reason)
    Logger.info("Circuit breaker #{state.config.name}: #{old} -> half_open")

    %{state |
      breaker_state: :half_open,
      metrics: metrics,
      half_open_successes: 0,
      half_open_calls: 0
    }
  end

  defp do_transition(:closed, old, state, reason) do
    now = System.system_time(:millisecond)

    recovery_time = if state.open_since do
      now - state.open_since
    else
      0
    end

    metrics = %{state.metrics |
      state_transitions: [{old, :closed, now, reason} | state.metrics.state_transitions],
      recovery_times: [recovery_time | Enum.take(state.metrics.recovery_times, 19)]
    }

    notify_transition(state, old, :closed, reason)
    Logger.info("Circuit breaker #{state.config.name}: #{old} -> closed (recovered in #{recovery_time}ms)")

    %{state |
      breaker_state: :closed,
      metrics: metrics,
      current_timeout: state.config.base_timeout_ms,
      open_since: nil,
      degraded: false,
      half_open_successes: 0,
      half_open_calls: 0
    }
  end

  # ---------------------------------------------------------------------------
  # Private: Degradation
  # ---------------------------------------------------------------------------
  defp maybe_degrade_or_open(state) do
    if state.config.degradation_enabled do
      if state.degraded do
        transition_to(:open, state, "degradation_failed")
      else
        Logger.warning("Circuit breaker #{state.config.name}: entering degraded mode")
        metrics = %{state.metrics | degraded_since: System.system_time(:millisecond)}
        %{state | degraded: true, breaker_state: :open, metrics: metrics}
        |> transition_to(:open, "degraded")
      end
    else
      transition_to(:open, state, "threshold_exceeded")
    end
  end

  defp should_allow_degraded(state) do
    counter = state.call_counter + 1
    state = %{state | call_counter: counter}
    rem(counter, 100) < state.config.degraded_rate_pct
  end

  # ---------------------------------------------------------------------------
  # Private: Adaptive threshold
  # ---------------------------------------------------------------------------
  defp update_adaptive_threshold(state) do
    failure_rate = Window.failure_rate(state.window)
    baseline = state.baseline_failure_rate

    new_baseline = baseline * 0.9 + failure_rate * 0.1

    total = Window.total(state.window)
    new_threshold = if total > 20 do
      # Set threshold to 3 standard deviations above baseline
      expected_failures = new_baseline * total
      std_dev = :math.sqrt(new_baseline * (1 - new_baseline) * total)
      max(round(expected_failures + 3 * std_dev), 3)
    else
      state.config.failure_threshold
    end

    %{state |
      baseline_failure_rate: new_baseline,
      adaptive_threshold: new_threshold
    }
  end

  # ---------------------------------------------------------------------------
  # Private: Predictive opening
  # ---------------------------------------------------------------------------
  defp maybe_predictive_open(state) do
    if state.breaker_state != :closed, do: state, else: do_predictive_check(state)
  end

  defp do_predictive_check(state) do
    rates = Window.recent_rates(state.window, state.config.predictive_lookback)

    if length(rates) >= 3 do
      # Linear regression on failure rate trend
      n = length(rates)
      xs = Enum.to_list(0..(n - 1))
      x_mean = Enum.sum(xs) / n
      y_mean = Enum.sum(rates) / n

      ss_xy = Enum.zip(xs, rates)
        |> Enum.reduce(0.0, fn {x, y}, acc -> acc + (x - x_mean) * (y - y_mean) end)
      ss_xx = Enum.reduce(xs, 0.0, fn x, acc -> acc + (x - x_mean) * (x - x_mean) end)

      slope = if ss_xx > 0, do: ss_xy / ss_xx, else: 0.0

      if slope > state.config.predictive_slope_threshold do
        Logger.warning("Circuit breaker #{state.config.name}: predictive opening (slope=#{Float.round(slope, 3)})")
        transition_to(:open, state, "predictive")
      else
        state
      end
    else
      state
    end
  end

  # ---------------------------------------------------------------------------
  # Private: Cascading
  # ---------------------------------------------------------------------------
  defp check_cascading(state) do
    any_open = Enum.any?(state.config.upstream_breakers, fn upstream ->
      try do
        case GenServer.call(upstream, :state, 1_000) do
          :open -> true
          _ -> false
        end
      catch
        _, _ -> false
      end
    end)

    if any_open and state.breaker_state == :closed do
      Logger.warning("Circuit breaker #{state.config.name}: cascading open from upstream")
      transition_to(:open, state, "cascading")
    else
      state
    end
  end

  # ---------------------------------------------------------------------------
  # Private: Health scoring
  # ---------------------------------------------------------------------------
  defp compute_health_score(state) do
    failure_rate = state.metrics.current_failure_rate
    latency_score = compute_latency_score(state.metrics.latency_window)
    timeout_rate = if state.metrics.total_calls > 0 do
      state.metrics.total_timeouts / state.metrics.total_calls
    else
      0.0
    end

    state_penalty = case state.breaker_state do
      :closed -> 0
      :half_open -> 30
      :open -> 60
    end

    base = 100
    score = base
      - round(failure_rate * 40)
      - round((1.0 - latency_score) * 20)
      - round(timeout_rate * 30)
      - state_penalty

    max(min(score, 100), 0)
  end

  defp compute_latency_score([]), do: 1.0
  defp compute_latency_score(latencies) do
    avg = Enum.sum(latencies) / length(latencies)
    # Normalize: < 100ms = 1.0, > 5000ms = 0.0
    score = 1.0 - min(max(avg - 100.0, 0.0) / 4900.0, 1.0)
    score
  end

  defp update_latency_avg(window, new_latency) do
    all = [new_latency | Enum.take(window, 99)]
    Enum.sum(all) / length(all)
  end

  # ---------------------------------------------------------------------------
  # Private: Notifications
  # ---------------------------------------------------------------------------
  defp notify_transition(state, from, to, reason) do
    if state.config.alert_callback do
      try do
        state.config.alert_callback.(%{
          name: state.config.name,
          from: from,
          to: to,
          reason: reason,
          timestamp: System.system_time(:millisecond),
          health: state.metrics.health_score,
          failure_rate: state.metrics.current_failure_rate
        })
      rescue
        _ -> :ok
      end
    end

    if state.config.metrics_callback do
      try do
        state.config.metrics_callback.(%{
          name: state.config.name,
          event: :state_transition,
          from: from,
          to: to,
          metrics: state.metrics
        })
      rescue
        _ -> :ok
      end
    end
  end

  defp schedule_health_check(interval) do
    Process.send_after(self(), :health_check, interval)
  end

  defp schedule_adaptive_update(interval) do
    Process.send_after(self(), :adaptive_update, interval)
  end
end
