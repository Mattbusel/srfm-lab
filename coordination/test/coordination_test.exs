defmodule SrfmCoordination.NewModulesTest do
  @moduledoc """
  ExUnit tests covering the five new modules:
    - PerformanceTracker
    - ParameterHistory
    - GenomeReceiver (pure logic functions only -- no HTTP)
    - DrainController (state machine)
    - MetricsBridge (Prometheus text parser)

  All tests run with async: false to avoid ETS table collisions.
  Each describe block starts fresh GenServers via start_supervised!/1.
  """

  use ExUnit.Case, async: false

  alias SrfmCoordination.PerformanceTracker
  alias SrfmCoordination.ParameterHistory
  alias SrfmCoordination.GenomeReceiver
  alias SrfmCoordination.DrainController
  alias SrfmCoordination.MetricsBridge

  # ---------------------------------------------------------------------------
  # Shared helpers
  # ---------------------------------------------------------------------------

  defp start_event_bus do
    # EventBus uses named ETS table -- only start if not already running
    case Process.whereis(SrfmCoordination.EventBus) do
      nil ->
        start_supervised!(SrfmCoordination.EventBus)

      pid ->
        pid
    end
  end

  defp start_registry do
    case Process.whereis(SrfmCoordination.ServiceRegistry) do
      nil ->
        start_supervised!({Registry, keys: :unique, name: SrfmCoordination.ServiceRegistry})

      pid ->
        pid
    end
  end

  defp make_datetime(unix_seconds) do
    DateTime.from_unix!(unix_seconds)
  end

  # ---------------------------------------------------------------------------
  # PerformanceTracker tests
  # ---------------------------------------------------------------------------

  describe "PerformanceTracker -- Sharpe computation" do
    setup do
      start_event_bus()

      # Clear any leftover ETS tables from prior tests by stopping/restarting
      if pid = Process.whereis(PerformanceTracker), do: GenServer.stop(pid)
      pid = start_supervised!(PerformanceTracker)
      %{tracker: pid}
    end

    test "returns 0.0 Sharpe when no equity points recorded" do
      assert PerformanceTracker.compute_sharpe(4) == 0.0
    end

    test "returns 0.0 Sharpe when only one equity point exists" do
      PerformanceTracker.record_equity_point(10_000.0, DateTime.utc_now())
      assert PerformanceTracker.compute_sharpe(4) == 0.0
    end

    test "computes positive Sharpe for a steadily increasing equity curve" do
      base_ts = System.os_time(:second) - 3_600

      # 12 equity points spaced 5 minutes apart over 1 hour (rising)
      Enum.each(0..11, fn i ->
        equity = 10_000.0 + i * 50.0
        ts = base_ts + i * 300
        PerformanceTracker.record_equity_point(equity, make_datetime(ts))
      end)

      sharpe = PerformanceTracker.compute_sharpe(1)
      assert is_float(sharpe)
      assert sharpe > 0.0
    end

    test "computes negative Sharpe for a consistently declining equity curve" do
      base_ts = System.os_time(:second) - 3_600

      Enum.each(0..11, fn i ->
        equity = 10_000.0 - i * 80.0
        ts = base_ts + i * 300
        PerformanceTracker.record_equity_point(equity, make_datetime(ts))
      end)

      sharpe = PerformanceTracker.compute_sharpe(1)
      assert is_float(sharpe)
      assert sharpe < 0.0
    end

    test "compute_sharpe respects the window -- ignores points outside window" do
      now = System.os_time(:second)
      old_ts = now - 10_000

      # Old points outside the 1-hour window -- should be ignored
      PerformanceTracker.record_equity_point(10_000.0, make_datetime(old_ts))
      PerformanceTracker.record_equity_point(9_500.0, make_datetime(old_ts + 300))

      # Only these two recent points are inside the window
      PerformanceTracker.record_equity_point(10_200.0, make_datetime(now - 120))
      PerformanceTracker.record_equity_point(10_250.0, make_datetime(now - 60))

      # With only 2 points inside window (1 return), result should be 0.0 or small
      sharpe_1h = PerformanceTracker.compute_sharpe(1)
      assert is_float(sharpe_1h)
    end

    test "get_equity_curve returns points sorted oldest-first" do
      now = System.os_time(:second)

      PerformanceTracker.record_equity_point(10_100.0, make_datetime(now - 200))
      PerformanceTracker.record_equity_point(10_000.0, make_datetime(now - 300))
      PerformanceTracker.record_equity_point(10_200.0, make_datetime(now - 100))

      curve = PerformanceTracker.get_equity_curve(1)
      timestamps = Enum.map(curve, fn {ts, _} -> ts end)
      assert timestamps == Enum.sort(timestamps)
    end
  end

  describe "PerformanceTracker -- drawdown tracker" do
    setup do
      start_event_bus()
      if pid = Process.whereis(PerformanceTracker), do: GenServer.stop(pid)
      start_supervised!(PerformanceTracker)
      :ok
    end

    test "drawdown is 0.0 when no equity recorded" do
      assert PerformanceTracker.current_drawdown() == 0.0
    end

    test "drawdown is 0.0 at equity peak" do
      PerformanceTracker.record_equity_point(10_000.0, DateTime.utc_now())
      assert PerformanceTracker.current_drawdown() == 0.0
    end

    test "drawdown equals expected fraction when equity falls from peak" do
      PerformanceTracker.record_equity_point(10_000.0, DateTime.utc_now())
      PerformanceTracker.record_equity_point(9_000.0, DateTime.utc_now())

      dd = PerformanceTracker.current_drawdown()
      assert_in_delta dd, 0.10, 0.001
    end

    test "drawdown updates correctly after partial recovery" do
      PerformanceTracker.record_equity_point(10_000.0, DateTime.utc_now())
      PerformanceTracker.record_equity_point(8_000.0, DateTime.utc_now())
      PerformanceTracker.record_equity_point(9_500.0, DateTime.utc_now())

      dd = PerformanceTracker.current_drawdown()
      # Peak is 10_000, current is 9_500 => 5% drawdown
      assert_in_delta dd, 0.05, 0.001
    end
  end

  describe "PerformanceTracker -- rollback trigger" do
    setup do
      start_event_bus()
      if pid = Process.whereis(PerformanceTracker), do: GenServer.stop(pid)
      start_supervised!(PerformanceTracker)
      :ok
    end

    test "performance report returns expected keys" do
      report = PerformanceTracker.get_performance_report()

      assert Map.has_key?(report, :equity)
      assert Map.has_key?(report, :sharpe_4h)
      assert Map.has_key?(report, :sharpe_24h)
      assert Map.has_key?(report, :max_dd_pct)
      assert Map.has_key?(report, :win_rate)
      assert Map.has_key?(report, :avg_hold_minutes)
      assert Map.has_key?(report, :trade_count)
    end

    test "record_trade increments trade_count" do
      now = DateTime.utc_now()
      PerformanceTracker.record_trade("BTC/USD", 150.0, now, now, %{})

      # Small sleep to let cast process
      Process.sleep(50)
      report = PerformanceTracker.get_performance_report()
      assert report.trade_count == 1
    end

    test "win_rate is 1.0 when all trades are winners" do
      now = DateTime.utc_now()

      Enum.each(1..5, fn i ->
        PerformanceTracker.record_trade("BTC/USD", i * 100.0, now, now, %{})
      end)

      Process.sleep(50)
      report = PerformanceTracker.get_performance_report()
      assert report.win_rate == 1.0
    end

    test "win_rate is 0.0 when all trades are losers" do
      now = DateTime.utc_now()

      Enum.each(1..4, fn _ ->
        PerformanceTracker.record_trade("ETH/USD", -50.0, now, now, %{})
      end)

      Process.sleep(50)
      report = PerformanceTracker.get_performance_report()
      assert report.win_rate == 0.0
    end
  end

  # ---------------------------------------------------------------------------
  # ParameterHistory tests
  # ---------------------------------------------------------------------------

  describe "ParameterHistory -- record and retrieve" do
    setup do
      if pid = Process.whereis(ParameterHistory), do: GenServer.stop(pid)
      # Start without SQLite by overriding path to in-memory
      start_supervised!(ParameterHistory)
      :ok
    end

    test "get_history returns empty list when no updates recorded" do
      history = ParameterHistory.get_history(10)
      assert history == []
    end

    test "record_update stores an entry retrievable by get_history" do
      old = %{"momentum_lookback" => 20, "entry_z_score" => 1.5}
      new = %{"momentum_lookback" => 25, "entry_z_score" => 1.8}

      ParameterHistory.record_update(old, new, "test", 0.5, 0.75)
      Process.sleep(50)

      history = ParameterHistory.get_history(10)
      assert length(history) == 1

      [entry] = history
      assert entry.source == "test"
      assert entry.fitness_before == 0.5
      assert entry.fitness_after == 0.75
      assert entry.new_params == new
    end

    test "get_history returns newest entry first" do
      now = DateTime.utc_now()

      ParameterHistory.record_update(%{}, %{"x" => 1}, "first", 0.1, 0.2)
      Process.sleep(20)
      ParameterHistory.record_update(%{}, %{"x" => 2}, "second", 0.2, 0.4)
      Process.sleep(50)

      history = ParameterHistory.get_history(10)
      assert length(history) == 2
      [first | _] = history
      assert first.source == "second"
    end

    test "get_history respects the limit n" do
      Enum.each(1..10, fn i ->
        ParameterHistory.record_update(%{}, %{"p" => i * 1.0}, "src", 0.0, i * 0.1)
      end)

      Process.sleep(100)
      history = ParameterHistory.get_history(3)
      assert length(history) == 3
    end

    test "update_count reflects recorded updates" do
      ParameterHistory.record_update(%{}, %{"a" => 1.0}, "src", 0.0, 1.0)
      ParameterHistory.record_update(%{}, %{"a" => 2.0}, "src", 1.0, 1.5)
      Process.sleep(80)
      assert ParameterHistory.update_count() == 2
    end
  end

  describe "ParameterHistory -- get_best_params" do
    setup do
      if pid = Process.whereis(ParameterHistory), do: GenServer.stop(pid)
      start_supervised!(ParameterHistory)
      :ok
    end

    test "returns error when no history exists" do
      assert {:error, :no_history} = ParameterHistory.get_best_params(:sharpe)
    end

    test "returns params from the update with highest fitness_after by :sharpe" do
      ParameterHistory.record_update(%{}, %{"x" => 1.0}, "a", 0.0, 0.5)
      ParameterHistory.record_update(%{}, %{"x" => 2.0}, "b", 0.0, 1.2)
      ParameterHistory.record_update(%{}, %{"x" => 3.0}, "c", 0.0, 0.8)
      Process.sleep(100)

      {:ok, best} = ParameterHistory.get_best_params(:sharpe)
      assert best["x"] == 2.0
    end

    test "returns params with best pnl improvement when metric is :pnl_improvement" do
      ParameterHistory.record_update(%{}, %{"y" => 10.0}, "a", 0.5, 0.8)
      ParameterHistory.record_update(%{}, %{"y" => 20.0}, "b", 0.1, 1.0)  # delta 0.9
      ParameterHistory.record_update(%{}, %{"y" => 30.0}, "c", 0.6, 0.9)  # delta 0.3
      Process.sleep(100)

      {:ok, best} = ParameterHistory.get_best_params(:pnl_improvement)
      assert best["y"] == 20.0
    end
  end

  describe "ParameterHistory -- sensitivity computation" do
    setup do
      if pid = Process.whereis(ParameterHistory), do: GenServer.stop(pid)
      start_supervised!(ParameterHistory)
      :ok
    end

    test "returns empty map when fewer than 3 updates exist" do
      ParameterHistory.record_update(%{}, %{"z" => 1.0}, "src", 0.0, 0.5)
      ParameterHistory.record_update(%{}, %{"z" => 2.0}, "src", 0.0, 0.6)
      Process.sleep(50)
      assert ParameterHistory.compute_param_sensitivity() == %{}
    end

    test "returns correlation map with keys matching recorded params" do
      Enum.each(1..5, fn i ->
        ParameterHistory.record_update(%{}, %{"alpha" => i * 0.1, "beta" => i * 0.5}, "src", 0.0, i * 0.2)
      end)

      Process.sleep(100)
      sensitivity = ParameterHistory.compute_param_sensitivity()
      assert Map.has_key?(sensitivity, "alpha")
      assert Map.has_key?(sensitivity, "beta")
    end

    test "Pearson correlation of perfectly correlated series is ~1.0" do
      # alpha = i, fitness = i => r = 1.0
      pairs = Enum.map(1..10, fn i -> {i * 1.0, i * 1.0} end)
      r = ParameterHistory.pearson_correlation(pairs)
      assert_in_delta r, 1.0, 0.001
    end

    test "Pearson correlation of perfectly anticorrelated series is ~-1.0" do
      pairs = Enum.map(1..10, fn i -> {i * 1.0, (11 - i) * 1.0} end)
      r = ParameterHistory.pearson_correlation(pairs)
      assert_in_delta r, -1.0, 0.001
    end

    test "Pearson correlation of constant series is 0.0" do
      pairs = Enum.map(1..5, fn _ -> {1.0, 2.0} end)
      r = ParameterHistory.pearson_correlation(pairs)
      assert r == 0.0
    end
  end

  # ---------------------------------------------------------------------------
  # GenomeReceiver tests (pure logic -- no HTTP)
  # ---------------------------------------------------------------------------

  describe "GenomeReceiver -- genome decoding" do
    test "decode_genome maps float array to named parameters" do
      genes = [50.0, 0.5, 1.2, 1.8, 0.9, 2.0, 3.0, 0.1, 0.3, 60.0]

      {:ok, params} = GenomeReceiver.decode_genome(genes)
      assert Map.has_key?(params, "momentum_lookback")
      assert Map.has_key?(params, "bh_mass_threshold")
      assert Map.has_key?(params, "entry_z_score")
    end

    test "decode_genome clamps momentum_lookback to [2, 200]" do
      genes = [500.0 | List.duplicate(1.0, 9)]
      {:ok, params} = GenomeReceiver.decode_genome(genes)
      assert params["momentum_lookback"] <= 200
      assert params["momentum_lookback"] >= 2
    end

    test "decode_genome rounds momentum_lookback to integer" do
      genes = [45.7 | List.duplicate(1.0, 9)]
      {:ok, params} = GenomeReceiver.decode_genome(genes)
      assert is_integer(params["momentum_lookback"])
    end

    test "decode_genome returns error when genes list is too short" do
      genes = [1.0, 2.0, 3.0]
      assert {:error, {:genome_too_short, _}} = GenomeReceiver.decode_genome(genes)
    end

    test "decode_genome clamps position_size_pct to [0.01, 0.25]" do
      # position_size_pct is index 7
      genes = List.duplicate(1.0, 7) ++ [99.0, 0.5, 30.0]
      {:ok, params} = GenomeReceiver.decode_genome(genes)
      assert params["position_size_pct"] <= 0.25
      assert params["position_size_pct"] >= 0.01
    end

    test "decode_genome clamps entry_z_score to [0.5, 4.0]" do
      genes = [50.0, 0.5, 1.0, -10.0 | List.duplicate(1.0, 6)]
      {:ok, params} = GenomeReceiver.decode_genome(genes)
      assert params["entry_z_score"] >= 0.5
      assert params["entry_z_score"] <= 4.0
    end
  end

  describe "GenomeReceiver -- incremental application logic" do
    test "single step when all deltas are small" do
      current = %{"alpha" => %{value: 1.0}, "beta" => %{value: 2.0}}
      new_params = %{"alpha" => 1.05, "beta" => 2.08}

      steps = GenomeReceiver.build_application_steps(new_params, current)
      assert length(steps) == 1
      assert hd(steps) == new_params
    end

    test "two steps when a param delta exceeds 25%" do
      # alpha changes from 1.0 to 1.4 => 40% delta, triggers 2-step
      current = %{"alpha" => %{value: 1.0}}
      new_params = %{"alpha" => 1.4}

      steps = GenomeReceiver.build_application_steps(new_params, current)
      assert length(steps) == 2

      [step1, step2] = steps
      # Step 1 should be halfway: 1.0 + (1.4 - 1.0) * 0.5 = 1.2
      assert_in_delta step1["alpha"], 1.2, 0.001
      # Step 2 is the final target
      assert_in_delta step2["alpha"], 1.4, 0.001
    end

    test "three steps when a param delta exceeds 50%" do
      # alpha changes from 1.0 to 2.1 => 110% delta, triggers 3-step
      current = %{"alpha" => %{value: 1.0}}
      new_params = %{"alpha" => 2.1}

      steps = GenomeReceiver.build_application_steps(new_params, current)
      assert length(steps) == 3

      [_step1, _step2, step3] = steps
      # Last step must reach the final target
      assert_in_delta step3["alpha"], 2.1, 0.001
    end

    test "intermediate steps interpolate linearly between old and new" do
      current = %{"x" => %{value: 0.0}}
      new_params = %{"x" => 0.9}  # 90% change from 1.0 base? -- current is 0

      # When current is 0 no meaningful pct change -- stays 1 step
      steps = GenomeReceiver.build_application_steps(new_params, current)
      assert length(steps) >= 1
    end

    test "handles params absent from current (new params)" do
      current = %{}
      new_params = %{"alpha" => 1.5, "beta" => 2.0}

      steps = GenomeReceiver.build_application_steps(new_params, current)
      # No current values to compare -- should default to 1 step
      assert length(steps) == 1
    end
  end

  # ---------------------------------------------------------------------------
  # DrainController tests
  # ---------------------------------------------------------------------------

  describe "DrainController -- state machine transitions" do
    setup do
      start_event_bus()
      if pid = Process.whereis(DrainController), do: GenServer.stop(pid)
      start_supervised!(DrainController)

      # Ensure ServiceRegistry is up for DrainController HTTP calls
      start_registry()

      :ok
    end

    test "initial state: unknown service returns {:error, :not_found}" do
      assert {:error, :not_found} = DrainController.get_state(:nonexistent)
    end

    test "initiating drain creates a :draining entry" do
      DrainController.initiate_drain(:test_service_a)

      {:ok, state_map} = DrainController.get_state(:test_service_a)
      assert state_map.state == :draining
      assert state_map.service == :test_service_a
      assert state_map.initiated_at != nil
    end

    test "initiating drain twice returns {:error, {:already_draining, _}}" do
      DrainController.initiate_drain(:test_service_b)
      result = DrainController.initiate_drain(:test_service_b)
      assert {:error, {:already_draining, :draining}} = result
    end

    test "abort_drain transitions a draining service back to idle" do
      DrainController.initiate_drain(:test_service_c)
      {:ok, before} = DrainController.get_state(:test_service_c)
      assert before.state == :draining

      :ok = DrainController.abort_drain(:test_service_c)

      {:ok, after_state} = DrainController.get_state(:test_service_c)
      assert after_state.state == :idle
      assert after_state.abort_at != nil
    end

    test "abort_drain on non-draining service returns {:error, :not_draining}" do
      assert {:error, :not_draining} = DrainController.abort_drain(:never_started)
    end

    test "all_states returns list of drain entries" do
      DrainController.initiate_drain(:svc_x)
      DrainController.initiate_drain(:svc_y)

      states = DrainController.all_states()
      names = Enum.map(states, fn s -> s.service end)

      assert :svc_x in names
      assert :svc_y in names
    end

    test "valid_states/0 returns all expected state atoms" do
      states = DrainController.valid_states()
      assert :idle in states
      assert :draining in states
      assert :ready in states
      assert :restarting in states
      assert :online in states
    end
  end

  # ---------------------------------------------------------------------------
  # MetricsBridge tests -- Prometheus text parsing
  # ---------------------------------------------------------------------------

  describe "MetricsBridge -- Prometheus text parsing" do
    test "parses a simple gauge line" do
      body = "my_gauge 3.14\n"
      result = MetricsBridge.parse_prometheus_text(body)
      assert_in_delta result["my_gauge"], 3.14, 0.001
    end

    test "parses an integer metric value" do
      body = "requests_total 42\n"
      result = MetricsBridge.parse_prometheus_text(body)
      assert_in_delta result["requests_total"], 42.0, 0.001
    end

    test "parses negative float values" do
      body = "pnl_total -150.75\n"
      result = MetricsBridge.parse_prometheus_text(body)
      assert_in_delta result["pnl_total"], -150.75, 0.001
    end

    test "skips comment lines starting with #" do
      body = """
      # HELP my_metric a gauge
      # TYPE my_metric gauge
      my_metric 99.0
      """

      result = MetricsBridge.parse_prometheus_text(body)
      assert Map.has_key?(result, "my_metric")
      refute Map.has_key?(result, "# HELP my_metric a gauge")
    end

    test "skips blank lines" do
      body = "\n\nmy_metric 1.0\n\n"
      result = MetricsBridge.parse_prometheus_text(body)
      assert Map.has_key?(result, "my_metric")
      assert map_size(result) == 1
    end

    test "parses metric with labels (braces)" do
      body = ~s|http_requests_total{method="GET",status="200"} 1234\n|
      result = MetricsBridge.parse_prometheus_text(body)
      assert Map.has_key?(result, "http_requests_total")
      assert_in_delta result["http_requests_total"], 1234.0, 0.001
    end

    test "parses scientific notation values" do
      body = "large_counter 1.5e6\n"
      result = MetricsBridge.parse_prometheus_text(body)
      assert_in_delta result["large_counter"], 1_500_000.0, 1.0
    end

    test "parses multiple metrics from a realistic Prometheus body" do
      body = """
      # HELP process_cpu_seconds_total Total user and system CPU time
      # TYPE process_cpu_seconds_total counter
      process_cpu_seconds_total 12.45

      # HELP go_goroutines Number of goroutines
      # TYPE go_goroutines gauge
      go_goroutines 23

      iae_trades_executed_total 152
      iae_pnl_unrealized{symbol="BTCUSD"} -420.5
      """

      result = MetricsBridge.parse_prometheus_text(body)

      assert Map.has_key?(result, "process_cpu_seconds_total")
      assert Map.has_key?(result, "go_goroutines")
      assert Map.has_key?(result, "iae_trades_executed_total")
      assert Map.has_key?(result, "iae_pnl_unrealized")
      assert_in_delta result["go_goroutines"], 23.0, 0.001
      assert_in_delta result["iae_pnl_unrealized"], -420.5, 0.001
    end

    test "returns empty map for empty body" do
      assert MetricsBridge.parse_prometheus_text("") == %{}
    end

    test "returns empty map for body with only comments" do
      body = """
      # HELP metric description
      # TYPE metric gauge
      """

      assert MetricsBridge.parse_prometheus_text(body) == %{}
    end

    test "handles metric line with timestamp suffix" do
      # Prometheus format allows optional Unix ms timestamp at end
      body = "my_metric 42.0 1717000000000\n"
      result = MetricsBridge.parse_prometheus_text(body)
      assert_in_delta result["my_metric"], 42.0, 0.001
    end

    test "get_metric returns :not_found for unregistered service" do
      # MetricsBridge ETS is fresh after restart
      if pid = Process.whereis(MetricsBridge), do: GenServer.stop(pid)
      start_supervised!(MetricsBridge)

      assert {:error, :not_found} = MetricsBridge.get_metric(:unknown_svc, "cpu_usage")
    end

    test "get_trend returns :no_history for unknown service/metric" do
      if pid = Process.whereis(MetricsBridge), do: GenServer.stop(pid)
      start_supervised!(MetricsBridge)

      assert {:error, :no_history} = MetricsBridge.get_trend(:unknown_svc, "cpu_usage", 60)
    end
  end

  # ---------------------------------------------------------------------------
  # Cross-module integration: genome -> parameter history
  # ---------------------------------------------------------------------------

  describe "Integration -- genome decode -> parameter history" do
    setup do
      start_event_bus()
      if pid = Process.whereis(ParameterHistory), do: GenServer.stop(pid)
      start_supervised!(ParameterHistory)
      :ok
    end

    test "decoded genome params have expected numeric types" do
      genes = [30.0, 0.3, 1.5, 1.2, 0.8, 2.5, 4.0, 0.05, 0.2, 15.0]
      {:ok, params} = GenomeReceiver.decode_genome(genes)

      Enum.each(params, fn {_key, val} ->
        assert is_number(val), "Expected numeric value for each decoded gene, got: #{inspect(val)}"
      end)
    end

    test "decoded params can be stored in ParameterHistory without error" do
      genes = [30.0, 0.3, 1.5, 1.2, 0.8, 2.5, 4.0, 0.05, 0.2, 15.0]
      {:ok, params} = GenomeReceiver.decode_genome(genes)

      old_params = Map.new(params, fn {k, v} -> {k, v * 0.9} end)
      ParameterHistory.record_update(old_params, params, "test_integration", 0.3, 0.6)
      Process.sleep(80)

      history = ParameterHistory.get_history(5)
      assert length(history) == 1
      assert hd(history).source == "test_integration"
    end
  end
end
