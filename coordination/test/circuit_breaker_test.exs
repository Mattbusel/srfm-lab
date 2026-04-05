defmodule SrfmCoordination.CircuitBreakerTest do
  @moduledoc """
  Tests for the CircuitBreaker state machine.

  State transitions tested:
    CLOSED -> OPEN      (after 3 failures in window — test config threshold)
    OPEN -> HALF_OPEN   (after cooldown)
    HALF_OPEN -> CLOSED (successful probe)
    HALF_OPEN -> OPEN   (failed probe)

  Circuit breakers are started in isolation using start_supervised!/1
  with a unique Registry per test to avoid name collisions.
  """

  use ExUnit.Case, async: false

  alias SrfmCoordination.CircuitBreaker

  # We start a standalone Registry and CircuitBreaker per test
  setup do
    start_supervised!({Registry, keys: :unique, name: SrfmCoordination.ServiceRegistry})
    :ok
  end

  defp start_breaker(name) do
    start_supervised!({CircuitBreaker, name: name})
    name
  end

  defp fail_n_times(circuit, n) do
    Enum.each(1..n, fn _ ->
      CircuitBreaker.call(circuit, fn -> raise "simulated failure" end)
    end)
  end

  describe "CLOSED state" do
    test "allows calls and returns {:ok, result}" do
      c = start_breaker(:test_cb_ok)
      assert {:ok, 42} = CircuitBreaker.call(c, fn -> 42 end)
    end

    test "accumulates failures without opening below threshold" do
      c = start_breaker(:test_cb_below)
      # threshold is 3 in test config; fire 2 failures
      CircuitBreaker.call(c, fn -> raise "err" end)
      CircuitBreaker.call(c, fn -> raise "err" end)

      status = CircuitBreaker.status(c)
      assert status.state == :closed
    end

    test "opens after reaching failure threshold" do
      c = start_breaker(:test_cb_open)
      fail_n_times(c, 3)

      status = CircuitBreaker.status(c)
      assert status.state == :open
    end

    test "success resets failure count" do
      c = start_breaker(:test_cb_reset)
      CircuitBreaker.call(c, fn -> raise "err" end)
      CircuitBreaker.call(c, fn -> raise "err" end)
      # A success should prevent opening
      CircuitBreaker.call(c, fn -> :ok end)
      fail_n_times(c, 2)

      status = CircuitBreaker.status(c)
      assert status.state == :closed
    end
  end

  describe "OPEN state" do
    test "rejects calls immediately" do
      c = start_breaker(:test_cb_reject)
      fail_n_times(c, 3)

      result = CircuitBreaker.call(c, fn -> :should_not_run end)
      assert result == {:error, :circuit_open}
    end

    test "tracks total rejections" do
      c = start_breaker(:test_cb_rejcount)
      fail_n_times(c, 3)

      CircuitBreaker.call(c, fn -> :nope end)
      CircuitBreaker.call(c, fn -> :nope end)

      status = CircuitBreaker.status(c)
      assert status.total_rejections >= 2
    end

    test "transitions to HALF_OPEN after cooldown message" do
      c = start_breaker(:test_cb_halfopen)
      fail_n_times(c, 3)

      # Simulate the cooldown expiry message directly
      pid = Process.whereis({:via, Registry, {SrfmCoordination.ServiceRegistry, {:circuit, c}}})
      send(pid, :cooldown_expired)
      Process.sleep(50)

      status = CircuitBreaker.status(c)
      assert status.state == :half_open
    end
  end

  describe "HALF_OPEN state" do
    defp force_half_open(circuit) do
      fail_n_times(circuit, 3)
      pid = Process.whereis({:via, Registry, {SrfmCoordination.ServiceRegistry, {:circuit, circuit}}})
      send(pid, :cooldown_expired)
      Process.sleep(50)
    end

    test "successful probe closes the circuit" do
      c = start_breaker(:test_cb_probe_ok)
      force_half_open(c)

      result = CircuitBreaker.call(c, fn -> :recovered end)
      assert {:ok, :recovered} = result

      status = CircuitBreaker.status(c)
      assert status.state == :closed
    end

    test "failed probe reopens the circuit" do
      c = start_breaker(:test_cb_probe_fail)
      force_half_open(c)

      CircuitBreaker.call(c, fn -> raise "still broken" end)

      status = CircuitBreaker.status(c)
      assert status.state == :open
    end
  end

  describe "manual reset" do
    test "reset/1 forces circuit to CLOSED from OPEN" do
      c = start_breaker(:test_cb_manual_reset)
      fail_n_times(c, 3)

      assert CircuitBreaker.status(c).state == :open
      CircuitBreaker.reset(c)
      assert CircuitBreaker.status(c).state == :closed
    end
  end

  describe "status/1" do
    test "returns all expected fields" do
      c = start_breaker(:test_cb_status)
      status = CircuitBreaker.status(c)

      assert Map.has_key?(status, :name)
      assert Map.has_key?(status, :state)
      assert Map.has_key?(status, :failure_count_in_window)
      assert Map.has_key?(status, :total_calls)
      assert Map.has_key?(status, :total_failures)
      assert Map.has_key?(status, :total_rejections)
      assert Map.has_key?(status, :time_in_state_ms)
    end

    test "total_calls increments on each call" do
      c = start_breaker(:test_cb_total_calls)
      CircuitBreaker.call(c, fn -> :a end)
      CircuitBreaker.call(c, fn -> :b end)
      CircuitBreaker.call(c, fn -> raise "c" end)

      status = CircuitBreaker.status(c)
      assert status.total_calls == 3
      assert status.total_failures == 1
    end
  end
end
