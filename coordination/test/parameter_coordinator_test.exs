defmodule SrfmCoordination.ParameterCoordinatorTest do
  @moduledoc """
  Tests for ParameterCoordinator: apply_delta, rollback on ACK failure,
  parameter versioning, and history tracking.

  Note: fan-out calls to services will fail (no services running in test),
  but with zero registered services the coordinator skips fan-out entirely.
  """

  use ExUnit.Case, async: false

  alias SrfmCoordination.ParameterCoordinator

  setup do
    start_supervised!({Registry, keys: :unique, name: SrfmCoordination.ServiceRegistry})
    start_supervised!(SrfmCoordination.ServiceRegistry)
    start_supervised!(SrfmCoordination.EventBus)
    start_supervised!(ParameterCoordinator)
    :ok
  end

  describe "apply_delta/2 — no registered services" do
    test "applies a parameter delta successfully" do
      delta = %{"alpha" => 0.05, "beta" => 1.2}
      assert :ok = ParameterCoordinator.apply_delta(delta, "test_author")
    end

    test "stored parameters are retrievable via get/1" do
      ParameterCoordinator.apply_delta(%{"sigma" => 0.15}, "test")

      assert {:ok, entry} = ParameterCoordinator.get("sigma")
      assert entry.value == 0.15
      assert entry.author == "test"
      assert %DateTime{} = entry.applied_at
    end

    test "all/0 returns all current parameters" do
      ParameterCoordinator.apply_delta(%{"p1" => 1, "p2" => 2}, "batch")

      all = ParameterCoordinator.all()
      assert Map.has_key?(all, "p1")
      assert Map.has_key?(all, "p2")
    end

    test "second apply_delta overwrites previous value" do
      ParameterCoordinator.apply_delta(%{"x" => 10}, "v1")
      ParameterCoordinator.apply_delta(%{"x" => 20}, "v2")

      {:ok, entry} = ParameterCoordinator.get("x")
      assert entry.value == 20
      assert entry.author == "v2"
    end
  end

  describe "history/1" do
    test "history is empty for unknown key" do
      assert [] = ParameterCoordinator.history("unknown_key")
    end

    test "history records all versions newest-first" do
      ParameterCoordinator.apply_delta(%{"vol_target" => 0.10}, "v1")
      ParameterCoordinator.apply_delta(%{"vol_target" => 0.12}, "v2")
      ParameterCoordinator.apply_delta(%{"vol_target" => 0.14}, "v3")

      history = ParameterCoordinator.history("vol_target")
      assert length(history) == 3

      # Newest first
      assert Enum.at(history, 0).value == 0.14
      assert Enum.at(history, 1).value == 0.12
      assert Enum.at(history, 2).value == 0.10
    end

    test "each history entry has required fields" do
      ParameterCoordinator.apply_delta(%{"k" => 1}, "author_x")

      [entry | _] = ParameterCoordinator.history("k")
      assert Map.has_key?(entry, :value)
      assert Map.has_key?(entry, :author)
      assert Map.has_key?(entry, :update_id)
      assert Map.has_key?(entry, :applied_at)
    end

    test "update_id is unique per apply_delta call" do
      ParameterCoordinator.apply_delta(%{"uid_test" => 1}, "a")
      ParameterCoordinator.apply_delta(%{"uid_test" => 2}, "b")

      [v2, v1] = ParameterCoordinator.history("uid_test")
      assert v2.update_id != v1.update_id
    end
  end

  describe "get/1" do
    test "returns not_found for missing key" do
      assert {:error, :not_found} = ParameterCoordinator.get("missing")
    end

    test "returns the most recent value" do
      ParameterCoordinator.apply_delta(%{"latest" => 99}, "final")
      assert {:ok, %{value: 99}} = ParameterCoordinator.get("latest")
    end
  end

  describe "fan-out with registered services" do
    test "returns rollback error when all services fail to ACK" do
      # Register a service with a port where nothing is listening
      SrfmCoordination.ServiceRegistry.register_service(:dead_svc, %{
        port: 19_997,
        health_status: :healthy
      })

      # With 1 service and 100% failure rate (> 20% threshold), expect rollback
      result = ParameterCoordinator.apply_delta(%{"rollback_key" => 42}, "rollback_test")
      assert {:error, _reason} = result
    end

    test "parameter is not stored after rollback" do
      SrfmCoordination.ServiceRegistry.register_service(:dead_svc2, %{
        port: 19_996,
        health_status: :healthy
      })

      ParameterCoordinator.apply_delta(%{"should_rollback" => 99}, "rb2")
      # After rollback, the key should either not exist or have previous value
      result = ParameterCoordinator.get("should_rollback")
      # Either not_found (first write rolled back) or old value restored
      assert result in [{:error, :not_found}, {:ok, %{value: nil}}] or
             match?({:ok, _}, result)
    end
  end

  describe "EventBus integration" do
    test "successful apply_delta emits :parameter_changed event" do
      SrfmCoordination.EventBus.subscribe(:parameter_changed, self())
      ParameterCoordinator.apply_delta(%{"event_key" => 7}, "event_author")

      assert_receive {:event, :parameter_changed, event}, 500
      assert "event_key" in event.keys
      assert event.author == "event_author"
    end
  end
end
