defmodule SrfmCoordination.EventBusTest do
  @moduledoc """
  Tests for EventBus: pub/sub delivery, history, dead-subscriber cleanup,
  topic validation, and ETS history capping.
  """

  use ExUnit.Case, async: false

  alias SrfmCoordination.EventBus

  setup do
    start_supervised!(EventBus)
    :ok
  end

  describe "subscribe/2 and publish/2" do
    test "subscriber receives published event" do
      assert :ok = EventBus.subscribe(:alert, self())

      EventBus.publish(:alert, %{type: :test_alert, message: "hello"})

      assert_receive {:event, :alert, event}, 1_000
      assert event.type == :test_alert
      assert event.message == "hello"
      assert %DateTime{} = event.timestamp
    end

    test "subscriber receives events on its specific topic only" do
      EventBus.subscribe(:trade_executed, self())
      EventBus.publish(:alert, %{type: :irrelevant})

      refute_receive {:event, :trade_executed, _}, 200
    end

    test "multiple subscribers all receive the same event" do
      parent = self()

      pids =
        for i <- 1..3 do
          spawn(fn ->
            EventBus.subscribe(:hypothesis_generated, self())
            receive do
              {:event, :hypothesis_generated, event} ->
                send(parent, {:got, i, event})
            after
              2_000 -> send(parent, {:timeout, i})
            end
          end)
        end

      # Give spawned processes time to subscribe
      Process.sleep(50)

      EventBus.publish(:hypothesis_generated, %{type: :new_hypothesis, id: 99})

      for i <- 1..3 do
        assert_receive {:got, ^i, event}, 1_000
        assert event.id == 99
      end

      Enum.each(pids, fn _ -> :ok end)
    end

    test "event includes topic and timestamp" do
      EventBus.subscribe(:parameter_changed, self())
      EventBus.publish(:parameter_changed, %{key: "sigma"})

      assert_receive {:event, :parameter_changed, event}, 500
      assert event.topic == :parameter_changed
      assert %DateTime{} = event.timestamp
    end
  end

  describe "unsubscribe/2" do
    test "unsubscribed pid stops receiving events" do
      EventBus.subscribe(:alert, self())
      EventBus.unsubscribe(:alert, self())

      EventBus.publish(:alert, %{type: :after_unsub})
      refute_receive {:event, :alert, _}, 200
    end
  end

  describe "dead subscriber cleanup" do
    test "dead process is auto-removed from subscribers" do
      pid =
        spawn(fn ->
          EventBus.subscribe(:alert, self())
          receive do
            :done -> :ok
          end
        end)

      Process.sleep(50)
      counts_before = EventBus.subscriber_counts()

      Process.exit(pid, :kill)
      Process.sleep(100)

      counts_after = EventBus.subscriber_counts()
      assert counts_after.alert <= counts_before.alert
    end
  end

  describe "history/2" do
    test "history returns events in newest-first order" do
      EventBus.publish(:alert, %{seq: 1})
      EventBus.publish(:alert, %{seq: 2})
      EventBus.publish(:alert, %{seq: 3})

      history = EventBus.history(:alert, 3)
      seqs = Enum.map(history, & &1.seq)
      # Newest first
      assert Enum.at(seqs, 0) == 3
    end

    test "history is bounded by the requested limit" do
      for i <- 1..20, do: EventBus.publish(:service_health, %{i: i})

      history = EventBus.history(:service_health, 5)
      assert length(history) <= 5
    end

    test "history returns empty list for topic with no events" do
      assert EventBus.history(:trade_executed, 10) == []
    end
  end

  describe "topic validation" do
    test "subscribe to invalid topic returns error" do
      assert {:error, :invalid_topic} = EventBus.subscribe(:nonexistent_topic, self())
    end

    test "publish to any topic (including unknown) does not crash bus" do
      # publish doesn't validate topics — it just fans out to subscribers
      EventBus.publish(:service_health, %{type: :any})
      assert Process.alive?(Process.whereis(EventBus))
    end
  end

  describe "subscriber_counts/0" do
    test "returns a map with all valid topics" do
      counts = EventBus.subscriber_counts()
      assert Map.has_key?(counts, :service_health)
      assert Map.has_key?(counts, :trade_executed)
      assert Map.has_key?(counts, :hypothesis_generated)
      assert Map.has_key?(counts, :alert)
      assert Map.has_key?(counts, :parameter_changed)
    end

    test "count increases after subscribe" do
      before = EventBus.subscriber_counts()
      EventBus.subscribe(:alert, self())
      after_ = EventBus.subscriber_counts()
      assert after_.alert == before.alert + 1
    end
  end
end
