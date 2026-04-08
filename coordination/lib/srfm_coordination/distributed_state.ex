defmodule SrfmCoordination.DistributedState do
  @moduledoc """
  Distributed state management with CRDTs, gossip sync, vector clocks,
  partition tolerance, snapshots, recovery, cluster membership, leader
  election, distributed locks, and health-aware routing.
  """
  use GenServer
  require Logger

  # ---------------------------------------------------------------------------
  # CRDT Types
  # ---------------------------------------------------------------------------
  defmodule GCounter do
    defstruct entries: %{}

    def new, do: %__MODULE__{}

    def increment(%__MODULE__{entries: e} = c, node, amount \\ 1) do
      %{c | entries: Map.update(e, node, amount, &(&1 + amount))}
    end

    def value(%__MODULE__{entries: e}), do: Map.values(e) |> Enum.sum()

    def merge(%__MODULE__{entries: a}, %__MODULE__{entries: b}) do
      merged = Map.merge(a, b, fn _k, v1, v2 -> max(v1, v2) end)
      %__MODULE__{entries: merged}
    end
  end

  defmodule PNCounter do
    defstruct positive: %GCounter{}, negative: %GCounter{}

    def new, do: %__MODULE__{positive: GCounter.new(), negative: GCounter.new()}

    def increment(%__MODULE__{} = c, node, amount \\ 1) do
      %{c | positive: GCounter.increment(c.positive, node, amount)}
    end

    def decrement(%__MODULE__{} = c, node, amount \\ 1) do
      %{c | negative: GCounter.increment(c.negative, node, amount)}
    end

    def value(%__MODULE__{positive: p, negative: n}),
      do: GCounter.value(p) - GCounter.value(n)

    def merge(%__MODULE__{} = a, %__MODULE__{} = b) do
      %__MODULE__{
        positive: GCounter.merge(a.positive, b.positive),
        negative: GCounter.merge(a.negative, b.negative)
      }
    end
  end

  defmodule LWWRegister do
    defstruct value: nil, timestamp: 0

    def new(value \\ nil), do: %__MODULE__{value: value, timestamp: System.system_time(:microsecond)}

    def set(%__MODULE__{} = r, value) do
      %{r | value: value, timestamp: System.system_time(:microsecond)}
    end

    def merge(%__MODULE__{} = a, %__MODULE__{} = b) do
      if a.timestamp >= b.timestamp, do: a, else: b
    end
  end

  defmodule ORSet do
    defstruct entries: %{}

    def new, do: %__MODULE__{}

    def add(%__MODULE__{entries: e} = s, element) do
      tag = :crypto.strong_rand_bytes(8) |> Base.encode16()
      existing = Map.get(e, element, MapSet.new())
      %{s | entries: Map.put(e, element, MapSet.put(existing, tag))}
    end

    def remove(%__MODULE__{entries: e} = s, element) do
      %{s | entries: Map.delete(e, element)}
    end

    def contains?(%__MODULE__{entries: e}, element), do: Map.has_key?(e, element)

    def elements(%__MODULE__{entries: e}), do: Map.keys(e)

    def merge(%__MODULE__{entries: a}, %__MODULE__{entries: b}) do
      merged = Map.merge(a, b, fn _k, v1, v2 -> MapSet.union(v1, v2) end)
      %__MODULE__{entries: merged}
    end
  end

  # ---------------------------------------------------------------------------
  # Vector Clock
  # ---------------------------------------------------------------------------
  defmodule VectorClock do
    defstruct entries: %{}

    def new, do: %__MODULE__{}

    def increment(%__MODULE__{entries: e} = vc, node) do
      %{vc | entries: Map.update(e, node, 1, &(&1 + 1))}
    end

    def merge(%__MODULE__{entries: a}, %__MODULE__{entries: b}) do
      merged = Map.merge(a, b, fn _k, v1, v2 -> max(v1, v2) end)
      %__MODULE__{entries: merged}
    end

    def compare(%__MODULE__{entries: a}, %__MODULE__{entries: b}) do
      all_keys = MapSet.union(MapSet.new(Map.keys(a)), MapSet.new(Map.keys(b)))
      {lt, gt} = Enum.reduce(all_keys, {false, false}, fn k, {lt_acc, gt_acc} ->
        va = Map.get(a, k, 0)
        vb = Map.get(b, k, 0)
        {lt_acc or va < vb, gt_acc or va > vb}
      end)
      cond do
        lt and not gt -> :before
        gt and not lt -> :after
        not lt and not gt -> :equal
        true -> :concurrent
      end
    end

    def dominates?(a, b), do: compare(a, b) == :after
  end

  # ---------------------------------------------------------------------------
  # State
  # ---------------------------------------------------------------------------
  defmodule ClusterMember do
    defstruct [:node, :status, :last_heartbeat, :health_score, :joined_at]
    @type status :: :alive | :suspect | :dead
  end

  defmodule Lock do
    defstruct [:key, :owner, :acquired_at, :ttl_ms]
  end

  defmodule ServerState do
    defstruct [
      node_id: nil,
      crdts: %{},
      vector_clock: %VectorClock{},
      members: %{},
      leader: nil,
      locks: %{},
      snapshot_dir: "/tmp/srfm_state",
      event_log: [],
      gossip_interval: 1_000,
      heartbeat_interval: 2_000,
      suspect_timeout: 6_000,
      dead_timeout: 15_000
    ]
  end

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  def get(server \\ __MODULE__, key), do: GenServer.call(server, {:get, key})
  def put_counter(server \\ __MODULE__, key), do: GenServer.call(server, {:put_counter, key})
  def increment(server \\ __MODULE__, key, amount \\ 1), do: GenServer.call(server, {:increment, key, amount})
  def decrement(server \\ __MODULE__, key, amount \\ 1), do: GenServer.call(server, {:decrement, key, amount})
  def set_register(server \\ __MODULE__, key, value), do: GenServer.call(server, {:set_register, key, value})
  def add_to_set(server \\ __MODULE__, key, element), do: GenServer.call(server, {:add_set, key, element})
  def remove_from_set(server \\ __MODULE__, key, element), do: GenServer.call(server, {:remove_set, key, element})

  def acquire_lock(server \\ __MODULE__, key, ttl_ms \\ 30_000) do
    GenServer.call(server, {:acquire_lock, key, ttl_ms})
  end

  def release_lock(server \\ __MODULE__, key) do
    GenServer.call(server, {:release_lock, key})
  end

  def get_leader(server \\ __MODULE__), do: GenServer.call(server, :get_leader)
  def get_members(server \\ __MODULE__), do: GenServer.call(server, :get_members)
  def snapshot(server \\ __MODULE__), do: GenServer.call(server, :snapshot)
  def recover(server \\ __MODULE__), do: GenServer.call(server, :recover)
  def route_request(server \\ __MODULE__, request), do: GenServer.call(server, {:route, request})

  def receive_gossip(server \\ __MODULE__, from_node, data) do
    GenServer.cast(server, {:gossip, from_node, data})
  end

  def receive_heartbeat(server \\ __MODULE__, from_node, health) do
    GenServer.cast(server, {:heartbeat, from_node, health})
  end

  # ---------------------------------------------------------------------------
  # Server Callbacks
  # ---------------------------------------------------------------------------
  @impl true
  def init(opts) do
    node_id = Keyword.get(opts, :node_id, node_name())
    snapshot_dir = Keyword.get(opts, :snapshot_dir, "/tmp/srfm_state")

    state = %ServerState{
      node_id: node_id,
      snapshot_dir: snapshot_dir,
      members: %{node_id => %ClusterMember{
        node: node_id, status: :alive,
        last_heartbeat: now(), health_score: 100,
        joined_at: now()
      }}
    }

    schedule_gossip(state.gossip_interval)
    schedule_heartbeat(state.heartbeat_interval)
    schedule_failure_detection(state.suspect_timeout)
    schedule_snapshot(300_000)
    schedule_lock_cleanup(10_000)

    {:ok, state}
  end

  @impl true
  def handle_call({:get, key}, _from, state) do
    result = case Map.get(state.crdts, key) do
      %GCounter{} = c -> {:ok, GCounter.value(c)}
      %PNCounter{} = c -> {:ok, PNCounter.value(c)}
      %LWWRegister{} = r -> {:ok, r.value}
      %ORSet{} = s -> {:ok, ORSet.elements(s)}
      nil -> {:error, :not_found}
    end
    {:reply, result, state}
  end

  def handle_call({:put_counter, key}, _from, state) do
    new_crdts = Map.put_new(state.crdts, key, GCounter.new())
    vc = VectorClock.increment(state.vector_clock, state.node_id)
    {:reply, :ok, %{state | crdts: new_crdts, vector_clock: vc}}
  end

  def handle_call({:increment, key, amount}, _from, state) do
    crdt = Map.get(state.crdts, key, PNCounter.new())
    updated = case crdt do
      %GCounter{} -> GCounter.increment(crdt, state.node_id, amount)
      %PNCounter{} -> PNCounter.increment(crdt, state.node_id, amount)
      _ -> crdt
    end
    vc = VectorClock.increment(state.vector_clock, state.node_id)
    event = {:increment, key, state.node_id, amount, now()}
    {:reply, :ok, %{state |
      crdts: Map.put(state.crdts, key, updated),
      vector_clock: vc,
      event_log: [event | state.event_log]
    }}
  end

  def handle_call({:decrement, key, amount}, _from, state) do
    crdt = Map.get(state.crdts, key, PNCounter.new())
    updated = case crdt do
      %PNCounter{} -> PNCounter.decrement(crdt, state.node_id, amount)
      _ -> crdt
    end
    vc = VectorClock.increment(state.vector_clock, state.node_id)
    event = {:decrement, key, state.node_id, amount, now()}
    {:reply, :ok, %{state |
      crdts: Map.put(state.crdts, key, updated),
      vector_clock: vc,
      event_log: [event | state.event_log]
    }}
  end

  def handle_call({:set_register, key, value}, _from, state) do
    reg = Map.get(state.crdts, key, LWWRegister.new())
    updated = case reg do
      %LWWRegister{} -> LWWRegister.set(reg, value)
      _ -> LWWRegister.new(value)
    end
    vc = VectorClock.increment(state.vector_clock, state.node_id)
    event = {:set, key, value, now()}
    {:reply, :ok, %{state |
      crdts: Map.put(state.crdts, key, updated),
      vector_clock: vc,
      event_log: [event | state.event_log]
    }}
  end

  def handle_call({:add_set, key, element}, _from, state) do
    set = Map.get(state.crdts, key, ORSet.new())
    updated = ORSet.add(set, element)
    vc = VectorClock.increment(state.vector_clock, state.node_id)
    {:reply, :ok, %{state |
      crdts: Map.put(state.crdts, key, updated),
      vector_clock: vc
    }}
  end

  def handle_call({:remove_set, key, element}, _from, state) do
    case Map.get(state.crdts, key) do
      %ORSet{} = set ->
        updated = ORSet.remove(set, element)
        vc = VectorClock.increment(state.vector_clock, state.node_id)
        {:reply, :ok, %{state |
          crdts: Map.put(state.crdts, key, updated),
          vector_clock: vc
        }}
      _ ->
        {:reply, {:error, :not_found}, state}
    end
  end

  def handle_call({:acquire_lock, key, ttl_ms}, _from, state) do
    case Map.get(state.locks, key) do
      nil ->
        lock = %Lock{key: key, owner: state.node_id, acquired_at: now(), ttl_ms: ttl_ms}
        {:reply, {:ok, lock}, %{state | locks: Map.put(state.locks, key, lock)}}

      %Lock{owner: owner} when owner == state.node_id ->
        lock = %Lock{key: key, owner: state.node_id, acquired_at: now(), ttl_ms: ttl_ms}
        {:reply, {:ok, lock}, %{state | locks: Map.put(state.locks, key, lock)}}

      %Lock{acquired_at: acq, ttl_ms: ttl} ->
        if now() - acq > ttl do
          lock = %Lock{key: key, owner: state.node_id, acquired_at: now(), ttl_ms: ttl_ms}
          {:reply, {:ok, lock}, %{state | locks: Map.put(state.locks, key, lock)}}
        else
          {:reply, {:error, :locked}, state}
        end
    end
  end

  def handle_call({:release_lock, key}, _from, state) do
    case Map.get(state.locks, key) do
      %Lock{owner: owner} when owner == state.node_id ->
        {:reply, :ok, %{state | locks: Map.delete(state.locks, key)}}
      _ ->
        {:reply, {:error, :not_owner}, state}
    end
  end

  def handle_call(:get_leader, _from, state) do
    {:reply, state.leader, state}
  end

  def handle_call(:get_members, _from, state) do
    {:reply, state.members, state}
  end

  def handle_call(:snapshot, _from, state) do
    result = do_snapshot(state)
    {:reply, result, state}
  end

  def handle_call(:recover, _from, state) do
    case do_recover(state) do
      {:ok, recovered} -> {:reply, :ok, recovered}
      err -> {:reply, err, state}
    end
  end

  def handle_call({:route, request}, _from, state) do
    target = select_healthiest_node(state)
    {:reply, {:routed, target, request}, state}
  end

  @impl true
  def handle_cast({:gossip, from_node, %{crdts: remote_crdts, vector_clock: remote_vc}}, state) do
    merged_crdts = merge_all_crdts(state.crdts, remote_crdts)
    merged_vc = VectorClock.merge(state.vector_clock, remote_vc)

    new_members = Map.update(state.members, from_node,
      %ClusterMember{node: from_node, status: :alive, last_heartbeat: now(),
                     health_score: 100, joined_at: now()},
      fn m -> %{m | status: :alive, last_heartbeat: now()} end)

    {:noreply, %{state |
      crdts: merged_crdts,
      vector_clock: merged_vc,
      members: new_members
    }}
  end

  def handle_cast({:heartbeat, from_node, health}, state) do
    new_members = Map.update(state.members, from_node,
      %ClusterMember{node: from_node, status: :alive, last_heartbeat: now(),
                     health_score: health, joined_at: now()},
      fn m -> %{m | last_heartbeat: now(), health_score: health, status: :alive} end)
    {:noreply, %{state | members: new_members}}
  end

  @impl true
  def handle_info(:gossip, state) do
    peers = state.members
      |> Map.keys()
      |> Enum.reject(&(&1 == state.node_id))

    if length(peers) > 0 do
      target = Enum.random(peers)
      gossip_data = %{
        crdts: state.crdts,
        vector_clock: state.vector_clock
      }
      send_gossip(target, state.node_id, gossip_data)
    end

    schedule_gossip(state.gossip_interval)
    {:noreply, state}
  end

  def handle_info(:heartbeat, state) do
    health = compute_health(state)
    peers = state.members
      |> Map.keys()
      |> Enum.reject(&(&1 == state.node_id))

    Enum.each(peers, fn peer -> send_heartbeat(peer, state.node_id, health) end)
    schedule_heartbeat(state.heartbeat_interval)
    {:noreply, state}
  end

  def handle_info(:failure_detection, state) do
    now_ts = now()
    new_members = Map.new(state.members, fn {id, member} ->
      if id == state.node_id do
        {id, member}
      else
        elapsed = now_ts - (member.last_heartbeat || 0)
        new_status = cond do
          elapsed > state.dead_timeout -> :dead
          elapsed > state.suspect_timeout -> :suspect
          true -> member.status
        end
        if new_status != member.status do
          Logger.warning("Node #{id} status: #{member.status} -> #{new_status}")
        end
        {id, %{member | status: new_status}}
      end
    end)

    new_state = %{state | members: new_members}
    new_state = maybe_elect_leader(new_state)

    schedule_failure_detection(state.suspect_timeout)
    {:noreply, new_state}
  end

  def handle_info(:snapshot_tick, state) do
    do_snapshot(state)
    schedule_snapshot(300_000)
    {:noreply, state}
  end

  def handle_info(:lock_cleanup, state) do
    now_ts = now()
    cleaned = state.locks
      |> Enum.reject(fn {_k, lock} -> now_ts - lock.acquired_at > lock.ttl_ms end)
      |> Map.new()
    schedule_lock_cleanup(10_000)
    {:noreply, %{state | locks: cleaned}}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # ---------------------------------------------------------------------------
  # Private
  # ---------------------------------------------------------------------------
  defp now, do: System.system_time(:millisecond)
  defp node_name, do: Atom.to_string(node()) |> String.split("@") |> hd()

  defp merge_all_crdts(local, remote) do
    all_keys = MapSet.union(MapSet.new(Map.keys(local)), MapSet.new(Map.keys(remote)))
    Map.new(all_keys, fn key ->
      l = Map.get(local, key)
      r = Map.get(remote, key)
      merged = case {l, r} do
        {nil, v} -> v
        {v, nil} -> v
        {%GCounter{} = a, %GCounter{} = b} -> GCounter.merge(a, b)
        {%PNCounter{} = a, %PNCounter{} = b} -> PNCounter.merge(a, b)
        {%LWWRegister{} = a, %LWWRegister{} = b} -> LWWRegister.merge(a, b)
        {%ORSet{} = a, %ORSet{} = b} -> ORSet.merge(a, b)
        {a, _b} -> a
      end
      {key, merged}
    end)
  end

  defp maybe_elect_leader(state) do
    alive = state.members
      |> Enum.filter(fn {_id, m} -> m.status == :alive end)
      |> Enum.map(fn {id, _m} -> id end)
      |> Enum.sort()

    new_leader = if length(alive) > 0 do
      # Bully algorithm: highest node ID wins
      Enum.max(alive)
    else
      state.node_id
    end

    if new_leader != state.leader do
      Logger.info("Leader elected: #{new_leader}")
    end

    %{state | leader: new_leader}
  end

  defp select_healthiest_node(state) do
    state.members
    |> Enum.filter(fn {_id, m} -> m.status == :alive end)
    |> Enum.max_by(fn {_id, m} -> m.health_score end, fn -> {state.node_id, nil} end)
    |> elem(0)
  end

  defp compute_health(state) do
    base = 100
    n_locks = map_size(state.locks)
    n_crdts = map_size(state.crdts)
    lock_penalty = min(n_locks * 2, 20)
    load_penalty = min(n_crdts, 30)
    memory_info = :erlang.memory(:total)
    mem_penalty = if memory_info > 500_000_000, do: 20, else: 0
    max(base - lock_penalty - load_penalty - mem_penalty, 0)
  end

  defp do_snapshot(state) do
    File.mkdir_p!(state.snapshot_dir)
    data = %{
      crdts: state.crdts,
      vector_clock: state.vector_clock,
      timestamp: now()
    }
    path = Path.join(state.snapshot_dir, "snapshot_#{now()}.bin")
    binary = :erlang.term_to_binary(data, [:compressed])
    case File.write(path, binary) do
      :ok ->
        cleanup_old_snapshots(state.snapshot_dir, 5)
        {:ok, path}
      err -> err
    end
  end

  defp do_recover(state) do
    case latest_snapshot(state.snapshot_dir) do
      nil -> {:error, :no_snapshot}
      path ->
        binary = File.read!(path)
        data = :erlang.binary_to_term(binary)
        recovered = %{state |
          crdts: data.crdts,
          vector_clock: data.vector_clock
        }
        # Replay event log after snapshot
        recovered = replay_events(recovered, data.timestamp)
        {:ok, recovered}
    end
  end

  defp latest_snapshot(dir) do
    case File.ls(dir) do
      {:ok, files} ->
        files
        |> Enum.filter(&String.starts_with?(&1, "snapshot_"))
        |> Enum.sort(:desc)
        |> List.first()
        |> case do
          nil -> nil
          f -> Path.join(dir, f)
        end
      _ -> nil
    end
  end

  defp cleanup_old_snapshots(dir, keep) do
    case File.ls(dir) do
      {:ok, files} ->
        files
        |> Enum.filter(&String.starts_with?(&1, "snapshot_"))
        |> Enum.sort(:desc)
        |> Enum.drop(keep)
        |> Enum.each(fn f -> File.rm(Path.join(dir, f)) end)
      _ -> :ok
    end
  end

  defp replay_events(state, since_timestamp) do
    events = state.event_log
      |> Enum.filter(fn event -> elem(event, tuple_size(event) - 1) > since_timestamp end)
      |> Enum.reverse()

    Enum.reduce(events, state, fn event, acc ->
      case event do
        {:increment, key, node, amount, _ts} ->
          crdt = Map.get(acc.crdts, key, PNCounter.new())
          updated = case crdt do
            %GCounter{} -> GCounter.increment(crdt, node, amount)
            %PNCounter{} -> PNCounter.increment(crdt, node, amount)
            _ -> crdt
          end
          %{acc | crdts: Map.put(acc.crdts, key, updated)}

        {:decrement, key, node, amount, _ts} ->
          crdt = Map.get(acc.crdts, key, PNCounter.new())
          updated = PNCounter.decrement(crdt, node, amount)
          %{acc | crdts: Map.put(acc.crdts, key, updated)}

        {:set, key, value, _ts} ->
          %{acc | crdts: Map.put(acc.crdts, key, LWWRegister.new(value))}

        _ -> acc
      end
    end)
  end

  defp send_gossip(target, from, data) do
    # In production, this would use :rpc.cast or distributed Erlang
    try do
      GenServer.cast({__MODULE__, target}, {:gossip, from, data})
    catch
      _, _ -> :ok
    end
  end

  defp send_heartbeat(target, from, health) do
    try do
      GenServer.cast({__MODULE__, target}, {:heartbeat, from, health})
    catch
      _, _ -> :ok
    end
  end

  defp schedule_gossip(interval), do: Process.send_after(self(), :gossip, interval)
  defp schedule_heartbeat(interval), do: Process.send_after(self(), :heartbeat, interval)
  defp schedule_failure_detection(interval), do: Process.send_after(self(), :failure_detection, interval)
  defp schedule_snapshot(interval), do: Process.send_after(self(), :snapshot_tick, interval)
  defp schedule_lock_cleanup(interval), do: Process.send_after(self(), :lock_cleanup, interval)
end
