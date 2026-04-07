defmodule SrfmCoordination.ParameterHistory do
  @moduledoc """
  Persistent parameter history with analytics.

  Dual-write strategy:
  -- ETS (:srfm_param_history_hot) for hot reads / in-memory analytics
  -- SQLite via Exqlite for persistence across restarts

  Analytics:
  -- get_history(n) returns last N updates with per-key delta and fitness change
  -- get_best_params(metric) returns the parameter snapshot from the update
     that achieved the best subsequent fitness (by sharpe or pnl)
  -- compute_param_sensitivity() correlates each param's value against
     subsequent Sharpe to identify high-leverage parameters

  REST:
    GET /params/history?n=20
    GET /params/best
    GET /params/sensitivity

  ETS table: :srfm_param_history_hot -- {update_id, record_map}
  SQLite: table param_updates (id, timestamp, source, fitness_before,
          fitness_after, old_params_json, new_params_json)
  """

  use GenServer
  require Logger

  @hot_table :srfm_param_history_hot
  @default_history_limit 20
  @sqlite_db_path Application.compile_env(
                    :srfm_coordination,
                    :param_history_db,
                    "/tmp/srfm_param_history.db"
                  )

  defstruct db: nil, insert_stmt: nil

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Record a parameter update event.
  `old_params` and `new_params` are maps of param_name => value.
  `source` is a string (e.g. "genome_receiver", "manual").
  `fitness_before` and `fitness_after` are Sharpe or comparable floats.
  """
  @spec record_update(map(), map(), String.t(), float(), float()) :: :ok
  def record_update(old_params, new_params, source, fitness_before, fitness_after) do
    GenServer.cast(__MODULE__, {:record_update, old_params, new_params, source, fitness_before, fitness_after})
  end

  @doc "Return the last `n` parameter updates, newest first, with delta and fitness impact."
  @spec get_history(pos_integer()) :: [map()]
  def get_history(n \\ @default_history_limit) when is_integer(n) and n > 0 do
    :ets.tab2list(@hot_table)
    |> Enum.map(fn {_id, record} -> record end)
    |> Enum.sort_by(fn r -> r.recorded_at end, {:desc, DateTime})
    |> Enum.take(n)
  end

  @doc """
  Return the parameter snapshot from the historical update that achieved the
  best subsequent fitness under `metric` (:sharpe or :pnl_improvement).
  """
  @spec get_best_params(atom()) :: {:ok, map()} | {:error, :no_history}
  def get_best_params(metric \\ :sharpe) when metric in [:sharpe, :pnl_improvement] do
    records =
      :ets.tab2list(@hot_table)
      |> Enum.map(fn {_id, r} -> r end)

    if records == [] do
      {:error, :no_history}
    else
      best =
        Enum.max_by(records, fn r ->
          case metric do
            :sharpe -> r.fitness_after
            :pnl_improvement -> r.fitness_after - r.fitness_before
          end
        end)

      {:ok, best.new_params}
    end
  end

  @doc """
  For each parameter, compute the Pearson correlation between its value
  across all recorded updates and the subsequent fitness_after value.
  Returns %{param_name => correlation_float}.
  """
  @spec compute_param_sensitivity() :: map()
  def compute_param_sensitivity do
    records =
      :ets.tab2list(@hot_table)
      |> Enum.map(fn {_id, r} -> r end)

    if length(records) < 3 do
      %{}
    else
      # Collect all param keys
      all_keys =
        records
        |> Enum.flat_map(fn r -> Map.keys(r.new_params) end)
        |> Enum.uniq()

      Map.new(all_keys, fn key ->
        pairs =
          Enum.flat_map(records, fn r ->
            case Map.fetch(r.new_params, key) do
              {:ok, v} when is_number(v) -> [{v * 1.0, r.fitness_after}]
              _ -> []
            end
          end)

        corr = pearson_correlation(pairs)
        {key, corr}
      end)
    end
  end

  @doc "Return the count of stored updates."
  @spec update_count() :: non_neg_integer()
  def update_count do
    :ets.info(@hot_table, :size)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@hot_table, [:named_table, :set, :public, read_concurrency: true])

    db_state = init_sqlite()

    # Warm ETS from SQLite on startup
    warm_ets_from_sqlite(db_state)

    Logger.info("[ParameterHistory] Initialized (SQLite: #{@sqlite_db_path})")
    {:ok, db_state}
  end

  @impl true
  def handle_cast({:record_update, old_params, new_params, source, fitness_before, fitness_after}, state) do
    update_id = generate_id()
    now = DateTime.utc_now()

    delta = compute_delta(old_params, new_params)
    fitness_impact = fitness_after - fitness_before

    record = %{
      id: update_id,
      old_params: old_params,
      new_params: new_params,
      delta: delta,
      source: source,
      fitness_before: fitness_before,
      fitness_after: fitness_after,
      fitness_impact: fitness_impact,
      recorded_at: now
    }

    :ets.insert(@hot_table, {update_id, record})
    persist_to_sqlite(state, record)

    Logger.info(
      "[ParameterHistory] Update #{update_id} recorded from #{source} " <>
      "fitness: #{Float.round(fitness_before, 4)} -> #{Float.round(fitness_after, 4)}"
    )

    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, %{db: db} = _state) do
    Logger.info("[ParameterHistory] Terminating: #{inspect(reason)}")

    if db do
      try do
        Exqlite.Sqlite3.close(db)
      catch
        _, _ -> :ok
      end
    end

    :ok
  end

  # ---------------------------------------------------------------------------
  # SQLite helpers
  # ---------------------------------------------------------------------------

  defp init_sqlite do
    case try_open_sqlite(@sqlite_db_path) do
      {:ok, db} ->
        create_tables(db)
        stmt = prepare_insert(db)
        %__MODULE__{db: db, insert_stmt: stmt}

      {:error, reason} ->
        Logger.warning("[ParameterHistory] SQLite unavailable: #{inspect(reason)} -- in-memory only")
        %__MODULE__{db: nil, insert_stmt: nil}
    end
  end

  defp try_open_sqlite(path) do
    try do
      Exqlite.Sqlite3.open(path)
    rescue
      _ -> {:error, :exqlite_not_available}
    catch
      _, reason -> {:error, reason}
    end
  end

  defp create_tables(db) do
    sql = """
    CREATE TABLE IF NOT EXISTS param_updates (
      id TEXT PRIMARY KEY,
      recorded_at TEXT NOT NULL,
      source TEXT NOT NULL,
      fitness_before REAL NOT NULL,
      fitness_after REAL NOT NULL,
      fitness_impact REAL NOT NULL,
      old_params_json TEXT NOT NULL,
      new_params_json TEXT NOT NULL,
      delta_json TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_param_updates_recorded_at ON param_updates(recorded_at);
    """

    try do
      Exqlite.Sqlite3.execute(db, sql)
    catch
      _, _ -> :ok
    end
  end

  defp prepare_insert(db) do
    sql = """
    INSERT INTO param_updates
      (id, recorded_at, source, fitness_before, fitness_after, fitness_impact,
       old_params_json, new_params_json, delta_json)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
    """

    try do
      case Exqlite.Sqlite3.prepare(db, sql) do
        {:ok, stmt} -> stmt
        _ -> nil
      end
    catch
      _, _ -> nil
    end
  end

  defp persist_to_sqlite(%{db: nil}, _record), do: :ok
  defp persist_to_sqlite(%{db: _db, insert_stmt: nil}, _record), do: :ok

  defp persist_to_sqlite(%{db: db, insert_stmt: stmt}, record) do
    try do
      Exqlite.Sqlite3.bind(db, stmt, [
        record.id,
        DateTime.to_iso8601(record.recorded_at),
        record.source,
        record.fitness_before,
        record.fitness_after,
        record.fitness_impact,
        Jason.encode!(record.old_params),
        Jason.encode!(record.new_params),
        Jason.encode!(record.delta)
      ])

      Exqlite.Sqlite3.step(db, stmt)
      Exqlite.Sqlite3.reset(db, stmt)
    catch
      _, reason ->
        Logger.warning("[ParameterHistory] SQLite write failed: #{inspect(reason)}")
    end

    :ok
  end

  defp warm_ets_from_sqlite(%{db: nil}), do: :ok

  defp warm_ets_from_sqlite(%{db: db}) do
    sql = "SELECT id, recorded_at, source, fitness_before, fitness_after, fitness_impact, old_params_json, new_params_json, delta_json FROM param_updates ORDER BY recorded_at DESC LIMIT 500"

    try do
      {:ok, stmt} = Exqlite.Sqlite3.prepare(db, sql)
      rows = collect_rows(db, stmt, [])
      Exqlite.Sqlite3.release(db, stmt)

      Enum.each(rows, fn [id, recorded_at_str, source, fb, fa, fi, old_j, new_j, delta_j] ->
        {:ok, recorded_at, _} = DateTime.from_iso8601(recorded_at_str)

        record = %{
          id: id,
          source: source,
          fitness_before: fb,
          fitness_after: fa,
          fitness_impact: fi,
          old_params: Jason.decode!(old_j),
          new_params: Jason.decode!(new_j),
          delta: Jason.decode!(delta_j),
          recorded_at: recorded_at
        }

        :ets.insert(@hot_table, {id, record})
      end)

      Logger.info("[ParameterHistory] Warmed #{length(rows)} records from SQLite")
    catch
      _, reason ->
        Logger.warning("[ParameterHistory] SQLite warm failed: #{inspect(reason)}")
    end
  end

  defp collect_rows(db, stmt, acc) do
    case Exqlite.Sqlite3.step(db, stmt) do
      {:row, row} -> collect_rows(db, stmt, [row | acc])
      :done -> Enum.reverse(acc)
      _ -> Enum.reverse(acc)
    end
  end

  # ---------------------------------------------------------------------------
  # Analytics helpers
  # ---------------------------------------------------------------------------

  defp compute_delta(old_params, new_params) do
    all_keys = Map.keys(old_params) ++ Map.keys(new_params) |> Enum.uniq()

    Map.new(all_keys, fn key ->
      old_val = Map.get(old_params, key)
      new_val = Map.get(new_params, key)

      diff =
        cond do
          is_number(old_val) and is_number(new_val) ->
            %{from: old_val, to: new_val, abs_delta: new_val - old_val,
              pct_delta: if(old_val != 0, do: (new_val - old_val) / abs(old_val) * 100, else: nil)}

          true ->
            %{from: old_val, to: new_val}
        end

      {key, diff}
    end)
  end

  @doc false
  def pearson_correlation([]), do: 0.0
  def pearson_correlation(pairs) when length(pairs) < 2, do: 0.0

  def pearson_correlation(pairs) do
    n = length(pairs) * 1.0
    xs = Enum.map(pairs, fn {x, _y} -> x end)
    ys = Enum.map(pairs, fn {_x, y} -> y end)

    mean_x = Enum.sum(xs) / n
    mean_y = Enum.sum(ys) / n

    numerator =
      Enum.zip(xs, ys)
      |> Enum.reduce(0.0, fn {x, y}, acc -> acc + (x - mean_x) * (y - mean_y) end)

    sum_sq_x = Enum.reduce(xs, 0.0, fn x, acc -> acc + (x - mean_x) * (x - mean_x) end)
    sum_sq_y = Enum.reduce(ys, 0.0, fn y, acc -> acc + (y - mean_y) * (y - mean_y) end)

    denom = :math.sqrt(sum_sq_x * sum_sq_y)

    if denom == 0.0, do: 0.0, else: numerator / denom
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
