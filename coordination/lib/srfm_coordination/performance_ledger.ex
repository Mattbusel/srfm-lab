defmodule SrfmCoordination.PerformanceLedger do
  @moduledoc """
  Append-only ledger of strategy performance metrics.
  Used by the rollback guard to detect post-update degradation.

  Stores observations in two tiers:
    1. ETS table :perf_ledger for sub-millisecond in-memory reads
    2. Exqlite-backed SQLite file for durability across restarts

  Each LedgerEntry records a Sharpe ratio, drawdown, PnL, and trade count
  at a given UNIX timestamp (milliseconds). Entries are never mutated or
  deleted -- only appended.

  Degradation detection rule:
    A period is considered degraded if the rolling 4h Sharpe in the 2 hours
    following a parameter update drops below -0.5.
  """

  use GenServer
  require Logger

  @table :perf_ledger
  @db_path "priv/perf_ledger.db"
  @degraded_sharpe_threshold -0.5
  @degraded_post_update_hours 2
  @degraded_rolling_hours 4

  -- LedgerEntry represents one observation appended to the ledger.
  defmodule LedgerEntry do
    @moduledoc "One row in the performance ledger."
    @enforce_keys [:ts, :sharpe, :drawdown, :pnl, :n_trades]
    defstruct [:ts, :sharpe, :drawdown, :pnl, :n_trades]

    @type t :: %__MODULE__{
      ts:       integer(),
      sharpe:   float(),
      drawdown: float(),
      pnl:      float(),
      n_trades: non_neg_integer()
    }
  end

  defstruct [
    db:           nil,
    entry_count:  0,
    db_available: false
  ]

  -- ---------------------------------------------------------------------------
  -- Public API
  -- ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Append a Sharpe ratio observation.
  `ts` is UNIX timestamp in milliseconds.
  `source` identifies the origin (e.g. :live_trader, :backtest).
  """
  @spec record_sharpe(integer(), float(), atom()) :: :ok
  def record_sharpe(ts, sharpe, source \\ :unknown) when is_integer(ts) and is_float(sharpe) do
    entry = %LedgerEntry{ts: ts, sharpe: sharpe, drawdown: 0.0, pnl: 0.0, n_trades: 0}
    GenServer.cast(__MODULE__, {:record, entry, source})
  end

  @doc """
  Append a drawdown observation.
  `drawdown` is a positive float representing the percentage decline (e.g. 0.05 = 5%).
  """
  @spec record_drawdown(integer(), float()) :: :ok
  def record_drawdown(ts, drawdown) when is_integer(ts) and is_float(drawdown) do
    entry = %LedgerEntry{ts: ts, sharpe: 0.0, drawdown: drawdown, pnl: 0.0, n_trades: 0}
    GenServer.cast(__MODULE__, {:record, entry, :unknown})
  end

  @doc """
  Append a full performance entry with all fields populated.
  """
  @spec record_entry(LedgerEntry.t()) :: :ok
  def record_entry(%LedgerEntry{} = entry) do
    GenServer.cast(__MODULE__, {:record, entry, :direct})
  end

  @doc """
  Compute the rolling average Sharpe ratio over the last `window_hours` hours.
  Returns 0.0 if no data is available in the window.
  """
  @spec get_rolling_sharpe(pos_integer()) :: float()
  def get_rolling_sharpe(window_hours \\ 4) when is_integer(window_hours) and window_hours > 0 do
    cutoff_ms = now_ms() - window_hours * 3_600_000
    entries   = entries_since(cutoff_ms)

    if entries == [] do
      0.0
    else
      sharpes = Enum.map(entries, & &1.sharpe)
      Float.round(Enum.sum(sharpes) / length(sharpes), 6)
    end
  end

  @doc """
  Compute the rolling average drawdown over the last `window_hours` hours.
  Returns 0.0 if no data is available in the window.
  """
  @spec get_rolling_drawdown(pos_integer()) :: float()
  def get_rolling_drawdown(window_hours \\ 4) when is_integer(window_hours) and window_hours > 0 do
    cutoff_ms = now_ms() - window_hours * 3_600_000
    entries   = entries_since(cutoff_ms)

    if entries == [] do
      0.0
    else
      drawdowns = Enum.map(entries, & &1.drawdown)
      Float.round(Enum.sum(drawdowns) / length(drawdowns), 6)
    end
  end

  @doc """
  Returns true if the system is degraded following the update at `post_update_ts`.

  Degradation rule: rolling #{@degraded_rolling_hours}h Sharpe in the
  #{@degraded_post_update_hours}h window after `post_update_ts` < #{@degraded_sharpe_threshold}.
  """
  @spec is_degraded?(integer()) :: boolean()
  def is_degraded?(post_update_ts) when is_integer(post_update_ts) do
    window_end_ms = post_update_ts + @degraded_post_update_hours * 3_600_000
    now           = now_ms()

    -- Only evaluate if enough time has passed
    if now < post_update_ts + 60_000 do
      false
    else
      effective_end = min(window_end_ms, now)
      entries = entries_in_range(post_update_ts, effective_end)

      if length(entries) < 3 do
        -- Not enough data to make a determination
        false
      else
        sharpes = Enum.map(entries, & &1.sharpe)
        avg = Enum.sum(sharpes) / length(sharpes)
        avg < @degraded_sharpe_threshold
      end
    end
  end

  @doc """
  Compute the difference in average Sharpe between a pre-update and post-update window.
  Both timestamps are UNIX milliseconds marking the start of each window.
  Window width defaults to 4 hours for each side.
  Returns a positive number if performance improved, negative if degraded.
  """
  @spec compute_sharpe_delta(integer(), integer()) :: float()
  def compute_sharpe_delta(pre_ts, post_ts) when is_integer(pre_ts) and is_integer(post_ts) do
    window_ms = @degraded_rolling_hours * 3_600_000

    pre_entries  = entries_in_range(pre_ts,  pre_ts  + window_ms)
    post_entries = entries_in_range(post_ts, post_ts + window_ms)

    pre_sharpe  = avg_sharpe(pre_entries)
    post_sharpe = avg_sharpe(post_entries)

    Float.round(post_sharpe - pre_sharpe, 6)
  end

  @doc "Return all entries since `cutoff_ms` (UNIX ms), newest first."
  @spec entries_since(integer()) :: [LedgerEntry.t()]
  def entries_since(cutoff_ms) when is_integer(cutoff_ms) do
    case :ets.lookup(@table, :all) do
      [{:all, entries}] ->
        Enum.filter(entries, fn e -> e.ts >= cutoff_ms end)

      [] ->
        []
    end
  end

  @doc "Return the total number of entries in the ledger."
  @spec entry_count() :: non_neg_integer()
  def entry_count do
    GenServer.call(__MODULE__, :entry_count)
  end

  -- ---------------------------------------------------------------------------
  -- GenServer callbacks
  -- ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])
    :ets.insert(@table, {:all, []})

    {db, available} = open_database()

    state = %__MODULE__{
      db:           db,
      db_available: available,
      entry_count:  0
    }

    state = if available, do: load_from_db(db, state), else: state

    Logger.info(
      "[PerformanceLedger] Initialized -- db=#{available}, " <>
      "entries=#{state.entry_count}"
    )

    {:ok, state}
  end

  @impl true
  def handle_cast({:record, entry, source}, state) do
    append_to_ets(entry)
    new_state = %{state | entry_count: state.entry_count + 1}

    if state.db_available do
      persist_entry(state.db, entry, source)
    end

    {:noreply, new_state}
  end

  @impl true
  def handle_call(:entry_count, _from, state) do
    {:reply, state.entry_count, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, state) do
    Logger.info("[PerformanceLedger] Terminating: #{inspect(reason)}")

    if state.db_available and state.db != nil do
      try do
        Exqlite.Sqlite3.close(state.db)
      catch
        _, _ -> :ok
      end
    end

    :ok
  end

  -- ---------------------------------------------------------------------------
  -- Private -- ETS operations
  -- ---------------------------------------------------------------------------

  defp append_to_ets(entry) do
    existing =
      case :ets.lookup(@table, :all) do
        [{:all, list}] -> list
        [] -> []
      end

    :ets.insert(@table, {:all, [entry | existing]})
  end

  defp entries_in_range(from_ms, to_ms) do
    case :ets.lookup(@table, :all) do
      [{:all, entries}] ->
        Enum.filter(entries, fn e -> e.ts >= from_ms and e.ts <= to_ms end)

      [] ->
        []
    end
  end

  defp avg_sharpe([]), do: 0.0

  defp avg_sharpe(entries) do
    sharpes = Enum.map(entries, & &1.sharpe)
    Enum.sum(sharpes) / length(sharpes)
  end

  -- ---------------------------------------------------------------------------
  -- Private -- SQLite persistence
  -- ---------------------------------------------------------------------------

  defp open_database do
    db_path = Application.get_env(:srfm_coordination, :perf_ledger_db, @db_path)

    try do
      File.mkdir_p!(Path.dirname(db_path))

      {:ok, db} = Exqlite.Sqlite3.open(db_path)

      {:ok, _stmt} =
        Exqlite.Sqlite3.execute(db, """
          CREATE TABLE IF NOT EXISTS perf_ledger (
            ts       INTEGER NOT NULL,
            sharpe   REAL    NOT NULL,
            drawdown REAL    NOT NULL,
            pnl      REAL    NOT NULL,
            n_trades INTEGER NOT NULL,
            source   TEXT    NOT NULL DEFAULT 'unknown',
            inserted_at INTEGER NOT NULL
          )
        """)

      {:ok, _} =
        Exqlite.Sqlite3.execute(db, "CREATE INDEX IF NOT EXISTS idx_ts ON perf_ledger(ts)")

      {db, true}
    catch
      kind, reason ->
        Logger.warning(
          "[PerformanceLedger] SQLite unavailable (#{kind}: #{inspect(reason)}) -- " <>
          "running in-memory only"
        )
        {nil, false}
    end
  end

  defp persist_entry(db, entry, source) do
    try do
      {:ok, stmt} =
        Exqlite.Sqlite3.prepare(db, """
          INSERT INTO perf_ledger (ts, sharpe, drawdown, pnl, n_trades, source, inserted_at)
          VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        """)

      :ok =
        Exqlite.Sqlite3.bind(db, stmt, [
          entry.ts,
          entry.sharpe,
          entry.drawdown,
          entry.pnl,
          entry.n_trades,
          to_string(source),
          now_ms()
        ])

      :done = Exqlite.Sqlite3.step(db, stmt)
      :ok
    catch
      kind, reason ->
        Logger.warning("[PerformanceLedger] DB write failed (#{kind}): #{inspect(reason)}")
        :ok
    end
  end

  defp load_from_db(db, state) do
    try do
      {:ok, stmt} =
        Exqlite.Sqlite3.prepare(db, """
          SELECT ts, sharpe, drawdown, pnl, n_trades
          FROM perf_ledger
          ORDER BY ts DESC
          LIMIT 10000
        """)

      entries = collect_rows(db, stmt, [])

      :ets.insert(@table, {:all, entries})

      Logger.info("[PerformanceLedger] Loaded #{length(entries)} entries from SQLite")
      %{state | entry_count: length(entries)}
    catch
      kind, reason ->
        Logger.warning("[PerformanceLedger] Failed to load from DB (#{kind}): #{inspect(reason)}")
        state
    end
  end

  defp collect_rows(db, stmt, acc) do
    case Exqlite.Sqlite3.step(db, stmt) do
      {:row, [ts, sharpe, drawdown, pnl, n_trades]} ->
        entry = %LedgerEntry{
          ts:       ts,
          sharpe:   sharpe / 1.0,
          drawdown: drawdown / 1.0,
          pnl:      pnl / 1.0,
          n_trades: n_trades
        }
        collect_rows(db, stmt, [entry | acc])

      :done ->
        acc

      {:error, reason} ->
        Logger.warning("[PerformanceLedger] Row fetch error: #{inspect(reason)}")
        acc
    end
  end

  defp now_ms, do: System.system_time(:millisecond)
end
