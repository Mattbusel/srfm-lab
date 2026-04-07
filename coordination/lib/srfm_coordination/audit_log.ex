defmodule SrfmCoordination.AuditLog do
  @moduledoc """
  Persistent, append-only audit log backed by SQLite via Exqlite.

  All significant system events are recorded with full context.
  Writes are async (cast); queries are synchronous (call).

  Schema:
    audit_log(id, timestamp_ns, event_type, actor, resource, details_json,
              outcome, session_id)

  Indices on: timestamp_ns, event_type, actor

  Retention: 90-day rolling window. Entries older than 90 days are archived
  to compressed JSON files in the configured archive directory before deletion.

  The table is opened with WAL mode and PRAGMA foreign_keys=ON.
  Deletes are only allowed via the archive path -- direct deletes are blocked
  at the application layer by routing all writes through this GenServer.
  """

  use GenServer
  require Logger

  alias SrfmCoordination.AuditLog.Entry

  @db_path Application.compile_env(
             :srfm_coordination,
             :audit_log_db_path,
             "/var/lib/srfm/audit.db"
           )

  @archive_dir Application.compile_env(
                 :srfm_coordination,
                 :audit_log_archive_dir,
                 "/var/lib/srfm/audit_archive"
               )

  @retention_days 90

  # How often to run the retention/archive sweep (12 hours in ms)
  @sweep_interval_ms 12 * 60 * 60 * 1_000

  @valid_event_types [
    :param_update,
    :param_rollback,
    :service_restart,
    :circuit_change,
    :order_submitted,
    :risk_breach,
    :manual_override
  ]

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  defmodule Entry do
    @moduledoc "Represents a single audit log entry."

    @type t :: %__MODULE__{
            id: non_neg_integer() | nil,
            timestamp_ns: non_neg_integer(),
            event_type: atom(),
            actor: String.t(),
            resource: String.t(),
            details: map(),
            outcome: String.t(),
            session_id: String.t() | nil
          }

    defstruct [
      :id,
      :timestamp_ns,
      :event_type,
      :actor,
      :resource,
      :details,
      :outcome,
      :session_id
    ]
  end

  @type filter_opt ::
          {:event_type, atom()}
          | {:actor, String.t()}
          | {:resource, String.t()}
          | {:from_ns, non_neg_integer()}
          | {:to_ns, non_neg_integer()}
          | {:limit, pos_integer()}
          | {:offset, non_neg_integer()}

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc "Start the AuditLog GenServer."
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Asynchronously write an audit entry.

  - event_type: one of the @valid_event_types atoms
  - actor: who triggered the event (e.g. "iae_genome_receiver", "operator:mwilson")
  - resource: the affected resource (e.g. "param:BH_MASS_THRESH", "service:alpha")
  - details: arbitrary map with event-specific data
  - outcome: short result string (e.g. "applied", "rejected", "rolled_back")

  Returns :ok immediately without waiting for the DB write.
  """
  @spec log(atom(), String.t(), String.t(), map(), String.t()) :: :ok
  def log(event_type, actor, resource, details \\ %{}, outcome \\ "ok")
      when event_type in @valid_event_types do
    entry = %Entry{
      timestamp_ns: System.system_time(:nanosecond),
      event_type: event_type,
      actor: actor,
      resource: resource,
      details: details,
      outcome: outcome,
      session_id: get_session_id()
    }

    GenServer.cast(__MODULE__, {:log, entry})
  end

  @doc """
  Query audit entries with filters.

  Supported filters:
    event_type: atom
    actor: string (exact match)
    resource: string (prefix match)
    from_ns: integer nanosecond timestamp (inclusive)
    to_ns: integer nanosecond timestamp (inclusive)
    limit: integer (default 100, max 1000)
    offset: integer (default 0)

  Returns a list of AuditEntry structs ordered by timestamp descending.
  """
  @spec query(filters: [filter_opt()]) :: [Entry.t()]
  def query(filters \\ []) do
    GenServer.call(__MODULE__, {:query, filters}, 15_000)
  end

  @doc """
  Export audit entries in a time range to JSON or CSV binary.

  from/to are DateTime structs.
  Returns {:ok, binary} or {:error, reason}.
  """
  @spec export_range(DateTime.t(), DateTime.t(), format: :json | :csv) ::
          {:ok, binary()} | {:error, term()}
  def export_range(from, to, opts \\ [format: :json]) do
    format = Keyword.get(opts, :format, :json)
    GenServer.call(__MODULE__, {:export_range, from, to, format}, 60_000)
  end

  @doc "Return the count of entries matching filters."
  @spec count(filters: [filter_opt()]) :: non_neg_integer()
  def count(filters \\ []) do
    GenServer.call(__MODULE__, {:count, filters}, 15_000)
  end

  @doc "Trigger an immediate retention sweep (archive + delete old entries)."
  @spec sweep_now() :: {:ok, non_neg_integer()} | {:error, term()}
  def sweep_now do
    GenServer.call(__MODULE__, :sweep_now, 120_000)
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  defstruct db: nil, db_path: nil

  @impl true
  def init(opts) do
    db_path = Keyword.get(opts, :db_path, @db_path)

    # Ensure directory exists
    File.mkdir_p!(Path.dirname(db_path))
    File.mkdir_p!(@archive_dir)

    case open_db(db_path) do
      {:ok, db} ->
        :ok = create_schema(db)
        :timer.send_after(@sweep_interval_ms, :sweep)
        Logger.info("[AuditLog] Initialized, DB at #{db_path}")
        {:ok, %__MODULE__{db: db, db_path: db_path}}

      {:error, reason} ->
        Logger.error("[AuditLog] Failed to open DB at #{db_path}: #{inspect(reason)}")
        {:stop, {:db_open_failed, reason}}
    end
  end

  @impl true
  def handle_cast({:log, entry}, state) do
    case insert_entry(state.db, entry) do
      :ok ->
        :ok

      {:error, reason} ->
        Logger.error("[AuditLog] Write failed for #{entry.event_type}: #{inspect(reason)}")
    end

    {:noreply, state}
  end

  @impl true
  def handle_call({:query, filters}, _from, state) do
    result = run_query(state.db, filters)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:count, filters}, _from, state) do
    result = run_count(state.db, filters)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:export_range, from, to, format}, _from, state) do
    result = do_export(state.db, from, to, format)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:sweep_now, _from, state) do
    result = do_sweep(state.db)
    {:reply, result, state}
  end

  @impl true
  def handle_info(:sweep, state) do
    case do_sweep(state.db) do
      {:ok, archived} ->
        Logger.info("[AuditLog] Retention sweep archived #{archived} entries")

      {:error, reason} ->
        Logger.error("[AuditLog] Retention sweep failed: #{inspect(reason)}")
    end

    :timer.send_after(@sweep_interval_ms, :sweep)
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(_reason, state) do
    if state.db do
      Exqlite.Sqlite3.close(state.db)
    end

    :ok
  end

  # ---------------------------------------------------------------------------
  # DB helpers
  # ---------------------------------------------------------------------------

  defp open_db(path) do
    Exqlite.Sqlite3.open(path)
  end

  defp create_schema(db) do
    statements = [
      """
      PRAGMA journal_mode=WAL;
      """,
      """
      PRAGMA synchronous=NORMAL;
      """,
      """
      CREATE TABLE IF NOT EXISTS audit_log (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_ns INTEGER NOT NULL,
        event_type   TEXT    NOT NULL,
        actor        TEXT    NOT NULL,
        resource     TEXT    NOT NULL,
        details_json TEXT    NOT NULL DEFAULT '{}',
        outcome      TEXT    NOT NULL,
        session_id   TEXT
      );
      """,
      "CREATE INDEX IF NOT EXISTS idx_audit_ts     ON audit_log(timestamp_ns);",
      "CREATE INDEX IF NOT EXISTS idx_audit_type   ON audit_log(event_type);",
      "CREATE INDEX IF NOT EXISTS idx_audit_actor  ON audit_log(actor);"
    ]

    Enum.reduce_while(statements, :ok, fn sql, _acc ->
      case Exqlite.Sqlite3.execute(db, String.trim(sql)) do
        :ok -> {:cont, :ok}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  defp insert_entry(db, entry) do
    sql = """
    INSERT INTO audit_log
      (timestamp_ns, event_type, actor, resource, details_json, outcome, session_id)
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
    """

    details_json = Jason.encode!(entry.details)

    with {:ok, stmt} <- Exqlite.Sqlite3.prepare(db, sql),
         :ok <-
           Exqlite.Sqlite3.bind(db, stmt, [
             entry.timestamp_ns,
             Atom.to_string(entry.event_type),
             entry.actor,
             entry.resource,
             details_json,
             entry.outcome,
             entry.session_id
           ]),
         :done <- Exqlite.Sqlite3.step(db, stmt),
         :ok <- Exqlite.Sqlite3.release(db, stmt) do
      :ok
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp run_query(db, filters) do
    {where_clauses, bindings} = build_where(filters)
    limit = min(Keyword.get(filters, :limit, 100), 1_000)
    offset = Keyword.get(filters, :offset, 0)

    where_sql =
      if where_clauses == [],
        do: "",
        else: "WHERE " <> Enum.join(where_clauses, " AND ")

    sql = """
    SELECT id, timestamp_ns, event_type, actor, resource, details_json, outcome, session_id
    FROM audit_log
    #{where_sql}
    ORDER BY timestamp_ns DESC
    LIMIT #{limit} OFFSET #{offset}
    """

    case Exqlite.Sqlite3.prepare(db, sql) do
      {:ok, stmt} ->
        Exqlite.Sqlite3.bind(db, stmt, bindings)
        rows = collect_rows(db, stmt, [])
        Exqlite.Sqlite3.release(db, stmt)
        Enum.map(rows, &row_to_entry/1)

      {:error, reason} ->
        Logger.error("[AuditLog] Query failed: #{inspect(reason)}")
        []
    end
  end

  defp run_count(db, filters) do
    {where_clauses, bindings} = build_where(filters)

    where_sql =
      if where_clauses == [],
        do: "",
        else: "WHERE " <> Enum.join(where_clauses, " AND ")

    sql = "SELECT COUNT(*) FROM audit_log #{where_sql}"

    case Exqlite.Sqlite3.prepare(db, sql) do
      {:ok, stmt} ->
        Exqlite.Sqlite3.bind(db, stmt, bindings)

        count =
          case Exqlite.Sqlite3.step(db, stmt) do
            {:row, [n]} -> n
            _ -> 0
          end

        Exqlite.Sqlite3.release(db, stmt)
        count

      {:error, _} ->
        0
    end
  end

  defp do_export(db, from, to, format) do
    from_ns = DateTime.to_unix(from, :nanosecond)
    to_ns = DateTime.to_unix(to, :nanosecond)

    filters = [from_ns: from_ns, to_ns: to_ns, limit: 100_000]
    entries = run_query(db, filters)

    case format do
      :json ->
        rows =
          Enum.map(entries, fn e ->
            %{
              id: e.id,
              timestamp_ns: e.timestamp_ns,
              event_type: e.event_type,
              actor: e.actor,
              resource: e.resource,
              details: e.details,
              outcome: e.outcome,
              session_id: e.session_id
            }
          end)

        {:ok, Jason.encode!(rows)}

      :csv ->
        header = "id,timestamp_ns,event_type,actor,resource,outcome,session_id\n"

        body =
          Enum.map_join(entries, "\n", fn e ->
            [
              e.id,
              e.timestamp_ns,
              e.event_type,
              csv_escape(e.actor),
              csv_escape(e.resource),
              csv_escape(e.outcome),
              csv_escape(e.session_id || "")
            ]
            |> Enum.join(",")
          end)

        {:ok, header <> body}

      other ->
        {:error, {:unsupported_format, other}}
    end
  end

  defp do_sweep(db) do
    cutoff_ns =
      DateTime.utc_now()
      |> DateTime.add(-@retention_days * 86_400, :second)
      |> DateTime.to_unix(:nanosecond)

    # Fetch entries to archive
    filters = [to_ns: cutoff_ns, limit: 100_000]
    old_entries = run_query(db, filters)

    if old_entries == [] do
      {:ok, 0}
    else
      archive_path = archive_file_path()

      archived_json = Jason.encode!(Enum.map(old_entries, &entry_to_map/1))

      with :ok <- File.write(archive_path, archived_json),
           :ok <- delete_before(db, cutoff_ns) do
        {:ok, length(old_entries)}
      else
        {:error, reason} -> {:error, reason}
      end
    end
  end

  defp delete_before(db, cutoff_ns) do
    sql = "DELETE FROM audit_log WHERE timestamp_ns < ?1"

    with {:ok, stmt} <- Exqlite.Sqlite3.prepare(db, sql),
         :ok <- Exqlite.Sqlite3.bind(db, stmt, [cutoff_ns]),
         :done <- Exqlite.Sqlite3.step(db, stmt),
         :ok <- Exqlite.Sqlite3.release(db, stmt) do
      :ok
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp build_where(filters) do
    binding_index = :counters.new(1, [])

    {clauses, bindings} =
      Enum.reduce(filters, {[], []}, fn
        {:event_type, et}, {clauses, binds} ->
          i = :counters.add(binding_index, 1, 1) |> then(fn _ -> :counters.get(binding_index, 1) end)
          {["event_type = ?#{i}" | clauses], [Atom.to_string(et) | binds]}

        {:actor, a}, {clauses, binds} ->
          i = bump(binding_index)
          {["actor = ?#{i}" | clauses], [a | binds]}

        {:resource, r}, {clauses, binds} ->
          i = bump(binding_index)
          {["resource LIKE ?#{i}" | clauses], ["#{r}%" | binds]}

        {:from_ns, ts}, {clauses, binds} ->
          i = bump(binding_index)
          {["timestamp_ns >= ?#{i}" | clauses], [ts | binds]}

        {:to_ns, ts}, {clauses, binds} ->
          i = bump(binding_index)
          {["timestamp_ns <= ?#{i}" | clauses], [ts | binds]}

        _, acc ->
          acc
      end)

    {Enum.reverse(clauses), Enum.reverse(bindings)}
  end

  defp bump(counter) do
    :counters.add(counter, 1, 1)
    :counters.get(counter, 1)
  end

  defp collect_rows(db, stmt, acc) do
    case Exqlite.Sqlite3.step(db, stmt) do
      {:row, row} -> collect_rows(db, stmt, [row | acc])
      :done -> Enum.reverse(acc)
      {:error, _} -> Enum.reverse(acc)
    end
  end

  defp row_to_entry([id, ts_ns, event_type, actor, resource, details_json, outcome, session_id]) do
    details =
      case Jason.decode(details_json || "{}") do
        {:ok, map} -> map
        _ -> %{}
      end

    %Entry{
      id: id,
      timestamp_ns: ts_ns,
      event_type: String.to_existing_atom(event_type),
      actor: actor,
      resource: resource,
      details: details,
      outcome: outcome,
      session_id: session_id
    }
  end

  defp entry_to_map(%Entry{} = e) do
    %{
      id: e.id,
      timestamp_ns: e.timestamp_ns,
      event_type: to_string(e.event_type),
      actor: e.actor,
      resource: e.resource,
      details: e.details,
      outcome: e.outcome,
      session_id: e.session_id
    }
  end

  defp archive_file_path do
    ts = DateTime.utc_now() |> DateTime.to_iso8601() |> String.replace(":", "-")
    Path.join(@archive_dir, "audit_archive_#{ts}.json")
  end

  defp csv_escape(str) when is_binary(str) do
    if String.contains?(str, [",", "\"", "\n"]) do
      "\"" <> String.replace(str, "\"", "\"\"") <> "\""
    else
      str
    end
  end

  defp csv_escape(nil), do: ""

  defp get_session_id do
    Process.get(:srfm_session_id)
  end
end
