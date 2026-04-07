defmodule SrfmCoordination.Scheduler do
  @moduledoc """
  Cron-like job scheduler implemented as a GenServer.

  Supports standard 5-field cron expressions: min hour dom mon dow.
  Each job runs in an isolated Task.Supervisor task so failures are contained.

  Retry logic: on failure, retries up to 3 times with 30-second backoff.

  Pre-scheduled jobs:
    "daily_health_report" -- 0 9 * * 1-5  (weekdays at 9am UTC)
    "genome_poll"         -- */5 * * * *  (every 5 minutes)
    "circuit_reset_check" -- */1 * * * *  (every minute)
    "metrics_flush"       -- */2 * * * *  (every 2 minutes)

  The scheduler ticks every 30 seconds. Jobs are fired when the current
  time matches the cron expression, deduplicated within the same minute.
  """

  use GenServer
  require Logger

  @tick_ms 30_000
  @max_retries 3
  @retry_backoff_ms 30_000

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  @type cron_expr :: String.t()
  @type mfa_tuple :: {module(), atom(), list()}

  @type job_def :: %{
          name: String.t(),
          cron_expr: cron_expr(),
          mfa: mfa_tuple(),
          opts: keyword(),
          last_run: DateTime.t() | nil,
          next_run: DateTime.t() | nil,
          run_count: non_neg_integer(),
          last_error: term() | nil,
          enabled: boolean()
        }

  # Parsed cron field -- either :any or a MapSet of integers
  @type cron_field :: :any | MapSet.t()

  @type parsed_cron :: %{
          minute: cron_field(),
          hour: cron_field(),
          dom: cron_field(),
          month: cron_field(),
          dow: cron_field()
        }

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc "Start the Scheduler GenServer."
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Register a recurring job.

  - name: unique string identifier
  - cron_expr: 5-field cron string (e.g. "*/5 * * * *")
  - mfa: {Module, :function, [args]} tuple
  - opts: optional keyword list
    - enabled: boolean (default true)
    - timeout_ms: max run time in ms (default 60_000)
    - max_retries: override retry limit (default #{@max_retries})

  Returns :ok or {:error, reason}.
  """
  @spec schedule(String.t(), cron_expr(), mfa_tuple(), keyword()) ::
          :ok | {:error, term()}
  def schedule(name, cron_expr, mfa, opts \\ []) do
    GenServer.call(__MODULE__, {:schedule, name, cron_expr, mfa, opts})
  end

  @doc "Remove a scheduled job by name."
  @spec cancel(String.t()) :: :ok | {:error, :not_found}
  def cancel(name) do
    GenServer.call(__MODULE__, {:cancel, name})
  end

  @doc "Return all registered job definitions."
  @spec list_jobs() :: [job_def()]
  def list_jobs do
    GenServer.call(__MODULE__, :list_jobs)
  end

  @doc "Immediately trigger a job regardless of its schedule."
  @spec trigger_now(String.t()) :: :ok | {:error, term()}
  def trigger_now(name) do
    GenServer.call(__MODULE__, {:trigger_now, name})
  end

  @doc "Enable or disable a job without removing it."
  @spec set_enabled(String.t(), boolean()) :: :ok | {:error, :not_found}
  def set_enabled(name, enabled) when is_boolean(enabled) do
    GenServer.call(__MODULE__, {:set_enabled, name, enabled})
  end

  @doc "Get a single job definition by name."
  @spec get_job(String.t()) :: {:ok, job_def()} | {:error, :not_found}
  def get_job(name) do
    GenServer.call(__MODULE__, {:get_job, name})
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  defstruct jobs: %{}, task_supervisor: nil, last_minute: nil

  @impl true
  def init(_opts) do
    {:ok, sup} = Task.Supervisor.start_link(name: SrfmCoordination.Scheduler.TaskSup)

    # Schedule first tick
    :timer.send_interval(@tick_ms, :tick)

    state = %__MODULE__{jobs: %{}, task_supervisor: sup, last_minute: nil}
    state = register_default_jobs(state)

    Logger.info("[Scheduler] Initialized with #{map_size(state.jobs)} default jobs")
    {:ok, state}
  end

  @impl true
  def handle_call({:schedule, name, cron_expr, mfa, opts}, _from, state) do
    case parse_cron(cron_expr) do
      {:ok, parsed} ->
        now = DateTime.utc_now()

        job = %{
          name: name,
          cron_expr: cron_expr,
          parsed_cron: parsed,
          mfa: mfa,
          opts: opts,
          last_run: nil,
          next_run: next_run_time(parsed, now),
          run_count: 0,
          last_error: nil,
          enabled: Keyword.get(opts, :enabled, true)
        }

        updated = Map.put(state.jobs, name, job)
        Logger.info("[Scheduler] Registered job #{name} cron=#{cron_expr}")
        {:reply, :ok, %{state | jobs: updated}}

      {:error, reason} ->
        {:reply, {:error, {:invalid_cron, reason}}, state}
    end
  end

  @impl true
  def handle_call({:cancel, name}, _from, state) do
    case Map.fetch(state.jobs, name) do
      {:ok, _} ->
        updated = Map.delete(state.jobs, name)
        Logger.info("[Scheduler] Cancelled job #{name}")
        {:reply, :ok, %{state | jobs: updated}}

      :error ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call(:list_jobs, _from, state) do
    jobs = Map.values(state.jobs) |> Enum.map(&public_job_def/1)
    {:reply, jobs, state}
  end

  @impl true
  def handle_call({:trigger_now, name}, _from, state) do
    case Map.fetch(state.jobs, name) do
      {:ok, job} ->
        pid = self()
        Task.Supervisor.start_child(state.task_supervisor, fn ->
          run_job_with_retry(job, 0, pid)
        end)
        {:reply, :ok, state}

      :error ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:set_enabled, name, enabled}, _from, state) do
    case Map.fetch(state.jobs, name) do
      {:ok, job} ->
        updated = Map.put(state.jobs, name, %{job | enabled: enabled})
        {:reply, :ok, %{state | jobs: updated}}

      :error ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:get_job, name}, _from, state) do
    case Map.fetch(state.jobs, name) do
      {:ok, job} -> {:reply, {:ok, public_job_def(job)}, state}
      :error -> {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_info(:tick, state) do
    now = DateTime.utc_now()
    current_minute = {now.year, now.month, now.day, now.hour, now.minute}

    # Skip if we already fired this minute
    if current_minute == state.last_minute do
      {:noreply, state}
    else
      {updated_jobs, _} =
        Enum.map_reduce(state.jobs, self(), fn {name, job}, scheduler_pid ->
          job =
            if job.enabled and cron_matches?(job.parsed_cron, now) do
              Task.Supervisor.start_child(state.task_supervisor, fn ->
                run_job_with_retry(job, 0, scheduler_pid)
              end)

              Logger.debug("[Scheduler] Fired job #{name}")
              %{job | last_run: now, run_count: job.run_count + 1}
            else
              job
            end

          {{name, job}, scheduler_pid}
        end)

      {:noreply, %{state | jobs: Map.new(updated_jobs), last_minute: current_minute}}
    end
  end

  @impl true
  def handle_info({:job_complete, name, result}, state) do
    case Map.fetch(state.jobs, name) do
      {:ok, job} ->
        job =
          case result do
            :ok ->
              %{job | last_error: nil}

            {:error, reason} ->
              Logger.error("[Scheduler] Job #{name} failed: #{inspect(reason)}")
              %{job | last_error: reason}
          end

        {:noreply, %{state | jobs: Map.put(state.jobs, name, job)}}

      :error ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # ---------------------------------------------------------------------------
  # Job execution with retry
  # ---------------------------------------------------------------------------

  defp run_job_with_retry(job, attempt, scheduler_pid) when attempt <= @max_retries do
    max_retries = Keyword.get(job.opts, :max_retries, @max_retries)
    timeout_ms = Keyword.get(job.opts, :timeout_ms, 60_000)

    {mod, fun, args} = job.mfa

    task = Task.async(fn -> apply(mod, fun, args) end)

    result =
      case Task.yield(task, timeout_ms) || Task.shutdown(task) do
        {:ok, :ok} -> :ok
        {:ok, {:ok, _}} -> :ok
        {:ok, {:error, reason}} -> {:error, reason}
        {:ok, other} -> {:ok, other}
        nil -> {:error, :timeout}
        {:exit, reason} -> {:error, {:exit, reason}}
      end

    case result do
      :ok ->
        send(scheduler_pid, {:job_complete, job.name, :ok})

      {:error, reason} when attempt < max_retries ->
        Logger.warning(
          "[Scheduler] Job #{job.name} attempt #{attempt + 1} failed: #{inspect(reason)}, retrying in #{@retry_backoff_ms}ms"
        )

        Process.sleep(@retry_backoff_ms)
        run_job_with_retry(job, attempt + 1, scheduler_pid)

      {:error, reason} ->
        Logger.error("[Scheduler] Job #{job.name} failed after #{attempt + 1} attempts: #{inspect(reason)}")
        send(scheduler_pid, {:job_complete, job.name, {:error, reason}})

      {:ok, _} ->
        send(scheduler_pid, {:job_complete, job.name, :ok})
    end
  end

  # ---------------------------------------------------------------------------
  # Cron parser
  # ---------------------------------------------------------------------------

  @doc false
  @spec parse_cron(String.t()) :: {:ok, parsed_cron()} | {:error, String.t()}
  def parse_cron(expr) when is_binary(expr) do
    parts = String.split(String.trim(expr), ~r/\s+/)

    if length(parts) != 5 do
      {:error, "expected 5 fields, got #{length(parts)}"}
    else
      [min, hour, dom, mon, dow] = parts

      with {:ok, pmin} <- parse_field(min, 0, 59),
           {:ok, phour} <- parse_field(hour, 0, 23),
           {:ok, pdom} <- parse_field(dom, 1, 31),
           {:ok, pmon} <- parse_field(mon, 1, 12),
           {:ok, pdow} <- parse_field(dow, 0, 6) do
        {:ok, %{minute: pmin, hour: phour, dom: pdom, month: pmon, dow: pdow}}
      end
    end
  end

  defp parse_field("*", _min, _max), do: {:ok, :any}

  defp parse_field("*/" <> step_str, min, max) do
    case Integer.parse(step_str) do
      {step, ""} when step > 0 ->
        values = Enum.filter(min..max, fn n -> rem(n - min, step) == 0 end)
        {:ok, MapSet.new(values)}

      _ ->
        {:error, "invalid step: #{step_str}"}
    end
  end

  defp parse_field(field, min, max) do
    if String.contains?(field, ",") do
      # List: "1,2,3"
      results =
        String.split(field, ",")
        |> Enum.map(&parse_single_or_range(&1, min, max))

      case Enum.find(results, &match?({:error, _}, &1)) do
        nil ->
          values = Enum.flat_map(results, fn {:ok, set} -> MapSet.to_list(set) end)
          {:ok, MapSet.new(values)}

        err ->
          err
      end
    else
      parse_single_or_range(field, min, max)
    end
  end

  defp parse_single_or_range(str, field_min, field_max) do
    if String.contains?(str, "-") do
      [lo, hi] = String.split(str, "-", parts: 2)

      with {lo_int, ""} <- Integer.parse(lo),
           {hi_int, ""} <- Integer.parse(hi),
           true <- lo_int >= field_min and hi_int <= field_max and lo_int <= hi_int do
        {:ok, MapSet.new(lo_int..hi_int)}
      else
        _ -> {:error, "invalid range: #{str}"}
      end
    else
      case Integer.parse(str) do
        {n, ""} when n >= field_min and n <= field_max ->
          {:ok, MapSet.new([n])}

        _ ->
          {:error, "invalid value: #{str} (expected #{field_min}-#{field_max})"}
      end
    end
  end

  defp cron_matches?(parsed, %DateTime{} = dt) do
    field_match?(parsed.minute, dt.minute) and
      field_match?(parsed.hour, dt.hour) and
      field_match?(parsed.dom, dt.day) and
      field_match?(parsed.month, dt.month) and
      field_match?(parsed.dow, day_of_week(dt))
  end

  defp field_match?(:any, _val), do: true
  defp field_match?(set, val), do: MapSet.member?(set, val)

  # Returns 0 (Sunday) through 6 (Saturday), matching cron convention
  defp day_of_week(%DateTime{} = dt) do
    # Calendar.ISO.day_of_week returns 1=Monday..7=Sunday
    # Convert to 0=Sunday..6=Saturday
    case Calendar.ISO.day_of_week(dt.year, dt.month, dt.day, :monday) do
      {7, _, _} -> 0
      {n, _, _} -> n
      n when is_integer(n) and n == 7 -> 0
      n when is_integer(n) -> n
    end
  end

  # Simple next-run calculation: finds next matching minute
  defp next_run_time(_parsed, now) do
    # Approximate -- just show next full minute
    DateTime.add(now, 60, :second)
  end

  # ---------------------------------------------------------------------------
  # Default job registration
  # ---------------------------------------------------------------------------

  defp register_default_jobs(state) do
    defaults = [
      {"daily_health_report", "0 9 * * 1-5",
       {SrfmCoordination.HealthMonitor, :generate_daily_report, []}, []},

      {"genome_poll", "*/5 * * * *",
       {SrfmCoordination.GenomeReceiver, :poll_now, []}, []},

      {"circuit_reset_check", "*/1 * * * *",
       {SrfmCoordination.CircuitBreakerSupervisor, :check_resets, []}, []},

      {"metrics_flush", "*/2 * * * *",
       {SrfmCoordination.MetricsBridge, :flush, []}, []}
    ]

    Enum.reduce(defaults, state, fn {name, expr, mfa, opts}, acc ->
      case parse_cron(expr) do
        {:ok, parsed} ->
          now = DateTime.utc_now()

          job = %{
            name: name,
            cron_expr: expr,
            parsed_cron: parsed,
            mfa: mfa,
            opts: opts,
            last_run: nil,
            next_run: next_run_time(parsed, now),
            run_count: 0,
            last_error: nil,
            enabled: true
          }

          %{acc | jobs: Map.put(acc.jobs, name, job)}

        {:error, reason} ->
          Logger.error("[Scheduler] Failed to parse default cron #{expr}: #{reason}")
          acc
      end
    end)
  end

  # Strip internal parsed_cron field from public-facing struct
  defp public_job_def(job) do
    Map.take(job, [:name, :cron_expr, :mfa, :last_run, :next_run, :run_count, :last_error, :enabled])
  end
end
