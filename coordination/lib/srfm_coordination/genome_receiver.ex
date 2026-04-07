defmodule SrfmCoordination.GenomeReceiver do
  @moduledoc """
  Polls the Go IAE genome evolution endpoint and applies validated parameter
  updates safely.

  Polling:
  -- GET :8780/genome/best every 5 minutes
  -- Decodes a float64 array to named parameters (mirrors Python GenomeDecoder)
  -- Validates decoded params via ParameterCoordinator.validate_schema/1

  Safety guards:
  -- Fitness gate: new_genome_fitness must exceed current_params_fitness + 0.05
  -- Anti-thrashing: no proposal if a rollback occurred within the last 2 hours
  -- Incremental application: if any single param delta > 25%, splits into
     2-3 incremental steps toward the target

  Events published:
  -- genome_applied  -- params accepted and fan-out initiated
  -- genome_rejected -- params rejected with reason

  REST:
    GET /genome/status  -- last received genome, decision, reason, timestamp

  ETS table: :srfm_genome_state
    :last_genome         -> map
    :last_decision       -> :applied | :rejected | :pending | nil
    :last_reason         -> string
    :last_polled_at      -> DateTime
    :current_fitness     -> float
    :last_rollback_at    -> DateTime | nil
  """

  use GenServer
  require Logger

  @poll_interval_ms 300_000   # 5 minutes
  @http_timeout_ms 10_000
  @iae_base_url Application.compile_env(:srfm_coordination, :iae_url, "http://localhost:8780")
  @fitness_improvement_threshold 0.05
  @anti_thrash_window_seconds 7_200   # 2 hours
  @large_delta_threshold 0.25         # 25% change triggers incremental steps

  @state_table :srfm_genome_state

  # Parameter layout mirrors Python GenomeDecoder (index -> name mapping).
  # Adjust to match the actual production genome spec.
  @param_layout [
    "momentum_lookback",
    "bh_mass_threshold",
    "vol_scaling_factor",
    "entry_z_score",
    "exit_z_score",
    "stop_loss_atr_mult",
    "take_profit_atr_mult",
    "position_size_pct",
    "regime_smooth_alpha",
    "rebalance_freq_minutes"
  ]

  defstruct poll_ref: nil, schema: nil

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Force an immediate genome poll."
  def poll_now do
    GenServer.cast(__MODULE__, :poll_now)
  end

  @doc "Return the last genome status snapshot."
  @spec get_status() :: map()
  def get_status do
    %{
      last_genome: ets_get(:last_genome, nil),
      last_decision: ets_get(:last_decision, nil),
      last_reason: ets_get(:last_reason, "none"),
      last_polled_at: ets_get(:last_polled_at, nil),
      current_fitness: ets_get(:current_fitness, 0.0),
      last_rollback_at: ets_get(:last_rollback_at, nil)
    }
  end

  @doc "Notify the receiver that a rollback occurred. Engages anti-thrash guard."
  @spec notify_rollback() :: :ok
  def notify_rollback do
    GenServer.cast(__MODULE__, :notify_rollback)
  end

  @doc "Update the tracked current fitness (called after successful param apply)."
  @spec set_current_fitness(float()) :: :ok
  def set_current_fitness(fitness) when is_float(fitness) do
    :ets.insert(@state_table, {:current_fitness, fitness})
    :ok
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@state_table, [:named_table, :set, :public, read_concurrency: true])

    :ets.insert(@state_table, {:last_genome, nil})
    :ets.insert(@state_table, {:last_decision, nil})
    :ets.insert(@state_table, {:last_reason, "not_yet_polled"})
    :ets.insert(@state_table, {:last_polled_at, nil})
    :ets.insert(@state_table, {:current_fitness, 0.0})
    :ets.insert(@state_table, {:last_rollback_at, nil})

    schema = load_param_schema()
    ref = schedule_poll()

    Logger.info("[GenomeReceiver] Initialized -- polling IAE at #{@iae_base_url} every #{div(@poll_interval_ms, 60_000)}min")
    {:ok, %__MODULE__{poll_ref: ref, schema: schema}}
  end

  @impl true
  def handle_cast(:poll_now, state) do
    do_poll(state)
    {:noreply, state}
  end

  @impl true
  def handle_cast(:notify_rollback, state) do
    now = DateTime.utc_now()
    :ets.insert(@state_table, {:last_rollback_at, now})
    Logger.info("[GenomeReceiver] Rollback recorded at #{now} -- anti-thrash guard active for 2 hours")
    {:noreply, state}
  end

  @impl true
  def handle_info(:poll, state) do
    ref = schedule_poll()
    do_poll(state)
    {:noreply, %{state | poll_ref: ref}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[GenomeReceiver] Terminating: #{inspect(reason)}")
    :ok
  end

  # ---------------------------------------------------------------------------
  # Poll logic
  # ---------------------------------------------------------------------------

  defp do_poll(state) do
    now = DateTime.utc_now()
    :ets.insert(@state_table, {:last_polled_at, now})

    case fetch_best_genome() do
      {:ok, genome_payload} ->
        process_genome(genome_payload, state)

      {:error, reason} ->
        Logger.warning("[GenomeReceiver] Failed to fetch genome: #{inspect(reason)}")
    end
  end

  defp fetch_best_genome do
    url = "#{@iae_base_url}/genome/best"

    try do
      case HTTPoison.get(url, [], recv_timeout: @http_timeout_ms) do
        {:ok, %{status_code: 200, body: body}} ->
          {:ok, Jason.decode!(body)}

        {:ok, %{status_code: 404}} ->
          {:error, :no_genome_available}

        {:ok, %{status_code: code}} ->
          {:error, {:bad_status, code}}

        {:error, %HTTPoison.Error{reason: reason}} ->
          {:error, reason}
      end
    catch
      kind, reason -> {:error, {kind, reason}}
    end
  end

  # ---------------------------------------------------------------------------
  # Genome processing pipeline
  # ---------------------------------------------------------------------------

  defp process_genome(%{"genome" => genes, "fitness" => new_fitness} = payload, state)
       when is_list(genes) and is_number(new_fitness) do

    :ets.insert(@state_table, {:last_genome, payload})

    current_fitness = ets_get(:current_fitness, 0.0)

    with :ok <- check_anti_thrash(),
         :ok <- check_fitness_improvement(new_fitness, current_fitness),
         {:ok, named_params} <- decode_genome(genes),
         :ok <- validate_params(named_params, state.schema) do

      current_params = SrfmCoordination.ParameterCoordinator.all()
      apply_genome(named_params, current_params, new_fitness)
    else
      {:error, reason} ->
        log_rejection(reason)
    end
  end

  defp process_genome(payload, _state) do
    Logger.warning("[GenomeReceiver] Unexpected genome payload shape: #{inspect(payload)}")
    log_rejection(:malformed_payload)
  end

  # ---------------------------------------------------------------------------
  # Guards
  # ---------------------------------------------------------------------------

  defp check_anti_thrash do
    case ets_get(:last_rollback_at, nil) do
      nil ->
        :ok

      last_rollback ->
        elapsed = DateTime.diff(DateTime.utc_now(), last_rollback, :second)

        if elapsed < @anti_thrash_window_seconds do
          remaining_min = div(@anti_thrash_window_seconds - elapsed, 60)
          {:error, {:anti_thrash, remaining_min}}
        else
          :ok
        end
    end
  end

  defp check_fitness_improvement(new_fitness, current_fitness) do
    if new_fitness > current_fitness + @fitness_improvement_threshold do
      :ok
    else
      {:error, {:insufficient_improvement,
        %{new: new_fitness, current: current_fitness,
          required_delta: @fitness_improvement_threshold}}}
    end
  end

  # ---------------------------------------------------------------------------
  # Genome decoding
  # ---------------------------------------------------------------------------

  @doc false
  def decode_genome(genes) when is_list(genes) do
    if length(genes) < length(@param_layout) do
      {:error, {:genome_too_short, %{expected: length(@param_layout), got: length(genes)}}}
    else
      params =
        @param_layout
        |> Enum.with_index()
        |> Map.new(fn {name, idx} ->
          raw = Enum.at(genes, idx)
          {name, normalize_gene(name, raw)}
        end)

      {:ok, params}
    end
  end

  # Apply domain-specific clamping and rounding per parameter
  defp normalize_gene("momentum_lookback", v),      do: clamp(round(abs(v)), 2, 200)
  defp normalize_gene("bh_mass_threshold", v),      do: clamp(v, 0.1, 10.0)
  defp normalize_gene("vol_scaling_factor", v),     do: clamp(v, 0.1, 5.0)
  defp normalize_gene("entry_z_score", v),          do: clamp(abs(v), 0.5, 4.0)
  defp normalize_gene("exit_z_score", v),           do: clamp(abs(v), 0.1, 3.0)
  defp normalize_gene("stop_loss_atr_mult", v),     do: clamp(abs(v), 0.5, 5.0)
  defp normalize_gene("take_profit_atr_mult", v),   do: clamp(abs(v), 0.5, 10.0)
  defp normalize_gene("position_size_pct", v),      do: clamp(abs(v), 0.01, 0.25)
  defp normalize_gene("regime_smooth_alpha", v),    do: clamp(abs(v), 0.01, 0.99)
  defp normalize_gene("rebalance_freq_minutes", v), do: clamp(round(abs(v)), 1, 1440)
  defp normalize_gene(_name, v) when is_number(v),  do: v * 1.0
  defp normalize_gene(_name, v),                    do: v

  defp clamp(v, lo, hi), do: max(lo, min(hi, v))

  # ---------------------------------------------------------------------------
  # Validation
  # ---------------------------------------------------------------------------

  defp validate_params(params, schema) do
    case Process.whereis(SrfmCoordination.ParameterCoordinator) do
      nil ->
        Logger.warning("[GenomeReceiver] ParameterCoordinator not running -- skipping schema validation")
        :ok

      _pid ->
        if function_exported?(SrfmCoordination.ParameterCoordinator, :validate_schema, 1) do
          SrfmCoordination.ParameterCoordinator.validate_schema(params)
        else
          validate_with_local_schema(params, schema)
        end
    end
  end

  defp validate_with_local_schema(_params, nil), do: :ok

  defp validate_with_local_schema(params, schema) when is_map(schema) do
    errors =
      Enum.flat_map(schema, fn {key, constraints} ->
        case Map.fetch(params, key) do
          {:ok, value} -> check_constraints(key, value, constraints)
          :error -> []
        end
      end)

    if errors == [] do
      :ok
    else
      {:error, {:schema_validation_failed, errors}}
    end
  end

  defp check_constraints(key, value, %{"min" => min, "max" => max}) do
    cond do
      not is_number(value) -> [{key, :not_numeric}]
      value < min -> [{key, {:below_min, value, min}}]
      value > max -> [{key, {:above_max, value, max}}]
      true -> []
    end
  end

  defp check_constraints(_key, _value, _constraints), do: []

  # ---------------------------------------------------------------------------
  # Incremental application
  # ---------------------------------------------------------------------------

  defp apply_genome(new_params, current_params, new_fitness) do
    steps = build_application_steps(new_params, current_params)
    step_count = length(steps)

    Logger.info("[GenomeReceiver] Applying genome in #{step_count} step(s)")

    results =
      steps
      |> Enum.with_index(1)
      |> Enum.reduce_while(:ok, fn {step_params, step_num}, _acc ->
        Logger.info("[GenomeReceiver] Applying step #{step_num}/#{step_count}")

        case SrfmCoordination.ParameterCoordinator.apply_delta(
               step_params,
               "genome_receiver_step_#{step_num}_of_#{step_count}"
             ) do
          :ok ->
            {:cont, :ok}

          {:error, reason} ->
            Logger.error("[GenomeReceiver] Step #{step_num} failed: #{inspect(reason)}")
            {:halt, {:error, reason}}
        end
      end)

    case results do
      :ok ->
        :ets.insert(@state_table, {:current_fitness, new_fitness * 1.0})
        :ets.insert(@state_table, {:last_decision, :applied})
        :ets.insert(@state_table, {:last_reason, "fitness_improved_by_#{Float.round(new_fitness - ets_get(:current_fitness, 0.0), 4)}"})

        # Record in parameter history
        record_in_history(current_params, new_params, new_fitness)

        emit_event(:genome_applied, %{
          fitness: new_fitness,
          step_count: step_count,
          param_count: map_size(new_params)
        })

        Logger.info("[GenomeReceiver] Genome applied successfully (fitness=#{new_fitness})")

      {:error, reason} ->
        log_rejection({:apply_failed, reason})
    end
  end

  # Build intermediate steps when any param delta exceeds the large-delta threshold.
  # Returns a list of delta maps to apply sequentially.
  @doc false
  def build_application_steps(new_params, current_params) when is_map(new_params) do
    # Find the maximum fractional change across all numeric params
    max_pct_change =
      new_params
      |> Enum.flat_map(fn {key, new_val} ->
        case Map.fetch(current_params, key) do
          {:ok, %{value: old_val}} when is_number(old_val) and is_number(new_val) and old_val != 0 ->
            [abs((new_val - old_val) / old_val)]

          {:ok, old_val} when is_number(old_val) and is_number(new_val) and old_val != 0 ->
            [abs((new_val - old_val) / old_val)]

          _ ->
            []
        end
      end)
      |> case do
        [] -> 0.0
        deltas -> Enum.max(deltas)
      end

    num_steps =
      cond do
        max_pct_change > 0.50 -> 3
        max_pct_change > @large_delta_threshold -> 2
        true -> 1
      end

    if num_steps == 1 do
      [new_params]
    else
      Enum.map(1..num_steps, fn step ->
        fraction = step / num_steps

        Map.new(new_params, fn {key, new_val} ->
          old_val =
            case Map.fetch(current_params, key) do
              {:ok, %{value: v}} when is_number(v) -> v
              {:ok, v} when is_number(v) -> v
              _ -> new_val
            end

          interpolated =
            if is_number(old_val) and is_number(new_val) do
              old_val + (new_val - old_val) * fraction
            else
              new_val
            end

          {key, interpolated}
        end)
      end)
    end
  end

  defp record_in_history(current_params, new_params, new_fitness) do
    case Process.whereis(SrfmCoordination.ParameterHistory) do
      nil -> :ok
      _pid ->
        # Extract plain values from current_params (which may be wrapped maps)
        old_plain = Map.new(current_params, fn
          {k, %{value: v}} -> {k, v}
          {k, v} -> {k, v}
        end)

        current_fitness = ets_get(:current_fitness, 0.0)

        SrfmCoordination.ParameterHistory.record_update(
          old_plain,
          new_params,
          "genome_receiver",
          current_fitness,
          new_fitness
        )
    end
  end

  # ---------------------------------------------------------------------------
  # Event emission
  # ---------------------------------------------------------------------------

  defp log_rejection(reason) do
    reason_str = inspect(reason)
    :ets.insert(@state_table, {:last_decision, :rejected})
    :ets.insert(@state_table, {:last_reason, reason_str})

    Logger.info("[GenomeReceiver] Genome rejected: #{reason_str}")

    emit_event(:genome_rejected, %{reason: reason_str})
  end

  defp emit_event(type, extra) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil -> :ok
      _pid ->
        SrfmCoordination.EventBus.publish(:alert, Map.merge(%{type: type}, extra))
    end
  end

  # ---------------------------------------------------------------------------
  # Schema loading
  # ---------------------------------------------------------------------------

  defp load_param_schema do
    path =
      Application.get_env(:srfm_coordination, :param_schema_path) ||
        Path.join([File.cwd!(), "param_schema.json"])

    case File.read(path) do
      {:ok, contents} ->
        case Jason.decode(contents) do
          {:ok, schema} ->
            Logger.info("[GenomeReceiver] Loaded param schema from #{path}")
            schema

          {:error, reason} ->
            Logger.warning("[GenomeReceiver] Failed to parse param_schema.json: #{inspect(reason)}")
            nil
        end

      {:error, _} ->
        Logger.info("[GenomeReceiver] param_schema.json not found at #{path} -- validation will be minimal")
        nil
    end
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp schedule_poll do
    Process.send_after(self(), :poll, @poll_interval_ms)
  end

  defp ets_get(key, default) do
    case :ets.lookup(@state_table, key) do
      [{^key, value}] -> value
      [] -> default
    end
  end
end
