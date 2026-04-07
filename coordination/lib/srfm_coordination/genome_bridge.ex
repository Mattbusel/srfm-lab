defmodule SrfmCoordination.GenomeBridge do
  @moduledoc """
  HTTP client bridge to the Go IAE genome engine.
  Handles genome proposal validation, parameter fan-out, and rollback coordination.

  Connects to the IAE genome service at @iae_base_url and polls for generation
  statistics on a configurable interval. All HTTP calls are wrapped in the
  :iae circuit breaker to prevent cascading failures.

  Events published to EventBus:
    :genome_updated        -- a new best genome was accepted
    :genome_rejected       -- a proposed genome update failed validation
    :generation_complete   -- IAE completed a full evolution generation
  """

  use GenServer
  require Logger

  @iae_base_url "http://localhost:8782"
  @poll_interval_ms 30_000
  @http_timeout_ms 10_000

  -- GenomeSummary struct captures the key metrics from one generation snapshot.
  defmodule GenomeSummary do
    @moduledoc "Snapshot of the current genome population state."
    @enforce_keys [:generation, :best_fitness, :avg_fitness, :diversity, :top_genome]
    defstruct [
      :generation,
      :best_fitness,
      :avg_fitness,
      :diversity,
      :top_genome
    ]

    @type t :: %__MODULE__{
      generation:   non_neg_integer(),
      best_fitness: float(),
      avg_fitness:  float(),
      diversity:    float(),
      top_genome:   map()
    }
  end

  defstruct [
    poll_ref:           nil,
    last_generation:    nil,
    last_best_fitness:  nil,
    poll_count:         0,
    error_count:        0
  ]

  -- ---------------------------------------------------------------------------
  -- Public API
  -- ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Return the best genome from the IAE engine.
  Hits GET /genome/best via the :iae circuit breaker.
  """
  @spec get_current_genome() :: {:ok, map()} | {:error, term()}
  def get_current_genome do
    GenServer.call(__MODULE__, :get_current_genome, 15_000)
  end

  @doc """
  Return population-level statistics from the IAE engine.
  Hits GET /genome/population via the :iae circuit breaker.
  """
  @spec get_population_stats() :: {:ok, map()} | {:error, term()}
  def get_population_stats do
    GenServer.call(__MODULE__, :get_population_stats, 15_000)
  end

  @doc """
  Validate a proposed genome update and forward it to the IAE engine.
  `genome_id` is a string identifier; `params` is a map of parameter overrides.
  Returns :ok on success or {:error, reason} if validation or delivery fails.
  """
  @spec propose_genome_update(String.t(), map()) :: :ok | {:error, term()}
  def propose_genome_update(genome_id, params) when is_binary(genome_id) and is_map(params) do
    GenServer.call(__MODULE__, {:propose_genome_update, genome_id, params}, 20_000)
  end

  @doc """
  Instruct the IAE engine to roll back to its previous stable genome.
  `reason` is a string describing why the rollback was triggered.
  """
  @spec force_rollback(String.t()) :: :ok
  def force_rollback(reason) when is_binary(reason) do
    GenServer.cast(__MODULE__, {:force_rollback, reason})
  end

  @doc "Return the current bridge state snapshot."
  @spec status() :: map()
  def status do
    GenServer.call(__MODULE__, :status)
  end

  -- ---------------------------------------------------------------------------
  -- GenServer callbacks
  -- ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    Logger.info("[GenomeBridge] Initializing IAE genome bridge -> #{iae_base_url()}")
    ref = schedule_poll()
    {:ok, %__MODULE__{poll_ref: ref}}
  end

  @impl true
  def handle_call(:get_current_genome, _from, state) do
    result = http_get("/genome/best")
    {:reply, result, state}
  end

  @impl true
  def handle_call(:get_population_stats, _from, state) do
    result = http_get("/genome/population")
    {:reply, result, state}
  end

  @impl true
  def handle_call({:propose_genome_update, genome_id, params}, _from, state) do
    result = do_propose_genome_update(genome_id, params)

    case result do
      :ok ->
        publish(:genome_updated, %{genome_id: genome_id, params: params})

      {:error, reason} ->
        Logger.warning("[GenomeBridge] Genome proposal #{genome_id} rejected: #{inspect(reason)}")
        publish(:genome_rejected, %{genome_id: genome_id, reason: reason})
    end

    {:reply, result, state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    info = %{
      last_generation:   state.last_generation,
      last_best_fitness: state.last_best_fitness,
      poll_count:        state.poll_count,
      error_count:       state.error_count,
      iae_url:           iae_base_url()
    }
    {:reply, info, state}
  end

  @impl true
  def handle_cast({:force_rollback, reason}, state) do
    Logger.warning("[GenomeBridge] Force rollback triggered: #{reason}")

    payload = Jason.encode!(%{reason: reason, triggered_at: DateTime.utc_now()})

    case circuit_post("/genome/rollback", payload) do
      {:ok, _} ->
        Logger.info("[GenomeBridge] IAE rollback acknowledged")

      {:error, err} ->
        Logger.error("[GenomeBridge] Rollback POST failed: #{inspect(err)}")
    end

    {:noreply, state}
  end

  @impl true
  def handle_info(:poll_generation, state) do
    new_state = do_poll_generation(state)
    ref = schedule_poll()
    {:noreply, %{new_state | poll_ref: ref}}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  @impl true
  def terminate(reason, _state) do
    Logger.info("[GenomeBridge] Terminating: #{inspect(reason)}")
    :ok
  end

  -- ---------------------------------------------------------------------------
  -- Private -- polling
  -- ---------------------------------------------------------------------------

  defp do_poll_generation(state) do
    case http_get("/genome/population") do
      {:ok, body} ->
        generation   = Map.get(body, "generation",    state.last_generation)
        best_fitness = Map.get(body, "best_fitness",  state.last_best_fitness)
        avg_fitness  = Map.get(body, "avg_fitness",   0.0)
        diversity    = Map.get(body, "diversity",     0.0)
        top_genome   = Map.get(body, "top_genome",    %{})

        summary = %GenomeSummary{
          generation:   generation,
          best_fitness: best_fitness,
          avg_fitness:  avg_fitness,
          diversity:    diversity,
          top_genome:   top_genome
        }

        Logger.debug(
          "[GenomeBridge] Generation #{generation} polled -- " <>
          "best=#{Float.round(best_fitness || 0.0, 4)} avg=#{Float.round(avg_fitness, 4)}"
        )

        -- Emit generation_complete when generation number advances
        if generation != nil and generation != state.last_generation and state.last_generation != nil do
          publish(:generation_complete, %{summary: summary})
        end

        %{state |
          last_generation:   generation,
          last_best_fitness: best_fitness,
          poll_count:        state.poll_count + 1
        }

      {:error, reason} ->
        Logger.warning("[GenomeBridge] Poll failed: #{inspect(reason)}")
        %{state | error_count: state.error_count + 1}
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- genome proposal
  -- ---------------------------------------------------------------------------

  defp do_propose_genome_update(genome_id, params) do
    -- Step 1: validate the genome params locally before forwarding
    with :ok <- validate_genome_params(params) do
      payload = Jason.encode!(%{
        genome_id: genome_id,
        parameters: params,
        proposed_at: DateTime.utc_now()
      })

      case circuit_post("/genome/propose", payload) do
        {:ok, %{"status" => "accepted"}} ->
          Logger.info("[GenomeBridge] Genome #{genome_id} accepted by IAE")
          :ok

        {:ok, %{"status" => "rejected", "reason" => reason}} ->
          {:error, {:iae_rejected, reason}}

        {:ok, body} ->
          Logger.debug("[GenomeBridge] Unexpected IAE response: #{inspect(body)}")
          :ok

        {:error, reason} ->
          {:error, reason}
      end
    end
  end

  defp validate_genome_params(params) when map_size(params) == 0 do
    {:error, :empty_params}
  end

  defp validate_genome_params(params) do
    -- Sanity-check that all values are numeric or boolean
    invalid =
      Enum.reject(params, fn {_k, v} ->
        is_number(v) or is_boolean(v) or is_binary(v)
      end)

    if invalid == [] do
      :ok
    else
      {:error, {:invalid_param_types, Enum.map(invalid, fn {k, _} -> k end)}}
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- HTTP helpers with circuit breaker
  -- ---------------------------------------------------------------------------

  defp http_get(path) do
    url = iae_base_url() <> path

    result =
      SrfmCoordination.CircuitBreaker.call(:iae, fn ->
        HTTPoison.get(url, [], recv_timeout: @http_timeout_ms, timeout: @http_timeout_ms)
      end)

    case result do
      {:ok, {:ok, %{status_code: code, body: body}}} when code in 200..299 ->
        case Jason.decode(body) do
          {:ok, decoded} -> {:ok, decoded}
          {:error, err}  -> {:error, {:json_decode, err}}
        end

      {:ok, {:ok, %{status_code: code}}} ->
        {:error, {:bad_status, code}}

      {:ok, {:error, reason}} ->
        {:error, reason}

      {:error, :circuit_open} ->
        {:error, :circuit_open}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp circuit_post(path, json_body) do
    url = iae_base_url() <> path

    result =
      SrfmCoordination.CircuitBreaker.call(:iae, fn ->
        HTTPoison.post(
          url,
          json_body,
          [{"content-type", "application/json"}],
          recv_timeout: @http_timeout_ms,
          timeout: @http_timeout_ms
        )
      end)

    case result do
      {:ok, {:ok, %{status_code: code, body: body}}} when code in 200..299 ->
        case Jason.decode(body) do
          {:ok, decoded} -> {:ok, decoded}
          {:error, _}    -> {:ok, %{}}
        end

      {:ok, {:ok, %{status_code: code, body: body}}} ->
        detail = safe_decode(body)
        {:error, {:bad_status, code, detail}}

      {:ok, {:error, reason}} ->
        {:error, reason}

      {:error, :circuit_open} ->
        {:error, :circuit_open}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp safe_decode(body) do
    case Jason.decode(body) do
      {:ok, map} -> map
      _ -> body
    end
  end

  -- ---------------------------------------------------------------------------
  -- Private -- EventBus publishing
  -- ---------------------------------------------------------------------------

  defp publish(type, payload) do
    case Process.whereis(SrfmCoordination.EventBus) do
      nil ->
        Logger.debug("[GenomeBridge] EventBus not running, dropping :#{type} event")

      _ ->
        SrfmCoordination.EventBus.publish(
          :hypothesis_generated,
          Map.merge(payload, %{type: type})
        )
    end
  end

  defp schedule_poll do
    interval = Application.get_env(:srfm_coordination, :genome_poll_interval_ms, @poll_interval_ms)
    Process.send_after(self(), :poll_generation, interval)
  end

  defp iae_base_url do
    Application.get_env(:srfm_coordination, :iae_base_url, @iae_base_url)
  end
end
