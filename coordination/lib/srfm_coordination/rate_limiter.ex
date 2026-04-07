defmodule SrfmCoordination.RateLimiter do
  @moduledoc """
  Token bucket rate limiter implemented as a GenServer with ETS backing.

  Each named limiter stores its state in an ETS row for O(1) concurrent reads.
  Tokens refill continuously at `rate` tokens/second up to `burst` maximum.
  The bucket state is updated lazily on each check_rate call.

  Pre-configured limiters (started automatically):
    :param_proposals  -- 10/minute per source (0.167/s, burst 10)
    :health_checks    -- 120/minute / 2/s, burst 10
    :api_requests     -- 1000/minute / 16.67/s, burst 100
    :broker_orders    -- 100/minute per broker / 1.67/s, burst 20

  Per-caller limiting: pass {name, caller_id} as the limiter key.
  """

  use GenServer
  require Logger

  @table :srfm_rate_limiter

  # Pre-configured limiter specs: {name, rate_per_second, burst}
  @default_limiters [
    {:param_proposals, 10 / 60, 10},
    {:health_checks, 2.0, 10},
    {:api_requests, 1000 / 60, 100},
    {:broker_orders, 100 / 60, 20}
  ]

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  @type limiter_name :: atom() | {atom(), term()}
  @type rate_result :: :ok | {:error, :rate_limited}

  @type bucket :: %{
          tokens: float(),
          rate: float(),
          burst: float(),
          last_refill_mono: integer()
        }

  # ---------------------------------------------------------------------------
  # Public API
  # ---------------------------------------------------------------------------

  @doc "Start the RateLimiter GenServer."
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Register a new named rate limiter.

  - name: atom or {atom, caller_id} tuple
  - rate: tokens per second (float)
  - burst: maximum token capacity

  Returns :ok if created, {:error, :already_exists} if already registered.
  Idempotent if the same config is re-registered.
  """
  @spec new(limiter_name(), float(), non_neg_integer()) :: :ok | {:error, term()}
  def new(name, rate, burst) when is_number(rate) and rate > 0 and is_integer(burst) and burst > 0 do
    GenServer.call(__MODULE__, {:new, name, float(rate), float(burst)})
  end

  @doc """
  Check whether `cost` tokens are available for `name`.

  Returns :ok and deducts tokens if sufficient tokens exist.
  Returns {:error, :rate_limited} if the bucket is empty.
  The bucket is refilled based on elapsed wall time before checking.
  """
  @spec check_rate(limiter_name(), pos_integer()) :: rate_result()
  def check_rate(name, cost \\ 1) when is_integer(cost) and cost > 0 do
    now_mono = System.monotonic_time(:millisecond)

    case :ets.lookup(@table, name) do
      [] ->
        {:error, :not_found}

      [{^name, bucket}] ->
        refilled = refill(bucket, now_mono)
        if refilled.tokens >= cost do
          updated = %{refilled | tokens: refilled.tokens - cost, last_refill_mono: now_mono}
          :ets.insert(@table, {name, updated})
          :ok
        else
          # Store the refilled state even on rejection so tokens accumulate
          :ets.insert(@table, {name, %{refilled | last_refill_mono: now_mono}})
          {:error, :rate_limited}
        end
    end
  end

  @doc "Return the number of whole tokens currently available for `name`."
  @spec remaining(limiter_name()) :: non_neg_integer() | {:error, :not_found}
  def remaining(name) do
    now_mono = System.monotonic_time(:millisecond)

    case :ets.lookup(@table, name) do
      [] ->
        {:error, :not_found}

      [{^name, bucket}] ->
        refilled = refill(bucket, now_mono)
        floor(refilled.tokens)
    end
  end

  @doc "Reset the bucket for `name` to full capacity."
  @spec reset(limiter_name()) :: :ok | {:error, :not_found}
  def reset(name) do
    case :ets.lookup(@table, name) do
      [] ->
        {:error, :not_found}

      [{^name, bucket}] ->
        full = %{bucket | tokens: bucket.burst, last_refill_mono: System.monotonic_time(:millisecond)}
        :ets.insert(@table, {name, full})
        :ok
    end
  end

  @doc "Delete a named limiter entirely."
  @spec delete(limiter_name()) :: :ok
  def delete(name) do
    :ets.delete(@table, name)
    :ok
  end

  @doc "List all registered limiter names."
  @spec list() :: [limiter_name()]
  def list do
    :ets.tab2list(@table)
    |> Enum.map(fn {name, _} -> name end)
  end

  @doc "Return the full bucket state for inspection."
  @spec inspect_bucket(limiter_name()) :: bucket() | nil
  def inspect_bucket(name) do
    case :ets.lookup(@table, name) do
      [{^name, bucket}] -> bucket
      [] -> nil
    end
  end

  @doc """
  Ensure a per-caller bucket exists for {name, caller_id}.
  Inherits the rate/burst from the parent named limiter.
  Creates the per-caller bucket if absent.
  """
  @spec ensure_caller_bucket(atom(), term()) :: :ok | {:error, :not_found}
  def ensure_caller_bucket(name, caller_id) do
    caller_key = {name, caller_id}

    case :ets.lookup(@table, caller_key) do
      [_] ->
        :ok

      [] ->
        case :ets.lookup(@table, name) do
          [] ->
            {:error, :not_found}

          [{^name, parent}] ->
            bucket = new_bucket(parent.rate, parent.burst)
            :ets.insert_new(@table, {caller_key, bucket})
            :ok
        end
    end
  end

  # ---------------------------------------------------------------------------
  # GenServer callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(_opts) do
    :ets.new(@table, [
      :named_table,
      :set,
      :public,
      write_concurrency: true,
      read_concurrency: true
    ])

    # Register all default limiters
    Enum.each(@default_limiters, fn {name, rate, burst} ->
      bucket = new_bucket(float(rate), float(burst))
      :ets.insert(@table, {name, bucket})
      Logger.debug("[RateLimiter] Registered #{inspect(name)} rate=#{rate}/s burst=#{burst}")
    end)

    Logger.info("[RateLimiter] Initialized with #{length(@default_limiters)} default limiters")
    {:ok, %{}}
  end

  @impl true
  def handle_call({:new, name, rate, burst}, _from, state) do
    case :ets.lookup(@table, name) do
      [{^name, _}] ->
        {:reply, {:error, :already_exists}, state}

      [] ->
        bucket = new_bucket(rate, burst)
        :ets.insert(@table, {name, bucket})
        Logger.debug("[RateLimiter] Created limiter #{inspect(name)} rate=#{rate}/s burst=#{burst}")
        {:reply, :ok, state}
    end
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  @spec new_bucket(float(), float()) :: bucket()
  defp new_bucket(rate, burst) do
    %{
      tokens: burst,
      rate: rate,
      burst: burst,
      last_refill_mono: System.monotonic_time(:millisecond)
    }
  end

  @spec refill(bucket(), integer()) :: bucket()
  defp refill(bucket, now_mono) do
    elapsed_sec = max(0, (now_mono - bucket.last_refill_mono) / 1_000)
    added = elapsed_sec * bucket.rate
    new_tokens = min(bucket.burst, bucket.tokens + added)
    %{bucket | tokens: new_tokens}
  end

  defp float(n) when is_integer(n), do: n * 1.0
  defp float(n) when is_float(n), do: n
end
