defmodule SrfmCoordination.HTTP.Router do
  @moduledoc """
  Plug router exposing the coordination API on port 8781.

  Endpoints:
    GET  /health                        — system health summary
    GET  /services                      — all registered services with status
    GET  /metrics                       — aggregated metrics snapshot
    GET  /events                        — recent events per topic
    POST /services/:name/restart        — trigger restart of a managed service
    POST /parameters                    — apply parameter delta (validated)
    GET  /circuit-breakers              — circuit breaker states
    POST /halt                          — emergency halt (sets global halt flag)
  """

  use Plug.Router
  require Logger

  plug Plug.Logger, log: :debug
  plug Plug.Parsers,
    parsers: [:json],
    pass: ["application/json"],
    json_decoder: Jason
  plug :match
  plug :dispatch

  # ---------------------------------------------------------------------------
  # Health
  # ---------------------------------------------------------------------------

  get "/health" do
    SrfmCoordination.HTTP.HealthController.get_health(conn)
  end

  get "/health/services" do
    SrfmCoordination.HTTP.HealthController.get_services_health(conn)
  end

  # ---------------------------------------------------------------------------
  # Services
  # ---------------------------------------------------------------------------

  get "/services" do
    services = SrfmCoordination.ServiceRegistry.list_all()
    json(conn, 200, %{services: services, count: length(services)})
  end

  post "/services/:name/restart" do
    service_name = String.to_existing_atom(conn.params["name"])

    case SrfmCoordination.ServiceSupervisor.restart_service(service_name) do
      {:ok, _pid} ->
        json(conn, 200, %{status: "restarting", service: conn.params["name"]})

      {:error, :not_found} ->
        json(conn, 404, %{error: "service not found", service: conn.params["name"]})

      {:error, reason} ->
        json(conn, 500, %{error: inspect(reason)})
    end
  rescue
    ArgumentError ->
      json(conn, 400, %{error: "unknown service name"})
  end

  # ---------------------------------------------------------------------------
  # Metrics
  # ---------------------------------------------------------------------------

  get "/metrics" do
    snapshot = SrfmCoordination.MetricsCollector.snapshot()
    json(conn, 200, %{metrics: snapshot, collected_at: DateTime.utc_now()})
  end

  # ---------------------------------------------------------------------------
  # Events
  # ---------------------------------------------------------------------------

  get "/events" do
    topic_param = conn.params["topic"]
    limit = parse_int(conn.params["limit"], 50)

    events =
      if topic_param do
        topic = String.to_existing_atom(topic_param)
        SrfmCoordination.EventBus.history(topic, limit)
      else
        SrfmCoordination.EventBus.topics()
        |> Enum.flat_map(fn t -> SrfmCoordination.EventBus.history(t, 20) end)
        |> Enum.sort_by(& &1.timestamp, {:desc, DateTime})
        |> Enum.take(limit)
      end

    json(conn, 200, %{events: events, count: length(events)})
  rescue
    ArgumentError ->
      json(conn, 400, %{error: "invalid topic"})
  end

  # ---------------------------------------------------------------------------
  # Parameters
  # ---------------------------------------------------------------------------

  post "/parameters" do
    case conn.body_params do
      %{"delta" => delta, "author" => author} when is_map(delta) ->
        string_delta = Map.new(delta, fn {k, v} -> {to_string(k), v} end)

        case SrfmCoordination.ParameterCoordinator.apply_delta(string_delta, author) do
          :ok ->
            json(conn, 200, %{status: "applied", keys: Map.keys(string_delta)})

          {:error, reason} ->
            json(conn, 422, %{error: inspect(reason)})
        end

      %{"delta" => delta} when is_map(delta) ->
        string_delta = Map.new(delta, fn {k, v} -> {to_string(k), v} end)

        case SrfmCoordination.ParameterCoordinator.apply_delta(string_delta) do
          :ok ->
            json(conn, 200, %{status: "applied", keys: Map.keys(string_delta)})

          {:error, reason} ->
            json(conn, 422, %{error: inspect(reason)})
        end

      _ ->
        json(conn, 400, %{error: "body must contain {delta: {key: value, ...}}"})
    end
  end

  get "/parameters" do
    all = SrfmCoordination.ParameterCoordinator.all()
    json(conn, 200, %{parameters: all, count: map_size(all)})
  end

  # ---------------------------------------------------------------------------
  # Circuit Breakers
  # ---------------------------------------------------------------------------

  get "/circuit-breakers" do
    statuses = SrfmCoordination.CircuitBreaker.all_statuses()
    json(conn, 200, %{circuit_breakers: statuses})
  end

  post "/circuit-breakers/:name/reset" do
    circuit = String.to_existing_atom(conn.params["name"])
    SrfmCoordination.CircuitBreaker.reset(circuit)
    json(conn, 200, %{status: "reset", circuit: conn.params["name"]})
  rescue
    ArgumentError ->
      json(conn, 400, %{error: "unknown circuit name"})
  end

  # ---------------------------------------------------------------------------
  # Emergency Halt
  # ---------------------------------------------------------------------------

  post "/halt" do
    reason = get_in(conn.body_params, ["reason"]) || "manual halt"
    Logger.error("[Router] EMERGENCY HALT requested: #{reason}")

    :persistent_term.put(:srfm_halt, true)

    SrfmCoordination.EventBus.publish(:alert, %{
      type: :emergency_halt,
      reason: reason,
      initiated_at: DateTime.utc_now()
    })

    SrfmCoordination.AlertManager.alert(
      :emergency_halt,
      :emergency,
      "EMERGENCY HALT: #{reason}",
      %{source: "http_api"}
    )

    json(conn, 200, %{status: "halted", reason: reason})
  end

  get "/halt" do
    halted = :persistent_term.get(:srfm_halt, false)
    json(conn, 200, %{halted: halted})
  end

  # ---------------------------------------------------------------------------
  # Catch-all
  # ---------------------------------------------------------------------------

  match _ do
    json(conn, 404, %{error: "not found", path: conn.request_path})
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp json(conn, status, data) do
    body = Jason.encode!(data, pretty: false)

    conn
    |> Plug.Conn.put_resp_content_type("application/json")
    |> Plug.Conn.send_resp(status, body)
  end

  defp parse_int(nil, default), do: default
  defp parse_int(str, default) do
    case Integer.parse(str) do
      {n, _} when n > 0 -> min(n, 1000)
      _ -> default
    end
  end
end
