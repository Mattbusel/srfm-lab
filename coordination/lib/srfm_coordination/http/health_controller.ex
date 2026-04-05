defmodule SrfmCoordination.HTTP.HealthController do
  @moduledoc """
  Handles health-related HTTP endpoints.

  GET /health  — Full system health summary:
    {
      overall: "healthy" | "degraded" | "critical" | "no_services",
      score: 97.5,
      services: [...],
      uptime_seconds: 3600,
      event_count_24h: 1042,
      circuit_breakers: [...],
      checked_at: "2025-01-01T00:00:00Z"
    }

  GET /health/services — Per-service health detail with history summary.
  """

  require Logger
  import Plug.Conn

  # ---------------------------------------------------------------------------
  # Public handlers (called from Router)
  # ---------------------------------------------------------------------------

  @doc "Full system health summary — the top-level liveness/readiness response."
  def get_health(conn) do
    summary = SrfmCoordination.HealthMonitor.system_health()
    services = SrfmCoordination.ServiceRegistry.list_all()
    circuits = safe_circuit_statuses()
    event_count = count_events_24h()

    body = %{
      overall: summary.overall,
      score: summary.score,
      uptime_seconds: summary.uptime_seconds,
      event_count_24h: event_count,
      service_counts: summary.counts,
      services: Enum.map(services, &format_service/1),
      circuit_breakers: circuits,
      checked_at: DateTime.to_iso8601(summary.checked_at)
    }

    status_code = health_status_to_http(summary.overall)

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status_code, Jason.encode!(body))
  end

  @doc "Per-service health detail with recent check history."
  def get_services_health(conn) do
    services = SrfmCoordination.ServiceRegistry.list_all()

    detailed =
      Enum.map(services, fn svc ->
        history = SrfmCoordination.HealthMonitor.history(svc.name)
        recent = Enum.take(history, 10)

        consecutive_failures =
          Enum.take_while(recent, fn h -> h.status == :error end) |> length()

        avg_response_ms =
          recent
          |> Enum.filter(fn h -> h.response_ms != nil end)
          |> case do
            [] -> nil
            samples ->
              total = Enum.sum(Enum.map(samples, & &1.response_ms))
              Float.round(total / length(samples), 1)
          end

        format_service(svc)
        |> Map.merge(%{
          consecutive_failures: consecutive_failures,
          avg_response_ms: avg_response_ms,
          recent_checks: Enum.map(recent, &format_check/1)
        })
      end)

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(200, Jason.encode!(%{services: detailed, count: length(detailed)}))
  end

  # ---------------------------------------------------------------------------
  # Private helpers
  # ---------------------------------------------------------------------------

  defp format_service(svc) do
    %{
      name: svc.name,
      port: svc.port,
      health_status: svc.health_status,
      last_heartbeat: format_dt(svc.last_heartbeat),
      restart_count: svc.restart_count,
      registered_at: format_dt(svc.registered_at)
    }
  end

  defp format_check(check) do
    %{
      status: check.status,
      response_ms: check.response_ms,
      checked_at: format_dt(check.checked_at),
      error: format_error(check.error)
    }
  end

  defp format_dt(nil), do: nil
  defp format_dt(%DateTime{} = dt), do: DateTime.to_iso8601(dt)
  defp format_dt(other), do: inspect(other)

  defp format_error(nil), do: nil
  defp format_error({kind, reason}), do: "#{kind}: #{inspect(reason)}"
  defp format_error(other), do: inspect(other)

  defp safe_circuit_statuses do
    try do
      SrfmCoordination.CircuitBreaker.all_statuses()
      |> Enum.map(fn s ->
        Map.update(s, :last_failure, nil, fn
          nil -> nil
          {dt, reason} -> %{at: format_dt(dt), reason: inspect(reason)}
        end)
      end)
    catch
      _, _ -> []
    end
  end

  defp count_events_24h do
    cutoff = DateTime.add(DateTime.utc_now(), -86_400, :second)

    SrfmCoordination.EventBus.topics()
    |> Enum.flat_map(fn topic -> SrfmCoordination.EventBus.history(topic, 1_000) end)
    |> Enum.count(fn event ->
      case Map.get(event, :timestamp) do
        nil -> false
        ts -> DateTime.compare(ts, cutoff) == :gt
      end
    end)
  end

  defp health_status_to_http(:healthy), do: 200
  defp health_status_to_http(:no_services), do: 200
  defp health_status_to_http(:degraded), do: 200
  defp health_status_to_http(:critical), do: 503
  defp health_status_to_http(_), do: 200
end
