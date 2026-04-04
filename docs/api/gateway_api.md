# Go Gateway API Reference

The Go gateway (`cmd/gateway/`) proxies requests to the FastAPI backend, adds JWT authentication, rate limiting, and WebSocket fan-out.

Base URL: `http://localhost:9000`

---

## Authentication

All endpoints (except `/health`) require a Bearer JWT token:

```
Authorization: Bearer <jwt_token>
```

### `POST /auth/token`

**Request**: `{ "username": "...", "password": "..." }`

**Response**: `{ "token": "eyJ...", "expires_at": "2024-01-16T00:00:00Z" }`

---

## Proxied Endpoints

All endpoints under `/api/*` and `/ws/*` are proxied to the FastAPI backend with authentication enforcement.

---

## Rate Limits

| Endpoint | Rate limit |
|---------|-----------|
| `POST /api/backtest` | 10 per minute |
| `POST /api/mc` | 5 per minute |
| `GET /api/*` | 60 per minute |
| `WS /ws/live` | 5 concurrent connections per user |

---

## `GET /health`

Returns gateway and backend health.

**Response**: `{ "gateway": "ok", "backend": "ok", "uptime_seconds": 3600 }`

---

## `GET /metrics`

Prometheus-format metrics for monitoring.

---

## Gateway Configuration

Configure in `config/gateway.yaml`:

```yaml
listen_addr: ":9000"
backend_url: "http://localhost:8000"
jwt_secret: "${JWT_SECRET}"
rate_limit:
  backtest_per_min: 10
  general_per_min: 60
cors_origins:
  - "http://localhost:5173"
  - "http://localhost:5174"
```
