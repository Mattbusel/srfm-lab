# Market Data Service

A Go microservice that aggregates L2 order book data from Alpaca (primary) and Binance
(fallback), assembles 15-minute OHLCV bars, and broadcasts them over WebSocket to all
subscribers. Provides the unified bar feed that the live trader and signal engine consume.

Status: **live in production**, running at `:8780`.

---

## Purpose

The live trader needs clean, synchronized 15-minute bars across 21 instruments. Sourcing
these from Alpaca alone creates a single point of failure: if the Alpaca WebSocket drops,
the live trader goes blind. This service abstracts the feed problem away: it manages dual
feeds, handles failover, and delivers a stable stream regardless of which upstream is live.

It also provides the L2 order book for the smart router: before submitting an order, the
execution layer queries the current bid-ask spread to select the routing tier
(Alpaca / Binance / block).

---

## Architecture

```
market-data/
  main.go                     -- service entry point, initializes all components
  feeds/
    alpaca_feed.go            -- Alpaca WebSocket client (primary feed)
    binance_feed.go           -- Binance @depth10@100ms fallback feed
  aggregator/
    bar_aggregator.go         -- tick-to-OHLCV bar assembly (15-minute windows)
  storage/
    bar_store.go              -- SQLite bar persistence (WAL mode)
  streaming/
    hub.go                    -- WebSocket broadcast hub, fan-out to subscribers
  api/
    handlers.go               -- REST endpoints: /bars, /snapshot, /stream, /health
  monitoring/
    metrics.go                -- Prometheus metrics exporter
    feed_monitor.go           -- spread sampling, failover state machine
```

---

## Dual Feed and Failover

The service maintains two concurrent L2 connections:

**Alpaca** (`feeds/alpaca_feed.go`):
- Subscribes to WebSocket messages type `'o'` (order book updates) and `'q'` (quotes)
- Applies exponential backoff on disconnect (1s, 2s, 4s... up to 60s)
- Marks itself as PRIMARY when connected

**Binance** (`feeds/binance_feed.go`):
- Subscribes to `<symbol>@depth10@100ms` WebSocket streams
- Symbol mapping: `BTCUSD` on Alpaca maps to `BTCUSDT` on Binance
- Marks itself as FALLBACK when Alpaca is connected, PRIMARY when Alpaca is down

**Failover logic** (`monitoring/feed_monitor.go`):
- Polls feed health every 30 seconds
- If Alpaca feed has not received an update in 30s: switch to Binance PRIMARY
- If Alpaca reconnects: wait 60s before switching back (hysteresis prevents flapping)
- Failover events are logged and exposed via Prometheus counter

The `BookManager` (exported to the Python execution layer) exposes a unified book that
always reflects the current PRIMARY feed.

---

## Bar Assembly

`aggregator/bar_aggregator.go` converts tick updates to OHLCV bars:

```go
type BarAggregator struct {
    windowSize time.Duration  // 15 minutes default
    bars       map[string]*Bar
    mu         sync.Mutex
}

func (a *BarAggregator) OnTick(symbol string, price, size float64, ts time.Time) {
    // accumulate into open/high/low/close/volume for current window
    // when window closes: emit completed bar, start new window
}
```

Bar windows are anchored to wall clock (00:00, 00:15, 00:30...), not to the first
tick received. This ensures bars from different instruments are aligned in time, which
is a requirement for the 3-timeframe BH engine (1h and 4h bars are assembled from 4
and 16 consecutive 15m bars respectively).

---

## Storage

`storage/bar_store.go` persists completed bars to `market_data.db` (SQLite, WAL mode):

```sql
CREATE TABLE bars (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL DEFAULT '15m',
    bar_time  TEXT    NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    vwap      REAL,
    trade_count INTEGER,
    UNIQUE(symbol, timeframe, bar_time)
);
```

The `bar_store` supports a `get_recent(symbol, timeframe, n)` query that returns the
last N completed bars. This is the bootstrap feed for the live trader on startup: it
reads recent bars to warm up the BH mass state before live processing begins.

---

## WebSocket Hub

`streaming/hub.go` implements a fan-out WebSocket hub:

```go
type Hub struct {
    clients    map[*Client]bool
    broadcast  chan []byte
    register   chan *Client
    unregister chan *Client
}
```

When a bar completes, the aggregator writes it to the broadcast channel. The hub
fan-outs to all connected clients. Clients that fall behind by more than 100 messages
are disconnected (slow consumer protection).

Message format:

```json
{
  "type":   "bar",
  "symbol": "BTCUSD",
  "tf":     "15m",
  "t":      "2026-04-06T14:15:00Z",
  "o":      82450.50,
  "h":      82510.00,
  "l":      82380.00,
  "c":      82490.00,
  "v":      142.35
}
```

---

## REST API

All endpoints served at `:8780`.

| Method | Path | Description |
|---|---|---|
| GET | `/bars?sym=BTCUSD&tf=15m&n=100` | Last N completed bars |
| GET | `/snapshot?sym=BTCUSD` | Current L2 book snapshot (top 10 levels) |
| GET | `/spread?sym=BTCUSD` | Current bid-ask spread in basis points |
| WS | `/stream` | Subscribe to bar events (all instruments) |
| GET | `/health` | Service health + feed status |
| GET | `/metrics` | Prometheus metrics endpoint |

The `/spread` endpoint is the primary integration point for the smart router. Before
each order submission, `execution/routing/smart_router.py` calls this endpoint to
determine the routing tier:
- <= 50 bps: Alpaca (best execution)
- 50-100 bps: Binance (better liquidity)
- > 100 bps: block trade or defer

---

## Prometheus Metrics

Exported at `/metrics`:

```
market_data_ticks_total{symbol, feed}       -- tick count per symbol per feed
market_data_bars_completed_total{symbol, tf} -- completed bars
market_data_failover_total                   -- number of primary feed switches
market_data_spread_bps{symbol}              -- current bid-ask spread
market_data_feed_latency_ms{feed}           -- last update latency
market_data_subscribers_active             -- WebSocket subscriber count
```

These are scraped by the Grafana dashboard and drive the `market_data_feed_latency_ms`
alert: if latency exceeds 5000ms, an alert fires to Slack.

---

## Configuration

```yaml
# config/instruments.yaml (relevant market-data section)
market_data:
  primary_feed: alpaca
  fallback_feed: binance
  failover_timeout_sec: 30
  bar_window_min: 15
  max_spread_bps: 200      # refuse to route if spread exceeds this
  symbols:
    - BTCUSD
    - ETHUSD
    - SOLUSD
    # ... 21 total
```

---

## Integration Points

| Consumer | Method | Data |
|---|---|---|
| `live_trader_alpaca.py` | REST `/bars` on bootstrap | Historical bars to warm BH state |
| `live_trader_alpaca.py` | WebSocket `/stream` | Real-time 15m bars |
| `execution/routing/smart_router.py` | REST `/spread` | Current spread for routing |
| `spacetime/api/main.py` | REST `/bars` | Bars for backtest replay |
| `research/live_monitor/` | WebSocket `/stream` | Live bar display |
| Grafana | `/metrics` | Feed health and latency monitoring |
| `cmd/alerter/` | `/health` | Health check for alert routing |

---

## Running

```bash
cd market-data
go run main.go

# Or from root via Makefile
make market-data

# Or via start_all.sh (recommended for production)
bash scripts/start_all.sh start
```

Health check:
```bash
curl http://localhost:8780/health
# {"status":"ok","primary_feed":"alpaca","connected":true,"subscribers":3}
```
