# Execution Stack -- Deep Dive

> SRFM Lab · LARSA v16 · Live Alpaca (paper) deployment  
> Last updated: 2026-04-05

This document traces every layer of the live execution infrastructure, from raw
WebSocket bytes to SQLite trade records, with code-level detail for each
component.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [L2 Orderbook System](#2-l2-orderbook-system)
   - 2.1 OrderBook data structure
   - 2.2 AlpacaL2Feed -- primary feed
   - 2.3 BinanceL2Feed -- hot standby
   - 2.4 BookManager -- failover orchestrator
   - 2.5 FeedMonitor -- metrics logger
3. [Spread-Tier Routing](#3-spread-tier-routing)
4. [SmartRouter](#4-smartrouter)
5. [Live Trader Order Flow](#5-live-trader-order-flow)
   - 5.1 on_bar() entry point
   - 5.2 compute_targets() -- LARSA v16 sizing
   - 5.3 _apply_signal_overrides()
   - 5.4 _place_order() -- Alpaca submission
   - 5.5 _on_fill() -- trade logging
6. [Signal Overrides](#6-signal-overrides)
7. [SQLite Trade Logging](#7-sqlite-trade-logging)
8. [Process Supervision](#8-process-supervision)
9. [Docker Compose](#9-docker-compose)
10. [IAE Improvements Applied Live](#10-iae-improvements-applied-live)
11. [Worked Example -- End-to-End Order Trace](#11-worked-example--end-to-end-order-trace)

---

## 1. System Overview

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         SRFM Lab -- Live Stack                       │
 │                                                                     │
 │  ┌──────────────┐     15m bars      ┌──────────────────────────┐   │
 │  │ Alpaca Data  │ ──────────────── ▶│  LiveTrader (LARSA v16)  │   │
 │  │ Stream (WSS) │                   │  tools/live_trader_       │   │
 │  └──────────────┘                   │  alpaca.py                │   │
 │                                     └────────────┬─────────────┘   │
 │  ┌──────────────┐  spread/depth      ▼            │ orders          │
 │  │ BookManager  │ ◀────────────── SmartRouter     │                 │
 │  │  (dual feed) │                   │             ▼                 │
 │  │  Alpaca L2   │                   │    Alpaca TradingClient       │
 │  │  Binance L2  │                   │    (market / IOC limit)       │
 │  └──────────────┘                   │                               │
 │                                     │ fills (trade_updates WSS)     │
 │  ┌──────────────┐                   ▼                               │
 │  │ FeedMonitor  │          _on_fill() → SQLite                      │
 │  │ logs/         │          execution/live_trades.db                │
 │  │ orderbook_    │                                                   │
 │  │ metrics.jsonl │  ┌──────────────────────────────────┐            │
 │  └──────────────┘  │ Supervisor :8790                  │            │
 │                    │  GET /status  POST /restart/{svc} │            │
 │                    └──────────────────────────────────┘            │
 └─────────────────────────────────────────────────────────────────────┘
```

Five services run under Docker Compose (or the Python supervisor directly):
`market-data`, `coordination`, `bridge`, `autonomous-loop`, `live-trader`.
A standalone `supervisor` container exposes an HTTP API at `:8790`.

---

## 2. L2 Orderbook System

### 2.1 OrderBook Data Structure

**File:** `execution/orderbook/orderbook.py`

The `OrderBook` class is a thread-safe, single-symbol Level-2 book. Internally
bids and asks are plain Python `dict[float, float]` (price → cumulative qty).

Key design decisions:

| Decision | Rationale |
|---|---|
| `dict` not heap | Max 25 levels; O(n) scan is faster than heap overhead for n < 30 |
| `threading.Lock` | Feed writers are asyncio tasks; strategy readers may be synchronous |
| `max_levels=25` cap | Prevents unbounded memory growth on noisy feeds |
| VWAP walk raises on thin books | Forces upstream code (SmartRouter) to handle liquidity failures explicitly |

**Core properties:**

```python
book.best_bid      # max of bid keys
book.best_ask      # min of ask keys
book.mid_price     # (best_bid + best_ask) / 2
book.spread_bps    # (ask - bid) / mid * 10_000
book.imbalance     # (bid_qty - ask_qty) / (bid_qty + ask_qty)  ∈ [-1, 1]
```

**VWAP fill estimator:**

```python
def vwap_to_fill(self, side: str, qty: float) -> float:
    # Walks ask levels (buy) or bid levels (sell)
    # Raises InsufficientLiquidityError if depth < qty
```

The formula is:

```
                  Σ (fill_i × price_i)
VWAP_fill  =  ─────────────────────────
                       qty
```

where the sum walks levels from best price outward until `qty` is exhausted.

**Snapshot vs incremental updates:**

```python
book.apply_snapshot(bids, asks)  # replaces entire book (Alpaca 'o' messages)
book.update(side, price, qty)    # single level (Alpaca 'q' messages; qty=0 removes level)
```

---

### 2.2 AlpacaL2Feed -- Primary Feed

**File:** `execution/orderbook/alpaca_l2_feed.py`

Connects to:
```
wss://stream.data.alpaca.markets/v1beta3/crypto/us
```

**Message types handled:**

| Type | Description | Action |
|---|---|---|
| `"o"` | Full orderbook snapshot | `book.apply_snapshot(bids, asks)` -- replaces all levels |
| `"q"` | Quote update (best bid/ask only) | `book.update("bid", ...)` + `book.update("ask", ...)` |
| `"subscription"` / `"success"` | Control messages | Logged at DEBUG, ignored |

**Authentication sequence:**

```python
await ws.send(json.dumps({"action": "auth", "key": key, "secret": secret}))
# Waits up to 10s for {"T": "success", "msg": "authenticated"}
await ws.send(json.dumps({
    "action": "subscribe",
    "orderbooks": symbols,
    "quotes": symbols,
}))
```

**Reconnect / backoff:**

```
attempt 1:  wait 1.0s × jitter(0.9–1.1)
attempt 2:  wait 2.0s × jitter
attempt 3:  wait 4.0s × jitter
...
cap:        60s maximum
```

**Silence detection:**

```python
SILENCE_TIMEOUT = 30.0  # seconds

@property
def is_silent(self) -> bool:
    return (time.time() - self._last_message_ts) > SILENCE_TIMEOUT
```

Every successfully parsed message (including 'q' quotes) resets
`_last_message_ts`. BookManager polls `is_silent` every 5 seconds.

---

### 2.3 BinanceL2Feed -- Hot Standby

**File:** `execution/orderbook/binance_l2_feed.py`

Connects to the Binance public partial-book stream:
```
wss://stream.binance.com:9443/stream?streams=btcusdt@depth10@100ms/ethusdt@depth10@100ms/...
```

For a single symbol, the simpler single-stream URL is used instead.

**Symbol mapping (Alpaca → Binance):**

```python
SYMBOL_MAP = {
    "BTC/USD":  "BTCUSDT",
    "ETH/USD":  "ETHUSDT",
    "SOL/USD":  "SOLUSDT",
    "AVAX/USD": "AVAXUSDT",
    # ... 15 symbols total
}
```

For symbols not in `SYMBOL_MAP`, a generic fallback strips the `/` and appends
`USDT`:  `"XYZ/USD"` → `"XYZUSDT"`.

**Message format:**

Binance sends a complete top-10 snapshot every 100ms. Unlike Alpaca's
incremental diff stream, each message replaces the book:

```json
{
  "stream": "btcusdt@depth10@100ms",
  "data": {
    "bids": [["67500.01", "0.42"], ...],
    "asks": [["67500.10", "0.31"], ...]
  }
}
```

This is applied directly via `book.apply_snapshot(bids, asks)`.

**Why not the Binance diff stream?**
The `@depth10@100ms` partial book is simpler to manage (no local state to
maintain, no sequence number gap detection) and sufficient for spread/liquidity
checks at the 15-minute bar frequency the live trader operates on.

---

### 2.4 BookManager -- Failover Orchestrator

**File:** `execution/orderbook/book_manager.py`

BookManager owns both feeds and exposes a single unified interface to the
execution layer. The failover logic is:

```
Every 5s: _health_monitor() runs

  if Alpaca is silent (>30s) AND currently using Alpaca:
      _use_alpaca = False
      log.warning("switching to Binance")

  elif Alpaca recovered AND currently using Binance:
      _use_alpaca = True
      log.info("switching back to Alpaca")
```

```
            Alpaca WSS
         ┌─────────────┐  normal path
         │ AlpacaL2Feed│ ──────────────┐
         └─────────────┘               ▼
                                  BookManager
         ┌─────────────┐               ▲
         │ BinanceFeed │ ──────────────┘  fallback (Alpaca silent >30s)
         └─────────────┘
```

**Spread fallback logic:**

Even when Alpaca is the active feed, `get_spread_bps()` falls back to Binance
for a specific symbol if the primary returns `None`. This handles mid-session
gaps where Alpaca has a book for BTC but not for a lower-liquidity altcoin.

**Market-impact model:**

```
impact_bps = k × √(notional / ADV) × vol_bps

  k        = 0.5  (empirical constant, square-root model)
  ADV      = daily_volume_estimates[symbol] or $5,000,000
  vol_bps  = current spread in bps (proxy for short-term realised vol)
```

This is the Almgren-Chriss square-root market-impact formula. Using spread as
a vol proxy is an approximation valid for liquid crypto markets where bid-ask
spread is the dominant short-term friction.

**Public API surface:**

```python
bm = BookManager(symbols=["BTC/USD", "ETH/USD"], ...)
await bm.start()

bm.get_spread_bps("BTC/USD")      # → float | None
bm.get_mid("BTC/USD")             # → float | None
bm.is_liquid("BTC/USD", min_depth_usd=10_000)  # → bool
bm.estimate_impact_bps("BTC/USD", notional_usd=50_000)  # → float
bm.get_bid_ask("BTC/USD")         # → (bid, ask) | None
bm.active_feed_name               # → "alpaca" | "binance"
```

---

### 2.5 FeedMonitor -- Metrics Logger

**File:** `execution/orderbook/feed_monitor.py`

Every 60 seconds, `FeedMonitor` samples each tracked symbol and writes a
JSON-Lines record to `logs/orderbook_metrics.jsonl`:

```json
{
  "ts": 1743820800.123,
  "symbol": "BTC/USD",
  "active_feed": "alpaca",
  "spread_bps": 4.23,
  "mid_price": 67483.420000,
  "imbalance": 0.1842,
  "bid_depth_usd": 342187.50,
  "ask_depth_usd": 289443.20,
  "best_bid": 67481.50,
  "best_ask": 67485.34,
  "book_age_sec": 0.042,
  "bids_top5": [[67481.50, 0.42], ...],
  "asks_top5": [[67485.34, 0.31], ...],
  "alert": false
}
```

**Alert logic:**

A rolling window of the last 20 spread observations per symbol is maintained
in a `collections.deque`. If the current spread exceeds `3×` the rolling
average, an alert is logged at `WARNING` level and `"alert": true` is set in
the record:

```python
if spread > 3.0 * avg_spread and avg_spread > 0:
    log.warning("ALERT: %s spread=%.2fbps is >3x avg=%.2fbps", ...)
    record["alert"] = True
```

This catches sudden liquidity deterioration -- e.g., an exchange-side outage
that widens crypto spreads from 5bps to 50bps -- without requiring absolute
thresholds that would need per-symbol calibration.

---

## 3. Spread-Tier Routing

SmartRouter applies three tiers based on real-time spread from BookManager:

```
              Spread (bps)
              ┌──────────────────────────────────────────────┐
     ≤ 50 bps │  MARKET ORDER  -- normal execution path       │
              ├──────────────────────────────────────────────┤
  50 – 100 bps│  IOC LIMIT @ MID -- convert market to limit   │
              │  at current midprice, immediate-or-cancel    │
              ├──────────────────────────────────────────────┤
    > 100 bps │  THIN MARKET -- wait 5s, alert, REJECT        │
              └──────────────────────────────────────────────┘
```

**Why these thresholds matter for crypto liquidity:**

- **≤ 50 bps** (~20–40 bps for BTC/ETH in normal conditions): Market-taking
  cost is acceptable. Market orders fill immediately without meaningful
  slippage beyond the half-spread.

- **50–100 bps**: Elevated spread typically indicates reduced book depth,
  post-news volatility, or an off-hours session. A limit at mid captures the
  midpoint without paying the full spread. The IOC flag prevents the order
  from resting and being adversely selected.

- **> 100 bps**: Books are thin enough that even limit orders at mid face
  significant impact. The strategy's expected edge is below the friction cost
  at this spread. Rejecting the order preserves capital for conditions where
  the edge is positive.

For reference, BTC/USD typically trades at 3–8 bps spread on Alpaca. A 100 bps
spread implies either a circuit-breaker event or an instrument where the
strategy should not be active (e.g., SHIB/USD during illiquid hours).

```python
SPREAD_LIMIT_AT_MID_BPS = 50.0
SPREAD_THIN_MARKET_BPS  = 100.0
THIN_MARKET_DELAY_SEC   = 5.0
```

---

## 4. SmartRouter

**File:** `execution/routing/smart_router.py`

SmartRouter is the execution decision engine. It wraps the Alpaca broker
adapter and applies the following checks in sequence before every order:

```
route(order)
  │
  ├─ 1. Time gate: is current UTC hour in BLOCKED_HOURS?
  │      {1, 13, 14, 15, 17, 18} → raise RuntimeError immediately
  │
  ├─ 2. Liquidity gate (BookManager required):
  │      is_liquid(symbol, min_depth_usd=10_000) → False → raise RuntimeError
  │
  ├─ 3. TWAP split:
  │      notional > 2% of estimated ADV → delegate to TWAPExecutor
  │
  ├─ 4. Spread tier (market orders only):
  │      spread_bps from BookManager (falls back to legacy pct method)
  │        > 100 bps  → sleep 5s → raise RuntimeError
  │        50–100 bps → _try_limit_at_mid_bm() → IOC limit at midprice
  │        ≤ 50 bps   → fall through to normal execution
  │
  └─ 5. Submit with retry (max 3 attempts, exponential backoff 1s base)
```

**IOC limit at mid -- BookManager path:**

```python
def _try_limit_at_mid_bm(self, order) -> Optional[str]:
    mid = self._book_manager.get_mid(order.symbol)
    broker_id = self._broker.submit_limit_order(
        symbol        = order.symbol,
        qty           = order.quantity,
        side          = side_str,
        limit_price   = mid,
        time_in_force = "ioc",
    )
    return broker_id
```

If the limit placement itself fails (e.g., Alpaca rejects the price), the
function returns `None` and execution falls through to a market order.

**InsufficientLiquidityError handling:**

`OrderBook.vwap_to_fill()` raises `InsufficientLiquidityError` when book depth
is less than the requested quantity. SmartRouter does not call `vwap_to_fill`
directly, but `is_liquid()` uses `bid_depth_usd(n=5)` and `ask_depth_usd(n=5)`
as a pre-check proxy. If `is_liquid()` returns `False`, the order is rejected
before any broker call is made.

---

## 5. Live Trader Order Flow

**File:** `tools/live_trader_alpaca.py`  
**Strategy version:** `larsa_v16`

### 5.1 on_bar() Entry Point

Alpaca's `CryptoDataStream` delivers 15-minute OHLCV bars via a subscribed
async handler. Each bar triggers the full pipeline:

```python
async def bar_handler(bar: Any) -> None:
    trader.on_bar(bar.symbol, bar)
```

Inside `on_bar(ticker, bar)`:

1. Convert `bar.symbol` → internal `sym` (e.g., `"BTC/USD"` → `"BTC"`)
2. Update 15m `BHState` with close price
3. Update `GARCHTracker` with log-return
4. Update `OUDetector` with close price
5. Append to 1h buffer; flush to `bh_1h` / `atr_1h` on hourly boundary
6. Append to 4h buffer; flush to `bh_4h` / `atr_4h` on 4-hour boundary
7. On UTC midnight bar: `_on_daily_close()` → update Mayer EMA-200, recompute dynamic CORR
8. `_act_on_targets(bar_time)` → compute and execute

### 5.2 compute_targets() -- LARSA v16 Sizing

The sizing model in full:

```
1. Effective correlation factor:
   corr_factor = √(N + N(N-1) × dynamic_corr)
   per_inst_risk = DAILY_RISK / corr_factor

   dynamic_corr = 0.25 (normal)  or  0.60 (stress, when avg_pairwise_corr > 0.60)

2. Timeframe score:
   tf = 4·(bh_4h.active) + 2·(bh_1h.active) + 1·(bh_15m.active)
   ceiling = min(TF_CAP[tf], CRYPTO_CAP_FRAC=0.40)

3. Vol-adjusted base size:
   vol = ATR_1h / price × √6.5      (annualised from 1h ATR)
   base = min(per_inst_risk / vol, min(ceiling, DELTA_MAX_FRAC))

4. GARCH scale:
   target = base × garch.vol_scale
   vol_scale = min(2.0, max(0.3, GARCH_TARGET_VOL / garch.vol))

5. Mayer Multiple dampener:
   if BTC/EMA200 > 2.4: damp = max(0.5, 1 - (mayer-2.4)/2.2)
   if BTC/EMA200 < 1.0: damp = min(1.2, 1 + (1-mayer)×0.3)

6. BTC cross-asset lead:
   if btc_4h.active AND btc_1h.active: raw[altcoin] *= 1.4

7. Hour boost (BOOST_HOURS = {3, 9, 16, 19}):
   new entries only: × 1.25

8. Blocked hours ({1, 13, 14, 15, 17, 18}):
   new entries → set to 0.0

9. Portfolio normalisation:
   if Σ|frac| > 1.0: scale all by 1/Σ|frac|

10. Signal overrides (hot-reload every 5min):
    target[sym] *= global_mult × per_sym_mult[sym]
```

`TF_CAP` maps the 3-bit timeframe score to a maximum allocation fraction:

| Score | Active TFs | Cap |
|---|---|---|
| 7 | 4h + 1h + 15m | 100% |
| 6 | 4h + 1h | 100% |
| 4 | 4h only | 60% |
| 3 | 1h + 15m | 50% |
| 2 | 1h only | 40% |
| 1 | 15m only | 20% |
| 0 | none | 0% |

### 5.3 _apply_signal_overrides()

After the core sizing logic, `_apply_signal_overrides(targets)` multiplies
each symbol's target fraction by the hot-loaded override multipliers. See
[Section 6](#6-signal-overrides) for the full override spec.

### 5.4 _place_order() -- Alpaca Submission

```python
def _place_order(self, sym, side, qty, price, new_frac):
    # Split large orders (> $195,000 notional) into slices
    while remaining > 1e-8:
        slice_qty = min(remaining, MAX_ORDER_NOTIONAL / price)
        req = MarketOrderRequest(
            symbol        = ticker,       # "BTC/USD"
            qty           = round(slice_qty, 8),
            side          = OrderSide.BUY / SELL,
            time_in_force = TimeInForce.GTC,
        )
        resp = self._trading_client.submit_order(req)
        remaining -= slice_qty

    # Optimistically update state (fills confirm via trade_updates stream)
    st.last_frac = new_frac
    st.bars_held = 0
```

The optimistic state update means the trader does not wait for fill
confirmation before updating its position model. True fill confirmation arrives
asynchronously via the `TradingStream` fill events.

### 5.5 _on_fill() -- Trade Logging

Fill events arrive from `TradingStream.subscribe_trade_updates()`. The handler
filters for `event_type == "fill"` and:

1. Writes a row to `live_trades` SQLite table
2. If the fill is a **sell**: performs FIFO P&L matching against the in-memory
   `InstrumentState._fifo` queue and writes matched trades to `trade_pnl`

```python
if side == "buy":
    st._fifo.append((qty, price, fill_time))

elif side == "sell":
    while remaining > 0 and st._fifo:
        entry_qty, entry_price, entry_time = st._fifo[0]
        matched = min(remaining, entry_qty)
        pnl = matched * (price - entry_price)
        # INSERT into trade_pnl
        remaining -= matched
```

---

## 6. Signal Overrides

**File:** `config/signal_overrides.json`  
**Hot-reload TTL:** 300 seconds (5 minutes)

The live trader supports two JSON formats for overrides:

**Format A (signal_injector.py output):**

```json
{
  "multipliers":   {"BTC": 0.5, "ETH": 1.2},
  "sizing_override": 0.8,
  "blocked_hours": [2, 3],
  "expires_at": "2026-04-06T00:00:00"
}
```

**Format B (direct per-symbol spec):**

```json
{
  "_global": {
    "size_multiplier": 0.75,
    "blocked_hours": [1, 2],
    "expiry": "2026-04-06T12:00:00"
  },
  "BTC": {
    "size_multiplier": 1.5,
    "expiry": "2026-04-05T20:00:00"
  }
}
```

**Override application order:**

```
final_target[sym] = raw_target[sym]
                    × global_multiplier
                    × per_sym_multiplier[sym]   (default 1.0 if absent)
```

Expired overrides (past `expires_at` / `expiry` timestamp) are silently
ignored. The cache is re-read every 5 minutes from disk, so operators can
drop a new `signal_overrides.json` to adjust sizing without restarting the
trader process.

Use cases:
- **Risk reduction before events**: set `_global.size_multiplier=0.3` ahead of
  a FOMC announcement
- **Symbol-level kill switch**: set `BTC.size_multiplier=0.0`
- **Extra blocked hours**: add hours not covered by the static
  `BLOCKED_ENTRY_HOURS_UTC` set

---

## 7. SQLite Trade Logging

**File:** `execution/live_trades.db`  
**WAL mode:** enabled (`PRAGMA journal_mode=WAL`)

WAL (Write-Ahead Logging) mode allows concurrent reads from other tools
(dashboards, analysis scripts) without blocking the writer. In practice the
fill handler and equity-refresh task are the only writers; multiple research
notebooks can read simultaneously.

**Schema:**

```sql
CREATE TABLE live_trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol           TEXT    NOT NULL,   -- e.g. "BTC"
    side             TEXT    NOT NULL,   -- "buy" | "sell"
    qty              REAL    NOT NULL,
    price            REAL    NOT NULL,
    notional         REAL    NOT NULL,   -- qty × price
    fill_time        TEXT    NOT NULL,   -- ISO-8601 UTC
    order_id         TEXT,              -- Alpaca order UUID
    strategy_version TEXT    DEFAULT 'larsa_v16'
);

CREATE TABLE trade_pnl (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT,
    entry_time   TEXT,
    exit_time    TEXT,
    entry_price  REAL,
    exit_price   REAL,
    qty          REAL,
    pnl          REAL,        -- (exit_price - entry_price) × qty
    hold_bars    INTEGER      -- number of 15m bars held
);
```

**FIFO P&L matching:**

FIFO (first-in, first-out) matches each sell fill against the oldest open
buy lots. A partial match leaves the remainder of the first FIFO entry in
place:

```
FIFO queue before sell:  [(1.0 BTC @ 60000), (0.5 BTC @ 61000)]
Sell 0.8 BTC @ 65000:
  match 0.8 of lot1: pnl = 0.8 × (65000 - 60000) = $4000
  FIFO queue after:  [(0.2 BTC @ 60000), (0.5 BTC @ 61000)]
```

---

## 8. Process Supervision

**File:** `scripts/supervisor.py`  
**Port:** 8790

The supervisor maintains one `ServiceState` per service and runs a monitor
thread that polls each process every 5 seconds.

**Service definitions:**

| Name | Command | Health URL |
|---|---|---|
| `market-data` | `./market-data.exe` | `http://localhost:8780/health` |
| `coordination` | `mix run --no-halt` | `http://localhost:8781/health` |
| `bridge` | `python bridge/heartbeat.py` | `http://localhost:8783/health` |
| `autonomous-loop` | `python -m idea_engine.autonomous_loop.orchestrator` | none |
| `live-trader` | `python tools/live_trader_alpaca.py` | none |

**Restart backoff sequence:**

```
crash 1 → wait 5s
crash 2 → wait 10s
crash 3 → wait 20s
crash 4 → wait 40s
crash 5+ → wait 60s (stays at max)
```

```python
BACKOFF_SEQUENCE = [5, 10, 20, 40, 60]
```

**HTTP API:**

```bash
# Check all service states
GET http://localhost:8790/status
→ {"services": {"live-trader": {"status": "running", "restarts": 0, ...}, ...}}

# Force restart a specific service
POST http://localhost:8790/restart/live-trader
→ {"ok": true}

# Stop a service (no auto-restart)
POST http://localhost:8790/stop/live-trader
→ {"ok": true}
```

State is also persisted to `logs/supervisor.json` on every update, so
dashboards can read the last-known state even if the supervisor itself
restarts.

---

## 9. Docker Compose

**File:** `docker-compose.yml`

Six services defined; `backtest`, `worker`, and `research` are in separate
Compose profiles and only start when explicitly requested.

```
                        srfm-net (bridge network)
 ┌──────────────┬────────────┬────────────┬────────────────┬──────────────┐
 │ market-data  │coordination│   bridge   │ autonomous-loop│ live-trader  │
 │   :8780      │   :8781    │   :8783    │                │              │
 │ Go binary    │ Elixir OTP │ Python     │ Python         │ Python       │
 │ healthcheck  │ healthcheck│ healthcheck│ pgrep check    │ pgrep check  │
 └──────────────┴────────────┴────────────┴────────────────┴──────────────┘
                                                    ┌──────────┐
                                                    │supervisor│
                                                    │   :8790  │
                                                    └──────────┘
```

**Shared volumes:**

All Python services mount:
- `./logs:/app/logs` -- shared log directory (FeedMonitor writes here)
- `./execution:/app/execution` -- shared SQLite DB path
- `./bridge:/app/bridge` -- signal_overrides.json pickup location

**Dependency ordering:**

```
market-data healthy → bridge healthy → live-trader starts
market-data healthy → bridge healthy → autonomous-loop starts
```

`live-trader` and `autonomous-loop` both declare `depends_on: bridge: condition: service_healthy`, ensuring the heartbeat service (which also acts as a
config proxy) is ready before trading begins.

**Healthchecks:**

Services with HTTP servers use `curl -f http://localhost:PORT/health`. Services
without HTTP servers (`autonomous-loop`, `live-trader`) use `pgrep -f <script_name>`.

---

## 10. IAE Improvements Applied Live

The Incremental Algorithm Evaluation (IAE) process produced the following
parameters, all reflected in `live_trader_alpaca.py`:

| Parameter | Value | Rationale |
|---|---|---|
| `MIN_HOLD` | 8 bars (= 2h) | Prevents high-frequency reversals; locks in trend signal |
| `BLOCKED_ENTRY_HOURS_UTC` | `{1, 13, 14, 15, 17, 18}` | UTC 1 = thin Asia session; 13–18 = US open volatility |
| `BOOST_ENTRY_HOURS_UTC` | `{3, 9, 16, 19}` | Pre-Asia, pre-Europe, post-US open, post-US close: strong trend initiation windows |
| `HOUR_BOOST_MULTIPLIER` | 1.25 | New entries in boost hours get 25% larger initial size |
| `CORR_NORMAL` | 0.25 | Conservative pairwise assumption during calm markets |
| `CORR_STRESS` | 0.60 | Elevated during periods of cross-asset correlation spike |
| `CORR_STRESS_THRESHOLD` | 0.60 | Average pairwise correlation at which regime switches |

**Dynamic CORR calculation:**

```python
mat  = np.array([daily_returns[sym] for sym in INSTRUMENTS
                 if len(daily_returns[sym]) >= 30])
corr = np.corrcoef(mat)
n    = corr.shape[0]
avg  = (np.sum(corr) - n) / (n * (n - 1))  # average off-diagonal

dynamic_corr = CORR_STRESS if avg > 0.60 else CORR_NORMAL
```

When the average pairwise 30-day correlation among the 17-instrument universe
exceeds 0.60 (characteristic of risk-off events), the correlation factor in
the Kelly-style sizing denominator increases from 0.25 to 0.60. This
mechanically reduces `per_inst_risk` and therefore all position sizes, without
requiring any manual intervention.

---

## 11. Worked Example -- End-to-End Order Trace

**Scenario:** BTC/USD 15m bar closes at 14:15 UTC on a Tuesday. The BH model
fires a buy signal.

```
Step 1 -- Bar arrival
  CryptoDataStream delivers bar: symbol="BTC/USD", close=67500, ts=14:15 UTC
  bar_handler() calls trader.on_bar("BTC/USD", bar)

Step 2 -- Indicator updates
  st.bh_15m.update(67500)           → bh_15m.active = True, bh_dir = +1
  st.garch.update(log(67500/67300)) → vol_scale ≈ 0.94
  st.ou.update(67500)               → zscore = -0.2 (no OU signal)
  15m buffer appended; hour boundary not reached yet

Step 3 -- _act_on_targets(14:15 UTC)
  compute_targets() called:
    bar_hour = 14 → in BLOCKED_ENTRY_HOURS? YES (14 ∈ {1,13,14,15,17,18})
    → new entries suppressed (last_frac["BTC"] == 0.0)
    → raw["BTC"] = 0.0
  No order placed.

  [Note: blocked hour prevents entry even though signal fired]

Step 4 -- Next bar at 15:15 UTC
  bar_hour = 15 → still blocked.

Step 5 -- Bar at 16:15 UTC
  bar_hour = 16 → in BOOST_ENTRY_HOURS ({3,9,16,19})
  blocked = False

  compute_targets():
    tf = 4·(bh_4h.active=T) + 2·(bh_1h.active=T) + 1·(bh_15m.active=T) = 7
    ceiling = min(TF_CAP[7]=1.0, CRYPTO_CAP=0.40) = 0.40
    direction = +1 (from bh_1h.bh_dir)
    vol = ATR_1h(≈400) / 67500 × √6.5 ≈ 0.0151
    per_inst_risk = 0.05 / corr_factor ≈ 0.0083  (CORR_NORMAL=0.25, N=17)
    base = min(0.0083 / 0.0151, 0.40) = min(0.55, 0.40) = 0.40  [capped]
    raw["BTC"] = 0.40 × vol_scale(0.94) = 0.376
    boost (16 in BOOST_HOURS, new entry): raw["BTC"] *= 1.25 = 0.470
    normalise (only BTC active): scale = 1.0 (sum = 0.470 < 1.0)
    signal_overrides: no override file → multiplier = 1.0
    target["BTC"] = 0.470

  _act_on_targets():
    delta = 0.470 - 0.0 = 0.470 > MIN_TRADE_FRAC(0.003) → proceed
    equity = $100,000
    tgt_dollar = 0.470 × 100,000 = $47,000
    qty = 47,000 / 67,500 = 0.6963 BTC
    side = "buy"
    asyncio.create_task(_place_order_async("BTC", "buy", 0.6963, 67500, 0.470))

Step 6 -- BookManager spread check (SmartRouter, if wired in)
  book_manager.get_spread_bps("BTC/USD") → 5.2 bps
  5.2 ≤ 50 bps → MARKET ORDER path

Step 7 -- Alpaca submission
  MarketOrderRequest(symbol="BTC/USD", qty=0.6963, side=BUY, tif=GTC)
  resp.id = "a1b2c3d4-..."
  log: "BTC order submitted: id=a1b2c3d4 side=buy qty=0.696300 notional=$47007"
  st.last_frac = 0.470; st.bars_held = 0; st.entry_px = 67500

Step 8 -- Fill arrives via TradingStream
  event.event = "fill"
  order.filled_qty = 0.6963, order.filled_avg_price = 67512.40
  _on_fill():
    INSERT INTO live_trades VALUES
      ("BTC", "buy", 0.6963, 67512.40, 47002.37, "2026-04-05T16:15:43Z",
       "a1b2c3d4-...", "larsa_v16")
    st._fifo.append((0.6963, 67512.40, "2026-04-05T16:15:43Z"))
    log: "Fill logged: BTC BUY 0.696300 @ $67512.40 notional=$47007"

Step 9 -- Position is live
  st.last_frac = 0.470
  st.entry_px  = 67500 (optimistic) / 67512.40 (actual from fill)
  st._fifo     = [(0.6963, 67512.40, "2026-04-05T16:15:43Z")]
  MIN_HOLD = 8 bars → no reversal before 18:15 UTC
```

---

*See also: `docs/wave4_backtest.md` for the research methodology underlying
the IAE parameter choices. The live constants in Section 10 directly reflect
Wave 4 findings.*
