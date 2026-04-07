# Order Management -- Algorithmic Execution Layer

## Overview

The `execution/order_management/` package implements the algorithmic order execution layer
for the SRFM trading lab. It sits between the strategy signal layer and the exchange-facing
SmartRouter, translating high-level trade intentions into a stream of carefully paced child
orders. The layer is designed around three goals: minimizing market impact, maintaining
auditability, and surviving process restarts without losing order state.

All algo types share a common order state machine, a unified scheduler, and a persistent
SQLite-backed tracker. Transaction cost analysis (TCA) is computed inline so every completed
algo immediately reports its execution quality against standard benchmarks.

---

## Order Types -- `order_types.py`

Four concrete order classes all inherit from `BaseOrder`.

### BaseOrder

`BaseOrder` is the abstract root of the hierarchy. It carries the fields common to every
algo:

| Field | Type | Description |
|---|---|---|
| `order_id` | `str` | UUID4 assigned at creation |
| `symbol` | `str` | Exchange-normalized ticker |
| `side` | `Side` | BUY or SELL enum |
| `total_qty` | `Decimal` | Shares/contracts to execute |
| `filled_qty` | `Decimal` | Cumulative fills received so far |
| `state` | `OrderState` | Current state machine position |
| `created_at` | `datetime` | UTC timestamp of creation |
| `parent_algo` | `str \| None` | Link to containing algo if child |

`BaseOrder` defines `remaining_qty`, `fill_pct`, and `is_terminal` as computed properties.
Subclasses must implement `next_slice_qty() -> Decimal` and `on_fill(fill: Fill) -> None`.

### TWAPOrder

`TWAPOrder` (time-weighted average price) divides `total_qty` into `n_slices` equal pieces
and submits one piece every `interval_seconds`. Fields added on top of `BaseOrder`:

- `n_slices` -- number of equal time intervals (default 12)
- `interval_seconds` -- target seconds between slices (computed from start/end window)
- `slice_qty` -- base quantity per slice (`total_qty / n_slices`)
- `slices_sent` -- counter incremented on each dispatch
- `start_time` / `end_time` -- execution window boundaries

`next_slice_qty()` returns `slice_qty` for all slices except the last, which returns
`remaining_qty` to avoid rounding residual.

### VWAPOrder

`VWAPOrder` (volume-weighted average price) targets execution proportional to the intraday
volume curve rather than uniform time slices. Extra fields:

- `volume_profile` -- dict mapping time bucket to fraction of daily volume
- `target_vwap` -- reference price at order creation for TCA
- `bucket_qty` -- precomputed qty for each time bucket

`next_slice_qty()` reads the current time bucket and returns the precomputed proportion.
If the actual market volume in a bucket is lower than expected, the engine carries forward
a deficit and adds it to the next bucket.

### IcebergOrder

`IcebergOrder` shows only a small visible slice to the order book while holding the
remaining quantity hidden. Extra fields:

- `visible_qty` -- quantity shown in the order book at any time (e.g. 100 shares)
- `min_replenish` / `max_replenish` -- bounds for randomized refill size
- `replenish_jitter` -- fraction applied to randomize visible qty on each refill

When a fill arrives and the visible child order is fully consumed, the engine replenishes
with a random quantity drawn uniformly from `[min_replenish, max_replenish]`. This
randomization prevents pattern detection by opposing HFT systems.

---

## TWAP Engine -- `twap_engine.py`

`TWAPEngine` runs a background thread that manages all live `TWAPOrder` instances.

### Slice Dispatch

On startup the engine computes an initial schedule: for each order, `n_slices` dispatch
times are spread evenly across the execution window. Each scheduled time is then jittered
by a random offset drawn from `Uniform(-0.20 * interval, +0.20 * interval)`. The 20%
jitter breaks the regular cadence that would otherwise be exploitable by front-running
algorithms.

The dispatch loop sleeps in short increments (100 ms default) and wakes to check whether
any order has a slice due. When a slice fires, `TWAPEngine` calls
`SmartRouter.submit_child(order, qty)` and increments `slices_sent`.

### Completion Tracking

After each fill callback the engine calls `order.on_fill(fill)` which updates
`filled_qty`. When `fill_pct >= 1.0` the engine transitions the order to `FILLED` and
emits a `TWAPCompletion` event carrying:

- Total elapsed time
- Average fill price vs. arrival price (IS)
- Average fill price vs. TWAP benchmark (TWAP deviation)

If the execution window expires with unfilled quantity, the engine transitions to
`CANCELLED` and logs the shortfall.

---

## VWAP Engine -- `twap_engine.py`

`VWAPEngine` shares the same file as `TWAPEngine` and uses the same background thread
infrastructure, but replaces the uniform slice schedule with an intraday volume profile.

### U-Shaped Volume Profile

The default profile allocates:

- **30%** of daily volume to the morning bucket (first 90 minutes)
- **40%** to the midday bucket (middle 3.5 hours)
- **30%** to the closing bucket (final 90 minutes)

Each broad bucket is further subdivided into 5-minute sub-buckets using an empirical
curve fitted to historical market data. The result is a vector of roughly 78 sub-bucket
weights summing to 1.0.

### Dynamic Adjustment

At the end of each sub-bucket, `VWAPEngine` compares the actual market volume observed
(reported by the market data feed) against the expected volume for that bucket. If actual
volume was lower, the remaining unexecuted quantity is redistributed forward
proportionally. If actual volume was higher and the engine executed too little, it logs a
VWAP deviation warning and increases the rate in the next sub-bucket.

---

## Iceberg Engine -- `algo_scheduler.py`

Iceberg replenishment logic lives inside `AlgoScheduler` rather than a separate engine
because replenishment is event-driven (triggered by fills) rather than time-driven.

### Replenishment Flow

1. `SmartRouter` reports a fill on a child order back to `AlgoScheduler`.
2. If the child order is the visible leg of an `IcebergOrder` and is now fully filled,
   `AlgoScheduler` calls `IcebergOrder.compute_next_visible()`.
3. `compute_next_visible()` draws a random qty from
   `[min_replenish, max_replenish]` using a seeded `random.uniform` call, capped at
   `remaining_qty` to avoid over-submitting.
4. `AlgoScheduler` submits a new child order for the computed visible qty.
5. Steps 1-4 repeat until `remaining_qty == 0`, at which point the iceberg transitions
   to `FILLED`.

The randomized replenish size means the order book observer cannot infer total hidden
quantity from the replenishment pattern.

---

## AlgoScheduler -- `algo_scheduler.py`

`AlgoScheduler` is the unified entry point for all algorithmic order types. Strategies
never talk to `TWAPEngine` or `VWAPEngine` directly; they go through `AlgoScheduler`.

### Priority Queue

Internally `AlgoScheduler` maintains a min-heap keyed on `next_dispatch_time`. The
background thread pops the earliest entry, dispatches the slice, then re-inserts the
order with its next scheduled time. Emergency cancellations bypass the heap and are
processed immediately on the next loop iteration.

### Public API

```python
scheduler.submit(order: BaseOrder) -> str          # returns order_id
scheduler.cancel(order_id: str) -> CancelResult
scheduler.status(order_id: str) -> OrderStatus
scheduler.list_active() -> list[OrderStatus]
scheduler.tca_report(order_id: str) -> TCAReport   # only for terminal orders
```

### Cancellation

`cancel()` transitions the order to `CANCELLING`, which prevents further slice dispatch.
When all in-flight child orders have been acknowledged as cancelled by SmartRouter, the
order transitions to `CANCELLED` and a `CancellationEvent` is emitted. Partial fills
received before cancellation are retained in `filled_qty`.

---

## Order State Machine

Every `BaseOrder` passes through the following states:

```
NEW
 |
 v
PENDING          -- submitted to AlgoScheduler, waiting for first dispatch
 |
 v
ACTIVE           -- at least one child order is live in SmartRouter
 |
 +-- partial fill received
 v
PARTIALLY_FILLED -- some qty done, algo still running
 |
 +-- all qty done --------> FILLED
 |
 +-- cancel requested ----> CANCELLING --> CANCELLED
 |
 +-- risk breach ----------> REJECTED
```

State transitions are enforced by a whitelist matrix in `BaseOrder._transition(new_state)`.
Illegal transitions raise `InvalidStateTransitionError` and are logged at ERROR level.

---

## OrderBookTracker -- `order_book_tracker.py`

`OrderBookTracker` maintains a persistent record of every order and state transition,
backed by an SQLite database in WAL (write-ahead log) mode.

### Schema

```sql
CREATE TABLE orders (
    order_id     TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    total_qty    TEXT NOT NULL,   -- stored as string to preserve Decimal precision
    filled_qty   TEXT NOT NULL,
    state        TEXT NOT NULL,
    algo_type    TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    params_json  TEXT            -- algo-specific params as JSON blob
);

CREATE TABLE order_events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id     TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    payload_json TEXT,
    ts           TEXT NOT NULL
);
```

### Thread Safety

All writes go through a single writer thread; reads use separate read connections with
`PRAGMA journal_mode=WAL` so readers never block the writer. `OrderBookTracker` exposes
`get_order()`, `get_events()`, and `list_by_state()` as the primary read API.

### Crash Recovery

On startup `AlgoScheduler` calls `OrderBookTracker.recover()`, which loads all orders
in non-terminal states (`PENDING`, `ACTIVE`, `PARTIALLY_FILLED`, `CANCELLING`). For
each recovered order it reconstructs the in-memory object and re-registers it with the
appropriate engine. Orders that were `ACTIVE` at crash time are placed back in
`PARTIALLY_FILLED` state with their last known `filled_qty` and resubmit their remaining
quantity.

---

## Integration with SmartRouter

Algo engines never submit orders directly to the exchange. Instead they call
`SmartRouter.submit_child(parent_order_id, qty, urgency)`. SmartRouter applies its
spread-tier routing logic:

- Tight spread (< 1 bps) -- route to primary lit venue
- Medium spread (1-5 bps) -- split between lit and dark pool
- Wide spread (> 5 bps) -- dark pool preferred, with a lit fallback after a timeout

Fill callbacks flow back as `SmartRouter -> AlgoScheduler -> engine -> order.on_fill()`.
Child order IDs are tracked by `OrderBookTracker` under the parent `order_id` in the
`order_events` table.

---

## Transaction Cost Analysis (TCA)

Every completed algo computes TCA metrics and emits a `TCAReport` accessible via
`AlgoScheduler.tca_report(order_id)`.

### Metrics

| Metric | Definition |
|---|---|
| **Implementation Shortfall (IS)** | `(avg_fill_price - arrival_price) * side_sign * total_qty` |
| **Arrival Price Slippage** | `(avg_fill_price - arrival_price) / arrival_price` in bps |
| **VWAP Benchmark** | For TWAP/Iceberg: difference vs. market VWAP over the execution window |
| **VWAP Deviation** | For VWAPOrder: `avg_fill_price - market_vwap` in bps |
| **Fill Rate** | `filled_qty / total_qty` at completion |
| **Duration** | Wall-clock seconds from first dispatch to last fill |

`arrival_price` is the mid-price at the moment `scheduler.submit()` was called, snapshot
from the market data feed and stored in the `params_json` blob at creation time.

TCA results are written to the `order_events` table as a `TCA_COMPLETE` event so they
survive process restarts and are queryable for post-trade analytics.
