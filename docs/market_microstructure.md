# Market Microstructure Analysis

## Overview

The SRFM lab's microstructure stack operates across three implementation layers, each
tuned for a different trade-off between analytical depth, raw performance, and research
flexibility:

| Layer                          | Language    | Primary Role                                   |
|--------------------------------|-------------|------------------------------------------------|
| `crates/microstructure-engine/`| Rust        | Production metrics: VPIN, OFI, regime signals  |
| `native/orderbook/`            | C           | L3 orderbook reconstruction, AVX2 price scans  |
| `research/micro_structure/`    | Python      | Research: latency arb, HFT flow analysis       |

Additionally, `native/zig/src/order_flow.zig` provides a Zig implementation of footprint
bars and cumulative delta for ultra-low-latency pre-trade checks, and
`crates/orderbook-sim/` provides a synthetic orderbook for adversarial stress testing.

---

## Trade Flow Analysis (`trade_flow_analysis.rs`)

### VPIN -- Volume-Synchronized Probability of Informed Trading

VPIN measures the fraction of trades that are order-flow-imbalanced within equal-volume
buckets. Unlike time-synchronized toxicity measures, VPIN is self-clocking and naturally
adjusts to varying trading activity levels.

**Construction:**

1. Divide the total cumulative volume into buckets of size V (the "volume clock")
2. Within each bucket, classify trades as buys (B) or sells (S) using tick rule or
   Lee-Ready algorithm
3. Compute the imbalance for bucket tau:

```
|B_tau - S_tau| / V
```

4. VPIN is the rolling average over the most recent n buckets:

```
VPIN = (1/n) * sum_{tau=t-n+1}^{t} |B_tau - S_tau| / V
```

Typical parameters: V = total daily volume / 50 (50 buckets per day), n = 50 (trailing
window of one day's worth of buckets).

**Interpretation:** VPIN > 0.7 is the threshold for the "trending" regime signal.
Elevated VPIN (> 0.85) historically precedes flash crash events and major directional
moves -- it is used as a position-size dampener in live trading.

### Order Flow Imbalance (OFI)

OFI captures the net pressure on the best bid and ask at each trade event:

```
OFI_t = Delta_bid_size_t * sign(bid_change_t) - Delta_ask_size_t * sign(ask_change_t)
```

More precisely, the OFI for a single event is:

```
e_t = (q^b_t - q^b_{t-1}) * 1{P^b_t >= P^b_{t-1}}
    - (q^a_t - q^a_{t-1}) * 1{P^a_t <= P^a_{t-1}}
```

Where `q^b`, `P^b` are best bid size and price, and `q^a`, `P^a` are best ask size and
price. Cumulative OFI over a window strongly predicts short-term price impact.

### Cumulative Delta

Running sum of signed volume -- buys are positive, sells are negative:

```
CumDelta(t) = sum_{i=0}^{t} v_i * side_i   (side = +1 for buy, -1 for sell)
```

Delta divergence is detected when price makes a new high/low but cumulative delta does
not confirm -- a classical footprint bar reversal signal.

### Delta Divergence Detection

The `trade_flow_analysis.rs` module scans for divergence within rolling windows:

```
divergence = (price_new_high AND cumDelta < cumDelta_prev_high)
          OR (price_new_low  AND cumDelta > cumDelta_prev_low)
```

Detected divergences are emitted as microstructure events consumed by the signal engine.

---

## Liquidity Metrics (`liquidity_metrics.rs`)

### Effective Spread

The effective spread measures the true cost of a round-trip trade relative to the
midpoint at time of execution:

```
EffSpread = 2 * |trade_price - midpoint|
```

Where midpoint = (best_bid + best_ask) / 2. Unlike the quoted spread (ask - bid), the
effective spread accounts for trades that execute inside the spread (due to hidden
liquidity or iceberg orders).

### Realized Spread

Measures how much of the effective spread is captured by the market maker after adverse
price movement:

```
RealizedSpread = 2 * side * (trade_price - midpoint_{t+5min})
```

Where `side = +1` for a buy trade and `-1` for a sell trade. The 5-minute midpoint is
used as the post-trade benchmark. If the realized spread is close to zero or negative,
the market maker is losing to informed flow.

### Price Impact

The permanent price impact of a trade -- the part of the effective spread that is not
recovered after the trade:

```
PriceImpact = EffSpread - RealizedSpread
            = 2 * side * (midpoint_{t+5min} - midpoint_t)
```

Large price impact relative to effective spread indicates high adverse selection -- the
counterparty was likely informed.

### Kyle's Lambda (Price Impact Coefficient)

Kyle's lambda is estimated via OLS regression of midpoint changes on signed order flow:

```
delta_mid_t = lambda * OFI_t + epsilon_t
```

A large lambda means each unit of net order flow moves prices significantly -- the market
has low depth or high information asymmetry. Lambda is updated every 5 minutes using a
rolling 1-hour window of trade-by-trade data.

### Amihud Illiquidity Ratio

Daily measure of price impact per unit of trading volume:

```
ILLIQ = (1/D) * sum_{d=1}^{D} |R_d| / Volume_d
```

Where R_d is the daily return and Volume_d is the daily dollar volume. Higher ILLIQ means
larger price moves per dollar traded -- the asset is less liquid.

### LiquiditySnapshot Struct

All metrics are packaged into a `LiquiditySnapshot` struct emitted every second to
downstream consumers:

```rust
pub struct LiquiditySnapshot {
    pub timestamp: i64,
    pub effective_spread: f64,
    pub realized_spread: f64,
    pub price_impact: f64,
    pub kyle_lambda: f64,
    pub amihud_illiq: f64,
    pub vpin: f64,
    pub ofi_cumulative: f64,
}
```

---

## Market Regime Signals (`market_regime_signals.rs`)

The regime module classifies current market conditions into one of three microstructure
regimes based on the metrics above:

### Trending Regime

Condition: `VPIN > 0.7`

Interpretation: a large fraction of recent volume is directional. The market is being
driven by informed or trend-following flow. Strategy response: widen profit targets,
reduce mean-reversion exposure, increase momentum signal weight.

### Fragmented Regime

Condition: `effective_spread > 3 * rolling_avg_effective_spread`

Interpretation: liquidity has deteriorated -- bid-ask spreads are abnormally wide,
possibly due to market maker withdrawal before a major announcement or low-liquidity
session hours. Strategy response: pause new entries, tighten existing stops, reduce
position sizes.

### Toxic Regime

Condition: `delta_divergence detected within last 3 bars`

Interpretation: price and order flow are decorrelating, suggesting spoofing, layering, or
informed accumulation/distribution. Strategy response: suspend position opens, flag for
human review if condition persists > 30 seconds.

Regimes are not mutually exclusive -- a market can be simultaneously trending and toxic
(e.g., a informed seller driving a sharp decline with diverging cumulative delta).

---

## C L3 Orderbook (`native/orderbook/`)

The C layer maintains a full Level 3 orderbook -- individual order ID tracking across
all price levels for supported venues.

### Individual Order Tracking

Each order is stored in a hash map keyed by exchange order ID. The price-level structure
holds a doubly-linked list of orders at that level for O(1) cancel/modify operations:

```
price_level -> { total_qty, order_count, head_order_ptr, tail_order_ptr }
order_node  -> { order_id, qty, side, timestamp, next_ptr, prev_ptr }
```

### AVX2 Vectorized Price Comparisons

Level scanning for best bid/ask and VWAP computation uses AVX2 SIMD intrinsics to
process 4 double-precision prices per cycle:

```c
__m256d v_prices = _mm256_loadu_pd(&level_prices[i]);
__m256d v_thresh = _mm256_set1_pd(threshold_price);
__m256d mask     = _mm256_cmp_pd(v_prices, v_thresh, _CMP_GT_OQ);
```

This achieves roughly 4x throughput for level scans compared to scalar code.

### VWAP Fill Estimation

For a hypothetical large order of size Q, the orderbook walks the book to estimate
the volume-weighted average fill price:

```
VWAP_fill = sum_{i=0}^{k} price_i * min(qty_i, remaining) / Q
```

Where levels are consumed in price priority order until Q is fully filled or book depth
is exhausted. Used for pre-trade cost estimation by the position sizer.

### Walk-the-Book Simulation

Full simulation of how a market order of size Q would execute level-by-level, returning
the complete fill ladder (price, quantity pairs). Used by the execution engine to decide
between market orders and limit order strategies for larger position changes.

---

## Zig Order Flow (`native/zig/src/order_flow.zig`)

The Zig implementation provides footprint bar analytics at extremely low latency
(target: < 100ns per update) for pre-trade decision support.

### FootprintBar

A `FootprintBar` accumulates trades within a price x time cell structure:

```zig
const FootprintBar = struct {
    price_levels:     [256]PriceLevel,
    point_of_control: f64,
    value_area_high:  f64,
    value_area_low:   f64,
    total_volume:     u64,
    delta:            i64,
};
```

The **Point of Control (POC)** is the price level with the highest traded volume in the
bar. The **Value Area** is the range of price levels accounting for 70% of bar volume,
centered on the POC.

### CumulativeDeltaTracker

Maintains running buy/sell volume counts and emits delta state at each trade:

```
delta = cum_buy_volume - cum_sell_volume
```

Delta reversals (sign changes) are emitted as discrete events for downstream signal
consumption.

### BuyPressureIndex

A normalized measure of buy-side aggression:

```
BPI = cum_buy_volume / (cum_buy_volume + cum_sell_volume)
```

BPI > 0.65 sustained over multiple bars is a bullish microstructure condition. BPI < 0.35
is bearish. The Zig tracker computes BPI on a rolling 20-bar window with O(1) update.

### VPINCalculator

The Zig VPIN implementation is the performance-critical path used during live trading
(the Rust implementation is used for research and backtesting). The Zig version processes
each trade tick in < 15ns by maintaining pre-allocated ring buffers for volume buckets
and using branchless bucket-fill logic.

---

## Python Latency Arbitrage Research (`research/micro_structure/latency_arbitrage.py`)

`latency_arbitrage.py` is a ~1,447 line research module covering cross-venue latency
analysis and HFT flow toxicity modeling.

### Cross-Venue Latency Detection

The module detects when one venue's price updates lead another by measuring the
cross-correlation of mid-price changes with varying lag:

```
rho(lag) = corr(delta_mid_venue_A(t), delta_mid_venue_B(t + lag))
```

The lag that maximizes rho is the estimated latency advantage of the leading venue. In
practice, CME crypto futures consistently lead spot venues by 15--50ms.

### Co-Location Simulation

Simulates the P&L available to a co-located HFT strategy that can:
1. Observe a price update on venue A
2. Immediately send an aggressive order to venue B (before venue B's price updates)
3. Exit within the next 1--5 ticks once venue B catches up

The simulation accounts for realistic one-way latency distributions (log-normal with
location parameter derived from observed cross-correlation lags).

### HFT Flow Toxicity

Classifies incoming flow on a per-trade basis using a logistic regression model trained
on labeled HFT vs. non-HFT flow (labels derived from order-to-trade ratios and order
lifetime distributions):

```
P(HFT | features) = sigmoid(w^T * x)
```

Features include: order lifetime, order-to-trade ratio, cancel rate, order size
relative to venue average, time since last same-side order from same IP subnet.

---

## Orderbook Simulation (`crates/orderbook-sim/`)

A synthetic orderbook generator used for adversarial testing of the microstructure
engine and signal models.

### Synthetic Orderbook Generation

Order sizes follow a **power-law distribution**:

```
P(size = x) ~ x^(-alpha)   for x >= x_min
```

With alpha typically 1.5--2.5, matching empirically observed crypto limit order size
distributions. Order arrival times follow a Poisson process with separate intensity
parameters for each side.

### Adversarial Scenarios

| Scenario             | Description                                                  |
|----------------------|--------------------------------------------------------------|
| Liquidity withdrawal | All orders beyond 2% of mid are cancelled simultaneously     |
| Spoofing             | Large orders placed at 3% off mid, then cancelled < 100ms   |
| Quote stuffing       | 1000x normal order arrival rate for 500ms, then normal       |

These scenarios are run automatically as part of the CI test suite to verify that VPIN,
OFI, and regime signal computations remain numerically stable and emit the correct regime
classification under adversarial conditions.

---

## Integration with Live Trader

### Pre-Trade Spread Check (Zig L2 Book, ~15ns)

Before every new order is submitted, the Zig L2 book is queried to confirm:
1. The current effective spread is within the pre-configured maximum (default: 3x
   rolling average effective spread)
2. There is sufficient depth to fill the intended order size within a 0.05% price impact
   budget

This check completes in approximately 15ns on the hot path, well within the 100ns total
pre-trade budget.

### VPIN as Position-Size Dampener

The live position sizer scales requested position sizes by a VPIN dampening factor:

```
size_dampened = size_requested * max(0, 1 - k * (VPIN - VPIN_threshold))
```

Where `VPIN_threshold = 0.5` and `k = 2.0` by default. At VPIN = 0.7 (trending regime
threshold), positions are reduced to 60% of requested size. At VPIN = 1.0, positions are
blocked entirely. This asymmetric dampening reduces exposure to adverse selection during
high-toxicity periods without eliminating participation entirely.

---

## Key Formulas Summary

| Metric             | Formula                                                            |
|--------------------|--------------------------------------------------------------------|
| VPIN               | `(1/n) * sum |B_tau - S_tau| / V`                                 |
| OFI                | `Delta_bid_qty * 1{bid_up} - Delta_ask_qty * 1{ask_down}`         |
| Effective Spread   | `2 * |trade_price - midpoint|`                                    |
| Realized Spread    | `2 * side * (trade_price - midpoint_{t+5min})`                    |
| Price Impact       | `EffSpread - RealizedSpread`                                      |
| Kyle's Lambda      | `delta_mid = lambda * OFI + epsilon` (OLS estimate of lambda)     |
| Amihud ILLIQ       | `mean(|R_d| / Volume_d)`                                          |
| BPI                | `cum_buy_vol / (cum_buy_vol + cum_sell_vol)`                      |

---

## Key Files

| Path                                              | Purpose                               |
|---------------------------------------------------|---------------------------------------|
| `crates/microstructure-engine/src/trade_flow_analysis.rs` | VPIN, OFI, cumulative delta  |
| `crates/microstructure-engine/src/liquidity_metrics.rs`   | Spread, impact, LiquiditySnapshot|
| `crates/microstructure-engine/src/market_regime_signals.rs` | Regime classification        |
| `native/orderbook/`                               | C L3 orderbook with AVX2 SIMD        |
| `native/zig/src/order_flow.zig`                   | Zig footprint bars, BPI, VPIN        |
| `research/micro_structure/latency_arbitrage.py`   | Cross-venue latency, HFT toxicity    |
| `crates/orderbook-sim/`                           | Synthetic orderbook, adversarial tests|
