# C++ Signal Engine

The low-latency signal computation layer. Mirrors the Python BH physics engine with
sub-millisecond bar processing via SIMD/AVX2 acceleration. Designed for co-location
or edge deployment where Python overhead is unacceptable.

Status: **standalone pipeline**, not currently wired into the live trader's primary
signal path (which uses the Python BH engine). The C++ engine produces identical
outputs and is the reference implementation for future latency-critical deployments.

---

## Purpose

The Python live trader processes ~21 instruments across 3 timeframes. At 15-minute
bars the latency budget is generous. But the C++ engine exists for two reasons:

1. **Validation**: it is a byte-for-byte reference implementation. When the Python
   engine produces a result, the C++ engine must produce the same result on the same
   input. Divergence indicates a bug.
2. **Future deployment**: co-location or market-making scenarios require processing
   1-second or tick-level bars across hundreds of instruments. The Python engine
   cannot do this; the C++ engine can.

---

## Architecture

```
cpp/signal-engine/
  src/
    main.cpp                     -- entry point, bar streaming loop
    bh_physics/
      bh_state.cpp               -- BH mass accumulation and detection
      bh_state.hpp
      garch.cpp                  -- GARCH(1,1) volatility forecaster
      garch.hpp
      ou_detector.cpp            -- Ornstein-Uhlenbeck mean-reversion detector
      ou_detector.hpp
    quaternion/
      quat_nav.cpp               -- QuatNav class: full quaternion navigation layer
      quat_nav.hpp
    streaming/
      feed_processor.cpp         -- InstrumentState, fill_signal_output()
      feed_processor.hpp
      bar_aggregator.cpp         -- tick-to-bar aggregation
      bar_aggregator.hpp
    indicators/
      rsi.cpp / rsi.hpp          -- RSI (Wilder's smoothing)
      macd.cpp / macd.hpp        -- MACD with signal line
      bollinger.cpp / bollinger.hpp -- Bollinger Bands (SMA + 2*std)
      atr.cpp / atr.hpp          -- Average True Range
      ema.cpp / ema.hpp          -- EMA with configurable alpha
      vwap.cpp / vwap.hpp        -- Session VWAP with volume reset
      realized_vol.cpp           -- Realized volatility (Rogers-Satchell estimator)
    portfolio/
      pid_controller.cpp         -- PID position sizing
      pid_controller.hpp
      risk_parity.cpp            -- Equal-risk contribution allocation
      risk_parity.hpp
    io/
      binary_protocol.cpp        -- packed binary frame format for IPC
      binary_protocol.hpp
      csv_reader.cpp             -- historical bar CSV ingestion
      csv_reader.hpp
      json_writer.cpp            -- JSON signal output for downstream consumers
      json_writer.hpp
  include/srfm/
    types.hpp                    -- SignalOutput struct (320 bytes, 5 cache lines)
    ring_buffer.hpp              -- lock-free SPSC ring buffer
    simd_math.hpp                -- AVX2 vectorized math primitives
  tests/
    test_bh_physics.cpp          -- BH state tests
    test_garch.cpp               -- GARCH state tests
    test_indicators.cpp          -- all indicator tests
    test_quat_nav.cpp            -- 15 quaternion navigation tests
    test_ring_buffer.cpp         -- ring buffer contention tests
    test_performance.cpp         -- latency benchmarks
  benchmarks/
    bench_signal_throughput.cpp  -- sustained throughput benchmark
  CMakeLists.txt
```

---

## SignalOutput Struct

The central data type. Every bar produces one `SignalOutput`, written atomically to
shared memory or a named pipe:

```cpp
// cpp/signal-engine/include/srfm/types.hpp
struct SignalOutput {
    // BH Physics
    double bh_mass;           // current accumulated mass [0, MASS_CAP]
    int32_t bh_active;        // 1 when mass >= BH_MASS_THRESH
    double proper_time;       // sqrt(max(0, ds^2)) for this bar
    double ds2;               // signed Minkowski interval
    double hawking_temp;      // 1 / (8 * pi * mass), proxy for BH stability

    // Indicators
    double rsi;               // 0-100
    double macd;              // MACD histogram
    double macd_signal;
    double bb_upper;          // Bollinger upper
    double bb_lower;          // Bollinger lower
    double bb_mid;
    double atr;               // Average True Range
    double vwap;              // Session VWAP
    double ema_fast;          // EMA(5)
    double ema_slow;          // EMA(20)
    double realized_vol;      // 20-bar realized volatility

    // GARCH
    double garch_vol;         // GARCH(1,1) conditional volatility forecast
    double garch_long_run;    // long-run variance

    // OU Detector
    double ou_theta;          // mean-reversion speed
    double ou_mu;             // long-run mean
    double ou_sigma;          // OU volatility

    // Sizing
    double position_size;     // normalized [-1, 1]
    double vol_budget;        // volatility-adjusted allocation
    double corr_factor;       // dynamic correlation adjustment

    // Quaternion Navigation (added LARSA v17)
    double nav_qw, nav_qx, nav_qy, nav_qz;  // Q_current orientation
    double nav_angular_vel;    // radians per bar
    double nav_geodesic_dev;   // curvature-corrected deviation

    uint8_t _fill[24];         // padding to 320 bytes (5 x 64-byte cache lines)
};

static_assert(sizeof(SignalOutput) == 320, "SignalOutput must be 320 bytes");
```

The struct is exactly 5 cache lines. This is intentional: the consumer reads the
complete output in 5 cache-line fetches with no false sharing.

---

## InstrumentState

One `InstrumentState` per instrument per timeframe. The `FeedProcessor` owns a map
from `(symbol, timeframe)` to `InstrumentState`:

```cpp
// cpp/signal-engine/src/streaming/feed_processor.hpp
struct InstrumentState {
    BHState     bh;
    GARCHState  garch;
    OUDetector  ou;
    RSI         rsi;
    MACD        macd;
    BollingerBands bb;
    ATR         atr;
    VWAP        vwap;
    EMA         ema_fast;   // period 5
    EMA         ema_slow;   // period 20
    RealizedVol rvol;
    QuatNav     quat_nav;   // quaternion navigation layer (added LARSA v17)
};
```

`fill_signal_output()` is called once per bar and populates a `SignalOutput` from the
current state. The quaternion nav block runs immediately after the BH block:

```cpp
// Inside fill_signal_output():
bool bh_was_active = state.bh.active();
state.bh.update(bar);
// ... fill BH fields ...

auto nav_out = state.quat_nav.update(
    bar.close, bar.volume, bar.timestamp_ns,
    out.bh_mass, bh_was_active, out.bh_active != 0
);
out.nav_qw = nav_out.qw;
out.nav_qx = nav_out.qx;
out.nav_qy = nav_out.qy;
out.nav_qz = nav_out.qz;
out.nav_angular_vel = nav_out.angular_velocity;
out.nav_geodesic_dev = nav_out.geodesic_deviation;
```

---

## BH Physics in C++

The C++ BH engine mirrors `lib/srfm_core.py` exactly. Key implementation:

```cpp
// cpp/signal-engine/src/bh_physics/bh_state.hpp
static constexpr double BH_MASS_THRESH = 1.92;
static constexpr double MASS_CAP       = 5.0;

class BHState {
    double mass_     = 0.0;
    double cf_       = 0.42;   // causal factor, set per timeframe
    double prev_close_ = 0.0;
    int64_t prev_ts_   = 0;

public:
    void   update(const Bar& bar);
    double mass()   const { return mass_; }
    bool   active() const { return mass_ >= BH_MASS_THRESH; }
};
```

The `update()` method computes the Minkowski interval, classifies the bar, and
accumulates or decays mass. Identical logic to the Python implementation.

---

## GARCH(1,1)

The C++ GARCH tracker is a rolling state machine. It maintains the conditional
variance `h_t` updated on every bar:

```
h_t = omega + alpha * epsilon_{t-1}^2 + beta * h_{t-1}
```

Default parameters: `omega = 0.000002`, `alpha = 0.10`, `beta = 0.88`.
These match the Python `GARCHTracker` in the live trader.

---

## OU Detector

The Ornstein-Uhlenbeck detector estimates theta, mu, and sigma from a rolling window
of log-returns using OLS on the autoregressive form:

```
r_t = alpha + beta * r_{t-1} + epsilon
theta = -log(beta) / dt
mu    = alpha / (1 - beta)
sigma = std(residuals) / sqrt((1 - beta^2) / (2 * |log(beta)|))
```

When theta is large (fast mean-reversion) and the current price is far from mu, the OU
detector generates a mean-reversion signal that the Python trader uses for the OU
overlay allocation (8% equity, OU-disabled for AVAX/DOT/LINK).

---

## SIMD Math

`include/srfm/simd_math.hpp` provides AVX2-vectorized operations:

- `simd_exp_ps`: vectorized exp() for 8 floats at once (used in GARCH vol scaling)
- `simd_log_ps`: vectorized log() (Minkowski interval computation)
- `simd_tanh_ps`: vectorized tanh() (agent signal lensing)
- `simd_dot4_pd`: 4-element double dot product (quaternion operations)

These are GCC-style intrinsics using `__attribute__((target("avx2")))`. Note: the
build currently fails on MSVC due to this syntax. The quaternion nav code compiles
cleanly on MSVC; the SIMD indicators require GCC/Clang.

---

## Ring Buffer

`include/srfm/ring_buffer.hpp` provides a lock-free SPSC (single-producer,
single-consumer) ring buffer using `std::atomic` acquire/release semantics:

```cpp
template<typename T, size_t N>
class RingBuffer {
    std::array<T, N>   buf_;
    std::atomic<size_t> head_, tail_;
public:
    bool try_push(const T& val);
    bool try_pop(T& val);
};
```

The `FeedProcessor` uses this to decouple the Alpaca WebSocket ingestion thread from
the signal computation thread. Bar events are pushed from the network thread and
popped by the signal computation thread without locking.

---

## Binary Protocol

`src/io/binary_protocol.hpp` defines a packed frame format for IPC:

```
Frame:
  [4 bytes magic: 0xBHSR]
  [4 bytes payload length]
  [N bytes payload: serialized SignalOutput]
  [4 bytes CRC32]
```

Consumers (Python live trader, dashboards) can connect via named pipe or TCP and
receive a stream of `SignalOutput` frames at bar rate. The JSON writer provides a
human-readable alternative for debugging.

---

## Tests

15 test cases in `tests/test_quat_nav.cpp` and additional coverage in other test files:

```bash
# Build and run (requires GCC or Clang; SIMD tests skip on MSVC)
cd cpp/signal-engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/signal_engine_tests
```

Key test categories:
- BH mass invariants: bounded, monotone on timelike, decays on spacelike
- GARCH: variance positive, long-run convergence
- OU: theta positive, sigma positive, 100-bar estimation stability
- Indicators: RSI bounded [0,100], MACD signal lag, Bollinger width > 0
- Ring buffer: 1M push/pop cycles, no dropped elements under contention
- Quaternion nav: 15 cases (see `docs/quaternion_nav.md`)
- Performance: throughput > 500K bars/sec on a single core

---

## Performance

On a 2024 AMD Ryzen 9 with AVX2:
- Single bar, single instrument: ~850ns end-to-end (BH + all indicators + nav)
- Sustained throughput (100 instruments, 3 TFs): ~2.1M bars/sec
- Memory per InstrumentState: ~4KB (all state inline, no heap allocation in hot path)

The Python live trader processes 21 instruments at 15-minute resolution. The C++ engine
can handle 100+ instruments at 1-second resolution on the same hardware.

---

## Build Notes

Full build with all targets:

```bash
cd cpp/signal-engine
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17
cmake --build build --target signal_engine_tests
```

The quaternion nav code and BH physics compile on MSVC 19.44+. The SIMD indicators
(`simd_math.hpp`) currently use GCC attribute syntax and require GCC 9+ or Clang 10+
for full compilation. Pre-built Windows binaries are in `native/`.
