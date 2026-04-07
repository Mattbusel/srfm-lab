# Native Layer (Zig and C)

Ultra-low-latency primitives written in Zig and C for sub-microsecond operations.
Used when Python, Rust, or C++ overhead is still too high: NASDAQ ITCH protocol
decoding, lock-free L2 order book maintenance, SIMD matrix multiply, and the lock-free
ring buffer used between the market-data service and the signal engine.

---

## Why Zig

Zig provides C-level performance with memory safety checks in debug mode, explicit
allocator control, and comptime generics. It compiles to shared libraries callable
from Python via `ctypes` or CFFI. The ITCH 5.0 decoder and the L2 order book are
the primary users.

---

## Components

### ITCH 5.0 Decoder (`native/zig/itch/`)

Decodes NASDAQ TotalView-ITCH 5.0 binary protocol messages. Used for direct NASDAQ
feed connections (not active in the current Alpaca paper trading setup, but available
for live market connectivity).

```
native/zig/itch/
  decoder.zig       -- top-level message dispatcher
  messages.zig      -- all ITCH message types (Add Order, Execute Order, etc.)
  types.zig         -- ITCH field types: MPID, OrderRef, Price4, etc.
  parser.zig        -- binary field extraction with endian handling
```

Key message types decoded:
- `S` (System Event) -- market open/close
- `A` (Add Order) -- new limit order, no MPID
- `F` (Add Order with MPID) -- new limit order with market participant ID
- `E` (Order Executed) -- partial or full fill
- `D` (Order Deleted) -- cancel
- `U` (Order Replaced) -- modify in place
- `P` (Trade) -- non-displayable trade
- `Q` (Cross Trade) -- opening/closing cross

Throughput: decodes full 4 GB/s NASDAQ ITCH binary stream on a single core. The
decoded order flow feeds the `order-flow-engine` and `microstructure-engine` Rust
crates.

Building the shared library:
```bash
cd native/zig/itch
zig build -Doptimize=ReleaseFast
# outputs libitch_decoder.so (Linux) or itch_decoder.dll (Windows)
```

### Lock-Free L2 Order Book (`native/zig/orderbook/`)

A lock-free L2 order book supporting up to 64 price levels per side. Uses atomic
compare-and-swap operations on a fixed-size price ladder to achieve sub-microsecond
update latency.

```
native/zig/orderbook/
  book.zig          -- L2Book struct: price ladder, bid/ask arrays
  level.zig         -- PriceLevel: price, size, order count
  atomics.zig       -- CAS wrappers, memory ordering utilities
```

The book stores levels as `[64]PriceLevel` arrays (bids descending, asks ascending).
Updates are single atomic writes to the relevant level. Best bid/ask are tracked
separately in atomic cells for O(1) access.

Latency (AMD Ryzen 9, DDR5):
- Add order: ~180ns
- Cancel order: ~190ns
- Best bid/ask read: ~15ns (from L1 cache after first access)

The Python execution layer calls this book via ctypes for the pre-trade spread check:

```python
# execution/orderbook/orderbook.py
_lib = ctypes.CDLL("native/zig/orderbook/liborderbook.so")
_lib.book_get_spread.restype = ctypes.c_double

def get_spread_bps(symbol: str) -> float:
    return _lib.book_get_spread(symbol.encode())
```

### SIMD Matrix Multiply (`native/matrix/`)

AVX2-vectorized dense matrix multiply for portfolio math. Used by the Python portfolio
optimizer when computing covariance matrices for large instrument sets.

```
native/matrix/
  matmul.c          -- 8x8 AVX2 blocked matmul kernel
  interface.c       -- Python ctypes-compatible C interface
  Makefile
```

For a 21x21 covariance matrix (21 instruments), NumPy is fast enough. For 200+
instruments (full universe backtest), this SIMD kernel is ~30x faster than NumPy's
BLAS call on this hardware due to the tight loop structure and zero allocation.

```c
// interface.c -- called from Python
void matmul_avx2(const double* A, const double* B, double* C, int n) {
    // 8-wide AVX2 FMA loop, n must be multiple of 8
}
```

### AVX2 L3 Order Book (`native/orderbook/`)

A C implementation of a full L3 order book (individual order tracking) using AVX2
vectorized price comparisons. Tracks every visible order by order ID, maintains sorted
bid/ask queues, and computes VWAP fills with market impact.

This is distinct from the Zig L2 book: the L3 book knows individual order IDs and
supports walking the book for large order VWAP estimation. The L2 book knows only
aggregate sizes at each level.

```
native/orderbook/
  l3_book.c         -- order ID hashmap, sorted price levels, VWAP walk
  avx_sort.c        -- AVX2 network sort for maintaining price level order
  book_interface.c  -- ctypes interface for Python
```

Used by `execution/routing/smart_router.py` for large equity orders where the full
depth walk is needed to estimate execution cost before committing.

### Lock-Free Ring Buffer (`native/ringbuffer/`)

A power-of-2 SPSC ring buffer in C with cache-line padding to eliminate false sharing.
Used between the market-data service's ingest thread and the signal computation thread.

```c
// native/ringbuffer/ring.h
typedef struct {
    _Alignas(64) atomic_size_t head;
    _Alignas(64) atomic_size_t tail;
    size_t mask;
    void** slots;
} RingBuffer;
```

Cache-line alignment ensures `head` and `tail` are never on the same cache line,
eliminating the primary performance bottleneck in SPSC ring buffers.

Throughput: 180M operations/second on a single producer/consumer pair (measured via
`bench_ring.c`).

---

## Building All Native Components

```bash
# Zig components
cd native/zig/itch && zig build -Doptimize=ReleaseFast
cd native/zig/orderbook && zig build -Doptimize=ReleaseFast

# C components
cd native/matrix && make
cd native/orderbook && make
cd native/ringbuffer && make
```

Or from the workspace root:
```bash
make native
```

Pre-built Windows DLLs for all components are checked in at `native/bin/windows/`.

---

## Python Integration

All native components are called from Python via ctypes. The pattern is consistent:

```python
# Load shared library once at module import
_lib = ctypes.CDLL("native/zig/orderbook/liborderbook.so")

# Declare argument and return types
_lib.book_add_order.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_int]
_lib.book_add_order.restype  = ctypes.c_int

# Call in hot path
_lib.book_add_order(symbol.encode(), price, size, side)
```

Error handling: all native functions return integer status codes (0 = success,
negative = error). Python wrappers raise `RuntimeError` on non-zero returns.

---

## Latency Summary

| Component | Operation | Latency |
|---|---|---|
| ITCH 5.0 decoder | Full message parse | ~40ns |
| Zig L2 book | Add/cancel | ~180-190ns |
| Zig L2 book | Best bid/ask read | ~15ns |
| C L3 book | VWAP walk (21 levels) | ~850ns |
| SIMD matmul | 21x21 double matrix | ~2.1us |
| Ring buffer | Push + pop round trip | ~5.5ns |

For comparison, a Python function call overhead is ~50-100ns, and a SQLite read from
WAL is ~10-50us. The native layer is relevant only for the sub-microsecond operations
that appear in the critical path.

---

## When to Use Native vs Rust vs C++

| Layer | Use when |
|---|---|
| Zig/C native | Sub-microsecond latency required, called from Python in tight loop |
| C++ signal engine | Sustained throughput for bar processing, SIMD indicator computation |
| Rust crates | Parallel batch computation (backtests, MC simulation, genome evolution) |
| Python | Anything that is not latency-critical, most of the live trader |

The live trader's hot path (15-minute bar processing) is Python. At 15-minute resolution
there is no need for native latency. The native layer matters for the orderbook spread
check (called on every order) and potential future tick-level signal computation.
