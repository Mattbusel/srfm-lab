// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// shm_bus.hpp — Zero-Copy Shared Memory Bus
// =============================================================================
// Architecture: Named channels backed by POSIX shared memory (mmap).
// Each channel is a ring of fixed-size slots (default 64 KB each).
// Writers claim a slot, populate it, then publish a sequence number.
// Readers track their own read cursor; they never block writers.
//
// Memory layout per slot (cache-line aligned):
//   [0..63]   SlotHeader (sequence, flags, tensor_shape, dtype, timestamp_ns)
//   [64..N]   Payload bytes
//
// Multiple named channels are managed by the ShmBus singleton.
// Backpressure: if ring is full, writer either spins (busy wait) or
// returns ShmBus::Error::RING_FULL depending on write policy.
// =============================================================================

#pragma once

#include <atomic>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Platform detection
#if defined(_WIN32)
#  define RTEL_PLATFORM_WINDOWS 1
#else
#  define RTEL_PLATFORM_POSIX   1
#endif

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr std::size_t kCacheLineSize       = 64;
static constexpr std::size_t kDefaultSlotBytes    = 64 * 1024;   // 64 KB
static constexpr std::size_t kDefaultRingCapacity = 1024;        // slots
static constexpr std::size_t kMaxChannelNameLen   = 64;
static constexpr std::size_t kMaxChannels         = 64;
static constexpr std::size_t kMagicHeader         = 0xAE7E4E55'52544C00ULL; // AETERNUS_RTEL

// ---------------------------------------------------------------------------
// Data-type tags (numpy-compatible codes)
// ---------------------------------------------------------------------------
enum class DType : uint8_t {
    FLOAT32 = 0x01,
    FLOAT16 = 0x02,
    FLOAT64 = 0x03,
    INT32   = 0x04,
    INT64   = 0x05,
    UINT8   = 0x06,
    COMPLEX64 = 0x07,
    UNKNOWN = 0xFF,
};

// Return element size in bytes for a given DType
inline std::size_t dtype_size(DType d) noexcept {
    switch (d) {
        case DType::FLOAT32:   return 4;
        case DType::FLOAT16:   return 2;
        case DType::FLOAT64:   return 8;
        case DType::INT32:     return 4;
        case DType::INT64:     return 8;
        case DType::UINT8:     return 1;
        case DType::COMPLEX64: return 8;
        default:               return 0;
    }
}

// Return numpy dtype string for a given DType
inline const char* dtype_numpy_str(DType d) noexcept {
    switch (d) {
        case DType::FLOAT32:   return "<f4";
        case DType::FLOAT16:   return "<f2";
        case DType::FLOAT64:   return "<f8";
        case DType::INT32:     return "<i4";
        case DType::INT64:     return "<i8";
        case DType::UINT8:     return "|u1";
        case DType::COMPLEX64: return "<c8";
        default:               return "unknown";
    }
}

// ---------------------------------------------------------------------------
// TensorDescriptor — describes shape and type of data in a slot
// ---------------------------------------------------------------------------
struct alignas(kCacheLineSize) TensorDescriptor {
    DType    dtype         = DType::UNKNOWN;
    uint8_t  ndim          = 0;
    uint8_t  _pad[6]       = {};
    uint64_t shape[8]      = {};   // up to 8 dimensions
    uint64_t strides[8]    = {};   // byte strides (C-order by default)
    uint64_t num_elements  = 0;
    uint64_t payload_bytes = 0;
    char     name[32]      = {};   // optional tensor name

    // Compute C-order strides given shape and dtype
    void compute_c_strides() noexcept {
        if (ndim == 0) return;
        strides[ndim - 1] = dtype_size(dtype);
        for (int i = static_cast<int>(ndim) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    // Total element count from shape
    uint64_t count_elements() const noexcept {
        if (ndim == 0) return 0;
        uint64_t n = 1;
        for (uint8_t i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }

    // Total payload size in bytes
    uint64_t bytes() const noexcept {
        return count_elements() * dtype_size(dtype);
    }
};
static_assert(sizeof(TensorDescriptor) <= 256, "TensorDescriptor too large");

// ---------------------------------------------------------------------------
// SlotHeader — fixed-size header at the start of every ring slot
// ---------------------------------------------------------------------------
struct alignas(kCacheLineSize) SlotHeader {
    uint64_t         magic        = kMagicHeader;
    std::atomic<uint64_t> sequence{0};  // even=available, odd=being-written
    uint64_t         timestamp_ns = 0;  // wall-clock nanoseconds at publish time
    uint64_t         producer_id  = 0;
    uint32_t         flags        = 0;
    uint32_t         schema_ver   = 1;
    TensorDescriptor tensor;
    uint8_t          _pad[kCacheLineSize - (sizeof(uint64_t) * 4
                           + sizeof(uint32_t) * 2
                           + sizeof(TensorDescriptor)) % kCacheLineSize];

    static constexpr uint32_t FLAG_VALID      = 1u << 0;
    static constexpr uint32_t FLAG_COMPRESSED = 1u << 1;
    static constexpr uint32_t FLAG_LAST       = 1u << 2;   // last in batch
    static constexpr uint32_t FLAG_HEARTBEAT  = 1u << 3;
};

// ---------------------------------------------------------------------------
// ChannelConfig — configuration for a single named channel
// ---------------------------------------------------------------------------
struct ChannelConfig {
    std::string name;
    std::size_t slot_bytes    = kDefaultSlotBytes;
    std::size_t ring_capacity = kDefaultRingCapacity;  // must be power of 2
    bool        create        = true;   // true=create, false=attach existing
    bool        persistent    = false;  // keep shm after last handle closes
    int         numa_node     = -1;     // -1 = OS decides
};

// ---------------------------------------------------------------------------
// ChannelStats — runtime statistics for a channel
// ---------------------------------------------------------------------------
struct ChannelStats {
    uint64_t published_total   = 0;
    uint64_t consumed_total    = 0;
    uint64_t dropped_slots     = 0;  // consumer too slow
    uint64_t backpressure_hits = 0;
    double   utilization_pct   = 0.0;
    uint64_t bytes_written     = 0;
    uint64_t bytes_read        = 0;
    uint64_t last_publish_ns   = 0;
    uint64_t last_consume_ns   = 0;
    uint64_t p50_latency_ns    = 0;
    uint64_t p95_latency_ns    = 0;
    uint64_t p99_latency_ns    = 0;
};

// ---------------------------------------------------------------------------
// WriteHandle — RAII handle for a claimed write slot
// ---------------------------------------------------------------------------
class ShmChannel;

class WriteHandle {
public:
    WriteHandle() = default;
    WriteHandle(const WriteHandle&) = delete;
    WriteHandle& operator=(const WriteHandle&) = delete;
    WriteHandle(WriteHandle&&) noexcept;
    WriteHandle& operator=(WriteHandle&&) noexcept;
    ~WriteHandle();

    bool valid() const noexcept { return slot_ != nullptr; }

    // Pointer to payload area (after SlotHeader)
    void*       data()       noexcept { return payload_; }
    const void* data() const noexcept { return payload_; }
    std::size_t capacity()   const noexcept { return capacity_; }

    // Set tensor descriptor before publishing
    void set_tensor(const TensorDescriptor& td) noexcept;
    void set_flags(uint32_t f) noexcept;
    void set_timestamp(uint64_t ns) noexcept;

    // Commit the write — makes slot visible to readers
    // Returns false if already committed or handle was moved-from
    bool publish() noexcept;

    // Abort without publishing (slot reclaimed)
    void abort() noexcept;

private:
    friend class ShmChannel;
    WriteHandle(ShmChannel* ch, SlotHeader* slot, void* payload,
                std::size_t cap, uint64_t seq);

    ShmChannel* channel_  = nullptr;
    SlotHeader* slot_     = nullptr;
    void*       payload_  = nullptr;
    std::size_t capacity_ = 0;
    uint64_t    seq_      = 0;
    bool        published_= false;
};

// ---------------------------------------------------------------------------
// ReadCursor — per-consumer cursor into a channel
// ---------------------------------------------------------------------------
struct ReadCursor {
    uint64_t next_seq = 1;   // next expected publish sequence
};

// ---------------------------------------------------------------------------
// ShmChannel — a single named shared-memory ring channel
// ---------------------------------------------------------------------------
class ShmChannel {
public:
    enum class Error {
        OK           = 0,
        RING_FULL    = 1,
        NO_DATA      = 2,
        STALE        = 3,
        BAD_MAGIC    = 4,
        SIZE_MISMATCH= 5,
        SHM_FAILED   = 6,
        INVALID_ARG  = 7,
    };

    explicit ShmChannel(const ChannelConfig& cfg);
    ~ShmChannel();

    ShmChannel(const ShmChannel&) = delete;
    ShmChannel& operator=(const ShmChannel&) = delete;

    const std::string& name() const noexcept { return cfg_.name; }
    bool               is_open() const noexcept { return base_ != nullptr; }

    // Writer API ---------------------------------------------------------------
    // Claim a slot for writing. Blocks if RING_FULL and block=true.
    std::pair<WriteHandle, Error> claim(bool block = false) noexcept;

    // Reader API ---------------------------------------------------------------
    // Peek at the next available slot at cursor; returns nullptr if none.
    const SlotHeader* peek(ReadCursor& cur) const noexcept;

    // Advance cursor past current slot.
    void advance(ReadCursor& cur) const noexcept;

    // Convenience: read into caller-supplied buffer (zero-copy path returns
    // pointer into shm if out_buf==nullptr).
    const void* read_data(ReadCursor& cur, TensorDescriptor* td_out = nullptr) noexcept;

    // Stats
    ChannelStats stats() const noexcept;
    void         reset_stats() noexcept;

    // Capacity monitoring
    double ring_utilization() const noexcept;
    bool   is_backpressured() const noexcept;

    // Internal: called by WriteHandle::publish()
    void commit_write(SlotHeader* slot, uint64_t seq) noexcept;
    void abort_write (SlotHeader* slot, uint64_t seq) noexcept;

private:
    void open_or_create();
    void map_shm();
    void unmap_shm();
    SlotHeader* slot_ptr(std::size_t idx) noexcept;
    const SlotHeader* slot_ptr(std::size_t idx) const noexcept;
    uint64_t now_ns() const noexcept;

    ChannelConfig cfg_;

    // Shared memory
    void*       base_       = nullptr;
    std::size_t total_bytes_= 0;
#if defined(RTEL_PLATFORM_POSIX)
    int         shm_fd_     = -1;
#endif

    // Ring control — in shared memory at base_[0]
    struct alignas(kCacheLineSize) RingControl {
        uint64_t magic        = kMagicHeader;
        uint64_t slot_bytes   = 0;
        uint64_t ring_cap     = 0;
        uint64_t schema_ver   = 1;
        std::atomic<uint64_t> write_seq{1};  // next sequence to publish
        std::atomic<uint64_t> min_read_seq{0}; // slowest reader (updated externally)
        char     channel_name[kMaxChannelNameLen] = {};
        uint8_t  _pad[kCacheLineSize - sizeof(uint64_t)*4
                      - sizeof(std::atomic<uint64_t>)*2
                      - kMaxChannelNameLen];
    };

    RingControl* ctrl_        = nullptr;  // pointer into shm
    SlotHeader*  ring_base_   = nullptr;  // first slot in ring

    // Per-process mutable stats (not in shm)
    mutable std::atomic<uint64_t> stat_published_{0};
    mutable std::atomic<uint64_t> stat_consumed_{0};
    mutable std::atomic<uint64_t> stat_dropped_{0};
    mutable std::atomic<uint64_t> stat_backpressure_{0};
    mutable std::atomic<uint64_t> stat_bytes_written_{0};
    mutable std::atomic<uint64_t> stat_bytes_read_{0};
};

// ---------------------------------------------------------------------------
// ShmBus — Singleton bus managing all channels
// ---------------------------------------------------------------------------
class ShmBus {
public:
    using Error = ShmChannel::Error;

    static ShmBus& instance();

    ShmBus(const ShmBus&) = delete;
    ShmBus& operator=(const ShmBus&) = delete;

    // Register/open a channel. Returns false if name already registered.
    bool register_channel(const ChannelConfig& cfg);

    // Get channel by name. Returns nullptr if not found.
    ShmChannel* channel(std::string_view name) noexcept;
    const ShmChannel* channel(std::string_view name) const noexcept;

    // Convenience: claim a write handle on named channel
    std::pair<WriteHandle, Error> claim(std::string_view channel_name,
                                         bool block = false) noexcept;

    // Convenience: read from named channel
    const void* read(std::string_view channel_name,
                     ReadCursor& cur,
                     TensorDescriptor* td_out = nullptr) noexcept;

    // Enumerate channels
    std::vector<std::string> channel_names() const;

    // Aggregate stats
    std::unordered_map<std::string, ChannelStats> all_stats() const;

    // Print stats to stdout (debugging)
    void print_stats() const;

    // Shutdown: close all channels
    void shutdown();

    // Create all standard AETERNUS channels
    void create_aeternus_channels();

private:
    ShmBus() = default;
    ~ShmBus();

    mutable std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    std::unordered_map<std::string, std::unique_ptr<ShmChannel>> channels_;

    void spin_lock() noexcept;
    void spin_unlock() noexcept;
};

// ---------------------------------------------------------------------------
// Standard AETERNUS channel names
// ---------------------------------------------------------------------------
namespace channels {
    inline constexpr const char* LOB_SNAPSHOT    = "aeternus.chronos.lob";
    inline constexpr const char* VOL_SURFACE     = "aeternus.neuro_sde.vol";
    inline constexpr const char* TENSOR_COMP     = "aeternus.tensornet.compressed";
    inline constexpr const char* GRAPH_ADJ       = "aeternus.omni_graph.adj";
    inline constexpr const char* LUMINA_PRED     = "aeternus.lumina.predictions";
    inline constexpr const char* AGENT_ACTIONS   = "aeternus.hyper_agent.actions";
    inline constexpr const char* AGENT_WEIGHTS   = "aeternus.hyper_agent.weights";
    inline constexpr const char* PIPELINE_EVENTS = "aeternus.rtel.pipeline_events";
    inline constexpr const char* HEARTBEAT       = "aeternus.rtel.heartbeat";
} // namespace channels

// ---------------------------------------------------------------------------
// Utility: current wall-clock time in nanoseconds
// ---------------------------------------------------------------------------
uint64_t now_ns() noexcept;

// Utility: RDTSC-based cycle counter
inline uint64_t rdtsc() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return 0;
#endif
}

// Align up to nearest multiple of alignment (must be power of 2)
inline constexpr std::size_t align_up(std::size_t v, std::size_t a) noexcept {
    return (v + a - 1) & ~(a - 1);
}

} // namespace aeternus::rtel
