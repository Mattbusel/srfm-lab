// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// shm_bus.cpp — Zero-Copy Shared Memory Bus Implementation
// =============================================================================

#include "rtel/shm_bus.hpp"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#if defined(RTEL_PLATFORM_POSIX)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  include <time.h>
#elif defined(RTEL_PLATFORM_WINDOWS)
#  define NOMINMAX
#  include <windows.h>
#endif

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// Time utilities
// ---------------------------------------------------------------------------
uint64_t now_ns() noexcept {
#if defined(RTEL_PLATFORM_POSIX)
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL
         + static_cast<uint64_t>(ts.tv_nsec);
#elif defined(RTEL_PLATFORM_WINDOWS)
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    ULARGE_INTEGER li;
    li.LowPart  = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;
    // FILETIME is in 100-nanosecond intervals since 1601-01-01
    // Subtract offset to Unix epoch: 116444736000000000 * 100ns
    constexpr uint64_t EPOCH_DIFF = 116444736000000000ULL;
    return (li.QuadPart - EPOCH_DIFF) * 100ULL;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// WriteHandle
// ---------------------------------------------------------------------------
WriteHandle::WriteHandle(ShmChannel* ch, SlotHeader* slot, void* payload,
                         std::size_t cap, uint64_t seq)
    : channel_(ch), slot_(slot), payload_(payload),
      capacity_(cap), seq_(seq), published_(false) {}

WriteHandle::WriteHandle(WriteHandle&& o) noexcept
    : channel_(o.channel_), slot_(o.slot_), payload_(o.payload_),
      capacity_(o.capacity_), seq_(o.seq_), published_(o.published_) {
    o.channel_   = nullptr;
    o.slot_      = nullptr;
    o.payload_   = nullptr;
    o.published_ = true;  // prevent double-commit in moved-from
}

WriteHandle& WriteHandle::operator=(WriteHandle&& o) noexcept {
    if (this != &o) {
        if (!published_ && slot_ != nullptr) abort();
        channel_   = o.channel_;
        slot_      = o.slot_;
        payload_   = o.payload_;
        capacity_  = o.capacity_;
        seq_       = o.seq_;
        published_ = o.published_;
        o.channel_   = nullptr;
        o.slot_      = nullptr;
        o.published_ = true;
    }
    return *this;
}

WriteHandle::~WriteHandle() {
    if (!published_ && slot_ != nullptr) {
        abort();
    }
}

void WriteHandle::set_tensor(const TensorDescriptor& td) noexcept {
    if (slot_) slot_->tensor = td;
}

void WriteHandle::set_flags(uint32_t f) noexcept {
    if (slot_) slot_->flags = f;
}

void WriteHandle::set_timestamp(uint64_t ns) noexcept {
    if (slot_) slot_->timestamp_ns = ns;
}

bool WriteHandle::publish() noexcept {
    if (published_ || !slot_ || !channel_) return false;
    published_ = true;
    slot_->timestamp_ns = now_ns();
    slot_->flags |= SlotHeader::FLAG_VALID;
    channel_->commit_write(slot_, seq_);
    slot_    = nullptr;
    payload_ = nullptr;
    return true;
}

void WriteHandle::abort() noexcept {
    if (!published_ && slot_ && channel_) {
        published_ = true;
        channel_->abort_write(slot_, seq_);
        slot_    = nullptr;
        payload_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// ShmChannel — helpers
// ---------------------------------------------------------------------------
static bool is_power_of_two(std::size_t n) { return n && !(n & (n - 1)); }

static std::size_t next_power_of_two(std::size_t n) {
    if (n == 0) return 1;
    --n;
    for (std::size_t shift = 1; shift < sizeof(n)*8; shift <<= 1)
        n |= n >> shift;
    return n + 1;
}

// ---------------------------------------------------------------------------
// ShmChannel — constructor / destructor
// ---------------------------------------------------------------------------
ShmChannel::ShmChannel(const ChannelConfig& cfg) : cfg_(cfg) {
    // Enforce power-of-2 ring capacity
    if (!is_power_of_two(cfg_.ring_capacity)) {
        cfg_.ring_capacity = next_power_of_two(cfg_.ring_capacity);
    }
    // Enforce slot_bytes is at least sizeof(SlotHeader) + 1 byte, and aligned
    if (cfg_.slot_bytes < sizeof(SlotHeader) + kCacheLineSize) {
        cfg_.slot_bytes = sizeof(SlotHeader) + kCacheLineSize;
    }
    cfg_.slot_bytes = align_up(cfg_.slot_bytes, kCacheLineSize);

    open_or_create();
}

ShmChannel::~ShmChannel() {
    unmap_shm();
}

// ---------------------------------------------------------------------------
// ShmChannel::open_or_create
// ---------------------------------------------------------------------------
void ShmChannel::open_or_create() {
    // Total bytes = RingControl + ring_capacity * slot_bytes
    const std::size_t ring_ctrl_bytes = align_up(sizeof(RingControl), kCacheLineSize);
    total_bytes_ = ring_ctrl_bytes + cfg_.ring_capacity * cfg_.slot_bytes;

#if defined(RTEL_PLATFORM_POSIX)
    std::string shm_name = "/aeternus_rtel_" + cfg_.name;
    // Replace any '.' with '_' for POSIX shm name compatibility
    for (char& c : shm_name) { if (c == '.') c = '_'; }

    int flags = O_RDWR;
    if (cfg_.create) flags |= O_CREAT;

    shm_fd_ = shm_open(shm_name.c_str(), flags, 0600);
    if (shm_fd_ < 0) {
        // Fall back to file-backed mmap (more portable, e.g. macOS strict limits)
        std::string fpath = "/tmp" + shm_name;
        shm_fd_ = ::open(fpath.c_str(), flags | O_RDWR, 0600);
        if (shm_fd_ < 0) {
            std::perror("ShmChannel: open");
            return;
        }
    }

    if (cfg_.create) {
        if (ftruncate(shm_fd_, static_cast<off_t>(total_bytes_)) != 0) {
            std::perror("ShmChannel: ftruncate");
            ::close(shm_fd_);
            shm_fd_ = -1;
            return;
        }
    }

    void* ptr = mmap(nullptr, total_bytes_,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED, shm_fd_, 0);
    if (ptr == MAP_FAILED) {
        std::perror("ShmChannel: mmap");
        ::close(shm_fd_);
        shm_fd_ = -1;
        return;
    }
    base_ = ptr;

#elif defined(RTEL_PLATFORM_WINDOWS)
    std::string map_name = "Local\\aeternus_rtel_" + cfg_.name;
    for (char& c : map_name) { if (c == '.') c = '_'; }

    HANDLE hMap = CreateFileMappingA(
        INVALID_HANDLE_VALUE, nullptr,
        PAGE_READWRITE,
        static_cast<DWORD>(total_bytes_ >> 32),
        static_cast<DWORD>(total_bytes_ & 0xFFFFFFFF),
        map_name.c_str());
    if (!hMap) {
        // Simulate with heap allocation (testing/non-IPC path)
        base_ = _aligned_malloc(total_bytes_, kCacheLineSize);
        if (base_) {
            std::memset(base_, 0, total_bytes_);
        }
    } else {
        base_ = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, total_bytes_);
        if (!base_) {
            CloseHandle(hMap);
        }
    }
#else
    // Fallback: heap (single-process only)
    base_ = std::aligned_alloc(kCacheLineSize, total_bytes_);
    if (base_) std::memset(base_, 0, total_bytes_);
#endif

    if (!base_) return;

    // Initialize control block
    ctrl_ = reinterpret_cast<RingControl*>(base_);
    ring_base_ = reinterpret_cast<SlotHeader*>(
        reinterpret_cast<uint8_t*>(base_) + ring_ctrl_bytes);

    if (cfg_.create) {
        ctrl_->magic      = kMagicHeader;
        ctrl_->slot_bytes = cfg_.slot_bytes;
        ctrl_->ring_cap   = cfg_.ring_capacity;
        ctrl_->schema_ver = 1;
        ctrl_->write_seq.store(1, std::memory_order_relaxed);
        ctrl_->min_read_seq.store(0, std::memory_order_relaxed);
        std::strncpy(ctrl_->channel_name, cfg_.name.c_str(),
                     kMaxChannelNameLen - 1);
    } else {
        // Validate existing shm
        if (ctrl_->magic != kMagicHeader) {
            std::fprintf(stderr, "ShmChannel: bad magic for '%s'\n",
                         cfg_.name.c_str());
            unmap_shm();
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// ShmChannel::unmap_shm
// ---------------------------------------------------------------------------
void ShmChannel::unmap_shm() {
    if (!base_) return;
#if defined(RTEL_PLATFORM_POSIX)
    munmap(base_, total_bytes_);
    if (shm_fd_ >= 0) {
        ::close(shm_fd_);
        shm_fd_ = -1;
    }
#elif defined(RTEL_PLATFORM_WINDOWS)
    UnmapViewOfFile(base_);
#else
    std::free(base_);
#endif
    base_      = nullptr;
    ctrl_      = nullptr;
    ring_base_ = nullptr;
}

// ---------------------------------------------------------------------------
// ShmChannel::slot_ptr
// ---------------------------------------------------------------------------
SlotHeader* ShmChannel::slot_ptr(std::size_t idx) noexcept {
    idx &= (cfg_.ring_capacity - 1);
    return reinterpret_cast<SlotHeader*>(
        reinterpret_cast<uint8_t*>(ring_base_) + idx * cfg_.slot_bytes);
}

const SlotHeader* ShmChannel::slot_ptr(std::size_t idx) const noexcept {
    idx &= (cfg_.ring_capacity - 1);
    return reinterpret_cast<const SlotHeader*>(
        reinterpret_cast<const uint8_t*>(ring_base_) + idx * cfg_.slot_bytes);
}

// ---------------------------------------------------------------------------
// ShmChannel::claim — writer claims a slot
// ---------------------------------------------------------------------------
std::pair<WriteHandle, ShmChannel::Error> ShmChannel::claim(bool block) noexcept {
    if (!base_) {
        return {WriteHandle{}, Error::SHM_FAILED};
    }

    // Sequence for next slot: odd while being written
    const uint64_t seq = ctrl_->write_seq.fetch_add(1, std::memory_order_acq_rel);
    const std::size_t idx = seq & (cfg_.ring_capacity - 1);
    SlotHeader* slot = slot_ptr(idx);

    // Spin until slot is free (previous write has been consumed)
    // A slot is "stale" if the reader cursor hasn't passed it.
    // We detect this by checking the slot's sequence number:
    //   if slot->sequence == seq, it hasn't been consumed yet → full ring.
    int spin = 0;
    while (true) {
        uint64_t prev = slot->sequence.load(std::memory_order_acquire);
        if (prev + cfg_.ring_capacity <= seq || prev == 0) {
            // Slot is old or uninitialized — safe to claim
            break;
        }
        if (!block) {
            // Return write_seq back (best-effort; approximate)
            stat_backpressure_.fetch_add(1, std::memory_order_relaxed);
            return {WriteHandle{}, Error::RING_FULL};
        }
        ++spin;
        if (spin < 100) {
            // Busy spin
            __asm__ volatile ("pause" ::: "memory");
        } else if (spin < 1000) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Mark slot as being written (sequence = odd: seq*2-1 convention is simplified here)
    slot->sequence.store(seq, std::memory_order_relaxed);
    slot->magic      = kMagicHeader;
    slot->flags      = 0;
    slot->schema_ver = 1;

    void* payload = reinterpret_cast<uint8_t*>(slot) + sizeof(SlotHeader);
    std::size_t cap = cfg_.slot_bytes - sizeof(SlotHeader);

    return {WriteHandle{this, slot, payload, cap, seq}, Error::OK};
}

// ---------------------------------------------------------------------------
// ShmChannel::commit_write / abort_write
// ---------------------------------------------------------------------------
void ShmChannel::commit_write(SlotHeader* slot, uint64_t seq) noexcept {
    // Store even sequence number (seq*2) to signal "readable"
    slot->sequence.store(seq + 1, std::memory_order_release);
    stat_published_.fetch_add(1, std::memory_order_relaxed);
    stat_bytes_written_.fetch_add(slot->tensor.payload_bytes,
                                  std::memory_order_relaxed);
}

void ShmChannel::abort_write(SlotHeader* slot, uint64_t /*seq*/) noexcept {
    // Mark slot as invalid by zeroing sequence
    slot->sequence.store(0, std::memory_order_release);
    slot->flags = 0;
}

// ---------------------------------------------------------------------------
// ShmChannel::peek — reader peeks at next available slot
// ---------------------------------------------------------------------------
const SlotHeader* ShmChannel::peek(ReadCursor& cur) const noexcept {
    if (!base_) return nullptr;
    const std::size_t idx = cur.next_seq & (cfg_.ring_capacity - 1);
    const SlotHeader* slot = slot_ptr(idx);
    uint64_t seq = slot->sequence.load(std::memory_order_acquire);
    // Published slots have seq == cur.next_seq + 1 (we stored seq+1 on commit)
    if (seq == cur.next_seq + 1) {
        return slot;
    }
    return nullptr;
}

void ShmChannel::advance(ReadCursor& cur) const noexcept {
    cur.next_seq += 1;
    stat_consumed_.fetch_add(1, std::memory_order_relaxed);
}

const void* ShmChannel::read_data(ReadCursor& cur,
                                   TensorDescriptor* td_out) noexcept {
    const SlotHeader* slot = peek(cur);
    if (!slot) return nullptr;
    if (td_out) *td_out = slot->tensor;
    const void* data = reinterpret_cast<const uint8_t*>(slot) + sizeof(SlotHeader);
    stat_bytes_read_.fetch_add(slot->tensor.payload_bytes,
                               std::memory_order_relaxed);
    advance(cur);
    return data;
}

// ---------------------------------------------------------------------------
// ShmChannel::stats
// ---------------------------------------------------------------------------
ChannelStats ShmChannel::stats() const noexcept {
    ChannelStats s{};
    s.published_total   = stat_published_.load(std::memory_order_relaxed);
    s.consumed_total    = stat_consumed_.load(std::memory_order_relaxed);
    s.dropped_slots     = stat_dropped_.load(std::memory_order_relaxed);
    s.backpressure_hits = stat_backpressure_.load(std::memory_order_relaxed);
    s.bytes_written     = stat_bytes_written_.load(std::memory_order_relaxed);
    s.bytes_read        = stat_bytes_read_.load(std::memory_order_relaxed);
    s.utilization_pct   = ring_utilization() * 100.0;
    return s;
}

void ShmChannel::reset_stats() noexcept {
    stat_published_.store(0, std::memory_order_relaxed);
    stat_consumed_.store(0, std::memory_order_relaxed);
    stat_dropped_.store(0, std::memory_order_relaxed);
    stat_backpressure_.store(0, std::memory_order_relaxed);
    stat_bytes_written_.store(0, std::memory_order_relaxed);
    stat_bytes_read_.store(0, std::memory_order_relaxed);
}

double ShmChannel::ring_utilization() const noexcept {
    if (!ctrl_) return 0.0;
    uint64_t w = ctrl_->write_seq.load(std::memory_order_relaxed);
    uint64_t r = ctrl_->min_read_seq.load(std::memory_order_relaxed);
    if (w <= r) return 0.0;
    double used = static_cast<double>(w - r);
    return std::min(1.0, used / static_cast<double>(cfg_.ring_capacity));
}

bool ShmChannel::is_backpressured() const noexcept {
    return ring_utilization() > 0.9;
}

// ---------------------------------------------------------------------------
// ShmBus
// ---------------------------------------------------------------------------
ShmBus& ShmBus::instance() {
    static ShmBus bus;
    return bus;
}

ShmBus::~ShmBus() {
    shutdown();
}

void ShmBus::spin_lock() noexcept {
    while (lock_.test_and_set(std::memory_order_acquire)) {
        __asm__ volatile ("pause" ::: "memory");
    }
}

void ShmBus::spin_unlock() noexcept {
    lock_.clear(std::memory_order_release);
}

bool ShmBus::register_channel(const ChannelConfig& cfg) {
    spin_lock();
    bool exists = (channels_.find(cfg.name) != channels_.end());
    if (!exists) {
        channels_.emplace(cfg.name,
                          std::make_unique<ShmChannel>(cfg));
    }
    spin_unlock();
    return !exists;
}

ShmChannel* ShmBus::channel(std::string_view name) noexcept {
    spin_lock();
    auto it = channels_.find(std::string(name));
    ShmChannel* ch = (it != channels_.end()) ? it->second.get() : nullptr;
    spin_unlock();
    return ch;
}

const ShmChannel* ShmBus::channel(std::string_view name) const noexcept {
    return const_cast<ShmBus*>(this)->channel(name);
}

std::pair<WriteHandle, ShmBus::Error>
ShmBus::claim(std::string_view channel_name, bool block) noexcept {
    ShmChannel* ch = channel(channel_name);
    if (!ch) return {WriteHandle{}, Error::INVALID_ARG};
    return ch->claim(block);
}

const void* ShmBus::read(std::string_view channel_name,
                          ReadCursor& cur,
                          TensorDescriptor* td_out) noexcept {
    ShmChannel* ch = channel(channel_name);
    if (!ch) return nullptr;
    return ch->read_data(cur, td_out);
}

std::vector<std::string> ShmBus::channel_names() const {
    spin_lock();
    std::vector<std::string> names;
    names.reserve(channels_.size());
    for (const auto& [k, v] : channels_) names.push_back(k);
    const_cast<ShmBus*>(this)->spin_unlock();
    return names;
}

std::unordered_map<std::string, ChannelStats> ShmBus::all_stats() const {
    std::unordered_map<std::string, ChannelStats> result;
    spin_lock();
    for (const auto& [k, v] : channels_) {
        result[k] = v->stats();
    }
    const_cast<ShmBus*>(this)->spin_unlock();
    return result;
}

void ShmBus::print_stats() const {
    auto stats = all_stats();
    std::printf("%-40s %10s %10s %10s %8s\n",
                "Channel", "Published", "Consumed", "Dropped", "Util%");
    std::printf("%s\n", std::string(80, '-').c_str());
    for (const auto& [name, s] : stats) {
        std::printf("%-40s %10lu %10lu %10lu %7.1f%%\n",
                    name.c_str(), s.published_total, s.consumed_total,
                    s.dropped_slots, s.utilization_pct);
    }
}

void ShmBus::shutdown() {
    spin_lock();
    channels_.clear();
    spin_unlock();
}

void ShmBus::create_aeternus_channels() {
    // LOB snapshots from Chronos: large slots for full order book
    register_channel({channels::LOB_SNAPSHOT,   128*1024, 512});
    // Volatility surface from Neuro-SDE
    register_channel({channels::VOL_SURFACE,    64*1024,  256});
    // Compressed tensor representations from TensorNet
    register_channel({channels::TENSOR_COMP,    256*1024, 256});
    // Graph adjacency from OmniGraph
    register_channel({channels::GRAPH_ADJ,      64*1024,  256});
    // Lumina predictions
    register_channel({channels::LUMINA_PRED,    64*1024,  512});
    // HyperAgent actions
    register_channel({channels::AGENT_ACTIONS,  16*1024,  1024});
    // Agent weight updates (infrequent, large)
    register_channel({channels::AGENT_WEIGHTS,  4*1024*1024, 8});
    // Pipeline events (small metadata)
    register_channel({channels::PIPELINE_EVENTS, 4*1024, 2048});
    // Heartbeat
    register_channel({channels::HEARTBEAT,       1*1024, 64});
}

} // namespace aeternus::rtel
