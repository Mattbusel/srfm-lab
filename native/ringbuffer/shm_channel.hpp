#pragma once
// Shared memory communication channel between C++ and Zig processes.
// Uses a lock-free SPSC ring buffer in shared memory for tick data.
// C++ producer writes; Zig consumer reads (or vice versa).

#include "ringbuffer.hpp"
#include "tick_store.cpp"
#include <string>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace tickstore {

// Extended tick format for IPC (same binary layout as Zig's SharedTick)
struct IpcTick {
    int64_t  timestamp;
    char     symbol[16];
    int64_t  price;
    uint64_t qty;
    int64_t  bid_price;
    int64_t  ask_price;
    uint64_t bid_qty;
    uint64_t ask_qty;
    uint64_t trade_id;
    uint8_t  side;
    uint8_t  tick_type;
    uint8_t  flags;
    uint8_t  _pad[5];
    uint32_t seq_num;
    uint8_t  _pad2[4];
};
static_assert(sizeof(IpcTick) == 128, "IpcTick must be 128 bytes");

// ============================================================
// SHM Channel: bidirectional tick data channel
// Uses two ring buffers (one per direction)
// ============================================================
class ShmChannel {
public:
    static constexpr size_t kDefaultCapacity = 1 << 16; // 64K ticks
    static constexpr const char* kSuffix     = ".shm";

    // Writer (producer) side
    ShmChannel(const std::string& name, size_t capacity = kDefaultCapacity)
        : name_(name), capacity_(capacity),
          ring_ab_(name + "_ab" + kSuffix, capacity),  // A → B
          ring_ba_(name + "_ba" + kSuffix, capacity)   // B → A
    {}

    // Write tick to A→B channel (C++ → Zig)
    bool write_to_b(const IpcTick& t) {
        seq_ab_++;
        return ring_ab_.push(t) != UINT64_MAX;
    }

    // Read tick from B→A channel (Zig → C++)
    bool read_from_b(IpcTick& t) {
        return ring_ba_.read(read_ba_seq_++, t);
    }

    // Batch write
    size_t write_batch(const IpcTick* ticks, size_t n) {
        return ring_ab_.push_batch(ticks, n);
    }

    // Batch read from B→A
    size_t read_batch(IpcTick* ticks, size_t max_n) {
        return ring_ba_.snapshot(read_ba_seq_, ticks, max_n);
    }

    uint64_t write_count() const noexcept { return seq_ab_; }
    uint64_t a_to_b_size() const noexcept { return ring_ab_.size(); }
    uint64_t b_to_a_size() const noexcept { return ring_ba_.size(); }

    // Helper: convert Tick → IpcTick
    static IpcTick from_tick(const Tick& t, uint32_t seq = 0) {
        IpcTick ipc{};
        ipc.timestamp  = t.timestamp;
        std::memcpy(ipc.symbol, t.symbol, 16);
        ipc.price      = t.price;
        ipc.qty        = t.qty;
        ipc.bid_price  = t.bid_price;
        ipc.ask_price  = t.ask_price;
        ipc.bid_qty    = t.bid_qty;
        ipc.ask_qty    = t.ask_qty;
        ipc.trade_id   = t.trade_id;
        ipc.side       = t.side;
        ipc.tick_type  = t.tick_type;
        ipc.seq_num    = seq;
        return ipc;
    }

    static Tick to_tick(const IpcTick& ipc) {
        Tick t{};
        t.timestamp  = ipc.timestamp;
        std::memcpy(t.symbol, ipc.symbol, 16);
        t.price      = ipc.price;
        t.qty        = ipc.qty;
        t.bid_price  = ipc.bid_price;
        t.ask_price  = ipc.ask_price;
        t.bid_qty    = ipc.bid_qty;
        t.ask_qty    = ipc.ask_qty;
        t.trade_id   = ipc.trade_id;
        t.side       = ipc.side;
        t.tick_type  = ipc.tick_type;
        return t;
    }

private:
    std::string        name_;
    size_t             capacity_;
    RingBuffer<IpcTick> ring_ab_;
    RingBuffer<IpcTick> ring_ba_;
    uint64_t           seq_ab_       = 0;
    uint64_t           read_ba_seq_  = 0;
};

// ============================================================
// Market Data Publisher: fans out ticks to multiple channels
// ============================================================
class MarketDataPublisher {
public:
    explicit MarketDataPublisher(const std::string& base_name)
        : base_name_(base_name) {}

    // Add a subscriber channel
    void add_channel(const std::string& subscriber_name) {
        std::string chan_name = base_name_ + "_" + subscriber_name;
        channels_.emplace(subscriber_name,
            std::make_unique<ShmChannel>(chan_name));
    }

    // Publish tick to all channels
    void publish(const IpcTick& t) {
        ++total_published_;
        for (auto& [name, chan] : channels_) {
            if (!chan->write_to_b(t)) ++dropped_;
        }
    }

    // Publish from Tick
    void publish(const Tick& t) {
        publish(ShmChannel::from_tick(t, static_cast<uint32_t>(total_published_)));
    }

    // Publish to specific subscriber only
    bool publish_to(const std::string& name, const IpcTick& t) {
        auto it = channels_.find(name);
        if (it == channels_.end()) return false;
        return it->second->write_to_b(t);
    }

    uint64_t total_published() const noexcept { return total_published_; }
    uint64_t total_dropped()   const noexcept { return dropped_; }
    size_t   channel_count()   const noexcept { return channels_.size(); }

private:
    std::string base_name_;
    std::unordered_map<std::string, std::unique_ptr<ShmChannel>> channels_;
    uint64_t total_published_ = 0;
    uint64_t dropped_         = 0;
};

// ============================================================
// Performance benchmark for IPC channel throughput
// ============================================================
struct ChannelBenchResult {
    double throughput_mps;   // million ticks per second
    double latency_ns_p50;
    double latency_ns_p99;
    double bandwidth_gbps;
};

ChannelBenchResult bench_channel(size_t n_ticks = 1000000) {
    std::string tmp_path = "/tmp/bench_channel";
    std::remove((tmp_path + "_ab" + ShmChannel::kSuffix).c_str());
    std::remove((tmp_path + "_ba" + ShmChannel::kSuffix).c_str());

    // In-memory benchmark using direct ring buffer
    static constexpr size_t RING_CAP = 1 << 18;
    RingBuffer<IpcTick> ring(tmp_path + "_bench.ring", RING_CAP);

    IpcTick t{};
    t.timestamp = 0;
    std::memcpy(t.symbol, "BENCH\0\0\0\0\0\0\0\0\0\0\0", 16);
    t.price = 15000000; t.qty = 100; t.side = 0; t.tick_type = 0;

    std::vector<uint64_t> latencies;
    latencies.reserve(n_ticks);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n_ticks; ++i) {
        auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        t.timestamp = ts;
        t.seq_num   = static_cast<uint32_t>(i);
        ring.push(t);

        IpcTick out;
        ring.peek_latest(out);
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        latencies.push_back(static_cast<uint64_t>(now - ts));
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::sort(latencies.begin(), latencies.end());

    ChannelBenchResult r{};
    r.throughput_mps = n_ticks / ms / 1000.0;
    r.latency_ns_p50 = static_cast<double>(latencies[n_ticks / 2]);
    r.latency_ns_p99 = static_cast<double>(latencies[n_ticks * 99 / 100]);
    r.bandwidth_gbps = r.throughput_mps * sizeof(IpcTick) / 1000.0;

    std::remove((tmp_path + "_bench.ring").c_str());
    return r;
}

} // namespace tickstore
