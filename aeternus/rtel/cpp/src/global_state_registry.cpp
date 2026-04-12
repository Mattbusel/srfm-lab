// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// global_state_registry.cpp — Global State Registry Implementation
// =============================================================================

#include "rtel/global_state_registry.hpp"
#include "rtel/shm_bus.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// GlobalStateRegistry — singleton
// ---------------------------------------------------------------------------
GlobalStateRegistry& GlobalStateRegistry::instance() {
    static GlobalStateRegistry gsr;
    return gsr;
}

GlobalStateRegistry::GlobalStateRegistry()
    : staging_(std::make_unique<WorldState>()) {
    // Initialize all generations to empty
    for (auto& s : states_) {
        std::memset(&s, 0, sizeof(WorldState));
        s.version = 0;
        s.flags   = 0;
    }
    current_idx_.store(0, std::memory_order_relaxed);
    version_.store(0, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Write lock (single writer)
// ---------------------------------------------------------------------------
void GlobalStateRegistry::write_spin_lock() noexcept {
    while (write_lock_.test_and_set(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void GlobalStateRegistry::write_spin_unlock() noexcept {
    write_lock_.clear(std::memory_order_release);
}

// ---------------------------------------------------------------------------
// begin_write / commit_write
// ---------------------------------------------------------------------------
WorldState& GlobalStateRegistry::begin_write() noexcept {
    write_spin_lock();
    // Copy current state to staging so caller can make incremental changes
    int cur = current_idx_.load(std::memory_order_relaxed);
    *staging_ = states_[cur];
    return *staging_;
}

uint64_t GlobalStateRegistry::commit_write() noexcept {
    // Determine next generation index
    int cur    = current_idx_.load(std::memory_order_relaxed);
    int next   = (cur + 1) % kGenerations;

    // Update version in staging
    uint64_t new_ver = version_.fetch_add(1, std::memory_order_relaxed) + 1;
    staging_->version      = new_ver;
    staging_->timestamp_ns = now_ns();

    // Seqlock write begin
    seqlock_.write_begin();

    // Copy staging into next generation slot
    states_[next] = *staging_;

    // Atomic swap: make next the current
    current_idx_.store(next, std::memory_order_release);

    // Seqlock write end
    seqlock_.write_end();

    stat_writes_.fetch_add(1, std::memory_order_relaxed);
    write_spin_unlock();

    // Publish version update to ShmBus pipeline events channel
    auto& bus = ShmBus::instance();
    auto* ch  = bus.channel(channels::PIPELINE_EVENTS);
    if (ch) {
        auto [handle, err] = ch->claim(false);
        if (err == ShmChannel::Error::OK) {
            struct VersionEvent {
                uint64_t version;
                uint64_t timestamp_ns;
                uint32_t flags;
                uint32_t source;  // 0=GSR
            };
            VersionEvent* ev = reinterpret_cast<VersionEvent*>(handle.data());
            ev->version      = new_ver;
            ev->timestamp_ns = staging_->timestamp_ns;
            ev->flags        = staging_->flags;
            ev->source       = 0;
            TensorDescriptor td{};
            td.dtype          = DType::UINT64;
            td.ndim           = 1;
            td.shape[0]       = sizeof(VersionEvent) / sizeof(uint64_t);
            td.payload_bytes  = sizeof(VersionEvent);
            handle.set_tensor(td);
            handle.publish();
        }
    }

    return new_ver;
}

// ---------------------------------------------------------------------------
// Read API — seqlock protected
// ---------------------------------------------------------------------------
uint64_t GlobalStateRegistry::read_world_state(WorldState& out) const noexcept {
    stat_reads_.fetch_add(1, std::memory_order_relaxed);
    uint64_t ver;
    do {
        uint64_t seq = seqlock_.read_begin();
        int cur = current_idx_.load(std::memory_order_acquire);
        out = states_[cur];
        if (seqlock_.read_consistent(seq)) {
            ver = out.version;
            break;
        }
        stat_retries_.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    } while (true);
    return ver;
}

bool GlobalStateRegistry::read_lob(uint32_t asset_id, LOBSnapshot& out) const noexcept {
    if (asset_id >= kMaxAssets) return false;
    stat_reads_.fetch_add(1, std::memory_order_relaxed);
    bool ok = false;
    do {
        uint64_t seq = seqlock_.read_begin();
        int cur = current_idx_.load(std::memory_order_acquire);
        if (states_[cur].lob_valid()) {
            out = states_[cur].lob[asset_id];
            ok = true;
        }
        if (seqlock_.read_consistent(seq)) break;
        stat_retries_.fetch_add(1, std::memory_order_relaxed);
    } while (true);
    return ok;
}

bool GlobalStateRegistry::read_vol_surface(uint32_t asset_id, VolSurface& out) const noexcept {
    if (asset_id >= kMaxAssets) return false;
    stat_reads_.fetch_add(1, std::memory_order_relaxed);
    bool ok = false;
    do {
        uint64_t seq = seqlock_.read_begin();
        int cur = current_idx_.load(std::memory_order_acquire);
        if (states_[cur].vol_valid()) {
            out = states_[cur].vol[asset_id];
            ok = true;
        }
        if (seqlock_.read_consistent(seq)) break;
        stat_retries_.fetch_add(1, std::memory_order_relaxed);
    } while (true);
    return ok;
}

bool GlobalStateRegistry::read_graph(GraphAdjacency& out) const noexcept {
    stat_reads_.fetch_add(1, std::memory_order_relaxed);
    bool ok = false;
    do {
        uint64_t seq = seqlock_.read_begin();
        int cur = current_idx_.load(std::memory_order_acquire);
        if (states_[cur].graph_valid()) {
            out = states_[cur].graph;
            ok = true;
        }
        if (seqlock_.read_consistent(seq)) break;
        stat_retries_.fetch_add(1, std::memory_order_relaxed);
    } while (true);
    return ok;
}

bool GlobalStateRegistry::read_tensor_comp(TensorCompressionState& out) const noexcept {
    stat_reads_.fetch_add(1, std::memory_order_relaxed);
    bool ok = false;
    do {
        uint64_t seq = seqlock_.read_begin();
        int cur = current_idx_.load(std::memory_order_acquire);
        if (states_[cur].tensor_valid()) {
            out = states_[cur].tensor_comp;
            ok = true;
        }
        if (seqlock_.read_consistent(seq)) break;
        stat_retries_.fetch_add(1, std::memory_order_relaxed);
    } while (true);
    return ok;
}

bool GlobalStateRegistry::read_weights(AgentWeightsManifest& out) const noexcept {
    stat_reads_.fetch_add(1, std::memory_order_relaxed);
    bool ok = false;
    do {
        uint64_t seq = seqlock_.read_begin();
        int cur = current_idx_.load(std::memory_order_acquire);
        if (states_[cur].weights_valid()) {
            out = states_[cur].weights;
            ok = true;
        }
        if (seqlock_.read_consistent(seq)) break;
        stat_retries_.fetch_add(1, std::memory_order_relaxed);
    } while (true);
    return ok;
}

uint64_t GlobalStateRegistry::version() const noexcept {
    return version_.load(std::memory_order_acquire);
}

// ---------------------------------------------------------------------------
// Convenience update methods
// ---------------------------------------------------------------------------
void GlobalStateRegistry::update_lob(uint32_t asset_id, const LOBSnapshot& snap) noexcept {
    if (asset_id >= kMaxAssets) return;
    WorldState& ws = begin_write();
    ws.lob[asset_id] = snap;
    ws.lob[asset_id].compute_derived();
    ws.flags |= WorldState::FLAG_LOB_VALID;
    if (ws.n_assets <= asset_id) ws.n_assets = asset_id + 1;
    commit_write();
}

void GlobalStateRegistry::update_vol_surface(uint32_t asset_id, const VolSurface& surf) noexcept {
    if (asset_id >= kMaxAssets) return;
    WorldState& ws = begin_write();
    ws.vol[asset_id] = surf;
    ws.flags |= WorldState::FLAG_VOL_VALID;
    commit_write();
}

void GlobalStateRegistry::update_graph(const GraphAdjacency& adj) noexcept {
    WorldState& ws = begin_write();
    ws.graph = adj;
    ws.flags |= WorldState::FLAG_GRAPH_VALID;
    commit_write();
}

void GlobalStateRegistry::update_tensor_comp(const TensorCompressionState& tc) noexcept {
    WorldState& ws = begin_write();
    ws.tensor_comp = tc;
    ws.flags |= WorldState::FLAG_TENSOR_VALID;
    commit_write();
}

void GlobalStateRegistry::update_weights(const AgentWeightsManifest& wm) noexcept {
    WorldState& ws = begin_write();
    ws.weights = wm;
    ws.flags |= WorldState::FLAG_WEIGHTS_VALID;
    commit_write();
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
void GlobalStateRegistry::print_summary() const noexcept {
    int cur = current_idx_.load(std::memory_order_relaxed);
    const WorldState& ws = states_[cur];
    std::printf("=== Global State Registry ===\n");
    std::printf("  Version:    %lu\n", ws.version);
    std::printf("  Timestamp:  %lu ns\n", ws.timestamp_ns);
    std::printf("  Assets:     %u\n", ws.n_assets);
    std::printf("  Flags:      0x%08X\n", ws.flags);
    if (ws.lob_valid() && ws.n_assets > 0) {
        const auto& lob = ws.lob[0];
        std::printf("  LOB[0]:     mid=%.4f spread=%.4f imbal=%.3f\n",
                    lob.mid_price, lob.spread, lob.bid_imbalance);
    }
    std::printf("  Writes:     %lu\n", stat_writes_.load(std::memory_order_relaxed));
    std::printf("  Reads:      %lu\n", stat_reads_.load(std::memory_order_relaxed));
    std::printf("  SeqRetries: %lu\n", stat_retries_.load(std::memory_order_relaxed));
}

GlobalStateRegistry::Stats GlobalStateRegistry::get_stats() const noexcept {
    Stats s{};
    s.total_writes    = stat_writes_.load(std::memory_order_relaxed);
    s.total_reads     = stat_reads_.load(std::memory_order_relaxed);
    s.seqlock_retries = stat_retries_.load(std::memory_order_relaxed);
    return s;
}

} // namespace aeternus::rtel
