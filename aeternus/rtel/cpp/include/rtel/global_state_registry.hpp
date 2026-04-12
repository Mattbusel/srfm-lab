// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// global_state_registry.hpp — Global State Registry (GSR)
// =============================================================================
// The GSR is the single source of truth for the current world state seen by
// all AETERNUS modules. It holds:
//
//   1. Order Book Snapshot    — bid/ask prices+sizes for N assets
//   2. Volatility Surface     — K strikes × T expiries per asset
//   3. Agent Weights Manifest — hash + version of current policy params
//   4. Graph Adjacency        — sparse CSR format for OmniGraph
//   5. Tensor Compression     — TT ranks per asset from TensorNet
//
// Concurrency model:
//   - All reads are lock-free via seqlock (64-bit version counter).
//     Reader retries if version changes during read.
//   - Writes use atomic pointer swap: writer builds new snapshot off-line,
//     then atomically swaps the pointer.
//   - Monotonic version counter incremented on every write.
//   - Three generations of snapshots maintained (current, prev, prev-prev)
//     to allow safe reads without memory reclamation concerns.
// =============================================================================

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "shm_bus.hpp"

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr std::size_t kMaxAssets     = 512;
static constexpr std::size_t kMaxLOBLevels  = 10;
static constexpr std::size_t kMaxStrikes    = 50;
static constexpr std::size_t kMaxExpiries   = 12;
static constexpr std::size_t kMaxTTRank     = 8;
static constexpr std::size_t kCSRMaxEdges   = 16384;
static constexpr std::size_t kCSRMaxNodes   = 512;

// ---------------------------------------------------------------------------
// Order Book Level
// ---------------------------------------------------------------------------
struct alignas(16) LOBLevel {
    double price = 0.0;
    double size  = 0.0;
};

// ---------------------------------------------------------------------------
// LOBSnapshot — order book snapshot for a single asset
// ---------------------------------------------------------------------------
struct alignas(kCacheLineSize) LOBSnapshot {
    uint32_t asset_id         = 0;
    uint32_t n_bid_levels     = 0;
    uint32_t n_ask_levels     = 0;
    uint32_t _pad             = 0;
    uint64_t exchange_ts_ns   = 0;
    uint64_t recv_ts_ns       = 0;
    uint64_t sequence         = 0;

    LOBLevel bids[kMaxLOBLevels];
    LOBLevel asks[kMaxLOBLevels];

    // Derived fields (computed, stored for fast access)
    double mid_price    = 0.0;
    double spread       = 0.0;
    double bid_imbalance= 0.0;  // (bid_depth - ask_depth) / (bid_depth + ask_depth)
    double vwap_bid     = 0.0;
    double vwap_ask     = 0.0;

    void compute_derived() noexcept {
        if (n_bid_levels == 0 || n_ask_levels == 0) return;
        mid_price = (bids[0].price + asks[0].price) * 0.5;
        spread    = asks[0].price - bids[0].price;

        double bid_depth = 0.0, ask_depth = 0.0;
        double bid_vol_sum = 0.0, ask_vol_sum = 0.0;

        for (uint32_t i = 0; i < n_bid_levels; ++i) {
            bid_depth    += bids[i].size;
            bid_vol_sum  += bids[i].price * bids[i].size;
        }
        for (uint32_t i = 0; i < n_ask_levels; ++i) {
            ask_depth    += asks[i].size;
            ask_vol_sum  += asks[i].price * asks[i].size;
        }
        double total_depth = bid_depth + ask_depth;
        bid_imbalance = (total_depth > 0.0)
                        ? (bid_depth - ask_depth) / total_depth
                        : 0.0;
        vwap_bid = (bid_depth > 0.0) ? bid_vol_sum / bid_depth : 0.0;
        vwap_ask = (ask_depth > 0.0) ? ask_vol_sum / ask_depth : 0.0;
    }
};

// ---------------------------------------------------------------------------
// VolatilitySurface — implied vol surface for a single asset
// ---------------------------------------------------------------------------
struct alignas(kCacheLineSize) VolSurface {
    uint32_t asset_id      = 0;
    uint32_t n_strikes     = 0;
    uint32_t n_expiries    = 0;
    uint32_t model_id      = 0;   // 0=BSM, 1=Heston, 2=SABR, 3=NeuralSDE
    uint64_t timestamp_ns  = 0;
    uint64_t version       = 0;

    double strikes[kMaxStrikes]  = {};   // absolute strike prices
    double expiries[kMaxExpiries]= {};   // time-to-expiry in years
    // vols[i*kMaxExpiries + j] = implied vol at strike i, expiry j
    double vols[kMaxStrikes * kMaxExpiries] = {};

    // SABR parameters (if model_id == 2)
    double sabr_alpha = 0.0;
    double sabr_beta  = 0.5;
    double sabr_rho   = 0.0;
    double sabr_nu    = 0.0;

    double vol_at(std::size_t i_strike, std::size_t j_expiry) const noexcept {
        if (i_strike >= n_strikes || j_expiry >= n_expiries) return 0.0;
        return vols[i_strike * kMaxExpiries + j_expiry];
    }
    double atm_vol(std::size_t j_expiry) const noexcept {
        // Return vol at middle strike
        return vol_at(n_strikes / 2, j_expiry);
    }
};

// ---------------------------------------------------------------------------
// AgentWeightsManifest — lightweight descriptor of current policy params
// ---------------------------------------------------------------------------
struct AgentWeightsManifest {
    uint64_t version          = 0;
    uint64_t timestamp_ns     = 0;
    uint64_t param_count      = 0;
    uint8_t  hash_sha256[32]  = {};   // SHA-256 of serialized weights
    char     model_name[64]   = {};
    char     checkpoint_path[256] = {};
    double   train_loss       = 0.0;
    double   val_loss         = 0.0;
    double   sharpe_ratio     = 0.0;
    uint32_t episode          = 0;
    uint32_t _pad             = 0;
};

// ---------------------------------------------------------------------------
// GraphAdjacency — sparse CSR adjacency for OmniGraph
// ---------------------------------------------------------------------------
struct alignas(kCacheLineSize) GraphAdjacency {
    uint32_t n_nodes          = 0;
    uint32_t n_edges          = 0;
    uint64_t version          = 0;
    uint64_t timestamp_ns     = 0;
    uint32_t graph_type       = 0;   // 0=correlation, 1=causality, 2=sector

    // CSR format: row_ptr[i]..row_ptr[i+1] gives edges for node i
    uint32_t row_ptr[kCSRMaxNodes + 1]    = {};
    uint32_t col_idx[kCSRMaxEdges]        = {};
    float    edge_weight[kCSRMaxEdges]    = {};
    char     node_names[kCSRMaxNodes][16] = {};

    // Get edges for node i → span of (col, weight) pairs
    std::size_t degree(uint32_t node) const noexcept {
        if (node >= n_nodes) return 0;
        return row_ptr[node + 1] - row_ptr[node];
    }
    const uint32_t* neighbors(uint32_t node) const noexcept {
        if (node >= n_nodes) return nullptr;
        return col_idx + row_ptr[node];
    }
    const float* weights(uint32_t node) const noexcept {
        if (node >= n_nodes) return nullptr;
        return edge_weight + row_ptr[node];
    }
};

// ---------------------------------------------------------------------------
// TensorCompressionState — TT ranks and compression stats per asset
// ---------------------------------------------------------------------------
struct TensorCompressionState {
    uint64_t version          = 0;
    uint64_t timestamp_ns     = 0;
    uint32_t n_assets         = 0;
    uint32_t tensor_ndim      = 0;

    struct AssetState {
        uint32_t asset_id = 0;
        uint32_t tt_ranks[kMaxTTRank] = {};
        double   compression_ratio = 1.0;
        double   reconstruction_error = 0.0;
        uint64_t last_update_ns = 0;
    };
    AssetState assets[kMaxAssets] = {};
};

// ---------------------------------------------------------------------------
// WorldState — the full GSR snapshot
// ---------------------------------------------------------------------------
struct alignas(kCacheLineSize) WorldState {
    uint64_t version          = 0;
    uint64_t timestamp_ns     = 0;
    uint32_t n_assets         = 0;
    uint32_t flags            = 0;

    static constexpr uint32_t FLAG_LOB_VALID    = 1u << 0;
    static constexpr uint32_t FLAG_VOL_VALID    = 1u << 1;
    static constexpr uint32_t FLAG_GRAPH_VALID  = 1u << 2;
    static constexpr uint32_t FLAG_TENSOR_VALID = 1u << 3;
    static constexpr uint32_t FLAG_WEIGHTS_VALID= 1u << 4;

    LOBSnapshot                lob[kMaxAssets];
    VolSurface                 vol[kMaxAssets];
    AgentWeightsManifest       weights;
    GraphAdjacency             graph;
    TensorCompressionState     tensor_comp;

    bool lob_valid()    const noexcept { return flags & FLAG_LOB_VALID; }
    bool vol_valid()    const noexcept { return flags & FLAG_VOL_VALID; }
    bool graph_valid()  const noexcept { return flags & FLAG_GRAPH_VALID; }
    bool tensor_valid() const noexcept { return flags & FLAG_TENSOR_VALID; }
    bool weights_valid()const noexcept { return flags & FLAG_WEIGHTS_VALID; }
};

// ---------------------------------------------------------------------------
// Seqlock — lightweight sequence-lock for readers
// ---------------------------------------------------------------------------
class Seqlock {
public:
    void write_begin() noexcept {
        seq_.fetch_add(1, std::memory_order_release);
        // Odd → write in progress
    }
    void write_end() noexcept {
        seq_.fetch_add(1, std::memory_order_release);
        // Even → write complete
    }
    // Returns current sequence; retry if odd
    uint64_t read_begin() const noexcept {
        uint64_t s;
        do { s = seq_.load(std::memory_order_acquire); }
        while (s & 1);  // spin while write in progress
        return s;
    }
    bool read_consistent(uint64_t s) const noexcept {
        std::atomic_thread_fence(std::memory_order_acquire);
        return seq_.load(std::memory_order_relaxed) == s;
    }

private:
    alignas(kCacheLineSize) std::atomic<uint64_t> seq_{0};
};

// ---------------------------------------------------------------------------
// GlobalStateRegistry — the GSR singleton
// ---------------------------------------------------------------------------
class GlobalStateRegistry {
public:
    static GlobalStateRegistry& instance();

    GlobalStateRegistry(const GlobalStateRegistry&) = delete;
    GlobalStateRegistry& operator=(const GlobalStateRegistry&) = delete;

    // -------------------------------------------------------------------------
    // Read API — lock-free seqlock reads
    // -------------------------------------------------------------------------

    // Read the full world state (copies into caller-supplied buffer)
    // Returns version number; if version==0, state not yet initialized.
    uint64_t read_world_state(WorldState& out) const noexcept;

    // Read just the LOB for a single asset (faster path)
    bool read_lob(uint32_t asset_id, LOBSnapshot& out) const noexcept;

    // Read vol surface for a single asset
    bool read_vol_surface(uint32_t asset_id, VolSurface& out) const noexcept;

    // Read graph adjacency
    bool read_graph(GraphAdjacency& out) const noexcept;

    // Read tensor compression state
    bool read_tensor_comp(TensorCompressionState& out) const noexcept;

    // Read agent weights manifest
    bool read_weights(AgentWeightsManifest& out) const noexcept;

    // Current version (monotonic)
    uint64_t version() const noexcept;

    // -------------------------------------------------------------------------
    // Write API — atomic pointer swap
    // -------------------------------------------------------------------------

    // Begin a write transaction: returns a mutable reference to the staging area.
    // Caller modifies the staging WorldState, then calls commit_write().
    WorldState& begin_write() noexcept;

    // Commit: increments version, swaps pointer, publishes to ShmBus
    uint64_t commit_write() noexcept;

    // Convenience: update just the LOB for one asset
    void update_lob(uint32_t asset_id, const LOBSnapshot& snap) noexcept;

    // Convenience: update vol surface for one asset
    void update_vol_surface(uint32_t asset_id, const VolSurface& surf) noexcept;

    // Convenience: update graph adjacency
    void update_graph(const GraphAdjacency& adj) noexcept;

    // Convenience: update tensor compression state
    void update_tensor_comp(const TensorCompressionState& tc) noexcept;

    // Convenience: update agent weights manifest
    void update_weights(const AgentWeightsManifest& wm) noexcept;

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------
    void print_summary() const noexcept;

    struct Stats {
        uint64_t total_writes        = 0;
        uint64_t total_reads         = 0;
        uint64_t seqlock_retries     = 0;
        uint64_t write_latency_p99ns = 0;
    };
    Stats get_stats() const noexcept;

private:
    GlobalStateRegistry();

    // Three generations: current, prev, prev-prev
    static constexpr std::size_t kGenerations = 3;
    alignas(kCacheLineSize) WorldState states_[kGenerations];

    std::atomic<int>      current_idx_{0};   // index of current read state
    std::atomic<uint64_t> version_{0};

    Seqlock seqlock_;

    // Staging area for writes (on heap to avoid stack overflow)
    std::unique_ptr<WorldState> staging_;
    int staging_gen_ = 1;  // which generation is staging

    // Stats
    mutable std::atomic<uint64_t> stat_writes_{0};
    mutable std::atomic<uint64_t> stat_reads_{0};
    mutable std::atomic<uint64_t> stat_retries_{0};

    // Spinlock for write serialization (single writer at a time)
    std::atomic_flag write_lock_ = ATOMIC_FLAG_INIT;
    void write_spin_lock() noexcept;
    void write_spin_unlock() noexcept;
};

// ---------------------------------------------------------------------------
// Helper: read-copy-update pattern for single asset LOB
// ---------------------------------------------------------------------------
inline void gsr_update_lob(uint32_t asset_id, const LOBSnapshot& snap) {
    GlobalStateRegistry::instance().update_lob(asset_id, snap);
}

inline bool gsr_read_lob(uint32_t asset_id, LOBSnapshot& out) {
    return GlobalStateRegistry::instance().read_lob(asset_id, out);
}

inline uint64_t gsr_version() {
    return GlobalStateRegistry::instance().version();
}

} // namespace aeternus::rtel
