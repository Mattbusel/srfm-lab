// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// module_wrapper.hpp — Plug-and-Play Module Wrapper (PPMW)
// =============================================================================
// Every AETERNUS module (Chronos, Neuro-SDE, TensorNet, OmniGraph, Lumina,
// HyperAgent) is wrapped in a concrete subclass of ModuleBase.
//
// Execution contract:
//   1. Scheduler calls module.forward(state, out) on every pipeline tick.
//   2. Module reads from state (shared reference to WorldState), runs its
//      compute, and writes results into OutputBuffer.
//   3. OutputBuffer is then flushed to the ShmBus by the scheduler.
//
// Module dependency graph:
//   Chronos → TensorNet → OmniGraph → Lumina → HyperAgent
//                    ↑
//               Neuro-SDE
//
// Topological execution order is computed once at startup and cached.
// =============================================================================

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "global_state_registry.hpp"
#include "ring_buffer.hpp"
#include "shm_bus.hpp"

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// ModuleID enum
// ---------------------------------------------------------------------------
enum class ModuleID : uint8_t {
    CHRONOS     = 0,
    NEURO_SDE   = 1,
    TENSORNET   = 2,
    OMNI_GRAPH  = 3,
    LUMINA      = 4,
    HYPER_AGENT = 5,
    COUNT       = 6,
};

inline const char* module_name(ModuleID id) noexcept {
    switch (id) {
        case ModuleID::CHRONOS:    return "Chronos";
        case ModuleID::NEURO_SDE:  return "Neuro-SDE";
        case ModuleID::TENSORNET:  return "TensorNet";
        case ModuleID::OMNI_GRAPH: return "OmniGraph";
        case ModuleID::LUMINA:     return "Lumina";
        case ModuleID::HYPER_AGENT:return "HyperAgent";
        default:                   return "Unknown";
    }
}

// ---------------------------------------------------------------------------
// OutputBuffer — module writes results here; scheduler flushes to ShmBus
// ---------------------------------------------------------------------------
struct OutputBuffer {
    // Opaque byte buffer for module-specific output
    std::vector<uint8_t>  data;
    TensorDescriptor      tensor;
    ModuleID              source_module = ModuleID::COUNT;
    uint64_t              timestamp_ns  = 0;
    uint64_t              pipeline_id   = 0;
    bool                  valid         = false;

    void reset() {
        data.clear();
        valid = false;
        tensor = {};
    }

    template<typename T>
    T* allocate(std::size_t n) {
        data.resize(n * sizeof(T));
        return reinterpret_cast<T*>(data.data());
    }

    template<typename T>
    const T* as() const {
        return reinterpret_cast<const T*>(data.data());
    }
};

// ---------------------------------------------------------------------------
// ModuleConfig
// ---------------------------------------------------------------------------
struct ModuleConfig {
    ModuleID    id;
    std::string name;
    bool        enabled          = true;
    uint32_t    priority         = 0;       // lower = higher priority
    uint64_t    timeout_us       = 1000;    // microsecond deadline
    bool        async            = false;   // run in background thread
    std::string output_channel;            // ShmBus channel to publish to
    std::unordered_map<std::string, std::string> params;
};

// ---------------------------------------------------------------------------
// ModuleStats — runtime metrics per module
// ---------------------------------------------------------------------------
struct ModuleStats {
    std::atomic<uint64_t> invocations{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> timeouts{0};
    std::atomic<uint64_t> total_cycles{0};
    LatencyHistogram       latency_hist;

    uint64_t mean_cycles() const noexcept {
        uint64_t inv = invocations.load(std::memory_order_relaxed);
        if (inv == 0) return 0;
        return total_cycles.load(std::memory_order_relaxed) / inv;
    }
};

// ---------------------------------------------------------------------------
// ModuleBase — abstract base for all AETERNUS modules
// ---------------------------------------------------------------------------
class ModuleBase {
public:
    explicit ModuleBase(ModuleConfig cfg) : cfg_(std::move(cfg)) {}
    virtual ~ModuleBase() = default;

    ModuleBase(const ModuleBase&) = delete;
    ModuleBase& operator=(const ModuleBase&) = delete;

    // Core execution method — called by scheduler on every pipeline tick
    // Returns true on success, false on error
    virtual bool forward(const WorldState& state, OutputBuffer& out) = 0;

    // Lifecycle
    virtual bool initialize()  { return true; }
    virtual void shutdown()    {}
    virtual bool is_healthy()  { return true; }

    // Warm up (optional pre-loading of model weights etc.)
    virtual void warmup(int n_iterations = 10) {
        (void)n_iterations;
    }

    // Dependencies — IDs of modules that must run before this one
    virtual std::vector<ModuleID> dependencies() const { return {}; }

    // Accessor
    const ModuleConfig& config() const noexcept { return cfg_; }
    ModuleID            id()     const noexcept { return cfg_.id; }
    const std::string&  name()   const noexcept { return cfg_.name; }
    bool                enabled()const noexcept { return cfg_.enabled; }

    ModuleStats& stats() noexcept { return stats_; }

    // Timed forward — wraps forward() with cycle counting
    bool timed_forward(const WorldState& state, OutputBuffer& out) noexcept {
        uint64_t t0 = rdtsc();
        bool ok = forward(state, out);
        uint64_t t1 = rdtsc();
        uint64_t cycles = t1 - t0;
        stats_.invocations.fetch_add(1, std::memory_order_relaxed);
        stats_.total_cycles.fetch_add(cycles, std::memory_order_relaxed);
        stats_.latency_hist.record(cycles);
        if (!ok) stats_.errors.fetch_add(1, std::memory_order_relaxed);
        return ok;
    }

protected:
    ModuleConfig cfg_;
    ModuleStats  stats_;

    // Publish output to ShmBus
    bool publish_output(const OutputBuffer& out) noexcept {
        if (!out.valid || out.data.empty()) return false;
        auto& bus = ShmBus::instance();
        auto* ch  = bus.channel(cfg_.output_channel);
        if (!ch) return false;
        auto [handle, err] = ch->claim(false);
        if (err != ShmChannel::Error::OK) return false;
        std::size_t n = std::min(out.data.size(), handle.capacity());
        std::memcpy(handle.data(), out.data.data(), n);
        handle.set_tensor(out.tensor);
        handle.publish();
        return true;
    }
};

// ---------------------------------------------------------------------------
// ChronosWrapper — wraps Chronos LOB engine
// ---------------------------------------------------------------------------
class ChronosWrapper : public ModuleBase {
public:
    explicit ChronosWrapper(ModuleConfig cfg = default_config());
    bool forward(const WorldState& state, OutputBuffer& out) override;
    bool initialize() override;
    std::vector<ModuleID> dependencies() const override { return {}; }

    // Inject a market event (called by market data feed)
    void inject_event(const MarketEvent& ev);

    static ModuleConfig default_config() {
        return {ModuleID::CHRONOS, "Chronos", true, 0, 500,
                false, std::string(channels::LOB_SNAPSHOT), {}};
    }

private:
    // Simulated LOB state (in production: binds to actual Chronos C++ lib)
    struct LOBState {
        LOBSnapshot snap;
        bool dirty = false;
    };
    std::array<LOBState, kMaxAssets> lob_states_{};
    uint32_t n_active_assets_ = 1;
    std::atomic<bool> has_new_data_{false};
    MarketEvent latest_event_{};
};

// ---------------------------------------------------------------------------
// NeuroSDEWrapper — wraps Neuro-SDE volatility surface engine
// ---------------------------------------------------------------------------
class NeuroSDEWrapper : public ModuleBase {
public:
    explicit NeuroSDEWrapper(ModuleConfig cfg = default_config());
    bool forward(const WorldState& state, OutputBuffer& out) override;
    bool initialize() override;
    std::vector<ModuleID> dependencies() const override {
        return {ModuleID::CHRONOS};
    }

    static ModuleConfig default_config() {
        return {ModuleID::NEURO_SDE, "Neuro-SDE", true, 1, 2000,
                false, std::string(channels::VOL_SURFACE), {}};
    }

private:
    // Heston SDE parameters per asset
    struct HestonParams {
        double kappa = 2.0;    // mean reversion speed
        double theta = 0.04;   // long-run variance
        double sigma = 0.3;    // vol of vol
        double rho   = -0.7;   // correlation
        double v0    = 0.04;   // initial variance
    };
    std::array<HestonParams, kMaxAssets> heston_{};

    VolSurface compute_heston_surface(uint32_t asset_id,
                                       const LOBSnapshot& lob,
                                       const HestonParams& p) const noexcept;
};

// ---------------------------------------------------------------------------
// TensorNetWrapper — wraps TensorNet compression engine
// ---------------------------------------------------------------------------
class TensorNetWrapper : public ModuleBase {
public:
    explicit TensorNetWrapper(ModuleConfig cfg = default_config());
    bool forward(const WorldState& state, OutputBuffer& out) override;
    bool initialize() override;
    std::vector<ModuleID> dependencies() const override {
        return {ModuleID::CHRONOS, ModuleID::NEURO_SDE};
    }

    static ModuleConfig default_config() {
        return {ModuleID::TENSORNET, "TensorNet", true, 2, 3000,
                false, std::string(channels::TENSOR_COMP), {}};
    }

private:
    std::array<uint32_t, kMaxAssets * kMaxTTRank> tt_ranks_{};
    TensorCompressionState compress(const WorldState& state) const noexcept;
};

// ---------------------------------------------------------------------------
// OmniGraphWrapper — wraps OmniGraph correlation/causality engine
// ---------------------------------------------------------------------------
class OmniGraphWrapper : public ModuleBase {
public:
    explicit OmniGraphWrapper(ModuleConfig cfg = default_config());
    bool forward(const WorldState& state, OutputBuffer& out) override;
    bool initialize() override;
    std::vector<ModuleID> dependencies() const override {
        return {ModuleID::TENSORNET};
    }

    static ModuleConfig default_config() {
        return {ModuleID::OMNI_GRAPH, "OmniGraph", true, 3, 5000,
                false, std::string(channels::GRAPH_ADJ), {}};
    }

private:
    // Rolling correlation matrix (n_assets × n_assets)
    std::vector<double> corr_matrix_;
    uint32_t n_assets_ = 0;
    std::vector<double> price_history_; // flattened: [asset][time]
    std::size_t history_len_ = 100;

    GraphAdjacency build_graph(const WorldState& state) const noexcept;
    double pearson_correlation(uint32_t a, uint32_t b,
                                const WorldState& state) const noexcept;
};

// ---------------------------------------------------------------------------
// LuminaWrapper — wraps Lumina neural inference engine (pybind11 bridge)
// ---------------------------------------------------------------------------
class LuminaWrapper : public ModuleBase {
public:
    explicit LuminaWrapper(ModuleConfig cfg = default_config());
    bool forward(const WorldState& state, OutputBuffer& out) override;
    bool initialize() override;
    std::vector<ModuleID> dependencies() const override {
        return {ModuleID::TENSORNET, ModuleID::OMNI_GRAPH};
    }

    static ModuleConfig default_config() {
        return {ModuleID::LUMINA, "Lumina", true, 4, 5000,
                false, std::string(channels::LUMINA_PRED), {}};
    }

    struct Prediction {
        float return_forecast[kMaxAssets]    = {};
        float risk_forecast[kMaxAssets]      = {};
        float confidence[kMaxAssets]         = {};
        uint64_t timestamp_ns                = 0;
        uint32_t model_version               = 0;
    };

private:
    bool python_initialized_ = false;
    uint32_t model_version_ = 0;
    // In production: pybind11 Python interpreter handle
    // For RTEL standalone: use a simple linear regression stub
    Prediction run_inference(const WorldState& state) const noexcept;
};

// ---------------------------------------------------------------------------
// HyperAgentWrapper — wraps HyperAgent RL policy
// ---------------------------------------------------------------------------
class HyperAgentWrapper : public ModuleBase {
public:
    explicit HyperAgentWrapper(ModuleConfig cfg = default_config());
    bool forward(const WorldState& state, OutputBuffer& out) override;
    bool initialize() override;
    std::vector<ModuleID> dependencies() const override {
        return {ModuleID::LUMINA};
    }

    static ModuleConfig default_config() {
        return {ModuleID::HYPER_AGENT, "HyperAgent", true, 5, 1000,
                false, std::string(channels::AGENT_ACTIONS), {}};
    }

    struct Action {
        float  position_delta[kMaxAssets] = {};  // target position change
        float  confidence                 = 0.0f;
        uint32_t action_type             = 0;    // 0=hold, 1=buy, 2=sell
        float  risk_budget               = 0.0f;
        uint64_t timestamp_ns            = 0;
    };

private:
    // Policy network (stub: linear in state features)
    std::vector<float> policy_weights_;
    Action compute_action(const WorldState& state,
                           const LuminaWrapper::Prediction& pred) const noexcept;
};

// ---------------------------------------------------------------------------
// ModuleRegistry — registers and owns all module wrappers
// ---------------------------------------------------------------------------
class ModuleRegistry {
public:
    static ModuleRegistry& instance();

    ModuleRegistry(const ModuleRegistry&) = delete;
    ModuleRegistry& operator=(const ModuleRegistry&) = delete;

    // Register a module (takes ownership)
    void register_module(std::unique_ptr<ModuleBase> module);

    // Get module by ID
    ModuleBase* get(ModuleID id) noexcept;
    const ModuleBase* get(ModuleID id) const noexcept;

    // Get topologically sorted execution order
    // Returns vector of ModuleIDs in dependency-resolved order
    std::vector<ModuleID> topological_order() const;

    // Initialize all registered modules
    bool initialize_all();

    // Shutdown all modules
    void shutdown_all();

    // Execute full pipeline on given world state
    // Returns number of successful module executions
    int execute_pipeline(const WorldState& state);

    std::vector<std::string> list_modules() const;

private:
    ModuleRegistry() = default;

    std::unordered_map<int, std::unique_ptr<ModuleBase>> modules_;

    // DFS topological sort
    void topo_dfs(ModuleID id,
                  std::unordered_map<int, int>& visited,
                  std::vector<ModuleID>& order) const;
};

} // namespace aeternus::rtel
