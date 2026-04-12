// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// module_wrapper.cpp — Plug-and-Play Module Wrapper Implementation
// =============================================================================

#include "rtel/module_wrapper.hpp"
#include "rtel/shm_bus.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

namespace aeternus::rtel {

// ---------------------------------------------------------------------------
// ChronosWrapper
// ---------------------------------------------------------------------------
ChronosWrapper::ChronosWrapper(ModuleConfig cfg)
    : ModuleBase(std::move(cfg)) {}

bool ChronosWrapper::initialize() {
    // Initialize simulated LOB state for 1 asset (AAPL-like)
    n_active_assets_ = 1;
    for (uint32_t i = 0; i < n_active_assets_; ++i) {
        auto& s = lob_states_[i].snap;
        s.asset_id = i;
        s.n_bid_levels = kMaxLOBLevels;
        s.n_ask_levels = kMaxLOBLevels;
        double base_price = 150.0 + i * 10.0;
        for (uint32_t j = 0; j < kMaxLOBLevels; ++j) {
            s.bids[j].price = base_price - (j + 1) * 0.01;
            s.bids[j].size  = 100.0 * (kMaxLOBLevels - j);
            s.asks[j].price = base_price + (j + 1) * 0.01;
            s.asks[j].size  = 100.0 * (kMaxLOBLevels - j);
        }
        s.compute_derived();
        lob_states_[i].dirty = true;
    }
    return true;
}

void ChronosWrapper::inject_event(const MarketEvent& ev) {
    latest_event_ = ev;
    if (ev.asset_id < kMaxAssets) {
        auto& s = lob_states_[ev.asset_id].snap;
        // Simple price impact: update best bid/ask
        if (ev.event_type == 0) {  // tick
            s.bids[0].price = ev.bid;
            s.asks[0].price = ev.ask;
            s.compute_derived();
            lob_states_[ev.asset_id].dirty = true;
        }
    }
    has_new_data_.store(true, std::memory_order_release);
}

bool ChronosWrapper::forward(const WorldState& /*state*/, OutputBuffer& out) {
    out.reset();
    out.source_module = ModuleID::CHRONOS;
    out.timestamp_ns  = now_ns();

    bool any_dirty = false;
    for (uint32_t i = 0; i < n_active_assets_; ++i) {
        if (!lob_states_[i].dirty) continue;
        any_dirty = true;

        auto& snap = lob_states_[i].snap;
        snap.recv_ts_ns = out.timestamp_ns;
        snap.sequence++;

        // Simulate micro-price movement
        std::mt19937_64 rng(out.timestamp_ns ^ snap.sequence);
        std::normal_distribution<double> noise(0.0, 0.0001);
        double delta = noise(rng);
        for (uint32_t j = 0; j < snap.n_bid_levels; ++j) {
            snap.bids[j].price += delta;
            snap.asks[j].price += delta;
        }
        snap.compute_derived();

        // Write to GSR
        GlobalStateRegistry::instance().update_lob(i, snap);
        lob_states_[i].dirty = false;
    }

    if (any_dirty) {
        // Pack LOB snapshots into output buffer
        std::size_t n = n_active_assets_;
        LOBSnapshot* buf = out.allocate<LOBSnapshot>(n);
        for (uint32_t i = 0; i < n; ++i) {
            buf[i] = lob_states_[i].snap;
        }
        out.tensor.dtype        = DType::FLOAT64;
        out.tensor.ndim         = 2;
        out.tensor.shape[0]     = n;
        out.tensor.shape[1]     = sizeof(LOBSnapshot) / sizeof(double);
        out.tensor.payload_bytes= n * sizeof(LOBSnapshot);
        out.valid = true;
        publish_output(out);
    }

    has_new_data_.store(false, std::memory_order_release);
    return true;
}

// ---------------------------------------------------------------------------
// NeuroSDEWrapper
// ---------------------------------------------------------------------------
NeuroSDEWrapper::NeuroSDEWrapper(ModuleConfig cfg)
    : ModuleBase(std::move(cfg)) {}

bool NeuroSDEWrapper::initialize() {
    // Set default Heston parameters for each asset
    for (auto& p : heston_) {
        p = {2.0, 0.04, 0.3, -0.7, 0.04};
    }
    return true;
}

VolSurface NeuroSDEWrapper::compute_heston_surface(
    uint32_t asset_id,
    const LOBSnapshot& lob,
    const HestonParams& p) const noexcept
{
    VolSurface surf{};
    surf.asset_id   = asset_id;
    surf.model_id   = 1;  // Heston
    surf.n_strikes  = 5;
    surf.n_expiries = 4;
    surf.timestamp_ns = now_ns();
    surf.version    = lob.sequence;

    double S = lob.mid_price;
    if (S <= 0.0) S = 100.0;

    // Strikes: 80%..120% of spot
    for (uint32_t i = 0; i < surf.n_strikes; ++i) {
        surf.strikes[i] = S * (0.80 + 0.10 * i);
    }
    // Expiries: 1w, 1m, 3m, 6m
    double expiries[4] = {7.0/365, 30.0/365, 91.0/365, 182.0/365};
    for (uint32_t j = 0; j < surf.n_expiries; ++j) {
        surf.expiries[j] = expiries[j];
    }

    // Approximate Heston implied vol using Gatheral approximation
    double v0 = p.v0;
    double sqrt_v0 = std::sqrt(v0);
    for (uint32_t i = 0; i < surf.n_strikes; ++i) {
        double log_moneyness = std::log(surf.strikes[i] / S);
        for (uint32_t j = 0; j < surf.n_expiries; ++j) {
            double T = surf.expiries[j];
            // Simple parameterization (not full Heston, but captures ATM and skew)
            double atm_vol = sqrt_v0 * std::sqrt(
                (1.0 - std::exp(-p.kappa * T)) / (p.kappa * T) * p.theta
                + (1.0 - (1.0 - std::exp(-p.kappa * T)) / (p.kappa * T)) * v0);
            double skew = p.rho * p.sigma / p.kappa
                        * (1.0 - std::exp(-p.kappa * T)) / T
                        * log_moneyness;
            double smile = 0.5 * p.sigma * p.sigma / (p.kappa * p.kappa)
                         * (log_moneyness * log_moneyness / T);
            double vol = std::max(0.01, atm_vol + skew + smile);
            surf.vols[i * kMaxExpiries + j] = vol;
        }
    }
    surf.sabr_alpha = sqrt_v0;
    surf.sabr_beta  = 0.5;
    surf.sabr_rho   = p.rho;
    surf.sabr_nu    = p.sigma;
    return surf;
}

bool NeuroSDEWrapper::forward(const WorldState& state, OutputBuffer& out) {
    out.reset();
    out.source_module = ModuleID::NEURO_SDE;
    out.timestamp_ns  = now_ns();

    if (!state.lob_valid()) return true;

    for (uint32_t i = 0; i < state.n_assets; ++i) {
        auto surf = compute_heston_surface(i, state.lob[i], heston_[i]);
        GlobalStateRegistry::instance().update_vol_surface(i, surf);
    }

    // Pack first asset surface into output
    if (state.n_assets > 0) {
        VolSurface* buf = out.allocate<VolSurface>(state.n_assets);
        for (uint32_t i = 0; i < state.n_assets; ++i) {
            buf[i] = compute_heston_surface(i, state.lob[i], heston_[i]);
        }
        out.tensor.dtype        = DType::FLOAT64;
        out.tensor.ndim         = 2;
        out.tensor.shape[0]     = state.n_assets;
        out.tensor.shape[1]     = kMaxStrikes * kMaxExpiries;
        out.tensor.payload_bytes= state.n_assets * sizeof(VolSurface);
        out.valid = true;
        publish_output(out);
    }
    return true;
}

// ---------------------------------------------------------------------------
// TensorNetWrapper
// ---------------------------------------------------------------------------
TensorNetWrapper::TensorNetWrapper(ModuleConfig cfg)
    : ModuleBase(std::move(cfg)) {}

bool TensorNetWrapper::initialize() {
    // Initialize TT ranks: 4 for all assets
    for (auto& r : tt_ranks_) r = 4;
    return true;
}

TensorCompressionState TensorNetWrapper::compress(const WorldState& state) const noexcept {
    TensorCompressionState tc{};
    tc.version      = state.version;
    tc.timestamp_ns = now_ns();
    tc.n_assets     = state.n_assets;
    tc.tensor_ndim  = 3;  // asset × strike × expiry

    for (uint32_t i = 0; i < state.n_assets; ++i) {
        auto& as = tc.assets[i];
        as.asset_id = i;
        // Default TT ranks
        for (std::size_t r = 0; r < kMaxTTRank; ++r) {
            as.tt_ranks[r] = tt_ranks_[i * kMaxTTRank + r];
        }
        // Compute compression ratio from vol surface shape
        uint32_t n_s = state.vol[i].n_strikes;
        uint32_t n_e = state.vol[i].n_expiries;
        if (n_s > 0 && n_e > 0) {
            double full_size = n_s * n_e;
            double tt_rank   = as.tt_ranks[0];
            double tt_size   = tt_rank * n_s + tt_rank * tt_rank + tt_rank * n_e;
            as.compression_ratio = full_size / std::max(1.0, tt_size);
        } else {
            as.compression_ratio = 1.0;
        }
        as.reconstruction_error = 1e-6;
        as.last_update_ns = tc.timestamp_ns;
    }
    return tc;
}

bool TensorNetWrapper::forward(const WorldState& state, OutputBuffer& out) {
    out.reset();
    out.source_module = ModuleID::TENSORNET;
    out.timestamp_ns  = now_ns();

    TensorCompressionState tc = compress(state);
    GlobalStateRegistry::instance().update_tensor_comp(tc);

    TensorCompressionState* buf = out.allocate<TensorCompressionState>(1);
    *buf = tc;
    out.tensor.dtype        = DType::FLOAT32;
    out.tensor.ndim         = 1;
    out.tensor.shape[0]     = sizeof(TensorCompressionState);
    out.tensor.payload_bytes= sizeof(TensorCompressionState);
    out.valid = true;
    publish_output(out);
    return true;
}

// ---------------------------------------------------------------------------
// OmniGraphWrapper
// ---------------------------------------------------------------------------
OmniGraphWrapper::OmniGraphWrapper(ModuleConfig cfg)
    : ModuleBase(std::move(cfg)) {}

bool OmniGraphWrapper::initialize() {
    n_assets_ = 1;
    corr_matrix_.assign(n_assets_ * n_assets_, 0.0);
    price_history_.assign(n_assets_ * history_len_, 0.0);
    return true;
}

double OmniGraphWrapper::pearson_correlation(uint32_t a, uint32_t b,
                                              const WorldState& state) const noexcept {
    (void)state;
    if (a == b) return 1.0;
    // Simplified: use mid-price history
    std::size_t n = std::min(history_len_, (std::size_t)30);
    if (n < 2) return 0.0;
    double sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    for (std::size_t t = 0; t < n; ++t) {
        double pa = price_history_[a * history_len_ + t];
        double pb = price_history_[b * history_len_ + t];
        sum_a  += pa;
        sum_b  += pb;
        sum_ab += pa * pb;
        sum_a2 += pa * pa;
        sum_b2 += pb * pb;
    }
    double mean_a = sum_a / n;
    double mean_b = sum_b / n;
    double num = sum_ab - n * mean_a * mean_b;
    double den = std::sqrt((sum_a2 - n*mean_a*mean_a) * (sum_b2 - n*mean_b*mean_b));
    return (den > 1e-10) ? (num / den) : 0.0;
}

GraphAdjacency OmniGraphWrapper::build_graph(const WorldState& state) const noexcept {
    GraphAdjacency adj{};
    adj.version      = state.version;
    adj.timestamp_ns = now_ns();
    adj.graph_type   = 0;  // correlation
    adj.n_nodes      = state.n_assets;

    // Build CSR: edge if |corr| > 0.3
    uint32_t edge_count = 0;
    adj.row_ptr[0] = 0;
    for (uint32_t i = 0; i < adj.n_nodes; ++i) {
        for (uint32_t j = 0; j < adj.n_nodes; ++j) {
            if (i == j) continue;
            double c = pearson_correlation(i, j, state);
            if (std::fabs(c) > 0.3 && edge_count < kCSRMaxEdges) {
                adj.col_idx[edge_count]    = j;
                adj.edge_weight[edge_count]= static_cast<float>(c);
                ++edge_count;
            }
        }
        adj.row_ptr[i + 1] = edge_count;
    }
    adj.n_edges = edge_count;

    // Fill node names
    for (uint32_t i = 0; i < adj.n_nodes; ++i) {
        std::snprintf(adj.node_names[i], 16, "ASSET_%03u", i);
    }
    return adj;
}

bool OmniGraphWrapper::forward(const WorldState& state, OutputBuffer& out) {
    out.reset();
    out.source_module = ModuleID::OMNI_GRAPH;
    out.timestamp_ns  = now_ns();

    // Update price history
    if (state.n_assets != n_assets_) {
        n_assets_ = state.n_assets;
        corr_matrix_.assign(n_assets_ * n_assets_, 0.0);
        price_history_.assign(n_assets_ * history_len_, 0.0);
    }
    for (uint32_t i = 0; i < state.n_assets; ++i) {
        // Shift history
        for (std::size_t t = history_len_ - 1; t > 0; --t) {
            price_history_[i * history_len_ + t] =
                price_history_[i * history_len_ + t - 1];
        }
        price_history_[i * history_len_] = state.lob[i].mid_price;
    }

    GraphAdjacency adj = build_graph(state);
    GlobalStateRegistry::instance().update_graph(adj);

    GraphAdjacency* buf = out.allocate<GraphAdjacency>(1);
    *buf = adj;
    out.tensor.dtype        = DType::FLOAT32;
    out.tensor.ndim         = 1;
    out.tensor.shape[0]     = sizeof(GraphAdjacency);
    out.tensor.payload_bytes= sizeof(GraphAdjacency);
    out.valid = true;
    publish_output(out);
    return true;
}

// ---------------------------------------------------------------------------
// LuminaWrapper
// ---------------------------------------------------------------------------
LuminaWrapper::LuminaWrapper(ModuleConfig cfg)
    : ModuleBase(std::move(cfg)) {}

bool LuminaWrapper::initialize() {
    // In production: initialize Python interpreter via pybind11
    // For standalone RTEL: use linear model stub
    python_initialized_ = true;
    model_version_       = 1;
    return true;
}

LuminaWrapper::Prediction LuminaWrapper::run_inference(const WorldState& state) const noexcept {
    Prediction pred{};
    pred.timestamp_ns  = now_ns();
    pred.model_version = model_version_;

    // Stub: linear model using LOB imbalance → return forecast
    for (uint32_t i = 0; i < state.n_assets && i < kMaxAssets; ++i) {
        const auto& lob = state.lob[i];
        // Simple alpha signal: imbalance × spread
        double signal = lob.bid_imbalance * lob.spread;
        pred.return_forecast[i] = static_cast<float>(signal * 0.01);
        pred.risk_forecast[i]   = static_cast<float>(std::fabs(signal) * 0.1 + 0.001);
        pred.confidence[i]      = static_cast<float>(
            std::min(1.0, std::fabs(lob.bid_imbalance) * 2.0));
    }
    return pred;
}

bool LuminaWrapper::forward(const WorldState& state, OutputBuffer& out) {
    out.reset();
    out.source_module = ModuleID::LUMINA;
    out.timestamp_ns  = now_ns();

    if (!state.lob_valid()) return true;

    Prediction pred = run_inference(state);
    Prediction* buf = out.allocate<Prediction>(1);
    *buf = pred;
    out.tensor.dtype        = DType::FLOAT32;
    out.tensor.ndim         = 2;
    out.tensor.shape[0]     = 3;               // return, risk, confidence
    out.tensor.shape[1]     = kMaxAssets;
    out.tensor.payload_bytes= sizeof(Prediction);
    out.valid = true;
    publish_output(out);
    return true;
}

// ---------------------------------------------------------------------------
// HyperAgentWrapper
// ---------------------------------------------------------------------------
HyperAgentWrapper::HyperAgentWrapper(ModuleConfig cfg)
    : ModuleBase(std::move(cfg)) {}

bool HyperAgentWrapper::initialize() {
    // Initialize linear policy weights
    policy_weights_.assign(kMaxAssets * 3 + 1, 0.0f);
    // Default: simple momentum-following policy
    for (std::size_t i = 0; i < kMaxAssets; ++i) {
        policy_weights_[i * 3 + 0] = 0.5f;   // return weight
        policy_weights_[i * 3 + 1] = -0.3f;  // risk weight
        policy_weights_[i * 3 + 2] = 0.2f;   // confidence weight
    }
    return true;
}

HyperAgentWrapper::Action HyperAgentWrapper::compute_action(
    const WorldState& state,
    const LuminaWrapper::Prediction& pred) const noexcept
{
    Action act{};
    act.timestamp_ns = now_ns();
    act.risk_budget  = 1.0f;

    float total_signal = 0.0f;
    for (uint32_t i = 0; i < state.n_assets && i < kMaxAssets; ++i) {
        float signal = policy_weights_[i * 3 + 0] * pred.return_forecast[i]
                     + policy_weights_[i * 3 + 1] * pred.risk_forecast[i]
                     + policy_weights_[i * 3 + 2] * pred.confidence[i];
        act.position_delta[i] = signal;
        total_signal += std::fabs(signal);
    }

    // Normalize by total signal
    if (total_signal > 1e-6f) {
        for (uint32_t i = 0; i < state.n_assets && i < kMaxAssets; ++i) {
            act.position_delta[i] /= total_signal;
        }
    }

    act.confidence = (total_signal > 0.01f) ? std::min(1.0f, total_signal) : 0.0f;
    // Determine dominant action type
    float net = 0.0f;
    for (uint32_t i = 0; i < state.n_assets && i < kMaxAssets; ++i)
        net += act.position_delta[i];
    act.action_type = (net > 0.05f) ? 1 : (net < -0.05f) ? 2 : 0;
    return act;
}

bool HyperAgentWrapper::forward(const WorldState& state, OutputBuffer& out) {
    out.reset();
    out.source_module = ModuleID::HYPER_AGENT;
    out.timestamp_ns  = now_ns();

    if (!state.lob_valid()) return true;

    // Read Lumina predictions from ShmBus
    LuminaWrapper::Prediction pred{};
    pred.timestamp_ns = now_ns();

    ReadCursor cur;
    auto& bus = ShmBus::instance();
    TensorDescriptor td{};
    const void* raw = bus.read(channels::LUMINA_PRED, cur, &td);
    if (raw) {
        std::memcpy(&pred, raw, std::min(sizeof(pred), (std::size_t)td.payload_bytes));
    }

    Action act = compute_action(state, pred);

    // Update agent weights manifest
    AgentWeightsManifest wm{};
    wm.version      = state.version;
    wm.timestamp_ns = act.timestamp_ns;
    wm.param_count  = policy_weights_.size();
    std::strncpy(wm.model_name, "HyperAgent-Linear-v1", sizeof(wm.model_name));
    GlobalStateRegistry::instance().update_weights(wm);

    Action* buf = out.allocate<Action>(1);
    *buf = act;
    out.tensor.dtype        = DType::FLOAT32;
    out.tensor.ndim         = 1;
    out.tensor.shape[0]     = kMaxAssets;
    out.tensor.payload_bytes= sizeof(Action);
    out.valid = true;
    publish_output(out);
    return true;
}

// ---------------------------------------------------------------------------
// ModuleRegistry
// ---------------------------------------------------------------------------
ModuleRegistry& ModuleRegistry::instance() {
    static ModuleRegistry reg;
    return reg;
}

void ModuleRegistry::register_module(std::unique_ptr<ModuleBase> module) {
    int id = static_cast<int>(module->id());
    modules_[id] = std::move(module);
}

ModuleBase* ModuleRegistry::get(ModuleID id) noexcept {
    auto it = modules_.find(static_cast<int>(id));
    return (it != modules_.end()) ? it->second.get() : nullptr;
}

const ModuleBase* ModuleRegistry::get(ModuleID id) const noexcept {
    auto it = modules_.find(static_cast<int>(id));
    return (it != modules_.end()) ? it->second.get() : nullptr;
}

void ModuleRegistry::topo_dfs(ModuleID id,
                               std::unordered_map<int, int>& visited,
                               std::vector<ModuleID>& order) const {
    int key = static_cast<int>(id);
    if (visited[key] == 2) return;
    if (visited[key] == 1) {
        // Cycle detected — skip
        return;
    }
    visited[key] = 1;
    const ModuleBase* m = get(id);
    if (m) {
        for (ModuleID dep : m->dependencies()) {
            topo_dfs(dep, visited, order);
        }
    }
    visited[key] = 2;
    order.push_back(id);
}

std::vector<ModuleID> ModuleRegistry::topological_order() const {
    std::unordered_map<int, int> visited;
    std::vector<ModuleID> order;
    for (const auto& [k, m] : modules_) {
        if (visited[k] == 0) {
            topo_dfs(static_cast<ModuleID>(k), visited, order);
        }
    }
    return order;
}

bool ModuleRegistry::initialize_all() {
    auto order = topological_order();
    bool ok = true;
    for (ModuleID id : order) {
        ModuleBase* m = get(id);
        if (m && m->enabled()) {
            if (!m->initialize()) {
                std::fprintf(stderr, "Module %s failed to initialize\n", m->name().c_str());
                ok = false;
            }
        }
    }
    return ok;
}

void ModuleRegistry::shutdown_all() {
    for (auto& [k, m] : modules_) {
        if (m) m->shutdown();
    }
}

int ModuleRegistry::execute_pipeline(const WorldState& state) {
    auto order = topological_order();
    int success = 0;
    for (ModuleID id : order) {
        ModuleBase* m = get(id);
        if (!m || !m->enabled()) continue;
        OutputBuffer out;
        if (m->timed_forward(state, out)) ++success;
    }
    return success;
}

std::vector<std::string> ModuleRegistry::list_modules() const {
    std::vector<std::string> names;
    for (const auto& [k, m] : modules_) {
        if (m) names.push_back(m->name());
    }
    return names;
}

} // namespace aeternus::rtel
