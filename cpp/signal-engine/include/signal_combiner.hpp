#pragma once
// signal_combiner.hpp -- Multi-signal combination with dynamic IC-based weighting.
// Supports four combination methods:
//   EQUAL_WEIGHT  -- simple average
//   IC_WEIGHT     -- weight proportional to rolling IC estimate
//   RANK_WEIGHT   -- Spearman rank combination
//   HEDGE_ALGORITHM -- multiplicative exponential hedge (Hedge algorithm / AdaHedge)
//
// Online IC update uses an exponential moving average of realized IC.
// portfolio_signal() maps SRFM-specific inputs to a final [-1,1] signal.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace srfm {
namespace combining {

// ============================================================
// Enumerations
// ============================================================

enum class CombineMethod : uint8_t {
    EQUAL_WEIGHT    = 0,
    IC_WEIGHT       = 1,
    RANK_WEIGHT     = 2,
    HEDGE_ALGORITHM = 3,
};

// ============================================================
// SignalInput -- per-signal descriptor passed to combine()
// ============================================================

struct SignalInput {
    std::string name;
    double      value;        // raw signal value (any range, combiner normalizes)
    double      weight;       // user-supplied static weight (used by EQUAL/IC weighting)
    double      ic_estimate;  // caller may provide own IC; 0 => use internally tracked IC
};

// ============================================================
// Per-signal state tracked internally
// ============================================================

struct SignalState {
    std::string name;
    double ic_ema;          // exponential moving average of |IC|
    double ic_ema_raw;      // signed IC EMA (for directional weighting)
    double hedge_weight;    // multiplicative hedge weight (unnormalized)
    double sum_signal;      // running sum for IC computation
    double sum_return;      // running sum of realized returns paired with signals
    int    count;           // number of IC updates seen

    static constexpr double IC_DECAY  = 0.95;   // EMA decay for IC
    static constexpr double HEDGE_ETA = 0.05;   // learning rate for hedge algorithm

    SignalState() noexcept
        : ic_ema(0.5), ic_ema_raw(0.5), hedge_weight(1.0),
          sum_signal(0.0), sum_return(0.0), count(0) {}

    explicit SignalState(std::string n) noexcept
        : name(std::move(n)), ic_ema(0.5), ic_ema_raw(0.5),
          hedge_weight(1.0), sum_signal(0.0), sum_return(0.0), count(0) {}
};

// ============================================================
// SignalCombiner
// ============================================================

class SignalCombiner {
public:
    // default_ic_decay -- EMA factor for online IC update.
    double ic_decay      = 0.95;
    // hedge_eta -- learning rate for multiplicative hedge update.
    double hedge_eta     = 0.05;
    // max_ic_weight_ratio -- max ratio any single IC weight can take.
    double max_ic_ratio  = 4.0;
    // min_weight_floor -- prevent any signal from being zeroed entirely.
    double min_weight_floor = 0.01;

    SignalCombiner() = default;

    // combine -- combine signals with the specified method.
    // Returns value in [-1, 1] (output is clipped).
    double combine(const std::vector<SignalInput>& signals, CombineMethod method) {
        if (signals.empty()) return 0.0;

        // Ensure state entries exist for all signals
        for (const auto& s : signals) {
            ensure_state(s.name);
        }

        double result = 0.0;
        switch (method) {
            case CombineMethod::EQUAL_WEIGHT:
                result = combine_equal(signals);
                break;
            case CombineMethod::IC_WEIGHT:
                result = combine_ic(signals);
                break;
            case CombineMethod::RANK_WEIGHT:
                result = combine_rank(signals);
                break;
            case CombineMethod::HEDGE_ALGORITHM:
                result = combine_hedge(signals);
                break;
        }
        return std::clamp(result, -1.0, 1.0);
    }

    // update_ic -- online update of IC estimate for a named signal.
    // realized_return: the actual return observed after the signal was emitted.
    // signal_value: the raw signal value that was emitted.
    void update_ic(const std::string& name,
                   double realized_return,
                   double signal_value) noexcept
    {
        auto it = states_.find(name);
        if (it == states_.end()) {
            states_.emplace(name, SignalState(name));
            it = states_.find(name);
        }
        SignalState& st = it->second;

        // Rank-biserial IC approximation: sign(signal) * sign(return)
        // This is a simplified IC that avoids needing cross-sectional data.
        double sign_s = (signal_value > 0.0) ? 1.0 : ((signal_value < 0.0) ? -1.0 : 0.0);
        double sign_r = (realized_return > 0.0) ? 1.0 : ((realized_return < 0.0) ? -1.0 : 0.0);
        const double ic_sample = sign_s * sign_r;  // in {-1, 0, 1}

        ++st.count;
        const double alpha = ic_decay;
        st.ic_ema_raw = alpha * st.ic_ema_raw + (1.0 - alpha) * ic_sample;
        st.ic_ema     = alpha * st.ic_ema     + (1.0 - alpha) * std::abs(ic_sample);

        // Hedge weight update: exponential multiplicative rule.
        // If ic_sample > 0 (signal predicted correctly) => increase weight.
        const double gain = std::exp(hedge_eta * ic_sample);
        st.hedge_weight *= gain;
        // Clip to prevent explosion
        st.hedge_weight = std::clamp(st.hedge_weight, 1e-6, 1e6);
    }

    // portfolio_signal -- SRFM-specific combination of 5 core signals.
    // bh_mass: black-hole mass signal [-1,1] or [0,1]
    // nav_curv: navigation curvature signal (geodesic deviation)
    // hurst: Hurst exponent [0,1] -- transforms to trend bias
    // garch_vol: normalized GARCH volatility (annualized)
    // pattern: CandlePattern composite signal [-1,1]
    //
    // Returns combined signal in [-1, 1].
    double portfolio_signal(
        double bh_mass,
        double nav_curv,
        double hurst,
        double garch_vol,
        double pattern) const noexcept
    {
        // Map hurst to directional signal:
        //   H > 0.5 => momentum bias (follow direction of bh_mass)
        //   H < 0.5 => mean-reversion bias (oppose direction)
        const double hurst_bias = 2.0 * (hurst - 0.5);  // [-1, 1]

        // Volatility scaling: high vol => reduce position size
        // garch_vol is annualized; normalize to [0,1] assuming 0..100% range.
        const double vol_scale = std::exp(-std::max(0.0, garch_vol) * 3.0);  // ~1 at low vol

        // Nav curvature: high curvature = regime change in progress => reduce confidence
        const double curv_damp = std::exp(-std::abs(nav_curv) * 2.0);

        // BH mass: primary directional signal.
        // Map from [0,1] space to [-1,1]: signal = 2*bh_mass - 1
        const double bh_dir = 2.0 * std::clamp(bh_mass, 0.0, 1.0) - 1.0;

        // Weighted sum:
        //   40% BH direction
        //   25% pattern
        //   20% hurst_bias (modulates BH)
        //   15% nav curvature-damped carry-through
        double raw = 0.40 * bh_dir
                   + 0.25 * pattern
                   + 0.20 * hurst_bias * std::abs(bh_dir)  // hurst amplifies bh
                   + 0.15 * bh_dir * curv_damp;            // curv damps near rotations

        // Apply vol scaling
        raw *= vol_scale;

        // Final tanh normalization
        return std::tanh(raw * 1.5);
    }

    // get_ic -- retrieve current IC estimate for a named signal.
    // Returns 0.5 if unknown (neutral IC).
    double get_ic(const std::string& name) const noexcept {
        auto it = states_.find(name);
        if (it == states_.end()) return 0.5;
        return it->second.ic_ema;
    }

    double get_ic_signed(const std::string& name) const noexcept {
        auto it = states_.find(name);
        if (it == states_.end()) return 0.5;
        return it->second.ic_ema_raw;
    }

    double get_hedge_weight(const std::string& name) const noexcept {
        auto it = states_.find(name);
        if (it == states_.end()) return 1.0;
        return it->second.hedge_weight;
    }

    // reset -- clear all stored state.
    void reset() noexcept {
        states_.clear();
    }

private:
    std::unordered_map<std::string, SignalState> states_;

    void ensure_state(const std::string& name) {
        if (states_.find(name) == states_.end()) {
            states_.emplace(name, SignalState(name));
        }
    }

    // EQUAL_WEIGHT: simple weighted average (weight from SignalInput.weight).
    double combine_equal(const std::vector<SignalInput>& sigs) const noexcept {
        double wsum = 0.0, total_w = 0.0;
        for (const auto& s : sigs) {
            const double w = std::max(min_weight_floor, s.weight);
            wsum    += w * s.value;
            total_w += w;
        }
        return (total_w > 1e-15) ? wsum / total_w : 0.0;
    }

    // IC_WEIGHT: weight by IC estimate.
    double combine_ic(const std::vector<SignalInput>& sigs) noexcept {
        // Gather IC weights
        std::vector<double> ic_weights;
        ic_weights.reserve(sigs.size());
        for (const auto& s : sigs) {
            double ic = s.ic_estimate;
            if (ic == 0.0) {
                auto it = states_.find(s.name);
                ic = (it != states_.end()) ? it->second.ic_ema : 0.5;
            }
            ic_weights.push_back(std::max(min_weight_floor, ic));
        }

        // Cap max ratio
        const double max_w = *std::max_element(ic_weights.begin(), ic_weights.end());
        const double min_allowed = max_w / max_ic_ratio;
        for (auto& w : ic_weights) w = std::max(w, min_allowed);

        // Normalize and combine
        const double total_w = std::accumulate(ic_weights.begin(), ic_weights.end(), 0.0);
        double result = 0.0;
        for (int i = 0; i < static_cast<int>(sigs.size()); ++i) {
            result += (ic_weights[i] / total_w) * sigs[i].value;
        }
        return result;
    }

    // RANK_WEIGHT: rank-normalize each signal, then average ranks.
    // Each signal's value is converted to a rank in [0,1] then averaged.
    double combine_rank(const std::vector<SignalInput>& sigs) const noexcept {
        if (sigs.size() == 1) return sigs[0].value;

        // For multi-signal: use signal magnitudes to build a rank-weighted combination.
        // Rank the abs(values) and weight by rank (higher absolute value => more weight).
        const int n = static_cast<int>(sigs.size());

        // Create index array sorted by abs(value) descending
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return std::abs(sigs[a].value) > std::abs(sigs[b].value);
        });

        // Assign ranks (1 = highest abs value), then compute rank weights
        // Rank weight = (n - rank + 1) / sum(1..n) to favor higher-confidence signals
        double denom = static_cast<double>(n * (n + 1)) / 2.0;
        double result = 0.0;
        for (int r = 0; r < n; ++r) {
            double rank_weight = static_cast<double>(n - r) / denom;
            result += rank_weight * sigs[idx[r]].value;
        }
        return result;
    }

    // HEDGE_ALGORITHM: exponential weights from online multiplicative update.
    double combine_hedge(const std::vector<SignalInput>& sigs) noexcept {
        double wsum = 0.0, total_w = 0.0;
        for (const auto& s : sigs) {
            auto it = states_.find(s.name);
            const double hw = (it != states_.end())
                ? it->second.hedge_weight
                : 1.0;
            wsum    += hw * s.value;
            total_w += hw;
        }
        return (total_w > 1e-15) ? wsum / total_w : 0.0;
    }
};

} // namespace combining
} // namespace srfm
