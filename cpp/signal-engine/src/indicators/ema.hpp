#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include "srfm/types.hpp"
#include "srfm/simd_math.hpp"

namespace srfm {

// ============================================================
// Single-instrument EMA
// ============================================================

class EMA {
public:
    /// alpha = 2/(period+1) by default. Custom alpha also supported.
    explicit EMA(int period) noexcept;
    explicit EMA(double alpha) noexcept;

    /// Update with a new price. Returns current EMA value.
    double update(double price) noexcept;

    /// Returns current EMA value without updating.
    double value() const noexcept { return value_; }

    /// True once warmup (period bars) has been observed.
    bool is_warm() const noexcept { return warm_; }

    /// Number of bars seen so far.
    int bar_count() const noexcept { return count_; }

    /// Reset state.
    void reset() noexcept;

    double alpha() const noexcept { return alpha_; }

private:
    double alpha_;
    double value_;
    int    period_;
    int    count_;
    bool   warm_;
};

// ============================================================
// Cross of two EMAs (fast/slow)
// ============================================================

struct EMACrossSignal {
    double fast_ema;
    double slow_ema;
    double diff;         // fast - slow
    int    cross;        // +1 = bullish cross, -1 = bearish cross, 0 = no cross
};

class EMACross {
public:
    EMACross(int fast_period, int slow_period) noexcept;

    EMACrossSignal update(double price) noexcept;

    bool is_warm() const noexcept { return fast_.is_warm() && slow_.is_warm(); }
    void reset() noexcept;

private:
    EMA  fast_;
    EMA  slow_;
    int  prev_sign_; // sign of last diff
};

// ============================================================
// Batch EMA: update N instruments with the same alpha
// ============================================================

template<std::size_t N>
class BatchEMA {
public:
    explicit BatchEMA(double alpha) noexcept : alpha_(alpha) {
        values_.fill(0.0);
        counts_.fill(0);
    }

    /// Update all N instruments simultaneously using SIMD.
    void update(const double* prices) noexcept {
        simd::ema_update_batch(prices, values_.data(), N, alpha_);
        for (std::size_t i = 0; i < N; ++i) ++counts_[i];
    }

    const double* values() const noexcept { return values_.data(); }
    double value(std::size_t i) const noexcept { return values_[i]; }

    void set_initial(std::size_t i, double v) noexcept {
        values_[i] = v;
        counts_[i] = 0;
    }

    static constexpr std::size_t size() noexcept { return N; }

private:
    double alpha_;
    std::array<double, N> values_;
    std::array<int, N>    counts_;
};

} // namespace srfm
