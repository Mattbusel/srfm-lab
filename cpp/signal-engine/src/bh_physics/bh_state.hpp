#pragma once
#include "srfm/types.hpp"

namespace srfm {

/// Output from the BH state machine for a single bar.
struct BHOutput {
    double mass;        // accumulated BH mass
    double bh_dir;      // directional bias from price momentum
    double cf_scale;    // BullScale: rolling EMA ratio (price scaling)
    bool   bh_active;   // true when mass >= BH_MASS_THRESH
    int    direction;   // +1 bull, -1 bear, 0 neutral
};

/// Black Hole physics signal.
///
/// Minkowski spacetime metric: ds^2 = c^2*dt^2 - dx^2
///   where dt = normalized time interval, dx = normalized price change.
///   If ds^2 > 0: timelike (causal/ordered); if < 0: spacelike (chaotic).
///
/// Mass accumulation:
///   - On each bar, compute ds^2.
///   - If timelike: mass += delta_mass * EMA_DECAY
///   - Otherwise:   mass *= COLLAPSE_RATE
///   - BH forms when mass >= BH_MASS_THRESH.
///
/// bh_dir is derived from a short EMA of log-returns.
/// cf_scale is the ratio of fast BullScale EMA to slow BullScale EMA.
class BHState {
public:
    BHState() noexcept;

    /// Update with a new bar. Returns current BH state.
    BHOutput update(const OHLCVBar& bar) noexcept;

    /// Update with explicit close price and volume.
    BHOutput update(double close, double volume, int64_t timestamp_ns) noexcept;

    double mass()     const noexcept { return mass_; }
    bool   active()   const noexcept { return mass_ >= constants::BH_MASS_THRESH; }
    double bh_dir()   const noexcept { return bh_dir_; }
    double cf_scale() const noexcept { return cf_scale_; }
    int    bar_count()const noexcept { return count_; }

    void reset() noexcept;

private:
    // Spacetime metric computation
    double compute_ds2(double dt_norm, double dx_norm) const noexcept;

    // Mass accumulation step
    void   update_mass(double ds2, double price_return) noexcept;

    // bh_dir update from price momentum EMA
    void   update_bh_dir(double log_ret) noexcept;

    // cf_scale (BullScale) update
    void   update_cf_scale(double close) noexcept;

    // State
    double mass_;
    double bh_dir_;
    double cf_scale_;

    // EMA for mass accumulation (decay = BH_EMA_DECAY)
    double mass_ema_;

    // Short EMA of log-return for bh_dir
    double dir_ema_;

    // BullScale: two EMAs on close price
    double bull_fast_ema_;   // fast EMA (period ~9)
    double bull_slow_ema_;   // slow EMA (period ~21)

    // Previous values
    double prev_close_;
    int64_t prev_ts_ns_;

    // Normalization reference (rolling max of |dx|)
    double norm_dx_max_;
    double norm_dt_ref_;   // reference dt (e.g. 60s for 1m bars)

    int count_;
    bool initialized_;

    // Speed of light reference for ds^2 computation
    // We use c = 1.0 in normalized units (so dt is in units where c*dt = 1 per bar)
    static constexpr double C_NORM = 1.0;
};

} // namespace srfm
