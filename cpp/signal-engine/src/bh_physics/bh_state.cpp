#include "bh_state.hpp"
#include <cmath>
#include <algorithm>

namespace srfm {

// ============================================================
// BHState implementation
// Matches Python BHState class from crypto_backtest_mc.py
// ============================================================

BHState::BHState() noexcept
    : mass_(0.0)
    , bh_dir_(0.0)
    , cf_scale_(1.0)
    , mass_ema_(0.0)
    , dir_ema_(0.0)
    , bull_fast_ema_(0.0)
    , bull_slow_ema_(0.0)
    , prev_close_(0.0)
    , prev_ts_ns_(0)
    , norm_dx_max_(1e-8)
    , norm_dt_ref_(60.0)  // 1m bar reference
    , count_(0)
    , initialized_(false)
{}

double BHState::compute_ds2(double dt_norm, double dx_norm) const noexcept {
    // Minkowski metric: ds^2 = c^2 * dt^2 - dx^2 (c=1 in normalized units)
    double ds2 = (C_NORM * C_NORM * dt_norm * dt_norm) - (dx_norm * dx_norm);
    return ds2;
}

void BHState::update_mass(double ds2, double price_return) noexcept {
    // delta_mass is the EMA of |return| weighted by timelike factor
    double delta_mass = std::abs(price_return);

    if (ds2 > 0.0) {
        // Timelike (causal): accumulate mass with EMA decay
        mass_ema_ = constants::BH_EMA_DECAY * mass_ema_
                  + (1.0 - constants::BH_EMA_DECAY) * delta_mass;
        mass_ += mass_ema_;
        // Cap mass to prevent overflow
        mass_ = std::min(mass_, 10.0);
    } else {
        // Spacelike (chaotic): collapse mass
        mass_ *= constants::BH_COLLAPSE_RATE;
    }

    // Also apply collapse when below threshold: gradual dissipation
    if (mass_ < constants::BH_MASS_THRESH) {
        mass_ *= constants::BH_COLLAPSE_RATE;
    }

    if (mass_ < 0.0) mass_ = 0.0;
}

void BHState::update_bh_dir(double log_ret) noexcept {
    // Short EMA (alpha ~0.2, period ~9) of log-return for direction
    constexpr double DIR_ALPHA = 0.2;
    dir_ema_ = DIR_ALPHA * log_ret + (1.0 - DIR_ALPHA) * dir_ema_;
    bh_dir_  = dir_ema_;
}

void BHState::update_cf_scale(double close) noexcept {
    // BullScale: ratio of fast EMA to slow EMA of close
    constexpr double FAST_ALPHA = 2.0 / (9.0  + 1.0);  // EMA(9)
    constexpr double SLOW_ALPHA = 2.0 / (21.0 + 1.0);  // EMA(21)

    if (count_ == 0) {
        bull_fast_ema_ = close;
        bull_slow_ema_ = close;
        cf_scale_      = 1.0;
        return;
    }

    bull_fast_ema_ = FAST_ALPHA * close + (1.0 - FAST_ALPHA) * bull_fast_ema_;
    bull_slow_ema_ = SLOW_ALPHA * close + (1.0 - SLOW_ALPHA) * bull_slow_ema_;

    if (bull_slow_ema_ > constants::EPSILON) {
        cf_scale_ = bull_fast_ema_ / bull_slow_ema_;
    } else {
        cf_scale_ = 1.0;
    }
}

BHOutput BHState::update(double close, double volume, int64_t timestamp_ns) noexcept {
    // Compute log-return
    double log_ret = 0.0;
    if (initialized_ && prev_close_ > constants::EPSILON) {
        log_ret = std::log(close / prev_close_);
    }

    // Compute dt (time step, normalized to 1.0 per bar)
    double dt_norm = 1.0;  // normalized: 1 bar = 1 unit of time
    if (initialized_ && prev_ts_ns_ > 0 && timestamp_ns > prev_ts_ns_) {
        double dt_sec = static_cast<double>(timestamp_ns - prev_ts_ns_) /
                        static_cast<double>(constants::NS_PER_SEC);
        dt_norm = dt_sec / norm_dt_ref_;  // normalize to 1m reference
    }

    // Compute dx (price change, normalized by rolling max)
    double dx = std::abs(log_ret);
    if (dx > norm_dx_max_) {
        norm_dx_max_ = dx;
    } else {
        // Decay the normalization reference slowly
        norm_dx_max_ *= 0.9999;
        if (norm_dx_max_ < 1e-10) norm_dx_max_ = 1e-10;
    }
    double dx_norm = (norm_dx_max_ > constants::EPSILON) ? dx / norm_dx_max_ : 0.0;

    // Minkowski metric
    double ds2 = compute_ds2(dt_norm, dx_norm);

    // Update mass
    if (initialized_) {
        update_mass(ds2, log_ret);
    }

    // Update bh_dir
    update_bh_dir(log_ret);

    // Update BullScale
    update_cf_scale(close);

    // Save state
    prev_close_  = close;
    prev_ts_ns_  = timestamp_ns;
    initialized_ = true;
    ++count_;

    int direction = 0;
    if (bh_dir_ > 0.001)       direction =  1;
    else if (bh_dir_ < -0.001) direction = -1;

    return { mass_, bh_dir_, cf_scale_, active(), direction };
}

BHOutput BHState::update(const OHLCVBar& bar) noexcept {
    return update(bar.close, bar.volume, bar.timestamp_ns);
}

void BHState::reset() noexcept {
    mass_          = 0.0;
    bh_dir_        = 0.0;
    cf_scale_      = 1.0;
    mass_ema_      = 0.0;
    dir_ema_       = 0.0;
    bull_fast_ema_ = 0.0;
    bull_slow_ema_ = 0.0;
    prev_close_    = 0.0;
    prev_ts_ns_    = 0;
    norm_dx_max_   = 1e-8;
    count_         = 0;
    initialized_   = false;
}

// ============================================================
// Batch BH computation over historical bars
// ============================================================

void bh_batch(const OHLCVBar* bars, std::size_t n, BHOutput* out) noexcept {
    BHState bh;
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = bh.update(bars[i]);
    }
}

// ============================================================
// BH signal interpretation
// ============================================================

/// Returns the signal strength: mass * bh_dir, scaled by cf_scale.
double bh_signal_strength(const BHOutput& bh) noexcept {
    if (!bh.bh_active) return 0.0;
    return bh.mass * bh.bh_dir * bh.cf_scale;
}

/// Returns true if the BH is forming (mass approaching threshold).
bool bh_forming(const BHOutput& bh, double approach_pct = 0.8) noexcept {
    return !bh.bh_active &&
           bh.mass >= constants::BH_MASS_THRESH * approach_pct;
}

/// Returns true if the BH just collapsed (mass dropped below threshold).
bool bh_collapsed(const BHOutput& prev, const BHOutput& curr) noexcept {
    return prev.bh_active && !curr.bh_active;
}

/// Returns true if the BH just formed.
bool bh_formed(const BHOutput& prev, const BHOutput& curr) noexcept {
    return !prev.bh_active && curr.bh_active;
}

} // namespace srfm
