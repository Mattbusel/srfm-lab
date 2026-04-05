#include "vwap.hpp"
#include <cmath>

namespace srfm {

// ============================================================
// VWAP
// ============================================================

VWAP::VWAP(int64_t session_ns) noexcept
    : session_ns_(session_ns)
    , session_start_ns_(0)
    , cum_tp_vol_(0.0)
    , cum_volume_(0.0)
    , vwap_(0.0)
    , bar_count_(0)
{}

bool VWAP::is_new_session(int64_t timestamp_ns) const noexcept {
    if (session_start_ns_ == 0) return true;
    return (timestamp_ns - session_start_ns_) >= session_ns_;
}

void VWAP::reset_session(int64_t new_session_start) noexcept {
    session_start_ns_ = new_session_start;
    cum_tp_vol_       = 0.0;
    cum_volume_       = 0.0;
    vwap_             = 0.0;
    bar_count_        = 0;
}

void VWAP::reset() noexcept {
    reset_session(0);
}

double VWAP::update(double typical_price, double volume, int64_t timestamp_ns) noexcept {
    if (is_new_session(timestamp_ns)) {
        // Align session start to nearest session boundary
        int64_t boundary = (timestamp_ns / session_ns_) * session_ns_;
        reset_session(boundary);
    }

    if (volume > 0.0) {
        cum_tp_vol_ += typical_price * volume;
        cum_volume_ += volume;
    }

    vwap_ = (cum_volume_ > constants::EPSILON) ? cum_tp_vol_ / cum_volume_ : typical_price;
    ++bar_count_;
    return vwap_;
}

double VWAP::update(const OHLCVBar& bar) noexcept {
    double tp = (bar.high + bar.low + bar.close) / 3.0;
    return update(tp, bar.volume, bar.timestamp_ns);
}

double VWAP::deviation(double close) const noexcept {
    if (vwap_ < constants::EPSILON) return 0.0;
    return (close - vwap_) / vwap_;
}

// ============================================================
// AnchoredVWAP
// ============================================================

AnchoredVWAP::AnchoredVWAP(int64_t anchor_ts_ns) noexcept
    : anchor_ts_ns_(anchor_ts_ns)
    , cum_tp_vol_(0.0)
    , cum_volume_(0.0)
    , vwap_(0.0)
    , active_(false)
{}

double AnchoredVWAP::update(double typical_price, double volume,
                             int64_t timestamp_ns) noexcept {
    if (timestamp_ns < anchor_ts_ns_) return vwap_;

    active_ = true;
    if (volume > 0.0) {
        cum_tp_vol_ += typical_price * volume;
        cum_volume_ += volume;
    }
    vwap_ = (cum_volume_ > constants::EPSILON) ? cum_tp_vol_ / cum_volume_ : typical_price;
    return vwap_;
}

double AnchoredVWAP::update(const OHLCVBar& bar) noexcept {
    double tp = (bar.high + bar.low + bar.close) / 3.0;
    return update(tp, bar.volume, bar.timestamp_ns);
}

// ============================================================
// Standalone batch VWAP
// ============================================================

void vwap_batch(const OHLCVBar* bars, std::size_t n,
                int64_t session_ns, double* out_vwap) noexcept {
    VWAP vwap(session_ns);
    for (std::size_t i = 0; i < n; ++i) {
        out_vwap[i] = vwap.update(bars[i]);
    }
}

// ============================================================
// VWAP standard deviation bands
// ============================================================

struct VWAPBands {
    double vwap;
    double upper1;  // +1 std dev
    double lower1;  // -1 std dev
    double upper2;  // +2 std dev
    double lower2;  // -2 std dev
};

/// Compute VWAP bands given cumulative data.
VWAPBands vwap_bands(double cum_tp_vol, double cum_tp2_vol,
                      double cum_volume, double vwap) noexcept {
    VWAPBands bands{};
    bands.vwap = vwap;
    if (cum_volume < constants::EPSILON) return bands;

    double variance = (cum_tp2_vol / cum_volume) - (vwap * vwap);
    double std_dev  = (variance > 0.0) ? std::sqrt(variance) : 0.0;

    bands.upper1 = vwap + 1.0 * std_dev;
    bands.lower1 = vwap - 1.0 * std_dev;
    bands.upper2 = vwap + 2.0 * std_dev;
    bands.lower2 = vwap - 2.0 * std_dev;
    return bands;
}

} // namespace srfm
