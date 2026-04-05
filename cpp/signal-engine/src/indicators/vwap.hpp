#pragma once
#include "srfm/types.hpp"

namespace srfm {

/// VWAP: Volume-Weighted Average Price, reset on each trading session.
/// Session detection: midnight UTC boundary (or custom interval).
class VWAP {
public:
    /// session_ns: session length in nanoseconds. Default = 1 day (86400s).
    explicit VWAP(int64_t session_ns = constants::NS_PER_DAY) noexcept;

    /// Update with a bar. Returns current VWAP for this session.
    double update(const OHLCVBar& bar) noexcept;

    /// Update with explicit tp (typical price), volume, and timestamp.
    double update(double typical_price, double volume, int64_t timestamp_ns) noexcept;

    double value()         const noexcept { return vwap_; }
    double session_volume()const noexcept { return cum_volume_; }
    int64_t session_start()const noexcept { return session_start_ns_; }
    int    bar_count()     const noexcept { return bar_count_; }
    bool   is_warm()       const noexcept { return bar_count_ >= 2; }

    /// Returns deviation of close from VWAP as a fraction.
    double deviation(double close) const noexcept;

    void reset_session(int64_t new_session_start) noexcept;
    void reset() noexcept;

private:
    bool is_new_session(int64_t timestamp_ns) const noexcept;

    int64_t session_ns_;
    int64_t session_start_ns_;
    double  cum_tp_vol_;    // sum(typical_price * volume)
    double  cum_volume_;
    double  vwap_;
    int     bar_count_;
};

// ============================================================
// Anchored VWAP: starts from a specific anchor time (e.g. session open, event)
// ============================================================

class AnchoredVWAP {
public:
    explicit AnchoredVWAP(int64_t anchor_ts_ns) noexcept;

    double update(double typical_price, double volume, int64_t timestamp_ns) noexcept;
    double update(const OHLCVBar& bar) noexcept;

    double value()  const noexcept { return vwap_; }
    bool   active() const noexcept { return active_; }

private:
    int64_t anchor_ts_ns_;
    double  cum_tp_vol_;
    double  cum_volume_;
    double  vwap_;
    bool    active_;
};

} // namespace srfm
