#pragma once
#include "srfm/types.hpp"

namespace srfm {

struct RealizedVolOutput {
    double parkinson;        // Parkinson estimator
    double garman_klass;     // Garman-Klass estimator
    double rogers_satchell;  // Rogers-Satchell estimator
    double yang_zhang;       // Yang-Zhang estimator (handles overnight gaps)
    double annualized_parkinson;
    double annualized_yz;
};

/// Parkinson estimator: uses high-low range only.
/// Assumes geometric Brownian motion, unbiased for continuous path.
class ParkinsonVol {
public:
    explicit ParkinsonVol(int period = 20, double ann_factor = 252.0) noexcept;
    double update(double high, double low) noexcept;
    double value()  const noexcept { return vol_; }
    bool   is_warm()const noexcept { return count_ >= period_; }
    void   reset()  noexcept;

private:
    static constexpr double K1 = 1.0 / (4.0 * 0.693147180559945); // 1/(4*ln2)
    int    period_;
    double ann_factor_;
    double sum_sq_;
    int    count_;
    double vol_;

    static constexpr int MAX_P = 500;
    double ring_[MAX_P];
    int    ring_idx_;
};

/// Garman-Klass estimator: uses OHLC.
/// More efficient than close-to-close (handles intraday variation).
class GarmanKlassVol {
public:
    explicit GarmanKlassVol(int period = 20, double ann_factor = 252.0) noexcept;
    double update(double open, double high, double low, double close) noexcept;
    double update(const OHLCVBar& bar) noexcept;
    double value()  const noexcept { return vol_; }
    bool   is_warm()const noexcept { return count_ >= period_; }
    void   reset()  noexcept;

private:
    int    period_;
    double ann_factor_;
    double sum_sq_;
    int    count_;
    double vol_;

    static constexpr int MAX_P = 500;
    double ring_[MAX_P];
    int    ring_idx_;
};

/// Rogers-Satchell estimator: unbiased for trending markets.
class RogersSatchellVol {
public:
    explicit RogersSatchellVol(int period = 20, double ann_factor = 252.0) noexcept;
    double update(double open, double high, double low, double close) noexcept;
    double update(const OHLCVBar& bar) noexcept;
    double value()  const noexcept { return vol_; }
    bool   is_warm()const noexcept { return count_ >= period_; }
    void   reset()  noexcept;

private:
    int    period_;
    double ann_factor_;
    double sum_sq_;
    int    count_;
    double vol_;

    static constexpr int MAX_P = 500;
    double ring_[MAX_P];
    int    ring_idx_;
};

/// Yang-Zhang estimator: handles overnight jumps; combines open-to-close and
/// Garman-Klass with Rogers-Satchell correction.
class YangZhangVol {
public:
    explicit YangZhangVol(int period = 20, double ann_factor = 252.0,
                          double k = 0.34) noexcept;
    double update(double open, double high, double low, double close) noexcept;
    double update(const OHLCVBar& bar) noexcept;
    double value()  const noexcept { return vol_; }
    bool   is_warm()const noexcept { return count_ > period_; }
    void   reset()  noexcept;

private:
    int    period_;
    double ann_factor_;
    double k_;           // weighting constant (0.34 by default)
    double sum_oc_sq_;   // open-to-close squared
    double sum_rs_;      // Rogers-Satchell
    double sum_co_sq_;   // close-to-open (overnight) squared
    double prev_close_;
    int    count_;
    double vol_;
    bool   has_prev_;

    static constexpr int MAX_P = 500;
    double ring_oc_[MAX_P];
    double ring_rs_[MAX_P];
    double ring_co_[MAX_P];
    int    ring_idx_;
};

/// Aggregates all four estimators in one pass.
class RealizedVolEstimator {
public:
    explicit RealizedVolEstimator(int period = 20, double ann_factor = 252.0) noexcept;
    RealizedVolOutput update(const OHLCVBar& bar) noexcept;
    const RealizedVolOutput& last() const noexcept { return last_; }
    void reset() noexcept;

private:
    ParkinsonVol     park_;
    GarmanKlassVol   gk_;
    RogersSatchellVol rs_;
    YangZhangVol     yz_;
    RealizedVolOutput last_;
};

} // namespace srfm
