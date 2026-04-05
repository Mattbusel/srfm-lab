#include "realized_vol.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace srfm {

// ============================================================
// ParkinsonVol
// ============================================================

ParkinsonVol::ParkinsonVol(int period, double ann_factor) noexcept
    : period_(std::min(period, MAX_P))
    , ann_factor_(ann_factor)
    , sum_sq_(0.0)
    , count_(0)
    , vol_(0.0)
    , ring_idx_(0)
{
    std::memset(ring_, 0, sizeof(ring_));
}

double ParkinsonVol::update(double high, double low) noexcept {
    if (high < low || low <= 0.0) return vol_;

    double ln_hl = std::log(high / low);
    double val   = K1 * ln_hl * ln_hl;

    if (count_ < period_) {
        sum_sq_         += val;
        ring_[ring_idx_] = val;
        ring_idx_        = (ring_idx_ + 1) % period_;
        ++count_;
    } else {
        sum_sq_         -= ring_[ring_idx_];
        sum_sq_         += val;
        ring_[ring_idx_] = val;
        ring_idx_        = (ring_idx_ + 1) % period_;
    }

    if (is_warm()) {
        double variance = sum_sq_ / count_;
        vol_            = std::sqrt(variance * ann_factor_);
    }
    return vol_;
}

void ParkinsonVol::reset() noexcept {
    sum_sq_  = 0.0;
    count_   = 0;
    vol_     = 0.0;
    ring_idx_= 0;
    std::memset(ring_, 0, sizeof(ring_));
}

// ============================================================
// GarmanKlassVol
// ============================================================

GarmanKlassVol::GarmanKlassVol(int period, double ann_factor) noexcept
    : period_(std::min(period, MAX_P))
    , ann_factor_(ann_factor)
    , sum_sq_(0.0)
    , count_(0)
    , vol_(0.0)
    , ring_idx_(0)
{
    std::memset(ring_, 0, sizeof(ring_));
}

double GarmanKlassVol::update(double open, double high, double low,
                               double close) noexcept {
    if (open <= 0.0 || high < low || low <= 0.0) return vol_;

    double hl   = std::log(high / low);
    double co   = std::log(close / open);
    double val  = 0.5 * hl * hl - (2.0 * std::log(2.0) - 1.0) * co * co;

    if (count_ < period_) {
        sum_sq_         += val;
        ring_[ring_idx_] = val;
        ring_idx_        = (ring_idx_ + 1) % period_;
        ++count_;
    } else {
        sum_sq_         -= ring_[ring_idx_];
        sum_sq_         += val;
        ring_[ring_idx_] = val;
        ring_idx_        = (ring_idx_ + 1) % period_;
    }

    if (is_warm()) {
        double variance = std::max(0.0, sum_sq_ / count_);
        vol_            = std::sqrt(variance * ann_factor_);
    }
    return vol_;
}

double GarmanKlassVol::update(const OHLCVBar& bar) noexcept {
    return update(bar.open, bar.high, bar.low, bar.close);
}

void GarmanKlassVol::reset() noexcept {
    sum_sq_  = 0.0;
    count_   = 0;
    vol_     = 0.0;
    ring_idx_= 0;
    std::memset(ring_, 0, sizeof(ring_));
}

// ============================================================
// RogersSatchellVol
// ============================================================

RogersSatchellVol::RogersSatchellVol(int period, double ann_factor) noexcept
    : period_(std::min(period, MAX_P))
    , ann_factor_(ann_factor)
    , sum_sq_(0.0)
    , count_(0)
    , vol_(0.0)
    , ring_idx_(0)
{
    std::memset(ring_, 0, sizeof(ring_));
}

double RogersSatchellVol::update(double open, double high, double low,
                                  double close) noexcept {
    if (open <= 0.0 || high < low || low <= 0.0) return vol_;

    double val = std::log(high / close) * std::log(high / open)
               + std::log(low  / close) * std::log(low  / open);

    if (count_ < period_) {
        sum_sq_         += val;
        ring_[ring_idx_] = val;
        ring_idx_        = (ring_idx_ + 1) % period_;
        ++count_;
    } else {
        sum_sq_         -= ring_[ring_idx_];
        sum_sq_         += val;
        ring_[ring_idx_] = val;
        ring_idx_        = (ring_idx_ + 1) % period_;
    }

    if (is_warm()) {
        double variance = std::max(0.0, sum_sq_ / count_);
        vol_            = std::sqrt(variance * ann_factor_);
    }
    return vol_;
}

double RogersSatchellVol::update(const OHLCVBar& bar) noexcept {
    return update(bar.open, bar.high, bar.low, bar.close);
}

void RogersSatchellVol::reset() noexcept {
    sum_sq_  = 0.0;
    count_   = 0;
    vol_     = 0.0;
    ring_idx_= 0;
    std::memset(ring_, 0, sizeof(ring_));
}

// ============================================================
// YangZhangVol
// ============================================================

YangZhangVol::YangZhangVol(int period, double ann_factor, double k) noexcept
    : period_(std::min(period, MAX_P))
    , ann_factor_(ann_factor)
    , k_(k)
    , sum_oc_sq_(0.0)
    , sum_rs_(0.0)
    , sum_co_sq_(0.0)
    , prev_close_(0.0)
    , count_(0)
    , vol_(0.0)
    , has_prev_(false)
    , ring_idx_(0)
{
    std::memset(ring_oc_, 0, sizeof(ring_oc_));
    std::memset(ring_rs_, 0, sizeof(ring_rs_));
    std::memset(ring_co_, 0, sizeof(ring_co_));
}

double YangZhangVol::update(double open, double high, double low,
                             double close) noexcept {
    if (open <= 0.0 || high < low || low <= 0.0) return vol_;

    double oc_sq = 0.0, co_sq = 0.0;
    double rs = std::log(high / close) * std::log(high / open)
              + std::log(low  / close) * std::log(low  / open);

    if (has_prev_ && prev_close_ > 0.0) {
        double oc = std::log(open  / prev_close_);
        double co = std::log(close / open);
        oc_sq = oc * oc;
        co_sq = co * co;
    }

    if (count_ < period_) {
        sum_oc_sq_      += oc_sq;
        sum_rs_         += rs;
        sum_co_sq_      += co_sq;
        ring_oc_[ring_idx_] = oc_sq;
        ring_rs_[ring_idx_] = rs;
        ring_co_[ring_idx_] = co_sq;
        ring_idx_        = (ring_idx_ + 1) % period_;
        ++count_;
    } else {
        sum_oc_sq_ -= ring_oc_[ring_idx_]; sum_oc_sq_ += oc_sq;
        sum_rs_    -= ring_rs_[ring_idx_]; sum_rs_    += rs;
        sum_co_sq_ -= ring_co_[ring_idx_]; sum_co_sq_ += co_sq;
        ring_oc_[ring_idx_] = oc_sq;
        ring_rs_[ring_idx_] = rs;
        ring_co_[ring_idx_] = co_sq;
        ring_idx_            = (ring_idx_ + 1) % period_;
    }

    prev_close_ = close;
    has_prev_   = true;

    if (is_warm()) {
        double var_oc = sum_oc_sq_ / count_;
        double var_co = sum_co_sq_ / count_;
        double var_rs = std::max(0.0, sum_rs_ / count_);
        double variance = var_oc + k_ * var_co + (1.0 - k_) * var_rs;
        vol_ = std::sqrt(std::max(0.0, variance) * ann_factor_);
    }
    return vol_;
}

double YangZhangVol::update(const OHLCVBar& bar) noexcept {
    return update(bar.open, bar.high, bar.low, bar.close);
}

void YangZhangVol::reset() noexcept {
    sum_oc_sq_ = sum_rs_ = sum_co_sq_ = 0.0;
    prev_close_ = 0.0;
    count_     = 0;
    vol_       = 0.0;
    has_prev_  = false;
    ring_idx_  = 0;
    std::memset(ring_oc_, 0, sizeof(ring_oc_));
    std::memset(ring_rs_, 0, sizeof(ring_rs_));
    std::memset(ring_co_, 0, sizeof(ring_co_));
}

// ============================================================
// RealizedVolEstimator
// ============================================================

RealizedVolEstimator::RealizedVolEstimator(int period, double ann_factor) noexcept
    : park_(period, ann_factor)
    , gk_(period, ann_factor)
    , rs_(period, ann_factor)
    , yz_(period, ann_factor)
{
    std::memset(&last_, 0, sizeof(last_));
}

RealizedVolOutput RealizedVolEstimator::update(const OHLCVBar& bar) noexcept {
    last_.parkinson        = park_.update(bar.high, bar.low);
    last_.garman_klass     = gk_.update(bar);
    last_.rogers_satchell  = rs_.update(bar);
    last_.yang_zhang       = yz_.update(bar);
    last_.annualized_parkinson = last_.parkinson;
    last_.annualized_yz        = last_.yang_zhang;
    return last_;
}

void RealizedVolEstimator::reset() noexcept {
    park_.reset(); gk_.reset(); rs_.reset(); yz_.reset();
    std::memset(&last_, 0, sizeof(last_));
}

} // namespace srfm
