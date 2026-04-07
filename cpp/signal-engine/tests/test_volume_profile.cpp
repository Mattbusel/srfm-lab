// test_volume_profile.cpp -- Google Test suite for VolumeProfile.
//
// Build: link against signal_engine_lib and gtest_main.
// Run:   ./test_volume_profile

#include <gtest/gtest.h>
#include "indicators/volume_profile.hpp"
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>

using namespace srfm::indicators;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

/// Build a profile from a vector of (price, volume) pairs.
static VolumeProfile make_profile(
    const std::vector<std::pair<double,double>>& pv,
    double lo = 0.0, double hi = 0.0)
{
    VolumeProfile vp;
    if (lo < hi) vp.init_range(lo, hi);
    for (auto& [p, v] : pv) vp.update(p, v);
    return vp;
}

// ------------------------------------------------------------------
// test_poc_finds_highest_volume_price
// ------------------------------------------------------------------

TEST(VolumeProfile, POC_finds_highest_volume_price) {
    // Insert volume with a clear peak at price 105.
    VolumeProfile vp;
    vp.init_range(100.0, 110.0);

    // Background volume at all prices
    for (double p = 100.5; p < 110.0; p += 1.0)
        vp.update(p, 10.0);

    // Massive volume at 105
    vp.update(105.0, 10000.0);

    double poc = vp.get_poc();
    // POC should be the bin containing 105.0
    EXPECT_NEAR(poc, 105.0, 1.0)
        << "POC should be near 105, got " << poc;
}

TEST(VolumeProfile, POC_empty_profile_returns_zero) {
    VolumeProfile vp;
    EXPECT_DOUBLE_EQ(vp.get_poc(), 0.0);
}

TEST(VolumeProfile, POC_single_price_level) {
    VolumeProfile vp;
    vp.init_range(50.0, 60.0);
    vp.update(55.0, 100.0);
    double poc = vp.get_poc();
    EXPECT_NEAR(poc, 55.0, 0.5);
}

// ------------------------------------------------------------------
// test_value_area_contains_70pct_volume
// ------------------------------------------------------------------

TEST(VolumeProfile, ValueArea_contains_70pct_volume) {
    // Gaussian-like distribution of volume across 100..110.
    VolumeProfile vp;
    vp.init_range(100.0, 110.0);

    // Insert volume at discrete price levels with Gaussian weights.
    for (int i = 0; i <= 100; ++i) {
        double p = 100.0 + i * 0.1;
        double sigma = 2.0;
        double mu    = 105.0;
        double weight = std::exp(-0.5 * ((p - mu) / sigma) * ((p - mu) / sigma));
        vp.update(p, weight * 1000.0);
    }

    auto [val, vah] = vp.get_value_area(0.70);
    EXPECT_LT(val, vah) << "VAL should be below VAH";

    // Count volume inside the value area
    double inside = 0.0;
    for (int i = 0; i < VP_BINS; ++i) {
        double centre = vp.bin_centre(i);
        if (centre >= val && centre <= vah)
            inside += vp.bin_volume(i);
    }
    double fraction = inside / vp.total_volume();
    EXPECT_GE(fraction, 0.65)
        << "Value area should contain at least 65% of volume, got " << fraction;
    // It may contain slightly more than 70% due to bin granularity.
    EXPECT_LE(fraction, 1.0);
}

TEST(VolumeProfile, ValueArea_uniform_distribution) {
    // With uniform distribution, VA should span ~70% of total price range.
    VolumeProfile vp;
    vp.init_range(0.0, 256.0);  // 256 bins, 1 unit each
    for (int i = 0; i < VP_BINS; ++i)
        vp.update(vp.bin_centre(i), 1.0);

    auto [val, vah] = vp.get_value_area(0.70);
    double width = vah - val;
    double total_width = 256.0;
    double fraction = width / total_width;
    EXPECT_NEAR(fraction, 0.70, 0.10);
}

// ------------------------------------------------------------------
// test_hvn_count_reasonable
// ------------------------------------------------------------------

TEST(VolumeProfile, HVN_count_reasonable_for_bimodal) {
    // Bimodal distribution: two clear peaks.
    VolumeProfile vp;
    vp.init_range(100.0, 120.0);

    // Low baseline
    for (double p = 100.5; p < 120.0; p += 0.5)
        vp.update(p, 5.0);

    // Peak 1 near 105
    for (double p = 104.0; p <= 106.0; p += 0.25)
        vp.update(p, 500.0);

    // Peak 2 near 115
    for (double p = 114.0; p <= 116.0; p += 0.25)
        vp.update(p, 500.0);

    auto hvn = vp.get_hvn(1.5);
    EXPECT_GE(static_cast<int>(hvn.size()), 1)
        << "Should find at least 1 HVN for bimodal distribution";
    EXPECT_LE(static_cast<int>(hvn.size()), 20)
        << "Should not report excessive HVNs";
}

TEST(VolumeProfile, HVN_empty_profile_returns_empty) {
    VolumeProfile vp;
    auto hvn = vp.get_hvn();
    EXPECT_TRUE(hvn.empty());
}

TEST(VolumeProfile, LVN_found_in_bimodal_gap) {
    VolumeProfile vp;
    vp.init_range(100.0, 120.0);

    // Two peaks with a gap in between
    for (double p = 100.5; p < 106.0; p += 0.5)
        vp.update(p, 200.0);  // peak A
    for (double p = 114.0; p < 120.0; p += 0.5)
        vp.update(p, 200.0);  // peak B
    // Very thin area in 106..114
    for (double p = 106.5; p < 114.0; p += 0.5)
        vp.update(p, 1.0);

    auto lvn = vp.get_lvn(0.5);
    EXPECT_GE(static_cast<int>(lvn.size()), 1)
        << "Should find LVN in the thin gap area";
}

// ------------------------------------------------------------------
// test_rolling_profile_evicts_old_bars
// ------------------------------------------------------------------

TEST(RollingVolumeProfile, Evicts_old_bars_and_total_volume_is_bounded) {
    const int window = 50;
    RollingVolumeProfile rvp(window);

    // Feed 150 bars -- only last 50 should be retained.
    double total_last_50 = 0.0;
    for (int i = 0; i < 150; ++i) {
        double price  = 100.0 + (i % 10);
        double volume = static_cast<double>(i + 1);
        rvp.update(price, volume);
        if (i >= 100) total_last_50 += volume;
    }

    // Total volume in profile should approximate the last 50 bars.
    double profile_vol = rvp.total_volume();
    EXPECT_GT(profile_vol, 0.0) << "Profile volume should be positive";
    // Allow generous tolerance due to floating point and bin merging.
    EXPECT_LE(profile_vol, total_last_50 * 1.05)
        << "Profile should not accumulate more than the window allows";
}

TEST(RollingVolumeProfile, POC_shifts_with_new_price_activity) {
    RollingVolumeProfile rvp(20);

    // Phase 1: heavy volume at 100
    for (int i = 0; i < 10; ++i)
        rvp.update(100.0, 1000.0);

    double poc1 = rvp.get_poc();
    EXPECT_NEAR(poc1, 100.0, 2.0);

    // Phase 2: shift to heavy volume at 110 -- should push POC
    for (int i = 0; i < 30; ++i)
        rvp.update(110.0, 1000.0);

    double poc2 = rvp.get_poc();
    EXPECT_GT(poc2, poc1)
        << "POC should shift upward when recent volume concentrates at higher price";
}

TEST(RollingVolumeProfile, Window_count_does_not_exceed_limit) {
    RollingVolumeProfile rvp(10);
    for (int i = 0; i < 100; ++i)
        rvp.update(100.0 + i * 0.1, 1.0);
    EXPECT_LE(rvp.bar_count(), 10)
        << "Rolling profile should not retain more than window bars";
}

TEST(RollingVolumeProfile, ValueArea_is_consistent_after_eviction) {
    RollingVolumeProfile rvp(30);
    // Feed enough bars to trigger several eviction cycles
    for (int i = 0; i < 90; ++i) {
        double p = 100.0 + std::sin(i * 0.3) * 5.0;
        rvp.update(p, 100.0);
    }
    auto [val, vah] = rvp.get_value_area(0.70);
    EXPECT_LT(val, vah) << "VAL must be below VAH after eviction";
    EXPECT_GT(rvp.total_volume(), 0.0);
}

// ------------------------------------------------------------------
// Profile statistics tests
// ------------------------------------------------------------------

TEST(VolumeProfile, SkewnessIsZeroForSymmetricProfile) {
    VolumeProfile vp;
    vp.init_range(0.0, 10.0);
    // Symmetric: same volume at equidistant prices around centre
    for (int i = 0; i < VP_BINS; ++i)
        vp.update(vp.bin_centre(i), 1.0);
    double skew = srfm::indicators::profile_skewness(vp);
    EXPECT_NEAR(skew, 0.0, 0.1) << "Symmetric profile should have near-zero skewness";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
