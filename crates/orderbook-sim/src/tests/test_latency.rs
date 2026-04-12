// test_latency.rs — Comprehensive tests for the latency model module.
// Covers Gaussian/LogNormal/Pareto distributions, network model, slippage,
// latency profiles, histogram statistics, jitter AR(1), and throughput counter.

#![allow(clippy::float_cmp)]

use crate::latency_model::{
    AtomicLatencyCounter, JitterModel, LatencyDistribution, LatencyHistogram, LatencyProfile,
    LatencySimulator, NetworkModel, SlippageModel, ThroughputCounter,
};

// ── Distribution tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod distribution_tests {
    use super::*;

    fn xorshift(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn test_gaussian_distribution_mean() {
        let dist = LatencyDistribution::Gaussian { mean_ns: 1000.0, std_ns: 100.0 };
        let mut rng = 42u64;
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| dist.sample_with_rng(&mut rng)).sum();
        let mean = sum / n as f64;
        // Should be within 5% of expected mean
        assert!((mean - 1000.0).abs() < 60.0, "Gaussian mean={:.1} expected ~1000", mean);
    }

    #[test]
    fn test_gaussian_distribution_non_negative() {
        let dist = LatencyDistribution::Gaussian { mean_ns: 500.0, std_ns: 50.0 };
        let mut rng = 123u64;
        for _ in 0..1000 {
            let sample = dist.sample_with_rng(&mut rng);
            assert!(sample >= 0.0, "Latency sample must be non-negative, got {}", sample);
        }
    }

    #[test]
    fn test_lognormal_distribution_positive() {
        let dist = LatencyDistribution::LogNormal { mu: 6.0, sigma: 0.5 };
        let mut rng = 99u64;
        for _ in 0..1000 {
            let sample = dist.sample_with_rng(&mut rng);
            assert!(sample > 0.0, "LogNormal must be positive, got {}", sample);
        }
    }

    #[test]
    fn test_lognormal_median() {
        // For LogNormal(mu, sigma), median = exp(mu)
        let mu = 6.908; // ln(1000) ≈ 6.908
        let dist = LatencyDistribution::LogNormal { mu, sigma: 0.3 };
        let mut rng = 777u64;
        let n = 10_000;
        let mut samples: Vec<f64> = (0..n).map(|_| dist.sample_with_rng(&mut rng)).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[n / 2];
        // Median should be close to exp(mu) = ~1000
        assert!((median - 1000.0).abs() < 200.0, "LogNormal median={:.1} expected ~1000", median);
    }

    #[test]
    fn test_pareto_tail_heavy() {
        let dist = LatencyDistribution::Pareto { x_min: 100.0, alpha: 1.5 };
        let mut rng = 55u64;
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| dist.sample_with_rng(&mut rng)).collect();
        // All values >= x_min
        for s in &samples {
            assert!(*s >= 100.0, "Pareto sample {} < x_min 100", s);
        }
        // Should have occasional large spikes (Pareto tail)
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max > 1000.0, "Pareto should have large tail values, max={}", max);
    }

    #[test]
    fn test_uniform_distribution_bounds() {
        let lo = 100.0f64;
        let hi = 500.0f64;
        let dist = LatencyDistribution::Uniform { min_ns: lo, max_ns: hi };
        let mut rng = 11u64;
        for _ in 0..1000 {
            let s = dist.sample_with_rng(&mut rng);
            assert!(s >= lo && s <= hi, "Uniform sample {} out of [{}, {}]", s, lo, hi);
        }
    }

    #[test]
    fn test_uniform_distribution_mean() {
        let lo = 0.0f64;
        let hi = 1000.0f64;
        let dist = LatencyDistribution::Uniform { min_ns: lo, max_ns: hi };
        let mut rng = 22u64;
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| dist.sample_with_rng(&mut rng)).sum();
        let mean = sum / n as f64;
        assert!((mean - 500.0).abs() < 30.0, "Uniform mean={:.1} expected ~500", mean);
    }

    #[test]
    fn test_constant_distribution() {
        let dist = LatencyDistribution::Constant { ns: 42.0 };
        let mut rng = 1u64;
        for _ in 0..100 {
            let s = dist.sample_with_rng(&mut rng);
            assert_eq!(s, 42.0, "Constant distribution must always return 42.0");
        }
    }

    #[test]
    fn test_bimodal_distribution() {
        let dist = LatencyDistribution::Bimodal {
            mean1_ns: 300.0,
            std1_ns: 30.0,
            mean2_ns: 3000.0,
            std2_ns: 300.0,
            mix_weight: 0.9,
        };
        let mut rng = 33u64;
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| dist.sample_with_rng(&mut rng)).collect();
        // Both modes should appear — check range
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min < 500.0, "Bimodal should have low-latency samples, min={}", min);
        assert!(max > 1500.0, "Bimodal should have high-latency samples, max={}", max);
    }

    #[test]
    fn test_bimodal_mix_weight() {
        // mix_weight=1.0 means 100% mode1
        let dist = LatencyDistribution::Bimodal {
            mean1_ns: 200.0,
            std1_ns: 10.0,
            mean2_ns: 10000.0,
            std2_ns: 100.0,
            mix_weight: 1.0,
        };
        let mut rng = 44u64;
        let n = 1000;
        let samples: Vec<f64> = (0..n).map(|_| dist.sample_with_rng(&mut rng)).collect();
        let above_5000 = samples.iter().filter(|&&s| s > 5000.0).count();
        assert_eq!(above_5000, 0, "mix_weight=1.0 should produce no mode2 samples");
    }
}

// ── Latency profile tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn test_colocation_profile_is_fast() {
        let profile = LatencyProfile::CoLocation;
        let (p50, p99) = profile.expected_percentiles_ns();
        assert!(p50 < 1_000, "CoLocation p50 should be sub-microsecond, got {}", p50);
        assert!(p99 < 10_000, "CoLocation p99 should be <10µs, got {}", p99);
    }

    #[test]
    fn test_retail_profile_is_slow() {
        let profile = LatencyProfile::Retail;
        let (p50, _p99) = profile.expected_percentiles_ns();
        assert!(p50 > 1_000_000, "Retail p50 should be >1ms, got {}", p50);
    }

    #[test]
    fn test_profile_ordering() {
        // CoLocation < DMA < Prime < Retail < Cloud < CrossContinent
        let profiles = [
            LatencyProfile::CoLocation,
            LatencyProfile::Dma,
            LatencyProfile::Prime,
            LatencyProfile::Retail,
            LatencyProfile::Cloud,
            LatencyProfile::CrossContinent,
        ];
        let mut prev_p50 = 0u64;
        for p in &profiles {
            let (p50, _) = p.expected_percentiles_ns();
            assert!(p50 >= prev_p50, "{:?} p50 {} should be >= prev {}", p, p50, prev_p50);
            prev_p50 = p50;
        }
    }

    #[test]
    fn test_cross_continent_is_very_slow() {
        let profile = LatencyProfile::CrossContinent;
        let (p50, _) = profile.expected_percentiles_ns();
        assert!(p50 > 100_000_000, "CrossContinent p50 should be >100ms, got {}", p50);
    }

    #[test]
    fn test_profile_samples_within_expected_range() {
        let profile = LatencyProfile::CoLocation;
        let dist = profile.to_distribution();
        let mut rng = 42u64;
        let n = 1000;
        let samples: Vec<f64> = (0..n).map(|_| dist.sample_with_rng(&mut rng)).collect();
        // CoLocation should mostly be sub-5µs
        let sub_5us = samples.iter().filter(|&&s| s < 5000.0).count();
        assert!(
            sub_5us > 900,
            "CoLocation should be mostly sub-5µs, got {}/{} sub-5µs",
            sub_5us,
            n
        );
    }

    #[test]
    fn test_dma_profile() {
        let profile = LatencyProfile::Dma;
        let (p50, p99) = profile.expected_percentiles_ns();
        assert!(p50 > 1_000, "DMA p50 should be >1µs");
        assert!(p50 < 100_000, "DMA p50 should be <100µs");
        assert!(p99 < 1_000_000, "DMA p99 should be <1ms");
    }
}

// ── Histogram tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod histogram_tests {
    use super::*;

    #[test]
    fn test_histogram_empty() {
        let h = LatencyHistogram::new(20);
        assert_eq!(h.count(), 0);
        assert_eq!(h.min_ns(), u64::MAX);
        assert_eq!(h.max_ns(), 0);
    }

    #[test]
    fn test_histogram_single_value() {
        let mut h = LatencyHistogram::new(20);
        h.record(1234);
        assert_eq!(h.count(), 1);
        assert_eq!(h.min_ns(), 1234);
        assert_eq!(h.max_ns(), 1234);
        assert_eq!(h.p50(), 1234);
        assert_eq!(h.p99(), 1234);
    }

    #[test]
    fn test_histogram_percentile_ordering() {
        let mut h = LatencyHistogram::new(20);
        for i in 1u64..=10000 {
            h.record(i * 100);
        }
        assert!(h.p50() <= h.p95(), "p50 {} > p95 {}", h.p50(), h.p95());
        assert!(h.p95() <= h.p99(), "p95 {} > p99 {}", h.p95(), h.p99());
        assert!(h.p99() <= h.p999(), "p99 {} > p999 {}", h.p99(), h.p999());
    }

    #[test]
    fn test_histogram_p50_approximately_median() {
        let mut h = LatencyHistogram::new(20);
        for i in 1u64..=1000 {
            h.record(i * 1000);
        }
        // Median of 1..=1000 * 1000 is ~500_000
        let p50 = h.p50();
        assert!(
            (p50 as i64 - 500_000).abs() < 100_000,
            "p50={} expected ~500000",
            p50
        );
    }

    #[test]
    fn test_histogram_min_max() {
        let mut h = LatencyHistogram::new(20);
        h.record(50_000);
        h.record(10_000);
        h.record(200_000);
        h.record(1_000);
        assert_eq!(h.min_ns(), 1_000);
        assert_eq!(h.max_ns(), 200_000);
    }

    #[test]
    fn test_histogram_reset() {
        let mut h = LatencyHistogram::new(20);
        h.record(1000);
        h.record(2000);
        assert_eq!(h.count(), 2);
        h.reset();
        assert_eq!(h.count(), 0);
        assert_eq!(h.min_ns(), u64::MAX);
        assert_eq!(h.max_ns(), 0);
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1 = LatencyHistogram::new(20);
        let mut h2 = LatencyHistogram::new(20);
        for i in 1u64..=500 {
            h1.record(i * 1000);
        }
        for i in 501u64..=1000 {
            h2.record(i * 1000);
        }
        h1.merge(&h2);
        assert_eq!(h1.count(), 1000);
        assert_eq!(h1.min_ns(), 1_000);
        assert_eq!(h1.max_ns(), 1_000_000);
    }

    #[test]
    fn test_histogram_summary_string() {
        let mut h = LatencyHistogram::new(20);
        for i in 1u64..=1000 {
            h.record(i * 1000);
        }
        let s = h.summary();
        assert!(s.contains("p50"), "Summary missing p50: {}", s);
        assert!(s.contains("p99"), "Summary missing p99: {}", s);
        assert!(s.contains("n=1000"), "Summary missing count: {}", s);
    }

    #[test]
    fn test_histogram_high_percentile_accuracy() {
        let mut h = LatencyHistogram::new(20);
        // Add 9999 values of 1000ns and 1 value of 1_000_000ns
        for _ in 0..9999 {
            h.record(1000);
        }
        h.record(1_000_000);
        // p99.9 should be 1_000_000
        let p999 = h.p999();
        assert!(
            p999 >= 500_000,
            "p999 should capture the outlier, got {}",
            p999
        );
    }
}

// ── Atomic latency counter tests ──────────────────────────────────────────────

#[cfg(test)]
mod atomic_counter_tests {
    use super::*;

    #[test]
    fn test_atomic_counter_basic() {
        let c = AtomicLatencyCounter::new();
        c.record(500);
        c.record(1000);
        c.record(200);
        assert_eq!(c.count(), 3);
        assert_eq!(c.min_ns(), 200);
        assert_eq!(c.max_ns(), 1000);
    }

    #[test]
    fn test_atomic_counter_mean() {
        let c = AtomicLatencyCounter::new();
        for v in [100u64, 200, 300, 400, 500] {
            c.record(v);
        }
        let mean = c.mean();
        assert!((mean - 300.0).abs() < 1.0, "mean={} expected 300", mean);
    }

    #[test]
    fn test_atomic_counter_reset() {
        let c = AtomicLatencyCounter::new();
        c.record(999);
        c.reset();
        assert_eq!(c.count(), 0);
        assert_eq!(c.min_ns(), u64::MAX);
        assert_eq!(c.max_ns(), 0);
    }

    #[test]
    fn test_atomic_counter_empty_mean() {
        let c = AtomicLatencyCounter::new();
        assert_eq!(c.mean(), 0.0);
    }

    #[test]
    fn test_atomic_counter_single() {
        let c = AtomicLatencyCounter::new();
        c.record(42);
        assert_eq!(c.min_ns(), 42);
        assert_eq!(c.max_ns(), 42);
        assert_eq!(c.mean(), 42.0);
    }
}

// ── Network model tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod network_model_tests {
    use super::*;

    #[test]
    fn test_network_model_base_latency() {
        let model = NetworkModel::new(LatencyProfile::CoLocation, 0.0);
        let mut rng = 42u64;
        let lat = model.sample_latency(&mut rng);
        // CoLocation should be fast
        assert!(lat < 100_000.0, "CoLocation latency too high: {}", lat);
    }

    #[test]
    fn test_network_model_congestion_increases_latency() {
        let base = NetworkModel::new(LatencyProfile::Dma, 0.0);
        let congested = NetworkModel::new(LatencyProfile::Dma, 0.8);
        let mut rng = 42u64;
        let n = 100;
        let base_sum: f64 = (0..n).map(|_| base.sample_latency(&mut rng)).sum();
        let mut rng2 = 42u64;
        let cong_sum: f64 = (0..n).map(|_| congested.sample_latency(&mut rng2)).sum();
        assert!(
            cong_sum > base_sum,
            "Congested latency {} should exceed base {}",
            cong_sum,
            base_sum
        );
    }

    #[test]
    fn test_network_model_packet_loss() {
        // With 50% packet loss, roughly half of transmissions should fail
        let model = NetworkModel::with_packet_loss(LatencyProfile::Retail, 0.0, 0.5);
        let mut rng = 42u64;
        let n = 1000;
        let losses = (0..n).filter(|_| model.is_packet_lost(&mut rng)).count();
        assert!(
            losses > 300 && losses < 700,
            "Expected ~50% packet loss, got {}/{}",
            losses,
            n
        );
    }

    #[test]
    fn test_network_model_no_packet_loss() {
        let model = NetworkModel::with_packet_loss(LatencyProfile::Retail, 0.0, 0.0);
        let mut rng = 42u64;
        for _ in 0..1000 {
            assert!(!model.is_packet_lost(&mut rng), "Should have no packet loss");
        }
    }

    #[test]
    fn test_network_model_congestion_factor_range() {
        // Congestion factor must be [0, 1]
        let model = NetworkModel::new(LatencyProfile::Prime, 1.0);
        let mut rng = 42u64;
        let lat = model.sample_latency(&mut rng);
        assert!(lat >= 0.0, "Latency must be non-negative, got {}", lat);
    }
}

// ── Slippage model tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod slippage_tests {
    use super::*;

    #[test]
    fn test_slippage_zero_volatility() {
        let model = SlippageModel::new(0.0, 0.5);
        // With zero volatility, slippage should be near zero
        let slip = model.compute_slippage_bps(100.0, 10.0, 1000.0);
        assert!(slip.abs() < 1.0, "Zero-vol slippage should be near 0, got {}", slip);
    }

    #[test]
    fn test_slippage_increases_with_order_size() {
        let model = SlippageModel::new(0.2, 0.5);
        let small = model.compute_slippage_bps(100.0, 1.0, 1000.0);
        let large = model.compute_slippage_bps(100.0, 100.0, 1000.0);
        assert!(
            large.abs() >= small.abs(),
            "Large order slippage {} < small order slippage {}",
            large,
            small
        );
    }

    #[test]
    fn test_slippage_increases_with_volatility() {
        let low_vol = SlippageModel::new(0.1, 0.5);
        let high_vol = SlippageModel::new(0.5, 0.5);
        let slip_low = low_vol.compute_slippage_bps(100.0, 10.0, 1000.0).abs();
        let slip_high = high_vol.compute_slippage_bps(100.0, 10.0, 1000.0).abs();
        assert!(
            slip_high >= slip_low,
            "High-vol slippage {} < low-vol slippage {}",
            slip_high,
            slip_low
        );
    }

    #[test]
    fn test_slippage_participation_rate_effect() {
        let model = SlippageModel::new(0.2, 0.5);
        // Small order relative to volume = low participation rate = less slippage
        let low_prate = model.compute_slippage_bps(100.0, 1.0, 10_000.0).abs();
        let high_prate = model.compute_slippage_bps(100.0, 1000.0, 10_000.0).abs();
        assert!(
            high_prate >= low_prate,
            "High participation rate slippage {} < low {}",
            high_prate,
            low_prate
        );
    }
}

// ── Jitter model tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod jitter_tests {
    use super::*;

    #[test]
    fn test_jitter_model_mean_reversion() {
        // AR(1) with high mean-reversion (phi close to 0) should stay near baseline
        let model = JitterModel::new(1000.0, 0.1, 50.0);
        let mut rng = 42u64;
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| model.next_sample(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        // Should be close to baseline of 1000
        assert!(
            (mean - 1000.0).abs() < 200.0,
            "Jitter mean={:.1} expected ~1000",
            mean
        );
    }

    #[test]
    fn test_jitter_model_autocorrelation() {
        // AR(1) with phi close to 1 should have high autocorrelation
        let model = JitterModel::new(1000.0, 0.95, 10.0);
        let mut rng = 42u64;
        let n = 1000;
        let samples: Vec<f64> = (0..n).map(|_| model.next_sample(&mut rng)).collect();
        // Compute lag-1 autocorrelation
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let cov: f64 = samples
            .windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>()
            / (n - 1) as f64;
        let autocorr = if var > 0.0 { cov / var } else { 0.0 };
        assert!(
            autocorr > 0.7,
            "High-phi AR(1) should have autocorr > 0.7, got {:.3}",
            autocorr
        );
    }

    #[test]
    fn test_jitter_model_non_negative() {
        let model = JitterModel::new(500.0, 0.8, 200.0);
        let mut rng = 123u64;
        for _ in 0..10_000 {
            let s = model.next_sample(&mut rng);
            assert!(s >= 0.0, "Jitter sample must be non-negative, got {}", s);
        }
    }

    #[test]
    fn test_jitter_model_reset() {
        let model = JitterModel::new(1000.0, 0.5, 100.0);
        model.reset();
        let mut rng = 42u64;
        let s = model.next_sample(&mut rng);
        assert!(s >= 0.0);
    }
}

// ── Throughput counter tests ──────────────────────────────────────────────────

#[cfg(test)]
mod throughput_tests {
    use super::*;

    #[test]
    fn test_throughput_counter_total() {
        let tc = ThroughputCounter::new(1.0);
        tc.record(100);
        tc.record(200);
        tc.record(300);
        assert_eq!(tc.total(), 600);
    }

    #[test]
    fn test_throughput_counter_count() {
        let tc = ThroughputCounter::new(1.0);
        tc.record(1);
        tc.record(1);
        tc.record(1);
        assert_eq!(tc.event_count(), 3);
    }

    #[test]
    fn test_throughput_counter_reset() {
        let tc = ThroughputCounter::new(1.0);
        tc.record(500);
        tc.record(500);
        assert_eq!(tc.total(), 1000);
        tc.reset();
        assert_eq!(tc.total(), 0);
        assert_eq!(tc.event_count(), 0);
    }

    #[test]
    fn test_throughput_counter_rate_positive() {
        let tc = ThroughputCounter::new(1.0);
        for _ in 0..100 {
            tc.record(1);
        }
        // Rate computation should return something reasonable
        let rate = tc.rate_per_sec();
        assert!(rate >= 0.0, "Rate must be non-negative, got {}", rate);
    }

    #[test]
    fn test_throughput_counter_empty_rate() {
        let tc = ThroughputCounter::new(1.0);
        // Rate of empty counter shouldn't panic
        let rate = tc.rate_per_sec();
        assert!(rate >= 0.0);
    }
}

// ── Latency simulator integration tests ──────────────────────────────────────

#[cfg(test)]
mod simulator_tests {
    use super::*;

    #[test]
    fn test_simulator_colocation_end_to_end() {
        let mut sim = LatencySimulator::new(LatencyProfile::CoLocation, 42);
        let n = 1000;
        for _ in 0..n {
            let lat = sim.simulate_order_rtt_ns();
            assert!(lat > 0, "Latency must be positive");
            assert!(lat < 10_000_000, "CoLocation RTT should be <10ms");
        }
    }

    #[test]
    fn test_simulator_retail_end_to_end() {
        let mut sim = LatencySimulator::new(LatencyProfile::Retail, 42);
        let n = 100;
        let mut total = 0u64;
        for _ in 0..n {
            total += sim.simulate_order_rtt_ns();
        }
        let mean = total / n as u64;
        assert!(mean > 500_000, "Retail mean RTT should be >500µs, got {}", mean);
    }

    #[test]
    fn test_simulator_histogram_recording() {
        let mut sim = LatencySimulator::new(LatencyProfile::Dma, 42);
        for _ in 0..1000 {
            sim.simulate_order_rtt_ns();
        }
        let hist = sim.histogram();
        assert_eq!(hist.count(), 1000);
        assert!(hist.p50() > 0, "p50 should be positive");
    }

    #[test]
    fn test_simulator_with_congestion() {
        let mut sim_base = LatencySimulator::with_congestion(LatencyProfile::Dma, 0.0, 42);
        let mut sim_cong = LatencySimulator::with_congestion(LatencyProfile::Dma, 0.9, 42);
        let n = 500;
        let base_total: u64 = (0..n).map(|_| sim_base.simulate_order_rtt_ns()).sum();
        let cong_total: u64 = (0..n).map(|_| sim_cong.simulate_order_rtt_ns()).sum();
        assert!(
            cong_total >= base_total,
            "Congested sim total {} < base {}",
            cong_total,
            base_total
        );
    }

    #[test]
    fn test_simulator_reset_clears_histogram() {
        let mut sim = LatencySimulator::new(LatencyProfile::Prime, 42);
        for _ in 0..100 {
            sim.simulate_order_rtt_ns();
        }
        sim.reset();
        let hist = sim.histogram();
        assert_eq!(hist.count(), 0, "Histogram should be cleared after reset");
    }

    #[test]
    fn test_simulator_deterministic_with_same_seed() {
        let mut sim1 = LatencySimulator::new(LatencyProfile::CoLocation, 999);
        let mut sim2 = LatencySimulator::new(LatencyProfile::CoLocation, 999);
        let n = 100;
        let r1: Vec<u64> = (0..n).map(|_| sim1.simulate_order_rtt_ns()).collect();
        let r2: Vec<u64> = (0..n).map(|_| sim2.simulate_order_rtt_ns()).collect();
        assert_eq!(r1, r2, "Same seed should produce identical sequence");
    }

    #[test]
    fn test_simulator_different_seeds_diverge() {
        let mut sim1 = LatencySimulator::new(LatencyProfile::CoLocation, 1);
        let mut sim2 = LatencySimulator::new(LatencyProfile::CoLocation, 2);
        let n = 100;
        let r1: Vec<u64> = (0..n).map(|_| sim1.simulate_order_rtt_ns()).collect();
        let r2: Vec<u64> = (0..n).map(|_| sim2.simulate_order_rtt_ns()).collect();
        // They should not be completely identical
        assert_ne!(r1, r2, "Different seeds should produce different sequences");
    }

    #[test]
    fn test_simulator_summary_report() {
        let mut sim = LatencySimulator::new(LatencyProfile::Prime, 42);
        for _ in 0..1000 {
            sim.simulate_order_rtt_ns();
        }
        let report = sim.summary_report();
        assert!(!report.is_empty(), "Summary report should not be empty");
        assert!(report.contains("p50") || report.contains("P50"), "Report should contain p50");
    }
}

// ── Edge case tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_histogram_large_count() {
        let mut h = LatencyHistogram::new(20);
        for i in 0u64..100_000 {
            h.record(i % 10_000 + 1);
        }
        assert_eq!(h.count(), 100_000);
        let p50 = h.p50();
        assert!(p50 > 0, "p50 must be positive");
    }

    #[test]
    fn test_histogram_all_same_value() {
        let mut h = LatencyHistogram::new(20);
        for _ in 0..1000 {
            h.record(5000);
        }
        assert_eq!(h.min_ns(), 5000);
        assert_eq!(h.max_ns(), 5000);
        assert_eq!(h.p50(), 5000);
        assert_eq!(h.p99(), 5000);
    }

    #[test]
    fn test_histogram_extreme_values() {
        let mut h = LatencyHistogram::new(20);
        h.record(1);
        h.record(u64::MAX / 2);
        assert_eq!(h.count(), 2);
        assert_eq!(h.min_ns(), 1);
        assert_eq!(h.max_ns(), u64::MAX / 2);
    }

    #[test]
    fn test_distribution_clamp_negative_to_zero() {
        // Gaussian with large std might produce negative values — must clamp
        let dist = LatencyDistribution::Gaussian { mean_ns: 10.0, std_ns: 1000.0 };
        let mut rng = 42u64;
        for _ in 0..10_000 {
            let s = dist.sample_with_rng(&mut rng);
            assert!(s >= 0.0, "Latency must not be negative, got {}", s);
        }
    }

    #[test]
    fn test_pareto_min_value_enforcement() {
        let dist = LatencyDistribution::Pareto { x_min: 250.0, alpha: 2.0 };
        let mut rng = 42u64;
        for _ in 0..10_000 {
            let s = dist.sample_with_rng(&mut rng);
            assert!(s >= 250.0, "Pareto sample {} < x_min 250", s);
        }
    }

    #[test]
    fn test_network_model_zero_congestion() {
        let model = NetworkModel::new(LatencyProfile::Dma, 0.0);
        let mut rng = 42u64;
        let lat = model.sample_latency(&mut rng);
        assert!(lat > 0.0);
        assert!(lat < 10_000_000.0, "DMA with no congestion should be fast");
    }

    #[test]
    fn test_simulator_outlier_handling() {
        // Test that simulator correctly tracks extreme outliers
        let mut sim = LatencySimulator::new(LatencyProfile::Retail, 42);
        // Record one very large latency
        sim.record_latency(1_000_000_000); // 1 second
        sim.record_latency(1_000); // 1 µs
        let hist = sim.histogram();
        assert_eq!(hist.max_ns(), 1_000_000_000);
        assert_eq!(hist.min_ns(), 1_000);
    }

    #[test]
    fn test_throughput_counter_large_values() {
        let tc = ThroughputCounter::new(1.0);
        tc.record(u64::MAX / 4);
        tc.record(u64::MAX / 4);
        let total = tc.total();
        assert!(total > 0, "Total should accumulate without overflow");
    }
}

// ── Percentile accuracy tests ─────────────────────────────────────────────────

#[cfg(test)]
mod percentile_tests {
    use super::*;

    #[test]
    fn test_p99_accuracy_with_known_distribution() {
        let mut h = LatencyHistogram::new(20);
        // 100 uniform values from 1000 to 100000 (step 990)
        for i in 0..100u64 {
            h.record(1000 + i * 990);
        }
        // p99 should be the 99th value out of 100
        let p99 = h.p99();
        // 99th value = 1000 + 98 * 990 = 98_020
        let expected = 1000 + 98 * 990;
        assert!(
            (p99 as i64 - expected as i64).abs() < 5000,
            "p99={} expected ~{}",
            p99,
            expected
        );
    }

    #[test]
    fn test_p50_accuracy_with_known_distribution() {
        let mut h = LatencyHistogram::new(20);
        // 1000 values, all 42000ns
        for _ in 0..1000 {
            h.record(42_000);
        }
        assert_eq!(h.p50(), 42_000);
    }

    #[test]
    fn test_percentile_monotonicity() {
        let mut h = LatencyHistogram::new(20);
        let mut rng = 42u64;
        for _ in 0..10_000 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            h.record((rng >> 11) % 1_000_000 + 100);
        }
        let p50 = h.p50();
        let p95 = h.p95();
        let p99 = h.p99();
        let p999 = h.p999();
        assert!(p50 <= p95, "p50 {} > p95 {}", p50, p95);
        assert!(p95 <= p99, "p95 {} > p99 {}", p95, p99);
        assert!(p99 <= p999, "p99 {} > p999 {}", p99, p999);
    }

    #[test]
    fn test_percentile_after_merge() {
        let mut h1 = LatencyHistogram::new(20);
        let mut h2 = LatencyHistogram::new(20);
        for i in 1u64..=100 {
            h1.record(i * 1000);
            h2.record(i * 1000 + 100_000);
        }
        h1.merge(&h2);
        // After merge, p50 should be in the lower half
        let p50 = h1.p50();
        assert!(p50 < 150_000, "p50 after merge should be in lower range, got {}", p50);
    }
}
