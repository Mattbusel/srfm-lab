//! latency_model.rs — Network/execution latency simulation for ultra-low-latency LOB engine.
//!
//! Models: Gaussian jitter, fat-tail network delays, co-location vs retail profiles,
//! execution slippage, and full percentile tracking (p50/p95/p99/p999).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ── Nanos re-export ──────────────────────────────────────────────────────────
pub type Nanos = u64;

// ── PRNG (xorshift64) ────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Rng { state: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self { Rng { state: seed ^ 0xdeadbeef_cafef00d } }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; x
    }

    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }

    /// Box-Muller transform for standard normal sample
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Exponential distribution with given mean
    pub fn next_exp(&mut self, mean: f64) -> f64 {
        -mean * self.next_f64().max(1e-15).ln()
    }

    /// Log-normal sample
    pub fn next_lognormal(&mut self, mu: f64, sigma: f64) -> f64 {
        (mu + sigma * self.next_normal()).exp()
    }
}

// ── Latency sample distribution ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum DistributionKind {
    Gaussian,
    LogNormal,
    Pareto,       // heavy-tail for burst delays
    Uniform,
    Constant,
    Bimodal,      // two Gaussians (normal + spike)
}

#[derive(Debug, Clone)]
pub struct LatencyDistribution {
    pub kind: DistributionKind,
    pub base_ns: f64,      // base (minimum) latency in nanoseconds
    pub mean_ns: f64,      // mean additional delay
    pub std_dev_ns: f64,   // standard deviation
    pub spike_prob: f64,   // probability of entering spike mode (Bimodal)
    pub spike_mean_ns: f64,
    pub spike_std_ns: f64,
    pub pareto_alpha: f64, // shape for Pareto
}

impl LatencyDistribution {
    pub fn gaussian(base_ns: f64, mean_ns: f64, std_dev_ns: f64) -> Self {
        LatencyDistribution { kind: DistributionKind::Gaussian, base_ns, mean_ns, std_dev_ns, spike_prob: 0.0, spike_mean_ns: 0.0, spike_std_ns: 0.0, pareto_alpha: 2.0 }
    }

    pub fn lognormal(base_ns: f64, mean_ns: f64, std_dev_ns: f64) -> Self {
        LatencyDistribution { kind: DistributionKind::LogNormal, base_ns, mean_ns, std_dev_ns, spike_prob: 0.0, spike_mean_ns: 0.0, spike_std_ns: 0.0, pareto_alpha: 2.0 }
    }

    pub fn bimodal(base_ns: f64, mean_ns: f64, std_dev_ns: f64, spike_prob: f64, spike_mean_ns: f64, spike_std_ns: f64) -> Self {
        LatencyDistribution { kind: DistributionKind::Bimodal, base_ns, mean_ns, std_dev_ns, spike_prob, spike_mean_ns, spike_std_ns, pareto_alpha: 2.0 }
    }

    pub fn sample(&self, rng: &mut Rng) -> Nanos {
        let delay = match self.kind {
            DistributionKind::Constant => self.mean_ns,
            DistributionKind::Uniform => self.mean_ns - self.std_dev_ns + 2.0 * self.std_dev_ns * rng.next_f64(),
            DistributionKind::Gaussian => self.mean_ns + self.std_dev_ns * rng.next_normal(),
            DistributionKind::LogNormal => {
                // Convert mean/std to lognormal parameters
                let m = self.mean_ns;
                let s = self.std_dev_ns;
                let sigma2 = (1.0 + (s / m).powi(2)).ln();
                let mu = m.ln() - sigma2 / 2.0;
                rng.next_lognormal(mu, sigma2.sqrt())
            }
            DistributionKind::Pareto => {
                // Pareto with scale = mean_ns * (alpha - 1) / alpha
                let alpha = self.pareto_alpha.max(1.01);
                let scale = self.mean_ns * (alpha - 1.0) / alpha;
                scale / rng.next_f64().max(1e-15).powf(1.0 / alpha)
            }
            DistributionKind::Bimodal => {
                if rng.next_f64() < self.spike_prob {
                    self.spike_mean_ns + self.spike_std_ns * rng.next_normal()
                } else {
                    self.mean_ns + self.std_dev_ns * rng.next_normal()
                }
            }
        };
        (self.base_ns + delay.max(0.0)) as Nanos
    }
}

// ── Latency profile presets ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LatencyProfile {
    CoLocation,       // FPGA co-lo: ~200ns base
    DirectMarketAccess, // DMA: ~5µs
    Prime,            // Prime broker: ~20µs
    Retail,           // Retail API: ~2ms
    Cloud,            // Cloud execution: ~10ms
    CrossContinent,   // Cross-continent: ~80ms
    Custom,
}

impl LatencyProfile {
    /// Returns (market_data_latency, order_submission_latency, ack_latency)
    pub fn distributions(&self) -> (LatencyDistribution, LatencyDistribution, LatencyDistribution) {
        match self {
            LatencyProfile::CoLocation => (
                LatencyDistribution::gaussian(150.0, 80.0, 20.0),       // md: ~230ns
                LatencyDistribution::gaussian(200.0, 100.0, 30.0),      // submit: ~300ns
                LatencyDistribution::gaussian(100.0, 50.0, 10.0),       // ack: ~150ns
            ),
            LatencyProfile::DirectMarketAccess => (
                LatencyDistribution::lognormal(4_000.0, 1_000.0, 500.0),
                LatencyDistribution::lognormal(5_000.0, 2_000.0, 800.0),
                LatencyDistribution::lognormal(2_000.0, 800.0, 300.0),
            ),
            LatencyProfile::Prime => (
                LatencyDistribution::lognormal(15_000.0, 5_000.0, 2_000.0),
                LatencyDistribution::lognormal(20_000.0, 8_000.0, 3_000.0),
                LatencyDistribution::lognormal(10_000.0, 3_000.0, 1_000.0),
            ),
            LatencyProfile::Retail => (
                LatencyDistribution::bimodal(1_500_000.0, 500_000.0, 200_000.0, 0.05, 10_000_000.0, 3_000_000.0),
                LatencyDistribution::bimodal(2_000_000.0, 800_000.0, 300_000.0, 0.05, 15_000_000.0, 5_000_000.0),
                LatencyDistribution::bimodal(1_000_000.0, 400_000.0, 150_000.0, 0.05, 8_000_000.0, 2_000_000.0),
            ),
            LatencyProfile::Cloud => (
                LatencyDistribution::lognormal(8_000_000.0, 2_000_000.0, 1_000_000.0),
                LatencyDistribution::lognormal(10_000_000.0, 3_000_000.0, 1_500_000.0),
                LatencyDistribution::lognormal(5_000_000.0, 1_500_000.0, 800_000.0),
            ),
            LatencyProfile::CrossContinent => (
                LatencyDistribution::gaussian(70_000_000.0, 10_000_000.0, 5_000_000.0),
                LatencyDistribution::gaussian(80_000_000.0, 12_000_000.0, 6_000_000.0),
                LatencyDistribution::gaussian(60_000_000.0, 8_000_000.0, 4_000_000.0),
            ),
            LatencyProfile::Custom => (
                LatencyDistribution::gaussian(0.0, 1_000.0, 100.0),
                LatencyDistribution::gaussian(0.0, 1_000.0, 100.0),
                LatencyDistribution::gaussian(0.0, 1_000.0, 100.0),
            ),
        }
    }

    pub fn round_trip_estimate_ns(&self) -> u64 {
        match self {
            LatencyProfile::CoLocation => 500,
            LatencyProfile::DirectMarketAccess => 10_000,
            LatencyProfile::Prime => 50_000,
            LatencyProfile::Retail => 4_000_000,
            LatencyProfile::Cloud => 25_000_000,
            LatencyProfile::CrossContinent => 180_000_000,
            LatencyProfile::Custom => 2_000,
        }
    }
}

// ── Network delay model ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NetworkModel {
    pub profile: LatencyProfile,
    pub md_dist: LatencyDistribution,
    pub submit_dist: LatencyDistribution,
    pub ack_dist: LatencyDistribution,
    /// Queue congestion state (0.0 = no congestion, 1.0 = fully saturated)
    pub congestion_factor: f64,
    /// Packet loss probability
    pub packet_loss_prob: f64,
    rng: Rng,
}

impl NetworkModel {
    pub fn new(profile: LatencyProfile, seed: u64) -> Self {
        let (md, submit, ack) = profile.distributions();
        NetworkModel {
            profile,
            md_dist: md,
            submit_dist: submit,
            ack_dist: ack,
            congestion_factor: 0.0,
            packet_loss_prob: 0.0,
            rng: Rng::new(seed),
        }
    }

    pub fn set_congestion(&mut self, factor: f64) {
        self.congestion_factor = factor.clamp(0.0, 1.0);
    }

    /// Sample market-data delivery latency
    pub fn sample_md_latency(&mut self) -> Option<Nanos> {
        if self.rng.next_f64() < self.packet_loss_prob { return None; }
        let base = self.md_dist.sample(&mut self.rng);
        let congestion_extra = (base as f64 * self.congestion_factor * 5.0) as Nanos;
        Some(base + congestion_extra)
    }

    /// Sample order submission latency
    pub fn sample_submit_latency(&mut self) -> Option<Nanos> {
        if self.rng.next_f64() < self.packet_loss_prob { return None; }
        let base = self.submit_dist.sample(&mut self.rng);
        let congestion_extra = (base as f64 * self.congestion_factor * 8.0) as Nanos;
        Some(base + congestion_extra)
    }

    /// Sample acknowledgment latency
    pub fn sample_ack_latency(&mut self) -> Option<Nanos> {
        let base = self.ack_dist.sample(&mut self.rng);
        Some(base)
    }

    /// Full round-trip latency: md + submit + ack
    pub fn sample_round_trip(&mut self) -> Option<Nanos> {
        let md = self.sample_md_latency()?;
        let sub = self.sample_submit_latency()?;
        let ack = self.sample_ack_latency()?;
        Some(md + sub + ack)
    }
}

// ── Execution slippage model ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SlippageParams {
    /// Base slippage in basis points at unit size
    pub base_bps: f64,
    /// Size scaling exponent (0.5 = sqrt, 1.0 = linear)
    pub size_exponent: f64,
    /// Volatility scaling factor (slippage increases with vol)
    pub vol_sensitivity: f64,
    /// Market impact scaling (LOB depth dependent)
    pub impact_scaling: f64,
    /// Adverse selection component (informed trading fraction)
    pub adverse_selection: f64,
}

impl Default for SlippageParams {
    fn default() -> Self {
        SlippageParams { base_bps: 0.5, size_exponent: 0.6, vol_sensitivity: 2.0, impact_scaling: 1.0, adverse_selection: 0.3 }
    }
}

impl SlippageParams {
    pub fn co_location() -> Self {
        SlippageParams { base_bps: 0.05, size_exponent: 0.5, vol_sensitivity: 1.2, impact_scaling: 0.5, adverse_selection: 0.1 }
    }
    pub fn retail() -> Self {
        SlippageParams { base_bps: 2.0, size_exponent: 0.7, vol_sensitivity: 3.0, impact_scaling: 2.0, adverse_selection: 0.5 }
    }
}

#[derive(Debug, Clone)]
pub struct SlippageModel {
    params: SlippageParams,
    realized_vols: VecDeque<f64>,
    vol_window: usize,
    running_vol: f64,
}

impl SlippageModel {
    pub fn new(params: SlippageParams) -> Self {
        SlippageModel { params, realized_vols: VecDeque::new(), vol_window: 100, running_vol: 0.01 }
    }

    pub fn update_vol(&mut self, ret: f64) {
        self.realized_vols.push_back(ret.abs());
        if self.realized_vols.len() > self.vol_window {
            self.realized_vols.pop_front();
        }
        self.running_vol = self.realized_vols.iter().sum::<f64>() / self.realized_vols.len() as f64;
    }

    /// Compute expected slippage in basis points for a given order.
    /// qty_fraction: fraction of average daily volume being executed
    pub fn expected_slippage_bps(&self, qty_fraction: f64, side_multiplier: f64, lob_depth_ratio: f64) -> f64 {
        let size_effect = qty_fraction.powf(self.params.size_exponent);
        let vol_effect = 1.0 + self.params.vol_sensitivity * self.running_vol * 100.0;
        let depth_effect = 1.0 / lob_depth_ratio.max(0.1);
        let base = self.params.base_bps * size_effect * vol_effect * depth_effect * self.params.impact_scaling;
        let adverse = self.params.adverse_selection * side_multiplier;
        base + adverse
    }

    /// Convert bps slippage to price ticks given a price and tick size
    pub fn slippage_to_ticks(&self, slippage_bps: f64, mid_price: f64, tick_size: f64) -> f64 {
        let price_move = mid_price * slippage_bps / 10_000.0;
        price_move / tick_size
    }
}

// ── Latency percentile tracker ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    // HDR-lite: fixed-size power-of-two buckets
    buckets: Vec<u64>,
    bucket_count: usize,
    min_val: u64,
    max_val: u64,
    total_count: u64,
    sum: u64,
    /// Resolution: each bucket covers 2^resolution nanoseconds
    resolution_bits: u32,
}

impl LatencyHistogram {
    pub fn new(max_ns: u64) -> Self {
        let resolution_bits = 8u32; // 256ns resolution
        let bucket_count = ((max_ns >> resolution_bits) + 1) as usize;
        LatencyHistogram {
            buckets: vec![0u64; bucket_count.min(1 << 20)], // cap at 1M buckets
            bucket_count: bucket_count.min(1 << 20),
            min_val: u64::MAX,
            max_val: 0,
            total_count: 0,
            sum: 0,
            resolution_bits,
        }
    }

    pub fn record(&mut self, latency_ns: u64) {
        let idx = (latency_ns >> self.resolution_bits) as usize;
        let idx = idx.min(self.bucket_count - 1);
        self.buckets[idx] += 1;
        self.total_count += 1;
        self.sum += latency_ns;
        if latency_ns < self.min_val { self.min_val = latency_ns; }
        if latency_ns > self.max_val { self.max_val = latency_ns; }
    }

    pub fn percentile(&self, pct: f64) -> u64 {
        if self.total_count == 0 { return 0; }
        let target = (pct / 100.0 * self.total_count as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return ((i as u64) << self.resolution_bits) + (1u64 << (self.resolution_bits - 1));
            }
        }
        self.max_val
    }

    pub fn p50(&self) -> u64 { self.percentile(50.0) }
    pub fn p95(&self) -> u64 { self.percentile(95.0) }
    pub fn p99(&self) -> u64 { self.percentile(99.0) }
    pub fn p999(&self) -> u64 { self.percentile(99.9) }
    pub fn p9999(&self) -> u64 { self.percentile(99.99) }
    pub fn min(&self) -> u64 { if self.min_val == u64::MAX { 0 } else { self.min_val } }
    pub fn max(&self) -> u64 { self.max_val }
    pub fn mean(&self) -> f64 { if self.total_count == 0 { 0.0 } else { self.sum as f64 / self.total_count as f64 } }
    pub fn count(&self) -> u64 { self.total_count }

    pub fn summary(&self) -> LatencySummary {
        LatencySummary {
            count: self.count(),
            min_ns: self.min(),
            p50_ns: self.p50(),
            p95_ns: self.p95(),
            p99_ns: self.p99(),
            p999_ns: self.p999(),
            p9999_ns: self.p9999(),
            max_ns: self.max(),
            mean_ns: self.mean(),
        }
    }

    pub fn reset(&mut self) {
        for b in self.buckets.iter_mut() { *b = 0; }
        self.total_count = 0;
        self.sum = 0;
        self.min_val = u64::MAX;
        self.max_val = 0;
    }

    pub fn merge(&mut self, other: &LatencyHistogram) {
        let len = self.buckets.len().min(other.buckets.len());
        for i in 0..len {
            self.buckets[i] += other.buckets[i];
        }
        self.total_count += other.total_count;
        self.sum += other.sum;
        if other.min_val < self.min_val { self.min_val = other.min_val; }
        if other.max_val > self.max_val { self.max_val = other.max_val; }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LatencySummary {
    pub count: u64,
    pub min_ns: u64,
    pub mean_ns: f64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub p9999_ns: u64,
    pub max_ns: u64,
}

impl std::fmt::Display for LatencySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n={} min={}ns mean={:.0}ns p50={}ns p95={}ns p99={}ns p999={}ns max={}ns",
               self.count, self.min_ns, self.mean_ns, self.p50_ns, self.p95_ns,
               self.p99_ns, self.p999_ns, self.max_ns)
    }
}

// ── Rolling latency tracker (lock-free atomic) ───────────────────────────────

pub struct AtomicLatencyCounter {
    sum: AtomicU64,
    count: AtomicU64,
    max: AtomicU64,
    min: AtomicU64,
}

impl AtomicLatencyCounter {
    pub fn new() -> Arc<Self> {
        Arc::new(AtomicLatencyCounter {
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
            max: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
        })
    }

    pub fn record(&self, ns: u64) {
        self.sum.fetch_add(ns, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        let mut cur = self.max.load(Ordering::Relaxed);
        while ns > cur {
            match self.max.compare_exchange_weak(cur, ns, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
        let mut cur = self.min.load(Ordering::Relaxed);
        while ns < cur {
            match self.min.compare_exchange_weak(cur, ns, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }

    pub fn mean(&self) -> f64 {
        let c = self.count.load(Ordering::Relaxed);
        if c == 0 { return 0.0; }
        self.sum.load(Ordering::Relaxed) as f64 / c as f64
    }

    pub fn count(&self) -> u64 { self.count.load(Ordering::Relaxed) }
    pub fn max(&self) -> u64 { self.max.load(Ordering::Relaxed) }
    pub fn min(&self) -> u64 {
        let v = self.min.load(Ordering::Relaxed);
        if v == u64::MAX { 0 } else { v }
    }
}

// ── Jitter model ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct JitterModel {
    /// Autocorrelation coefficient: how much previous jitter persists
    pub phi: f64,
    pub innovation_std: f64,
    state: f64,
    rng: Rng,
}

impl JitterModel {
    /// phi in [0, 1), higher = more correlated jitter (AR(1))
    pub fn new(phi: f64, innovation_std: f64, seed: u64) -> Self {
        JitterModel { phi: phi.clamp(0.0, 0.99), innovation_std, state: 0.0, rng: Rng::new(seed) }
    }

    pub fn next_jitter_ns(&mut self) -> f64 {
        self.state = self.phi * self.state + self.innovation_std * self.rng.next_normal();
        self.state
    }

    pub fn apply_to_latency(&mut self, base_ns: Nanos) -> Nanos {
        let jitter = self.next_jitter_ns();
        ((base_ns as f64 + jitter).max(0.0)) as Nanos
    }
}

// ── End-to-end latency simulator ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LatencySimulator {
    pub network: NetworkModel,
    pub slippage: SlippageModel,
    pub jitter: JitterModel,
    pub histogram: LatencyHistogram,
    pub profile: LatencyProfile,
    sample_count: u64,
}

impl LatencySimulator {
    pub fn new(profile: LatencyProfile, seed: u64) -> Self {
        let network = NetworkModel::new(profile.clone(), seed);
        let slippage = SlippageModel::new(SlippageParams::default());
        let jitter = JitterModel::new(0.3, 100.0, seed ^ 0x1234);
        let histogram = LatencyHistogram::new(1_000_000_000); // 1 second max
        LatencySimulator { network, slippage, jitter, histogram, profile, sample_count: 0 }
    }

    pub fn with_slippage_params(mut self, params: SlippageParams) -> Self {
        self.slippage = SlippageModel::new(params); self
    }

    pub fn with_congestion(mut self, factor: f64) -> Self {
        self.network.set_congestion(factor); self
    }

    /// Simulate a complete order round-trip. Returns (total_latency_ns, slippage_bps)
    pub fn simulate_order(
        &mut self,
        qty_fraction: f64,
        is_aggressive: bool,
        lob_depth_ratio: f64,
    ) -> Option<(Nanos, f64)> {
        let rtt = self.network.sample_round_trip()?;
        let rtt_with_jitter = self.jitter.apply_to_latency(rtt);
        self.histogram.record(rtt_with_jitter);
        self.sample_count += 1;

        let side_mult = if is_aggressive { 1.0 } else { 0.5 };
        let slippage = self.slippage.expected_slippage_bps(qty_fraction, side_mult, lob_depth_ratio);

        Some((rtt_with_jitter, slippage))
    }

    /// Run N simulations and collect statistics
    pub fn run_simulation(&mut self, n: usize, qty_fraction: f64) -> Vec<(Nanos, f64)> {
        (0..n).filter_map(|_| self.simulate_order(qty_fraction, true, 1.0)).collect()
    }

    pub fn summary(&self) -> LatencySummary { self.histogram.summary() }
    pub fn sample_count(&self) -> u64 { self.sample_count }
}

// ── Co-location vs retail comparison ─────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ProfileComparison {
    pub profile: LatencyProfile,
    pub summary: LatencySummary,
    pub slippage_bps_mean: f64,
    pub fill_rate: f64, // fraction of orders that got filled (not lost)
}

pub fn compare_profiles(profiles: &[LatencyProfile], n_samples: usize, seed: u64) -> Vec<ProfileComparison> {
    profiles.iter().map(|profile| {
        let mut sim = LatencySimulator::new(profile.clone(), seed);
        let results = sim.run_simulation(n_samples, 0.001);
        let n_fills = results.len();
        let fill_rate = n_fills as f64 / n_samples as f64;
        let slippage_mean = if results.is_empty() { 0.0 }
            else { results.iter().map(|(_, s)| s).sum::<f64>() / n_fills as f64 };
        ProfileComparison {
            profile: profile.clone(),
            summary: sim.summary(),
            slippage_bps_mean: slippage_mean,
            fill_rate,
        }
    }).collect()
}

// ── Throughput counter ───────────────────────────────────────────────────────

pub struct ThroughputCounter {
    counts: VecDeque<(std::time::Instant, u64)>,
    window_secs: u64,
    total: u64,
}

impl ThroughputCounter {
    pub fn new(window_secs: u64) -> Self {
        ThroughputCounter { counts: VecDeque::new(), window_secs, total: 0 }
    }

    pub fn record(&mut self, count: u64) {
        let now = std::time::Instant::now();
        self.counts.push_back((now, count));
        self.total += count;
        // prune old
        let cutoff = now - std::time::Duration::from_secs(self.window_secs);
        while self.counts.front().map_or(false, |(t, _)| *t < cutoff) {
            self.counts.pop_front();
        }
    }

    pub fn rate_per_second(&self) -> f64 {
        if self.counts.len() < 2 { return 0.0; }
        let window_count: u64 = self.counts.iter().map(|(_, c)| c).sum();
        window_count as f64 / self.window_secs as f64
    }

    pub fn total(&self) -> u64 { self.total }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_normal() {
        let mut rng = Rng::new(42);
        let samples: Vec<f64> = (0..1000).map(|_| rng.next_normal()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.2, "mean = {}", mean);
    }

    #[test]
    fn test_gaussian_distribution_positive() {
        let dist = LatencyDistribution::gaussian(1000.0, 500.0, 100.0);
        let mut rng = Rng::new(1);
        for _ in 0..100 {
            let sample = dist.sample(&mut rng);
            assert!(sample >= 0);
        }
    }

    #[test]
    fn test_lognormal_mean() {
        let dist = LatencyDistribution::lognormal(0.0, 1000.0, 200.0);
        let mut rng = Rng::new(7);
        let samples: Vec<u64> = (0..1000).map(|_| dist.sample(&mut rng)).collect();
        let mean = samples.iter().sum::<u64>() as f64 / samples.len() as f64;
        // should be roughly near the target mean (loose check due to lognormal spread)
        assert!(mean > 500.0 && mean < 5000.0, "mean={}", mean);
    }

    #[test]
    fn test_latency_histogram() {
        let mut h = LatencyHistogram::new(1_000_000_000);
        for i in 0..1000u64 {
            h.record(i * 1000);
        }
        let p50 = h.p50();
        let p99 = h.p99();
        assert!(p50 < p99, "p50={} p99={}", p50, p99);
        assert!(h.count() == 1000);
    }

    #[test]
    fn test_histogram_percentile_monotone() {
        let mut h = LatencyHistogram::new(10_000_000);
        let mut rng = Rng::new(99);
        for _ in 0..10000 {
            h.record((rng.next_f64() * 1_000_000.0) as u64);
        }
        let p50 = h.p50();
        let p95 = h.p95();
        let p99 = h.p99();
        let p999 = h.p999();
        assert!(p50 <= p95, "p50={} p95={}", p50, p95);
        assert!(p95 <= p99, "p95={} p99={}", p95, p99);
        assert!(p99 <= p999, "p99={} p999={}", p99, p999);
    }

    #[test]
    fn test_network_model_co_location() {
        let mut model = NetworkModel::new(LatencyProfile::CoLocation, 42);
        let rtt = model.sample_round_trip();
        assert!(rtt.is_some());
        let ns = rtt.unwrap();
        assert!(ns > 0 && ns < 10_000_000, "rtt={}ns", ns);
    }

    #[test]
    fn test_slippage_model() {
        let mut model = SlippageModel::new(SlippageParams::default());
        for _ in 0..20 { model.update_vol(0.001); }
        let slippage = model.expected_slippage_bps(0.01, 1.0, 1.0);
        assert!(slippage > 0.0);
    }

    #[test]
    fn test_jitter_model_mean_zero() {
        let mut jitter = JitterModel::new(0.5, 200.0, 42);
        let samples: Vec<f64> = (0..1000).map(|_| jitter.next_jitter_ns()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 200.0, "mean={}", mean);
    }

    #[test]
    fn test_latency_simulator_end_to_end() {
        let mut sim = LatencySimulator::new(LatencyProfile::DirectMarketAccess, 42);
        let results = sim.run_simulation(1000, 0.001);
        assert!(!results.is_empty());
        let summary = sim.summary();
        assert!(summary.p99_ns > summary.p50_ns, "p50={} p99={}", summary.p50_ns, summary.p99_ns);
    }

    #[test]
    fn test_throughput_counter() {
        let mut tc = ThroughputCounter::new(10);
        tc.record(100);
        tc.record(200);
        assert_eq!(tc.total(), 300);
    }

    #[test]
    fn test_atomic_latency_counter() {
        let counter = AtomicLatencyCounter::new();
        for i in 0..100u64 {
            counter.record(i * 1000);
        }
        assert_eq!(counter.count(), 100);
        assert!(counter.max() > counter.min());
        assert!(counter.mean() > 0.0);
    }

    #[test]
    fn test_profile_comparison() {
        let profiles = vec![LatencyProfile::CoLocation, LatencyProfile::Retail];
        let comparisons = compare_profiles(&profiles, 500, 42);
        assert_eq!(comparisons.len(), 2);
        // co-location should be faster
        let colo = comparisons.iter().find(|c| c.profile == LatencyProfile::CoLocation).unwrap();
        let retail = comparisons.iter().find(|c| c.profile == LatencyProfile::Retail).unwrap();
        assert!(colo.summary.p99_ns < retail.summary.p99_ns,
                "co-lo p99={}ns, retail p99={}ns", colo.summary.p99_ns, retail.summary.p99_ns);
    }
}
