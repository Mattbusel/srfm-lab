// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// sim_engine.rs — Full simulation engine for AETERNUS pipeline testing
// =============================================================================
//! Provides end-to-end simulation:
//! - `SimMarket` — multi-asset GBM + GARCH with correlated shocks
//! - `SimLOB` — simulated limit order book per asset
//! - `SimPipeline` — drives data through full RTEL pipeline
//! - Performance benchmarking utilities

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::now_ns;
use crate::metrics::global_metrics;
use crate::order_book::{LimitOrderBook, MultiAssetLOB, OrderEvent, OrderAction, Side};

// ---------------------------------------------------------------------------
// Random number generation (xorshift64 + Box-Muller)
// ---------------------------------------------------------------------------

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn next_normal_pair(&mut self) -> (f64, f64) {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        let r  = (-2.0 * u1.ln()).sqrt();
        let t  = 2.0 * std::f64::consts::PI * u2;
        (r * t.cos(), r * t.sin())
    }
}

// ---------------------------------------------------------------------------
// GARCH(1,1) volatility process
// ---------------------------------------------------------------------------

pub struct GARCH11 {
    pub omega: f64,
    pub alpha: f64,
    pub beta:  f64,
    pub var:   f64,   // current conditional variance
}

impl GARCH11 {
    pub fn new(long_run_var: f64) -> Self {
        let omega = long_run_var * 0.05;
        let alpha = 0.10;
        let beta  = 0.85;
        Self { omega, alpha, beta, var: long_run_var }
    }

    pub fn update(&mut self, shock: f64) -> f64 {
        self.var = self.omega + self.alpha * shock * shock + self.beta * self.var;
        self.var
    }

    pub fn std_dev(&self) -> f64 { self.var.sqrt() }
}

// ---------------------------------------------------------------------------
// SimAsset — single-asset GBM + GARCH simulation
// ---------------------------------------------------------------------------

pub struct SimAsset {
    pub asset_id:   u32,
    pub price:      f64,
    pub mu:         f64,
    pub sigma_long: f64,
    garch:          GARCH11,
    rng:            Rng,
    pub volume:     f64,
    pub n_ticks:    u64,
}

impl SimAsset {
    pub fn new(asset_id: u32, initial_price: f64, mu: f64, sigma: f64) -> Self {
        let long_run_var = sigma * sigma;
        Self {
            asset_id,
            price: initial_price,
            mu,
            sigma_long: sigma,
            garch:  GARCH11::new(long_run_var),
            rng:    Rng::new(12345678 + asset_id as u64 * 987654321),
            volume: 0.0,
            n_ticks: 0,
        }
    }

    pub fn step(&mut self, dt: f64) -> f64 {
        let z   = self.rng.next_normal();
        let std = self.garch.std_dev();
        self.garch.update(z * std);

        let drift = (self.mu - 0.5 * std * std) * dt;
        let shock = std * dt.sqrt() * z;
        self.price *= (drift + shock).exp();
        self.price = self.price.max(0.001);

        let vol_noise = self.rng.next_f64();
        self.volume += 1000.0 * (0.5 + vol_noise) * std / self.sigma_long;
        self.n_ticks += 1;
        self.price
    }

    pub fn current_vol(&self) -> f64 { self.garch.std_dev() }
}

// ---------------------------------------------------------------------------
// Correlated multi-asset simulation
// ---------------------------------------------------------------------------

pub struct SimMarket {
    n_assets:   usize,
    assets:     Vec<SimAsset>,
    cholesky:   Vec<Vec<f64>>,  // lower triangular
    master_rng: Rng,
    pub step_count: u64,
}

impl SimMarket {
    pub fn new(n_assets: usize, base_price: f64, mu: f64, sigma: f64,
               correlation: f64, seed: u64) -> Self
    {
        let assets: Vec<SimAsset> = (0..n_assets).map(|i| {
            let price = base_price * (1.0 + 0.1 * i as f64);
            let sigma_i = sigma * (1.0 + 0.2 * ((i % 3) as f64));
            SimAsset::new(i as u32, price, mu, sigma_i)
        }).collect();

        // Cholesky decomposition of correlation matrix (factor model)
        // C = rho * 11' + (1-rho) * I
        let mut chol = vec![vec![0.0f64; n_assets]; n_assets];
        let sqrt_rho = correlation.sqrt();
        let sqrt_1mr = (1.0 - correlation).sqrt();
        for i in 0..n_assets {
            chol[i][0] = sqrt_rho;
            chol[i][i] = if i == 0 { sqrt_1mr } else { sqrt_1mr };
        }

        Self {
            n_assets,
            assets,
            cholesky: chol,
            master_rng: Rng::new(seed),
            step_count: 0,
        }
    }

    pub fn step(&mut self, dt: f64) -> Vec<f64> {
        // Generate correlated normals
        let common = self.master_rng.next_normal();
        let prices: Vec<f64> = self.assets.iter_mut().enumerate().map(|(i, asset)| {
            let idio = asset.rng.next_normal();
            let _rho = self.cholesky[i][0];
            // Use mixed common/idiosyncratic shock
            let z = self.cholesky[i][0] * common + self.cholesky[i][i] * idio;
            let std = asset.garch.std_dev();
            asset.garch.update(z * std);
            let drift = (asset.mu - 0.5 * std * std) * dt;
            let shock = std * dt.sqrt() * z;
            asset.price *= (drift + shock).exp();
            asset.price = asset.price.max(0.001);
            asset.n_ticks += 1;
            asset.price
        }).collect();
        self.step_count += 1;
        prices
    }

    pub fn prices(&self) -> Vec<f64> {
        self.assets.iter().map(|a| a.price).collect()
    }

    pub fn vols(&self) -> Vec<f64> {
        self.assets.iter().map(|a| a.current_vol()).collect()
    }

    pub fn n_assets(&self) -> usize { self.n_assets }
}

// ---------------------------------------------------------------------------
// SimLOB — simulated limit order book feed
// ---------------------------------------------------------------------------

pub struct SimLOBFeed {
    pub lob:       MultiAssetLOB,
    rng:           Rng,
    spread_bps:    f64,
    n_levels:      i64,
    level_size:    i64,   // scaled
}

impl SimLOBFeed {
    pub fn new(spread_bps: f64, n_levels: i64, seed: u64) -> Self {
        Self {
            lob:        MultiAssetLOB::new(1e-8, 1e-8),
            rng:        Rng::new(seed),
            spread_bps,
            n_levels,
            level_size: 100_000_000,  // 1.0 in 1e-8 scale
        }
    }

    pub fn seed_book(&mut self, asset_id: u32, mid_price: f64) {
        // Convert to integer (scaled ×1e8)
        let mid_int = (mid_price * 1e8) as i64;
        let spread_half = ((mid_price * self.spread_bps / 2e4) * 1e8) as i64;
        let tick = spread_half.max(1);

        for i in 1..=self.n_levels {
            let bid_ev = OrderEvent {
                order_id:  self.rng.next_u64(),
                timestamp: now_ns(),
                side:      Side::Bid,
                action:    OrderAction::Add,
                price:     mid_int - i * tick,
                size:      self.level_size + (self.rng.next_u64() as i64 % self.level_size),
                asset_id,
            };
            let ask_ev = OrderEvent {
                price:  mid_int + i * tick,
                side:   Side::Ask,
                ..bid_ev
            };
            self.lob.process(&bid_ev);
            self.lob.process(&ask_ev);
        }
    }

    pub fn update_mid(&mut self, asset_id: u32, new_mid: f64, old_mid: f64) {
        // Add new orders near new mid
        let new_mid_int = (new_mid * 1e8) as i64;
        let spread_half = ((new_mid * self.spread_bps / 2e4) * 1e8) as i64;
        let tick = spread_half.max(1);

        // Add a new level
        let add_bid = OrderEvent {
            order_id:  self.rng.next_u64(),
            timestamp: now_ns(),
            side:      Side::Bid,
            action:    OrderAction::Add,
            price:     new_mid_int - tick,
            size:      self.level_size,
            asset_id,
        };
        let add_ask = OrderEvent {
            side:  Side::Ask,
            price: new_mid_int + tick,
            ..add_bid
        };
        self.lob.process(&add_bid);
        self.lob.process(&add_ask);

        // Simulate a small trade
        if self.rng.next_f64() > 0.7 {
            let trade_side = if new_mid > old_mid { Side::Bid } else { Side::Ask };
            let trade_ev = OrderEvent {
                order_id:  self.rng.next_u64(),
                timestamp: now_ns(),
                side:      trade_side,
                action:    OrderAction::Trade,
                price:     if new_mid > old_mid { new_mid_int + tick } else { new_mid_int - tick },
                size:      self.level_size / 10,
                asset_id,
            };
            self.lob.process(&trade_ev);
        }
    }

    pub fn snapshot_all(&self) -> Vec<crate::state_publisher::LobSnapshot> {
        self.lob.all_snapshots()
    }
}

// ---------------------------------------------------------------------------
// SimPipeline — drives full RTEL simulation
// ---------------------------------------------------------------------------

pub struct SimPipelineConfig {
    pub n_assets:      usize,
    pub n_steps:       usize,
    pub base_price:    f64,
    pub mu:            f64,
    pub sigma:         f64,
    pub correlation:   f64,
    pub dt:            f64,
    pub spread_bps:    f64,
    pub seed:          u64,
}

impl Default for SimPipelineConfig {
    fn default() -> Self {
        Self {
            n_assets:    10,
            n_steps:     1000,
            base_price:  100.0,
            mu:          0.0,
            sigma:       0.01,
            correlation: 0.3,
            dt:          1.0 / 252.0,
            spread_bps:  5.0,
            seed:        42,
        }
    }
}

pub struct SimPipelineResult {
    pub n_steps:        usize,
    pub n_lob_updates:  u64,
    pub elapsed_us:     f64,
    pub throughput:     f64,   // LOB updates / second
    pub final_prices:  Vec<f64>,
    pub price_history: Vec<Vec<f64>>,  // [step][asset]
    pub vol_history:   Vec<Vec<f64>>,
}

pub struct SimPipeline {
    config:   SimPipelineConfig,
    market:   SimMarket,
    lob_feed: SimLOBFeed,
}

impl SimPipeline {
    pub fn new(config: SimPipelineConfig) -> Self {
        let market = SimMarket::new(
            config.n_assets, config.base_price, config.mu, config.sigma,
            config.correlation, config.seed,
        );
        let mut lob_feed = SimLOBFeed::new(config.spread_bps, 5, config.seed + 1);

        // Seed order books
        for i in 0..config.n_assets {
            let price = config.base_price * (1.0 + 0.1 * i as f64);
            lob_feed.seed_book(i as u32, price);
        }

        Self { config, market, lob_feed }
    }

    pub fn run(&mut self) -> SimPipelineResult {
        let t0 = Instant::now();
        let n  = self.config.n_steps;
        let na = self.config.n_assets;
        let dt = self.config.dt;

        let mut price_history = Vec::with_capacity(n);
        let mut vol_history   = Vec::with_capacity(n);
        let mut n_lob_updates = 0u64;
        let mut prev_prices   = self.market.prices();

        for _step in 0..n {
            // Step market
            let prices = self.market.step(dt);
            let vols   = self.market.vols();

            // Update LOB for each asset
            for i in 0..na {
                self.lob_feed.update_mid(i as u32, prices[i], prev_prices[i]);
                n_lob_updates += 1;
            }

            price_history.push(prices.clone());
            vol_history.push(vols);
            prev_prices = prices;

            // Update global metrics
            if _step % 100 == 0 {
                global_metrics().pipeline_runs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        let elapsed_us = t0.elapsed().as_micros() as f64;
        let throughput  = if elapsed_us > 0.0 {
            n_lob_updates as f64 / elapsed_us * 1_000_000.0
        } else {
            0.0
        };

        SimPipelineResult {
            n_steps:       n,
            n_lob_updates,
            elapsed_us,
            throughput,
            final_prices:  self.market.prices(),
            price_history,
            vol_history,
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

pub struct BenchmarkResult {
    pub name:        String,
    pub n_ops:       u64,
    pub elapsed_ns:  u64,
    pub ops_per_sec: f64,
    pub ns_per_op:   f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:.1}M ops/s ({:.1} ns/op)",
               self.name, self.ops_per_sec / 1e6, self.ns_per_op)
    }
}

pub fn benchmark<F: FnMut() -> ()>(name: &str, n_ops: u64, mut f: F) -> BenchmarkResult {
    // Warmup
    for _ in 0..100 { f(); }

    let t0 = Instant::now();
    for _ in 0..n_ops { f(); }
    let elapsed_ns = t0.elapsed().as_nanos() as u64;

    let ops_per_sec = if elapsed_ns > 0 {
        n_ops as f64 / elapsed_ns as f64 * 1e9
    } else { 0.0 };
    let ns_per_op = if n_ops > 0 { elapsed_ns as f64 / n_ops as f64 } else { 0.0 };

    BenchmarkResult {
        name: name.to_owned(),
        n_ops,
        elapsed_ns,
        ops_per_sec,
        ns_per_op,
    }
}

pub fn run_standard_benchmarks() -> Vec<BenchmarkResult> {
    let mut results = vec![];

    // Benchmark RNG
    let mut rng = Rng::new(42);
    results.push(benchmark("rng_next_normal", 1_000_000, || {
        std::hint::black_box(rng.next_normal());
    }));

    // Benchmark GARCH update
    let mut garch = GARCH11::new(0.01 * 0.01);
    results.push(benchmark("garch_update", 1_000_000, || {
        std::hint::black_box(garch.update(0.01));
    }));

    // Benchmark LOB process
    let mut lob = LimitOrderBook::new(0, 1e-8, 1e-8);
    let mut ev_rng = Rng::new(123);
    results.push(benchmark("lob_process_add", 100_000, || {
        let ev = OrderEvent {
            order_id:  ev_rng.next_u64(),
            timestamp: now_ns(),
            side:      if ev_rng.next_f64() > 0.5 { Side::Bid } else { Side::Ask },
            action:    OrderAction::Add,
            price:     (ev_rng.next_f64() * 1000.0) as i64 * 1_000_000,
            size:      100_000_000,
            asset_id:  0,
        };
        lob.process(&ev);
    }));

    // Benchmark market step
    let mut market = SimMarket::new(10, 100.0, 0.0, 0.01, 0.3, 0);
    results.push(benchmark("market_step_10_assets", 10_000, || {
        std::hint::black_box(market.step(1.0/252.0));
    }));

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_distribution() {
        let mut rng = Rng::new(42);
        let samples: Vec<f64> = (0..10000).map(|_| rng.next_normal()).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let var:  f64 = samples.iter().map(|&x| (x-mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1, "mean should be ~0, got {}", mean);
        assert!((var - 1.0).abs() < 0.1, "variance should be ~1, got {}", var);
    }

    #[test]
    fn test_garch_positive_variance() {
        let mut garch = GARCH11::new(0.0001);
        for _ in 0..100 {
            let v = garch.update(0.01);
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_sim_asset_positive_prices() {
        let mut asset = SimAsset::new(0, 100.0, 0.0, 0.01);
        for _ in 0..100 {
            let p = asset.step(1.0/252.0);
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_sim_market_all_positive() {
        let mut market = SimMarket::new(5, 100.0, 0.0, 0.01, 0.3, 42);
        for _ in 0..100 {
            let prices = market.step(1.0/252.0);
            assert!(prices.iter().all(|&p| p > 0.0));
        }
    }

    #[test]
    fn test_sim_pipeline_runs() {
        let config = SimPipelineConfig {
            n_assets: 3,
            n_steps:  50,
            ..Default::default()
        };
        let mut pipeline = SimPipeline::new(config);
        let result = pipeline.run();
        assert_eq!(result.n_steps, 50);
        assert!(result.n_lob_updates > 0);
        assert!(result.throughput > 0.0);
        assert_eq!(result.final_prices.len(), 3);
        println!("SimPipeline: {} LOB updates in {:.0}µs = {:.0} ops/s",
                 result.n_lob_updates, result.elapsed_us, result.throughput);
    }

    #[test]
    fn test_sim_lob_feed_seeds() {
        let mut feed = SimLOBFeed::new(5.0, 5, 42);
        feed.seed_book(0, 100.0);
        let snaps = feed.snapshot_all();
        assert_eq!(snaps.len(), 1);
        let snap = &snaps[0];
        assert!(!snap.bids.is_empty(), "should have bid levels");
        assert!(!snap.asks.is_empty(), "should have ask levels");
    }

    #[test]
    fn test_benchmark_runs() {
        let results = run_standard_benchmarks();
        for r in &results {
            println!("{}", r);
            assert!(r.ops_per_sec > 0.0);
        }
    }

    #[test]
    fn test_market_correlated_returns() {
        let mut market = SimMarket::new(2, 100.0, 0.0, 0.02, 0.8, 7);
        let mut prices0 = vec![];
        let mut prices1 = vec![];
        for _ in 0..200 {
            let p = market.step(1.0/252.0);
            prices0.push(p[0]);
            prices1.push(p[1]);
        }
        // Compute correlation of returns
        let r0: Vec<f64> = prices0.windows(2).map(|w| (w[1]-w[0])/w[0]).collect();
        let r1: Vec<f64> = prices1.windows(2).map(|w| (w[1]-w[0])/w[0]).collect();
        let n  = r0.len() as f64;
        let m0 = r0.iter().sum::<f64>() / n;
        let m1 = r1.iter().sum::<f64>() / n;
        let cov: f64  = r0.iter().zip(r1.iter()).map(|(a,b)| (a-m0)*(b-m1)).sum::<f64>() / n;
        let v0:  f64  = r0.iter().map(|r| (r-m0).powi(2)).sum::<f64>() / n;
        let v1:  f64  = r1.iter().map(|r| (r-m1).powi(2)).sum::<f64>() / n;
        let corr = cov / (v0.sqrt() * v1.sqrt() + 1e-12);
        // With high correlation=0.8 in inputs, realized corr should be >0
        println!("Realized correlation: {:.3}", corr);
        assert!(corr > 0.0, "correlated assets should have positive return correlation");
    }
}
