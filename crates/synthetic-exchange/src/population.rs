/// Population of heterogeneous agents with configurable mix and population dynamics.
///
/// Agent types:
/// - MarketMaker: Avellaneda-Stoikov optimal quoting
/// - Arbitrageur: Statistical arbitrage between instrument pairs
/// - MomentumTrader: EWMA trend following
/// - NoiseTrader: Random order submission (simulates uninformed flow)
///
/// Population dynamics:
/// - Fitness-based selection (agents with poor PnL reduced in activity)
/// - Mutation of agent parameters
/// - Performance attribution and cross-agent statistics

use std::collections::HashMap;
use crate::exchange::{Exchange, Order, Fill, Side, OrderKind, TimeInForce,
                     OrderId, InstrumentId, AgentId, Qty, Price, Nanos,
                     InstrumentConfig, MarketDataSnapshot};
use crate::market_maker::{MarketMakerAgent, AvellanedaStoikovParams, MarketMakerSummary};
use crate::momentum_agent::{MomentumAgent, MomentumConfig};
use crate::arbitrageur::{ArbitrageAgent, ArbitrageConfig};

// ── Noise Trader ──────────────────────────────────────────────────────────────

pub struct NoiseTrader {
    pub agent_id: AgentId,
    pub instrument_id: InstrumentId,
    /// Probability of submitting an order each step.
    pub activity_prob: f64,
    /// Order size distribution: uniform [min_qty, max_qty].
    pub min_qty: Qty,
    pub max_qty: Qty,
    /// Market order fraction.
    pub market_fraction: f64,
    /// Max price offset for limit orders (in ticks).
    pub max_offset_ticks: u32,
    /// Tick size.
    pub tick_size: f64,
    pub position: Qty,
    pub cash: f64,
    pub fill_count: u64,
    order_id_base: u64,
    order_id_ctr: u64,
    rng_state: u64,  // Simple LCG state.
}

impl NoiseTrader {
    pub fn new(agent_id: AgentId, instrument_id: InstrumentId, seed: u64) -> Self {
        NoiseTrader {
            agent_id,
            instrument_id,
            activity_prob: 0.3,
            min_qty: 10.0,
            max_qty: 100.0,
            market_fraction: 0.4,
            max_offset_ticks: 3,
            tick_size: 0.01,
            position: 0.0,
            cash: 100_000.0,
            fill_count: 0,
            order_id_base: (agent_id as u64) * 400_000_000,
            order_id_ctr: 0,
            rng_state: seed,
        }
    }

    fn lcg(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_order_id(&mut self) -> OrderId {
        self.order_id_ctr += 1;
        self.order_id_base + self.order_id_ctr
    }

    pub fn step(&mut self, mid: Price, ts_ns: Nanos) -> Vec<Order> {
        if self.lcg() > self.activity_prob { return vec![]; }

        let is_buy = self.lcg() < 0.5;
        let side = if is_buy { Side::Buy } else { Side::Sell };
        let qty = self.min_qty + self.lcg() * (self.max_qty - self.min_qty);
        let is_market = self.lcg() < self.market_fraction;

        let order = if is_market {
            Order::new_market(self.next_order_id(), self.instrument_id, self.agent_id, side, qty, ts_ns)
        } else {
            let offset = (self.lcg() * self.max_offset_ticks as f64) as u32;
            let price = match side {
                Side::Buy => mid - offset as f64 * self.tick_size,
                Side::Sell => mid + offset as f64 * self.tick_size,
            };
            let price = (price / self.tick_size).round() * self.tick_size;
            let mut o = Order::new_limit(self.next_order_id(), self.instrument_id, self.agent_id, side, price, qty, ts_ns);
            o.tif = TimeInForce::IOC; // noise traders often cancel quickly
            o
        };

        vec![order]
    }

    pub fn on_fill(&mut self, fill: &Fill) {
        if fill.aggressor_agent != self.agent_id && fill.passive_agent != self.agent_id { return; }
        let our_side = if fill.passive_agent == self.agent_id {
            match fill.side { Side::Buy => Side::Sell, Side::Sell => Side::Buy }
        } else { fill.side };

        match our_side {
            Side::Buy => {
                self.position += fill.qty;
                self.cash -= fill.price * fill.qty;
            }
            Side::Sell => {
                self.position -= fill.qty;
                self.cash += fill.price * fill.qty;
            }
        }
        self.fill_count += 1;
    }

    pub fn mark_to_market(&self, mid: Price) -> f64 {
        self.cash + self.position * mid
    }
}

// ── Agent Enum ────────────────────────────────────────────────────────────────

pub enum AgentVariant {
    MarketMaker(MarketMakerAgent),
    Momentum(MomentumAgent),
    Noise(NoiseTrader),
    // Arbitrageur handled separately due to multi-instrument nature.
}

impl AgentVariant {
    pub fn agent_id(&self) -> AgentId {
        match self {
            AgentVariant::MarketMaker(a) => a.agent_id,
            AgentVariant::Momentum(a) => a.agent_id,
            AgentVariant::Noise(a) => a.agent_id,
        }
    }

    pub fn step(&mut self, mid: Price, ts_ns: Nanos, tick_size: f64) -> Vec<Order> {
        match self {
            AgentVariant::MarketMaker(a) => {
                let (orders, _cancels) = a.step(mid, ts_ns, tick_size);
                orders
            }
            AgentVariant::Momentum(a) => a.step(mid, ts_ns),
            AgentVariant::Noise(a) => a.step(mid, ts_ns),
        }
    }

    pub fn on_fill(&mut self, fill: &Fill) {
        match self {
            AgentVariant::MarketMaker(a) => a.on_fill(fill),
            AgentVariant::Momentum(a) => a.on_fill(fill),
            AgentVariant::Noise(a) => a.on_fill(fill),
        }
    }

    pub fn mark_to_market(&self, mid: Price) -> f64 {
        match self {
            AgentVariant::MarketMaker(a) => a.mark_to_market(mid),
            AgentVariant::Momentum(a) => a.mark_to_market(mid),
            AgentVariant::Noise(a) => a.mark_to_market(mid),
        }
    }

    pub fn agent_type(&self) -> &'static str {
        match self {
            AgentVariant::MarketMaker(_) => "MarketMaker",
            AgentVariant::Momentum(_) => "Momentum",
            AgentVariant::Noise(_) => "Noise",
        }
    }
}

// ── Population Config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PopulationConfig {
    pub n_market_makers: usize,
    pub n_momentum: usize,
    pub n_noise: usize,
    pub n_arb_pairs: usize,
    /// Whether to run population dynamics (fitness-based selection).
    pub enable_dynamics: bool,
    /// How often to run population dynamics (in steps).
    pub dynamics_interval: usize,
    /// Fraction of poorly-performing agents to reduce activity.
    pub selection_pressure: f64,
    /// Initial cash for each agent type.
    pub initial_cash: f64,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        PopulationConfig {
            n_market_makers: 3,
            n_momentum: 5,
            n_noise: 10,
            n_arb_pairs: 0,
            enable_dynamics: false,
            dynamics_interval: 1000,
            selection_pressure: 0.1,
            initial_cash: 1_000_000.0,
        }
    }
}

// ── Performance Record ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AgentPerformance {
    pub agent_id: AgentId,
    pub agent_type: String,
    pub mtm: f64,
    pub fill_count: u64,
    pub fitness_score: f64,
}

// ── Population ────────────────────────────────────────────────────────────────

pub struct Population {
    pub config: PopulationConfig,
    pub agents: Vec<AgentVariant>,
    pub arb_agents: Vec<ArbitrageAgent>,
    step_count: usize,
    id_counter: u32,
    pub performance_history: Vec<Vec<AgentPerformance>>,
    rng_state: u64,
}

impl Population {
    pub fn new(config: PopulationConfig, instrument_id: InstrumentId, seed: u64) -> Self {
        let mut pop = Population {
            config: config.clone(),
            agents: Vec::new(),
            arb_agents: Vec::new(),
            step_count: 0,
            id_counter: 0,
            performance_history: Vec::new(),
            rng_state: seed,
        };

        let initial_cash = config.initial_cash;
        let session_start = 0u64;

        // Create market makers.
        for i in 0..config.n_market_makers {
            let id = pop.next_id();
            let params = AvellanedaStoikovParams {
                gamma: 0.05 + (i as f64) * 0.02,
                sigma: 0.2,
                kappa: 1.5,
                horizon: 3600.0,
                max_inventory: 1000.0,
                order_qty: 50.0 + (i as f64) * 10.0,
                requote_interval_secs: 1.0 + (i as f64) * 0.5,
                ..Default::default()
            };
            let mm = MarketMakerAgent::new(id, instrument_id, params, initial_cash, session_start);
            pop.agents.push(AgentVariant::MarketMaker(mm));
        }

        // Create momentum traders.
        for i in 0..config.n_momentum {
            let id = pop.next_id();
            let cfg = MomentumConfig {
                fast_halflife: 3.0 + (i as f64) * 2.0,
                slow_halflife: 15.0 + (i as f64) * 5.0,
                max_position: 500.0,
                base_qty: 50.0 + (i as f64) * 10.0,
                warmup_periods: 20 + i * 5,
                ..Default::default()
            };
            let mom = MomentumAgent::new(id, instrument_id, cfg, initial_cash);
            pop.agents.push(AgentVariant::Momentum(mom));
        }

        // Create noise traders.
        for i in 0..config.n_noise {
            let id = pop.next_id();
            let mut noise = NoiseTrader::new(id, instrument_id, seed.wrapping_add(i as u64 * 0x9e3779b9));
            noise.activity_prob = 0.2 + (i as f64 * 0.05).min(0.6);
            pop.agents.push(AgentVariant::Noise(noise));
        }

        pop
    }

    fn next_id(&mut self) -> AgentId {
        self.id_counter += 1;
        self.id_counter
    }

    fn lcg(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.rng_state >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn n_agents(&self) -> usize {
        self.agents.len() + self.arb_agents.len()
    }

    /// Run one simulation step. Agents observe market data and submit orders.
    /// Returns all orders generated this step.
    pub fn step(
        &mut self,
        snapshot: &MarketDataSnapshot,
        tick_size: f64,
        ts_ns: Nanos,
    ) -> Vec<Order> {
        self.step_count += 1;
        let mid = snapshot.mid.unwrap_or(snapshot.last_price);

        let mut all_orders = Vec::new();

        for agent in self.agents.iter_mut() {
            let orders = agent.step(mid, ts_ns, tick_size);
            all_orders.extend(orders);
        }

        // Population dynamics.
        if self.config.enable_dynamics
            && self.step_count % self.config.dynamics_interval == 0
        {
            self.run_dynamics(mid);
        }

        all_orders
    }

    /// Broadcast fills to all relevant agents.
    pub fn on_fills(&mut self, fills: &[Fill]) {
        for fill in fills {
            for agent in self.agents.iter_mut() {
                if agent.agent_id() == fill.aggressor_agent
                    || agent.agent_id() == fill.passive_agent
                {
                    agent.on_fill(fill);
                }
            }
            for arb in self.arb_agents.iter_mut() {
                if arb.agent_id == fill.aggressor_agent || arb.agent_id == fill.passive_agent {
                    arb.on_fill(fill);
                }
            }
        }
    }

    /// Run fitness-based population dynamics.
    fn run_dynamics(&mut self, mid: Price) {
        let performances: Vec<(usize, f64)> = self.agents.iter().enumerate()
            .map(|(i, a)| (i, a.mark_to_market(mid)))
            .collect();

        // Sort by MTM ascending (worst first).
        let mut sorted = performances.clone();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Reduce activity of bottom performers (simplistic dynamics).
        let n_reduce = ((self.agents.len() as f64 * self.config.selection_pressure) as usize).max(1);
        for (i, _mtm) in sorted.iter().take(n_reduce) {
            if let AgentVariant::Noise(ref mut noise) = self.agents[*i] {
                noise.activity_prob = (noise.activity_prob * 0.9).max(0.05);
            }
        }
    }

    /// Performance snapshot of all agents.
    pub fn performance_snapshot(&self, mid: Price) -> Vec<AgentPerformance> {
        let mut perfs: Vec<AgentPerformance> = self.agents.iter().map(|a| {
            let mtm = a.mark_to_market(mid);
            let fills = match a {
                AgentVariant::MarketMaker(mm) => mm.position.total_fills,
                AgentVariant::Momentum(mom) => mom.total_fills,
                AgentVariant::Noise(noise) => noise.fill_count,
            };
            AgentPerformance {
                agent_id: a.agent_id(),
                agent_type: a.agent_type().to_string(),
                mtm,
                fill_count: fills,
                fitness_score: mtm / self.config.initial_cash - 1.0,
            }
        }).collect();

        // Sort by fitness descending.
        perfs.sort_by(|a, b| b.fitness_score.partial_cmp(&a.fitness_score).unwrap_or(std::cmp::Ordering::Equal));
        perfs
    }

    /// Aggregate statistics across all agents.
    pub fn aggregate_stats(&self, mid: Price) -> PopulationStats {
        let perfs = self.performance_snapshot(mid);
        let n = perfs.len() as f64;
        if n < 1.0 { return PopulationStats::default(); }

        let total_mtm: f64 = perfs.iter().map(|p| p.mtm).sum();
        let mean_mtm = total_mtm / n;
        let variance_mtm = perfs.iter().map(|p| (p.mtm - mean_mtm).powi(2)).sum::<f64>() / n;

        let total_fills: u64 = perfs.iter().map(|p| p.fill_count).sum();
        let positive = perfs.iter().filter(|p| p.fitness_score > 0.0).count();

        let by_type = {
            let mut map: HashMap<String, (usize, f64)> = HashMap::new();
            for p in &perfs {
                let e = map.entry(p.agent_type.clone()).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += p.mtm;
            }
            map
        };

        PopulationStats {
            n_agents: perfs.len(),
            mean_mtm,
            std_mtm: variance_mtm.sqrt(),
            total_fills,
            pct_positive: positive as f64 / n,
            by_type,
            step: self.step_count,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PopulationStats {
    pub n_agents: usize,
    pub mean_mtm: f64,
    pub std_mtm: f64,
    pub total_fills: u64,
    pub pct_positive: f64,
    pub by_type: HashMap<String, (usize, f64)>,
    pub step: usize,
}

// ── Full Simulation Runner ────────────────────────────────────────────────────

pub struct PopulationSimulation {
    pub exchange: Exchange,
    pub population: Population,
    pub instrument_id: InstrumentId,
    pub tick_size: f64,
    pub step_count: u64,
    pub price_path: Vec<f64>,
}

impl PopulationSimulation {
    pub fn new(
        instrument_config: InstrumentConfig,
        pop_config: PopulationConfig,
        tick_size: f64,
        seed: u64,
    ) -> Self {
        let inst_id = instrument_config.id;
        let mut exchange = Exchange::new();
        exchange.add_instrument(instrument_config);

        let population = Population::new(pop_config, inst_id, seed);

        PopulationSimulation {
            exchange,
            population,
            instrument_id: inst_id,
            tick_size,
            step_count: 0,
            price_path: Vec::new(),
        }
    }

    /// Run N simulation steps. Returns price path and population stats.
    pub fn run(
        &mut self,
        n_steps: usize,
        initial_mid: Price,
        ts_start_ns: Nanos,
        step_duration_ns: Nanos,
    ) -> SimulationResult {
        let mut ts = ts_start_ns;
        let mut all_perf = Vec::new();

        // Seed the book with initial quotes.
        self.seed_book(initial_mid, ts);

        for step in 0..n_steps {
            ts += step_duration_ns;
            self.step_count += 1;

            // Get market snapshot.
            let snap = self.exchange.snapshot(self.instrument_id)
                .unwrap_or_else(|| self.default_snapshot(initial_mid, ts));

            // Get mid price.
            let mid = snap.mid.unwrap_or(snap.last_price);
            if mid > 1e-9 { self.price_path.push(mid); }

            // Population step.
            let orders = self.population.step(&snap, self.tick_size, ts);

            // Submit orders to exchange.
            let mut fills = Vec::new();
            for order in orders {
                let f = self.exchange.submit(order, ts);
                fills.extend(f);
            }

            // Broadcast fills.
            self.population.on_fills(&fills);

            // Drain exchange fills (dedup).
            self.exchange.drain_fills();

            // Replenish thin books.
            if step % 10 == 0 {
                self.replenish_book(mid, ts);
            }

            // Collect performance snapshot.
            if step % 100 == 0 && mid > 1e-9 {
                let perf = self.population.performance_snapshot(mid);
                all_perf.push((step, perf));
            }
        }

        let final_mid = self.price_path.last().copied().unwrap_or(initial_mid);
        let final_stats = self.population.aggregate_stats(final_mid);

        SimulationResult {
            price_path: self.price_path.clone(),
            final_stats,
            performance_snapshots: all_perf,
            n_steps,
        }
    }

    fn seed_book(&mut self, mid: Price, ts_ns: Nanos) {
        let tick = self.tick_size;
        let inst = self.instrument_id;
        let mut id = 900_000_000u64;
        for i in 1..=5u32 {
            let bid = Order::new_limit(id, inst, 0, Side::Buy, mid - i as f64 * tick, 200.0, ts_ns);
            id += 1;
            let ask = Order::new_limit(id, inst, 0, Side::Sell, mid + i as f64 * tick, 200.0, ts_ns);
            id += 1;
            self.exchange.submit(bid, ts_ns);
            self.exchange.submit(ask, ts_ns);
        }
    }

    fn replenish_book(&mut self, mid: Price, ts_ns: Nanos) {
        let snap = match self.exchange.snapshot(self.instrument_id) {
            Some(s) => s,
            None => return,
        };
        let tick = self.tick_size;
        let inst = self.instrument_id;
        let mut id = 990_000_000u64 + self.step_count as u64;

        if snap.bid_depth.len() < 2 {
            for i in 1..=3u32 {
                let b = Order::new_limit(id, inst, 0, Side::Buy, mid - i as f64 * tick, 100.0, ts_ns);
                id += 1;
                self.exchange.submit(b, ts_ns);
            }
        }
        if snap.ask_depth.len() < 2 {
            for i in 1..=3u32 {
                let a = Order::new_limit(id, inst, 0, Side::Sell, mid + i as f64 * tick, 100.0, ts_ns);
                id += 1;
                self.exchange.submit(a, ts_ns);
            }
        }
    }

    fn default_snapshot(&self, mid: Price, ts_ns: Nanos) -> MarketDataSnapshot {
        MarketDataSnapshot {
            instrument_id: self.instrument_id,
            timestamp_ns: ts_ns,
            best_bid: Some(mid - self.tick_size),
            best_ask: Some(mid + self.tick_size),
            mid: Some(mid),
            spread: Some(self.tick_size * 2.0),
            last_price: mid,
            last_qty: 0.0,
            bid_depth: vec![(mid - self.tick_size, 100.0)],
            ask_depth: vec![(mid + self.tick_size, 100.0)],
            total_bid_qty: 100.0,
            total_ask_qty: 100.0,
            imbalance: 0.0,
            session_volume: 0.0,
            session_vwap: mid,
        }
    }
}

#[derive(Debug)]
pub struct SimulationResult {
    pub price_path: Vec<f64>,
    pub final_stats: PopulationStats,
    pub performance_snapshots: Vec<(usize, Vec<AgentPerformance>)>,
    pub n_steps: usize,
}

impl SimulationResult {
    pub fn price_return(&self) -> f64 {
        if self.price_path.len() < 2 { return 0.0; }
        let p0 = self.price_path[0];
        let p1 = self.price_path.last().unwrap();
        if p0 < 1e-9 { return 0.0; }
        (p1 / p0).ln()
    }

    pub fn realized_vol(&self) -> f64 {
        if self.price_path.len() < 2 { return 0.0; }
        let n = self.price_path.len() - 1;
        let returns: Vec<f64> = self.price_path.windows(2)
            .map(|w| if w[0] > 1e-9 { (w[1] / w[0]).ln() } else { 0.0 })
            .collect();
        let mean = returns.iter().sum::<f64>() / n as f64;
        let var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snap(mid: Price, ts: Nanos) -> MarketDataSnapshot {
        MarketDataSnapshot {
            instrument_id: 1,
            timestamp_ns: ts,
            best_bid: Some(mid - 0.01),
            best_ask: Some(mid + 0.01),
            mid: Some(mid),
            spread: Some(0.02),
            last_price: mid,
            last_qty: 100.0,
            bid_depth: vec![(mid - 0.01, 100.0), (mid - 0.02, 200.0)],
            ask_depth: vec![(mid + 0.01, 100.0), (mid + 0.02, 200.0)],
            total_bid_qty: 300.0,
            total_ask_qty: 300.0,
            imbalance: 0.0,
            session_volume: 0.0,
            session_vwap: mid,
        }
    }

    #[test]
    fn test_noise_trader_step() {
        let mut noise = NoiseTrader::new(1, 1, 42);
        noise.activity_prob = 1.0; // Always active.
        let orders = noise.step(100.0, 0);
        assert!(!orders.is_empty());
    }

    #[test]
    fn test_population_creation() {
        let config = PopulationConfig {
            n_market_makers: 2,
            n_momentum: 3,
            n_noise: 5,
            ..Default::default()
        };
        let pop = Population::new(config.clone(), 1, 42);
        assert_eq!(pop.n_agents(), config.n_market_makers + config.n_momentum + config.n_noise);
    }

    #[test]
    fn test_population_step() {
        let config = PopulationConfig {
            n_market_makers: 1,
            n_momentum: 1,
            n_noise: 2,
            ..Default::default()
        };
        let mut pop = Population::new(config, 1, 99);
        let snap = make_snap(100.0, 1_000_000_000);

        // Step 50 times to warm up agents.
        let mut ts = 1_000_000_000u64;
        for _ in 0..50 {
            pop.step(&snap, 0.01, ts);
            ts += 1_000_000_000;
        }
        // No panic = success.
        assert!(pop.step_count >= 50);
    }

    #[test]
    fn test_performance_snapshot() {
        let config = PopulationConfig {
            n_market_makers: 2,
            n_noise: 3,
            ..Default::default()
        };
        let pop = Population::new(config, 1, 7);
        let perfs = pop.performance_snapshot(100.0);
        assert_eq!(perfs.len(), 5);
    }

    #[test]
    fn test_simulation_run() {
        let inst_config = InstrumentConfig::default_equity(1, "TEST", 100.0);
        let pop_config = PopulationConfig {
            n_market_makers: 1,
            n_noise: 2,
            n_momentum: 1,
            ..Default::default()
        };
        let mut sim = PopulationSimulation::new(inst_config, pop_config, 0.01, 42);
        let result = sim.run(100, 100.0, 0, 1_000_000_000);
        assert_eq!(result.n_steps, 100);
        assert!(!result.price_path.is_empty());
    }

    #[test]
    fn test_population_dynamics() {
        let config = PopulationConfig {
            n_noise: 10,
            enable_dynamics: true,
            dynamics_interval: 5,
            selection_pressure: 0.2,
            ..Default::default()
        };
        let mut pop = Population::new(config, 1, 13);
        let snap = make_snap(100.0, 0);
        for i in 0..20u64 {
            pop.step(&snap, 0.01, i * 1_000_000_000);
        }
        // No panic = success. Dynamics ran at steps 5, 10, 15, 20.
        assert!(pop.step_count == 20);
    }
}
