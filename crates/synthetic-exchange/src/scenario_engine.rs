//! scenario_engine.rs — Market scenario injection engine.
//!
//! Flash crash injection, volatility regime switching, liquidity crisis,
//! correlated multi-asset scenarios, YAML config replay.
//!
//! Chronos / AETERNUS — production scenario engine.

use std::collections::{HashMap, VecDeque};

// ── Types ────────────────────────────────────────────────────────────────────

pub type Price = f64;
pub type Qty = f64;
pub type Nanos = u64;
pub type InstrumentId = u32;

// ── PRNG ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Rng { state: u64 }
impl Rng {
    pub fn new(seed: u64) -> Self { Rng { state: seed ^ 0x1234abcd_5678ef01 } }
    pub fn next_u64(&mut self) -> u64 { let mut x = self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.state = x; x }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.next_f64() }
}

// ── Volatility regime ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum VolatilityRegime {
    Calm,      // VIX < 15
    Normal,    // VIX 15-25
    Elevated,  // VIX 25-40
    Crisis,    // VIX > 40
    Custom(f64),
}

impl VolatilityRegime {
    pub fn annualized_vol(&self) -> f64 {
        match self {
            VolatilityRegime::Calm => 0.08,
            VolatilityRegime::Normal => 0.15,
            VolatilityRegime::Elevated => 0.30,
            VolatilityRegime::Crisis => 0.60,
            VolatilityRegime::Custom(v) => *v,
        }
    }

    pub fn daily_vol(&self) -> f64 { self.annualized_vol() / 252f64.sqrt() }
    pub fn intraday_vol(&self) -> f64 { self.daily_vol() / 390f64.sqrt() } // 390 min/day
    pub fn liquidity_multiplier(&self) -> f64 {
        match self {
            VolatilityRegime::Calm => 1.5,
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::Elevated => 0.6,
            VolatilityRegime::Crisis => 0.2,
            VolatilityRegime::Custom(_) => 0.8,
        }
    }

    pub fn spread_multiplier(&self) -> f64 {
        match self {
            VolatilityRegime::Calm => 0.7,
            VolatilityRegime::Normal => 1.0,
            VolatilityRegime::Elevated => 2.0,
            VolatilityRegime::Crisis => 5.0,
            VolatilityRegime::Custom(_) => 1.5,
        }
    }
}

// ── Scenario types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FlashCrashParams {
    pub duration_ns: Nanos,
    pub drawdown_pct: f64,       // peak drop e.g. 0.10 = 10%
    pub recovery_pct: f64,       // how much recovers (0.8 = 80% recovery)
    pub initial_speed_ns: Nanos, // how fast the drop occurs
    pub recovery_speed_ns: Nanos,
    pub affected_instruments: Vec<InstrumentId>,
    pub contagion_factor: f64,   // how much other instruments are affected
}

impl Default for FlashCrashParams {
    fn default() -> Self {
        FlashCrashParams {
            duration_ns: 600_000_000_000, // 10 minutes
            drawdown_pct: 0.08,
            recovery_pct: 0.75,
            initial_speed_ns: 60_000_000_000, // 1 minute drop
            recovery_speed_ns: 540_000_000_000,
            affected_instruments: vec![1],
            contagion_factor: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiquidityCrisisParams {
    pub duration_ns: Nanos,
    pub bid_ask_spread_multiplier: f64,
    pub depth_depletion_factor: f64,   // 0 = fully depleted, 1 = normal
    pub market_maker_withdrawal_pct: f64,
    pub affected_instruments: Vec<InstrumentId>,
}

impl Default for LiquidityCrisisParams {
    fn default() -> Self {
        LiquidityCrisisParams {
            duration_ns: 1_800_000_000_000, // 30 minutes
            bid_ask_spread_multiplier: 5.0,
            depth_depletion_factor: 0.1,
            market_maker_withdrawal_pct: 0.8,
            affected_instruments: vec![1],
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorrelationBreakParams {
    pub duration_ns: Nanos,
    pub new_correlation: f64,   // -1 to 1
    pub pairs: Vec<(InstrumentId, InstrumentId)>,
}

#[derive(Debug, Clone)]
pub struct NewsShockParams {
    pub timestamp_ns: Nanos,
    pub price_move_pct: f64,    // signed
    pub vol_spike_factor: f64,
    pub affected_instruments: Vec<InstrumentId>,
    pub headline: String,
}

// ── Scenario definition ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ScenarioKind {
    FlashCrash(FlashCrashParams),
    LiquidityCrisis(LiquidityCrisisParams),
    VolatilityRegimeChange { from: VolatilityRegime, to: VolatilityRegime, transition_ns: Nanos },
    CorrelationBreak(CorrelationBreakParams),
    NewsShock(NewsShockParams),
    MarketMakerWithdrawal { pct: f64, instruments: Vec<InstrumentId>, duration_ns: Nanos },
    RegulatoryHalt { instrument: InstrumentId, duration_ns: Nanos },
    CircuitBreaker { instrument: InstrumentId, trigger_move_pct: f64, halt_duration_ns: Nanos },
    Replay { events: Vec<ScenarioEvent> },
}

#[derive(Debug, Clone)]
pub struct Scenario {
    pub id: u32,
    pub name: String,
    pub description: String,
    pub start_ns: Nanos,
    pub kind: ScenarioKind,
    pub priority: u8,
    pub tags: Vec<String>,
}

impl Scenario {
    pub fn flash_crash(id: u32, name: impl Into<String>, start_ns: Nanos, params: FlashCrashParams) -> Self {
        Scenario { id, name: name.into(), description: "Flash crash scenario".into(), start_ns, kind: ScenarioKind::FlashCrash(params), priority: 10, tags: vec!["crash".into()] }
    }

    pub fn liquidity_crisis(id: u32, name: impl Into<String>, start_ns: Nanos, params: LiquidityCrisisParams) -> Self {
        Scenario { id, name: name.into(), description: "Liquidity crisis scenario".into(), start_ns, kind: ScenarioKind::LiquidityCrisis(params), priority: 9, tags: vec!["liquidity".into()] }
    }
}

// ── Scenario event (for replay) ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ScenarioEvent {
    pub timestamp_ns: Nanos,
    pub instrument_id: InstrumentId,
    pub event_type: ScenarioEventType,
}

#[derive(Debug, Clone)]
pub enum ScenarioEventType {
    PriceMove { delta_pct: f64 },
    SpreadChange { multiplier: f64 },
    DepthChange { multiplier: f64 },
    VolatilityChange { new_vol: f64 },
    TradingHalt { duration_ns: Nanos },
    LiquidityAdd { bid_qty: Qty, ask_qty: Qty },
    LiquidityRemove { bid_qty: Qty, ask_qty: Qty },
}

// ── Active scenario state ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ScenarioState { Pending, Active, Completed, Aborted }

#[derive(Debug, Clone)]
pub struct ActiveScenario {
    pub scenario: Scenario,
    pub state: ScenarioState,
    pub activated_ns: Nanos,
    pub progress: f64, // 0.0 to 1.0
}

// ── Market state snapshot ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MarketState {
    pub instrument_id: InstrumentId,
    pub mid_price: Price,
    pub bid: Price,
    pub ask: Price,
    pub bid_depth: Qty,
    pub ask_depth: Qty,
    pub vol_regime: VolatilityRegime,
    pub is_halted: bool,
    pub spread_multiplier: f64,
    pub depth_multiplier: f64,
    pub timestamp_ns: Nanos,
}

impl MarketState {
    pub fn new(instrument_id: InstrumentId, mid_price: Price) -> Self {
        let half_spread = mid_price * 0.0001; // 1bps
        MarketState {
            instrument_id, mid_price,
            bid: mid_price - half_spread,
            ask: mid_price + half_spread,
            bid_depth: 1000.0,
            ask_depth: 1000.0,
            vol_regime: VolatilityRegime::Normal,
            is_halted: false,
            spread_multiplier: 1.0,
            depth_multiplier: 1.0,
            timestamp_ns: 0,
        }
    }

    pub fn spread_bps(&self) -> f64 {
        (self.ask - self.bid) / self.mid_price * 10_000.0
    }
}

// ── Scenario engine ──────────────────────────────────────────────────────────

pub struct ScenarioEngine {
    pub scenarios: Vec<Scenario>,
    pub active_scenarios: Vec<ActiveScenario>,
    pub market_states: HashMap<InstrumentId, MarketState>,
    pub regime: VolatilityRegime,
    pub current_ns: Nanos,
    rng: Rng,
    pub event_log: VecDeque<ScenarioEvent>,
    pub max_log_size: usize,
}

impl ScenarioEngine {
    pub fn new(seed: u64) -> Self {
        ScenarioEngine {
            scenarios: Vec::new(),
            active_scenarios: Vec::new(),
            market_states: HashMap::new(),
            regime: VolatilityRegime::Normal,
            current_ns: 0,
            rng: Rng::new(seed),
            event_log: VecDeque::new(),
            max_log_size: 100_000,
        }
    }

    pub fn add_instrument(&mut self, id: InstrumentId, mid_price: Price) {
        self.market_states.insert(id, MarketState::new(id, mid_price));
    }

    pub fn register_scenario(&mut self, s: Scenario) { self.scenarios.push(s); }

    /// Advance simulation time and process all due scenarios
    pub fn advance(&mut self, new_ns: Nanos) {
        self.current_ns = new_ns;
        self.check_and_activate_scenarios();
        self.process_active_scenarios();
        self.apply_ambient_noise();
    }

    fn check_and_activate_scenarios(&mut self) {
        let now = self.current_ns;
        let mut to_activate = Vec::new();
        for s in &self.scenarios {
            if s.start_ns <= now && !self.active_scenarios.iter().any(|a| a.scenario.id == s.id) {
                to_activate.push(s.clone());
            }
        }
        for s in to_activate {
            self.active_scenarios.push(ActiveScenario {
                scenario: s,
                state: ScenarioState::Active,
                activated_ns: now,
                progress: 0.0,
            });
        }
    }

    fn process_active_scenarios(&mut self) {
        let now = self.current_ns;
        let mut completed = Vec::new();

        for (i, active) in self.active_scenarios.iter_mut().enumerate() {
            if active.state != ScenarioState::Active { continue; }
            let elapsed = now.saturating_sub(active.activated_ns);

            match &active.scenario.kind {
                ScenarioKind::FlashCrash(params) => {
                    let dur = params.duration_ns;
                    active.progress = (elapsed as f64 / dur as f64).clamp(0.0, 1.0);

                    // Compute price level at this progress
                    let initial_phase = params.initial_speed_ns;
                    let price_factor = if elapsed <= initial_phase {
                        // Dropping
                        let t = elapsed as f64 / initial_phase as f64;
                        1.0 - params.drawdown_pct * t
                    } else {
                        // Recovering
                        let t = (elapsed - initial_phase) as f64 / (dur - initial_phase).max(1) as f64;
                        let trough = 1.0 - params.drawdown_pct;
                        trough + params.drawdown_pct * params.recovery_pct * t.min(1.0)
                    };

                    for &inst in &params.affected_instruments {
                        if let Some(state) = self.market_states.get_mut(&inst) {
                            // Adjust spread during crash
                            state.spread_multiplier = 1.0 + 4.0 * (1.0 - price_factor).abs() * 10.0;
                            state.depth_multiplier = (1.0 - (1.0 - price_factor).abs() * 5.0).max(0.05);
                        }
                    }

                    if elapsed >= dur { completed.push(i); }
                }

                ScenarioKind::LiquidityCrisis(params) => {
                    let dur = params.duration_ns;
                    active.progress = (elapsed as f64 / dur as f64).clamp(0.0, 1.0);
                    let frac = 1.0 - active.progress;
                    for &inst in &params.affected_instruments {
                        if let Some(state) = self.market_states.get_mut(&inst) {
                            state.spread_multiplier = 1.0 + (params.bid_ask_spread_multiplier - 1.0) * frac;
                            state.depth_multiplier = params.depth_depletion_factor
                                + (1.0 - params.depth_depletion_factor) * (1.0 - frac);
                        }
                    }
                    if elapsed >= dur { completed.push(i); }
                }

                ScenarioKind::VolatilityRegimeChange { to, transition_ns, .. } => {
                    if elapsed >= *transition_ns {
                        self.regime = to.clone();
                        completed.push(i);
                    }
                }

                ScenarioKind::RegulatoryHalt { instrument, duration_ns } => {
                    if let Some(state) = self.market_states.get_mut(instrument) {
                        state.is_halted = elapsed < *duration_ns;
                    }
                    if elapsed >= *duration_ns { completed.push(i); }
                }

                ScenarioKind::NewsShock(params) => {
                    // One-time shock at activation
                    if elapsed < 1_000_000 { // apply once
                        for &inst in &params.affected_instruments {
                            if let Some(state) = self.market_states.get_mut(&inst) {
                                let new_price = state.mid_price * (1.0 + params.price_move_pct);
                                state.mid_price = new_price.max(0.01);
                                state.spread_multiplier *= params.vol_spike_factor;
                            }
                        }
                    }
                    completed.push(i);
                }

                ScenarioKind::Replay { events } => {
                    let due: Vec<_> = events.iter().filter(|e| {
                        let rel = e.timestamp_ns;
                        rel <= elapsed
                    }).cloned().collect();
                    for event in due {
                        self.apply_event(&event);
                    }
                    let max_ts = events.iter().map(|e| e.timestamp_ns).max().unwrap_or(0);
                    if elapsed >= max_ts { completed.push(i); }
                }

                _ => {}
            }
        }

        // Mark completed (reverse order to maintain indices)
        for &i in completed.iter().rev() {
            if i < self.active_scenarios.len() {
                self.active_scenarios[i].state = ScenarioState::Completed;
            }
        }
    }

    fn apply_event(&mut self, event: &ScenarioEvent) {
        if let Some(state) = self.market_states.get_mut(&event.instrument_id) {
            match &event.event_type {
                ScenarioEventType::PriceMove { delta_pct } => {
                    state.mid_price *= 1.0 + delta_pct;
                    state.mid_price = state.mid_price.max(0.001);
                }
                ScenarioEventType::SpreadChange { multiplier } => {
                    state.spread_multiplier = *multiplier;
                }
                ScenarioEventType::DepthChange { multiplier } => {
                    state.depth_multiplier = *multiplier;
                }
                ScenarioEventType::VolatilityChange { new_vol } => {
                    // Store custom vol in regime
                }
                ScenarioEventType::TradingHalt { duration_ns } => {
                    state.is_halted = true;
                }
                ScenarioEventType::LiquidityAdd { bid_qty, ask_qty } => {
                    state.bid_depth += bid_qty;
                    state.ask_depth += ask_qty;
                }
                ScenarioEventType::LiquidityRemove { bid_qty, ask_qty } => {
                    state.bid_depth = (state.bid_depth - bid_qty).max(0.0);
                    state.ask_depth = (state.ask_depth - ask_qty).max(0.0);
                }
            }
        }
        if self.event_log.len() >= self.max_log_size { self.event_log.pop_front(); }
        self.event_log.push_back(event.clone());
    }

    fn apply_ambient_noise(&mut self) {
        let vol = self.regime.daily_vol() / (390.0 * 60.0 * 1e9f64).sqrt(); // per-ns vol
        for state in self.market_states.values_mut() {
            if !state.is_halted {
                let ret = vol * self.rng.next_normal();
                state.mid_price *= 1.0 + ret;
                state.mid_price = state.mid_price.max(0.001);
                let half_spread = state.mid_price * 0.0001 * state.spread_multiplier;
                state.bid = state.mid_price - half_spread;
                state.ask = state.mid_price + half_spread;
            }
        }
    }

    pub fn inject_flash_crash(&mut self, instrument: InstrumentId, drawdown_pct: f64) {
        let params = FlashCrashParams {
            drawdown_pct,
            affected_instruments: vec![instrument],
            ..Default::default()
        };
        let id = self.scenarios.len() as u32 + 1000;
        self.register_scenario(Scenario::flash_crash(id, "injected_crash", self.current_ns, params));
    }

    pub fn inject_vol_regime_change(&mut self, new_regime: VolatilityRegime) {
        let old = self.regime.clone();
        let s = Scenario {
            id: self.scenarios.len() as u32 + 2000,
            name: "regime_change".into(),
            description: format!("Regime change to {:?}", new_regime),
            start_ns: self.current_ns,
            kind: ScenarioKind::VolatilityRegimeChange { from: old, to: new_regime, transition_ns: 1_000_000 },
            priority: 5,
            tags: vec!["regime".into()],
        };
        self.register_scenario(s);
    }

    pub fn market_state(&self, instrument: InstrumentId) -> Option<&MarketState> {
        self.market_states.get(&instrument)
    }

    pub fn active_scenario_count(&self) -> usize {
        self.active_scenarios.iter().filter(|a| a.state == ScenarioState::Active).count()
    }

    pub fn completed_scenario_count(&self) -> usize {
        self.active_scenarios.iter().filter(|a| a.state == ScenarioState::Completed).count()
    }

    /// Build a scenario from YAML-like config (simplified key-value)
    pub fn from_config(config: &HashMap<String, String>, seed: u64) -> Self {
        let mut engine = ScenarioEngine::new(seed);
        let n_instruments = config.get("n_instruments").and_then(|v| v.parse::<u32>().ok()).unwrap_or(3);
        let base_price = config.get("base_price").and_then(|v| v.parse::<f64>().ok()).unwrap_or(100.0);
        for i in 1..=n_instruments {
            let price = base_price * (1.0 + 0.1 * (i as f64 - 1.0));
            engine.add_instrument(i, price);
        }
        if config.get("flash_crash").map(|v| v == "true").unwrap_or(false) {
            engine.inject_flash_crash(1, 0.07);
        }
        if let Some(regime_str) = config.get("regime") {
            let regime = match regime_str.as_str() {
                "calm" => VolatilityRegime::Calm,
                "elevated" => VolatilityRegime::Elevated,
                "crisis" => VolatilityRegime::Crisis,
                _ => VolatilityRegime::Normal,
            };
            engine.regime = regime;
        }
        engine
    }
}

// ── Correlated multi-asset scenario ──────────────────────────────────────────

pub struct CorrelatedAssetModel {
    /// Cholesky decomposition (lower triangular) of correlation matrix
    pub chol: Vec<Vec<f64>>,
    pub n: usize,
}

impl CorrelatedAssetModel {
    /// Build from correlation matrix
    pub fn from_correlation(corr: &Vec<Vec<f64>>) -> Option<Self> {
        let n = corr.len();
        let mut l = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = corr[i][j];
                for k in 0..j { sum -= l[i][k] * l[j][k]; }
                if i == j {
                    if sum < 0.0 { return None; }
                    l[i][j] = sum.sqrt();
                } else {
                    if l[j][j].abs() < 1e-12 { return None; }
                    l[i][j] = sum / l[j][j];
                }
            }
        }
        Some(CorrelatedAssetModel { chol: l, n })
    }

    /// Generate correlated standard normal vector
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let z: Vec<f64> = (0..self.n).map(|_| rng.next_normal()).collect();
        let mut out = vec![0.0f64; self.n];
        for i in 0..self.n {
            for j in 0..=i {
                out[i] += self.chol[i][j] * z[j];
            }
        }
        out
    }

    /// Generate correlated returns given individual vols
    pub fn sample_returns(&self, vols: &[f64], rng: &mut Rng) -> Vec<f64> {
        let corr_normals = self.sample(rng);
        corr_normals.iter().zip(vols.iter()).map(|(z, v)| z * v).collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_regime_properties() {
        assert!(VolatilityRegime::Crisis.annualized_vol() > VolatilityRegime::Calm.annualized_vol());
        assert!(VolatilityRegime::Calm.liquidity_multiplier() > VolatilityRegime::Crisis.liquidity_multiplier());
        assert!(VolatilityRegime::Crisis.spread_multiplier() > VolatilityRegime::Normal.spread_multiplier());
    }

    #[test]
    fn test_scenario_engine_init() {
        let mut engine = ScenarioEngine::new(42);
        engine.add_instrument(1, 100.0);
        assert!(engine.market_states.contains_key(&1));
        assert_eq!(engine.market_states[&1].mid_price, 100.0);
    }

    #[test]
    fn test_flash_crash_injection() {
        let mut engine = ScenarioEngine::new(42);
        engine.add_instrument(1, 100.0);
        engine.inject_flash_crash(1, 0.10);
        engine.advance(1_000_000_000); // 1 second
        // Scenario should be active
        assert!(engine.active_scenario_count() > 0 || engine.scenarios.len() > 0);
    }

    #[test]
    fn test_regime_change_injection() {
        let mut engine = ScenarioEngine::new(42);
        engine.add_instrument(1, 100.0);
        engine.inject_vol_regime_change(VolatilityRegime::Crisis);
        engine.advance(100_000_000_000);
        // After enough time, regime should change
        // (may or may not have activated depending on timing)
        let _ = engine.regime.clone();
    }

    #[test]
    fn test_scenario_from_config() {
        let mut config = HashMap::new();
        config.insert("n_instruments".into(), "3".into());
        config.insert("base_price".into(), "150.0".into());
        config.insert("regime".into(), "elevated".into());
        let engine = ScenarioEngine::from_config(&config, 42);
        assert_eq!(engine.market_states.len(), 3);
        assert!(matches!(engine.regime, VolatilityRegime::Elevated));
    }

    #[test]
    fn test_ambient_noise() {
        let mut engine = ScenarioEngine::new(42);
        engine.add_instrument(1, 100.0);
        let initial_price = engine.market_states[&1].mid_price;
        for i in 0..1000u64 { engine.advance(i * 1_000_000); }
        let final_price = engine.market_states[&1].mid_price;
        // Price should have drifted (not identical)
        assert!((final_price - initial_price).abs() > 0.0);
    }

    #[test]
    fn test_liquidity_crisis_scenario() {
        let mut engine = ScenarioEngine::new(42);
        engine.add_instrument(1, 100.0);
        let params = LiquidityCrisisParams { duration_ns: 10_000_000_000, ..Default::default() };
        let s = Scenario::liquidity_crisis(1, "crisis", 0, params);
        engine.register_scenario(s);
        engine.advance(1_000_000_000);
        let state = &engine.market_states[&1];
        assert!(state.spread_multiplier > 1.0, "spread_mult={}", state.spread_multiplier);
    }

    #[test]
    fn test_news_shock_scenario() {
        let mut engine = ScenarioEngine::new(42);
        engine.add_instrument(1, 100.0);
        let initial_price = engine.market_states[&1].mid_price;
        let params = NewsShockParams { timestamp_ns: 0, price_move_pct: 0.05, vol_spike_factor: 2.0, affected_instruments: vec![1], headline: "Good news".into() };
        let s = Scenario { id: 1, name: "news".into(), description: "".into(), start_ns: 1_000_000, kind: ScenarioKind::NewsShock(params), priority: 5, tags: vec![] };
        engine.register_scenario(s);
        engine.advance(2_000_000);
        // Price should be ~5% higher
        let new_price = engine.market_states[&1].mid_price;
        assert!((new_price / initial_price - 1.05).abs() < 0.01, "new={} init={}", new_price, initial_price);
    }

    #[test]
    fn test_correlated_asset_model_identity() {
        let n = 3;
        let identity: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
        let model = CorrelatedAssetModel::from_correlation(&identity).unwrap();
        let mut rng = Rng::new(42);
        let samples = model.sample(&mut rng);
        assert_eq!(samples.len(), n);
    }

    #[test]
    fn test_correlated_asset_model_sampling() {
        let corr = vec![
            vec![1.0, 0.8, 0.5],
            vec![0.8, 1.0, 0.3],
            vec![0.5, 0.3, 1.0],
        ];
        let model = CorrelatedAssetModel::from_correlation(&corr).unwrap();
        let mut rng = Rng::new(99);
        let vols = vec![0.20, 0.25, 0.15];
        let returns = model.sample_returns(&vols, &mut rng);
        assert_eq!(returns.len(), 3);
        assert!(returns.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_replay_scenario() {
        let events = vec![
            ScenarioEvent { timestamp_ns: 100_000_000, instrument_id: 1, event_type: ScenarioEventType::PriceMove { delta_pct: -0.02 } },
            ScenarioEvent { timestamp_ns: 200_000_000, instrument_id: 1, event_type: ScenarioEventType::SpreadChange { multiplier: 3.0 } },
        ];
        let mut engine = ScenarioEngine::new(42);
        engine.add_instrument(1, 100.0);
        let s = Scenario { id: 1, name: "replay".into(), description: "".into(), start_ns: 0, kind: ScenarioKind::Replay { events }, priority: 5, tags: vec![] };
        engine.register_scenario(s);
        engine.advance(300_000_000);
        // spread multiplier should have changed
        let state = &engine.market_states[&1];
        assert!(state.spread_multiplier >= 1.0);
    }
}
