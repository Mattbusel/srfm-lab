/// Signal genome representation for the SRFM evolutionary engine.
///
/// A SignalGenome encodes all tunable parameters of one tradeable signal as
/// an ordered set of genes. The genome can be serialized to a flat f64
/// vector for compatibility with genetic algorithm operators, and decoded
/// back to a typed struct for backtesting.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Gene bounds -- hard limits for feasibility checking
// ---------------------------------------------------------------------------

const BH_MASS_THRESHOLD_MIN: f64 = 0.1;
const BH_MASS_THRESHOLD_MAX: f64 = 5.0;

const HURST_WINDOW_MIN: u32 = 10;
const HURST_WINDOW_MAX: u32 = 252;

const GARCH_ALPHA_MIN: f64 = 0.01;
const GARCH_ALPHA_MAX: f64 = 0.49;

const NAV_OMEGA_SCALE_MIN: f64 = 0.1;
const NAV_OMEGA_SCALE_MAX: f64 = 10.0;

const ENTRY_GATE_MIN: f64 = 0.01;
const ENTRY_GATE_MAX: f64 = 1.0;

const EXIT_MULTIPLIER_MIN: f64 = 0.5;
const EXIT_MULTIPLIER_MAX: f64 = 5.0;

const MIN_HOLD_BARS_MIN: u32 = 1;
const MIN_HOLD_BARS_MAX: u32 = 96; // max 1 trading day at 15m

// Hours 0-23; any subset may be blocked.
const MAX_BLOCKED_HOURS: usize = 24;

// ---------------------------------------------------------------------------
// DecodeError
// ---------------------------------------------------------------------------

/// Errors returned by `SignalGenome::decode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecodeError {
    /// The flat vector has the wrong length.
    WrongLength { expected: usize, got: usize },
    /// A gene value is outside its allowed range.
    OutOfRange { gene: String, value: f64 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::WrongLength { expected, got } => {
                write!(f, "decode error: expected {expected} floats, got {got}")
            }
            DecodeError::OutOfRange { gene, value } => {
                write!(f, "decode error: gene '{gene}' value {value} out of range")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SignalGene
// ---------------------------------------------------------------------------

/// One evolvable gene within a signal genome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SignalGene {
    /// Minimum BH mass required to enter (0.1 to 5.0).
    BHMassThreshold(f64),
    /// Rolling window for Hurst exponent computation (10 to 252 bars).
    HurstWindow(u32),
    /// GARCH alpha (ARCH coefficient) in (0.01, 0.49).
    GarchAlpha(f64),
    /// Scale factor applied to NAV omega in position sizing (0.1 to 10.0).
    NavOmegaScale(f64),
    /// Minimum absolute signal value required for entry (0.01 to 1.0).
    EntryGate(f64),
    /// Multiplier applied to entry gate for exit (0.5 to 5.0).
    ExitMultiplier(f64),
    /// Minimum bars to hold a position before allowing exit (1 to 96).
    MinHoldBars(u32),
    /// Hours of the trading day (0-23) during which entry is blocked.
    BlockedHours(Vec<u8>),
}

// Encoding layout (flat f64 vector):
//   [0] BHMassThreshold
//   [1] HurstWindow (as f64, will be rounded to u32)
//   [2] GarchAlpha
//   [3] NavOmegaScale
//   [4] EntryGate
//   [5] ExitMultiplier
//   [6] MinHoldBars (as f64, rounded to u32)
//   [7..30] BlockedHours -- 24 slots, each 0.0 or 1.0 (blocked=1)
const ENCODE_LEN: usize = 7 + MAX_BLOCKED_HOURS; // 31

// ---------------------------------------------------------------------------
// SignalGenome
// ---------------------------------------------------------------------------

/// A complete tradeable signal encoded as an evolvable genome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGenome {
    pub bh_mass_threshold: f64,
    pub hurst_window: u32,
    pub garch_alpha: f64,
    pub nav_omega_scale: f64,
    pub entry_gate: f64,
    pub exit_multiplier: f64,
    pub min_hold_bars: u32,
    /// Hours blocked from entry (values 0-23).
    pub blocked_hours: Vec<u8>,
    /// Cached fitness estimate (not serialized as part of GA state).
    #[serde(skip)]
    pub fitness_cache: Option<f64>,
}

impl SignalGenome {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a genome with safe default values.
    pub fn default_genome() -> Self {
        Self {
            bh_mass_threshold: 1.5,
            hurst_window: 63,
            garch_alpha: 0.10,
            nav_omega_scale: 1.0,
            entry_gate: 0.3,
            exit_multiplier: 1.5,
            min_hold_bars: 4,
            blocked_hours: vec![],
            fitness_cache: None,
        }
    }

    /// Generate a random genome using the provided RNG.
    pub fn random(rng: &mut impl Rng) -> Self {
        let n_blocked: usize = rng.gen_range(0..=6);
        let mut blocked = Vec::with_capacity(n_blocked);
        for _ in 0..n_blocked {
            let h: u8 = rng.gen_range(0..24);
            if !blocked.contains(&h) {
                blocked.push(h);
            }
        }
        Self {
            bh_mass_threshold: rng.gen_range(BH_MASS_THRESHOLD_MIN..=BH_MASS_THRESHOLD_MAX),
            hurst_window: rng.gen_range(HURST_WINDOW_MIN..=HURST_WINDOW_MAX),
            garch_alpha: rng.gen_range(GARCH_ALPHA_MIN..=GARCH_ALPHA_MAX),
            nav_omega_scale: rng.gen_range(NAV_OMEGA_SCALE_MIN..=NAV_OMEGA_SCALE_MAX),
            entry_gate: rng.gen_range(ENTRY_GATE_MIN..=ENTRY_GATE_MAX),
            exit_multiplier: rng.gen_range(EXIT_MULTIPLIER_MIN..=EXIT_MULTIPLIER_MAX),
            min_hold_bars: rng.gen_range(MIN_HOLD_BARS_MIN..=MIN_HOLD_BARS_MAX),
            blocked_hours: blocked,
            fitness_cache: None,
        }
    }

    // -----------------------------------------------------------------------
    // Encoding / Decoding
    // -----------------------------------------------------------------------

    /// Encode the genome as a flat Vec<f64> for genetic algorithm operators.
    pub fn encode(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(ENCODE_LEN);
        v.push(self.bh_mass_threshold);
        v.push(self.hurst_window as f64);
        v.push(self.garch_alpha);
        v.push(self.nav_omega_scale);
        v.push(self.entry_gate);
        v.push(self.exit_multiplier);
        v.push(self.min_hold_bars as f64);
        // 24-slot binary encoding for blocked_hours.
        let mut blocked_slots = [0.0f64; MAX_BLOCKED_HOURS];
        for &h in &self.blocked_hours {
            if (h as usize) < MAX_BLOCKED_HOURS {
                blocked_slots[h as usize] = 1.0;
            }
        }
        v.extend_from_slice(&blocked_slots);
        debug_assert_eq!(v.len(), ENCODE_LEN);
        v
    }

    /// Decode a flat vector back into a SignalGenome.
    /// Returns `Err(DecodeError)` if the vector has the wrong length or any
    /// value is outside the allowed range.
    pub fn decode(values: &[f64]) -> Result<SignalGenome, DecodeError> {
        if values.len() != ENCODE_LEN {
            return Err(DecodeError::WrongLength {
                expected: ENCODE_LEN,
                got: values.len(),
            });
        }

        let bh_mass_threshold = values[0];
        let hurst_window = values[1].round() as u32;
        let garch_alpha = values[2];
        let nav_omega_scale = values[3];
        let entry_gate = values[4];
        let exit_multiplier = values[5];
        let min_hold_bars = values[6].round() as u32;

        // Decode blocked_hours from binary slots.
        let mut blocked_hours: Vec<u8> = Vec::new();
        for (i, &slot) in values[7..7 + MAX_BLOCKED_HOURS].iter().enumerate() {
            if slot > 0.5 {
                blocked_hours.push(i as u8);
            }
        }

        let genome = SignalGenome {
            bh_mass_threshold,
            hurst_window,
            garch_alpha,
            nav_omega_scale,
            entry_gate,
            exit_multiplier,
            min_hold_bars,
            blocked_hours,
            fitness_cache: None,
        };

        // Validate feasibility after decode.
        if !genome.is_feasible() {
            // Find which gene is out of range.
            let reason = genome.feasibility_reason();
            return Err(DecodeError::OutOfRange {
                gene: reason,
                value: 0.0, // generic; reason contains detail
            });
        }

        Ok(genome)
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Mutate each gene independently with probability `rate`.
    ///
    /// Float genes use Gaussian perturbation (sigma = 10% of range).
    /// Integer genes use +/-1 random walk.
    /// BlockedHours: randomly toggle one hour.
    pub fn mutate(&mut self, rate: f64, rng: &mut impl Rng) {
        // BHMassThreshold
        if rng.gen_bool(rate) {
            let range = BH_MASS_THRESHOLD_MAX - BH_MASS_THRESHOLD_MIN;
            let sigma = range * 0.10;
            self.bh_mass_threshold = (self.bh_mass_threshold + rng.gen_range(-sigma..=sigma))
                .clamp(BH_MASS_THRESHOLD_MIN, BH_MASS_THRESHOLD_MAX);
        }

        // HurstWindow
        if rng.gen_bool(rate) {
            let delta: i32 = rng.gen_range(-5..=5);
            self.hurst_window = ((self.hurst_window as i32 + delta)
                .max(HURST_WINDOW_MIN as i32)
                .min(HURST_WINDOW_MAX as i32)) as u32;
        }

        // GarchAlpha
        if rng.gen_bool(rate) {
            let range = GARCH_ALPHA_MAX - GARCH_ALPHA_MIN;
            let sigma = range * 0.10;
            self.garch_alpha = (self.garch_alpha + rng.gen_range(-sigma..=sigma))
                .clamp(GARCH_ALPHA_MIN, GARCH_ALPHA_MAX);
        }

        // NavOmegaScale
        if rng.gen_bool(rate) {
            let range = NAV_OMEGA_SCALE_MAX - NAV_OMEGA_SCALE_MIN;
            let sigma = range * 0.10;
            self.nav_omega_scale = (self.nav_omega_scale + rng.gen_range(-sigma..=sigma))
                .clamp(NAV_OMEGA_SCALE_MIN, NAV_OMEGA_SCALE_MAX);
        }

        // EntryGate
        if rng.gen_bool(rate) {
            let range = ENTRY_GATE_MAX - ENTRY_GATE_MIN;
            let sigma = range * 0.10;
            self.entry_gate = (self.entry_gate + rng.gen_range(-sigma..=sigma))
                .clamp(ENTRY_GATE_MIN, ENTRY_GATE_MAX);
        }

        // ExitMultiplier
        if rng.gen_bool(rate) {
            let range = EXIT_MULTIPLIER_MAX - EXIT_MULTIPLIER_MIN;
            let sigma = range * 0.10;
            self.exit_multiplier = (self.exit_multiplier + rng.gen_range(-sigma..=sigma))
                .clamp(EXIT_MULTIPLIER_MIN, EXIT_MULTIPLIER_MAX);
        }

        // MinHoldBars
        if rng.gen_bool(rate) {
            let delta: i32 = rng.gen_range(-3..=3);
            self.min_hold_bars = ((self.min_hold_bars as i32 + delta)
                .max(MIN_HOLD_BARS_MIN as i32)
                .min(MIN_HOLD_BARS_MAX as i32)) as u32;
        }

        // BlockedHours: toggle a random hour
        if rng.gen_bool(rate) {
            let h: u8 = rng.gen_range(0..24);
            if let Some(pos) = self.blocked_hours.iter().position(|&x| x == h) {
                self.blocked_hours.remove(pos);
            } else if self.blocked_hours.len() < 12 {
                // Cap at 12 blocked hours to keep the signal tradeable.
                self.blocked_hours.push(h);
            }
        }

        // Invalidate fitness cache after mutation.
        self.fitness_cache = None;
    }

    // -----------------------------------------------------------------------
    // Crossover
    // -----------------------------------------------------------------------

    /// Uniform crossover: for each gene, randomly assign from parent a or b.
    /// Returns two children.
    pub fn crossover(
        a: &SignalGenome,
        b: &SignalGenome,
        rng: &mut impl Rng,
    ) -> (SignalGenome, SignalGenome) {
        macro_rules! cross_f64 {
            ($field:ident) => {
                if rng.gen_bool(0.5) {
                    (a.$field, b.$field)
                } else {
                    (b.$field, a.$field)
                }
            };
        }
        macro_rules! cross_u32 {
            ($field:ident) => {
                if rng.gen_bool(0.5) {
                    (a.$field, b.$field)
                } else {
                    (b.$field, a.$field)
                }
            };
        }

        let (bh1, bh2) = cross_f64!(bh_mass_threshold);
        let (hw1, hw2) = cross_u32!(hurst_window);
        let (ga1, ga2) = cross_f64!(garch_alpha);
        let (ns1, ns2) = cross_f64!(nav_omega_scale);
        let (eg1, eg2) = cross_f64!(entry_gate);
        let (em1, em2) = cross_f64!(exit_multiplier);
        let (mh1, mh2) = cross_u32!(min_hold_bars);

        // For blocked_hours, take union of each parent's set split by RNG.
        let bh_hours_1 = mix_hours(&a.blocked_hours, &b.blocked_hours, rng);
        let bh_hours_2 = mix_hours(&b.blocked_hours, &a.blocked_hours, rng);

        let child_a = SignalGenome {
            bh_mass_threshold: bh1,
            hurst_window: hw1,
            garch_alpha: ga1,
            nav_omega_scale: ns1,
            entry_gate: eg1,
            exit_multiplier: em1,
            min_hold_bars: mh1,
            blocked_hours: bh_hours_1,
            fitness_cache: None,
        };
        let child_b = SignalGenome {
            bh_mass_threshold: bh2,
            hurst_window: hw2,
            garch_alpha: ga2,
            nav_omega_scale: ns2,
            entry_gate: eg2,
            exit_multiplier: em2,
            min_hold_bars: mh2,
            blocked_hours: bh_hours_2,
            fitness_cache: None,
        };
        (child_a, child_b)
    }

    // -----------------------------------------------------------------------
    // Fitness heuristic
    // -----------------------------------------------------------------------

    /// Fast heuristic fitness estimate without full backtest.
    ///
    /// Penalizes extreme parameter values that historically degrade performance.
    /// Returns a value in (0.0, 1.0]; 1.0 = theoretically ideal configuration.
    pub fn fitness_estimate(&self) -> f64 {
        if let Some(cached) = self.fitness_cache {
            return cached;
        }

        let mut score = 1.0f64;

        // BH mass threshold: reward moderate values (0.5 to 2.5).
        if self.bh_mass_threshold < 0.5 || self.bh_mass_threshold > 3.5 {
            score *= 0.8;
        }

        // Hurst window: reward windows of 20-100 bars.
        if self.hurst_window < 20 || self.hurst_window > 150 {
            score *= 0.85;
        }

        // GARCH alpha: penalize very high values (unstable GARCH).
        if self.garch_alpha > 0.40 {
            score *= 0.75;
        }

        // Entry gate: too low = overtrading, too high = undertrading.
        if self.entry_gate < 0.05 {
            score *= 0.80;
        } else if self.entry_gate > 0.85 {
            score *= 0.85;
        }

        // Exit multiplier: penalize extreme values.
        if self.exit_multiplier < 0.8 || self.exit_multiplier > 4.0 {
            score *= 0.85;
        }

        // Min hold bars: very short holds are likely overfit.
        if self.min_hold_bars < 2 {
            score *= 0.75;
        }

        // Blocked hours: penalize blocking many hours (reduces capacity).
        let n_blocked = self.blocked_hours.len();
        if n_blocked > 8 {
            score *= 1.0 - (n_blocked - 8) as f64 * 0.05;
        }

        score.max(0.01)
    }

    // -----------------------------------------------------------------------
    // Feasibility
    // -----------------------------------------------------------------------

    /// Returns true if all genes are within their allowed ranges.
    pub fn is_feasible(&self) -> bool {
        self.bh_mass_threshold >= BH_MASS_THRESHOLD_MIN
            && self.bh_mass_threshold <= BH_MASS_THRESHOLD_MAX
            && self.hurst_window >= HURST_WINDOW_MIN
            && self.hurst_window <= HURST_WINDOW_MAX
            && self.garch_alpha >= GARCH_ALPHA_MIN
            && self.garch_alpha <= GARCH_ALPHA_MAX
            && self.nav_omega_scale >= NAV_OMEGA_SCALE_MIN
            && self.nav_omega_scale <= NAV_OMEGA_SCALE_MAX
            && self.entry_gate >= ENTRY_GATE_MIN
            && self.entry_gate <= ENTRY_GATE_MAX
            && self.exit_multiplier >= EXIT_MULTIPLIER_MIN
            && self.exit_multiplier <= EXIT_MULTIPLIER_MAX
            && self.min_hold_bars >= MIN_HOLD_BARS_MIN
            && self.min_hold_bars <= MIN_HOLD_BARS_MAX
            && self.blocked_hours.iter().all(|&h| h < 24)
    }

    /// Returns a description of the first feasibility violation, or "ok".
    fn feasibility_reason(&self) -> String {
        if !(self.bh_mass_threshold >= BH_MASS_THRESHOLD_MIN
            && self.bh_mass_threshold <= BH_MASS_THRESHOLD_MAX)
        {
            return format!("bh_mass_threshold={}", self.bh_mass_threshold);
        }
        if !(self.hurst_window >= HURST_WINDOW_MIN && self.hurst_window <= HURST_WINDOW_MAX) {
            return format!("hurst_window={}", self.hurst_window);
        }
        if !(self.garch_alpha >= GARCH_ALPHA_MIN && self.garch_alpha <= GARCH_ALPHA_MAX) {
            return format!("garch_alpha={}", self.garch_alpha);
        }
        if !(self.nav_omega_scale >= NAV_OMEGA_SCALE_MIN
            && self.nav_omega_scale <= NAV_OMEGA_SCALE_MAX)
        {
            return format!("nav_omega_scale={}", self.nav_omega_scale);
        }
        if !(self.entry_gate >= ENTRY_GATE_MIN && self.entry_gate <= ENTRY_GATE_MAX) {
            return format!("entry_gate={}", self.entry_gate);
        }
        if !(self.exit_multiplier >= EXIT_MULTIPLIER_MIN
            && self.exit_multiplier <= EXIT_MULTIPLIER_MAX)
        {
            return format!("exit_multiplier={}", self.exit_multiplier);
        }
        if !(self.min_hold_bars >= MIN_HOLD_BARS_MIN && self.min_hold_bars <= MIN_HOLD_BARS_MAX) {
            return format!("min_hold_bars={}", self.min_hold_bars);
        }
        "ok".to_string()
    }
}

// ---------------------------------------------------------------------------
// mix_hours helper
// ---------------------------------------------------------------------------

/// For crossover: randomly pick each hour from one of the two parents' sets.
fn mix_hours(primary: &[u8], secondary: &[u8], rng: &mut impl Rng) -> Vec<u8> {
    let mut combined: Vec<u8> = primary
        .iter()
        .chain(secondary.iter())
        .copied()
        .collect::<std::collections::HashSet<u8>>()
        .into_iter()
        .collect();
    combined.sort();
    // Randomly keep each hour with 50% probability.
    combined
        .into_iter()
        .filter(|_| rng.gen_bool(0.5))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn test_default_genome_is_feasible() {
        let g = SignalGenome::default_genome();
        assert!(g.is_feasible(), "default genome must be feasible");
    }

    #[test]
    fn test_random_genome_is_feasible() {
        let mut rng = seeded_rng();
        for _ in 0..50 {
            let g = SignalGenome::random(&mut rng);
            assert!(g.is_feasible(), "random genome must be feasible: {:?}", g);
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut rng = seeded_rng();
        for _ in 0..20 {
            let g = SignalGenome::random(&mut rng);
            let encoded = g.encode();
            assert_eq!(encoded.len(), ENCODE_LEN);
            let decoded = SignalGenome::decode(&encoded).expect("decode must succeed");
            assert!(
                (decoded.bh_mass_threshold - g.bh_mass_threshold).abs() < 1e-9,
                "bh_mass_threshold roundtrip"
            );
            assert_eq!(decoded.hurst_window, g.hurst_window);
            assert!((decoded.garch_alpha - g.garch_alpha).abs() < 1e-9);
            assert_eq!(decoded.min_hold_bars, g.min_hold_bars);
        }
    }

    #[test]
    fn test_decode_wrong_length_returns_error() {
        let short = vec![0.0f64; 10];
        let result = SignalGenome::decode(&short);
        assert!(matches!(result, Err(DecodeError::WrongLength { .. })));
    }

    #[test]
    fn test_mutate_stays_feasible() {
        let mut rng = seeded_rng();
        let mut g = SignalGenome::default_genome();
        for _ in 0..200 {
            g.mutate(0.5, &mut rng);
            assert!(g.is_feasible(), "post-mutation genome must be feasible: {:?}", g);
        }
    }

    #[test]
    fn test_crossover_children_feasible() {
        let mut rng = seeded_rng();
        let a = SignalGenome::random(&mut rng);
        let b = SignalGenome::random(&mut rng);
        for _ in 0..20 {
            let (c1, c2) = SignalGenome::crossover(&a, &b, &mut rng);
            assert!(c1.is_feasible(), "child1 must be feasible");
            assert!(c2.is_feasible(), "child2 must be feasible");
        }
    }

    #[test]
    fn test_fitness_estimate_in_range() {
        let mut rng = seeded_rng();
        for _ in 0..30 {
            let g = SignalGenome::random(&mut rng);
            let f = g.fitness_estimate();
            assert!(f > 0.0 && f <= 1.0, "fitness must be in (0, 1], got {f}");
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let g = SignalGenome::default_genome();
        let json = serde_json::to_string(&g).unwrap();
        let decoded: SignalGenome = serde_json::from_str(&json).unwrap();
        assert!((decoded.bh_mass_threshold - g.bh_mass_threshold).abs() < 1e-12);
        assert_eq!(decoded.hurst_window, g.hurst_window);
    }

    #[test]
    fn test_blocked_hours_in_valid_range() {
        let mut rng = seeded_rng();
        for _ in 0..30 {
            let g = SignalGenome::random(&mut rng);
            for &h in &g.blocked_hours {
                assert!(h < 24, "blocked hour must be 0-23, got {h}");
            }
        }
    }

    #[test]
    fn test_encode_blocked_hours_binary() {
        let mut g = SignalGenome::default_genome();
        g.blocked_hours = vec![0, 3, 23];
        let enc = g.encode();
        // Slots 7+0, 7+3, 7+23 should be 1.0; others 0.0.
        assert!((enc[7] - 1.0).abs() < 1e-12, "hour 0 should be blocked");
        assert!((enc[10] - 1.0).abs() < 1e-12, "hour 3 should be blocked");
        assert!((enc[30] - 1.0).abs() < 1e-12, "hour 23 should be blocked");
        assert!((enc[8] - 0.0).abs() < 1e-12, "hour 1 should not be blocked");
    }
}
