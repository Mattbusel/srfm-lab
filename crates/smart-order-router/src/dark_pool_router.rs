// dark_pool_router.rs
// Dark pool and crossing network routing logic.
// Ranks venues by fill probability adjusted for information leakage,
// allocates greedily, and tracks adverse selection signals.

use std::collections::VecDeque;
use crate::{OrderSide, SorError};

// ---- DarkVenue ------------------------------------------------------------

/// A dark pool or crossing network venue.
#[derive(Debug, Clone)]
pub struct DarkVenue {
    /// Unique venue identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Historical fill rate (fraction of orders that receive any fill)
    pub avg_fill_rate: f64,
    /// Average fill size when filled (absolute quantity)
    pub avg_fill_size: f64,
    /// One-way latency in microseconds
    pub latency_us: u64,
    /// Information leakage score: 0.0 = no leakage, 1.0 = full leakage
    pub information_leakage_score: f64,
    /// Minimum order size accepted
    pub min_order_size: f64,
    /// Maximum order size accepted
    pub max_order_size: f64,
    /// Average daily dark volume (used for fill probability model)
    pub avg_daily_dark_vol: f64,
}

impl DarkVenue {
    pub fn new(id: &str, name: &str) -> Self {
        DarkVenue {
            id: id.to_string(),
            name: name.to_string(),
            avg_fill_rate: 0.50,
            avg_fill_size: 5_000.0,
            latency_us: 100,
            information_leakage_score: 0.10,
            min_order_size: 100.0,
            max_order_size: f64::INFINITY,
            avg_daily_dark_vol: 500_000.0,
        }
    }

    pub fn with_fill_rate(mut self, rate: f64) -> Self {
        self.avg_fill_rate = rate.min(1.0).max(0.0);
        self
    }

    pub fn with_avg_fill_size(mut self, size: f64) -> Self {
        self.avg_fill_size = size;
        self
    }

    pub fn with_latency(mut self, latency_us: u64) -> Self {
        self.latency_us = latency_us;
        self
    }

    pub fn with_leakage(mut self, leakage: f64) -> Self {
        self.information_leakage_score = leakage.min(1.0).max(0.0);
        self
    }

    pub fn with_dark_volume(mut self, vol: f64) -> Self {
        self.avg_daily_dark_vol = vol;
        self
    }

    pub fn with_size_limits(mut self, min: f64, max: f64) -> Self {
        self.min_order_size = min;
        self.max_order_size = max;
        self
    }
}

// ---- FillProbabilityModel --------------------------------------------------

/// Logistic regression model for dark fill probability.
/// P(fill) = sigmoid(a * ln(size / avg_vol) + b)
/// where a < 0 (larger orders relative to dark volume are harder to fill)
/// and b is the intercept calibrated to observed fill rates.
#[derive(Debug, Clone)]
pub struct FillProbabilityModel {
    /// Coefficient on ln(size / avg_vol)
    pub coef_a: f64,
    /// Intercept
    pub coef_b: f64,
}

impl FillProbabilityModel {
    /// Default calibration: typical dark pool empirics.
    pub fn default_calibration() -> Self {
        FillProbabilityModel {
            coef_a: -0.70, // negative: larger relative size -> lower fill prob
            coef_b: 0.50,  // intercept -> sigmoid(0.5) ~= 0.62 at median size
        }
    }

    pub fn new(coef_a: f64, coef_b: f64) -> Self {
        FillProbabilityModel { coef_a, coef_b }
    }

    /// sigmoid(x) = 1 / (1 + exp(-x))
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Compute P(fill) for a given order size vs average dark volume.
    pub fn fill_probability(&self, size: f64, avg_vol: f64) -> f64 {
        if avg_vol <= 0.0 || size <= 0.0 {
            return 0.0;
        }
        let log_ratio = (size / avg_vol).ln();
        Self::sigmoid(self.coef_a * log_ratio + self.coef_b)
    }

    /// Expected fill size = P(fill) * min(size, avg_fill_size).
    pub fn expected_fill(&self, size: f64, avg_vol: f64, avg_fill_size: f64) -> f64 {
        let p = self.fill_probability(size, avg_vol);
        p * size.min(avg_fill_size)
    }
}

// ---- DarkVenueAllocation --------------------------------------------------

/// Recommended allocation to a single dark venue.
#[derive(Debug, Clone)]
pub struct DarkVenueAllocation {
    pub venue_id: String,
    pub quantity: f64,
    /// Estimated probability of fill at this quantity
    pub fill_probability: f64,
    /// Expected fill size (probability * qty)
    pub expected_fill: f64,
    /// Ranking score used to select this venue
    pub score: f64,
}

// ---- DarkPoolRouter -------------------------------------------------------

/// Routes dark orders across a set of dark venues.
/// Allocates greedily by venue score until expected fill covers the order.
#[derive(Debug, Clone)]
pub struct DarkPoolRouter {
    pub venues: Vec<DarkVenue>,
    pub fill_probability_model: FillProbabilityModel,
    /// Weight applied to the leakage penalty when scoring venues.
    pub information_leakage_penalty: f64,
    /// Minimum fill probability required to use a venue
    pub min_fill_prob_threshold: f64,
}

impl DarkPoolRouter {
    pub fn new(fill_probability_model: FillProbabilityModel, leakage_penalty: f64) -> Self {
        DarkPoolRouter {
            venues: Vec::new(),
            fill_probability_model,
            information_leakage_penalty: leakage_penalty.min(1.0).max(0.0),
            min_fill_prob_threshold: 0.05,
        }
    }

    pub fn add_venue(&mut self, venue: DarkVenue) {
        self.venues.push(venue);
    }

    pub fn with_min_fill_prob(mut self, threshold: f64) -> Self {
        self.min_fill_prob_threshold = threshold;
        self
    }

    /// Score a venue for a given order size.
    /// score = fill_probability * (1 - leakage_penalty * leakage_score) / sqrt(latency_us)
    /// Higher is better.
    pub fn venue_score(&self, venue: &DarkVenue, order_size: f64) -> f64 {
        let fill_prob = self.fill_probability_model.fill_probability(
            order_size,
            venue.avg_daily_dark_vol,
        );
        let leakage_adj = 1.0 - self.information_leakage_penalty * venue.information_leakage_score;
        let latency_penalty = (venue.latency_us as f64).sqrt().max(1.0);
        fill_prob * leakage_adj / latency_penalty
    }

    /// Route an order to dark venues.
    /// Returns allocations sorted by priority (best venue first).
    /// Allocates greedily until the sum of expected fills >= order_qty.
    pub fn route(
        &self,
        order_qty: f64,
        side: OrderSide,
        symbol: &str,
    ) -> Result<Vec<DarkVenueAllocation>, SorError> {
        let _ = (side, symbol); // Not venue-specific in this model; used by callers

        if self.venues.is_empty() {
            return Err(SorError::NoVenues);
        }
        if order_qty <= 0.0 {
            return Err(SorError::InvalidParameter("order_qty must be > 0".to_string()));
        }

        // Score all venues for this order
        let mut scored: Vec<(f64, &DarkVenue)> = self.venues.iter()
            .filter(|v| v.min_order_size <= order_qty)
            .map(|v| (self.venue_score(v, order_qty), v))
            .filter(|(s, v)| {
                // Also filter by raw fill probability floor
                let fp = self.fill_probability_model.fill_probability(order_qty, v.avg_daily_dark_vol);
                fp >= self.min_fill_prob_threshold && *s > 0.0
            })
            .collect();

        if scored.is_empty() {
            return Err(SorError::InsufficientLiquidity {
                needed: order_qty,
                available: 0.0,
            });
        }

        // Sort descending by score
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut allocations = Vec::new();
        let mut expected_filled = 0.0;

        for (score, venue) in &scored {
            if expected_filled >= order_qty { break; }

            // Allocate the remaining unfilled quantity, capped by venue max
            let remaining = order_qty - expected_filled;
            let venue_qty = remaining.min(venue.max_order_size);

            let fill_prob = self.fill_probability_model.fill_probability(
                venue_qty,
                venue.avg_daily_dark_vol,
            );
            let exp_fill = self.fill_probability_model.expected_fill(
                venue_qty,
                venue.avg_daily_dark_vol,
                venue.avg_fill_size,
            );

            expected_filled += exp_fill;

            allocations.push(DarkVenueAllocation {
                venue_id: venue.id.clone(),
                quantity: venue_qty,
                fill_probability: fill_prob,
                expected_fill: exp_fill,
                score: *score,
            });
        }

        Ok(allocations)
    }

    /// Sum of expected fills across all allocations.
    pub fn total_expected_fill(allocations: &[DarkVenueAllocation]) -> f64 {
        allocations.iter().map(|a| a.expected_fill).sum()
    }

    /// Total quantity sent across all allocations.
    pub fn total_allocated(allocations: &[DarkVenueAllocation]) -> f64 {
        allocations.iter().map(|a| a.quantity).sum()
    }
}

// ---- InformationLeakageTracker -------------------------------------------

/// Detects whether dark orders are being front-run by monitoring price moves
/// after order submission.
///
/// Methodology: for each dark order submitted, record the mid-price at entry.
/// After a `lookback_secs` interval, record the mid-price and compute the
/// adverse price move (bps). If the rolling average adverse move exceeds
/// `alert_threshold_bps`, an alert is raised.
#[derive(Debug, Clone)]
pub struct InformationLeakageTracker {
    /// Seconds after submission to measure adverse move
    pub lookback_secs: u64,
    /// Rolling window of adverse moves (bps) for triggered events
    history: VecDeque<f64>,
    /// Maximum history entries
    max_history: usize,
    /// Threshold in bps -- above this avg the venue is flagged
    pub alert_threshold_bps: f64,
    /// Total orders tracked
    pub n_orders: usize,
    /// Orders flagged with adverse moves exceeding threshold
    pub n_flagged: usize,
}

/// A single dark order observation for leakage analysis.
#[derive(Debug, Clone)]
pub struct DarkOrderEvent {
    pub venue_id: String,
    pub submit_time: u64,
    pub side: OrderSide,
    /// Mid-price at order submission
    pub entry_mid: f64,
    /// Mid-price after `lookback_secs`
    pub post_mid: Option<f64>,
}

impl DarkOrderEvent {
    /// Adverse price move in basis points.
    /// Positive means the price moved against our order.
    pub fn adverse_move_bps(&self) -> Option<f64> {
        let post = self.post_mid?;
        if self.entry_mid <= 0.0 { return None; }
        let raw_move = match self.side {
            OrderSide::Buy  => (post - self.entry_mid) / self.entry_mid * 10_000.0,
            OrderSide::Sell => (self.entry_mid - post) / self.entry_mid * 10_000.0,
        };
        Some(raw_move)
    }
}

impl InformationLeakageTracker {
    pub fn new(lookback_secs: u64, alert_threshold_bps: f64) -> Self {
        InformationLeakageTracker {
            lookback_secs,
            history: VecDeque::new(),
            max_history: 500,
            alert_threshold_bps,
            n_orders: 0,
            n_flagged: 0,
        }
    }

    /// Record a completed observation (entry + post mid known).
    pub fn record(&mut self, event: &DarkOrderEvent) {
        self.n_orders += 1;
        if let Some(adv_bps) = event.adverse_move_bps() {
            if self.history.len() >= self.max_history {
                self.history.pop_front();
            }
            self.history.push_back(adv_bps);
            if adv_bps > self.alert_threshold_bps {
                self.n_flagged += 1;
            }
        }
    }

    /// Rolling average adverse move over all retained observations.
    pub fn avg_adverse_move_bps(&self) -> f64 {
        if self.history.is_empty() { return 0.0; }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    /// Fraction of observations where price moved adversely above threshold.
    pub fn flag_rate(&self) -> f64 {
        if self.n_orders == 0 { return 0.0; }
        self.n_flagged as f64 / self.n_orders as f64
    }

    /// True when average adverse move exceeds the alert threshold.
    pub fn is_leaking(&self) -> bool {
        self.avg_adverse_move_bps() > self.alert_threshold_bps
    }

    /// Leakage score in [0, 1]: 0 = no leakage detected, 1 = severe leakage.
    /// Defined as min(avg_adverse_move / (2 * threshold), 1.0).
    pub fn leakage_score(&self) -> f64 {
        if self.alert_threshold_bps <= 0.0 { return 0.0; }
        let score = self.avg_adverse_move_bps() / (2.0 * self.alert_threshold_bps);
        score.min(1.0).max(0.0)
    }

    /// Update a DarkVenue's leakage score based on observed data.
    pub fn apply_to_venue(&self, venue: &mut DarkVenue) {
        let new_score = self.leakage_score();
        // Exponential smoothing: blend historical score with new evidence
        venue.information_leakage_score =
            0.70 * venue.information_leakage_score + 0.30 * new_score;
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dark_venue(id: &str, fill_rate: f64, leakage: f64, latency: u64, dark_vol: f64) -> DarkVenue {
        DarkVenue::new(id, id)
            .with_fill_rate(fill_rate)
            .with_leakage(leakage)
            .with_latency(latency)
            .with_dark_volume(dark_vol)
            .with_avg_fill_size(10_000.0)
    }

    fn make_router() -> DarkPoolRouter {
        let model = FillProbabilityModel::default_calibration();
        let mut router = DarkPoolRouter::new(model, 0.5);
        router.add_venue(make_dark_venue("SIGMA_X", 0.60, 0.10, 100, 1_000_000.0));
        router.add_venue(make_dark_venue("CROSSFINDER", 0.50, 0.05, 90, 800_000.0));
        router.add_venue(make_dark_venue("UBS_MTF", 0.55, 0.20, 120, 600_000.0));
        router
    }

    #[test]
    fn test_fill_probability_model() {
        let model = FillProbabilityModel::default_calibration();
        // Order size == avg vol: ln(1) = 0, score = sigmoid(b) ~= 0.62
        let p = model.fill_probability(1_000.0, 1_000.0);
        assert!((p - 0.6225).abs() < 0.01, "p={}", p);
        // Very large order: low fill prob
        let p_large = model.fill_probability(100_000.0, 1_000.0);
        assert!(p_large < 0.30, "p_large={}", p_large);
        // Very small order: high fill prob
        let p_small = model.fill_probability(10.0, 1_000.0);
        assert!(p_small > 0.70, "p_small={}", p_small);
    }

    #[test]
    fn test_route_produces_allocations() {
        let router = make_router();
        let allocs = router.route(5_000.0, OrderSide::Buy, "AAPL").unwrap();
        assert!(!allocs.is_empty());
        for a in &allocs {
            assert!(a.fill_probability > 0.0);
            assert!(a.expected_fill >= 0.0);
            assert!(a.quantity > 0.0);
        }
    }

    #[test]
    fn test_route_no_venues_error() {
        let model = FillProbabilityModel::default_calibration();
        let router = DarkPoolRouter::new(model, 0.5);
        let err = router.route(1_000.0, OrderSide::Buy, "AAPL");
        assert!(matches!(err, Err(SorError::NoVenues)));
    }

    #[test]
    fn test_route_sorted_by_score() {
        let router = make_router();
        let allocs = router.route(1_000.0, OrderSide::Buy, "AAPL").unwrap();
        // Scores should be non-increasing
        let scores: Vec<f64> = allocs.iter().map(|a| a.score).collect();
        for i in 1..scores.len() {
            assert!(scores[i - 1] >= scores[i] - 1e-9,
                "Scores not sorted: {} < {}", scores[i - 1], scores[i]);
        }
    }

    #[test]
    fn test_leakage_tracker_no_leakage() {
        let mut tracker = InformationLeakageTracker::new(30, 2.0);
        for i in 0..20 {
            let event = DarkOrderEvent {
                venue_id: "SIGMA_X".to_string(),
                submit_time: i * 60,
                side: OrderSide::Buy,
                entry_mid: 100.0,
                post_mid: Some(100.01), // 1 bps move -- below threshold
            };
            tracker.record(&event);
        }
        assert!(!tracker.is_leaking());
        assert!(tracker.avg_adverse_move_bps() < 2.0);
    }

    #[test]
    fn test_leakage_tracker_leakage_detected() {
        let mut tracker = InformationLeakageTracker::new(30, 2.0);
        for _ in 0..20 {
            let event = DarkOrderEvent {
                venue_id: "BAD_POOL".to_string(),
                submit_time: 0,
                side: OrderSide::Buy,
                entry_mid: 100.0,
                post_mid: Some(100.50), // 50 bps adverse
            };
            tracker.record(&event);
        }
        assert!(tracker.is_leaking());
        assert!(tracker.leakage_score() > 0.5);
    }

    #[test]
    fn test_leakage_apply_to_venue() {
        let mut venue = make_dark_venue("TEST", 0.50, 0.10, 100, 500_000.0);
        let original_score = venue.information_leakage_score;
        let mut tracker = InformationLeakageTracker::new(30, 2.0);
        // Record significant leakage
        for _ in 0..50 {
            tracker.record(&DarkOrderEvent {
                venue_id: "TEST".to_string(),
                submit_time: 0,
                side: OrderSide::Sell,
                entry_mid: 100.0,
                post_mid: Some(99.50), // 50 bps adverse for sell
            });
        }
        tracker.apply_to_venue(&mut venue);
        assert!(venue.information_leakage_score > original_score,
            "leakage score should have increased");
    }

    #[test]
    fn test_venue_score_low_leakage_wins() {
        let model = FillProbabilityModel::default_calibration();
        let router = DarkPoolRouter::new(model, 1.0); // full leakage penalty
        let clean = make_dark_venue("CLEAN", 0.55, 0.01, 100, 500_000.0);
        let leaky = make_dark_venue("LEAKY", 0.55, 0.90, 100, 500_000.0);
        let score_clean = router.venue_score(&clean, 5_000.0);
        let score_leaky = router.venue_score(&leaky, 5_000.0);
        assert!(score_clean > score_leaky,
            "clean score {} should beat leaky score {}", score_clean, score_leaky);
    }
}
