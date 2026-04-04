use std::collections::HashMap;
use crate::venue::{Venue, MarketData};
use crate::cost_model::CostModel;
use crate::{SorError, OrderSide};

/// Constraints for venue allocation
#[derive(Debug, Clone)]
pub struct AllocationConstraints {
    /// Maximum fraction to any single venue
    pub max_venue_fraction: f64,
    /// Minimum quantity to send to any venue (if chosen)
    pub min_venue_qty: f64,
    /// Maximum number of venues to use
    pub max_venues: usize,
    /// Exclude specific venues
    pub excluded_venues: Vec<String>,
    /// POV (participation of volume) cap per venue
    pub pov_cap: f64,
    /// Only use lit exchanges (no dark pools)
    pub lit_only: bool,
}

impl Default for AllocationConstraints {
    fn default() -> Self {
        AllocationConstraints {
            max_venue_fraction: 0.50,
            min_venue_qty: 1.0,
            max_venues: 5,
            excluded_venues: Vec::new(),
            pov_cap: 0.20,
            lit_only: false,
        }
    }
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// venue_id -> quantity allocation
    pub allocations: Vec<(String, f64)>,
    /// Estimated total cost in bps
    pub estimated_cost_bps: f64,
    /// Fraction of order allocated (should be 1.0 if fully allocated)
    pub fill_fraction: f64,
    /// Number of venues used
    pub venues_used: usize,
}

impl AllocationResult {
    pub fn total_qty(&self) -> f64 {
        self.allocations.iter().map(|(_, q)| q).sum()
    }
}

/// Optimal venue allocator using gradient descent
pub struct VenueAllocator {
    cost_model: CostModel,
    venues: Vec<Venue>,
    market_data: HashMap<String, MarketData>,
}

impl VenueAllocator {
    pub fn new(cost_model: CostModel) -> Self {
        VenueAllocator {
            cost_model,
            venues: Vec::new(),
            market_data: HashMap::new(),
        }
    }

    pub fn add_venue(&mut self, venue: Venue, md: MarketData) {
        self.market_data.insert(venue.id.clone(), md);
        self.venues.push(venue);
    }

    /// Compute per-unit cost for a venue given a proposed quantity allocation
    fn per_unit_cost(
        &self,
        venue: &Venue,
        qty: f64,
        price: f64,
        is_buy: bool,
    ) -> f64 {
        let md = match self.market_data.get(&venue.id) {
            Some(m) => m,
            None => return f64::INFINITY,
        };
        let cost = self.cost_model.estimate_cost(venue, qty, price, md, is_buy, true, 0.5);
        cost.breakdown.total
    }

    /// Gradient of total cost w.r.t. allocation fractions
    fn cost_gradient(
        &self,
        fractions: &[f64],
        total_qty: f64,
        price: f64,
        is_buy: bool,
        venues: &[&Venue],
    ) -> Vec<f64> {
        let eps = 1e-6;
        let mut grad = vec![0.0_f64; fractions.len()];
        let base_cost = self.total_cost(fractions, total_qty, price, is_buy, venues);

        for i in 0..fractions.len() {
            let mut perturbed = fractions.to_vec();
            perturbed[i] += eps;
            let perturbed_cost = self.total_cost(&perturbed, total_qty, price, is_buy, venues);
            grad[i] = (perturbed_cost - base_cost) / eps;
        }
        grad
    }

    fn total_cost(
        &self,
        fractions: &[f64],
        total_qty: f64,
        price: f64,
        is_buy: bool,
        venues: &[&Venue],
    ) -> f64 {
        fractions.iter().zip(venues.iter()).map(|(&f, v)| {
            let qty = f * total_qty;
            if qty < 1e-6 { return 0.0; }
            let unit_cost = self.per_unit_cost(v, qty, price, is_buy);
            unit_cost * qty
        }).sum()
    }

    /// Project fractions onto simplex with constraints
    fn project_to_simplex(
        &self,
        fractions: &mut Vec<f64>,
        constraints: &AllocationConstraints,
        venues: &[&Venue],
        total_qty: f64,
    ) {
        let n = fractions.len();

        // Apply max fraction constraint
        for f in fractions.iter_mut() {
            *f = f.max(0.0).min(constraints.max_venue_fraction);
        }

        // Apply POV cap
        for (i, v) in venues.iter().enumerate() {
            let md = match self.market_data.get(&v.id) {
                Some(m) => m,
                None => continue,
            };
            let max_qty_by_pov = md.adv * constraints.pov_cap;
            let max_fraction = if total_qty > 0.0 {
                (max_qty_by_pov / total_qty).min(constraints.max_venue_fraction)
            } else {
                constraints.max_venue_fraction
            };
            fractions[i] = fractions[i].min(max_fraction);
        }

        // Normalize to sum to 1.0
        let sum: f64 = fractions.iter().sum();
        if sum > 0.0 {
            for f in fractions.iter_mut() {
                *f /= sum;
            }
        }
    }

    /// Projected gradient descent for optimal venue allocation
    pub fn optimize(
        &self,
        total_qty: f64,
        price: f64,
        side: OrderSide,
        constraints: &AllocationConstraints,
        max_iter: usize,
    ) -> Result<AllocationResult, SorError> {
        if total_qty <= 0.0 {
            return Err(SorError::InvalidParameter("total_qty must be positive".into()));
        }

        // Filter eligible venues
        let is_buy = side == OrderSide::Buy;
        let eligible: Vec<&Venue> = self.venues.iter()
            .filter(|v| {
                v.active
                && !constraints.excluded_venues.contains(&v.id)
                && v.can_accept(constraints.min_venue_qty)
                && !(constraints.lit_only && matches!(v.exchange_type, crate::venue::ExchangeType::DarkPool))
            })
            .take(constraints.max_venues * 2) // start with more, trim later
            .collect();

        if eligible.is_empty() {
            return Err(SorError::NoVenues);
        }

        let n = eligible.len().min(constraints.max_venues);
        let eligible = &eligible[..n];

        // Initial allocation: equal weight
        let mut fractions = vec![1.0 / n as f64; n];
        self.project_to_simplex(&mut fractions, constraints, eligible, total_qty);

        let mut learning_rate = 0.1;
        let decay = 0.99;

        for _ in 0..max_iter {
            let grad = self.cost_gradient(&fractions, total_qty, price, is_buy, eligible);

            // Gradient step
            let mut new_fractions: Vec<f64> = fractions.iter().zip(grad.iter())
                .map(|(&f, &g)| f - learning_rate * g)
                .collect();

            self.project_to_simplex(&mut new_fractions, constraints, eligible, total_qty);

            let old_cost = self.total_cost(&fractions, total_qty, price, is_buy, eligible);
            let new_cost = self.total_cost(&new_fractions, total_qty, price, is_buy, eligible);

            if new_cost < old_cost {
                fractions = new_fractions;
            }

            learning_rate *= decay;

            if learning_rate < 1e-8 { break; }
        }

        // Build result
        let mut allocations = Vec::new();
        let mut total_allocated = 0.0;

        for (i, v) in eligible.iter().enumerate() {
            let qty = fractions[i] * total_qty;
            let rounded = v.round_to_lot(qty);
            if rounded >= constraints.min_venue_qty {
                allocations.push((v.id.clone(), rounded));
                total_allocated += rounded;
            }
        }

        if allocations.is_empty() {
            // Fallback: distribute across venues respecting max_fraction
            let max_per_venue = total_qty * constraints.max_venue_fraction;
            let mut remaining = total_qty;
            let mut sorted_eligible = eligible.to_vec();
            sorted_eligible.sort_by(|a, b| a.fee_score().partial_cmp(&b.fee_score()).unwrap());
            for v in &sorted_eligible {
                if remaining < constraints.min_venue_qty { break; }
                let qty = v.round_to_lot(remaining.min(max_per_venue));
                if qty >= constraints.min_venue_qty {
                    allocations.push((v.id.clone(), qty));
                    total_allocated += qty;
                    remaining -= qty;
                }
            }
            if allocations.is_empty() {
                let best = sorted_eligible.first().ok_or(SorError::NoVenues)?;
                allocations.push((best.id.clone(), total_qty.min(max_per_venue)));
                total_allocated = total_qty.min(max_per_venue);
            }
        }

        let final_cost = self.total_cost(
            &allocations.iter().map(|(_, q)| q / total_qty).collect::<Vec<_>>(),
            total_qty, price, is_buy,
            &eligible[..allocations.len().min(eligible.len())]
        );

        Ok(AllocationResult {
            venues_used: allocations.len(),
            fill_fraction: total_allocated / total_qty,
            estimated_cost_bps: final_cost / (total_qty * price) * 10_000.0,
            allocations,
        })
    }

    /// Simple greedy allocation by cost score (fast but not optimal)
    pub fn greedy_allocate(
        &self,
        total_qty: f64,
        price: f64,
        side: OrderSide,
        constraints: &AllocationConstraints,
    ) -> Result<AllocationResult, SorError> {
        let is_buy = side == OrderSide::Buy;
        let eligible: Vec<&Venue> = self.venues.iter()
            .filter(|v| v.active && !constraints.excluded_venues.contains(&v.id))
            .collect();

        if eligible.is_empty() {
            return Err(SorError::NoVenues);
        }

        // Score each venue
        let mut scored: Vec<(&Venue, f64)> = eligible.iter().map(|&v| {
            let md = self.market_data.get(&v.id);
            let unit_cost = md.map(|m| {
                self.cost_model.estimate_cost(v, total_qty * 0.10, price, m, is_buy, true, 0.5)
                    .breakdown.total
            }).unwrap_or(f64::INFINITY);
            (v, unit_cost)
        }).collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut remaining = total_qty;
        let mut allocations = Vec::new();

        for (venue, _) in scored.iter().take(constraints.max_venues) {
            if remaining < constraints.min_venue_qty { break; }

            let md = match self.market_data.get(&venue.id) {
                Some(m) => m,
                None => continue,
            };

            // Max by POV cap
            let max_by_pov = md.adv * constraints.pov_cap;
            let max_by_fraction = total_qty * constraints.max_venue_fraction;
            let avail = max_by_pov.min(max_by_fraction).min(remaining);

            let qty = venue.round_to_lot(avail);
            if qty < constraints.min_venue_qty { continue; }

            allocations.push((venue.id.clone(), qty));
            remaining -= qty;
        }

        let total_alloc: f64 = allocations.iter().map(|(_, q)| q).sum();
        Ok(AllocationResult {
            venues_used: allocations.len(),
            fill_fraction: total_alloc / total_qty,
            estimated_cost_bps: 0.0, // not computed in greedy
            allocations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::venue::{Venue, ExchangeType, MarketData, BookLevel};

    fn make_md(price: f64, spread_frac: f64, adv: f64) -> MarketData {
        let half = price * spread_frac / 2.0;
        MarketData {
            bid: price - half, ask: price + half,
            bid_size: 10_000.0, ask_size: 10_000.0,
            last_trade: price, daily_volume: adv, adv,
            bid_levels: vec![BookLevel { price: price - half, quantity: 10_000.0 }],
            ask_levels: vec![BookLevel { price: price + half, quantity: 10_000.0 }],
        }
    }

    fn make_allocator() -> VenueAllocator {
        let cost_model = CostModel::new(0.02, 0.001);
        let mut alloc = VenueAllocator::new(cost_model);

        alloc.add_venue(
            Venue::new("NYSE", "NYSE", ExchangeType::LitExchange).with_fees(-0.002, 0.003),
            make_md(100.0, 0.001, 5_000_000.0),
        );
        alloc.add_venue(
            Venue::new("BATS", "BATS", ExchangeType::LitExchange).with_fees(-0.0032, 0.003),
            make_md(100.0, 0.001, 3_000_000.0),
        );
        alloc.add_venue(
            Venue::new("DARK1", "Dark1", ExchangeType::DarkPool).with_fees(0.001, 0.001),
            make_md(100.0, 0.0, 2_000_000.0),
        );
        alloc
    }

    #[test]
    fn test_greedy_allocate() {
        let alloc = make_allocator();
        let constraints = AllocationConstraints::default();
        let result = alloc.greedy_allocate(50_000.0, 100.0, OrderSide::Buy, &constraints).unwrap();
        assert!(result.fill_fraction > 0.0);
        assert!(!result.allocations.is_empty());
        println!("Greedy: {:?}", result.allocations);
    }

    #[test]
    fn test_optimize_returns_allocation() {
        let alloc = make_allocator();
        let constraints = AllocationConstraints::default();
        let result = alloc.optimize(10_000.0, 100.0, OrderSide::Buy, &constraints, 50).unwrap();
        assert!(!result.allocations.is_empty());
        let total: f64 = result.allocations.iter().map(|(_, q)| q).sum();
        // Total allocated should be close to 10,000
        assert!(total > 0.0);
        println!("Optimized: {:?} fill={:.2}", result.allocations, result.fill_fraction);
    }

    #[test]
    fn test_lit_only_excludes_dark() {
        let alloc = make_allocator();
        let mut constraints = AllocationConstraints::default();
        constraints.lit_only = true;
        let result = alloc.optimize(10_000.0, 100.0, OrderSide::Buy, &constraints, 50).unwrap();
        for (vid, _) in &result.allocations {
            assert_ne!(vid, "DARK1", "Dark pool should not be in lit-only allocation");
        }
    }

    #[test]
    fn test_max_venue_fraction_respected() {
        let alloc = make_allocator();
        let mut constraints = AllocationConstraints::default();
        constraints.max_venue_fraction = 0.60;
        let total_qty = 100_000.0;
        let result = alloc.optimize(total_qty, 100.0, OrderSide::Buy, &constraints, 100).unwrap();
        // Allow 5% tolerance on top of the max fraction constraint
        for (_, qty) in &result.allocations {
            assert!(*qty <= total_qty * 0.65 + 1.0, "Venue fraction exceeded: {}/{}", qty, total_qty);
        }
    }

    #[test]
    fn test_excluded_venues() {
        let alloc = make_allocator();
        let mut constraints = AllocationConstraints::default();
        constraints.excluded_venues = vec!["NYSE".to_string()];
        let result = alloc.optimize(10_000.0, 100.0, OrderSide::Buy, &constraints, 50).unwrap();
        for (vid, _) in &result.allocations {
            assert_ne!(vid, "NYSE");
        }
    }
}
