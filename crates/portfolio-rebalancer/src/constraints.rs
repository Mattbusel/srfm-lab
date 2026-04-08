// ═══════════════════════════════════════════════════════════════════════════
// CONSTRAINT TYPES
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum Constraint {
    BoxConstraint(BoxConstraint),
    GroupConstraint(GroupConstraint),
    TurnoverConstraint(TurnoverConstraint),
    CardinalityConstraint(CardinalityConstraint),
    TrackingErrorConstraint(TrackingErrorConstraint),
    FactorExposureConstraint(FactorExposureConstraint),
    ShortSellConstraint(ShortSellConstraint),
    LeverageConstraint(LeverageConstraint),
    PairConstraint(PairConstraint),
    LinearConstraint(LinearConstraint),
}

/// Check if all constraints are satisfied.
pub fn check_constraints(weights: &[f64], constraints: &[Constraint], old_weights: Option<&[f64]>) -> ConstraintCheckResult {
    let mut result = ConstraintCheckResult {
        feasible: true,
        violations: Vec::new(),
        max_violation: 0.0,
    };

    for constraint in constraints {
        match constraint {
            Constraint::BoxConstraint(c) => {
                let violations = c.check(weights);
                for v in violations {
                    result.add_violation(v);
                }
            }
            Constraint::GroupConstraint(c) => {
                if let Some(v) = c.check(weights) {
                    result.add_violation(v);
                }
            }
            Constraint::TurnoverConstraint(c) => {
                if let Some(old) = old_weights {
                    if let Some(v) = c.check(weights, old) {
                        result.add_violation(v);
                    }
                }
            }
            Constraint::CardinalityConstraint(c) => {
                if let Some(v) = c.check(weights) {
                    result.add_violation(v);
                }
            }
            Constraint::TrackingErrorConstraint(c) => {
                if let Some(v) = c.check(weights) {
                    result.add_violation(v);
                }
            }
            Constraint::FactorExposureConstraint(c) => {
                if let Some(v) = c.check(weights) {
                    result.add_violation(v);
                }
            }
            Constraint::ShortSellConstraint(c) => {
                let violations = c.check(weights);
                for v in violations {
                    result.add_violation(v);
                }
            }
            Constraint::LeverageConstraint(c) => {
                if let Some(v) = c.check(weights) {
                    result.add_violation(v);
                }
            }
            Constraint::PairConstraint(c) => {
                if let Some(v) = c.check(weights) {
                    result.add_violation(v);
                }
            }
            Constraint::LinearConstraint(c) => {
                if let Some(v) = c.check(weights) {
                    result.add_violation(v);
                }
            }
        }
    }

    result
}

#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    pub constraint_name: String,
    pub violation_amount: f64,
    pub details: String,
}

#[derive(Debug, Clone)]
pub struct ConstraintCheckResult {
    pub feasible: bool,
    pub violations: Vec<ConstraintViolation>,
    pub max_violation: f64,
}

impl ConstraintCheckResult {
    fn add_violation(&mut self, v: ConstraintViolation) {
        if v.violation_amount > self.max_violation {
            self.max_violation = v.violation_amount;
        }
        self.feasible = false;
        self.violations.push(v);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BOX CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Per-asset weight bounds: lower_i <= w_i <= upper_i
#[derive(Debug, Clone)]
pub struct BoxConstraint {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl BoxConstraint {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        Self { lower, upper }
    }

    pub fn uniform(n: usize, lower: f64, upper: f64) -> Self {
        Self {
            lower: vec![lower; n],
            upper: vec![upper; n],
        }
    }

    pub fn long_only(n: usize) -> Self {
        Self::uniform(n, 0.0, 1.0)
    }

    pub fn long_only_capped(n: usize, max_weight: f64) -> Self {
        Self::uniform(n, 0.0, max_weight)
    }

    pub fn check(&self, weights: &[f64]) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();
        for (i, &w) in weights.iter().enumerate() {
            if i < self.lower.len() && w < self.lower[i] - 1e-10 {
                violations.push(ConstraintViolation {
                    constraint_name: format!("Box lower bound asset {}", i),
                    violation_amount: self.lower[i] - w,
                    details: format!("w[{}] = {:.6} < lower = {:.6}", i, w, self.lower[i]),
                });
            }
            if i < self.upper.len() && w > self.upper[i] + 1e-10 {
                violations.push(ConstraintViolation {
                    constraint_name: format!("Box upper bound asset {}", i),
                    violation_amount: w - self.upper[i],
                    details: format!("w[{}] = {:.6} > upper = {:.6}", i, w, self.upper[i]),
                });
            }
        }
        violations
    }

    /// Project weights onto box constraints.
    pub fn project(&self, weights: &mut [f64]) {
        for (i, w) in weights.iter_mut().enumerate() {
            if i < self.lower.len() {
                *w = w.max(self.lower[i]);
            }
            if i < self.upper.len() {
                *w = w.min(self.upper[i]);
            }
        }
    }

    /// Clamp a single weight.
    pub fn clamp(&self, idx: usize, w: f64) -> f64 {
        let lo = if idx < self.lower.len() { self.lower[idx] } else { f64::NEG_INFINITY };
        let hi = if idx < self.upper.len() { self.upper[idx] } else { f64::INFINITY };
        w.max(lo).min(hi)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GROUP CONSTRAINTS (SECTOR / COUNTRY / FACTOR GROUPS)
// ═══════════════════════════════════════════════════════════════════════════

/// Group constraint: lower <= sum(w_i for i in group) <= upper
#[derive(Debug, Clone)]
pub struct GroupConstraint {
    pub name: String,
    pub members: Vec<usize>,  // indices of assets in this group
    pub lower: f64,
    pub upper: f64,
}

impl GroupConstraint {
    pub fn new(name: String, members: Vec<usize>, lower: f64, upper: f64) -> Self {
        Self { name, members, lower, upper }
    }

    pub fn sector(name: &str, members: Vec<usize>, lower: f64, upper: f64) -> Self {
        Self::new(format!("Sector:{}", name), members, lower, upper)
    }

    pub fn country(name: &str, members: Vec<usize>, lower: f64, upper: f64) -> Self {
        Self::new(format!("Country:{}", name), members, lower, upper)
    }

    pub fn group_weight(&self, weights: &[f64]) -> f64 {
        self.members.iter()
            .filter_map(|&i| weights.get(i))
            .sum()
    }

    pub fn check(&self, weights: &[f64]) -> Option<ConstraintViolation> {
        let gw = self.group_weight(weights);
        if gw < self.lower - 1e-10 {
            Some(ConstraintViolation {
                constraint_name: self.name.clone(),
                violation_amount: self.lower - gw,
                details: format!("Group weight {:.6} < lower {:.6}", gw, self.lower),
            })
        } else if gw > self.upper + 1e-10 {
            Some(ConstraintViolation {
                constraint_name: self.name.clone(),
                violation_amount: gw - self.upper,
                details: format!("Group weight {:.6} > upper {:.6}", gw, self.upper),
            })
        } else {
            None
        }
    }

    /// Project: scale group weights proportionally to meet bounds.
    pub fn project(&self, weights: &mut [f64]) {
        let gw = self.group_weight(weights);
        if gw > self.upper && gw > 1e-15 {
            let scale = self.upper / gw;
            for &i in &self.members {
                if i < weights.len() {
                    weights[i] *= scale;
                }
            }
        } else if gw < self.lower {
            let deficit = self.lower - gw;
            let n = self.members.len() as f64;
            for &i in &self.members {
                if i < weights.len() {
                    weights[i] += deficit / n;
                }
            }
        }
    }
}

/// Multiple group constraints system.
#[derive(Debug, Clone)]
pub struct GroupConstraintSet {
    pub groups: Vec<GroupConstraint>,
}

impl GroupConstraintSet {
    pub fn new(groups: Vec<GroupConstraint>) -> Self {
        Self { groups }
    }

    pub fn check_all(&self, weights: &[f64]) -> Vec<ConstraintViolation> {
        self.groups.iter()
            .filter_map(|g| g.check(weights))
            .collect()
    }

    pub fn is_feasible(&self, weights: &[f64]) -> bool {
        self.groups.iter().all(|g| g.check(weights).is_none())
    }

    /// Iterative projection onto all group constraints.
    pub fn project(&self, weights: &mut [f64], max_iter: usize) {
        for _ in 0..max_iter {
            let mut any_violation = false;
            for group in &self.groups {
                if group.check(weights).is_some() {
                    group.project(weights);
                    any_violation = true;
                }
            }
            if !any_violation {
                break;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TURNOVER CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Turnover constraint: sum |w_new - w_old| <= max_turnover
#[derive(Debug, Clone)]
pub struct TurnoverConstraint {
    pub max_one_way_turnover: f64,  // max sum of buys (or sells)
    pub max_two_way_turnover: f64,  // max sum of |trades|
    pub per_asset_max_trade: Option<Vec<f64>>,  // per-asset trade limits
}

impl TurnoverConstraint {
    pub fn new(max_two_way: f64) -> Self {
        Self {
            max_one_way_turnover: max_two_way / 2.0,
            max_two_way_turnover: max_two_way,
            per_asset_max_trade: None,
        }
    }

    pub fn with_per_asset(max_two_way: f64, per_asset: Vec<f64>) -> Self {
        Self {
            max_one_way_turnover: max_two_way / 2.0,
            max_two_way_turnover: max_two_way,
            per_asset_max_trade: Some(per_asset),
        }
    }

    pub fn turnover(&self, new_weights: &[f64], old_weights: &[f64]) -> f64 {
        new_weights.iter().zip(old_weights.iter())
            .map(|(n, o)| (n - o).abs())
            .sum()
    }

    pub fn one_way_turnover(&self, new_weights: &[f64], old_weights: &[f64]) -> (f64, f64) {
        let mut buys = 0.0;
        let mut sells = 0.0;
        for (n, o) in new_weights.iter().zip(old_weights.iter()) {
            let trade = n - o;
            if trade > 0.0 { buys += trade; }
            else { sells += -trade; }
        }
        (buys, sells)
    }

    pub fn check(&self, new_weights: &[f64], old_weights: &[f64]) -> Option<ConstraintViolation> {
        let turnover = self.turnover(new_weights, old_weights);
        if turnover > self.max_two_way_turnover + 1e-10 {
            return Some(ConstraintViolation {
                constraint_name: "Turnover".into(),
                violation_amount: turnover - self.max_two_way_turnover,
                details: format!("Turnover {:.6} > max {:.6}", turnover, self.max_two_way_turnover),
            });
        }

        if let Some(ref per_asset) = self.per_asset_max_trade {
            for (i, ((n, o), &max_t)) in new_weights.iter().zip(old_weights.iter()).zip(per_asset.iter()).enumerate() {
                let trade = (n - o).abs();
                if trade > max_t + 1e-10 {
                    return Some(ConstraintViolation {
                        constraint_name: format!("Per-asset turnover [{}]", i),
                        violation_amount: trade - max_t,
                        details: format!("Trade {:.6} > max {:.6} for asset {}", trade, max_t, i),
                    });
                }
            }
        }

        None
    }

    /// Project new weights to satisfy turnover constraint.
    pub fn project(&self, new_weights: &mut [f64], old_weights: &[f64]) {
        let turnover = self.turnover(new_weights, old_weights);
        if turnover <= self.max_two_way_turnover {
            return;
        }

        // Scale trades proportionally
        let scale = self.max_two_way_turnover / turnover;
        for (i, w) in new_weights.iter_mut().enumerate() {
            let trade = *w - old_weights[i];
            *w = old_weights[i] + trade * scale;
        }

        // Apply per-asset limits
        if let Some(ref per_asset) = self.per_asset_max_trade {
            for (i, w) in new_weights.iter_mut().enumerate() {
                if i < per_asset.len() {
                    let trade = *w - old_weights[i];
                    if trade.abs() > per_asset[i] {
                        *w = old_weights[i] + trade.signum() * per_asset[i];
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CARDINALITY CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Cardinality constraint: max number of non-zero positions, min position size.
#[derive(Debug, Clone)]
pub struct CardinalityConstraint {
    pub max_positions: usize,
    pub min_position_size: f64,
    pub max_position_size: f64,
}

impl CardinalityConstraint {
    pub fn new(max_positions: usize, min_size: f64, max_size: f64) -> Self {
        Self { max_positions, min_position_size: min_size, max_position_size: max_size }
    }

    pub fn active_count(&self, weights: &[f64]) -> usize {
        weights.iter().filter(|&&w| w.abs() > self.min_position_size * 0.5).count()
    }

    pub fn check(&self, weights: &[f64]) -> Option<ConstraintViolation> {
        let count = self.active_count(weights);
        if count > self.max_positions {
            return Some(ConstraintViolation {
                constraint_name: "Cardinality".into(),
                violation_amount: (count - self.max_positions) as f64,
                details: format!("{} positions > max {}", count, self.max_positions),
            });
        }

        // Check min/max position sizes
        for (i, &w) in weights.iter().enumerate() {
            if w.abs() > 1e-10 && w.abs() < self.min_position_size {
                return Some(ConstraintViolation {
                    constraint_name: format!("Min position size [{}]", i),
                    violation_amount: self.min_position_size - w.abs(),
                    details: format!("|w[{}]| = {:.6} < min {:.6}", i, w.abs(), self.min_position_size),
                });
            }
        }

        None
    }

    /// Enforce cardinality: zero out smallest positions, redistribute.
    pub fn project(&self, weights: &mut [f64]) {
        let n = weights.len();

        // Sort by absolute weight
        let mut indexed: Vec<(usize, f64)> = weights.iter().enumerate().map(|(i, &w)| (i, w.abs())).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Zero out positions beyond max_positions
        let mut zeroed_sum = 0.0;
        for i in self.max_positions..indexed.len() {
            let idx = indexed[i].0;
            zeroed_sum += weights[idx];
            weights[idx] = 0.0;
        }

        // Redistribute zeroed weight proportionally among remaining
        if self.max_positions > 0 && indexed.len() > self.max_positions {
            let active_sum: f64 = (0..self.max_positions).map(|i| weights[indexed[i].0]).sum::<f64>();
            if active_sum.abs() > 1e-15 {
                for i in 0..self.max_positions {
                    let idx = indexed[i].0;
                    weights[idx] += zeroed_sum * weights[idx] / active_sum;
                }
            }
        }

        // Enforce minimum position size
        for w in weights.iter_mut() {
            if w.abs() > 0.0 && w.abs() < self.min_position_size {
                *w = 0.0;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRACKING ERROR CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Tracking error constraint: TE = sqrt((w-b)'Σ(w-b)) <= max_te
#[derive(Debug, Clone)]
pub struct TrackingErrorConstraint {
    pub benchmark_weights: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub max_tracking_error: f64,
}

impl TrackingErrorConstraint {
    pub fn new(benchmark: Vec<f64>, cov: Vec<Vec<f64>>, max_te: f64) -> Self {
        Self { benchmark_weights: benchmark, covariance: cov, max_tracking_error: max_te }
    }

    pub fn tracking_error(&self, weights: &[f64]) -> f64 {
        let n = weights.len();
        let mut active: Vec<f64> = weights.iter().zip(self.benchmark_weights.iter())
            .map(|(w, b)| w - b)
            .collect();

        let mut te_var = 0.0;
        for i in 0..n {
            for j in 0..n {
                te_var += active[i] * self.covariance[i][j] * active[j];
            }
        }
        te_var.max(0.0).sqrt()
    }

    pub fn active_weights(&self, weights: &[f64]) -> Vec<f64> {
        weights.iter().zip(self.benchmark_weights.iter())
            .map(|(w, b)| w - b)
            .collect()
    }

    pub fn check(&self, weights: &[f64]) -> Option<ConstraintViolation> {
        let te = self.tracking_error(weights);
        if te > self.max_tracking_error + 1e-10 {
            Some(ConstraintViolation {
                constraint_name: "Tracking Error".into(),
                violation_amount: te - self.max_tracking_error,
                details: format!("TE {:.4}% > max {:.4}%", te * 100.0, self.max_tracking_error * 100.0),
            })
        } else {
            None
        }
    }

    /// Project: move weights towards benchmark to reduce TE.
    pub fn project(&self, weights: &mut [f64]) {
        let te = self.tracking_error(weights);
        if te <= self.max_tracking_error {
            return;
        }

        let scale = self.max_tracking_error / te;
        for (i, w) in weights.iter_mut().enumerate() {
            if i < self.benchmark_weights.len() {
                let active = *w - self.benchmark_weights[i];
                *w = self.benchmark_weights[i] + active * scale;
            }
        }
    }

    /// Information ratio: excess return / tracking error.
    pub fn information_ratio(&self, weights: &[f64], returns: &[f64], benchmark_returns: &[f64]) -> f64 {
        let n = weights.len();
        let excess_return: f64 = weights.iter().zip(returns.iter())
            .map(|(w, r)| w * r)
            .sum::<f64>()
            - self.benchmark_weights.iter().zip(benchmark_returns.iter())
                .map(|(w, r)| w * r)
                .sum::<f64>();
        let te = self.tracking_error(weights);
        if te > 1e-15 { excess_return / te } else { 0.0 }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FACTOR EXPOSURE CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

/// Factor exposure constraint: lower <= w'β_k <= upper for each factor k
#[derive(Debug, Clone)]
pub struct FactorExposureConstraint {
    pub factor_name: String,
    pub factor_loadings: Vec<f64>,  // β_k for each asset
    pub lower: f64,
    pub upper: f64,
    pub benchmark_exposure: f64,    // for relative constraints
}

impl FactorExposureConstraint {
    pub fn new(name: String, loadings: Vec<f64>, lower: f64, upper: f64) -> Self {
        Self {
            factor_name: name,
            factor_loadings: loadings,
            lower,
            upper,
            benchmark_exposure: 0.0,
        }
    }

    pub fn relative(name: String, loadings: Vec<f64>, benchmark_weights: &[f64], max_active: f64) -> Self {
        let bench_exp: f64 = loadings.iter().zip(benchmark_weights.iter())
            .map(|(b, w)| b * w)
            .sum();
        Self {
            factor_name: name,
            factor_loadings: loadings,
            lower: bench_exp - max_active,
            upper: bench_exp + max_active,
            benchmark_exposure: bench_exp,
        }
    }

    pub fn exposure(&self, weights: &[f64]) -> f64 {
        weights.iter().zip(self.factor_loadings.iter())
            .map(|(w, b)| w * b)
            .sum()
    }

    pub fn active_exposure(&self, weights: &[f64]) -> f64 {
        self.exposure(weights) - self.benchmark_exposure
    }

    pub fn check(&self, weights: &[f64]) -> Option<ConstraintViolation> {
        let exp = self.exposure(weights);
        if exp < self.lower - 1e-10 {
            Some(ConstraintViolation {
                constraint_name: format!("Factor:{} lower", self.factor_name),
                violation_amount: self.lower - exp,
                details: format!("{} exposure {:.4} < lower {:.4}", self.factor_name, exp, self.lower),
            })
        } else if exp > self.upper + 1e-10 {
            Some(ConstraintViolation {
                constraint_name: format!("Factor:{} upper", self.factor_name),
                violation_amount: exp - self.upper,
                details: format!("{} exposure {:.4} > upper {:.4}", self.factor_name, exp, self.upper),
            })
        } else {
            None
        }
    }
}

/// Multi-factor exposure constraints.
#[derive(Debug, Clone)]
pub struct MultiFactorConstraints {
    pub factors: Vec<FactorExposureConstraint>,
}

impl MultiFactorConstraints {
    pub fn new(factors: Vec<FactorExposureConstraint>) -> Self {
        Self { factors }
    }

    pub fn check_all(&self, weights: &[f64]) -> Vec<ConstraintViolation> {
        self.factors.iter()
            .filter_map(|f| f.check(weights))
            .collect()
    }

    pub fn exposures(&self, weights: &[f64]) -> Vec<(String, f64)> {
        self.factors.iter()
            .map(|f| (f.factor_name.clone(), f.exposure(weights)))
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SHORT-SELL CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ShortSellConstraint {
    pub allow_short: bool,
    pub max_short_per_asset: f64,      // maximum short weight per asset
    pub max_total_short: f64,          // maximum total short exposure
    pub hard_to_borrow: Vec<usize>,    // indices of assets that cannot be shorted
    pub borrow_costs: Vec<f64>,        // annualized borrow cost per asset
}

impl ShortSellConstraint {
    pub fn no_shorting(n: usize) -> Self {
        Self {
            allow_short: false,
            max_short_per_asset: 0.0,
            max_total_short: 0.0,
            hard_to_borrow: vec![],
            borrow_costs: vec![0.0; n],
        }
    }

    pub fn with_limits(max_per_asset: f64, max_total: f64) -> Self {
        Self {
            allow_short: true,
            max_short_per_asset: max_per_asset,
            max_total_short: max_total,
            hard_to_borrow: vec![],
            borrow_costs: vec![],
        }
    }

    pub fn total_short(&self, weights: &[f64]) -> f64 {
        weights.iter().filter(|&&w| w < 0.0).map(|w| w.abs()).sum()
    }

    pub fn total_long(&self, weights: &[f64]) -> f64 {
        weights.iter().filter(|&&w| w > 0.0).sum()
    }

    pub fn gross_exposure(&self, weights: &[f64]) -> f64 {
        weights.iter().map(|w| w.abs()).sum()
    }

    pub fn net_exposure(&self, weights: &[f64]) -> f64 {
        weights.iter().sum()
    }

    pub fn check(&self, weights: &[f64]) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        if !self.allow_short {
            for (i, &w) in weights.iter().enumerate() {
                if w < -1e-10 {
                    violations.push(ConstraintViolation {
                        constraint_name: format!("No short-sell [{}]", i),
                        violation_amount: -w,
                        details: format!("w[{}] = {:.6} < 0", i, w),
                    });
                }
            }
            return violations;
        }

        // Per-asset short limits
        for (i, &w) in weights.iter().enumerate() {
            if w < -self.max_short_per_asset - 1e-10 {
                violations.push(ConstraintViolation {
                    constraint_name: format!("Max short [{}]", i),
                    violation_amount: -w - self.max_short_per_asset,
                    details: format!("w[{}] = {:.6} < -{:.6}", i, w, self.max_short_per_asset),
                });
            }
        }

        // Hard-to-borrow
        for &i in &self.hard_to_borrow {
            if i < weights.len() && weights[i] < -1e-10 {
                violations.push(ConstraintViolation {
                    constraint_name: format!("Hard to borrow [{}]", i),
                    violation_amount: -weights[i],
                    details: format!("Cannot short asset {}: w = {:.6}", i, weights[i]),
                });
            }
        }

        // Total short
        let total = self.total_short(weights);
        if total > self.max_total_short + 1e-10 {
            violations.push(ConstraintViolation {
                constraint_name: "Total short exposure".into(),
                violation_amount: total - self.max_total_short,
                details: format!("Total short {:.4} > max {:.4}", total, self.max_total_short),
            });
        }

        violations
    }

    /// Total borrow cost for the portfolio.
    pub fn borrow_cost(&self, weights: &[f64]) -> f64 {
        if self.borrow_costs.is_empty() {
            return 0.0;
        }
        weights.iter().enumerate().map(|(i, &w)| {
            if w < 0.0 && i < self.borrow_costs.len() {
                w.abs() * self.borrow_costs[i]
            } else {
                0.0
            }
        }).sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LEVERAGE CONSTRAINT
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct LeverageConstraint {
    pub max_gross_leverage: f64,  // sum |w_i|
    pub max_net_leverage: f64,    // |sum w_i|
}

impl LeverageConstraint {
    pub fn new(max_gross: f64, max_net: f64) -> Self {
        Self { max_gross_leverage: max_gross, max_net_leverage: max_net }
    }

    pub fn long_only() -> Self {
        Self { max_gross_leverage: 1.0, max_net_leverage: 1.0 }
    }

    pub fn check(&self, weights: &[f64]) -> Option<ConstraintViolation> {
        let gross: f64 = weights.iter().map(|w| w.abs()).sum();
        if gross > self.max_gross_leverage + 1e-10 {
            return Some(ConstraintViolation {
                constraint_name: "Gross leverage".into(),
                violation_amount: gross - self.max_gross_leverage,
                details: format!("Gross {:.4} > max {:.4}", gross, self.max_gross_leverage),
            });
        }
        let net: f64 = weights.iter().sum::<f64>().abs();
        if net > self.max_net_leverage + 1e-10 {
            return Some(ConstraintViolation {
                constraint_name: "Net leverage".into(),
                violation_amount: net - self.max_net_leverage,
                details: format!("Net {:.4} > max {:.4}", net, self.max_net_leverage),
            });
        }
        None
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PAIR CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PairConstraint {
    pub asset_a: usize,
    pub asset_b: usize,
    pub constraint_type: PairConstraintType,
}

#[derive(Debug, Clone)]
pub enum PairConstraintType {
    WeightRatio { min: f64, max: f64 },      // w_a / w_b in [min, max]
    WeightDifference { min: f64, max: f64 },  // w_a - w_b in [min, max]
    MutuallyExclusive,                         // at most one can be non-zero
}

impl PairConstraint {
    pub fn check(&self, weights: &[f64]) -> Option<ConstraintViolation> {
        let wa = if self.asset_a < weights.len() { weights[self.asset_a] } else { 0.0 };
        let wb = if self.asset_b < weights.len() { weights[self.asset_b] } else { 0.0 };

        match &self.constraint_type {
            PairConstraintType::WeightRatio { min, max } => {
                if wb.abs() < 1e-15 {
                    if wa.abs() > 1e-10 {
                        return Some(ConstraintViolation {
                            constraint_name: "Pair ratio".into(),
                            violation_amount: wa.abs(),
                            details: format!("w[{}]/w[{}]: denominator is zero", self.asset_a, self.asset_b),
                        });
                    }
                    return None;
                }
                let ratio = wa / wb;
                if ratio < *min - 1e-10 || ratio > *max + 1e-10 {
                    Some(ConstraintViolation {
                        constraint_name: "Pair ratio".into(),
                        violation_amount: (ratio - (*min + *max) / 2.0).abs(),
                        details: format!("w[{}]/w[{}] = {:.4} not in [{:.4}, {:.4}]", self.asset_a, self.asset_b, ratio, min, max),
                    })
                } else {
                    None
                }
            }
            PairConstraintType::WeightDifference { min, max } => {
                let diff = wa - wb;
                if diff < *min - 1e-10 || diff > *max + 1e-10 {
                    Some(ConstraintViolation {
                        constraint_name: "Pair difference".into(),
                        violation_amount: (diff - (*min + *max) / 2.0).abs(),
                        details: format!("w[{}]-w[{}] = {:.4} not in [{:.4}, {:.4}]", self.asset_a, self.asset_b, diff, min, max),
                    })
                } else {
                    None
                }
            }
            PairConstraintType::MutuallyExclusive => {
                if wa.abs() > 1e-10 && wb.abs() > 1e-10 {
                    Some(ConstraintViolation {
                        constraint_name: "Mutually exclusive".into(),
                        violation_amount: wa.abs().min(wb.abs()),
                        details: format!("Both w[{}]={:.4} and w[{}]={:.4} are non-zero", self.asset_a, wa, self.asset_b, wb),
                    })
                } else {
                    None
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LINEAR CONSTRAINTS (GENERIC)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct LinearConstraint {
    pub name: String,
    pub coefficients: Vec<f64>,  // a'w
    pub lower: f64,               // lower <= a'w
    pub upper: f64,               // a'w <= upper
}

impl LinearConstraint {
    pub fn new(name: String, coefficients: Vec<f64>, lower: f64, upper: f64) -> Self {
        Self { name, coefficients, lower, upper }
    }

    pub fn equality(name: String, coefficients: Vec<f64>, target: f64) -> Self {
        Self { name, coefficients, lower: target, upper: target }
    }

    pub fn evaluate(&self, weights: &[f64]) -> f64 {
        self.coefficients.iter().zip(weights.iter()).map(|(a, w)| a * w).sum()
    }

    pub fn check(&self, weights: &[f64]) -> Option<ConstraintViolation> {
        let val = self.evaluate(weights);
        if val < self.lower - 1e-10 {
            Some(ConstraintViolation {
                constraint_name: self.name.clone(),
                violation_amount: self.lower - val,
                details: format!("{} = {:.6} < lower {:.6}", self.name, val, self.lower),
            })
        } else if val > self.upper + 1e-10 {
            Some(ConstraintViolation {
                constraint_name: self.name.clone(),
                violation_amount: val - self.upper,
                details: format!("{} = {:.6} > upper {:.6}", self.name, val, self.upper),
            })
        } else {
            None
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTRAINED OPTIMIZATION WRAPPER
// ═══════════════════════════════════════════════════════════════════════════

/// Apply constraints to weights via iterative projection.
pub fn project_constraints(
    weights: &mut Vec<f64>,
    box_constraint: Option<&BoxConstraint>,
    group_constraints: Option<&GroupConstraintSet>,
    turnover: Option<(&TurnoverConstraint, &[f64])>,
    cardinality: Option<&CardinalityConstraint>,
    max_iter: usize,
) {
    for _ in 0..max_iter {
        let mut changed = false;
        let old = weights.clone();

        // Box constraints
        if let Some(bc) = box_constraint {
            bc.project(weights);
        }

        // Normalize to sum=1
        let sum: f64 = weights.iter().sum();
        if sum.abs() > 1e-15 {
            for w in weights.iter_mut() { *w /= sum; }
        }

        // Group constraints
        if let Some(gc) = group_constraints {
            gc.project(weights, 10);
        }

        // Turnover
        if let Some((tc, old_w)) = turnover {
            tc.project(weights, old_w);
        }

        // Cardinality
        if let Some(cc) = cardinality {
            cc.project(weights);
        }

        // Re-normalize
        let sum: f64 = weights.iter().sum();
        if sum.abs() > 1e-15 && (sum - 1.0).abs() > 1e-10 {
            for w in weights.iter_mut() { *w /= sum; }
        }

        // Check convergence
        let max_diff: f64 = weights.iter().zip(old.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if max_diff < 1e-12 {
            break;
        }
    }
}

/// Penalty function for constraints (for use in penalty-based optimization).
pub fn constraint_penalty(
    weights: &[f64],
    constraints: &[Constraint],
    old_weights: Option<&[f64]>,
    penalty_coeff: f64,
) -> f64 {
    let result = check_constraints(weights, constraints, old_weights);
    result.violations.iter()
        .map(|v| penalty_coeff * v.violation_amount * v.violation_amount)
        .sum()
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_constraint() {
        let bc = BoxConstraint::long_only_capped(4, 0.4);
        let w = vec![0.3, 0.3, 0.3, 0.1];
        assert!(bc.check(&w).is_empty());
        let w2 = vec![0.5, 0.3, 0.1, 0.1];
        assert!(!bc.check(&w2).is_empty());
    }

    #[test]
    fn test_group_constraint() {
        let gc = GroupConstraint::sector("Tech", vec![0, 1], 0.0, 0.5);
        let w = vec![0.3, 0.3, 0.2, 0.2];
        assert!(gc.check(&w).is_some()); // 0.6 > 0.5
        let w2 = vec![0.2, 0.2, 0.3, 0.3];
        assert!(gc.check(&w2).is_none()); // 0.4 <= 0.5
    }

    #[test]
    fn test_turnover_constraint() {
        let tc = TurnoverConstraint::new(0.2);
        let old = vec![0.25, 0.25, 0.25, 0.25];
        let new = vec![0.35, 0.15, 0.25, 0.25];
        assert!(tc.check(&new, &old).is_none()); // turnover = 0.2
        let new2 = vec![0.50, 0.00, 0.25, 0.25];
        assert!(tc.check(&new2, &old).is_some()); // turnover = 0.5
    }

    #[test]
    fn test_cardinality() {
        let cc = CardinalityConstraint::new(2, 0.1, 0.8);
        let w = vec![0.5, 0.5, 0.0, 0.0];
        assert!(cc.check(&w).is_none());
        let w2 = vec![0.4, 0.3, 0.2, 0.1];
        assert!(cc.check(&w2).is_some()); // 4 positions > 2
    }

    #[test]
    fn test_constraint_check() {
        let constraints = vec![
            Constraint::BoxConstraint(BoxConstraint::long_only(4)),
            Constraint::LeverageConstraint(LeverageConstraint::long_only()),
        ];
        let w = vec![0.3, 0.3, 0.3, 0.1];
        let result = check_constraints(&w, &constraints, None);
        assert!(result.feasible);
    }
}
