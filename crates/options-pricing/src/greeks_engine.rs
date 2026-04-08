use crate::black_scholes::{
    BSParams, OptionType, FullGreeks, full_greeks, bs_price, delta, gamma, vega, theta, rho,
    vanna, volga, charm, speed, color, zomma, div_rho, dvanna_dvol,
};

// ═══════════════════════════════════════════════════════════════════════════
// PORTFOLIO POSITION
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct OptionPosition {
    pub id: usize,
    pub underlying_id: usize,
    pub spot: f64,
    pub strike: f64,
    pub rate: f64,
    pub dividend: f64,
    pub vol: f64,
    pub time_to_expiry: f64,
    pub opt_type: OptionType,
    pub quantity: f64,
    pub market_price: Option<f64>,
}

impl OptionPosition {
    pub fn params(&self) -> BSParams {
        BSParams::new(self.spot, self.strike, self.rate, self.dividend, self.vol, self.time_to_expiry)
    }

    pub fn price(&self) -> f64 {
        bs_price(&self.params(), self.opt_type) * self.quantity
    }

    pub fn greeks(&self) -> FullGreeks {
        full_greeks(&self.params(), self.opt_type).scale(self.quantity)
    }

    pub fn intrinsic(&self) -> f64 {
        let phi = match self.opt_type { OptionType::Call => 1.0, OptionType::Put => -1.0 };
        (phi * (self.spot - self.strike)).max(0.0) * self.quantity
    }

    pub fn time_value(&self) -> f64 {
        self.price() - self.intrinsic()
    }

    pub fn moneyness(&self) -> f64 {
        self.spot / self.strike
    }

    pub fn log_moneyness(&self) -> f64 {
        (self.spot / self.strike).ln()
    }

    pub fn notional(&self) -> f64 {
        self.spot * self.quantity.abs()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PORTFOLIO-LEVEL GREEKS AGGREGATION
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PortfolioGreeks {
    pub total_delta: f64,
    pub total_gamma: f64,
    pub total_vega: f64,
    pub total_theta: f64,
    pub total_rho: f64,
    pub total_vanna: f64,
    pub total_volga: f64,
    pub total_charm: f64,
    pub total_speed: f64,
    pub total_color: f64,
    pub total_zomma: f64,
    pub total_value: f64,
    pub total_notional: f64,
    pub dollar_delta: f64,
    pub dollar_gamma: f64,
    pub dollar_vega: f64,
    pub dollar_theta: f64,
    pub pct_delta: f64,
    pub gamma_pct: f64,
    pub vega_pct: f64,
}

/// Aggregate Greeks across a portfolio of options.
pub fn aggregate_greeks(positions: &[OptionPosition]) -> PortfolioGreeks {
    let mut pg = PortfolioGreeks {
        total_delta: 0.0, total_gamma: 0.0, total_vega: 0.0,
        total_theta: 0.0, total_rho: 0.0, total_vanna: 0.0,
        total_volga: 0.0, total_charm: 0.0, total_speed: 0.0,
        total_color: 0.0, total_zomma: 0.0, total_value: 0.0,
        total_notional: 0.0, dollar_delta: 0.0, dollar_gamma: 0.0,
        dollar_vega: 0.0, dollar_theta: 0.0, pct_delta: 0.0,
        gamma_pct: 0.0, vega_pct: 0.0,
    };

    for pos in positions {
        let g = pos.greeks();
        let p = pos.params();

        pg.total_delta += g.delta;
        pg.total_gamma += g.gamma;
        pg.total_vega += g.vega;
        pg.total_theta += g.theta;
        pg.total_rho += g.rho;
        pg.total_vanna += g.vanna;
        pg.total_volga += g.volga;
        pg.total_charm += g.charm;
        pg.total_speed += g.speed;
        pg.total_color += g.color;
        pg.total_zomma += g.zomma;
        pg.total_value += g.price;
        pg.total_notional += pos.notional();

        // Dollar Greeks
        pg.dollar_delta += g.delta * pos.spot;
        pg.dollar_gamma += 0.5 * g.gamma * pos.spot * pos.spot * 0.01; // per 1% move
        pg.dollar_vega += g.vega * 0.01; // per 1 vol point
        pg.dollar_theta += g.theta / 365.0; // per day
    }

    if pg.total_notional > 0.0 {
        pg.pct_delta = pg.dollar_delta / pg.total_notional;
        pg.gamma_pct = pg.dollar_gamma / pg.total_notional;
        pg.vega_pct = pg.dollar_vega / pg.total_notional;
    }

    pg
}

/// Per-underlying Greeks aggregation.
pub fn aggregate_by_underlying(positions: &[OptionPosition]) -> Vec<(usize, PortfolioGreeks)> {
    let mut underlying_ids: Vec<usize> = positions.iter().map(|p| p.underlying_id).collect();
    underlying_ids.sort();
    underlying_ids.dedup();

    underlying_ids.iter().map(|&uid| {
        let sub: Vec<OptionPosition> = positions.iter()
            .filter(|p| p.underlying_id == uid)
            .cloned()
            .collect();
        (uid, aggregate_greeks(&sub))
    }).collect()
}

/// Per-expiry Greeks bucketing.
pub fn aggregate_by_expiry(positions: &[OptionPosition], buckets: &[f64]) -> Vec<(f64, PortfolioGreeks)> {
    buckets.iter().map(|&bucket_end| {
        let sub: Vec<OptionPosition> = positions.iter()
            .filter(|p| p.time_to_expiry <= bucket_end)
            .cloned()
            .collect();
        (bucket_end, aggregate_greeks(&sub))
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// SCENARIO GREEKS (BUMP AND REPRICE)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub spot_bump: f64,
    pub vol_bump: f64,
    pub rate_bump: f64,
    pub time_bump: f64,
    pub pnl: f64,
    pub new_value: f64,
}

/// Compute P&L for a single scenario (spot bump, vol bump, etc).
pub fn scenario_pnl(
    positions: &[OptionPosition],
    spot_bump_pct: f64,
    vol_bump_abs: f64,
    rate_bump_abs: f64,
    time_bump_days: f64,
) -> ScenarioResult {
    let base_value: f64 = positions.iter().map(|p| p.price()).sum();

    let new_value: f64 = positions.iter().map(|pos| {
        let new_spot = pos.spot * (1.0 + spot_bump_pct);
        let new_vol = (pos.vol + vol_bump_abs).max(0.001);
        let new_rate = pos.rate + rate_bump_abs;
        let new_tte = (pos.time_to_expiry - time_bump_days / 365.0).max(0.0);
        let p = BSParams::new(new_spot, pos.strike, new_rate, pos.dividend, new_vol, new_tte);
        bs_price(&p, pos.opt_type) * pos.quantity
    }).sum();

    ScenarioResult {
        spot_bump: spot_bump_pct,
        vol_bump: vol_bump_abs,
        rate_bump: rate_bump_abs,
        time_bump: time_bump_days,
        pnl: new_value - base_value,
        new_value,
    }
}

/// Spot-vol scenario grid.
pub fn spot_vol_scenario_grid(
    positions: &[OptionPosition],
    spot_bumps: &[f64],
    vol_bumps: &[f64],
) -> Vec<Vec<ScenarioResult>> {
    spot_bumps.iter().map(|&ds| {
        vol_bumps.iter().map(|&dv| {
            scenario_pnl(positions, ds, dv, 0.0, 0.0)
        }).collect()
    }).collect()
}

/// Spot ladder: P&L across spot moves.
pub fn spot_ladder(
    positions: &[OptionPosition],
    spot_bumps: &[f64],
) -> Vec<ScenarioResult> {
    spot_bumps.iter().map(|&ds| {
        scenario_pnl(positions, ds, 0.0, 0.0, 0.0)
    }).collect()
}

/// Vol ladder: P&L across vol moves.
pub fn vol_ladder(
    positions: &[OptionPosition],
    vol_bumps: &[f64],
) -> Vec<ScenarioResult> {
    vol_bumps.iter().map(|&dv| {
        scenario_pnl(positions, 0.0, dv, 0.0, 0.0)
    }).collect()
}

/// Time decay ladder: P&L over time.
pub fn time_ladder(
    positions: &[OptionPosition],
    days_forward: &[f64],
) -> Vec<ScenarioResult> {
    days_forward.iter().map(|&dt| {
        scenario_pnl(positions, 0.0, 0.0, 0.0, dt)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// P&L EXPLAIN (ATTRIBUTION)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PnlAttribution {
    pub delta_pnl: f64,
    pub gamma_pnl: f64,
    pub vega_pnl: f64,
    pub theta_pnl: f64,
    pub rho_pnl: f64,
    pub vanna_pnl: f64,
    pub volga_pnl: f64,
    pub charm_pnl: f64,
    pub higher_order_pnl: f64,
    pub cross_gamma_pnl: f64,
    pub total_explained: f64,
    pub actual_pnl: f64,
    pub unexplained: f64,
}

/// Full P&L attribution for portfolio.
pub fn pnl_attribution(
    positions: &[OptionPosition],
    ds_pct: f64,   // spot return
    dvol: f64,     // vol change (absolute)
    dt: f64,       // time elapsed (years)
    dr: f64,       // rate change
) -> PnlAttribution {
    let mut attr = PnlAttribution {
        delta_pnl: 0.0, gamma_pnl: 0.0, vega_pnl: 0.0,
        theta_pnl: 0.0, rho_pnl: 0.0, vanna_pnl: 0.0,
        volga_pnl: 0.0, charm_pnl: 0.0, higher_order_pnl: 0.0,
        cross_gamma_pnl: 0.0, total_explained: 0.0,
        actual_pnl: 0.0, unexplained: 0.0,
    };

    for pos in positions {
        let g = pos.greeks();
        let ds = pos.spot * ds_pct;

        attr.delta_pnl += g.delta * ds;
        attr.gamma_pnl += 0.5 * g.gamma * ds * ds;
        attr.vega_pnl += g.vega * dvol;
        attr.theta_pnl += g.theta * dt;
        attr.rho_pnl += g.rho * dr;
        attr.vanna_pnl += g.vanna * ds * dvol;
        attr.volga_pnl += 0.5 * g.volga * dvol * dvol;
        attr.charm_pnl += g.charm * ds * dt;
        attr.higher_order_pnl += g.speed * ds * ds * ds / 6.0
            + g.color * ds * ds * dt / 2.0
            + g.zomma * ds * ds * dvol / 2.0;
    }

    // Compute actual P&L via full reprice
    let base: f64 = positions.iter().map(|p| p.price()).sum();
    let new_val: f64 = positions.iter().map(|pos| {
        let new_spot = pos.spot * (1.0 + ds_pct);
        let new_vol = (pos.vol + dvol).max(0.001);
        let new_tte = (pos.time_to_expiry - dt).max(0.0);
        let new_rate = pos.rate + dr;
        let p = BSParams::new(new_spot, pos.strike, new_rate, pos.dividend, new_vol, new_tte);
        bs_price(&p, pos.opt_type) * pos.quantity
    }).sum();

    attr.actual_pnl = new_val - base;
    attr.total_explained = attr.delta_pnl + attr.gamma_pnl + attr.vega_pnl
        + attr.theta_pnl + attr.rho_pnl + attr.vanna_pnl + attr.volga_pnl
        + attr.charm_pnl + attr.higher_order_pnl;
    attr.unexplained = attr.actual_pnl - attr.total_explained;

    attr
}

/// P&L explain per position.
pub fn pnl_attribution_per_position(
    positions: &[OptionPosition],
    ds_pct: f64, dvol: f64, dt: f64, dr: f64,
) -> Vec<(usize, PnlAttribution)> {
    positions.iter().map(|pos| {
        let single = vec![pos.clone()];
        (pos.id, pnl_attribution(&single, ds_pct, dvol, dt, dr))
    }).collect()
}

/// Daily P&L explain (1-day theta + realized moves).
pub fn daily_pnl_explain(
    positions: &[OptionPosition],
    new_spots: &[f64],       // new spot prices per underlying
    new_vols: &[f64],        // new implied vols per position
) -> PnlAttribution {
    assert_eq!(positions.len(), new_vols.len());

    let mut total_attr = PnlAttribution {
        delta_pnl: 0.0, gamma_pnl: 0.0, vega_pnl: 0.0,
        theta_pnl: 0.0, rho_pnl: 0.0, vanna_pnl: 0.0,
        volga_pnl: 0.0, charm_pnl: 0.0, higher_order_pnl: 0.0,
        cross_gamma_pnl: 0.0, total_explained: 0.0,
        actual_pnl: 0.0, unexplained: 0.0,
    };

    let dt = 1.0 / 365.0;
    let mut base_total = 0.0;
    let mut new_total = 0.0;

    for (idx, pos) in positions.iter().enumerate() {
        let g = pos.greeks();
        let new_spot = if pos.underlying_id < new_spots.len() {
            new_spots[pos.underlying_id]
        } else {
            pos.spot
        };
        let ds = new_spot - pos.spot;
        let dvol = new_vols[idx] - pos.vol;

        total_attr.delta_pnl += g.delta * ds;
        total_attr.gamma_pnl += 0.5 * g.gamma * ds * ds;
        total_attr.vega_pnl += g.vega * dvol;
        total_attr.theta_pnl += g.theta * dt;
        total_attr.vanna_pnl += g.vanna * ds * dvol;
        total_attr.volga_pnl += 0.5 * g.volga * dvol * dvol;

        base_total += pos.price();
        let new_tte = (pos.time_to_expiry - dt).max(0.0);
        let p = BSParams::new(new_spot, pos.strike, pos.rate, pos.dividend, new_vols[idx], new_tte);
        new_total += bs_price(&p, pos.opt_type) * pos.quantity;
    }

    total_attr.actual_pnl = new_total - base_total;
    total_attr.total_explained = total_attr.delta_pnl + total_attr.gamma_pnl
        + total_attr.vega_pnl + total_attr.theta_pnl + total_attr.vanna_pnl
        + total_attr.volga_pnl;
    total_attr.unexplained = total_attr.actual_pnl - total_attr.total_explained;

    total_attr
}

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-GAMMA MATRIX
// ═══════════════════════════════════════════════════════════════════════════

/// Compute cross-gamma matrix for multi-underlying portfolio.
/// cross_gamma[i][j] = d²V / (dS_i dS_j)
pub fn cross_gamma_matrix(
    positions: &[OptionPosition],
    underlying_spots: &[f64],
    n_underlyings: usize,
    ds_pct: f64,
) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n_underlyings]; n_underlyings];

    // Diagonal: sum of gammas for each underlying
    for i in 0..n_underlyings {
        let ds = underlying_spots[i] * ds_pct;
        let mut gamma_sum = 0.0;

        for pos in positions {
            if pos.underlying_id == i {
                let g = gamma(&pos.params());
                gamma_sum += g * pos.quantity;
            }
        }
        matrix[i][i] = gamma_sum;
    }

    // Off-diagonal: cross-gammas via finite difference on multi-asset positions
    // For single-asset options, cross-gammas are zero
    // This would be non-zero for basket/rainbow options

    matrix
}

/// Dollar cross-gamma matrix: Γ_ij * S_i * S_j * 0.01²
pub fn dollar_cross_gamma_matrix(
    positions: &[OptionPosition],
    underlying_spots: &[f64],
    n_underlyings: usize,
) -> Vec<Vec<f64>> {
    let gamma_mat = cross_gamma_matrix(positions, underlying_spots, n_underlyings, 0.01);
    let mut dollar_mat = vec![vec![0.0; n_underlyings]; n_underlyings];

    for i in 0..n_underlyings {
        for j in 0..n_underlyings {
            dollar_mat[i][j] = gamma_mat[i][j]
                * underlying_spots[i] * underlying_spots[j]
                * 0.01 * 0.01;
        }
    }
    dollar_mat
}

// ═══════════════════════════════════════════════════════════════════════════
// GREEKS BUCKETING (TERM STRUCTURE)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct GreeksBucket {
    pub label: String,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub vanna: f64,
    pub volga: f64,
    pub count: usize,
    pub notional: f64,
}

impl GreeksBucket {
    pub fn new(label: String) -> Self {
        Self {
            label, delta: 0.0, gamma: 0.0, vega: 0.0, theta: 0.0,
            vanna: 0.0, volga: 0.0, count: 0, notional: 0.0,
        }
    }

    pub fn add_position(&mut self, pos: &OptionPosition) {
        let g = pos.greeks();
        self.delta += g.delta;
        self.gamma += g.gamma;
        self.vega += g.vega;
        self.theta += g.theta;
        self.vanna += g.vanna;
        self.volga += g.volga;
        self.count += 1;
        self.notional += pos.notional();
    }
}

/// Bucket Greeks by expiry tenor.
pub fn bucket_by_expiry(positions: &[OptionPosition]) -> Vec<GreeksBucket> {
    let tenors = [
        (1.0 / 52.0, "1W"),
        (1.0 / 12.0, "1M"),
        (3.0 / 12.0, "3M"),
        (6.0 / 12.0, "6M"),
        (1.0, "1Y"),
        (2.0, "2Y"),
        (5.0, "5Y"),
        (f64::INFINITY, "5Y+"),
    ];

    let mut buckets: Vec<GreeksBucket> = tenors.iter()
        .map(|(_, label)| GreeksBucket::new(label.to_string()))
        .collect();

    for pos in positions {
        let mut prev_cutoff = 0.0;
        for (i, &(cutoff, _)) in tenors.iter().enumerate() {
            if pos.time_to_expiry > prev_cutoff && pos.time_to_expiry <= cutoff {
                buckets[i].add_position(pos);
                break;
            }
            prev_cutoff = cutoff;
        }
    }

    buckets
}

/// Bucket Greeks by moneyness.
pub fn bucket_by_moneyness(positions: &[OptionPosition]) -> Vec<GreeksBucket> {
    let ranges = [
        (0.0, 0.8, "Deep OTM"),
        (0.8, 0.95, "OTM"),
        (0.95, 1.05, "ATM"),
        (1.05, 1.2, "ITM"),
        (1.2, f64::INFINITY, "Deep ITM"),
    ];

    let mut buckets: Vec<GreeksBucket> = ranges.iter()
        .map(|(_, _, label)| GreeksBucket::new(label.to_string()))
        .collect();

    for pos in positions {
        let m = pos.moneyness();
        for (i, &(lo, hi, _)) in ranges.iter().enumerate() {
            if m >= lo && m < hi {
                buckets[i].add_position(pos);
                break;
            }
        }
    }

    buckets
}

// ═══════════════════════════════════════════════════════════════════════════
// RISK METRICS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute delta-equivalent position (in shares of underlying).
pub fn delta_equivalent(positions: &[OptionPosition]) -> f64 {
    positions.iter().map(|p| {
        let g = p.greeks();
        g.delta
    }).sum()
}

/// Compute gamma-equivalent (gamma * S² / 100).
pub fn gamma_equivalent(positions: &[OptionPosition]) -> f64 {
    positions.iter().map(|p| {
        let g = p.greeks();
        g.gamma * p.spot * p.spot * 0.01
    }).sum()
}

/// Compute vega-equivalent (vega per 1 vol point).
pub fn vega_equivalent(positions: &[OptionPosition]) -> f64 {
    positions.iter().map(|p| {
        let g = p.greeks();
        g.vega * 0.01
    }).sum()
}

/// Compute theta-per-day.
pub fn theta_per_day(positions: &[OptionPosition]) -> f64 {
    positions.iter().map(|p| {
        let g = p.greeks();
        g.theta / 365.0
    }).sum()
}

/// Delta-neutral hedge ratio: how many shares to sell to be delta-flat.
pub fn delta_hedge_shares(positions: &[OptionPosition]) -> f64 {
    -delta_equivalent(positions)
}

/// Gamma scalp P&L estimate: expected gamma P&L from realized vol.
pub fn gamma_scalp_pnl(
    positions: &[OptionPosition],
    realized_move_pct: f64,
) -> f64 {
    let gamma_eq = gamma_equivalent(positions);
    gamma_eq * realized_move_pct * realized_move_pct * 100.0 / 2.0
}

/// Vega P&L from vol move.
pub fn vega_pnl(positions: &[OptionPosition], vol_move: f64) -> f64 {
    vega_equivalent(positions) * vol_move * 100.0
}

/// Break-even realized vol: the vol at which gamma P&L offsets theta.
pub fn breakeven_vol(positions: &[OptionPosition]) -> f64 {
    let theta_day = theta_per_day(positions);
    let gamma_eq = gamma_equivalent(positions);
    if gamma_eq.abs() < 1e-15 {
        return 0.0;
    }
    // theta_day + 0.5 * gamma_eq * S² * σ²_daily = 0
    // σ_daily = sqrt(-2 * theta / (gamma * S²))
    let total_gamma_s2: f64 = positions.iter().map(|p| {
        let g = gamma(&p.params());
        g * p.quantity * p.spot * p.spot
    }).sum();

    if total_gamma_s2.abs() < 1e-15 {
        return 0.0;
    }

    let var_daily = (-2.0 * theta_day / total_gamma_s2).max(0.0);
    var_daily.sqrt() * 252.0_f64.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// STRESS TESTING
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct StressScenario {
    pub name: String,
    pub spot_shock: f64,     // as percentage
    pub vol_shock: f64,      // absolute
    pub rate_shock: f64,     // absolute
    pub time_decay_days: f64,
}

#[derive(Debug, Clone)]
pub struct StressResult {
    pub scenario: String,
    pub base_value: f64,
    pub stressed_value: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
}

/// Run stress tests across multiple scenarios.
pub fn stress_test(
    positions: &[OptionPosition],
    scenarios: &[StressScenario],
) -> Vec<StressResult> {
    let base_value: f64 = positions.iter().map(|p| p.price()).sum();

    scenarios.iter().map(|scenario| {
        let result = scenario_pnl(
            positions,
            scenario.spot_shock,
            scenario.vol_shock,
            scenario.rate_shock,
            scenario.time_decay_days,
        );

        StressResult {
            scenario: scenario.name.clone(),
            base_value,
            stressed_value: result.new_value,
            pnl: result.pnl,
            pnl_pct: if base_value.abs() > 1e-10 { result.pnl / base_value } else { 0.0 },
        }
    }).collect()
}

/// Standard stress scenarios for options portfolio.
pub fn standard_stress_scenarios() -> Vec<StressScenario> {
    vec![
        StressScenario { name: "Spot +10%".into(), spot_shock: 0.10, vol_shock: 0.0, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Spot -10%".into(), spot_shock: -0.10, vol_shock: 0.0, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Spot +20%".into(), spot_shock: 0.20, vol_shock: 0.0, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Spot -20%".into(), spot_shock: -0.20, vol_shock: 0.0, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Vol +5%".into(), spot_shock: 0.0, vol_shock: 0.05, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Vol -5%".into(), spot_shock: 0.0, vol_shock: -0.05, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Vol +10%".into(), spot_shock: 0.0, vol_shock: 0.10, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Crash: -25% spot, +15% vol".into(), spot_shock: -0.25, vol_shock: 0.15, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Rally: +15% spot, -5% vol".into(), spot_shock: 0.15, vol_shock: -0.05, rate_shock: 0.0, time_decay_days: 0.0 },
        StressScenario { name: "Rate +100bp".into(), spot_shock: 0.0, vol_shock: 0.0, rate_shock: 0.01, time_decay_days: 0.0 },
        StressScenario { name: "Rate -100bp".into(), spot_shock: 0.0, vol_shock: 0.0, rate_shock: -0.01, time_decay_days: 0.0 },
        StressScenario { name: "1 week decay".into(), spot_shock: 0.0, vol_shock: 0.0, rate_shock: 0.0, time_decay_days: 7.0 },
        StressScenario { name: "1 month decay".into(), spot_shock: 0.0, vol_shock: 0.0, rate_shock: 0.0, time_decay_days: 30.0 },
        StressScenario { name: "Black Monday: -22% spot, +30% vol".into(), spot_shock: -0.22, vol_shock: 0.30, rate_shock: -0.005, time_decay_days: 1.0 },
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// HEDGING ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute optimal hedge ratios to minimize portfolio risk.
pub fn delta_gamma_hedge(
    portfolio: &[OptionPosition],
    hedge_instruments: &[OptionPosition],
) -> Vec<f64> {
    let n = hedge_instruments.len();
    if n == 0 {
        return vec![];
    }

    let port_greeks = aggregate_greeks(portfolio);

    if n == 1 {
        // Delta hedge only
        let h_delta = delta(&hedge_instruments[0].params(), hedge_instruments[0].opt_type);
        if h_delta.abs() < 1e-15 {
            return vec![0.0];
        }
        return vec![-port_greeks.total_delta / h_delta];
    }

    // For 2+ instruments, solve delta-gamma system
    let mut a = vec![vec![0.0; n]; n.min(2)];
    let mut b = vec![0.0; n.min(2)];

    // Delta equation
    b[0] = -port_greeks.total_delta;
    for j in 0..n {
        a[0][j] = delta(&hedge_instruments[j].params(), hedge_instruments[j].opt_type);
    }

    if n >= 2 {
        // Gamma equation
        b[1] = -port_greeks.total_gamma;
        for j in 0..n {
            a[1][j] = gamma(&hedge_instruments[j].params());
        }
    }

    // Simple least-squares for overdetermined system
    if n == 2 {
        // Direct solve 2x2
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        if det.abs() < 1e-15 {
            return vec![0.0; n];
        }
        let x0 = (b[0] * a[1][1] - b[1] * a[0][1]) / det;
        let x1 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
        return vec![x0, x1];
    }

    // For n > 2, use delta hedge on first instrument
    let h_delta = a[0][0];
    if h_delta.abs() < 1e-15 {
        return vec![0.0; n];
    }
    let mut result = vec![0.0; n];
    result[0] = b[0] / h_delta;
    result
}

/// Delta-vega hedge using two instruments.
pub fn delta_vega_hedge(
    portfolio: &[OptionPosition],
    hedge_option: &OptionPosition,  // option for vega hedge
    hedge_spot: f64,                 // spot for delta hedge (delta=1)
) -> (f64, f64) {
    let port = aggregate_greeks(portfolio);
    let h_vega = vega(&hedge_option.params());
    let h_delta = delta(&hedge_option.params(), hedge_option.opt_type);

    if h_vega.abs() < 1e-15 {
        return (0.0, -port.total_delta);
    }

    let n_options = -port.total_vega / h_vega;
    let residual_delta = port.total_delta + n_options * h_delta;
    let n_shares = -residual_delta;

    (n_options, n_shares)
}

// ═══════════════════════════════════════════════════════════════════════════
// PORTFOLIO RISK REPORT
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RiskReport {
    pub n_positions: usize,
    pub total_value: f64,
    pub total_notional: f64,
    pub net_delta: f64,
    pub net_gamma: f64,
    pub net_vega: f64,
    pub net_theta_daily: f64,
    pub dollar_delta: f64,
    pub dollar_gamma_1pct: f64,
    pub dollar_vega_1pt: f64,
    pub breakeven_vol: f64,
    pub max_loss_10pct_down: f64,
    pub max_loss_crash: f64,
    pub gamma_theta_ratio: f64,
    pub vega_theta_ratio: f64,
}

/// Generate comprehensive risk report for options portfolio.
pub fn risk_report(positions: &[OptionPosition]) -> RiskReport {
    let pg = aggregate_greeks(positions);
    let be_vol = breakeven_vol(positions);

    let loss_10_down = scenario_pnl(positions, -0.10, 0.0, 0.0, 0.0).pnl;
    let loss_crash = scenario_pnl(positions, -0.25, 0.15, 0.0, 0.0).pnl;

    let theta_daily = pg.dollar_theta;
    let g_t_ratio = if theta_daily.abs() > 1e-10 {
        pg.dollar_gamma / theta_daily
    } else {
        0.0
    };
    let v_t_ratio = if theta_daily.abs() > 1e-10 {
        pg.dollar_vega / theta_daily
    } else {
        0.0
    };

    RiskReport {
        n_positions: positions.len(),
        total_value: pg.total_value,
        total_notional: pg.total_notional,
        net_delta: pg.total_delta,
        net_gamma: pg.total_gamma,
        net_vega: pg.total_vega,
        net_theta_daily: theta_daily,
        dollar_delta: pg.dollar_delta,
        dollar_gamma_1pct: pg.dollar_gamma,
        dollar_vega_1pt: pg.dollar_vega,
        breakeven_vol: be_vol,
        max_loss_10pct_down: loss_10_down,
        max_loss_crash: loss_crash,
        gamma_theta_ratio: g_t_ratio,
        vega_theta_ratio: v_t_ratio,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_positions() -> Vec<OptionPosition> {
        vec![
            OptionPosition {
                id: 0, underlying_id: 0, spot: 100.0, strike: 100.0,
                rate: 0.05, dividend: 0.02, vol: 0.20, time_to_expiry: 0.5,
                opt_type: OptionType::Call, quantity: 100.0, market_price: None,
            },
            OptionPosition {
                id: 1, underlying_id: 0, spot: 100.0, strike: 95.0,
                rate: 0.05, dividend: 0.02, vol: 0.22, time_to_expiry: 0.25,
                opt_type: OptionType::Put, quantity: -50.0, market_price: None,
            },
            OptionPosition {
                id: 2, underlying_id: 0, spot: 100.0, strike: 105.0,
                rate: 0.05, dividend: 0.02, vol: 0.18, time_to_expiry: 1.0,
                opt_type: OptionType::Call, quantity: 75.0, market_price: None,
            },
        ]
    }

    #[test]
    fn test_aggregate_greeks() {
        let positions = sample_positions();
        let pg = aggregate_greeks(&positions);
        // Should have some non-zero values
        assert!(pg.total_value != 0.0);
        assert!(pg.total_delta != 0.0);
    }

    #[test]
    fn test_pnl_attribution_sums() {
        let positions = sample_positions();
        let attr = pnl_attribution(&positions, 0.05, 0.02, 1.0/365.0, 0.0);
        // Explained should be close to actual
        let err = (attr.unexplained / attr.actual_pnl.abs().max(0.01)).abs();
        assert!(err < 0.5, "Unexplained too large: {} vs {}", attr.unexplained, attr.actual_pnl);
    }

    #[test]
    fn test_scenario_grid() {
        let positions = sample_positions();
        let spot_bumps = vec![-0.10, -0.05, 0.0, 0.05, 0.10];
        let vol_bumps = vec![-0.05, 0.0, 0.05];
        let grid = spot_vol_scenario_grid(&positions, &spot_bumps, &vol_bumps);
        assert_eq!(grid.len(), 5);
        assert_eq!(grid[0].len(), 3);
        // Center should be ~0
        assert!(grid[2][1].pnl.abs() < 1e-8);
    }

    #[test]
    fn test_stress_test() {
        let positions = sample_positions();
        let scenarios = standard_stress_scenarios();
        let results = stress_test(&positions, &scenarios);
        assert_eq!(results.len(), scenarios.len());
    }

    #[test]
    fn test_risk_report() {
        let positions = sample_positions();
        let report = risk_report(&positions);
        assert!(report.n_positions == 3);
        assert!(report.total_notional > 0.0);
    }

    #[test]
    fn test_bucket_by_expiry() {
        let positions = sample_positions();
        let buckets = bucket_by_expiry(&positions);
        let total_count: usize = buckets.iter().map(|b| b.count).sum();
        assert_eq!(total_count, 3);
    }
}
