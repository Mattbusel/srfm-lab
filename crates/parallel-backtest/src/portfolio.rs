use crate::bh_engine::{BHState, GARCHState, OUState};
use crate::params::StrategyParams;
use std::collections::HashMap;

/// Live portfolio state.
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Total equity in USD (or normalised base currency).
    pub equity: f64,
    /// Current fractional allocation per symbol (sums to ≤ 1.0).
    pub positions: HashMap<String, f64>,
    /// Entry price per symbol (for P&L tracking).
    pub entry_prices: HashMap<String, f64>,
    /// Number of bars each position has been held.
    pub bars_held: HashMap<String, u32>,
    /// Minimum fractional position size (avoid dust).
    pub pos_floor: f64,
    /// Last seen 15-min price per symbol (for stale-price detection).
    pub last_15m_px: HashMap<String, f64>,
    /// Peak unrealised equity per symbol (for winner protection).
    pub peak_equity: HashMap<String, f64>,
    /// Cumulative equity curve (for drawdown computation).
    pub equity_curve: Vec<f64>,
}

impl Portfolio {
    pub fn new(initial_equity: f64) -> Self {
        Self {
            equity: initial_equity,
            positions: HashMap::new(),
            entry_prices: HashMap::new(),
            bars_held: HashMap::new(),
            pos_floor: 0.001,
            last_15m_px: HashMap::new(),
            peak_equity: HashMap::new(),
            equity_curve: vec![initial_equity],
        }
    }

    /// Compute target fractional allocations for each symbol.
    ///
    /// Decision tree:
    /// 1. Check if BH is active and direction is set.
    /// 2. Check blocked/boost hours.
    /// 3. Scale position size by GARCH inverse-vol relative to target.
    /// 4. Reduce if OU z-score is extreme (mean reversion warning).
    /// 5. Apply dynamic correlation penalty to avoid over-concentration.
    pub fn compute_targets(
        &self,
        bh_states: &HashMap<String, BHState>,
        garch_states: &HashMap<String, GARCHState>,
        ou_states: &HashMap<String, OUState>,
        params: &StrategyParams,
        curr_prices: &HashMap<String, f64>,
        hour_utc: u8,
        daily_returns: &HashMap<String, Vec<f64>>,
    ) -> HashMap<String, f64> {
        // Skip trading entirely during blocked hours.
        if params.blocked_hours.contains(&hour_utc) {
            // Return current positions (hold) or zero if this is a new signal check.
            return HashMap::new();
        }

        let hour_boost = if params.boost_hours.contains(&hour_utc) {
            params.hour_boost_multiplier
        } else {
            1.0
        };

        let n = bh_states.len() as f64;
        if n == 0.0 {
            return HashMap::new();
        }

        // Compute dynamic correlation for portfolio-wide sizing.
        let corr = self.dynamic_corr(daily_returns);
        // Determine regime.
        let btc_ret = daily_returns
            .get("BTC")
            .and_then(|r| r.last().copied())
            .unwrap_or(0.0)
            .abs();
        let regime_corr = if btc_ret > params.corr_stress_threshold {
            params.corr_stress
        } else {
            params.corr_normal
        };
        // Use the larger of realised and regime-assumption correlation.
        let effective_corr = corr.max(regime_corr);

        // Kelly-inspired diversification: max alloc per instrument shrinks as corr rises.
        // With N assets at correlation rho, the diversified fraction ~ 1/(N + rho*(N^2-N))
        // approximated as 1/(1 + (n-1)*rho) per instrument.
        let base_alloc = 1.0 / (1.0 + (n - 1.0) * effective_corr);

        let mut targets: HashMap<String, f64> = HashMap::new();
        let mut total_alloc = 0.0f64;

        for (sym, bh) in bh_states {
            if !bh.active || bh.bh_dir == 0 {
                continue;
            }

            // GARCH vol scaling.
            let vol_scale = garch_states
                .get(sym)
                .map(|g| g.inv_vol_scale(params.garch_target_vol))
                .unwrap_or(1.0);

            // OU z-score penalty: shrink position if price is very extended.
            let ou_penalty = if params.ou_disabled_syms.contains(sym) {
                1.0
            } else if let Some(ou) = ou_states.get(sym) {
                let price = curr_prices.get(sym).copied().unwrap_or(1.0);
                match ou.z_score(price) {
                    Some(z) if z.abs() > 2.0 => 0.5, // extended, reduce size
                    Some(z) if z.abs() > 3.0 => 0.25,
                    _ => 1.0,
                }
            } else {
                1.0
            };

            let alloc = (base_alloc * vol_scale * ou_penalty * hour_boost).clamp(0.0, 0.30);
            if alloc >= self.pos_floor {
                total_alloc += alloc;
                targets.insert(sym.clone(), alloc * bh.bh_dir as f64);
            }
        }

        // Normalise if total exceeds 1.0 (no leverage).
        if total_alloc > 1.0 {
            for v in targets.values_mut() {
                *v /= total_alloc;
            }
        }

        targets
    }

    /// Apply target allocations: compute dollar trades, update positions, settle P&L.
    pub fn apply_targets(
        &mut self,
        targets: &HashMap<String, f64>,
        curr_prices: &HashMap<String, f64>,
        transaction_cost: f64,
    ) {
        let equity = self.equity;

        // Close positions no longer in targets.
        let to_close: Vec<String> = self
            .positions
            .keys()
            .filter(|s| !targets.contains_key(*s))
            .cloned()
            .collect();

        for sym in to_close {
            if let (Some(&frac), Some(&price)) =
                (self.positions.get(&sym), curr_prices.get(&sym))
            {
                if frac.abs() >= self.pos_floor {
                    let entry = self.entry_prices.get(&sym).copied().unwrap_or(price);
                    let pnl_pct = (price - entry) / entry * frac.signum();
                    self.equity += equity * frac.abs() * pnl_pct;
                    self.equity -= equity * frac.abs() * transaction_cost;
                }
                self.positions.remove(&sym);
                self.entry_prices.remove(&sym);
                self.bars_held.remove(&sym);
                self.peak_equity.remove(&sym);
            }
        }

        // Open/adjust positions.
        for (sym, &target_frac) in targets {
            let price = match curr_prices.get(sym) {
                Some(&p) if p > 0.0 => p,
                _ => continue,
            };

            let current_frac = self.positions.get(sym).copied().unwrap_or(0.0);
            let delta = target_frac - current_frac;

            if delta.abs() >= self.pos_floor {
                // Transaction cost on the traded notional.
                self.equity -= self.equity * delta.abs() * transaction_cost;
            }

            if target_frac.abs() < self.pos_floor {
                // Effectively flat.
                self.positions.remove(sym);
                self.entry_prices.remove(sym);
                self.bars_held.remove(sym);
                self.peak_equity.remove(sym);
            } else {
                self.positions.insert(sym.clone(), target_frac);
                self.entry_prices.entry(sym.clone()).or_insert(price);
                *self.bars_held.entry(sym.clone()).or_insert(0) += 1;
                self.peak_equity
                    .entry(sym.clone())
                    .and_modify(|pk| *pk = pk.max(self.equity * target_frac.abs()))
                    .or_insert(self.equity * target_frac.abs());
            }
        }

        // Mark-to-market unrealised P&L: handled lazily at close.
        self.equity = self.equity.max(0.0);
        self.equity_curve.push(self.equity);
    }

    /// Increment bars_held for open positions and check winner-protection.
    /// Returns list of symbols that should be force-closed.
    pub fn tick_positions(
        &mut self,
        curr_prices: &HashMap<String, f64>,
        params: &StrategyParams,
    ) -> Vec<String> {
        let mut force_close = vec![];
        let equity = self.equity;

        for (sym, &frac) in &self.positions {
            *self.bars_held.entry(sym.clone()).or_insert(0) += 1;

            let price = match curr_prices.get(sym) {
                Some(&p) => p,
                None => continue,
            };
            let entry = self.entry_prices.get(sym).copied().unwrap_or(price);
            let unrealised_pnl_pct = (price - entry) / entry * frac.signum();
            let unrealised_dollar = equity * frac.abs() * unrealised_pnl_pct;

            // Winner protection: if unrealised profit has been, but drawback > threshold, close.
            let peak = self.peak_equity.get(sym).copied().unwrap_or(0.0);
            if unrealised_dollar > 0.0 {
                let new_peak = peak.max(unrealised_dollar);
                self.peak_equity.insert(sym.clone(), new_peak);
            } else if peak > 0.0 {
                let drawback = (peak - unrealised_dollar) / (peak + 1e-10);
                if drawback > params.winner_protection_pct {
                    force_close.push(sym.clone());
                }
            }
        }

        force_close
    }

    /// Rolling 30-day average pairwise Pearson correlation across daily return series.
    pub fn dynamic_corr(&self, daily_returns: &HashMap<String, Vec<f64>>) -> f64 {
        let syms: Vec<&String> = daily_returns.keys().collect();
        let n = syms.len();
        if n < 2 {
            return 0.0;
        }

        let window = 30;
        let mut sum_corr = 0.0;
        let mut count = 0u32;

        for i in 0..n {
            for j in (i + 1)..n {
                let rx = daily_returns[syms[i]].as_slice();
                let ry = daily_returns[syms[j]].as_slice();
                if let Some(c) = pearson_corr_window(rx, ry, window) {
                    sum_corr += c;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            (sum_corr / count as f64).clamp(-1.0, 1.0)
        }
    }
}

/// Pearson correlation over the last `window` elements of two slices.
fn pearson_corr_window(x: &[f64], y: &[f64], window: usize) -> Option<f64> {
    let len = x.len().min(y.len());
    if len < 5 {
        return None;
    }
    let start = len.saturating_sub(window);
    let xs = &x[start..];
    let ys = &y[start..];
    let n = xs.len() as f64;

    let mx: f64 = xs.iter().sum::<f64>() / n;
    let my: f64 = ys.iter().sum::<f64>() / n;

    let num: f64 = xs.iter().zip(ys).map(|(a, b)| (a - mx) * (b - my)).sum();
    let dx: f64 = xs.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = ys.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();

    if dx < 1e-12 || dy < 1e-12 {
        None
    } else {
        Some((num / (dx * dy)).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bh_active(dir: i8) -> BHState {
        let mut bh = crate::bh_engine::BHState::new(0.005);
        bh.active = true;
        bh.bh_dir = dir;
        bh.mass = 0.8;
        bh
    }

    #[test]
    fn test_compute_targets_blocked_hour() {
        let p = Portfolio::new(10_000.0);
        let params = StrategyParams::default(); // blocked_hours: [2,3,4]
        let bh: HashMap<String, BHState> = [("BTC".to_string(), make_bh_active(1))].into();
        let targets = p.compute_targets(&bh, &Default::default(), &Default::default(), &params, &Default::default(), 2, &Default::default());
        assert!(targets.is_empty(), "no trades during blocked hours");
    }

    #[test]
    fn test_compute_targets_active_bh() {
        let p = Portfolio::new(10_000.0);
        let params = StrategyParams::default();
        let bh: HashMap<String, BHState> = [("BTC".to_string(), make_bh_active(1))].into();
        let prices: HashMap<String, f64> = [("BTC".to_string(), 50000.0)].into();
        let daily: HashMap<String, Vec<f64>> = Default::default();
        let targets = p.compute_targets(&bh, &Default::default(), &Default::default(), &params, &prices, 10, &daily);
        assert!(targets.contains_key("BTC"));
        assert!(targets["BTC"] > 0.0);
    }

    #[test]
    fn test_apply_targets_reduces_equity_on_cost() {
        let mut p = Portfolio::new(10_000.0);
        let prices: HashMap<String, f64> = [("BTC".to_string(), 50000.0)].into();
        let targets: HashMap<String, f64> = [("BTC".to_string(), 0.10)].into();
        p.apply_targets(&targets, &prices, 0.001); // 10 bps
        assert!(p.equity < 10_000.0, "equity should drop by transaction cost");
    }

    #[test]
    fn test_dynamic_corr_perfect() {
        let p = Portfolio::new(10_000.0);
        let r: Vec<f64> = (0..40).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let daily: HashMap<String, Vec<f64>> = [
            ("BTC".to_string(), r.clone()),
            ("ETH".to_string(), r.clone()),
        ].into();
        let corr = p.dynamic_corr(&daily);
        assert!((corr - 1.0).abs() < 1e-9, "identical series => corr=1.0, got {}", corr);
    }

    #[test]
    fn test_dynamic_corr_anticorrelated() {
        let p = Portfolio::new(10_000.0);
        let r: Vec<f64> = (0..40).map(|i| (i as f64) * 0.001).collect();
        let neg: Vec<f64> = r.iter().map(|x| -x).collect();
        let daily: HashMap<String, Vec<f64>> = [
            ("BTC".to_string(), r),
            ("ETH".to_string(), neg),
        ].into();
        let corr = p.dynamic_corr(&daily);
        assert!(corr < -0.9, "anti-correlated series, got {}", corr);
    }
}
