/// Risk limits management system.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Severity ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Ok,
    Warning,
    Breach,
    Critical,
}

// ── Limit Types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLimit {
    /// Maximum 1-day 99% Value at Risk (fraction of portfolio value).
    MaxVar { threshold: f64, warning_pct: f64 },
    /// Maximum drawdown from peak (fraction).
    MaxDrawdown { threshold: f64, warning_pct: f64 },
    /// Maximum single-asset concentration (fraction of portfolio).
    MaxConcentration { threshold: f64, warning_pct: f64 },
    /// Maximum gross leverage (total absolute exposure / NAV).
    MaxLeverage { threshold: f64, warning_pct: f64 },
    /// Maximum pairwise correlation between any two positions.
    MaxCorrelation { threshold: f64, warning_pct: f64 },
    /// Maximum net exposure (long - short).
    MaxNetExposure { threshold: f64, warning_pct: f64 },
    /// Minimum diversification ratio.
    MinDiversification { threshold: f64, warning_pct: f64 },
    /// Maximum sector concentration.
    MaxSectorConcentration { threshold: f64, warning_pct: f64, sector: String },
    /// Daily loss limit (stop-loss on daily PnL as fraction of NAV).
    DailyLossLimit { threshold: f64, warning_pct: f64 },
    /// Maximum position count.
    MaxPositions { threshold: usize },
}

impl RiskLimit {
    pub fn name(&self) -> &str {
        match self {
            RiskLimit::MaxVar { .. } => "MaxVar",
            RiskLimit::MaxDrawdown { .. } => "MaxDrawdown",
            RiskLimit::MaxConcentration { .. } => "MaxConcentration",
            RiskLimit::MaxLeverage { .. } => "MaxLeverage",
            RiskLimit::MaxCorrelation { .. } => "MaxCorrelation",
            RiskLimit::MaxNetExposure { .. } => "MaxNetExposure",
            RiskLimit::MinDiversification { .. } => "MinDiversification",
            RiskLimit::MaxSectorConcentration { .. } => "MaxSectorConcentration",
            RiskLimit::DailyLossLimit { .. } => "DailyLossLimit",
            RiskLimit::MaxPositions { .. } => "MaxPositions",
        }
    }
}

// ── Portfolio State ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PortfolioState {
    /// Asset → weight mapping.
    pub weights: HashMap<String, f64>,
    /// Asset → sector mapping (optional).
    pub sectors: HashMap<String, String>,
    /// Current 1-day 99% VaR estimate.
    pub current_var: f64,
    /// Current drawdown from peak.
    pub current_drawdown: f64,
    /// Today's PnL (fraction of NAV).
    pub daily_pnl: f64,
    /// Pairwise correlation matrix (asset × asset).
    pub correlations: HashMap<(String, String), f64>,
    /// Current diversification ratio.
    pub diversification_ratio: f64,
}

impl PortfolioState {
    pub fn gross_leverage(&self) -> f64 {
        self.weights.values().map(|w| w.abs()).sum()
    }

    pub fn net_exposure(&self) -> f64 {
        self.weights.values().sum::<f64>().abs()
    }

    pub fn max_single_concentration(&self) -> f64 {
        self.weights.values().copied().fold(0.0_f64, f64::max)
    }

    pub fn max_sector_concentration(&self, sector: &str) -> f64 {
        self.weights
            .iter()
            .filter(|(asset, _)| self.sectors.get(*asset).map_or(false, |s| s == sector))
            .map(|(_, w)| w.abs())
            .sum()
    }

    pub fn max_pairwise_correlation(&self) -> f64 {
        self.correlations
            .values()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn position_count(&self) -> usize {
        self.weights.values().filter(|&&w| w.abs() > 1e-6).count()
    }
}

// ── Limit Breach ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LimitBreach {
    pub limit_name: String,
    pub current_value: f64,
    pub limit_value: f64,
    pub utilisation: f64, // current / limit
    pub severity: Severity,
    pub message: String,
}

// ── Limit Checker ─────────────────────────────────────────────────────────────

pub struct LimitChecker {
    pub limits: Vec<RiskLimit>,
}

impl LimitChecker {
    pub fn new(limits: Vec<RiskLimit>) -> Self {
        LimitChecker { limits }
    }

    /// Evaluate all limits against the current portfolio state.
    pub fn breach_report(&self, state: &PortfolioState) -> Vec<LimitBreach> {
        let mut breaches = Vec::new();
        for limit in &self.limits {
            if let Some(breach) = self.check_limit(limit, state) {
                breaches.push(breach);
            }
        }
        // Sort by severity descending.
        breaches.sort_by(|a, b| b.severity.cmp(&a.severity));
        breaches
    }

    fn check_limit(&self, limit: &RiskLimit, state: &PortfolioState) -> Option<LimitBreach> {
        match limit {
            RiskLimit::MaxVar { threshold, warning_pct } => {
                let val = state.current_var;
                Self::check_upper(limit.name(), val, *threshold, *warning_pct)
            }
            RiskLimit::MaxDrawdown { threshold, warning_pct } => {
                let val = state.current_drawdown;
                Self::check_upper(limit.name(), val, *threshold, *warning_pct)
            }
            RiskLimit::MaxConcentration { threshold, warning_pct } => {
                let val = state.max_single_concentration();
                Self::check_upper(limit.name(), val, *threshold, *warning_pct)
            }
            RiskLimit::MaxLeverage { threshold, warning_pct } => {
                let val = state.gross_leverage();
                Self::check_upper(limit.name(), val, *threshold, *warning_pct)
            }
            RiskLimit::MaxCorrelation { threshold, warning_pct } => {
                let val = state.max_pairwise_correlation();
                if val == f64::NEG_INFINITY { return None; }
                Self::check_upper(limit.name(), val, *threshold, *warning_pct)
            }
            RiskLimit::MaxNetExposure { threshold, warning_pct } => {
                let val = state.net_exposure();
                Self::check_upper(limit.name(), val, *threshold, *warning_pct)
            }
            RiskLimit::MinDiversification { threshold, warning_pct } => {
                let val = state.diversification_ratio;
                // Lower is worse for diversification.
                if val >= *threshold { return None; }
                let utilisation = threshold / val.max(1e-12);
                let severity = if val < threshold * (1.0 - warning_pct) {
                    Severity::Critical
                } else {
                    Severity::Breach
                };
                Some(LimitBreach {
                    limit_name: limit.name().to_string(),
                    current_value: val,
                    limit_value: *threshold,
                    utilisation,
                    severity,
                    message: format!(
                        "Diversification ratio {:.3} below minimum {:.3}",
                        val, threshold
                    ),
                })
            }
            RiskLimit::MaxSectorConcentration { threshold, warning_pct, sector } => {
                let val = state.max_sector_concentration(sector);
                Self::check_upper(
                    &format!("MaxSectorConcentration[{}]", sector),
                    val, *threshold, *warning_pct
                )
            }
            RiskLimit::DailyLossLimit { threshold, warning_pct } => {
                let val = (-state.daily_pnl).max(0.0);
                Self::check_upper(limit.name(), val, *threshold, *warning_pct)
            }
            RiskLimit::MaxPositions { threshold } => {
                let val = state.position_count();
                if val <= *threshold { return None; }
                Some(LimitBreach {
                    limit_name: limit.name().to_string(),
                    current_value: val as f64,
                    limit_value: *threshold as f64,
                    utilisation: val as f64 / *threshold as f64,
                    severity: Severity::Breach,
                    message: format!("Position count {} exceeds max {}", val, threshold),
                })
            }
        }
    }

    fn check_upper(name: &str, val: f64, threshold: f64, warning_pct: f64) -> Option<LimitBreach> {
        if val <= threshold * warning_pct {
            return None; // well within limit
        }
        let utilisation = val / threshold.max(1e-12);
        let severity = if val >= threshold * 1.0 {
            if val >= threshold * 1.2 {
                Severity::Critical
            } else {
                Severity::Breach
            }
        } else {
            Severity::Warning
        };

        Some(LimitBreach {
            limit_name: name.to_string(),
            current_value: val,
            limit_value: threshold,
            utilisation,
            severity,
            message: format!("{}: {:.4} vs limit {:.4} ({:.1}%)", name, val, threshold, utilisation * 100.0),
        })
    }

    /// Check a single specific limit type.
    pub fn check_var_limit(&self, var: f64, max_var: f64) -> bool {
        var <= max_var
    }

    pub fn check_drawdown_limit(&self, drawdown: f64, max_dd: f64) -> bool {
        drawdown <= max_dd
    }
}

// ── Pre-trade Risk Check ──────────────────────────────────────────────────────

/// Result of a pre-trade risk check.
#[derive(Debug)]
pub struct PreTradeCheck {
    pub approved: bool,
    pub breaches: Vec<LimitBreach>,
    pub warnings: Vec<String>,
}

/// Check whether a proposed trade would breach any limits.
pub fn pre_trade_check(
    current_state: &PortfolioState,
    trade_asset: &str,
    trade_weight_delta: f64,
    checker: &LimitChecker,
) -> PreTradeCheck {
    // Build a hypothetical state with the trade applied.
    let mut hypo_state = current_state.clone();
    let current_w = hypo_state.weights.get(trade_asset).copied().unwrap_or(0.0);
    hypo_state.weights.insert(trade_asset.to_string(), current_w + trade_weight_delta);

    let breaches = checker.breach_report(&hypo_state);
    let approved = breaches.iter().all(|b| b.severity < Severity::Breach);
    let warnings = breaches
        .iter()
        .map(|b| b.message.clone())
        .collect();

    PreTradeCheck { approved, breaches, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_state() -> PortfolioState {
        let mut weights = HashMap::new();
        weights.insert("A".to_string(), 0.6);
        weights.insert("B".to_string(), 0.4);
        PortfolioState {
            weights,
            sectors: HashMap::new(),
            current_var: 0.025,
            current_drawdown: 0.05,
            daily_pnl: -0.01,
            correlations: HashMap::new(),
            diversification_ratio: 1.2,
        }
    }

    #[test]
    fn breach_concentration_limit() {
        let state = basic_state();
        let checker = LimitChecker::new(vec![
            RiskLimit::MaxConcentration { threshold: 0.5, warning_pct: 0.9 },
        ]);
        let report = checker.breach_report(&state);
        assert!(!report.is_empty(), "Should have concentration breach");
        assert!(report[0].severity >= Severity::Breach);
    }

    #[test]
    fn no_breach_when_within_limits() {
        let state = basic_state();
        let checker = LimitChecker::new(vec![
            RiskLimit::MaxVar { threshold: 0.05, warning_pct: 0.8 },
            RiskLimit::MaxDrawdown { threshold: 0.10, warning_pct: 0.8 },
        ]);
        let report = checker.breach_report(&state);
        assert!(report.is_empty(), "Should be within limits");
    }
}
