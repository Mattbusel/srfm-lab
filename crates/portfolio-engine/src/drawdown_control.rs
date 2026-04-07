// drawdown_control.rs -- Drawdown control and position scaling
// Tracks equity curves, peaks, and computes position size scalars
// that reduce exposure as drawdown deepens.

/// Snapshot of current drawdown state.
#[derive(Debug, Clone)]
pub struct DrawdownState {
    /// Current portfolio equity (USD or index-normalized).
    pub current_equity: f64,
    /// Peak equity observed since inception (high-water mark).
    pub peak_equity: f64,
    /// Current drawdown as a positive percentage (0.0 = at peak, 0.15 = 15% down).
    pub current_drawdown_pct: f64,
    /// Number of consecutive days the portfolio has been below the peak.
    pub days_in_drawdown: u32,
    /// Maximum drawdown observed over the entire history.
    pub max_drawdown_pct: f64,
}

impl DrawdownState {
    /// Is the portfolio currently at its high-water mark?
    pub fn is_at_peak(&self) -> bool {
        self.current_drawdown_pct < 1e-9
    }
}

/// Drawdown controller -- maintains equity history and computes position scalars.
#[derive(Debug, Clone)]
pub struct DrawdownController {
    /// Recorded equity values, oldest first.
    equity_history: Vec<f64>,
    /// High-water mark.
    peak_equity: f64,
    /// Number of consecutive days below peak.
    days_in_drawdown: u32,
    /// Maximum drawdown (as positive fraction) observed.
    max_drawdown_pct: f64,
}

impl DrawdownController {
    /// Create a new controller with an initial equity value.
    pub fn new(initial_equity: f64) -> Self {
        DrawdownController {
            equity_history: vec![initial_equity],
            peak_equity: initial_equity,
            days_in_drawdown: 0,
            max_drawdown_pct: 0.0,
        }
    }

    /// Create a controller with no history. First call to `update` sets the peak.
    pub fn empty() -> Self {
        DrawdownController {
            equity_history: Vec::new(),
            peak_equity: 0.0,
            days_in_drawdown: 0,
            max_drawdown_pct: 0.0,
        }
    }

    /// Record a new equity observation and return the updated DrawdownState.
    pub fn update(&mut self, equity: f64) -> DrawdownState {
        // Initialize peak on first observation.
        if self.equity_history.is_empty() {
            self.peak_equity = equity;
        }

        self.equity_history.push(equity);

        // Update peak (high-water mark).
        if equity > self.peak_equity {
            self.peak_equity = equity;
            self.days_in_drawdown = 0;
        } else {
            self.days_in_drawdown += 1;
        }

        // Current drawdown as fraction.
        let drawdown_pct = if self.peak_equity > 0.0 {
            (self.peak_equity - equity) / self.peak_equity
        } else {
            0.0
        }
        .max(0.0);

        // Update max drawdown.
        if drawdown_pct > self.max_drawdown_pct {
            self.max_drawdown_pct = drawdown_pct;
        }

        DrawdownState {
            current_equity: equity,
            peak_equity: self.peak_equity,
            current_drawdown_pct: drawdown_pct,
            days_in_drawdown: self.days_in_drawdown,
            max_drawdown_pct: self.max_drawdown_pct,
        }
    }

    /// Return the current drawdown state without updating.
    pub fn current_state(&self) -> DrawdownState {
        let equity = self.equity_history.last().copied().unwrap_or(0.0);
        let drawdown_pct = if self.peak_equity > 0.0 {
            ((self.peak_equity - equity) / self.peak_equity).max(0.0)
        } else {
            0.0
        };
        DrawdownState {
            current_equity: equity,
            peak_equity: self.peak_equity,
            current_drawdown_pct: drawdown_pct,
            days_in_drawdown: self.days_in_drawdown,
            max_drawdown_pct: self.max_drawdown_pct,
        }
    }

    /// Compute the position scalar based on current drawdown:
    ///
    /// | Drawdown range | Scalar | Mode              |
    /// |----------------|--------|-------------------|
    /// | 0 -- 3%        | 1.00   | Full size         |
    /// | 3 -- 7%        | 0.75   | Reduced           |
    /// | 7 -- 12%       | 0.50   | Half size         |
    /// | > 12%          | 0.25   | Preservation mode |
    pub fn get_position_scalar(&self) -> f64 {
        let state = self.current_state();
        let dd = state.current_drawdown_pct;
        if dd <= 0.03 {
            1.00
        } else if dd <= 0.07 {
            0.75
        } else if dd <= 0.12 {
            0.50
        } else {
            0.25
        }
    }

    /// Return true if the portfolio is in recovery mode (drawdown > 10%).
    pub fn is_in_recovery_mode(&self) -> bool {
        let state = self.current_state();
        state.current_drawdown_pct > 0.10
    }

    /// Estimate the number of days to recover from current drawdown,
    /// given the current daily return rate.
    ///
    /// Returns None if current_rate <= 0 (cannot recover) or if at peak.
    ///
    /// Formula: peak = equity * (1 + r)^d
    /// d = log(peak / equity) / log(1 + r)
    pub fn days_to_recover(&self, current_rate: f64) -> Option<f64> {
        let state = self.current_state();
        if state.current_drawdown_pct < 1e-9 {
            // Already at peak.
            return Some(0.0);
        }
        if current_rate <= 0.0 {
            return None;
        }
        // Need to grow by (1 / (1 - drawdown_pct)) ratio.
        let ratio = state.peak_equity / state.current_equity;
        if ratio <= 1.0 {
            return Some(0.0);
        }
        let days = ratio.ln() / (1.0 + current_rate).ln();
        Some(days)
    }

    /// Return the number of equity observations recorded.
    pub fn history_len(&self) -> usize {
        self.equity_history.len()
    }

    /// Return a copy of the equity history.
    pub fn equity_history(&self) -> &[f64] {
        &self.equity_history
    }

    /// Compute annualized volatility of the equity curve (using log returns).
    /// Assumes daily observations. Returns None if fewer than 2 observations.
    pub fn annualized_vol(&self) -> Option<f64> {
        let n = self.equity_history.len();
        if n < 2 {
            return None;
        }
        let log_returns: Vec<f64> = self.equity_history
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 && w[1] > 0.0 {
                    Some((w[1] / w[0]).ln())
                } else {
                    None
                }
            })
            .collect();
        if log_returns.is_empty() {
            return None;
        }
        let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        let variance = log_returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
            / (log_returns.len() as f64 - 1.0).max(1.0);
        Some((variance * 252.0_f64).sqrt())
    }

    /// Compute the Calmar ratio: annualized return / max drawdown.
    /// Returns None if max drawdown is zero or history is too short.
    pub fn calmar_ratio(&self) -> Option<f64> {
        if self.equity_history.len() < 2 || self.max_drawdown_pct < 1e-9 {
            return None;
        }
        let first = *self.equity_history.first().unwrap();
        let last = *self.equity_history.last().unwrap();
        if first <= 0.0 {
            return None;
        }
        let n_days = (self.equity_history.len() - 1) as f64;
        let total_return = last / first - 1.0;
        let ann_return = (1.0 + total_return).powf(252.0 / n_days) - 1.0;
        Some(ann_return / self.max_drawdown_pct)
    }

    /// Return the duration of the worst drawdown (in observations).
    pub fn max_drawdown_duration(&self) -> u32 {
        let mut peak = self.equity_history.first().copied().unwrap_or(0.0);
        let mut duration = 0u32;
        let mut max_duration = 0u32;
        let mut in_dd = false;

        for &eq in &self.equity_history {
            if eq > peak {
                peak = eq;
                if in_dd {
                    if duration > max_duration {
                        max_duration = duration;
                    }
                    duration = 0;
                    in_dd = false;
                }
            } else if eq < peak {
                in_dd = true;
                duration += 1;
            }
        }
        if in_dd && duration > max_duration {
            max_duration = duration;
        }
        max_duration
    }

    /// Compute a scaled position size in USD given full-size position and equity.
    /// Returns full_size_usd * get_position_scalar().
    pub fn scaled_position_usd(&self, full_size_usd: f64) -> f64 {
        full_size_usd * self.get_position_scalar()
    }

    /// Reset history and peak. Useful for starting a new trading period.
    pub fn reset(&mut self, new_initial_equity: f64) {
        self.equity_history.clear();
        self.equity_history.push(new_initial_equity);
        self.peak_equity = new_initial_equity;
        self.days_in_drawdown = 0;
        self.max_drawdown_pct = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_drawdown_at_start() {
        let ctrl = DrawdownController::new(100_000.0);
        let state = ctrl.current_state();
        assert_eq!(state.current_drawdown_pct, 0.0);
        assert!(state.is_at_peak());
    }

    #[test]
    fn test_update_tracks_peak() {
        let mut ctrl = DrawdownController::new(100_000.0);
        ctrl.update(105_000.0);
        ctrl.update(103_000.0);
        let state = ctrl.current_state();
        assert_eq!(state.peak_equity, 105_000.0);
        let expected_dd = (105_000.0 - 103_000.0) / 105_000.0;
        assert!((state.current_drawdown_pct - expected_dd).abs() < 1e-9);
    }

    #[test]
    fn test_scalar_full_size() {
        let ctrl = DrawdownController::new(100_000.0);
        assert_eq!(ctrl.get_position_scalar(), 1.0);
    }

    #[test]
    fn test_scalar_reduced_after_drawdown() {
        let mut ctrl = DrawdownController::new(100_000.0);
        ctrl.update(95_000.0); // 5% drawdown -> scalar 0.75
        assert_eq!(ctrl.get_position_scalar(), 0.75);
    }

    #[test]
    fn test_recovery_mode() {
        let mut ctrl = DrawdownController::new(100_000.0);
        ctrl.update(88_000.0); // 12% drawdown -> recovery mode
        assert!(ctrl.is_in_recovery_mode());
    }

    #[test]
    fn test_not_recovery_mode_shallow_dd() {
        let mut ctrl = DrawdownController::new(100_000.0);
        ctrl.update(95_000.0); // 5% -> not recovery mode
        assert!(!ctrl.is_in_recovery_mode());
    }
}
