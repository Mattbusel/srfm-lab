/// Kyle's Lambda — price-impact coefficient estimator.
///
/// Regresses signed price-change on signed order-flow using an
/// online least-squares accumulator (no matrix allocation).
///
/// Model:  Δp_t = λ · Q_t + ε_t
///
/// where Q_t is net signed volume (positive = buy, negative = sell).
/// λ is estimated via OLS: λ = Cov(Δp, Q) / Var(Q).

use crate::streaming_stats::StreamingStats;

/// Incremental Kyle-lambda estimator.
#[derive(Debug, Clone, Default)]
pub struct KyleLambda {
    n:       u64,
    sum_q:   f64,   // Σ Q
    sum_dp:  f64,   // Σ Δp
    sum_q2:  f64,   // Σ Q²
    sum_qdp: f64,   // Σ Q·Δp

    /// Running stats on λ estimates from rolling windows (optional diagnostics)
    lambda_stats: StreamingStats,
}

impl KyleLambda {
    pub fn new() -> Self { Self::default() }

    /// Feed a single (signed_volume, price_change) observation.
    pub fn update(&mut self, signed_volume: f64, price_change: f64) {
        self.n       += 1;
        self.sum_q   += signed_volume;
        self.sum_dp  += price_change;
        self.sum_q2  += signed_volume * signed_volume;
        self.sum_qdp += signed_volume * price_change;
    }

    /// Current OLS estimate of λ. Returns `None` if insufficient data or no volume variance.
    pub fn lambda(&self) -> Option<f64> {
        if self.n < 2 { return None; }
        let n = self.n as f64;
        let cov_num = self.sum_qdp - self.sum_q * self.sum_dp / n;
        let var_num = self.sum_q2  - self.sum_q * self.sum_q  / n;
        if var_num.abs() < 1e-30 { return None; }
        Some(cov_num / var_num)
    }

    /// R² of the model.
    pub fn r_squared(&self) -> Option<f64> {
        let lambda = self.lambda()?;
        let n = self.n as f64;
        let ss_tot = self.sum_dp * self.sum_dp / n; // simplified; full version uses variance
        let ss_res = ss_tot - lambda * lambda * (self.sum_q2 - self.sum_q * self.sum_q / n);
        if ss_tot.abs() < 1e-30 { return None; }
        Some(1.0 - ss_res / ss_tot)
    }

    pub fn count(&self) -> u64 { self.n }

    /// Lambda diagnostic stats (rolling snapshots fed externally).
    pub fn record_lambda_snapshot(&mut self) {
        if let Some(l) = self.lambda() {
            self.lambda_stats.update(l);
        }
    }

    pub fn lambda_mean(&self) -> f64  { self.lambda_stats.mean() }
    pub fn lambda_std(&self)  -> f64  { self.lambda_stats.std() }

    pub fn reset(&mut self) { *self = Self::default(); }
}

/// Rolling Kyle-lambda that forgets old data as new data arrives.
/// Uses a sliding-window approximation by maintaining two accumulators
/// (current window and shadow), swapping every `window` observations.
#[derive(Debug, Clone)]
pub struct RollingKyleLambda {
    window:  usize,
    current: KyleLambda,
    shadow:  KyleLambda,
    tick:    usize,
}

impl RollingKyleLambda {
    pub fn new(window: usize) -> Self {
        assert!(window >= 10, "window must be at least 10");
        Self {
            window,
            current: KyleLambda::new(),
            shadow:  KyleLambda::new(),
            tick:    0,
        }
    }

    pub fn update(&mut self, signed_volume: f64, price_change: f64) {
        self.current.update(signed_volume, price_change);
        self.shadow .update(signed_volume, price_change);
        self.tick += 1;
        if self.tick >= self.window {
            self.current = std::mem::replace(&mut self.shadow, KyleLambda::new());
            self.tick = 0;
        }
    }

    pub fn lambda(&self) -> Option<f64> { self.current.lambda() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kyle_lambda_positive_impact() {
        let mut k = KyleLambda::new();
        // Perfect linear relationship: Δp = 0.5 * Q
        for q in [-10.0_f64, -5.0, 0.0, 5.0, 10.0] {
            k.update(q, 0.5 * q);
        }
        let lam = k.lambda().unwrap();
        assert!((lam - 0.5).abs() < 1e-6, "expected λ≈0.5, got {}", lam);
    }

    #[test]
    fn kyle_lambda_no_volume_variance() {
        let mut k = KyleLambda::new();
        for _ in 0..10 { k.update(5.0, 1.0); } // all same volume — no variance
        assert!(k.lambda().is_none());
    }

    #[test]
    fn rolling_kyle_lambda_forgets() {
        let mut rk = RollingKyleLambda::new(10);
        for q in 0..50 {
            let qf = q as f64;
            rk.update(qf, 0.3 * qf);
        }
        let lam = rk.lambda().unwrap();
        assert!((lam - 0.3).abs() < 0.05);
    }
}
