// adaptive_learning_rate.rs -- Adaptive learning rate schedulers.
//
// Provides Adam, AdaGrad, RMSProp, and CyclicLR.
// Each optimizer exposes a `step(&mut self, gradient: &[f64]) -> Vec<f64>`
// that returns the parameter *update* (delta to add to parameters).
// Applied to FTRL and PA-II by passing in the relevant gradient vector.

// ---------------------------------------------------------------------------
// AdamOptimizer
// ---------------------------------------------------------------------------

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Update rule:
///   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
///   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
///   m_hat = m_t / (1 - beta1^t)
///   v_hat = v_t / (1 - beta2^t)
///   update = alpha * m_hat / (sqrt(v_hat) + eps)
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    /// First-moment vector (momentum).
    pub m:     Vec<f64>,
    /// Second-moment vector (uncentered variance).
    pub v:     Vec<f64>,
    /// Step counter (starts at 0, incremented before each update).
    pub t:     u64,
    /// Base learning rate.
    pub alpha: f64,
    /// Exponential decay rate for first moment (default 0.9).
    pub beta1: f64,
    /// Exponential decay rate for second moment (default 0.999).
    pub beta2: f64,
    /// Numerical stability constant (default 1e-8).
    pub eps:   f64,
}

impl AdamOptimizer {
    /// Construct Adam with standard defaults.
    pub fn new(dim: usize, alpha: f64) -> Self {
        AdamOptimizer {
            m:     vec![0.0; dim],
            v:     vec![0.0; dim],
            t:     0,
            alpha,
            beta1: 0.9,
            beta2: 0.999,
            eps:   1e-8,
        }
    }

    /// Construct Adam with custom beta values.
    pub fn with_betas(dim: usize, alpha: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        AdamOptimizer { m: vec![0.0; dim], v: vec![0.0; dim], t: 0, alpha, beta1, beta2, eps }
    }

    /// Compute the parameter update for one gradient step.
    ///
    /// Returns a Vec of the same length as `gradient` -- add this to the
    /// parameters to apply the update (i.e. params -= step(...) for minimization).
    pub fn step(&mut self, gradient: &[f64]) -> Vec<f64> {
        self.t += 1;
        let t = self.t as f64;

        // Ensure buffers are large enough.
        if gradient.len() > self.m.len() {
            self.m.resize(gradient.len(), 0.0);
            self.v.resize(gradient.len(), 0.0);
        }

        let bias_corr1 = 1.0 - self.beta1.powf(t);
        let bias_corr2 = 1.0 - self.beta2.powf(t);

        let mut updates = Vec::with_capacity(gradient.len());
        for (i, &g) in gradient.iter().enumerate() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            let m_hat = self.m[i] / bias_corr1;
            let v_hat = self.v[i] / bias_corr2;
            updates.push(self.alpha * m_hat / (v_hat.sqrt() + self.eps));
        }
        updates
    }

    /// Reset all state (useful for re-training).
    pub fn reset(&mut self) {
        for x in self.m.iter_mut() { *x = 0.0; }
        for x in self.v.iter_mut() { *x = 0.0; }
        self.t = 0;
    }
}

// ---------------------------------------------------------------------------
// AdaGrad
// ---------------------------------------------------------------------------

/// AdaGrad optimizer.
///
/// Update rule:
///   G_t += g_t^2
///   update = lr * g_t / (sqrt(G_t) + eps)
#[derive(Debug, Clone)]
pub struct AdaGrad {
    /// Accumulated squared gradients.
    pub sum_sq_grad: Vec<f64>,
    /// Base learning rate.
    pub lr:  f64,
    /// Numerical stability constant.
    pub eps: f64,
}

impl AdaGrad {
    /// Construct AdaGrad with default eps=1e-8.
    pub fn new(dim: usize, lr: f64) -> Self {
        AdaGrad { sum_sq_grad: vec![0.0; dim], lr, eps: 1e-8 }
    }

    /// Construct with custom eps.
    pub fn with_eps(dim: usize, lr: f64, eps: f64) -> Self {
        AdaGrad { sum_sq_grad: vec![0.0; dim], lr, eps }
    }

    /// Compute the parameter update for one gradient step.
    pub fn step(&mut self, gradient: &[f64]) -> Vec<f64> {
        if gradient.len() > self.sum_sq_grad.len() {
            self.sum_sq_grad.resize(gradient.len(), 0.0);
        }
        let mut updates = Vec::with_capacity(gradient.len());
        for (i, &g) in gradient.iter().enumerate() {
            self.sum_sq_grad[i] += g * g;
            updates.push(self.lr * g / (self.sum_sq_grad[i].sqrt() + self.eps));
        }
        updates
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        for x in self.sum_sq_grad.iter_mut() { *x = 0.0; }
    }

    /// Effective per-dimension learning rate at the current step.
    pub fn effective_lr(&self, dim: usize) -> f64 {
        if dim >= self.sum_sq_grad.len() {
            return self.lr;
        }
        self.lr / (self.sum_sq_grad[dim].sqrt() + self.eps)
    }
}

// ---------------------------------------------------------------------------
// RMSProp
// ---------------------------------------------------------------------------

/// RMSProp optimizer.
///
/// Update rule:
///   E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
///   update = lr * g_t / (sqrt(E[g^2]_t) + eps)
#[derive(Debug, Clone)]
pub struct RMSProp {
    /// Exponential moving average of squared gradients.
    pub sq_grad_ema: Vec<f64>,
    /// Base learning rate.
    pub lr:  f64,
    /// Decay coefficient for the EMA (default 0.9).
    pub rho: f64,
    /// Numerical stability constant.
    pub eps: f64,
}

impl RMSProp {
    /// Construct RMSProp with standard defaults (rho=0.9, eps=1e-8).
    pub fn new(dim: usize, lr: f64) -> Self {
        RMSProp { sq_grad_ema: vec![0.0; dim], lr, rho: 0.9, eps: 1e-8 }
    }

    /// Construct with custom rho and eps.
    pub fn with_params(dim: usize, lr: f64, rho: f64, eps: f64) -> Self {
        RMSProp { sq_grad_ema: vec![0.0; dim], lr, rho, eps }
    }

    /// Compute the parameter update for one gradient step.
    pub fn step(&mut self, gradient: &[f64]) -> Vec<f64> {
        if gradient.len() > self.sq_grad_ema.len() {
            self.sq_grad_ema.resize(gradient.len(), 0.0);
        }
        let mut updates = Vec::with_capacity(gradient.len());
        for (i, &g) in gradient.iter().enumerate() {
            self.sq_grad_ema[i] = self.rho * self.sq_grad_ema[i] + (1.0 - self.rho) * g * g;
            updates.push(self.lr * g / (self.sq_grad_ema[i].sqrt() + self.eps));
        }
        updates
    }

    /// Reset accumulated state.
    pub fn reset(&mut self) {
        for x in self.sq_grad_ema.iter_mut() { *x = 0.0; }
    }
}

// ---------------------------------------------------------------------------
// CyclicLR
// ---------------------------------------------------------------------------

/// Triangular cyclic learning rate schedule.
///
/// Cycle: lr goes from base_lr up to max_lr and back over 2 * step_size steps.
///
/// Formula (triangular1 policy):
///   cycle = floor(1 + t / (2 * step_size))
///   x     = |t / step_size - 2 * cycle + 1|
///   lr    = base_lr + (max_lr - base_lr) * max(0, 1 - x)
#[derive(Debug, Clone)]
pub struct CyclicLR {
    /// Minimum (base) learning rate.
    pub base_lr:   f64,
    /// Maximum learning rate reached at cycle peak.
    pub max_lr:    f64,
    /// Half-cycle length in steps.
    pub step_size: u64,
    /// Current step counter.
    pub t:         u64,
}

impl CyclicLR {
    /// Construct a CyclicLR scheduler.
    pub fn new(base_lr: f64, max_lr: f64, step_size: u64) -> Self {
        assert!(max_lr >= base_lr, "max_lr must be >= base_lr");
        assert!(step_size >= 1, "step_size must be >= 1");
        CyclicLR { base_lr, max_lr, step_size, t: 0 }
    }

    /// Current learning rate.
    pub fn lr(&self) -> f64 {
        let ss = self.step_size as f64;
        let t  = self.t as f64;
        let cycle = (1.0 + t / (2.0 * ss)).floor();
        let x = (t / ss - 2.0 * cycle + 1.0).abs();
        self.base_lr + (self.max_lr - self.base_lr) * (1.0_f64 - x).max(0.0)
    }

    /// Advance the step counter by one and return the new learning rate.
    pub fn step(&mut self) -> f64 {
        self.t += 1;
        self.lr()
    }

    /// Reset to the start of the first cycle.
    pub fn reset(&mut self) {
        self.t = 0;
    }

    /// True when the scheduler is at the start of a new full cycle.
    pub fn is_cycle_start(&self) -> bool {
        self.t % (2 * self.step_size) == 0
    }
}

// ---------------------------------------------------------------------------
// Plug-in: apply adaptive LR to FTRL / PA-II gradient vectors
// ---------------------------------------------------------------------------

/// Apply an Adam update to an external weight vector in-place.
///
/// Convenience function for wiring Adam into FTRL and PA-II.
/// `weights` -- current parameter vector (modified in place)
/// `gradient` -- gradient wrt loss
/// `adam` -- mutable reference to the optimizer state
pub fn adam_apply(weights: &mut Vec<f64>, gradient: &[f64], adam: &mut AdamOptimizer) {
    let updates = adam.step(gradient);
    if weights.len() < updates.len() {
        weights.resize(updates.len(), 0.0);
    }
    for (w, u) in weights.iter_mut().zip(updates.iter()) {
        *w -= u; // gradient descent: subtract update
    }
}

/// Apply an AdaGrad update to an external weight vector in-place.
pub fn adagrad_apply(weights: &mut Vec<f64>, gradient: &[f64], adagrad: &mut AdaGrad) {
    let updates = adagrad.step(gradient);
    if weights.len() < updates.len() {
        weights.resize(updates.len(), 0.0);
    }
    for (w, u) in weights.iter_mut().zip(updates.iter()) {
        *w -= u;
    }
}

/// Apply an RMSProp update to an external weight vector in-place.
pub fn rmsprop_apply(weights: &mut Vec<f64>, gradient: &[f64], rmsprop: &mut RMSProp) {
    let updates = rmsprop.step(gradient);
    if weights.len() < updates.len() {
        weights.resize(updates.len(), 0.0);
    }
    for (w, u) in weights.iter_mut().zip(updates.iter()) {
        *w -= u;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- AdamOptimizer tests -------------------------------------------------

    #[test]
    fn test_adam_update_is_nonzero() {
        let mut adam = AdamOptimizer::new(3, 0.01);
        let g = vec![1.0, -2.0, 0.5];
        let u = adam.step(&g);
        assert_eq!(u.len(), 3);
        for &v in &u {
            assert!(v.abs() > 0.0);
        }
    }

    #[test]
    fn test_adam_step_counter_increments() {
        let mut adam = AdamOptimizer::new(2, 0.001);
        for _ in 0..5 {
            adam.step(&[1.0, 1.0]);
        }
        assert_eq!(adam.t, 5);
    }

    #[test]
    fn test_adam_reset_zeroes_state() {
        let mut adam = AdamOptimizer::new(2, 0.001);
        adam.step(&[3.0, 3.0]);
        adam.reset();
        assert_eq!(adam.t, 0);
        assert!(adam.m.iter().all(|&x| x == 0.0));
        assert!(adam.v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_adam_update_decreases_loss_on_simple_quadratic() {
        // Minimize f(w) = w^2 starting from w=5.
        let mut w = 5.0_f64;
        let mut adam = AdamOptimizer::new(1, 0.1);
        for _ in 0..200 {
            let g = 2.0 * w;
            let u = adam.step(&[g]);
            w -= u[0];
        }
        assert!(w.abs() < 0.1, "w should converge near 0, got {}", w);
    }

    // -- AdaGrad tests -------------------------------------------------------

    #[test]
    fn test_adagrad_step_shape() {
        let mut ag = AdaGrad::new(4, 0.1);
        let u = ag.step(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(u.len(), 4);
    }

    #[test]
    fn test_adagrad_lr_decreases_over_time() {
        let mut ag = AdaGrad::new(1, 1.0);
        let u1 = ag.step(&[1.0])[0];
        let u2 = ag.step(&[1.0])[0];
        // Second update should have lower effective LR.
        assert!(u2.abs() < u1.abs(), "AdaGrad LR should decrease: u1={} u2={}", u1, u2);
    }

    #[test]
    fn test_adagrad_reset() {
        let mut ag = AdaGrad::new(2, 0.1);
        ag.step(&[10.0, 10.0]);
        ag.reset();
        assert!(ag.sum_sq_grad.iter().all(|&x| x == 0.0));
    }

    // -- RMSProp tests -------------------------------------------------------

    #[test]
    fn test_rmsprop_step_shape() {
        let mut rms = RMSProp::new(3, 0.01);
        let u = rms.step(&[1.0, -1.0, 0.5]);
        assert_eq!(u.len(), 3);
    }

    #[test]
    fn test_rmsprop_ema_updates() {
        let mut rms = RMSProp::new(1, 0.01);
        rms.step(&[2.0]);
        let after_one = rms.sq_grad_ema[0];
        rms.step(&[2.0]);
        let after_two = rms.sq_grad_ema[0];
        // EMA should increase toward g^2 = 4.
        assert!(after_two > after_one);
        assert!(after_two < 4.0);
    }

    // -- CyclicLR tests ------------------------------------------------------

    #[test]
    fn test_cyclic_lr_starts_at_base() {
        let clr = CyclicLR::new(0.001, 0.01, 100);
        // At t=0, lr should be base_lr.
        assert!((clr.lr() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_cyclic_lr_peak_at_step_size() {
        let mut clr = CyclicLR::new(0.001, 0.01, 10);
        // Peak should be at t = step_size.
        for _ in 0..10 {
            clr.step();
        }
        let peak = clr.lr();
        assert!((peak - 0.01).abs() < 1e-6, "peak lr={}", peak);
    }

    #[test]
    fn test_cyclic_lr_returns_to_base() {
        let mut clr = CyclicLR::new(0.001, 0.01, 10);
        // After a full cycle (2 * step_size steps) we should be back at base.
        for _ in 0..(2 * 10) {
            clr.step();
        }
        let lr = clr.lr();
        assert!((lr - 0.001).abs() < 1e-6, "lr after full cycle={}", lr);
    }

    #[test]
    fn test_cyclic_lr_stays_in_bounds() {
        let mut clr = CyclicLR::new(0.001, 0.01, 50);
        for _ in 0..300 {
            let lr = clr.step();
            assert!(lr >= 0.001 - 1e-9 && lr <= 0.01 + 1e-9, "lr out of bounds: {}", lr);
        }
    }

    // -- Plug-in helpers -----------------------------------------------------

    #[test]
    fn test_adam_apply_modifies_weights() {
        let mut weights = vec![1.0, 2.0, 3.0];
        let grad = vec![0.1, 0.2, 0.3];
        let mut adam = AdamOptimizer::new(3, 0.01);
        adam_apply(&mut weights, &grad, &mut adam);
        // Weights should have changed.
        assert!(weights[0] != 1.0 || weights[1] != 2.0 || weights[2] != 3.0);
    }

    #[test]
    fn test_rmsprop_apply_modifies_weights() {
        let mut weights = vec![0.5, -0.5];
        let grad = vec![1.0, -1.0];
        let mut rms = RMSProp::new(2, 0.01);
        rmsprop_apply(&mut weights, &grad, &mut rms);
        assert!(weights[0] < 0.5, "weight should decrease for positive gradient");
        assert!(weights[1] > -0.5, "weight should increase for negative gradient");
    }
}
