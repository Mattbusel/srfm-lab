// streaming/adaptive_filter.rs -- Adaptive signal filters for noise reduction on streaming
// price data. Implements Kalman (1D and 2D), LMS, RLS, Adaptive EMA, and a rolling
// Hodrick-Prescott trend extractor.

use std::collections::VecDeque;

// ─── KalmanFilter1D ───────────────────────────────────────────────────────────

/// One-dimensional Kalman filter treating price as scalar state.
///
/// State transition: x_k = x_{k-1} + w_k  (random walk)
/// Measurement:      z_k = x_k + v_k
///
/// Q = process noise variance, R = measurement noise variance.
#[derive(Debug, Clone)]
pub struct KalmanFilter1D {
    /// Current state estimate (price).
    state: f64,
    /// Current error variance (P).
    variance: f64,
    /// Process noise covariance Q.
    process_noise: f64,
    /// Measurement noise covariance R.
    measurement_noise: f64,
    /// Number of updates processed.
    n: u64,
}

impl KalmanFilter1D {
    /// Create a new 1D Kalman filter.
    ///
    /// - `initial_price`   -- starting state estimate
    /// - `process_noise`   -- Q; controls how quickly the filter tracks changes
    /// - `measurement_noise` -- R; reflects sensor/tick noise level
    pub fn new(initial_price: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            state: initial_price,
            variance: 1.0,
            process_noise,
            measurement_noise,
            n: 0,
        }
    }

    /// Feed a new price measurement and return the filtered estimate.
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Predict
        let p_pred = self.variance + self.process_noise;

        // Kalman gain
        let gain = p_pred / (p_pred + self.measurement_noise);

        // Update
        self.state = self.state + gain * (measurement - self.state);
        self.variance = (1.0 - gain) * p_pred;
        self.n += 1;

        self.state
    }

    /// Return the current state estimate.
    pub fn get_state(&self) -> f64 {
        self.state
    }

    /// Return the current error variance.
    pub fn get_variance(&self) -> f64 {
        self.variance
    }

    /// Return the number of updates processed.
    pub fn n_updates(&self) -> u64 {
        self.n
    }

    /// Reset the filter to a new initial state.
    pub fn reset(&mut self, initial_price: f64) {
        self.state = initial_price;
        self.variance = 1.0;
        self.n = 0;
    }
}

// ─── KalmanFilter2D ───────────────────────────────────────────────────────────

/// Two-dimensional Kalman filter with state [price, velocity].
///
/// State transition (constant-velocity model):
///   price_{k} = price_{k-1} + velocity_{k-1} * dt
///   velocity_{k} = velocity_{k-1}
///
/// Process noise Q is auto-tuned via an EM-style running estimate of
/// innovation variance after every `em_window` observations.
#[derive(Debug, Clone)]
pub struct KalmanFilter2D {
    /// State vector [price, velocity].
    state: [f64; 2],
    /// 2x2 covariance matrix stored row-major [P00, P01, P10, P11].
    covariance: [f64; 4],
    /// Process noise scale (applied to identity).
    q_scale: f64,
    /// Measurement noise variance.
    measurement_noise: f64,
    /// Time step between observations (seconds, default 1.0).
    dt: f64,
    /// Circular buffer of recent innovations for EM tuning.
    innovation_buf: VecDeque<f64>,
    /// Window size for EM Q estimation.
    em_window: usize,
    /// Update counter.
    n: u64,
}

impl KalmanFilter2D {
    pub fn new(initial_price: f64, measurement_noise: f64, dt: f64) -> Self {
        Self {
            state: [initial_price, 0.0],
            covariance: [1.0, 0.0, 0.0, 1.0],
            q_scale: 1e-4,
            measurement_noise,
            dt,
            innovation_buf: VecDeque::with_capacity(50),
            em_window: 50,
            n: 0,
        }
    }

    /// Feed a price measurement and return (filtered_price, velocity).
    pub fn update(&mut self, measurement: f64) -> (f64, f64) {
        let dt = self.dt;

        // State transition matrix F = [[1, dt], [0, 1]]
        // Predicted state
        let x0 = self.state[0] + self.state[1] * dt;
        let x1 = self.state[1];

        // Predicted covariance P_pred = F*P*F^T + Q
        let [p00, p01, p10, p11] = self.covariance;
        // F*P: row 0 = [p00 + dt*p10, p01 + dt*p11], row 1 = [p10, p11]
        let fp00 = p00 + dt * p10;
        let fp01 = p01 + dt * p11;
        let fp10 = p10;
        let fp11 = p11;
        // (F*P)*F^T: col 0 stays, col 1 adds dt * col 0 from F^T perspective
        let pp00 = fp00 + dt * fp01;
        let pp01 = fp01;
        let pp10 = fp10 + dt * fp11;
        let pp11 = fp11;
        // Add Q (diagonal)
        let q = self.q_scale;
        let pp00 = pp00 + q;
        let pp11 = pp11 + q;

        // Innovation -- only price is observed, H = [1, 0]
        let innovation = measurement - x0;

        // Innovation covariance S = H*P_pred*H^T + R = pp00 + R
        let s = pp00 + self.measurement_noise;

        // Kalman gain K = P_pred * H^T / S  => 2x1 vector
        let k0 = pp00 / s;
        let k1 = pp10 / s;

        // Updated state
        self.state[0] = x0 + k0 * innovation;
        self.state[1] = x1 + k1 * innovation;

        // Updated covariance P = (I - K*H)*P_pred
        self.covariance[0] = pp00 - k0 * pp00;
        self.covariance[1] = pp01 - k0 * pp01;
        self.covariance[2] = pp10 - k1 * pp00;
        self.covariance[3] = pp11 - k1 * pp01;

        // EM update: accumulate innovation^2 and re-estimate q_scale
        if self.innovation_buf.len() == self.em_window {
            self.innovation_buf.pop_front();
        }
        self.innovation_buf.push_back(innovation);
        if self.n > 0 && self.n % self.em_window as u64 == 0 {
            self.em_tune_q();
        }

        self.n += 1;
        (self.state[0], self.state[1])
    }

    /// EM step: re-estimate q_scale from innovation variance.
    fn em_tune_q(&mut self) {
        if self.innovation_buf.is_empty() {
            return;
        }
        let n = self.innovation_buf.len() as f64;
        let mean = self.innovation_buf.iter().sum::<f64>() / n;
        let var = self.innovation_buf.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        // Q ~ innovation variance - R (clamped to small positive)
        let new_q = (var - self.measurement_noise).max(1e-8);
        // Smooth update
        self.q_scale = 0.9 * self.q_scale + 0.1 * new_q;
    }

    pub fn get_price(&self) -> f64 { self.state[0] }
    pub fn get_velocity(&self) -> f64 { self.state[1] }
    pub fn get_q_scale(&self) -> f64 { self.q_scale }
}

// ─── LMSFilter ────────────────────────────────────────────────────────────────

/// Least Mean Squares (LMS) adaptive FIR filter.
///
/// Maintains a weight vector w of length `order`. On each sample:
///   y = w^T * x_buffer
///   error = desired - y   (desired defaults to 0 -- noise cancellation mode)
///   w += mu * error * x_buffer
#[derive(Debug, Clone)]
pub struct LMSFilter {
    weights: Vec<f64>,
    buffer: VecDeque<f64>,
    /// Step size (learning rate).
    mu: f64,
    /// Filter order (number of taps).
    order: usize,
    /// Last filtered output.
    last_output: f64,
}

impl LMSFilter {
    /// Create a new LMS filter.
    ///
    /// - `order` -- number of FIR taps
    /// - `mu`    -- learning rate (typical: 0.001 - 0.1)
    pub fn new(order: usize, mu: f64) -> Self {
        Self {
            weights: vec![0.0; order],
            buffer: VecDeque::from(vec![0.0; order]),
            mu,
            order,
            last_output: 0.0,
        }
    }

    /// Filter input sample x. Uses previous output as desired signal
    /// (1-step predictor mode for smoothing).
    pub fn filter(&mut self, x: f64) -> f64 {
        // Slide buffer
        if self.buffer.len() == self.order {
            self.buffer.pop_back();
        }
        self.buffer.push_front(x);

        // Compute output y = w^T * x_buf
        let y: f64 = self.weights.iter().zip(self.buffer.iter()).map(|(w, b)| w * b).sum();

        // Error against current input (noise cancellation: desired = x, signal = y)
        let error = x - y;

        // Update weights
        for (i, w) in self.weights.iter_mut().enumerate() {
            let xi = self.buffer[i];
            *w += self.mu * error * xi;
        }

        self.last_output = y;
        y
    }

    /// Return the current weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Return last filtered output.
    pub fn last_output(&self) -> f64 {
        self.last_output
    }

    /// Reset weights and buffer to zero.
    pub fn reset(&mut self) {
        self.weights.iter_mut().for_each(|w| *w = 0.0);
        self.buffer.iter_mut().for_each(|b| *b = 0.0);
        self.last_output = 0.0;
    }
}

// ─── RLSFilter ────────────────────────────────────────────────────────────────

/// Recursive Least Squares (RLS) adaptive filter.
///
/// Maintains inverse covariance matrix P and weights w.
/// Forgetting factor lambda in (0, 1] controls how fast old data is discarded.
/// lambda = 1.0 means equal weighting (standard LS); lambda = 0.99 tracks
/// non-stationary signals well.
#[derive(Debug, Clone)]
pub struct RLSFilter {
    weights: Vec<f64>,
    /// Inverse covariance matrix P (order x order), stored as flat Vec row-major.
    p_inv: Vec<f64>,
    /// Forgetting factor.
    lambda: f64,
    order: usize,
    buffer: VecDeque<f64>,
}

impl RLSFilter {
    /// Create a new RLS filter.
    ///
    /// - `order`  -- number of taps
    /// - `lambda` -- forgetting factor (0.95 - 0.999 typical)
    /// - `delta`  -- initial P = delta * I (large delta = small initial confidence)
    pub fn new(order: usize, lambda: f64, delta: f64) -> Self {
        let mut p_inv = vec![0.0_f64; order * order];
        for i in 0..order {
            p_inv[i * order + i] = delta;
        }
        Self {
            weights: vec![0.0; order],
            p_inv,
            lambda,
            order,
            buffer: VecDeque::from(vec![0.0; order]),
        }
    }

    /// Filter one sample and return the predicted output.
    pub fn filter(&mut self, x: f64) -> f64 {
        let n = self.order;

        // Slide buffer
        if self.buffer.len() == n {
            self.buffer.pop_back();
        }
        self.buffer.push_front(x);

        // x_vec: input vector from buffer
        let xv: Vec<f64> = self.buffer.iter().cloned().collect();

        // Output: y = w^T * x
        let y: f64 = self.weights.iter().zip(xv.iter()).map(|(w, xi)| w * xi).sum();

        // Error: use x as desired (1-step predictor)
        let error = x - y;

        // k = P * x / (lambda + x^T * P * x)
        // Compute Px = P * xv
        let mut px = vec![0.0_f64; n];
        for i in 0..n {
            for j in 0..n {
                px[i] += self.p_inv[i * n + j] * xv[j];
            }
        }
        // x^T * Px
        let xtp_x: f64 = xv.iter().zip(px.iter()).map(|(xi, pxi)| xi * pxi).sum();
        let denom = self.lambda + xtp_x;

        // Gain vector k
        let k: Vec<f64> = px.iter().map(|pxi| pxi / denom).collect();

        // Update weights
        for (i, w) in self.weights.iter_mut().enumerate() {
            *w += k[i] * error;
        }

        // Update P: P = (P - k * x^T * P) / lambda
        // k * x^T is outer product
        let mut new_p = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                // (k * x^T * P)[i,j] = k[i] * sum_l (xv[l] * P[l,j])
                // But standard RLS update is P = (P - k*(Px)^T) / lambda
                new_p[i * n + j] = (self.p_inv[i * n + j] - k[i] * px[j]) / self.lambda;
            }
        }
        self.p_inv = new_p;

        y
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

// ─── AdaptiveEMA ─────────────────────────────────────────────────────────────

/// Exponential Moving Average with adaptive alpha.
///
/// Alpha adapts based on market efficiency:
///   alpha = |momentum| / (|momentum| + sigma)
///
/// In an efficient (noisy) market momentum ~ 0 -> alpha -> 0 (slow EMA).
/// In a trending market momentum >> sigma -> alpha -> 1 (fast EMA).
#[derive(Debug, Clone)]
pub struct AdaptiveEMA {
    value: f64,
    /// Window for computing sigma (rolling std of returns).
    window: usize,
    price_buf: VecDeque<f64>,
    /// Current effective alpha.
    current_alpha: f64,
    initialized: bool,
}

impl AdaptiveEMA {
    pub fn new(window: usize) -> Self {
        Self {
            value: 0.0,
            window,
            price_buf: VecDeque::with_capacity(window + 2),
            current_alpha: 0.5,
            initialized: false,
        }
    }

    /// Update with new price and return the smoothed value.
    pub fn update(&mut self, price: f64) -> f64 {
        if !self.initialized {
            self.value = price;
            self.initialized = true;
            self.price_buf.push_back(price);
            return price;
        }

        self.price_buf.push_back(price);
        if self.price_buf.len() > self.window + 1 {
            self.price_buf.pop_front();
        }

        // Compute momentum and sigma from recent prices
        if self.price_buf.len() >= 2 {
            let n = self.price_buf.len();
            // Momentum = last price - first price in window (net change)
            let momentum = (self.price_buf[n - 1] - self.price_buf[0]).abs();

            // Sigma = sum of |individual bar changes|
            let price_slice: Vec<f64> = self.price_buf.iter().cloned().collect();
            let sigma: f64 = price_slice.windows(2).map(|w| (w[1] - w[0]).abs()).sum();

            if sigma > 1e-12 {
                self.current_alpha = momentum / (momentum + sigma);
            } else {
                self.current_alpha = 0.01;
            }
            // Clamp alpha to reasonable range
            self.current_alpha = self.current_alpha.max(0.01).min(0.99);
        }

        self.value = self.current_alpha * price + (1.0 - self.current_alpha) * self.value;
        self.value
    }

    pub fn get_value(&self) -> f64 { self.value }
    pub fn get_alpha(&self) -> f64 { self.current_alpha }
}

// ─── HodrickPrescottFilter ────────────────────────────────────────────────────

/// Rolling Hodrick-Prescott filter for trend extraction.
///
/// The HP filter minimizes:
///   sum((y_t - tau_t)^2) + lambda * sum((tau_{t+1} - 2*tau_t + tau_{t-1})^2)
///
/// This rolling implementation maintains a fixed-length window and solves the
/// resulting tridiagonal system each time a new observation arrives.
/// lambda = 1600 (standard quarterly), 6.25 (annual), 100 (monthly).
#[derive(Debug, Clone)]
pub struct HodrickPrescottFilter {
    buffer: VecDeque<f64>,
    window: usize,
    lambda: f64,
    /// Last estimated trend value for the most recent point.
    last_trend: f64,
}

impl HodrickPrescottFilter {
    /// Create a new rolling HP filter.
    ///
    /// - `window` -- number of bars in the rolling window (minimum 5)
    /// - `lambda` -- smoothing parameter (1600 for quarterly data)
    pub fn new(window: usize, lambda: f64) -> Self {
        assert!(window >= 5, "HP filter requires window >= 5");
        Self {
            buffer: VecDeque::with_capacity(window),
            window,
            lambda,
            last_trend: 0.0,
        }
    }

    /// Add a new observation and return the estimated trend for the latest point.
    /// Returns None until the window is full.
    pub fn update(&mut self, y: f64) -> Option<f64> {
        if self.buffer.len() == self.window {
            self.buffer.pop_front();
        }
        self.buffer.push_back(y);

        if self.buffer.len() < self.window {
            return None;
        }

        let n = self.window;
        let y_vec: Vec<f64> = self.buffer.iter().cloned().collect();
        let trend = self.solve_hp(&y_vec, n);
        self.last_trend = trend[n - 1];
        Some(self.last_trend)
    }

    /// Solve the HP filter system (A + lambda*D'D) * tau = y via band matrix.
    ///
    /// Uses the pentadiagonal banded structure of the system for O(n) solve.
    fn solve_hp(&self, y: &[f64], n: usize) -> Vec<f64> {
        let lam = self.lambda;

        // Build diagonal bands of (I + lambda * D2'*D2):
        // The second-difference matrix D2 has the property that D2'*D2 is a
        // pentadiagonal matrix. Coefficients for row i (0-indexed):
        //   main diagonal: 1 + lambda * c_ii
        //   off-diagonal +/-1: lambda * c_i1
        //   off-diagonal +/-2: lambda * c_i2
        //
        // c_ii values: [1, 5, 6, ...6, 5, 1] for i = 0,1,2..n-3,n-2,n-1
        // c_i1 values: [-2, -4, -4, ...-4, -2] for off-diag 1
        // c_i2 values: [1, 1, ..., 1] for off-diag 2
        //
        // We use Thomas algorithm extended to pentadiagonal.

        let mut diag = vec![0.0_f64; n];
        let mut d1 = vec![0.0_f64; n]; // off-diag at +/-1
        let mut d2 = vec![0.0_f64; n]; // off-diag at +/-2

        for i in 0..n {
            let c = match i {
                0 | 1 => if i == 0 { 1.0 } else { 5.0 },
                _ if i >= n - 2 => if i == n - 1 { 1.0 } else { 5.0 },
                _ => 6.0,
            };
            diag[i] = 1.0 + lam * c;
        }
        for i in 0..n - 1 {
            let c = if i == 0 || i == n - 2 { -2.0 } else { -4.0 };
            d1[i] = lam * c;
        }
        for i in 0..n - 2 {
            d2[i] = lam * 1.0;
        }

        // Gaussian elimination for banded system (bandwidth = 2)
        // Forward sweep
        let mut a = diag.clone();
        let mut b = d1.clone();
        let mut c = d2.clone();
        let mut rhs = y.to_vec();

        for i in 1..n {
            if a[i - 1].abs() < 1e-15 {
                continue;
            }
            let m1 = b[i - 1] / a[i - 1];
            a[i] -= m1 * b[i - 1];
            rhs[i] -= m1 * rhs[i - 1];
            if i < n - 1 {
                b[i] -= m1 * c[i - 1];
            }

            if i >= 2 {
                if a[i - 2].abs() < 1e-15 {
                    continue;
                }
                let m2 = c[i - 2] / a[i - 2];
                a[i] -= m2 * b[i - 2];
                rhs[i] -= m2 * rhs[i - 2];
                if i < n - 1 {
                    b[i] -= m2 * c[i - 2];
                }
            }
        }

        // Back substitution
        let mut tau = vec![0.0_f64; n];
        tau[n - 1] = rhs[n - 1] / a[n - 1];
        if n >= 2 {
            tau[n - 2] = (rhs[n - 2] - b[n - 2] * tau[n - 1]) / a[n - 2];
        }
        for i in (0..n - 2).rev() {
            tau[i] = (rhs[i] - b[i] * tau[i + 1] - c[i] * tau[i + 2]) / a[i];
        }
        tau
    }

    pub fn last_trend(&self) -> f64 { self.last_trend }
    pub fn is_ready(&self) -> bool { self.buffer.len() == self.window }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman1d_converges_to_constant() {
        let mut kf = KalmanFilter1D::new(50.0, 1e-4, 1.0);
        for _ in 0..200 {
            kf.update(50.0);
        }
        assert!((kf.get_state() - 50.0).abs() < 0.1);
    }

    #[test]
    fn lms_output_is_finite() {
        let mut lms = LMSFilter::new(8, 0.01);
        for i in 0..100 {
            let v = lms.filter(i as f64 * 0.1);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rls_output_is_finite() {
        let mut rls = RLSFilter::new(4, 0.99, 100.0);
        for i in 0..50 {
            let v = rls.filter(i as f64);
            assert!(v.is_finite());
        }
    }
}
