use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Matrix utilities (small dense matrices, row-major)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    pub fn from_scalar(val: f64) -> Self {
        Self { rows: 1, cols: 1, data: vec![val] }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::new(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.get(r, c));
            }
        }
        t
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a + b).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a - b).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| v * s).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    pub fn inverse(&self) -> Option<Self> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut aug = vec![0.0; n * 2 * n];
        for r in 0..n {
            for c in 0..n {
                aug[r * 2 * n + c] = self.get(r, c);
            }
            aug[r * 2 * n + n + r] = 1.0;
        }

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col * 2 * n + col].abs();
            for row in (col + 1)..n {
                let v = aug[row * 2 * n + col].abs();
                if v > max_val { max_val = v; max_row = row; }
            }
            if max_val < 1e-15 { return None; }
            if max_row != col {
                for c in 0..2 * n {
                    let tmp = aug[col * 2 * n + c];
                    aug[col * 2 * n + c] = aug[max_row * 2 * n + c];
                    aug[max_row * 2 * n + c] = tmp;
                }
            }
            let pivot = aug[col * 2 * n + col];
            for c in 0..2 * n { aug[col * 2 * n + c] /= pivot; }
            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row * 2 * n + col];
                for c in 0..2 * n {
                    aug[row * 2 * n + c] -= factor * aug[col * 2 * n + c];
                }
            }
        }

        let mut result = Self::new(n, n);
        for r in 0..n {
            for c in 0..n {
                result.set(r, c, aug[r * 2 * n + n + c]);
            }
        }
        Some(result)
    }

    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        let mut s = 0.0;
        for i in 0..n { s += self.get(i, i); }
        s
    }

    pub fn col_vec(data: &[f64]) -> Self {
        Self { rows: data.len(), cols: 1, data: data.to_vec() }
    }

    pub fn as_scalar(&self) -> f64 {
        assert_eq!(self.rows, 1);
        assert_eq!(self.cols, 1);
        self.data[0]
    }

    pub fn cholesky(&self) -> Option<Self> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut l = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    let val = self.get(i, i) - sum;
                    if val < 0.0 { return None; }
                    l.set(i, j, val.sqrt());
                } else {
                    let ljj = l.get(j, j);
                    if ljj.abs() < 1e-15 { return None; }
                    l.set(i, j, (self.get(i, j) - sum) / ljj);
                }
            }
        }
        Some(l)
    }
}

// ---------------------------------------------------------------------------
// Kalman Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub f_mat: Matrix,    // state transition
    pub h_mat: Matrix,    // observation
    pub q_mat: Matrix,    // process noise covariance
    pub r_mat: Matrix,    // observation noise covariance
    pub b_mat: Option<Matrix>, // control input
    pub x: Matrix,        // state estimate
    pub p: Matrix,        // error covariance
}

#[derive(Debug, Clone)]
pub struct KalmanOutput {
    pub x_prior: Matrix,
    pub p_prior: Matrix,
    pub x_post: Matrix,
    pub p_post: Matrix,
    pub innovation: Matrix,
    pub innovation_cov: Matrix,
    pub kalman_gain: Matrix,
    pub log_likelihood: f64,
}

impl KalmanFilter {
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            state_dim,
            obs_dim,
            f_mat: Matrix::identity(state_dim),
            h_mat: Matrix::zeros(obs_dim, state_dim),
            q_mat: Matrix::identity(state_dim).scale(0.01),
            r_mat: Matrix::identity(obs_dim).scale(1.0),
            b_mat: None,
            x: Matrix::zeros(state_dim, 1),
            p: Matrix::identity(state_dim),
        }
    }

    pub fn predict(&mut self, u: Option<&Matrix>) {
        // x = F * x + B * u
        self.x = self.f_mat.mul(&self.x);
        if let (Some(b), Some(u)) = (&self.b_mat, u) {
            self.x = self.x.add(&b.mul(u));
        }
        // P = F * P * F^T + Q
        self.p = self.f_mat.mul(&self.p).mul(&self.f_mat.transpose()).add(&self.q_mat);
    }

    pub fn update(&mut self, z: &Matrix) -> KalmanOutput {
        let x_prior = self.x.clone();
        let p_prior = self.p.clone();

        // Innovation: y = z - H * x
        let innovation = z.sub(&self.h_mat.mul(&self.x));

        // Innovation covariance: S = H * P * H^T + R
        let s = self.h_mat.mul(&self.p).mul(&self.h_mat.transpose()).add(&self.r_mat);

        // Kalman gain: K = P * H^T * S^{-1}
        let s_inv = s.inverse().unwrap_or(Matrix::identity(self.obs_dim));
        let k = self.p.mul(&self.h_mat.transpose()).mul(&s_inv);

        // Update state: x = x + K * y
        self.x = self.x.add(&k.mul(&innovation));

        // Update covariance: P = (I - K * H) * P
        let i_kh = Matrix::identity(self.state_dim).sub(&k.mul(&self.h_mat));
        self.p = i_kh.mul(&self.p);

        // Log-likelihood contribution
        let n = self.obs_dim as f64;
        let det_s = if self.obs_dim == 1 { s.data[0] } else { s.trace() }; // simplified
        let ll = -0.5 * (n * (2.0 * PI).ln() + det_s.abs().max(1e-30).ln()
            + innovation.transpose().mul(&s_inv).mul(&innovation).as_scalar());

        KalmanOutput {
            x_prior,
            p_prior,
            x_post: self.x.clone(),
            p_post: self.p.clone(),
            innovation,
            innovation_cov: s,
            kalman_gain: k,
            log_likelihood: ll,
        }
    }

    pub fn predict_and_update(&mut self, z: &Matrix, u: Option<&Matrix>) -> KalmanOutput {
        self.predict(u);
        self.update(z)
    }

    pub fn filter_batch(&mut self, observations: &[Vec<f64>]) -> Vec<KalmanOutput> {
        observations.iter().map(|obs| {
            let z = Matrix::col_vec(obs);
            self.predict_and_update(&z, None)
        }).collect()
    }

    pub fn log_likelihood(&mut self, observations: &[Vec<f64>]) -> f64 {
        let results = self.filter_batch(observations);
        results.iter().map(|r| r.log_likelihood).sum()
    }

    pub fn reset(&mut self) {
        self.x = Matrix::zeros(self.state_dim, 1);
        self.p = Matrix::identity(self.state_dim);
    }
}

// ---------------------------------------------------------------------------
// Kalman Smoother (RTS - Rauch-Tung-Striebel)
// ---------------------------------------------------------------------------
pub fn kalman_smoother(
    filter: &mut KalmanFilter,
    observations: &[Vec<f64>],
) -> Vec<(Matrix, Matrix)> {
    let n = observations.len();

    // Forward pass
    let mut x_filt = Vec::with_capacity(n);
    let mut p_filt = Vec::with_capacity(n);
    let mut x_pred = Vec::with_capacity(n);
    let mut p_pred = Vec::with_capacity(n);

    filter.reset();
    for obs in observations {
        let z = Matrix::col_vec(obs);
        filter.predict(None);
        x_pred.push(filter.x.clone());
        p_pred.push(filter.p.clone());
        filter.update(&z);
        x_filt.push(filter.x.clone());
        p_filt.push(filter.p.clone());
    }

    // Backward pass
    let mut x_smooth = vec![Matrix::zeros(filter.state_dim, 1); n];
    let mut p_smooth = vec![Matrix::zeros(filter.state_dim, filter.state_dim); n];

    x_smooth[n - 1] = x_filt[n - 1].clone();
    p_smooth[n - 1] = p_filt[n - 1].clone();

    for t in (0..n - 1).rev() {
        let p_pred_inv = p_pred[t + 1].inverse().unwrap_or(Matrix::identity(filter.state_dim));
        let gain = p_filt[t].mul(&filter.f_mat.transpose()).mul(&p_pred_inv);

        x_smooth[t] = x_filt[t].add(&gain.mul(&x_smooth[t + 1].sub(&x_pred[t + 1])));
        p_smooth[t] = p_filt[t].add(&gain.mul(&p_smooth[t + 1].sub(&p_pred[t + 1])).mul(&gain.transpose()));
    }

    x_smooth.into_iter().zip(p_smooth.into_iter()).collect()
}

// ---------------------------------------------------------------------------
// Extended Kalman Filter (EKF)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ExtendedKalmanFilter {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub q_mat: Matrix,
    pub r_mat: Matrix,
    pub x: Matrix,
    pub p: Matrix,
    pub f_func: fn(&Matrix) -> Matrix,       // state transition function
    pub h_func: fn(&Matrix) -> Matrix,       // observation function
    pub f_jacobian: fn(&Matrix) -> Matrix,   // Jacobian of f
    pub h_jacobian: fn(&Matrix) -> Matrix,   // Jacobian of h
}

impl ExtendedKalmanFilter {
    pub fn new(
        state_dim: usize, obs_dim: usize,
        f_func: fn(&Matrix) -> Matrix,
        h_func: fn(&Matrix) -> Matrix,
        f_jac: fn(&Matrix) -> Matrix,
        h_jac: fn(&Matrix) -> Matrix,
    ) -> Self {
        Self {
            state_dim,
            obs_dim,
            q_mat: Matrix::identity(state_dim).scale(0.01),
            r_mat: Matrix::identity(obs_dim).scale(1.0),
            x: Matrix::zeros(state_dim, 1),
            p: Matrix::identity(state_dim),
            f_func,
            h_func,
            f_jacobian: f_jac,
            h_jacobian: h_jac,
        }
    }

    pub fn predict(&mut self) {
        let f_jac = (self.f_jacobian)(&self.x);
        self.x = (self.f_func)(&self.x);
        self.p = f_jac.mul(&self.p).mul(&f_jac.transpose()).add(&self.q_mat);
    }

    pub fn update(&mut self, z: &Matrix) -> Matrix {
        let h_jac = (self.h_jacobian)(&self.x);
        let z_pred = (self.h_func)(&self.x);
        let innovation = z.sub(&z_pred);

        let s = h_jac.mul(&self.p).mul(&h_jac.transpose()).add(&self.r_mat);
        let s_inv = s.inverse().unwrap_or(Matrix::identity(self.obs_dim));
        let k = self.p.mul(&h_jac.transpose()).mul(&s_inv);

        self.x = self.x.add(&k.mul(&innovation));
        let i_kh = Matrix::identity(self.state_dim).sub(&k.mul(&h_jac));
        self.p = i_kh.mul(&self.p);

        innovation
    }

    pub fn predict_and_update(&mut self, z: &Matrix) -> Matrix {
        self.predict();
        self.update(z)
    }
}

// ---------------------------------------------------------------------------
// Unscented Kalman Filter (UKF)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct UnscentedKalmanFilter {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub q_mat: Matrix,
    pub r_mat: Matrix,
    pub x: Matrix,
    pub p: Matrix,
    pub alpha: f64,
    pub beta: f64,
    pub kappa: f64,
    pub f_func: fn(&Matrix) -> Matrix,
    pub h_func: fn(&Matrix) -> Matrix,
}

impl UnscentedKalmanFilter {
    pub fn new(
        state_dim: usize, obs_dim: usize,
        f_func: fn(&Matrix) -> Matrix,
        h_func: fn(&Matrix) -> Matrix,
    ) -> Self {
        Self {
            state_dim,
            obs_dim,
            q_mat: Matrix::identity(state_dim).scale(0.01),
            r_mat: Matrix::identity(obs_dim).scale(1.0),
            x: Matrix::zeros(state_dim, 1),
            p: Matrix::identity(state_dim),
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
            f_func,
            h_func,
        }
    }

    fn compute_sigma_points(&self) -> (Vec<Matrix>, Vec<f64>, Vec<f64>) {
        let n = self.state_dim;
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;
        let num_sigma = 2 * n + 1;

        let mut sigma_points = Vec::with_capacity(num_sigma);
        let mut w_mean = vec![0.0; num_sigma];
        let mut w_cov = vec![0.0; num_sigma];

        // Mean weights
        w_mean[0] = lambda / (n as f64 + lambda);
        w_cov[0] = lambda / (n as f64 + lambda) + (1.0 - self.alpha * self.alpha + self.beta);
        for i in 1..num_sigma {
            w_mean[i] = 1.0 / (2.0 * (n as f64 + lambda));
            w_cov[i] = 1.0 / (2.0 * (n as f64 + lambda));
        }

        // Square root of (n + lambda) * P
        let scaled_p = self.p.scale(n as f64 + lambda);
        let sqrt_p = scaled_p.cholesky().unwrap_or(Matrix::identity(n));

        // Sigma point 0: mean
        sigma_points.push(self.x.clone());

        // Sigma points 1..n: x + column of sqrt_P
        for i in 0..n {
            let mut col = vec![0.0; n];
            for r in 0..n { col[r] = sqrt_p.get(r, i); }
            let offset = Matrix::col_vec(&col);
            sigma_points.push(self.x.add(&offset));
        }

        // Sigma points n+1..2n: x - column of sqrt_P
        for i in 0..n {
            let mut col = vec![0.0; n];
            for r in 0..n { col[r] = sqrt_p.get(r, i); }
            let offset = Matrix::col_vec(&col);
            sigma_points.push(self.x.sub(&offset));
        }

        (sigma_points, w_mean, w_cov)
    }

    pub fn predict(&mut self) {
        let n = self.state_dim;
        let (sigma_points, w_mean, w_cov) = self.compute_sigma_points();

        // Transform sigma points through f
        let transformed: Vec<Matrix> = sigma_points.iter().map(|sp| (self.f_func)(sp)).collect();

        // Weighted mean
        let mut x_pred = Matrix::zeros(n, 1);
        for (i, t) in transformed.iter().enumerate() {
            x_pred = x_pred.add(&t.scale(w_mean[i]));
        }

        // Weighted covariance
        let mut p_pred = self.q_mat.clone();
        for (i, t) in transformed.iter().enumerate() {
            let diff = t.sub(&x_pred);
            p_pred = p_pred.add(&diff.mul(&diff.transpose()).scale(w_cov[i]));
        }

        self.x = x_pred;
        self.p = p_pred;
    }

    pub fn update(&mut self, z: &Matrix) {
        let n = self.state_dim;
        let m = self.obs_dim;
        let (sigma_points, w_mean, w_cov) = self.compute_sigma_points();

        // Transform through h
        let z_sigma: Vec<Matrix> = sigma_points.iter().map(|sp| (self.h_func)(sp)).collect();

        // Predicted observation mean
        let mut z_pred = Matrix::zeros(m, 1);
        for (i, zs) in z_sigma.iter().enumerate() {
            z_pred = z_pred.add(&zs.scale(w_mean[i]));
        }

        // Innovation covariance S = sum(w * (z_i - z_pred)(z_i - z_pred)^T) + R
        let mut s = self.r_mat.clone();
        for (i, zs) in z_sigma.iter().enumerate() {
            let dz = zs.sub(&z_pred);
            s = s.add(&dz.mul(&dz.transpose()).scale(w_cov[i]));
        }

        // Cross-covariance Pxz = sum(w * (x_i - x_pred)(z_i - z_pred)^T)
        let f_sigma: Vec<Matrix> = sigma_points.iter().map(|sp| (self.f_func)(sp)).collect();
        let mut pxz = Matrix::zeros(n, m);
        for (i, (xs, zs)) in f_sigma.iter().zip(z_sigma.iter()).enumerate() {
            let dx = xs.sub(&self.x);
            let dz = zs.sub(&z_pred);
            pxz = pxz.add(&dx.mul(&dz.transpose()).scale(w_cov[i]));
        }

        // Kalman gain: K = Pxz * S^{-1}
        let s_inv = s.inverse().unwrap_or(Matrix::identity(m));
        let k = pxz.mul(&s_inv);

        // Update
        let innovation = z.sub(&z_pred);
        self.x = self.x.add(&k.mul(&innovation));
        self.p = self.p.sub(&k.mul(&s).mul(&k.transpose()));
    }

    pub fn predict_and_update(&mut self, z: &Matrix) {
        self.predict();
        self.update(z);
    }
}

// ---------------------------------------------------------------------------
// Square-Root Kalman Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct SquareRootKalmanFilter {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub f_mat: Matrix,
    pub h_mat: Matrix,
    pub sq_q: Matrix,  // Cholesky of Q
    pub sq_r: Matrix,  // Cholesky of R
    pub x: Matrix,
    pub sq_p: Matrix,  // Cholesky factor of P
}

impl SquareRootKalmanFilter {
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            state_dim,
            obs_dim,
            f_mat: Matrix::identity(state_dim),
            h_mat: Matrix::zeros(obs_dim, state_dim),
            sq_q: Matrix::identity(state_dim).scale(0.1),
            sq_r: Matrix::identity(obs_dim),
            x: Matrix::zeros(state_dim, 1),
            sq_p: Matrix::identity(state_dim),
        }
    }

    pub fn predict(&mut self) {
        self.x = self.f_mat.mul(&self.x);
        // S_p = chol(F * S * S^T * F^T + Q)
        let p = self.f_mat.mul(&self.sq_p).mul(&self.sq_p.transpose()).mul(&self.f_mat.transpose())
            .add(&self.sq_q.mul(&self.sq_q.transpose()));
        self.sq_p = p.cholesky().unwrap_or(Matrix::identity(self.state_dim));
    }

    pub fn update(&mut self, z: &Matrix) {
        let p = self.sq_p.mul(&self.sq_p.transpose());
        let innovation = z.sub(&self.h_mat.mul(&self.x));
        let s = self.h_mat.mul(&p).mul(&self.h_mat.transpose())
            .add(&self.sq_r.mul(&self.sq_r.transpose()));
        let s_inv = s.inverse().unwrap_or(Matrix::identity(self.obs_dim));
        let k = p.mul(&self.h_mat.transpose()).mul(&s_inv);

        self.x = self.x.add(&k.mul(&innovation));
        let i_kh = Matrix::identity(self.state_dim).sub(&k.mul(&self.h_mat));
        let p_new = i_kh.mul(&p);
        self.sq_p = p_new.cholesky().unwrap_or(Matrix::identity(self.state_dim));
    }
}

// ---------------------------------------------------------------------------
// Information Filter (inverse covariance form)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct InformationFilter {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub f_mat: Matrix,
    pub h_mat: Matrix,
    pub q_mat: Matrix,
    pub r_mat: Matrix,
    pub info_state: Matrix,   // P^{-1} * x
    pub info_matrix: Matrix,  // P^{-1}
}

impl InformationFilter {
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        Self {
            state_dim,
            obs_dim,
            f_mat: Matrix::identity(state_dim),
            h_mat: Matrix::zeros(obs_dim, state_dim),
            q_mat: Matrix::identity(state_dim).scale(0.01),
            r_mat: Matrix::identity(obs_dim),
            info_state: Matrix::zeros(state_dim, 1),
            info_matrix: Matrix::identity(state_dim).scale(0.01), // small initial info
        }
    }

    pub fn predict(&mut self) {
        // Convert to state space for prediction
        let p = self.info_matrix.inverse().unwrap_or(Matrix::identity(self.state_dim).scale(100.0));
        let x = p.mul(&self.info_state);

        let x_pred = self.f_mat.mul(&x);
        let p_pred = self.f_mat.mul(&p).mul(&self.f_mat.transpose()).add(&self.q_mat);

        self.info_matrix = p_pred.inverse().unwrap_or(Matrix::identity(self.state_dim).scale(0.01));
        self.info_state = self.info_matrix.mul(&x_pred);
    }

    pub fn update(&mut self, z: &Matrix) {
        let r_inv = self.r_mat.inverse().unwrap_or(Matrix::identity(self.obs_dim));
        let ht_rinv = self.h_mat.transpose().mul(&r_inv);

        // Information update
        self.info_matrix = self.info_matrix.add(&ht_rinv.mul(&self.h_mat));
        self.info_state = self.info_state.add(&ht_rinv.mul(z));
    }

    pub fn state_estimate(&self) -> Matrix {
        let p = self.info_matrix.inverse().unwrap_or(Matrix::identity(self.state_dim));
        p.mul(&self.info_state)
    }

    pub fn covariance(&self) -> Matrix {
        self.info_matrix.inverse().unwrap_or(Matrix::identity(self.state_dim))
    }
}

// ---------------------------------------------------------------------------
// EM for State Space Parameter Estimation
// ---------------------------------------------------------------------------
pub struct StateSpaceEM;

impl StateSpaceEM {
    /// Estimate parameters F, H, Q, R via EM iterations.
    pub fn fit(
        observations: &[Vec<f64>],
        state_dim: usize,
        obs_dim: usize,
        max_iter: usize,
    ) -> KalmanFilter {
        let n = observations.len();
        let mut kf = KalmanFilter::new(state_dim, obs_dim);

        // Initialize H as [I 0...] if obs_dim <= state_dim
        for i in 0..obs_dim.min(state_dim) {
            kf.h_mat.set(i, i, 1.0);
        }

        for _iter in 0..max_iter {
            // E-step: run Kalman smoother
            let smoothed = kalman_smoother(&mut kf, observations);

            // M-step: update parameters
            // Update R
            let mut r_new = Matrix::zeros(obs_dim, obs_dim);
            for t in 0..n {
                let z = Matrix::col_vec(&observations[t]);
                let (x_s, p_s) = &smoothed[t];
                let innovation = z.sub(&kf.h_mat.mul(x_s));
                let contrib = innovation.mul(&innovation.transpose())
                    .add(&kf.h_mat.mul(p_s).mul(&kf.h_mat.transpose()));
                r_new = r_new.add(&contrib);
            }
            kf.r_mat = r_new.scale(1.0 / n as f64);
            // Ensure positive diagonal
            for i in 0..obs_dim {
                if kf.r_mat.get(i, i) < 1e-10 {
                    kf.r_mat.set(i, i, 1e-10);
                }
            }

            // Update Q
            let mut q_new = Matrix::zeros(state_dim, state_dim);
            for t in 1..n {
                let (x_t, p_t) = &smoothed[t];
                let (x_prev, p_prev) = &smoothed[t - 1];
                let diff = x_t.sub(&kf.f_mat.mul(x_prev));
                let contrib = diff.mul(&diff.transpose()).add(p_t)
                    .add(&kf.f_mat.mul(p_prev).mul(&kf.f_mat.transpose()).scale(-1.0))
                    .add(p_t);
                q_new = q_new.add(&contrib.scale(1.0 / (n - 1) as f64));
            }
            kf.q_mat = q_new;
            for i in 0..state_dim {
                if kf.q_mat.get(i, i) < 1e-10 {
                    kf.q_mat.set(i, i, 1e-10);
                }
            }

            kf.reset();
        }

        kf
    }
}

// ---------------------------------------------------------------------------
// Structural Time Series Models
// ---------------------------------------------------------------------------

/// Local Level Model: y_t = mu_t + eps_t, mu_t = mu_{t-1} + eta_t
pub fn local_level_model(observations: &[f64], sigma2_eps: f64, sigma2_eta: f64) -> Vec<f64> {
    let n = observations.len();
    let mut kf = KalmanFilter::new(1, 1);
    kf.f_mat = Matrix::from_vec(1, 1, vec![1.0]);
    kf.h_mat = Matrix::from_vec(1, 1, vec![1.0]);
    kf.q_mat = Matrix::from_vec(1, 1, vec![sigma2_eta]);
    kf.r_mat = Matrix::from_vec(1, 1, vec![sigma2_eps]);
    kf.x = Matrix::from_vec(1, 1, vec![observations[0]]);
    kf.p = Matrix::from_vec(1, 1, vec![1.0]);

    let obs_vecs: Vec<Vec<f64>> = observations.iter().map(|&o| vec![o]).collect();
    let smoothed = kalman_smoother(&mut kf, &obs_vecs);
    smoothed.iter().map(|(x, _)| x.data[0]).collect()
}

/// Local Linear Trend: y_t = mu_t + eps_t, mu_t = mu_{t-1} + nu_{t-1} + eta_t, nu_t = nu_{t-1} + zeta_t
pub fn local_linear_trend(observations: &[f64], sigma2_eps: f64, sigma2_eta: f64, sigma2_zeta: f64) -> (Vec<f64>, Vec<f64>) {
    let n = observations.len();
    let mut kf = KalmanFilter::new(2, 1);
    kf.f_mat = Matrix::from_vec(2, 2, vec![1.0, 1.0, 0.0, 1.0]);
    kf.h_mat = Matrix::from_vec(1, 2, vec![1.0, 0.0]);
    kf.q_mat = Matrix::from_vec(2, 2, vec![sigma2_eta, 0.0, 0.0, sigma2_zeta]);
    kf.r_mat = Matrix::from_vec(1, 1, vec![sigma2_eps]);
    kf.x = Matrix::from_vec(2, 1, vec![observations[0], 0.0]);

    let obs_vecs: Vec<Vec<f64>> = observations.iter().map(|&o| vec![o]).collect();
    let smoothed = kalman_smoother(&mut kf, &obs_vecs);
    let level: Vec<f64> = smoothed.iter().map(|(x, _)| x.data[0]).collect();
    let trend: Vec<f64> = smoothed.iter().map(|(x, _)| x.data[1]).collect();
    (level, trend)
}

/// Seasonal component via state space (trigonometric form)
pub fn seasonal_model(
    observations: &[f64],
    period: usize,
    sigma2_eps: f64,
    sigma2_season: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n_harmonics = period / 2;
    let state_dim = 1 + 2 * n_harmonics; // level + seasonal harmonics
    let mut kf = KalmanFilter::new(state_dim, 1);

    // F matrix: identity for level, rotation for each harmonic
    let mut f = Matrix::identity(state_dim);
    for j in 0..n_harmonics {
        let freq = 2.0 * PI * (j + 1) as f64 / period as f64;
        let idx = 1 + 2 * j;
        f.set(idx, idx, freq.cos());
        f.set(idx, idx + 1, freq.sin());
        f.set(idx + 1, idx, -freq.sin());
        f.set(idx + 1, idx + 1, freq.cos());
    }
    kf.f_mat = f;

    // H matrix: observe level + sum of cosine components
    let mut h = Matrix::zeros(1, state_dim);
    h.set(0, 0, 1.0);
    for j in 0..n_harmonics {
        h.set(0, 1 + 2 * j, 1.0);
    }
    kf.h_mat = h;

    kf.q_mat = Matrix::identity(state_dim).scale(sigma2_season);
    kf.q_mat.set(0, 0, 0.01); // level noise
    kf.r_mat = Matrix::from_vec(1, 1, vec![sigma2_eps]);
    kf.x.data[0] = observations[0];

    let obs_vecs: Vec<Vec<f64>> = observations.iter().map(|&o| vec![o]).collect();
    let smoothed = kalman_smoother(&mut kf, &obs_vecs);

    let level: Vec<f64> = smoothed.iter().map(|(x, _)| x.data[0]).collect();
    let seasonal: Vec<f64> = smoothed.iter().map(|(x, _)| {
        let mut s = 0.0;
        for j in 0..n_harmonics {
            s += x.data[1 + 2 * j];
        }
        s
    }).collect();

    (level, seasonal)
}

// ---------------------------------------------------------------------------
// Missing Data Handling
// ---------------------------------------------------------------------------
pub fn kalman_filter_missing(
    kf: &mut KalmanFilter,
    observations: &[Option<Vec<f64>>],
) -> Vec<(Matrix, Matrix)> {
    let mut results = Vec::with_capacity(observations.len());

    for obs in observations {
        kf.predict(None);
        match obs {
            Some(z) => {
                let z_mat = Matrix::col_vec(z);
                kf.update(&z_mat);
            }
            None => {
                // No update — prediction only
            }
        }
        results.push((kf.x.clone(), kf.p.clone()));
    }
    results
}

// ---------------------------------------------------------------------------
// Scalar Kalman convenience
// ---------------------------------------------------------------------------
pub fn scalar_kalman_filter(data: &[f64], q: f64, r: f64) -> Vec<f64> {
    let n = data.len();
    let mut x = data[0];
    let mut p = 1.0;
    let mut result = Vec::with_capacity(n);
    result.push(x);

    for t in 1..n {
        // Predict
        let x_pred = x;
        let p_pred = p + q;

        // Update
        let k = p_pred / (p_pred + r);
        x = x_pred + k * (data[t] - x_pred);
        p = (1.0 - k) * p_pred;

        result.push(x);
    }
    result
}

pub fn scalar_kalman_smoother(data: &[f64], q: f64, r: f64) -> Vec<f64> {
    let n = data.len();

    // Forward
    let mut x_filt = vec![0.0; n];
    let mut p_filt = vec![0.0; n];
    let mut x_pred_arr = vec![0.0; n];
    let mut p_pred_arr = vec![0.0; n];

    x_filt[0] = data[0];
    p_filt[0] = 1.0;

    for t in 1..n {
        x_pred_arr[t] = x_filt[t - 1];
        p_pred_arr[t] = p_filt[t - 1] + q;
        let k = p_pred_arr[t] / (p_pred_arr[t] + r);
        x_filt[t] = x_pred_arr[t] + k * (data[t] - x_pred_arr[t]);
        p_filt[t] = (1.0 - k) * p_pred_arr[t];
    }

    // Backward (RTS)
    let mut x_smooth = vec![0.0; n];
    x_smooth[n - 1] = x_filt[n - 1];

    for t in (0..n - 1).rev() {
        let gain = if p_pred_arr[t + 1].abs() > 1e-15 {
            p_filt[t] / p_pred_arr[t + 1]
        } else {
            0.0
        };
        x_smooth[t] = x_filt[t] + gain * (x_smooth[t + 1] - x_pred_arr[t + 1]);
    }

    x_smooth
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_mul() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.mul(&b);
        assert!((c.get(0, 0) - 19.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 22.0).abs() < 1e-10);
        assert!((c.get(1, 0) - 43.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = Matrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = a.inverse().unwrap();
        let product = a.mul(&inv);
        assert!((product.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((product.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((product.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        let a = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]);
        let l = a.cholesky().unwrap();
        let reconstruct = l.mul(&l.transpose());
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstruct.get(i, j) - a.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_scalar_kalman() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 + ((i as f64 * 0.3).sin() * 3.0)).collect();
        let filtered = scalar_kalman_filter(&data, 0.1, 1.0);
        assert_eq!(filtered.len(), 50);
    }

    #[test]
    fn test_scalar_kalman_smoother() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 + ((i as f64 * 0.3).sin() * 3.0)).collect();
        let smoothed = scalar_kalman_smoother(&data, 0.1, 1.0);
        assert_eq!(smoothed.len(), 50);
    }

    #[test]
    fn test_kalman_filter() {
        let mut kf = KalmanFilter::new(2, 1);
        kf.f_mat = Matrix::from_vec(2, 2, vec![1.0, 1.0, 0.0, 1.0]);
        kf.h_mat = Matrix::from_vec(1, 2, vec![1.0, 0.0]);
        kf.q_mat = Matrix::identity(2).scale(0.01);
        kf.r_mat = Matrix::from_vec(1, 1, vec![1.0]);

        let observations: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 + 0.5]).collect();
        let results = kf.filter_batch(&observations);
        assert_eq!(results.len(), 20);
    }

    #[test]
    fn test_kalman_smoother() {
        let mut kf = KalmanFilter::new(1, 1);
        kf.f_mat = Matrix::from_vec(1, 1, vec![1.0]);
        kf.h_mat = Matrix::from_vec(1, 1, vec![1.0]);
        kf.q_mat = Matrix::from_vec(1, 1, vec![0.1]);
        kf.r_mat = Matrix::from_vec(1, 1, vec![1.0]);

        let observations: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let smoothed = kalman_smoother(&mut kf, &observations);
        assert_eq!(smoothed.len(), 20);
    }

    #[test]
    fn test_local_level() {
        let data: Vec<f64> = (0..50).map(|i| 10.0 + ((i as f64 * 0.2).sin() * 2.0)).collect();
        let level = local_level_model(&data, 1.0, 0.01);
        assert_eq!(level.len(), 50);
    }

    #[test]
    fn test_local_linear_trend() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 * 0.5 + ((i as f64 * 0.3).sin())).collect();
        let (level, trend) = local_linear_trend(&data, 1.0, 0.01, 0.001);
        assert_eq!(level.len(), 50);
        assert_eq!(trend.len(), 50);
    }

    #[test]
    fn test_information_filter() {
        let mut inf = InformationFilter::new(1, 1);
        inf.f_mat = Matrix::from_vec(1, 1, vec![1.0]);
        inf.h_mat = Matrix::from_vec(1, 1, vec![1.0]);
        inf.q_mat = Matrix::from_vec(1, 1, vec![0.1]);
        inf.r_mat = Matrix::from_vec(1, 1, vec![1.0]);

        for i in 0..10 {
            inf.predict();
            let z = Matrix::from_vec(1, 1, vec![i as f64]);
            inf.update(&z);
        }
        let x = inf.state_estimate();
        assert!(x.data[0] > 0.0);
    }

    #[test]
    fn test_missing_data() {
        let mut kf = KalmanFilter::new(1, 1);
        kf.f_mat = Matrix::from_vec(1, 1, vec![1.0]);
        kf.h_mat = Matrix::from_vec(1, 1, vec![1.0]);
        kf.q_mat = Matrix::from_vec(1, 1, vec![0.1]);
        kf.r_mat = Matrix::from_vec(1, 1, vec![1.0]);

        let obs: Vec<Option<Vec<f64>>> = vec![
            Some(vec![1.0]), Some(vec![2.0]), None, Some(vec![4.0]), None, Some(vec![6.0]),
        ];
        let results = kalman_filter_missing(&mut kf, &obs);
        assert_eq!(results.len(), 6);
    }

    #[test]
    fn test_square_root_kf() {
        let mut kf = SquareRootKalmanFilter::new(1, 1);
        kf.f_mat = Matrix::from_vec(1, 1, vec![1.0]);
        kf.h_mat = Matrix::from_vec(1, 1, vec![1.0]);

        for i in 0..10 {
            kf.predict();
            let z = Matrix::from_vec(1, 1, vec![i as f64]);
            kf.update(&z);
        }
    }
}
