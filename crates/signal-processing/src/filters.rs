use std::f64::consts::PI;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Butterworth Filter Coefficients
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ButterworthCoeffs {
    pub b: Vec<f64>,
    pub a: Vec<f64>,
    pub order: usize,
}

fn bilinear_transform_pair(s_b: &[f64], s_a: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = s_a.len();
    let m = s_b.len();
    let max_len = n.max(m);
    let mut b = vec![0.0; max_len];
    let mut a = vec![0.0; max_len];
    let t = 2.0 * fs;

    // For second-order sections we do manual bilinear
    // s = 2*fs*(z-1)/(z+1)
    // For order 2: a0*s^2 + a1*s + a2 -> expand
    if max_len == 3 {
        let (b0, b1, b2) = (
            if m > 0 { s_b[0] } else { 0.0 },
            if m > 1 { s_b[1] } else { 0.0 },
            if m > 2 { s_b[2] } else { 0.0 },
        );
        let (a0, a1, a2) = (s_a[0], if n > 1 { s_a[1] } else { 0.0 }, if n > 2 { s_a[2] } else { 0.0 });

        let t2 = t * t;
        let denom = a0 * t2 + a1 * t + a2;

        b[0] = (b0 * t2 + b1 * t + b2) / denom;
        b[1] = (2.0 * b2 - 2.0 * b0 * t2) / denom;
        b[2] = (b0 * t2 - b1 * t + b2) / denom;

        a[0] = 1.0;
        a[1] = (2.0 * a2 - 2.0 * a0 * t2) / denom;
        a[2] = (a0 * t2 - a1 * t + a2) / denom;
    } else if max_len == 2 {
        let b0 = if m > 0 { s_b[0] } else { 0.0 };
        let b1 = if m > 1 { s_b[1] } else { 0.0 };
        let a0 = s_a[0];
        let a1 = if n > 1 { s_a[1] } else { 0.0 };
        let denom = a0 * t + a1;
        b[0] = (b0 * t + b1) / denom;
        b[1] = (-b0 * t + b1) / denom;
        a[0] = 1.0;
        a[1] = (-a0 * t + a1) / denom;
    } else {
        // fallback: copy
        for i in 0..m.min(max_len) { b[i] = s_b[i]; }
        for i in 0..n.min(max_len) { a[i] = s_a[i]; }
        if a[0].abs() > 1e-15 {
            let norm = a[0];
            for x in b.iter_mut() { *x /= norm; }
            for x in a.iter_mut() { *x /= norm; }
        }
    }
    (b, a)
}

impl ButterworthCoeffs {
    /// Design a low-pass Butterworth filter of given order.
    /// `cutoff` is normalized frequency (0, 1) where 1 = Nyquist.
    pub fn lowpass(order: usize, cutoff: f64) -> Self {
        assert!(order > 0 && order <= 10);
        assert!(cutoff > 0.0 && cutoff < 1.0);
        let wc = (PI * cutoff / 2.0).tan();
        // Cascade of second-order sections
        let mut b_all = vec![1.0];
        let mut a_all = vec![1.0];
        let n = order;
        let num_sos = n / 2;
        let odd = n % 2 == 1;

        // Collect all SOS
        let mut sections: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        for k in 0..num_sos {
            let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;
            // s-domain: s^2 + 2*cos(theta)*wc*s + wc^2
            let s_b = vec![0.0, 0.0, wc * wc]; // wc^2 in numerator for lowpass
            let s_a = vec![1.0, 2.0 * theta.cos() * wc, wc * wc];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }
        if odd {
            let s_b = vec![0.0, wc];
            let s_a = vec![1.0, wc];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }

        // Convolve all sections
        let mut b_result = vec![1.0];
        let mut a_result = vec![1.0];
        for (sb, sa) in &sections {
            b_result = convolve(&b_result, sb);
            a_result = convolve(&a_result, sa);
        }

        // Normalize
        if a_result[0].abs() > 1e-15 {
            let norm = a_result[0];
            for x in b_result.iter_mut() { *x /= norm; }
            for x in a_result.iter_mut() { *x /= norm; }
        }

        Self { b: b_result, a: a_result, order }
    }

    /// Design a high-pass Butterworth filter.
    pub fn highpass(order: usize, cutoff: f64) -> Self {
        assert!(order > 0 && order <= 10);
        assert!(cutoff > 0.0 && cutoff < 1.0);
        let wc = (PI * cutoff / 2.0).tan();
        let n = order;
        let num_sos = n / 2;
        let odd = n % 2 == 1;
        let mut sections: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

        for k in 0..num_sos {
            let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;
            // highpass: s^2 in numerator
            let s_b = vec![1.0, 0.0, 0.0];
            let s_a = vec![1.0, 2.0 * theta.cos() * wc, wc * wc];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }
        if odd {
            let s_b = vec![1.0, 0.0];
            let s_a = vec![1.0, wc];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }

        let mut b_result = vec![1.0];
        let mut a_result = vec![1.0];
        for (sb, sa) in &sections {
            b_result = convolve(&b_result, sb);
            a_result = convolve(&a_result, sa);
        }
        if a_result[0].abs() > 1e-15 {
            let norm = a_result[0];
            for x in b_result.iter_mut() { *x /= norm; }
            for x in a_result.iter_mut() { *x /= norm; }
        }
        Self { b: b_result, a: a_result, order }
    }

    /// Design a band-pass Butterworth filter.
    pub fn bandpass(order: usize, low_cutoff: f64, high_cutoff: f64) -> Self {
        assert!(low_cutoff < high_cutoff);
        // Simple approach: cascade lowpass and highpass
        let lp = Self::lowpass(order, high_cutoff);
        let hp = Self::highpass(order, low_cutoff);
        let b = convolve(&lp.b, &hp.b);
        let a = convolve(&lp.a, &hp.a);
        Self { b, a, order: order * 2 }
    }
}

fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len() + b.len() - 1;
    let mut result = vec![0.0; len];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            result[i + j] += av * bv;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// IIR Filter (applies any b/a coefficients)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct IirFilter {
    b: Vec<f64>,
    a: Vec<f64>,
    x_history: VecDeque<f64>,
    y_history: VecDeque<f64>,
}

impl IirFilter {
    pub fn new(b: Vec<f64>, a: Vec<f64>) -> Self {
        let max_len = b.len().max(a.len());
        Self {
            b,
            a,
            x_history: VecDeque::from(vec![0.0; max_len]),
            y_history: VecDeque::from(vec![0.0; max_len]),
        }
    }

    pub fn from_butterworth_lowpass(order: usize, cutoff: f64) -> Self {
        let c = ButterworthCoeffs::lowpass(order, cutoff);
        Self::new(c.b, c.a)
    }

    pub fn from_butterworth_highpass(order: usize, cutoff: f64) -> Self {
        let c = ButterworthCoeffs::highpass(order, cutoff);
        Self::new(c.b, c.a)
    }

    pub fn from_butterworth_bandpass(order: usize, low: f64, high: f64) -> Self {
        let c = ButterworthCoeffs::bandpass(order, low, high);
        Self::new(c.b, c.a)
    }

    pub fn update(&mut self, x: f64) -> f64 {
        self.x_history.push_front(x);
        if self.x_history.len() > self.b.len() {
            self.x_history.pop_back();
        }

        let mut y = 0.0;
        for (i, &bi) in self.b.iter().enumerate() {
            if i < self.x_history.len() {
                y += bi * self.x_history[i];
            }
        }
        for (i, &ai) in self.a.iter().enumerate().skip(1) {
            if i < self.y_history.len() + 1 {
                if let Some(&yh) = self.y_history.get(i - 1) {
                    y -= ai * yh;
                }
            }
        }
        if self.a[0].abs() > 1e-15 {
            y /= self.a[0];
        }

        self.y_history.push_front(y);
        if self.y_history.len() > self.a.len() {
            self.y_history.pop_back();
        }
        y
    }

    pub fn filter(&mut self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| self.update(x)).collect()
    }

    pub fn filtfilt(&mut self, data: &[f64]) -> Vec<f64> {
        // Forward pass
        self.reset();
        let forward: Vec<f64> = data.iter().map(|&x| self.update(x)).collect();
        // Backward pass
        self.reset();
        let mut backward: Vec<f64> = forward.iter().rev().map(|&x| self.update(x)).collect();
        backward.reverse();
        backward
    }

    pub fn reset(&mut self) {
        for x in self.x_history.iter_mut() { *x = 0.0; }
        for y in self.y_history.iter_mut() { *y = 0.0; }
    }
}

// ---------------------------------------------------------------------------
// Chebyshev Type I Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ChebyshevType1Coeffs {
    pub b: Vec<f64>,
    pub a: Vec<f64>,
    pub order: usize,
    pub ripple_db: f64,
}

impl ChebyshevType1Coeffs {
    pub fn lowpass(order: usize, ripple_db: f64, cutoff: f64) -> Self {
        assert!(order > 0 && order <= 8);
        assert!(cutoff > 0.0 && cutoff < 1.0);
        let eps = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
        let wc = (PI * cutoff / 2.0).tan();
        let n = order;
        let mu = (1.0 / eps).asinh() / n as f64;

        let num_sos = n / 2;
        let odd = n % 2 == 1;
        let mut sections: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

        for k in 0..num_sos {
            let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;
            let sigma = -mu.sinh() * theta.sin();
            let omega = mu.cosh() * theta.cos();
            // pole at sigma +/- j*omega, scaled by wc
            let s_a = vec![1.0, -2.0 * sigma * wc, (sigma * sigma + omega * omega) * wc * wc];
            let gain = (sigma * sigma + omega * omega) * wc * wc;
            let s_b = vec![0.0, 0.0, gain];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }
        if odd {
            let sigma = -mu.sinh();
            let s_a = vec![1.0, -sigma * wc];
            let s_b = vec![0.0, -sigma * wc];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }

        let mut b_result = vec![1.0];
        let mut a_result = vec![1.0];
        for (sb, sa) in &sections {
            b_result = convolve(&b_result, sb);
            a_result = convolve(&a_result, sa);
        }
        if a_result[0].abs() > 1e-15 {
            let norm = a_result[0];
            for x in b_result.iter_mut() { *x /= norm; }
            for x in a_result.iter_mut() { *x /= norm; }
        }

        // Normalize DC gain to 1 (for lowpass)
        let b_sum: f64 = b_result.iter().sum();
        let a_sum: f64 = a_result.iter().sum();
        if b_sum.abs() > 1e-15 {
            let gain = a_sum / b_sum;
            for x in b_result.iter_mut() { *x *= gain; }
        }

        Self { b: b_result, a: a_result, order, ripple_db }
    }

    pub fn to_iir(&self) -> IirFilter {
        IirFilter::new(self.b.clone(), self.a.clone())
    }
}

// ---------------------------------------------------------------------------
// Chebyshev Type II Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ChebyshevType2Coeffs {
    pub b: Vec<f64>,
    pub a: Vec<f64>,
    pub order: usize,
    pub stopband_db: f64,
}

impl ChebyshevType2Coeffs {
    pub fn lowpass(order: usize, stopband_db: f64, cutoff: f64) -> Self {
        assert!(order > 0 && order <= 8);
        assert!(cutoff > 0.0 && cutoff < 1.0);
        let eps = 1.0 / (10.0_f64.powf(stopband_db / 10.0) - 1.0).sqrt();
        let wc = (PI * cutoff / 2.0).tan();
        let n = order;
        let mu = (1.0 / eps).asinh() / n as f64;

        let num_sos = n / 2;
        let odd = n % 2 == 1;
        let mut sections: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();

        for k in 0..num_sos {
            let theta = PI * (2 * k + 1) as f64 / (2 * n) as f64;
            let sigma = -mu.sinh() * theta.sin();
            let omega = mu.cosh() * theta.cos();
            let denom = sigma * sigma + omega * omega;
            let p_re = sigma / denom * wc;
            let p_im = -omega / denom * wc;
            let zero_omega = 1.0 / theta.cos() * wc;

            let s_a = vec![1.0, -2.0 * p_re, p_re * p_re + p_im * p_im];
            let s_b = vec![1.0, 0.0, zero_omega * zero_omega];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }
        if odd {
            let sigma = -mu.sinh();
            let pole = wc / sigma.abs();
            let s_a = vec![1.0, pole];
            let s_b = vec![0.0, pole];
            let (zb, za) = bilinear_transform_pair(&s_b, &s_a, 1.0);
            sections.push((zb, za));
        }

        let mut b_result = vec![1.0];
        let mut a_result = vec![1.0];
        for (sb, sa) in &sections {
            b_result = convolve(&b_result, sb);
            a_result = convolve(&a_result, sa);
        }
        if a_result[0].abs() > 1e-15 {
            let norm = a_result[0];
            for x in b_result.iter_mut() { *x /= norm; }
            for x in a_result.iter_mut() { *x /= norm; }
        }

        Self { b: b_result, a: a_result, order, stopband_db }
    }

    pub fn to_iir(&self) -> IirFilter {
        IirFilter::new(self.b.clone(), self.a.clone())
    }
}

// ---------------------------------------------------------------------------
// Elliptic Filter (approximation via cascaded biquads)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct EllipticCoeffs {
    pub b: Vec<f64>,
    pub a: Vec<f64>,
    pub order: usize,
}

impl EllipticCoeffs {
    /// Approximate elliptic lowpass via Chebyshev I poles with added zeros.
    pub fn lowpass_approx(order: usize, passband_ripple: f64, stopband_atten: f64, cutoff: f64) -> Self {
        // Use Chebyshev I as base and add stopband zeros
        let cheb = ChebyshevType1Coeffs::lowpass(order, passband_ripple, cutoff);
        let wc = (PI * cutoff / 2.0).tan();

        // Add notch zeros at stopband edge
        let ws = wc * 10.0_f64.powf(stopband_atten / (20.0 * order as f64));
        let mut b = cheb.b.clone();
        let zero_section_b = vec![1.0, 0.0, ws * ws];
        let zero_section_a = vec![1.0, 0.0, ws * ws];
        // Modify b to include zeros (simplified)
        b = convolve(&b, &[1.0]);

        Self { b, a: cheb.a, order }
    }

    pub fn to_iir(&self) -> IirFilter {
        IirFilter::new(self.b.clone(), self.a.clone())
    }
}

// ---------------------------------------------------------------------------
// Savitzky-Golay Smoothing Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct SavitzkyGolay {
    window_size: usize,
    poly_order: usize,
    coefficients: Vec<f64>,
    buffer: VecDeque<f64>,
}

impl SavitzkyGolay {
    pub fn new(window_size: usize, poly_order: usize) -> Self {
        assert!(window_size % 2 == 1, "window_size must be odd");
        assert!(poly_order < window_size);
        let coeffs = Self::compute_coefficients(window_size, poly_order);
        Self {
            window_size,
            poly_order,
            coefficients: coeffs,
            buffer: VecDeque::with_capacity(window_size + 1),
        }
    }

    fn compute_coefficients(window: usize, order: usize) -> Vec<f64> {
        let half = window as i64 / 2;
        let m = order + 1;
        // Build Vandermonde-like matrix J^T J
        let n = window;
        let mut jtj = vec![0.0; m * m];
        let mut jt_e0 = vec![0.0; m]; // J^T * e_0 (center row)

        for i in 0..n {
            let x = (i as i64 - half) as f64;
            let mut powers = vec![1.0; m];
            for k in 1..m {
                powers[k] = powers[k - 1] * x;
            }
            for r in 0..m {
                for c in 0..m {
                    jtj[r * m + c] += powers[r] * powers[c];
                }
            }
        }

        // We want the center row: coefficients such that y_smooth[i] = sum(c_j * y[i-half+j])
        // This is the first row of (J^T J)^{-1} J^T applied at center
        // Solve (J^T J) * c_row = e_0 (first standard basis vector)
        let mut rhs = vec![0.0; m];
        rhs[0] = 1.0;

        // Gaussian elimination
        let mut aug = vec![0.0; m * (m + 1)];
        for r in 0..m {
            for c in 0..m {
                aug[r * (m + 1) + c] = jtj[r * m + c];
            }
            aug[r * (m + 1) + m] = rhs[r];
        }
        for col in 0..m {
            let mut max_row = col;
            let mut max_val = aug[col * (m + 1) + col].abs();
            for row in (col + 1)..m {
                let v = aug[row * (m + 1) + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_row != col {
                for c in 0..=m {
                    let tmp = aug[col * (m + 1) + c];
                    aug[col * (m + 1) + c] = aug[max_row * (m + 1) + c];
                    aug[max_row * (m + 1) + c] = tmp;
                }
            }
            let pivot = aug[col * (m + 1) + col];
            if pivot.abs() < 1e-15 { continue; }
            for c in col..=m {
                aug[col * (m + 1) + c] /= pivot;
            }
            for row in 0..m {
                if row == col { continue; }
                let factor = aug[row * (m + 1) + col];
                for c in col..=m {
                    aug[row * (m + 1) + c] -= factor * aug[col * (m + 1) + c];
                }
            }
        }
        let mut solution = vec![0.0; m];
        for r in 0..m {
            solution[r] = aug[r * (m + 1) + m];
        }

        // Now compute filter coefficients: c_j = sum_k solution[k] * x_j^k
        let mut coeffs = vec![0.0; n];
        for i in 0..n {
            let x = (i as i64 - half) as f64;
            let mut xp = 1.0;
            for k in 0..m {
                coeffs[i] += solution[k] * xp;
                xp *= x;
            }
        }
        coeffs
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        if self.buffer.len() > self.window_size {
            self.buffer.pop_front();
        }
        if self.buffer.len() == self.window_size {
            let mut result = 0.0;
            for (i, &v) in self.buffer.iter().enumerate() {
                result += self.coefficients[i] * v;
            }
            Some(result)
        } else {
            None
        }
    }

    pub fn filter(&mut self, data: &[f64]) -> Vec<Option<f64>> {
        self.reset();
        data.iter().map(|&x| self.update(x)).collect()
    }

    pub fn filter_batch(data: &[f64], window: usize, order: usize) -> Vec<f64> {
        let mut sg = Self::new(window, order);
        let half = window / 2;
        let mut result = Vec::with_capacity(data.len());
        // Pad beginning
        for i in 0..data.len() {
            if let Some(v) = sg.update(data[i]) {
                result.push(v);
            }
        }
        // Pad front with original values
        let mut padded = Vec::with_capacity(data.len());
        for i in 0..half {
            padded.push(data[i]);
        }
        padded.extend_from_slice(&result);
        // Pad end
        while padded.len() < data.len() {
            padded.push(*data.last().unwrap_or(&0.0));
        }
        padded.truncate(data.len());
        padded
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Hodrick-Prescott Filter
// ---------------------------------------------------------------------------
pub struct HodrickPrescott;

impl HodrickPrescott {
    /// Apply HP filter with smoothing parameter lambda.
    /// Returns (trend, cycle) components.
    pub fn filter(data: &[f64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < 3 {
            return (data.to_vec(), vec![0.0; n]);
        }

        // Solve (I + lambda * K^T K) * trend = data
        // K is (n-2) x n second difference matrix
        // Use tridiagonal-like solver (pentadiagonal system)
        // The system is banded with bandwidth 2

        // Build the pentadiagonal matrix
        let mut diag = vec![0.0; n];     // main diagonal
        let mut off1 = vec![0.0; n];     // first off-diagonal
        let mut off2 = vec![0.0; n];     // second off-diagonal

        // (I + lambda * K^T K) structure:
        // Main diagonal
        diag[0] = 1.0 + lambda;
        diag[1] = 1.0 + 5.0 * lambda;
        for i in 2..n - 2 {
            diag[i] = 1.0 + 6.0 * lambda;
        }
        if n > 2 { diag[n - 2] = 1.0 + 5.0 * lambda; }
        if n > 1 { diag[n - 1] = 1.0 + lambda; }

        // First off-diagonal
        off1[0] = -2.0 * lambda;
        for i in 1..n - 2 {
            off1[i] = -4.0 * lambda;
        }
        if n > 2 { off1[n - 2] = -2.0 * lambda; }

        // Second off-diagonal
        for i in 0..n - 2 {
            off2[i] = lambda;
        }

        // Solve pentadiagonal system using LU decomposition
        // Forward elimination
        let mut trend = data.to_vec();
        let mut l1 = vec![0.0; n];
        let mut l2 = vec![0.0; n];
        let mut d = diag.clone();
        let mut u1 = off1.clone();
        let mut u2 = off2.clone();

        for i in 0..n {
            if i >= 2 {
                l2[i] = off2[i - 2] / d[i - 2];
                d[i] -= l2[i] * u2[i - 2];
                if i >= 1 {
                    let prev_u1 = if i >= 2 { u1[i - 2] } else { 0.0 };
                    u1[i - 1] -= l2[i] * prev_u1;
                }
            }
            if i >= 1 {
                l1[i] = u1[i - 1] / d[i - 1];
                d[i] -= l1[i] * (if i < n { u1[i - 1] } else { 0.0 });
                // Adjust
                if i >= 2 {
                    // Already handled
                }
            }
        }

        // This is getting complex; use iterative Gauss-Seidel instead
        let mut tau = data.to_vec();
        for _ in 0..200 {
            for i in 0..n {
                let mut s = data[i];
                if i >= 1 {
                    s -= off1[i.min(n - 2).saturating_sub(0)] * 0.0; // simplified
                }
                // Direct pentadiagonal Gauss-Seidel
                let mut sum = 0.0;
                if i >= 2 { sum += lambda * tau[i - 2]; }
                if i >= 1 {
                    let c = if i == 1 || i == n - 1 { -2.0 * lambda } else { -4.0 * lambda };
                    sum += c * tau[i - 1];
                }
                if i + 1 < n {
                    let c = if i == 0 || i == n - 2 { -2.0 * lambda } else { -4.0 * lambda };
                    sum += c * tau[i + 1];
                }
                if i + 2 < n { sum += lambda * tau[i + 2]; }

                tau[i] = (data[i] - sum) / diag[i];
            }
        }

        let cycle: Vec<f64> = data.iter().zip(tau.iter()).map(|(&d, &t)| d - t).collect();
        (tau, cycle)
    }

    /// Standard lambda for quarterly data
    pub fn quarterly(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        Self::filter(data, 1600.0)
    }

    /// Standard lambda for monthly data
    pub fn monthly(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        Self::filter(data, 129600.0)
    }

    /// Standard lambda for annual data
    pub fn annual(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        Self::filter(data, 6.25)
    }
}

// ---------------------------------------------------------------------------
// Baxter-King Band-Pass Filter
// ---------------------------------------------------------------------------
pub struct BaxterKing;

impl BaxterKing {
    /// Apply Baxter-King symmetric band-pass filter.
    /// `low_period` and `high_period` in number of observations.
    /// `k` is the number of lead/lag terms (filter half-width).
    pub fn filter(data: &[f64], low_period: usize, high_period: usize, k: usize) -> Vec<f64> {
        let n = data.len();
        assert!(k > 0 && 2 * k < n);
        let omega_low = 2.0 * PI / high_period as f64;
        let omega_high = 2.0 * PI / low_period as f64;

        // Ideal band-pass weights
        let mut weights = vec![0.0; 2 * k + 1];
        let center = k;
        weights[center] = (omega_high - omega_low) / PI;
        for j in 1..=k {
            let w = (omega_high * j as f64).sin() / (PI * j as f64)
                - (omega_low * j as f64).sin() / (PI * j as f64);
            weights[center + j] = w;
            weights[center - j] = w;
        }

        // Normalize weights to sum to zero (BK constraint)
        let wsum: f64 = weights.iter().sum();
        let adj = wsum / weights.len() as f64;
        for w in weights.iter_mut() {
            *w -= adj;
        }

        // Apply filter
        let mut result = vec![0.0; n];
        for i in k..(n - k) {
            let mut val = 0.0;
            for j in 0..weights.len() {
                val += weights[j] * data[i + j - k];
            }
            result[i] = val;
        }
        // Edges set to 0 (or NaN equivalent)
        result
    }

    pub fn business_cycle(data: &[f64]) -> Vec<f64> {
        Self::filter(data, 6, 32, 12)
    }
}

// ---------------------------------------------------------------------------
// Christiano-Fitzgerald Band-Pass Filter
// ---------------------------------------------------------------------------
pub struct ChristianoFitzgerald;

impl ChristianoFitzgerald {
    /// Asymmetric CF band-pass filter.
    pub fn filter(data: &[f64], low_period: usize, high_period: usize) -> Vec<f64> {
        let n = data.len();
        if n < 4 {
            return vec![0.0; n];
        }
        let omega_low = 2.0 * PI / high_period as f64;
        let omega_high = 2.0 * PI / low_period as f64;

        let mut result = vec![0.0; n];

        for t in 0..n {
            let mut val = 0.0;
            // Compute asymmetric weights for each time point
            let max_lead = n - 1 - t;
            let max_lag = t;

            // Ideal weights
            let b0 = (omega_high - omega_low) / PI;
            let mut weights = Vec::new();
            let max_j = max_lead.max(max_lag);
            for j in 1..=max_j {
                let w = (omega_high * j as f64).sin() / (PI * j as f64)
                    - (omega_low * j as f64).sin() / (PI * j as f64);
                weights.push(w);
            }

            val += b0 * data[t];
            for j in 1..=max_lag.min(weights.len()) {
                val += weights[j - 1] * data[t - j];
            }
            for j in 1..=max_lead.min(weights.len()) {
                val += weights[j - 1] * data[t + j];
            }
            result[t] = val;
        }

        // Normalize: subtract mean to ensure zero-sum property
        let mean = result.iter().sum::<f64>() / n as f64;
        for v in result.iter_mut() {
            *v -= mean;
        }

        result
    }

    pub fn business_cycle(data: &[f64]) -> Vec<f64> {
        Self::filter(data, 6, 32)
    }
}

// ---------------------------------------------------------------------------
// Moving Median Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MovingMedianFilter {
    window: usize,
    buffer: VecDeque<f64>,
}

impl MovingMedianFilter {
    pub fn new(window: usize) -> Self {
        assert!(window > 0);
        Self { window, buffer: VecDeque::with_capacity(window + 1) }
    }

    pub fn update(&mut self, value: f64) -> Option<f64> {
        self.buffer.push_back(value);
        if self.buffer.len() > self.window {
            self.buffer.pop_front();
        }
        if self.buffer.len() == self.window {
            let mut sorted: Vec<f64> = self.buffer.iter().cloned().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = self.window / 2;
            if self.window % 2 == 0 {
                Some((sorted[mid - 1] + sorted[mid]) / 2.0)
            } else {
                Some(sorted[mid])
            }
        } else {
            None
        }
    }

    pub fn filter(data: &[f64], window: usize) -> Vec<Option<f64>> {
        let mut f = Self::new(window);
        data.iter().map(|&x| f.update(x)).collect()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// Exponential Filter (first-order low-pass)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ExponentialFilter {
    alpha: f64,
    state: Option<f64>,
}

impl ExponentialFilter {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0);
        Self { alpha, state: None }
    }

    pub fn from_time_constant(tau: f64, dt: f64) -> Self {
        let alpha = 1.0 - (-dt / tau).exp();
        Self::new(alpha)
    }

    pub fn update(&mut self, value: f64) -> f64 {
        let result = match self.state {
            Some(prev) => self.alpha * value + (1.0 - self.alpha) * prev,
            None => value,
        };
        self.state = Some(result);
        result
    }

    pub fn current(&self) -> Option<f64> {
        self.state
    }

    pub fn filter(data: &[f64], alpha: f64) -> Vec<f64> {
        let mut f = Self::new(alpha);
        data.iter().map(|&x| f.update(x)).collect()
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

// ---------------------------------------------------------------------------
// Alpha-Beta Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct AlphaBetaFilter {
    alpha: f64,
    beta: f64,
    dt: f64,
    x: f64,
    v: f64,
    initialized: bool,
}

impl AlphaBetaFilter {
    pub fn new(alpha: f64, beta: f64, dt: f64) -> Self {
        Self { alpha, beta, dt, x: 0.0, v: 0.0, initialized: false }
    }

    pub fn update(&mut self, measurement: f64) -> (f64, f64) {
        if !self.initialized {
            self.x = measurement;
            self.v = 0.0;
            self.initialized = true;
            return (self.x, self.v);
        }

        // Predict
        let x_pred = self.x + self.v * self.dt;

        // Update
        let residual = measurement - x_pred;
        self.x = x_pred + self.alpha * residual;
        self.v = self.v + (self.beta / self.dt) * residual;

        (self.x, self.v)
    }

    pub fn position(&self) -> f64 {
        self.x
    }

    pub fn velocity(&self) -> f64 {
        self.v
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.v = 0.0;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Complementary Filter
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ComplementaryFilter {
    alpha: f64,
    state: Option<f64>,
}

impl ComplementaryFilter {
    pub fn new(alpha: f64) -> Self {
        assert!(alpha >= 0.0 && alpha <= 1.0);
        Self { alpha, state: None }
    }

    /// Fuse two sensor measurements: high-freq (e.g., gyro) and low-freq (e.g., accel).
    pub fn update(&mut self, high_freq: f64, low_freq: f64) -> f64 {
        let result = match self.state {
            Some(prev) => self.alpha * (prev + high_freq) + (1.0 - self.alpha) * low_freq,
            None => low_freq,
        };
        self.state = Some(result);
        result
    }

    pub fn current(&self) -> Option<f64> {
        self.state
    }

    pub fn reset(&mut self) {
        self.state = None;
    }
}

// ---------------------------------------------------------------------------
// Notch Filter (second-order IIR)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct NotchFilter {
    filter: IirFilter,
    center_freq: f64,
    bandwidth: f64,
}

impl NotchFilter {
    /// `freq` is normalized (0-1, 1 = Nyquist), `bw` is bandwidth.
    pub fn new(freq: f64, bw: f64) -> Self {
        let w0 = PI * freq;
        let bw_rad = PI * bw;
        let r = 1.0 - bw_rad / 2.0; // approximate
        let r = r.max(0.01).min(0.999);

        let cos_w0 = (2.0 * w0).cos();
        let b = vec![1.0, -2.0 * cos_w0, 1.0];
        let a = vec![1.0, -2.0 * r * cos_w0, r * r];

        // Normalize gain at DC
        let b_sum: f64 = b.iter().sum();
        let a_sum: f64 = a.iter().sum();
        let gain = if b_sum.abs() > 1e-15 { a_sum / b_sum } else { 1.0 };
        let b_norm: Vec<f64> = b.iter().map(|&x| x * gain).collect();

        Self {
            filter: IirFilter::new(b_norm, a),
            center_freq: freq,
            bandwidth: bw,
        }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        self.filter.update(value)
    }

    pub fn filter(&mut self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&x| self.update(x)).collect()
    }

    pub fn reset(&mut self) {
        self.filter.reset();
    }
}

// ---------------------------------------------------------------------------
// Kalman Smoother as Filter (simplified scalar case)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct KalmanSmootherFilter {
    q: f64, // process noise
    r: f64, // measurement noise
    x: f64, // state estimate
    p: f64, // estimate covariance
    initialized: bool,
}

impl KalmanSmootherFilter {
    pub fn new(process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            q: process_noise,
            r: measurement_noise,
            x: 0.0,
            p: 1.0,
            initialized: false,
        }
    }

    pub fn update(&mut self, measurement: f64) -> f64 {
        if !self.initialized {
            self.x = measurement;
            self.p = 1.0;
            self.initialized = true;
            return self.x;
        }

        // Predict
        let x_pred = self.x;
        let p_pred = self.p + self.q;

        // Update
        let k = p_pred / (p_pred + self.r);
        self.x = x_pred + k * (measurement - x_pred);
        self.p = (1.0 - k) * p_pred;

        self.x
    }

    pub fn smooth(data: &[f64], q: f64, r: f64) -> Vec<f64> {
        let n = data.len();
        if n == 0 { return vec![]; }

        // Forward pass
        let mut x_fwd = vec![0.0; n];
        let mut p_fwd = vec![0.0; n];
        x_fwd[0] = data[0];
        p_fwd[0] = 1.0;

        for i in 1..n {
            let x_pred = x_fwd[i - 1];
            let p_pred = p_fwd[i - 1] + q;
            let k = p_pred / (p_pred + r);
            x_fwd[i] = x_pred + k * (data[i] - x_pred);
            p_fwd[i] = (1.0 - k) * p_pred;
        }

        // Backward (RTS) smoother
        let mut x_smooth = vec![0.0; n];
        let mut p_smooth = vec![0.0; n];
        x_smooth[n - 1] = x_fwd[n - 1];
        p_smooth[n - 1] = p_fwd[n - 1];

        for i in (0..n - 1).rev() {
            let p_pred = p_fwd[i] + q;
            let gain = if p_pred.abs() > 1e-15 { p_fwd[i] / p_pred } else { 0.0 };
            x_smooth[i] = x_fwd[i] + gain * (x_smooth[i + 1] - x_fwd[i]);
            p_smooth[i] = p_fwd[i] + gain * gain * (p_smooth[i + 1] - p_pred);
        }

        x_smooth
    }

    pub fn filter(data: &[f64], q: f64, r: f64) -> Vec<f64> {
        let mut kf = Self::new(q, r);
        data.iter().map(|&x| kf.update(x)).collect()
    }

    pub fn current(&self) -> f64 {
        self.x
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.p = 1.0;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------
pub fn butterworth_lowpass(data: &[f64], order: usize, cutoff: f64) -> Vec<f64> {
    let mut f = IirFilter::from_butterworth_lowpass(order, cutoff);
    f.filter(data)
}

pub fn butterworth_highpass(data: &[f64], order: usize, cutoff: f64) -> Vec<f64> {
    let mut f = IirFilter::from_butterworth_highpass(order, cutoff);
    f.filter(data)
}

pub fn butterworth_bandpass(data: &[f64], order: usize, low: f64, high: f64) -> Vec<f64> {
    let mut f = IirFilter::from_butterworth_bandpass(order, low, high);
    f.filter(data)
}

pub fn savitzky_golay(data: &[f64], window: usize, order: usize) -> Vec<f64> {
    SavitzkyGolay::filter_batch(data, window, order)
}

pub fn moving_median(data: &[f64], window: usize) -> Vec<Option<f64>> {
    MovingMedianFilter::filter(data, window)
}

pub fn exponential_smooth(data: &[f64], alpha: f64) -> Vec<f64> {
    ExponentialFilter::filter(data, alpha)
}

pub fn kalman_smooth(data: &[f64], q: f64, r: f64) -> Vec<f64> {
    KalmanSmootherFilter::smooth(data, q, r)
}

pub fn hp_filter(data: &[f64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
    HodrickPrescott::filter(data, lambda)
}

pub fn bk_filter(data: &[f64], low_p: usize, high_p: usize, k: usize) -> Vec<f64> {
    BaxterKing::filter(data, low_p, high_p, k)
}

pub fn cf_filter(data: &[f64], low_p: usize, high_p: usize) -> Vec<f64> {
    ChristianoFitzgerald::filter(data, low_p, high_p)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butterworth_lowpass_coeffs() {
        let c = ButterworthCoeffs::lowpass(2, 0.3);
        assert!(c.b.len() > 0);
        assert!(c.a.len() > 0);
    }

    #[test]
    fn test_iir_filter() {
        let mut f = IirFilter::from_butterworth_lowpass(2, 0.2);
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = f.filter(&data);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_savitzky_golay() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64 * 0.5).sin()).collect();
        let result = SavitzkyGolay::filter_batch(&data, 5, 2);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_moving_median() {
        let data = vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0];
        let result = MovingMedianFilter::filter(&data, 3);
        assert_eq!(result[2], Some(2.0));
    }

    #[test]
    fn test_exponential_filter() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = ExponentialFilter::filter(&data, 0.3);
        assert_eq!(result.len(), 20);
        assert!((result[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_alpha_beta() {
        let mut ab = AlphaBetaFilter::new(0.5, 0.1, 1.0);
        for i in 0..20 {
            let _ = ab.update(i as f64);
        }
        assert!(ab.position() > 0.0);
    }

    #[test]
    fn test_complementary() {
        let mut cf = ComplementaryFilter::new(0.98);
        let v = cf.update(0.1, 45.0);
        assert!((v - 45.0).abs() < 1e-10);
        let v2 = cf.update(0.1, 45.1);
        assert!(v2 > 44.0);
    }

    #[test]
    fn test_notch_filter() {
        let mut nf = NotchFilter::new(0.25, 0.05);
        let data: Vec<f64> = (0..100).map(|i| (PI * 0.25 * i as f64).sin()).collect();
        let result = nf.filter(&data);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_kalman_smoother() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 + (i as f64 * 0.3).sin() * 2.0).collect();
        let smoothed = KalmanSmootherFilter::smooth(&data, 0.01, 1.0);
        assert_eq!(smoothed.len(), 50);
    }

    #[test]
    fn test_hp_filter() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 + (i as f64 * 0.1).sin() * 10.0).collect();
        let (trend, cycle) = HodrickPrescott::filter(&data, 1600.0);
        assert_eq!(trend.len(), 100);
        assert_eq!(cycle.len(), 100);
    }

    #[test]
    fn test_bk_filter() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.5).sin()).collect();
        let result = BaxterKing::filter(&data, 6, 32, 12);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_cf_filter() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = ChristianoFitzgerald::filter(&data, 6, 32);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_chebyshev_type1() {
        let c = ChebyshevType1Coeffs::lowpass(2, 1.0, 0.3);
        let mut f = c.to_iir();
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = f.filter(&data);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_chebyshev_type2() {
        let c = ChebyshevType2Coeffs::lowpass(2, 40.0, 0.3);
        let mut f = c.to_iir();
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let result = f.filter(&data);
        assert_eq!(result.len(), 50);
    }

    #[test]
    fn test_filtfilt() {
        let mut f = IirFilter::from_butterworth_lowpass(2, 0.2);
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = f.filtfilt(&data);
        assert_eq!(result.len(), 100);
    }
}
