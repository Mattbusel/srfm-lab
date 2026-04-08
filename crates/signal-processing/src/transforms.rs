use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Complex number type (minimal, no external deps)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conjugate(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    pub fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn div(self, other: Self) -> Self {
        let denom = other.magnitude_sq();
        if denom < 1e-30 {
            return Self::zero();
        }
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }

    pub fn exp(self) -> Self {
        let e = self.re.exp();
        Self { re: e * self.im.cos(), im: e * self.im.sin() }
    }

    pub fn norm(&self) -> f64 {
        self.magnitude()
    }

    pub fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Complex::add(self, rhs) }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Complex::sub(self, rhs) }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { Complex::mul(self, rhs) }
}

impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self { Complex::div(self, rhs) }
}

impl std::ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self { Self { re: -self.re, im: -self.im } }
}

// ---------------------------------------------------------------------------
// FFT (Cooley-Tukey radix-2 DIT)
// ---------------------------------------------------------------------------
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }

    // Pad to next power of 2
    let n_padded = n.next_power_of_two();
    let mut data: Vec<Complex> = input.to_vec();
    data.resize(n_padded, Complex::zero());

    fft_in_place(&mut data, false);
    data
}

pub fn fft_real(input: &[f64]) -> Vec<Complex> {
    let complexified: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft(&complexified)
}

fn fft_in_place(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be power of 2");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_sign = if inverse { 1.0 } else { -1.0 };
        let angle = angle_sign * 2.0 * PI / len as f64;
        let wn = Complex::from_polar(1.0, angle);

        let mut i = 0;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = data[i + k];
                let t = w * data[i + k + half];
                data[i + k] = u + t;
                data[i + k + half] = u - t;
                w = w * wn;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x = x.scale(scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Inverse FFT
// ---------------------------------------------------------------------------
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let n_padded = n.next_power_of_two();
    let mut data = input.to_vec();
    data.resize(n_padded, Complex::zero());

    fft_in_place(&mut data, true);
    data
}

pub fn ifft_real(input: &[Complex]) -> Vec<f64> {
    let result = ifft(input);
    result.iter().map(|c| c.re).collect()
}

// ---------------------------------------------------------------------------
// DFT for arbitrary length (O(N^2))
// ---------------------------------------------------------------------------
pub fn dft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = vec![Complex::zero(); n];
    for k in 0..n {
        for j in 0..n {
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            let w = Complex::from_polar(1.0, angle);
            output[k] = output[k] + input[j] * w;
        }
    }
    output
}

pub fn dft_real(input: &[f64]) -> Vec<Complex> {
    let complexified: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    dft(&complexified)
}

pub fn idft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = vec![Complex::zero(); n];
    for k in 0..n {
        for j in 0..n {
            let angle = 2.0 * PI * k as f64 * j as f64 / n as f64;
            let w = Complex::from_polar(1.0, angle);
            output[k] = output[k] + input[j] * w;
        }
        output[k] = output[k].scale(1.0 / n as f64);
    }
    output
}

// ---------------------------------------------------------------------------
// Power Spectral Density
// ---------------------------------------------------------------------------
pub fn power_spectral_density(data: &[f64]) -> Vec<f64> {
    let spectrum = fft_real(data);
    let n = data.len() as f64;
    spectrum.iter().map(|c| c.magnitude_sq() / n).collect()
}

pub fn power_spectral_density_welch(data: &[f64], segment_len: usize, overlap: usize) -> Vec<f64> {
    let n = data.len();
    if n < segment_len {
        return power_spectral_density(data);
    }

    let step = segment_len - overlap;
    let n_padded = segment_len.next_power_of_two();
    let mut psd_sum = vec![0.0; n_padded];
    let mut count = 0;

    // Hanning window
    let window: Vec<f64> = (0..segment_len)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (segment_len - 1) as f64).cos()))
        .collect();
    let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>() / segment_len as f64;

    let mut start = 0;
    while start + segment_len <= n {
        let mut segment: Vec<Complex> = Vec::with_capacity(n_padded);
        for i in 0..segment_len {
            segment.push(Complex::new(data[start + i] * window[i], 0.0));
        }
        segment.resize(n_padded, Complex::zero());

        fft_in_place(&mut segment, false);

        for (i, c) in segment.iter().enumerate() {
            psd_sum[i] += c.magnitude_sq();
        }
        count += 1;
        start += step;
    }

    if count > 0 {
        let scale = 1.0 / (count as f64 * segment_len as f64 * window_power);
        for v in psd_sum.iter_mut() {
            *v *= scale;
        }
    }
    psd_sum
}

// ---------------------------------------------------------------------------
// Cross-Spectral Density
// ---------------------------------------------------------------------------
pub fn cross_spectral_density(x: &[f64], y: &[f64]) -> Vec<Complex> {
    assert_eq!(x.len(), y.len());
    let fx = fft_real(x);
    let fy = fft_real(y);
    let n = x.len() as f64;
    fx.iter()
        .zip(fy.iter())
        .map(|(&a, &b)| a.conjugate() * b.scale(1.0 / n))
        .collect()
}

// ---------------------------------------------------------------------------
// Coherence
// ---------------------------------------------------------------------------
pub fn coherence(x: &[f64], y: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    let csd = cross_spectral_density(x, y);
    let psd_x = power_spectral_density(x);
    let psd_y = power_spectral_density(y);

    csd.iter()
        .zip(psd_x.iter().zip(psd_y.iter()))
        .map(|(c, (&px, &py))| {
            let denom = px * py;
            if denom > 1e-30 {
                c.magnitude_sq() / denom
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hilbert Transform (via FFT)
// ---------------------------------------------------------------------------
pub fn hilbert_transform(data: &[f64]) -> Vec<Complex> {
    let n = data.len();
    let n_padded = n.next_power_of_two();
    let mut spectrum: Vec<Complex> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    spectrum.resize(n_padded, Complex::zero());

    fft_in_place(&mut spectrum, false);

    // Double positive frequencies, zero negative frequencies
    // DC and Nyquist unchanged
    let half = n_padded / 2;
    for i in 1..half {
        spectrum[i] = spectrum[i].scale(2.0);
    }
    for i in (half + 1)..n_padded {
        spectrum[i] = Complex::zero();
    }

    fft_in_place(&mut spectrum, true);
    spectrum.truncate(n);
    spectrum
}

// ---------------------------------------------------------------------------
// Analytic Signal
// ---------------------------------------------------------------------------
pub struct AnalyticSignal;

impl AnalyticSignal {
    pub fn compute(data: &[f64]) -> Vec<Complex> {
        hilbert_transform(data)
    }

    pub fn instantaneous_amplitude(data: &[f64]) -> Vec<f64> {
        let analytic = hilbert_transform(data);
        analytic.iter().map(|c| c.magnitude()).collect()
    }

    pub fn instantaneous_phase(data: &[f64]) -> Vec<f64> {
        let analytic = hilbert_transform(data);
        analytic.iter().map(|c| c.phase()).collect()
    }

    pub fn instantaneous_frequency(data: &[f64], sample_rate: f64) -> Vec<f64> {
        let phase = Self::instantaneous_phase(data);
        let n = phase.len();
        if n < 2 {
            return vec![0.0; n];
        }
        let mut freq = vec![0.0; n];
        for i in 1..n {
            let mut dp = phase[i] - phase[i - 1];
            // Unwrap phase
            while dp > PI { dp -= 2.0 * PI; }
            while dp < -PI { dp += 2.0 * PI; }
            freq[i] = dp * sample_rate / (2.0 * PI);
        }
        freq[0] = freq[1];
        freq
    }
}

// ---------------------------------------------------------------------------
// Wavelet Transform (Morlet)
// ---------------------------------------------------------------------------
pub struct MorletWavelet;

impl MorletWavelet {
    /// Compute continuous wavelet transform using Morlet wavelet.
    /// Returns a 2D matrix (scales x time).
    pub fn cwt(data: &[f64], scales: &[f64], omega0: f64) -> Vec<Vec<Complex>> {
        let n = data.len();
        let n_padded = n.next_power_of_two();

        // FFT of signal
        let mut signal_fft: Vec<Complex> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
        signal_fft.resize(n_padded, Complex::zero());
        fft_in_place(&mut signal_fft, false);

        let mut result = Vec::with_capacity(scales.len());

        for &scale in scales {
            // Compute Morlet wavelet in frequency domain
            let mut wavelet_fft = vec![Complex::zero(); n_padded];
            let norm = (2.0 * PI * scale / 1.0).sqrt().powf(0.5); // normalization

            for k in 0..n_padded {
                let omega = if k <= n_padded / 2 {
                    2.0 * PI * k as f64 / n_padded as f64
                } else {
                    2.0 * PI * (k as f64 - n_padded as f64) / n_padded as f64
                };
                let scaled_omega = scale * omega;
                // Morlet wavelet in freq domain: pi^{-1/4} * exp(-0.5*(s*w - w0)^2)
                let val = PI.powf(-0.25) * (-(scaled_omega - omega0).powi(2) / 2.0).exp();
                let val = val * scale.sqrt();
                wavelet_fft[k] = signal_fft[k].scale(val);
            }

            // Inverse FFT
            fft_in_place(&mut wavelet_fft, true);
            wavelet_fft.truncate(n);
            result.push(wavelet_fft);
        }

        result
    }

    pub fn default_scales(n_scales: usize, min_scale: f64, max_scale: f64) -> Vec<f64> {
        let mut scales = Vec::with_capacity(n_scales);
        let log_min = min_scale.ln();
        let log_max = max_scale.ln();
        for i in 0..n_scales {
            let t = i as f64 / (n_scales - 1).max(1) as f64;
            scales.push((log_min + t * (log_max - log_min)).exp());
        }
        scales
    }

    pub fn scalogram(data: &[f64], scales: &[f64], omega0: f64) -> Vec<Vec<f64>> {
        let cwt_result = Self::cwt(data, scales, omega0);
        cwt_result
            .iter()
            .map(|row| row.iter().map(|c| c.magnitude_sq()).collect())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Short-Time Fourier Transform (STFT)
// ---------------------------------------------------------------------------
pub struct Stft;

impl Stft {
    /// Compute STFT with given window size and hop size.
    /// Returns frames x frequency bins.
    pub fn compute(data: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<Complex>> {
        let n = data.len();
        if n < window_size {
            return vec![];
        }

        let n_padded = window_size.next_power_of_two();
        // Hanning window
        let window: Vec<f64> = (0..window_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (window_size - 1) as f64).cos()))
            .collect();

        let mut frames = Vec::new();
        let mut start = 0;
        while start + window_size <= n {
            let mut frame: Vec<Complex> = Vec::with_capacity(n_padded);
            for i in 0..window_size {
                frame.push(Complex::new(data[start + i] * window[i], 0.0));
            }
            frame.resize(n_padded, Complex::zero());
            fft_in_place(&mut frame, false);
            // Keep only positive frequencies
            frame.truncate(n_padded / 2 + 1);
            frames.push(frame);
            start += hop_size;
        }
        frames
    }

    pub fn spectrogram(data: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<f64>> {
        let stft = Self::compute(data, window_size, hop_size);
        stft.iter()
            .map(|frame| frame.iter().map(|c| c.magnitude_sq()).collect())
            .collect()
    }

    pub fn istft(frames: &[Vec<Complex>], window_size: usize, hop_size: usize, output_len: usize) -> Vec<f64> {
        let n_padded = window_size.next_power_of_two();
        let window: Vec<f64> = (0..window_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (window_size - 1) as f64).cos()))
            .collect();

        let mut output = vec![0.0; output_len];
        let mut window_sum = vec![0.0; output_len];

        for (frame_idx, frame) in frames.iter().enumerate() {
            // Reconstruct full spectrum (mirror)
            let mut full: Vec<Complex> = Vec::with_capacity(n_padded);
            for c in frame.iter() {
                full.push(*c);
            }
            while full.len() < n_padded {
                let idx = n_padded - full.len();
                full.push(frame.get(idx).map(|c| c.conjugate()).unwrap_or(Complex::zero()));
            }

            fft_in_place(&mut full, true);

            let start = frame_idx * hop_size;
            for i in 0..window_size {
                if start + i < output_len {
                    output[start + i] += full[i].re * window[i];
                    window_sum[start + i] += window[i] * window[i];
                }
            }
        }

        for i in 0..output_len {
            if window_sum[i] > 1e-10 {
                output[i] /= window_sum[i];
            }
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Cepstrum
// ---------------------------------------------------------------------------
pub struct Cepstrum;

impl Cepstrum {
    /// Real cepstrum: ifft(log(|fft(x)|))
    pub fn real_cepstrum(data: &[f64]) -> Vec<f64> {
        let spectrum = fft_real(data);
        let log_magnitude: Vec<Complex> = spectrum
            .iter()
            .map(|c| {
                let mag = c.magnitude().max(1e-30);
                Complex::new(mag.ln(), 0.0)
            })
            .collect();
        let result = ifft(&log_magnitude);
        result.iter().map(|c| c.re).collect()
    }

    /// Complex cepstrum: ifft(log(fft(x))) with phase unwrapping.
    pub fn complex_cepstrum(data: &[f64]) -> Vec<Complex> {
        let spectrum = fft_real(data);
        let log_spectrum: Vec<Complex> = spectrum
            .iter()
            .map(|c| {
                let mag = c.magnitude().max(1e-30);
                let phase = c.phase();
                Complex::new(mag.ln(), phase)
            })
            .collect();
        ifft(&log_spectrum)
    }

    /// Power cepstrum: |ifft(log(|fft(x)|^2))|^2
    pub fn power_cepstrum(data: &[f64]) -> Vec<f64> {
        let spectrum = fft_real(data);
        let log_power: Vec<Complex> = spectrum
            .iter()
            .map(|c| {
                let power = c.magnitude_sq().max(1e-30);
                Complex::new(power.ln(), 0.0)
            })
            .collect();
        let result = ifft(&log_power);
        result.iter().map(|c| c.magnitude_sq()).collect()
    }
}

// ---------------------------------------------------------------------------
// Autocorrelation via FFT
// ---------------------------------------------------------------------------
pub fn autocorrelation(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let n_padded = (2 * n).next_power_of_two();

    let mean = data.iter().sum::<f64>() / n as f64;
    let mut padded: Vec<Complex> = data
        .iter()
        .map(|&x| Complex::new(x - mean, 0.0))
        .collect();
    padded.resize(n_padded, Complex::zero());

    fft_in_place(&mut padded, false);

    // Multiply by conjugate (power spectrum)
    for c in padded.iter_mut() {
        let mag_sq = c.magnitude_sq();
        *c = Complex::new(mag_sq, 0.0);
    }

    fft_in_place(&mut padded, true);

    // Normalize
    let var = padded[0].re;
    if var.abs() > 1e-30 {
        padded.iter().take(n).map(|c| c.re / var).collect()
    } else {
        vec![0.0; n]
    }
}

pub fn autocorrelation_biased(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut result = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag.min(n - 1) {
        let mut sum = 0.0;
        for i in 0..n - lag {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        result[lag] = sum / n as f64;
    }
    // Normalize by variance
    if result[0].abs() > 1e-30 {
        let var = result[0];
        for v in result.iter_mut() {
            *v /= var;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Cross-Correlation
// ---------------------------------------------------------------------------
pub fn cross_correlation(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len().max(y.len());
    let n_padded = (2 * n).next_power_of_two();

    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;

    let mut fx: Vec<Complex> = x.iter().map(|&v| Complex::new(v - mean_x, 0.0)).collect();
    let mut fy: Vec<Complex> = y.iter().map(|&v| Complex::new(v - mean_y, 0.0)).collect();
    fx.resize(n_padded, Complex::zero());
    fy.resize(n_padded, Complex::zero());

    fft_in_place(&mut fx, false);
    fft_in_place(&mut fy, false);

    // Cross-power spectrum: conj(X) * Y
    let mut cross: Vec<Complex> = fx
        .iter()
        .zip(fy.iter())
        .map(|(&a, &b)| a.conjugate() * b)
        .collect();

    fft_in_place(&mut cross, true);

    // Normalize
    let var_x: f64 = x.iter().map(|&v| (v - mean_x).powi(2)).sum::<f64>();
    let var_y: f64 = y.iter().map(|&v| (v - mean_y).powi(2)).sum::<f64>();
    let norm = (var_x * var_y).sqrt();

    if norm > 1e-30 {
        cross.iter().take(n).map(|c| c.re / norm).collect()
    } else {
        vec![0.0; n]
    }
}

pub fn cross_correlation_full(x: &[f64], y: &[f64]) -> Vec<f64> {
    let nx = x.len();
    let ny = y.len();
    let out_len = nx + ny - 1;
    let n_padded = out_len.next_power_of_two();

    let mut fx: Vec<Complex> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut fy: Vec<Complex> = y.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fx.resize(n_padded, Complex::zero());
    fy.resize(n_padded, Complex::zero());

    fft_in_place(&mut fx, false);
    fft_in_place(&mut fy, false);

    let mut cross: Vec<Complex> = fx
        .iter()
        .zip(fy.iter())
        .map(|(&a, &b)| a.conjugate() * b)
        .collect();

    fft_in_place(&mut cross, true);

    // Rearrange: negative lags first, then positive
    let mut result = Vec::with_capacity(out_len);
    for i in (n_padded - ny + 1)..n_padded {
        result.push(cross[i].re);
    }
    for i in 0..nx {
        result.push(cross[i].re);
    }
    result.truncate(out_len);
    result
}

// ---------------------------------------------------------------------------
// Windowing functions
// ---------------------------------------------------------------------------
pub fn hanning_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

pub fn hamming_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

pub fn blackman_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = 2.0 * PI * i as f64 / (n - 1) as f64;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

pub fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
            let arg = beta * (1.0 - x * x).max(0.0).sqrt();
            bessel_i0(arg) / i0_beta
        })
        .collect()
}

pub fn flat_top_window(n: usize) -> Vec<f64> {
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;
    (0..n)
        .map(|i| {
            let t = 2.0 * PI * i as f64 / (n - 1) as f64;
            a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + a4 * (4.0 * t).cos()
        })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..25 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-16 * sum { break; }
    }
    sum
}

// ---------------------------------------------------------------------------
// Apply window to data
// ---------------------------------------------------------------------------
pub fn apply_window(data: &[f64], window: &[f64]) -> Vec<f64> {
    data.iter()
        .zip(window.iter())
        .map(|(&d, &w)| d * w)
        .collect()
}

// ---------------------------------------------------------------------------
// Frequency axis helper
// ---------------------------------------------------------------------------
pub fn frequency_axis(n: usize, sample_rate: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * sample_rate / n as f64).collect()
}

pub fn frequency_axis_centered(n: usize, sample_rate: f64) -> Vec<f64> {
    let half = n / 2;
    (0..n)
        .map(|i| {
            if i <= half {
                i as f64 * sample_rate / n as f64
            } else {
                (i as f64 - n as f64) * sample_rate / n as f64
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Phase unwrapping
// ---------------------------------------------------------------------------
pub fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; phase.len()];
    if phase.is_empty() { return result; }
    result[0] = phase[0];
    for i in 1..phase.len() {
        let mut diff = phase[i] - phase[i - 1];
        while diff > PI { diff -= 2.0 * PI; }
        while diff < -PI { diff += 2.0 * PI; }
        result[i] = result[i - 1] + diff;
    }
    result
}

// ---------------------------------------------------------------------------
// Zero-padding
// ---------------------------------------------------------------------------
pub fn zero_pad(data: &[f64], target_len: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    result.resize(target_len, 0.0);
    result
}

pub fn zero_pad_power_of_two(data: &[f64]) -> Vec<f64> {
    let target = data.len().next_power_of_two();
    zero_pad(data, target)
}

// ---------------------------------------------------------------------------
// Convolution via FFT
// ---------------------------------------------------------------------------
pub fn fft_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let out_len = a.len() + b.len() - 1;
    let n_padded = out_len.next_power_of_two();

    let mut fa: Vec<Complex> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut fb: Vec<Complex> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fa.resize(n_padded, Complex::zero());
    fb.resize(n_padded, Complex::zero());

    fft_in_place(&mut fa, false);
    fft_in_place(&mut fb, false);

    let mut product: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&a, &b)| a * b).collect();
    fft_in_place(&mut product, true);

    product.iter().take(out_len).map(|c| c.re).collect()
}

// ---------------------------------------------------------------------------
// Deconvolution via FFT (Wiener-like)
// ---------------------------------------------------------------------------
pub fn fft_deconvolve(signal: &[f64], kernel: &[f64], noise_power: f64) -> Vec<f64> {
    let n = signal.len();
    let n_padded = n.next_power_of_two();

    let mut fs: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut fk: Vec<Complex> = kernel.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fs.resize(n_padded, Complex::zero());
    fk.resize(n_padded, Complex::zero());

    fft_in_place(&mut fs, false);
    fft_in_place(&mut fk, false);

    // Wiener deconvolution: H^* / (|H|^2 + noise)
    let mut result: Vec<Complex> = fs
        .iter()
        .zip(fk.iter())
        .map(|(&s, &h)| {
            let h_conj = h.conjugate();
            let h_power = h.magnitude_sq();
            let wiener = h_conj.scale(1.0 / (h_power + noise_power));
            s * wiener
        })
        .collect();

    fft_in_place(&mut result, true);
    result.iter().take(n).map(|c| c.re).collect()
}

// ---------------------------------------------------------------------------
// Goertzel algorithm (single frequency DFT bin)
// ---------------------------------------------------------------------------
pub fn goertzel(data: &[f64], target_freq: f64, sample_rate: f64) -> Complex {
    let n = data.len();
    let k = (target_freq * n as f64 / sample_rate).round();
    let omega = 2.0 * PI * k / n as f64;
    let coeff = 2.0 * omega.cos();

    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;

    for &x in data {
        s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    Complex::new(
        s1 - s2 * omega.cos(),
        s2 * omega.sin(),
    )
}

pub fn goertzel_magnitude(data: &[f64], target_freq: f64, sample_rate: f64) -> f64 {
    goertzel(data, target_freq, sample_rate).magnitude()
}

// ---------------------------------------------------------------------------
// Periodogram
// ---------------------------------------------------------------------------
pub fn periodogram(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let spectrum = fft_real(data);
    let n_freq = n / 2 + 1;
    let mut psd = Vec::with_capacity(n_freq);
    for i in 0..n_freq {
        let mut val = spectrum[i].magnitude_sq() / n as f64;
        if i > 0 && i < n / 2 {
            val *= 2.0; // one-sided
        }
        psd.push(val);
    }
    psd
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = a * b;
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        let data: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 64.0).sin()).collect();
        let spectrum = fft_real(&data);
        let recovered = ifft_real(&spectrum);
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                i, orig, rec
            );
        }
    }

    #[test]
    fn test_dft_matches_fft() {
        let data: Vec<Complex> = (0..8)
            .map(|i| Complex::new((i as f64 * 0.5).sin(), 0.0))
            .collect();
        let dft_result = dft(&data);
        let fft_result = fft(&data);
        for (d, f) in dft_result.iter().zip(fft_result.iter()) {
            assert!((d.re - f.re).abs() < 1e-8);
            assert!((d.im - f.im).abs() < 1e-8);
        }
    }

    #[test]
    fn test_power_spectral_density() {
        let data: Vec<f64> = (0..128).map(|i| (2.0 * PI * 10.0 * i as f64 / 128.0).sin()).collect();
        let psd = power_spectral_density(&data);
        assert!(psd.len() > 0);
    }

    #[test]
    fn test_autocorrelation() {
        let data: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let ac = autocorrelation(&data);
        assert!((ac[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_correlation() {
        let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin()).collect();
        let y = x.clone();
        let cc = cross_correlation(&x, &y);
        assert!(cc[0] > 0.9);
    }

    #[test]
    fn test_hilbert_transform() {
        let data: Vec<f64> = (0..64).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let analytic = hilbert_transform(&data);
        assert_eq!(analytic.len(), 64);
        for (i, (c, &d)) in analytic.iter().zip(data.iter()).enumerate() {
            assert!(
                (c.re - d).abs() < 1e-8,
                "Real part mismatch at {}: {} vs {}",
                i, c.re, d
            );
        }
    }

    #[test]
    fn test_stft() {
        let data: Vec<f64> = (0..256).map(|i| (2.0 * PI * i as f64 / 32.0).sin()).collect();
        let frames = Stft::compute(&data, 64, 32);
        assert!(frames.len() > 0);
    }

    #[test]
    fn test_cepstrum() {
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin() + 0.5).collect();
        let cep = Cepstrum::real_cepstrum(&data);
        assert_eq!(cep.len(), 64);
    }

    #[test]
    fn test_windowing() {
        let h = hanning_window(64);
        assert_eq!(h.len(), 64);
        assert!((h[0]).abs() < 1e-10);
        assert!((h[32] - 1.0).abs() < 0.01);

        let hm = hamming_window(64);
        assert_eq!(hm.len(), 64);

        let bk = blackman_window(64);
        assert_eq!(bk.len(), 64);
    }

    #[test]
    fn test_fft_convolve() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 0.5];
        let result = fft_convolve(&a, &b);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_goertzel() {
        let n = 128;
        let freq = 10.0;
        let sr = 128.0;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 / sr).sin()).collect();
        let mag = goertzel_magnitude(&data, freq, sr);
        assert!(mag > 50.0); // Strong peak
    }

    #[test]
    fn test_morlet_cwt() {
        let data: Vec<f64> = (0..128).map(|i| (2.0 * PI * i as f64 / 16.0).sin()).collect();
        let scales = MorletWavelet::default_scales(10, 1.0, 32.0);
        let cwt = MorletWavelet::cwt(&data, &scales, 6.0);
        assert_eq!(cwt.len(), 10);
        assert_eq!(cwt[0].len(), 128);
    }

    #[test]
    fn test_phase_unwrap() {
        let phase = vec![0.0, 0.5, 1.0, 1.5, -2.8, -2.3, -1.8];
        let unwrapped = unwrap_phase(&phase);
        for i in 1..unwrapped.len() {
            let diff = (unwrapped[i] - unwrapped[i - 1]).abs();
            assert!(diff < PI + 0.1);
        }
    }

    #[test]
    fn test_coherence() {
        let x: Vec<f64> = (0..64).map(|i| (i as f64 * 0.2).sin()).collect();
        let y = x.clone();
        let coh = coherence(&x, &y);
        // Self-coherence should be ~1
        for &c in coh.iter().take(32) {
            assert!(c >= -0.01 && c <= 1.01);
        }
    }

    #[test]
    fn test_welch_psd() {
        let data: Vec<f64> = (0..256).map(|i| (2.0 * PI * 10.0 * i as f64 / 256.0).sin()).collect();
        let psd = power_spectral_density_welch(&data, 64, 32);
        assert!(psd.len() > 0);
    }

    #[test]
    fn test_periodogram() {
        let data: Vec<f64> = (0..64).map(|i| (2.0 * PI * 5.0 * i as f64 / 64.0).sin()).collect();
        let p = periodogram(&data);
        assert_eq!(p.len(), 33);
    }

    #[test]
    fn test_instantaneous_freq() {
        let data: Vec<f64> = (0..64).map(|i| (2.0 * PI * 5.0 * i as f64 / 64.0).sin()).collect();
        let freq = AnalyticSignal::instantaneous_frequency(&data, 64.0);
        assert_eq!(freq.len(), 64);
    }
}
