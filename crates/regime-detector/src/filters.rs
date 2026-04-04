/// Kalman filter, particle filter, Hodrick-Prescott, Baxter-King.

// ── Kalman Filter (1D) ────────────────────────────────────────────────────────

/// 1D Kalman filter for tracking a random-walk state.
///
/// State model: x_t = x_{t-1} + w_t,  w_t ~ N(0, process_noise)
/// Observation: z_t = x_t + v_t,      v_t ~ N(0, measurement_noise)
///
/// Returns (filtered_state, predicted_state) series.
pub fn kalman_filter_1d(
    observations: &[f64],
    process_noise: f64,
    measurement_noise: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = observations.len();
    let mut filtered = vec![0.0_f64; n];
    let mut predicted = vec![0.0_f64; n];

    if n == 0 { return (filtered, predicted); }

    // Initialise with first observation.
    let mut x_est = observations[0];
    let mut p_est = measurement_noise; // initial error covariance

    filtered[0] = x_est;
    predicted[0] = x_est;

    for i in 1..n {
        // Predict.
        let x_pred = x_est;
        let p_pred = p_est + process_noise;
        predicted[i] = x_pred;

        // Update.
        let k = p_pred / (p_pred + measurement_noise); // Kalman gain
        x_est = x_pred + k * (observations[i] - x_pred);
        p_est = (1.0 - k) * p_pred;
        filtered[i] = x_est;
    }
    (filtered, predicted)
}

// ── Particle Filter ───────────────────────────────────────────────────────────

/// Generic particle filter for 1D state estimation.
///
/// * `transition_fn` — samples x_t from p(x_t | x_{t-1}).
/// * `likelihood_fn` — evaluates p(z_t | x_t).
///
/// Returns the MMSE (weighted mean) estimate at each step.
pub fn particle_filter<T, L>(
    observations: &[f64],
    n_particles: usize,
    transition_fn: T,
    likelihood_fn: L,
) -> Vec<f64>
where
    T: Fn(f64, u64) -> f64,
    L: Fn(f64, f64) -> f64,
{
    let n = observations.len();
    let mut estimates = vec![0.0_f64; n];
    if n == 0 { return estimates; }

    // Initialise particles around first observation.
    let mut particles: Vec<f64> = (0..n_particles)
        .map(|i| {
            let noise = (i as f64 / n_particles as f64 - 0.5) * 0.1;
            observations[0] + noise
        })
        .collect();
    let mut weights = vec![1.0 / n_particles as f64; n_particles];

    estimates[0] = observations[0];

    let mut rng_state = 54321_u64;

    for t in 1..n {
        // Propagate each particle through transition.
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        for (i, p) in particles.iter_mut().enumerate() {
            let seed = rng_state.wrapping_add(i as u64 * 1000003);
            *p = transition_fn(*p, seed);
        }

        // Update weights with likelihood.
        for (i, p) in particles.iter().enumerate() {
            weights[i] *= likelihood_fn(observations[t], *p);
        }

        // Normalise weights.
        let w_sum: f64 = weights.iter().sum::<f64>().max(1e-300);
        for w in &mut weights { *w /= w_sum; }

        // Weighted mean estimate.
        estimates[t] = particles.iter().zip(weights.iter()).map(|(x, w)| x * w).sum();

        // Systematic resampling if ESS < N/2.
        let ess: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>().max(1e-300);
        if ess < n_particles as f64 / 2.0 {
            particles = systematic_resample(&particles, &weights, &mut rng_state);
            weights = vec![1.0 / n_particles as f64; n_particles];
        }
    }
    estimates
}

fn systematic_resample(particles: &[f64], weights: &[f64], state: &mut u64) -> Vec<f64> {
    let n = particles.len();
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u0 = (*state >> 11) as f64 / (1u64 << 53) as f64 / n as f64;
    let mut cumsum = 0.0_f64;
    let mut j = 0usize;
    let mut new_particles = Vec::with_capacity(n);

    for i in 0..n {
        let threshold = u0 + i as f64 / n as f64;
        while cumsum < threshold && j < n - 1 {
            cumsum += weights[j];
            j += 1;
        }
        new_particles.push(particles[j.min(n - 1)]);
    }
    new_particles
}

// ── Hodrick-Prescott Filter ────────────────────────────────────────────────────

/// Hodrick-Prescott filter: decompose `series` into (trend, cycle).
///
/// Solves: min_tau sum[(y_t - tau_t)^2] + lambda * sum[(tau_{t+1} - 2tau_t + tau_{t-1})^2]
/// via the closed-form tridiagonal system.
pub fn hp_filter(series: &[f64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let n = series.len();
    if n < 3 {
        return (series.to_vec(), vec![0.0; n]);
    }

    // Build the (n×n) system (I + lambda * D'D) tau = y
    // where D is the second-difference operator.
    // For efficiency, build a pentadiagonal system and solve with banded Gaussian elimination.
    let mut a = vec![0.0_f64; n]; // diagonal
    let mut b = vec![0.0_f64; n]; // super/sub diagonal 1
    let mut c = vec![0.0_f64; n]; // super/sub diagonal 2

    for i in 0..n {
        a[i] = 1.0 + 6.0 * lambda;
        if i == 0 || i == n - 1 {
            a[i] = 1.0 + lambda;
        } else if i == 1 || i == n - 2 {
            a[i] = 1.0 + 5.0 * lambda;
        }
    }
    for i in 0..n - 1 {
        b[i] = if i == 0 || i == n - 2 { -2.0 * lambda } else { -4.0 * lambda };
    }
    for i in 0..n - 2 {
        c[i] = lambda;
    }

    // Solve the pentadiagonal system via Thomas algorithm (banded).
    let trend = solve_pentadiagonal(&a, &b, &c, series);
    let cycle: Vec<f64> = series.iter().zip(trend.iter()).map(|(y, t)| y - t).collect();
    (trend, cycle)
}

/// Solve a symmetric pentadiagonal system Ax = d.
/// a = main diagonal, b = ±1 off-diagonal, c = ±2 off-diagonal.
fn solve_pentadiagonal(a: &[f64], b: &[f64], c: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    let mut c = c.to_vec();
    let mut d = rhs.to_vec();

    // Forward sweep.
    for i in 0..n {
        if i >= 1 {
            let factor1 = b[i - 1] / a[i - 1];
            a[i] -= factor1 * b[i - 1];
            d[i] -= factor1 * d[i - 1];
            if i >= 2 {
                // Also eliminate c component.
                if a[i - 1] != 0.0 {
                    let factor2 = if i >= 2 { c[i - 2] / a[i - 2] } else { 0.0 };
                    a[i] -= factor2 * c[i - 2];
                    d[i] -= factor2 * d[i - 2];
                }
            }
        }
        if a[i].abs() < 1e-14 { a[i] = 1e-14; }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    x[n - 1] = d[n - 1] / a[n - 1];
    if n >= 2 {
        x[n - 2] = (d[n - 2] - b[n - 2] * x[n - 1]) / a[n - 2].max(1e-14);
    }
    for i in (0..n.saturating_sub(2)).rev() {
        let mut xi = d[i];
        if i + 1 < n { xi -= b[i] * x[i + 1]; }
        if i + 2 < n { xi -= c[i] * x[i + 2]; }
        x[i] = xi / a[i].max(1e-14);
    }
    x
}

// ── Baxter-King Band-Pass Filter ──────────────────────────────────────────────

/// Baxter-King band-pass filter.
///
/// Extracts cycles with periods between `low_freq` and `high_freq` using
/// a symmetric moving average of order `2K+1`.
///
/// Typical parameters for quarterly data: low=6, high=32, K=12.
pub fn bk_filter(series: &[f64], low_freq: f64, high_freq: f64, k: usize) -> Vec<f64> {
    let n = series.len();
    if n <= 2 * k {
        return vec![0.0; n];
    }

    // Compute ideal bandpass weights.
    let omega_low = 2.0 * std::f64::consts::PI / high_freq;
    let omega_high = 2.0 * std::f64::consts::PI / low_freq;

    let mut b: Vec<f64> = (0..=k)
        .map(|h| {
            if h == 0 {
                (omega_high - omega_low) / std::f64::consts::PI
            } else {
                let hf = h as f64;
                ((omega_high * hf).sin() - (omega_low * hf).sin()) / (std::f64::consts::PI * hf)
            }
        })
        .collect();

    // Leakage-free adjustment: ensure sum of all weights = 0.
    let sum: f64 = b[0] + 2.0 * b[1..].iter().sum::<f64>();
    let correction = sum / (2 * k + 1) as f64;
    for bi in &mut b { *bi -= correction; }

    // Apply the symmetric filter (only valid for indices k..n-k).
    let mut result = vec![0.0_f64; n];
    for t in k..n - k {
        let mut val = b[0] * series[t];
        for h in 1..=k {
            val += b[h] * (series[t - h] + series[t + h]);
        }
        result[t] = val;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_smooths_noisy_constant() {
        let true_val = 5.0_f64;
        let noisy: Vec<f64> = (0..50).map(|i| true_val + (i as f64 * 0.3).sin()).collect();
        let (filtered, _) = kalman_filter_1d(&noisy, 0.001, 0.5);
        let last = filtered.last().unwrap();
        assert!((last - true_val).abs() < 1.0, "last={last}");
    }

    #[test]
    fn hp_filter_trend_length() {
        let series: Vec<f64> = (0..100).map(|i| i as f64 * 0.01 + (i as f64).sin() * 0.05).collect();
        let (trend, cycle) = hp_filter(&series, 1600.0);
        assert_eq!(trend.len(), series.len());
        assert_eq!(cycle.len(), series.len());
    }

    #[test]
    fn bk_filter_output_length() {
        let series: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let cycle = bk_filter(&series, 6.0, 32.0, 12);
        assert_eq!(cycle.len(), series.len());
    }
}
