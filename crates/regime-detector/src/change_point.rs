/// Change point detection: CUSUM, binary segmentation, PELT, online detection.

// ── CUSUM ─────────────────────────────────────────────────────────────────────

/// Cumulative sum (CUSUM) control chart for detecting a mean shift.
///
/// Returns (positive_cusum, negative_cusum) series and indices where
/// either series exceeds `threshold`.
pub fn cusum(
    series: &[f64],
    target_mean: f64,
    allowance: f64,
    threshold: f64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let n = series.len();
    let mut c_pos = vec![0.0_f64; n];
    let mut c_neg = vec![0.0_f64; n];
    let mut alarms = Vec::new();

    for i in 1..n {
        let x = series[i];
        c_pos[i] = (c_pos[i - 1] + x - target_mean - allowance).max(0.0);
        c_neg[i] = (c_neg[i - 1] - x + target_mean - allowance).max(0.0);
        if c_pos[i] > threshold || c_neg[i] > threshold {
            alarms.push(i);
        }
    }
    (c_pos, c_neg, alarms)
}

// ── Binary Segmentation ───────────────────────────────────────────────────────

fn mean_f(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn var_f(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean_f(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}

/// Compute the within-segment variance for a split at `tau` in `data[start..end]`.
fn split_cost(data: &[f64], start: usize, end: usize, tau: usize) -> f64 {
    if tau <= start || tau >= end { return f64::INFINITY; }
    let left = &data[start..tau];
    let right = &data[tau..end];
    let cost_l = var_f(left) * (tau - start) as f64;
    let cost_r = var_f(right) * (end - tau) as f64;
    cost_l + cost_r
}

/// Find the best split point within `data[start..end]`.
fn best_split(data: &[f64], start: usize, end: usize, min_seg: usize) -> Option<(usize, f64)> {
    if end - start < 2 * min_seg { return None; }
    let base_cost = var_f(&data[start..end]) * (end - start) as f64;
    let mut best_tau = start + min_seg;
    let mut best_gain = f64::NEG_INFINITY;

    for tau in (start + min_seg)..(end - min_seg) {
        let cost = split_cost(data, start, end, tau);
        let gain = base_cost - cost;
        if gain > best_gain {
            best_gain = gain;
            best_tau = tau;
        }
    }
    if best_gain > 0.0 { Some((best_tau, best_gain)) } else { None }
}

/// Binary segmentation change-point detection.
///
/// Returns sorted list of change point indices.
pub fn binary_segmentation(
    series: &[f64],
    min_segment_len: usize,
    penalty: f64,
) -> Vec<usize> {
    let n = series.len();
    if n < 2 * min_segment_len { return vec![]; }

    let mut cps: Vec<usize> = Vec::new();
    let mut segments: Vec<(usize, usize)> = vec![(0, n)];

    loop {
        let mut best: Option<(usize, usize, usize, f64)> = None; // (seg_i, tau, gain)
        for (si, &(start, end)) in segments.iter().enumerate() {
            if let Some((tau, gain)) = best_split(series, start, end, min_segment_len) {
                if gain > penalty {
                    match best {
                        None => best = Some((si, start, tau, gain)),
                        Some((_, _, _, bg)) if gain > bg => best = Some((si, start, tau, gain)),
                        _ => {}
                    }
                }
            }
        }
        match best {
            None => break,
            Some((si, start, tau, _)) => {
                let end = segments[si].1;
                segments.remove(si);
                segments.push((start, tau));
                segments.push((tau, end));
                cps.push(tau);
            }
        }
    }
    cps.sort_unstable();
    cps
}

// ── PELT (Pruned Exact Linear Time) ──────────────────────────────────────────

/// Simplified PELT algorithm for change point detection.
/// Uses piecewise constant model (penalised variance reduction).
///
/// Returns sorted change point indices.
pub fn pelt(series: &[f64], penalty: f64, min_seg: usize) -> Vec<usize> {
    let n = series.len();
    if n < 2 * min_seg { return vec![]; }

    // Precompute cumulative sums for O(1) mean/variance computation.
    let mut cum_sum = vec![0.0_f64; n + 1];
    let mut cum_sq = vec![0.0_f64; n + 1];
    for i in 0..n {
        cum_sum[i + 1] = cum_sum[i] + series[i];
        cum_sq[i + 1] = cum_sq[i] + series[i] * series[i];
    }

    let seg_cost = |start: usize, end: usize| -> f64 {
        let len = (end - start) as f64;
        if len < 1.0 { return 0.0; }
        let s = cum_sum[end] - cum_sum[start];
        let sq = cum_sq[end] - cum_sq[start];
        sq - s * s / len
    };

    let mut f = vec![f64::INFINITY; n + 1];
    let mut cp = vec![0usize; n + 1];
    f[0] = -penalty;
    let mut admissible = vec![0usize];

    for t in min_seg..=n {
        let mut best_f = f64::INFINITY;
        let mut best_s = 0usize;
        for &s in &admissible {
            if t - s < min_seg { continue; }
            let cost = f[s] + seg_cost(s, t) + penalty;
            if cost < best_f {
                best_f = cost;
                best_s = s;
            }
        }
        f[t] = best_f;
        cp[t] = best_s;

        // PELT pruning: remove admissible points where f[s] + seg_cost(s,t) + penalty >= f[t].
        admissible.retain(|&s| f[s] + seg_cost(s, t) <= f[t]);
        admissible.push(t);
    }

    // Backtrack.
    let mut cps = Vec::new();
    let mut t = n;
    while t > 0 {
        let s = cp[t];
        if s > 0 { cps.push(s); }
        t = s;
    }
    cps.sort_unstable();
    cps
}

// ── Online Change Point Detection ─────────────────────────────────────────────

/// Simple online change-point detector using a CUSUM-based alarm.
///
/// Returns the index within `history` where a change point is detected,
/// or None if no change is detected.
pub fn online_change_point(
    new_obs: f64,
    history: &[f64],
    threshold: f64,
) -> Option<usize> {
    if history.len() < 10 { return None; }

    // Estimate baseline from first half of history.
    let half = history.len() / 2;
    let mu0 = mean_f(&history[..half]);
    let sigma0 = {
        let v = var_f(&history[..half]);
        v.sqrt().max(1e-10)
    };

    // CUSUM on second half + new obs.
    let k = 0.5 * sigma0; // allowance = half sigma
    let mut c_pos = 0.0_f64;
    let mut c_neg = 0.0_f64;

    let test_series: Vec<f64> = history[half..].iter().copied().chain(std::iter::once(new_obs)).collect();
    for (i, &x) in test_series.iter().enumerate() {
        c_pos = (c_pos + x - mu0 - k).max(0.0);
        c_neg = (c_neg - x + mu0 - k).max(0.0);
        if c_pos > threshold * sigma0 || c_neg > threshold * sigma0 {
            return Some(half + i);
        }
    }
    None
}

// ── Breakpoint Validation ────────────────────────────────────────────────────

/// Given detected change points, compute per-segment statistics.
#[derive(Debug, Clone)]
pub struct SegmentStats {
    pub start: usize,
    pub end: usize,
    pub mean: f64,
    pub std: f64,
    pub n_obs: usize,
}

pub fn segment_statistics(series: &[f64], change_points: &[usize]) -> Vec<SegmentStats> {
    let n = series.len();
    let mut boundaries: Vec<usize> = std::iter::once(0)
        .chain(change_points.iter().copied())
        .chain(std::iter::once(n))
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();

    boundaries
        .windows(2)
        .map(|w| {
            let (start, end) = (w[0], w[1]);
            let seg = &series[start..end];
            let m = mean_f(seg);
            let s = var_f(seg).sqrt();
            SegmentStats { start, end, mean: m, std: s, n_obs: seg.len() }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_seg_detects_mean_shift() {
        let mut series: Vec<f64> = (0..50).map(|_| 0.01_f64).collect();
        series.extend((0..50).map(|_| -0.02_f64));
        let cps = binary_segmentation(&series, 10, 0.001);
        assert!(!cps.is_empty(), "No change points detected");
        // There should be a change point near index 50.
        let near_50 = cps.iter().any(|&cp| (cp as i64 - 50).abs() < 5);
        assert!(near_50, "Change point not near 50: {:?}", cps);
    }

    #[test]
    fn pelt_detects_mean_shift() {
        let mut series: Vec<f64> = vec![0.0; 60];
        for x in series[30..].iter_mut() { *x = 1.0; }
        let cps = pelt(&series, 10.0, 5);
        assert!(!cps.is_empty(), "pelt found no change points");
    }
}
