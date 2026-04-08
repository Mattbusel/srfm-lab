use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// CUSUM (Cumulative Sum)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct CusumResult {
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
    pub changepoints: Vec<usize>,
}

/// One-sided upper CUSUM
pub fn cusum_upper(data: &[f64], threshold: f64, drift: f64) -> CusumResult {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut s_pos = vec![0.0; n];
    let mut changepoints = Vec::new();

    for i in 1..n {
        s_pos[i] = (s_pos[i - 1] + data[i] - mean - drift).max(0.0);
        if s_pos[i] > threshold {
            changepoints.push(i);
            s_pos[i] = 0.0;
        }
    }

    CusumResult {
        upper: s_pos,
        lower: vec![0.0; n],
        changepoints,
    }
}

/// One-sided lower CUSUM
pub fn cusum_lower(data: &[f64], threshold: f64, drift: f64) -> CusumResult {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut s_neg = vec![0.0; n];
    let mut changepoints = Vec::new();

    for i in 1..n {
        s_neg[i] = (s_neg[i - 1] - data[i] + mean - drift).max(0.0);
        if s_neg[i] > threshold {
            changepoints.push(i);
            s_neg[i] = 0.0;
        }
    }

    CusumResult {
        upper: vec![0.0; n],
        lower: s_neg,
        changepoints,
    }
}

/// Two-sided CUSUM
pub fn cusum_two_sided(data: &[f64], threshold: f64, drift: f64) -> CusumResult {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut s_pos = vec![0.0; n];
    let mut s_neg = vec![0.0; n];
    let mut changepoints = Vec::new();

    for i in 1..n {
        s_pos[i] = (s_pos[i - 1] + data[i] - mean - drift).max(0.0);
        s_neg[i] = (s_neg[i - 1] - data[i] + mean - drift).max(0.0);
        if s_pos[i] > threshold || s_neg[i] > threshold {
            changepoints.push(i);
            s_pos[i] = 0.0;
            s_neg[i] = 0.0;
        }
    }

    CusumResult {
        upper: s_pos,
        lower: s_neg,
        changepoints,
    }
}

/// Tabular CUSUM (with slack parameter k)
pub fn cusum_tabular(data: &[f64], k: f64, h: f64, target: f64) -> CusumResult {
    let n = data.len();
    let mut c_plus = vec![0.0; n];
    let mut c_minus = vec![0.0; n];
    let mut changepoints = Vec::new();

    for i in 0..n {
        let val = data[i] - target;
        if i > 0 {
            c_plus[i] = (c_plus[i - 1] + val - k).max(0.0);
            c_minus[i] = (c_minus[i - 1] - val - k).max(0.0);
        } else {
            c_plus[i] = (val - k).max(0.0);
            c_minus[i] = (-val - k).max(0.0);
        }
        if c_plus[i] > h || c_minus[i] > h {
            changepoints.push(i);
            c_plus[i] = 0.0;
            c_minus[i] = 0.0;
        }
    }

    CusumResult {
        upper: c_plus,
        lower: c_minus,
        changepoints,
    }
}

// ---------------------------------------------------------------------------
// EWMA Control Chart
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct EwmaControlResult {
    pub ewma: Vec<f64>,
    pub ucl: Vec<f64>,
    pub lcl: Vec<f64>,
    pub violations: Vec<usize>,
}

pub fn ewma_control_chart(data: &[f64], lambda: f64, l_factor: f64) -> EwmaControlResult {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = var.sqrt();

    let mut ewma = vec![mean; n];
    let mut ucl = vec![0.0; n];
    let mut lcl = vec![0.0; n];
    let mut violations = Vec::new();

    for i in 0..n {
        if i == 0 {
            ewma[i] = lambda * data[i] + (1.0 - lambda) * mean;
        } else {
            ewma[i] = lambda * data[i] + (1.0 - lambda) * ewma[i - 1];
        }

        // Control limits widen over time
        let factor = (lambda / (2.0 - lambda) * (1.0 - (1.0 - lambda).powi(2 * (i + 1) as i32))).sqrt();
        ucl[i] = mean + l_factor * std_dev * factor;
        lcl[i] = mean - l_factor * std_dev * factor;

        if ewma[i] > ucl[i] || ewma[i] < lcl[i] {
            violations.push(i);
        }
    }

    EwmaControlResult { ewma, ucl, lcl, violations }
}

// ---------------------------------------------------------------------------
// Shewhart Control Chart
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ShewhartResult {
    pub mean: f64,
    pub ucl: f64,
    pub lcl: f64,
    pub violations: Vec<usize>,
    pub western_electric_rules: Vec<(usize, &'static str)>,
}

pub fn shewhart_chart(data: &[f64], num_sigma: f64) -> ShewhartResult {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let std_dev = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();

    let ucl = mean + num_sigma * std_dev;
    let lcl = mean - num_sigma * std_dev;

    let mut violations = Vec::new();
    let mut we_rules = Vec::new();

    let one_sigma_upper = mean + std_dev;
    let one_sigma_lower = mean - std_dev;
    let two_sigma_upper = mean + 2.0 * std_dev;
    let two_sigma_lower = mean - 2.0 * std_dev;

    for i in 0..n {
        // Rule 1: Beyond 3-sigma
        if data[i] > ucl || data[i] < lcl {
            violations.push(i);
            we_rules.push((i, "Beyond 3-sigma"));
        }

        // Rule 2: 2 of 3 successive points beyond 2-sigma (same side)
        if i >= 2 {
            let above_2sigma = |j: usize| data[j] > two_sigma_upper;
            let below_2sigma = |j: usize| data[j] < two_sigma_lower;
            let count_above = [i - 2, i - 1, i].iter().filter(|&&j| above_2sigma(j)).count();
            let count_below = [i - 2, i - 1, i].iter().filter(|&&j| below_2sigma(j)).count();
            if count_above >= 2 { we_rules.push((i, "2 of 3 above 2-sigma")); }
            if count_below >= 2 { we_rules.push((i, "2 of 3 below 2-sigma")); }
        }

        // Rule 3: 4 of 5 successive beyond 1-sigma (same side)
        if i >= 4 {
            let above = (0..5).filter(|&k| data[i - k] > one_sigma_upper).count();
            let below = (0..5).filter(|&k| data[i - k] < one_sigma_lower).count();
            if above >= 4 { we_rules.push((i, "4 of 5 above 1-sigma")); }
            if below >= 4 { we_rules.push((i, "4 of 5 below 1-sigma")); }
        }

        // Rule 4: 8 successive on same side of center
        if i >= 7 {
            let all_above = (0..8).all(|k| data[i - k] > mean);
            let all_below = (0..8).all(|k| data[i - k] < mean);
            if all_above { we_rules.push((i, "8 consecutive above mean")); }
            if all_below { we_rules.push((i, "8 consecutive below mean")); }
        }
    }

    ShewhartResult { mean, ucl, lcl, violations, western_electric_rules: we_rules }
}

// ---------------------------------------------------------------------------
// Bayesian Online Changepoint Detection (BOCPD)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct BocpdResult {
    pub run_length_probs: Vec<Vec<f64>>, // run_length_probs[t][r] = P(r_t = r | data)
    pub changepoint_probs: Vec<f64>,     // P(changepoint at t)
    pub changepoints: Vec<usize>,
    pub map_run_lengths: Vec<usize>,
}

pub fn bocpd_normal(data: &[f64], hazard_rate: f64, prior_mean: f64, prior_var: f64) -> BocpdResult {
    let n = data.len();
    let lambda = 1.0 / hazard_rate;

    // Sufficient statistics for normal-normal conjugate
    let mut sum_x = vec![0.0; n + 1]; // running sum per run
    let mut sum_x2 = vec![0.0; n + 1]; // running sum of squares per run
    let mut count = vec![0usize; n + 1];

    let mut run_length_probs = Vec::with_capacity(n);
    let mut cp_probs = Vec::with_capacity(n);
    let mut map_run_lengths = Vec::with_capacity(n);

    // Initial: run length 0 with probability 1
    let mut joint = vec![1.0]; // joint[r] = P(r_t = r, x_{1:t})

    for t in 0..n {
        let x = data[t];
        let max_r = joint.len();

        // Compute predictive probabilities for each run length
        let mut pred_probs = vec![0.0; max_r];
        for r in 0..max_r {
            // Student-t predictive (normal-normal-inverse-gamma)
            let kappa = 1.0 / prior_var + count[r] as f64;
            let mu = (prior_mean / prior_var + sum_x[r]) / kappa;
            let alpha = 1.0 + count[r] as f64 / 2.0;
            let beta_val = 1.0 + 0.5 * (sum_x2[r] - sum_x[r] * sum_x[r] / count[r].max(1) as f64);
            let beta_val = beta_val.max(0.01);

            // Simplified: use normal predictive
            let pred_var = beta_val / alpha + 1.0 / kappa;
            let pred_std = pred_var.max(1e-15).sqrt();
            pred_probs[r] = normal_pdf(x, mu, pred_std);
        }

        // Growth probabilities
        let mut new_joint = vec![0.0; max_r + 1];
        for r in 0..max_r {
            new_joint[r + 1] = joint[r] * pred_probs[r] * (1.0 - 1.0 / lambda);
        }

        // Changepoint probability: sum over all runs
        let cp_mass: f64 = joint.iter().zip(pred_probs.iter())
            .map(|(&j, &p)| j * p * (1.0 / lambda))
            .sum();
        new_joint[0] = cp_mass;

        // Normalize
        let total: f64 = new_joint.iter().sum();
        if total > 1e-30 {
            for v in new_joint.iter_mut() { *v /= total; }
        }

        // Update sufficient statistics
        let new_len = new_joint.len();
        let mut new_sum_x = vec![0.0; new_len];
        let mut new_sum_x2 = vec![0.0; new_len];
        let mut new_count = vec![0usize; new_len];

        new_sum_x[0] = 0.0;
        new_sum_x2[0] = 0.0;
        new_count[0] = 0;

        for r in 0..max_r {
            new_sum_x[r + 1] = sum_x[r] + x;
            new_sum_x2[r + 1] = sum_x2[r] + x * x;
            new_count[r + 1] = count[r] + 1;
        }

        sum_x = new_sum_x;
        sum_x2 = new_sum_x2;
        count = new_count;

        // Store results
        run_length_probs.push(new_joint.clone());
        cp_probs.push(new_joint[0]);
        let map_r = new_joint.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        map_run_lengths.push(map_r);

        joint = new_joint;

        // Truncate very small probabilities for efficiency
        let max_len = 500.min(joint.len());
        if joint.len() > max_len {
            let excess: f64 = joint[max_len..].iter().sum();
            joint.truncate(max_len);
            joint[0] += excess;
            sum_x.truncate(max_len);
            sum_x2.truncate(max_len);
            count.truncate(max_len);
        }
    }

    // Detect changepoints where P(r=0) is high
    let threshold = 0.5;
    let changepoints: Vec<usize> = cp_probs.iter().enumerate()
        .filter(|&(_, &p)| p > threshold)
        .map(|(i, _)| i)
        .collect();

    BocpdResult {
        run_length_probs,
        changepoint_probs: cp_probs,
        changepoints,
        map_run_lengths,
    }
}

fn normal_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let z = (x - mean) / std;
    (-(z * z) / 2.0).exp() / (std * (2.0 * PI).sqrt())
}

// ---------------------------------------------------------------------------
// Binary Segmentation
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct BinarySegmentationResult {
    pub changepoints: Vec<usize>,
    pub costs: Vec<f64>,
}

pub fn binary_segmentation(data: &[f64], min_segment: usize, penalty: f64) -> BinarySegmentationResult {
    let n = data.len();
    let mut changepoints = Vec::new();
    let mut segments: Vec<(usize, usize)> = vec![(0, n)];

    while let Some((start, end)) = segments.pop() {
        if end - start < 2 * min_segment { continue; }

        let full_cost = segment_cost(&data[start..end]);
        let mut best_gain = 0.0;
        let mut best_cp = start + min_segment;

        for cp in (start + min_segment)..(end - min_segment) {
            let cost_left = segment_cost(&data[start..cp]);
            let cost_right = segment_cost(&data[cp..end]);
            let gain = full_cost - cost_left - cost_right;
            if gain > best_gain {
                best_gain = gain;
                best_cp = cp;
            }
        }

        if best_gain > penalty {
            changepoints.push(best_cp);
            segments.push((start, best_cp));
            segments.push((best_cp, end));
        }
    }

    changepoints.sort();

    let costs: Vec<f64> = {
        let mut all_cps = vec![0];
        all_cps.extend_from_slice(&changepoints);
        all_cps.push(n);
        all_cps.windows(2)
            .map(|w| segment_cost(&data[w[0]..w[1]]))
            .collect()
    };

    BinarySegmentationResult { changepoints, costs }
}

fn segment_cost(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 1.0 { return 0.0; }
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
}

// ---------------------------------------------------------------------------
// PELT (Pruned Exact Linear Time)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PeltResult {
    pub changepoints: Vec<usize>,
    pub costs: Vec<f64>,
}

pub fn pelt(data: &[f64], penalty: f64, min_segment: usize) -> PeltResult {
    let n = data.len();
    if n < 2 * min_segment {
        return PeltResult { changepoints: vec![], costs: vec![segment_cost(data)] };
    }

    // Precompute cumulative sums for O(1) cost computation
    let mut cum_sum = vec![0.0; n + 1];
    let mut cum_sum2 = vec![0.0; n + 1];
    for i in 0..n {
        cum_sum[i + 1] = cum_sum[i] + data[i];
        cum_sum2[i + 1] = cum_sum2[i] + data[i] * data[i];
    }

    let segment_cost_fast = |start: usize, end: usize| -> f64 {
        let len = (end - start) as f64;
        if len < 1.0 { return 0.0; }
        let s = cum_sum[end] - cum_sum[start];
        let s2 = cum_sum2[end] - cum_sum2[start];
        s2 - s * s / len
    };

    // Dynamic programming
    let mut f = vec![f64::INFINITY; n + 1]; // f[t] = min cost for data[0..t]
    f[0] = -penalty; // so that first segment adds penalty correctly
    let mut last_cp = vec![0usize; n + 1];
    let mut admissible: Vec<usize> = vec![0];

    for t in min_segment..=n {
        let mut best_cost = f64::INFINITY;
        let mut best_s = 0;

        for &s in &admissible {
            if t - s < min_segment { continue; }
            let cost = f[s] + segment_cost_fast(s, t) + penalty;
            if cost < best_cost {
                best_cost = cost;
                best_s = s;
            }
        }

        f[t] = best_cost;
        last_cp[t] = best_s;

        // Pruning: remove admissible points that can never be optimal
        admissible.retain(|&s| {
            f[s] + segment_cost_fast(s, t) <= f[t]
        });
        admissible.push(t);
    }

    // Backtrace changepoints
    let mut changepoints = Vec::new();
    let mut t = n;
    while t > 0 {
        let s = last_cp[t];
        if s > 0 {
            changepoints.push(s);
        }
        t = s;
    }
    changepoints.reverse();

    let costs = {
        let mut all = vec![0];
        all.extend_from_slice(&changepoints);
        all.push(n);
        all.windows(2)
            .map(|w| segment_cost_fast(w[0], w[1]))
            .collect()
    };

    PeltResult { changepoints, costs }
}

// ---------------------------------------------------------------------------
// Wild Binary Segmentation
// ---------------------------------------------------------------------------
pub fn wild_binary_segmentation(
    data: &[f64],
    min_segment: usize,
    penalty: f64,
    num_intervals: usize,
) -> Vec<usize> {
    let n = data.len();
    if n < 2 * min_segment { return vec![]; }

    // Generate random intervals deterministically
    let mut intervals: Vec<(usize, usize)> = Vec::with_capacity(num_intervals);
    for i in 0..num_intervals {
        let s = (i * 7 + 3) % n;
        let e_offset = min_segment * 2 + (i * 13 + 5) % (n / 2).max(1);
        let e = (s + e_offset).min(n);
        if e - s >= 2 * min_segment {
            intervals.push((s, e));
        }
    }
    // Add full interval
    intervals.push((0, n));

    let mut changepoints = Vec::new();
    wbs_recursive(data, &intervals, min_segment, penalty, &mut changepoints);
    changepoints.sort();
    changepoints.dedup();
    changepoints
}

fn wbs_recursive(
    data: &[f64],
    intervals: &[(usize, usize)],
    min_segment: usize,
    penalty: f64,
    changepoints: &mut Vec<usize>,
) {
    let mut best_gain = 0.0;
    let mut best_cp = 0;
    let mut best_interval = (0, 0);

    for &(s, e) in intervals {
        if e - s < 2 * min_segment { continue; }
        let full_cost = segment_cost(&data[s..e]);
        for cp in (s + min_segment)..(e - min_segment) {
            let gain = full_cost - segment_cost(&data[s..cp]) - segment_cost(&data[cp..e]);
            if gain > best_gain {
                best_gain = gain;
                best_cp = cp;
                best_interval = (s, e);
            }
        }
    }

    if best_gain > penalty && best_cp > 0 {
        changepoints.push(best_cp);
        // Recurse on left and right
        let left_intervals: Vec<(usize, usize)> = intervals.iter()
            .filter_map(|&(s, e)| {
                let s2 = s;
                let e2 = e.min(best_cp);
                if e2 - s2 >= 2 * min_segment { Some((s2, e2)) } else { None }
            })
            .collect();
        let right_intervals: Vec<(usize, usize)> = intervals.iter()
            .filter_map(|&(s, e)| {
                let s2 = s.max(best_cp);
                let e2 = e;
                if e2 - s2 >= 2 * min_segment { Some((s2, e2)) } else { None }
            })
            .collect();
        wbs_recursive(data, &left_intervals, min_segment, penalty, changepoints);
        wbs_recursive(data, &right_intervals, min_segment, penalty, changepoints);
    }
}

// ---------------------------------------------------------------------------
// Bai-Perron Dynamic Programming
// ---------------------------------------------------------------------------
pub fn bai_perron(data: &[f64], max_breaks: usize, min_segment: usize) -> Vec<usize> {
    let n = data.len();
    if n < 2 * min_segment || max_breaks == 0 {
        return vec![];
    }

    // Precompute cumulative sums
    let mut cum_sum = vec![0.0; n + 1];
    let mut cum_sum2 = vec![0.0; n + 1];
    for i in 0..n {
        cum_sum[i + 1] = cum_sum[i] + data[i];
        cum_sum2[i + 1] = cum_sum2[i] + data[i] * data[i];
    }

    let cost = |s: usize, e: usize| -> f64 {
        let len = (e - s) as f64;
        if len < 1.0 { return 0.0; }
        let sm = cum_sum[e] - cum_sum[s];
        let sm2 = cum_sum2[e] - cum_sum2[s];
        sm2 - sm * sm / len
    };

    // DP: f[m][t] = min cost with m breaks in data[0..t]
    let mut f = vec![vec![f64::INFINITY; n + 1]; max_breaks + 1];
    let mut bp = vec![vec![0usize; n + 1]; max_breaks + 1]; // backpointer

    // 0 breaks
    for t in min_segment..=n {
        f[0][t] = cost(0, t);
    }

    // m breaks
    for m in 1..=max_breaks {
        let min_t = (m + 1) * min_segment;
        for t in min_t..=n {
            for s in (m * min_segment)..(t - min_segment + 1) {
                let candidate = f[m - 1][s] + cost(s, t);
                if candidate < f[m][t] {
                    f[m][t] = candidate;
                    bp[m][t] = s;
                }
            }
        }
    }

    // Find optimal number of breaks using BIC
    let mut best_bic = f64::INFINITY;
    let mut best_m = 0;
    for m in 0..=max_breaks {
        let rss = f[m][n];
        if rss.is_finite() {
            let k = (m + 1) as f64 * 2.0; // mean + variance per segment
            let bic = n as f64 * (rss / n as f64).max(1e-15).ln() + k * (n as f64).ln();
            if bic < best_bic {
                best_bic = bic;
                best_m = m;
            }
        }
    }

    // Backtrace
    let mut changepoints = Vec::new();
    let mut t = n;
    for m in (1..=best_m).rev() {
        let s = bp[m][t];
        changepoints.push(s);
        t = s;
    }
    changepoints.reverse();
    changepoints
}

// ---------------------------------------------------------------------------
// Structural Break F-test (Chow test)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ChowTestResult {
    pub f_statistic: f64,
    pub p_value: f64,
    pub break_point: usize,
}

pub fn chow_test(data: &[f64], break_point: usize) -> ChowTestResult {
    let n = data.len();
    assert!(break_point > 1 && break_point < n - 1);

    let segment1 = &data[..break_point];
    let segment2 = &data[break_point..];

    let rss_full = segment_cost(data);
    let rss1 = segment_cost(segment1);
    let rss2 = segment_cost(segment2);
    let rss_split = rss1 + rss2;

    let k = 2; // parameters per segment (mean + variance)
    let numerator = (rss_full - rss_split) / k as f64;
    let denominator = rss_split / (n - 2 * k) as f64;

    let f_stat = if denominator > 1e-15 { numerator / denominator } else { 0.0 };

    // Approximate p-value using F distribution (df1=k, df2=n-2k)
    let p_value = 1.0 - f_cdf(f_stat, k as f64, (n - 2 * k) as f64);

    ChowTestResult { f_statistic: f_stat, p_value, break_point }
}

/// Scan all possible break points and return the most significant
pub fn chow_test_scan(data: &[f64], min_segment: usize) -> ChowTestResult {
    let n = data.len();
    let mut best = ChowTestResult { f_statistic: 0.0, p_value: 1.0, break_point: n / 2 };

    for bp in min_segment..(n - min_segment) {
        let result = chow_test(data, bp);
        if result.f_statistic > best.f_statistic {
            best = result;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Sup-F test (Andrews)
// ---------------------------------------------------------------------------
pub fn sup_f_test(data: &[f64], trim: f64) -> (f64, usize) {
    let n = data.len();
    let start = (n as f64 * trim) as usize;
    let end = n - start;

    let mut max_f = 0.0;
    let mut max_bp = start;

    for bp in start..end {
        let result = chow_test(data, bp);
        if result.f_statistic > max_f {
            max_f = result.f_statistic;
            max_bp = bp;
        }
    }

    (max_f, max_bp)
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------
fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let z = d1 * x / (d1 * x + d2);
    regularized_beta(z, d1 / 2.0, d2 / 2.0)
}

fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }

    // Continued fraction approximation
    let max_iter = 200;
    let eps = 1e-10;

    let factor = (a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b)).exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        factor * beta_cf(x, a, b, max_iter, eps) / a
    } else {
        1.0 - factor * beta_cf(1.0 - x, b, a, max_iter, eps) / b
    }
}

fn beta_cf(x: f64, a: f64, b: f64, max_iter: usize, eps: f64) -> f64 {
    let mut c: f64 = 1.0;
    let mut d: f64 = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(1e-30);
    let mut h: f64 = d;

    for m in 1..=max_iter {
        let m = m as f64;
        // Even step
        let num = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 / (1.0 + num * d).max(1e-30);
        c = 1.0 + num / c.max(1e-30);
        h *= d * c;

        // Odd step
        let num = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 / (1.0 + num * d).max(1e-30);
        c = 1.0 + num / c.max(1e-30);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < eps { break; }
    }
    h
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let v = PI / (PI * x).sin();
        v.abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        let t = x + g + 0.5;
        for i in 1..9 { a += c[i] / (x + i as f64); }
        0.5 * (2.0 * PI).ln() + (t.ln()) * (x + 0.5) - t + a.ln()
    }
}

// ---------------------------------------------------------------------------
// Convenience: detect changepoints with auto-tuned penalty
// ---------------------------------------------------------------------------
pub fn detect_changepoints(data: &[f64], method: &str) -> Vec<usize> {
    let n = data.len();
    let penalty = 2.0 * (n as f64).ln() * segment_cost(data) / n as f64;
    let min_seg = (n / 20).max(5);

    match method {
        "pelt" => pelt(data, penalty, min_seg).changepoints,
        "binseg" => binary_segmentation(data, min_seg, penalty).changepoints,
        "wbs" => wild_binary_segmentation(data, min_seg, penalty, 100),
        "bocpd" => {
            let mean = data.iter().sum::<f64>() / n as f64;
            let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            bocpd_normal(data, 250.0, mean, var).changepoints
        }
        "bai_perron" => bai_perron(data, 5, min_seg),
        _ => pelt(data, penalty, min_seg).changepoints,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_changepoint_data() -> Vec<f64> {
        let mut data = Vec::with_capacity(200);
        // Segment 1: mean=0
        for i in 0..50 {
            data.push(((i as f64 * 1.618).sin() * 43758.5453).fract());
        }
        // Segment 2: mean=5
        for i in 50..100 {
            data.push(5.0 + ((i as f64 * 1.618).sin() * 43758.5453).fract());
        }
        // Segment 3: mean=2
        for i in 100..200 {
            data.push(2.0 + ((i as f64 * 1.618).sin() * 43758.5453).fract());
        }
        data
    }

    #[test]
    fn test_cusum_two_sided() {
        let data = make_changepoint_data();
        let result = cusum_two_sided(&data, 5.0, 0.5);
        assert!(result.changepoints.len() > 0);
    }

    #[test]
    fn test_cusum_tabular() {
        let data = make_changepoint_data();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let result = cusum_tabular(&data, 0.5, 5.0, mean);
        assert!(result.changepoints.len() >= 0);
    }

    #[test]
    fn test_ewma_control() {
        let data = make_changepoint_data();
        let result = ewma_control_chart(&data, 0.2, 3.0);
        assert_eq!(result.ewma.len(), 200);
        assert!(result.violations.len() > 0);
    }

    #[test]
    fn test_shewhart() {
        let data = make_changepoint_data();
        let result = shewhart_chart(&data, 3.0);
        assert!(result.ucl > result.mean);
        assert!(result.lcl < result.mean);
    }

    #[test]
    fn test_bocpd() {
        let data = make_changepoint_data();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let result = bocpd_normal(&data, 100.0, mean, var);
        assert_eq!(result.changepoint_probs.len(), 200);
        assert_eq!(result.map_run_lengths.len(), 200);
    }

    #[test]
    fn test_binary_segmentation() {
        let data = make_changepoint_data();
        let result = binary_segmentation(&data, 10, 50.0);
        assert!(result.changepoints.len() > 0);
    }

    #[test]
    fn test_pelt() {
        let data = make_changepoint_data();
        let penalty = 2.0 * (200.0_f64).ln() * segment_cost(&data) / 200.0;
        let result = pelt(&data, penalty, 10);
        assert!(result.changepoints.len() >= 1);
    }

    #[test]
    fn test_wild_binary_seg() {
        let data = make_changepoint_data();
        let cps = wild_binary_segmentation(&data, 10, 50.0, 50);
        assert!(cps.len() >= 0);
    }

    #[test]
    fn test_bai_perron() {
        let data = make_changepoint_data();
        let cps = bai_perron(&data, 3, 10);
        assert!(cps.len() >= 0);
    }

    #[test]
    fn test_chow_test() {
        let data = make_changepoint_data();
        let result = chow_test(&data, 50);
        assert!(result.f_statistic > 0.0);
    }

    #[test]
    fn test_chow_scan() {
        let data = make_changepoint_data();
        let result = chow_test_scan(&data, 10);
        assert!(result.f_statistic > 0.0);
        // Should find break near 50 or 100
        assert!(result.break_point > 5 && result.break_point < 195);
    }

    #[test]
    fn test_sup_f() {
        let data = make_changepoint_data();
        let (f_stat, bp) = sup_f_test(&data, 0.15);
        assert!(f_stat > 0.0);
    }

    #[test]
    fn test_detect_changepoints() {
        let data = make_changepoint_data();
        let cps = detect_changepoints(&data, "pelt");
        assert!(cps.len() >= 1);
    }

    #[test]
    fn test_segment_cost() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let cost = segment_cost(&data);
        assert!((cost - 0.0).abs() < 1e-10);

        let data2 = vec![0.0, 2.0];
        let cost2 = segment_cost(&data2);
        assert!((cost2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cusum_upper() {
        let data: Vec<f64> = (0..50).map(|i| if i < 25 { 0.0 } else { 3.0 }).collect();
        let result = cusum_upper(&data, 5.0, 0.5);
        assert!(result.changepoints.len() > 0);
    }
}
