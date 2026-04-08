// statistics.rs — Descriptive statistics, correlation, covariance, online stats, bootstrap

/// Descriptive statistics for a sample
pub struct Descriptive {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
}

impl Descriptive {
    /// Compute all descriptive statistics from a sample
    pub fn compute(data: &[f64]) -> Self {
        let n = data.len();
        assert!(n > 0, "empty data");
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = data.iter().sum::<f64>() / n as f64;
        let m2: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        let m3: f64 = data.iter().map(|x| (x - mean).powi(3)).sum();
        let m4: f64 = data.iter().map(|x| (x - mean).powi(4)).sum();

        let variance = m2 / (n as f64 - 1.0).max(1.0);
        let std_dev = variance.sqrt();

        let skewness = if std_dev > 1e-15 {
            (m3 / n as f64) / (m2 / n as f64).powf(1.5)
        } else { 0.0 };

        let kurtosis = if std_dev > 1e-15 {
            (m4 / n as f64) / (m2 / n as f64).powi(2) - 3.0
        } else { 0.0 };

        let median = percentile_sorted(&sorted, 0.5);
        let q1 = percentile_sorted(&sorted, 0.25);
        let q3 = percentile_sorted(&sorted, 0.75);

        Self {
            n,
            mean,
            variance,
            std_dev,
            skewness,
            kurtosis,
            min: sorted[0],
            max: sorted[n - 1],
            median,
            q1,
            q3,
            iqr: q3 - q1,
        }
    }
}

/// Mean of a slice
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Variance (sample, Bessel-corrected)
pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 { return 0.0; }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

/// Standard deviation (sample)
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Population variance
pub fn variance_pop(data: &[f64]) -> f64 {
    let n = data.len();
    if n == 0 { return 0.0; }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64
}

/// Skewness (Fisher)
pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return 0.0; }
    let nf = n as f64;
    let g1: f64 = data.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / nf;
    let adj = (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
    g1 * adj
}

/// Kurtosis (excess, Fisher)
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 4 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return 0.0; }
    let nf = n as f64;
    let g2: f64 = data.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / nf;
    g2 - 3.0
}

/// Median
pub fn median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, 0.5)
}

/// Percentile (0..1) from sorted data using linear interpolation
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n == 1 { return sorted[0]; }
    let idx = p * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo >= n { return sorted[n - 1]; }
    if hi >= n { return sorted[n - 1]; }
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Percentile from unsorted data
pub fn percentile(data: &[f64], p: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, p)
}

/// Mode (most frequent value, binned for continuous data)
pub fn mode_binned(data: &[f64], bins: usize) -> f64 {
    if data.is_empty() { return 0.0; }
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-15 { return min; }
    let bin_width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for &x in data {
        let b = ((x - min) / bin_width).floor() as usize;
        let b = b.min(bins - 1);
        counts[b] += 1;
    }
    let max_bin = counts.iter().enumerate().max_by_key(|&(_, c)| c).unwrap().0;
    min + (max_bin as f64 + 0.5) * bin_width
}

/// Geometric mean
pub fn geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let log_sum: f64 = data.iter().map(|x| x.ln()).sum();
    (log_sum / data.len() as f64).exp()
}

/// Harmonic mean
pub fn harmonic_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let inv_sum: f64 = data.iter().map(|x| 1.0 / x).sum();
    data.len() as f64 / inv_sum
}

/// Trimmed mean (trim fraction from each end)
pub fn trimmed_mean(data: &[f64], trim: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let k = (n as f64 * trim).floor() as usize;
    if 2 * k >= n { return mean(data); }
    let trimmed = &sorted[k..n - k];
    mean(trimmed)
}

/// Winsorized mean
pub fn winsorized_mean(data: &[f64], trim: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let k = (n as f64 * trim).floor() as usize;
    if k == 0 || n < 3 { return mean(data); }
    let lo = sorted[k];
    let hi = sorted[n - 1 - k];
    let mut result = sorted.clone();
    for x in &mut result {
        if *x < lo { *x = lo; }
        if *x > hi { *x = hi; }
    }
    mean(&result)
}

/// Covariance between two samples (sample covariance)
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 { return 0.0; }
    let mx = mean(x);
    let my = mean(y);
    x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / (n as f64 - 1.0)
}

/// Pearson correlation
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let cov = covariance(x, y);
    let sx = std_dev(x);
    let sy = std_dev(y);
    if sx < 1e-15 || sy < 1e-15 { return 0.0; }
    cov / (sx * sy)
}

/// Spearman rank correlation
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let rx = ranks(x);
    let ry = ranks(y);
    correlation(&rx, &ry)
}

/// Kendall's tau-b
pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len());
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut ties_x = 0i64;
    let mut ties_y = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            let product = dx * dy;
            if product > 1e-15 { concordant += 1; }
            else if product < -1e-15 { discordant += 1; }
            else {
                if dx.abs() < 1e-15 { ties_x += 1; }
                if dy.abs() < 1e-15 { ties_y += 1; }
            }
        }
    }
    let n0 = (n * (n - 1) / 2) as f64;
    let denom = ((n0 - ties_x as f64) * (n0 - ties_y as f64)).sqrt();
    if denom < 1e-15 { return 0.0; }
    (concordant - discordant) as f64 / denom
}

/// Rank values (average rank for ties)
pub fn ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 { j += 1; }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based
        for k in i..j { result[indexed[k].0] = avg_rank; }
        i = j;
    }
    result
}

/// Covariance matrix from column-oriented data (each inner Vec is a variable)
pub fn covariance_matrix(vars: &[&[f64]]) -> Vec<Vec<f64>> {
    let p = vars.len();
    let mut cov = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in i..p {
            let c = covariance(vars[i], vars[j]);
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    cov
}

/// Correlation matrix
pub fn correlation_matrix(vars: &[&[f64]]) -> Vec<Vec<f64>> {
    let p = vars.len();
    let mut cor = vec![vec![0.0; p]; p];
    for i in 0..p {
        cor[i][i] = 1.0;
        for j in (i + 1)..p {
            let c = correlation(vars[i], vars[j]);
            cor[i][j] = c;
            cor[j][i] = c;
        }
    }
    cor
}

/// Welford's online algorithm for mean and variance
#[derive(Clone, Debug)]
pub struct WelfordOnline {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min: f64,
    pub max: f64,
}

impl WelfordOnline {
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY }
    }

    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        if x < self.min { self.min = x; }
        if x > self.max { self.max = x; }
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.m2 / (self.count as f64 - 1.0) }
    }

    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }

    pub fn merge(&self, other: &WelfordOnline) -> WelfordOnline {
        let count = self.count + other.count;
        if count == 0 { return WelfordOnline::new(); }
        let delta = other.mean - self.mean;
        let mean = (self.mean * self.count as f64 + other.mean * other.count as f64) / count as f64;
        let m2 = self.m2 + other.m2 + delta * delta * (self.count as f64) * (other.count as f64) / count as f64;
        WelfordOnline {
            count,
            mean,
            m2,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

impl Default for WelfordOnline {
    fn default() -> Self { Self::new() }
}

/// Extended Welford for skewness and kurtosis
#[derive(Clone, Debug)]
pub struct WelfordExtended {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
}

impl WelfordExtended {
    pub fn new() -> Self {
        Self { n: 0, mean: 0.0, m2: 0.0, m3: 0.0, m4: 0.0 }
    }

    pub fn update(&mut self, x: f64) {
        let n1 = self.n;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1 as f64;

        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 { 0.0 } else { self.m2 / (self.n as f64 - 1.0) }
    }

    pub fn skewness(&self) -> f64 {
        if self.n < 3 || self.m2 < 1e-30 { return 0.0; }
        let n = self.n as f64;
        (n.sqrt() * self.m3) / self.m2.powf(1.5)
    }

    pub fn kurtosis(&self) -> f64 {
        if self.n < 4 || self.m2 < 1e-30 { return 0.0; }
        let n = self.n as f64;
        (n * self.m4) / (self.m2 * self.m2) - 3.0
    }
}

impl Default for WelfordExtended {
    fn default() -> Self { Self::new() }
}

/// Exponentially weighted moving average/variance
#[derive(Clone, Debug)]
pub struct ExponentiallyWeighted {
    pub alpha: f64,
    pub mean: f64,
    pub variance: f64,
    pub initialized: bool,
}

impl ExponentiallyWeighted {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, mean: 0.0, variance: 0.0, initialized: false }
    }

    /// Create with span (alpha = 2/(span+1))
    pub fn from_span(span: f64) -> Self {
        Self::new(2.0 / (span + 1.0))
    }

    /// Create with halflife (alpha = 1 - exp(-ln(2)/halflife))
    pub fn from_halflife(halflife: f64) -> Self {
        Self::new(1.0 - (-0.693147180559945 / halflife).exp())
    }

    pub fn update(&mut self, x: f64) {
        if !self.initialized {
            self.mean = x;
            self.variance = 0.0;
            self.initialized = true;
            return;
        }
        let diff = x - self.mean;
        let incr = self.alpha * diff;
        self.mean += incr;
        self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff);
    }

    pub fn std_dev(&self) -> f64 { self.variance.sqrt() }

    /// Process entire series, return (means, variances)
    pub fn process(&mut self, data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut means = Vec::with_capacity(data.len());
        let mut vars = Vec::with_capacity(data.len());
        for &x in data {
            self.update(x);
            means.push(self.mean);
            vars.push(self.variance);
        }
        (means, vars)
    }
}

/// Exponentially weighted covariance between two series
pub fn ewm_covariance(x: &[f64], y: &[f64], alpha: f64) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut cov = Vec::with_capacity(n);
    let mut mx = x[0];
    let mut my = y[0];
    let mut c = 0.0;
    cov.push(0.0);
    for i in 1..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        mx += alpha * dx;
        my += alpha * dy;
        c = (1.0 - alpha) * (c + alpha * dx * dy);
        cov.push(c);
    }
    cov
}

/// Exponentially weighted correlation
pub fn ewm_correlation(x: &[f64], y: &[f64], alpha: f64) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let mut ewx = ExponentiallyWeighted::new(alpha);
    let mut ewy = ExponentiallyWeighted::new(alpha);
    let cov = ewm_covariance(x, y, alpha);
    let mut corr = Vec::with_capacity(n);
    for i in 0..n {
        ewx.update(x[i]);
        ewy.update(y[i]);
        let sx = ewx.std_dev();
        let sy = ewy.std_dev();
        if sx > 1e-15 && sy > 1e-15 {
            corr.push(cov[i] / (sx * sy));
        } else {
            corr.push(0.0);
        }
    }
    corr
}

/// Order statistics: k-th smallest (0-indexed), using quickselect
pub fn order_statistic(data: &mut [f64], k: usize) -> f64 {
    assert!(k < data.len());
    let n = data.len();
    if n == 1 { return data[0]; }
    quickselect(data, 0, n - 1, k)
}

fn quickselect(arr: &mut [f64], lo: usize, hi: usize, k: usize) -> f64 {
    if lo == hi { return arr[lo]; }
    let pivot_idx = lo + (hi - lo) / 2;
    let pivot = arr[pivot_idx];
    arr.swap(pivot_idx, hi);
    let mut store = lo;
    for i in lo..hi {
        if arr[i] < pivot {
            arr.swap(i, store);
            store += 1;
        }
    }
    arr.swap(store, hi);
    if k == store { arr[store] }
    else if k < store {
        if store == 0 { return arr[0]; }
        quickselect(arr, lo, store - 1, k)
    }
    else { quickselect(arr, store + 1, hi, k) }
}

/// Histogram: compute bin edges and counts
pub struct Histogram {
    pub bin_edges: Vec<f64>,
    pub counts: Vec<usize>,
    pub density: Vec<f64>,
}

impl Histogram {
    /// Create histogram with specified number of bins
    pub fn new(data: &[f64], bins: usize) -> Self {
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::with_range(data, bins, min, max)
    }

    /// Create histogram with specified range
    pub fn with_range(data: &[f64], bins: usize, min: f64, max: f64) -> Self {
        let width = (max - min) / bins as f64;
        let mut bin_edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins { bin_edges.push(min + i as f64 * width); }

        let mut counts = vec![0usize; bins];
        let n = data.len();
        for &x in data {
            let b = ((x - min) / width).floor() as usize;
            let b = b.min(bins - 1);
            counts[b] += 1;
        }

        let density: Vec<f64> = counts.iter().map(|&c| c as f64 / (n as f64 * width)).collect();
        Self { bin_edges, counts, density }
    }

    /// Sturges' rule for number of bins
    pub fn sturges_bins(n: usize) -> usize {
        ((n as f64).log2().ceil() + 1.0) as usize
    }

    /// Scott's rule for bin width
    pub fn scott_width(data: &[f64]) -> f64 {
        let s = std_dev(data);
        let n = data.len() as f64;
        3.5 * s / n.cbrt()
    }

    /// Freedman-Diaconis rule
    pub fn fd_width(data: &[f64]) -> f64 {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let iqr = percentile_sorted(&sorted, 0.75) - percentile_sorted(&sorted, 0.25);
        let n = data.len() as f64;
        2.0 * iqr / n.cbrt()
    }

    /// Cumulative distribution from histogram
    pub fn cumulative(&self) -> Vec<f64> {
        let total: usize = self.counts.iter().sum();
        let mut cum = Vec::with_capacity(self.counts.len());
        let mut running = 0;
        for &c in &self.counts {
            running += c;
            cum.push(running as f64 / total as f64);
        }
        cum
    }
}

/// Kernel density estimation (Gaussian kernel)
pub struct KernelDensity {
    pub data: Vec<f64>,
    pub bandwidth: f64,
}

impl KernelDensity {
    /// Create with Silverman's rule of thumb bandwidth
    pub fn new(data: &[f64]) -> Self {
        let s = std_dev(data);
        let n = data.len() as f64;
        let iqr_val = {
            let mut sorted = data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            percentile_sorted(&sorted, 0.75) - percentile_sorted(&sorted, 0.25)
        };
        let sigma = s.min(iqr_val / 1.34);
        let h = 0.9 * sigma * n.powf(-0.2);
        Self { data: data.to_vec(), bandwidth: h.max(1e-10) }
    }

    /// Create with explicit bandwidth
    pub fn with_bandwidth(data: &[f64], bandwidth: f64) -> Self {
        Self { data: data.to_vec(), bandwidth }
    }

    /// Evaluate density at point x
    pub fn evaluate(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let h = self.bandwidth;
        let mut sum = 0.0;
        for &xi in &self.data {
            let z = (x - xi) / h;
            sum += (-0.5 * z * z).exp();
        }
        sum / (n * h * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// Evaluate at multiple points
    pub fn evaluate_grid(&self, grid: &[f64]) -> Vec<f64> {
        grid.iter().map(|&x| self.evaluate(x)).collect()
    }

    /// Generate evaluation grid
    pub fn auto_grid(&self, n_points: usize) -> Vec<f64> {
        let min = self.data.iter().copied().fold(f64::INFINITY, f64::min) - 3.0 * self.bandwidth;
        let max = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 3.0 * self.bandwidth;
        let step = (max - min) / (n_points - 1) as f64;
        (0..n_points).map(|i| min + i as f64 * step).collect()
    }

    /// Integrated squared error bandwidth selection (LSCV)
    pub fn lscv_bandwidth(data: &[f64], h_min: f64, h_max: f64, n_grid: usize) -> f64 {
        let step = (h_max - h_min) / n_grid as f64;
        let mut best_h = h_min;
        let mut best_score = f64::INFINITY;
        let n = data.len();

        for g in 0..=n_grid {
            let h = h_min + g as f64 * step;
            if h < 1e-12 { continue; }
            // Leave-one-out cross-validation score
            let mut score = 0.0;
            for i in 0..n {
                let mut f_i = 0.0;
                for j in 0..n {
                    if i == j { continue; }
                    let z = (data[i] - data[j]) / h;
                    f_i += (-0.5 * z * z).exp();
                }
                f_i /= ((n - 1) as f64) * h * (2.0 * std::f64::consts::PI).sqrt();
                score -= f_i.ln().max(-700.0);
            }
            if score < best_score { best_score = score; best_h = h; }
        }
        best_h
    }
}

/// Bootstrap confidence interval
pub struct Bootstrap {
    pub n_resamples: usize,
    pub seed: u64,
}

impl Bootstrap {
    pub fn new(n_resamples: usize, seed: u64) -> Self {
        Self { n_resamples, seed }
    }

    /// Bootstrap a statistic function, return (point_estimate, ci_low, ci_high)
    pub fn confidence_interval<F: Fn(&[f64]) -> f64>(
        &self, data: &[f64], statistic: F, alpha: f64,
    ) -> (f64, f64, f64) {
        let n = data.len();
        let point_est = statistic(data);
        let mut rng = Lcg::new(self.seed);
        let mut estimates = Vec::with_capacity(self.n_resamples);

        for _ in 0..self.n_resamples {
            let sample: Vec<f64> = (0..n).map(|_| data[rng.next_usize(n)]).collect();
            estimates.push(statistic(&sample));
        }

        estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = percentile_sorted(&estimates, alpha / 2.0);
        let hi = percentile_sorted(&estimates, 1.0 - alpha / 2.0);
        (point_est, lo, hi)
    }

    /// BCa (bias-corrected and accelerated) bootstrap
    pub fn bca_interval<F: Fn(&[f64]) -> f64 + Clone>(
        &self, data: &[f64], statistic: F, alpha: f64,
    ) -> (f64, f64, f64) {
        let n = data.len();
        let point_est = statistic(data);
        let mut rng = Lcg::new(self.seed);
        let mut estimates = Vec::with_capacity(self.n_resamples);

        for _ in 0..self.n_resamples {
            let sample: Vec<f64> = (0..n).map(|_| data[rng.next_usize(n)]).collect();
            estimates.push(statistic(&sample));
        }

        estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Bias correction
        let count_below = estimates.iter().filter(|&&x| x < point_est).count();
        let z0 = norm_ppf(count_below as f64 / self.n_resamples as f64);

        // Acceleration (jackknife)
        let mut jackknife = Vec::with_capacity(n);
        for i in 0..n {
            let jk_sample: Vec<f64> = data.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &x)| x)
                .collect();
            jackknife.push(statistic(&jk_sample));
        }
        let jk_mean = mean(&jackknife);
        let num: f64 = jackknife.iter().map(|x| (jk_mean - x).powi(3)).sum();
        let den: f64 = jackknife.iter().map(|x| (jk_mean - x).powi(2)).sum();
        let a_hat = if den.abs() > 1e-30 { num / (6.0 * den.powf(1.5)) } else { 0.0 };

        // Adjusted quantiles
        let z_alpha = norm_ppf(alpha / 2.0);
        let z_1alpha = norm_ppf(1.0 - alpha / 2.0);
        let adj_lo = norm_cdf(z0 + (z0 + z_alpha) / (1.0 - a_hat * (z0 + z_alpha)));
        let adj_hi = norm_cdf(z0 + (z0 + z_1alpha) / (1.0 - a_hat * (z0 + z_1alpha)));

        let lo = percentile_sorted(&estimates, adj_lo.clamp(0.0, 1.0));
        let hi = percentile_sorted(&estimates, adj_hi.clamp(0.0, 1.0));
        (point_est, lo, hi)
    }

    /// Permutation test for difference in means
    pub fn permutation_test(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let observed = mean(x) - mean(y);
        let mut combined: Vec<f64> = x.iter().chain(y.iter()).copied().collect();
        let nx = x.len();
        let mut rng = Lcg::new(self.seed);
        let mut count = 0usize;

        for _ in 0..self.n_resamples {
            // Fisher-Yates shuffle
            for i in (1..combined.len()).rev() {
                let j = rng.next_usize(i + 1);
                combined.swap(i, j);
            }
            let perm_diff = mean(&combined[..nx]) - mean(&combined[nx..]);
            if perm_diff.abs() >= observed.abs() { count += 1; }
        }
        (observed, count as f64 / self.n_resamples as f64)
    }
}

/// Simple normal CDF (Abramowitz & Stegun approximation)
fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();
    0.5 * (1.0 + sign * y)
}

/// Normal quantile (Beasley-Springer-Moro)
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 { return -8.0; }
    if p >= 1.0 { return 8.0; }
    let a = [
        -3.969683028665376e1, 2.209460984245205e2,
        -2.759285104469687e2, 1.383577518672690e2,
        -3.066479806614716e1, 2.506628277459239e0,
    ];
    let b = [
        -5.447609879822406e1, 1.615858368580409e2,
        -1.556989798598866e2, 6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3, -3.223964580411365e-1,
        -2.400758277161838e0, -2.549732539343734e0,
        4.374664141464968e0, 2.938163982698783e0,
    ];
    let d = [
        7.784695709041462e-3, 3.224671290700398e-1,
        2.445134137142996e0, 3.754408661907416e0,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    }
}

/// Simple LCG for bootstrap reproducibility
struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() >> 33) as usize % max
    }
}

/// Weighted statistics
pub fn weighted_mean(data: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(data.len(), weights.len());
    let sum_w: f64 = weights.iter().sum();
    if sum_w.abs() < 1e-30 { return 0.0; }
    data.iter().zip(weights).map(|(x, w)| x * w).sum::<f64>() / sum_w
}

pub fn weighted_variance(data: &[f64], weights: &[f64]) -> f64 {
    let wm = weighted_mean(data, weights);
    let sum_w: f64 = weights.iter().sum();
    if sum_w.abs() < 1e-30 { return 0.0; }
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let var_num: f64 = data.iter().zip(weights).map(|(x, w)| w * (x - wm).powi(2)).sum();
    var_num * sum_w / (sum_w * sum_w - sum_w2)
}

/// Rolling statistics with a fixed window
pub struct RollingStats {
    pub window: usize,
}

impl RollingStats {
    pub fn new(window: usize) -> Self { Self { window } }

    pub fn rolling_mean(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let w = self.window;
        let mut result = Vec::with_capacity(n);
        let mut sum = 0.0;
        for i in 0..n {
            sum += data[i];
            if i >= w { sum -= data[i - w]; }
            if i + 1 >= w {
                result.push(sum / w as f64);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    pub fn rolling_std(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let w = self.window;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if i + 1 < w {
                result.push(f64::NAN);
            } else {
                let window_data = &data[i + 1 - w..=i];
                result.push(std_dev(window_data));
            }
        }
        result
    }

    pub fn rolling_min(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let w = self.window;
        let mut result = Vec::with_capacity(n);
        // Simple O(nw) implementation
        for i in 0..n {
            if i + 1 < w { result.push(f64::NAN); continue; }
            let start = i + 1 - w;
            let min_val = data[start..=i].iter().copied().fold(f64::INFINITY, f64::min);
            result.push(min_val);
        }
        result
    }

    pub fn rolling_max(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let w = self.window;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if i + 1 < w { result.push(f64::NAN); continue; }
            let start = i + 1 - w;
            let max_val = data[start..=i].iter().copied().fold(f64::NEG_INFINITY, f64::max);
            result.push(max_val);
        }
        result
    }

    pub fn rolling_correlation(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let w = self.window;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if i + 1 < w { result.push(f64::NAN); continue; }
            let start = i + 1 - w;
            result.push(correlation(&x[start..=i], &y[start..=i]));
        }
        result
    }

    pub fn rolling_skew(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let w = self.window;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if i + 1 < w { result.push(f64::NAN); continue; }
            let start = i + 1 - w;
            result.push(skewness(&data[start..=i]));
        }
        result
    }

    pub fn rolling_kurtosis(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let w = self.window;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            if i + 1 < w { result.push(f64::NAN); continue; }
            let start = i + 1 - w;
            result.push(kurtosis(&data[start..=i]));
        }
        result
    }
}

/// Z-score normalization
pub fn z_score(data: &[f64]) -> Vec<f64> {
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|x| (x - m) / s).collect()
}

/// Min-max normalization to [0, 1]
pub fn min_max_normalize(data: &[f64]) -> Vec<f64> {
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range < 1e-15 { return vec![0.5; data.len()]; }
    data.iter().map(|x| (x - min) / range).collect()
}

/// Rank-based normalization (normal scores)
pub fn rank_normalize(data: &[f64]) -> Vec<f64> {
    let r = ranks(data);
    let n = data.len() as f64;
    r.iter().map(|&ri| norm_ppf((ri - 0.5) / n)).collect()
}

/// Robust z-score using median and MAD
pub fn robust_z_score(data: &[f64]) -> Vec<f64> {
    let med = median(data);
    let abs_dev: Vec<f64> = data.iter().map(|x| (x - med).abs()).collect();
    let mad = median(&abs_dev);
    let scale = mad * 1.4826; // consistency constant for normal
    if scale < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|x| (x - med) / scale).collect()
}

/// Compute entropy of a discrete distribution (probabilities)
pub fn entropy(probs: &[f64]) -> f64 {
    probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Kullback-Leibler divergence D(P || Q)
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    p.iter().zip(q).map(|(&pi, &qi)| {
        if pi > 1e-30 && qi > 1e-30 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q).map(|(a, b)| 0.5 * (a + b)).collect();
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Mutual information (from joint and marginal counts)
pub fn mutual_information(joint: &[Vec<usize>], n_total: usize) -> f64 {
    let rows = joint.len();
    let cols = if rows > 0 { joint[0].len() } else { 0 };
    let n = n_total as f64;
    let row_sums: Vec<f64> = joint.iter().map(|r| r.iter().sum::<usize>() as f64).collect();
    let col_sums: Vec<f64> = (0..cols).map(|j| joint.iter().map(|r| r[j]).sum::<usize>() as f64).collect();

    let mut mi = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let pij = joint[i][j] as f64 / n;
            let pi = row_sums[i] / n;
            let pj = col_sums[j] / n;
            if pij > 1e-30 && pi > 1e-30 && pj > 1e-30 {
                mi += pij * (pij / (pi * pj)).ln();
            }
        }
    }
    mi
}

/// Empirical CDF
pub fn ecdf(data: &[f64], x: f64) -> f64 {
    let count = data.iter().filter(|&&v| v <= x).count();
    count as f64 / data.len() as f64
}

/// Kolmogorov-Smirnov statistic (one sample, against standard normal)
pub fn ks_statistic_normal(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let m = mean(data);
    let s = std_dev(data);

    let mut max_d = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let z = if s > 1e-15 { (x - m) / s } else { 0.0 };
        let f_x = norm_cdf(z);
        let d1 = ((i + 1) as f64 / n - f_x).abs();
        let d2 = (f_x - i as f64 / n).abs();
        max_d = max_d.max(d1).max(d2);
    }
    max_d
}

/// Two-sample KS statistic
pub fn ks_statistic_two_sample(x: &[f64], y: &[f64]) -> f64 {
    let mut sx = x.to_vec();
    let mut sy = y.to_vec();
    sx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let nx = sx.len() as f64;
    let ny = sy.len() as f64;
    let mut i = 0;
    let mut j = 0;
    let mut max_d = 0.0;

    while i < sx.len() && j < sy.len() {
        let d = ((i + 1) as f64 / nx - (j + 1) as f64 / ny).abs();
        max_d = max_d.max(d);
        if sx[i] <= sy[j] { i += 1; } else { j += 1; }
    }
    max_d
}

/// Anderson-Darling test statistic (normal)
pub fn anderson_darling_normal(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 8 { return 0.0; }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 { return 0.0; }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut sum = 0.0;
    for i in 0..n {
        let z = (sorted[i] - m) / s;
        let fi = norm_cdf(z).clamp(1e-10, 1.0 - 1e-10);
        let fn_i = norm_cdf((sorted[n - 1 - i] - m) / s).clamp(1e-10, 1.0 - 1e-10);
        sum += (2 * i + 1) as f64 * (fi.ln() + (1.0 - fn_i).ln());
    }
    -(n as f64) - sum / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-12);
        assert!((variance(&data) - 2.5).abs() < 1e-12);
        assert!((median(&data) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_welford() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut w = WelfordOnline::new();
        for &x in &data { w.update(x); }
        assert!((w.mean - mean(&data)).abs() < 1e-10);
        assert!((w.variance() - variance(&data)).abs() < 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((correlation(&x, &y) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_histogram() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let h = Histogram::new(&data, 10);
        assert_eq!(h.counts.len(), 10);
        let total: usize = h.counts.iter().sum();
        assert_eq!(total, 100);
    }
}
