/// Probability of Informed Trading (PIN) — Easley, Kiefer, O'Hara, Paperman (1996).
///
/// Structural model parameters:
///   α   = probability of an information event
///   δ   = probability that the event is bad news (given event)
///   μ   = arrival rate of informed traders
///   ε_b = arrival rate of uninformed buys
///   ε_s = arrival rate of uninformed sells
///
/// PIN = α·μ / (α·μ + ε_b + ε_s)
///
/// Estimated via EM algorithm on (B_i, S_i) buy/sell count pairs per trading day.

/// Parameters of the PIN structural model.
#[derive(Debug, Clone)]
pub struct PinParams {
    pub alpha:   f64,   // P(information event)
    pub delta:   f64,   // P(bad news | event)
    pub mu:      f64,   // informed order arrival rate
    pub eps_b:   f64,   // uninformed buy arrival rate
    pub eps_s:   f64,   // uninformed sell arrival rate
}

impl PinParams {
    /// PIN value derived from current parameters.
    pub fn pin(&self) -> f64 {
        let denom = self.alpha * self.mu + self.eps_b + self.eps_s;
        if denom < 1e-15 { return 0.0; }
        self.alpha * self.mu / denom
    }

    /// Log-likelihood of observed (buys, sells) under current params.
    pub fn log_likelihood(&self, days: &[(u64, u64)]) -> f64 {
        days.iter().map(|&(b, s)| {
            let b = b as f64;
            let s = s as f64;
            let mu  = self.mu;
            let eb  = self.eps_b;
            let es  = self.eps_s;
            let a   = self.alpha;
            let d   = self.delta;

            // Three regimes: no event, good news event, bad news event
            let l_no  = (1.0 - a) * poisson_pmf_log(b, eb)    * poisson_pmf_log(s, es).exp();
            let l_good= a * (1.0 - d) * poisson_pmf_log(b, eb + mu).exp() * poisson_pmf_log(s, es).exp();
            let l_bad = a * d         * poisson_pmf_log(b, eb).exp()       * poisson_pmf_log(s, es + mu).exp();
            let l_tot = l_no + l_good + l_bad;
            if l_tot <= 0.0 { -1e30 } else { l_tot.ln() }
        }).sum()
    }
}

fn poisson_pmf_log(k: f64, lambda: f64) -> f64 {
    if lambda <= 0.0 { return if k == 0.0 { 0.0 } else { f64::NEG_INFINITY }; }
    k * lambda.ln() - lambda - lgamma(k + 1.0)
}

/// Log-gamma via Stirling approximation (sufficient for large k).
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x < 1.0  { return lgamma(x + 1.0) - x.ln(); }
    // Lanczos approximation (g=7, n=9)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// EM-based PIN estimator.
pub struct PinEstimator {
    days:      Vec<(u64, u64)>,  // (buys, sells) per day
    params:    PinParams,
    converged: bool,
    iterations: u32,
}

impl PinEstimator {
    /// Create a new estimator with initial parameter guesses.
    pub fn new() -> Self {
        Self {
            days: Vec::new(),
            params: PinParams {
                alpha: 0.4,
                delta: 0.5,
                mu:    20.0,
                eps_b: 40.0,
                eps_s: 40.0,
            },
            converged:  false,
            iterations: 0,
        }
    }

    /// Add a day's observed (buy_count, sell_count).
    pub fn push_day(&mut self, buys: u64, sells: u64) {
        self.days.push((buys, sells));
        self.converged = false;
    }

    /// Run EM until convergence or max iterations.
    pub fn fit(&mut self, max_iter: u32, tol: f64) {
        if self.days.len() < 5 {
            return;
        }
        let mut ll_prev = f64::NEG_INFINITY;
        for iter in 0..max_iter {
            self.em_step();
            let ll = self.params.log_likelihood(&self.days);
            if (ll - ll_prev).abs() < tol {
                self.converged  = true;
                self.iterations = iter + 1;
                break;
            }
            ll_prev = ll;
            self.iterations = iter + 1;
        }
    }

    fn em_step(&mut self) {
        let a  = self.params.alpha;
        let d  = self.params.delta;
        let mu = self.params.mu;
        let eb = self.params.eps_b;
        let es = self.params.eps_s;

        let mut sum_p_no   = 0.0;
        let mut sum_p_good = 0.0;
        let mut sum_p_bad  = 0.0;

        let mut sum_b_no   = 0.0; let mut sum_b_good = 0.0; let mut sum_b_bad = 0.0;
        let mut sum_s_no   = 0.0; let mut sum_s_good = 0.0; let mut sum_s_bad = 0.0;

        for &(b, s) in &self.days {
            let bf = b as f64; let sf = s as f64;

            let ln_no   = (1.0 - a).ln() + poisson_pmf_log(bf, eb)      + poisson_pmf_log(sf, es);
            let ln_good = (a * (1.0 - d)).ln() + poisson_pmf_log(bf, eb + mu) + poisson_pmf_log(sf, es);
            let ln_bad  = (a * d).ln()          + poisson_pmf_log(bf, eb)      + poisson_pmf_log(sf, es + mu);

            // Log-sum-exp for numerical stability
            let max_ln = ln_no.max(ln_good).max(ln_bad);
            let denom  = (ln_no - max_ln).exp() + (ln_good - max_ln).exp() + (ln_bad - max_ln).exp();

            let p_no   = (ln_no   - max_ln).exp() / denom;
            let p_good = (ln_good - max_ln).exp() / denom;
            let p_bad  = (ln_bad  - max_ln).exp() / denom;

            sum_p_no   += p_no;
            sum_p_good += p_good;
            sum_p_bad  += p_bad;

            sum_b_no   += p_no   * bf; sum_b_good += p_good * bf; sum_b_bad += p_bad * bf;
            sum_s_no   += p_no   * sf; sum_s_good += p_good * sf; sum_s_bad += p_bad * sf;
        }

        let n = self.days.len() as f64;

        self.params.alpha = (sum_p_good + sum_p_bad) / n;
        self.params.delta = if (sum_p_good + sum_p_bad) > 1e-15 {
            sum_p_bad / (sum_p_good + sum_p_bad)
        } else { 0.5 };

        // μ: arrival rate of informed (from good+bad regimes)
        let mu_num_b = sum_b_good; let mu_num_s = sum_s_bad;
        let mu_den   = sum_p_good + sum_p_bad;
        self.params.mu = if mu_den > 1e-15 {
            ((mu_num_b + mu_num_s) / mu_den).max(1e-6)
        } else { mu };

        // ε_b, ε_s from no-event regime
        self.params.eps_b = if n > 0.0 {
            (sum_b_no + sum_b_good) / (sum_p_no + sum_p_good).max(1e-15)
        } else { eb };
        self.params.eps_s = if n > 0.0 {
            (sum_s_no + sum_s_bad) / (sum_p_no + sum_p_bad).max(1e-15)
        } else { es };

        // Clamp to valid range
        self.params.alpha = self.params.alpha.clamp(1e-6, 1.0 - 1e-6);
        self.params.delta = self.params.delta.clamp(1e-6, 1.0 - 1e-6);
        self.params.mu    = self.params.mu.max(1e-6);
        self.params.eps_b = self.params.eps_b.max(1e-6);
        self.params.eps_s = self.params.eps_s.max(1e-6);
    }

    pub fn pin(&self)         -> f64        { self.params.pin() }
    pub fn params(&self)      -> &PinParams { &self.params }
    pub fn converged(&self)   -> bool       { self.converged }
    pub fn iterations(&self)  -> u32        { self.iterations }
    pub fn n_days(&self)      -> usize      { self.days.len() }
}

impl Default for PinEstimator {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lgamma_accuracy() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6
        assert!((lgamma(1.0).exp() - 1.0).abs() < 1e-5);
        assert!((lgamma(2.0).exp() - 1.0).abs() < 1e-5);
        assert!((lgamma(4.0).exp() - 6.0).abs() < 1e-3);
    }

    #[test]
    fn pin_converges_on_synthetic_data() {
        let mut est = PinEstimator::new();
        // Simulate: mostly quiet days, occasional informed-trading spikes
        let days: Vec<(u64, u64)> = (0..60).map(|i| {
            if i % 7 == 0 { (80, 20) }  // informed buys
            else           { (40, 38) }  // noise
        }).collect();
        for (b, s) in days { est.push_day(b, s); }
        est.fit(500, 1e-6);
        let pin = est.pin();
        assert!((0.0..=1.0).contains(&pin), "PIN={} out of range", pin);
    }

    #[test]
    fn pin_params_valid_range() {
        let p = PinParams { alpha: 0.3, delta: 0.5, mu: 20.0, eps_b: 40.0, eps_s: 40.0 };
        let pin = p.pin();
        assert!(pin > 0.0 && pin < 1.0);
    }
}
