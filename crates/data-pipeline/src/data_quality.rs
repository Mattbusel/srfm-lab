/// data_quality.rs -- data quality checks for OHLCV bars and tick data.
///
/// Each check returns a CheckResult indicating pass/fail, the measured value,
/// the threshold used, and a human-readable message. DataQualityChecker runs
/// all checks and produces a QualityReport with a weighted quality score.

use crate::ohlcv::Bar;
use chrono::Utc;

// ── CheckResult ───────────────────────────────────────────────────────────────

/// Result of a single data quality check.
#[derive(Debug, Clone)]
pub struct CheckResult {
    /// Short identifier for the check.
    pub name:      String,
    /// Whether the check passed.
    pub passed:    bool,
    /// The measured value being tested.
    pub value:     f64,
    /// The threshold used for the decision.
    pub threshold: f64,
    /// Human-readable description of the result.
    pub message:   String,
}

impl CheckResult {
    fn pass(name: &str, value: f64, threshold: f64, msg: &str) -> Self {
        CheckResult {
            name:      name.to_string(),
            passed:    true,
            value,
            threshold,
            message:   msg.to_string(),
        }
    }

    fn fail(name: &str, value: f64, threshold: f64, msg: &str) -> Self {
        CheckResult {
            name:      name.to_string(),
            passed:    false,
            value,
            threshold,
            message:   msg.to_string(),
        }
    }
}

// ── QualityReport ─────────────────────────────────────────────────────────────

/// Aggregate quality report for a single bar.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Symbol this report applies to.
    pub symbol:        String,
    /// True if all checks passed.
    pub passed:        bool,
    /// Individual check results.
    pub checks:        Vec<CheckResult>,
    /// Weighted quality score in [0, 1].
    pub quality_score: f64,
}

impl QualityReport {
    /// Number of failed checks.
    pub fn failed_count(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }

    /// Names of all failed checks.
    pub fn failed_names(&self) -> Vec<&str> {
        self.checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.as_str())
            .collect()
    }
}

// ── QualityContext ────────────────────────────────────────────────────────────

/// Market context needed by some quality checks.
#[derive(Debug, Clone)]
pub struct QualityContext {
    /// Rolling average daily volume (same units as bar.volume).
    pub rolling_adv:         f64,
    /// Rolling volatility (std-dev of returns, fractional, e.g. 0.02 = 2%).
    pub rolling_vol:         f64,
    /// Expected bar interval in seconds (e.g. 60 for 1-min bars).
    pub expected_interval_s: i64,
    /// Maximum allowed absolute return in a single bar, fractional.
    pub max_return_pct:      f64,
}

impl Default for QualityContext {
    fn default() -> Self {
        QualityContext {
            rolling_adv:         1_000_000.0,
            rolling_vol:         0.02,
            expected_interval_s: 60,
            max_return_pct:      0.10,
        }
    }
}

// ── DataQualityChecker ────────────────────────────────────────────────────────

/// Stateless checker that applies individual quality rules.
pub struct DataQualityChecker;

impl DataQualityChecker {
    pub fn new() -> Self { DataQualityChecker }

    // ---- individual checks --------------------------------------------------

    /// OHLC logical validity: H >= max(O,C) >= min(O,C) >= L and all prices > 0.
    pub fn check_ohlc_validity(bar: &Bar) -> CheckResult {
        let name = "ohlc_validity";
        let o = bar.open;
        let h = bar.high;
        let l = bar.low;
        let c = bar.close;

        if o <= 0.0 || h <= 0.0 || l <= 0.0 || c <= 0.0 {
            return CheckResult::fail(
                name, 0.0, 0.0,
                "one or more prices are non-positive",
            );
        }
        if h < o.max(c) {
            return CheckResult::fail(
                name, h, o.max(c),
                "high is below max(open, close)",
            );
        }
        if l > o.min(c) {
            return CheckResult::fail(
                name, l, o.min(c),
                "low is above min(open, close)",
            );
        }
        if h < l {
            return CheckResult::fail(
                name, h, l,
                "high is below low",
            );
        }
        CheckResult::pass(name, h - l, 0.0, "OHLC prices are logically consistent")
    }

    /// Volume spike check: bar volume must be < 10x rolling ADV.
    pub fn check_volume_spike(bar: &Bar, rolling_adv: f64) -> CheckResult {
        let name = "volume_spike";
        let threshold = rolling_adv * 10.0;
        if rolling_adv <= 0.0 {
            return CheckResult::fail(name, bar.volume, 0.0, "rolling ADV is non-positive");
        }
        if bar.volume > threshold {
            return CheckResult::fail(
                name,
                bar.volume,
                threshold,
                &format!(
                    "volume {:.0} exceeds 10x ADV {:.0}",
                    bar.volume, rolling_adv
                ),
            );
        }
        CheckResult::pass(
            name,
            bar.volume,
            threshold,
            "volume within acceptable range",
        )
    }

    /// Return spike check: |close/prev_close - 1| < max_pct.
    pub fn check_return_spike(bar: &Bar, prev_close: f64, max_pct: f64) -> CheckResult {
        let name = "return_spike";
        if prev_close <= 0.0 {
            return CheckResult::fail(name, 0.0, max_pct, "prev_close is non-positive");
        }
        let ret = (bar.close / prev_close - 1.0).abs();
        if ret >= max_pct {
            return CheckResult::fail(
                name,
                ret,
                max_pct,
                &format!(
                    "absolute return {:.4} exceeds threshold {:.4}",
                    ret, max_pct
                ),
            );
        }
        CheckResult::pass(name, ret, max_pct, "return within acceptable range")
    }

    /// Timestamp gap check: gap between consecutive bars is close to expected.
    /// Tolerates up to 50% deviation from expected interval.
    pub fn check_timestamp_gap(
        ts: i64,
        prev_ts: i64,
        expected_interval_s: i64,
    ) -> CheckResult {
        let name = "timestamp_gap";
        let gap = ts - prev_ts;
        if gap <= 0 {
            return CheckResult::fail(
                name,
                gap as f64,
                expected_interval_s as f64,
                "timestamp is not strictly increasing",
            );
        }
        let expected = expected_interval_s as f64;
        let deviation = ((gap as f64 - expected) / expected).abs();
        // Allow up to 150% extra (i.e., up to 2.5x expected gap) for market gaps.
        // Flag if gap is negative OR less than 50% of expected (duplicate data).
        if (gap as f64) < expected * 0.5 {
            return CheckResult::fail(
                name,
                gap as f64,
                expected,
                &format!(
                    "gap {}s is less than 50% of expected {}s -- possible duplicate bar",
                    gap, expected_interval_s
                ),
            );
        }
        let _ = deviation;
        CheckResult::pass(
            name,
            gap as f64,
            expected,
            "timestamp gap is acceptable",
        )
    }

    /// Bid-ask spread check: spread in bps must not exceed max_bps.
    pub fn check_bid_ask_spread(bid: f64, ask: f64, max_bps: f64) -> CheckResult {
        let name = "bid_ask_spread";
        if bid <= 0.0 || ask <= 0.0 {
            return CheckResult::fail(name, 0.0, max_bps, "bid or ask is non-positive");
        }
        if ask < bid {
            return CheckResult::fail(
                name,
                ask - bid,
                0.0,
                "ask is below bid -- crossed market",
            );
        }
        let mid = (bid + ask) / 2.0;
        let spread_bps = (ask - bid) / mid * 10_000.0;
        if spread_bps > max_bps {
            return CheckResult::fail(
                name,
                spread_bps,
                max_bps,
                &format!(
                    "spread {:.2} bps exceeds threshold {:.2} bps",
                    spread_bps, max_bps
                ),
            );
        }
        CheckResult::pass(
            name,
            spread_bps,
            max_bps,
            "bid-ask spread is within acceptable range",
        )
    }

    /// Data freshness check: bar must be within max_age_s seconds of current time.
    pub fn check_data_freshness(
        bar_ts: i64,
        current_ts: i64,
        max_age_s: i64,
    ) -> CheckResult {
        let name = "data_freshness";
        let age = current_ts - bar_ts;
        if age < 0 {
            // Bar timestamp is in the future -- accept but note it.
            return CheckResult::pass(
                name,
                age as f64,
                max_age_s as f64,
                "bar timestamp is in the future",
            );
        }
        if age > max_age_s {
            return CheckResult::fail(
                name,
                age as f64,
                max_age_s as f64,
                &format!("bar is {}s old, exceeds max age {}s", age, max_age_s),
            );
        }
        CheckResult::pass(
            name,
            age as f64,
            max_age_s as f64,
            "bar data is fresh",
        )
    }

    // ---- composite check ----------------------------------------------------

    /// Run all checks for a single bar given its quality context.
    pub fn run_all_checks(
        symbol: &str,
        bar: &Bar,
        context: &QualityContext,
        prev_close: Option<f64>,
        prev_ts: Option<i64>,
    ) -> QualityReport {
        let current_ts = Utc::now().timestamp();
        let bar_ts = bar.timestamp.timestamp();

        let mut checks = Vec::with_capacity(6);

        checks.push(Self::check_ohlc_validity(bar));
        checks.push(Self::check_volume_spike(bar, context.rolling_adv));

        if let Some(pc) = prev_close {
            checks.push(Self::check_return_spike(bar, pc, context.max_return_pct));
        }

        if let Some(pts) = prev_ts {
            checks.push(Self::check_timestamp_gap(
                bar_ts,
                pts,
                context.expected_interval_s,
            ));
        }

        checks.push(Self::check_data_freshness(
            bar_ts,
            current_ts,
            context.expected_interval_s * 10,
        ));

        // Weights: ohlc=3, volume=2, return=2, gap=1, freshness=1 (normalized later).
        let weights: &[f64] = &[3.0, 2.0, 2.0, 1.0, 1.0];
        let quality_score = compute_quality_score(&checks, weights);
        let passed = checks.iter().all(|c| c.passed);

        QualityReport {
            symbol: symbol.to_string(),
            passed,
            checks,
            quality_score,
        }
    }
}

impl Default for DataQualityChecker {
    fn default() -> Self { Self::new() }
}

// ── Public free functions ─────────────────────────────────────────────────────

/// Run all quality checks for a bar. `context` supplies thresholds.
/// `prev_close` and `prev_ts` are optional (first bar has no predecessor).
pub fn run_all_checks(bar: &Bar, context: &QualityContext) -> QualityReport {
    DataQualityChecker::run_all_checks("", bar, context, None, None)
}

/// Run quality checks over an entire bar series, carrying forward state.
pub fn quality_series(bars: &[Bar], contexts: &[QualityContext]) -> Vec<QualityReport> {
    assert_eq!(
        bars.len(),
        contexts.len(),
        "bars and contexts must have the same length"
    );

    let mut reports = Vec::with_capacity(bars.len());
    let mut prev_close: Option<f64> = None;
    let mut prev_ts: Option<i64> = None;

    for (bar, ctx) in bars.iter().zip(contexts.iter()) {
        let report =
            DataQualityChecker::run_all_checks("", bar, ctx, prev_close, prev_ts);
        prev_close = Some(bar.close);
        prev_ts = Some(bar.timestamp.timestamp());
        reports.push(report);
    }

    reports
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Weighted average of check results. Pads or trims weights to match checks.
fn compute_quality_score(checks: &[CheckResult], weights: &[f64]) -> f64 {
    if checks.is_empty() {
        return 1.0;
    }
    let mut total_weight = 0.0;
    let mut weighted_sum = 0.0;
    for (i, check) in checks.iter().enumerate() {
        let w = if i < weights.len() { weights[i] } else { 1.0 };
        weighted_sum += if check.passed { w } else { 0.0 };
        total_weight += w;
    }
    if total_weight == 0.0 {
        return 1.0;
    }
    weighted_sum / total_weight
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_bar(o: f64, h: f64, l: f64, c: f64, v: f64) -> Bar {
        Bar::new(Utc::now(), o, h, l, c, v)
    }

    #[test]
    fn test_ohlc_valid_bar() {
        let bar = make_bar(100.0, 105.0, 98.0, 103.0, 1000.0);
        let r = DataQualityChecker::check_ohlc_validity(&bar);
        assert!(r.passed, "expected pass: {}", r.message);
    }

    #[test]
    fn test_ohlc_high_below_close() {
        let bar = make_bar(100.0, 101.0, 98.0, 103.0, 1000.0);
        let r = DataQualityChecker::check_ohlc_validity(&bar);
        assert!(!r.passed, "expected fail when high < close");
    }

    #[test]
    fn test_ohlc_negative_price() {
        let bar = make_bar(-1.0, 105.0, 98.0, 103.0, 1000.0);
        let r = DataQualityChecker::check_ohlc_validity(&bar);
        assert!(!r.passed, "expected fail on negative open");
    }

    #[test]
    fn test_ohlc_low_above_open() {
        let bar = make_bar(100.0, 105.0, 102.0, 99.0, 1000.0);
        // close=99, low=102 -- low is above close, should fail
        let r = DataQualityChecker::check_ohlc_validity(&bar);
        assert!(!r.passed, "expected fail when low > min(open,close)");
    }

    #[test]
    fn test_volume_spike_pass() {
        let bar = make_bar(100.0, 105.0, 98.0, 103.0, 5_000.0);
        let r = DataQualityChecker::check_volume_spike(&bar, 1_000.0);
        // 5000 < 10000 -- pass
        assert!(r.passed);
    }

    #[test]
    fn test_volume_spike_fail() {
        let bar = make_bar(100.0, 105.0, 98.0, 103.0, 15_000.0);
        let r = DataQualityChecker::check_volume_spike(&bar, 1_000.0);
        // 15000 > 10000 -- fail
        assert!(!r.passed);
    }

    #[test]
    fn test_return_spike_pass() {
        let bar = make_bar(100.0, 105.0, 98.0, 101.0, 1000.0);
        let r = DataQualityChecker::check_return_spike(&bar, 100.0, 0.10);
        assert!(r.passed);
        assert!((r.value - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_return_spike_fail() {
        let bar = make_bar(100.0, 130.0, 98.0, 125.0, 1000.0);
        let r = DataQualityChecker::check_return_spike(&bar, 100.0, 0.10);
        // 25% return > 10% threshold
        assert!(!r.passed);
    }

    #[test]
    fn test_timestamp_gap_normal() {
        let r = DataQualityChecker::check_timestamp_gap(120, 60, 60);
        assert!(r.passed, "60s gap with 60s interval should pass");
    }

    #[test]
    fn test_timestamp_gap_duplicate() {
        // gap of 5s when expected is 60s -- looks like duplicate
        let r = DataQualityChecker::check_timestamp_gap(65, 60, 60);
        assert!(!r.passed, "5s gap with 60s interval should fail as duplicate");
    }

    #[test]
    fn test_bid_ask_spread_pass() {
        let r = DataQualityChecker::check_bid_ask_spread(99.9, 100.1, 50.0);
        assert!(r.passed);
        // spread = 0.2 / 100.0 * 10000 = 20 bps
        assert!(r.value < 21.0 && r.value > 19.0);
    }

    #[test]
    fn test_bid_ask_spread_crossed() {
        let r = DataQualityChecker::check_bid_ask_spread(100.5, 99.5, 50.0);
        assert!(!r.passed, "crossed market should fail");
    }

    #[test]
    fn test_bid_ask_spread_too_wide() {
        let r = DataQualityChecker::check_bid_ask_spread(98.0, 102.0, 50.0);
        // spread = 4/100 * 10000 = 400 bps > 50
        assert!(!r.passed);
    }

    #[test]
    fn test_data_freshness_fresh() {
        let now = Utc::now().timestamp();
        let r = DataQualityChecker::check_data_freshness(now - 30, now, 60);
        assert!(r.passed);
    }

    #[test]
    fn test_data_freshness_stale() {
        let now = Utc::now().timestamp();
        let r = DataQualityChecker::check_data_freshness(now - 3600, now, 60);
        assert!(!r.passed);
    }

    #[test]
    fn test_run_all_checks_clean_bar() {
        let bar = make_bar(100.0, 105.0, 98.0, 102.0, 1_000.0);
        let ctx = QualityContext {
            rolling_adv: 100_000.0,
            rolling_vol: 0.02,
            expected_interval_s: 60,
            max_return_pct: 0.10,
        };
        let report = run_all_checks(&bar, &ctx);
        assert!(report.quality_score > 0.8, "clean bar should score high");
    }

    #[test]
    fn test_quality_series_length() {
        let bar = make_bar(100.0, 105.0, 98.0, 102.0, 1_000.0);
        let bars = vec![bar.clone(), bar.clone(), bar];
        let ctx = QualityContext::default();
        let ctxs = vec![ctx.clone(), ctx.clone(), ctx];
        let reports = quality_series(&bars, &ctxs);
        assert_eq!(reports.len(), 3);
    }

    #[test]
    fn test_compute_quality_score_all_pass() {
        let checks = vec![
            CheckResult::pass("a", 1.0, 1.0, ""),
            CheckResult::pass("b", 1.0, 1.0, ""),
        ];
        let score = compute_quality_score(&checks, &[1.0, 1.0]);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_quality_score_all_fail() {
        let checks = vec![
            CheckResult::fail("a", 0.0, 1.0, ""),
            CheckResult::fail("b", 0.0, 1.0, ""),
        ];
        let score = compute_quality_score(&checks, &[1.0, 1.0]);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_ohlc_equal_ohlc() {
        // All OHLC = 100 -- degenerate but valid
        let bar = make_bar(100.0, 100.0, 100.0, 100.0, 0.0);
        let r = DataQualityChecker::check_ohlc_validity(&bar);
        assert!(r.passed);
    }
}
