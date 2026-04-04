/// Train/validation/test splitting for time-series ML pipelines.
/// Implements walk-forward, purged k-fold, and combinatorial purged CV.

use std::ops::Range;

// ── Walk-Forward Splits ───────────────────────────────────────────────────────

/// A single walk-forward split containing (train, val, test) index ranges.
#[derive(Debug, Clone)]
pub struct WalkForwardSplit {
    pub train: Range<usize>,
    pub val: Range<usize>,
    pub test: Range<usize>,
}

/// Generate walk-forward splits with expanding or rolling training window.
///
/// * `n_bars` — total number of observations.
/// * `train_size` — initial (minimum) training set size.
/// * `val_size` — validation window size.
/// * `test_size` — test (out-of-sample) window size.
/// * `step` — step between successive splits.
/// * `expanding` — if true, training window expands; if false, rolling window.
pub fn walk_forward_splits(
    n_bars: usize,
    train_size: usize,
    val_size: usize,
    test_size: usize,
    step: usize,
    expanding: bool,
) -> Vec<WalkForwardSplit> {
    let mut splits = Vec::new();
    let total_required = train_size + val_size + test_size;
    if n_bars < total_required { return splits; }

    let mut train_start = 0usize;
    let mut train_end = train_size;

    loop {
        let val_end = train_end + val_size;
        let test_end = val_end + test_size;
        if test_end > n_bars { break; }

        splits.push(WalkForwardSplit {
            train: train_start..train_end,
            val: train_end..val_end,
            test: val_end..test_end,
        });

        train_end += step;
        if !expanding {
            train_start += step;
        }
    }
    splits
}

// ── Purged K-Fold ─────────────────────────────────────────────────────────────

/// A single k-fold split with (train_indices, test_indices).
#[derive(Debug, Clone)]
pub struct KFoldSplit {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
    pub fold: usize,
}

/// Purged k-fold cross-validation for time series.
///
/// Unlike standard k-fold, this:
/// 1. Respects time ordering (test folds are contiguous).
/// 2. Purges `embargo_bars` from the training set immediately before and after each test fold
///    to avoid data leakage from overlapping features.
///
/// * `n_bars` — number of observations.
/// * `k` — number of folds.
/// * `embargo_bars` — number of bars to purge from training on each side of the test fold.
pub fn purged_kfold(n_bars: usize, k: usize, embargo_bars: usize) -> Vec<KFoldSplit> {
    assert!(k >= 2, "Need at least 2 folds");
    let fold_size = n_bars / k;
    if fold_size == 0 { return vec![]; }

    (0..k)
        .map(|fold| {
            let test_start = fold * fold_size;
            let test_end = if fold == k - 1 { n_bars } else { (fold + 1) * fold_size };
            let test: Vec<usize> = (test_start..test_end).collect();

            // Purge: exclude embargo_bars before and after test fold.
            let purge_start = test_start.saturating_sub(embargo_bars);
            let purge_end = (test_end + embargo_bars).min(n_bars);

            let train: Vec<usize> = (0..n_bars)
                .filter(|&i| i < purge_start || i >= purge_end)
                .collect();

            KFoldSplit { train, test, fold }
        })
        .collect()
}

// ── Combinatorial Purged CV (CPCV) ────────────────────────────────────────────

/// A single CPCV split pair.
#[derive(Debug, Clone)]
pub struct SplitPair {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
    pub path_id: usize,
}

/// Combinatorial Purged Cross-Validation (López de Prado 2018).
///
/// Generates all C(k, n_paths) combinations of test folds, with purging.
/// This allows backtesting many distinct "paths" while controlling for overfitting.
///
/// * `n_bars` — number of observations.
/// * `k` — number of groups to split into.
/// * `n_test_groups` — number of groups in each test set (usually 2).
/// * `embargo_bars` — bars to embargo on each test-group boundary.
pub fn combinatorial_purged_cv(
    n_bars: usize,
    k: usize,
    n_test_groups: usize,
    embargo_bars: usize,
) -> Vec<SplitPair> {
    assert!(n_test_groups <= k);
    let fold_size = n_bars / k;
    if fold_size == 0 { return vec![]; }

    // Group boundaries.
    let groups: Vec<Range<usize>> = (0..k)
        .map(|i| {
            let start = i * fold_size;
            let end = if i == k - 1 { n_bars } else { (i + 1) * fold_size };
            start..end
        })
        .collect();

    // Generate all combinations of n_test_groups from k groups.
    let combos = combinations(k, n_test_groups);

    combos
        .into_iter()
        .enumerate()
        .map(|(path_id, test_group_indices)| {
            // Test: union of selected groups.
            let test: Vec<usize> = test_group_indices
                .iter()
                .flat_map(|&gi| groups[gi].clone())
                .collect();

            // Purge region: all test indices + embargo.
            let test_min = test.iter().copied().min().unwrap_or(0);
            let test_max = test.iter().copied().max().unwrap_or(0);
            let purge_start = test_min.saturating_sub(embargo_bars);
            let purge_end = (test_max + 1 + embargo_bars).min(n_bars);

            let train: Vec<usize> = (0..n_bars)
                .filter(|&i| i < purge_start || i >= purge_end)
                .collect();

            SplitPair { train, test, path_id }
        })
        .collect()
}

/// Generate all C(n, k) combinations.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut combo = vec![0usize; k];
    for i in 0..k { combo[i] = i; }

    loop {
        result.push(combo.clone());
        // Find rightmost element that can be incremented.
        let mut i = k as isize - 1;
        while i >= 0 && combo[i as usize] == n - k + i as usize {
            i -= 1;
        }
        if i < 0 { break; }
        combo[i as usize] += 1;
        for j in (i as usize + 1)..k {
            combo[j] = combo[j - 1] + 1;
        }
    }
    result
}

// ── Embargo-only splits ────────────────────────────────────────────────────────

/// Simple time-series train/test split with an embargo gap between them.
pub fn train_test_split_with_embargo(
    n_bars: usize,
    train_frac: f64,
    embargo_bars: usize,
) -> (Vec<usize>, Vec<usize>) {
    let train_n = (n_bars as f64 * train_frac) as usize;
    let test_start = (train_n + embargo_bars).min(n_bars);
    let train: Vec<usize> = (0..train_n).collect();
    let test: Vec<usize> = (test_start..n_bars).collect();
    (train, test)
}

// ── Split Statistics ──────────────────────────────────────────────────────────

/// Summary of a split set.
#[derive(Debug)]
pub struct SplitStats {
    pub n_splits: usize,
    pub avg_train_size: f64,
    pub avg_test_size: f64,
    pub min_train_size: usize,
    pub max_train_size: usize,
}

pub fn walk_forward_stats(splits: &[WalkForwardSplit]) -> SplitStats {
    let n = splits.len();
    if n == 0 {
        return SplitStats { n_splits: 0, avg_train_size: 0.0, avg_test_size: 0.0, min_train_size: 0, max_train_size: 0 };
    }
    let train_sizes: Vec<usize> = splits.iter().map(|s| s.train.end - s.train.start).collect();
    let test_sizes: Vec<usize> = splits.iter().map(|s| s.test.end - s.test.start).collect();
    SplitStats {
        n_splits: n,
        avg_train_size: train_sizes.iter().sum::<usize>() as f64 / n as f64,
        avg_test_size: test_sizes.iter().sum::<usize>() as f64 / n as f64,
        min_train_size: *train_sizes.iter().min().unwrap(),
        max_train_size: *train_sizes.iter().max().unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_forward_expanding_count() {
        let splits = walk_forward_splits(252, 60, 20, 21, 21, true);
        assert!(!splits.is_empty());
        // Each split's test should be non-overlapping.
        for s in &splits {
            assert!(s.train.end <= s.val.start);
            assert!(s.val.end <= s.test.start);
        }
    }

    #[test]
    fn purged_kfold_no_leakage() {
        let splits = purged_kfold(100, 5, 5);
        assert_eq!(splits.len(), 5);
        for split in &splits {
            // No overlap between train and test.
            let test_set: std::collections::HashSet<usize> = split.test.iter().copied().collect();
            for &i in &split.train {
                assert!(!test_set.contains(&i), "Leakage at index {i}");
            }
        }
    }

    #[test]
    fn combinations_c5k2() {
        let combos = combinations(5, 2);
        assert_eq!(combos.len(), 10); // C(5,2) = 10
    }

    #[test]
    fn cpcv_basic() {
        let splits = combinatorial_purged_cv(100, 5, 2, 3);
        assert_eq!(splits.len(), 10); // C(5,2)
    }
}
