use crate::backtest::{run_backtest, BacktestResult};
use crate::bar_data::DataStore;
use crate::params::StrategyParams;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Run a parameter sweep in parallel over all provided parameter sets.
///
/// Uses Rayon for work-stealing parallelism across `n_threads` threads.
/// Progress is printed to stderr every 100 completions.
/// Results with Sharpe < -0.5 are filtered out.
pub fn sweep(
    data: &DataStore,
    params_list: Vec<StrategyParams>,
    n_threads: usize,
) -> Vec<(StrategyParams, BacktestResult)> {
    // Configure Rayon thread pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("failed to build Rayon thread pool");

    let total = params_list.len();
    let counter = Arc::new(AtomicUsize::new(0));

    eprintln!(
        "[sweep] Starting {} variants on {} threads",
        total, n_threads
    );

    let results: Vec<(StrategyParams, BacktestResult)> = pool.install(|| {
        params_list
            .into_par_iter()
            .filter_map(|params| {
                let result = run_backtest(data, &params);

                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 100 == 0 || done == total {
                    eprintln!("[sweep] {}/{} completed (latest sharpe={:.3})", done, total, result.sharpe);
                }

                // Filter garbage runs.
                if result.sharpe < -0.5 {
                    return None;
                }

                Some((params, result))
            })
            .collect()
    });

    eprintln!(
        "[sweep] Done. {} / {} variants passed sharpe filter.",
        results.len(),
        total
    );

    results
}

/// Sort results by Sharpe ratio descending and return top-N.
pub fn top_n_by_sharpe(
    mut results: Vec<(StrategyParams, BacktestResult)>,
    n: usize,
) -> Vec<(StrategyParams, BacktestResult)> {
    results.sort_unstable_by(|a, b| {
        b.1.sharpe
            .partial_cmp(&a.1.sharpe)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(n);
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bar_data::BarData;
    use crate::params::ParameterSpace;

    fn tiny_data() -> DataStore {
        let mut data = DataStore::new();
        let bars: Vec<BarData> = (0..300)
            .map(|i| {
                let t = 1_600_000_000_i64 + (i as i64) * 900;
                let price = 50000.0 * (1.0 + 0.0002_f64 * i as f64);
                BarData {
                    symbol: "BTC".to_string(),
                    timestamp: t,
                    open: price,
                    high: price * 1.001,
                    low: price * 0.999,
                    close: price,
                    volume: 5.0,
                }
            })
            .collect();
        data.insert("BTC".to_string(), bars);
        data
    }

    #[test]
    fn test_sweep_returns_results() {
        let data = tiny_data();
        let space = ParameterSpace::default();
        let params_list = space.sample(20);
        let results = sweep(&data, params_list, 2);
        // We don't mandate a specific count because of the sharpe filter, just no panic.
        assert!(results.len() <= 20);
    }

    #[test]
    fn test_sweep_filters_low_sharpe() {
        let data = tiny_data();
        let space = ParameterSpace::default();
        let params_list = space.sample(10);
        let results = sweep(&data, params_list, 1);
        for (_, res) in &results {
            assert!(res.sharpe >= -0.5, "sharpe filter violated: {}", res.sharpe);
        }
    }

    #[test]
    fn test_top_n_sorted() {
        let params = StrategyParams::default();
        let make_result = |sharpe: f64| {
            let mut r = BacktestResult::default();
            r.sharpe = sharpe;
            (params.clone(), r)
        };
        let results = vec![make_result(0.5), make_result(2.0), make_result(1.0)];
        let top = top_n_by_sharpe(results, 2);
        assert_eq!(top.len(), 2);
        assert!((top[0].1.sharpe - 2.0).abs() < 1e-9);
        assert!((top[1].1.sharpe - 1.0).abs() < 1e-9);
    }
}
