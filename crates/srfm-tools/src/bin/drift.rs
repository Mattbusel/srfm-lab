use std::fs;

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let num: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum();
    let dx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = y.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    if dx * dy < 1e-10 { 0.0 } else { num / (dx * dy) }
}

fn read_closes(path: &str) -> Vec<f64> {
    let content = fs::read_to_string(path).unwrap_or_default();
    let mut prices = Vec::new();
    let mut header_seen = false;
    let mut close_idx = 4usize;

    for line in content.lines() {
        if !header_seen {
            let cols: Vec<&str> = line.split(',').collect();
            for (i, col) in cols.iter().enumerate() {
                if col.to_lowercase().contains("close") {
                    close_idx = i;
                    break;
                }
            }
            header_seen = true;
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if let Some(v) = cols.get(close_idx).and_then(|s| s.trim().parse::<f64>().ok()) {
            prices.push(v);
        }
    }
    prices
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: drift file1.csv file2.csv [--window N] [--summary]");
        return;
    }

    let window: usize = args.iter().position(|a| a == "--window")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let summary = args.iter().any(|a| a == "--summary");

    let prices_a = read_closes(&args[1]);
    let prices_b = read_closes(&args[2]);
    let n = prices_a.len().min(prices_b.len());

    let mut corrs = Vec::new();
    for i in window..n {
        let c = pearson(&prices_a[i - window..i], &prices_b[i - window..i]);
        println!("{:.4}", c);
        corrs.push(c);
    }

    if summary && !corrs.is_empty() {
        let mean = corrs.iter().sum::<f64>() / corrs.len() as f64;
        eprintln!(
            "min={:.4} max={:.4} mean={:.4}",
            corrs.iter().cloned().fold(f64::INFINITY, f64::min),
            corrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            mean
        );
    }
}
