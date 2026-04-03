use std::io::{self, BufRead};
use colored::*;

fn main() {
    let cf: f64 = std::env::args()
        .position(|a| a == "--cf")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.001);

    let stream = std::env::args().any(|a| a == "--stream");

    let stdin = io::stdin();
    let mut prev_price: Option<f64> = None;
    let mut mass: f64 = 0.0;
    let bh_form = 1.5;

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let price: f64 = match line.trim().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        if let Some(prev) = prev_price {
            let ret = (price - prev).abs() / prev;
            let beta = ret / cf;
            let timelike = beta < 1.0;

            if timelike {
                mass = mass * 0.95 + beta * (1.0 - 0.95) * 10.0;
            } else {
                mass *= 0.7;
            }

            let filled = ((mass / 2.5) * 20.0).min(20.0) as usize;
            let empty = 20 - filled;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

            let active = mass >= bh_form;
            let status = if active {
                format!("mass={:.2} BH ACTIVE", mass)
            } else {
                format!("mass={:.2} (needs {:.2} to form)", mass, bh_form)
            };

            let bar_str = if active { bar.green().to_string() } else { bar.normal().to_string() };
            let status_str = if active { status.green().bold().to_string() } else { status.normal().to_string() };

            if stream {
                print!("\r{} {}", bar_str, status_str);
                use std::io::Write;
                io::stdout().flush().unwrap();
            } else {
                println!("{} {}", bar_str, status_str);
            }
        }
        prev_price = Some(price);
    }
    if stream {
        println!();
    }
}
