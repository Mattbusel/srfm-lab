use plotters::prelude::*;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Well {
    start: String,
    end: String,
    total_pnl: f64,
    instruments: Vec<String>,
    is_win: bool,
    duration_h: f64,
}

#[derive(Debug, Deserialize)]
struct TradeData {
    wells: Vec<Well>,
}

#[allow(dead_code)]
struct GroupStats {
    label: String,
    count: usize,
    win_rate: f64,
    avg_pnl: f64,
    pct_total_pnl: f64,
}

pub fn run(json_path: &str, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let content = std::fs::read_to_string(json_path)?;
    let data: TradeData = serde_json::from_str(&content)?;

    let total_pnl: f64 = data.wells.iter().map(|w| w.total_pnl).sum();

    let mut groups: Vec<(&str, Vec<&Well>)> = vec![
        ("Single", vec![]),
        ("Dual", vec![]),
        ("Triple+", vec![]),
    ];

    for well in &data.wells {
        let n = well.instruments.len();
        if n == 1 {
            groups[0].1.push(well);
        } else if n == 2 {
            groups[1].1.push(well);
        } else {
            groups[2].1.push(well);
        }
    }

    let stats: Vec<GroupStats> = groups
        .iter()
        .map(|(label, wells)| {
            let count = wells.len();
            let wins = wells.iter().filter(|w| w.is_win).count();
            let win_rate = if count > 0 {
                100.0 * wins as f64 / count as f64
            } else {
                0.0
            };
            let sum_pnl: f64 = wells.iter().map(|w| w.total_pnl).sum();
            let avg_pnl = if count > 0 { sum_pnl / count as f64 } else { 0.0 };
            let pct = if total_pnl.abs() > 0.0 {
                100.0 * sum_pnl / total_pnl
            } else {
                0.0
            };
            GroupStats {
                label: label.to_string(),
                count,
                win_rate,
                avg_pnl,
                pct_total_pnl: pct,
            }
        })
        .collect();

    // We'll render a grouped bar chart: 3 groups × 4 bars
    // Map to positions: group_idx * 5 + metric_idx (with gap of 1 between groups)
    // metrics: count, win_rate, avg_pnl($k), pct_total_pnl
    let n_groups = stats.len();
    let n_metrics = 4usize;
    let group_width = n_metrics + 1; // 4 bars + 1 gap
    let total_bars = n_groups * group_width;

    // Normalize each metric to 0-100 range for display on same axis
    let max_count = stats.iter().map(|s| s.count).max().unwrap_or(1) as f64;
    let max_pnl_k = stats
        .iter()
        .map(|s| s.avg_pnl.abs() / 1000.0)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let max_pct = stats
        .iter()
        .map(|s| s.pct_total_pnl.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);

    let root = SVGBackend::new(out_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Well Convergence — Instrument Count Groups",
            ("sans-serif", 22).into_font(),
        )
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..total_bars, 0.0f64..110.0f64)?;

    chart
        .configure_mesh()
        .x_labels(n_groups * 2)
        .x_label_formatter(&|v| {
            let group = *v / group_width;
            let offset = *v % group_width;
            if offset == 1 {
                // middle-ish of group — label it
                match group {
                    0 => "Single".to_string(),
                    1 => "Dual".to_string(),
                    2 => "Triple+".to_string(),
                    _ => String::new(),
                }
            } else {
                String::new()
            }
        })
        .y_desc("Normalized Value (%) / Count")
        .draw()?;

    let blue = RGBColor(70, 130, 180);
    let gold = RGBColor(218, 165, 32);
    let metric_colors = [blue, blue, gold, gold];
    let metric_labels = ["Count (norm)", "Win Rate (%)", "Avg P&L ($k norm)", "% Total P&L"];

    for (gi, grp) in stats.iter().enumerate() {
        let base = gi * group_width;
        let normalized = [
            100.0 * grp.count as f64 / max_count,
            grp.win_rate,
            100.0 * (grp.avg_pnl / 1000.0) / max_pnl_k,
            100.0 * grp.pct_total_pnl / max_pct,
        ];
        for (mi, &val) in normalized.iter().enumerate() {
            let x = base + mi;
            let y0 = 0.0f64.min(val);
            let y1 = 0.0f64.max(val);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, y0), (x + 1, y1)],
                metric_colors[mi].mix(0.8).filled(),
            )))?;
        }
    }

    // Legend
    for (i, label) in metric_labels.iter().enumerate() {
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(total_bars - 6 + i, 100.0), (total_bars - 5 + i, 105.0)],
                metric_colors[i].filled(),
            )))?
            .label(*label)
            .legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 12, y + 5)], metric_colors[i].filled())
            });
    }

    chart.configure_series_labels().border_style(BLACK).draw()?;

    root.present()?;
    println!("Convergence chart written to: {}", out_path);
    Ok(())
}
