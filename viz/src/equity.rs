use chrono::NaiveDateTime;
use plotters::prelude::*;
use serde::Deserialize;

use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct YearStats {
    pnl: f64,
}

#[derive(Debug, Deserialize)]
struct TradeData {
    equity_curve: Vec<(String, f64)>,
    by_year: HashMap<String, YearStats>,
}

pub fn run(json_path: &str, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let content = std::fs::read_to_string(json_path)?;
    let data: TradeData = serde_json::from_str(&content)?;

    let equity: Vec<(i64, f64)> = data
        .equity_curve
        .iter()
        .filter_map(|(ts, val)| {
            parse_timestamp(ts).map(|t| (t, *val))
        })
        .collect();

    if equity.is_empty() {
        println!("No equity curve data.");
        return Ok(());
    }

    let mut years: Vec<(i32, f64)> = data
        .by_year
        .iter()
        .filter_map(|(yr, stats)| yr.parse::<i32>().ok().map(|y| (y, stats.pnl)))
        .collect();
    years.sort_by_key(|(y, _)| *y);

    let x_min = equity.first().map(|(t, _)| *t).unwrap_or(0);
    let x_max = equity.last().map(|(t, _)| *t).unwrap_or(1);
    let y_min = equity.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let y_max = equity.iter().map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min) * 0.05;

    let root = SVGBackend::new(out_path, (1600, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let (top_area, bottom_area) = root.split_vertically(500);

    // --- Top: equity curve ---
    let mut top_chart = ChartBuilder::on(&top_area)
        .caption("SRFM Equity Curve — LARSA 274% Backtest", ("sans-serif", 22).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(x_min..x_max, (y_min - y_pad)..(y_max + y_pad))?;

    top_chart
        .configure_mesh()
        .x_label_formatter(&|v| {
            let secs = *v;
            let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(secs, 0);
            match dt {
                Some(dt) => dt.format("%Y").to_string(),
                None => String::new(),
            }
        })
        .y_label_formatter(&|v| format!("${:.0}K", v / 1000.0))
        .y_desc("Portfolio Value")
        .draw()?;

    // Year background bands
    let band_colors = [
        RGBColor(245, 245, 255),
        RGBColor(255, 255, 245),
    ];
    for (i, (yr, _)) in years.iter().enumerate() {
        let yr_start = chrono::NaiveDate::from_ymd_opt(*yr, 1, 1)
            .and_then(|d| d.and_hms_opt(0, 0, 0))
            .map(|dt| dt.and_utc().timestamp())
            .unwrap_or(x_min);
        let yr_end = chrono::NaiveDate::from_ymd_opt(*yr, 12, 31)
            .and_then(|d| d.and_hms_opt(23, 59, 59))
            .map(|dt| dt.and_utc().timestamp())
            .unwrap_or(x_max);
        let x0 = yr_start.max(x_min);
        let x1 = yr_end.min(x_max);
        let color = band_colors[i % 2];
        top_chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y_min - y_pad), (x1, y_max + y_pad)],
            color.filled(),
        )))?;
    }

    // Equity line
    top_chart.draw_series(LineSeries::new(
        equity.iter().map(|(t, v)| (*t, *v)),
        Into::<ShapeStyle>::into(&BLACK).stroke_width(2),
    ))?;

    // --- Bottom: annual P&L bars ---
    if !years.is_empty() {
        let pnl_min = years.iter().map(|(_, p)| *p).fold(0.0f64, f64::min).min(0.0);
        let pnl_max = years.iter().map(|(_, p)| *p).fold(0.0f64, f64::max).max(0.0);
        let pnl_pad = (pnl_max - pnl_min).max(1.0) * 0.1;

        let year_labels: Vec<String> = years.iter().map(|(y, _)| y.to_string()).collect();
        let n = years.len();

        let mut bottom_chart = ChartBuilder::on(&bottom_area)
            .caption("Annual P&L", ("sans-serif", 16).into_font())
            .margin(20)
            .x_label_area_size(30)
            .y_label_area_size(80)
            .build_cartesian_2d(0usize..n, (pnl_min - pnl_pad)..(pnl_max + pnl_pad))?;

        bottom_chart
            .configure_mesh()
            .x_labels(n)
            .x_label_formatter(&|v| {
                year_labels.get(*v).cloned().unwrap_or_default()
            })
            .y_label_formatter(&|v| format!("${:.0}K", v / 1000.0))
            .draw()?;

        for (i, (_, pnl)) in years.iter().enumerate() {
            let color = if *pnl >= 0.0 {
                RGBColor(50, 168, 82)
            } else {
                RGBColor(220, 50, 47)
            };
            let y0 = 0.0f64.min(*pnl);
            let y1 = 0.0f64.max(*pnl);
            bottom_chart.draw_series(std::iter::once(Rectangle::new(
                [(i, y0), (i + 1, y1)],
                color.mix(0.8).filled(),
            )))?;
        }
    }

    root.present()?;
    println!("Equity chart written to: {}", out_path);
    Ok(())
}

fn parse_timestamp(s: &str) -> Option<i64> {
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.and_utc().timestamp());
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(dt.and_utc().timestamp());
    }
    None
}
