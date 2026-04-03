use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime};
use plotters::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
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

fn parse_dt(s: &str) -> Option<NaiveDate> {
    // Try full datetime first, then date-only
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(dt.date());
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(dt.date());
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(d);
    }
    None
}

pub fn run(json_path: &str, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let content = std::fs::read_to_string(json_path)?;
    let data: TradeData = serde_json::from_str(&content)?;

    // Build day -> pnl sum map
    let mut day_pnl: HashMap<NaiveDate, f64> = HashMap::new();
    for well in &data.wells {
        let start = match parse_dt(&well.start) {
            Some(d) => d,
            None => continue,
        };
        let end = match parse_dt(&well.end) {
            Some(d) => d,
            None => continue,
        };
        // Distribute pnl evenly across days in the well
        let mut d = start;
        let mut day_count = 0i64;
        while d <= end {
            day_count += 1;
            d = d + Duration::days(1);
        }
        let pnl_per_day = if day_count > 0 {
            well.total_pnl / day_count as f64
        } else {
            well.total_pnl
        };
        let mut d = start;
        while d <= end {
            *day_pnl.entry(d).or_insert(0.0) += pnl_per_day;
            d = d + Duration::days(1);
        }
    }

    if day_pnl.is_empty() {
        println!("No well data to render.");
        return Ok(());
    }

    let min_date = *day_pnl.keys().min().unwrap();
    let max_date = *day_pnl.keys().max().unwrap();

    // Find Monday of the week containing min_date
    let days_from_mon = min_date.weekday().num_days_from_monday() as i64;
    let week_start = min_date - Duration::days(days_from_mon);

    // Find Sunday of the week containing max_date
    let days_to_sun = 6 - max_date.weekday().num_days_from_monday() as i64;
    let week_end = max_date + Duration::days(days_to_sun);

    let total_days = (week_end - week_start).num_days() + 1;
    let total_weeks = (total_days / 7) as usize;

    // Collect pnl max for scaling
    let max_abs_pnl = day_pnl.values().map(|v| v.abs()).fold(0.0f64, f64::max);
    let scale = if max_abs_pnl > 0.0 { max_abs_pnl } else { 1.0 };

    // SVG: 1800x400
    // Layout: left margin for day labels, top margin for year labels, cells
    let width = 1800u32;
    let height = 400u32;
    let margin_left = 60i32;
    let margin_top = 60i32;
    let margin_right = 20i32;
    let margin_bottom = 20i32;
    let draw_w = width as i32 - margin_left - margin_right;
    let draw_h = height as i32 - margin_top - margin_bottom;

    let cell_w = if total_weeks > 0 {
        draw_w / total_weeks as i32
    } else {
        10
    };
    let cell_h = draw_h / 7;

    let root = SVGBackend::new(out_path, (width, height)).into_drawing_area();
    root.fill(&RGBColor(30, 30, 30))?;

    // Title
    root.draw(&Text::new(
        "Well Calendar — LARSA 274% Backtest",
        (width as i32 / 2 - 200, 18),
        ("sans-serif", 22).into_font().color(&WHITE),
    ))?;

    // Day of week labels
    let days_label = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    for (i, label) in days_label.iter().enumerate() {
        root.draw(&Text::new(
            *label,
            (5, margin_top + cell_h * i as i32 + cell_h / 2 - 6),
            ("sans-serif", 11).into_font().color(&RGBColor(180, 180, 180)),
        ))?;
    }

    // Draw cells
    let mut last_year = 0i32;
    for week_idx in 0..total_weeks {
        for dow in 0..7usize {
            let date = week_start + Duration::days((week_idx * 7 + dow) as i64);
            let x = margin_left + cell_w * week_idx as i32;
            let y = margin_top + cell_h * dow as i32;

            // Year label at first week of year
            if date.year() != last_year && dow == 0 {
                root.draw(&Text::new(
                    format!("{}", date.year()),
                    (x, margin_top - 12),
                    ("sans-serif", 11).into_font().color(&RGBColor(200, 200, 200)),
                ))?;
                last_year = date.year();
            }

            let pnl = day_pnl.get(&date).copied().unwrap_or(0.0);
            let cell_color = if pnl > 0.0 {
                let intensity = ((pnl / scale) * 200.0).min(200.0) as u8 + 55;
                RGBColor(0, intensity, 0)
            } else if pnl < 0.0 {
                let intensity = ((pnl.abs() / scale) * 200.0).min(200.0) as u8 + 55;
                RGBColor(intensity, 0, 0)
            } else {
                RGBColor(50, 50, 50)
            };

            let rect = Rectangle::new(
                [(x + 1, y + 1), (x + cell_w - 2, y + cell_h - 2)],
                cell_color.filled(),
            );
            root.draw(&rect)?;
        }
    }

    root.present()?;
    println!("Wells calendar written to: {}", out_path);
    Ok(())
}
