use plotters::prelude::*;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Experiment {
    pub exp: String,
    pub flags: Option<String>,
    pub arena_sharpe: f64,
    pub synth_sharpe: f64,
    pub arena_return: f64,
    pub combined_score: f64,
    pub overfit: Option<String>,
}

pub fn run(json_path: &str, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    let content = std::fs::read_to_string(json_path)?;
    let mut exps: Vec<Experiment> = serde_json::from_str(&content)?;

    // Sort by combined_score descending for bar chart
    exps.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

    let root = SVGBackend::new(out_path, (1600, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    root.draw(&Text::new(
        "SRFM Experiments — Arena vs Synth Sharpe",
        (400, 20),
        ("sans-serif", 24).into_font(),
    ))?;

    let (left_area, right_area) = root.split_horizontally(800);

    // --- Left panel: horizontal bar chart ranked by combined_score ---
    let n = exps.len();
    let score_min = exps
        .iter()
        .map(|e| e.combined_score)
        .fold(0.0f64, f64::min)
        .min(-0.1);
    let score_max = exps
        .iter()
        .map(|e| e.combined_score)
        .fold(0.0f64, f64::max)
        .max(0.1);

    let mut left_chart = ChartBuilder::on(&left_area)
        .caption("Combined Score (ranked)", ("sans-serif", 18).into_font())
        .margin(20)
        .margin_top(50)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(score_min..score_max, 0i32..n as i32)?;

    left_chart
        .configure_mesh()
        .y_labels(n.min(20))
        .y_label_formatter(&|v| {
            let idx = *v as usize;
            if idx < exps.len() {
                exps[idx].exp.clone()
            } else {
                String::new()
            }
        })
        .x_desc("Combined Score")
        .draw()?;

    for (i, exp) in exps.iter().enumerate() {
        let color = if exp.combined_score >= 0.0 {
            RGBColor(50, 168, 82)
        } else {
            RGBColor(220, 50, 47)
        };
        let bar_start = score_min.min(0.0);
        let x0 = bar_start.min(exp.combined_score);
        let x1 = bar_start.max(exp.combined_score);
        left_chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, i as i32), (x1, i as i32 + 1)],
            color.mix(0.8).filled(),
        )))?;
    }

    // --- Right panel: scatter arena_sharpe vs synth_sharpe ---
    let ax_min = exps
        .iter()
        .map(|e| e.arena_sharpe.min(e.synth_sharpe))
        .fold(f64::INFINITY, f64::min)
        - 0.2;
    let ax_max = exps
        .iter()
        .map(|e| e.arena_sharpe.max(e.synth_sharpe))
        .fold(f64::NEG_INFINITY, f64::max)
        + 0.2;

    let mut right_chart = ChartBuilder::on(&right_area)
        .caption("Arena Sharpe vs Synth Sharpe", ("sans-serif", 18).into_font())
        .margin(20)
        .margin_top(50)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(ax_min..ax_max, ax_min..ax_max)?;

    right_chart
        .configure_mesh()
        .x_desc("Arena Sharpe")
        .y_desc("Synth Sharpe")
        .draw()?;

    // Diagonal reference line
    right_chart.draw_series(std::iter::once(PathElement::new(
        vec![(ax_min, ax_min), (ax_max, ax_max)],
        Into::<ShapeStyle>::into(&BLACK.mix(0.3)).stroke_width(1),
    )))?;

    // Zero lines
    right_chart.draw_series(std::iter::once(PathElement::new(
        vec![(0.0, ax_min), (0.0, ax_max)],
        Into::<ShapeStyle>::into(&BLACK.mix(0.2)).stroke_width(1),
    )))?;
    right_chart.draw_series(std::iter::once(PathElement::new(
        vec![(ax_min, 0.0), (ax_max, 0.0)],
        Into::<ShapeStyle>::into(&BLACK.mix(0.2)).stroke_width(1),
    )))?;

    for exp in &exps {
        let color = if exp.combined_score >= 0.0 {
            RGBColor(50, 168, 82)
        } else {
            RGBColor(220, 50, 47)
        };
        right_chart.draw_series(std::iter::once(Circle::new(
            (exp.arena_sharpe, exp.synth_sharpe),
            5,
            color.filled(),
        )))?;
        right_chart.draw_series(std::iter::once(Text::new(
            exp.exp.clone(),
            (exp.arena_sharpe + 0.02, exp.synth_sharpe + 0.02),
            ("sans-serif", 10).into_font(),
        )))?;
    }

    root.present()?;
    println!("Experiments chart written to: {}", out_path);
    Ok(())
}
