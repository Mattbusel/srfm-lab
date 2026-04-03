use plotters::prelude::*;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct OhlcvRow {
    timestamp: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

pub fn run(csv_path: &str, cf: f64, out_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure output directory exists
    if let Some(parent) = Path::new(out_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Read CSV
    let mut rdr = csv::Reader::from_path(csv_path)?;
    let mut closes: Vec<f64> = Vec::new();
    for result in rdr.deserialize::<OhlcvRow>() {
        let row = result?;
        closes.push(row.close);
    }

    // Compute betas
    let mut betas: Vec<(usize, f64)> = Vec::new();
    for i in 1..closes.len() {
        let delta = (closes[i] - closes[i - 1]).abs();
        let beta = (delta / closes[i - 1]) / cf;
        let beta = beta.max(1e-4); // clamp for log scale
        betas.push((i, beta));
    }

    let filename = Path::new(csv_path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(csv_path);

    let title = format!("SRFM Spacetime Diagram — {}", filename);

    // Draw SVG
    let root = SVGBackend::new(out_path, (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let x_max = betas.len();
    let y_min_log = -1.0f64; // log10(0.1)
    let y_max_log = 1.0f64;  // log10(10)

    let mut chart = ChartBuilder::on(&root)
        .caption(&title, ("sans-serif", 28).into_font())
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0usize..x_max, y_min_log..y_max_log)?;

    chart
        .configure_mesh()
        .x_desc("Bar Index")
        .y_desc("Beta (log10 scale)")
        .y_label_formatter(&|v| format!("{:.2}", 10f64.powf(*v)))
        .draw()?;

    // Horizontal dashed line at beta=1.0 (log10=0)
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(0usize, 0.0f64), (x_max, 0.0f64)],
        Into::<ShapeStyle>::into(&BLACK.mix(0.5)).stroke_width(2),
    )))?;

    // TIMELIKE label
    chart.draw_series(std::iter::once(Text::new(
        "TIMELIKE (causal)",
        (10usize, -0.3f64),
        ("sans-serif", 16).into_font().color(&RGBColor(70, 130, 180)),
    )))?;

    // SPACELIKE label
    chart.draw_series(std::iter::once(Text::new(
        "SPACELIKE (forbidden)",
        (10usize, 0.7f64),
        ("sans-serif", 16).into_font().color(&RGBColor(220, 50, 47)),
    )))?;

    // Plot points
    let timelike_color = RGBColor(70, 130, 180);
    let spacelike_color = RGBColor(220, 50, 47);

    let timelike: Vec<(usize, f64)> = betas
        .iter()
        .filter(|(_, b)| *b < 1.0)
        .map(|(i, b)| (*i, b.log10()))
        .collect();

    let spacelike: Vec<(usize, f64)> = betas
        .iter()
        .filter(|(_, b)| *b >= 1.0)
        .map(|(i, b)| (*i, b.log10().min(y_max_log - 0.01)))
        .collect();

    chart.draw_series(timelike.into_iter().map(|(x, y)| {
        Circle::new((x, y), 3, timelike_color.filled())
    }))?
    .label("TIMELIKE (beta < 1)")
    .legend(|(x, y)| Circle::new((x, y), 4, RGBColor(70, 130, 180).filled()));

    chart.draw_series(spacelike.into_iter().map(|(x, y)| {
        Circle::new((x, y), 3, spacelike_color.filled())
    }))?
    .label("SPACELIKE (beta >= 1)")
    .legend(|(x, y)| Circle::new((x, y), 4, RGBColor(220, 50, 47).filled()));

    chart.configure_series_labels().border_style(BLACK).draw()?;

    root.present()?;
    println!("Spacetime diagram written to: {}", out_path);
    Ok(())
}
