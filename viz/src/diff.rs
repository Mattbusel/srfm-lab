use std::fs;
use colored::*;

pub fn run_diff(file_a: &str, file_b: &str) -> Result<(), Box<dyn std::error::Error>> {
    let content_a = fs::read_to_string(file_a)?;
    let content_b = fs::read_to_string(file_b)?;

    let lines_a: Vec<&str> = content_a.lines().collect();
    let lines_b: Vec<&str> = content_b.lines().collect();

    // Find header comment (line 1 starting with #)
    let header_a = lines_a.first().unwrap_or(&"");
    let header_b = lines_b.first().unwrap_or(&"");

    // Find lines only in B (additions)
    let set_a: std::collections::HashSet<&str> = lines_a.iter().map(|l| l.trim()).collect();
    let additions: Vec<&str> = lines_b.iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !set_a.contains(l))
        .collect();

    // Find lines only in A (removals)
    let set_b: std::collections::HashSet<&str> = lines_b.iter().map(|l| l.trim()).collect();
    let removals: Vec<&str> = lines_a.iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !set_b.contains(l))
        .collect();

    println!("{}", "srfm-viz diff — strategy semantic diff".bold());
    println!("{}", "\u{2501}".repeat(50));
    println!("  {}  \u{2192}  {}", file_a.cyan(), file_b.cyan());
    println!("{}", "\u{2501}".repeat(50));

    if header_a != header_b {
        println!("\n  {}", "HEADER".bold().yellow());
        println!("  {} {}", "-".red(), header_a.red());
        println!("  {} {}", "+".green(), header_b.green());
    }

    if !additions.is_empty() {
        println!("\n  {} (+{} chars, +{} lines)",
            "ADDITIONS".bold().green(),
            content_b.len().saturating_sub(content_a.len()),
            lines_b.len().saturating_sub(lines_a.len()));
        for line in &additions[..additions.len().min(20)] {
            if !line.is_empty() {
                println!("  {} {}", "+".green().bold(), line.green());
            }
        }
    }

    if !removals.is_empty() {
        println!("\n  {}", "REMOVALS".bold().red());
        for line in &removals[..removals.len().min(20)] {
            if !line.is_empty() {
                println!("  {} {}", "-".red().bold(), line.red());
            }
        }
    }

    println!("\n  {}", "STATS".bold());
    println!("  File A: {} chars | File B: {} chars | Delta: {:+} chars",
        content_a.len(), content_b.len(),
        content_b.len() as i64 - content_a.len() as i64);
    println!("  Lines A: {} | Lines B: {} | Delta: {:+}",
        lines_a.len(), lines_b.len(),
        lines_b.len() as i64 - lines_a.len() as i64);

    let complexity = if removals.is_empty() { "LOW (additions only)" }
                     else if removals.len() < 5 { "MEDIUM" }
                     else { "HIGH (significant rewrites)" };
    println!("  Complexity: {}", complexity.yellow());
    println!("{}", "\u{2501}".repeat(50));

    Ok(())
}
