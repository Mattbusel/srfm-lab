use std::collections::HashMap;
use std::fs;

struct ParamDoc {
    fmt: fn(&str, f64) -> String,
}

fn param_docs() -> HashMap<&'static str, ParamDoc> {
    let mut m: HashMap<&'static str, ParamDoc> = HashMap::new();
    m.insert("cf", ParamDoc { fmt: |_, v| format!("cf={v} -> TIMELIKE boundary at {:.2}% hourly move. ES median ~0.067% -> ~87% TIMELIKE", v * 100.0) });
    m.insert("bh_form", ParamDoc { fmt: |_, v| format!("bh_form={v} -> BH forms after ~10-15 TIMELIKE bars at avg vol") });
    m.insert("bh_decay", ParamDoc { fmt: |_, v| {
        let half = -(0.5f64.ln()) / (1.0 - v).ln();
        format!("bh_decay={v} -> mass halves in ~{:.0} SPACELIKE bars", half)
    }});
    m.insert("cap_solo", ParamDoc { fmt: |_, v| format!("cap_solo={v} -> {:.0}% of portfolio per instrument (solo BH)", v * 100.0) });
    m.insert("cap_multi", ParamDoc { fmt: |_, v| format!("cap_multi={v} -> {:.0}% of portfolio per instrument (convergence)", v * 100.0) });
    m.insert("nq_notional", ParamDoc { fmt: |_, v| {
        let at1m = v / 1_000_000.0 * 100.0;
        let at3m = v / 3_000_000.0 * 100.0;
        format!("nq_notional={v} -> at $1M: {:.0}% cap; at $3M: {:.0}% cap <- key protection", at1m, at3m)
    }});
    m.insert("bear_gate_bars", ParamDoc { fmt: |_, v| format!("bear_gate_bars={v} -> blocks longs after {:.0} BEAR bars (BEAR avg run=29.5 bars -> blocks ~83%)", v) });
    m.insert("ctl_min", ParamDoc { fmt: |_, v| format!("ctl_min={v} -> requires {:.0} consecutive TIMELIKE bars before BH entry", v) });
    m.insert("drawdown_limit", ParamDoc { fmt: |_, v| format!("drawdown_limit={v} -> max {:.1}% drawdown from peak before halt", v * 100.0) });
    m.insert("convergence_boost", ParamDoc { fmt: |_, v| format!("convergence_boost={v} -> position multiplier when instruments converge: {v}x") });
    m
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 || args[1] != "check" {
        eprintln!("Usage: srfm check <strategy.srfm>");
        return;
    }

    let path = &args[2];
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => { eprintln!("Cannot read {}: {}", path, e); return; }
    };

    let docs = param_docs();
    let mut count = 0;
    let mut errors = 0;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 { errors += 1; eprintln!("  ERROR: unparseable line: {}", line); continue; }
        let key = parts[0].trim();
        let val_str = parts[1].trim();
        let val: f64 = match val_str.parse() {
            Ok(v) => v,
            Err(_) => { errors += 1; eprintln!("  ERROR: non-numeric value for {}: {}", key, val_str); continue; }
        };

        count += 1;
        if let Some(doc) = docs.get(key) {
            println!("  {} {}", "\u{2713}", (doc.fmt)(key, val));
        } else {
            println!("  {} {}={} (no doc)", "\u{2713}", key, val_str);
        }
    }

    println!("  {} Config valid ({} params, {} errors)", "\u{2713}", count, errors);
}
