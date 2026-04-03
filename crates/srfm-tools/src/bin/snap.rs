use std::io::{self, Read};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C, packed)]
struct SRFMState {
    timestamp: i64,
    bh_mass: f64,
    ctl: i64,
    price: f64,
    portfolio_value: f64,
    peak_equity: f64,
    convergence: f64,
    flags: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let save = args.iter().position(|a| a == "--save").and_then(|i| args.get(i + 1));
    let load = args.iter().position(|a| a == "--load").and_then(|i| args.get(i + 1));

    if let Some(path) = save {
        let mut input = String::new();
        io::stdin().read_to_string(&mut input).unwrap();
        let v: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::json!({}));

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let state = SRFMState {
            timestamp: now,
            bh_mass: v["bh_mass"].as_f64().unwrap_or(0.0),
            ctl: v["ctl"].as_i64().unwrap_or(0),
            price: v["price"].as_f64().unwrap_or(0.0),
            portfolio_value: v["portfolio_value"].as_f64().unwrap_or(1_000_000.0),
            peak_equity: v["peak_equity"].as_f64().unwrap_or(1_000_000.0),
            convergence: v["convergence"].as_f64().unwrap_or(1.0),
            flags: if v["bh_active"].as_bool().unwrap_or(false) { 1 } else { 0 },
        };

        let bytes: [u8; 64] = unsafe { std::mem::transmute(state) };
        fs::write(path, &bytes).unwrap();
        eprintln!("Saved 64-byte SRFM state -> {}", path);
    } else if let Some(path) = load {
        let bytes = fs::read(path).unwrap();
        if bytes.len() < 64 {
            eprintln!("Invalid state file");
            return;
        }
        let state: SRFMState = unsafe { std::mem::transmute_copy(&bytes[0]) };
        // Copy packed fields to locals to avoid unaligned reference errors
        let ts = state.timestamp;
        let bh_mass = state.bh_mass;
        let ctl = state.ctl;
        let price = state.price;
        let pv = state.portfolio_value;
        let pe = state.peak_equity;
        let conv = state.convergence;
        let flags = state.flags;
        println!("{{");
        println!("  \"timestamp\": {},", ts);
        println!("  \"bh_mass\": {:.4},", bh_mass);
        println!("  \"ctl\": {},", ctl);
        println!("  \"price\": {:.2},", price);
        println!("  \"portfolio_value\": {:.0},", pv);
        println!("  \"peak_equity\": {:.0},", pe);
        println!("  \"convergence\": {:.2},", conv);
        println!("  \"bh_active\": {}", (flags & 1) == 1);
        println!("}}");
    } else {
        eprintln!("Usage: snap --save state.bin  |  snap --load state.bin");
    }
}
