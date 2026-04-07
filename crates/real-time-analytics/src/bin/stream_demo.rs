// bin/stream_demo.rs -- Real-time streaming analytics demo.
//
// Generates synthetic OHLCV bar data in a loop and applies all filters,
// detectors, and regime estimators. Prints a formatted real-time dashboard.

use real_time_analytics::{
    // Filters
    KalmanFilter1D, KalmanFilter2D, AdaptiveEMA,
    // Detectors
    VolatilityBreakout, MomentumShift, VolumeAnomaly, OrderFlowReversal,
    // Analytics
    VPINEstimator, ToxicityMeter,
    CompositeRegime, RegimeDetector,
    // Bar types
    Bar, BarType,
};
use chrono::Utc;
use std::f64::consts::PI;

// ─── Synthetic bar generator ──────────────────────────────────────────────────

struct SyntheticMarket {
    price: f64,
    trend: f64,
    vol: f64,
    t: usize,
    rng_state: u64,
}

impl SyntheticMarket {
    fn new(seed: u64) -> Self {
        Self {
            price: 100.0,
            trend: 0.0001,
            vol: 0.005,
            t: 0,
            rng_state: seed,
        }
    }

    // LCG pseudo-random number generator returning float in [-1, 1]
    fn rand(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = self.rng_state >> 11;
        // Map to [0, 1] then shift to [-1, 1]
        (bits as f64 / (u64::MAX >> 11) as f64) * 2.0 - 1.0
    }

    fn rand_pos(&mut self) -> f64 {
        (self.rand() + 1.0) / 2.0 // [0, 1]
    }

    fn next_bar(&mut self) -> (Bar, f64, bool) {
        self.t += 1;

        // Inject a volatility spike every 80 bars
        let vol = if self.t % 80 < 5 {
            self.vol * 6.0
        } else {
            self.vol
        };

        // Slow sine trend cycle
        let cycle = (self.t as f64 * 2.0 * PI / 200.0).sin() * 0.0003;
        let noise = self.rand() * vol;
        let ret = self.trend + cycle + noise;
        self.price *= 1.0 + ret;

        let hl_range = self.price * vol * (1.0 + self.rand_pos() * 0.5);
        let open = self.price * (1.0 - self.rand() * vol * 0.3);
        let close = self.price;
        let high = open.max(close) + hl_range * 0.5;
        let low = open.min(close) - hl_range * 0.5;

        // Volume: base 10k with occasional anomalies
        let base_vol = 10_000.0;
        let volume = if self.t % 50 == 0 {
            base_vol * 15.0 // volume anomaly
        } else {
            base_vol * (0.5 + self.rand_pos())
        };

        let is_buy = ret > 0.0;
        let signed_vol = if is_buy { volume } else { -volume };

        let bar = Bar {
            symbol: "SYN".into(),
            bar_type: BarType::Time,
            open_time: Utc::now(),
            close_time: Utc::now(),
            open,
            high,
            low,
            close,
            volume,
            dollar_volume: close * volume,
            vwap: (open + high + low + close) / 4.0,
            num_ticks: 50,
            imbalance: if is_buy { volume } else { -volume },
            tick_imbalance: 0.0,
        };

        (bar, signed_vol, is_buy)
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Real-Time Analytics Stream Demo ===");
    println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>12} {:>14}",
        "Bar#", "Raw", "Kalman1D", "KalVel", "AdapEMA",
        "VPIN", "Toxic", "Regime", "Events");
    println!("{}", "-".repeat(100));

    let mut market = SyntheticMarket::new(42);

    // Filters
    let mut kf1 = KalmanFilter1D::new(100.0, 1e-4, 0.01);
    let mut kf2 = KalmanFilter2D::new(100.0, 0.01, 1.0);
    let mut ema = AdaptiveEMA::new(20);

    // Detectors
    let mut vol_breakout = VolatilityBreakout::new("SYN");
    let mut mom_shift = MomentumShift::new("SYN");
    let mut vol_anomaly = VolumeAnomaly::new("SYN", 50);
    let mut of_reversal = OrderFlowReversal::new("SYN");

    // Microstructure
    let mut toxicity = ToxicityMeter::new(5000.0);

    // Regime
    let mut composite = CompositeRegime::new();

    let num_bars = 300;

    for i in 0..num_bars {
        let (bar, signed_vol, is_buy) = market.next_bar();

        // Filters
        let k1 = kf1.update(bar.close);
        let (k2_price, k2_vel) = kf2.update(bar.close);
        let ema_val = ema.update(bar.close);

        // Detectors
        vol_breakout.update(bar.high, bar.low, bar.close);
        mom_shift.update(bar.close);
        vol_anomaly.update(bar.volume);
        of_reversal.update(signed_vol);

        // Microstructure
        toxicity.update(bar.close, bar.volume, is_buy);

        // Regime
        composite.update(&bar);

        // Collect events
        let mut event_strs: Vec<String> = Vec::new();
        for e in vol_breakout.emitter.drain() {
            event_strs.push(format!("VOL_BRK({:.1}x)", e.magnitude));
        }
        for _e in mom_shift.emitter.drain() {
            event_strs.push("MOM_SHIFT".to_string());
        }
        for _e in vol_anomaly.emitter.drain() {
            event_strs.push("VOL_ANOM".to_string());
        }
        for _e in of_reversal.emitter.drain() {
            event_strs.push("OF_REV".to_string());
        }

        let vpin_str = toxicity.vpin.vpin().map_or("-".to_string(), |v| format!("{:.3}", v));
        let tox_str = toxicity.toxicity().map_or("-".to_string(), |v| format!("{:.3}", v));
        let regime_str = composite.current_regime().to_string();
        let events_str = if event_strs.is_empty() {
            ".".to_string()
        } else {
            event_strs.join(",")
        };

        // Print every 10 bars or when events occur
        if i % 10 == 0 || !event_strs.is_empty() {
            println!(
                "{:<6} {:>10.4} {:>10.4} {:>10.6} {:>10.4} {:>8} {:>8} {:>12} {:>14}",
                i + 1,
                bar.close,
                k1,
                k2_vel,
                ema_val,
                vpin_str,
                tox_str,
                regime_str,
                events_str,
            );
        }
    }

    println!("{}", "-".repeat(100));
    println!("Demo complete. Processed {} bars.", num_bars);
    println!("Final Kalman1D state:  {:.4}", kf1.get_state());
    println!("Final Kalman2D price:  {:.4}", kf2.get_price());
    println!("Final Kalman2D vel:    {:.6}", kf2.get_velocity());
    println!("Final AdaptiveEMA:     {:.4} (alpha={:.3})", ema.get_value(), ema.get_alpha());
    println!("Final regime:          {}", composite.current_regime());
    println!("Final confidence:      {:.3}", composite.confidence());
    if let Some(vpin) = toxicity.vpin.vpin() {
        println!("Final VPIN:            {:.4}", vpin);
    }
    if let Some(tox) = toxicity.toxicity() {
        println!("Final toxicity:        {:.4}", tox);
    }
}
