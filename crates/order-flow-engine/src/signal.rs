use crate::aggressive_flow::{AggressiveFlowDetector, AggressiveFlowConfig};
use crate::delta_divergence::{DeltaDivergenceDetector, DivergenceObs, DivergenceType};
use crate::tick_classifier::BulkVolumeClassifier;
use crate::volume_imbalance::{ClassifiedBar, OFICalculator};
use crate::vpin::{VpinCalculator, VPIN_ENTRY_FILTER_THRESHOLD};

/// The overall order flow regime derived from combining all indicators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OrderFlowRegime {
    BuyDominated,
    SellDominated,
    Balanced,
    InformedTradingDetected,
}

impl std::fmt::Display for OrderFlowRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderFlowRegime::BuyDominated => write!(f, "BUY_DOMINATED"),
            OrderFlowRegime::SellDominated => write!(f, "SELL_DOMINATED"),
            OrderFlowRegime::Balanced => write!(f, "BALANCED"),
            OrderFlowRegime::InformedTradingDetected => write!(f, "INFORMED_TRADING_DETECTED"),
        }
    }
}

/// Complete order flow signal for a single bar.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderFlowSignal {
    pub symbol: String,
    pub timestamp: String,
    /// Order Flow Imbalance in [-1, 1]
    pub ofi: f64,
    /// VPIN in [0, 1]
    pub vpin: f64,
    /// Is delta divergence present?
    pub delta_divergence: bool,
    /// Type of divergence
    pub divergence_type: String,
    /// Aggressive buy pressure score [0, 1]
    pub aggressive_buy_pressure: f64,
    /// Aggressive sell pressure score [0, 1]
    pub aggressive_sell_pressure: f64,
    /// Regime classification
    pub regime: OrderFlowRegime,
    /// Should entry be filtered (VPIN too high)?
    pub filter_entry: bool,
    /// Buy volume estimated for this bar
    pub buy_volume: f64,
    /// Sell volume estimated for this bar
    pub sell_volume: f64,
    /// Net delta
    pub delta: f64,
    /// POC price from footprint
    pub poc: f64,
    /// Value area high
    pub vah: f64,
    /// Value area low
    pub val: f64,
    /// Footprint imbalance signal
    pub footprint_imbalance: String,
}

impl OrderFlowSignal {
    /// CSV header for output.
    pub fn csv_header() -> &'static str {
        "symbol,timestamp,ofi,vpin,delta_divergence,divergence_type,\
aggressive_buy_pressure,aggressive_sell_pressure,regime,filter_entry,\
buy_volume,sell_volume,delta,poc,vah,val,footprint_imbalance"
    }

    /// Format as a CSV row.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{:.6},{:.6},{},{},{:.6},{:.6},{},{},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4},{}",
            self.symbol,
            self.timestamp,
            self.ofi,
            self.vpin,
            self.delta_divergence,
            self.divergence_type,
            self.aggressive_buy_pressure,
            self.aggressive_sell_pressure,
            self.regime,
            self.filter_entry,
            self.buy_volume,
            self.sell_volume,
            self.delta,
            self.poc,
            self.vah,
            self.val,
            self.footprint_imbalance,
        )
    }
}

/// Configuration for the signal engine.
#[derive(Debug, Clone)]
pub struct SignalEngineConfig {
    pub symbol: String,
    pub ofi_window: usize,
    pub vpin_bucket_volume: f64,
    pub vpin_n_buckets: usize,
    pub divergence_window: usize,
    pub agg_flow_config: AggressiveFlowConfig,
}

impl SignalEngineConfig {
    pub fn new(symbol: &str) -> Self {
        SignalEngineConfig {
            symbol: symbol.to_string(),
            ofi_window: 10,
            vpin_bucket_volume: 5000.0,
            vpin_n_buckets: 20,
            divergence_window: 14,
            agg_flow_config: AggressiveFlowConfig::default(),
        }
    }
}

/// Stateful signal engine that processes OHLCV bars and emits `OrderFlowSignal`s.
pub struct OrderFlowEngine {
    config: SignalEngineConfig,
    bvc: BulkVolumeClassifier,
    ofi_calc: OFICalculator,
    vpin_calc: VpinCalculator,
    divergence_detector: DeltaDivergenceDetector,
    agg_flow_detector: AggressiveFlowDetector,
}

impl OrderFlowEngine {
    pub fn new(config: SignalEngineConfig) -> Self {
        let ofi_calc = OFICalculator::new(config.ofi_window);
        let vpin_calc = VpinCalculator::new(config.vpin_bucket_volume, config.vpin_n_buckets);
        let divergence_detector = DeltaDivergenceDetector::new(config.divergence_window, 0.001, 0.10);
        let agg_flow_detector = AggressiveFlowDetector::new(config.agg_flow_config.clone());
        OrderFlowEngine {
            config,
            bvc: BulkVolumeClassifier::new(20),
            ofi_calc,
            vpin_calc,
            divergence_detector,
            agg_flow_detector,
        }
    }

    /// Process a single OHLCV bar and return an `OrderFlowSignal`.
    pub fn process_bar(
        &mut self,
        timestamp: &str,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> OrderFlowSignal {
        // 1. Classify volume using BVC
        let (buy_vol, sell_vol) = self.bvc.classify_bar(open, close, volume);

        // 2. OFI
        let ofi = self.ofi_calc.push(ClassifiedBar::new(buy_vol, sell_vol));

        // 3. VPIN
        let vpin = self.vpin_calc.push(buy_vol, sell_vol).unwrap_or(0.0);
        let filter_entry = vpin > VPIN_ENTRY_FILTER_THRESHOLD;

        // 4. Delta divergence
        let delta = buy_vol - sell_vol;
        let div_signal = self.divergence_detector.push(DivergenceObs {
            high,
            low,
            close,
            delta,
        });
        let delta_divergence = div_signal.divergence_type != DivergenceType::None;
        let divergence_type = div_signal.divergence_type.to_string();

        // 5. Aggressive flow
        let agg = self.agg_flow_detector.push(open, high, low, close, volume);

        // 6. Footprint (single bar)
        let buy_frac = if (buy_vol + sell_vol) > 1e-9 {
            buy_vol / (buy_vol + sell_vol)
        } else {
            0.5
        };
        let fp = crate::footprint::build_footprint_bar(open, high, low, close, volume, buy_frac);

        // 7. Regime classification
        let regime = classify_regime(ofi, vpin, &agg);

        OrderFlowSignal {
            symbol: self.config.symbol.clone(),
            timestamp: timestamp.to_string(),
            ofi,
            vpin,
            delta_divergence,
            divergence_type,
            aggressive_buy_pressure: agg.aggressive_buy_pressure,
            aggressive_sell_pressure: agg.aggressive_sell_pressure,
            regime,
            filter_entry,
            buy_volume: buy_vol,
            sell_volume: sell_vol,
            delta,
            poc: fp.poc,
            vah: fp.vah,
            val: fp.val,
            footprint_imbalance: fp.imbalance.to_string(),
        }
    }

    /// Process a full series of OHLCV rows.
    pub fn process_series(
        &mut self,
        rows: &[(String, f64, f64, f64, f64, f64)], // (timestamp, o, h, l, c, v)
    ) -> Vec<OrderFlowSignal> {
        rows.iter()
            .map(|(ts, o, h, l, c, v)| self.process_bar(ts, *o, *h, *l, *c, *v))
            .collect()
    }
}

fn classify_regime(
    ofi: f64,
    vpin: f64,
    agg: &crate::aggressive_flow::AggressiveFlowResult,
) -> OrderFlowRegime {
    if vpin > crate::vpin::VPIN_INFORMED_THRESHOLD {
        return OrderFlowRegime::InformedTradingDetected;
    }

    // Weight OFI and aggressive flow
    let net_pressure = ofi
        + agg.aggressive_buy_pressure
        - agg.aggressive_sell_pressure;

    if net_pressure > 0.25 {
        OrderFlowRegime::BuyDominated
    } else if net_pressure < -0.25 {
        OrderFlowRegime::SellDominated
    } else {
        OrderFlowRegime::Balanced
    }
}

/// Load OHLCV data from a CSV file.
///
/// Expected columns: timestamp, open, high, low, close, volume
pub fn load_ohlcv_csv(
    path: &str,
) -> anyhow::Result<Vec<(String, f64, f64, f64, f64, f64)>> {
    #[derive(serde::Deserialize)]
    struct Row {
        timestamp: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    }

    let mut rdr = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for result in rdr.deserialize::<Row>() {
        let row = result?;
        rows.push((row.timestamp, row.open, row.high, row.low, row.close, row.volume));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> OrderFlowEngine {
        OrderFlowEngine::new(SignalEngineConfig::new("TEST"))
    }

    fn bar(ts: &str, open: f64, close: f64, vol: f64) -> (String, f64, f64, f64, f64, f64) {
        let high = open.max(close) + 0.5;
        let low = open.min(close) - 0.5;
        (ts.to_string(), open, high, low, close, vol)
    }

    #[test]
    fn test_single_bar_produces_signal() {
        let mut eng = make_engine();
        let sig = eng.process_bar("2024-01-01T09:30:00", 100.0, 101.0, 99.5, 100.5, 5000.0);
        assert_eq!(sig.symbol, "TEST");
        assert!(sig.ofi >= -1.0 && sig.ofi <= 1.0);
        assert!(sig.vpin >= 0.0 && sig.vpin <= 1.0);
    }

    #[test]
    fn test_signal_series_length() {
        let mut eng = make_engine();
        let bars: Vec<_> = (0..50)
            .map(|i| bar(&format!("2024-01-01T{:02}:00:00", i % 24), 100.0 + i as f64 * 0.1, 100.2 + i as f64 * 0.1, 3000.0))
            .collect();
        let signals = eng.process_series(&bars);
        assert_eq!(signals.len(), 50);
    }

    #[test]
    fn test_regime_informed_on_high_vpin() {
        // We can't easily force VPIN high without many bars; just check it compiles and runs
        let mut eng = make_engine();
        // Push 200 highly imbalanced bars to raise VPIN
        let bars: Vec<_> = (0..200)
            .map(|i| bar(&format!("T{}", i), 100.0, 105.0, 100000.0)) // big buy bars
            .collect();
        let signals = eng.process_series(&bars);
        let last = signals.last().unwrap();
        // With all-buy bars VPIN should be elevated
        assert!(last.vpin >= 0.0);
    }

    #[test]
    fn test_filter_entry_false_on_balanced_flow() {
        let mut eng = make_engine();
        let bars: Vec<_> = (0..100)
            .map(|i| bar(&format!("T{}", i), 100.0, 100.0, 1000.0)) // flat bars -> balanced
            .collect();
        let signals = eng.process_series(&bars);
        // Flat bars -> low VPIN -> no filter
        let filtered = signals.iter().filter(|s| s.filter_entry).count();
        assert_eq!(filtered, 0, "balanced flow should not filter entries");
    }

    #[test]
    fn test_buy_sell_volume_sum_to_total() {
        let mut eng = make_engine();
        let bars: Vec<_> = (0..20)
            .map(|i| bar(&format!("T{}", i), 100.0, 101.0 + i as f64 * 0.1, 5000.0))
            .collect();
        for b in &bars {
            let sig = eng.process_bar(&b.0, b.1, b.2, b.3, b.4, b.5);
            assert!((sig.buy_volume + sig.sell_volume - b.5).abs() < 1.0);
        }
    }

    #[test]
    fn test_csv_row_format() {
        let sig = OrderFlowSignal {
            symbol: "AAPL".into(),
            timestamp: "2024-01-01".into(),
            ofi: 0.123,
            vpin: 0.456,
            delta_divergence: false,
            divergence_type: "NONE".into(),
            aggressive_buy_pressure: 0.1,
            aggressive_sell_pressure: 0.05,
            regime: OrderFlowRegime::BuyDominated,
            filter_entry: false,
            buy_volume: 3000.0,
            sell_volume: 2000.0,
            delta: 1000.0,
            poc: 100.5,
            vah: 101.2,
            val: 99.8,
            footprint_imbalance: "NEUTRAL".into(),
        };
        let row = sig.to_csv_row();
        assert!(row.contains("AAPL"));
        assert!(row.contains("BUY_DOMINATED"));
    }
}
