// types.rs — Core data types for the tick-backtest engine

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Bar
// ---------------------------------------------------------------------------

/// OHLCV bar with Unix-millisecond timestamp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bar {
    /// Unix timestamp in milliseconds.
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    pub fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self { timestamp, open, high, low, close, volume }
    }

    /// True range for this bar given the previous close.
    #[inline]
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }
}

// ---------------------------------------------------------------------------
// Regime
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regime {
    Bull,
    Bear,
    Sideways,
    HighVol,
}

impl Regime {
    pub fn from_str(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "bull" => Self::Bull,
            "bear" => Self::Bear,
            "sideways" => Self::Sideways,
            "highvol" | "high_vol" | "high-vol" => Self::HighVol,
            _ => Self::Sideways,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Self::Bull => "bull",
            Self::Bear => "bear",
            Self::Sideways => "sideways",
            Self::HighVol => "highvol",
        }
    }
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_str())
    }
}

// ---------------------------------------------------------------------------
// TFScore — multi-timeframe alignment score
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TFScore {
    /// Daily timeframe component (0.0 – 1.0).
    pub daily: f64,
    /// Hourly timeframe component (0.0 – 1.0).
    pub hourly: f64,
    /// 15-minute timeframe component (0.0 – 1.0).
    pub m15: f64,
    /// Weighted total (0.0 – 1.0).
    pub total: f64,
}

impl TFScore {
    pub fn new(daily: f64, hourly: f64, m15: f64) -> Self {
        // Weights: daily 50 %, hourly 30 %, 15 m 20 %
        let total = 0.50 * daily + 0.30 * hourly + 0.20 * m15;
        Self { daily, hourly, m15, total }
    }

    pub fn zero() -> Self {
        Self { daily: 0.0, hourly: 0.0, m15: 0.0, total: 0.0 }
    }
}

// ---------------------------------------------------------------------------
// DeltaScore — composite entry signal
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeltaScore {
    pub tf_score: TFScore,
    /// BH mass at signal time.
    pub mass: f64,
    /// ATR at signal time.
    pub atr: f64,
    /// Composite value used for position sizing.
    pub value: f64,
}

impl DeltaScore {
    pub fn new(tf_score: TFScore, mass: f64, atr: f64) -> Self {
        let value = tf_score.total * mass.min(2.0) / 2.0;
        Self { tf_score, mass, atr, value }
    }
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub sym: String,
    /// Positive = long, negative = short (shares / contracts).
    pub size: f64,
    pub avg_cost: f64,
    pub unrealized_pnl: f64,
}

impl Position {
    pub fn new(sym: impl Into<String>, size: f64, avg_cost: f64) -> Self {
        Self { sym: sym.into(), size, avg_cost, unrealized_pnl: 0.0 }
    }

    pub fn is_flat(&self) -> bool {
        self.size.abs() < 1e-10
    }

    pub fn mark_to_market(&mut self, current_price: f64) {
        self.unrealized_pnl = (current_price - self.avg_cost) * self.size;
    }
}

// ---------------------------------------------------------------------------
// Trade
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trade {
    pub sym: String,
    /// Unix-ms entry timestamp.
    pub entry_time: i64,
    /// Unix-ms exit timestamp.
    pub exit_time: i64,
    pub entry_price: f64,
    pub exit_price: f64,
    /// Realised P&L net of transaction costs (in dollar terms).
    pub pnl: f64,
    /// Dollar position size at entry.
    pub dollar_pos: f64,
    /// Number of bars held.
    pub hold_bars: usize,
    /// Market regime at entry.
    pub regime: Regime,
    /// Multi-timeframe signal score at entry.
    pub tf_score: TFScore,
    /// BH mass at entry.
    pub mass: f64,
}

impl Trade {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sym: impl Into<String>,
        entry_time: i64,
        exit_time: i64,
        entry_price: f64,
        exit_price: f64,
        pnl: f64,
        dollar_pos: f64,
        hold_bars: usize,
        regime: Regime,
        tf_score: TFScore,
        mass: f64,
    ) -> Self {
        Self {
            sym: sym.into(),
            entry_time,
            exit_time,
            entry_price,
            exit_price,
            pnl,
            dollar_pos,
            hold_bars,
            regime,
            tf_score,
            mass,
        }
    }

    /// Return as fraction of dollar position (net of costs).
    pub fn return_frac(&self) -> f64 {
        if self.dollar_pos.abs() < 1e-10 {
            0.0
        } else {
            self.pnl / self.dollar_pos
        }
    }

    pub fn is_winner(&self) -> bool {
        self.pnl > 0.0
    }
}

// ---------------------------------------------------------------------------
// RegimeStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct RegimeStats {
    pub count: usize,
    pub winners: usize,
    pub total_pnl: f64,
    pub avg_return: f64,
}

impl RegimeStats {
    pub fn win_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.winners as f64 / self.count as f64
        }
    }

    pub fn add_trade(&mut self, trade: &Trade) {
        self.count += 1;
        if trade.is_winner() {
            self.winners += 1;
        }
        self.total_pnl += trade.pnl;
        // Incremental running mean
        let r = trade.return_frac();
        self.avg_return += (r - self.avg_return) / self.count as f64;
    }
}

// ---------------------------------------------------------------------------
// BacktestMetrics — performance statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BacktestMetrics {
    pub total_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub cagr: f64,
    pub calmar_ratio: f64,
    pub avg_hold_bars: f64,
    pub avg_return_per_trade: f64,
}

impl Default for BacktestMetrics {
    fn default() -> Self {
        Self {
            total_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            sharpe: 0.0,
            max_drawdown: 0.0,
            cagr: 0.0,
            calmar_ratio: 0.0,
            avg_hold_bars: 0.0,
            avg_return_per_trade: 0.0,
        }
    }
}
