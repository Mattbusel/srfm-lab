// intraday_patterns.rs -- Intraday pattern analysis for SRFM tick backtest engine.
// Detects opening drives, intraday reversals, session stats, and hourly profiles.

use crate::types::Bar;

// ---------------------------------------------------------------------------
// HourlyProfile
// ---------------------------------------------------------------------------

/// Aggregated statistics for a single hour-of-day bucket (0..=23).
#[derive(Debug, Clone, PartialEq)]
pub struct HourlyProfile {
    pub hour: u8,
    /// Average volume across bars in this hour.
    pub avg_volume: f64,
    /// Average realised volatility (close-to-close absolute return).
    pub avg_vol: f64,
    /// Average bar return: (close - open) / open.
    pub avg_return: f64,
    /// Average effective spread estimate in basis points.
    pub avg_spread_bps: f64,
}

// ---------------------------------------------------------------------------
// OpeningDrive
// ---------------------------------------------------------------------------

/// Detected opening-drive pattern: strong directional move in the first 30 min
/// followed by a reversal pullback.
#[derive(Debug, Clone, PartialEq)]
pub struct OpeningDrive {
    /// +1 for bullish drive, -1 for bearish drive.
    pub direction: i8,
    /// Magnitude of the initial move expressed in ATR units.
    pub magnitude_atr: f64,
    /// Bar index (not timestamp) at which the reversal was observed.
    pub reversal_time: u32,
}

// ---------------------------------------------------------------------------
// Reversal
// ---------------------------------------------------------------------------

/// A detected intraday reversal: price moved > 2 ATR from the open during the
/// session and then returned to within range of the open by the close.
#[derive(Debug, Clone, PartialEq)]
pub struct Reversal {
    /// +1 if price first spiked up then reversed down, -1 for the opposite.
    pub direction: i8,
    /// Maximum distance from the session open in ATR units.
    pub max_extension_atr: f64,
    /// Bar index where the maximum extension was observed.
    pub peak_bar: u32,
    /// Hour in which the reversal setup was triggered.
    pub trigger_hour: u8,
}

// ---------------------------------------------------------------------------
// SessionStats
// ---------------------------------------------------------------------------

/// Key session-level statistics derived from a set of intraday bars.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionStats {
    /// High - Low of the first 30 minutes (first ~2 bars at 15 min resolution).
    pub open_range: f64,
    /// Distance between the last-bar close and the first-bar open.
    pub close_range: f64,
    /// Bar index at which the session high was set.
    pub high_time: u32,
    /// Bar index at which the session low was set.
    pub low_time: u32,
    /// Bar index at which the range expanded beyond the opening range.
    pub range_expansion_time: u32,
}

// ---------------------------------------------------------------------------
// IntradayPatternAnalyzer
// ---------------------------------------------------------------------------

/// Stateless analyser -- all methods take bar slices and return results.
/// Bars must be sorted ascending by timestamp. Bar timestamps are Unix ms
/// and the hour is derived by (timestamp / 3_600_000) % 24 with the caller
/// expected to pre-adjust for exchange local time if needed.
pub struct IntradayPatternAnalyzer;

impl IntradayPatternAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for IntradayPatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ATR helper
// ---------------------------------------------------------------------------

/// Compute a simple average true range over the slice.
/// Returns 0.0 for slices shorter than 2 bars.
fn compute_atr(bars: &[Bar]) -> f64 {
    if bars.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 1..bars.len() {
        let hl = bars[i].high - bars[i].low;
        let hc = (bars[i].high - bars[i - 1].close).abs();
        let lc = (bars[i].low - bars[i - 1].close).abs();
        sum += hl.max(hc).max(lc);
    }
    sum / (bars.len() - 1) as f64
}

/// Extract the hour-of-day (0..=23) from a Unix-millisecond timestamp.
#[inline]
fn hour_from_ms(ts: i64) -> u8 {
    ((ts / 3_600_000) % 24) as u8
}

// ---------------------------------------------------------------------------
// build_hourly_profile
// ---------------------------------------------------------------------------

/// Build per-hour statistics from a collection of bars.
/// Hours with no bars are omitted from the output vector.
pub fn build_hourly_profile(bars: &[Bar]) -> Vec<HourlyProfile> {
    // Accumulate per-hour buckets
    struct Bucket {
        volume_sum: f64,
        vol_sum: f64,      // sum of |close - prev_close| / prev_close
        return_sum: f64,   // sum of (close - open) / open
        spread_sum: f64,   // sum of spread estimate in bps
        count: usize,
        prev_close: Option<f64>,
    }

    let mut buckets: [Bucket; 24] = std::array::from_fn(|_| Bucket {
        volume_sum: 0.0,
        vol_sum: 0.0,
        return_sum: 0.0,
        spread_sum: 0.0,
        count: 0,
        prev_close: None,
    });

    // Track last close to compute close-to-close vol across hours
    let mut global_prev_close: Option<f64> = None;

    for bar in bars {
        let h = hour_from_ms(bar.timestamp) as usize;
        let b = &mut buckets[h];

        b.volume_sum += bar.volume;

        if let Some(pc) = global_prev_close {
            if pc > 1e-12 {
                b.vol_sum += ((bar.close - pc) / pc).abs();
            }
        }

        if bar.open > 1e-12 {
            b.return_sum += (bar.close - bar.open) / bar.open;
        }

        // Spread estimate: (high - low) / close * 10_000 bps
        if bar.close > 1e-12 {
            b.spread_sum += (bar.high - bar.low) / bar.close * 10_000.0;
        }

        b.count += 1;
        global_prev_close = Some(bar.close);
    }

    let mut profiles = Vec::with_capacity(24);
    for (h, b) in buckets.iter().enumerate() {
        if b.count == 0 {
            continue;
        }
        let n = b.count as f64;
        profiles.push(HourlyProfile {
            hour: h as u8,
            avg_volume: b.volume_sum / n,
            avg_vol: b.vol_sum / n,
            avg_return: b.return_sum / n,
            avg_spread_bps: b.spread_sum / n,
        });
    }
    profiles
}

// ---------------------------------------------------------------------------
// detect_opening_drive
// ---------------------------------------------------------------------------

/// Detect an opening drive: the first ~30 minutes must contain a directional
/// move > 1 ATR from the session open, followed by a pullback.
///
/// "30 minutes" is approximated as the first 2 bars when bars are 15-minute
/// bars, or the first 4 bars for 5-minute bars. This implementation uses the
/// first 2 bars unconditionally to keep it timeframe-agnostic, which is
/// correct for hourly+ resolutions too -- callers should pass the right slice.
pub fn detect_opening_drive(bars: &[Bar]) -> Option<OpeningDrive> {
    if bars.len() < 4 {
        return None;
    }

    let atr = compute_atr(bars);
    if atr < 1e-12 {
        return None;
    }

    // The "drive" window is bars[0..2]
    let drive_open = bars[0].open;
    let drive_high = bars[0].high.max(bars[1].high);
    let drive_low = bars[0].low.min(bars[1].low);
    let drive_close = bars[1].close;

    let up_move = drive_high - drive_open;
    let down_move = drive_open - drive_low;

    let (direction, magnitude_atr): (i8, f64) = if up_move > down_move && up_move > atr {
        (1, up_move / atr)
    } else if down_move > up_move && down_move > atr {
        (-1, down_move / atr)
    } else {
        return None;
    };

    // Look for pullback: after the drive close, price must retrace >= 38.2% of
    // the drive move within the next bars.
    let drive_range = if direction == 1 { up_move } else { down_move };
    let retrace_threshold = drive_range * 0.382;

    for (i, bar) in bars.iter().enumerate().skip(2) {
        let retrace = if direction == 1 {
            drive_close - bar.low
        } else {
            bar.high - drive_close
        };
        if retrace >= retrace_threshold {
            return Some(OpeningDrive {
                direction,
                magnitude_atr,
                reversal_time: i as u32,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// detect_intraday_reversal
// ---------------------------------------------------------------------------

/// Detect a 10am-style exhaustion reversal: price extends > 2 ATR from the
/// session open (bar[0].open) at any point, then returns to within 0.5 ATR of
/// the open by the close of some later bar.
///
/// `hour` constrains which bars can act as the peak: only bars whose timestamp
/// falls in `hour` or `hour + 1` are considered for the extension peak.
pub fn detect_intraday_reversal(bars: &[Bar], hour: u8) -> Option<Reversal> {
    if bars.len() < 3 {
        return None;
    }

    let atr = compute_atr(bars);
    if atr < 1e-12 {
        return None;
    }

    let session_open = bars[0].open;
    let return_threshold = 0.5 * atr;

    // Find the largest extension that occurs in the target hour window
    let mut best_peak: Option<(usize, f64, i8)> = None; // (bar_idx, extension, direction)

    for (i, bar) in bars.iter().enumerate() {
        let h = hour_from_ms(bar.timestamp);
        if h != hour && h != hour.saturating_add(1) {
            continue;
        }
        let up_ext = bar.high - session_open;
        let dn_ext = session_open - bar.low;

        if up_ext > 2.0 * atr {
            if best_peak.map_or(true, |(_, ext, _)| up_ext > ext) {
                best_peak = Some((i, up_ext, 1));
            }
        }
        if dn_ext > 2.0 * atr {
            if best_peak.map_or(true, |(_, ext, _)| dn_ext > ext) {
                best_peak = Some((i, dn_ext, -1));
            }
        }
    }

    let (peak_idx, max_ext, direction) = best_peak?;

    // After the peak, check if price returns within return_threshold of session open
    for bar in bars.iter().skip(peak_idx + 1) {
        let dist = if direction == 1 {
            bar.low - session_open // should become small/negative if reversal
        } else {
            session_open - bar.high
        };
        if dist.abs() <= return_threshold || dist < 0.0 {
            return Some(Reversal {
                direction,
                max_extension_atr: max_ext / atr,
                peak_bar: peak_idx as u32,
                trigger_hour: hour,
            });
        }
    }

    None
}

// ---------------------------------------------------------------------------
// compute_session_stats
// ---------------------------------------------------------------------------

/// Compute key session-level statistics from a sorted bar slice.
pub fn compute_session_stats(bars: &[Bar]) -> SessionStats {
    if bars.is_empty() {
        return SessionStats {
            open_range: 0.0,
            close_range: 0.0,
            high_time: 0,
            low_time: 0,
            range_expansion_time: 0,
        };
    }

    let session_open = bars[0].open;

    // Opening range: high/low of first 2 bars (or just 1 if that's all we have)
    let or_end = bars.len().min(2);
    let or_high = bars[..or_end].iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
    let or_low = bars[..or_end].iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
    let open_range = or_high - or_low;

    let session_close = bars.last().unwrap().close;
    let close_range = (session_close - session_open).abs();

    // Session high/low bar indices
    let mut high_time = 0u32;
    let mut low_time = 0u32;
    let mut session_high = f64::NEG_INFINITY;
    let mut session_low = f64::INFINITY;

    for (i, bar) in bars.iter().enumerate() {
        if bar.high > session_high {
            session_high = bar.high;
            high_time = i as u32;
        }
        if bar.low < session_low {
            session_low = bar.low;
            low_time = i as u32;
        }
    }

    // Range expansion: first bar beyond the opening range
    let mut range_expansion_time = 0u32;
    for (i, bar) in bars.iter().enumerate().skip(or_end) {
        if bar.high > or_high || bar.low < or_low {
            range_expansion_time = i as u32;
            break;
        }
    }

    SessionStats {
        open_range,
        close_range,
        high_time,
        low_time,
        range_expansion_time,
    }
}

// ---------------------------------------------------------------------------
// volume_weighted_direction
// ---------------------------------------------------------------------------

/// Compute a volume-weighted directional signal over a rolling window.
/// Returns sum(vol_i * sign(close_i - open_i)) / sum(vol_i).
/// Returns 0.0 if total volume is zero or window is empty.
pub fn volume_weighted_direction(bars: &[Bar], window: usize) -> f64 {
    if window == 0 || bars.is_empty() {
        return 0.0;
    }
    let start = bars.len().saturating_sub(window);
    let slice = &bars[start..];

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;

    for bar in slice {
        if bar.volume < 1e-12 {
            continue;
        }
        let dir = if bar.close > bar.open {
            1.0_f64
        } else if bar.close < bar.open {
            -1.0_f64
        } else {
            0.0_f64
        };
        num += bar.volume * dir;
        den += bar.volume;
    }

    if den < 1e-12 {
        0.0
    } else {
        num / den
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;

    fn make_bar(ts_ms: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Bar {
        Bar::new(ts_ms, open, high, low, close, volume)
    }

    // Bars in the 9am hour (UTC): ts = 9 * 3_600_000 = 32_400_000
    fn ts_h(hour: u8, minute: u8) -> i64 {
        (hour as i64) * 3_600_000 + (minute as i64) * 60_000
    }

    fn flat_bars(n: usize) -> Vec<Bar> {
        (0..n)
            .map(|i| make_bar(ts_h(9, i as u8), 100.0, 101.0, 99.0, 100.0, 1000.0))
            .collect()
    }

    // ---------------------------------------------------------------------------
    // build_hourly_profile
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hourly_profile_basic() {
        let bars = vec![
            make_bar(ts_h(9, 0), 100.0, 102.0, 99.0, 101.0, 5000.0),
            make_bar(ts_h(9, 15), 101.0, 103.0, 100.0, 102.0, 4000.0),
            make_bar(ts_h(10, 0), 102.0, 104.0, 101.0, 103.0, 3000.0),
        ];
        let profiles = build_hourly_profile(&bars);
        assert_eq!(profiles.len(), 2);
        let h9 = profiles.iter().find(|p| p.hour == 9).unwrap();
        assert!((h9.avg_volume - 4500.0).abs() < 1.0);
        let h10 = profiles.iter().find(|p| p.hour == 10).unwrap();
        assert!((h10.avg_volume - 3000.0).abs() < 1.0);
    }

    #[test]
    fn test_hourly_profile_returns_positive_for_up_bars() {
        let bars = vec![
            make_bar(ts_h(10, 0), 100.0, 105.0, 99.0, 104.0, 1000.0),
            make_bar(ts_h(10, 15), 104.0, 108.0, 103.0, 107.0, 1000.0),
        ];
        let profiles = build_hourly_profile(&bars);
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0].avg_return > 0.0);
    }

    #[test]
    fn test_hourly_profile_empty_slice() {
        let profiles = build_hourly_profile(&[]);
        assert!(profiles.is_empty());
    }

    #[test]
    fn test_hourly_profile_spread_bps_positive() {
        let bars = vec![make_bar(ts_h(9, 0), 100.0, 102.0, 98.0, 100.0, 500.0)];
        let profiles = build_hourly_profile(&bars);
        assert_eq!(profiles.len(), 1);
        // spread = (102-98)/100 * 10000 = 400 bps
        assert!((profiles[0].avg_spread_bps - 400.0).abs() < 0.1);
    }

    // ---------------------------------------------------------------------------
    // detect_opening_drive
    // ---------------------------------------------------------------------------

    #[test]
    fn test_opening_drive_bullish_detected() {
        // Strong up move in first 2 bars, then pullback
        let bars = vec![
            make_bar(ts_h(9, 0), 100.0, 104.0, 99.5, 103.5, 5000.0),  // big up
            make_bar(ts_h(9, 15), 103.5, 105.0, 103.0, 104.5, 4000.0), // continuation
            make_bar(ts_h(9, 30), 104.5, 104.5, 101.0, 101.5, 3000.0), // pullback
            make_bar(ts_h(9, 45), 101.5, 102.0, 100.5, 101.0, 2000.0),
            make_bar(ts_h(10, 0), 101.0, 102.0, 100.0, 101.0, 2000.0),
        ];
        let drive = detect_opening_drive(&bars);
        assert!(drive.is_some());
        let d = drive.unwrap();
        assert_eq!(d.direction, 1);
        assert!(d.magnitude_atr > 1.0);
    }

    #[test]
    fn test_opening_drive_not_detected_for_small_move() {
        // Move < 1 ATR -- should not detect
        let bars: Vec<Bar> = (0..6)
            .map(|i| make_bar(ts_h(9, i * 15), 100.0 + i as f64 * 0.05, 100.1 + i as f64 * 0.05, 99.9 + i as f64 * 0.05, 100.0 + i as f64 * 0.05, 1000.0))
            .collect();
        let drive = detect_opening_drive(&bars);
        assert!(drive.is_none());
    }

    #[test]
    fn test_opening_drive_requires_min_bars() {
        let bars = vec![
            make_bar(ts_h(9, 0), 100.0, 110.0, 99.0, 109.0, 1000.0),
            make_bar(ts_h(9, 15), 109.0, 111.0, 108.0, 110.0, 1000.0),
            make_bar(ts_h(9, 30), 110.0, 110.5, 108.0, 108.5, 1000.0),
        ];
        // Only 3 bars -- less than required minimum of 4
        let drive = detect_opening_drive(&bars);
        assert!(drive.is_none());
    }

    // ---------------------------------------------------------------------------
    // detect_intraday_reversal
    // ---------------------------------------------------------------------------

    #[test]
    fn test_intraday_reversal_detected() {
        let mut bars = Vec::new();
        // Session open at 100
        bars.push(make_bar(ts_h(9, 0), 100.0, 101.0, 99.0, 100.5, 1000.0));
        // Spike up in 10am hour
        bars.push(make_bar(ts_h(10, 0), 100.5, 107.0, 100.0, 106.5, 2000.0));
        bars.push(make_bar(ts_h(10, 15), 106.5, 107.5, 105.0, 106.0, 1500.0));
        // Reversal back toward open
        bars.push(make_bar(ts_h(11, 0), 106.0, 106.0, 99.0, 100.3, 1800.0));
        bars.push(make_bar(ts_h(11, 15), 100.3, 101.0, 99.5, 100.1, 1200.0));
        let rev = detect_intraday_reversal(&bars, 10);
        assert!(rev.is_some());
        let r = rev.unwrap();
        assert_eq!(r.direction, 1);
        assert!(r.max_extension_atr > 2.0);
    }

    #[test]
    fn test_intraday_reversal_wrong_hour_not_detected() {
        let mut bars = Vec::new();
        bars.push(make_bar(ts_h(9, 0), 100.0, 101.0, 99.0, 100.5, 1000.0));
        // Spike at 14:00 (2pm), not at 10am
        bars.push(make_bar(ts_h(14, 0), 100.5, 107.0, 100.0, 106.5, 2000.0));
        bars.push(make_bar(ts_h(14, 30), 106.5, 107.0, 99.5, 100.2, 1500.0));
        // Looking for reversal in hour 10 -- nothing should match
        let rev = detect_intraday_reversal(&bars, 10);
        assert!(rev.is_none());
    }

    // ---------------------------------------------------------------------------
    // compute_session_stats
    // ---------------------------------------------------------------------------

    #[test]
    fn test_session_stats_basic() {
        let bars = vec![
            make_bar(ts_h(9, 0), 100.0, 102.0, 99.0, 101.0, 1000.0),
            make_bar(ts_h(9, 15), 101.0, 103.0, 100.0, 102.0, 1000.0),
            make_bar(ts_h(9, 30), 102.0, 108.0, 101.5, 107.0, 2000.0),
            make_bar(ts_h(9, 45), 107.0, 107.5, 105.0, 106.0, 1500.0),
        ];
        let stats = compute_session_stats(&bars);
        assert!(stats.open_range > 0.0);
        // Session high is 108, at bar index 2
        assert_eq!(stats.high_time, 2);
        // Range expansion happens at bar 2 (beyond 99..103 opening range)
        assert_eq!(stats.range_expansion_time, 2);
    }

    #[test]
    fn test_session_stats_empty() {
        let stats = compute_session_stats(&[]);
        assert_eq!(stats.open_range, 0.0);
        assert_eq!(stats.high_time, 0);
    }

    // ---------------------------------------------------------------------------
    // volume_weighted_direction
    // ---------------------------------------------------------------------------

    #[test]
    fn test_vwd_all_up_bars() {
        let bars = vec![
            make_bar(0, 100.0, 101.0, 99.5, 101.0, 1000.0),
            make_bar(1, 101.0, 102.0, 100.5, 102.0, 2000.0),
            make_bar(2, 102.0, 103.0, 101.5, 103.0, 1500.0),
        ];
        let vwd = volume_weighted_direction(&bars, 3);
        assert!((vwd - 1.0).abs() < 1e-9, "All up bars should yield vwd = 1.0, got {vwd}");
    }

    #[test]
    fn test_vwd_all_down_bars() {
        let bars = vec![
            make_bar(0, 103.0, 103.5, 101.0, 101.0, 1000.0),
            make_bar(1, 101.0, 101.5, 99.0, 99.0, 1000.0),
        ];
        let vwd = volume_weighted_direction(&bars, 2);
        assert!((vwd + 1.0).abs() < 1e-9, "All down bars should yield vwd = -1.0, got {vwd}");
    }

    #[test]
    fn test_vwd_mixed_bars() {
        // 1000 vol up, 1000 vol down -> net 0
        let bars = vec![
            make_bar(0, 100.0, 101.0, 99.0, 101.0, 1000.0),
            make_bar(1, 101.0, 102.0, 99.5, 99.5, 1000.0),
        ];
        let vwd = volume_weighted_direction(&bars, 2);
        assert!(vwd.abs() < 1e-9, "Equal up/down volume should yield vwd = 0.0, got {vwd}");
    }

    #[test]
    fn test_vwd_window_limits_lookback() {
        // 5 bars, window=2 -- only last 2 bars should count
        let bars = vec![
            make_bar(0, 100.0, 101.0, 99.0, 99.0, 9999.0), // down, huge vol -- should be excluded
            make_bar(1, 99.0, 100.0, 98.0, 98.0, 9999.0),  // down, huge vol -- should be excluded
            make_bar(2, 98.0, 99.0, 97.0, 97.0, 9999.0),   // down, huge vol -- should be excluded
            make_bar(3, 97.0, 99.0, 96.5, 99.0, 500.0),    // up
            make_bar(4, 99.0, 100.0, 98.5, 100.0, 500.0),  // up
        ];
        let vwd = volume_weighted_direction(&bars, 2);
        assert!((vwd - 1.0).abs() < 1e-9, "Window=2 should see only up bars, got {vwd}");
    }

    #[test]
    fn test_vwd_empty_returns_zero() {
        let vwd = volume_weighted_direction(&[], 5);
        assert_eq!(vwd, 0.0);
    }

    #[test]
    fn test_flat_bars_vwd_zero() {
        // Bars where close == open -> direction = 0
        let bars = flat_bars(4);
        let vwd = volume_weighted_direction(&bars, 4);
        assert_eq!(vwd, 0.0);
    }
}
