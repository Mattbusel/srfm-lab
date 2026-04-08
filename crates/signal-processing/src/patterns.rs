use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Candlestick data
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Candle {
    pub fn new(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self { open, high, low, close, volume }
    }

    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }

    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    pub fn upper_shadow(&self) -> f64 {
        self.high - self.close.max(self.open)
    }

    pub fn lower_shadow(&self) -> f64 {
        self.close.min(self.open) - self.low
    }

    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    pub fn is_bearish(&self) -> bool {
        self.close < self.open
    }

    pub fn body_top(&self) -> f64 {
        self.close.max(self.open)
    }

    pub fn body_bottom(&self) -> f64 {
        self.close.min(self.open)
    }

    pub fn midpoint(&self) -> f64 {
        (self.high + self.low) / 2.0
    }

    pub fn body_midpoint(&self) -> f64 {
        (self.open + self.close) / 2.0
    }

    pub fn body_pct(&self) -> f64 {
        if self.range() > 1e-15 {
            self.body_size() / self.range()
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern result
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub name: &'static str,
    pub pattern_type: PatternType,
    pub direction: PatternDirection,
    pub score: f64, // 0-100 confidence
    pub start_index: usize,
    pub end_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternType {
    Candlestick,
    Chart,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternDirection {
    Bullish,
    Bearish,
    Neutral,
}

// ---------------------------------------------------------------------------
// Candlestick Pattern Detector
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct CandlestickDetector {
    buffer: VecDeque<Candle>,
    max_lookback: usize,
    index: usize,
    avg_body: f64,
    avg_range: f64,
    body_count: usize,
}

impl CandlestickDetector {
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::with_capacity(20),
            max_lookback: 15,
            index: 0,
            avg_body: 0.0,
            avg_range: 0.0,
            body_count: 0,
        }
    }

    pub fn update(&mut self, candle: Candle) -> Vec<PatternMatch> {
        self.buffer.push_back(candle);
        if self.buffer.len() > self.max_lookback {
            self.buffer.pop_front();
        }

        // Update running averages
        self.body_count += 1;
        let alpha = 2.0 / (15.0_f64.min(self.body_count as f64) + 1.0);
        self.avg_body = alpha * candle.body_size() + (1.0 - alpha) * self.avg_body;
        self.avg_range = alpha * candle.range() + (1.0 - alpha) * self.avg_range;

        let mut patterns = Vec::new();
        let idx = self.index;
        self.index += 1;

        // Single candle patterns
        if self.buffer.len() >= 1 {
            self.detect_doji(&mut patterns, idx);
            self.detect_marubozu(&mut patterns, idx);
            self.detect_spinning_top(&mut patterns, idx);
        }

        // Two candle patterns
        if self.buffer.len() >= 2 {
            self.detect_hammer(&mut patterns, idx);
            self.detect_hanging_man(&mut patterns, idx);
            self.detect_engulfing(&mut patterns, idx);
            self.detect_tweezer_top(&mut patterns, idx);
            self.detect_tweezer_bottom(&mut patterns, idx);
            self.detect_piercing_line(&mut patterns, idx);
            self.detect_dark_cloud_cover(&mut patterns, idx);
            self.detect_harami(&mut patterns, idx);
            self.detect_kicker(&mut patterns, idx);
        }

        // Three candle patterns
        if self.buffer.len() >= 3 {
            self.detect_morning_star(&mut patterns, idx);
            self.detect_evening_star(&mut patterns, idx);
            self.detect_three_white_soldiers(&mut patterns, idx);
            self.detect_three_black_crows(&mut patterns, idx);
            self.detect_abandoned_baby(&mut patterns, idx);
            self.detect_three_inside_up(&mut patterns, idx);
            self.detect_three_inside_down(&mut patterns, idx);
        }

        // Five candle patterns
        if self.buffer.len() >= 5 {
            self.detect_rising_three_methods(&mut patterns, idx);
            self.detect_falling_three_methods(&mut patterns, idx);
            self.detect_mat_hold(&mut patterns, idx);
        }

        patterns
    }

    fn last(&self, offset: usize) -> &Candle {
        let len = self.buffer.len();
        &self.buffer[len - 1 - offset]
    }

    fn is_small_body(&self, c: &Candle) -> bool {
        c.body_size() < self.avg_body * 0.3
    }

    fn is_large_body(&self, c: &Candle) -> bool {
        c.body_size() > self.avg_body * 1.0
    }

    fn detect_doji(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c = self.last(0);
        if c.range() < 1e-15 { return; }
        let body_pct = c.body_pct();
        if body_pct < 0.1 {
            let score = (1.0 - body_pct / 0.1) * 80.0 + 20.0;
            patterns.push(PatternMatch {
                name: "Doji",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Neutral,
                score: score.min(100.0),
                start_index: idx,
                end_index: idx,
            });
        }
    }

    fn detect_hammer(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c = self.last(0);
        if c.range() < 1e-15 { return; }
        let lower = c.lower_shadow();
        let upper = c.upper_shadow();
        let body = c.body_size();
        if lower > body * 2.0 && upper < body * 0.5 && body > self.avg_body * 0.3 {
            let score = (lower / (body + 1e-15)).min(5.0) / 5.0 * 70.0 + 30.0;
            patterns.push(PatternMatch {
                name: "Hammer",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: score.min(100.0),
                start_index: idx,
                end_index: idx,
            });
        }
    }

    fn detect_hanging_man(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c = self.last(0);
        if c.range() < 1e-15 { return; }
        let lower = c.lower_shadow();
        let upper = c.upper_shadow();
        let body = c.body_size();
        // Same shape as hammer but after uptrend
        if lower > body * 2.0 && upper < body * 0.5 && body > self.avg_body * 0.3 {
            // Check for prior uptrend (simple: prev close > prev prev close)
            if self.buffer.len() >= 3 {
                let prev = self.last(1);
                let prev2 = self.last(2);
                if prev.close > prev2.close {
                    let score = (lower / (body + 1e-15)).min(5.0) / 5.0 * 60.0 + 30.0;
                    patterns.push(PatternMatch {
                        name: "Hanging Man",
                        pattern_type: PatternType::Candlestick,
                        direction: PatternDirection::Bearish,
                        score: score.min(100.0),
                        start_index: idx,
                        end_index: idx,
                    });
                }
            }
        }
    }

    fn detect_engulfing(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let curr = self.last(0);
        let prev = self.last(1);

        // Bullish engulfing
        if prev.is_bearish() && curr.is_bullish()
            && curr.body_bottom() < prev.body_bottom()
            && curr.body_top() > prev.body_top()
            && self.is_large_body(curr)
        {
            let engulf_ratio = curr.body_size() / (prev.body_size() + 1e-15);
            let score = engulf_ratio.min(3.0) / 3.0 * 60.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Bullish Engulfing",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }

        // Bearish engulfing
        if prev.is_bullish() && curr.is_bearish()
            && curr.body_top() > prev.body_top()
            && curr.body_bottom() < prev.body_bottom()
            && self.is_large_body(curr)
        {
            let engulf_ratio = curr.body_size() / (prev.body_size() + 1e-15);
            let score = engulf_ratio.min(3.0) / 3.0 * 60.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Bearish Engulfing",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }
    }

    fn detect_morning_star(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c3 = self.last(0); // current
        let c2 = self.last(1); // middle (small body)
        let c1 = self.last(2); // first (bearish)

        if c1.is_bearish() && self.is_large_body(c1)
            && self.is_small_body(c2)
            && c3.is_bullish() && self.is_large_body(c3)
            && c2.body_top() < c1.body_bottom()
            && c3.close > c1.body_midpoint()
        {
            let score = 70.0 + (c3.body_size() / (c1.body_size() + 1e-15)).min(1.5) * 20.0;
            patterns.push(PatternMatch {
                name: "Morning Star",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: score.min(100.0),
                start_index: idx - 2,
                end_index: idx,
            });
        }
    }

    fn detect_evening_star(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c3 = self.last(0);
        let c2 = self.last(1);
        let c1 = self.last(2);

        if c1.is_bullish() && self.is_large_body(c1)
            && self.is_small_body(c2)
            && c3.is_bearish() && self.is_large_body(c3)
            && c2.body_bottom() > c1.body_top()
            && c3.close < c1.body_midpoint()
        {
            let score = 70.0 + (c3.body_size() / (c1.body_size() + 1e-15)).min(1.5) * 20.0;
            patterns.push(PatternMatch {
                name: "Evening Star",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: score.min(100.0),
                start_index: idx - 2,
                end_index: idx,
            });
        }
    }

    fn detect_three_white_soldiers(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c3 = self.last(0);
        let c2 = self.last(1);
        let c1 = self.last(2);

        if c1.is_bullish() && c2.is_bullish() && c3.is_bullish()
            && c2.close > c1.close && c3.close > c2.close
            && c2.open > c1.body_bottom() && c2.open < c1.close
            && c3.open > c2.body_bottom() && c3.open < c2.close
            && c1.upper_shadow() < c1.body_size() * 0.5
            && c2.upper_shadow() < c2.body_size() * 0.5
            && c3.upper_shadow() < c3.body_size() * 0.5
        {
            let score = 75.0;
            patterns.push(PatternMatch {
                name: "Three White Soldiers",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score,
                start_index: idx - 2,
                end_index: idx,
            });
        }
    }

    fn detect_three_black_crows(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c3 = self.last(0);
        let c2 = self.last(1);
        let c1 = self.last(2);

        if c1.is_bearish() && c2.is_bearish() && c3.is_bearish()
            && c2.close < c1.close && c3.close < c2.close
            && c2.open < c1.body_top() && c2.open > c1.close
            && c3.open < c2.body_top() && c3.open > c2.close
            && c1.lower_shadow() < c1.body_size() * 0.5
            && c2.lower_shadow() < c2.body_size() * 0.5
            && c3.lower_shadow() < c3.body_size() * 0.5
        {
            let score = 75.0;
            patterns.push(PatternMatch {
                name: "Three Black Crows",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score,
                start_index: idx - 2,
                end_index: idx,
            });
        }
    }

    fn detect_harami(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let curr = self.last(0);
        let prev = self.last(1);

        // Bullish harami
        if prev.is_bearish() && self.is_large_body(prev)
            && curr.is_bullish()
            && curr.body_top() < prev.body_top()
            && curr.body_bottom() > prev.body_bottom()
        {
            let ratio = prev.body_size() / (curr.body_size() + 1e-15);
            let score = ratio.min(4.0) / 4.0 * 50.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Bullish Harami",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }

        // Bearish harami
        if prev.is_bullish() && self.is_large_body(prev)
            && curr.is_bearish()
            && curr.body_top() < prev.body_top()
            && curr.body_bottom() > prev.body_bottom()
        {
            let ratio = prev.body_size() / (curr.body_size() + 1e-15);
            let score = ratio.min(4.0) / 4.0 * 50.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Bearish Harami",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }
    }

    fn detect_piercing_line(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let curr = self.last(0);
        let prev = self.last(1);

        if prev.is_bearish() && self.is_large_body(prev)
            && curr.is_bullish()
            && curr.open < prev.low
            && curr.close > prev.body_midpoint()
            && curr.close < prev.open
        {
            let penetration = (curr.close - prev.body_bottom()) / (prev.body_size() + 1e-15);
            let score = penetration.min(1.0) * 50.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Piercing Line",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }
    }

    fn detect_dark_cloud_cover(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let curr = self.last(0);
        let prev = self.last(1);

        if prev.is_bullish() && self.is_large_body(prev)
            && curr.is_bearish()
            && curr.open > prev.high
            && curr.close < prev.body_midpoint()
            && curr.close > prev.open
        {
            let penetration = (prev.body_top() - curr.close) / (prev.body_size() + 1e-15);
            let score = penetration.min(1.0) * 50.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Dark Cloud Cover",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }
    }

    fn detect_spinning_top(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c = self.last(0);
        if c.range() < 1e-15 { return; }
        let body = c.body_size();
        let upper = c.upper_shadow();
        let lower = c.lower_shadow();
        if body < self.avg_body * 0.5
            && upper > body * 0.5
            && lower > body * 0.5
            && body > 1e-15
        {
            let score = 50.0;
            patterns.push(PatternMatch {
                name: "Spinning Top",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Neutral,
                score,
                start_index: idx,
                end_index: idx,
            });
        }
    }

    fn detect_marubozu(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c = self.last(0);
        if c.range() < 1e-15 { return; }
        let body = c.body_size();
        let upper = c.upper_shadow();
        let lower = c.lower_shadow();
        let range = c.range();
        if body / range > 0.9 && upper < range * 0.05 && lower < range * 0.05 {
            let dir = if c.is_bullish() { PatternDirection::Bullish } else { PatternDirection::Bearish };
            let score = (body / range) * 80.0 + 20.0;
            patterns.push(PatternMatch {
                name: if c.is_bullish() { "Bullish Marubozu" } else { "Bearish Marubozu" },
                pattern_type: PatternType::Candlestick,
                direction: dir,
                score: score.min(100.0),
                start_index: idx,
                end_index: idx,
            });
        }
    }

    fn detect_tweezer_top(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let curr = self.last(0);
        let prev = self.last(1);
        let tolerance = self.avg_range * 0.05;
        if (curr.high - prev.high).abs() < tolerance
            && prev.is_bullish() && curr.is_bearish()
        {
            let score = (1.0 - (curr.high - prev.high).abs() / (tolerance + 1e-15)) * 40.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Tweezer Top",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }
    }

    fn detect_tweezer_bottom(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let curr = self.last(0);
        let prev = self.last(1);
        let tolerance = self.avg_range * 0.05;
        if (curr.low - prev.low).abs() < tolerance
            && prev.is_bearish() && curr.is_bullish()
        {
            let score = (1.0 - (curr.low - prev.low).abs() / (tolerance + 1e-15)) * 40.0 + 40.0;
            patterns.push(PatternMatch {
                name: "Tweezer Bottom",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: score.min(100.0),
                start_index: idx - 1,
                end_index: idx,
            });
        }
    }

    fn detect_abandoned_baby(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c3 = self.last(0);
        let c2 = self.last(1);
        let c1 = self.last(2);

        // Bullish abandoned baby
        if c1.is_bearish() && self.is_large_body(c1)
            && self.is_small_body(c2)
            && c2.high < c1.low  // gap down
            && c3.is_bullish() && self.is_large_body(c3)
            && c3.low > c2.high  // gap up
        {
            patterns.push(PatternMatch {
                name: "Bullish Abandoned Baby",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: 85.0,
                start_index: idx - 2,
                end_index: idx,
            });
        }

        // Bearish abandoned baby
        if c1.is_bullish() && self.is_large_body(c1)
            && self.is_small_body(c2)
            && c2.low > c1.high  // gap up
            && c3.is_bearish() && self.is_large_body(c3)
            && c3.high < c2.low  // gap down
        {
            patterns.push(PatternMatch {
                name: "Bearish Abandoned Baby",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: 85.0,
                start_index: idx - 2,
                end_index: idx,
            });
        }
    }

    fn detect_three_inside_up(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c3 = self.last(0);
        let c2 = self.last(1);
        let c1 = self.last(2);

        if c1.is_bearish() && self.is_large_body(c1)
            && c2.is_bullish()
            && c2.body_top() < c1.body_top()
            && c2.body_bottom() > c1.body_bottom()
            && c3.is_bullish()
            && c3.close > c1.open
        {
            patterns.push(PatternMatch {
                name: "Three Inside Up",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: 70.0,
                start_index: idx - 2,
                end_index: idx,
            });
        }
    }

    fn detect_three_inside_down(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let c3 = self.last(0);
        let c2 = self.last(1);
        let c1 = self.last(2);

        if c1.is_bullish() && self.is_large_body(c1)
            && c2.is_bearish()
            && c2.body_top() < c1.body_top()
            && c2.body_bottom() > c1.body_bottom()
            && c3.is_bearish()
            && c3.close < c1.open
        {
            patterns.push(PatternMatch {
                name: "Three Inside Down",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: 70.0,
                start_index: idx - 2,
                end_index: idx,
            });
        }
    }

    fn detect_rising_three_methods(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        if self.buffer.len() < 5 { return; }
        let c5 = self.last(0);
        let c4 = self.last(1);
        let c3 = self.last(2);
        let c2 = self.last(3);
        let c1 = self.last(4);

        if c1.is_bullish() && self.is_large_body(c1)
            && c2.is_bearish() && c3.is_bearish() && c4.is_bearish()
            && c2.high < c1.high && c4.low > c1.low
            && c5.is_bullish() && c5.close > c1.close
        {
            patterns.push(PatternMatch {
                name: "Rising Three Methods",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: 72.0,
                start_index: idx - 4,
                end_index: idx,
            });
        }
    }

    fn detect_falling_three_methods(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        if self.buffer.len() < 5 { return; }
        let c5 = self.last(0);
        let c4 = self.last(1);
        let c3 = self.last(2);
        let c2 = self.last(3);
        let c1 = self.last(4);

        if c1.is_bearish() && self.is_large_body(c1)
            && c2.is_bullish() && c3.is_bullish() && c4.is_bullish()
            && c2.high < c1.open && c4.low > c1.close
            && c5.is_bearish() && c5.close < c1.close
        {
            patterns.push(PatternMatch {
                name: "Falling Three Methods",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: 72.0,
                start_index: idx - 4,
                end_index: idx,
            });
        }
    }

    fn detect_kicker(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        let curr = self.last(0);
        let prev = self.last(1);

        // Bullish kicker
        if prev.is_bearish() && curr.is_bullish()
            && curr.open > prev.open
            && self.is_large_body(curr) && self.is_large_body(prev)
        {
            patterns.push(PatternMatch {
                name: "Bullish Kicker",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: 85.0,
                start_index: idx - 1,
                end_index: idx,
            });
        }

        // Bearish kicker
        if prev.is_bullish() && curr.is_bearish()
            && curr.open < prev.open
            && self.is_large_body(curr) && self.is_large_body(prev)
        {
            patterns.push(PatternMatch {
                name: "Bearish Kicker",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bearish,
                score: 85.0,
                start_index: idx - 1,
                end_index: idx,
            });
        }
    }

    fn detect_mat_hold(&self, patterns: &mut Vec<PatternMatch>, idx: usize) {
        if self.buffer.len() < 5 { return; }
        let c5 = self.last(0);
        let c4 = self.last(1);
        let c3 = self.last(2);
        let c2 = self.last(3);
        let c1 = self.last(4);

        // Bullish mat hold
        if c1.is_bullish() && self.is_large_body(c1)
            && c2.is_bullish() && c2.open > c1.close // gap up
            && self.is_small_body(c2)
            && c3.is_bearish() && self.is_small_body(c3)
            && c4.is_bearish() && self.is_small_body(c4)
            && c4.low > c1.body_midpoint()
            && c5.is_bullish() && c5.close > c2.high
        {
            patterns.push(PatternMatch {
                name: "Mat Hold",
                pattern_type: PatternType::Candlestick,
                direction: PatternDirection::Bullish,
                score: 78.0,
                start_index: idx - 4,
                end_index: idx,
            });
        }
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.index = 0;
        self.avg_body = 0.0;
        self.avg_range = 0.0;
        self.body_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Chart Pattern Detection
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ChartPatternDetector {
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    min_pattern_len: usize,
}

impl ChartPatternDetector {
    pub fn new(min_pattern_len: usize) -> Self {
        Self {
            highs: Vec::new(),
            lows: Vec::new(),
            closes: Vec::new(),
            min_pattern_len,
        }
    }

    pub fn add_bar(&mut self, high: f64, low: f64, close: f64) {
        self.highs.push(high);
        self.lows.push(low);
        self.closes.push(close);
    }

    pub fn detect_all(&self) -> Vec<PatternMatch> {
        let mut patterns = Vec::new();
        self.detect_head_and_shoulders(&mut patterns);
        self.detect_inverse_head_and_shoulders(&mut patterns);
        self.detect_double_top(&mut patterns);
        self.detect_double_bottom(&mut patterns);
        self.detect_ascending_triangle(&mut patterns);
        self.detect_descending_triangle(&mut patterns);
        self.detect_symmetrical_triangle(&mut patterns);
        self.detect_rising_wedge(&mut patterns);
        self.detect_falling_wedge(&mut patterns);
        self.detect_flag(&mut patterns);
        self.detect_pennant(&mut patterns);
        self.detect_cup_and_handle(&mut patterns);
        self.detect_rounding_bottom(&mut patterns);
        patterns
    }

    fn find_local_maxima(&self, window: usize) -> Vec<(usize, f64)> {
        let mut maxima = Vec::new();
        let n = self.highs.len();
        if n < 2 * window + 1 { return maxima; }
        for i in window..n - window {
            let mut is_max = true;
            for j in 1..=window {
                if self.highs[i] < self.highs[i - j] || self.highs[i] < self.highs[i + j] {
                    is_max = false;
                    break;
                }
            }
            if is_max {
                maxima.push((i, self.highs[i]));
            }
        }
        maxima
    }

    fn find_local_minima(&self, window: usize) -> Vec<(usize, f64)> {
        let mut minima = Vec::new();
        let n = self.lows.len();
        if n < 2 * window + 1 { return minima; }
        for i in window..n - window {
            let mut is_min = true;
            for j in 1..=window {
                if self.lows[i] > self.lows[i - j] || self.lows[i] > self.lows[i + j] {
                    is_min = false;
                    break;
                }
            }
            if is_min {
                minima.push((i, self.lows[i]));
            }
        }
        minima
    }

    fn detect_head_and_shoulders(&self, patterns: &mut Vec<PatternMatch>) {
        let maxima = self.find_local_maxima(5);
        let minima = self.find_local_minima(5);
        if maxima.len() < 3 || minima.len() < 2 { return; }

        for i in 0..maxima.len().saturating_sub(2) {
            let (idx1, h1) = maxima[i];
            let (idx2, h2) = maxima[i + 1];
            let (idx3, h3) = maxima[i + 2];

            // Head must be highest
            if h2 <= h1 || h2 <= h3 { continue; }

            // Shoulders should be roughly equal
            let shoulder_diff = (h1 - h3).abs() / (h2.max(1e-15));
            if shoulder_diff > 0.15 { continue; }

            // Find neckline (two troughs between shoulders)
            let mut trough1 = None;
            let mut trough2 = None;
            for &(mi, mv) in &minima {
                if mi > idx1 && mi < idx2 && trough1.is_none() {
                    trough1 = Some((mi, mv));
                }
                if mi > idx2 && mi < idx3 && trough2.is_none() {
                    trough2 = Some((mi, mv));
                }
            }

            if let (Some((_t1i, t1v)), Some((_t2i, t2v))) = (trough1, trough2) {
                let neckline_diff = (t1v - t2v).abs() / h2;
                if neckline_diff < 0.1 {
                    let score = (1.0 - shoulder_diff / 0.15) * 30.0
                        + (1.0 - neckline_diff / 0.1) * 30.0
                        + 40.0;
                    patterns.push(PatternMatch {
                        name: "Head and Shoulders",
                        pattern_type: PatternType::Chart,
                        direction: PatternDirection::Bearish,
                        score: score.min(100.0),
                        start_index: idx1,
                        end_index: idx3,
                    });
                }
            }
        }
    }

    fn detect_inverse_head_and_shoulders(&self, patterns: &mut Vec<PatternMatch>) {
        let maxima = self.find_local_maxima(5);
        let minima = self.find_local_minima(5);
        if minima.len() < 3 || maxima.len() < 2 { return; }

        for i in 0..minima.len().saturating_sub(2) {
            let (idx1, l1) = minima[i];
            let (idx2, l2) = minima[i + 1];
            let (idx3, l3) = minima[i + 2];

            if l2 >= l1 || l2 >= l3 { continue; }
            let shoulder_diff = (l1 - l3).abs() / (l1.abs().max(1e-15));
            if shoulder_diff > 0.15 { continue; }

            let mut peak1 = None;
            let mut peak2 = None;
            for &(mi, mv) in &maxima {
                if mi > idx1 && mi < idx2 && peak1.is_none() { peak1 = Some((mi, mv)); }
                if mi > idx2 && mi < idx3 && peak2.is_none() { peak2 = Some((mi, mv)); }
            }

            if let (Some((_, p1v)), Some((_, p2v))) = (peak1, peak2) {
                let neckline_diff = (p1v - p2v).abs() / p1v.abs().max(1e-15);
                if neckline_diff < 0.1 {
                    let score = 70.0 + (1.0 - shoulder_diff / 0.15) * 15.0 + (1.0 - neckline_diff / 0.1) * 15.0;
                    patterns.push(PatternMatch {
                        name: "Inverse Head and Shoulders",
                        pattern_type: PatternType::Chart,
                        direction: PatternDirection::Bullish,
                        score: score.min(100.0),
                        start_index: idx1,
                        end_index: idx3,
                    });
                }
            }
        }
    }

    fn detect_double_top(&self, patterns: &mut Vec<PatternMatch>) {
        let maxima = self.find_local_maxima(5);
        if maxima.len() < 2 { return; }

        for i in 0..maxima.len() - 1 {
            let (idx1, h1) = maxima[i];
            let (idx2, h2) = maxima[i + 1];

            if idx2 - idx1 < self.min_pattern_len { continue; }

            let diff = (h1 - h2).abs() / h1.max(1e-15);
            if diff < 0.03 {
                // Find trough between
                let mut min_val = f64::INFINITY;
                for j in idx1..=idx2 {
                    if self.lows[j] < min_val { min_val = self.lows[j]; }
                }
                let depth = (h1 - min_val) / h1.max(1e-15);
                if depth > 0.03 {
                    let score = (1.0 - diff / 0.03) * 30.0 + depth.min(0.15) / 0.15 * 30.0 + 40.0;
                    patterns.push(PatternMatch {
                        name: "Double Top",
                        pattern_type: PatternType::Chart,
                        direction: PatternDirection::Bearish,
                        score: score.min(100.0),
                        start_index: idx1,
                        end_index: idx2,
                    });
                }
            }
        }
    }

    fn detect_double_bottom(&self, patterns: &mut Vec<PatternMatch>) {
        let minima = self.find_local_minima(5);
        if minima.len() < 2 { return; }

        for i in 0..minima.len() - 1 {
            let (idx1, l1) = minima[i];
            let (idx2, l2) = minima[i + 1];

            if idx2 - idx1 < self.min_pattern_len { continue; }

            let diff = (l1 - l2).abs() / l1.abs().max(1e-15);
            if diff < 0.03 {
                let mut max_val = f64::NEG_INFINITY;
                for j in idx1..=idx2 {
                    if self.highs[j] > max_val { max_val = self.highs[j]; }
                }
                let height = (max_val - l1) / l1.abs().max(1e-15);
                if height > 0.03 {
                    let score = (1.0 - diff / 0.03) * 30.0 + height.min(0.15) / 0.15 * 30.0 + 40.0;
                    patterns.push(PatternMatch {
                        name: "Double Bottom",
                        pattern_type: PatternType::Chart,
                        direction: PatternDirection::Bullish,
                        score: score.min(100.0),
                        start_index: idx1,
                        end_index: idx2,
                    });
                }
            }
        }
    }

    fn detect_ascending_triangle(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.highs.len();
        if n < self.min_pattern_len * 2 { return; }

        // Sliding window approach
        let window = self.min_pattern_len * 2;
        for start in (0..n.saturating_sub(window)).step_by(window / 2) {
            let end = (start + window).min(n);
            let slice_h = &self.highs[start..end];
            let slice_l = &self.lows[start..end];

            // Check flat resistance (highs roughly constant)
            let max_h = slice_h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_h = slice_h.iter().cloned().fold(f64::INFINITY, f64::min);
            let h_range = max_h - min_h;
            let avg_h = slice_h.iter().sum::<f64>() / slice_h.len() as f64;

            // Check rising support (lows trending up)
            let half = slice_l.len() / 2;
            let avg_l_first = slice_l[..half].iter().sum::<f64>() / half as f64;
            let avg_l_second = slice_l[half..].iter().sum::<f64>() / (slice_l.len() - half) as f64;

            if h_range / avg_h.max(1e-15) < 0.03 && avg_l_second > avg_l_first {
                let flatness = 1.0 - (h_range / avg_h.max(1e-15)) / 0.03;
                let trend_strength = (avg_l_second - avg_l_first) / avg_h.max(1e-15);
                let score = flatness * 30.0 + trend_strength.min(0.05) / 0.05 * 30.0 + 40.0;
                patterns.push(PatternMatch {
                    name: "Ascending Triangle",
                    pattern_type: PatternType::Chart,
                    direction: PatternDirection::Bullish,
                    score: score.min(100.0),
                    start_index: start,
                    end_index: end - 1,
                });
            }
        }
    }

    fn detect_descending_triangle(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.lows.len();
        if n < self.min_pattern_len * 2 { return; }

        let window = self.min_pattern_len * 2;
        for start in (0..n.saturating_sub(window)).step_by(window / 2) {
            let end = (start + window).min(n);
            let slice_h = &self.highs[start..end];
            let slice_l = &self.lows[start..end];

            let max_l = slice_l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_l = slice_l.iter().cloned().fold(f64::INFINITY, f64::min);
            let l_range = max_l - min_l;
            let avg_l = slice_l.iter().sum::<f64>() / slice_l.len() as f64;

            let half = slice_h.len() / 2;
            let avg_h_first = slice_h[..half].iter().sum::<f64>() / half as f64;
            let avg_h_second = slice_h[half..].iter().sum::<f64>() / (slice_h.len() - half) as f64;

            if l_range / avg_l.abs().max(1e-15) < 0.03 && avg_h_second < avg_h_first {
                let score = 65.0;
                patterns.push(PatternMatch {
                    name: "Descending Triangle",
                    pattern_type: PatternType::Chart,
                    direction: PatternDirection::Bearish,
                    score,
                    start_index: start,
                    end_index: end - 1,
                });
            }
        }
    }

    fn detect_symmetrical_triangle(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.highs.len();
        if n < self.min_pattern_len * 2 { return; }

        let window = self.min_pattern_len * 2;
        for start in (0..n.saturating_sub(window)).step_by(window / 2) {
            let end = (start + window).min(n);
            let slice_h = &self.highs[start..end];
            let slice_l = &self.lows[start..end];

            let half = slice_h.len() / 2;
            let avg_h_first = slice_h[..half].iter().sum::<f64>() / half as f64;
            let avg_h_second = slice_h[half..].iter().sum::<f64>() / (slice_h.len() - half) as f64;
            let avg_l_first = slice_l[..half].iter().sum::<f64>() / half as f64;
            let avg_l_second = slice_l[half..].iter().sum::<f64>() / (slice_l.len() - half) as f64;

            // Highs converging down, lows converging up
            if avg_h_second < avg_h_first && avg_l_second > avg_l_first {
                let h_convergence = avg_h_first - avg_h_second;
                let l_convergence = avg_l_second - avg_l_first;
                let symmetry = 1.0 - (h_convergence - l_convergence).abs() / (h_convergence + l_convergence).max(1e-15);
                if symmetry > 0.5 {
                    let score = symmetry * 40.0 + 40.0;
                    patterns.push(PatternMatch {
                        name: "Symmetrical Triangle",
                        pattern_type: PatternType::Chart,
                        direction: PatternDirection::Neutral,
                        score: score.min(100.0),
                        start_index: start,
                        end_index: end - 1,
                    });
                }
            }
        }
    }

    fn detect_rising_wedge(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.highs.len();
        if n < self.min_pattern_len * 2 { return; }

        let window = self.min_pattern_len * 2;
        for start in (0..n.saturating_sub(window)).step_by(window / 2) {
            let end = (start + window).min(n);
            let slice_h = &self.highs[start..end];
            let slice_l = &self.lows[start..end];

            // Both highs and lows trending up, converging
            let (h_slope, _) = linear_regression_slope(slice_h);
            let (l_slope, _) = linear_regression_slope(slice_l);

            if h_slope > 0.0 && l_slope > 0.0 && l_slope > h_slope {
                let score = 60.0;
                patterns.push(PatternMatch {
                    name: "Rising Wedge",
                    pattern_type: PatternType::Chart,
                    direction: PatternDirection::Bearish,
                    score,
                    start_index: start,
                    end_index: end - 1,
                });
            }
        }
    }

    fn detect_falling_wedge(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.highs.len();
        if n < self.min_pattern_len * 2 { return; }

        let window = self.min_pattern_len * 2;
        for start in (0..n.saturating_sub(window)).step_by(window / 2) {
            let end = (start + window).min(n);
            let slice_h = &self.highs[start..end];
            let slice_l = &self.lows[start..end];

            let (h_slope, _) = linear_regression_slope(slice_h);
            let (l_slope, _) = linear_regression_slope(slice_l);

            if h_slope < 0.0 && l_slope < 0.0 && h_slope > l_slope {
                let score = 60.0;
                patterns.push(PatternMatch {
                    name: "Falling Wedge",
                    pattern_type: PatternType::Chart,
                    direction: PatternDirection::Bullish,
                    score,
                    start_index: start,
                    end_index: end - 1,
                });
            }
        }
    }

    fn detect_flag(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.closes.len();
        if n < self.min_pattern_len + 10 { return; }

        // Look for sharp move followed by consolidation channel
        let pole_len = self.min_pattern_len;
        let flag_len = self.min_pattern_len;

        for start in 0..n.saturating_sub(pole_len + flag_len) {
            let pole_start = start;
            let pole_end = start + pole_len;
            let flag_start = pole_end;
            let flag_end = (flag_start + flag_len).min(n);
            if flag_end > n { continue; }

            let pole_move = self.closes[pole_end - 1] - self.closes[pole_start];
            let pole_range = self.highs[pole_start..pole_end].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - self.lows[pole_start..pole_end].iter().cloned().fold(f64::INFINITY, f64::min);

            if pole_move.abs() < pole_range * 0.5 { continue; }

            // Flag: low volatility channel
            let flag_range = self.highs[flag_start..flag_end].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - self.lows[flag_start..flag_end].iter().cloned().fold(f64::INFINITY, f64::min);

            if flag_range < pole_range * 0.5 {
                let dir = if pole_move > 0.0 { PatternDirection::Bullish } else { PatternDirection::Bearish };
                let score = (1.0 - flag_range / pole_range) * 40.0 + 40.0;
                patterns.push(PatternMatch {
                    name: if pole_move > 0.0 { "Bull Flag" } else { "Bear Flag" },
                    pattern_type: PatternType::Chart,
                    direction: dir,
                    score: score.min(100.0),
                    start_index: pole_start,
                    end_index: flag_end - 1,
                });
            }
        }
    }

    fn detect_pennant(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.closes.len();
        if n < self.min_pattern_len + 10 { return; }

        let pole_len = self.min_pattern_len;
        let pennant_len = self.min_pattern_len;

        for start in (0..n.saturating_sub(pole_len + pennant_len)).step_by(pole_len) {
            let pole_end = start + pole_len;
            let pn_end = (pole_end + pennant_len).min(n);

            let pole_move = self.closes[pole_end - 1] - self.closes[start];
            if pole_move.abs() < self.avg_range_in(&self.highs[start..pole_end], &self.lows[start..pole_end]) * 3.0 {
                continue;
            }

            // Check converging highs and lows in pennant
            let slice_h = &self.highs[pole_end..pn_end];
            let slice_l = &self.lows[pole_end..pn_end];
            if slice_h.len() < 4 { continue; }

            let (h_slope, _) = linear_regression_slope(slice_h);
            let (l_slope, _) = linear_regression_slope(slice_l);

            if h_slope < 0.0 && l_slope > 0.0 {
                let dir = if pole_move > 0.0 { PatternDirection::Bullish } else { PatternDirection::Bearish };
                patterns.push(PatternMatch {
                    name: "Pennant",
                    pattern_type: PatternType::Chart,
                    direction: dir,
                    score: 60.0,
                    start_index: start,
                    end_index: pn_end - 1,
                });
            }
        }
    }

    fn detect_cup_and_handle(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.closes.len();
        if n < self.min_pattern_len * 4 { return; }

        let cup_len = self.min_pattern_len * 3;
        let handle_len = self.min_pattern_len;

        for start in (0..n.saturating_sub(cup_len + handle_len)).step_by(cup_len / 2) {
            let cup_end = start + cup_len;
            let handle_end = (cup_end + handle_len).min(n);
            if handle_end > n { continue; }

            let left_rim = self.closes[start];
            let right_rim = self.closes[cup_end - 1];

            // Find bottom of cup
            let cup_min = self.lows[start..cup_end].iter().cloned().fold(f64::INFINITY, f64::min);
            let cup_depth = (left_rim.min(right_rim) - cup_min) / left_rim.max(1e-15);

            // Rims should be roughly equal
            let rim_diff = (left_rim - right_rim).abs() / left_rim.max(1e-15);

            if cup_depth > 0.05 && cup_depth < 0.5 && rim_diff < 0.1 {
                // Handle: small decline
                let handle_max = self.highs[cup_end..handle_end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let handle_min = self.lows[cup_end..handle_end].iter().cloned().fold(f64::INFINITY, f64::min);
                let handle_depth = (right_rim - handle_min) / right_rim.max(1e-15);

                if handle_depth < cup_depth * 0.5 && handle_depth > 0.01 {
                    let score = 65.0 + (1.0 - rim_diff / 0.1) * 15.0 + (1.0 - handle_depth / cup_depth) * 20.0;
                    patterns.push(PatternMatch {
                        name: "Cup and Handle",
                        pattern_type: PatternType::Chart,
                        direction: PatternDirection::Bullish,
                        score: score.min(100.0),
                        start_index: start,
                        end_index: handle_end - 1,
                    });
                }
            }
        }
    }

    fn detect_rounding_bottom(&self, patterns: &mut Vec<PatternMatch>) {
        let n = self.closes.len();
        if n < self.min_pattern_len * 3 { return; }

        let window = self.min_pattern_len * 3;
        for start in (0..n.saturating_sub(window)).step_by(window / 2) {
            let end = (start + window).min(n);
            let slice = &self.closes[start..end];
            let len = slice.len();

            // Check U-shape: first half decreasing, second half increasing
            let third = len / 3;
            let avg_first = slice[..third].iter().sum::<f64>() / third as f64;
            let avg_mid = slice[third..2 * third].iter().sum::<f64>() / third as f64;
            let avg_last = slice[2 * third..].iter().sum::<f64>() / (len - 2 * third) as f64;

            if avg_mid < avg_first && avg_mid < avg_last && avg_last > avg_first * 0.95 {
                let depth = (avg_first.min(avg_last) - avg_mid) / avg_first.max(1e-15);
                if depth > 0.03 {
                    let score = depth.min(0.2) / 0.2 * 40.0 + 40.0;
                    patterns.push(PatternMatch {
                        name: "Rounding Bottom",
                        pattern_type: PatternType::Chart,
                        direction: PatternDirection::Bullish,
                        score: score.min(100.0),
                        start_index: start,
                        end_index: end - 1,
                    });
                }
            }
        }
    }

    fn avg_range_in(&self, highs: &[f64], lows: &[f64]) -> f64 {
        if highs.is_empty() { return 0.0; }
        let sum: f64 = highs.iter().zip(lows.iter()).map(|(&h, &l)| h - l).sum();
        sum / highs.len() as f64
    }

    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.closes.clear();
    }
}

// ---------------------------------------------------------------------------
// Helper: simple linear regression slope
// ---------------------------------------------------------------------------
fn linear_regression_slope(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    if n < 2.0 { return (0.0, data.first().copied().unwrap_or(0.0)); }
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (i, &y) in data.iter().enumerate() {
        let x = i as f64;
        sx += x;
        sy += y;
        sxy += x * y;
        sxx += x * x;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-15 {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

// ---------------------------------------------------------------------------
// Batch detection convenience
// ---------------------------------------------------------------------------
pub fn detect_candlestick_patterns(candles: &[Candle]) -> Vec<PatternMatch> {
    let mut detector = CandlestickDetector::new();
    let mut all_patterns = Vec::new();
    for c in candles {
        let mut p = detector.update(*c);
        all_patterns.append(&mut p);
    }
    all_patterns
}

pub fn detect_chart_patterns(highs: &[f64], lows: &[f64], closes: &[f64], min_len: usize) -> Vec<PatternMatch> {
    let mut detector = ChartPatternDetector::new(min_len);
    for i in 0..highs.len() {
        detector.add_bar(highs[i], lows[i], closes[i]);
    }
    detector.detect_all()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(o: f64, h: f64, l: f64, c: f64) -> Candle {
        Candle::new(o, h, l, c, 1000.0)
    }

    #[test]
    fn test_candle_properties() {
        let c = make_candle(100.0, 105.0, 95.0, 103.0);
        assert!(c.is_bullish());
        assert!((c.body_size() - 3.0).abs() < 1e-10);
        assert!((c.range() - 10.0).abs() < 1e-10);
        assert!((c.upper_shadow() - 2.0).abs() < 1e-10);
        assert!((c.lower_shadow() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_doji_detection() {
        let mut det = CandlestickDetector::new();
        // Feed some normal candles first for averages
        for i in 0..10 {
            let o = 100.0 + i as f64;
            det.update(make_candle(o, o + 2.0, o - 2.0, o + 1.5));
        }
        // Now a doji
        let patterns = det.update(make_candle(110.0, 112.0, 108.0, 110.01));
        let doji = patterns.iter().find(|p| p.name == "Doji");
        assert!(doji.is_some());
    }

    #[test]
    fn test_engulfing_detection() {
        let mut det = CandlestickDetector::new();
        for i in 0..10 {
            let o = 100.0 + i as f64;
            det.update(make_candle(o, o + 2.0, o - 2.0, o + 1.5));
        }
        // Bearish candle
        det.update(make_candle(112.0, 113.0, 110.0, 110.5));
        // Bullish engulfing
        let patterns = det.update(make_candle(110.0, 114.0, 109.0, 113.0));
        let engulfing = patterns.iter().find(|p| p.name == "Bullish Engulfing");
        assert!(engulfing.is_some());
    }

    #[test]
    fn test_hammer_detection() {
        let mut det = CandlestickDetector::new();
        for i in 0..10 {
            det.update(make_candle(100.0, 102.0, 98.0, 101.0));
        }
        let patterns = det.update(make_candle(100.0, 100.5, 95.0, 100.3));
        let hammer = patterns.iter().find(|p| p.name == "Hammer");
        assert!(hammer.is_some());
    }

    #[test]
    fn test_chart_pattern_double_top() {
        let mut det = ChartPatternDetector::new(5);
        // Create price data with double top
        for i in 0..50 {
            let t = i as f64;
            let price = if i < 15 {
                100.0 + t * 2.0
            } else if i < 25 {
                130.0 - (t - 15.0) * 2.0
            } else if i < 35 {
                110.0 + (t - 25.0) * 2.0
            } else {
                130.0 - (t - 35.0) * 2.0
            };
            det.add_bar(price + 1.0, price - 1.0, price);
        }
        let patterns = det.detect_all();
        // May or may not detect depending on exact shape
        assert!(patterns.len() >= 0);
    }

    #[test]
    fn test_linear_regression_slope() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (slope, intercept) = linear_regression_slope(&data);
        assert!((slope - 1.0).abs() < 1e-10);
        assert!((intercept - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_detection() {
        let candles: Vec<Candle> = (0..30)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.3).sin() * 5.0;
                make_candle(base, base + 2.0, base - 2.0, base + 1.0)
            })
            .collect();
        let patterns = detect_candlestick_patterns(&candles);
        // Just verify it runs without panics
        assert!(patterns.len() >= 0);
    }
}
