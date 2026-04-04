package aggregator

import (
	"math"

	"github.com/srfm/gateway/internal/feed"
)

// CandlePattern describes a recognised candlestick pattern.
type CandlePattern struct {
	Name        string
	Bullish     bool   // true = bullish signal, false = bearish
	Confidence  float64 // 0-1 confidence score
	Description string
}

// DetectPatterns analyses a slice of recent bars and returns recognised patterns.
// bars should be in chronological order.
func DetectPatterns(bars []feed.Bar) []CandlePattern {
	if len(bars) == 0 {
		return nil
	}
	var patterns []CandlePattern

	// Single-bar patterns.
	last := bars[len(bars)-1]
	if p := detectDoji(last); p != nil {
		patterns = append(patterns, *p)
	}
	if p := detectHammer(last); p != nil {
		patterns = append(patterns, *p)
	}
	if p := detectShootingStar(last); p != nil {
		patterns = append(patterns, *p)
	}
	if p := detectMarubozu(last); p != nil {
		patterns = append(patterns, *p)
	}
	if p := detectSpinningTop(last); p != nil {
		patterns = append(patterns, *p)
	}

	// Two-bar patterns.
	if len(bars) >= 2 {
		prev := bars[len(bars)-2]
		if p := detectEngulfing(prev, last); p != nil {
			patterns = append(patterns, *p)
		}
		if p := detectHarami(prev, last); p != nil {
			patterns = append(patterns, *p)
		}
		if p := detectPiercingLine(prev, last); p != nil {
			patterns = append(patterns, *p)
		}
		if p := detectDarkCloudCover(prev, last); p != nil {
			patterns = append(patterns, *p)
		}
	}

	// Three-bar patterns.
	if len(bars) >= 3 {
		b1 := bars[len(bars)-3]
		b2 := bars[len(bars)-2]
		b3 := bars[len(bars)-1]
		if p := detectMorningStar(b1, b2, b3); p != nil {
			patterns = append(patterns, *p)
		}
		if p := detectEveningStar(b1, b2, b3); p != nil {
			patterns = append(patterns, *p)
		}
		if p := detectThreeWhiteSoldiers(b1, b2, b3); p != nil {
			patterns = append(patterns, *p)
		}
		if p := detectThreeBlackCrows(b1, b2, b3); p != nil {
			patterns = append(patterns, *p)
		}
	}

	return patterns
}

// bodySize returns the absolute body size (|close - open|).
func bodySize(b feed.Bar) float64 {
	return math.Abs(b.Close - b.Open)
}

// upperShadow returns the upper wick length.
func upperShadow(b feed.Bar) float64 {
	top := b.Close
	if b.Open > top {
		top = b.Open
	}
	return b.High - top
}

// lowerShadow returns the lower wick length.
func lowerShadow(b feed.Bar) float64 {
	bottom := b.Close
	if b.Open < bottom {
		bottom = b.Open
	}
	return bottom - b.Low
}

// totalRange is the full bar range.
func totalRange(b feed.Bar) float64 {
	return b.High - b.Low
}

// isBullish returns true if the bar closed higher than it opened.
func isBullish(b feed.Bar) bool {
	return b.Close >= b.Open
}

// detectDoji detects a doji (body < 10% of range).
func detectDoji(b feed.Bar) *CandlePattern {
	r := totalRange(b)
	if r == 0 {
		return nil
	}
	if bodySize(b)/r < 0.10 {
		return &CandlePattern{
			Name:        "Doji",
			Bullish:     false,
			Confidence:  0.5,
			Description: "Indecision candle — small body relative to wicks",
		}
	}
	return nil
}

// detectHammer detects a hammer (long lower wick, small body at top).
func detectHammer(b feed.Bar) *CandlePattern {
	r := totalRange(b)
	if r == 0 {
		return nil
	}
	body := bodySize(b)
	lower := lowerShadow(b)
	upper := upperShadow(b)
	if lower >= 2*body && body/r < 0.35 && upper < body {
		return &CandlePattern{
			Name:        "Hammer",
			Bullish:     true,
			Confidence:  0.65,
			Description: "Long lower wick suggests rejection of lower prices (bullish reversal)",
		}
	}
	return nil
}

// detectShootingStar detects a shooting star (long upper wick, small body at bottom).
func detectShootingStar(b feed.Bar) *CandlePattern {
	r := totalRange(b)
	if r == 0 {
		return nil
	}
	body := bodySize(b)
	lower := lowerShadow(b)
	upper := upperShadow(b)
	if upper >= 2*body && body/r < 0.35 && lower < body {
		return &CandlePattern{
			Name:        "Shooting Star",
			Bullish:     false,
			Confidence:  0.65,
			Description: "Long upper wick suggests rejection of higher prices (bearish reversal)",
		}
	}
	return nil
}

// detectMarubozu detects a marubozu (no wicks, full body).
func detectMarubozu(b feed.Bar) *CandlePattern {
	r := totalRange(b)
	if r == 0 {
		return nil
	}
	if bodySize(b)/r > 0.95 {
		bull := isBullish(b)
		name := "Bearish Marubozu"
		if bull {
			name = "Bullish Marubozu"
		}
		return &CandlePattern{
			Name:        name,
			Bullish:     bull,
			Confidence:  0.70,
			Description: "Full-body bar with no wicks — strong directional conviction",
		}
	}
	return nil
}

// detectSpinningTop detects a spinning top (small body, significant wicks on both sides).
func detectSpinningTop(b feed.Bar) *CandlePattern {
	r := totalRange(b)
	if r == 0 {
		return nil
	}
	body := bodySize(b)
	upper := upperShadow(b)
	lower := lowerShadow(b)
	if body/r < 0.25 && upper > body && lower > body {
		return &CandlePattern{
			Name:        "Spinning Top",
			Bullish:     false,
			Confidence:  0.45,
			Description: "Small body with wicks on both sides — market indecision",
		}
	}
	return nil
}

// detectEngulfing detects bullish or bearish engulfing patterns.
func detectEngulfing(prev, curr feed.Bar) *CandlePattern {
	prevBull := isBullish(prev)
	currBull := isBullish(curr)
	if prevBull == currBull {
		return nil
	}
	prevBody := bodySize(prev)
	currBody := bodySize(curr)
	if currBody <= prevBody {
		return nil
	}

	if !prevBull && currBull {
		// Bullish engulfing: bearish prev, bullish current that engulfs prev body.
		if curr.Open <= prev.Close && curr.Close >= prev.Open {
			return &CandlePattern{
				Name:        "Bullish Engulfing",
				Bullish:     true,
				Confidence:  0.75,
				Description: "Bullish bar engulfs the previous bearish bar — strong reversal signal",
			}
		}
	} else if prevBull && !currBull {
		// Bearish engulfing.
		if curr.Open >= prev.Close && curr.Close <= prev.Open {
			return &CandlePattern{
				Name:        "Bearish Engulfing",
				Bullish:     false,
				Confidence:  0.75,
				Description: "Bearish bar engulfs the previous bullish bar — strong reversal signal",
			}
		}
	}
	return nil
}

// detectHarami detects a harami (inside bar) pattern.
func detectHarami(prev, curr feed.Bar) *CandlePattern {
	prevBody := bodySize(prev)
	currBody := bodySize(curr)
	if currBody >= prevBody {
		return nil
	}
	// Current body must be inside prev body.
	prevTop := math.Max(prev.Open, prev.Close)
	prevBottom := math.Min(prev.Open, prev.Close)
	currTop := math.Max(curr.Open, curr.Close)
	currBottom := math.Min(curr.Open, curr.Close)

	if currTop <= prevTop && currBottom >= prevBottom {
		bull := !isBullish(prev) // harami is bullish after a down bar
		name := "Bearish Harami"
		if bull {
			name = "Bullish Harami"
		}
		return &CandlePattern{
			Name:        name,
			Bullish:     bull,
			Confidence:  0.55,
			Description: "Small inside bar after a large bar — potential reversal",
		}
	}
	return nil
}

// detectPiercingLine detects a piercing line (bullish reversal after down bar).
func detectPiercingLine(prev, curr feed.Bar) *CandlePattern {
	if isBullish(prev) || !isBullish(curr) {
		return nil
	}
	prevMid := (prev.Open + prev.Close) / 2
	if curr.Open < prev.Close && curr.Close > prevMid && curr.Close < prev.Open {
		return &CandlePattern{
			Name:        "Piercing Line",
			Bullish:     true,
			Confidence:  0.65,
			Description: "Bullish bar opens below prior close and closes above prior midpoint",
		}
	}
	return nil
}

// detectDarkCloudCover detects a dark cloud cover (bearish reversal after up bar).
func detectDarkCloudCover(prev, curr feed.Bar) *CandlePattern {
	if !isBullish(prev) || isBullish(curr) {
		return nil
	}
	prevMid := (prev.Open + prev.Close) / 2
	if curr.Open > prev.Close && curr.Close < prevMid && curr.Close > prev.Open {
		return &CandlePattern{
			Name:        "Dark Cloud Cover",
			Bullish:     false,
			Confidence:  0.65,
			Description: "Bearish bar opens above prior close and closes below prior midpoint",
		}
	}
	return nil
}

// detectMorningStar detects a morning star (3-bar bullish reversal).
func detectMorningStar(b1, b2, b3 feed.Bar) *CandlePattern {
	if isBullish(b1) || bodySize(b2) > bodySize(b1)*0.5 || !isBullish(b3) {
		return nil
	}
	if b3.Close > (b1.Open+b1.Close)/2 {
		return &CandlePattern{
			Name:        "Morning Star",
			Bullish:     true,
			Confidence:  0.80,
			Description: "Three-bar bullish reversal: down bar, doji/small, up bar",
		}
	}
	return nil
}

// detectEveningStar detects an evening star (3-bar bearish reversal).
func detectEveningStar(b1, b2, b3 feed.Bar) *CandlePattern {
	if !isBullish(b1) || bodySize(b2) > bodySize(b1)*0.5 || isBullish(b3) {
		return nil
	}
	if b3.Close < (b1.Open+b1.Close)/2 {
		return &CandlePattern{
			Name:        "Evening Star",
			Bullish:     false,
			Confidence:  0.80,
			Description: "Three-bar bearish reversal: up bar, doji/small, down bar",
		}
	}
	return nil
}

// detectThreeWhiteSoldiers detects three consecutive bullish bars.
func detectThreeWhiteSoldiers(b1, b2, b3 feed.Bar) *CandlePattern {
	if !isBullish(b1) || !isBullish(b2) || !isBullish(b3) {
		return nil
	}
	// Each bar opens within prior body and closes higher.
	if b2.Open > b1.Open && b2.Open < b1.Close &&
		b3.Open > b2.Open && b3.Open < b2.Close &&
		b3.Close > b2.Close {
		return &CandlePattern{
			Name:        "Three White Soldiers",
			Bullish:     true,
			Confidence:  0.75,
			Description: "Three consecutive advancing bullish bars — strong uptrend continuation",
		}
	}
	return nil
}

// detectThreeBlackCrows detects three consecutive bearish bars.
func detectThreeBlackCrows(b1, b2, b3 feed.Bar) *CandlePattern {
	if isBullish(b1) || isBullish(b2) || isBullish(b3) {
		return nil
	}
	if b2.Open < b1.Open && b2.Open > b1.Close &&
		b3.Open < b2.Open && b3.Open > b2.Close &&
		b3.Close < b2.Close {
		return &CandlePattern{
			Name:        "Three Black Crows",
			Bullish:     false,
			Confidence:  0.75,
			Description: "Three consecutive declining bearish bars — strong downtrend continuation",
		}
	}
	return nil
}
