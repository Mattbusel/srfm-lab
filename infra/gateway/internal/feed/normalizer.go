package feed

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

// NormalizedSymbol maps exchange-specific tickers to canonical symbols.
// e.g. Binance "BTCUSDT" -> canonical "BTC/USD".
type NormalizedSymbol struct {
	Canonical string
	Exchange  string
	Native    string
}

// SymbolNormalizer maps native feed symbols to canonical identifiers and
// applies basic sanity checks on incoming bars.
type SymbolNormalizer struct {
	mu      sync.RWMutex
	mapping map[string]NormalizedSymbol // native -> canonical info
	log     *zap.Logger
}

// NewSymbolNormalizer creates a SymbolNormalizer with a default mapping table.
func NewSymbolNormalizer(log *zap.Logger) *SymbolNormalizer {
	sn := &SymbolNormalizer{
		log:     log,
		mapping: make(map[string]NormalizedSymbol),
	}
	sn.loadDefaults()
	return sn
}

// loadDefaults populates well-known symbol mappings.
func (sn *SymbolNormalizer) loadDefaults() {
	defaults := []NormalizedSymbol{
		// Binance crypto.
		{Canonical: "BTCUSD", Exchange: "binance", Native: "BTCUSDT"},
		{Canonical: "ETHUSD", Exchange: "binance", Native: "ETHUSDT"},
		{Canonical: "SOLUSD", Exchange: "binance", Native: "SOLUSDT"},
		{Canonical: "BNBUSD", Exchange: "binance", Native: "BNBUSDT"},
		{Canonical: "XRPUSD", Exchange: "binance", Native: "XRPUSDT"},
		{Canonical: "ADAUSD", Exchange: "binance", Native: "ADAUSDT"},
		// Alpaca stocks are already canonical.
		{Canonical: "AAPL", Exchange: "alpaca", Native: "AAPL"},
		{Canonical: "MSFT", Exchange: "alpaca", Native: "MSFT"},
		{Canonical: "TSLA", Exchange: "alpaca", Native: "TSLA"},
		{Canonical: "NVDA", Exchange: "alpaca", Native: "NVDA"},
		{Canonical: "SPY", Exchange: "alpaca", Native: "SPY"},
		{Canonical: "QQQ", Exchange: "alpaca", Native: "QQQ"},
		{Canonical: "AMZN", Exchange: "alpaca", Native: "AMZN"},
		{Canonical: "GOOGL", Exchange: "alpaca", Native: "GOOGL"},
	}
	for _, ns := range defaults {
		sn.mapping[ns.Native] = ns
	}
}

// AddMapping registers a custom symbol mapping.
func (sn *SymbolNormalizer) AddMapping(ns NormalizedSymbol) {
	sn.mu.Lock()
	defer sn.mu.Unlock()
	sn.mapping[ns.Native] = ns
}

// Normalize translates a native symbol to canonical form.
// If the symbol is unknown, it is returned as-is.
func (sn *SymbolNormalizer) Normalize(native string) string {
	sn.mu.RLock()
	ns, ok := sn.mapping[native]
	sn.mu.RUnlock()
	if ok {
		return ns.Canonical
	}
	// Try upper-case lookup.
	upper := strings.ToUpper(native)
	sn.mu.RLock()
	ns, ok = sn.mapping[upper]
	sn.mu.RUnlock()
	if ok {
		return ns.Canonical
	}
	return native
}

// NormalizeBar returns a copy of the bar with the symbol normalised.
func (sn *SymbolNormalizer) NormalizeBar(b Bar) Bar {
	b.Symbol = sn.Normalize(b.Symbol)
	return b
}

// ----- BarValidator -----

// ValidationError describes a specific bar data issue.
type ValidationError struct {
	Field   string
	Message string
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("field %q: %s", e.Field, e.Message)
}

// BarValidator applies sanity checks to incoming bars.
type BarValidator struct {
	// MaxPriceChange is the maximum allowed single-bar return (fraction).
	// Default: 0.50 (50%).
	MaxPriceChange float64
	// MinPrice is the minimum acceptable price.
	MinPrice float64
	// MaxVolume is the maximum acceptable volume per bar (0 = unlimited).
	MaxVolume float64
	// MaxAgeSecs is the maximum bar age in seconds (0 = unlimited).
	MaxAgeSecs float64

	log *zap.Logger

	mu       sync.RWMutex
	prevClose map[string]float64
}

// NewBarValidator creates a BarValidator with sensible defaults.
func NewBarValidator(log *zap.Logger) *BarValidator {
	return &BarValidator{
		MaxPriceChange: 0.50,
		MinPrice:       0.0001,
		MaxVolume:      1e12,
		MaxAgeSecs:     300,
		log:            log,
		prevClose:      make(map[string]float64),
	}
}

// Validate checks a bar for data quality issues.
// Returns nil if the bar is valid, otherwise a slice of validation errors.
func (v *BarValidator) Validate(b Bar) []ValidationError {
	var errs []ValidationError

	// Price checks.
	if b.Open <= v.MinPrice {
		errs = append(errs, ValidationError{"open", fmt.Sprintf("open=%.6f below min=%.6f", b.Open, v.MinPrice)})
	}
	if b.High < b.Open || b.High < b.Close || b.High < b.Low {
		errs = append(errs, ValidationError{"high", "high is not the highest OHLC value"})
	}
	if b.Low > b.Open || b.Low > b.Close || b.Low > b.High {
		errs = append(errs, ValidationError{"low", "low is not the lowest OHLC value"})
	}
	if b.Close <= v.MinPrice {
		errs = append(errs, ValidationError{"close", fmt.Sprintf("close=%.6f below min=%.6f", b.Close, v.MinPrice)})
	}
	if b.Volume < 0 {
		errs = append(errs, ValidationError{"volume", "negative volume"})
	}
	if v.MaxVolume > 0 && b.Volume > v.MaxVolume {
		errs = append(errs, ValidationError{"volume", fmt.Sprintf("volume %.0f exceeds max %.0f", b.Volume, v.MaxVolume)})
	}

	// Timestamp.
	if b.Timestamp.IsZero() {
		errs = append(errs, ValidationError{"timestamp", "zero timestamp"})
	}
	if v.MaxAgeSecs > 0 {
		age := time.Since(b.Timestamp).Seconds()
		if age > v.MaxAgeSecs && age > 0 {
			// Don't reject historical bars, just log.
			v.log.Debug("bar is old",
				zap.String("symbol", b.Symbol),
				zap.Float64("age_seconds", age))
		}
	}

	// Consecutive return check.
	if v.MaxPriceChange > 0 {
		v.mu.RLock()
		prev, hasPrev := v.prevClose[b.Symbol]
		v.mu.RUnlock()
		if hasPrev && prev > 0 {
			ret := math.Abs(b.Close-prev) / prev
			if ret > v.MaxPriceChange {
				errs = append(errs, ValidationError{
					"close",
					fmt.Sprintf("return %.2f%% exceeds max %.2f%% (prev=%.4f close=%.4f)",
						ret*100, v.MaxPriceChange*100, prev, b.Close),
				})
			}
		}
	}

	// Update previous close.
	if len(errs) == 0 && !b.IsPartial {
		v.mu.Lock()
		v.prevClose[b.Symbol] = b.Close
		v.mu.Unlock()
	}

	return errs
}

// IsValid returns true if the bar passes all validation checks.
func (v *BarValidator) IsValid(b Bar) bool {
	return len(v.Validate(b)) == 0
}

// ----- PriceAdjuster -----

// SplitEvent represents a stock split or dividend adjustment.
type SplitEvent struct {
	Symbol    string
	Date      time.Time
	Ratio     float64 // post/pre ratio, e.g. 4.0 for 4:1 split
	EventType string  // "split", "dividend"
}

// PriceAdjuster applies split and dividend adjustments to historical bars.
type PriceAdjuster struct {
	mu     sync.RWMutex
	events map[string][]SplitEvent // symbol -> sorted events
}

// NewPriceAdjuster creates a PriceAdjuster.
func NewPriceAdjuster() *PriceAdjuster {
	return &PriceAdjuster{
		events: make(map[string][]SplitEvent),
	}
}

// AddEvent registers a split or dividend event.
func (pa *PriceAdjuster) AddEvent(evt SplitEvent) {
	pa.mu.Lock()
	defer pa.mu.Unlock()
	pa.events[evt.Symbol] = append(pa.events[evt.Symbol], evt)
}

// AdjustedClose returns the adjusted close price for a bar given all events
// that occurred after the bar's timestamp.
func (pa *PriceAdjuster) AdjustedClose(b Bar) float64 {
	pa.mu.RLock()
	evts := pa.events[b.Symbol]
	pa.mu.RUnlock()

	adjFactor := 1.0
	for _, evt := range evts {
		if evt.Date.After(b.Timestamp) {
			if evt.EventType == "split" && evt.Ratio != 0 {
				adjFactor /= evt.Ratio
			}
		}
	}
	return b.Close * adjFactor
}

// AdjustBar returns a copy of the bar with split-adjusted OHLCV.
func (pa *PriceAdjuster) AdjustBar(b Bar) Bar {
	pa.mu.RLock()
	evts := pa.events[b.Symbol]
	pa.mu.RUnlock()

	adjFactor := 1.0
	volFactor := 1.0
	for _, evt := range evts {
		if evt.Date.After(b.Timestamp) && evt.EventType == "split" && evt.Ratio != 0 {
			adjFactor /= evt.Ratio
			volFactor *= evt.Ratio
		}
	}

	if adjFactor == 1.0 {
		return b
	}

	return Bar{
		Symbol:    b.Symbol,
		Timestamp: b.Timestamp,
		Open:      b.Open * adjFactor,
		High:      b.High * adjFactor,
		Low:       b.Low * adjFactor,
		Close:     b.Close * adjFactor,
		Volume:    b.Volume * volFactor,
		Source:    b.Source,
		IsPartial: b.IsPartial,
	}
}
