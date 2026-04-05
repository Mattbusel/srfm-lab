package aggregator

import (
	"log"
	"math"
)

// SymbolSpec defines precision and size constraints for a symbol.
type SymbolSpec struct {
	MinTickSize float64 // minimum price increment
	LotSize     float64 // minimum volume increment
	Precision   int     // decimal places for price
}

// defaultSpecs holds per-symbol normalization specs.
var defaultSpecs = map[string]SymbolSpec{
	"BTC":   {MinTickSize: 0.01, LotSize: 0.00001, Precision: 2},
	"ETH":   {MinTickSize: 0.01, LotSize: 0.0001, Precision: 2},
	"SOL":   {MinTickSize: 0.001, LotSize: 0.001, Precision: 3},
	"BNB":   {MinTickSize: 0.001, LotSize: 0.001, Precision: 3},
	"XRP":   {MinTickSize: 0.0001, LotSize: 0.1, Precision: 4},
	"ADA":   {MinTickSize: 0.0001, LotSize: 0.1, Precision: 4},
	"AVAX":  {MinTickSize: 0.001, LotSize: 0.01, Precision: 3},
	"DOGE":  {MinTickSize: 0.00001, LotSize: 1.0, Precision: 5},
	"MATIC": {MinTickSize: 0.0001, LotSize: 0.1, Precision: 4},
	"DOT":   {MinTickSize: 0.001, LotSize: 0.01, Precision: 3},
	"LINK":  {MinTickSize: 0.001, LotSize: 0.01, Precision: 3},
	"UNI":   {MinTickSize: 0.001, LotSize: 0.01, Precision: 3},
	"ATOM":  {MinTickSize: 0.001, LotSize: 0.01, Precision: 3},
	"LTC":   {MinTickSize: 0.01, LotSize: 0.001, Precision: 2},
	"BCH":   {MinTickSize: 0.01, LotSize: 0.001, Precision: 2},
	"ALGO":  {MinTickSize: 0.0001, LotSize: 0.1, Precision: 4},
	"XLM":   {MinTickSize: 0.00001, LotSize: 1.0, Precision: 5},
	"VET":   {MinTickSize: 0.000001, LotSize: 10.0, Precision: 6},
	"FIL":   {MinTickSize: 0.001, LotSize: 0.01, Precision: 3},
	"AAVE":  {MinTickSize: 0.01, LotSize: 0.001, Precision: 2},
}

// Normalizer applies price/volume normalization rules.
type Normalizer struct {
	specs map[string]SymbolSpec
}

// NewNormalizer creates a Normalizer with default specs.
func NewNormalizer() *Normalizer {
	return &Normalizer{specs: defaultSpecs}
}

// ValidateTick returns false if the tick contains impossible values.
func (n *Normalizer) ValidateTick(tick RawTick) bool {
	if tick.Symbol == "" {
		return false
	}
	if tick.Timestamp.IsZero() {
		return false
	}
	if tick.IsBar {
		if tick.High < tick.Low {
			log.Printf("[norm] reject %s: high(%.8f) < low(%.8f)", tick.Symbol, tick.High, tick.Low)
			return false
		}
		if tick.Close < tick.Low || tick.Close > tick.High {
			log.Printf("[norm] reject %s: close(%.8f) outside [%.8f, %.8f]", tick.Symbol, tick.Close, tick.Low, tick.High)
			return false
		}
		if tick.Open < tick.Low || tick.Open > tick.High {
			// Open occasionally slightly outside due to exchange precision; clamp rather than reject
			// but log it
			log.Printf("[norm] warn %s: open(%.8f) outside [%.8f, %.8f], clamping", tick.Symbol, tick.Open, tick.Low, tick.High)
		}
	}
	if tick.Volume < 0 {
		log.Printf("[norm] reject %s: negative volume(%.8f)", tick.Symbol, tick.Volume)
		return false
	}
	if tick.Close <= 0 {
		log.Printf("[norm] reject %s: non-positive close(%.8f)", tick.Symbol, tick.Close)
		return false
	}
	return true
}

// NormalizeTick rounds prices to the symbol's tick size and precision.
func (n *Normalizer) NormalizeTick(tick RawTick) RawTick {
	spec, ok := n.specs[tick.Symbol]
	if !ok {
		return tick // no spec, pass through
	}

	if tick.IsBar {
		tick.Open = roundToTick(tick.Open, spec.MinTickSize)
		tick.High = roundToTick(tick.High, spec.MinTickSize)
		tick.Low = roundToTick(tick.Low, spec.MinTickSize)
	}
	tick.Close = roundToTick(tick.Close, spec.MinTickSize)
	tick.Volume = roundToLot(tick.Volume, spec.LotSize)

	// Clamp open within [low, high] if out of range after rounding
	if tick.IsBar {
		if tick.Open < tick.Low {
			tick.Open = tick.Low
		}
		if tick.Open > tick.High {
			tick.Open = tick.High
		}
	}

	return tick
}

func roundToTick(price, tick float64) float64 {
	if tick <= 0 {
		return price
	}
	return math.Round(price/tick) * tick
}

func roundToLot(volume, lot float64) float64 {
	if lot <= 0 {
		return volume
	}
	return math.Round(volume/lot) * lot
}
