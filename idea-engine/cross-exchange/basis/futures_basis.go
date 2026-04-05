// Package basis computes futures/spot basis signals.
package basis

import (
	"time"
)

// Regime describes the market structure for futures pricing.
type Regime string

const (
	RegimeContango     Regime = "CONTANGO"
	RegimeBackwardation Regime = "BACKWARDATION"
	RegimeNormal       Regime = "NORMAL"

	// Thresholds for regime classification (annualised basis).
	contangoThreshold     = 0.15 // 15% annualised
	backwardationThreshold = 0.0
)

// BasisSignal carries basis analysis for one symbol.
type BasisSignal struct {
	Symbol           string    `json:"symbol"`
	SpotPrice        float64   `json:"spot_price"`
	FuturesPrice     float64   `json:"futures_price"`
	RawBasis         float64   `json:"raw_basis"`         // (fut - spot) / spot
	AnnualizedBasis  float64   `json:"annualized_basis"`  // annualised as fraction
	DaysToExpiry     float64   `json:"days_to_expiry"`
	Regime           Regime    `json:"regime"`
	FundingRate      float64   `json:"funding_rate"`       // 8h funding rate (perpetuals)
	AnnualFunding    float64   `json:"annual_funding"`     // funding * 3 * 365
	ComputedAt       time.Time `json:"computed_at"`
}

// ComputeBasis calculates the annualised futures/spot basis.
// For perpetual futures pass daysToExpiry = 0; funding rate is used instead.
func ComputeBasis(symbol string, spotPrice, futuresPrice, daysToExpiry, fundingRate float64) BasisSignal {
	now := time.Now().UTC()
	if spotPrice <= 0 {
		return BasisSignal{Symbol: symbol, ComputedAt: now}
	}

	rawBasis := (futuresPrice - spotPrice) / spotPrice

	var annualizedBasis float64
	if daysToExpiry > 0 {
		annualizedBasis = rawBasis * (365.0 / daysToExpiry)
	} else {
		// Perpetual: use funding rate. Funding is every 8h, 3x per day.
		annualFunding := fundingRate * 3.0 * 365.0
		annualizedBasis = annualFunding
	}

	var regime Regime
	switch {
	case annualizedBasis > contangoThreshold:
		regime = RegimeContango
	case annualizedBasis < backwardationThreshold:
		regime = RegimeBackwardation
	default:
		regime = RegimeNormal
	}

	annualFunding := fundingRate * 3.0 * 365.0

	return BasisSignal{
		Symbol:          symbol,
		SpotPrice:       spotPrice,
		FuturesPrice:    futuresPrice,
		RawBasis:        rawBasis,
		AnnualizedBasis: annualizedBasis,
		DaysToExpiry:    daysToExpiry,
		Regime:          regime,
		FundingRate:     fundingRate,
		AnnualFunding:   annualFunding,
		ComputedAt:      now,
	}
}

// ComputeAllBasis computes basis signals for all symbols given spot prices,
// futures prices, and funding rates.
func ComputeAllBasis(
	spotPrices map[string]float64,
	futuresPrices map[string]float64,
	fundingRates map[string]float64,
) []BasisSignal {
	signals := make([]BasisSignal, 0, len(spotPrices))
	for symbol, spot := range spotPrices {
		futures, ok := futuresPrices[symbol]
		if !ok {
			continue
		}
		funding := fundingRates[symbol] // 0 if not present
		sig := ComputeBasis(symbol, spot, futures, 0, funding)
		signals = append(signals, sig)
	}
	return signals
}
