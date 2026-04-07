// Package aggregator contains the core computation engines for the risk-aggregator service.
// This file implements exposure calculations: gross, net, sector, beta-adjusted,
// and dollar-delta for options portfolios.
package aggregator

import (
	"math"
)

// OptionPosition represents a single options contract position used for
// dollar-delta calculations.
type OptionPosition struct {
	Symbol   string  `json:"symbol"`
	Delta    float64 `json:"delta"`    // [-1, 1]
	Notional float64 `json:"notional"` // USD notional of the position
}

// ExposureCalculator provides stateless exposure computations.
// It requires no initialization; a zero value is ready to use.
type ExposureCalculator struct{}

// GrossExposure returns the sum of absolute position values in USD.
// Long 100k + short 50k -> 150k gross exposure.
func (c *ExposureCalculator) GrossExposure(positions map[string]float64) float64 {
	total := 0.0
	for _, v := range positions {
		total += math.Abs(v)
	}
	return total
}

// NetExposure returns the algebraic sum of position values.
// Long 100k + short 50k -> +50k net exposure.
func (c *ExposureCalculator) NetExposure(positions map[string]float64) float64 {
	total := 0.0
	for _, v := range positions {
		total += v
	}
	return total
}

// SectorExposure aggregates position USD values by sector.
// Positions whose symbol is not present in sectorMap are grouped under "unknown".
func (c *ExposureCalculator) SectorExposure(
	positions map[string]float64,
	sectorMap map[string]string,
) map[string]float64 {
	out := make(map[string]float64)
	for sym, val := range positions {
		sector, ok := sectorMap[sym]
		if !ok || sector == "" {
			sector = "unknown"
		}
		out[sector] += val
	}
	return out
}

// BetaAdjustedExposure returns the net exposure multiplied by each position's
// market beta, giving a market-equivalent dollar exposure.
//
//	beta-adjusted = sum_i (value_i * beta_i)
//
// If a symbol has no entry in betas it is treated as beta = 1.0.
func (c *ExposureCalculator) BetaAdjustedExposure(
	positions map[string]float64,
	betas map[string]float64,
) float64 {
	total := 0.0
	for sym, val := range positions {
		beta, ok := betas[sym]
		if !ok {
			beta = 1.0
		}
		total += val * beta
	}
	return total
}

// DollarDelta returns the aggregate dollar-delta for an options portfolio.
//
//	dollar_delta = sum_i (notional_i * delta_i)
//
// A fully hedged book (net delta == 0) returns ~0.
func (c *ExposureCalculator) DollarDelta(options []OptionPosition) float64 {
	total := 0.0
	for _, opt := range options {
		total += opt.Notional * opt.Delta
	}
	return total
}

// Leverage returns the ratio of gross exposure to net asset value (equity).
// Returns 0 when equity is zero to avoid division by zero.
func (c *ExposureCalculator) Leverage(positions map[string]float64, equity float64) float64 {
	if equity == 0 {
		return 0
	}
	return c.GrossExposure(positions) / equity
}

// LongExposure returns the sum of all positive (long) position values.
func (c *ExposureCalculator) LongExposure(positions map[string]float64) float64 {
	total := 0.0
	for _, v := range positions {
		if v > 0 {
			total += v
		}
	}
	return total
}

// ShortExposure returns the absolute sum of all negative (short) position values.
func (c *ExposureCalculator) ShortExposure(positions map[string]float64) float64 {
	total := 0.0
	for _, v := range positions {
		if v < 0 {
			total += math.Abs(v)
		}
	}
	return total
}

// ExposurePosition holds a symbol and its USD position value, used by TopExposures.
type ExposurePosition struct {
	Symbol string  `json:"symbol"`
	Value  float64 `json:"value_usd"`
}

// TopExposures returns up to n symbols sorted by absolute USD value descending.
// Useful for identifying concentration risk.
func (c *ExposureCalculator) TopExposures(positions map[string]float64, n int) []ExposurePosition {
	type symVal struct {
		sym string
		val float64
	}

	items := make([]symVal, 0, len(positions))
	for sym, val := range positions {
		items = append(items, symVal{sym, val})
	}

	// Simple insertion sort -- portfolios are typically small (<500 symbols).
	for i := 1; i < len(items); i++ {
		key := items[i]
		j := i - 1
		for j >= 0 && math.Abs(items[j].val) < math.Abs(key.val) {
			items[j+1] = items[j]
			j--
		}
		items[j+1] = key
	}

	if n > len(items) {
		n = len(items)
	}

	out := make([]ExposurePosition, n)
	for i := 0; i < n; i++ {
		out[i] = ExposurePosition{
			Symbol: items[i].sym,
			Value:  items[i].val,
		}
	}
	return out
}
