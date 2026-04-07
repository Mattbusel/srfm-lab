package handler

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"

	"github.com/srfm-lab/risk-aggregator/aggregator"
	"github.com/srfm-lab/risk-aggregator/model"
)

// AttributionHandler handles performance and risk attribution requests.
type AttributionHandler struct {
	agg *aggregator.Aggregator
}

// NewAttributionHandler constructs an AttributionHandler.
func NewAttributionHandler(agg *aggregator.Aggregator) *AttributionHandler {
	return &AttributionHandler{agg: agg}
}

// GetDailyAttribution handles GET /attribution/daily.
// It fetches today's portfolio and benchmark returns, computes Brinson
// allocation/selection/interaction effects by sector, then overlays a
// factor model attribution.
//
// Query parameters:
//
//	date       string  ISO-8601 date, default today
//	benchmark  string  benchmark ticker, default "SPY"
func (h *AttributionHandler) GetDailyAttribution(c *gin.Context) {
	ctx := c.Request.Context()

	dateStr := queryString(c, "date", time.Now().UTC().Format("2006-01-02"))
	benchmark := queryString(c, "benchmark", "SPY")

	date, err := time.Parse("2006-01-02", dateStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid date format, use YYYY-MM-DD"})
		return
	}

	// Fetch raw attribution data from the risk API.
	raw, err := h.agg.FetchAttributionData(ctx, date, benchmark)
	if err != nil {
		log.Error().Err(err).Str("date", dateStr).Msg("attribution: fetch failed")
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to fetch attribution data", "detail": err.Error()})
		return
	}

	report := computeAttribution(raw, date)

	log.Info().
		Str("date", dateStr).
		Float64("total_return", report.TotalReturn).
		Float64("active_return", report.ActiveReturn).
		Float64("factor_return", report.FactorReturn).
		Float64("specific_return", report.SpecificReturn).
		Msg("attribution computed")

	c.JSON(http.StatusOK, report)
}

// ---------------------------------------------------------------------------
// Attribution computation
// ---------------------------------------------------------------------------

// RawAttributionData is the aggregator.AttributionData type aliased here
// for readability.
type RawAttributionData = aggregator.AttributionData

// computeAttribution applies Brinson decomposition and factor model attribution
// to the raw data fetched from the risk API.
func computeAttribution(raw RawAttributionData, date time.Time) model.AttributionReport {
	segments := computeBrinson(raw.Segments)

	// Sum Brinson effects.
	var totalAlloc, totalSel, totalInteract float64
	for _, s := range segments {
		totalAlloc += s.AllocationEffect
		totalSel += s.SelectionEffect
		totalInteract += s.InteractionEffect
	}

	// Factor attribution -- contribution = exposure * factor_return.
	factorContribs := make([]model.FactorContribution, 0, len(raw.FactorExposures))
	totalFactorReturn := 0.0
	for factor, exposure := range raw.FactorExposures {
		factorRet, ok := raw.FactorReturns[factor]
		if !ok {
			continue
		}
		contrib := exposure * factorRet
		totalFactorReturn += contrib
		factorContribs = append(factorContribs, model.FactorContribution{
			Factor:       factor,
			Exposure:     exposure,
			FactorReturn: factorRet,
			Contribution: contrib,
		})
	}

	specificReturn := raw.TotalReturn - totalFactorReturn
	activeReturn := raw.TotalReturn - raw.BenchmarkReturn

	return model.AttributionReport{
		Date:                date,
		TotalReturn:         raw.TotalReturn,
		BenchmarkReturn:     raw.BenchmarkReturn,
		ActiveReturn:        activeReturn,
		BrinsonSegments:     segments,
		TotalAllocation:     totalAlloc,
		TotalSelection:      totalSel,
		TotalInteraction:    totalInteract,
		FactorContributions: factorContribs,
		FactorReturn:        totalFactorReturn,
		SpecificReturn:      specificReturn,
	}
}

// computeBrinson applies the Brinson-Hood-Beebower decomposition for each segment.
// Notation:
//
//	w_p = portfolio weight, w_b = benchmark weight
//	r_p = portfolio segment return, r_b = benchmark segment return
//	R_b = total benchmark return
//
// Effects:
//
//	Allocation  = (w_p - w_b) * (r_b - R_b)
//	Selection   = w_b * (r_p - r_b)
//	Interaction = (w_p - w_b) * (r_p - r_b)
func computeBrinson(segs []aggregator.SegmentData) []model.BrinsonSegment {
	// Compute total benchmark return as the weighted sum of segment benchmark returns.
	totalBenchmarkReturn := 0.0
	for _, s := range segs {
		totalBenchmarkReturn += s.BenchmarkWeight * s.BenchmarkReturn
	}

	out := make([]model.BrinsonSegment, 0, len(segs))
	for _, s := range segs {
		wDiff := s.PortfolioWeight - s.BenchmarkWeight
		rDiff := s.PortfolioReturn - s.BenchmarkReturn

		allocation := wDiff * (s.BenchmarkReturn - totalBenchmarkReturn)
		selection := s.BenchmarkWeight * rDiff
		interaction := wDiff * rDiff
		total := allocation + selection + interaction

		out = append(out, model.BrinsonSegment{
			Segment:           s.Segment,
			PortfolioWeight:   s.PortfolioWeight,
			BenchmarkWeight:   s.BenchmarkWeight,
			PortfolioReturn:   s.PortfolioReturn,
			BenchmarkReturn:   s.BenchmarkReturn,
			AllocationEffect:  allocation,
			SelectionEffect:   selection,
			InteractionEffect: interaction,
			TotalEffect:       total,
		})
	}
	return out
}
