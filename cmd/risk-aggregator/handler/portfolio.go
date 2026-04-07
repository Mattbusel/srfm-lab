// Package handler contains the HTTP handler implementations for the
// risk-aggregator service.
package handler

import (
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"golang.org/x/sync/errgroup"

	"github.com/srfm-lab/risk-aggregator/aggregator"
	"github.com/srfm-lab/risk-aggregator/model"
)

// PortfolioHandler handles requests for the consolidated portfolio risk view.
type PortfolioHandler struct {
	agg *aggregator.Aggregator
}

// NewPortfolioHandler constructs a PortfolioHandler.
func NewPortfolioHandler(agg *aggregator.Aggregator) *PortfolioHandler {
	return &PortfolioHandler{agg: agg}
}

// GetPortfolioRisk handles GET /portfolio/risk.
// It fans out to the VaR API, options analytics, and live-trader positions
// concurrently, then assembles a consolidated PortfolioRiskReport.
//
// Query parameters:
//
//	confidence  float64  default 0.95 -- VaR confidence level
//	horizon     int      default 1    -- VaR horizon in days
func (h *PortfolioHandler) GetPortfolioRisk(c *gin.Context) {
	confidence := queryFloat(c, "confidence", 0.95)
	horizon := queryInt(c, "horizon", 1)

	ctx := c.Request.Context()

	var (
		varConsensus model.VaRConsensus
		greeks       model.GreeksReport
		limits       model.LimitsReport
		corrMatrix   model.CorrelationMatrix
	)

	// Fan out -- all four calls can proceed concurrently.
	g, gctx := errgroup.WithContext(ctx)

	g.Go(func() error {
		v, err := h.agg.FetchVaRConsensus(gctx, confidence, horizon)
		if err != nil {
			log.Error().Err(err).Msg("VaR consensus fetch failed")
			return err
		}
		varConsensus = v
		return nil
	})

	g.Go(func() error {
		gr, err := h.agg.FetchGreeksReport(gctx)
		if err != nil {
			log.Error().Err(err).Msg("Greeks fetch failed")
			return err
		}
		greeks = gr
		return nil
	})

	g.Go(func() error {
		lim, err := h.agg.FetchLimitsReport(gctx)
		if err != nil {
			log.Warn().Err(err).Msg("limits fetch failed, using empty report")
			// Non-fatal -- return empty limits rather than failing the whole request.
			limits = model.LimitsReport{AsOf: time.Now().UTC()}
			return nil
		}
		limits = lim
		return nil
	})

	g.Go(func() error {
		cm, err := h.agg.GetCorrelationMatrix(gctx)
		if err != nil {
			log.Warn().Err(err).Msg("correlation fetch failed, using empty matrix")
			corrMatrix = model.CorrelationMatrix{AsOf: time.Now().UTC(), WindowDays: 30}
			return nil
		}
		corrMatrix = cm
		return nil
	})

	if err := g.Wait(); err != nil {
		// Only VaR and Greeks are hard dependencies.
		c.JSON(http.StatusBadGateway, gin.H{
			"error":   "upstream service error",
			"detail":  err.Error(),
		})
		return
	}

	// Determine NAV from Greeks report (the options API is the authoritative source).
	nav := greeks.NAV

	report := model.PortfolioRiskReport{
		AsOf:        time.Now().UTC(),
		NAV:         nav,
		VaR:         varConsensus,
		Greeks:      greeks,
		Limits:      limits,
		Correlation: corrMatrix,
	}

	log.Info().
		Float64("nav", nav).
		Float64("var_95_1d", varConsensus.Consensus.VaRAbsolute).
		Float64("net_delta", greeks.NetDelta).
		Int("breach_count", limits.BreachCount).
		Msg("portfolio risk assembled")

	c.JSON(http.StatusOK, report)
}

// ---------------------------------------------------------------------------
// Helper utilities shared across handlers
// ---------------------------------------------------------------------------

// queryFloat parses a float64 query parameter, returning def if absent or invalid.
func queryFloat(c *gin.Context, key string, def float64) float64 {
	raw := c.Query(key)
	if raw == "" {
		return def
	}
	var v float64
	if _, err := fmt.Sscanf(raw, "%f", &v); err != nil {
		return def
	}
	return v
}

// queryInt parses an int query parameter, returning def if absent or invalid.
func queryInt(c *gin.Context, key string, def int) int {
	raw := c.Query(key)
	if raw == "" {
		return def
	}
	var v int
	if _, err := fmt.Sscanf(raw, "%d", &v); err != nil {
		return def
	}
	return v
}

// queryString parses a string query parameter, returning def if absent.
func queryString(c *gin.Context, key, def string) string {
	if v := c.Query(key); v != "" {
		return v
	}
	return def
}
