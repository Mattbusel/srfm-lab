package handler

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"

	"github.com/srfm-lab/risk-aggregator/aggregator"
	"github.com/srfm-lab/risk-aggregator/model"
)

// LimitsHandler handles requests related to risk limit monitoring.
type LimitsHandler struct {
	agg *aggregator.Aggregator
}

// NewLimitsHandler constructs a LimitsHandler.
func NewLimitsHandler(agg *aggregator.Aggregator) *LimitsHandler {
	return &LimitsHandler{agg: agg}
}

// GetBreaches handles GET /limits/breaches.
// Returns only the limits that are currently breached or in warning state.
// Query params:
//
//	severity  string  filter to "warning", "critical", or "breached" (default: all)
//	scope     string  filter to "portfolio", "strategy:<id>", or "instrument:<id>"
func (h *LimitsHandler) GetBreaches(c *gin.Context) {
	ctx := c.Request.Context()

	severityFilter := queryString(c, "severity", "")
	scopeFilter := queryString(c, "scope", "")

	report, err := h.agg.FetchLimitsReport(ctx)
	if err != nil {
		log.Error().Err(err).Msg("limits: fetch failed")
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to fetch limits", "detail": err.Error()})
		return
	}

	// Apply optional filters.
	filtered := filterBreaches(report.Breaches, severityFilter, scopeFilter)

	out := model.LimitsReport{
		AsOf:          report.AsOf,
		Breaches:      filtered,
		TotalLimits:   report.TotalLimits,
		BreachCount:   countBySeverity(filtered, model.SeverityBreached),
		WarningCount:  countBySeverity(filtered, model.SeverityWarning),
		CriticalCount: countBySeverity(filtered, model.SeverityCritical),
	}

	log.Info().
		Int("total_limits", report.TotalLimits).
		Int("breaches", out.BreachCount).
		Int("warnings", out.WarningCount).
		Msg("limits report returned")

	c.JSON(http.StatusOK, out)
}

// GetAllLimits handles GET /limits/all -- returns every configured limit
// and its current utilization, including those not in breach.
func (h *LimitsHandler) GetAllLimits(c *gin.Context) {
	ctx := c.Request.Context()

	all, err := h.agg.FetchAllLimits(ctx)
	if err != nil {
		log.Error().Err(err).Msg("limits: fetch all failed")
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"limits": all,
		"count":  len(all),
	})
}

// ---------------------------------------------------------------------------
// Filter helpers
// ---------------------------------------------------------------------------

// filterBreaches returns only the breaches that match the given severity and
// scope filters. Empty filter strings match everything.
func filterBreaches(breaches []model.LimitBreach, severityFilter, scopeFilter string) []model.LimitBreach {
	if severityFilter == "" && scopeFilter == "" {
		return breaches
	}

	out := make([]model.LimitBreach, 0, len(breaches))
	for _, b := range breaches {
		if severityFilter != "" && string(b.Severity) != severityFilter {
			continue
		}
		if scopeFilter != "" && b.Limit.Scope != scopeFilter {
			continue
		}
		out = append(out, b)
	}
	return out
}

// countBySeverity counts breaches matching the given severity level.
func countBySeverity(breaches []model.LimitBreach, sev model.LimitSeverity) int {
	count := 0
	for _, b := range breaches {
		if b.Severity == sev {
			count++
		}
	}
	return count
}
