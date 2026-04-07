package handler

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"

	"github.com/srfm-lab/risk-aggregator/aggregator"
	"github.com/srfm-lab/risk-aggregator/model"
)

// StressHandler handles requests for running stress test scenarios.
type StressHandler struct {
	agg *aggregator.Aggregator
}

// NewStressHandler constructs a StressHandler.
func NewStressHandler(agg *aggregator.Aggregator) *StressHandler {
	return &StressHandler{agg: agg}
}

// stressRunRequest is the body for POST /stress/run.
type stressRunRequest struct {
	// Scenarios is an optional list of scenario names to run.
	// If empty, all built-in scenarios are run.
	Scenarios []string `json:"scenarios"`
	// CustomShocks allows ad-hoc factor shocks not in the built-in set.
	CustomShocks []model.ScenarioShock `json:"custom_shocks,omitempty"`
}

// RunStress handles POST /stress/run.
// It accepts an optional list of scenario names and runs each against the
// current portfolio positions, applying pre-defined historical factor shocks.
func (h *StressHandler) RunStress(c *gin.Context) {
	var req stressRunRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		// Allow empty body -- default to all scenarios.
		req = stressRunRequest{}
	}

	ctx := c.Request.Context()

	// Resolve which scenarios to run.
	scenarios := selectScenarios(req.Scenarios)
	scenarios = append(scenarios, req.CustomShocks...)

	if len(scenarios) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "no scenarios selected"})
		return
	}

	// Fetch current positions from trader API.
	positions, nav, err := h.agg.FetchPositions(ctx)
	if err != nil {
		log.Error().Err(err).Msg("stress: failed to fetch positions")
		c.JSON(http.StatusBadGateway, gin.H{"error": "failed to fetch positions", "detail": err.Error()})
		return
	}

	stressResults := make([]model.StressResult, 0, len(scenarios))
	for _, sc := range scenarios {
		result := runScenario(sc, positions, nav)
		stressResults = append(stressResults, result)
		log.Info().
			Str("scenario", sc.Name).
			Float64("total_pnl", result.TotalPnL).
			Float64("total_pnl_pct", result.TotalPnLPct).
			Msg("stress scenario complete")
	}

	suite := model.StressSuiteResult{
		RunAt:   time.Now().UTC(),
		Results: stressResults,
		NAV:     nav,
	}

	c.JSON(http.StatusOK, suite)
}

// ListScenarios handles GET /stress/scenarios and returns all built-in scenarios.
func (h *StressHandler) ListScenarios(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"scenarios": builtinScenarios()})
}

// ---------------------------------------------------------------------------
// Scenario definitions
// ---------------------------------------------------------------------------

// builtinScenarios returns the three hardcoded historical stress scenarios.
func builtinScenarios() []model.ScenarioShock {
	return []model.ScenarioShock{
		{
			Name:        "GFC_2008",
			Description: "Global Financial Crisis peak stress (Sep-Nov 2008)",
			StartDate:   "2008-09-01",
			EndDate:     "2008-11-30",
			VolMultiplier: 3.5,
			Shocks: map[string]float64{
				"SPX":    -0.37,
				"NDX":    -0.41,
				"RTY":    -0.40,
				"EEM":    -0.55,
				"EURUSD": -0.18,
				"USDJPY": 0.14,  // yen strengthened
				"GLD":    0.12,  // gold as safe haven
				"TLT":    0.22,  // bonds rallied
				"VIX":    5.80,  // multiplier (x current VIX)
				"OIL":    -0.68,
				"HYG":    -0.28,
				"LQD":    -0.12,
			},
		},
		{
			Name:        "COVID_2020",
			Description: "COVID-19 crash (Feb-Mar 2020)",
			StartDate:   "2020-02-19",
			EndDate:     "2020-03-23",
			VolMultiplier: 4.2,
			Shocks: map[string]float64{
				"SPX":    -0.34,
				"NDX":    -0.29,
				"RTY":    -0.43,
				"EEM":    -0.31,
				"EURUSD": 0.05,
				"USDJPY": 0.03,
				"GLD":    -0.03, // gold initially sold off
				"TLT":    0.18,
				"VIX":    7.10,
				"OIL":    -0.65,
				"HYG":    -0.21,
				"LQD":    -0.08,
			},
		},
		{
			Name:        "RATES_2022",
			Description: "Fed rate hike cycle / Ukraine shock (Jan-Oct 2022)",
			StartDate:   "2022-01-03",
			EndDate:     "2022-10-14",
			VolMultiplier: 2.0,
			Shocks: map[string]float64{
				"SPX":    -0.25,
				"NDX":    -0.35,
				"RTY":    -0.26,
				"EEM":    -0.28,
				"EURUSD": -0.14,
				"USDJPY": -0.20, // yen weakened dramatically
				"GLD":    -0.08,
				"TLT":    -0.32, // bonds crashed on rate hikes
				"VIX":    2.50,
				"OIL":    0.30,  // energy spike from Ukraine
				"HYG":    -0.16,
				"LQD":    -0.23,
			},
		},
	}
}

// selectScenarios filters the built-in scenario list by name.
// If names is empty, all built-in scenarios are returned.
func selectScenarios(names []string) []model.ScenarioShock {
	all := builtinScenarios()
	if len(names) == 0 {
		return all
	}

	wanted := make(map[string]bool, len(names))
	for _, n := range names {
		wanted[n] = true
	}

	out := make([]model.ScenarioShock, 0, len(names))
	for _, s := range all {
		if wanted[s.Name] {
			out = append(out, s)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Scenario application logic
// ---------------------------------------------------------------------------

// TraderPosition is a simplified position row from the live trader API.
// (Full type lives in aggregator package; this mirrors what FetchPositions returns.)
type TraderPosition = aggregator.Position

// runScenario applies the factor shocks from sc to the list of positions and
// returns a StressResult with per-position and total P&L.
// For equity positions the shock is: pnl = quantity * price * shock[underlying].
// For options positions we apply a first-order Greek approximation:
//
//	pnl = delta * dS + 0.5 * gamma * dS^2 + vega * d_sigma
func runScenario(sc model.ScenarioShock, positions []TraderPosition, nav float64) model.StressResult {
	posResults := make([]model.StressPositionResult, 0, len(positions))
	totalPnL := 0.0
	worstPnL := 0.0
	worstInstrument := ""

	for _, pos := range positions {
		shock, ok := sc.Shocks[pos.Underlying]
		if !ok {
			shock = 0 // no shock for this underlying
		}

		var pnl float64
		baseValue := pos.Quantity * pos.Price

		switch pos.AssetClass {
		case "equity", "etf", "future":
			pnl = baseValue * shock

		case "option":
			dS := pos.UnderlyingPrice * shock
			dSigma := (sc.VolMultiplier - 1.0) * pos.ImpliedVol
			pnl = pos.Quantity * pos.Multiplier * (
				pos.Delta*dS +
					0.5*pos.Gamma*dS*dS +
					pos.Vega*dSigma)

		default:
			pnl = baseValue * shock * 0.5 // generic half-shock for unknown types
		}

		stressValue := baseValue + pnl
		pnlPct := 0.0
		if baseValue != 0 {
			pnlPct = pnl / baseValue
		}

		posResults = append(posResults, model.StressPositionResult{
			Instrument:  pos.Instrument,
			Underlying:  pos.Underlying,
			BaseValue:   baseValue,
			StressValue: stressValue,
			PnL:         pnl,
			PnLPercent:  pnlPct,
		})

		totalPnL += pnl
		if pnl < worstPnL {
			worstPnL = pnl
			worstInstrument = pos.Instrument
		}
	}

	totalPnLPct := 0.0
	if nav != 0 {
		totalPnLPct = totalPnL / nav
	}

	return model.StressResult{
		Scenario:      sc,
		RunAt:         time.Now().UTC(),
		Positions:     posResults,
		TotalPnL:      totalPnL,
		TotalPnLPct:   totalPnLPct,
		NAV:           nav,
		WorstPosition: worstInstrument,
	}
}
