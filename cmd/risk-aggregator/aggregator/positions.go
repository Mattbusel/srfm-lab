package aggregator

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/srfm-lab/risk-aggregator/model"
)

// Position represents a single live position from the trader API.
type Position struct {
	Instrument      string  `json:"instrument"`
	Underlying      string  `json:"underlying"`
	AssetClass      string  `json:"asset_class"` // equity, etf, future, option
	Quantity        float64 `json:"quantity"`    // signed (negative = short)
	Price           float64 `json:"price"`       // current mark price
	Multiplier      float64 `json:"multiplier"`  // contract multiplier (1 for stocks)
	UnderlyingPrice float64 `json:"underlying_price"`
	ImpliedVol      float64 `json:"implied_vol"`
	Delta           float64 `json:"delta"`
	Gamma           float64 `json:"gamma"`
	Vega            float64 `json:"vega"`
	Theta           float64 `json:"theta"`
}

// traderPositionsResponse mirrors the JSON from GET /positions on the trader API.
type traderPositionsResponse struct {
	AsOf      time.Time  `json:"as_of"`
	NAV       float64    `json:"nav"`
	Positions []Position `json:"positions"`
}

// FetchPositions retrieves the current live positions from the trader API.
// Returns the position list and current NAV.
func (a *Aggregator) FetchPositions(ctx context.Context) ([]Position, float64, error) {
	url := fmt.Sprintf("%s/positions", a.cfg.TraderAPIBase)

	resp, err := a.httpClient.R().SetContext(ctx).Get(url)
	if err != nil {
		return nil, 0, fmt.Errorf("GET trader/positions: %w", err)
	}
	if resp.IsError() {
		return nil, 0, fmt.Errorf("trader API returned %d: %s", resp.StatusCode(), resp.String())
	}

	var payload traderPositionsResponse
	if err := json.Unmarshal(resp.Body(), &payload); err != nil {
		return nil, 0, fmt.Errorf("unmarshal positions: %w", err)
	}

	log.Debug().Int("positions", len(payload.Positions)).Float64("nav", payload.NAV).
		Msg("positions fetched")

	return payload.Positions, payload.NAV, nil
}

// ---------------------------------------------------------------------------
// Limits fetching
// ---------------------------------------------------------------------------

// limitsAPIResponse mirrors GET /limits from the risk API.
type limitsAPIResponse struct {
	AsOf        time.Time       `json:"as_of"`
	TotalLimits int             `json:"total_limits"`
	Breaches    []limitBreachAPI `json:"breaches"`
}

type limitBreachAPI struct {
	Name           string  `json:"name"`
	Description    string  `json:"description"`
	Metric         string  `json:"metric"`
	Scope          string  `json:"scope"`
	HardLimit      float64 `json:"hard_limit"`
	SoftLimit      float64 `json:"soft_limit"`
	CurrentValue   float64 `json:"current_value"`
	UtilizationPct float64 `json:"utilization_pct"`
	Severity       string  `json:"severity"`
	Note           string  `json:"note"`
}

// FetchLimitsReport calls the risk API and returns only active breaches.
func (a *Aggregator) FetchLimitsReport(ctx context.Context) (model.LimitsReport, error) {
	url := fmt.Sprintf("%s/limits/breaches", a.cfg.RiskAPIBase)

	resp, err := a.httpClient.R().SetContext(ctx).Get(url)
	if err != nil {
		return model.LimitsReport{}, fmt.Errorf("GET limits/breaches: %w", err)
	}
	if resp.IsError() {
		return model.LimitsReport{}, fmt.Errorf("risk API limits returned %d", resp.StatusCode())
	}

	var apiResp limitsAPIResponse
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return model.LimitsReport{}, fmt.Errorf("unmarshal limits: %w", err)
	}

	breaches := make([]model.LimitBreach, 0, len(apiResp.Breaches))
	warningCount, criticalCount, breachCount := 0, 0, 0

	for _, b := range apiResp.Breaches {
		sev := model.LimitSeverity(b.Severity)
		breach := model.LimitBreach{
			Limit: model.RiskLimit{
				Name:        b.Name,
				Description: b.Description,
				Metric:      b.Metric,
				Scope:       b.Scope,
				HardLimit:   b.HardLimit,
				SoftLimit:   b.SoftLimit,
			},
			CurrentValue:   b.CurrentValue,
			UtilizationPct: b.UtilizationPct,
			Severity:       sev,
			DetectedAt:     apiResp.AsOf,
			Note:           b.Note,
		}
		breaches = append(breaches, breach)

		switch sev {
		case model.SeverityWarning:
			warningCount++
		case model.SeverityCritical:
			criticalCount++
		case model.SeverityBreached:
			breachCount++
		}
	}

	return model.LimitsReport{
		AsOf:          apiResp.AsOf,
		Breaches:      breaches,
		TotalLimits:   apiResp.TotalLimits,
		BreachCount:   breachCount,
		WarningCount:  warningCount,
		CriticalCount: criticalCount,
	}, nil
}

// FetchAllLimits retrieves every configured limit regardless of breach status.
func (a *Aggregator) FetchAllLimits(ctx context.Context) ([]model.RiskLimit, error) {
	url := fmt.Sprintf("%s/limits", a.cfg.RiskAPIBase)

	resp, err := a.httpClient.R().SetContext(ctx).Get(url)
	if err != nil {
		return nil, fmt.Errorf("GET limits: %w", err)
	}
	if resp.IsError() {
		return nil, fmt.Errorf("risk API returned %d", resp.StatusCode())
	}

	var payload struct {
		Limits []struct {
			Name        string  `json:"name"`
			Description string  `json:"description"`
			Metric      string  `json:"metric"`
			Scope       string  `json:"scope"`
			HardLimit   float64 `json:"hard_limit"`
			SoftLimit   float64 `json:"soft_limit"`
		} `json:"limits"`
	}
	if err := json.Unmarshal(resp.Body(), &payload); err != nil {
		return nil, fmt.Errorf("unmarshal all limits: %w", err)
	}

	out := make([]model.RiskLimit, 0, len(payload.Limits))
	for _, l := range payload.Limits {
		out = append(out, model.RiskLimit{
			Name:        l.Name,
			Description: l.Description,
			Metric:      l.Metric,
			Scope:       l.Scope,
			HardLimit:   l.HardLimit,
			SoftLimit:   l.SoftLimit,
		})
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Attribution data fetching
// ---------------------------------------------------------------------------

// SegmentData is one sector/segment row returned by the attribution API.
type SegmentData struct {
	Segment         string  `json:"segment"`
	PortfolioWeight float64 `json:"portfolio_weight"`
	BenchmarkWeight float64 `json:"benchmark_weight"`
	PortfolioReturn float64 `json:"portfolio_return"`
	BenchmarkReturn float64 `json:"benchmark_return"`
}

// AttributionData is the raw payload from the risk API attribution endpoint.
type AttributionData struct {
	Date            time.Time          `json:"date"`
	TotalReturn     float64            `json:"total_return"`
	BenchmarkReturn float64            `json:"benchmark_return"`
	Segments        []SegmentData      `json:"segments"`
	FactorExposures map[string]float64 `json:"factor_exposures"`
	FactorReturns   map[string]float64 `json:"factor_returns"`
}

// FetchAttributionData retrieves the raw attribution data for one date
// from the risk API.
func (a *Aggregator) FetchAttributionData(ctx context.Context, date time.Time, benchmark string) (AttributionData, error) {
	dateStr := date.Format("2006-01-02")
	url := fmt.Sprintf("%s/attribution?date=%s&benchmark=%s", a.cfg.RiskAPIBase, dateStr, benchmark)

	resp, err := a.httpClient.R().SetContext(ctx).Get(url)
	if err != nil {
		return AttributionData{}, fmt.Errorf("GET attribution: %w", err)
	}
	if resp.IsError() {
		return AttributionData{}, fmt.Errorf("risk API attribution returned %d", resp.StatusCode())
	}

	var data AttributionData
	if err := json.Unmarshal(resp.Body(), &data); err != nil {
		return AttributionData{}, fmt.Errorf("unmarshal attribution: %w", err)
	}

	return data, nil
}
