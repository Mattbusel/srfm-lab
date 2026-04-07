package aggregator

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/srfm-lab/risk-aggregator/model"
)

// optionsAPIPositionsResponse is the JSON shape returned by the options
// analytics service at GET /positions/greeks.
type optionsAPIPositionsResponse struct {
	AsOf      time.Time            `json:"as_of"`
	NAV       float64              `json:"nav"`
	Positions []optionsAPIPosition `json:"positions"`
}

// optionsAPIPosition mirrors one position row from the options API.
type optionsAPIPosition struct {
	Instrument string  `json:"instrument"`
	Underlying string  `json:"underlying"`
	Delta      float64 `json:"delta"`
	Gamma      float64 `json:"gamma"`
	Vega       float64 `json:"vega"`
	Theta      float64 `json:"theta"`
	Rho        float64 `json:"rho"`
	Vanna      float64 `json:"vanna"`
	Volga      float64 `json:"volga"`
	Notional   float64 `json:"notional"`
	Quantity   float64 `json:"quantity"`
}

// FetchGreeksReport calls the options analytics service to retrieve
// per-position Greeks, then sums them into portfolio-level aggregates.
// Dollar Greeks are computed as the sum of (quantity * notional * greek).
func (a *Aggregator) FetchGreeksReport(ctx context.Context) (model.GreeksReport, error) {
	url := fmt.Sprintf("%s/positions/greeks", a.cfg.OptionsAPIBase)

	resp, err := a.httpClient.R().SetContext(ctx).Get(url)
	if err != nil {
		return model.GreeksReport{}, fmt.Errorf("GET options/greeks: %w", err)
	}
	if resp.IsError() {
		return model.GreeksReport{}, fmt.Errorf("options API returned %d: %s",
			resp.StatusCode(), resp.String())
	}

	var apiResp optionsAPIPositionsResponse
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return model.GreeksReport{}, fmt.Errorf("unmarshal greeks response: %w", err)
	}

	return aggregateGreeks(apiResp), nil
}

// aggregateGreeks converts the raw API payload into a model.GreeksReport,
// computing net and dollar Greeks as portfolio sums.
func aggregateGreeks(apiResp optionsAPIPositionsResponse) model.GreeksReport {
	positions := make([]model.PositionGreeks, 0, len(apiResp.Positions))

	var (
		netDelta float64
		netGamma float64
		netVega  float64
		netTheta float64
		netRho   float64

		dollarDelta float64
		dollarGamma float64
		dollarVega  float64
		dollarTheta float64
	)

	nav := apiResp.NAV
	if nav == 0 {
		nav = 1 // avoid divide-by-zero for dollar Greek normalization
	}

	for _, p := range apiResp.Positions {
		pos := model.PositionGreeks{
			Instrument: p.Instrument,
			Underlying: p.Underlying,
			Delta:      p.Delta,
			Gamma:      p.Gamma,
			Vega:       p.Vega,
			Theta:      p.Theta,
			Rho:        p.Rho,
			Vanna:      p.Vanna,
			Volga:      p.Volga,
			Notional:   p.Notional,
			Quantity:   p.Quantity,
		}
		positions = append(positions, pos)

		// Net Greeks -- plain sum across all positions (already quantity-weighted
		// by the options API, which returns portfolio-level Greeks per position).
		netDelta += p.Delta
		netGamma += p.Gamma
		netVega += p.Vega
		netTheta += p.Theta
		netRho += p.Rho

		// Dollar Greeks: sensitivity in dollars per $1 NAV move.
		// delta dollar = delta * notional * quantity / nav
		scale := p.Quantity * p.Notional / nav
		dollarDelta += p.Delta * scale
		dollarGamma += p.Gamma * scale
		dollarVega += p.Vega * scale
		dollarTheta += p.Theta * scale
	}

	log.Debug().
		Float64("net_delta", netDelta).
		Float64("net_vega", netVega).
		Float64("net_theta", netTheta).
		Int("positions", len(positions)).
		Msg("greeks aggregated")

	return model.GreeksReport{
		AsOf:        apiResp.AsOf,
		Positions:   positions,
		NetDelta:    netDelta,
		NetGamma:    netGamma,
		NetVega:     netVega,
		NetTheta:    netTheta,
		NetRho:      netRho,
		DollarDelta: dollarDelta,
		DollarGamma: dollarGamma,
		DollarVega:  dollarVega,
		DollarTheta: dollarTheta,
		NAV:         apiResp.NAV,
	}
}

// AggregateByUnderlying groups position Greeks by underlying asset and
// returns a map of underlying -> summed GreeksReport. Useful for
// per-underlying exposure views.
func AggregateByUnderlying(report model.GreeksReport) map[string]model.GreeksReport {
	grouped := make(map[string][]model.PositionGreeks)
	for _, p := range report.Positions {
		grouped[p.Underlying] = append(grouped[p.Underlying], p)
	}

	out := make(map[string]model.GreeksReport, len(grouped))
	for underlying, positions := range grouped {
		var (
			netDelta, netGamma, netVega float64
			netTheta, netRho            float64
		)
		for _, p := range positions {
			netDelta += p.Delta
			netGamma += p.Gamma
			netVega += p.Vega
			netTheta += p.Theta
			netRho += p.Rho
		}
		out[underlying] = model.GreeksReport{
			AsOf:      report.AsOf,
			Positions: positions,
			NetDelta:  netDelta,
			NetGamma:  netGamma,
			NetVega:   netVega,
			NetTheta:  netTheta,
			NetRho:    netRho,
			NAV:       report.NAV,
		}
	}
	return out
}
