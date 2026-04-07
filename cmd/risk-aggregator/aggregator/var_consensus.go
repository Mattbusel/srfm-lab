// Package aggregator contains the core logic for fetching and aggregating
// risk metrics from upstream microservices.
package aggregator

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/srfm-lab/risk-aggregator/model"
)

// varAPIResponse mirrors the JSON returned by the Python risk API
// GET /var?confidence=0.95&horizon=1
type varAPIResponse struct {
	Method          string    `json:"method"`
	ConfidenceLevel float64   `json:"confidence_level"`
	HorizonDays     int       `json:"horizon_days"`
	VaRAbsolute     float64   `json:"var_absolute"`
	VaRPercent      float64   `json:"var_percent"`
	CVaR            float64   `json:"cvar_absolute"`
	CVaRPercent     float64   `json:"cvar_percent"`
	ComputedAt      time.Time `json:"computed_at"`
}

// toModel converts a varAPIResponse to model.VaRResult.
func (r varAPIResponse) toModel(method model.VaRMethod) model.VaRResult {
	return model.VaRResult{
		Method:          method,
		ConfidenceLevel: r.ConfidenceLevel,
		HorizonDays:     r.HorizonDays,
		VaRAbsolute:     r.VaRAbsolute,
		VaRPercent:      r.VaRPercent,
		CVaR:            r.CVaR,
		CVaRPercent:     r.CVaRPercent,
		ComputedAt:      r.ComputedAt,
	}
}

// defaultVaRWeights are the consensus blending weights:
// 40% parametric, 30% historical, 30% Monte Carlo.
var defaultVaRWeights = struct{ param, hist, mc float64 }{0.40, 0.30, 0.30}

// divergenceThreshold is the maximum allowed relative difference between
// any two VaR estimates before the consensus is flagged as divergent.
const divergenceThreshold = 0.25 // 25%

// FetchVaRConsensus retrieves parametric, historical, and MC VaR from the
// Python risk API and blends them into a weighted consensus estimate.
// It also checks for divergence and annotates the result accordingly.
func (a *Aggregator) FetchVaRConsensus(ctx context.Context, confidence float64, horizon int) (model.VaRConsensus, error) {
	// Fire three requests concurrently.
	type result struct {
		method model.VaRMethod
		resp   varAPIResponse
		err    error
	}

	methods := []struct {
		method   model.VaRMethod
		apiParam string
	}{
		{model.VaRMethodParametric, "parametric"},
		{model.VaRMethodHistorical, "historical"},
		{model.VaRMethodMonteCarlo, "monte_carlo"},
	}

	results := make([]result, len(methods))
	ch := make(chan result, len(methods))

	for _, m := range methods {
		m := m // capture loop var
		go func() {
			url := fmt.Sprintf("%s/var?method=%s&confidence=%.4f&horizon=%d",
				a.cfg.RiskAPIBase, m.apiParam, confidence, horizon)

			resp, err := a.httpClient.R().
				SetContext(ctx).
				Get(url)
			if err != nil {
				ch <- result{method: m.method, err: fmt.Errorf("GET %s: %w", url, err)}
				return
			}
			if resp.IsError() {
				ch <- result{method: m.method, err: fmt.Errorf("risk API %s returned %d", m.apiParam, resp.StatusCode())}
				return
			}

			var v varAPIResponse
			if err := json.Unmarshal(resp.Body(), &v); err != nil {
				ch <- result{method: m.method, err: fmt.Errorf("unmarshal %s VaR: %w", m.apiParam, err)}
				return
			}
			ch <- result{method: m.method, resp: v}
		}()
	}

	// Collect results, allow partial success -- fall back to available methods.
	errs := make([]error, 0)
	varMap := make(map[model.VaRMethod]varAPIResponse)
	for range methods {
		r := <-ch
		if r.err != nil {
			log.Warn().Err(r.err).Str("method", string(r.method)).Msg("VaR fetch failed")
			errs = append(errs, r.err)
		} else {
			varMap[r.method] = r.resp
		}
	}

	if len(varMap) == 0 {
		return model.VaRConsensus{}, fmt.Errorf("all VaR methods failed: %v", errs)
	}

	// Use available estimates; fall back to whichever succeeded for missing ones.
	fallback := pickFallback(varMap)
	param := getOrFallback(varMap, model.VaRMethodParametric, fallback)
	hist := getOrFallback(varMap, model.VaRMethodHistorical, fallback)
	mc := getOrFallback(varMap, model.VaRMethodMonteCarlo, fallback)

	w := defaultVaRWeights
	consensus := blendVaR(param, hist, mc, w.param, w.hist, w.mc, confidence, horizon)

	divergent, note := detectDivergence(param, hist, mc)

	return model.VaRConsensus{
		Parametric:     param.toModel(model.VaRMethodParametric),
		Historical:     hist.toModel(model.VaRMethodHistorical),
		MonteCarlo:     mc.toModel(model.VaRMethodMonteCarlo),
		Consensus:      consensus,
		WeightParam:    w.param,
		WeightHist:     w.hist,
		WeightMC:       w.mc,
		Divergent:      divergent,
		DivergenceNote: note,
	}, nil
}

// blendVaR computes a weighted average VaR from three method estimates.
func blendVaR(param, hist, mc varAPIResponse, wp, wh, wm float64,
	confidence float64, horizon int) model.VaRResult {

	totalW := wp + wh + wm
	if totalW == 0 {
		totalW = 1
	}
	wp /= totalW
	wh /= totalW
	wm /= totalW

	varAbs := wp*param.VaRAbsolute + wh*hist.VaRAbsolute + wm*mc.VaRAbsolute
	varPct := wp*param.VaRPercent + wh*hist.VaRPercent + wm*mc.VaRPercent
	cvarAbs := wp*param.CVaR + wh*hist.CVaR + wm*mc.CVaR
	cvarPct := wp*param.CVaRPercent + wh*hist.CVaRPercent + wm*mc.CVaRPercent

	return model.VaRResult{
		Method:          model.VaRMethodConsensus,
		ConfidenceLevel: confidence,
		HorizonDays:     horizon,
		VaRAbsolute:     varAbs,
		VaRPercent:      varPct,
		CVaR:            cvarAbs,
		CVaRPercent:     cvarPct,
		ComputedAt:      time.Now().UTC(),
	}
}

// detectDivergence checks all pairwise relative differences of VaR estimates.
// Returns true and a description if any pair diverges beyond the threshold.
func detectDivergence(param, hist, mc varAPIResponse) (bool, string) {
	vals := []struct {
		name  string
		value float64
	}{
		{"parametric", param.VaRAbsolute},
		{"historical", hist.VaRAbsolute},
		{"monte_carlo", mc.VaRAbsolute},
	}

	for i := 0; i < len(vals); i++ {
		for j := i + 1; j < len(vals); j++ {
			a, b := vals[i].value, vals[j].value
			if a == 0 && b == 0 {
				continue
			}
			mid := (math.Abs(a) + math.Abs(b)) / 2.0
			if mid == 0 {
				continue
			}
			relDiff := math.Abs(a-b) / mid
			if relDiff > divergenceThreshold {
				return true, fmt.Sprintf(
					"%s vs %s differ by %.1f%% (threshold %.0f%%)",
					vals[i].name, vals[j].name, relDiff*100, divergenceThreshold*100,
				)
			}
		}
	}
	return false, ""
}

// pickFallback returns the first available VaR response from the map.
func pickFallback(m map[model.VaRMethod]varAPIResponse) varAPIResponse {
	for _, v := range m {
		return v
	}
	return varAPIResponse{}
}

// getOrFallback returns the VaR response for the requested method, or the
// fallback if it is not present (e.g. that method's request failed).
func getOrFallback(m map[model.VaRMethod]varAPIResponse, method model.VaRMethod, fallback varAPIResponse) varAPIResponse {
	if v, ok := m[method]; ok {
		return v
	}
	return fallback
}

