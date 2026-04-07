// api_client.go -- HTTP client for fetching live data from SRFM microservices.
// Endpoints:
//   live-trader:8080  -- positions, signals, trades
//   risk-aggregator:8783  -- risk metrics
//   coordination:8781  -- circuit breakers
//
// All requests timeout at 2s. CachedAPIClient wraps with a 5s TTL per endpoint.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"srfm-tui/views"
)

// ── Base endpoints ────────────────────────────────────────────────────────────

const (
	liveTraderBase    = "http://live-trader:8080"
	riskAggBase       = "http://risk-aggregator:8783"
	coordinationBase  = "http://coordination:8781"
	requestTimeout    = 2 * time.Second
	cacheTTL          = 5 * time.Second
)

// ── Wire types ────────────────────────────────────────────────────────────────
// These match the JSON schema returned by each service.

type wirePosition struct {
	Symbol    string  `json:"symbol"`
	Qty       float64 `json:"qty"`
	AvgCost   float64 `json:"avg_cost"`
	Mark      float64 `json:"mark"`
	UnrealPnL float64 `json:"unreal_pnl"`
	PctPnL    float64 `json:"pct_pnl"`
	DailyPnL  float64 `json:"daily_pnl"`
}

type wirePositionsResp struct {
	Positions []wirePosition `json:"positions"`
}

type wireBHMass struct {
	Symbol  string     `json:"symbol"`
	Mass    float64    `json:"mass"`
	Active  bool       `json:"active"`
	History [5]float64 `json:"history"`
}

type wireSignalState struct {
	BHMasses []wireBHMass `json:"bh_masses"`
	Hurst    struct {
		Value   float64    `json:"value"`
		History [5]float64 `json:"history"`
	} `json:"hurst"`
	GARCH struct {
		CurrentForecast float64    `json:"current_forecast"`
		HistoricalMean  float64    `json:"historical_mean"`
		History         [5]float64 `json:"history"`
	} `json:"garch"`
	NavCurvature struct {
		W              float64 `json:"w"`
		X              float64 `json:"x"`
		Y              float64 `json:"y"`
		Z              float64 `json:"z"`
		CurvatureAngle float64 `json:"curvature_angle"`
		Regime         string  `json:"regime"`
	} `json:"nav_curvature"`
	RLQValues []struct {
		Action string  `json:"action"`
		QValue float64 `json:"q_value"`
	} `json:"rl_q_values"`
	Regime     string  `json:"regime"`
	Confidence float64 `json:"confidence"`
	UpdatedAt  string  `json:"updated_at"` // RFC3339
}

type wireRiskMetrics struct {
	VaR95         float64 `json:"var_95"`
	VaRLimit      float64 `json:"var_limit"`
	Drawdown      float64 `json:"drawdown"`
	DrawdownLimit float64 `json:"drawdown_limit"`
	MarginUsed    float64 `json:"margin_used"`
	MarginLimit   float64 `json:"margin_limit"`
	UpdatedAt     string  `json:"updated_at"`
}

type wireTrade struct {
	Time     string  `json:"time"`
	Symbol   string  `json:"symbol"`
	Side     string  `json:"side"`
	Qty      float64 `json:"qty"`
	Price    float64 `json:"price"`
	Slippage float64 `json:"slippage_bps"`
	Strategy string  `json:"strategy"`
}

type wireTradesResp struct {
	Trades []wireTrade `json:"trades"`
}

type wireCircuitBreaker struct {
	Name      string `json:"name"`
	State     string `json:"state"`
	Reason    string `json:"reason"`
	TrippedAt string `json:"tripped_at"`
	TripCount int    `json:"trip_count"`
}

type wireCircuitBreakersResp struct {
	Breakers map[string]wireCircuitBreaker `json:"breakers"`
}

// ── APIClient ─────────────────────────────────────────────────────────────────

// APIClient performs direct HTTP calls to SRFM services.
type APIClient struct {
	http *http.Client
}

// NewAPIClient constructs an APIClient with a 2s timeout.
func NewAPIClient() *APIClient {
	return &APIClient{
		http: &http.Client{Timeout: requestTimeout},
	}
}

// get performs an HTTP GET to url and unmarshals JSON into dest.
func (c *APIClient) get(url string, dest interface{}) error {
	resp, err := c.http.Get(url)
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: status %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("GET %s read body: %w", url, err)
	}
	if err := json.Unmarshal(body, dest); err != nil {
		return fmt.Errorf("GET %s unmarshal: %w", url, err)
	}
	return nil
}

// GetPositions fetches live positions from live-trader.
func (c *APIClient) GetPositions() ([]views.Position, error) {
	var resp wirePositionsResp
	if err := c.get(liveTraderBase+"/positions", &resp); err != nil {
		return nil, err
	}
	out := make([]views.Position, len(resp.Positions))
	for i, p := range resp.Positions {
		out[i] = views.Position{
			Symbol:    p.Symbol,
			Qty:       p.Qty,
			AvgCost:   p.AvgCost,
			Mark:      p.Mark,
			UnrealPnL: p.UnrealPnL,
			PctPnL:    p.PctPnL,
			DailyPnL:  p.DailyPnL,
		}
	}
	return out, nil
}

// GetSignals fetches the current signal state from live-trader.
func (c *APIClient) GetSignals() (views.SignalState, error) {
	var resp wireSignalState
	if err := c.get(liveTraderBase+"/signals", &resp); err != nil {
		return views.SignalState{}, err
	}

	bh := make([]views.BHMassSignal, len(resp.BHMasses))
	for i, b := range resp.BHMasses {
		bh[i] = views.BHMassSignal{
			Symbol:  b.Symbol,
			Mass:    b.Mass,
			Active:  b.Active,
			History: views.SignalHistory(b.History),
		}
	}

	rlq := make([]views.RLQValue, len(resp.RLQValues))
	for i, q := range resp.RLQValues {
		rlq[i] = views.RLQValue{Action: q.Action, QValue: q.QValue, IsTop: i == 0}
	}

	updatedAt := time.Now()
	if t, err := time.Parse(time.RFC3339, resp.UpdatedAt); err == nil {
		updatedAt = t
	}

	return views.SignalState{
		BHMasses: bh,
		Hurst: views.HurstSignal{
			Value:   resp.Hurst.Value,
			History: views.SignalHistory(resp.Hurst.History),
		},
		GARCH: views.GARCHSignal{
			CurrentForecast: resp.GARCH.CurrentForecast,
			HistoricalMean:  resp.GARCH.HistoricalMean,
			History:         views.SignalHistory(resp.GARCH.History),
		},
		NavCurve: views.NavCurvature{
			W:              resp.NavCurvature.W,
			X:              resp.NavCurvature.X,
			Y:              resp.NavCurvature.Y,
			Z:              resp.NavCurvature.Z,
			CurvatureAngle: resp.NavCurvature.CurvatureAngle,
			Regime:         resp.NavCurvature.Regime,
		},
		RLQValues:  rlq,
		Regime:     resp.Regime,
		Confidence: resp.Confidence,
		UpdatedAt:  updatedAt,
	}, nil
}

// GetRiskMetrics fetches risk metrics from risk-aggregator.
func (c *APIClient) GetRiskMetrics() (views.RiskMetrics, error) {
	var resp wireRiskMetrics
	if err := c.get(riskAggBase+"/metrics", &resp); err != nil {
		return views.RiskMetrics{}, err
	}
	updatedAt := time.Now()
	if t, err := time.Parse(time.RFC3339, resp.UpdatedAt); err == nil {
		updatedAt = t
	}
	return views.RiskMetrics{
		VaR95:         resp.VaR95,
		VaRLimit:      resp.VaRLimit,
		Drawdown:      resp.Drawdown,
		DrawdownLimit: resp.DrawdownLimit,
		MarginUsed:    resp.MarginUsed,
		MarginLimit:   resp.MarginLimit,
		UpdatedAt:     updatedAt,
	}, nil
}

// GetRecentTrades fetches recent fills from live-trader.
func (c *APIClient) GetRecentTrades(n int) ([]views.Fill, error) {
	url := fmt.Sprintf("%s/trades?n=%d", liveTraderBase, n)
	var resp wireTradesResp
	if err := c.get(url, &resp); err != nil {
		return nil, err
	}
	out := make([]views.Fill, 0, len(resp.Trades))
	for _, t := range resp.Trades {
		ts := time.Now()
		if parsed, err := time.Parse(time.RFC3339Nano, t.Time); err == nil {
			ts = parsed
		}
		out = append(out, views.Fill{
			Time:     ts,
			Symbol:   t.Symbol,
			Side:     t.Side,
			Qty:      t.Qty,
			Price:    t.Price,
			Slippage: t.Slippage,
			Strategy: t.Strategy,
		})
	}
	return out, nil
}

// GetCircuitBreakers fetches all circuit breaker states from coordination.
func (c *APIClient) GetCircuitBreakers() (map[string]string, error) {
	var resp wireCircuitBreakersResp
	if err := c.get(coordinationBase+"/circuit/all", &resp); err != nil {
		return nil, err
	}
	out := make(map[string]string, len(resp.Breakers))
	for k, cb := range resp.Breakers {
		out[k] = cb.State
	}
	return out, nil
}

// GetCircuitBreakersDetail fetches full circuit breaker detail for RiskView.
func (c *APIClient) GetCircuitBreakersDetail() ([]views.CircuitBreaker, error) {
	var resp wireCircuitBreakersResp
	if err := c.get(coordinationBase+"/circuit/all", &resp); err != nil {
		return nil, err
	}
	out := make([]views.CircuitBreaker, 0, len(resp.Breakers))
	for key, cb := range resp.Breakers {
		// prefer map key as canonical name; fall back to Name field in body
		name := key
		if cb.Name != "" {
			name = cb.Name
		}
		ta := time.Time{}
		if cb.TrippedAt != "" {
			if t, err := time.Parse(time.RFC3339, cb.TrippedAt); err == nil {
				ta = t
			}
		}
		out = append(out, views.CircuitBreaker{
			Name:      name,
			State:     cb.State,
			Reason:    cb.Reason,
			TrippedAt: ta,
			TripCount: cb.TripCount,
		})
	}
	return out, nil
}

// ── CachedAPIClient ───────────────────────────────────────────────────────────

// cacheEntry holds a cached value and its expiry time.
type cacheEntry struct {
	value     interface{}
	expiresAt time.Time
}

// CachedAPIClient wraps APIClient with a per-endpoint 5s TTL cache.
// On fetch failure it returns the last successful cached value (if any).
type CachedAPIClient struct {
	inner *APIClient
	mu    sync.Mutex
	cache map[string]cacheEntry
}

// NewCachedAPIClient constructs a CachedAPIClient.
func NewCachedAPIClient() *CachedAPIClient {
	return &CachedAPIClient{
		inner: NewAPIClient(),
		cache: make(map[string]cacheEntry),
	}
}

// load retrieves a cached value if still fresh.
func (c *CachedAPIClient) load(key string) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.cache[key]
	if !ok {
		return nil, false
	}
	if time.Now().After(e.expiresAt) {
		return e.value, false // stale -- return for fallback but signal miss
	}
	return e.value, true
}

// store saves a value to cache.
func (c *CachedAPIClient) store(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache[key] = cacheEntry{value: value, expiresAt: time.Now().Add(cacheTTL)}
}

// loadStale returns the stale cached value regardless of TTL (fallback on error).
func (c *CachedAPIClient) loadStale(key string) (interface{}, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.cache[key]
	if !ok {
		return nil, false
	}
	return e.value, true
}

// GetPositions returns cached positions or fetches fresh ones.
func (c *CachedAPIClient) GetPositions() ([]views.Position, error) {
	const key = "positions"
	if v, ok := c.load(key); ok {
		return v.([]views.Position), nil
	}
	result, err := c.inner.GetPositions()
	if err != nil {
		if stale, ok := c.loadStale(key); ok {
			return stale.([]views.Position), nil
		}
		return nil, err
	}
	c.store(key, result)
	return result, nil
}

// GetSignals returns cached signals or fetches fresh ones.
func (c *CachedAPIClient) GetSignals() (views.SignalState, error) {
	const key = "signals"
	if v, ok := c.load(key); ok {
		return v.(views.SignalState), nil
	}
	result, err := c.inner.GetSignals()
	if err != nil {
		if stale, ok := c.loadStale(key); ok {
			return stale.(views.SignalState), nil
		}
		return views.SignalState{}, err
	}
	c.store(key, result)
	return result, nil
}

// GetRiskMetrics returns cached risk metrics or fetches fresh ones.
func (c *CachedAPIClient) GetRiskMetrics() (views.RiskMetrics, error) {
	const key = "risk_metrics"
	if v, ok := c.load(key); ok {
		return v.(views.RiskMetrics), nil
	}
	result, err := c.inner.GetRiskMetrics()
	if err != nil {
		if stale, ok := c.loadStale(key); ok {
			return stale.(views.RiskMetrics), nil
		}
		return views.RiskMetrics{}, err
	}
	c.store(key, result)
	return result, nil
}

// GetRecentTrades returns cached trades or fetches fresh ones.
func (c *CachedAPIClient) GetRecentTrades(n int) ([]views.Fill, error) {
	key := fmt.Sprintf("trades_%d", n)
	if v, ok := c.load(key); ok {
		return v.([]views.Fill), nil
	}
	result, err := c.inner.GetRecentTrades(n)
	if err != nil {
		if stale, ok := c.loadStale(key); ok {
			return stale.([]views.Fill), nil
		}
		return nil, err
	}
	c.store(key, result)
	return result, nil
}

// GetCircuitBreakers returns cached circuit breaker state map or fetches fresh.
func (c *CachedAPIClient) GetCircuitBreakers() (map[string]string, error) {
	const key = "circuit_breakers"
	if v, ok := c.load(key); ok {
		return v.(map[string]string), nil
	}
	result, err := c.inner.GetCircuitBreakers()
	if err != nil {
		if stale, ok := c.loadStale(key); ok {
			return stale.(map[string]string), nil
		}
		return nil, err
	}
	c.store(key, result)
	return result, nil
}

// GetCircuitBreakersDetail returns cached detail circuit breakers or fetches fresh.
func (c *CachedAPIClient) GetCircuitBreakersDetail() ([]views.CircuitBreaker, error) {
	const key = "circuit_breakers_detail"
	if v, ok := c.load(key); ok {
		return v.([]views.CircuitBreaker), nil
	}
	result, err := c.inner.GetCircuitBreakersDetail()
	if err != nil {
		if stale, ok := c.loadStale(key); ok {
			return stale.([]views.CircuitBreaker), nil
		}
		return nil, err
	}
	c.store(key, result)
	return result, nil
}
