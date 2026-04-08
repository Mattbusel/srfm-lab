package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Data types shared across the service
// ---------------------------------------------------------------------------

// Portfolio represents a portfolio of assets with weights.
type Portfolio struct {
	ID          string             `json:"id"`
	Weights     map[string]float64 `json:"weights"`
	CreatedAt   time.Time          `json:"created_at"`
	UpdatedAt   time.Time          `json:"updated_at"`
	Metadata    map[string]string  `json:"metadata,omitempty"`
}

// OptimizeRequest is the request body for POST /optimize.
type OptimizeRequest struct {
	Method       string              `json:"method"` // mean_variance, min_variance, max_sharpe, risk_parity, black_litterman, max_diversification, equal_weight, inverse_vol
	Assets       []string            `json:"assets"`
	Returns      [][]float64         `json:"returns"` // T x N matrix
	RiskFreeRate float64             `json:"risk_free_rate"`
	TargetReturn float64             `json:"target_return,omitempty"`
	Constraints  ConstraintSet       `json:"constraints,omitempty"`
	CostModel    CostModelConfig     `json:"cost_model,omitempty"`
	CovMethod    string              `json:"cov_method,omitempty"` // sample, ledoit_wolf, ewma
	EWMALambda   float64             `json:"ewma_lambda,omitempty"`
	Views        []BlackLittermanView `json:"views,omitempty"`
}

// BlackLittermanView represents an investor view for BL model.
type BlackLittermanView struct {
	Assets  []string  `json:"assets"`
	Weights []float64 `json:"weights"`
	Return  float64   `json:"return"`
	Confidence float64 `json:"confidence"`
}

// ConstraintSet holds optimization constraints.
type ConstraintSet struct {
	MinWeight       float64            `json:"min_weight"`
	MaxWeight       float64            `json:"max_weight"`
	GroupConstraints []GroupConstraint  `json:"group_constraints,omitempty"`
	MaxTurnover     float64            `json:"max_turnover,omitempty"`
	MaxTrackingError float64           `json:"max_tracking_error,omitempty"`
	FactorExposure  []FactorConstraint `json:"factor_exposure,omitempty"`
}

// GroupConstraint limits weight for a group of assets.
type GroupConstraint struct {
	Assets    []string `json:"assets"`
	MinWeight float64  `json:"min_weight"`
	MaxWeight float64  `json:"max_weight"`
}

// FactorConstraint limits factor exposure.
type FactorConstraint struct {
	FactorName string    `json:"factor_name"`
	Loadings   []float64 `json:"loadings"` // per asset
	MinExposure float64  `json:"min_exposure"`
	MaxExposure float64  `json:"max_exposure"`
}

// CostModelConfig configures transaction costs.
type CostModelConfig struct {
	Type       string  `json:"type"` // proportional, sqrt_impact
	FixedCost  float64 `json:"fixed_cost"`
	PropCost   float64 `json:"prop_cost"`
	ImpactCoeff float64 `json:"impact_coeff"`
}

// OptimizeResponse is the response for POST /optimize.
type OptimizeResponse struct {
	Weights        map[string]float64 `json:"weights"`
	ExpectedReturn float64            `json:"expected_return"`
	Volatility     float64            `json:"volatility"`
	Sharpe         float64            `json:"sharpe"`
	Method         string             `json:"method"`
	CovMethod      string             `json:"cov_method"`
}

// RiskResponse is the response for GET /risk/{portfolio_id}.
type RiskResponse struct {
	PortfolioID    string             `json:"portfolio_id"`
	VaR95          float64            `json:"var_95"`
	VaR99          float64            `json:"var_99"`
	CVaR95         float64            `json:"cvar_95"`
	ComponentVaR   map[string]float64 `json:"component_var"`
	Volatility     float64            `json:"volatility"`
	MaxDrawdown    float64            `json:"max_drawdown"`
	HHI            float64            `json:"hhi"`
	TopNConc       float64            `json:"top_n_concentration"`
}

// StressTestRequest is the request body for POST /stress-test.
type StressTestRequest struct {
	PortfolioID string              `json:"portfolio_id"`
	Scenarios   []StressScenario    `json:"scenarios"`
}

// StressScenario defines a stress test scenario.
type StressScenario struct {
	Name    string             `json:"name"`
	Shocks  map[string]float64 `json:"shocks"` // asset -> return shock
}

// StressTestResponse is the response for POST /stress-test.
type StressTestResponse struct {
	Results []ScenarioResult `json:"results"`
}

// ScenarioResult holds the result of a single stress scenario.
type ScenarioResult struct {
	Name       string  `json:"name"`
	PortReturn float64 `json:"portfolio_return"`
	PortLoss   float64 `json:"portfolio_loss"`
}

// RebalanceRequest is the request body for POST /rebalance.
type RebalanceRequest struct {
	PortfolioID    string             `json:"portfolio_id"`
	TargetWeights  map[string]float64 `json:"target_weights"`
	CurrentWeights map[string]float64 `json:"current_weights"`
	Threshold      float64            `json:"threshold"`
	MinTradeSize   float64            `json:"min_trade_size"`
	CostModel      CostModelConfig    `json:"cost_model"`
}

// RebalanceResponse is the response for POST /rebalance.
type RebalanceResponse struct {
	Trades       map[string]float64 `json:"trades"`
	TotalCost    float64            `json:"total_cost"`
	Turnover     float64            `json:"turnover"`
	Rebalanced   bool               `json:"rebalanced"`
}

// ---------------------------------------------------------------------------
// Portfolio store
// ---------------------------------------------------------------------------

type portfolioStore struct {
	mu         sync.RWMutex
	portfolios map[string]*Portfolio
	returns    map[string][][]float64 // portfolio_id -> historical returns matrix
}

func newPortfolioStore() *portfolioStore {
	return &portfolioStore{
		portfolios: make(map[string]*Portfolio),
		returns:    make(map[string][][]float64),
	}
}

func (s *portfolioStore) get(id string) (*Portfolio, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	p, ok := s.portfolios[id]
	return p, ok
}

func (s *portfolioStore) put(p *Portfolio) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.portfolios[p.ID] = p
}

func (s *portfolioStore) putReturns(id string, returns [][]float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.returns[id] = returns
}

func (s *portfolioStore) getReturns(id string) ([][]float64, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	r, ok := s.returns[id]
	return r, ok
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

type server struct {
	store     *portfolioStore
	optimizer *Optimizer
	risk      *RiskEngine
}

func newServer() *server {
	return &server{
		store:     newPortfolioStore(),
		optimizer: NewOptimizer(),
		risk:      NewRiskEngine(),
	}
}

func (s *server) handleOptimize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req OptimizeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	if len(req.Assets) == 0 || len(req.Returns) == 0 {
		http.Error(w, "assets and returns are required", http.StatusBadRequest)
		return
	}
	n := len(req.Assets)
	if len(req.Returns[0]) != n {
		http.Error(w, "returns columns must match assets count", http.StatusBadRequest)
		return
	}
	// Covariance estimation
	covMethod := req.CovMethod
	if covMethod == "" {
		covMethod = "sample"
	}
	lambda := req.EWMALambda
	if lambda == 0 {
		lambda = 0.94
	}
	var cov [][]float64
	switch covMethod {
	case "ledoit_wolf":
		cov = LedoitWolfShrinkage(req.Returns, n)
	case "ewma":
		cov = EWMACov(req.Returns, n, lambda)
	default:
		cov = SampleCov(req.Returns, n)
	}
	mu := MeanReturns(req.Returns, n)
	// Apply constraints defaults
	if req.Constraints.MaxWeight == 0 {
		req.Constraints.MaxWeight = 1.0
	}
	var weights []float64
	switch req.Method {
	case "min_variance":
		weights = s.optimizer.MinVariance(cov, n, req.Constraints)
	case "max_sharpe":
		weights = s.optimizer.MaxSharpe(mu, cov, n, req.RiskFreeRate, req.Constraints)
	case "risk_parity":
		weights = s.optimizer.RiskParity(cov, n)
	case "black_litterman":
		weights = s.optimizer.BlackLitterman(mu, cov, n, req.Views, req.RiskFreeRate, req.Constraints)
	case "max_diversification":
		weights = s.optimizer.MaxDiversification(cov, n, req.Constraints)
	case "equal_weight":
		weights = s.optimizer.EqualWeight(n)
	case "inverse_vol":
		weights = s.optimizer.InverseVol(cov, n)
	default:
		weights = s.optimizer.MeanVariance(mu, cov, n, req.TargetReturn, req.Constraints)
	}
	// Build response
	wMap := make(map[string]float64, n)
	for i, a := range req.Assets {
		wMap[a] = weights[i]
	}
	expRet := dotProduct(weights, mu)
	vol := portfolioVol(weights, cov)
	sharpe := 0.0
	if vol > 0 {
		sharpe = (expRet - req.RiskFreeRate) / vol
	}
	// Store portfolio
	portID := fmt.Sprintf("port_%d", time.Now().UnixNano())
	port := &Portfolio{
		ID:        portID,
		Weights:   wMap,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	s.store.put(port)
	s.store.putReturns(portID, req.Returns)
	resp := OptimizeResponse{
		Weights:        wMap,
		ExpectedReturn: expRet,
		Volatility:     vol,
		Sharpe:         sharpe,
		Method:         req.Method,
		CovMethod:      covMethod,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleRisk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	path := r.URL.Path
	parts := strings.Split(strings.TrimPrefix(path, "/risk/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		http.Error(w, "portfolio_id required", http.StatusBadRequest)
		return
	}
	portID := parts[0]
	port, ok := s.store.get(portID)
	if !ok {
		http.Error(w, "portfolio not found", http.StatusNotFound)
		return
	}
	returns, ok := s.store.getReturns(portID)
	if !ok {
		http.Error(w, "returns not found", http.StatusNotFound)
		return
	}
	assets := make([]string, 0, len(port.Weights))
	weights := make([]float64, 0, len(port.Weights))
	for a, wt := range port.Weights {
		assets = append(assets, a)
		weights = append(weights, wt)
	}
	n := len(assets)
	riskResult := s.risk.ComputeRisk(weights, returns, n)
	compVaR := make(map[string]float64, n)
	for i, a := range assets {
		if i < len(riskResult.ComponentVaR) {
			compVaR[a] = riskResult.ComponentVaR[i]
		}
	}
	resp := RiskResponse{
		PortfolioID:  portID,
		VaR95:        riskResult.VaR95,
		VaR99:        riskResult.VaR99,
		CVaR95:       riskResult.CVaR95,
		ComponentVaR: compVaR,
		Volatility:   riskResult.Volatility,
		MaxDrawdown:  riskResult.MaxDrawdown,
		HHI:          riskResult.HHI,
		TopNConc:     riskResult.TopNConc,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleStressTest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req StressTestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	port, ok := s.store.get(req.PortfolioID)
	if !ok {
		http.Error(w, "portfolio not found", http.StatusNotFound)
		return
	}
	var results []ScenarioResult
	for _, scenario := range req.Scenarios {
		portRet := 0.0
		for asset, weight := range port.Weights {
			if shock, ok := scenario.Shocks[asset]; ok {
				portRet += weight * shock
			}
		}
		loss := 0.0
		if portRet < 0 {
			loss = -portRet
		}
		results = append(results, ScenarioResult{
			Name:       scenario.Name,
			PortReturn: portRet,
			PortLoss:   loss,
		})
	}
	resp := StressTestResponse{Results: results}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleRebalance(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req RebalanceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}
	defer r.Body.Close()
	trades := make(map[string]float64)
	turnover := 0.0
	rebalanced := false
	// Check if any weight deviates beyond threshold
	needsRebalance := false
	allAssets := make(map[string]bool)
	for a := range req.TargetWeights {
		allAssets[a] = true
	}
	for a := range req.CurrentWeights {
		allAssets[a] = true
	}
	for a := range allAssets {
		target := req.TargetWeights[a]
		current := req.CurrentWeights[a]
		diff := target - current
		if diff < 0 {
			diff = -diff
		}
		if diff > req.Threshold {
			needsRebalance = true
			break
		}
	}
	totalCost := 0.0
	if needsRebalance {
		rebalanced = true
		for a := range allAssets {
			target := req.TargetWeights[a]
			current := req.CurrentWeights[a]
			trade := target - current
			absTrade := trade
			if absTrade < 0 {
				absTrade = -absTrade
			}
			if absTrade < req.MinTradeSize {
				continue
			}
			trades[a] = trade
			turnover += absTrade
			// Cost
			switch req.CostModel.Type {
			case "sqrt_impact":
				cost := req.CostModel.PropCost*absTrade + req.CostModel.ImpactCoeff*sqrt(absTrade)
				totalCost += cost
			default:
				cost := req.CostModel.FixedCost + req.CostModel.PropCost*absTrade
				totalCost += cost
			}
		}
	}
	resp := RebalanceResponse{
		Trades:     trades,
		TotalCost:  totalCost,
		Turnover:   turnover / 2,
		Rebalanced: rebalanced,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	srv := newServer()
	mux := http.NewServeMux()
	mux.HandleFunc("/optimize", srv.handleOptimize)
	mux.HandleFunc("/risk/", srv.handleRisk)
	mux.HandleFunc("/stress-test", srv.handleStressTest)
	mux.HandleFunc("/rebalance", srv.handleRebalance)
	mux.HandleFunc("/health", srv.handleHealth)
	addr := ":8090"
	log.Printf("portfolio-engine listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
