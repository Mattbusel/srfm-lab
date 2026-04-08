package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

// ──────────────────────────────────────────────────────────────────────────────
// Configuration & domain types
// ──────────────────────────────────────────────────────────────────────────────

type BacktestConfig struct {
	ID         string            `json:"id"`
	Strategy   string            `json:"strategy"` // momentum, mean_rev, breakout
	Symbols    []string          `json:"symbols"`
	StartDate  string            `json:"start_date"`
	EndDate    string            `json:"end_date"`
	Parameters map[string]float64 `json:"parameters"`
	CostModel  CostModel         `json:"cost_model"`
	Sizing     SizingConfig      `json:"sizing"`
	WalkForward *WalkForwardCfg  `json:"walk_forward,omitempty"`
	MonteCarlo  *MonteCarloCfg   `json:"monte_carlo,omitempty"`
	InitialCash float64          `json:"initial_cash"`
}

type CostModel struct {
	CommissionPerShare float64 `json:"commission_per_share"`
	FixedSlippage      float64 `json:"fixed_slippage"`
	PropSlippage       float64 `json:"prop_slippage"`
	SqrtImpactCoeff    float64 `json:"sqrt_impact_coeff"`
}

type SizingConfig struct {
	Method     string  `json:"method"` // fixed_frac, vol_target, kelly
	FixedFrac  float64 `json:"fixed_frac"`
	VolTarget  float64 `json:"vol_target"`
	KellyFrac  float64 `json:"kelly_frac"`
	MaxPosPct  float64 `json:"max_pos_pct"`
}

type WalkForwardCfg struct {
	TrainBars  int `json:"train_bars"`
	TestBars   int `json:"test_bars"`
	NumFolds   int `json:"num_folds"`
}

type MonteCarloCfg struct {
	NumSims    int     `json:"num_sims"`
	Confidence float64 `json:"confidence"`
}

type Trade struct {
	Symbol    string  `json:"symbol"`
	Side      string  `json:"side"` // buy, sell
	Quantity  float64 `json:"quantity"`
	Price     float64 `json:"price"`
	Slippage  float64 `json:"slippage"`
	Commission float64 `json:"commission"`
	Bar       int     `json:"bar"`
	Timestamp string  `json:"timestamp"`
	PnL       float64 `json:"pnl"`
}

type EquityPoint struct {
	Bar       int     `json:"bar"`
	Equity    float64 `json:"equity"`
	Drawdown  float64 `json:"drawdown"`
	Timestamp string  `json:"timestamp"`
}

type BacktestMetrics struct {
	TotalReturn    float64 `json:"total_return"`
	AnnualReturn   float64 `json:"annual_return"`
	Sharpe         float64 `json:"sharpe"`
	Sortino        float64 `json:"sortino"`
	Calmar         float64 `json:"calmar"`
	MaxDrawdown    float64 `json:"max_drawdown"`
	MaxDDDuration  int     `json:"max_dd_duration_bars"`
	WinRate        float64 `json:"win_rate"`
	ProfitFactor   float64 `json:"profit_factor"`
	AvgWin         float64 `json:"avg_win"`
	AvgLoss        float64 `json:"avg_loss"`
	NumTrades      int     `json:"num_trades"`
	AvgBarsHeld    float64 `json:"avg_bars_held"`
	Volatility     float64 `json:"volatility"`
	Skewness       float64 `json:"skewness"`
	Kurtosis       float64 `json:"kurtosis"`
	TurnoverAnn    float64 `json:"turnover_annual"`
}

type MonthlyReturn struct {
	Year  int     `json:"year"`
	Month int     `json:"month"`
	Ret   float64 `json:"return"`
}

type WalkForwardResult struct {
	Fold       int             `json:"fold"`
	TrainMetrics BacktestMetrics `json:"train_metrics"`
	TestMetrics  BacktestMetrics `json:"test_metrics"`
	OptParams    map[string]float64 `json:"opt_params"`
}

type MonteCarloResult struct {
	MedianReturn float64   `json:"median_return"`
	P5Return     float64   `json:"p5_return"`
	P95Return    float64   `json:"p95_return"`
	MedianDD     float64   `json:"median_max_dd"`
	P95DD        float64   `json:"p95_max_dd"`
	Runs         []float64 `json:"terminal_values"`
}

type BacktestResult struct {
	ID             string              `json:"id"`
	Config         BacktestConfig      `json:"config"`
	EquityCurve    []EquityPoint       `json:"equity_curve"`
	Trades         []Trade             `json:"trades"`
	Metrics        BacktestMetrics     `json:"metrics"`
	MonthlyReturns []MonthlyReturn     `json:"monthly_returns"`
	WalkForward    []WalkForwardResult `json:"walk_forward,omitempty"`
	MonteCarlo     *MonteCarloResult   `json:"monte_carlo,omitempty"`
	Status         string              `json:"status"` // pending, running, done, error
	Progress       float64             `json:"progress"`
	Error          string              `json:"error,omitempty"`
	StartedAt      time.Time           `json:"started_at"`
	CompletedAt    time.Time           `json:"completed_at,omitempty"`
}

// ──────────────────────────────────────────────────────────────────────────────
// Backtest store (in-memory + SQLite-like persistence)
// ──────────────────────────────────────────────────────────────────────────────

type BacktestStore struct {
	mu      sync.RWMutex
	results map[string]*BacktestResult
}

func NewBacktestStore() *BacktestStore {
	return &BacktestStore{results: make(map[string]*BacktestResult)}
}

func (s *BacktestStore) Put(r *BacktestResult) {
	s.mu.Lock()
	s.results[r.ID] = r
	s.mu.Unlock()
}

func (s *BacktestStore) Get(id string) (*BacktestResult, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	r, ok := s.results[id]
	return r, ok
}

func (s *BacktestStore) UpdateProgress(id string, progress float64) {
	s.mu.Lock()
	if r, ok := s.results[id]; ok {
		r.Progress = progress
	}
	s.mu.Unlock()
}

func (s *BacktestStore) List() []*BacktestResult {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*BacktestResult, 0, len(s.results))
	for _, r := range s.results {
		out = append(out, r)
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// HTTP server
// ──────────────────────────────────────────────────────────────────────────────

type Server struct {
	store  *BacktestStore
	engine *BacktestEngine
	mux    *http.ServeMux
}

func NewServer() *Server {
	s := &Server{
		store:  NewBacktestStore(),
		engine: NewBacktestEngine(),
		mux:    http.NewServeMux(),
	}
	s.mux.HandleFunc("/backtest/run", s.handleRun)
	s.mux.HandleFunc("/backtest/status/", s.handleStatus)
	s.mux.HandleFunc("/backtest/results/", s.handleResults)
	s.mux.HandleFunc("/backtest/list", s.handleList)
	s.mux.HandleFunc("/health", s.handleHealth)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func (s *Server) handleRun(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var cfg BacktestConfig
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	r.Body.Close()

	if cfg.ID == "" {
		cfg.ID = fmt.Sprintf("bt_%d", time.Now().UnixNano())
	}
	if cfg.InitialCash <= 0 {
		cfg.InitialCash = 100000
	}

	result := &BacktestResult{
		ID:        cfg.ID,
		Config:    cfg,
		Status:    "pending",
		StartedAt: time.Now(),
	}
	s.store.Put(result)

	go func() {
		s.store.mu.Lock()
		result.Status = "running"
		s.store.mu.Unlock()

		err := s.engine.Run(cfg, result, s.store)

		s.store.mu.Lock()
		if err != nil {
			result.Status = "error"
			result.Error = err.Error()
		} else {
			result.Status = "done"
		}
		result.CompletedAt = time.Now()
		s.store.mu.Unlock()
	}()

	writeJSON(w, http.StatusAccepted, map[string]string{"id": cfg.ID, "status": "pending"})
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Path[len("/backtest/status/"):]
	res, ok := s.store.Get(id)
	if !ok {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"id": res.ID, "status": res.Status, "progress": res.Progress,
	})
}

func (s *Server) handleResults(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Path[len("/backtest/results/"):]
	res, ok := s.store.Get(id)
	if !ok {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func (s *Server) handleList(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.store.List())
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func main() {
	addr := ":8090"
	if v := os.Getenv("BACKTEST_ADDR"); v != "" {
		addr = v
	}
	srv := NewServer()
	log.Printf("backtest-server listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, srv))
}
