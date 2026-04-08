package backtest

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

// Bar represents a single OHLCV bar.
type Bar struct {
	Timestamp int64   `json:"timestamp"`
	Symbol    string  `json:"symbol"`
	Open      float64 `json:"open"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Close     float64 `json:"close"`
	Volume    float64 `json:"volume"`
}

// Trade represents a single executed trade.
type Trade struct {
	Timestamp int64   `json:"timestamp"`
	Symbol    string  `json:"symbol"`
	Side      string  `json:"side"` // "buy" or "sell"
	Quantity  float64 `json:"quantity"`
	Price     float64 `json:"price"`
	Cost      float64 `json:"cost"`
	SlippageBps float64 `json:"slippage_bps"`
}

// Position represents a held position.
type Position struct {
	Symbol    string  `json:"symbol"`
	Quantity  float64 `json:"quantity"`
	AvgPrice  float64 `json:"avg_price"`
	MarketVal float64 `json:"market_value"`
	UnrealPnL float64 `json:"unrealized_pnl"`
	RealPnL   float64 `json:"realized_pnl"`
}

// OrderSignal is the output of a strategy for a given bar.
type OrderSignal struct {
	Symbol   string
	Weight   float64 // target portfolio weight
	Quantity float64 // or absolute quantity
	Side     string  // "buy", "sell", or "" for weight-based
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// BacktestConfig holds all configuration for a backtest run.
type BacktestConfig struct {
	StartDate        int64           `json:"start_date"`
	EndDate          int64           `json:"end_date"`
	Symbols          []string        `json:"symbols"`
	InitialCapital   float64         `json:"initial_capital"`
	RebalanceFreq    int             `json:"rebalance_freq"` // bars between rebalances
	CostModel        CostModelConfig `json:"cost_model"`
	Slippage         float64         `json:"slippage_bps"`
	MarginReq        float64         `json:"margin_requirement"` // 0 = no margin
	MaxPositionSize  float64         `json:"max_position_size"`  // max weight per symbol
	AllowShort       bool            `json:"allow_short"`
	RiskFreeRate     float64         `json:"risk_free_rate"`
	BenchmarkSymbol  string          `json:"benchmark_symbol"`
}

// CostModelConfig configures transaction costs.
type CostModelConfig struct {
	Type        string  `json:"type"` // "fixed", "proportional", "sqrt_impact"
	FixedCost   float64 `json:"fixed_cost"`
	PropCost    float64 `json:"prop_cost"`
	ImpactCoeff float64 `json:"impact_coeff"`
	MinCost     float64 `json:"min_cost"`
}

// DefaultConfig returns a sensible default configuration.
func DefaultConfig() BacktestConfig {
	return BacktestConfig{
		InitialCapital:  1000000,
		RebalanceFreq:   21,
		CostModel:       CostModelConfig{Type: "proportional", PropCost: 0.001},
		Slippage:        5,
		MaxPositionSize: 0.2,
		RiskFreeRate:    0.02,
	}
}

// ---------------------------------------------------------------------------
// Strategy interface
// ---------------------------------------------------------------------------

// Strategy defines the interface for a trading strategy.
type Strategy interface {
	Name() string
	OnBar(bars map[string]Bar, positions map[string]*Position, equity float64) []OrderSignal
	OnInit(config BacktestConfig)
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

func computeCost(config CostModelConfig, notional float64) float64 {
	absNotional := math.Abs(notional)
	var cost float64
	switch config.Type {
	case "fixed":
		cost = config.FixedCost
	case "sqrt_impact":
		cost = config.FixedCost + config.PropCost*absNotional + config.ImpactCoeff*math.Sqrt(absNotional)
	default:
		cost = config.FixedCost + config.PropCost*absNotional
	}
	if cost < config.MinCost && absNotional > 0 {
		cost = config.MinCost
	}
	return cost
}

func computeSlippage(price float64, slippageBps float64, side string) float64 {
	slip := price * slippageBps / 10000
	if side == "buy" {
		return price + slip
	}
	return price - slip
}

// ---------------------------------------------------------------------------
// PositionTracker
// ---------------------------------------------------------------------------

// PositionTracker manages positions, cash, and P&L.
type PositionTracker struct {
	mu        sync.RWMutex
	positions map[string]*Position
	cash      float64
	margin    float64
	trades    []Trade
}

// NewPositionTracker creates a new tracker with initial cash.
func NewPositionTracker(initialCash float64) *PositionTracker {
	return &PositionTracker{
		positions: make(map[string]*Position),
		cash:      initialCash,
	}
}

// Cash returns available cash.
func (pt *PositionTracker) Cash() float64 {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	return pt.cash
}

// Equity returns total portfolio value.
func (pt *PositionTracker) Equity() float64 {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	equity := pt.cash
	for _, p := range pt.positions {
		equity += p.MarketVal
	}
	return equity
}

// Positions returns a copy of all positions.
func (pt *PositionTracker) Positions() map[string]*Position {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	out := make(map[string]*Position, len(pt.positions))
	for k, v := range pt.positions {
		cp := *v
		out[k] = &cp
	}
	return out
}

// UpdateMarketPrices updates position market values.
func (pt *PositionTracker) UpdateMarketPrices(prices map[string]float64) {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	for sym, pos := range pt.positions {
		if price, ok := prices[sym]; ok {
			pos.MarketVal = pos.Quantity * price
			pos.UnrealPnL = (price - pos.AvgPrice) * pos.Quantity
		}
	}
}

// ExecuteTrade executes a trade and updates positions.
func (pt *PositionTracker) ExecuteTrade(trade Trade) {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	pos, exists := pt.positions[trade.Symbol]
	if !exists {
		pos = &Position{Symbol: trade.Symbol}
		pt.positions[trade.Symbol] = pos
	}
	notional := trade.Quantity * trade.Price
	if trade.Side == "buy" {
		// Update average price
		totalCost := pos.AvgPrice*pos.Quantity + notional
		pos.Quantity += trade.Quantity
		if pos.Quantity != 0 {
			pos.AvgPrice = totalCost / pos.Quantity
		}
		pos.MarketVal = pos.Quantity * trade.Price
		pt.cash -= notional + trade.Cost
	} else {
		// Sell
		if pos.Quantity > 0 {
			realPnL := (trade.Price - pos.AvgPrice) * trade.Quantity
			pos.RealPnL += realPnL
		}
		pos.Quantity -= trade.Quantity
		pos.MarketVal = pos.Quantity * trade.Price
		pt.cash += notional - trade.Cost
	}
	// Clean up zero positions
	if math.Abs(pos.Quantity) < 1e-10 {
		pos.Quantity = 0
		pos.MarketVal = 0
		pos.UnrealPnL = 0
		pos.AvgPrice = 0
	}
	pt.trades = append(pt.trades, trade)
}

// Trades returns all executed trades.
func (pt *PositionTracker) Trades() []Trade {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	out := make([]Trade, len(pt.trades))
	copy(out, pt.trades)
	return out
}

// TotalRealizedPnL returns total realized P&L.
func (pt *PositionTracker) TotalRealizedPnL() float64 {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	total := 0.0
	for _, p := range pt.positions {
		total += p.RealPnL
	}
	return total
}

// Weights returns current portfolio weights.
func (pt *PositionTracker) Weights() map[string]float64 {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	equity := pt.cash
	for _, p := range pt.positions {
		equity += p.MarketVal
	}
	if equity <= 0 {
		return nil
	}
	w := make(map[string]float64, len(pt.positions))
	for sym, p := range pt.positions {
		w[sym] = p.MarketVal / equity
	}
	return w
}

// ---------------------------------------------------------------------------
// Engine: bar-by-bar event loop
// ---------------------------------------------------------------------------

// Engine runs backtests.
type Engine struct {
	config   BacktestConfig
	strategy Strategy
	tracker  *PositionTracker
	barCount int
	equityCurve []float64
	returns     []float64
	timestamps  []int64
	barData     map[string][]Bar
}

// NewEngine creates a new backtest engine.
func NewEngine(config BacktestConfig, strategy Strategy) *Engine {
	return &Engine{
		config:   config,
		strategy: strategy,
		tracker:  NewPositionTracker(config.InitialCapital),
		barData:  make(map[string][]Bar),
	}
}

// LoadBars loads bar data for all symbols.
func (e *Engine) LoadBars(data map[string][]Bar) {
	e.barData = data
}

// Run executes the backtest.
func (e *Engine) Run() Report {
	e.strategy.OnInit(e.config)
	// Determine time range
	allTimestamps := make(map[int64]bool)
	for _, bars := range e.barData {
		for _, b := range bars {
			if (e.config.StartDate == 0 || b.Timestamp >= e.config.StartDate) &&
				(e.config.EndDate == 0 || b.Timestamp <= e.config.EndDate) {
				allTimestamps[b.Timestamp] = true
			}
		}
	}
	timestamps := make([]int64, 0, len(allTimestamps))
	for ts := range allTimestamps {
		timestamps = append(timestamps, ts)
	}
	sort.Slice(timestamps, func(i, j int) bool { return timestamps[i] < timestamps[j] })
	// Build index: symbol -> timestamp -> bar
	barIndex := make(map[string]map[int64]Bar)
	for sym, bars := range e.barData {
		barIndex[sym] = make(map[int64]Bar, len(bars))
		for _, b := range bars {
			barIndex[sym][b.Timestamp] = b
		}
	}
	prevEquity := e.config.InitialCapital
	for i, ts := range timestamps {
		// Get current bars
		currentBars := make(map[string]Bar)
		prices := make(map[string]float64)
		for _, sym := range e.config.Symbols {
			if b, ok := barIndex[sym][ts]; ok {
				currentBars[sym] = b
				prices[sym] = b.Close
			}
		}
		// Update market prices
		e.tracker.UpdateMarketPrices(prices)
		equity := e.tracker.Equity()
		// Record equity
		e.equityCurve = append(e.equityCurve, equity)
		e.timestamps = append(e.timestamps, ts)
		if prevEquity > 0 {
			ret := math.Log(equity / prevEquity)
			e.returns = append(e.returns, ret)
		}
		prevEquity = equity
		// Rebalance check
		if e.config.RebalanceFreq > 0 && i%e.config.RebalanceFreq == 0 {
			signals := e.strategy.OnBar(currentBars, e.tracker.Positions(), equity)
			e.executeSignals(signals, prices, ts)
		} else if e.config.RebalanceFreq == 0 {
			signals := e.strategy.OnBar(currentBars, e.tracker.Positions(), equity)
			e.executeSignals(signals, prices, ts)
		}
		e.barCount++
	}
	return e.generateReport()
}

func (e *Engine) executeSignals(signals []OrderSignal, prices map[string]float64, ts int64) {
	equity := e.tracker.Equity()
	for _, sig := range signals {
		price, ok := prices[sig.Symbol]
		if !ok || price <= 0 {
			continue
		}
		// Determine target quantity from weight
		var targetQty float64
		if sig.Weight != 0 {
			targetNotional := sig.Weight * equity
			targetQty = targetNotional / price
		} else {
			targetQty = sig.Quantity
		}
		// Max position size constraint
		if e.config.MaxPositionSize > 0 {
			maxQty := e.config.MaxPositionSize * equity / price
			if targetQty > maxQty {
				targetQty = maxQty
			}
			if targetQty < -maxQty {
				targetQty = -maxQty
			}
		}
		// Short constraint
		if !e.config.AllowShort && targetQty < 0 {
			targetQty = 0
		}
		// Current position
		positions := e.tracker.Positions()
		currentQty := 0.0
		if pos, exists := positions[sig.Symbol]; exists {
			currentQty = pos.Quantity
		}
		deltaQty := targetQty - currentQty
		if math.Abs(deltaQty) < 1e-10 {
			continue
		}
		side := "buy"
		qty := deltaQty
		if deltaQty < 0 {
			side = "sell"
			qty = -deltaQty
		}
		execPrice := computeSlippage(price, e.config.Slippage, side)
		notional := qty * execPrice
		cost := computeCost(e.config.CostModel, notional)
		trade := Trade{
			Timestamp:   ts,
			Symbol:      sig.Symbol,
			Side:        side,
			Quantity:    qty,
			Price:       execPrice,
			Cost:        cost,
			SlippageBps: e.config.Slippage,
		}
		e.tracker.ExecuteTrade(trade)
	}
}

// ---------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------

// Analytics holds all computed performance metrics.
type Analytics struct {
	TotalReturn      float64            `json:"total_return"`
	AnnualizedReturn float64            `json:"annualized_return"`
	Volatility       float64            `json:"volatility"`
	Sharpe           float64            `json:"sharpe"`
	Sortino          float64            `json:"sortino"`
	Calmar           float64            `json:"calmar"`
	MaxDrawdown      float64            `json:"max_drawdown"`
	MaxDDDuration    int                `json:"max_dd_duration"`
	WinRate          float64            `json:"win_rate"`
	ProfitFactor     float64            `json:"profit_factor"`
	AvgWin           float64            `json:"avg_win"`
	AvgLoss          float64            `json:"avg_loss"`
	TotalTrades      int                `json:"total_trades"`
	TotalCosts       float64            `json:"total_costs"`
	Turnover         float64            `json:"turnover"`
	MonthlyReturns   []float64          `json:"monthly_returns"`
	SkewReturn       float64            `json:"skew_return"`
	KurtReturn       float64            `json:"kurt_return"`
}

func computeAnalytics(returns []float64, trades []Trade, rf float64) Analytics {
	a := Analytics{}
	n := len(returns)
	if n == 0 {
		return a
	}
	// Total return
	cumRet := 0.0
	for _, r := range returns {
		cumRet += r
	}
	a.TotalReturn = math.Exp(cumRet) - 1
	// Annualized
	years := float64(n) / 252
	if years > 0 {
		a.AnnualizedReturn = math.Pow(1+a.TotalReturn, 1/years) - 1
	}
	// Vol
	mu := cumRet / float64(n)
	ss := 0.0
	for _, r := range returns {
		d := r - mu
		ss += d * d
	}
	if n > 1 {
		a.Volatility = math.Sqrt(ss/float64(n-1)) * math.Sqrt(252)
	}
	// Sharpe
	if a.Volatility > 0 {
		a.Sharpe = (a.AnnualizedReturn - rf) / a.Volatility
	}
	// Sortino
	downSS := 0.0
	for _, r := range returns {
		if r < 0 {
			downSS += r * r
		}
	}
	downVol := math.Sqrt(downSS/float64(n)) * math.Sqrt(252)
	if downVol > 0 {
		a.Sortino = (a.AnnualizedReturn - rf) / downVol
	}
	// Max drawdown
	peak := 0.0
	cum := 0.0
	maxDD := 0.0
	ddStart := 0
	maxDDDur := 0
	curDDStart := 0
	inDD := false
	for i, r := range returns {
		cum += r
		if cum > peak {
			peak = cum
			if inDD {
				dur := i - curDDStart
				if dur > maxDDDur {
					maxDDDur = dur
				}
				inDD = false
			}
		}
		dd := peak - cum
		if dd > 0 && !inDD {
			inDD = true
			curDDStart = i
		}
		if dd > maxDD {
			maxDD = dd
			ddStart = curDDStart
		}
	}
	if inDD {
		dur := n - ddStart
		if dur > maxDDDur {
			maxDDDur = dur
		}
	}
	a.MaxDrawdown = maxDD
	a.MaxDDDuration = maxDDDur
	// Calmar
	if maxDD > 0 {
		a.Calmar = a.AnnualizedReturn / maxDD
	}
	// Win/loss analysis on returns
	wins := 0
	grossProfit := 0.0
	grossLoss := 0.0
	totalWin := 0.0
	totalLoss := 0.0
	winCount := 0
	lossCount := 0
	for _, r := range returns {
		if r > 0 {
			wins++
			grossProfit += r
			totalWin += r
			winCount++
		} else if r < 0 {
			grossLoss -= r
			totalLoss -= r
			lossCount++
		}
	}
	a.WinRate = float64(wins) / float64(n)
	if grossLoss > 0 {
		a.ProfitFactor = grossProfit / grossLoss
	}
	if winCount > 0 {
		a.AvgWin = totalWin / float64(winCount)
	}
	if lossCount > 0 {
		a.AvgLoss = totalLoss / float64(lossCount)
	}
	// Trades
	a.TotalTrades = len(trades)
	for _, t := range trades {
		a.TotalCosts += t.Cost
	}
	// Turnover
	if n > 0 {
		totalNotional := 0.0
		for _, t := range trades {
			totalNotional += t.Quantity * t.Price
		}
		a.Turnover = totalNotional / float64(n)
	}
	// Monthly returns
	monthLen := 21
	for i := 0; i < n; i += monthLen {
		end := i + monthLen
		if end > n {
			end = n
		}
		mRet := 0.0
		for j := i; j < end; j++ {
			mRet += returns[j]
		}
		a.MonthlyReturns = append(a.MonthlyReturns, mRet)
	}
	// Higher moments
	a.SkewReturn = skewness(returns)
	a.KurtReturn = kurtosis(returns)
	return a
}

func skewness(vals []float64) float64 {
	n := float64(len(vals))
	if n < 3 {
		return 0
	}
	m := 0.0
	for _, v := range vals {
		m += v
	}
	m /= n
	m2, m3 := 0.0, 0.0
	for _, v := range vals {
		d := v - m
		m2 += d * d
		m3 += d * d * d
	}
	m2 /= n
	m3 /= n
	if m2 == 0 {
		return 0
	}
	return m3 / math.Pow(m2, 1.5)
}

func kurtosis(vals []float64) float64 {
	n := float64(len(vals))
	if n < 4 {
		return 0
	}
	m := 0.0
	for _, v := range vals {
		m += v
	}
	m /= n
	m2, m4 := 0.0, 0.0
	for _, v := range vals {
		d := v - m
		d2 := d * d
		m2 += d2
		m4 += d2 * d2
	}
	m2 /= n
	m4 /= n
	if m2 == 0 {
		return 0
	}
	return m4/(m2*m2) - 3
}

func (e *Engine) generateReport() Report {
	analytics := computeAnalytics(e.returns, e.tracker.Trades(), e.config.RiskFreeRate)
	return Report{
		Strategy:    e.strategy.Name(),
		Config:      e.config,
		Analytics:   analytics,
		EquityCurve: e.equityCurve,
		Returns:     e.returns,
		Trades:      e.tracker.Trades(),
		FinalEquity: e.tracker.Equity(),
		BarCount:    e.barCount,
	}
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

// Report holds the complete backtest results.
type Report struct {
	Strategy    string         `json:"strategy"`
	Config      BacktestConfig `json:"config"`
	Analytics   Analytics      `json:"analytics"`
	EquityCurve []float64      `json:"equity_curve"`
	Returns     []float64      `json:"returns"`
	Trades      []Trade        `json:"trades"`
	FinalEquity float64        `json:"final_equity"`
	BarCount    int            `json:"bar_count"`
}

// JSON returns the report as JSON bytes.
func (r *Report) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// Summary returns a text summary of the report.
func (r *Report) Summary() string {
	return fmt.Sprintf(
		"Strategy: %s | Return: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%% | Trades: %d | Final: %.2f",
		r.Strategy,
		r.Analytics.TotalReturn*100,
		r.Analytics.Sharpe,
		r.Analytics.MaxDrawdown*100,
		r.Analytics.TotalTrades,
		r.FinalEquity,
	)
}

// ---------------------------------------------------------------------------
// WalkForward: train/test splits
// ---------------------------------------------------------------------------

// WalkForwardConfig configures walk-forward analysis.
type WalkForwardConfig struct {
	TrainPeriod int // bars for training
	TestPeriod  int // bars for testing
	Step        int // step size between windows
}

// WalkForwardResult holds results from one walk-forward window.
type WalkForwardResult struct {
	TrainStart int
	TrainEnd   int
	TestStart  int
	TestEnd    int
	TrainReport Report
	TestReport  Report
}

// WalkForward runs walk-forward optimization.
type WalkForward struct {
	config    WalkForwardConfig
	btConfig  BacktestConfig
	barData   map[string][]Bar
}

// NewWalkForward creates a walk-forward analyzer.
func NewWalkForward(config WalkForwardConfig, btConfig BacktestConfig) *WalkForward {
	return &WalkForward{
		config:   config,
		btConfig: btConfig,
		barData:  make(map[string][]Bar),
	}
}

// LoadBars loads bar data.
func (wf *WalkForward) LoadBars(data map[string][]Bar) {
	wf.barData = data
}

// Run executes the walk-forward analysis with a strategy constructor.
func (wf *WalkForward) Run(newStrategy func() Strategy) []WalkForwardResult {
	// Find total length (use first symbol)
	var totalLen int
	for _, bars := range wf.barData {
		totalLen = len(bars)
		break
	}
	trainLen := wf.config.TrainPeriod
	testLen := wf.config.TestPeriod
	step := wf.config.Step
	if step == 0 {
		step = testLen
	}
	var results []WalkForwardResult
	for start := 0; start+trainLen+testLen <= totalLen; start += step {
		trainEnd := start + trainLen
		testEnd := trainEnd + testLen
		// Split data
		trainData := make(map[string][]Bar)
		testData := make(map[string][]Bar)
		for sym, bars := range wf.barData {
			if trainEnd <= len(bars) {
				trainData[sym] = bars[start:trainEnd]
			}
			if testEnd <= len(bars) {
				testData[sym] = bars[trainEnd:testEnd]
			}
		}
		// Train
		trainStrategy := newStrategy()
		trainEngine := NewEngine(wf.btConfig, trainStrategy)
		trainEngine.LoadBars(trainData)
		trainReport := trainEngine.Run()
		// Test
		testStrategy := newStrategy()
		testEngine := NewEngine(wf.btConfig, testStrategy)
		testEngine.LoadBars(testData)
		testReport := testEngine.Run()
		results = append(results, WalkForwardResult{
			TrainStart:  start,
			TrainEnd:    trainEnd,
			TestStart:   trainEnd,
			TestEnd:     testEnd,
			TrainReport: trainReport,
			TestReport:  testReport,
		})
	}
	return results
}

// WalkForwardSummary summarizes walk-forward results.
type WalkForwardSummary struct {
	Windows        int     `json:"windows"`
	AvgTrainSharpe float64 `json:"avg_train_sharpe"`
	AvgTestSharpe  float64 `json:"avg_test_sharpe"`
	TrainTestRatio float64 `json:"train_test_ratio"`
	Stability      float64 `json:"stability"`
}

// Summarize produces a summary of walk-forward results.
func Summarize(results []WalkForwardResult) WalkForwardSummary {
	if len(results) == 0 {
		return WalkForwardSummary{}
	}
	var trainSharpes, testSharpes []float64
	for _, r := range results {
		trainSharpes = append(trainSharpes, r.TrainReport.Analytics.Sharpe)
		testSharpes = append(testSharpes, r.TestReport.Analytics.Sharpe)
	}
	avgTrain := mean(trainSharpes)
	avgTest := mean(testSharpes)
	ratio := 0.0
	if avgTrain != 0 {
		ratio = avgTest / avgTrain
	}
	// Stability: fraction of windows with positive test Sharpe
	positive := 0
	for _, s := range testSharpes {
		if s > 0 {
			positive++
		}
	}
	stability := float64(positive) / float64(len(testSharpes))
	return WalkForwardSummary{
		Windows:        len(results),
		AvgTrainSharpe: avgTrain,
		AvgTestSharpe:  avgTest,
		TrainTestRatio: ratio,
		Stability:      stability,
	}
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	s := 0.0
	for _, v := range vals {
		s += v
	}
	return s / float64(len(vals))
}

// ---------------------------------------------------------------------------
// MonteCarloBootstrap: resample trades, confidence intervals
// ---------------------------------------------------------------------------

// MonteCarloConfig configures bootstrap simulation.
type MonteCarloConfig struct {
	NumSimulations int
	BlockSize      int // for block bootstrap
	Seed           int64
}

// MonteCarloResult holds bootstrap results.
type MonteCarloResult struct {
	SimulatedSharpes  []float64          `json:"simulated_sharpes"`
	SimulatedReturns  []float64          `json:"simulated_returns"`
	SimulatedMaxDDs   []float64          `json:"simulated_max_dds"`
	MeanSharpe        float64            `json:"mean_sharpe"`
	MedianSharpe      float64            `json:"median_sharpe"`
	Sharpe5th         float64            `json:"sharpe_5th"`
	Sharpe95th        float64            `json:"sharpe_95th"`
	MeanReturn        float64            `json:"mean_return"`
	MeanMaxDD         float64            `json:"mean_max_dd"`
	ProbPositive      float64            `json:"prob_positive_return"`
	ConfidenceIntervals map[string][2]float64 `json:"confidence_intervals"`
}

// MonteCarloBootstrap runs bootstrap analysis.
type MonteCarloBootstrap struct {
	config MonteCarloConfig
}

// NewMonteCarloBootstrap creates a bootstrap analyzer.
func NewMonteCarloBootstrap(config MonteCarloConfig) *MonteCarloBootstrap {
	return &MonteCarloBootstrap{config: config}
}

// Run executes bootstrap analysis on returns.
func (mc *MonteCarloBootstrap) Run(returns []float64, rf float64) MonteCarloResult {
	result := MonteCarloResult{
		ConfidenceIntervals: make(map[string][2]float64),
	}
	n := len(returns)
	if n == 0 {
		return result
	}
	seed := mc.config.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))
	numSim := mc.config.NumSimulations
	if numSim == 0 {
		numSim = 1000
	}
	blockSize := mc.config.BlockSize
	if blockSize == 0 {
		blockSize = 1
	}
	simSharpes := make([]float64, numSim)
	simReturns := make([]float64, numSim)
	simMaxDDs := make([]float64, numSim)
	for sim := 0; sim < numSim; sim++ {
		// Block bootstrap
		simRet := make([]float64, 0, n)
		for len(simRet) < n {
			start := rng.Intn(n - blockSize + 1)
			for j := 0; j < blockSize && len(simRet) < n; j++ {
				simRet = append(simRet, returns[start+j])
			}
		}
		// Compute metrics
		a := computeAnalytics(simRet, nil, rf)
		simSharpes[sim] = a.Sharpe
		simReturns[sim] = a.TotalReturn
		simMaxDDs[sim] = a.MaxDrawdown
	}
	result.SimulatedSharpes = simSharpes
	result.SimulatedReturns = simReturns
	result.SimulatedMaxDDs = simMaxDDs
	// Statistics
	sort.Float64s(simSharpes)
	sort.Float64s(simReturns)
	sort.Float64s(simMaxDDs)
	result.MeanSharpe = mean(simSharpes)
	result.MedianSharpe = percentile(simSharpes, 50)
	result.Sharpe5th = percentile(simSharpes, 5)
	result.Sharpe95th = percentile(simSharpes, 95)
	result.MeanReturn = mean(simReturns)
	result.MeanMaxDD = mean(simMaxDDs)
	// Prob positive
	posCount := 0
	for _, r := range simReturns {
		if r > 0 {
			posCount++
		}
	}
	result.ProbPositive = float64(posCount) / float64(numSim)
	// Confidence intervals
	result.ConfidenceIntervals["sharpe_90"] = [2]float64{
		percentile(simSharpes, 5), percentile(simSharpes, 95),
	}
	result.ConfidenceIntervals["return_90"] = [2]float64{
		percentile(simReturns, 5), percentile(simReturns, 95),
	}
	result.ConfidenceIntervals["maxdd_90"] = [2]float64{
		percentile(simMaxDDs, 5), percentile(simMaxDDs, 95),
	}
	return result
}

func percentile(sorted []float64, pct float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := pct / 100.0 * float64(len(sorted)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi || hi >= len(sorted) {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

// ---------------------------------------------------------------------------
// Simple built-in strategies
// ---------------------------------------------------------------------------

// BuyAndHoldStrategy holds equal weight across all symbols.
type BuyAndHoldStrategy struct {
	symbols []string
	entered bool
}

func NewBuyAndHoldStrategy() *BuyAndHoldStrategy {
	return &BuyAndHoldStrategy{}
}

func (s *BuyAndHoldStrategy) Name() string { return "buy_and_hold" }

func (s *BuyAndHoldStrategy) OnInit(config BacktestConfig) {
	s.symbols = config.Symbols
}

func (s *BuyAndHoldStrategy) OnBar(bars map[string]Bar, _ map[string]*Position, _ float64) []OrderSignal {
	if s.entered {
		return nil
	}
	s.entered = true
	n := len(s.symbols)
	if n == 0 {
		return nil
	}
	signals := make([]OrderSignal, 0, n)
	w := 1.0 / float64(n)
	for _, sym := range s.symbols {
		if _, ok := bars[sym]; ok {
			signals = append(signals, OrderSignal{Symbol: sym, Weight: w})
		}
	}
	return signals
}

// MomentumStrategy buys top N momentum stocks.
type MomentumStrategy struct {
	symbols  []string
	lookback int
	topN     int
	history  map[string][]float64
	barCount int
	rebalFreq int
}

func NewMomentumStrategy(lookback, topN, rebalFreq int) *MomentumStrategy {
	return &MomentumStrategy{
		lookback:  lookback,
		topN:      topN,
		rebalFreq: rebalFreq,
		history:   make(map[string][]float64),
	}
}

func (s *MomentumStrategy) Name() string { return "momentum" }

func (s *MomentumStrategy) OnInit(config BacktestConfig) {
	s.symbols = config.Symbols
	for _, sym := range s.symbols {
		s.history[sym] = make([]float64, 0)
	}
}

func (s *MomentumStrategy) OnBar(bars map[string]Bar, _ map[string]*Position, _ float64) []OrderSignal {
	s.barCount++
	for _, sym := range s.symbols {
		if b, ok := bars[sym]; ok {
			s.history[sym] = append(s.history[sym], b.Close)
		}
	}
	if s.barCount < s.lookback || s.barCount%s.rebalFreq != 0 {
		return nil
	}
	type momScore struct {
		sym string
		mom float64
	}
	var scores []momScore
	for _, sym := range s.symbols {
		h := s.history[sym]
		if len(h) >= s.lookback {
			old := h[len(h)-s.lookback]
			if old > 0 {
				mom := h[len(h)-1]/old - 1
				scores = append(scores, momScore{sym, mom})
			}
		}
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].mom > scores[j].mom
	})
	signals := make([]OrderSignal, 0)
	topN := s.topN
	if topN > len(scores) {
		topN = len(scores)
	}
	selected := make(map[string]bool)
	w := 1.0 / float64(topN)
	for i := 0; i < topN; i++ {
		signals = append(signals, OrderSignal{Symbol: scores[i].sym, Weight: w})
		selected[scores[i].sym] = true
	}
	// Zero out non-selected
	for _, sym := range s.symbols {
		if !selected[sym] {
			signals = append(signals, OrderSignal{Symbol: sym, Weight: 0})
		}
	}
	return signals
}

// MeanReversionStrategy buys oversold and sells overbought.
type MeanReversionStrategy struct {
	symbols   []string
	period    int
	threshold float64
	history   map[string][]float64
	barCount  int
}

func NewMeanReversionStrategy(period int, threshold float64) *MeanReversionStrategy {
	return &MeanReversionStrategy{
		period:    period,
		threshold: threshold,
		history:   make(map[string][]float64),
	}
}

func (s *MeanReversionStrategy) Name() string { return "mean_reversion" }

func (s *MeanReversionStrategy) OnInit(config BacktestConfig) {
	s.symbols = config.Symbols
	for _, sym := range s.symbols {
		s.history[sym] = make([]float64, 0)
	}
}

func (s *MeanReversionStrategy) OnBar(bars map[string]Bar, _ map[string]*Position, _ float64) []OrderSignal {
	s.barCount++
	for _, sym := range s.symbols {
		if b, ok := bars[sym]; ok {
			s.history[sym] = append(s.history[sym], b.Close)
		}
	}
	if s.barCount < s.period {
		return nil
	}
	var signals []OrderSignal
	n := float64(len(s.symbols))
	for _, sym := range s.symbols {
		h := s.history[sym]
		if len(h) < s.period {
			continue
		}
		window := h[len(h)-s.period:]
		m := mean(window)
		sd := 0.0
		for _, v := range window {
			d := v - m
			sd += d * d
		}
		sd = math.Sqrt(sd / float64(s.period-1))
		if sd == 0 {
			continue
		}
		z := (h[len(h)-1] - m) / sd
		weight := 0.0
		if z < -s.threshold {
			weight = 1.0 / n // buy oversold
		} else if z > s.threshold {
			weight = 0 // sell overbought
		} else {
			weight = (1.0 / n) * (1 - z/s.threshold) / 2
		}
		signals = append(signals, OrderSignal{Symbol: sym, Weight: weight})
	}
	return signals
}

// ---------------------------------------------------------------------------
// Multi-backtest runner
// ---------------------------------------------------------------------------

// MultiBacktestResult holds results from multiple strategy backtests.
type MultiBacktestResult struct {
	Reports []Report `json:"reports"`
}

// RunMultiple runs multiple strategies on the same data.
func RunMultiple(config BacktestConfig, data map[string][]Bar, strategies []Strategy) MultiBacktestResult {
	var mu sync.Mutex
	var wg sync.WaitGroup
	reports := make([]Report, len(strategies))
	for i, strat := range strategies {
		wg.Add(1)
		go func(idx int, s Strategy) {
			defer wg.Done()
			engine := NewEngine(config, s)
			engine.LoadBars(data)
			report := engine.Run()
			mu.Lock()
			reports[idx] = report
			mu.Unlock()
		}(i, strat)
	}
	wg.Wait()
	return MultiBacktestResult{Reports: reports}
}

// CompareStrategies produces a comparison table.
func CompareStrategies(results MultiBacktestResult) []map[string]interface{} {
	var rows []map[string]interface{}
	for _, r := range results.Reports {
		row := map[string]interface{}{
			"strategy":     r.Strategy,
			"total_return": r.Analytics.TotalReturn,
			"sharpe":       r.Analytics.Sharpe,
			"sortino":      r.Analytics.Sortino,
			"max_drawdown": r.Analytics.MaxDrawdown,
			"calmar":       r.Analytics.Calmar,
			"win_rate":     r.Analytics.WinRate,
			"trades":       r.Analytics.TotalTrades,
			"final_equity": r.FinalEquity,
		}
		rows = append(rows, row)
	}
	return rows
}

// ---------------------------------------------------------------------------
// EquityCurveAnalysis: detailed equity curve analysis
// ---------------------------------------------------------------------------

// EquityCurveStats holds detailed equity curve statistics.
type EquityCurveStats struct {
	CAGR               float64   `json:"cagr"`
	TotalDays          int       `json:"total_days"`
	PositiveDays       int       `json:"positive_days"`
	NegativeDays       int       `json:"negative_days"`
	BestDay            float64   `json:"best_day"`
	WorstDay           float64   `json:"worst_day"`
	BestMonth          float64   `json:"best_month"`
	WorstMonth         float64   `json:"worst_month"`
	AvgDailyReturn     float64   `json:"avg_daily_return"`
	DailyVolatility    float64   `json:"daily_volatility"`
	UlcerIndex         float64   `json:"ulcer_index"`
	PainIndex          float64   `json:"pain_index"`
	BurkeRatio         float64   `json:"burke_ratio"`
	MartinRatio        float64   `json:"martin_ratio"`
	TailRatio          float64   `json:"tail_ratio"`
	CommonSenseRatio   float64   `json:"common_sense_ratio"`
	ConsecutiveWins    int       `json:"consecutive_wins"`
	ConsecutiveLosses  int       `json:"consecutive_losses"`
	RecoveryFactor     float64   `json:"recovery_factor"`
	PayoffRatio        float64   `json:"payoff_ratio"`
	ExpectedReturn     float64   `json:"expected_return"`
	KellyFraction      float64   `json:"kelly_fraction"`
	DrawdownPeriods    []DrawdownInfo `json:"drawdown_periods"`
}

// DrawdownInfo describes a single drawdown episode.
type DrawdownInfo struct {
	Start    int     `json:"start"`
	Trough   int     `json:"trough"`
	End      int     `json:"end"`
	Depth    float64 `json:"depth"`
	Duration int     `json:"duration"`
	Recovery int     `json:"recovery"`
}

// AnalyzeEquityCurve computes detailed statistics from an equity curve.
func AnalyzeEquityCurve(equityCurve []float64, returns []float64, rf float64) EquityCurveStats {
	stats := EquityCurveStats{}
	n := len(returns)
	if n == 0 {
		return stats
	}
	stats.TotalDays = n
	// Daily stats
	bestDay := -math.MaxFloat64
	worstDay := math.MaxFloat64
	sumRet := 0.0
	for _, r := range returns {
		sumRet += r
		if r > bestDay {
			bestDay = r
		}
		if r < worstDay {
			worstDay = r
		}
		if r > 0 {
			stats.PositiveDays++
		} else if r < 0 {
			stats.NegativeDays++
		}
	}
	stats.BestDay = bestDay
	stats.WorstDay = worstDay
	stats.AvgDailyReturn = sumRet / float64(n)
	// Daily vol
	ss := 0.0
	for _, r := range returns {
		d := r - stats.AvgDailyReturn
		ss += d * d
	}
	if n > 1 {
		stats.DailyVolatility = math.Sqrt(ss / float64(n-1))
	}
	// CAGR
	years := float64(n) / 252
	totalRet := math.Exp(sumRet) - 1
	if years > 0 && totalRet > -1 {
		stats.CAGR = math.Pow(1+totalRet, 1/years) - 1
	}
	// Monthly returns
	monthLen := 21
	bestMonth := -math.MaxFloat64
	worstMonth := math.MaxFloat64
	for i := 0; i < n; i += monthLen {
		end := i + monthLen
		if end > n {
			end = n
		}
		mRet := 0.0
		for j := i; j < end; j++ {
			mRet += returns[j]
		}
		if mRet > bestMonth {
			bestMonth = mRet
		}
		if mRet < worstMonth {
			worstMonth = mRet
		}
	}
	stats.BestMonth = bestMonth
	stats.WorstMonth = worstMonth
	// Drawdown analysis
	cum := 0.0
	peak := 0.0
	maxDD := 0.0
	ddSquaredSum := 0.0
	ddSum := 0.0
	ddCount := 0
	inDD := false
	ddStart := 0
	ddTrough := 0
	ddDepth := 0.0
	var ddPeriods []DrawdownInfo
	for i, r := range returns {
		cum += r
		if cum > peak {
			if inDD {
				ddPeriods = append(ddPeriods, DrawdownInfo{
					Start:    ddStart,
					Trough:   ddTrough,
					End:      i,
					Depth:    ddDepth,
					Duration: i - ddStart,
					Recovery: i - ddTrough,
				})
				inDD = false
			}
			peak = cum
		}
		dd := peak - cum
		if dd > 0 {
			if !inDD {
				inDD = true
				ddStart = i
				ddTrough = i
				ddDepth = dd
			}
			if dd > ddDepth {
				ddDepth = dd
				ddTrough = i
			}
			ddSquaredSum += dd * dd
			ddSum += dd
			ddCount++
		}
		if dd > maxDD {
			maxDD = dd
		}
	}
	if inDD {
		ddPeriods = append(ddPeriods, DrawdownInfo{
			Start:    ddStart,
			Trough:   ddTrough,
			End:      n - 1,
			Depth:    ddDepth,
			Duration: n - 1 - ddStart,
		})
	}
	stats.DrawdownPeriods = ddPeriods
	// Ulcer Index
	if n > 0 {
		stats.UlcerIndex = math.Sqrt(ddSquaredSum / float64(n))
	}
	// Pain Index
	if n > 0 {
		stats.PainIndex = ddSum / float64(n)
	}
	// Martin Ratio (return / ulcer index)
	annRet := stats.CAGR
	if stats.UlcerIndex > 0 {
		stats.MartinRatio = (annRet - rf) / stats.UlcerIndex
	}
	// Burke Ratio
	if len(ddPeriods) > 0 {
		sqSum := 0.0
		for _, d := range ddPeriods {
			sqSum += d.Depth * d.Depth
		}
		burkeDD := math.Sqrt(sqSum / float64(len(ddPeriods)))
		if burkeDD > 0 {
			stats.BurkeRatio = (annRet - rf) / burkeDD
		}
	}
	// Consecutive wins/losses
	maxConsWin := 0
	maxConsLoss := 0
	curWin := 0
	curLoss := 0
	for _, r := range returns {
		if r > 0 {
			curWin++
			curLoss = 0
		} else if r < 0 {
			curLoss++
			curWin = 0
		} else {
			curWin = 0
			curLoss = 0
		}
		if curWin > maxConsWin {
			maxConsWin = curWin
		}
		if curLoss > maxConsLoss {
			maxConsLoss = curLoss
		}
	}
	stats.ConsecutiveWins = maxConsWin
	stats.ConsecutiveLosses = maxConsLoss
	// Recovery factor
	if maxDD > 0 {
		stats.RecoveryFactor = totalRet / maxDD
	}
	// Payoff ratio
	avgWin := 0.0
	avgLoss := 0.0
	winCount := 0
	lossCount := 0
	for _, r := range returns {
		if r > 0 {
			avgWin += r
			winCount++
		} else if r < 0 {
			avgLoss -= r
			lossCount++
		}
	}
	if winCount > 0 {
		avgWin /= float64(winCount)
	}
	if lossCount > 0 {
		avgLoss /= float64(lossCount)
	}
	if avgLoss > 0 {
		stats.PayoffRatio = avgWin / avgLoss
	}
	// Tail ratio (95th / 5th percentile)
	sorted := make([]float64, n)
	copy(sorted, returns)
	sort.Float64s(sorted)
	p5 := sorted[int(0.05*float64(n))]
	p95 := sorted[int(0.95*float64(n))]
	if p5 != 0 {
		stats.TailRatio = math.Abs(p95 / p5)
	}
	// Common sense ratio = profit factor * tail ratio
	grossProfit := 0.0
	grossLoss := 0.0
	for _, r := range returns {
		if r > 0 {
			grossProfit += r
		} else {
			grossLoss -= r
		}
	}
	pf := 0.0
	if grossLoss > 0 {
		pf = grossProfit / grossLoss
	}
	stats.CommonSenseRatio = pf * stats.TailRatio
	// Expected return per trade
	winRate := float64(winCount) / float64(n)
	stats.ExpectedReturn = winRate*avgWin - (1-winRate)*avgLoss
	// Kelly fraction
	if avgLoss > 0 {
		stats.KellyFraction = winRate - (1-winRate)/(avgWin/avgLoss)
	}
	return stats
}

// ---------------------------------------------------------------------------
// RollingMetrics: computes rolling performance metrics
// ---------------------------------------------------------------------------

// RollingMetrics computes rolling statistics over the backtest.
type RollingMetrics struct {
	window int
}

// NewRollingMetrics creates a rolling metrics calculator.
func NewRollingMetrics(window int) *RollingMetrics {
	return &RollingMetrics{window: window}
}

// RollingSharpe computes rolling Sharpe ratio.
func (rm *RollingMetrics) RollingSharpe(returns []float64, rf float64) []float64 {
	n := len(returns)
	if n < rm.window {
		return nil
	}
	result := make([]float64, n-rm.window+1)
	for i := 0; i <= n-rm.window; i++ {
		window := returns[i : i+rm.window]
		m := mean(window)
		sd := 0.0
		for _, v := range window {
			d := v - m
			sd += d * d
		}
		if rm.window > 1 {
			sd = math.Sqrt(sd / float64(rm.window-1))
		}
		annRet := m * 252
		annVol := sd * math.Sqrt(252)
		if annVol > 0 {
			result[i] = (annRet - rf) / annVol
		}
	}
	return result
}

// RollingVol computes rolling volatility.
func (rm *RollingMetrics) RollingVol(returns []float64) []float64 {
	n := len(returns)
	if n < rm.window {
		return nil
	}
	result := make([]float64, n-rm.window+1)
	for i := 0; i <= n-rm.window; i++ {
		window := returns[i : i+rm.window]
		m := mean(window)
		sd := 0.0
		for _, v := range window {
			d := v - m
			sd += d * d
		}
		if rm.window > 1 {
			sd = math.Sqrt(sd / float64(rm.window-1))
		}
		result[i] = sd * math.Sqrt(252)
	}
	return result
}

// RollingDrawdown computes rolling maximum drawdown.
func (rm *RollingMetrics) RollingDrawdown(returns []float64) []float64 {
	n := len(returns)
	if n < rm.window {
		return nil
	}
	result := make([]float64, n-rm.window+1)
	for i := 0; i <= n-rm.window; i++ {
		window := returns[i : i+rm.window]
		cum := 0.0
		peak := 0.0
		maxDD := 0.0
		for _, r := range window {
			cum += r
			if cum > peak {
				peak = cum
			}
			dd := peak - cum
			if dd > maxDD {
				maxDD = dd
			}
		}
		result[i] = maxDD
	}
	return result
}

// RollingBeta computes rolling beta vs benchmark.
func (rm *RollingMetrics) RollingBeta(returns, benchmark []float64) []float64 {
	n := len(returns)
	if n < rm.window || len(benchmark) < n {
		return nil
	}
	result := make([]float64, n-rm.window+1)
	for i := 0; i <= n-rm.window; i++ {
		rWin := returns[i : i+rm.window]
		bWin := benchmark[i : i+rm.window]
		mR := mean(rWin)
		mB := mean(bWin)
		cov := 0.0
		varB := 0.0
		for j := 0; j < rm.window; j++ {
			dr := rWin[j] - mR
			db := bWin[j] - mB
			cov += dr * db
			varB += db * db
		}
		if varB > 0 {
			result[i] = cov / varB
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// TradeAnalyzer: detailed trade-level analysis
// ---------------------------------------------------------------------------

// TradeStats holds trade-level statistics.
type TradeStats struct {
	TotalTrades    int     `json:"total_trades"`
	BuyTrades      int     `json:"buy_trades"`
	SellTrades     int     `json:"sell_trades"`
	AvgTradeSize   float64 `json:"avg_trade_size"`
	MaxTradeSize   float64 `json:"max_trade_size"`
	TotalVolume    float64 `json:"total_volume"`
	TotalCosts     float64 `json:"total_costs"`
	CostPerTrade   float64 `json:"cost_per_trade"`
	CostBps        float64 `json:"cost_bps"`
	AvgSlippage    float64 `json:"avg_slippage_bps"`
	TradesPerDay   float64 `json:"trades_per_day"`
	SymbolBreakdown map[string]int `json:"symbol_breakdown"`
}

// AnalyzeTrades computes trade-level statistics.
func AnalyzeTrades(trades []Trade, totalDays int) TradeStats {
	stats := TradeStats{
		SymbolBreakdown: make(map[string]int),
	}
	if len(trades) == 0 {
		return stats
	}
	stats.TotalTrades = len(trades)
	totalNotional := 0.0
	maxNotional := 0.0
	totalSlip := 0.0
	for _, t := range trades {
		notional := t.Quantity * t.Price
		totalNotional += notional
		if notional > maxNotional {
			maxNotional = notional
		}
		stats.TotalCosts += t.Cost
		totalSlip += t.SlippageBps
		stats.SymbolBreakdown[t.Symbol]++
		if t.Side == "buy" {
			stats.BuyTrades++
		} else {
			stats.SellTrades++
		}
	}
	stats.TotalVolume = totalNotional
	stats.AvgTradeSize = totalNotional / float64(len(trades))
	stats.MaxTradeSize = maxNotional
	stats.CostPerTrade = stats.TotalCosts / float64(len(trades))
	if totalNotional > 0 {
		stats.CostBps = stats.TotalCosts / totalNotional * 10000
	}
	stats.AvgSlippage = totalSlip / float64(len(trades))
	if totalDays > 0 {
		stats.TradesPerDay = float64(len(trades)) / float64(totalDays)
	}
	return stats
}

// ---------------------------------------------------------------------------
// ParameterSensitivity: parameter grid sensitivity analysis
// ---------------------------------------------------------------------------

// ParameterGrid defines a parameter to sweep.
type ParameterGrid struct {
	Name   string
	Values []float64
}

// SensitivityResult holds results from parameter sensitivity analysis.
type SensitivityResult struct {
	ParamName  string             `json:"param_name"`
	ParamValue float64            `json:"param_value"`
	Sharpe     float64            `json:"sharpe"`
	Return     float64            `json:"return"`
	MaxDD      float64            `json:"max_dd"`
	Trades     int                `json:"trades"`
}

// RunSensitivity runs a strategy across parameter values.
func RunSensitivity(config BacktestConfig, data map[string][]Bar, grid ParameterGrid, newStrategy func(paramVal float64) Strategy) []SensitivityResult {
	var mu sync.Mutex
	var wg sync.WaitGroup
	results := make([]SensitivityResult, len(grid.Values))
	for i, val := range grid.Values {
		wg.Add(1)
		go func(idx int, v float64) {
			defer wg.Done()
			strat := newStrategy(v)
			engine := NewEngine(config, strat)
			engine.LoadBars(data)
			report := engine.Run()
			mu.Lock()
			results[idx] = SensitivityResult{
				ParamName:  grid.Name,
				ParamValue: v,
				Sharpe:     report.Analytics.Sharpe,
				Return:     report.Analytics.TotalReturn,
				MaxDD:      report.Analytics.MaxDrawdown,
				Trades:     report.Analytics.TotalTrades,
			}
			mu.Unlock()
		}(i, val)
	}
	wg.Wait()
	return results
}

// BestParameter returns the parameter value with the best Sharpe.
func BestParameter(results []SensitivityResult) SensitivityResult {
	best := SensitivityResult{Sharpe: -math.MaxFloat64}
	for _, r := range results {
		if r.Sharpe > best.Sharpe {
			best = r
		}
	}
	return best
}

// ParameterStability returns the standard deviation of Sharpes across parameter values.
func ParameterStability(results []SensitivityResult) float64 {
	if len(results) < 2 {
		return 0
	}
	sharpes := make([]float64, len(results))
	for i, r := range results {
		sharpes[i] = r.Sharpe
	}
	m := mean(sharpes)
	ss := 0.0
	for _, s := range sharpes {
		d := s - m
		ss += d * d
	}
	return math.Sqrt(ss / float64(len(sharpes)-1))
}
