package main

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// ──────────────────────────────────────────────────────────────────────────────
// Synthetic market data (for standalone operation)
// ──────────────────────────────────────────────────────────────────────────────

type Bar struct {
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume float64
	Date   string
}

// generateBars produces synthetic daily bars using geometric Brownian motion.
func generateBars(symbol string, nBars int, seed int64) []Bar {
	bars := make([]Bar, nBars)
	price := 100.0 + float64(hashStr(symbol)%50)
	vol := 0.02 + float64(hashStr(symbol)%10)*0.002
	drift := 0.0003

	rng := seed ^ int64(hashStr(symbol))
	for i := 0; i < nBars; i++ {
		rng = lcg(rng)
		z := boxMullerOne(rng)
		ret := drift + vol*z
		newPrice := price * math.Exp(ret)

		hi := math.Max(price, newPrice) * (1 + math.Abs(z)*0.005)
		lo := math.Min(price, newPrice) * (1 - math.Abs(z)*0.005)
		rng = lcg(rng)
		v := 1e6 * (1 + math.Abs(z)*0.5)

		bars[i] = Bar{
			Open:   price,
			High:   hi,
			Low:    lo,
			Close:  newPrice,
			Volume: v,
			Date:   syntheticDate(i),
		}
		price = newPrice
	}
	return bars
}

func syntheticDate(barIdx int) string {
	year := 2020 + barIdx/252
	dayOfYear := barIdx%252 + 1
	month := dayOfYear / 21
	if month > 11 {
		month = 11
	}
	day := dayOfYear%21 + 1
	return fmt.Sprintf("%04d-%02d-%02d", year, month+1, day)
}

func lcg(s int64) int64 {
	return s*6364136223846793005 + 1442695040888963407
}

func boxMullerOne(seed int64) float64 {
	// Approximate standard normal from LCG
	u1 := float64((seed>>16)&0x7FFFFFFF) / float64(0x7FFFFFFF)
	s2 := lcg(seed)
	u2 := float64((s2>>16)&0x7FFFFFFF) / float64(0x7FFFFFFF)
	if u1 < 1e-10 {
		u1 = 1e-10
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func hashStr(s string) int {
	h := 0
	for _, c := range s {
		h = h*31 + int(c)
	}
	if h < 0 {
		h = -h
	}
	return h
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal generators
// ──────────────────────────────────────────────────────────────────────────────

type Signal struct {
	Symbol    string
	Direction float64 // -1 to +1
	Strength  float64 // 0 to 1
}

func momentumSignal(bars []Bar, idx int, lookback int) Signal {
	if idx < lookback {
		return Signal{}
	}
	ret := (bars[idx].Close - bars[idx-lookback].Close) / bars[idx-lookback].Close
	dir := math.Copysign(1, ret)
	str := math.Min(math.Abs(ret)/0.10, 1.0)
	return Signal{Direction: dir, Strength: str}
}

func meanRevSignal(bars []Bar, idx int, lookback int) Signal {
	if idx < lookback {
		return Signal{}
	}
	sum := 0.0
	for i := idx - lookback + 1; i <= idx; i++ {
		sum += bars[i].Close
	}
	sma := sum / float64(lookback)
	dev := (bars[idx].Close - sma) / sma
	dir := -math.Copysign(1, dev) // Mean reversion: sell above, buy below
	str := math.Min(math.Abs(dev)/0.05, 1.0)
	return Signal{Direction: dir, Strength: str}
}

func breakoutSignal(bars []Bar, idx int, lookback int) Signal {
	if idx < lookback {
		return Signal{}
	}
	hi, lo := bars[idx-lookback].High, bars[idx-lookback].Low
	for i := idx - lookback + 1; i < idx; i++ {
		if bars[i].High > hi {
			hi = bars[i].High
		}
		if bars[i].Low < lo {
			lo = bars[i].Low
		}
	}
	price := bars[idx].Close
	rng := hi - lo
	if rng <= 0 {
		return Signal{}
	}
	if price > hi {
		return Signal{Direction: 1, Strength: math.Min((price-hi)/rng, 1.0)}
	}
	if price < lo {
		return Signal{Direction: -1, Strength: math.Min((lo-price)/rng, 1.0)}
	}
	return Signal{}
}

func generateSignal(strategy string, bars []Bar, idx int, params map[string]float64) Signal {
	lb := int(params["lookback"])
	if lb <= 0 {
		lb = 20
	}
	switch strategy {
	case "momentum":
		return momentumSignal(bars, idx, lb)
	case "mean_rev":
		return meanRevSignal(bars, idx, lb)
	case "breakout":
		return breakoutSignal(bars, idx, lb)
	default:
		return momentumSignal(bars, idx, lb)
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Slippage models
// ──────────────────────────────────────────────────────────────────────────────

func computeSlippage(cost CostModel, price, qty float64) float64 {
	slip := cost.FixedSlippage
	slip += price * cost.PropSlippage
	if cost.SqrtImpactCoeff > 0 {
		slip += cost.SqrtImpactCoeff * math.Sqrt(math.Abs(qty))
	}
	return slip
}

func computeCommission(cost CostModel, qty float64) float64 {
	return math.Abs(qty) * cost.CommissionPerShare
}

// ──────────────────────────────────────────────────────────────────────────────
// Position sizing
// ──────────────────────────────────────────────────────────────────────────────

func computePositionSize(cfg SizingConfig, equity, price, volatility, winRate float64) float64 {
	if price <= 0 {
		return 0
	}
	var shares float64

	switch cfg.Method {
	case "vol_target":
		if volatility <= 0 {
			volatility = 0.02
		}
		target := cfg.VolTarget
		if target <= 0 {
			target = 0.01
		}
		dollarVol := price * volatility
		riskBudget := equity * target
		shares = riskBudget / dollarVol

	case "kelly":
		// Simplified Kelly: f = (p*b - q) / b where b = avg_win/avg_loss
		kf := cfg.KellyFrac
		if kf <= 0 {
			kf = 0.25
		}
		// Use half-Kelly by default
		frac := kf * 0.5
		shares = (equity * frac) / price

	default: // fixed_frac
		frac := cfg.FixedFrac
		if frac <= 0 {
			frac = 0.02
		}
		shares = (equity * frac) / price
	}

	// Max position limit
	maxPct := cfg.MaxPosPct
	if maxPct <= 0 {
		maxPct = 0.20
	}
	maxShares := (equity * maxPct) / price
	if shares > maxShares {
		shares = maxShares
	}
	return math.Floor(shares)
}

// ──────────────────────────────────────────────────────────────────────────────
// Realized volatility helper
// ──────────────────────────────────────────────────────────────────────────────

func rollingVol(bars []Bar, idx, window int) float64 {
	if idx < window {
		return 0.02 // default
	}
	rets := make([]float64, window)
	for i := 0; i < window; i++ {
		prev := bars[idx-window+i].Close
		cur := bars[idx-window+i+1].Close
		if prev > 0 {
			rets[i] = math.Log(cur / prev)
		}
	}
	return stddev(rets)
}

// ──────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ──────────────────────────────────────────────────────────────────────────────

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func stddev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := mean(xs)
	s := 0.0
	for _, x := range xs {
		d := x - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(xs)-1))
}

func downdev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	s := 0.0
	n := 0
	for _, x := range xs {
		if x < 0 {
			s += x * x
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(s / float64(n))
}

func skewness(xs []float64) float64 {
	n := float64(len(xs))
	if n < 3 {
		return 0
	}
	m := mean(xs)
	sd := stddev(xs)
	if sd == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += math.Pow((x-m)/sd, 3)
	}
	return (n / ((n - 1) * (n - 2))) * s
}

func kurtosis(xs []float64) float64 {
	n := float64(len(xs))
	if n < 4 {
		return 0
	}
	m := mean(xs)
	sd := stddev(xs)
	if sd == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += math.Pow((x-m)/sd, 4)
	}
	return (n*(n+1))/((n-1)*(n-2)*(n-3))*s - 3*(n-1)*(n-1)/((n-2)*(n-3))
}

func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	idx := p * float64(n-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi || hi >= n {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func absFloat(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// ──────────────────────────────────────────────────────────────────────────────
// Backtest engine
// ──────────────────────────────────────────────────────────────────────────────

type BacktestEngine struct {
	mu sync.Mutex
}

func NewBacktestEngine() *BacktestEngine {
	return &BacktestEngine{}
}

type positionState struct {
	symbol   string
	qty      float64
	avgCost  float64
	entryBar int
}

func (e *BacktestEngine) Run(cfg BacktestConfig, result *BacktestResult, store *BacktestStore) error {
	// Generate synthetic data for each symbol
	nBars := 504 // ~2 years
	if lb, ok := cfg.Parameters["num_bars"]; ok && lb > 0 {
		nBars = int(lb)
	}

	allBars := make(map[string][]Bar)
	seed := int64(42)
	for _, sym := range cfg.Symbols {
		allBars[sym] = generateBars(sym, nBars, seed)
		seed = lcg(seed)
	}

	if cfg.WalkForward != nil {
		return e.runWalkForward(cfg, allBars, nBars, result, store)
	}

	metrics, equity, trades := e.runSinglePass(cfg, allBars, nBars, 0, nBars, store, result)
	result.Metrics = metrics
	result.EquityCurve = equity
	result.Trades = trades
	result.MonthlyReturns = computeMonthlyReturns(equity)

	// Monte Carlo if requested
	if cfg.MonteCarlo != nil && len(trades) > 0 {
		result.MonteCarlo = runMonteCarlo(trades, cfg.InitialCash, cfg.MonteCarlo)
	}

	store.UpdateProgress(cfg.ID, 1.0)
	return nil
}

func (e *BacktestEngine) runSinglePass(cfg BacktestConfig, allBars map[string][]Bar, nBars, startBar, endBar int, store *BacktestStore, result *BacktestResult) (BacktestMetrics, []EquityPoint, []Trade) {
	cash := cfg.InitialCash
	positions := make(map[string]*positionState)
	var trades []Trade
	var equity []EquityPoint
	var dailyReturns []float64
	prevEquity := cash
	hwm := cash
	maxDD := 0.0
	maxDDDur := 0
	ddStart := 0
	totalTurnover := 0.0

	totalBars := endBar - startBar
	for bar := startBar; bar < endBar; bar++ {
		// Mark to market
		portfolioVal := cash
		for sym, pos := range positions {
			bars := allBars[sym]
			if bar < len(bars) {
				portfolioVal += pos.qty * bars[bar].Close
			}
		}

		// Equity curve
		dd := 0.0
		if portfolioVal > hwm {
			hwm = portfolioVal
			ddStart = bar
		} else if hwm > 0 {
			dd = (portfolioVal - hwm) / hwm
			if dd < maxDD {
				maxDD = dd
			}
			dur := bar - ddStart
			if dur > maxDDDur {
				maxDDDur = dur
			}
		}
		equity = append(equity, EquityPoint{
			Bar:      bar,
			Equity:   portfolioVal,
			Drawdown: dd,
			Date:     allBars[cfg.Symbols[0]][min(bar, len(allBars[cfg.Symbols[0]])-1)].Date,
		})

		if prevEquity > 0 {
			dailyReturns = append(dailyReturns, (portfolioVal-prevEquity)/prevEquity)
		}
		prevEquity = portfolioVal

		// Generate signals and trade
		for _, sym := range cfg.Symbols {
			bars := allBars[sym]
			if bar >= len(bars) {
				continue
			}
			sig := generateSignal(cfg.Strategy, bars, bar, cfg.Parameters)
			sig.Symbol = sym

			pos := positions[sym]
			price := bars[bar].Close
			vol := rollingVol(bars, bar, 20)

			// Entry logic
			if sig.Strength > 0.3 && pos == nil {
				qty := computePositionSize(cfg.Sizing, portfolioVal, price, vol, 0.5)
				if sig.Direction < 0 {
					qty = -qty
				}
				if qty == 0 {
					continue
				}
				slip := computeSlippage(cfg.CostModel, price, qty)
				comm := computeCommission(cfg.CostModel, qty)
				fillPrice := price + math.Copysign(slip, qty)
				cost := qty*fillPrice + comm

				if math.Abs(cost) > cash*0.95 {
					continue
				}
				cash -= cost
				positions[sym] = &positionState{symbol: sym, qty: qty, avgCost: fillPrice, entryBar: bar}
				totalTurnover += math.Abs(qty * fillPrice)

				trades = append(trades, Trade{
					Symbol: sym, Side: sideStr(qty), Quantity: qty,
					Price: fillPrice, Slippage: slip, Commission: comm, Bar: bar,
					Timestamp: bars[bar].Date,
				})
			}

			// Exit logic: signal reversal or stop
			if pos != nil {
				shouldExit := false
				if sig.Strength > 0.3 && math.Signbit(sig.Direction) != math.Signbit(pos.qty) {
					shouldExit = true
				}
				// Time stop: exit after 40 bars
				if bar-pos.entryBar > 40 {
					shouldExit = true
				}
				// Stop loss: -5%
				pnlPct := (price - pos.avgCost) / math.Abs(pos.avgCost)
				if pos.qty < 0 {
					pnlPct = -pnlPct
				}
				if pnlPct < -0.05 {
					shouldExit = true
				}

				if shouldExit {
					slip := computeSlippage(cfg.CostModel, price, -pos.qty)
					comm := computeCommission(cfg.CostModel, pos.qty)
					fillPrice := price - math.Copysign(slip, pos.qty)
					pnl := pos.qty*(fillPrice-pos.avgCost) - comm
					cash += pos.qty*fillPrice - comm
					totalTurnover += math.Abs(pos.qty * fillPrice)

					trades = append(trades, Trade{
						Symbol: sym, Side: sideStr(-pos.qty), Quantity: -pos.qty,
						Price: fillPrice, Slippage: slip, Commission: comm, Bar: bar,
						Timestamp: bars[bar].Date, PnL: pnl,
					})
					delete(positions, sym)
				}
			}
		}

		// Progress
		if store != nil && result != nil && totalBars > 0 {
			store.UpdateProgress(result.ID, float64(bar-startBar)/float64(totalBars))
		}
	}

	// Close remaining positions at last bar
	for sym, pos := range positions {
		bars := allBars[sym]
		lastBar := min(endBar-1, len(bars)-1)
		price := bars[lastBar].Close
		slip := computeSlippage(cfg.CostModel, price, -pos.qty)
		comm := computeCommission(cfg.CostModel, pos.qty)
		fillPrice := price - math.Copysign(slip, pos.qty)
		pnl := pos.qty*(fillPrice-pos.avgCost) - comm
		cash += pos.qty*fillPrice - comm

		trades = append(trades, Trade{
			Symbol: sym, Side: sideStr(-pos.qty), Quantity: -pos.qty,
			Price: fillPrice, Slippage: slip, Commission: comm, Bar: lastBar,
			Timestamp: bars[lastBar].Date, PnL: pnl,
		})
	}

	metrics := computeMetrics(dailyReturns, trades, cfg.InitialCash, cash, maxDD, maxDDDur, totalTurnover, float64(totalBars))

	return metrics, equity, trades
}

func (e *BacktestEngine) runWalkForward(cfg BacktestConfig, allBars map[string][]Bar, nBars int, result *BacktestResult, store *BacktestStore) error {
	wf := cfg.WalkForward
	if wf.NumFolds <= 0 {
		wf.NumFolds = 5
	}
	if wf.TrainBars <= 0 {
		wf.TrainBars = 252
	}
	if wf.TestBars <= 0 {
		wf.TestBars = 63
	}

	foldSize := wf.TrainBars + wf.TestBars
	var wfResults []WalkForwardResult
	var allEquity []EquityPoint
	var allTrades []Trade

	for fold := 0; fold < wf.NumFolds; fold++ {
		trainStart := fold * wf.TestBars
		trainEnd := trainStart + wf.TrainBars
		testStart := trainEnd
		testEnd := testStart + wf.TestBars

		if testEnd > nBars {
			break
		}

		// Optimize on train period: try different lookbacks
		bestSharpe := math.Inf(-1)
		bestLB := 20.0
		for lb := 10.0; lb <= 60; lb += 5 {
			params := copyParams(cfg.Parameters)
			params["lookback"] = lb
			trialCfg := cfg
			trialCfg.Parameters = params
			m, _, _ := e.runSinglePass(trialCfg, allBars, nBars, trainStart, trainEnd, nil, nil)
			if m.Sharpe > bestSharpe {
				bestSharpe = m.Sharpe
				bestLB = lb
			}
		}

		optParams := copyParams(cfg.Parameters)
		optParams["lookback"] = bestLB

		trainCfg := cfg
		trainCfg.Parameters = optParams
		trainMetrics, _, _ := e.runSinglePass(trainCfg, allBars, nBars, trainStart, trainEnd, nil, nil)

		testCfg := cfg
		testCfg.Parameters = optParams
		testMetrics, testEquity, testTrades := e.runSinglePass(testCfg, allBars, nBars, testStart, testEnd, nil, nil)

		wfResults = append(wfResults, WalkForwardResult{
			Fold:         fold,
			TrainMetrics: trainMetrics,
			TestMetrics:  testMetrics,
			OptParams:    optParams,
		})
		allEquity = append(allEquity, testEquity...)
		allTrades = append(allTrades, testTrades...)

		if store != nil {
			store.UpdateProgress(cfg.ID, float64(fold+1)/float64(wf.NumFolds))
		}

		_ = foldSize // suppress unused
	}

	result.WalkForward = wfResults
	result.EquityCurve = allEquity
	result.Trades = allTrades
	if len(allEquity) > 0 {
		var rets []float64
		for i := 1; i < len(allEquity); i++ {
			if allEquity[i-1].Equity > 0 {
				rets = append(rets, (allEquity[i].Equity-allEquity[i-1].Equity)/allEquity[i-1].Equity)
			}
		}
		result.Metrics = computeMetrics(rets, allTrades, cfg.InitialCash, allEquity[len(allEquity)-1].Equity, 0, 0, 0, float64(len(allEquity)))
	}
	result.MonthlyReturns = computeMonthlyReturns(allEquity)
	return nil
}

func copyParams(m map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func sideStr(qty float64) string {
	if qty >= 0 {
		return "buy"
	}
	return "sell"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ──────────────────────────────────────────────────────────────────────────────
// Metrics computation
// ──────────────────────────────────────────────────────────────────────────────

func computeMetrics(dailyReturns []float64, trades []Trade, initialCash, finalEquity, maxDD float64, maxDDDur int, turnover, nBars float64) BacktestMetrics {
	totalRet := 0.0
	if initialCash > 0 {
		totalRet = (finalEquity - initialCash) / initialCash
	}
	years := nBars / 252
	if years <= 0 {
		years = 1
	}
	annualRet := math.Pow(1+totalRet, 1/years) - 1

	vol := stddev(dailyReturns) * math.Sqrt(252)
	sharpe := 0.0
	if vol > 0 {
		sharpe = annualRet / vol
	}

	dd := downdev(dailyReturns) * math.Sqrt(252)
	sortino := 0.0
	if dd > 0 {
		sortino = annualRet / dd
	}

	calmar := 0.0
	if maxDD != 0 {
		calmar = annualRet / math.Abs(maxDD)
	}

	// Trade stats
	var wins, losses int
	var grossWin, grossLoss float64
	var totalBarsHeld int
	for _, t := range trades {
		if t.PnL > 0 {
			wins++
			grossWin += t.PnL
		} else if t.PnL < 0 {
			losses++
			grossLoss += math.Abs(t.PnL)
		}
	}
	numTrades := wins + losses
	winRate := 0.0
	if numTrades > 0 {
		winRate = float64(wins) / float64(numTrades)
	}
	pf := 0.0
	if grossLoss > 0 {
		pf = grossWin / grossLoss
	}
	avgWin := 0.0
	if wins > 0 {
		avgWin = grossWin / float64(wins)
	}
	avgLoss := 0.0
	if losses > 0 {
		avgLoss = grossLoss / float64(losses)
	}
	avgBars := 0.0
	if numTrades > 0 {
		avgBars = float64(totalBarsHeld) / float64(numTrades)
	}

	turnoverAnn := 0.0
	if initialCash > 0 && years > 0 {
		turnoverAnn = turnover / (initialCash * years)
	}

	return BacktestMetrics{
		TotalReturn:   totalRet,
		AnnualReturn:  annualRet,
		Sharpe:        sharpe,
		Sortino:       sortino,
		Calmar:        calmar,
		MaxDrawdown:   maxDD,
		MaxDDDuration: maxDDDur,
		WinRate:       winRate,
		ProfitFactor:  pf,
		AvgWin:        avgWin,
		AvgLoss:       avgLoss,
		NumTrades:     numTrades,
		AvgBarsHeld:   avgBars,
		Volatility:    vol,
		Skewness:      skewness(dailyReturns),
		Kurtosis:      kurtosis(dailyReturns),
		TurnoverAnn:   turnoverAnn,
	}
}

func computeMonthlyReturns(equity []EquityPoint) []MonthlyReturn {
	if len(equity) == 0 {
		return nil
	}
	type monthKey struct{ y, m int }
	first := make(map[monthKey]float64)
	last := make(map[monthKey]float64)
	var keys []monthKey

	for _, ep := range equity {
		// Parse "YYYY-MM-DD"
		if len(ep.Date) < 7 {
			continue
		}
		y, m := 0, 0
		fmt.Sscanf(ep.Date, "%d-%d", &y, &m)
		k := monthKey{y, m}
		if _, ok := first[k]; !ok {
			first[k] = ep.Equity
			keys = append(keys, k)
		}
		last[k] = ep.Equity
	}

	var monthly []MonthlyReturn
	for _, k := range keys {
		f := first[k]
		l := last[k]
		ret := 0.0
		if f > 0 {
			ret = (l - f) / f
		}
		monthly = append(monthly, MonthlyReturn{Year: k.y, Month: k.m, Ret: ret})
	}
	return monthly
}

// ──────────────────────────────────────────────────────────────────────────────
// Monte Carlo simulation
// ──────────────────────────────────────────────────────────────────────────────

func runMonteCarlo(trades []Trade, initialCash float64, cfg *MonteCarloCfg) *MonteCarloResult {
	if cfg.NumSims <= 0 {
		cfg.NumSims = 1000
	}
	if cfg.Confidence <= 0 {
		cfg.Confidence = 0.95
	}

	// Extract trade P&Ls
	pnls := make([]float64, 0, len(trades))
	for _, t := range trades {
		if t.PnL != 0 {
			pnls = append(pnls, t.PnL)
		}
	}
	if len(pnls) == 0 {
		return nil
	}

	nTrades := len(pnls)
	terminals := make([]float64, cfg.NumSims)
	maxDDs := make([]float64, cfg.NumSims)

	var wg sync.WaitGroup
	batchSize := 100
	for start := 0; start < cfg.NumSims; start += batchSize {
		end := start + batchSize
		if end > cfg.NumSims {
			end = cfg.NumSims
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for sim := s; sim < e; sim++ {
				equity := initialCash
				hwm := equity
				mdd := 0.0
				rng := int64(sim * 7919)
				for t := 0; t < nTrades; t++ {
					rng = lcg(rng)
					idx := int(absInt64(rng>>16) % int64(nTrades))
					equity += pnls[idx]
					if equity > hwm {
						hwm = equity
					}
					dd := (equity - hwm) / hwm
					if dd < mdd {
						mdd = dd
					}
				}
				terminals[sim] = equity
				maxDDs[sim] = mdd
			}
		}(start, end)
	}
	wg.Wait()

	sort.Float64s(terminals)
	sortedDD := make([]float64, len(maxDDs))
	copy(sortedDD, maxDDs)
	sort.Float64s(sortedDD)

	p5 := (1 - cfg.Confidence) / 2
	p95 := 1 - p5

	return &MonteCarloResult{
		MedianReturn: (percentile(terminals, 0.50) - initialCash) / initialCash,
		P5Return:     (percentile(terminals, p5) - initialCash) / initialCash,
		P95Return:    (percentile(terminals, p95) - initialCash) / initialCash,
		MedianDD:     percentile(sortedDD, 0.50),
		P95DD:        percentile(sortedDD, 0.05), // worst 5% of DDs
		Runs:         terminals,
	}
}

func absInt64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

// EquityPoint.Date alias for monthly returns
func (ep EquityPoint) DateStr() string { return ep.Date }
