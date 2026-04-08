package main

import (
	"math"
	"sort"
)

// ---------------------------------------------------------------------------
// Risk computation results
// ---------------------------------------------------------------------------

// RiskResult holds computed risk metrics for a portfolio.
type RiskResult struct {
	VaR95        float64
	VaR99        float64
	CVaR95       float64
	CVaR99       float64
	ComponentVaR []float64
	MarginalVaR  []float64
	Volatility   float64
	MaxDrawdown  float64
	HHI          float64
	TopNConc     float64
	FactorRisk   map[string]float64
}

// DrawdownResult holds drawdown analysis.
type DrawdownResult struct {
	MaxDrawdown      float64
	MaxDrawdownStart int
	MaxDrawdownEnd   int
	CurrentDrawdown  float64
	AvgDrawdown      float64
	DrawdownDuration int
	Drawdowns        []DrawdownPeriod
}

// DrawdownPeriod is a single drawdown episode.
type DrawdownPeriod struct {
	Start    int
	Trough   int
	End      int
	Depth    float64
	Duration int
	Recovery int
}

// ---------------------------------------------------------------------------
// RiskEngine
// ---------------------------------------------------------------------------

// RiskEngine computes portfolio risk metrics.
type RiskEngine struct{}

// NewRiskEngine creates a new risk engine.
func NewRiskEngine() *RiskEngine {
	return &RiskEngine{}
}

// ComputeRisk computes comprehensive risk metrics.
func (r *RiskEngine) ComputeRisk(weights []float64, returns [][]float64, n int) RiskResult {
	result := RiskResult{
		FactorRisk: make(map[string]float64),
	}
	t := len(returns)
	if t == 0 || n == 0 {
		return result
	}
	// Portfolio returns
	portReturns := computePortfolioReturns(weights, returns, n)
	// Volatility
	result.Volatility = annualizedVol(portReturns)
	// VaR
	result.VaR95 = historicalVaR(portReturns, 0.05)
	result.VaR99 = historicalVaR(portReturns, 0.01)
	// CVaR
	result.CVaR95 = historicalCVaR(portReturns, 0.05)
	result.CVaR99 = historicalCVaR(portReturns, 0.01)
	// Component VaR
	cov := SampleCov(returns, n)
	result.ComponentVaR = componentVaR(weights, cov, result.VaR95)
	result.MarginalVaR = marginalVaR(weights, cov)
	// Drawdown
	dd := analyzeDrawdowns(portReturns)
	result.MaxDrawdown = dd.MaxDrawdown
	// Concentration
	result.HHI = hhi(weights)
	result.TopNConc = topNConcentration(weights, 3)
	return result
}

// ---------------------------------------------------------------------------
// Portfolio returns
// ---------------------------------------------------------------------------

func computePortfolioReturns(weights []float64, returns [][]float64, n int) []float64 {
	t := len(returns)
	portRet := make([]float64, t)
	for i := 0; i < t; i++ {
		for j := 0; j < n && j < len(returns[i]); j++ {
			portRet[i] += weights[j] * returns[i][j]
		}
	}
	return portRet
}

func annualizedVol(returns []float64) float64 {
	n := len(returns)
	if n < 2 {
		return 0
	}
	mu := 0.0
	for _, r := range returns {
		mu += r
	}
	mu /= float64(n)
	ss := 0.0
	for _, r := range returns {
		d := r - mu
		ss += d * d
	}
	dailyVol := math.Sqrt(ss / float64(n-1))
	return dailyVol * math.Sqrt(252)
}

// ---------------------------------------------------------------------------
// Value at Risk
// ---------------------------------------------------------------------------

func historicalVaR(returns []float64, alpha float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	sorted := make([]float64, len(returns))
	copy(sorted, returns)
	sort.Float64s(sorted)
	idx := int(math.Floor(alpha * float64(len(sorted))))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return -sorted[idx]
}

func historicalCVaR(returns []float64, alpha float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	sorted := make([]float64, len(returns))
	copy(sorted, returns)
	sort.Float64s(sorted)
	cutoff := int(math.Ceil(alpha * float64(len(sorted))))
	if cutoff == 0 {
		cutoff = 1
	}
	sum := 0.0
	for i := 0; i < cutoff && i < len(sorted); i++ {
		sum += sorted[i]
	}
	return -sum / float64(cutoff)
}

func parametricVaR(portVol float64, alpha float64) float64 {
	// Using normal distribution approximation
	zScores := map[float64]float64{
		0.01: 2.326,
		0.05: 1.645,
		0.10: 1.282,
	}
	z, ok := zScores[alpha]
	if !ok {
		z = 1.645
	}
	return z * portVol / math.Sqrt(252)
}

// componentVaR computes each asset's contribution to portfolio VaR.
func componentVaR(weights []float64, cov [][]float64, portfolioVaR float64) []float64 {
	n := len(weights)
	pVar := portfolioVariance(weights, cov)
	if pVar <= 0 {
		return make([]float64, n)
	}
	pVol := math.Sqrt(pVar)
	// Marginal contribution
	marginal := matVecMul(cov, weights)
	compVaR := make([]float64, n)
	for i := 0; i < n; i++ {
		compVaR[i] = weights[i] * marginal[i] / pVol * portfolioVaR / pVol
	}
	return compVaR
}

func marginalVaR(weights []float64, cov [][]float64) []float64 {
	n := len(weights)
	pVar := portfolioVariance(weights, cov)
	if pVar <= 0 {
		return make([]float64, n)
	}
	pVol := math.Sqrt(pVar)
	sigmaW := matVecMul(cov, weights)
	mvar := make([]float64, n)
	for i := 0; i < n; i++ {
		mvar[i] = sigmaW[i] / pVol
	}
	return mvar
}

// ---------------------------------------------------------------------------
// Drawdown analysis
// ---------------------------------------------------------------------------

func analyzeDrawdowns(returns []float64) DrawdownResult {
	result := DrawdownResult{}
	if len(returns) == 0 {
		return result
	}
	cumReturn := 0.0
	peak := 0.0
	maxDD := 0.0
	maxDDStart := 0
	maxDDEnd := 0
	ddStart := 0
	inDrawdown := false
	totalDD := 0.0
	ddCount := 0
	var periods []DrawdownPeriod
	currentStart := 0
	currentTrough := 0
	currentDepth := 0.0
	for i, r := range returns {
		cumReturn += r
		if cumReturn > peak {
			if inDrawdown {
				// Drawdown recovered
				periods = append(periods, DrawdownPeriod{
					Start:    currentStart,
					Trough:   currentTrough,
					End:      i,
					Depth:    currentDepth,
					Duration: i - currentStart,
					Recovery: i - currentTrough,
				})
				inDrawdown = false
			}
			peak = cumReturn
			ddStart = i
		}
		dd := peak - cumReturn
		if dd > 0 {
			if !inDrawdown {
				inDrawdown = true
				currentStart = ddStart
				currentTrough = i
				currentDepth = dd
			}
			if dd > currentDepth {
				currentDepth = dd
				currentTrough = i
			}
			totalDD += dd
			ddCount++
		}
		if dd > maxDD {
			maxDD = dd
			maxDDStart = ddStart
			maxDDEnd = i
		}
	}
	// Handle ongoing drawdown
	if inDrawdown {
		periods = append(periods, DrawdownPeriod{
			Start:    currentStart,
			Trough:   currentTrough,
			End:      len(returns) - 1,
			Depth:    currentDepth,
			Duration: len(returns) - 1 - currentStart,
		})
	}
	result.MaxDrawdown = maxDD
	result.MaxDrawdownStart = maxDDStart
	result.MaxDrawdownEnd = maxDDEnd
	result.CurrentDrawdown = peak - cumReturn
	if ddCount > 0 {
		result.AvgDrawdown = totalDD / float64(ddCount)
	}
	if inDrawdown {
		result.DrawdownDuration = len(returns) - 1 - currentStart
	}
	result.Drawdowns = periods
	return result
}

// ---------------------------------------------------------------------------
// Concentration metrics
// ---------------------------------------------------------------------------

func hhi(weights []float64) float64 {
	s := 0.0
	for _, w := range weights {
		s += w * w
	}
	return s
}

func topNConcentration(weights []float64, topN int) float64 {
	sorted := make([]float64, len(weights))
	copy(sorted, weights)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))
	s := 0.0
	for i := 0; i < topN && i < len(sorted); i++ {
		s += sorted[i]
	}
	return s
}

// effectiveN returns the effective number of assets (1/HHI).
func effectiveN(weights []float64) float64 {
	h := hhi(weights)
	if h == 0 {
		return 0
	}
	return 1.0 / h
}

// ---------------------------------------------------------------------------
// Factor decomposition
// ---------------------------------------------------------------------------

// FactorDecomposition decomposes portfolio risk by factors.
type FactorDecomposition struct {
	FactorNames     []string
	FactorLoadings  [][]float64 // N x K
	FactorCov       [][]float64 // K x K
	SpecificRisk    []float64   // N
}

// Decompose computes factor risk decomposition.
func (fd *FactorDecomposition) Decompose(weights []float64) map[string]float64 {
	result := make(map[string]float64)
	n := len(weights)
	k := len(fd.FactorNames)
	if k == 0 || n == 0 {
		return result
	}
	// Portfolio factor exposures: b_p = B^T * w
	exposure := make([]float64, k)
	for f := 0; f < k; f++ {
		for i := 0; i < n && i < len(fd.FactorLoadings); i++ {
			if f < len(fd.FactorLoadings[i]) {
				exposure[f] += weights[i] * fd.FactorLoadings[i][f]
			}
		}
	}
	// Factor variance contribution: b_p^T * F * b_p
	totalFactorVar := 0.0
	for f1 := 0; f1 < k; f1++ {
		for f2 := 0; f2 < k; f2++ {
			fv := exposure[f1] * exposure[f2] * fd.FactorCov[f1][f2]
			totalFactorVar += fv
		}
	}
	// Individual factor contributions (diagonal approximation)
	for f := 0; f < k; f++ {
		contrib := exposure[f] * exposure[f] * fd.FactorCov[f][f]
		result[fd.FactorNames[f]] = contrib
	}
	result["systematic"] = totalFactorVar
	// Specific risk
	specificVar := 0.0
	for i := 0; i < n && i < len(fd.SpecificRisk); i++ {
		specificVar += weights[i] * weights[i] * fd.SpecificRisk[i] * fd.SpecificRisk[i]
	}
	result["specific"] = specificVar
	result["total"] = totalFactorVar + specificVar
	return result
}

// ---------------------------------------------------------------------------
// Stress testing engine
// ---------------------------------------------------------------------------

// StressEngine runs stress tests on portfolios.
type StressEngine struct{}

// NewStressEngine creates a new stress engine.
func NewStressEngine() *StressEngine {
	return &StressEngine{}
}

// ScenarioTest runs a single scenario stress test.
func (s *StressEngine) ScenarioTest(weights []float64, assets []string, shocks map[string]float64) float64 {
	portReturn := 0.0
	assetIdx := make(map[string]int, len(assets))
	for i, a := range assets {
		assetIdx[a] = i
	}
	for asset, shock := range shocks {
		if idx, ok := assetIdx[asset]; ok && idx < len(weights) {
			portReturn += weights[idx] * shock
		}
	}
	return portReturn
}

// HistoricalScenarios generates stress scenarios from historical data.
func (s *StressEngine) HistoricalScenarios(returns [][]float64, assets []string, worstN int) []StressScenario {
	t := len(returns)
	n := len(assets)
	if t == 0 || n == 0 {
		return nil
	}
	type dayReturn struct {
		day     int
		portRet float64
		returns []float64
	}
	// Compute equal-weighted portfolio returns for each day
	days := make([]dayReturn, t)
	for i := 0; i < t; i++ {
		pr := 0.0
		for j := 0; j < n && j < len(returns[i]); j++ {
			pr += returns[i][j] / float64(n)
		}
		row := make([]float64, n)
		for j := 0; j < n && j < len(returns[i]); j++ {
			row[j] = returns[i][j]
		}
		days[i] = dayReturn{day: i, portRet: pr, returns: row}
	}
	// Sort by portfolio return (worst first)
	sort.Slice(days, func(i, j int) bool {
		return days[i].portRet < days[j].portRet
	})
	if worstN > t {
		worstN = t
	}
	scenarios := make([]StressScenario, 0, worstN)
	for i := 0; i < worstN; i++ {
		shocks := make(map[string]float64, n)
		for j := 0; j < n; j++ {
			shocks[assets[j]] = days[i].returns[j]
		}
		scenarios = append(scenarios, StressScenario{
			Name:   fmt.Sprintf("historical_day_%d", days[i].day),
			Shocks: shocks,
		})
	}
	return scenarios
}

// ---------------------------------------------------------------------------
// Performance analytics
// ---------------------------------------------------------------------------

// PerformanceMetrics holds computed performance metrics.
type PerformanceMetrics struct {
	TotalReturn     float64
	AnnualizedReturn float64
	Volatility      float64
	Sharpe          float64
	Sortino         float64
	Calmar          float64
	MaxDrawdown     float64
	WinRate         float64
	ProfitFactor    float64
	MonthlyReturns  []float64
	DailyReturns    []float64
}

// ComputePerformance computes full performance metrics from daily returns.
func ComputePerformance(dailyReturns []float64, riskFreeRate float64) PerformanceMetrics {
	result := PerformanceMetrics{
		DailyReturns: dailyReturns,
	}
	n := len(dailyReturns)
	if n == 0 {
		return result
	}
	// Total return (compounded)
	cumRet := 0.0
	for _, r := range dailyReturns {
		cumRet += r
	}
	result.TotalReturn = math.Exp(cumRet) - 1
	// Annualized return
	years := float64(n) / 252
	if years > 0 {
		result.AnnualizedReturn = math.Pow(1+result.TotalReturn, 1/years) - 1
	}
	// Volatility
	result.Volatility = annualizedVol(dailyReturns)
	// Sharpe
	if result.Volatility > 0 {
		result.Sharpe = (result.AnnualizedReturn - riskFreeRate) / result.Volatility
	}
	// Sortino (downside deviation)
	mu := cumRet / float64(n)
	downSS := 0.0
	downCount := 0
	for _, r := range dailyReturns {
		if r < 0 {
			downSS += r * r
			downCount++
		}
	}
	if downCount > 0 {
		downsideVol := math.Sqrt(downSS/float64(n)) * math.Sqrt(252)
		if downsideVol > 0 {
			result.Sortino = (result.AnnualizedReturn - riskFreeRate) / downsideVol
		}
	}
	// Drawdown
	dd := analyzeDrawdowns(dailyReturns)
	result.MaxDrawdown = dd.MaxDrawdown
	// Calmar
	if result.MaxDrawdown > 0 {
		result.Calmar = result.AnnualizedReturn / result.MaxDrawdown
	}
	// Win rate
	wins := 0
	grossProfit := 0.0
	grossLoss := 0.0
	for _, r := range dailyReturns {
		if r > 0 {
			wins++
			grossProfit += r
		} else if r < 0 {
			grossLoss -= r
		}
	}
	result.WinRate = float64(wins) / float64(n)
	// Profit factor
	if grossLoss > 0 {
		result.ProfitFactor = grossProfit / grossLoss
	}
	// Monthly returns (approximate: 21 trading days per month)
	monthLen := 21
	for i := 0; i < n; i += monthLen {
		end := i + monthLen
		if end > n {
			end = n
		}
		monthRet := 0.0
		for j := i; j < end; j++ {
			monthRet += dailyReturns[j]
		}
		result.MonthlyReturns = append(result.MonthlyReturns, monthRet)
	}
	_ = mu
	return result
}

// ---------------------------------------------------------------------------
// Risk budget
// ---------------------------------------------------------------------------

// RiskBudget computes risk contributions for each asset.
type RiskBudget struct {
	Assets       []string
	Weights      []float64
	RiskContrib  []float64
	PctContrib   []float64
	MarginalRisk []float64
	TotalRisk    float64
}

// ComputeRiskBudget computes the risk budget for a portfolio.
func ComputeRiskBudget(weights []float64, cov [][]float64, assets []string) RiskBudget {
	n := len(weights)
	pVar := portfolioVariance(weights, cov)
	pVol := math.Sqrt(math.Max(pVar, 0))
	sigmaW := matVecMul(cov, weights)
	rc := make([]float64, n)
	pctRC := make([]float64, n)
	marginal := make([]float64, n)
	for i := 0; i < n; i++ {
		if pVol > 0 {
			marginal[i] = sigmaW[i] / pVol
			rc[i] = weights[i] * marginal[i]
			pctRC[i] = rc[i] / pVol
		}
	}
	return RiskBudget{
		Assets:       assets,
		Weights:      weights,
		RiskContrib:  rc,
		PctContrib:   pctRC,
		MarginalRisk: marginal,
		TotalRisk:    pVol,
	}
}

// ---------------------------------------------------------------------------
// Correlation analysis
// ---------------------------------------------------------------------------

// CorrelationMatrix computes the correlation matrix from returns.
func CorrelationMatrix(returns [][]float64, n int) [][]float64 {
	cov := SampleCov(returns, n)
	corr := make([][]float64, n)
	for i := range corr {
		corr[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			denom := math.Sqrt(cov[i][i] * cov[j][j])
			if denom > 0 {
				corr[i][j] = cov[i][j] / denom
			}
		}
	}
	return corr
}

// MaxCorrelation returns the max off-diagonal correlation.
func MaxCorrelation(corr [][]float64) float64 {
	n := len(corr)
	maxC := -1.0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if corr[i][j] > maxC {
				maxC = corr[i][j]
			}
		}
	}
	return maxC
}

// AverageCorrelation returns the average off-diagonal correlation.
func AverageCorrelation(corr [][]float64) float64 {
	n := len(corr)
	sum := 0.0
	count := 0
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sum += corr[i][j]
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

// ---------------------------------------------------------------------------
// Rolling risk metrics
// ---------------------------------------------------------------------------

// RollingVaR computes rolling VaR over a window.
func RollingVaR(returns []float64, window int, alpha float64) []float64 {
	n := len(returns)
	if n < window {
		return nil
	}
	result := make([]float64, n-window+1)
	for i := 0; i <= n-window; i++ {
		result[i] = historicalVaR(returns[i:i+window], alpha)
	}
	return result
}

// RollingVol computes rolling annualized volatility.
func RollingVol(returns []float64, window int) []float64 {
	n := len(returns)
	if n < window {
		return nil
	}
	result := make([]float64, n-window+1)
	for i := 0; i <= n-window; i++ {
		result[i] = annualizedVol(returns[i : i+window])
	}
	return result
}

// RollingSharpe computes rolling Sharpe ratio.
func RollingSharpe(returns []float64, window int, rf float64) []float64 {
	n := len(returns)
	if n < window {
		return nil
	}
	result := make([]float64, n-window+1)
	for i := 0; i <= n-window; i++ {
		slice := returns[i : i+window]
		vol := annualizedVol(slice)
		sum := 0.0
		for _, r := range slice {
			sum += r
		}
		annRet := sum / float64(window) * 252
		if vol > 0 {
			result[i] = (annRet - rf) / vol
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// Tail risk metrics
// ---------------------------------------------------------------------------

// Skewness computes the skewness of returns.
func Skewness(returns []float64) float64 {
	n := float64(len(returns))
	if n < 3 {
		return 0
	}
	mu := 0.0
	for _, r := range returns {
		mu += r
	}
	mu /= n
	m2, m3 := 0.0, 0.0
	for _, r := range returns {
		d := r - mu
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

// Kurtosis computes the excess kurtosis of returns.
func Kurtosis(returns []float64) float64 {
	n := float64(len(returns))
	if n < 4 {
		return 0
	}
	mu := 0.0
	for _, r := range returns {
		mu += r
	}
	mu /= n
	m2, m4 := 0.0, 0.0
	for _, r := range returns {
		d := r - mu
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

// TailRatio computes the ratio of right tail to left tail.
func TailRatio(returns []float64, pct float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	sorted := make([]float64, len(returns))
	copy(sorted, returns)
	sort.Float64s(sorted)
	n := len(sorted)
	rightIdx := int(math.Floor((1 - pct/100) * float64(n)))
	leftIdx := int(math.Floor(pct / 100 * float64(n)))
	if rightIdx >= n {
		rightIdx = n - 1
	}
	rightTail := sorted[rightIdx]
	leftTail := sorted[leftIdx]
	if leftTail == 0 {
		return 0
	}
	return math.Abs(rightTail / leftTail)
}

// ---------------------------------------------------------------------------
// Parametric risk (normal assumption)
// ---------------------------------------------------------------------------

// ParametricRisk computes parametric risk metrics assuming normality.
type ParametricRisk struct{}

// VaR computes parametric VaR.
func (p *ParametricRisk) VaR(weights []float64, cov [][]float64, alpha float64, horizon int) float64 {
	pVol := portfolioVol(weights, cov)
	dailyVol := pVol / math.Sqrt(252)
	z := parametricVaRZ(alpha)
	return z * dailyVol * math.Sqrt(float64(horizon))
}

// parametricVaRZ returns the z-score for a given confidence level.
func parametricVaRZ(alpha float64) float64 {
	// Rational approximation for normal inverse CDF
	if alpha <= 0 || alpha >= 1 {
		return 0
	}
	p := alpha
	if p > 0.5 {
		p = 1 - p
	}
	t := math.Sqrt(-2 * math.Log(p))
	// Abramowitz and Stegun approximation
	c0 := 2.515517
	c1 := 0.802853
	c2 := 0.010328
	d1 := 1.432788
	d2 := 0.189269
	d3 := 0.001308
	z := t - (c0+c1*t+c2*t*t)/(1+d1*t+d2*t*t+d3*t*t*t)
	if alpha > 0.5 {
		z = -z
	}
	return z
}

// CornishFisherVaR adjusts VaR for skewness and kurtosis.
func CornishFisherVaR(returns []float64, alpha float64) float64 {
	vol := annualizedVol(returns) / math.Sqrt(252) // daily vol
	z := parametricVaRZ(alpha)
	s := Skewness(returns)
	k := Kurtosis(returns)
	// Cornish-Fisher expansion
	zCF := z + (z*z-1)*s/6 + (z*z*z-3*z)*k/24 - (2*z*z*z-5*z)*s*s/36
	return zCF * vol
}
