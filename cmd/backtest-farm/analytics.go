package main

import (
	"math"
	"sort"
	"strings"
)

// =============================================================================
// analytics.go — statistical analysis of backtest farm results:
// distribution analysis, deflated Sharpe, FDR correction, ensemble
// construction, parameter stability, overfitting detection, leaderboard,
// heatmap generation, comparison reports.
// =============================================================================

// ---- result analyzer -------------------------------------------------------

type ResultAnalyzer struct {
	engine *FarmEngine
}

func NewResultAnalyzer(engine *FarmEngine) *ResultAnalyzer {
	return &ResultAnalyzer{engine: engine}
}

// AggregateStatistics returns overall farm statistics including best-ever
// results, average duration, and distribution of Sharpes.
func (ra *ResultAnalyzer) AggregateStatistics() map[string]interface{} {
	completed := ra.engine.store.Completed()
	metrics := ra.engine.metrics.Snapshot()

	stats := map[string]interface{}{
		"total_backtests":    metrics.TotalSubmitted,
		"completed":          metrics.TotalCompleted,
		"failed":             metrics.TotalFailed,
		"avg_duration_ms":    metrics.AvgDurationMs,
		"throughput_per_sec": metrics.ThroughputPerSec,
	}

	if len(completed) == 0 {
		stats["best_ever"] = nil
		stats["sharpe_distribution"] = nil
		stats["strategy_breakdown"] = nil
		stats["deflated_sharpe"] = nil
		stats["leaderboard"] = nil
		return stats
	}

	// Best ever
	var bestJob *BacktestJob
	for _, j := range completed {
		if bestJob == nil || j.Result.Sharpe > bestJob.Result.Sharpe {
			bestJob = j
		}
	}
	stats["best_ever"] = map[string]interface{}{
		"job_id":       bestJob.ID,
		"strategy":     bestJob.Config.Strategy,
		"symbol":       bestJob.Config.Symbol,
		"sharpe":       bestJob.Result.Sharpe,
		"total_return": bestJob.Result.TotalReturn,
		"max_drawdown": bestJob.Result.MaxDrawdown,
		"parameters":   bestJob.Config.Parameters,
	}

	// Sharpe distribution
	sharpes := make([]float64, len(completed))
	for i, j := range completed {
		sharpes[i] = j.Result.Sharpe
	}
	stats["sharpe_distribution"] = computeDistribution(sharpes)

	// Strategy breakdown
	stratMap := make(map[string][]float64)
	for _, j := range completed {
		stratMap[j.Config.Strategy] = append(stratMap[j.Config.Strategy], j.Result.Sharpe)
	}
	breakdown := make(map[string]interface{})
	for strat, sharpeList := range stratMap {
		dist := computeDistribution(sharpeList)
		breakdown[strat] = map[string]interface{}{
			"count":    len(sharpeList),
			"mean":     dist.Mean,
			"std":      dist.Std,
			"median":   dist.Median,
			"best":     dist.Max,
			"worst":    dist.Min,
		}
	}
	stats["strategy_breakdown"] = breakdown

	// Deflated Sharpe for best strategy
	dsr := ra.computeDeflatedSharpe(completed)
	stats["deflated_sharpe"] = dsr

	// Leaderboard
	lb := ra.BuildLeaderboard(completed)
	stats["leaderboard"] = lb

	// Strategy correlations
	corr := ra.computeStrategyCorrelations(completed)
	stats["strategy_correlations"] = corr

	// FDR correction
	fdr := ra.computeFDRCorrection(completed)
	stats["fdr_correction"] = fdr

	// Optimal ensemble
	ensemble := ra.computeOptimalEnsemble(completed, 5)
	stats["optimal_ensemble"] = ensemble

	// Parameter stability
	stability := ra.computeParameterStability(completed)
	stats["parameter_stability"] = stability

	// Overfitting flags
	overfit := ra.detectOverfitting(completed)
	stats["overfitting_flags"] = overfit

	return stats
}

// ---- deflated Sharpe ratio -------------------------------------------------
// Adjusts the best Sharpe for the number of trials to avoid selection bias.
// Based on Bailey & Lopez de Prado (2014).

type DeflatedSharpeResult struct {
	OriginalSharpe   float64 `json:"original_sharpe"`
	DeflatedSharpe   float64 `json:"deflated_sharpe"`
	NumTrials        int     `json:"num_trials"`
	SharpeStd        float64 `json:"sharpe_std"`
	ExpectedMaxSR    float64 `json:"expected_max_sharpe"`
	IsSignificant    bool    `json:"is_significant"`
	PValue           float64 `json:"p_value"`
	HaircutPct       float64 `json:"haircut_pct"`
}

func (ra *ResultAnalyzer) computeDeflatedSharpe(completed []*BacktestJob) *DeflatedSharpeResult {
	if len(completed) < 2 {
		return nil
	}

	sharpes := make([]float64, len(completed))
	for i, j := range completed {
		sharpes[i] = j.Result.Sharpe
	}

	mean, std := meanStd(sharpes)
	nTrials := len(sharpes)

	// Find best Sharpe
	bestSR := sharpes[0]
	for _, s := range sharpes[1:] {
		if s > bestSR {
			bestSR = s
		}
	}

	// Expected maximum Sharpe ratio under null (all strategies have SR=0)
	// E[max(Z_1,...,Z_N)] ~ sqrt(2 * ln(N)) - (ln(pi) + ln(ln(N))) / (2 * sqrt(2 * ln(N)))
	// where Z_i ~ N(0,1)
	logN := math.Log(float64(nTrials))
	if logN < 0.01 {
		logN = 0.01
	}
	sqrtTwoLogN := math.Sqrt(2 * logN)
	eulerMascheroni := 0.5772156649
	expectedMaxZ := sqrtTwoLogN - (math.Log(math.Pi)+math.Log(logN))/(2*sqrtTwoLogN) + eulerMascheroni/sqrtTwoLogN

	// Expected max Sharpe = mean + std * expectedMaxZ
	expectedMaxSR := mean + std*expectedMaxZ

	// Deflated Sharpe: test if bestSR > expectedMaxSR
	// Under the null, bestSR ~ N(expectedMaxSR, std^2 / N)
	testStat := 0.0
	se := std / math.Sqrt(float64(nTrials))
	if se > 1e-10 {
		testStat = (bestSR - expectedMaxSR) / se
	}

	// P-value from standard normal
	pValue := 1.0 - normalCDF(testStat)

	// Deflated Sharpe
	deflated := bestSR - expectedMaxSR
	if deflated < 0 {
		deflated = 0
	}

	haircut := 0.0
	if bestSR != 0 {
		haircut = (1 - deflated/bestSR) * 100
	}

	return &DeflatedSharpeResult{
		OriginalSharpe: roundTo(bestSR, 4),
		DeflatedSharpe: roundTo(deflated, 4),
		NumTrials:      nTrials,
		SharpeStd:      roundTo(std, 4),
		ExpectedMaxSR:  roundTo(expectedMaxSR, 4),
		IsSignificant:  pValue < 0.05,
		PValue:         roundTo(pValue, 6),
		HaircutPct:     roundTo(haircut, 2),
	}
}

// normalCDF approximation using Abramowitz and Stegun formula 7.1.26
func normalCDF(x float64) float64 {
	if x < -8 {
		return 0
	}
	if x > 8 {
		return 1
	}
	t := 1.0 / (1.0 + 0.2316419*math.Abs(x))
	d := 0.3989422804014327 // 1/sqrt(2*pi)
	p := d * math.Exp(-x*x/2.0) * (t * (0.319381530 + t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429)))))
	if x > 0 {
		return 1 - p
	}
	return p
}

// normalPDF for standard normal
func normalPDF(x float64) float64 {
	return math.Exp(-x*x/2.0) / math.Sqrt(2*math.Pi)
}

// ---- false discovery rate (Benjamini-Hochberg) -----------------------------

type FDRResult struct {
	Strategy     string  `json:"strategy"`
	Symbol       string  `json:"symbol"`
	Sharpe       float64 `json:"sharpe"`
	PValue       float64 `json:"p_value"`
	AdjPValue    float64 `json:"adjusted_p_value"`
	IsDiscovery  bool    `json:"is_discovery"`
	Rank         int     `json:"rank"`
}

func (ra *ResultAnalyzer) computeFDRCorrection(completed []*BacktestJob) []FDRResult {
	if len(completed) == 0 {
		return nil
	}

	// Group by strategy+symbol to avoid double-counting parameter variations
	type stratKey struct {
		strategy string
		symbol   string
	}
	grouped := make(map[stratKey]*BacktestJob)
	for _, j := range completed {
		key := stratKey{j.Config.Strategy, j.Config.Symbol}
		existing, ok := grouped[key]
		if !ok || j.Result.Sharpe > existing.Result.Sharpe {
			grouped[key] = j
		}
	}

	// Compute p-values: test H0: Sharpe <= 0
	// Under null, Sharpe ~ N(0, 1/sqrt(T)) approximately
	type pValEntry struct {
		key    stratKey
		job    *BacktestJob
		pValue float64
	}
	entries := make([]pValEntry, 0, len(grouped))
	for key, j := range grouped {
		// Approximate: SE of Sharpe ~ 1/sqrt(T) where T ~ n_trades
		T := float64(j.Result.NTrades)
		if T < 10 {
			T = 10
		}
		se := 1.0 / math.Sqrt(T)
		z := j.Result.Sharpe / se
		pVal := 1.0 - normalCDF(z)
		entries = append(entries, pValEntry{key: key, job: j, pValue: pVal})
	}

	// Sort by p-value ascending
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].pValue < entries[j].pValue
	})

	m := len(entries)
	alpha := 0.05
	results := make([]FDRResult, m)

	for i, e := range entries {
		rank := i + 1
		adjP := e.pValue * float64(m) / float64(rank)
		if adjP > 1 {
			adjP = 1
		}
		results[i] = FDRResult{
			Strategy:    e.key.strategy,
			Symbol:      e.key.symbol,
			Sharpe:      e.job.Result.Sharpe,
			PValue:      roundTo(e.pValue, 6),
			AdjPValue:   roundTo(adjP, 6),
			IsDiscovery: adjP < alpha,
			Rank:        rank,
		}
	}

	// Step-up: ensure monotonicity of adjusted p-values
	for i := m - 2; i >= 0; i-- {
		if results[i].AdjPValue > results[i+1].AdjPValue {
			results[i].AdjPValue = results[i+1].AdjPValue
		}
		results[i].IsDiscovery = results[i].AdjPValue < alpha
	}

	return results
}

// ---- strategy correlations -------------------------------------------------

type CorrelationEntry struct {
	Strategy1   string  `json:"strategy_1"`
	Strategy2   string  `json:"strategy_2"`
	Correlation float64 `json:"correlation"`
	Redundant   bool    `json:"redundant"`
}

func (ra *ResultAnalyzer) computeStrategyCorrelations(completed []*BacktestJob) []CorrelationEntry {
	// Group Sharpe vectors by strategy
	stratSharpes := make(map[string][]float64)
	stratSymbols := make(map[string]map[string]float64) // strategy -> symbol -> best sharpe

	for _, j := range completed {
		s := j.Config.Strategy
		sym := j.Config.Symbol
		if stratSymbols[s] == nil {
			stratSymbols[s] = make(map[string]float64)
		}
		if existing, ok := stratSymbols[s][sym]; !ok || j.Result.Sharpe > existing {
			stratSymbols[s][sym] = j.Result.Sharpe
		}
	}

	// Build common symbol set
	allSymbols := make(map[string]bool)
	for _, symMap := range stratSymbols {
		for sym := range symMap {
			allSymbols[sym] = true
		}
	}
	symbolList := make([]string, 0, len(allSymbols))
	for sym := range allSymbols {
		symbolList = append(symbolList, sym)
	}
	sort.Strings(symbolList)

	strategies := make([]string, 0, len(stratSymbols))
	for s := range stratSymbols {
		strategies = append(strategies, s)
	}
	sort.Strings(strategies)

	// Build vectors aligned by symbol
	for _, s := range strategies {
		vec := make([]float64, len(symbolList))
		for i, sym := range symbolList {
			if v, ok := stratSymbols[s][sym]; ok {
				vec[i] = v
			}
		}
		stratSharpes[s] = vec
	}

	// Compute pairwise correlations
	results := make([]CorrelationEntry, 0)
	for i := 0; i < len(strategies); i++ {
		for j := i + 1; j < len(strategies); j++ {
			s1 := strategies[i]
			s2 := strategies[j]
			corr := pearsonCorrelation(stratSharpes[s1], stratSharpes[s2])
			results = append(results, CorrelationEntry{
				Strategy1:   s1,
				Strategy2:   s2,
				Correlation: roundTo(corr, 4),
				Redundant:   corr > 0.8,
			})
		}
	}

	return results
}

func pearsonCorrelation(x, y []float64) float64 {
	n := len(x)
	if n != len(y) || n < 2 {
		return 0
	}
	sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	nf := float64(n)
	num := nf*sumXY - sumX*sumY
	den := math.Sqrt((nf*sumX2 - sumX*sumX) * (nf*sumY2 - sumY*sumY))
	if den < 1e-15 {
		return 0
	}
	return num / den
}

// ---- optimal ensemble (inverse-vol weighted) -------------------------------

type EnsembleMember struct {
	Strategy string  `json:"strategy"`
	Symbol   string  `json:"symbol"`
	Weight   float64 `json:"weight"`
	Sharpe   float64 `json:"sharpe"`
	Return   float64 `json:"total_return"`
}

type EnsembleResult struct {
	Members        []EnsembleMember `json:"members"`
	EnsembleSharpe float64          `json:"ensemble_sharpe"`
	EnsembleReturn float64          `json:"ensemble_return"`
	Diversification float64         `json:"diversification_ratio"`
}

func (ra *ResultAnalyzer) computeOptimalEnsemble(completed []*BacktestJob, topN int) *EnsembleResult {
	if len(completed) == 0 {
		return nil
	}

	// Pick best config per strategy
	bestPerStrategy := make(map[string]*BacktestJob)
	for _, j := range completed {
		s := j.Config.Strategy
		if existing, ok := bestPerStrategy[s]; !ok || j.Result.Sharpe > existing.Result.Sharpe {
			bestPerStrategy[s] = j
		}
	}

	// Sort by Sharpe descending, take top N
	candidates := make([]*BacktestJob, 0, len(bestPerStrategy))
	for _, j := range bestPerStrategy {
		candidates = append(candidates, j)
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Result.Sharpe > candidates[j].Result.Sharpe
	})
	if len(candidates) > topN {
		candidates = candidates[:topN]
	}

	if len(candidates) == 0 {
		return nil
	}

	// Inverse volatility weighting
	// vol ~ 1/Sharpe * return, but simplified: use max_drawdown as vol proxy
	totalInvVol := 0.0
	invVols := make([]float64, len(candidates))
	for i, j := range candidates {
		vol := j.Result.MaxDrawdown
		if vol < 0.01 {
			vol = 0.01
		}
		invVols[i] = 1.0 / vol
		totalInvVol += invVols[i]
	}

	members := make([]EnsembleMember, len(candidates))
	ensembleReturn := 0.0
	ensembleSharpe := 0.0
	sumWeightSq := 0.0

	for i, j := range candidates {
		w := invVols[i] / totalInvVol
		members[i] = EnsembleMember{
			Strategy: j.Config.Strategy,
			Symbol:   j.Config.Symbol,
			Weight:   roundTo(w, 4),
			Sharpe:   j.Result.Sharpe,
			Return:   j.Result.TotalReturn,
		}
		ensembleReturn += w * j.Result.TotalReturn
		ensembleSharpe += w * j.Result.Sharpe
		sumWeightSq += w * w
	}

	// Diversification ratio: 1/sqrt(sum(w_i^2)) / N
	divRatio := 0.0
	if sumWeightSq > 0 {
		divRatio = 1.0 / (math.Sqrt(sumWeightSq) * float64(len(members)))
	}

	return &EnsembleResult{
		Members:         members,
		EnsembleSharpe:  roundTo(ensembleSharpe, 4),
		EnsembleReturn:  roundTo(ensembleReturn, 6),
		Diversification: roundTo(divRatio, 4),
	}
}

// ---- parameter stability ---------------------------------------------------

type ParamStabilityResult struct {
	Strategy        string             `json:"strategy"`
	OverallStdSharpe float64           `json:"overall_std_sharpe"`
	IsStable        bool               `json:"is_stable"`
	ParamRanges     map[string]ParamRange `json:"param_ranges"`
}

type ParamRange struct {
	MinSharpe    float64 `json:"min_sharpe"`
	MaxSharpe    float64 `json:"max_sharpe"`
	MeanSharpe   float64 `json:"mean_sharpe"`
	StdSharpe    float64 `json:"std_sharpe"`
	NumValues    int     `json:"num_values"`
	Sensitivity  float64 `json:"sensitivity"`
}

func (ra *ResultAnalyzer) computeParameterStability(completed []*BacktestJob) []ParamStabilityResult {
	// Group by strategy
	byStrategy := make(map[string][]*BacktestJob)
	for _, j := range completed {
		byStrategy[j.Config.Strategy] = append(byStrategy[j.Config.Strategy], j)
	}

	results := make([]ParamStabilityResult, 0, len(byStrategy))

	for strategy, jobs := range byStrategy {
		if len(jobs) < 3 {
			continue
		}

		sharpes := make([]float64, len(jobs))
		for i, j := range jobs {
			sharpes[i] = j.Result.Sharpe
		}
		_, overallStd := meanStd(sharpes)

		// Per-parameter analysis
		paramRanges := make(map[string]ParamRange)

		// Collect all parameter names
		paramNames := make(map[string]bool)
		for _, j := range jobs {
			for k := range j.Config.Parameters {
				paramNames[k] = true
			}
		}

		for param := range paramNames {
			// Group sharpes by this param value
			byValue := make(map[string][]float64)
			for _, j := range jobs {
				val := ""
				if v, ok := j.Config.Parameters[param]; ok {
					val = formatParamValue(v)
				}
				byValue[val] = append(byValue[val], j.Result.Sharpe)
			}

			if len(byValue) < 2 {
				continue
			}

			// Compute mean Sharpe per value, then std of means
			means := make([]float64, 0, len(byValue))
			allSharpes := make([]float64, 0)
			for _, slist := range byValue {
				sum := 0.0
				for _, s := range slist {
					sum += s
					allSharpes = append(allSharpes, s)
				}
				means = append(means, sum/float64(len(slist)))
			}

			_, std := meanStd(means)
			overallMean, _ := meanStd(allSharpes)

			minS, maxS := allSharpes[0], allSharpes[0]
			for _, s := range allSharpes[1:] {
				if s < minS {
					minS = s
				}
				if s > maxS {
					maxS = s
				}
			}

			// Sensitivity: std of group means / overall mean (coefficient of variation)
			sensitivity := 0.0
			if math.Abs(overallMean) > 1e-10 {
				sensitivity = std / math.Abs(overallMean)
			}

			paramRanges[param] = ParamRange{
				MinSharpe:   roundTo(minS, 4),
				MaxSharpe:   roundTo(maxS, 4),
				MeanSharpe:  roundTo(overallMean, 4),
				StdSharpe:   roundTo(std, 4),
				NumValues:   len(byValue),
				Sensitivity: roundTo(sensitivity, 4),
			}
		}

		// Stable if overall Sharpe std is low relative to mean
		meanSharpe, _ := meanStd(sharpes)
		isStable := overallStd < 0.5*math.Abs(meanSharpe) || overallStd < 0.3

		results = append(results, ParamStabilityResult{
			Strategy:         strategy,
			OverallStdSharpe: roundTo(overallStd, 4),
			IsStable:         isStable,
			ParamRanges:      paramRanges,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Strategy < results[j].Strategy
	})

	return results
}

func formatParamValue(v interface{}) string {
	switch t := v.(type) {
	case float64:
		if t == math.Floor(t) {
			return strings.TrimRight(strings.TrimRight(
				strings.Replace(
					strings.Replace(
						formatFloat(t), ".", "d", 1),
					"-", "n", 1),
				"0"), "d")
		}
		return formatFloat(t)
	default:
		return strings.Replace(strings.Replace(
			strings.Replace(formatAny(v), " ", "_", -1),
			"[", "", -1), "]", "", -1)
	}
}

func formatFloat(f float64) string {
	s := ""
	if f < 0 {
		s = "-"
		f = -f
	}
	intPart := int64(f)
	fracPart := f - float64(intPart)
	s += intToStr(intPart)
	if fracPart > 1e-10 {
		s += "."
		for i := 0; i < 6; i++ {
			fracPart *= 10
			digit := int(fracPart)
			s += string(rune('0' + digit))
			fracPart -= float64(digit)
			if fracPart < 1e-10 {
				break
			}
		}
	}
	return s
}

func intToStr(n int64) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}

func formatAny(v interface{}) string {
	switch t := v.(type) {
	case string:
		return t
	case float64:
		return formatFloat(t)
	case int:
		return intToStr(int64(t))
	case bool:
		if t {
			return "true"
		}
		return "false"
	default:
		return "unknown"
	}
}

// ---- regime robustness -----------------------------------------------------

type RegimeRobustness struct {
	Strategy string             `json:"strategy"`
	Regimes  map[string]float64 `json:"regime_sharpes"`
	IsRobust bool               `json:"is_robust"`
	WorstRegime string          `json:"worst_regime"`
	BestRegime  string          `json:"best_regime"`
	Spread      float64         `json:"regime_spread"`
}

func (ra *ResultAnalyzer) computeRegimeRobustness(completed []*BacktestJob) []RegimeRobustness {
	byStrategy := make(map[string][]*BacktestJob)
	for _, j := range completed {
		byStrategy[j.Config.Strategy] = append(byStrategy[j.Config.Strategy], j)
	}

	results := make([]RegimeRobustness, 0)

	for strategy, jobs := range byStrategy {
		regimes := make(map[string][]float64)

		for _, j := range jobs {
			// Classify into regimes based on result characteristics
			if j.Result.TotalReturn > 0.1 {
				regimes["bull"] = append(regimes["bull"], j.Result.Sharpe)
			} else if j.Result.TotalReturn < -0.05 {
				regimes["bear"] = append(regimes["bear"], j.Result.Sharpe)
			} else {
				regimes["sideways"] = append(regimes["sideways"], j.Result.Sharpe)
			}

			if j.Result.MaxDrawdown > 0.2 {
				regimes["high_vol"] = append(regimes["high_vol"], j.Result.Sharpe)
			} else {
				regimes["low_vol"] = append(regimes["low_vol"], j.Result.Sharpe)
			}
		}

		regimeMeans := make(map[string]float64)
		bestRegime := ""
		worstRegime := ""
		bestMean := -math.MaxFloat64
		worstMean := math.MaxFloat64

		for regime, sharpes := range regimes {
			if len(sharpes) == 0 {
				continue
			}
			sum := 0.0
			for _, s := range sharpes {
				sum += s
			}
			mean := sum / float64(len(sharpes))
			regimeMeans[regime] = roundTo(mean, 4)
			if mean > bestMean {
				bestMean = mean
				bestRegime = regime
			}
			if mean < worstMean {
				worstMean = mean
				worstRegime = regime
			}
		}

		spread := bestMean - worstMean
		isRobust := spread < 1.0 && worstMean > -0.5

		results = append(results, RegimeRobustness{
			Strategy:    strategy,
			Regimes:     regimeMeans,
			IsRobust:    isRobust,
			WorstRegime: worstRegime,
			BestRegime:  bestRegime,
			Spread:      roundTo(spread, 4),
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Spread < results[j].Spread
	})

	return results
}

// ---- heatmap generator -----------------------------------------------------

type HeatmapGenerator struct {
	store *ResultStore
}

func NewHeatmapGenerator(store *ResultStore) *HeatmapGenerator {
	return &HeatmapGenerator{store: store}
}

type HeatmapOutput struct {
	Strategy   string          `json:"strategy"`
	Symbol     string          `json:"symbol"`
	Param1Name string          `json:"param1_name"`
	Param2Name string          `json:"param2_name"`
	Param1Vals []interface{}   `json:"param1_values"`
	Param2Vals []interface{}   `json:"param2_values"`
	Grid       [][]float64     `json:"grid"`
	BestParam1 interface{}     `json:"best_param1"`
	BestParam2 interface{}     `json:"best_param2"`
	BestSharpe float64         `json:"best_sharpe"`
	MinSharpe  float64         `json:"min_sharpe"`
	MaxSharpe  float64         `json:"max_sharpe"`
	AvgSharpe  float64         `json:"avg_sharpe"`
}

func (hg *HeatmapGenerator) Generate(strategy, symbol, param1, param2 string) *HeatmapOutput {
	completed := hg.store.Completed()

	// Filter and collect unique param values
	param1Set := make(map[string]interface{})
	param2Set := make(map[string]interface{})
	type dataPoint struct {
		p1Key  string
		p2Key  string
		p1Val  interface{}
		p2Val  interface{}
		sharpe float64
	}
	points := make([]dataPoint, 0)

	for _, j := range completed {
		if !strings.EqualFold(j.Config.Strategy, strategy) {
			continue
		}
		if symbol != "" && !strings.EqualFold(j.Config.Symbol, symbol) {
			continue
		}
		p1, ok1 := j.Config.Parameters[param1]
		p2, ok2 := j.Config.Parameters[param2]
		if !ok1 || !ok2 {
			continue
		}
		p1Key := formatAny(p1)
		p2Key := formatAny(p2)
		param1Set[p1Key] = p1
		param2Set[p2Key] = p2
		points = append(points, dataPoint{
			p1Key: p1Key, p2Key: p2Key,
			p1Val: p1, p2Val: p2,
			sharpe: j.Result.Sharpe,
		})
	}

	if len(points) == 0 {
		return &HeatmapOutput{Strategy: strategy, Symbol: symbol, Param1Name: param1, Param2Name: param2}
	}

	// Build sorted unique value lists
	p1Keys := sortedKeys(param1Set)
	p2Keys := sortedKeys(param2Set)

	p1Idx := make(map[string]int)
	p2Idx := make(map[string]int)
	p1Vals := make([]interface{}, len(p1Keys))
	p2Vals := make([]interface{}, len(p2Keys))
	for i, k := range p1Keys {
		p1Idx[k] = i
		p1Vals[i] = param1Set[k]
	}
	for i, k := range p2Keys {
		p2Idx[k] = i
		p2Vals[i] = param2Set[k]
	}

	// Build grid (NaN for missing)
	grid := make([][]float64, len(p1Keys))
	for i := range grid {
		grid[i] = make([]float64, len(p2Keys))
		for j := range grid[i] {
			grid[i][j] = math.NaN()
		}
	}

	bestSharpe := -math.MaxFloat64
	minSharpe := math.MaxFloat64
	sumSharpe := 0.0
	var bestP1, bestP2 interface{}

	for _, dp := range points {
		i := p1Idx[dp.p1Key]
		j := p2Idx[dp.p2Key]
		// Keep best if multiple
		if math.IsNaN(grid[i][j]) || dp.sharpe > grid[i][j] {
			grid[i][j] = dp.sharpe
		}
		if dp.sharpe > bestSharpe {
			bestSharpe = dp.sharpe
			bestP1 = dp.p1Val
			bestP2 = dp.p2Val
		}
		if dp.sharpe < minSharpe {
			minSharpe = dp.sharpe
		}
		sumSharpe += dp.sharpe
	}

	return &HeatmapOutput{
		Strategy:   strategy,
		Symbol:     symbol,
		Param1Name: param1,
		Param2Name: param2,
		Param1Vals: p1Vals,
		Param2Vals: p2Vals,
		Grid:       grid,
		BestParam1: bestP1,
		BestParam2: bestP2,
		BestSharpe: roundTo(bestSharpe, 4),
		MinSharpe:  roundTo(minSharpe, 4),
		MaxSharpe:  roundTo(bestSharpe, 4),
		AvgSharpe:  roundTo(sumSharpe/float64(len(points)), 4),
	}
}

func sortedKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ---- leaderboard generator -------------------------------------------------

type LeaderboardEntry struct {
	Rank           int     `json:"rank"`
	Strategy       string  `json:"strategy"`
	Symbol         string  `json:"symbol"`
	Timeframe      string  `json:"timeframe"`
	Sharpe         float64 `json:"sharpe"`
	Sortino        float64 `json:"sortino"`
	Return         float64 `json:"total_return"`
	MaxDD          float64 `json:"max_drawdown"`
	WinRate        float64 `json:"win_rate"`
	NTrades        int     `json:"n_trades"`
	IsSignificant  bool    `json:"is_significant"`
	PValue         float64 `json:"p_value"`
	StabilityScore float64 `json:"stability_score"`
}

func (ra *ResultAnalyzer) BuildLeaderboard(completed []*BacktestJob) []LeaderboardEntry {
	if len(completed) == 0 {
		return nil
	}

	// Best result per strategy+symbol
	type key struct{ s, sym string }
	best := make(map[key]*BacktestJob)
	for _, j := range completed {
		k := key{j.Config.Strategy, j.Config.Symbol}
		if existing, ok := best[k]; !ok || j.Result.Sharpe > existing.Result.Sharpe {
			best[k] = j
		}
	}

	entries := make([]LeaderboardEntry, 0, len(best))
	for _, j := range best {
		T := float64(j.Result.NTrades)
		if T < 10 {
			T = 10
		}
		se := 1.0 / math.Sqrt(T)
		z := j.Result.Sharpe / se
		pVal := 1.0 - normalCDF(z)

		// Stability: inverse of Sharpe variability for this strategy
		stabilityScore := computeStrategyStability(j.Config.Strategy, completed)

		entries = append(entries, LeaderboardEntry{
			Strategy:       j.Config.Strategy,
			Symbol:         j.Config.Symbol,
			Timeframe:      j.Config.Timeframe,
			Sharpe:         j.Result.Sharpe,
			Sortino:        j.Result.Sortino,
			Return:         j.Result.TotalReturn,
			MaxDD:          j.Result.MaxDrawdown,
			WinRate:        j.Result.WinRate,
			NTrades:        j.Result.NTrades,
			IsSignificant:  pVal < 0.05,
			PValue:         roundTo(pVal, 6),
			StabilityScore: roundTo(stabilityScore, 4),
		})
	}

	// Sort by Sharpe descending
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Sharpe > entries[j].Sharpe
	})

	for i := range entries {
		entries[i].Rank = i + 1
	}

	return entries
}

func computeStrategyStability(strategy string, completed []*BacktestJob) float64 {
	sharpes := make([]float64, 0)
	for _, j := range completed {
		if j.Config.Strategy == strategy {
			sharpes = append(sharpes, j.Result.Sharpe)
		}
	}
	if len(sharpes) < 2 {
		return 1.0
	}
	mean, std := meanStd(sharpes)
	if math.Abs(mean) < 1e-10 {
		return 0
	}
	// Stability = 1 / (1 + CV)
	cv := std / math.Abs(mean)
	return 1.0 / (1.0 + cv)
}

// ---- overfitting detector --------------------------------------------------

type OverfitFlag struct {
	Strategy     string  `json:"strategy"`
	Symbol       string  `json:"symbol"`
	ISSharpe     float64 `json:"in_sample_sharpe"`
	OOSSharpe    float64 `json:"out_of_sample_sharpe"`
	SharpeDecay  float64 `json:"sharpe_decay_pct"`
	IsOverfit    bool    `json:"is_overfit"`
	Severity     string  `json:"severity"`
	NVariations  int     `json:"n_variations"`
}

func (ra *ResultAnalyzer) detectOverfitting(completed []*BacktestJob) []OverfitFlag {
	// For each strategy, simulate IS vs OOS by splitting parameter variations
	// If the best parameter set has much higher Sharpe than the median, flag it
	byStrategy := make(map[string][]*BacktestJob)
	for _, j := range completed {
		byStrategy[j.Config.Strategy] = append(byStrategy[j.Config.Strategy], j)
	}

	flags := make([]OverfitFlag, 0)

	for strategy, jobs := range byStrategy {
		if len(jobs) < 5 {
			continue
		}

		// Sort by Sharpe
		sort.Slice(jobs, func(i, j int) bool {
			return jobs[i].Result.Sharpe > jobs[j].Result.Sharpe
		})

		// "In-sample" = best result, "Out-of-sample" = median result
		bestSharpe := jobs[0].Result.Sharpe
		medianIdx := len(jobs) / 2
		medianSharpe := jobs[medianIdx].Result.Sharpe

		// Also compute mean of bottom half as "OOS proxy"
		bottomHalf := jobs[len(jobs)/2:]
		sumBottom := 0.0
		for _, j := range bottomHalf {
			sumBottom += j.Result.Sharpe
		}
		oosSharpe := sumBottom / float64(len(bottomHalf))

		decay := 0.0
		if math.Abs(bestSharpe) > 1e-10 {
			decay = ((bestSharpe - oosSharpe) / math.Abs(bestSharpe)) * 100
		}

		severity := "none"
		isOverfit := false
		if decay > 80 {
			severity = "severe"
			isOverfit = true
		} else if decay > 50 {
			severity = "moderate"
			isOverfit = true
		} else if decay > 30 {
			severity = "mild"
			isOverfit = true
		}

		// Also flag if best >> median by more than 2 std
		allSharpes := make([]float64, len(jobs))
		for i, j := range jobs {
			allSharpes[i] = j.Result.Sharpe
		}
		_, std := meanStd(allSharpes)
		if bestSharpe > medianSharpe+2*std && std > 0.1 {
			if severity == "none" {
				severity = "mild"
				isOverfit = true
			}
		}

		flags = append(flags, OverfitFlag{
			Strategy:    strategy,
			Symbol:      jobs[0].Config.Symbol,
			ISSharpe:    roundTo(bestSharpe, 4),
			OOSSharpe:   roundTo(oosSharpe, 4),
			SharpeDecay: roundTo(decay, 2),
			IsOverfit:   isOverfit,
			Severity:    severity,
			NVariations: len(jobs),
		})
	}

	// Sort by decay descending
	sort.Slice(flags, func(i, j int) bool {
		return flags[i].SharpeDecay > flags[j].SharpeDecay
	})

	return flags
}

// ---- comparison report (paired t-test) -------------------------------------

type ComparisonReport struct {
	Strategy1     string  `json:"strategy_1"`
	Strategy2     string  `json:"strategy_2"`
	MeanSharpe1   float64 `json:"mean_sharpe_1"`
	MeanSharpe2   float64 `json:"mean_sharpe_2"`
	MeanDiff      float64 `json:"mean_diff"`
	TStatistic    float64 `json:"t_statistic"`
	PValue        float64 `json:"p_value"`
	IsSignificant bool    `json:"is_significant"`
	Winner        string  `json:"winner"`
	EffectSize    float64 `json:"cohens_d"`
	CommonSymbols int     `json:"common_symbols"`
}

func (ra *ResultAnalyzer) CompareStrategies(strat1, strat2 string) *ComparisonReport {
	completed := ra.engine.store.Completed()

	// Get best Sharpe per symbol for each strategy
	s1BySym := make(map[string]float64)
	s2BySym := make(map[string]float64)

	for _, j := range completed {
		switch j.Config.Strategy {
		case strat1:
			if existing, ok := s1BySym[j.Config.Symbol]; !ok || j.Result.Sharpe > existing {
				s1BySym[j.Config.Symbol] = j.Result.Sharpe
			}
		case strat2:
			if existing, ok := s2BySym[j.Config.Symbol]; !ok || j.Result.Sharpe > existing {
				s2BySym[j.Config.Symbol] = j.Result.Sharpe
			}
		}
	}

	// Paired differences on common symbols
	diffs := make([]float64, 0)
	s1Vals := make([]float64, 0)
	s2Vals := make([]float64, 0)
	for sym, v1 := range s1BySym {
		if v2, ok := s2BySym[sym]; ok {
			diffs = append(diffs, v1-v2)
			s1Vals = append(s1Vals, v1)
			s2Vals = append(s2Vals, v2)
		}
	}

	if len(diffs) < 2 {
		mean1, _ := meanStd(s1Vals)
		mean2, _ := meanStd(s2Vals)
		winner := strat1
		if mean2 > mean1 {
			winner = strat2
		}
		return &ComparisonReport{
			Strategy1:     strat1,
			Strategy2:     strat2,
			MeanSharpe1:   roundTo(mean1, 4),
			MeanSharpe2:   roundTo(mean2, 4),
			Winner:        winner,
			CommonSymbols: len(diffs),
		}
	}

	meanDiff, stdDiff := meanStd(diffs)
	n := float64(len(diffs))
	se := stdDiff / math.Sqrt(n)

	tStat := 0.0
	if se > 1e-10 {
		tStat = meanDiff / se
	}

	// Approximate p-value using normal (for large n)
	pValue := 2 * (1 - normalCDF(math.Abs(tStat)))

	mean1, std1 := meanStd(s1Vals)
	mean2, std2 := meanStd(s2Vals)

	// Cohen's d
	pooledStd := math.Sqrt((std1*std1 + std2*std2) / 2)
	cohensD := 0.0
	if pooledStd > 1e-10 {
		cohensD = meanDiff / pooledStd
	}

	winner := strat1
	if mean2 > mean1 {
		winner = strat2
	}

	return &ComparisonReport{
		Strategy1:     strat1,
		Strategy2:     strat2,
		MeanSharpe1:   roundTo(mean1, 4),
		MeanSharpe2:   roundTo(mean2, 4),
		MeanDiff:      roundTo(meanDiff, 4),
		TStatistic:    roundTo(tStat, 4),
		PValue:        roundTo(pValue, 6),
		IsSignificant: pValue < 0.05,
		Winner:        winner,
		EffectSize:    roundTo(cohensD, 4),
		CommonSymbols: len(diffs),
	}
}

// ---- cross-validation splitter for robustness ------------------------------

type CVFold struct {
	FoldIndex   int     `json:"fold_index"`
	TrainSharpe float64 `json:"train_sharpe"`
	TestSharpe  float64 `json:"test_sharpe"`
	SharpeDecay float64 `json:"sharpe_decay_pct"`
}

type CVResult struct {
	Strategy    string   `json:"strategy"`
	NFolds      int      `json:"n_folds"`
	Folds       []CVFold `json:"folds"`
	MeanTrain   float64  `json:"mean_train_sharpe"`
	MeanTest    float64  `json:"mean_test_sharpe"`
	AvgDecay    float64  `json:"avg_decay_pct"`
	IsRobust    bool     `json:"is_robust"`
}

func (ra *ResultAnalyzer) CrossValidateStrategy(strategy string, nFolds int) *CVResult {
	completed := ra.engine.store.Completed()

	// Filter for this strategy
	jobs := make([]*BacktestJob, 0)
	for _, j := range completed {
		if strings.EqualFold(j.Config.Strategy, strategy) {
			jobs = append(jobs, j)
		}
	}

	if len(jobs) < nFolds*2 {
		return &CVResult{Strategy: strategy, NFolds: 0}
	}

	// Sort by submitted time for temporal splitting
	sort.Slice(jobs, func(i, j int) bool {
		return jobs[i].SubmittedAt.Before(jobs[j].SubmittedAt)
	})

	foldSize := len(jobs) / nFolds
	folds := make([]CVFold, nFolds)
	sumTrain, sumTest := 0.0, 0.0

	for f := 0; f < nFolds; f++ {
		testStart := f * foldSize
		testEnd := testStart + foldSize
		if f == nFolds-1 {
			testEnd = len(jobs)
		}

		trainSharpes := make([]float64, 0)
		testSharpes := make([]float64, 0)

		for i, j := range jobs {
			if i >= testStart && i < testEnd {
				testSharpes = append(testSharpes, j.Result.Sharpe)
			} else {
				trainSharpes = append(trainSharpes, j.Result.Sharpe)
			}
		}

		trainMean, _ := meanStd(trainSharpes)
		testMean, _ := meanStd(testSharpes)

		decay := 0.0
		if math.Abs(trainMean) > 1e-10 {
			decay = ((trainMean - testMean) / math.Abs(trainMean)) * 100
		}

		folds[f] = CVFold{
			FoldIndex:   f,
			TrainSharpe: roundTo(trainMean, 4),
			TestSharpe:  roundTo(testMean, 4),
			SharpeDecay: roundTo(decay, 2),
		}
		sumTrain += trainMean
		sumTest += testMean
	}

	avgTrain := sumTrain / float64(nFolds)
	avgTest := sumTest / float64(nFolds)
	avgDecay := 0.0
	if math.Abs(avgTrain) > 1e-10 {
		avgDecay = ((avgTrain - avgTest) / math.Abs(avgTrain)) * 100
	}

	return &CVResult{
		Strategy:  strategy,
		NFolds:    nFolds,
		Folds:     folds,
		MeanTrain: roundTo(avgTrain, 4),
		MeanTest:  roundTo(avgTest, 4),
		AvgDecay:  roundTo(avgDecay, 2),
		IsRobust:  avgDecay < 30 && avgTest > 0,
	}
}

// ---- Monte Carlo significance test -----------------------------------------

type MonteCarloResult struct {
	Strategy      string  `json:"strategy"`
	ObservedSR    float64 `json:"observed_sharpe"`
	MeanRandomSR  float64 `json:"mean_random_sharpe"`
	StdRandomSR   float64 `json:"std_random_sharpe"`
	Percentile    float64 `json:"percentile"`
	PValue        float64 `json:"p_value"`
	IsSignificant bool    `json:"is_significant"`
	NSimulations  int     `json:"n_simulations"`
}

func (ra *ResultAnalyzer) MonteCarloTest(strategy string, nSims int) *MonteCarloResult {
	completed := ra.engine.store.Completed()

	// Find best Sharpe for this strategy
	bestSharpe := -math.MaxFloat64
	allSharpes := make([]float64, 0)
	for _, j := range completed {
		allSharpes = append(allSharpes, j.Result.Sharpe)
		if strings.EqualFold(j.Config.Strategy, strategy) && j.Result.Sharpe > bestSharpe {
			bestSharpe = j.Result.Sharpe
		}
	}

	if bestSharpe == -math.MaxFloat64 || len(allSharpes) < 10 {
		return &MonteCarloResult{Strategy: strategy, NSimulations: 0}
	}

	// Bootstrap: randomly sample from all Sharpes and take max
	rng := newLCG(uint64(len(allSharpes)) * 12345)
	randomMaxes := make([]float64, nSims)

	nPerSample := len(allSharpes)
	if nPerSample > 100 {
		nPerSample = 100
	}

	for sim := 0; sim < nSims; sim++ {
		maxSR := -math.MaxFloat64
		for k := 0; k < nPerSample; k++ {
			idx := int(rng.Next() % uint64(len(allSharpes)))
			if allSharpes[idx] > maxSR {
				maxSR = allSharpes[idx]
			}
		}
		randomMaxes[sim] = maxSR
	}

	sort.Float64s(randomMaxes)

	// Percentile of observed in random distribution
	rank := sort.SearchFloat64s(randomMaxes, bestSharpe)
	percentile := float64(rank) / float64(nSims) * 100

	meanRand, stdRand := meanStd(randomMaxes)
	pValue := 1.0 - percentile/100.0
	if pValue < 0 {
		pValue = 0
	}

	return &MonteCarloResult{
		Strategy:      strategy,
		ObservedSR:    roundTo(bestSharpe, 4),
		MeanRandomSR:  roundTo(meanRand, 4),
		StdRandomSR:   roundTo(stdRand, 4),
		Percentile:    roundTo(percentile, 2),
		PValue:        roundTo(pValue, 4),
		IsSignificant: pValue < 0.05,
		NSimulations:  nSims,
	}
}

// ---- tail risk analysis ----------------------------------------------------

type TailRiskMetrics struct {
	Strategy   string  `json:"strategy"`
	VaR95      float64 `json:"var_95"`
	VaR99      float64 `json:"var_99"`
	CVaR95     float64 `json:"cvar_95"`
	CVaR99     float64 `json:"cvar_99"`
	MaxLoss    float64 `json:"max_loss"`
	SkewReturn float64 `json:"skew_return"`
	KurtReturn float64 `json:"kurtosis_return"`
}

func (ra *ResultAnalyzer) ComputeTailRisk(completed []*BacktestJob) []TailRiskMetrics {
	byStrategy := make(map[string][]float64)
	for _, j := range completed {
		byStrategy[j.Config.Strategy] = append(byStrategy[j.Config.Strategy], j.Result.TotalReturn)
	}

	results := make([]TailRiskMetrics, 0, len(byStrategy))

	for strategy, returns := range byStrategy {
		if len(returns) < 10 {
			continue
		}

		sorted := make([]float64, len(returns))
		copy(sorted, returns)
		sort.Float64s(sorted)

		n := len(sorted)

		// VaR: loss at given percentile (lower tail)
		idx95 := int(math.Floor(0.05 * float64(n)))
		idx99 := int(math.Floor(0.01 * float64(n)))
		if idx95 < 0 {
			idx95 = 0
		}
		if idx99 < 0 {
			idx99 = 0
		}

		var95 := -sorted[idx95] // positive number = loss
		var99 := -sorted[idx99]

		// CVaR: mean of returns below VaR
		cvar95Sum := 0.0
		cvar95N := 0
		cvar99Sum := 0.0
		cvar99N := 0
		for i := 0; i <= idx95 && i < n; i++ {
			cvar95Sum += sorted[i]
			cvar95N++
		}
		for i := 0; i <= idx99 && i < n; i++ {
			cvar99Sum += sorted[i]
			cvar99N++
		}
		cvar95 := 0.0
		if cvar95N > 0 {
			cvar95 = -cvar95Sum / float64(cvar95N)
		}
		cvar99 := 0.0
		if cvar99N > 0 {
			cvar99 = -cvar99Sum / float64(cvar99N)
		}

		mean, std := meanStd(returns)
		skew := 0.0
		kurt := 0.0
		if std > 1e-10 {
			for _, r := range returns {
				d := (r - mean) / std
				skew += d * d * d
				kurt += d * d * d * d
			}
			skew /= float64(n)
			kurt = kurt/float64(n) - 3
		}

		results = append(results, TailRiskMetrics{
			Strategy:   strategy,
			VaR95:      roundTo(var95, 6),
			VaR99:      roundTo(var99, 6),
			CVaR95:     roundTo(cvar95, 6),
			CVaR99:     roundTo(cvar99, 6),
			MaxLoss:    roundTo(-sorted[0], 6),
			SkewReturn: roundTo(skew, 4),
			KurtReturn: roundTo(kurt, 4),
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].CVaR95 < results[j].CVaR95
	})

	return results
}

// ---- portfolio construction helper -----------------------------------------

type PortfolioCandidate struct {
	Strategy string  `json:"strategy"`
	Symbol   string  `json:"symbol"`
	Weight   float64 `json:"weight"`
	Sharpe   float64 `json:"sharpe"`
	Return   float64 `json:"return"`
	MaxDD    float64 `json:"max_dd"`
}

type PortfolioResult struct {
	Candidates       []PortfolioCandidate `json:"candidates"`
	ExpectedReturn   float64              `json:"expected_return"`
	ExpectedSharpe   float64              `json:"expected_sharpe"`
	ExpectedMaxDD    float64              `json:"expected_max_dd"`
	NStrategies      int                  `json:"n_strategies"`
	HHI              float64              `json:"concentration_hhi"`
	MaxWeight        float64              `json:"max_weight"`
}

func (ra *ResultAnalyzer) BuildPortfolio(maxStrategies int, minSharpe float64) *PortfolioResult {
	completed := ra.engine.store.Completed()
	if len(completed) == 0 {
		return nil
	}

	// Best per strategy
	bestPerStrat := make(map[string]*BacktestJob)
	for _, j := range completed {
		if j.Result.Sharpe < minSharpe {
			continue
		}
		if existing, ok := bestPerStrat[j.Config.Strategy]; !ok || j.Result.Sharpe > existing.Result.Sharpe {
			bestPerStrat[j.Config.Strategy] = j
		}
	}

	candidates := make([]*BacktestJob, 0, len(bestPerStrat))
	for _, j := range bestPerStrat {
		candidates = append(candidates, j)
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Result.Sharpe > candidates[j].Result.Sharpe
	})
	if len(candidates) > maxStrategies {
		candidates = candidates[:maxStrategies]
	}

	if len(candidates) == 0 {
		return &PortfolioResult{}
	}

	// Equal risk contribution (simplified: inverse max_dd weighting)
	totalInvRisk := 0.0
	invRisks := make([]float64, len(candidates))
	for i, j := range candidates {
		risk := j.Result.MaxDrawdown
		if risk < 0.01 {
			risk = 0.01
		}
		invRisks[i] = 1.0 / risk
		totalInvRisk += invRisks[i]
	}

	portfolio := make([]PortfolioCandidate, len(candidates))
	expReturn := 0.0
	expSharpe := 0.0
	expDD := 0.0
	sumWtSq := 0.0
	maxWt := 0.0

	for i, j := range candidates {
		w := invRisks[i] / totalInvRisk
		portfolio[i] = PortfolioCandidate{
			Strategy: j.Config.Strategy,
			Symbol:   j.Config.Symbol,
			Weight:   roundTo(w, 4),
			Sharpe:   j.Result.Sharpe,
			Return:   j.Result.TotalReturn,
			MaxDD:    j.Result.MaxDrawdown,
		}
		expReturn += w * j.Result.TotalReturn
		expSharpe += w * j.Result.Sharpe
		expDD += w * j.Result.MaxDrawdown
		sumWtSq += w * w
		if w > maxWt {
			maxWt = w
		}
	}

	return &PortfolioResult{
		Candidates:     portfolio,
		ExpectedReturn: roundTo(expReturn, 6),
		ExpectedSharpe: roundTo(expSharpe, 4),
		ExpectedMaxDD:  roundTo(expDD, 6),
		NStrategies:    len(portfolio),
		HHI:           roundTo(sumWtSq, 4),
		MaxWeight:      roundTo(maxWt, 4),
	}
}
