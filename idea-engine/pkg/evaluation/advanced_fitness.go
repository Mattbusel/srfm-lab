package evaluation

import (
	"math"
	"sort"
	"sync"
)

// ──────────────────────────────────────────────────────────────────────────────
// Core types
// ──────────────────────────────────────────────────────────────────────────────

type StrategyReturns struct {
	Daily    []float64 `json:"daily"`
	Monthly  []float64 `json:"monthly,omitempty"`
	Trades   []float64 `json:"trades,omitempty"`
	Turnover []float64 `json:"turnover,omitempty"`
	IC       []float64 `json:"ic,omitempty"` // Information coefficient time series
}

type FitnessVector struct {
	Sharpe          float64 `json:"sharpe"`
	Calmar          float64 `json:"calmar"`
	MaxDrawdown     float64 `json:"max_drawdown"`
	Stability       float64 `json:"stability"` // R-squared of cumulative returns
	Turnover        float64 `json:"turnover"`
	Sortino         float64 `json:"sortino"`
	TailRatio       float64 `json:"tail_ratio"`
	InformationRatio float64 `json:"information_ratio"`
	AlphaDecay      float64 `json:"alpha_decay"`
	Robustness      float64 `json:"robustness"`
	CostAdjSharpe   float64 `json:"cost_adj_sharpe"`
	CorrelationPen  float64 `json:"correlation_penalty"`
	DiversifBonus   float64 `json:"diversification_bonus"`
	WalkForwardEff  float64 `json:"walk_forward_efficiency"`
	Composite       float64 `json:"composite"`
}

type RegimeLabel int

const (
	RegimeBull RegimeLabel = iota
	RegimeBear
	RegimeSideways
	RegimeVolatile
	RegimeCount
)

func (r RegimeLabel) String() string {
	return [...]string{"bull", "bear", "sideways", "volatile"}[r]
}

type RegimeReturns struct {
	Label   RegimeLabel `json:"label"`
	Returns []float64   `json:"returns"`
	Sharpe  float64     `json:"sharpe"`
	MaxDD   float64     `json:"max_dd"`
	Count   int         `json:"count"`
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-objective fitness
// ──────────────────────────────────────────────────────────────────────────────

type MultiObjectiveFitness struct {
	Weights map[string]float64
}

func NewMultiObjectiveFitness() *MultiObjectiveFitness {
	return &MultiObjectiveFitness{
		Weights: map[string]float64{
			"sharpe":      0.25,
			"calmar":      0.15,
			"max_dd":      0.15,
			"stability":   0.15,
			"turnover":    0.10,
			"sortino":     0.10,
			"tail_ratio":  0.10,
		},
	}
}

func (mof *MultiObjectiveFitness) Evaluate(rets StrategyReturns) FitnessVector {
	var fv FitnessVector
	if len(rets.Daily) < 20 {
		return fv
	}

	dailyMean := meanF(rets.Daily) * 252
	dailyVol := stddevF(rets.Daily) * math.Sqrt(252)
	if dailyVol > 0 {
		fv.Sharpe = dailyMean / dailyVol
	}

	dd := maxDrawdownF(rets.Daily)
	fv.MaxDrawdown = dd
	if dd != 0 {
		fv.Calmar = dailyMean / math.Abs(dd)
	}

	fv.Stability = rSquared(rets.Daily)

	if len(rets.Turnover) > 0 {
		fv.Turnover = meanF(rets.Turnover)
	}

	downDev := downDeviation(rets.Daily) * math.Sqrt(252)
	if downDev > 0 {
		fv.Sortino = dailyMean / downDev
	}

	fv.TailRatio = tailRatio(rets.Daily)

	// Composite weighted score
	fv.Composite = mof.Weights["sharpe"]*clamp(fv.Sharpe/3, -1, 1) +
		mof.Weights["calmar"]*clamp(fv.Calmar/5, -1, 1) +
		mof.Weights["max_dd"]*(1+clamp(fv.MaxDrawdown*5, -1, 0)) +
		mof.Weights["stability"]*fv.Stability +
		mof.Weights["turnover"]*clamp(1-fv.Turnover/10, 0, 1) +
		mof.Weights["sortino"]*clamp(fv.Sortino/4, -1, 1) +
		mof.Weights["tail_ratio"]*clamp(fv.TailRatio/3, -1, 1)

	return fv
}

// ──────────────────────────────────────────────────────────────────────────────
// Pareto ranking: fast non-dominated sort + crowding distance
// ──────────────────────────────────────────────────────────────────────────────

type Individual struct {
	ID       int
	Fitness  FitnessVector
	Rank     int
	Crowding float64
}

type ParetoRanker struct {
	Objectives []func(FitnessVector) float64 // higher = better
}

func NewParetoRanker() *ParetoRanker {
	return &ParetoRanker{
		Objectives: []func(FitnessVector) float64{
			func(f FitnessVector) float64 { return f.Sharpe },
			func(f FitnessVector) float64 { return f.Calmar },
			func(f FitnessVector) float64 { return -math.Abs(f.MaxDrawdown) }, // minimize DD
			func(f FitnessVector) float64 { return f.Stability },
			func(f FitnessVector) float64 { return -f.Turnover }, // minimize turnover
		},
	}
}

func (pr *ParetoRanker) dominates(a, b FitnessVector) bool {
	anyBetter := false
	for _, obj := range pr.Objectives {
		va := obj(a)
		vb := obj(b)
		if va < vb {
			return false
		}
		if va > vb {
			anyBetter = true
		}
	}
	return anyBetter
}

// FastNonDominatedSort implements NSGA-II's fast non-dominated sort.
func (pr *ParetoRanker) FastNonDominatedSort(pop []Individual) [][]int {
	n := len(pop)
	dominatedBy := make([][]int, n)
	dominationCount := make([]int, n)
	var fronts [][]int

	for i := 0; i < n; i++ {
		dominatedBy[i] = nil
		dominationCount[i] = 0
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			if pr.dominates(pop[i].Fitness, pop[j].Fitness) {
				dominatedBy[i] = append(dominatedBy[i], j)
			} else if pr.dominates(pop[j].Fitness, pop[i].Fitness) {
				dominationCount[i]++
			}
		}
	}

	// First front
	var front []int
	for i := 0; i < n; i++ {
		if dominationCount[i] == 0 {
			pop[i].Rank = 0
			front = append(front, i)
		}
	}
	fronts = append(fronts, front)

	rank := 0
	for len(fronts[rank]) > 0 {
		var nextFront []int
		for _, i := range fronts[rank] {
			for _, j := range dominatedBy[i] {
				dominationCount[j]--
				if dominationCount[j] == 0 {
					pop[j].Rank = rank + 1
					nextFront = append(nextFront, j)
				}
			}
		}
		rank++
		fronts = append(fronts, nextFront)
		if len(nextFront) == 0 {
			break
		}
	}

	return fronts
}

// CrowdingDistance computes crowding distance for a front.
func (pr *ParetoRanker) CrowdingDistance(pop []Individual, front []int) {
	n := len(front)
	if n <= 2 {
		for _, i := range front {
			pop[i].Crowding = math.Inf(1)
		}
		return
	}
	for _, i := range front {
		pop[i].Crowding = 0
	}

	for _, obj := range pr.Objectives {
		// Sort front by this objective
		indices := make([]int, n)
		copy(indices, front)
		sort.Slice(indices, func(a, b int) bool {
			return obj(pop[indices[a]].Fitness) < obj(pop[indices[b]].Fitness)
		})

		// Boundary points get infinite distance
		pop[indices[0]].Crowding = math.Inf(1)
		pop[indices[n-1]].Crowding = math.Inf(1)

		fmin := obj(pop[indices[0]].Fitness)
		fmax := obj(pop[indices[n-1]].Fitness)
		rng := fmax - fmin
		if rng == 0 {
			continue
		}

		for k := 1; k < n-1; k++ {
			pop[indices[k]].Crowding += (obj(pop[indices[k+1]].Fitness) - obj(pop[indices[k-1]].Fitness)) / rng
		}
	}
}

// RankPopulation performs full NSGA-II ranking.
func (pr *ParetoRanker) RankPopulation(pop []Individual) {
	fronts := pr.FastNonDominatedSort(pop)
	for _, front := range fronts {
		if len(front) == 0 {
			continue
		}
		pr.CrowdingDistance(pop, front)
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Regime-conditional fitness
// ──────────────────────────────────────────────────────────────────────────────

type RegimeConditionalFitness struct {
	RegimeWeights [int(RegimeCount)]float64
}

func NewRegimeConditionalFitness() *RegimeConditionalFitness {
	return &RegimeConditionalFitness{
		RegimeWeights: [4]float64{0.25, 0.35, 0.20, 0.20}, // bear gets more weight
	}
}

func (rcf *RegimeConditionalFitness) ClassifyRegimes(marketReturns []float64, windowSize int) []RegimeLabel {
	n := len(marketReturns)
	labels := make([]RegimeLabel, n)
	for i := 0; i < n; i++ {
		end := i + 1
		start := end - windowSize
		if start < 0 {
			start = 0
		}
		window := marketReturns[start:end]
		cumRet := sumF(window)
		vol := stddevF(window)

		if vol > 0.025 {
			labels[i] = RegimeVolatile
		} else if cumRet > 0.02 {
			labels[i] = RegimeBull
		} else if cumRet < -0.02 {
			labels[i] = RegimeBear
		} else {
			labels[i] = RegimeSideways
		}
	}
	return labels
}

func (rcf *RegimeConditionalFitness) Evaluate(stratReturns, marketReturns []float64) (float64, []RegimeReturns) {
	n := minInt(len(stratReturns), len(marketReturns))
	if n < 20 {
		return 0, nil
	}
	labels := rcf.ClassifyRegimes(marketReturns[:n], 21)

	regimeRets := make([][]float64, int(RegimeCount))
	for i := 0; i < n; i++ {
		l := int(labels[i])
		regimeRets[l] = append(regimeRets[l], stratReturns[i])
	}

	var results []RegimeReturns
	composite := 0.0
	for r := 0; r < int(RegimeCount); r++ {
		rr := RegimeReturns{Label: RegimeLabel(r), Returns: regimeRets[r], Count: len(regimeRets[r])}
		if len(regimeRets[r]) > 5 {
			vol := stddevF(regimeRets[r]) * math.Sqrt(252)
			mu := meanF(regimeRets[r]) * 252
			if vol > 0 {
				rr.Sharpe = mu / vol
			}
			rr.MaxDD = maxDrawdownF(regimeRets[r])
		}
		results = append(results, rr)
		composite += rcf.RegimeWeights[r] * rr.Sharpe
	}
	return composite, results
}

// ──────────────────────────────────────────────────────────────────────────────
// Robustness fitness: bootstrap resampling + parameter sensitivity
// ──────────────────────────────────────────────────────────────────────────────

type RobustnessFitness struct {
	NumBootstrap   int
	BlockSize      int
	ParamPerturbPct float64
}

func NewRobustnessFitness() *RobustnessFitness {
	return &RobustnessFitness{
		NumBootstrap:    500,
		BlockSize:       21,
		ParamPerturbPct: 0.10,
	}
}

// BootstrapSharpe computes bootstrap confidence interval for Sharpe ratio.
func (rf *RobustnessFitness) BootstrapSharpe(returns []float64) (median, p5, p95, stability float64) {
	n := len(returns)
	if n < rf.BlockSize*2 {
		return 0, 0, 0, 0
	}

	sharpes := make([]float64, rf.NumBootstrap)
	nBlocks := n / rf.BlockSize

	for b := 0; b < rf.NumBootstrap; b++ {
		sample := make([]float64, 0, n)
		seed := int64(b * 104729)
		for block := 0; block < nBlocks; block++ {
			seed = lcg(seed)
			startIdx := int(absI64(seed>>16) % int64(n-rf.BlockSize))
			sample = append(sample, returns[startIdx:startIdx+rf.BlockSize]...)
		}
		mu := meanF(sample) * 252
		vol := stddevF(sample) * math.Sqrt(252)
		if vol > 0 {
			sharpes[b] = mu / vol
		}
	}

	sort.Float64s(sharpes)
	median = percentileSorted(sharpes, 0.50)
	p5 = percentileSorted(sharpes, 0.05)
	p95 = percentileSorted(sharpes, 0.95)

	// Stability: what fraction of bootstrap samples have Sharpe > 0
	pos := 0
	for _, s := range sharpes {
		if s > 0 {
			pos++
		}
	}
	stability = float64(pos) / float64(rf.NumBootstrap)
	return
}

// ParameterSensitivity evaluates how sensitive Sharpe is to parameter perturbation.
// evalFn takes parameter vector and returns daily returns.
func (rf *RobustnessFitness) ParameterSensitivity(baseParams []float64, evalFn func([]float64) []float64) float64 {
	baseRets := evalFn(baseParams)
	baseSharpe := sharpeRatio(baseRets)
	if baseSharpe == 0 {
		return 0
	}

	var diffs []float64
	for i := range baseParams {
		perturbed := make([]float64, len(baseParams))
		copy(perturbed, baseParams)

		// Positive perturbation
		perturbed[i] *= (1 + rf.ParamPerturbPct)
		rets := evalFn(perturbed)
		s := sharpeRatio(rets)
		diffs = append(diffs, math.Abs(s-baseSharpe)/math.Abs(baseSharpe))

		// Negative perturbation
		perturbed[i] = baseParams[i] * (1 - rf.ParamPerturbPct)
		rets = evalFn(perturbed)
		s = sharpeRatio(rets)
		diffs = append(diffs, math.Abs(s-baseSharpe)/math.Abs(baseSharpe))
	}

	// Average relative change: lower = more robust
	avgSens := meanF(diffs)
	return math.Max(0, 1-avgSens) // robustness score in [0,1]
}

// ──────────────────────────────────────────────────────────────────────────────
// Cost-adjusted fitness
// ──────────────────────────────────────────────────────────────────────────────

type CostAdjustedFitness struct {
	CommissionBps   float64 // basis points per trade
	SlippageBps     float64
	BorrowCostAnn   float64 // annual cost for shorts
	SpreadCostBps   float64
}

func NewCostAdjustedFitness() *CostAdjustedFitness {
	return &CostAdjustedFitness{
		CommissionBps: 1.0,
		SlippageBps:   2.0,
		BorrowCostAnn: 0.005,
		SpreadCostBps: 1.0,
	}
}

func (caf *CostAdjustedFitness) Evaluate(rets StrategyReturns) FitnessVector {
	if len(rets.Daily) < 20 {
		return FitnessVector{}
	}

	// Total cost per unit turnover (in return terms)
	costPerTurn := (caf.CommissionBps + caf.SlippageBps + caf.SpreadCostBps) / 10000

	adjustedReturns := make([]float64, len(rets.Daily))
	copy(adjustedReturns, rets.Daily)

	for i := range adjustedReturns {
		turnover := 0.0
		if i < len(rets.Turnover) {
			turnover = rets.Turnover[i]
		}
		adjustedReturns[i] -= turnover * costPerTurn
		// Borrow cost (daily)
		adjustedReturns[i] -= caf.BorrowCostAnn / 252
	}

	mu := meanF(adjustedReturns) * 252
	vol := stddevF(adjustedReturns) * math.Sqrt(252)

	var fv FitnessVector
	if vol > 0 {
		fv.CostAdjSharpe = mu / vol
		fv.Sharpe = fv.CostAdjSharpe
	}
	fv.MaxDrawdown = maxDrawdownF(adjustedReturns)
	fv.Stability = rSquared(adjustedReturns)
	if len(rets.Turnover) > 0 {
		fv.Turnover = meanF(rets.Turnover)
	}
	return fv
}

// ──────────────────────────────────────────────────────────────────────────────
// Alpha decay fitness
// ──────────────────────────────────────────────────────────────────────────────

type AlphaDecayFitness struct {
	MinIC       float64
	DecayWindow int
}

func NewAlphaDecayFitness() *AlphaDecayFitness {
	return &AlphaDecayFitness{MinIC: 0.02, DecayWindow: 63}
}

// Evaluate computes alpha decay penalty. Returns (decay_rate, penalty).
func (adf *AlphaDecayFitness) Evaluate(icSeries []float64) (decayRate, penalty float64) {
	n := len(icSeries)
	if n < adf.DecayWindow {
		return 0, 0
	}

	// Linear regression of IC over time
	xs := make([]float64, n)
	for i := range xs {
		xs[i] = float64(i)
	}
	slope, _ := linearRegression(xs, icSeries)
	decayRate = slope

	// Penalty: higher if IC is declining
	if slope < 0 {
		// Annualize the decay
		annualDecay := slope * 252
		penalty = math.Abs(annualDecay)
	}

	// Additional penalty if recent IC is below minimum
	recentIC := meanF(icSeries[n-adf.DecayWindow:])
	if recentIC < adf.MinIC {
		penalty += (adf.MinIC - recentIC) * 10
	}

	return decayRate, penalty
}

// ──────────────────────────────────────────────────────────────────────────────
// Correlation penalty
// ──────────────────────────────────────────────────────────────────────────────

type CorrelationPenalty struct {
	MaxCorrelation float64
	PenaltyScale   float64
}

func NewCorrelationPenalty() *CorrelationPenalty {
	return &CorrelationPenalty{MaxCorrelation: 0.50, PenaltyScale: 2.0}
}

// Evaluate computes the correlation penalty against existing book strategies.
func (cp *CorrelationPenalty) Evaluate(candidateReturns []float64, bookReturns [][]float64) float64 {
	if len(bookReturns) == 0 || len(candidateReturns) < 20 {
		return 0
	}

	maxCorr := 0.0
	avgCorr := 0.0
	for _, existing := range bookReturns {
		n := minInt(len(candidateReturns), len(existing))
		if n < 20 {
			continue
		}
		c := math.Abs(pearsonCorr(candidateReturns[:n], existing[:n]))
		if c > maxCorr {
			maxCorr = c
		}
		avgCorr += c
	}
	avgCorr /= float64(len(bookReturns))

	penalty := 0.0
	if maxCorr > cp.MaxCorrelation {
		penalty = (maxCorr - cp.MaxCorrelation) * cp.PenaltyScale
	}
	// Additional penalty for high average correlation
	if avgCorr > cp.MaxCorrelation*0.7 {
		penalty += (avgCorr - cp.MaxCorrelation*0.7) * cp.PenaltyScale * 0.5
	}
	return penalty
}

// ──────────────────────────────────────────────────────────────────────────────
// Diversification bonus
// ──────────────────────────────────────────────────────────────────────────────

type DiversificationBonus struct {
	FactorNames []string
}

func NewDiversificationBonus() *DiversificationBonus {
	return &DiversificationBonus{
		FactorNames: []string{"market", "size", "value", "momentum", "quality", "volatility"},
	}
}

// Evaluate computes diversification bonus based on factor exposures.
// factorExposures: map[factor_name]beta to that factor.
func (db *DiversificationBonus) Evaluate(factorExposures map[string]float64, existingExposures []map[string]float64) float64 {
	if len(existingExposures) == 0 {
		return 0.5 // neutral bonus for first strategy
	}

	// Compute average existing exposure per factor
	avgExposure := make(map[string]float64)
	for _, exp := range existingExposures {
		for f, v := range exp {
			avgExposure[f] += v
		}
	}
	for f := range avgExposure {
		avgExposure[f] /= float64(len(existingExposures))
	}

	// Bonus for unique/orthogonal exposure
	bonus := 0.0
	for _, f := range db.FactorNames {
		candidateExp := factorExposures[f]
		avgExp := avgExposure[f]
		// Reward difference in exposure
		diff := math.Abs(candidateExp - avgExp)
		bonus += diff * 0.1
	}

	// Cap bonus
	if bonus > 1.0 {
		bonus = 1.0
	}
	return bonus
}

// ──────────────────────────────────────────────────────────────────────────────
// Walk-forward fitness with deflated Sharpe
// ──────────────────────────────────────────────────────────────────────────────

type WalkForwardFitness struct {
	TrainPct       float64
	NumFolds       int
	DeflateTrials  int // number of strategy trials for deflated Sharpe
}

func NewWalkForwardFitness() *WalkForwardFitness {
	return &WalkForwardFitness{
		TrainPct:      0.60,
		NumFolds:      5,
		DeflateTrials: 100,
	}
}

type WalkForwardResult struct {
	ISMetrics  FitnessVector `json:"is_metrics"`
	OOSMetrics FitnessVector `json:"oos_metrics"`
	Efficiency float64       `json:"efficiency"` // OOS_Sharpe / IS_Sharpe
	DeflatedSharpe float64   `json:"deflated_sharpe"`
}

func (wff *WalkForwardFitness) Evaluate(returns []float64) WalkForwardResult {
	n := len(returns)
	if n < 100 {
		return WalkForwardResult{}
	}

	foldSize := n / wff.NumFolds
	trainSize := int(float64(foldSize) * wff.TrainPct)
	testSize := foldSize - trainSize

	var isSharpes, oosSharpes []float64
	mof := NewMultiObjectiveFitness()

	for fold := 0; fold < wff.NumFolds; fold++ {
		start := fold * foldSize
		trainEnd := start + trainSize
		testEnd := trainEnd + testSize
		if testEnd > n {
			break
		}
		trainRets := returns[start:trainEnd]
		testRets := returns[trainEnd:testEnd]

		isFV := mof.Evaluate(StrategyReturns{Daily: trainRets})
		oosFV := mof.Evaluate(StrategyReturns{Daily: testRets})
		isSharpes = append(isSharpes, isFV.Sharpe)
		oosSharpes = append(oosSharpes, oosFV.Sharpe)
	}

	result := WalkForwardResult{}
	if len(isSharpes) > 0 {
		result.ISMetrics.Sharpe = meanF(isSharpes)
		result.OOSMetrics.Sharpe = meanF(oosSharpes)
		if result.ISMetrics.Sharpe != 0 {
			result.Efficiency = result.OOSMetrics.Sharpe / result.ISMetrics.Sharpe
		}
	}

	// Deflated Sharpe ratio (Harvey, Liu, Zhu 2016 approximation)
	result.DeflatedSharpe = wff.deflatedSharpe(returns)

	return result
}

func (wff *WalkForwardFitness) deflatedSharpe(returns []float64) float64 {
	sr := sharpeRatio(returns)
	n := float64(len(returns))
	if n < 10 {
		return 0
	}

	sk := skewnessF(returns)
	ku := kurtosisExcessF(returns)

	// Variance of Sharpe ratio estimate
	srVar := (1 - sk*sr + (ku-1)/4*sr*sr) / n

	// Expected max Sharpe from wff.DeflateTrials independent trials
	// E[max(SR)] ~ sqrt(2*log(trials)) (Bonferroni-like adjustment)
	trials := float64(wff.DeflateTrials)
	if trials < 1 {
		trials = 1
	}
	expectedMaxSR := math.Sqrt(2 * math.Log(trials))

	// Deflated Sharpe: prob that observed SR > E[max(SR)]
	if srVar <= 0 {
		return 0
	}
	z := (sr - expectedMaxSR) / math.Sqrt(srVar)
	// Approximate normal CDF
	return normCDF(z)
}

// ──────────────────────────────────────────────────────────────────────────────
// Fitness landscape analysis
// ──────────────────────────────────────────────────────────────────────────────

type LandscapePoint struct {
	Params  []float64 `json:"params"`
	Fitness float64   `json:"fitness"`
}

type FitnessLandscape struct {
	Points      []LandscapePoint `json:"points"`
	Ruggedness  float64          `json:"ruggedness"`
	Smoothness  float64          `json:"smoothness"`
	NumOptima   int              `json:"num_optima"`
	GlobalMax   LandscapePoint   `json:"global_max"`
	Gradient    []float64        `json:"gradient"`
}

type FitnessLandscapeAnalyzer struct {
	StepsPerDim int
	ParamRanges [][2]float64 // min,max per parameter
}

func NewFitnessLandscapeAnalyzer(ranges [][2]float64) *FitnessLandscapeAnalyzer {
	return &FitnessLandscapeAnalyzer{StepsPerDim: 20, ParamRanges: ranges}
}

// Analyze evaluates the fitness landscape by grid sampling.
func (fla *FitnessLandscapeAnalyzer) Analyze(evalFn func([]float64) float64) FitnessLandscape {
	nDims := len(fla.ParamRanges)
	if nDims == 0 {
		return FitnessLandscape{}
	}

	// Total grid points
	totalPoints := 1
	for d := 0; d < nDims; d++ {
		totalPoints *= fla.StepsPerDim
	}
	if totalPoints > 100000 {
		totalPoints = 100000 // safety cap
	}

	var landscape FitnessLandscape
	landscape.GlobalMax.Fitness = math.Inf(-1)

	// Concurrent evaluation
	type result struct {
		point LandscapePoint
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	batchSize := 100

	for start := 0; start < totalPoints; start += batchSize {
		end := start + batchSize
		if end > totalPoints {
			end = totalPoints
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for idx := s; idx < e; idx++ {
				params := fla.indexToParams(idx, nDims)
				fitness := evalFn(params)
				pt := LandscapePoint{Params: params, Fitness: fitness}

				mu.Lock()
				landscape.Points = append(landscape.Points, pt)
				if fitness > landscape.GlobalMax.Fitness {
					landscape.GlobalMax = pt
				}
				mu.Unlock()
			}
		}(start, end)
	}
	wg.Wait()

	// Compute ruggedness: average absolute difference between adjacent points
	landscape.Ruggedness = fla.computeRuggedness(landscape.Points)
	landscape.NumOptima = fla.countLocalOptima(landscape.Points)
	if landscape.Ruggedness > 0 {
		landscape.Smoothness = 1.0 / (1.0 + landscape.Ruggedness)
	}

	// Gradient at global max (finite differences)
	landscape.Gradient = fla.finiteDiffGradient(evalFn, landscape.GlobalMax.Params)

	return landscape
}

func (fla *FitnessLandscapeAnalyzer) indexToParams(idx, nDims int) []float64 {
	params := make([]float64, nDims)
	remainder := idx
	for d := nDims - 1; d >= 0; d-- {
		step := remainder % fla.StepsPerDim
		remainder /= fla.StepsPerDim
		lo := fla.ParamRanges[d][0]
		hi := fla.ParamRanges[d][1]
		params[d] = lo + (hi-lo)*float64(step)/float64(fla.StepsPerDim-1)
	}
	return params
}

func (fla *FitnessLandscapeAnalyzer) computeRuggedness(points []LandscapePoint) float64 {
	if len(points) < 2 {
		return 0
	}
	totalDiff := 0.0
	count := 0
	for i := 1; i < len(points); i++ {
		totalDiff += math.Abs(points[i].Fitness - points[i-1].Fitness)
		count++
	}
	if count == 0 {
		return 0
	}
	return totalDiff / float64(count)
}

func (fla *FitnessLandscapeAnalyzer) countLocalOptima(points []LandscapePoint) int {
	if len(points) < 3 {
		return 0
	}
	count := 0
	for i := 1; i < len(points)-1; i++ {
		if points[i].Fitness > points[i-1].Fitness && points[i].Fitness > points[i+1].Fitness {
			count++
		}
	}
	return count
}

func (fla *FitnessLandscapeAnalyzer) finiteDiffGradient(evalFn func([]float64) float64, params []float64) []float64 {
	grad := make([]float64, len(params))
	f0 := evalFn(params)
	for i := range params {
		perturbed := make([]float64, len(params))
		copy(perturbed, params)
		h := math.Abs(params[i]) * 0.01
		if h < 1e-8 {
			h = 1e-8
		}
		perturbed[i] += h
		f1 := evalFn(perturbed)
		grad[i] = (f1 - f0) / h
	}
	return grad
}

// ──────────────────────────────────────────────────────────────────────────────
// Composite fitness evaluator (orchestrator)
// ──────────────────────────────────────────────────────────────────────────────

type CompositeFitnessEvaluator struct {
	MultiObj   *MultiObjectiveFitness
	Regime     *RegimeConditionalFitness
	Robust     *RobustnessFitness
	CostAdj    *CostAdjustedFitness
	AlphaDecay *AlphaDecayFitness
	CorrPen    *CorrelationPenalty
	DivBonus   *DiversificationBonus
	WalkFwd    *WalkForwardFitness

	// Weight for each component in final score
	Weights map[string]float64
}

func NewCompositeFitnessEvaluator() *CompositeFitnessEvaluator {
	return &CompositeFitnessEvaluator{
		MultiObj:   NewMultiObjectiveFitness(),
		Regime:     NewRegimeConditionalFitness(),
		Robust:     NewRobustnessFitness(),
		CostAdj:    NewCostAdjustedFitness(),
		AlphaDecay: NewAlphaDecayFitness(),
		CorrPen:    NewCorrelationPenalty(),
		DivBonus:   NewDiversificationBonus(),
		WalkFwd:    NewWalkForwardFitness(),
		Weights: map[string]float64{
			"multi_obj":   0.20,
			"regime":      0.15,
			"robust":      0.15,
			"cost_adj":    0.15,
			"alpha_decay": 0.10,
			"corr_pen":    0.10,
			"div_bonus":   0.05,
			"walk_fwd":    0.10,
		},
	}
}

type CompositeFitnessResult struct {
	Vector         FitnessVector      `json:"vector"`
	RegimeScore    float64            `json:"regime_score"`
	RegimeDetails  []RegimeReturns    `json:"regime_details"`
	BootstrapP5    float64            `json:"bootstrap_p5"`
	BootstrapP95   float64            `json:"bootstrap_p95"`
	BootstrapStab  float64            `json:"bootstrap_stability"`
	CostAdjVector  FitnessVector      `json:"cost_adj_vector"`
	AlphaDecayRate float64            `json:"alpha_decay_rate"`
	AlphaDecayPen  float64            `json:"alpha_decay_penalty"`
	CorrPenalty    float64            `json:"correlation_penalty"`
	DivBonus       float64            `json:"diversification_bonus"`
	WalkFwdResult  WalkForwardResult  `json:"walk_forward"`
	FinalScore     float64            `json:"final_score"`
}

func (cfe *CompositeFitnessEvaluator) Evaluate(
	rets StrategyReturns,
	marketReturns []float64,
	bookReturns [][]float64,
	factorExposures map[string]float64,
	existingExposures []map[string]float64,
) CompositeFitnessResult {
	var result CompositeFitnessResult

	// Multi-objective
	result.Vector = cfe.MultiObj.Evaluate(rets)

	// Regime conditional
	result.RegimeScore, result.RegimeDetails = cfe.Regime.Evaluate(rets.Daily, marketReturns)

	// Robustness (bootstrap)
	_, result.BootstrapP5, result.BootstrapP95, result.BootstrapStab = cfe.Robust.BootstrapSharpe(rets.Daily)

	// Cost-adjusted
	result.CostAdjVector = cfe.CostAdj.Evaluate(rets)

	// Alpha decay
	if len(rets.IC) > 0 {
		result.AlphaDecayRate, result.AlphaDecayPen = cfe.AlphaDecay.Evaluate(rets.IC)
	}

	// Correlation penalty
	result.CorrPenalty = cfe.CorrPen.Evaluate(rets.Daily, bookReturns)

	// Diversification bonus
	result.DivBonus = cfe.DivBonus.Evaluate(factorExposures, existingExposures)

	// Walk-forward
	result.WalkFwdResult = cfe.WalkFwd.Evaluate(rets.Daily)

	// Final composite score
	score := 0.0
	score += cfe.Weights["multi_obj"] * result.Vector.Composite
	score += cfe.Weights["regime"] * clamp(result.RegimeScore/3, -1, 1)
	score += cfe.Weights["robust"] * result.BootstrapStab
	score += cfe.Weights["cost_adj"] * clamp(result.CostAdjVector.CostAdjSharpe/3, -1, 1)
	score -= cfe.Weights["alpha_decay"] * result.AlphaDecayPen
	score -= cfe.Weights["corr_pen"] * result.CorrPenalty
	score += cfe.Weights["div_bonus"] * result.DivBonus
	score += cfe.Weights["walk_fwd"] * clamp(result.WalkFwdResult.Efficiency, -1, 1)

	result.FinalScore = score
	result.Vector.Composite = score
	return result
}

// ──────────────────────────────────────────────────────────────────────────────
// Math utilities
// ──────────────────────────────────────────────────────────────────────────────

func meanF(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func sumF(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s
}

func stddevF(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := meanF(xs)
	s := 0.0
	for _, x := range xs {
		d := x - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(xs)-1))
}

func downDeviation(xs []float64) float64 {
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

func maxDrawdownF(dailyReturns []float64) float64 {
	cum := 1.0
	peak := 1.0
	maxDD := 0.0
	for _, r := range dailyReturns {
		cum *= (1 + r)
		if cum > peak {
			peak = cum
		}
		dd := (cum - peak) / peak
		if dd < maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

func rSquared(dailyReturns []float64) float64 {
	n := len(dailyReturns)
	if n < 10 {
		return 0
	}
	// R^2 of cumulative returns regressed on time
	cumRets := make([]float64, n)
	cumRets[0] = dailyReturns[0]
	for i := 1; i < n; i++ {
		cumRets[i] = cumRets[i-1] + dailyReturns[i]
	}
	xs := make([]float64, n)
	for i := range xs {
		xs[i] = float64(i)
	}
	_, r2 := linearRegression(xs, cumRets)
	return r2
}

func tailRatio(returns []float64) float64 {
	sorted := make([]float64, len(returns))
	copy(sorted, returns)
	sort.Float64s(sorted)
	p95 := percentileSorted(sorted, 0.95)
	p5 := math.Abs(percentileSorted(sorted, 0.05))
	if p5 == 0 {
		return 0
	}
	return p95 / p5
}

func sharpeRatio(returns []float64) float64 {
	if len(returns) < 2 {
		return 0
	}
	mu := meanF(returns) * 252
	vol := stddevF(returns) * math.Sqrt(252)
	if vol == 0 {
		return 0
	}
	return mu / vol
}

func skewnessF(xs []float64) float64 {
	n := float64(len(xs))
	if n < 3 {
		return 0
	}
	m := meanF(xs)
	sd := stddevF(xs)
	if sd == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += math.Pow((x-m)/sd, 3)
	}
	return (n / ((n - 1) * (n - 2))) * s
}

func kurtosisExcessF(xs []float64) float64 {
	n := float64(len(xs))
	if n < 4 {
		return 0
	}
	m := meanF(xs)
	sd := stddevF(xs)
	if sd == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += math.Pow((x-m)/sd, 4)
	}
	return (n*(n+1))/((n-1)*(n-2)*(n-3))*s - 3*(n-1)*(n-1)/((n-2)*(n-3))
}

func pearsonCorr(xs, ys []float64) float64 {
	n := len(xs)
	if n < 2 || n != len(ys) {
		return 0
	}
	mx := meanF(xs)
	my := meanF(ys)
	var num, dx2, dy2 float64
	for i := 0; i < n; i++ {
		dx := xs[i] - mx
		dy := ys[i] - my
		num += dx * dy
		dx2 += dx * dx
		dy2 += dy * dy
	}
	denom := math.Sqrt(dx2 * dy2)
	if denom == 0 {
		return 0
	}
	return num / denom
}

func linearRegression(xs, ys []float64) (slope, r2 float64) {
	n := float64(len(xs))
	if n < 2 {
		return 0, 0
	}
	mx := meanF(xs)
	my := meanF(ys)
	var ssxy, ssxx, ssyy float64
	for i := 0; i < int(n); i++ {
		dx := xs[i] - mx
		dy := ys[i] - my
		ssxy += dx * dy
		ssxx += dx * dx
		ssyy += dy * dy
	}
	if ssxx == 0 {
		return 0, 0
	}
	slope = ssxy / ssxx
	if ssyy == 0 {
		return slope, 1
	}
	r2 = (ssxy * ssxy) / (ssxx * ssyy)
	return
}

func percentileSorted(sorted []float64, p float64) float64 {
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

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func lcg(s int64) int64 {
	return s*6364136223846793005 + 1442695040888963407
}

func absI64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

func normCDF(x float64) float64 {
	// Abramowitz & Stegun approximation
	if x < -8 {
		return 0
	}
	if x > 8 {
		return 1
	}
	t := 1.0 / (1.0 + 0.2316419*math.Abs(x))
	d := 0.3989422804014327 * math.Exp(-x*x/2)
	p := d * t * (0.3193815 + t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274))))
	if x > 0 {
		return 1 - p
	}
	return p
}
