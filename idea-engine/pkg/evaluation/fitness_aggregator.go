package evaluation

import (
	"math"
	"sort"
)

// ---------------------------------------------------------------------------
// AggregatorConfig
// ---------------------------------------------------------------------------

// AggregatorConfig controls the weighting and penalty parameters used by
// FitnessAggregator. The zero value applies the defaults documented on each
// field.
type AggregatorConfig struct {
	// Weight3M is the weight given to the 3-month period evaluation.
	// Default: 0.40.
	Weight3M float64
	// Weight6M is the weight given to the 6-month period evaluation.
	// Default: 0.35.
	Weight6M float64
	// Weight12M is the weight given to the 12-month period evaluation.
	// Default: 0.25.
	Weight12M float64
	// DrawdownPenaltyThreshold is the drawdown level above which a penalty
	// is applied. Default: 0.15 (15%).
	DrawdownPenaltyThreshold float64
	// DrawdownPenaltyRange is the denominator for the penalty multiplier.
	// Default: 0.10.
	DrawdownPenaltyRange float64
	// MinFillsThreshold is the fill count below which the genome is
	// considered too thinly traded. Default: 10.
	MinFillsThreshold int
	// SharpeDeflationBase is the target trade count used in AdjustedSharpe.
	// Default: 100.
	SharpeDeflationBase float64
}

// withDefaults returns a copy of c with zero fields replaced by defaults.
func (c AggregatorConfig) withDefaults() AggregatorConfig {
	if c.Weight3M == 0 {
		c.Weight3M = 0.40
	}
	if c.Weight6M == 0 {
		c.Weight6M = 0.35
	}
	if c.Weight12M == 0 {
		c.Weight12M = 0.25
	}
	if c.DrawdownPenaltyThreshold == 0 {
		c.DrawdownPenaltyThreshold = 0.15
	}
	if c.DrawdownPenaltyRange == 0 {
		c.DrawdownPenaltyRange = 0.10
	}
	if c.MinFillsThreshold == 0 {
		c.MinFillsThreshold = 10
	}
	if c.SharpeDeflationBase == 0 {
		c.SharpeDeflationBase = 100
	}
	return c
}

// ---------------------------------------------------------------------------
// FitnessAggregator
// ---------------------------------------------------------------------------

// FitnessAggregator combines multi-period EvalResult slices into scalar
// fitness scores suitable for driving genetic selection.
type FitnessAggregator struct {
	cfg AggregatorConfig
}

// NewFitnessAggregator constructs a FitnessAggregator with the given config.
// Fields left at zero in cfg receive their default values.
func NewFitnessAggregator(cfg AggregatorConfig) *FitnessAggregator {
	return &FitnessAggregator{cfg: cfg.withDefaults()}
}

// Aggregate computes a weighted fitness score from a slice of EvalResult.
//
// The slice is expected to contain results from periods in the order:
//   [0] 3-month period
//   [1] 6-month period
//   [2] 12-month period
//
// If the slice has fewer than 3 elements, the available results are weighted
// proportionally. Each result's Sharpe is adjusted for trade count via
// AdjustedSharpe, then the weighted sum is penalised for excess drawdown.
//
// Returns 0 if results is empty.
func (a *FitnessAggregator) Aggregate(results []EvalResult) float64 {
	if len(results) == 0 {
		return 0
	}

	weights := []float64{a.cfg.Weight3M, a.cfg.Weight6M, a.cfg.Weight12M}

	// Use only as many weights as we have results.
	n := len(results)
	if n > 3 {
		n = 3
	}
	w := weights[:n]
	// Renormalise weights to sum to 1.
	var wSum float64
	for _, wi := range w {
		wSum += wi
	}
	if wSum == 0 {
		return 0
	}

	var score float64
	for i := 0; i < n; i++ {
		r := results[i]
		if !r.IsValid() {
			// Invalid result contributes zero to the weighted sum.
			continue
		}
		adj := a.AdjustedSharpe(r.Sharpe, r.NFills)
		penalised := a.applyDrawdownPenalty(adj, r.MaxDrawdown)
		score += (w[i] / wSum) * penalised
	}

	return score
}

// applyDrawdownPenalty multiplies score by a penalty factor when drawdown
// exceeds the configured threshold.
//
// Penalty factor = 1 - max(0, (dd - threshold) / range)
// The factor is clamped to [0, 1].
func (a *FitnessAggregator) applyDrawdownPenalty(score, drawdown float64) float64 {
	excess := drawdown - a.cfg.DrawdownPenaltyThreshold
	if excess <= 0 {
		return score
	}
	penalty := excess / a.cfg.DrawdownPenaltyRange
	factor := 1.0 - penalty
	if factor < 0 {
		factor = 0
	}
	return score * factor
}

// AdjustedSharpe deflates a raw Sharpe ratio when the trade count is low.
//
// Deflation factor = min(1, sqrt(nTrades / SharpeDeflationBase))
//
// This prevents the optimiser from over-selecting strategies that happened
// to be lucky over a small number of trades.
func (a *FitnessAggregator) AdjustedSharpe(sharpe float64, nTrades int) float64 {
	if nTrades <= 0 {
		return 0
	}
	factor := math.Sqrt(float64(nTrades) / a.cfg.SharpeDeflationBase)
	if factor > 1 {
		factor = 1
	}
	return sharpe * factor
}

// ParetoScore assigns a non-dominated Pareto rank to the given results and
// returns a normalised score in [0, 1] where higher is better.
//
// Three objectives are maximised simultaneously:
//   - Sharpe ratio (adjusted for trade count)
//   - Calmar ratio
//   - Negative max drawdown (i.e. lower drawdown is better)
//
// The score for each result is: 1 - (rank - 1) / N, where rank=1 is the
// Pareto front. If results has fewer than 2 elements, the single result
// returns 1.0 (or 0 for an invalid result).
//
// When results contains more than 3 periods, all are included in the Pareto
// comparison and the score is averaged across periods that contributed to
// the front.
func (a *FitnessAggregator) ParetoScore(results []EvalResult) float64 {
	if len(results) == 0 {
		return 0
	}
	if len(results) == 1 {
		if !results[0].IsValid() {
			return 0
		}
		return 1.0
	}

	// Build objective vectors. Objectives are all maximised.
	type obj struct {
		sharpe  float64 // adjusted Sharpe
		calmar  float64
		negDD   float64 // -MaxDrawdown (higher = lower drawdown)
		origIdx int
	}
	objs := make([]obj, len(results))
	for i, r := range results {
		objs[i] = obj{
			sharpe:  a.AdjustedSharpe(r.Sharpe, r.NFills),
			calmar:  r.Calmar,
			negDD:   -r.MaxDrawdown,
			origIdx: i,
		}
	}

	// Compute Pareto ranks using iterative front extraction.
	ranks := make([]int, len(objs))
	remaining := make([]int, len(objs))
	for i := range objs {
		remaining[i] = i
	}
	rank := 1
	for len(remaining) > 0 {
		front := paretoFront(objs, remaining)
		for _, idx := range front {
			ranks[idx] = rank
		}
		rank++
		// Remove front from remaining.
		frontSet := make(map[int]bool, len(front))
		for _, idx := range front {
			frontSet[idx] = true
		}
		next := remaining[:0]
		for _, idx := range remaining {
			if !frontSet[idx] {
				next = append(next, idx)
			}
		}
		remaining = next
	}

	// Average score across valid periods.
	maxRank := rank - 1
	var totalScore float64
	var count int
	for i, r := range results {
		if !r.IsValid() {
			continue
		}
		// Score = 1 - (rank-1)/maxRank normalised to [0,1].
		s := 1.0 - float64(ranks[i]-1)/float64(maxRank)
		totalScore += s
		count++
	}
	if count == 0 {
		return 0
	}
	return totalScore / float64(count)
}

// paretoFront returns the indices from `candidates` that are non-dominated
// within `objs`. All three objectives (sharpe, calmar, negDD) are maximised.
func paretoFront(objs []obj, candidates []int) []int {
	var front []int
	for _, i := range candidates {
		dominated := false
		for _, j := range candidates {
			if i == j {
				continue
			}
			if dominates(objs[j], objs[i]) {
				dominated = true
				break
			}
		}
		if !dominated {
			front = append(front, i)
		}
	}
	return front
}

// dominates returns true when a dominates b (a is at least as good on all
// objectives and strictly better on at least one).
func dominates(a, b obj) bool {
	atLeastAsGood := a.sharpe >= b.sharpe && a.calmar >= b.calmar && a.negDD >= b.negDD
	strictlyBetter := a.sharpe > b.sharpe || a.calmar > b.calmar || a.negDD > b.negDD
	return atLeastAsGood && strictlyBetter
}

// ---------------------------------------------------------------------------
// MultiPeriodEvaluator -- convenience wrapper
// ---------------------------------------------------------------------------

// PeriodConfig describes the date range for one evaluation period.
type PeriodConfig struct {
	// Label is a human-readable name (e.g. "3m", "6m", "12m").
	Label string
	// StartDate in "YYYY-MM-DD" format.
	StartDate string
	// EndDate in "YYYY-MM-DD" format.
	EndDate string
}

// MultiPeriodEvaluator runs a genome against multiple date ranges and
// aggregates the results into a single fitness score.
type MultiPeriodEvaluator struct {
	periods []PeriodConfig
	baseEval *GenomeEvaluator
	agg      *FitnessAggregator
}

// NewMultiPeriodEvaluator constructs a MultiPeriodEvaluator.
// periods must be in the order expected by FitnessAggregator.Aggregate
// (shortest to longest). If periods is empty, three default periods are used.
func NewMultiPeriodEvaluator(baseCfg EvalConfig, periods []PeriodConfig, aggCfg AggregatorConfig) *MultiPeriodEvaluator {
	if len(periods) == 0 {
		periods = defaultPeriods()
	}
	return &MultiPeriodEvaluator{
		periods:  periods,
		baseEval: NewGenomeEvaluator(baseCfg),
		agg:      NewFitnessAggregator(aggCfg),
	}
}

// Evaluate runs the genome against all configured periods and returns the
// aggregated fitness score along with the raw per-period results.
func (m *MultiPeriodEvaluator) Evaluate(genome map[string]float64) (float64, []EvalResult, error) {
	results := make([]EvalResult, len(m.periods))
	for i, p := range m.periods {
		cfg := m.baseEval.cfg
		cfg.StartDate = p.StartDate
		cfg.EndDate = p.EndDate
		ev := NewGenomeEvaluator(cfg)
		r, err := ev.Evaluate(genome)
		if err != nil {
			// Non-fatal: record the error in the result and continue.
			r.Error = err.Error()
		}
		results[i] = r
	}
	score := m.agg.Aggregate(results)
	return score, results, nil
}

// defaultPeriods returns the standard 3m / 6m / 12m evaluation periods.
// The exact dates are intentionally left empty so the Python script uses
// its own defaults; callers should override with actual dates.
func defaultPeriods() []PeriodConfig {
	return []PeriodConfig{
		{Label: "3m"},
		{Label: "6m"},
		{Label: "12m"},
	}
}

// ---------------------------------------------------------------------------
// Utility -- sort EvalResult by Sharpe descending
// ---------------------------------------------------------------------------

// SortBySharpeDESC sorts a slice of EvalResult by Sharpe ratio, highest first.
func SortBySharpeDESC(results []EvalResult) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Sharpe > results[j].Sharpe
	})
}
