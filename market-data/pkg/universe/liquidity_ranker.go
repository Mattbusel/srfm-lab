// Package universe manages the set of tradeable instruments.
// This file implements a composite liquidity score and ranking for symbols.
package universe

import (
	"sort"
)

// RankedSymbol holds a symbol's composite liquidity score and component ranks.
type RankedSymbol struct {
	Symbol         string  `json:"symbol"`
	LiquidityScore float64 `json:"liquidity_score"` // higher is more liquid
	ADVRank        int     `json:"adv_rank"`        // 1 == highest ADV
	SpreadRank     int     `json:"spread_rank"`     // 1 == tightest spread
}

// LiquidityRanker ranks a set of symbols by a composite liquidity score:
//
//	score = 0.6 * (1 - spread_rank/n) + 0.4 * (adv_rank_inverted/n)
//
// where spread_rank is 1-based (1 = tightest spread) and adv_rank is 1-based
// (1 = largest ADV).  The formula rewards both tight spreads (60% weight)
// and large ADV (40% weight).
//
// All public methods are stateless and safe for concurrent use.
type LiquidityRanker struct {
	manager *UniverseManager // optional, used by IsLiquidEnough
	ranked  []RankedSymbol   // result of the last Rank call
}

// NewLiquidityRanker constructs a ranker.  manager may be nil if only the
// stateless Rank/TopN methods are used.
func NewLiquidityRanker(manager *UniverseManager) *LiquidityRanker {
	return &LiquidityRanker{manager: manager}
}

// Rank computes a composite liquidity score for each symbol and returns the
// list sorted from most liquid to least liquid.
//
//   - adv:    symbol -> 30-day average daily volume in USD
//   - spread: symbol -> quoted spread as a fraction (e.g. 0.001 = 10 bps)
//
// Symbols not present in either map receive the worst rank for that dimension.
func (r *LiquidityRanker) Rank(
	symbols []string,
	adv map[string]float64,
	spread map[string]float64,
) []RankedSymbol {
	n := len(symbols)
	if n == 0 {
		return nil
	}

	// Build working slice for sorting.
	type entry struct {
		symbol string
		adv    float64
		spread float64
	}
	entries := make([]entry, n)
	for i, sym := range symbols {
		entries[i] = entry{
			symbol: sym,
			adv:    adv[sym],    // 0 if absent -- treated as worst
			spread: spread[sym], // 0 if absent -- treated as best; handled below
		}
		// A spread of 0 from a missing entry should be penalised, not rewarded.
		// Use a sentinel large value that will be ranked last.
		if _, ok := spread[sym]; !ok {
			entries[i].spread = 1e18
		}
	}

	// Rank by ADV descending (rank 1 = largest ADV).
	advSorted := make([]entry, n)
	copy(advSorted, entries)
	sort.SliceStable(advSorted, func(i, j int) bool {
		return advSorted[i].adv > advSorted[j].adv
	})
	advRank := make(map[string]int, n)
	for i, e := range advSorted {
		advRank[e.symbol] = i + 1
	}

	// Rank by spread ascending (rank 1 = tightest spread).
	spreadSorted := make([]entry, n)
	copy(spreadSorted, entries)
	sort.SliceStable(spreadSorted, func(i, j int) bool {
		return spreadSorted[i].spread < spreadSorted[j].spread
	})
	spreadRank := make(map[string]int, n)
	for i, e := range spreadSorted {
		spreadRank[e.symbol] = i + 1
	}

	fn := float64(n)
	result := make([]RankedSymbol, n)
	for i, sym := range symbols {
		ar := advRank[sym]
		sr := spreadRank[sym]
		// adv_rank is 1=best so invert: (n - adv_rank + 1)/n gives 1 for best.
		advScore := float64(n-ar+1) / fn
		// spread_rank is 1=best so invert similarly.
		spreadScore := float64(n-sr+1) / fn
		score := 0.4*advScore + 0.6*spreadScore
		result[i] = RankedSymbol{
			Symbol:         sym,
			LiquidityScore: score,
			ADVRank:        ar,
			SpreadRank:     sr,
		}
	}

	// Sort result by score descending.
	sort.SliceStable(result, func(i, j int) bool {
		return result[i].LiquidityScore > result[j].LiquidityScore
	})

	// Cache for TopN / IsLiquidEnough.
	r.ranked = result
	return result
}

// TopN returns the symbols of the top n most liquid instruments from the last
// Rank call.  If n exceeds the number of ranked symbols, all are returned.
func (r *LiquidityRanker) TopN(n int) []string {
	if n <= 0 {
		return nil
	}
	if n > len(r.ranked) {
		n = len(r.ranked)
	}
	out := make([]string, n)
	for i := 0; i < n; i++ {
		out[i] = r.ranked[i].Symbol
	}
	return out
}

// IsLiquidEnough returns true when an order of orderSizeUSD is less than 5%
// of the symbol's ADV30 as reported by the UniverseManager.
// If the symbol is not in the manager or the manager is nil, it returns false.
func (r *LiquidityRanker) IsLiquidEnough(symbol string, orderSizeUSD float64) bool {
	if r.manager == nil {
		return false
	}
	info, ok := r.manager.SymbolInfo(symbol)
	if !ok || info.ADV30 <= 0 {
		return false
	}
	return orderSizeUSD < 0.05*info.ADV30
}

// ScoreFor returns the LiquidityScore for a symbol from the last Rank call.
// Returns 0 and false if the symbol was not ranked.
func (r *LiquidityRanker) ScoreFor(symbol string) (float64, bool) {
	for _, rs := range r.ranked {
		if rs.Symbol == symbol {
			return rs.LiquidityScore, true
		}
	}
	return 0, false
}
