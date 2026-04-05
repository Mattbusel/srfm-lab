// Package scorer computes composite liquidity scores.
package scorer

import (
	"math"
	"time"

	"srfm-lab/idea-engine/liquidity-oracle/monitors"
)

// Recommendation classifies what action is appropriate given liquidity.
type Recommendation string

const (
	RecommendationDoNotTrade Recommendation = "DO_NOT_TRADE"
	RecommendationCaution    Recommendation = "CAUTION"
	RecommendationOK         Recommendation = "OK"
)

const (
	doNotTradeThreshold = 0.40
	cautionThreshold    = 0.70

	weightSpread = 0.40
	weightDepth  = 0.30
	weightVolume = 0.30
)

// LiquidityScore is the full composite score for one symbol.
type LiquidityScore struct {
	Symbol          string         `json:"symbol"`
	Composite       float64        `json:"composite"`        // 0–1
	SpreadScore     float64        `json:"spread_score"`     // 0–1
	DepthScore      float64        `json:"depth_score"`      // 0–1
	VolumeScore     float64        `json:"volume_score"`     // 0–1
	Recommendation  Recommendation `json:"recommendation"`
	SpreadPct       float64        `json:"spread_pct"`
	BidDepth        float64        `json:"bid_depth"`
	AskDepth        float64        `json:"ask_depth"`
	VolumeRatio     float64        `json:"volume_ratio"`
	TradeRateRatio  float64        `json:"trade_rate_ratio"`
	ScoredAt        time.Time      `json:"scored_at"`
}

// Scorer computes LiquidityScore from monitor data.
type Scorer struct {
	spreadMon    *monitors.SpreadMonitor
	depthMon     *monitors.DepthMonitor
	volumeMon    *monitors.VolumeMonitor
	tradeRateMon *monitors.TradeRateMonitor
}

// New creates a Scorer.
func New(
	spreadMon *monitors.SpreadMonitor,
	depthMon *monitors.DepthMonitor,
	volumeMon *monitors.VolumeMonitor,
	tradeRateMon *monitors.TradeRateMonitor,
) *Scorer {
	return &Scorer{
		spreadMon:    spreadMon,
		depthMon:     depthMon,
		volumeMon:    volumeMon,
		tradeRateMon: tradeRateMon,
	}
}

// Score returns the LiquidityScore for a Binance symbol (e.g. "BTCUSDT").
func (s *Scorer) Score(symbol string) LiquidityScore {
	now := time.Now().UTC()

	spreadScore := 0.5
	spreadPct := 0.0
	if obs, ok := s.spreadMon.Current(symbol); ok {
		spreadPct = obs.EffectiveSpread
		worst := s.spreadMon.WorstSpread(symbol)
		if worst > 0 {
			spreadScore = clamp01(1.0 - (spreadPct / worst))
		}
	}

	depthScore := 0.5
	bidDepth, askDepth := 0.0, 0.0
	if snap, ok := s.depthMon.Current(symbol); ok {
		bidDepth = snap.BidDepth
		askDepth = snap.AskDepth
		p20 := s.depthMon.Percentile20(symbol)
		if snap.TotalDepth > 0 && p20 > 0 {
			// Score 1 when at p80 or better; score 0 when at or below p20.
			depthScore = clamp01(snap.TotalDepth / (p20 * 4.0))
		}
	}

	volumeScore := 0.5
	volumeRatio := 0.0
	if snap, ok := s.volumeMon.Current(symbol); ok {
		volumeRatio = snap.Ratio
		// Clip ratio to [0, 2]; 1.0 ratio → score 0.5, 2.0 ratio → score 1.0.
		volumeScore = clamp01(snap.Ratio / 2.0)
	}

	tradeRateRatio := 0.0
	if snap, ok := s.tradeRateMon.Current(symbol); ok {
		tradeRateRatio = snap.Ratio
		// Blend trade rate into volume score (simple average).
		tradeScore := clamp01(snap.Ratio / 2.0)
		volumeScore = (volumeScore + tradeScore) / 2.0
	}

	composite := weightSpread*spreadScore + weightDepth*depthScore + weightVolume*volumeScore

	rec := RecommendationOK
	switch {
	case composite < doNotTradeThreshold:
		rec = RecommendationDoNotTrade
	case composite < cautionThreshold:
		rec = RecommendationCaution
	}

	return LiquidityScore{
		Symbol:         symbol,
		Composite:      math.Round(composite*1000) / 1000,
		SpreadScore:    math.Round(spreadScore*1000) / 1000,
		DepthScore:     math.Round(depthScore*1000) / 1000,
		VolumeScore:    math.Round(volumeScore*1000) / 1000,
		Recommendation: rec,
		SpreadPct:      spreadPct,
		BidDepth:       bidDepth,
		AskDepth:       askDepth,
		VolumeRatio:    volumeRatio,
		TradeRateRatio: tradeRateRatio,
		ScoredAt:       now,
	}
}

// ScoreAll computes scores for all tracked symbols.
func (s *Scorer) ScoreAll(symbols []string) []LiquidityScore {
	out := make([]LiquidityScore, 0, len(symbols))
	for _, sym := range symbols {
		out = append(out, s.Score(sym))
	}
	return out
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
