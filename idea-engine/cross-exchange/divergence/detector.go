// Package divergence detects price divergences across exchanges.
package divergence

import (
	"math"
	"sync"
	"time"

	"srfm-lab/idea-engine/cross-exchange/aggregator"
)

const historySize = 1000

// DivergenceSignal describes a detected price spread event.
type DivergenceSignal struct {
	Symbol           string    `json:"symbol"`
	MaxSpread        float64   `json:"max_spread"`
	LeadingExchange  string    `json:"leading_exchange"`
	LaggingExchange  string    `json:"lagging_exchange"`
	SpreadPct        float64   `json:"spread_pct"`
	Zscore           float64   `json:"zscore"`
	ConsensusPrice   float64   `json:"consensus_price"`
	DetectedAt       time.Time `json:"detected_at"`
}

// spreadHistory holds a rolling buffer of spread observations for one symbol.
type spreadHistory struct {
	observations []float64
	pos          int
	full         bool
}

func (h *spreadHistory) push(v float64) {
	if len(h.observations) < historySize {
		h.observations = append(h.observations, v)
	} else {
		h.observations[h.pos] = v
		h.pos = (h.pos + 1) % historySize
		h.full = true
	}
}

func (h *spreadHistory) stats() (mean, stddev float64) {
	n := len(h.observations)
	if n == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range h.observations {
		sum += v
	}
	mean = sum / float64(n)
	if n < 2 {
		return mean, 0
	}
	variance := 0.0
	for _, v := range h.observations {
		diff := v - mean
		variance += diff * diff
	}
	stddev = math.Sqrt(variance / float64(n-1))
	return mean, stddev
}

// Detector computes divergence signals from an aggregator.
type Detector struct {
	mu      sync.Mutex
	history map[string]*spreadHistory
}

// New creates a new Detector.
func New() *Detector {
	return &Detector{
		history: make(map[string]*spreadHistory),
	}
}

// Detect computes divergence signals from the current consensus prices.
func (d *Detector) Detect(consensus []aggregator.ConsensusPrices) []DivergenceSignal {
	signals := make([]DivergenceSignal, 0)
	now := time.Now().UTC()

	for _, cp := range consensus {
		if len(cp.Prices) < 2 {
			continue
		}
		if cp.ConsensusMid == 0 {
			continue
		}

		maxMid := -math.MaxFloat64
		minMid := math.MaxFloat64
		var leadExchange, lagExchange string

		for _, p := range cp.Prices {
			if p.Mid > maxMid {
				maxMid = p.Mid
				leadExchange = p.Exchange
			}
			if p.Mid < minMid {
				minMid = p.Mid
				lagExchange = p.Exchange
			}
		}

		spread := maxMid - minMid
		spreadPct := spread / cp.ConsensusMid

		d.mu.Lock()
		h, ok := d.history[cp.Symbol]
		if !ok {
			h = &spreadHistory{}
			d.history[cp.Symbol] = h
		}
		h.push(spreadPct)
		mean, stddev := h.stats()
		d.mu.Unlock()

		zscore := 0.0
		if stddev > 0 {
			zscore = (spreadPct - mean) / stddev
		}

		signals = append(signals, DivergenceSignal{
			Symbol:          cp.Symbol,
			MaxSpread:       spread,
			LeadingExchange: leadExchange,
			LaggingExchange: lagExchange,
			SpreadPct:       spreadPct,
			Zscore:          zscore,
			ConsensusPrice:  cp.ConsensusMid,
			DetectedAt:      now,
		})
	}
	return signals
}
