// Package historical builds and exposes hourly liquidity baseline profiles.
package historical

import (
	"sync"
	"time"
)

const minObservationsForProfile = 48 // at least 2 days of data per hour

// HourlyProfile holds the baseline statistics for one symbol at one hour of day.
type HourlyProfile struct {
	Symbol       string
	Hour         int
	AvgSpread    float64
	AvgDepth     float64
	AvgVolume    float64
	AvgTradeRate float64
	Observations int
	LastUpdated  time.Time
	Mature       bool // true once we have enough data to trust the profile
}

// profileKey uniquely identifies a symbol+hour combination.
type profileKey struct {
	Symbol string
	Hour   int
}

// accumulator holds running sums for one profile key.
type accumulator struct {
	sumSpread    float64
	sumDepth     float64
	sumVolume    float64
	sumTradeRate float64
	count        int
}

// ProfileStore stores and computes hourly profiles.
type ProfileStore struct {
	mu   sync.RWMutex
	acc  map[profileKey]*accumulator
}

// NewProfileStore creates an empty ProfileStore.
func NewProfileStore() *ProfileStore {
	return &ProfileStore{
		acc: make(map[profileKey]*accumulator),
	}
}

// Record adds one observation to the profile for symbol at the current UTC hour.
func (p *ProfileStore) Record(symbol string, spread, depth, volume, tradeRate float64) {
	hour := time.Now().UTC().Hour()
	key := profileKey{Symbol: symbol, Hour: hour}

	p.mu.Lock()
	a, ok := p.acc[key]
	if !ok {
		a = &accumulator{}
		p.acc[key] = a
	}
	a.sumSpread += spread
	a.sumDepth += depth
	a.sumVolume += volume
	a.sumTradeRate += tradeRate
	a.count++
	p.mu.Unlock()
}

// Profile returns the current profile for symbol at hour. If not enough data
// has been collected, Mature will be false.
func (p *ProfileStore) Profile(symbol string, hour int) HourlyProfile {
	key := profileKey{Symbol: symbol, Hour: hour}
	p.mu.RLock()
	a, ok := p.acc[key]
	p.mu.RUnlock()

	if !ok || a.count == 0 {
		return HourlyProfile{Symbol: symbol, Hour: hour}
	}
	n := float64(a.count)
	return HourlyProfile{
		Symbol:       symbol,
		Hour:         hour,
		AvgSpread:    a.sumSpread / n,
		AvgDepth:     a.sumDepth / n,
		AvgVolume:    a.sumVolume / n,
		AvgTradeRate: a.sumTradeRate / n,
		Observations: a.count,
		LastUpdated:  time.Now().UTC(),
		Mature:       a.count >= minObservationsForProfile,
	}
}

// AllProfiles returns all currently stored profiles.
func (p *ProfileStore) AllProfiles() []HourlyProfile {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]HourlyProfile, 0, len(p.acc))
	for key, a := range p.acc {
		if a.count == 0 {
			continue
		}
		n := float64(a.count)
		out = append(out, HourlyProfile{
			Symbol:       key.Symbol,
			Hour:         key.Hour,
			AvgSpread:    a.sumSpread / n,
			AvgDepth:     a.sumDepth / n,
			AvgVolume:    a.sumVolume / n,
			AvgTradeRate: a.sumTradeRate / n,
			Observations: a.count,
			LastUpdated:  time.Now().UTC(),
			Mature:       a.count >= minObservationsForProfile,
		})
	}
	return out
}

// SpreadDeviation returns how many multiples of average the current spread is
// for this symbol and hour. Returns 1.0 if no profile exists.
func (p *ProfileStore) SpreadDeviation(symbol string, currentSpread float64) float64 {
	hour := time.Now().UTC().Hour()
	prof := p.Profile(symbol, hour)
	if !prof.Mature || prof.AvgSpread == 0 {
		return 1.0
	}
	return currentSpread / prof.AvgSpread
}
