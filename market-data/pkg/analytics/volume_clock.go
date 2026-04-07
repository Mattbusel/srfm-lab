package analytics

import (
	"math"
	"sync"
	"time"
)

// VolumeBar is a bar closed when a target volume threshold is reached,
// rather than when a fixed time interval elapses.
type VolumeBar struct {
	Symbol    string    `json:"symbol"`
	Open      float64   `json:"open"`
	High      float64   `json:"high"`
	Low       float64   `json:"low"`
	Close     float64   `json:"close"`
	Volume    float64   `json:"volume"`
	TickCount int       `json:"tick_count"`   // number of ticks in this bar
	OpenTime  time.Time `json:"open_time"`
	CloseTime time.Time `json:"close_time"`
}

// volumeBarBuffer is a fixed-capacity circular buffer of completed VolumeBar values.
const volumeBarCapacity = 500

type volumeBarBuffer struct {
	buf  [volumeBarCapacity]VolumeBar
	head int // next write index
	size int
}

func (b *volumeBarBuffer) push(vb VolumeBar) {
	b.buf[b.head] = vb
	b.head = (b.head + 1) % volumeBarCapacity
	if b.size < volumeBarCapacity {
		b.size++
	}
}

// last returns the most recent n bars, oldest first.
// If n <= 0 or n > size, all available bars are returned.
func (b *volumeBarBuffer) last(n int) []VolumeBar {
	if b.size == 0 {
		return nil
	}
	if n <= 0 || n > b.size {
		n = b.size
	}

	out := make([]VolumeBar, n)
	// newest entry is at (head-1), oldest entry we want is at (head - n)
	startOffset := b.size - n // how many entries to skip from the oldest

	var absStart int
	if b.size < volumeBarCapacity {
		absStart = (startOffset) % volumeBarCapacity
	} else {
		absStart = (b.head + startOffset) % volumeBarCapacity
	}

	for i := 0; i < n; i++ {
		out[i] = b.buf[(absStart+i)%volumeBarCapacity]
	}
	return out
}

// symbolVolumeClock tracks the in-progress volume bar for one symbol and
// maintains the calibration state for target_volume.
type symbolVolumeClock struct {
	mu sync.Mutex

	symbol        string
	targetVolume  float64 // closes bar when accumulated volume >= this
	barsPerDay    int     // calibration parameter (default 50)
	autoCalibrate bool

	// Calibration: track daily volume via EWMA of per-bar volumes.
	// target_volume is re-set after each closed bar:
	//   target = adv_ema / barsPerDay
	advEMA   float64 // average daily volume EWMA (using closed bars)
	advAlpha float64 // EWMA alpha for ADV
	advCount int64

	// In-progress bar state
	inProgress bool
	openPrice  float64
	high       float64
	low        float64
	closePrice float64
	volume     float64
	tickCount  int
	openTime   time.Time

	// Completed bar history
	completed volumeBarBuffer
}

func newSymbolVolumeClock(symbol string, barsPerDay int) *symbolVolumeClock {
	if barsPerDay <= 0 {
		barsPerDay = 50
	}
	// ADV EWMA alpha: 2 / (N+1) with N = barsPerDay * 20 trading days
	advN := float64(barsPerDay * 20)
	return &symbolVolumeClock{
		symbol:        symbol,
		barsPerDay:    barsPerDay,
		autoCalibrate: true,
		advAlpha:      2.0 / (advN + 1.0),
		targetVolume:  1.0, // bootstrap: accept any volume as first bar
	}
}

// onTick processes a single trade tick (price, volume) and returns a
// completed VolumeBar if the target was reached, or nil otherwise.
func (sc *symbolVolumeClock) onTick(price, volume float64, ts time.Time) *VolumeBar {
	if price <= 0 || volume <= 0 {
		return nil
	}

	sc.mu.Lock()
	defer sc.mu.Unlock()

	if !sc.inProgress {
		sc.openPrice = price
		sc.high = price
		sc.low = price
		sc.closePrice = price
		sc.volume = 0
		sc.tickCount = 0
		sc.openTime = ts
		sc.inProgress = true
	}

	// Update running bar
	if price > sc.high {
		sc.high = price
	}
	if price < sc.low {
		sc.low = price
	}
	sc.closePrice = price
	sc.volume += volume
	sc.tickCount++

	// Check if bar is complete
	if sc.volume < sc.targetVolume {
		return nil
	}

	// Close the bar
	bar := VolumeBar{
		Symbol:    sc.symbol,
		Open:      sc.openPrice,
		High:      sc.high,
		Low:       sc.low,
		Close:     sc.closePrice,
		Volume:    sc.volume,
		TickCount: sc.tickCount,
		OpenTime:  sc.openTime,
		CloseTime: ts,
	}

	// Recalibrate target volume
	if sc.autoCalibrate {
		sc.updateTargetVolume(sc.volume)
	}

	sc.completed.push(bar)
	sc.inProgress = false

	return &bar
}

// updateTargetVolume updates the ADV EWMA and recalculates target_volume.
// Called with sc.mu held.
func (sc *symbolVolumeClock) updateTargetVolume(closedBarVolume float64) {
	sc.advCount++
	if sc.advCount == 1 {
		// Bootstrap: estimate one full day's volume as barsPerDay * first bar volume
		sc.advEMA = closedBarVolume * float64(sc.barsPerDay)
	} else {
		sc.advEMA = closedBarVolume*sc.advAlpha + sc.advEMA*(1-sc.advAlpha)
	}
	newTarget := sc.advEMA / float64(sc.barsPerDay)
	// Clamp: never go below a minimum meaningful volume to avoid empty bar spam
	sc.targetVolume = math.Max(newTarget, 1e-9)
}

// getBars returns the last n completed volume bars (oldest first).
func (sc *symbolVolumeClock) getBars(n int) []VolumeBar {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.completed.last(n)
}

// targetVol returns the current target volume.
func (sc *symbolVolumeClock) targetVol() float64 {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.targetVolume
}

// VolumeClock constructs equal-volume bars across multiple symbols.
// It is safe for concurrent use.
//
// Volume bars are essential for VPIN computation because they yield
// buckets of equal volume, making the flow-toxicity estimate stationary.
type VolumeClock struct {
	clocks     sync.Map // symbol -> *symbolVolumeClock
	barsPerDay int
}

// NewVolumeClock creates a VolumeClock.
// barsPerDay controls auto-calibration: target_volume = ADV / barsPerDay.
// Pass 0 to use the default of 50 bars per day.
func NewVolumeClock(barsPerDay int) *VolumeClock {
	if barsPerDay <= 0 {
		barsPerDay = 50
	}
	return &VolumeClock{barsPerDay: barsPerDay}
}

// OnTick processes a single trade tick for the given symbol.
// Returns a pointer to the completed VolumeBar if the target volume was
// reached, or nil if the bar is still in progress.
func (vc *VolumeClock) OnTick(symbol string, price, volume float64) *VolumeBar {
	return vc.OnTickAt(symbol, price, volume, time.Now().UTC())
}

// OnTickAt is like OnTick but accepts an explicit timestamp -- useful for
// back-testing and deterministic unit tests.
func (vc *VolumeClock) OnTickAt(symbol string, price, volume float64, ts time.Time) *VolumeBar {
	val, _ := vc.clocks.LoadOrStore(symbol, newSymbolVolumeClock(symbol, vc.barsPerDay))
	clock := val.(*symbolVolumeClock)
	return clock.onTick(price, volume, ts)
}

// GetBars returns the last n completed volume bars for a symbol, oldest first.
// Returns nil if the symbol has no history.
func (vc *VolumeClock) GetBars(symbol string, n int) []VolumeBar {
	val, ok := vc.clocks.Load(symbol)
	if !ok {
		return nil
	}
	clock := val.(*symbolVolumeClock)
	return clock.getBars(n)
}

// GetTargetVolume returns the current auto-calibrated target volume for a symbol.
// Returns 0 if the symbol has not been seen yet.
func (vc *VolumeClock) GetTargetVolume(symbol string) float64 {
	val, ok := vc.clocks.Load(symbol)
	if !ok {
		return 0
	}
	clock := val.(*symbolVolumeClock)
	return clock.targetVol()
}

// Symbols returns all symbols tracked by the clock.
func (vc *VolumeClock) Symbols() []string {
	var out []string
	vc.clocks.Range(func(k, _ interface{}) bool {
		out = append(out, k.(string))
		return true
	})
	return out
}
