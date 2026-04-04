package feed

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"time"

	"go.uber.org/zap"
)

// SimulatorConfig controls the GBM + GARCH simulator.
type SimulatorConfig struct {
	// Symbols to simulate.
	Symbols []string
	// Mu is the annualised drift (GBM).
	Mu float64
	// Sigma is the annualised base volatility.
	Sigma float64
	// TickSize is the minimum price increment.
	TickSize float64
	// VolumeMean is the mean volume per bar (log-normal).
	VolumeMean float64
	// SpeedMult is how many real-time minutes each simulated minute takes.
	// 1.0 = wall-clock real-time.  1000.0 = 1000x faster.
	SpeedMult float64
	// BarInterval is the simulated bar width (default 1 minute).
	BarInterval time.Duration
	// InitialPrice is the starting price (default 100.0).
	InitialPrice float64
}

// DefaultSimulatorConfig returns a reasonable default configuration.
func DefaultSimulatorConfig(symbols []string) SimulatorConfig {
	return SimulatorConfig{
		Symbols:      symbols,
		Mu:           0.0001,
		Sigma:        0.015,
		TickSize:     0.01,
		VolumeMean:   100000,
		SpeedMult:    1.0,
		BarInterval:  time.Minute,
		InitialPrice: 100.0,
	}
}

// symbolState tracks the per-symbol GBM + GARCH state.
type symbolState struct {
	price   float64
	// GARCH(1,1) parameters
	omega   float64 // long-run variance constant
	alpha   float64 // ARCH coefficient
	beta    float64 // GARCH coefficient
	varT    float64 // current conditional variance
}

// SimulatorFeed generates synthetic OHLCV bars using GBM with GARCH volatility.
type SimulatorFeed struct {
	cfg    SimulatorConfig
	out    chan<- Event
	log    *zap.Logger
	cancel context.CancelFunc
	wg     sync.WaitGroup
	rng    *rand.Rand
}

// NewSimulatorFeed creates a new SimulatorFeed.
func NewSimulatorFeed(cfg SimulatorConfig, out chan<- Event, log *zap.Logger) *SimulatorFeed {
	return &SimulatorFeed{
		cfg: cfg,
		out: out,
		log: log,
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Start begins generating bars in a background goroutine per symbol.
func (f *SimulatorFeed) Start(ctx context.Context) {
	ctx, f.cancel = context.WithCancel(ctx)
	for _, sym := range f.cfg.Symbols {
		f.wg.Add(1)
		go f.runSymbol(ctx, sym)
	}
}

// Stop gracefully terminates all goroutines.
func (f *SimulatorFeed) Stop() {
	if f.cancel != nil {
		f.cancel()
	}
	f.wg.Wait()
}

func (f *SimulatorFeed) runSymbol(ctx context.Context, symbol string) {
	defer f.wg.Done()

	initPrice := f.cfg.InitialPrice
	if initPrice <= 0 {
		initPrice = 100.0
	}

	// GARCH(1,1) long-run variance calibrated to daily sigma ≈ cfg.Sigma.
	// We assume minutely returns; daily = sqrt(390) * minutely sigma.
	minutelySigma := f.cfg.Sigma / math.Sqrt(252*390)
	longRunVar := minutelySigma * minutelySigma

	state := symbolState{
		price: initPrice,
		omega: longRunVar * 0.1,
		alpha: 0.05,
		beta:  0.90,
		varT:  longRunVar,
	}

	barInterval := f.cfg.BarInterval
	if barInterval == 0 {
		barInterval = time.Minute
	}

	tickDur := time.Duration(float64(barInterval) / f.cfg.SpeedMult)
	if tickDur <= 0 {
		tickDur = time.Millisecond
	}

	ticker := time.NewTicker(tickDur)
	defer ticker.Stop()

	now := time.Now().UTC().Truncate(barInterval)

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-ticker.C:
			barTime := t.UTC().Truncate(barInterval)
			if f.cfg.SpeedMult > 1 {
				barTime = now
				now = now.Add(barInterval)
			}

			bar := f.generateBar(symbol, barTime, &state)
			f.emit(Event{Kind: EventBar, Bar: bar})
		}
	}
}

// generateBar produces a single synthetic bar using GBM + GARCH.
func (f *SimulatorFeed) generateBar(symbol string, ts time.Time, s *symbolState) *Bar {
	rng := f.rng

	// GARCH(1,1) variance update.
	//   epsilon_{t-1} is the last innovation; we approximate it as 0 initially.
	// Update: h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
	// For simplicity we draw the epsilon from the previous step.
	eps := rng.NormFloat64()
	prevReturn := math.Sqrt(s.varT) * eps
	s.varT = s.omega + s.alpha*prevReturn*prevReturn + s.beta*s.varT
	if s.varT < 1e-12 {
		s.varT = 1e-12
	}

	sigma := math.Sqrt(s.varT)
	mu := f.cfg.Mu / (252 * 390) // per-bar drift

	// GBM: ln(S_t/S_{t-1}) = (mu - sigma^2/2)*dt + sigma*dW
	logReturn := (mu-0.5*sigma*sigma) + sigma*rng.NormFloat64()
	closePrice := s.price * math.Exp(logReturn)

	// Build intra-bar OHLC using Brownian bridge approximation (4 sub-steps).
	prices := [5]float64{s.price}
	for i := 1; i <= 4; i++ {
		subSigma := sigma / 2
		subRet := (mu/4 - 0.5*subSigma*subSigma) + subSigma*rng.NormFloat64()
		prices[i] = prices[i-1] * math.Exp(subRet)
	}
	prices[4] = closePrice

	high := prices[0]
	low := prices[0]
	for _, p := range prices[1:] {
		if p > high {
			high = p
		}
		if p < low {
			low = p
		}
	}

	// Round to tick size.
	round := func(v float64) float64 {
		if f.cfg.TickSize <= 0 {
			return v
		}
		return math.Round(v/f.cfg.TickSize) * f.cfg.TickSize
	}

	open := round(s.price)
	high = round(high)
	low = round(low)
	closePrice = round(closePrice)

	// Ensure OHLC consistency after rounding.
	if high < open {
		high = open
	}
	if low > open {
		low = open
	}
	if high < closePrice {
		high = closePrice
	}
	if low > closePrice {
		low = closePrice
	}

	// Log-normal volume.
	volMean := f.cfg.VolumeMean
	if volMean <= 0 {
		volMean = 100000
	}
	volSigma := 0.5
	logVol := math.Log(volMean) - 0.5*volSigma*volSigma
	volume := math.Exp(logVol + volSigma*rng.NormFloat64())

	// Advance price.
	s.price = closePrice

	return &Bar{
		Symbol:    symbol,
		Timestamp: ts,
		Open:      open,
		High:      high,
		Low:       low,
		Close:     closePrice,
		Volume:    math.Round(volume),
		Source:    "simulator",
	}
}

func (f *SimulatorFeed) emit(e Event) {
	select {
	case f.out <- e:
	default:
		f.log.Warn("simulator channel full, dropping bar",
			zap.String("symbol", func() string {
				if e.Bar != nil {
					return e.Bar.Symbol
				}
				return ""
			}()))
	}
}
